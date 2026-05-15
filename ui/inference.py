import os
import sys
import glob
import json
import shutil
import subprocess
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from PIL import Image
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.model import generate_model

EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
SAMPLE_RATE = 22050
TARGET_SAMPLES = int(SAMPLE_RATE * 3.6)  # 79,380
N_FRAMES = 15
N_MELS = 64
N_MFCC = 10


def _load_run_metadata(run_dir):
    opts_files = sorted(glob.glob(os.path.join(run_dir, "opts*.json")))
    if not opts_files:
        return {}
    try:
        with open(opts_files[-1], "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def discover_checkpoints(repo_root):
    """Return discovered checkpoints and saved run metadata under results*/."""
    result = {}
    pattern = os.path.join(repo_root, "results*", "*", "*.pth")
    preferred_names = [
        "RAVDESS_multimodal_cnn_15_best.pth",
        "model.pth",
        "RAVDESS_multimodal_cnn_15_checkpoint.pth",
    ]

    by_run = {}
    for path in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(path)
        by_run.setdefault(run_dir, []).append(path)

    for run_dir, paths in sorted(by_run.items()):
        metadata = _load_run_metadata(run_dir)
        chosen = None
        for preferred in preferred_names:
            chosen = next((p for p in paths if os.path.basename(p) == preferred), None)
            if chosen:
                break
        if chosen is None:
            chosen = paths[0]

        run_name = os.path.basename(run_dir)
        display = run_name
        audio_feature = _infer_audio_feature(run_name, metadata)
        if metadata:
            num_heads = metadata.get("num_heads")
            fusion = metadata.get("fusion")
            lr = metadata.get("learning_rate")
            if num_heads is not None and fusion is not None and lr is not None:
                display = (
                    f"{run_name}  |  feature={audio_feature}  "
                    f"heads={num_heads}  fusion={fusion}  lr={lr}"
                )

        result[display] = {
            "path": chosen,
            "run_dir": run_dir,
            "metadata": metadata,
            "audio_feature": audio_feature,
        }
    return result


def load_model(pth_path, num_heads, fusion, device):
    """Load a MultiModalCNN checkpoint.

    Handles both state_dict-only and full-checkpoint dict formats, and strips
    DataParallel 'module.' prefixes automatically.  Uses device='cpu' in the
    opts so generate_model() never wraps in DataParallel (not needed for
    single-sample inference), then moves the bare model to the target device.
    """
    opt = SimpleNamespace(
        model='multimodal_cnn',
        n_classes=8,
        fusion=fusion,
        sample_duration=N_FRAMES,
        num_heads=num_heads,
        pretrain_path='None',  # skip EfficientFace init; we load fine-tuned weights
        device='cpu',          # avoid DataParallel; inference never needs it
    )
    model, _ = generate_model(opt)
    model = model.to(device)

    raw = torch.load(pth_path, map_location=device)
    state_dict = raw.get('state_dict', raw) if isinstance(raw, dict) else raw
    # Strip DataParallel 'module.' prefix so we can load into the bare model
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _infer_audio_feature(run_name, metadata):
    feature = metadata.get("audio_feature") if metadata else None
    if isinstance(feature, str) and feature.strip():
        return feature.strip().lower()

    lowered = run_name.lower()
    if "mfcc" in lowered:
        return "mfcc"
    if "mel" in lowered:
        return "mel"
    return "mel"


def preprocess_audio(video_path, feature_type="mel"):
    """Extract mono audio and return audio tensor (1, F, T) for the chosen feature."""
    ffmpeg_exe = _resolve_ffmpeg()
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp_wav = f.name
    try:
        subprocess.run(
            [ffmpeg_exe, '-y', '-i', video_path, '-ac', '1', '-ar', str(SAMPLE_RATE), tmp_wav],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        audio, _ = librosa.load(tmp_wav, sr=SAMPLE_RATE)
    except subprocess.CalledProcessError:
        # Video has no audio track — use silence
        audio = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

    # Crop or pad to exactly 3.6 s (mirrors datasets/ravdess.py: load_audio)
    if len(audio) < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
    else:
        excess = len(audio) - TARGET_SAMPLES
        audio = audio[excess // 2: len(audio) - (excess - excess // 2)]

    feature_type = (feature_type or "mel").lower()

    if feature_type == "mfcc":
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=1024,
            hop_length=512,
        )
        return torch.tensor(mfcc.astype(np.float32), dtype=torch.float32).unsqueeze(0)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=512,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db + 40) / 40
    return torch.tensor(mel_db.astype(np.float32), dtype=torch.float32).unsqueeze(0)


def _resolve_ffmpeg():
    """Return an ffmpeg executable path that works inside the current env."""
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        return ffmpeg_exe

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise FileNotFoundError("ffmpeg executable not available") from exc


def preprocess_video(video_path):
    """Sample 15 face-cropped frames. Returns tensor (3, 15, 224, 224) in [0, 1].

    Matches the shape that datasets/ravdess.py __getitem__ produces:
      torch.stack(clip, 0).permute(1, 0, 2, 3)  →  (C, T, H, W) = (3, 15, 224, 224)
    """
    try:
        from facenet_pytorch import MTCNN as _MTCNN
        mtcnn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mtcnn = _MTCNN(keep_all=False, device=mtcnn_device)
    except ImportError:
        mtcnn = None  # fall back to full-frame resize

    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        raw_frames.append(frame_rgb)
    cap.release()

    if not raw_frames:
        raw_frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * N_FRAMES

    indices = np.linspace(0, len(raw_frames) - 1, N_FRAMES, dtype=int)
    sampled = [raw_frames[i] for i in indices]

    face_tensors = [_extract_face(frame, mtcnn) for frame in sampled]
    stacked = torch.stack(face_tensors, dim=0)  # (15, 3, 224, 224) [T, C, H, W]
    return stacked.permute(1, 0, 2, 3)          # (3, 15, 224, 224) [C, T, H, W]


def _extract_face(frame_rgb, mtcnn):
    """Detect and crop face from an RGB numpy frame. Returns (3, 224, 224) in [0, 1].

    Falls back to a full-frame resize if MTCNN is unavailable or detects nothing.
    """
    if mtcnn is not None:
        boxes, _ = mtcnn.detect(Image.fromarray(frame_rgb))
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = [int(round(c)) for c in boxes[0]]
            h, w = frame_rgb.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                face = cv2.resize(frame_rgb[y1:y2, x1:x2], (224, 224))
                return torch.tensor(face, dtype=torch.float32).permute(2, 0, 1) / 255.0

    # No face found — resize full frame so the model still gets visual signal
    resized = cv2.resize(frame_rgb, (224, 224))
    return torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0


def predict(model, audio_tensor, video_tensor, device):
    """Run a single-sample forward pass."""
    audio = audio_tensor.to(device)
    # Replicate the permute + reshape from validation.py
    video = video_tensor.unsqueeze(0).to(device)          # (1, 3, 15, 224, 224)
    video = video.permute(0, 2, 1, 3, 4)                  # (1, 15, 3, 224, 224)
    video = video.reshape(video.shape[0] * N_FRAMES, 3, 224, 224)  # (15, 3, 224, 224)

    with torch.no_grad():
        logits = model(audio, video)                      # (1, 8)

    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return {label: float(p) for label, p in zip(EMOTION_LABELS, probs)}

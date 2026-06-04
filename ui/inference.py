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

from src.engine.checkpointing import load_state_dict_flexible
from src.models.factory import generate_model

EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
SAMPLE_RATE = 22050
TARGET_SAMPLES = int(SAMPLE_RATE * 3.6)  # 79,380
N_FRAMES = 15
N_MELS = 64
N_MFCC = 10
TARGET_SECONDS = 3.6


def _load_run_metadata(run_dir):
    opts_files = sorted(glob.glob(os.path.join(run_dir, "opts*.json")))
    if not opts_files:
        return {}
    try:
        with open(opts_files[-1], "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def checkpoint_metadata(pth_path):
    run_dir = os.path.dirname(os.path.abspath(pth_path))
    metadata = _load_run_metadata(run_dir)
    return {
        "run_dir": run_dir,
        "metadata": metadata,
        "audio_feature": _infer_audio_feature(os.path.basename(run_dir), metadata),
    }


def discover_checkpoints(repo_root):
    """Return discovered checkpoints and saved run metadata under results*/."""
    result = {}
    pattern = os.path.join(repo_root, "results*", "*", "*.pth")
    preferred_names = [
        "RAVDESS_multimodal_cnn_15_best.pth",
        "RAVDESS_multimodalcnn_15_best0.pth",
        "model.pth",
        "RAVDESS_multimodal_cnn_15_checkpoint.pth",
        "RAVDESS_multimodalcnn_15_checkpoint0.pth",
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
                visual_backbone = _metadata_value(metadata, "visual_backbone", "efficientface")
                display = (
                    f"{run_name}  |  feature={audio_feature}  "
                    f"heads={num_heads}  fusion={fusion}  "
                    f"visual={visual_backbone}  lr={lr}"
                )

        result[display] = {
            "path": chosen,
            "run_dir": run_dir,
            "metadata": metadata,
            "audio_feature": audio_feature,
        }
    return result


def _checkpoint_state_keys(pth_path, map_location='cpu'):
    try:
        checkpoint_obj = torch.load(pth_path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint_obj = torch.load(pth_path, map_location=map_location)
    state_dict = checkpoint_obj.get('state_dict', checkpoint_obj) if isinstance(checkpoint_obj, dict) else checkpoint_obj
    if not isinstance(state_dict, dict):
        return []
    return [key.replace('module.', '', 1) for key in state_dict.keys()]


def _infer_visual_backbone_from_checkpoint(pth_path):
    keys = _checkpoint_state_keys(pth_path)
    if any(
        key.startswith('visual_model.attention_reduce.')
        or key.startswith('visual_model.channel_att.')
        or key.startswith('visual_model.spatial_att.')
        for key in keys
    ):
        return 'attention_local'
    if any(
        key.startswith('visual_model.stage2.')
        or key.startswith('visual_model.modulator.')
        for key in keys
    ):
        return 'efficientface'
    return None


def _metadata_value(metadata, key, default):
    value = metadata.get(key, default) if metadata else default
    return value if value not in [None, ""] else default


def load_model(pth_path, num_heads, fusion, device, metadata=None):
    """Load a MultiModalCNN checkpoint.

    Handles both state_dict-only and full-checkpoint dict formats, and strips
    DataParallel 'module.' prefixes automatically.  Uses device='cpu' in the
    opts so generate_model() never wraps in DataParallel (not needed for
    single-sample inference), then moves the bare model to the target device.
    """
    metadata = metadata or {}
    visual_backbone = _metadata_value(
        metadata,
        'visual_backbone',
        _infer_visual_backbone_from_checkpoint(pth_path) or 'efficientface',
    )
    visual_stem_pooling = _metadata_value(metadata, 'visual_stem_pooling', 'maxpool')
    opt = SimpleNamespace(
        model='multimodal_cnn',
        n_classes=8,
        fusion=fusion,
        sample_duration=N_FRAMES,
        num_heads=num_heads,
        pretrain_path='None',  # skip EfficientFace init; we load fine-tuned weights
        device='cpu',          # avoid DataParallel; inference never needs it
        audio_channel_attention=bool(metadata.get('audio_channel_attention', False)),
        visual_backbone=visual_backbone,
        visual_stem_pooling=visual_stem_pooling,
    )
    model, _ = generate_model(opt)
    model = model.to(device)

    load_state_dict_flexible(model, pth_path, map_location=device)
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
    """Build v3-faithful video input for RAVDESS-style inference.

    This mirrors the original preprocessing flow more closely:
    - center-crop the clip to ~3.6 seconds
    - pick 15 distributed frames from that window
    - detect/crop one face per selected frame with MTCNN
    - fall back to full-frame resize if detection fails
    """
    try:
        from facenet_pytorch import MTCNN as _MTCNN
        mtcnn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mtcnn = _MTCNN(image_size=(720, 1280), keep_all=False, device=mtcnn_device)
    except ImportError:
        mtcnn = None  # fall back to full-frame resize

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
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
    else:
        fps = float(fps) if fps and fps > 0 else 30.0
        target_frame_count = int(round(TARGET_SECONDS * fps))
        if len(raw_frames) > target_frame_count:
            skip_begin = (len(raw_frames) - target_frame_count) // 2
            raw_frames = raw_frames[skip_begin:skip_begin + target_frame_count]

    selected_indices = _select_distributed_indices(N_FRAMES, len(raw_frames))
    sampled = [raw_frames[i] for i in selected_indices]

    face_tensors = [_extract_face_v3(frame, mtcnn) for frame in sampled]
    stacked = torch.stack(face_tensors, dim=0)  # (15, 3, 224, 224) [T, C, H, W]
    return stacked.permute(1, 0, 2, 3)          # (3, 15, 224, 224) [C, T, H, W]


def _select_distributed_indices(n_samples, n_frames):
    if n_frames <= 0:
        return [0] * n_samples
    return [min(n_frames - 1, (i * n_frames) // n_samples + n_frames // (2 * n_samples)) for i in range(n_samples)]


def _extract_face_v3(frame_rgb, mtcnn):
    """Match the original facecroppad extraction as closely as practical."""
    if mtcnn is not None:
        boxes, _ = mtcnn.detect(torch.tensor(frame_rgb))
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = [int(round(coord)) for coord in boxes[0]]
            h, w = frame_rgb.shape[:2]
            x1 = max(0, min(w, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h, y1))
            y2 = max(0, min(h, y2))
            if x2 > x1 and y2 > y1:
                frame_rgb = frame_rgb[y1:y2, x1:x2, :]

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

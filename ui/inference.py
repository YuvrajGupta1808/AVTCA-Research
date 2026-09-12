"""Engagement inference for the Streamlit UI.

Runs the EngageNet-trained AVT-CA checkpoints on an arbitrary uploaded video.

The training pipeline consumed fixed 10-second EngageNet clips.  A real
recording is minutes long, so this module slices the upload into 10-second
windows, reproduces the EngageNet preprocessing contract for each window, and
returns a per-window engagement level plus a session-level summary.

Preprocessing contract reproduced here (must stay in sync with
``preprocessing/engagenet/`` and ``src/data/temporal.py``):

* video — frames taken at a fixed 5 fps stride (``extract_faces.py
  --target_fps 5``, ≈50 frames per 10 s clip), each face-cropped with MTCNN
  (Haar fallback, full-frame last resort), resized to 224x224, **kept in
  OpenCV BGR order** because ``extract_faces.py`` saves BGR into the
  ``*_facecroppad.npy`` arrays, scaled to [0, 1], and capped at 96 frames the
  way ``maybe_temporal_subsample`` does it.
* audio — mono 22050 Hz, 64-bin mel spectrogram in dB with ``ref=np.max``.
  Checkpoints trained on ``annotations_engagement_a10.txt`` see the full
  10 s window; checkpoints trained on ``annotations_engagement.txt`` see the
  legacy 3.6 s centre crop.
* decoding — softmax expected class index compared against the calibrated
  ``refined_expected_thresholds`` from the run's ``calibration_results.json``.
  This is the decision rule that produced the reported test accuracy.
"""

import os
import subprocess
import shutil
import sys
import tempfile
from types import SimpleNamespace

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.engine.checkpointing import load_state_dict_flexible
from src.models.factory import generate_model

# EngageNet label order — see preprocessing/engagenet/label_utils.py
ENGAGEMENT_LABELS = ["Not Engaged", "Barely Engaged", "Engaged", "Highly Engaged"]
ENGAGEMENT_SHORT = ["Not", "Barely", "Engaged", "Highly"]
N_CLASSES = 4

SAMPLE_RATE = 22050
N_MELS = 64
MAX_VIDEO_FRAMES = 96          # opts.max_video_frames for every EngageNet run
TARGET_FPS = 5.0               # extract_faces.py --target_fps 5 → ~50 frames per 10 s clip
FRAME_SIZE = 224
WINDOW_SECONDS = 10.0          # EngageNet clip length
MIN_WINDOW_SECONDS = 2.0       # discard a trailing sliver shorter than this


# ── Curated checkpoints ───────────────────────────────────────────────────────
# Only models that beat the published EngageNet video-only baselines are listed.
# Metrics are copied from the matching results/exp2026/*/calibration_results.json
# ("refined_expected_thresholds" block), which is the decoder used below.
MODEL_REGISTRY = {
    "e04_ft10s": {
        "label": "E04 · 10 s audio, ordinal fine-tune  (best — 66.36% test)",
        "checkpoint": "results/exp2026/E04_ft10s_ord_lr5e5_e8/ENGAGENET_multimodal_cnn_15_best.pth",
        "calibration": "results/exp2026/R04_e04_av_fixed/calibration_results.json",
        "num_heads": 8,
        "fusion": "it",
        "audio_features": "mel",
        "audio_seconds": None,          # full 10 s window (annotations_..._a10.txt)
        "thresholds": [1.02, 1.405, 2.04],
        "metrics": {
            "test_top1": 66.36,
            "test_adjacent": 90.96,
            "val_top1": 68.91,
            "test_f1_macro": 52.03,
        },
        "notes": (
            "Fine-tuned on full-length 10 s audio with the ordinal-distance loss. "
            "Best top-1 and best adjacent-accuracy trade-off of every run in this repo."
        ),
    },
    "v12_05": {
        "label": "V12-05 · 3.6 s audio, ordinal fine-tune  (66.13% test)",
        "checkpoint": "results/v12_05_finetune_uniform_best_ord_nosampler_lr0001_e30/model.pth",
        "calibration": "results/exp2026/R01_v12_av_fixed/calibration_results.json",
        "num_heads": 8,
        "fusion": "it",
        "audio_features": "mel",
        "audio_seconds": 3.6,           # legacy centre-cropped audio contract
        "thresholds": [1.22, 1.47, 1.995],
        "metrics": {
            "test_top1": 66.13,
            "test_adjacent": 91.00,
            "val_top1": 68.72,
            "test_f1_macro": 52.32,
        },
        "notes": (
            "Second-best run. Its audio branch only ever saw a 3.6 s centre crop of each "
            "clip, so speech outside that crop is invisible to it."
        ),
    },
}


def mtcnn_available():
    """True when the same face detector used to build the training arrays is importable."""
    try:
        import facenet_pytorch  # noqa: F401
        return True
    except Exception:
        return False


def available_models(repo_root=REPO_ROOT):
    """Registry entries whose checkpoint file actually exists on disk."""
    found = {}
    for key, spec in MODEL_REGISTRY.items():
        path = os.path.join(repo_root, spec["checkpoint"])
        if os.path.isfile(path):
            resolved = dict(spec)
            resolved["key"] = key
            resolved["checkpoint_path"] = path
            found[key] = resolved
    return found


def load_model(spec, device):
    """Build MultiModalCNN with the run's config and load its weights."""
    opt = SimpleNamespace(
        model="multimodal_cnn",
        n_classes=N_CLASSES,
        fusion=spec["fusion"],
        sample_duration=15,
        max_video_frames=MAX_VIDEO_FRAMES,   # seq_length for the visual stem
        num_heads=spec["num_heads"],
        pretrain_path="None",                # fine-tuned weights come from the checkpoint
        device="cpu",                        # never wrap in DataParallel for single-clip use
        audio_channel_attention=False,
        visual_backbone="efficientface",
        visual_stem_pooling="maxpool",
        it_fusion_mode="modern",
        late_text_fusion=False,
        text_vocab_size=4096,
    )
    model, _ = generate_model(opt)
    load_state_dict_flexible(model, spec["checkpoint_path"], map_location="cpu")
    model = model.to(device)
    model.eval()
    model.ablate_modality = "none"
    return model


# ── Video probing and windowing ───────────────────────────────────────────────
def probe_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    fps = float(fps) if fps and fps > 0 else 30.0
    if frame_count <= 0:
        frame_count = _count_frames(video_path)
    return {"fps": fps, "frame_count": frame_count, "duration": frame_count / fps}


def _count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.grab():
        count += 1
    cap.release()
    return count


def plan_windows(duration, window_seconds=WINDOW_SECONDS, hop_seconds=None):
    """Split a clip duration into analysis windows."""
    hop_seconds = hop_seconds or window_seconds
    if duration <= 0:
        return []
    if duration <= window_seconds:
        return [(0.0, duration)]

    windows = []
    start = 0.0
    while start < duration:
        end = min(start + window_seconds, duration)
        if end - start >= MIN_WINDOW_SECONDS or not windows:
            windows.append((start, end))
        if end >= duration:
            break
        start += hop_seconds
    return windows


def _uniform_indices(length, target):
    """Mirror src.data.temporal.maybe_temporal_subsample(mode='uniform')."""
    if length <= 0:
        return []
    if length <= target:
        return list(range(length))
    return torch.linspace(0, length - 1, steps=target).round().long().tolist()


def window_frame_indices(first, last, fps):
    """Frame indices for one window, matching the stored EngageNet arrays.

    ``extract_faces.py`` was run with ``--target_fps 5``, so every training clip
    is a fixed 5 fps stride (≈50 frames per 10 s) rather than a uniform pick of
    96.  Sampling any denser here would hand the visual stem a temporal
    resolution it never saw during training.  The 96-frame cap is the
    ``maybe_temporal_subsample`` step that follows in the collate.
    """
    stride = max(int(round(fps / TARGET_FPS)), 1)
    picks = list(range(first, last, stride)) or [first]
    if len(picks) > MAX_VIDEO_FRAMES:
        picks = [picks[i] for i in _uniform_indices(len(picks), MAX_VIDEO_FRAMES)]
    return picks


# ── Face cropping ─────────────────────────────────────────────────────────────
class FaceCropper:
    """MTCNN face crop with Haar fallback, matching extract_faces.py.

    ``redetect_every`` > 1 reuses the previous bounding box for intermediate
    frames.  Detection is the dominant cost on CPU; reuse trades a little
    fidelity for a large speedup on long recordings.
    """

    def __init__(self, device="cpu", redetect_every=1):
        self.redetect_every = max(int(redetect_every), 1)
        self.detector = None
        self.device = device
        self._seen = 0
        self._box = None
        try:
            from facenet_pytorch import MTCNN
            self.detector = MTCNN(image_size=(720, 1280), device=torch.device(device))
        except Exception:
            self.detector = None

        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(cascade_path)
        self.cascade = None if cascade.empty() else cascade

    @property
    def backend(self):
        if self.detector is not None:
            return "MTCNN"
        if self.cascade is not None:
            return "Haar cascade"
        return "no detector (full frame)"

    def reset(self):
        self._seen = 0
        self._box = None

    def _detect(self, frame_bgr):
        if self.detector is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(frame_rgb).to(self.device)
            boxes, _ = self.detector.detect(tensor)
            if boxes is not None and len(boxes) > 0:
                return boxes[0]
        if self.cascade is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            found = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            if len(found) > 0:
                x, y, w, h = found[0]
                return np.array([x, y, x + w, y + h], dtype=np.float32)
        return None

    def crop(self, frame_bgr):
        if self._seen % self.redetect_every == 0:
            box = self._detect(frame_bgr)
            if box is not None:
                self._box = box
        self._seen += 1

        box, found = self._box, self._box is not None
        if found:
            h, w = frame_bgr.shape[:2]
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))
            frame_bgr = frame_bgr[y1:y2, x1:x2]
        # BGR is intentional — the training arrays were saved straight from cv2.
        return cv2.resize(frame_bgr, (FRAME_SIZE, FRAME_SIZE)), found


def collect_window_frames(video_path, windows, fps, cropper, progress_cb=None):
    """One sequential decode pass; returns {window_index: (T, 224, 224, 3) uint8}."""
    wanted = {}          # global frame index -> list of (window_index, slot)
    per_window_counts = {}
    for w_idx, (start, end) in enumerate(windows):
        first = int(round(start * fps))
        last = max(int(round(end * fps)), first + 1)
        picks = window_frame_indices(first, last, fps)
        per_window_counts[w_idx] = len(picks)
        for slot, frame_idx in enumerate(picks):
            wanted.setdefault(frame_idx, []).append((w_idx, slot))

    frames = {w_idx: [None] * count for w_idx, count in per_window_counts.items()}
    face_hits = {w_idx: 0 for w_idx in per_window_counts}

    cap = cv2.VideoCapture(video_path)
    cropper.reset()
    current = 0
    processed = 0
    total = len(wanted)
    while True:
        ok = cap.grab()
        if not ok:
            break
        if current in wanted:
            ok, frame = cap.retrieve()
            if ok:
                cropped, found = cropper.crop(frame)
                for w_idx, slot in wanted[current]:
                    frames[w_idx][slot] = cropped
                    face_hits[w_idx] += int(found)
            processed += 1
            if progress_cb and total:
                progress_cb(processed / total)
        current += 1
    cap.release()

    output = {}
    face_rates = {}
    for w_idx, slots in frames.items():
        filled = [f for f in slots if f is not None]
        if not filled:
            filled = [np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)]
        output[w_idx] = np.asarray(filled, dtype=np.uint8)
        face_rates[w_idx] = face_hits[w_idx] / max(len(filled), 1)
    return output, face_rates


def video_tensor(frames_bgr):
    """(T, H, W, 3) uint8 BGR  ->  (T, 3, 224, 224) float in [0, 1]."""
    tensor = torch.from_numpy(frames_bgr.astype(np.float32) / 255.0)
    return tensor.permute(0, 3, 1, 2).contiguous()


# ── Audio ─────────────────────────────────────────────────────────────────────
def _resolve_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise FileNotFoundError("ffmpeg executable not available") from exc


def extract_waveform(video_path):
    """Mono 22050 Hz waveform for the whole file; silence if there is no track."""
    ffmpeg = _resolve_ffmpeg()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        tmp_wav = handle.name
    try:
        subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", video_path,
             "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "wav", tmp_wav],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        waveform, _ = librosa.load(tmp_wav, sr=SAMPLE_RATE)
        return waveform, True
    except subprocess.CalledProcessError:
        return np.zeros(0, dtype=np.float32), False
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


def window_audio_features(waveform, start, end, audio_seconds=None):
    """Mel-dB features for one window, following the run's audio contract."""
    first = int(round(start * SAMPLE_RATE))
    last = int(round(end * SAMPLE_RATE))
    segment = waveform[first:last] if waveform.size else np.zeros(0, dtype=np.float32)

    if audio_seconds:
        target = int(SAMPLE_RATE * audio_seconds)
        if len(segment) < target:
            segment = np.pad(segment, (0, target - len(segment)))
        else:                                     # centre crop, as load_audio() does
            excess = len(segment) - target
            segment = segment[excess // 2: len(segment) - (excess - excess // 2)]
    elif len(segment) < int(SAMPLE_RATE * MIN_WINDOW_SECONDS):
        segment = np.pad(segment, (0, int(SAMPLE_RATE * MIN_WINDOW_SECONDS) - len(segment)))

    mel = librosa.feature.melspectrogram(y=segment.astype(np.float32), sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return torch.from_numpy(mel_db.astype(np.float32)), float(np.sqrt(np.mean(segment ** 2)) if segment.size else 0.0)


# ── Forward pass and decoding ─────────────────────────────────────────────────
def forward_window(model, audio_features, video_frames, device, ablate="none"):
    """Single-clip forward pass; mirrors the batch-of-one collate output."""
    audio = audio_features.unsqueeze(0).to(device)                 # (1, 64, T_a)
    video = video_frames.unsqueeze(0).to(device)                   # (1, T_v, 3, 224, 224)
    audio_len = torch.tensor([audio.shape[-1]], dtype=torch.long, device=device)
    video_len = torch.tensor([video.shape[1]], dtype=torch.long, device=device)
    audio_mask = torch.ones(1, audio.shape[-1], dtype=torch.bool, device=device)
    video_mask = torch.ones(1, video.shape[1], dtype=torch.bool, device=device)

    previous = getattr(model, "ablate_modality", "none")
    model.ablate_modality = ablate
    try:
        with torch.no_grad():
            logits = model(
                audio, video,
                audio_mask=audio_mask, video_mask=video_mask,
                audio_lengths=audio_len, video_lengths=video_len,
            )
    finally:
        model.ablate_modality = previous
    return logits.squeeze(0).float().cpu().numpy()


def expected_score(probs):
    """Softmax expected class index — the ordinal score the calibration fits."""
    return float(np.dot(probs, np.arange(len(probs), dtype=np.float32)))


def decode(logits, thresholds):
    probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
    score = expected_score(probs)
    level = int(np.digitize([score], np.asarray(thresholds, dtype=np.float32))[0])
    return {
        "probs": probs,
        "score": score,
        "level": level,
        "label": ENGAGEMENT_LABELS[level],
        "argmax_level": int(np.argmax(probs)),
        "confidence": float(probs.max()),
    }


def analyze_video(
    video_path,
    model,
    spec,
    device,
    window_seconds=WINDOW_SECONDS,
    hop_seconds=None,
    redetect_every=1,
    compute_modality_split=False,
    progress_cb=None,
):
    """Full pipeline: window -> preprocess -> predict -> decode.

    ``progress_cb(fraction, message)`` is optional and used by the UI.
    """
    def report(fraction, message):
        if progress_cb:
            progress_cb(max(0.0, min(1.0, fraction)), message)

    info = probe_video(video_path)
    windows = plan_windows(info["duration"], window_seconds, hop_seconds)
    if not windows:
        raise ValueError("Video appears to contain no frames.")

    report(0.02, "Extracting audio…")
    waveform, has_audio = extract_waveform(video_path)

    cropper = FaceCropper(device=device if device == "cuda" else "cpu", redetect_every=redetect_every)
    report(0.05, f"Detecting faces with {cropper.backend}…")
    frames_by_window, face_rates = collect_window_frames(
        video_path, windows, info["fps"], cropper,
        progress_cb=lambda frac: report(0.05 + 0.65 * frac, "Detecting and cropping faces…"),
    )

    results = []
    for w_idx, (start, end) in enumerate(windows):
        report(0.70 + 0.30 * (w_idx / len(windows)),
               f"Scoring window {w_idx + 1}/{len(windows)}…")
        audio_features, rms = window_audio_features(waveform, start, end, spec.get("audio_seconds"))
        frames = video_tensor(frames_by_window[w_idx])

        logits = forward_window(model, audio_features, frames, device)
        decoded = decode(logits, spec["thresholds"])
        entry = {
            "index": w_idx,
            "start": start,
            "end": end,
            "n_frames": int(frames.shape[0]),
            "face_rate": face_rates[w_idx],
            "audio_rms": rms,
            **decoded,
        }

        if compute_modality_split:
            audio_only = decode(
                forward_window(model, audio_features, frames, device, ablate="audio_only"),
                spec["thresholds"],
            )
            video_only = decode(
                forward_window(model, audio_features, frames, device, ablate="video_only"),
                spec["thresholds"],
            )
            entry["audio_only"] = audio_only
            entry["video_only"] = video_only

        results.append(entry)

    report(1.0, "Done")
    return {
        "windows": results,
        "video_info": info,
        "has_audio": bool(has_audio and waveform.size),
        "face_backend": cropper.backend,
        "summary": summarize(results, spec["thresholds"]),
        "thresholds": spec["thresholds"],
    }


def summarize(results, thresholds):
    """Session-level rollup across windows, decoded with the same thresholds."""
    if not results:
        return {}
    scores = np.array([r["score"] for r in results], dtype=np.float32)
    levels = np.array([r["level"] for r in results], dtype=np.int64)
    durations = np.array([r["end"] - r["start"] for r in results], dtype=np.float32)
    total = float(durations.sum()) or 1.0

    time_per_level = {
        ENGAGEMENT_LABELS[i]: float(durations[levels == i].sum()) for i in range(N_CLASSES)
    }
    mean_score = float(scores.mean())
    overall_level = int(np.digitize([mean_score], np.asarray(thresholds, dtype=np.float32))[0])

    return {
        "mean_score": mean_score,
        "overall_level": overall_level,
        "overall_label": ENGAGEMENT_LABELS[overall_level],
        "engaged_fraction": float(durations[levels >= 2].sum() / total),
        "disengaged_fraction": float(durations[levels <= 1].sum() / total),
        "time_per_level": time_per_level,
        "total_seconds": total,
        "n_windows": len(results),
        "score_std": float(scores.std()),
        "lowest": min(results, key=lambda r: r["score"]),
        "highest": max(results, key=lambda r: r["score"]),
    }

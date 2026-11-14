#!/usr/bin/env python3
"""Extract face-focused frame tensors for EngageNet clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    from facenet_pytorch import MTCNN  # type: ignore
except ImportError:  # pragma: no cover
    MTCNN = None


RAW_SPLITS = ("Train", "Validation", "Test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("datasets/EngageNet"))
    parser.add_argument("--splits", nargs="+", choices=RAW_SPLITS, default=list(RAW_SPLITS))
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--max_video_seconds", default=0.0, type=float)
    parser.add_argument("--max_frames", default=0, type=int)
    parser.add_argument("--target_fps", default=0.0, type=float)
    parser.add_argument("--frame_stride", default=1, type=int)
    parser.add_argument("--output_size", default=224, type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow_opencv_fallback", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def iter_video_files(data_root: Path, splits: list[str]) -> list[Path]:
    files: list[Path] = []
    for split_name in splits:
        split_dir = data_root / split_name
        if split_dir.exists():
            files.extend(sorted(split_dir.rglob("*.mp4")))
    return files


def select_shard(files: list[Path], num_shards: int, shard_index: int) -> list[Path]:
    if num_shards <= 1:
        return files
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards - 1}], got {shard_index}")
    return files[shard_index::num_shards]


def select_distributed(count: int, total: int) -> list[int]:
    if total <= 0:
        return []
    if count <= 0 or total <= count:
        return list(range(total))
    return np.linspace(0, total - 1, num=count, dtype=int).tolist()


def crop_face_or_resize(frame_bgr: np.ndarray, detector, output_size: int, device: torch.device, face_cascade) -> np.ndarray:
    boxes = None
    if detector is not None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).to(device)
        boxes, _ = detector.detect(tensor)

    if boxes is None and face_cascade is not None and not face_cascade.empty():
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        if len(detections) > 0:
            x, y, w, h = detections[0]
            boxes = np.array([[x, y, x + w, y + h]], dtype=np.float32)

    if boxes is not None and len(boxes) > 0:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in boxes[0]]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        frame_bgr = frame_bgr[y1:y2, x1:x2]
    return cv2.resize(frame_bgr, (output_size, output_size))


def extract_clip_faces(
    video_path: Path,
    detector,
    max_frames: int,
    max_video_seconds: float,
    target_fps: float,
    frame_stride: int,
    output_size: int,
    device: torch.device,
    face_cascade,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_window_frames = frame_count
    if max_video_seconds and max_video_seconds > 0:
        max_window_frames = min(frame_count, int(round(max_video_seconds * fps)))
    start_frame = 0
    effective_stride = max(int(frame_stride), 1)
    raw_selected = list(range(start_frame, start_frame + max(max_window_frames, 1), effective_stride))
    if target_fps and target_fps > 0:
        effective_stride = max(int(round(fps / target_fps)), 1)
        raw_selected = list(range(start_frame, start_frame + max(max_window_frames, 1), effective_stride))
    raw_selected = [idx for idx in raw_selected if idx < frame_count]
    if max_frames and max_frames > 0:
        keep_indices = select_distributed(max_frames, len(raw_selected))
        raw_selected = [raw_selected[idx] for idx in keep_indices]
    selected_set = set(raw_selected)
    frames: list[np.ndarray] = []
    current = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if current in selected_set:
            processed = crop_face_or_resize(frame, detector, output_size, device, face_cascade)
            frames.append(processed)
        current += 1
    cap.release()

    if not frames:
        frames = [np.zeros((output_size, output_size, 3), dtype=np.uint8)]

    return np.asarray(frames, dtype=np.uint8)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if MTCNN is None and not args.allow_opencv_fallback:
        raise RuntimeError(
            "facenet_pytorch is required for EngageNet face extraction. "
            "Activate the correct environment or rerun with --allow_opencv_fallback."
        )
    detector = MTCNN(image_size=(720, 1280), device=device) if MTCNN is not None else None
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))

    files = iter_video_files(data_root, args.splits)
    files = select_shard(files, args.num_shards, args.shard_index)
    if args.limit > 0:
        files = files[:args.limit]

    for video_path in tqdm(files, desc="Extracting EngageNet face clips"):
        target_path = video_path.with_name(video_path.stem + "_facecroppad.npy")
        if target_path.exists() and not args.force:
            continue
        clip = extract_clip_faces(
            video_path,
            detector=detector,
            max_frames=args.max_frames,
            max_video_seconds=args.max_video_seconds,
            target_fps=args.target_fps,
            frame_stride=args.frame_stride,
            output_size=args.output_size,
            device=device,
            face_cascade=face_cascade,
        )
        np.save(target_path, clip)


if __name__ == "__main__":
    main()

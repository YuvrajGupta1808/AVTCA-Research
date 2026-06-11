#!/usr/bin/env python3
"""Extract face-focused frame tensors for DAiSEE clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    from facenet_pytorch import MTCNN  # type: ignore
except ImportError:  # pragma: no cover - optional dependency in some envs
    MTCNN = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("datasets/DAISEE"),
        help="DAiSEE root directory containing DataSet/.",
    )
    parser.add_argument(
        "--target_time",
        default=3.6,
        type=float,
        help="Centered temporal window to keep before frame sampling.",
    )
    parser.add_argument(
        "--save_frames",
        default=15,
        type=int,
        help="Number of frames to save per clip.",
    )
    parser.add_argument(
        "--output_size",
        default=224,
        type=int,
        help="Output frame height and width.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite existing *_facecroppad.npy files.",
    )
    parser.add_argument(
        "--allow_opencv_fallback",
        action="store_true",
        help="Use OpenCV Haar face detection if facenet_pytorch is unavailable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most this many videos for a smoke run. 0 means all files.",
    )
    return parser.parse_args()


def iter_video_files(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "DataSet"
    files = [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".avi", ".mp4"}
    ]
    return sorted(files)


def select_distributed(count: int, total: int) -> list[int]:
    if total <= 0:
        return []
    if total <= count:
        return list(range(total))
    return [i * total // count + total // (2 * count) for i in range(count)]


def crop_face_or_resize(
    frame_bgr: np.ndarray,
    detector,
    output_size: int,
    device: torch.device,
    face_cascade: cv2.CascadeClassifier | None,
) -> np.ndarray:
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
    frame_bgr = cv2.resize(frame_bgr, (output_size, output_size))
    return frame_bgr


def extract_clip_faces(video_path: Path, detector, save_frames: int, target_time: float, output_size: int, device: torch.device, face_cascade) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    window_frames = max(save_frames, int(round(target_time * fps)))
    if frame_count > window_frames:
        start_frame = max(0, (frame_count - window_frames) // 2)
    else:
        start_frame = 0
        window_frames = frame_count

    selected = select_distributed(save_frames, max(window_frames, 1))
    selected_set = set(start_frame + idx for idx in selected)
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
        frames = [np.zeros((output_size, output_size, 3), dtype=np.uint8) for _ in range(save_frames)]
    elif len(frames) < save_frames:
        filler = frames[-1]
        frames.extend([filler.copy() for _ in range(save_frames - len(frames))])

    return np.asarray(frames[:save_frames], dtype=np.uint8)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if MTCNN is None and not args.allow_opencv_fallback:
        raise RuntimeError(
            "facenet_pytorch is required for DAiSEE face extraction. "
            "Activate the correct environment or rerun with --allow_opencv_fallback."
        )
    detector = MTCNN(image_size=(720, 1280), device=device) if MTCNN is not None else None
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))

    files = iter_video_files(data_root)
    if args.limit > 0:
        files = files[:args.limit]

    for video_path in tqdm(files, desc="Extracting DAiSEE face clips"):
        target_path = video_path.with_name(video_path.stem + "_facecroppad.npy")
        if target_path.exists() and not args.force:
            continue
        clip = extract_clip_faces(
            video_path,
            detector=detector,
            save_frames=args.save_frames,
            target_time=args.target_time,
            output_size=args.output_size,
            device=device,
            face_cascade=face_cascade,
        )
        np.save(target_path, clip)


if __name__ == "__main__":
    main()

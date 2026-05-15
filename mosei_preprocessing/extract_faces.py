#!/usr/bin/env python3
"""Extract 224x224 face-centered frames from CMU-MOSEI segment clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


SAVE_FRAMES = 15
OUTPUT_SIZE = 224


def select_distributed(m: int, n: int) -> list[int]:
    return [i * n // m + n // (2 * m) for i in range(m)]


def detect_face_box(frame: np.ndarray, cascade: cv2.CascadeClassifier) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    return int(x), int(y), int(w), int(h)


def center_square_crop(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    side = min(height, width)
    x1 = (width - side) // 2
    y1 = (height - side) // 2
    return frame[y1 : y1 + side, x1 : x1 + side]


def crop_face(frame: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    if box is None:
        return center_square_crop(frame)

    x, y, w, h = box
    cx = x + w // 2
    cy = y + h // 2
    side = int(max(w, h) * 1.4)
    half = side // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(frame.shape[1], cx + half)
    y2 = min(frame.shape[0], cy + half)

    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return center_square_crop(frame)
    return cropped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segment_dir",
        type=Path,
        default=Path("CMU-MOSEI/raw_segments"),
    )
    args = parser.parse_args()

    segment_dir = args.segment_dir.resolve()
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for clip_path in sorted(segment_dir.rglob("*.mp4")):
        capture = cv2.VideoCapture(str(clip_path))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frames_to_select = set(select_distributed(SAVE_FRAMES, max(total_frames, SAVE_FRAMES)))

        selected_frames: list[np.ndarray] = []
        frame_index = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            if frame_index in frames_to_select:
                box = detect_face_box(frame, cascade)
                cropped = crop_face(frame, box)
                resized = cv2.resize(cropped, (OUTPUT_SIZE, OUTPUT_SIZE))
                selected_frames.append(resized)
            frame_index += 1

        capture.release()

        while len(selected_frames) < SAVE_FRAMES:
            selected_frames.append(np.zeros((OUTPUT_SIZE, OUTPUT_SIZE, 3), dtype=np.uint8))

        output_path = clip_path.with_name(clip_path.stem + "_facecrop.npy")
        np.save(output_path, np.asarray(selected_frames[:SAVE_FRAMES], dtype=np.uint8))

    print(f"Extracted face crops under {segment_dir}")


if __name__ == "__main__":
    main()

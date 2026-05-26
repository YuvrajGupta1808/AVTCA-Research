# -*- coding: utf-8 -*-
"""
Extract face crops from CREMA-D FLV videos.

Expected layout:
  <data_root>/
    VideoFlash/
      1001_IEO_ANG_HI.flv
      ...

For each .flv, writes:
  <stem>_facecroppad.npy   — (15, 224, 224, 3) uint8 array
  <stem>_facecroppad.avi   — MJPG AVI at reduced fps (optional, save_avi=True)
"""

import argparse
import os

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=(720, 1280), device=device)

SAVE_FRAMES = 15
INPUT_FPS = 30
SAVE_LENGTH = 3.6  # seconds
SAVE_AVI = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        default=os.environ.get('CREMAD_ROOT', 'datasets/CREMAD'),
        help='Root directory containing VideoFlash/',
    )
    return parser.parse_args()


def select_distributed(m, n):
    return [i * n // m + n // (2 * m) for i in range(m)]


def count_frames(cap):
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    return n


def extract_video(video_path, out_npy, out_avi):
    cap = cv2.VideoCapture(video_path)
    framen = count_frames(cap)
    cap = cv2.VideoCapture(video_path)

    if SAVE_LENGTH * INPUT_FPS < framen:
        skip_begin = int((framen - SAVE_LENGTH * INPUT_FPS) // 2)
        for _ in range(skip_begin):
            cap.read()
        framen = int(SAVE_LENGTH * INPUT_FPS)

    frames_to_select = select_distributed(SAVE_FRAMES, framen)
    save_fps = max(1, SAVE_FRAMES // max(1, framen // INPUT_FPS))

    writer = None
    if SAVE_AVI:
        writer = cv2.VideoWriter(
            out_avi, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            save_fps, (224, 224)
        )

    numpy_video = []
    frame_ctr = 0
    failed = False

    while True:
        ret, im = cap.read()
        if not ret:
            break
        if frame_ctr not in frames_to_select:
            frame_ctr += 1
            continue

        frames_to_select.remove(frame_ctr)
        frame_ctr += 1

        try:
            cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        except Exception:
            failed = True
            break

        # BGR → RGB for MTCNN
        im_rgb = im[:, :, ::-1].copy()
        im_tensor = torch.tensor(im_rgb).to(device)

        bbox = mtcnn.detect(im_tensor)
        if bbox[0] is not None:
            x1, y1, x2, y2 = [round(x) for x in bbox[0][0]]
            x1, y1 = max(0, x1), max(0, y1)
            im = im[y1:y2, x1:x2, :]

        im = cv2.resize(im, (224, 224))
        if writer is not None:
            writer.write(im)
        numpy_video.append(im)

    cap.release()

    # Pad with blank frames if fewer than SAVE_FRAMES were captured
    blank = np.zeros((224, 224, 3), dtype=np.uint8)
    while len(numpy_video) < SAVE_FRAMES:
        if writer is not None:
            writer.write(blank)
        numpy_video.append(blank)

    if writer is not None:
        writer.release()

    np.save(out_npy, np.array(numpy_video))

    if len(numpy_video) != SAVE_FRAMES or failed:
        return False
    return True


def main():
    args = parse_args()
    root = os.path.abspath(os.path.expanduser(args.data_root))
    video_dir = os.path.join(root, 'VideoFlash')

    if not os.path.isdir(video_dir):
        raise FileNotFoundError(
            f'VideoFlash/ not found under {root}. '
            'Pass --data_root pointing to the CREMA-D root.'
        )

    flv_files = sorted(f for f in os.listdir(video_dir) if f.endswith('.flv'))
    failed_videos = []

    for fname in tqdm(flv_files):
        stem = fname[:-4]
        out_npy = os.path.join(video_dir, stem + '_facecroppad.npy')
        out_avi = os.path.join(video_dir, stem + '_facecroppad.avi')

        if os.path.isfile(out_npy):
            continue

        ok = extract_video(os.path.join(video_dir, fname), out_npy, out_avi)
        if not ok:
            failed_videos.append(fname)

    if failed_videos:
        print(f'\nFailed ({len(failed_videos)}):')
        for v in failed_videos:
            print(' ', v)
    else:
        print('All videos processed successfully.')


if __name__ == '__main__':
    main()

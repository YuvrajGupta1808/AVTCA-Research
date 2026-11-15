# -*- coding: utf-8 -*-
"""RAVDESS face clips (legacy emotion pipeline): 15 frames from the centred 3.6 s of each video.

For every <video>.mp4 under <data_root>/<actor>/, writes <video>_facecroppad.npy (15 x 224 x 224 x 3,
BGR uint8: the first MTCNN face box, clamped to the frame, or the whole frame when no face is found;
missing frames are zero) and, by default, a matching MJPG .avi for inspection.
"""
import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

SAVE_FRAMES = 15
INPUT_FPS = 30
SAVE_LENGTH = 3.6  # seconds
HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data_root',
        default=os.environ.get('RAVDESS_ROOT', 'datasets/RAVDESS'),
        help='Directory containing ACTORxx folders',
    )
    parser.add_argument('--no_avi', action='store_true', help='do not write the inspection .avi files')
    return parser.parse_args()


def select_distributed(m, n):
    return [i * n // m + n // (2 * m) for i in range(m)]


def clamp_box(box, height, width):
    """Round a detector box and clamp it to the frame; None if nothing of it is inside the frame."""
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, y1 = max(0, min(width - 1, x1)), max(0, min(height - 1, y1))
    x2, y2 = max(x1 + 1, min(width, x2)), max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def process_video(path, mtcnn, device, save_avi):
    cap = cv2.VideoCapture(path)
    framen = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        framen += 1
    cap = cv2.VideoCapture(path)

    if SAVE_LENGTH * INPUT_FPS < framen:
        skip_begin = int((framen - (SAVE_LENGTH * INPUT_FPS)) // 2)
        for _ in range(skip_begin):
            cap.read()
        framen = int(SAVE_LENGTH * INPUT_FPS)

    frames_to_select = select_distributed(SAVE_FRAMES, framen)
    seconds = framen // INPUT_FPS
    save_fps = SAVE_FRAMES // seconds if seconds > 0 else SAVE_FRAMES  # clips under 1 s used to divide by zero
    out = None
    if save_avi:
        out = cv2.VideoWriter(path[:-4] + '_facecroppad.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), save_fps, (224, 224))

    numpy_video, frame_ctr, failed = [], 0, False
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
        im_rgb = torch.tensor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).to(device)
        bbox = mtcnn.detect(im_rgb)
        if bbox[0] is not None:
            x1, y1, x2, y2 = clamp_box(bbox[0][0], im.shape[0], im.shape[1])
            im = im[y1:y2, x1:x2, :]
        im = cv2.resize(im, (224, 224))
        if out is not None:
            out.write(im)
        numpy_video.append(im)
    for _ in range(len(frames_to_select)):
        if out is not None:
            out.write(np.zeros((224, 224, 3), dtype=np.uint8))
        numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))
    if out is not None:
        out.release()
    np.save(path[:-4] + '_facecroppad.npy', np.array(numpy_video))
    return failed, len(numpy_video)


def main():
    from facenet_pytorch import MTCNN

    args = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(image_size=(720, 1280), device=device)
    root = os.path.abspath(os.path.expanduser(args.data_root))
    failed_videos = []
    for sess in tqdm(sorted(os.listdir(root))):
        sess_dir = os.path.join(root, sess)
        if not os.path.isdir(sess_dir):
            continue
        for filename in sorted(os.listdir(sess_dir)):
            if not filename.endswith('.mp4'):
                continue
            failed, n = process_video(os.path.join(sess_dir, filename), mtcnn, device, not args.no_avi)
            if failed:
                failed_videos.append((sess, filename))
            if n != SAVE_FRAMES:
                print('Error', sess, filename)
        with open(os.path.join(HERE, 'processed.txt'), 'a') as f:
            f.write(sess + '\n')
    print(failed_videos)


if __name__ == '__main__':
    main()

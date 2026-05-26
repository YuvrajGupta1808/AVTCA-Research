# -*- coding: utf-8 -*-
"""
Extract and crop/pad audio directly from CREMA-D FLV video files.

Expected layout:
  <data_root>/
    VideoFlash/
      1001_IEO_ANG_HI.flv
      ...

Writes <stem>_croppad.wav alongside each source FLV (no AudioWAV/ needed).
librosa reads audio from FLV via ffmpeg — ffmpeg must be on PATH.
"""

import argparse
import os

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        default=os.environ.get('CREMAD_ROOT', 'datasets/CREMAD'),
        help='Root directory containing VideoFlash/',
    )
    parser.add_argument('--target_time', default=3.6, type=float,
                        help='Target clip duration in seconds')
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(os.path.expanduser(args.data_root))
    video_dir = os.path.join(root, 'VideoFlash')
    target_length_sec = args.target_time

    if not os.path.isdir(video_dir):
        raise FileNotFoundError(
            f'VideoFlash/ not found under {root}. '
            'Pass --data_root pointing to the CREMA-D root.'
        )

    flv_files = sorted(f for f in os.listdir(video_dir) if f.endswith('.flv'))

    for flv_file in tqdm(flv_files):
        out_path = os.path.join(video_dir, flv_file[:-4] + '_croppad.wav')
        if os.path.isfile(out_path):
            continue

        in_path = os.path.join(video_dir, flv_file)
        y, sr = librosa.core.load(in_path, sr=22050)
        target_length = int(sr * target_length_sec)

        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            remain = len(y) - target_length
            y = y[remain // 2: len(y) - (remain - remain // 2)]

        sf.write(out_path, y, sr)

    print(f'Done. Processed {len(flv_files)} files from {video_dir}.')


if __name__ == '__main__':
    main()

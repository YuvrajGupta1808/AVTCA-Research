# -*- coding: utf-8 -*-

import argparse
import librosa
import os
import soundfile as sf
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        default=os.environ.get('RAVDESS_ROOT', 'datasets/RAVDESS'),
        help='Directory containing ACTORxx folders',
    )
    parser.add_argument('--target_time', default=3.6, type=float, help='Target clip duration in seconds')
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(os.path.expanduser(args.data_root))
    target_time = args.target_time

    for actor in sorted(os.listdir(root)):
        actor_dir = os.path.join(root, actor)
        if not os.path.isdir(actor_dir):
            continue

        for audiofile in os.listdir(actor_dir):
            if not audiofile.endswith('.wav') or 'croppad' in audiofile:
                continue

            y, sr = librosa.core.load(os.path.join(actor_dir, audiofile), sr=22050)
            target_length = int(sr * target_time)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)))
            else:
                remain = len(y) - target_length
                y = y[remain // 2:len(y) - (remain - remain // 2)]

            sf.write(os.path.join(actor_dir, audiofile[:-4] + '_croppad.wav'), y, sr)


if __name__ == '__main__':
    main()

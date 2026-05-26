# -*- coding: utf-8 -*-
"""
Build annotations.txt for CREMA-D from the collected repo data.

File naming: <ActorID>_<Sentence>_<Emotion>_<Level>.flv
  Emotions: ANG DIS FEA HAP NEU SAD -> labels 1-6

Actor split (91 actors, sorted by ID):
  test  : first 13  (indices  0-12)
  val   : next  13  (indices 13-25)
  train : rest  65  (indices 26-90)

Audio is expected as <stem>_croppad.wav in VideoFlash/.
Video prefers <stem>_facecroppad.npy when present, otherwise falls back to
the raw <stem>.flv so the dataset remains usable before face extraction.

Output line format:
  <video_path>;<audio_croppad_wav_path>;<label>;<split>
"""

import argparse
import os
from collections import Counter


EMOTION_MAP = {
    'ANG': 1,
    'DIS': 2,
    'FEA': 3,
    'HAP': 4,
    'NEU': 5,
    'SAD': 6,
}

N_TEST = 13
N_VAL = 13


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        default=os.environ.get('CREMAD_ROOT', 'datasets/CREMAD'),
        help='Root directory containing VideoFlash/',
    )
    parser.add_argument(
        '--annotation_file',
        default='preprocessing/cremad/annotations.txt',
        help='Output annotation file path',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(os.path.expanduser(args.data_root))
    video_dir = os.path.join(root, 'VideoFlash')
    annotation_file = os.path.abspath(args.annotation_file)

    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f'VideoFlash/ not found under {root}')

    # Use raw FLVs as the source of truth; face-crop NPYs are optional.
    flv_files = sorted(
        f for f in os.listdir(video_dir)
        if f.endswith('.flv')
    )

    # Derive unique actor IDs and build sorted list for deterministic split
    actor_ids = sorted({f.split('_')[0] for f in flv_files})

    test_actors = set(actor_ids[:N_TEST])
    val_actors = set(actor_ids[N_TEST: N_TEST + N_VAL])

    skipped = 0
    split_counts = Counter()
    emotion_counts = Counter()
    with open(annotation_file, 'w') as ann:
        for flv_file in flv_files:
            stem = flv_file[:-4]
            parts = stem.split('_')
            # parts: [ActorID, Sentence, Emotion, Level]
            if len(parts) != 4:
                skipped += 1
                continue

            actor_id, _, emotion_code, _ = parts
            if emotion_code not in EMOTION_MAP:
                skipped += 1
                continue

            label = EMOTION_MAP[emotion_code]
            audio_path = os.path.join(video_dir, stem + '_croppad.wav')
            npy_path = os.path.join(video_dir, stem + '_facecroppad.npy')
            flv_path = os.path.join(video_dir, flv_file)

            if not os.path.isfile(audio_path):
                skipped += 1
                continue

            video_path = npy_path if os.path.isfile(npy_path) else flv_path

            if actor_id in test_actors:
                split = 'testing'
            elif actor_id in val_actors:
                split = 'validation'
            else:
                split = 'training'

            ann.write(f'{video_path};{audio_path};{label};{split}\n')
            split_counts[split] += 1
            emotion_counts[emotion_code] += 1

    total = len(flv_files) - skipped
    print(f'Wrote {total} entries to {annotation_file} ({skipped} skipped).')
    print(f'  test actors : {N_TEST}')
    print(f'  val actors  : {N_VAL}')
    print(f'  train actors: {len(actor_ids) - N_TEST - N_VAL}')
    print(f'  split counts: {dict(split_counts)}')
    print(f'  emotion counts: {dict(sorted(emotion_counts.items()))}')


if __name__ == '__main__':
    main()

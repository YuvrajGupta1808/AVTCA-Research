# -*- coding: utf-8 -*-

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        default=os.environ.get('RAVDESS_ROOT', 'RAVDESS'),
        help='Directory containing ACTORxx folders',
    )
    parser.add_argument(
        '--annotation_file',
        default='annotations.txt',
        help='Output annotation filename',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(os.path.expanduser(args.data_root))
    annotation_file = os.path.abspath(args.annotation_file)

    n_folds = 1
    folds = [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]]
    actors = sorted([actor for actor in os.listdir(root) if os.path.isdir(os.path.join(root, actor))])

    with open(annotation_file, 'w') as _:
        pass

    for fold in range(n_folds):
        test_ids, val_ids, train_ids = folds[fold]

        for i, actor in enumerate(actors):
            actor_dir = os.path.join(root, actor)
            for video in os.listdir(actor_dir):
                if not video.endswith('.npy') or 'croppad' not in video:
                    continue
                label = str(int(video.split('-')[2]))
                audio = '01' + video.split('_face')[0][2:] + '_croppad.wav'

                if i in train_ids:
                    split = 'training'
                elif i in val_ids:
                    split = 'validation'
                else:
                    split = 'testing'

                with open(annotation_file, 'a') as f:
                    f.write(
                        os.path.join(root, actor, video)
                        + ';'
                        + os.path.join(root, actor, audio)
                        + ';'
                        + label
                        + ';'
                        + split
                        + '\n'
                    )


if __name__ == '__main__':
    main()

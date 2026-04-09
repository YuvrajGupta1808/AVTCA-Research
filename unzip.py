import argparse
import os
import zipfile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root',
        default=os.environ.get('RAVDESS_ROOT', 'RAVDESS'),
        help='Directory containing the downloaded RAVDESS zip files',
    )
    parser.add_argument(
        '--delete_zip',
        action='store_true',
        help='Delete each zip file after extraction',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    folder_path = os.path.abspath(os.path.expanduser(args.data_root))

    file_list = os.listdir(folder_path)
    for file_name in file_list:
        if not file_name.endswith('.zip'):
            continue

        file_path = os.path.join(folder_path, file_name)
        extract_dir = os.path.join(folder_path, os.path.splitext(file_name)[0])
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        if args.delete_zip:
            os.remove(file_path)


if __name__ == '__main__':
    main()

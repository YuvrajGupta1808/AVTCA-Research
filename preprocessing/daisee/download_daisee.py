#!/usr/bin/env python3
"""Download the official DAiSEE archive from a user-provided approval link."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import os

import gdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default=os.environ.get("DAISEE_URL", ""),
        help="Official DAiSEE archive URL from your approval email. Can also be set via DAISEE_URL.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("datasets/DAISEE"),
        help="Directory where DAiSEE should live.",
    )
    parser.add_argument(
        "--archive_name",
        default="DAiSEE.zip",
        help="Filename to use for the downloaded archive.",
    )
    parser.add_argument(
        "--skip_extract",
        action="store_true",
        help="Only download the archive; do not extract it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if the archive already exists.",
    )
    return parser.parse_args()


def extract_archive(archive_path: Path, data_root: Path) -> None:
    bad_members: list[str] = []
    with zipfile.ZipFile(archive_path) as zf:
        for info in zf.infolist():
            parts = Path(info.filename).parts
            if not parts or parts[0] != "DAiSEE":
                continue

            rel_path = Path(*parts[1:])
            if not rel_path.parts:
                continue

            destination = data_root / rel_path
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue

            if destination.exists() and destination.stat().st_size == info.file_size:
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(info) as src, destination.open("wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
            except Exception:
                bad_members.append(info.filename)

    if bad_members:
        print("Warning: some archive members could not be extracted cleanly:", file=sys.stderr)
        for member in bad_members[:10]:
            print(f"  - {member}", file=sys.stderr)
        if len(bad_members) > 10:
            print(f"  ... plus {len(bad_members) - 10} more", file=sys.stderr)


def main() -> None:
    args = parse_args()
    if not args.url:
        raise SystemExit(
            "Missing DAiSEE archive URL. Pass --url or set the DAISEE_URL environment variable."
        )
    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    archive_path = data_root / args.archive_name

    if args.force or not archive_path.exists():
        print(f"Downloading official DAiSEE archive to {archive_path}")
        output = gdown.download(args.url, str(archive_path), quiet=False, fuzzy=True)
        if output is None:
            raise SystemExit("DAiSEE download failed.")
    else:
        print(f"Using existing archive: {archive_path}")

    if args.skip_extract:
        return

    print(f"Extracting {archive_path} into {data_root}")
    extract_archive(archive_path, data_root)
    print("DAiSEE extraction complete.")


if __name__ == "__main__":
    main()

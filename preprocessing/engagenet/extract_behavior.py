"""Run OpenFace FeatureExtraction over face clips -> per-clip behavior .npy.

Offline preprocessing step (NOT imported by tests). Requires OpenFace 2.0
installed; point ``OPENFACE_BIN`` at its ``FeatureExtraction`` binary.

Each output ``.npy`` is a ``(T, 22)`` float32 array in ``FEATURE_NAMES`` order,
consumed later by ``datasets.engagenet`` via ``BehaviorFeatures.process``.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import tempfile

import numpy as np

from models.behavior_features import load_openface_csv


def extract_clip(video_path: str, out_npy: str, openface_bin: str | None = None) -> np.ndarray:
    """Extract AUs/gaze/pose for one clip and save to ``out_npy``. Returns the array."""
    openface_bin = openface_bin or os.environ.get("OPENFACE_BIN", "FeatureExtraction")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [openface_bin, "-f", video_path, "-out_dir", tmp, "-aus", "-gaze", "-pose"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        csv_matches = glob.glob(os.path.join(tmp, "*.csv"))
        if not csv_matches:
            raise RuntimeError(f"OpenFace produced no CSV for {video_path}")
        feats = load_openface_csv(csv_matches[0])
    os.makedirs(os.path.dirname(os.path.abspath(out_npy)), exist_ok=True)
    np.save(out_npy, feats)
    return feats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos_glob", required=True, help="glob of face-clip videos")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--openface_bin", default=None)
    args = parser.parse_args()

    for video_path in sorted(glob.glob(args.videos_glob, recursive=True)):
        stem = os.path.splitext(os.path.basename(video_path))[0]
        out_npy = os.path.join(args.out_dir, f"{stem}.npy")
        try:
            extract_clip(video_path, out_npy, openface_bin=args.openface_bin)
            print(f"ok  {stem}")
        except Exception as exc:  # keep going on per-clip failures
            print(f"ERR {stem}: {exc}")


if __name__ == "__main__":
    main()

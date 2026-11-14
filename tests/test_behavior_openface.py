"""Tests for the OpenFace CSV loader (models/behavior_features.load_openface_csv)."""

import numpy as np
import pytest

from models.behavior_features import FEATURE_DIM, load_openface_csv

HEADER = (
    "frame, AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r,"
    " AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r, AU45_r,"
    " gaze_angle_x, gaze_angle_y, pose_Rx, pose_Ry, pose_Rz"
)


def _write(path, rows):
    path.write_text("\n".join([HEADER] + rows))
    return str(path)


def test_load_openface_csv_maps_columns(tmp_path):
    rows = ["1," + ",".join(["1.0"] * 22), "2," + ",".join(["2.0"] * 22)]
    feats = load_openface_csv(_write(tmp_path / "clip.csv", rows))
    assert feats.shape == (2, FEATURE_DIM)
    assert np.allclose(feats[0], 1.0)
    assert np.allclose(feats[1], 2.0)


def test_load_openface_csv_reorders_to_feature_names(tmp_path):
    # AU01_r=0..AU45_r=16 then gaze/pose; put a distinctive AU12 value.
    vals = [f"{i}.0" for i in range(22)]
    rows = ["1," + ",".join(vals)]
    feats = load_openface_csv(_write(tmp_path / "clip.csv", rows))
    # AU12 is the 9th AU (index 8) -> value "8.0"
    from models.behavior_features import feature_index
    assert feats[0, feature_index("AU12")] == 8.0
    assert feats[0, feature_index("pose_Rz")] == 21.0


def test_load_openface_csv_missing_column_raises(tmp_path):
    bad_header = HEADER.replace(", AU45_r", "")
    p = tmp_path / "bad.csv"
    p.write_text(bad_header + "\n1," + ",".join(["1.0"] * 21))
    with pytest.raises(ValueError):
        load_openface_csv(str(p))

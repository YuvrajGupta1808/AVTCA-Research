"""Segment statistics and their cache (scripts/fullclip/segment_features.py), on synthetic OpenFace series."""
import csv
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('segment_features_under_test',
                                              os.path.join(ROOT, 'scripts', 'fullclip', 'segment_features.py'))
sf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sf)


def test_segment_stats_mean_and_std_per_segment():
    x = np.arange(30, dtype=np.float32).reshape(10, 3)  # 10 frames, 3 channels
    tok = sf.segment_stats(x, 2, 3)
    assert tok.shape == (2, 6) and tok.dtype == np.float32
    np.testing.assert_allclose(tok[0, :3], x[:5].mean(0))
    np.testing.assert_allclose(tok[0, 3:], x[:5].std(0))
    np.testing.assert_allclose(tok[1, :3], x[5:].mean(0))
    np.testing.assert_allclose(tok[1, 3:], x[5:].std(0))


def test_segment_stats_uneven_split_extra_channels_and_empty_series():
    x = np.random.default_rng(0).normal(size=(7, 5)).astype(np.float32)
    tok = sf.segment_stats(x, 3, 2)  # only the first 2 channels; 7 rows -> segments of 3, 2, 2
    np.testing.assert_allclose(tok[0, :2], x[:3, :2].mean(0), rtol=1e-6)
    np.testing.assert_allclose(tok[2, 2:], x[5:, :2].std(0), rtol=1e-6)
    assert not sf.segment_stats(np.zeros((0, 5), np.float32), 4, 2).any()
    # more segments than rows: the empty segments are zeros
    assert not sf.segment_stats(x[:2], 4, 2)[2:].any()


@pytest.fixture
def tiny_tree(tmp_path, monkeypatch):
    beh = tmp_path / 'behavior'
    beh.mkdir()
    rows = []
    rng = np.random.default_rng(0)
    for i, (label, split) in enumerate([(0, 'training'), (1, 'training'), (3, 'validation'), (2, 'testing')]):
        stem = f'subject_{i}_abc_vid_0_{i}'
        np.save(beh / f'{stem}.npy', rng.normal(size=(12, 4)).astype(np.float32))
        rows.append([f'/data/{stem}_facecroppad.npy', f'/data/{stem}_croppad10s.wav', str(label), split])
    ann = tmp_path / 'ann.txt'
    with open(ann, 'w', newline='') as handle:
        csv.writer(handle, delimiter=';').writerows(rows)
    monkeypatch.setattr(sf, 'ROOT', str(tmp_path))
    monkeypatch.setattr(sf, 'ANN', str(ann))
    monkeypatch.setattr(sf, 'FEATURE_SETS', {'22': (str(beh), 3)})
    return tmp_path, beh


def test_build_caches_with_its_parameters_and_reuses_them(tiny_tree):
    tmp_path, beh = tiny_tree
    d = sf.build(2)
    assert d['X'].shape == (4, 2, 6)
    assert list(d['y']) == [0, 1, 3, 2] and list(d['split']) == ['training', 'training', 'validation', 'testing']
    assert d['keys'][0] == 'subject_0_abc_vid_0_0'
    z = np.load(sf.cache_path(2), allow_pickle=True)
    assert 'meta' in z.files
    # unchanged parameters: the cache is reused, even if a series changed on disk
    np.save(beh / 'subject_0_abc_vid_0_0.npy', np.zeros((12, 4), np.float32))
    np.testing.assert_array_equal(sf.build(2)['X'], d['X'])
    # changed parameters (number of channels): rebuilt, and the new series is read
    sf.FEATURE_SETS['22'] = (str(beh), 2)
    rebuilt = sf.build(2)
    assert rebuilt['X'].shape == (4, 2, 4) and not rebuilt['X'][0].any()


def test_build_refuses_a_missing_series_unless_allowed(tiny_tree):
    _, beh = tiny_tree
    os.remove(beh / 'subject_3_abc_vid_0_3.npy')
    with pytest.raises(FileNotFoundError, match='no OpenFace series'):
        sf.build(2, force=True)
    d = sf.build(2, force=True, allow_missing=True)
    assert not d['X'][3].any()

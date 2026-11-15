"""Late fusion (scripts/fullclip/fuse_members.py): clip-key alignment, threshold search, fixed weights."""
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fm = _load('fuse_members_under_test', 'scripts/fullclip/fuse_members.py')


def _member(path, keys_val, keys_test, y_val, y_test, seed=0, layout='keys'):
    rng = np.random.default_rng(seed)
    out = dict(validation_logits=rng.normal(size=(len(keys_val), 4)).astype(np.float32),
               testing_logits=rng.normal(size=(len(keys_test), 4)).astype(np.float32),
               validation_targets=np.asarray(y_val), testing_targets=np.asarray(y_test))
    if layout == 'keys':
        out.update(validation_keys=np.asarray(keys_val), testing_keys=np.asarray(keys_test))
    elif layout == 'subj':  # the context_analysis.py layout
        for split, keys in (('validation', keys_val), ('testing', keys_test)):
            parts = [k.rsplit('_vid_', 1) for k in keys]
            out[f'{split}_subj'] = np.array([p[0] for p in parts])
            out[f'{split}_vid'] = np.array([int(p[1].split('_')[0]) for p in parts])
            out[f'{split}_idx'] = np.array([int(p[1].split('_')[1]) for p in parts])
    np.savez(path, **out)
    return fm.load(path)


KV = [f'subject_{i}_ab_vid_0_{i}' for i in range(6)]
KT = [f'subject_{i}_cd_vid_1_{i}' for i in range(5)]
YV, YT = [0, 1, 2, 3, 0, 1], [3, 2, 1, 0, 3]


def test_keys_from_subject_layout_match_keys_layout(tmp_path):
    a = _member(tmp_path / 'a.npz', KV, KT, YV, YT, layout='subj')
    assert list(a['validation_keys']) == KV and list(a['testing_keys']) == KT


def test_align_reorders_a_member_stored_in_another_order(tmp_path):
    a = _member(tmp_path / 'a.npz', KV, KT, YV, YT, layout='subj')
    perm = [3, 0, 5, 1, 4, 2]
    b = _member(tmp_path / 'b.npz', [KV[i] for i in perm], KT, [YV[i] for i in perm], YT, seed=1)
    probs_by_key = dict(zip(b['validation_keys'], b['validation']))
    fm.align_members([a, b], ['a', 'b'])
    assert list(b['validation_keys']) == KV
    assert np.array_equal(b['validation_y'], np.asarray(YV))
    for k, p in zip(b['validation_keys'], b['validation']):
        assert np.array_equal(p, probs_by_key[k])


def test_align_rejects_a_missing_clip(tmp_path):
    a = _member(tmp_path / 'a.npz', KV, KT, YV, YT)
    b = _member(tmp_path / 'b.npz', KV[:-1], KT, YV[:-1], YT, seed=1)
    with pytest.raises(ValueError, match='missing'):
        fm.align_members([a, b], ['a', 'b'])


def test_align_rejects_an_extra_clip_even_with_equal_labels(tmp_path):
    a = _member(tmp_path / 'a.npz', KV, KT, YV, YT)
    other = KT[:-1] + ['subject_99_zz_vid_0_0']
    b = _member(tmp_path / 'b.npz', KV, other, YV, YT, seed=1)
    with pytest.raises(ValueError, match='extra'):
        fm.align_members([a, b], ['a', 'b'])


def test_align_rejects_different_labels_for_the_same_clips(tmp_path):
    a = _member(tmp_path / 'a.npz', KV, KT, YV, YT)
    b = _member(tmp_path / 'b.npz', KV, KT, YV[:-1] + [3], YT, seed=1)
    with pytest.raises(ValueError, match='labels differ'):
        fm.align_members([a, b], ['a', 'b'])


def test_unkeyed_members_need_the_explicit_fallback(tmp_path):
    a = _member(tmp_path / 'a.npz', KV, KT, YV, YT, layout='none')
    b = _member(tmp_path / 'b.npz', KV, KT, YV, YT, seed=1, layout='none')
    with pytest.raises(ValueError, match='no clip keys'):
        fm.align_members([a, b], ['a', 'b'])
    fm.align_members([a, b], ['a', 'b'], allow_unkeyed=True)


def test_threshold_search_finds_a_perfect_cut_on_separable_scores():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(4), 25)
    scores = y + 0.8 + rng.uniform(-0.05, 0.05, size=y.size)  # every level shifted up: the default cut fails
    assert (np.digitize(scores, [0.5, 1.5, 2.5]) == y).mean() < 0.5
    th = fm.fit_thresholds(scores, y)
    assert np.all(np.diff(th) > 0)
    assert (np.digitize(scores, th) == y).all()
    for k in range(3):  # each threshold lies in the gap between level k and level k + 1
        assert scores[y == k].max() < th[k] <= scores[y == k + 1].min()


def test_threshold_search_keeps_the_start_when_nothing_is_better():
    y = np.array([0, 1, 2, 3])
    th = fm.fit_thresholds(np.array([0.0, 1.0, 2.0, 3.0]), y)
    assert np.array_equal(th, [0.5, 1.5, 2.5])


def test_fixed_weights():
    assert fm.parse_fixed('equal', 3) == (1 / 3, 1 / 3, 1 / 3)
    assert fm.parse_fixed('1/3, 1/3, 1/3', 3) == (1 / 3, 1 / 3, 1 / 3)
    assert fm.parse_fixed('0.6,0.2,0.2', 3) == (0.6, 0.2, 0.2)
    with pytest.raises(ValueError):
        fm.parse_fixed('0.5,0.2,0.2', 3)
    with pytest.raises(ValueError):
        fm.parse_fixed('0.5,0.5', 3)

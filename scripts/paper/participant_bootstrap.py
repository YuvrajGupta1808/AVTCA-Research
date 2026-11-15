#!/usr/bin/env python
"""Participant-level bootstrap of the two main paired comparisons (JAIHC manuscript, Section 5.4 / S13).

The 2,256 test clips come from 26 participants, so clips are not independent. This resamples
participants instead of clips: each of the 20,000 draws picks 26 participant numbers (the n of
``subject_<n>_...``) with replacement from the sorted unique list, keeps every clip of each drawn
participant, and computes BOTH differences on that same resample:

  1. AVB-Engage minus the audio-visual network (top-1, points; per-clip correctness averaged over the
     three seeds). AVB-Engage = fixed 0.6/0.2/0.2 fusion decoded with expected-level thresholds fitted on
     validation (fuse_members.fit_thresholds); the network = its refined thresholds from the run's
     calibration JSON. Same decoders as Table 2 (springer_analysis.py).
  2. Audio + video minus video only, on the same trained weights (results/paper_jaihc/sigtest/*.npz from
     significance_tests.py collect), each condition decoded with its own refined thresholds
     (calibration/ and calibration_video_only/), averaged over the nine networks of Section 5.4.1.

Reads stored outputs only (no training, no GPU). Output: results/paper_jaihc/participant_bootstrap.json
Usage: python scripts/paper/participant_bootstrap.py [--draws 20000] [--seed 0]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import springer_analysis as sa  # noqa: E402  (also puts scripts/fullclip on sys.path)
from fuse_members import align_members, load  # noqa: E402
from src.engine.calibration import decode_expected_thresholds  # noqa: E402

FC = sa.FC
OUT = os.path.join(ROOT, 'results', 'paper_jaihc', 'participant_bootstrap.json')
SIGTEST = os.path.join(ROOT, 'results', 'paper_jaihc', 'sigtest')
TF_CONFIG = 'T7_d64_L2_lr1e3'
NETWORKS = [f'{arm}_s{s}' for arm in ('A_av', 'B_beh', 'C_behtext') for s in (1, 2, 3)]


def participant_of(key):
    """'subject_105_jwdq6sdejx_vid_0_1' -> '105' (the participant number)."""
    return str(key).split('_')[1]


def fusion_and_network_correct():
    """Per-clip correctness (3 seeds x 2,256) of AVB-Engage and of the network, plus clip keys and labels."""
    gbm = load(os.path.join(FC, 'gbm', 'probs.npz'))
    fused, net, keys, y = [], [], None, None
    for s in sa.SEEDS:
        av = sa.av_member(s)
        tf = load(os.path.join(FC, 'segtf', f'{TF_CONFIG}_s{s}', 'logits.npz'))
        align_members([av, tf, gbm], [f'A_av_s{s}', f'{TF_CONFIG}_s{s}', 'gbm'])
        yv, yt = av['validation_y'], av['testing_y']
        w = sa.FIXED
        pred, _ = sa.fitted_pred(w[0] * av['validation'] + w[1] * tf['validation'] + w[2] * gbm['validation'], yv,
                                 w[0] * av['testing'] + w[1] * tf['testing'] + w[2] * gbm['testing'])
        fused.append(pred == yt)
        net.append(av['pred_test'] == yt)
        if keys is None:
            keys, y = av['testing_keys'], yt
        assert np.array_equal(keys, av['testing_keys'])
    return np.array(fused), np.array(net), keys, y


def audio_correct(y_ref):
    """Per-clip correctness (9 networks x 2,256) with audio + video and with video only."""
    with_audio, video_only = [], []
    for run in NETWORKS:
        z = np.load(os.path.join(SIGTEST, f'{run}.npz'))
        y = z['testing_y']
        assert np.array_equal(y, y_ref), f'{run}: sigtest clip order differs from the annotation order'
        for cond, sub, sink in (('none', 'calibration', with_audio), ('video_only', 'calibration_video_only', video_only)):
            cj = json.load(open(os.path.join(FC, run, sub, 'calibration_results.json')))['refined_expected_thresholds']
            pred = decode_expected_thresholds(z[f'testing_{cond}'], cj['thresholds'])
            sa.assert_same_count(pred, y, cj['testing']['top1_accuracy'], f'{run} {cond}')
            sink.append(pred == y)
    return np.array(with_audio), np.array(video_only)


def bootstrap(pairs, part, draws=20000, seed=0):
    """Participant bootstrap of paired differences on shared resamples.

    pairs: list of (a, b) per-clip correctness arrays (each length n_clips); part: participant of each clip.
    Each draw resamples the sorted unique participants with replacement (one rng.choice per draw) and
    returns, for every pair, 100 * (mean a - mean b) over the clips of the drawn participants.
    Returns (units, idx, diffs) with diffs of shape (len(pairs), draws).
    """
    part = np.asarray(part)
    units = np.unique(part)
    idx = {u: np.where(part == u)[0] for u in units}
    rng = np.random.default_rng(seed)
    out = np.empty((len(pairs), draws))
    for b in range(draws):
        ii = np.concatenate([idx[u] for u in rng.choice(units, len(units))])
        for k, (x, y) in enumerate(pairs):
            out[k, b] = 100 * (x[ii].mean() - y[ii].mean())
    return units, idx, out


def summarise(diff_by_participant, draws, point):
    return dict(point=float(point), ci95=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                share_of_draws_le_0=float((draws <= 0).mean()),
                participants_better=int((diff_by_participant > 0).sum()),
                participants_worse=int((diff_by_participant < 0).sum()),
                participants_tied=int((diff_by_participant == 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    fused, net, keys, y = fusion_and_network_correct()
    with_audio, video_only = audio_correct(y)
    F, N = fused.mean(0), net.mean(0)              # per-clip correctness averaged over the 3 seeds
    A, V = with_audio.mean(0), video_only.mean(0)  # ... over the 9 networks
    part = np.array([participant_of(k) for k in keys])  # units sorted as strings, the order the draws index into
    units, idx, (d_fusion, d_audio) = bootstrap([(F, N), (A, V)], part, a.draws, a.seed)

    per_f = np.array([100 * (F[idx[u]].mean() - N[idx[u]].mean()) for u in units])
    per_a = np.array([100 * (A[idx[u]].mean() - V[idx[u]].mean()) for u in units])
    clips = {u: int(len(idx[u])) for u in units}
    fused_acc = {u: float(100 * F[idx[u]].mean()) for u in units}
    res = dict(
        description='Participant-level bootstrap: resample the test participants with replacement; both differences '
                    'on the same resample; top-1 points.',
        n_test_clips=int(len(y)), n_participants=int(len(units)), draws=a.draws, rng='numpy default_rng',
        rng_seed=a.seed, participant_order='sorted unique participant numbers (as strings)',
        fusion_minus_network=summarise(per_f, d_fusion, 100 * (F.mean() - N.mean())),
        audio_video_minus_video_only=summarise(per_a, d_audio, 100 * (A.mean() - V.mean())),
        clips_per_participant=dict(min=min(clips.values()), median=float(np.median(list(clips.values()))),
                                   max=max(clips.values()), by_participant=clips),
        fused_accuracy_per_participant=dict(min=min(fused_acc.values()), max=max(fused_acc.values()),
                                            by_participant=fused_acc),
        fused_top1_mean=float(100 * F.mean()), network_top1_mean=float(100 * N.mean()),
        audio_video_top1_mean=float(100 * A.mean()), video_only_top1_mean=float(100 * V.mean()))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    for name in ('fusion_minus_network', 'audio_video_minus_video_only'):
        r = res[name]
        print(f"{name:30s} {r['point']:+.2f}  95% interval [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]  "
              f"share <= 0 {r['share_of_draws_le_0']:.4f}  participants better/worse/tied "
              f"{r['participants_better']}/{r['participants_worse']}/{r['participants_tied']}")
    c, f = res['clips_per_participant'], res['fused_accuracy_per_participant']
    print(f"{res['n_participants']} participants; clips per participant {c['min']}-{c['max']} (median {c['median']:.0f}); "
          f"AVB-Engage accuracy per participant {f['min']:.1f}-{f['max']:.1f}%")
    print('wrote', a.out)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Model size, FLOPs and inference time of the AVB-Engage members (Springer/MTAP complexity table).

Measured on real EngageNet test clips (batch 1, 50 face frames + full-clip log-mel) with the trained
seed-1 audio-visual checkpoint of the fixed full-clip protocol (results/fullclip/A_av_s1). FLOPs are
counted with torch.utils.flop_counter (one multiply-add = 2 FLOPs). Latency excludes data loading,
face detection and OpenFace extraction; those are upstream preprocessing steps reported separately.

The segment Transformer is counted in eval mode with PyTorch's fused multi-head-attention fast path
disabled and the math attention kernel forced: the fused paths are invisible to the FLOP counter, which
then reports 0.11 M instead of 3.09 M (results/paper_springer/complexity.json, written before this fix,
still holds the 0.11 M value). The count is checked against the analytic count.

Run with the training environment (conda activate avtca):
  python scripts/paper/complexity_benchmark.py                # everything (GPU); results/paper_springer/complexity.json
  python scripts/paper/complexity_benchmark.py --flops_only   # segment-Transformer FLOPs only (CPU);
                                                              # results/paper_jaihc/segment_transformer_flops.json
"""
import argparse
import importlib.util
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'fullclip'))
os.chdir(ROOT)
RUN = os.path.join(ROOT, 'results', 'fullclip', 'A_av_s1')
OUT = os.path.join(ROOT, 'results', 'paper_springer', 'complexity.json')
FLOPS_OUT = os.path.join(ROOT, 'results', 'paper_jaihc', 'segment_transformer_flops.json')
# The reported segment Transformer (T7): 20 tokens of 44 features, width 64, 2 layers, 8 heads, FFN 128.
TF_CFG = dict(in_dim=44, n_seg=20, d_model=64, n_layers=2, n_heads=8, ffn=128, n_classes=4)


def analytic_transformer_flops(in_dim, n_seg, d_model, n_layers, n_heads, ffn, n_classes):
    """Matmul FLOPs of SegmentTransformer for one clip (2 per multiply-add; CLS token included)."""
    t = n_seg + 1
    per_layer = (2 * t * d_model * 3 * d_model      # Q, K, V projections
                 + 2 * 2 * t * t * d_model           # Q K^T and attention x V, summed over the heads
                 + 2 * t * d_model * d_model         # output projection
                 + 2 * 2 * t * d_model * ffn)        # two feed-forward layers
    return 2 * n_seg * in_dim * d_model + n_layers * per_layer + 2 * d_model * n_classes


def count_transformer_flops(model, x):
    """FLOPs of one forward pass with the attention matmuls visible to the counter."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    prev = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as fc:
            model(x)
    finally:
        torch.backends.mha.set_fastpath_enabled(prev)
    return int(fc.get_total_flops())


def segment_transformer_flops():
    from segment_transformer import SegmentTransformer
    c = TF_CFG
    tf = SegmentTransformer(c['in_dim'], c['n_seg'], c['d_model'], c['n_layers'], c['n_heads'], c['ffn'], 0.3, pool='cls')
    counted = count_transformer_flops(tf, torch.randn(1, c['n_seg'], c['in_dim']))
    analytic = analytic_transformer_flops(**c)
    assert abs(counted - analytic) <= 0.01 * analytic, (counted, analytic)
    return tf, dict(config=c, params=count(tf), flops_counted=counted, flops_analytic=analytic,
                    mflops_per_clip=counted / 1e6, torch=torch.__version__,
                    method='eval mode, MHA fast path disabled, math SDPA kernel, torch.utils.flop_counter (multiply-add = 2)')


def load_cal():
    spec = importlib.util.spec_from_file_location('cal', os.path.join(ROOT, 'scripts', 'calibrate_engagement_logits.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def count(module):
    return int(sum(p.numel() for p in module.parameters()))


def build_network(cal, device):
    opts = json.load(open(sorted(f for f in (os.path.join(RUN, x) for x in os.listdir(RUN)) if os.path.basename(f).startswith('opts'))[-1]))
    ann = str(opts['annotation_path'])
    if not os.path.isfile(ann):  # the run's worktree copy is gone; the main-tree file has the same clips
        ann = f'{ROOT}/preprocessing/engagenet/annotations_engagement_a10.txt'
    sys.argv = ['x', '--annotation_path', ann,
                '--checkpoint_path', os.path.join(RUN, 'ENGAGENET_multimodal_cnn_15_best.pth'),
                '--result_path', os.path.join(ROOT, 'results', 'paper_springer', 'tmp_bench'), '--dataset', 'ENGAGENET',
                '--n_classes', '4', '--model', 'multimodal_cnn', '--fusion', 'it', '--audio_features', 'mel',
                '--num_heads', '8', '--visual_backbone', 'efficientface', '--data_root', f'{ROOT}/datasets/EngageNet',
                '--device', device, '--n_threads', '4',
                '--pretrain_path', f'{ROOT}/pretrained/EfficientFace_Trained_on_AffectNet7.pth', '--mask', 'nodropout',
                '--no_late_text_fusion', '--full_video_preprocessing', '--max_audio_steps', '0',
                '--max_video_frames', str(opts.get('max_video_frames', 50)),
                '--frame_sampling', str(opts.get('frame_sampling', 'uniform')),
                '--batch_size', '1', '--loss', 'ordinal_distance', '--ordinal_distance_weight', '0.15']
    args = cal.parse_args()
    opt = cal.make_opt(args)
    model, _ = cal.generate_model(opt)
    target = model.module if hasattr(model, 'module') else model
    target.ablate_modality = 'none'
    cal.load_state_dict_flexible(model, opt.checkpoint_path, map_location=torch.device(device))
    model.eval()
    return model, target, opt


def batches(cal, opt, n):
    _, loader = cal.build_eval_loader(opt, 'testing')
    out = []
    for i, batch in enumerate(loader):
        if i >= n:
            break
        audio, video, _, al, vl, am, vm, *_ = cal._unpack_multimodal_batch(batch)
        out.append((audio, video, al, vl, am, vm))
    return out


def run(model, b, device):
    audio, video, al, vl, am, vm = (t.to(device) for t in b)
    return model(audio, video, audio_mask=am, video_mask=vm, audio_lengths=al, video_lengths=vl,
                 text_tokens=None, text_mask=None, behavior_feats=None, behavior_present=None)


def time_model(model, data, device, warmup=10):
    with torch.no_grad():
        for b in data[:warmup]:
            run(model, b, device)
        if device == 'cuda':
            torch.cuda.synchronize()
        ts = []
        for b in data:
            t0 = time.perf_counter()
            run(model, b, device)
            if device == 'cuda':
                torch.cuda.synchronize()
            ts.append(1000 * (time.perf_counter() - t0))
    return dict(mean_ms=float(np.mean(ts)), sd_ms=float(np.std(ts, ddof=1)), median_ms=float(np.median(ts)), n=len(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flops_only', action='store_true', help='only count the segment-Transformer FLOPs (CPU) and write ' + FLOPS_OUT)
    ap.add_argument('--out', default='', help='output path (default depends on --flops_only)')
    a = ap.parse_args()
    if a.flops_only:
        _, r = segment_transformer_flops()
        out = a.out or FLOPS_OUT
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(r, open(out, 'w'), indent=1)
        print(json.dumps(r, indent=1))
        return
    res = {}
    cal = load_cal()
    torch.manual_seed(0)
    model, target, opt = build_network(cal, 'cuda')
    groups = {}
    for name, child in target.named_children():
        groups[name] = count(child)
    res['network_params'] = dict(total=count(target), by_module=groups)
    backbone = sum(p.numel() for n, p in target.visual_model.named_parameters()
                   if not n.startswith(('conv1d_0', 'conv1d_1', 'conv1d_2', 'conv1d_3', 'classifier')))
    res['network_params']['visual_frame_encoder'] = int(backbone)
    data = batches(cal, opt, 110)
    first = data[0]
    res['input_shapes'] = dict(audio=list(first[0].shape), video=list(first[1].shape))
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        run(model, first, 'cuda')
    res['network_gflops_per_clip'] = fc.get_total_flops() / 1e9
    by_mod = fc.get_flop_counts()
    res['network_gflops_by_module'] = {k: sum(v.values()) / 1e9 for k, v in by_mod.items() if k.count('.') <= 1}
    res['gpu'] = dict(device=torch.cuda.get_device_name(0), **time_model(model, data, 'cuda'))
    res['gpu']['peak_mem_mb'] = torch.cuda.max_memory_allocated() / 2**20
    torch.set_num_threads(8)
    cpu_model = model.module if hasattr(model, 'module') else model
    cpu_model = cpu_model.to('cpu')
    res['cpu'] = dict(threads=8, **time_model(cpu_model, data[:30], 'cpu', warmup=3))
    # segment Transformer member (T7: d64, L2, ffn 128, 8 heads, CLS pooling); timed with the fast path on
    tf, flops = segment_transformer_flops()
    tf.eval()
    x = torch.randn(1, TF_CFG['n_seg'], TF_CFG['in_dim'])
    ts = []
    with torch.no_grad():
        for _ in range(10):
            tf(x)
        for _ in range(200):
            t0 = time.perf_counter(); tf(x); ts.append(1000 * (time.perf_counter() - t0))
    res['segment_transformer'] = dict(params=flops['params'], mflops_per_clip=flops['mflops_per_clip'],
                                      mflops_analytic=flops['flops_analytic'] / 1e6,
                                      cpu_mean_ms=float(np.mean(ts)), cpu_threads=8)
    # gradient-boosting member: same recipe as gbm_member.py (three fits), timed on CPU
    from segment_features import build
    from sklearn.ensemble import HistGradientBoostingClassifier
    d = build(20)
    import csv, re
    from concurrent.futures import ThreadPoolExecutor
    ann = f'{ROOT}/preprocessing/engagenet/annotations_engagement_a10.txt'
    rows = [r for r in csv.reader(open(ann), delimiter=';') if r]
    stems = [re.sub(r'_facecroppad\.npy$', '', os.path.basename(r[0])) for r in rows]

    def clipstat(stem):
        p = os.path.join(ROOT, 'datasets', 'EngageNet', 'behavior', stem + '.npy')
        s = np.load(p)[:, :22] if os.path.isfile(p) else np.zeros((0, 22), np.float32)
        return np.concatenate([s.mean(0), s.std(0)]).astype(np.float32) if len(s) else np.zeros(44, np.float32)
    with ThreadPoolExecutor(8) as ex:
        C = np.stack(list(ex.map(clipstat, stems)))
    X = np.concatenate([d['X'].reshape(len(d['X']), -1), C], 1)
    tr, te = d['split'] == 'training', d['split'] == 'testing'
    fits, t_fit = [], []
    for seed, (lr_, it) in enumerate([(0.05, 300), (0.03, 600), (0.05, 300)]):
        t0 = time.perf_counter()
        fits.append(HistGradientBoostingClassifier(learning_rate=lr_, max_iter=it, max_leaf_nodes=15, l2_regularization=1.0,
                                                   early_stopping=False, random_state=seed).fit(X[tr], d['y'][tr]))
        t_fit.append(time.perf_counter() - t0)
    one = X[te][:1]
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        np.mean([g.predict_proba(one) for g in fits], 0)
        ts.append(1000 * (time.perf_counter() - t0))
    n_trees = int(sum(g.n_iter_ * len(g.classes_) for g in fits))
    res['gbm'] = dict(input_dims=int(X.shape[1]), trees=n_trees, max_leaves_per_tree=15,
                      pickled_mb=len(pickle.dumps(fits)) / 2**20, fit_seconds_total=float(sum(t_fit)),
                      predict_ms_per_clip=float(np.median(ts)),
                      test_argmax_top1=float(100 * (np.mean([g.predict_proba(X[te]) for g in fits], 0).argmax(1) == d['y'][te]).mean()))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

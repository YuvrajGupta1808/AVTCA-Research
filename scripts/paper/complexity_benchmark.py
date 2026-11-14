#!/usr/bin/env python
"""Model size, FLOPs and inference time of the AVB-Engage members (Springer/MTAP complexity table).

Measured on real EngageNet test clips (batch 1, 50 face frames + full-clip log-mel) with the trained
seed-1 audio-visual checkpoint of the fixed full-clip protocol (results/fullclip/A_av_s1). FLOPs are
counted with torch.utils.flop_counter (one multiply-add = 2 FLOPs). Latency excludes data loading,
face detection and OpenFace extraction; those are upstream preprocessing steps reported separately.

Run with the training environment:  ~/.conda/envs/avtca/bin/python scripts/paper/complexity_benchmark.py
Output: results/paper_springer/complexity.json
"""
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
    # segment Transformer member (T7: d64, L2, ffn 128, 8 heads, CLS pooling)
    from segment_transformer import SegmentTransformer
    tf = SegmentTransformer(44, 20, 64, 2, 8, 128, 0.3, pool='cls').eval()
    x = torch.randn(1, 20, 44)
    with torch.no_grad(), FlopCounterMode(display=False) as fc2:
        tf(x)
    ts = []
    with torch.no_grad():
        for _ in range(10):
            tf(x)
        for _ in range(200):
            t0 = time.perf_counter(); tf(x); ts.append(1000 * (time.perf_counter() - t0))
    res['segment_transformer'] = dict(params=count(tf), mflops_per_clip=fc2.get_total_flops() / 1e6,
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

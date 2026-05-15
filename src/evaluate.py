"""
Computes all metrics for an existing AVTCA checkpoint.

Usage:
  python src/evaluate.py \
      --checkpoint results/mel_h8_lr001_e75/RAVDESS_multimodal_cnn_15_best.pth \
      --result_path results/mel_h8_lr001_e75 --num_heads 8 --test_subset test
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from torch import nn
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, accuracy_score,
)

from src import transforms
from src.model import generate_model
from src.dataset import get_test_set

CLASS_NAMES = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def parse_args():
    parser = argparse.ArgumentParser(description='Compute metrics for an AVTCA checkpoint')
    parser.add_argument('--checkpoint',      required=True, type=str)
    parser.add_argument('--result_path',     default='results', type=str)
    parser.add_argument('--test_subset',     default='test', choices=['test', 'val'])
    parser.add_argument('--annotation_path', default='preprocessing/ravdess/annotations.txt', type=str)
    parser.add_argument('--data_root',       default='', type=str)
    parser.add_argument('--dataset',         default='RAVDESS', type=str)
    parser.add_argument('--n_classes',       default=8, type=int)
    parser.add_argument('--num_heads',       default=8, type=int)
    parser.add_argument('--fusion',          default='it', type=str)
    parser.add_argument('--sample_duration', default=15, type=int)
    parser.add_argument('--sample_size',     default=224, type=int)
    parser.add_argument('--batch_size',      default=8, type=int)
    parser.add_argument('--n_threads',       default=4, type=int)
    parser.add_argument('--video_norm_value',default=255, type=int)
    parser.add_argument('--pretrain_path',   default='None', type=str)
    parser.add_argument('--model',           default='multimodal_cnn', type=str)
    parser.add_argument('--audio_features',  default='mel', type=str, choices=['mfcc', 'mel'])
    parser.add_argument('--device',          default='cuda', type=str)
    parser.add_argument('--mask',            default='softhard', type=str)
    parser.add_argument('--manual_seed',     default=1, type=int)
    parser.add_argument('--store_name',      default='model', type=str)
    return parser.parse_args()


def topk_accuracy(outputs_np, targets_np, k):
    topk_preds = np.argsort(outputs_np, axis=1)[:, -k:]
    correct = [targets_np[i] in topk_preds[i] for i in range(len(targets_np))]
    return float(np.mean(correct)) * 100


def run():
    opt = parse_args()

    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    opt.arch = opt.model
    torch.manual_seed(opt.manual_seed)

    print(f'Building model: fusion={opt.fusion}, num_heads={opt.num_heads}, '
          f'pretrain_path={opt.pretrain_path}')
    model, _ = generate_model(opt)

    print(f'Loading checkpoint: {opt.checkpoint}')
    ckpt = torch.load(opt.checkpoint, map_location=opt.device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
        print(f'  Epoch {ckpt.get("epoch", "?")}  best_prec1={ckpt.get("best_prec1", "?")}')
    else:
        model.load_state_dict(ckpt)
        print('  Loaded flat state_dict')

    model.eval()
    criterion = nn.CrossEntropyLoss().to(opt.device)

    video_transform = transforms.Compose([transforms.ToTensor(opt.video_norm_value)])
    data = get_test_set(opt, spatial_transform=video_transform)
    loader = torch.utils.data.DataLoader(
        data, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.n_threads, pin_memory=True)

    all_preds, all_targets, all_logits = [], [], []
    total_loss = 0.0
    n_batches = 0

    print(f'Running inference on {opt.test_subset} split ({len(data)} samples)...')
    with torch.no_grad():
        for i, (inputs_audio, inputs_visual, targets) in enumerate(loader):
            inputs_visual = inputs_visual.permute(0, 2, 1, 3, 4)
            inputs_visual = inputs_visual.reshape(
                inputs_visual.shape[0] * inputs_visual.shape[1],
                inputs_visual.shape[2], inputs_visual.shape[3], inputs_visual.shape[4])
            inputs_audio  = inputs_audio.to(opt.device)
            inputs_visual = inputs_visual.to(opt.device)
            targets       = targets.to(opt.device)
            outputs = model(inputs_audio, inputs_visual)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            n_batches += 1
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_targets.append(targets.cpu())
            all_logits.append(outputs.cpu())
            if (i + 1) % 10 == 0:
                print(f'  batch {i+1}/{len(loader)}')

    all_preds_np   = torch.cat(all_preds).numpy()
    all_targets_np = torch.cat(all_targets).numpy()
    all_logits_np  = torch.cat(all_logits).numpy()
    avg_loss       = total_loss / n_batches

    top1  = accuracy_score(all_targets_np, all_preds_np) * 100
    top5  = topk_accuracy(all_logits_np, all_targets_np, k=5)

    f1_macro      = f1_score(all_targets_np, all_preds_np, average='macro',    zero_division=0)
    f1_weighted   = f1_score(all_targets_np, all_preds_np, average='weighted', zero_division=0)
    prec_weighted = precision_score(all_targets_np, all_preds_np, average='weighted', zero_division=0)
    rec_weighted  = recall_score(all_targets_np, all_preds_np,  average='weighted', zero_division=0)
    report_str    = classification_report(all_targets_np, all_preds_np, target_names=CLASS_NAMES, zero_division=0)
    report_dict   = classification_report(all_targets_np, all_preds_np, target_names=CLASS_NAMES, zero_division=0, output_dict=True)

    per_class_acc = {}
    for c, name in enumerate(CLASS_NAMES):
        mask = all_targets_np == c
        if mask.sum() > 0:
            per_class_acc[name] = round(float((all_preds_np[mask] == c).mean()), 4)

    print(f'\n{"="*55}')
    print(f'Checkpoint : {opt.checkpoint}')
    print(f'Split      : {opt.test_subset}  ({len(data)} samples)')
    print(f'{"="*55}')
    print(f'Top-1 Accuracy : {top1:.2f}%')
    print(f'Top-5 Accuracy : {top5:.2f}%')
    print(f'Loss           : {avg_loss:.4f}')
    print(f'F1 Macro       : {f1_macro*100:.2f}%')
    print(f'F1 Weighted    : {f1_weighted*100:.2f}%')
    print(f'Prec Weighted  : {prec_weighted*100:.2f}%')
    print(f'Rec Weighted   : {rec_weighted*100:.2f}%')
    print(f'\nClassification Report:\n{report_str}')
    print('Per-class accuracy:')
    for name, acc in per_class_acc.items():
        n = int((all_targets_np == CLASS_NAMES.index(name)).sum())
        print(f'  {name:>10s}: {acc:.3f}  ({n} samples)')

    out = {
        'checkpoint':       opt.checkpoint,
        'split':            opt.test_subset,
        'n_samples':        len(data),
        'top1_accuracy':    round(top1, 4),
        'top5_accuracy':    round(top5, 4),
        'loss':             round(avg_loss, 6),
        'f1_macro':         round(f1_macro * 100, 4),
        'f1_weighted':      round(f1_weighted * 100, 4),
        'prec_weighted':    round(prec_weighted * 100, 4),
        'rec_weighted':     round(rec_weighted * 100, 4),
        'classification_report_str':  report_str,
        'classification_report_dict': report_dict,
        'per_class_accuracy': per_class_acc,
    }
    os.makedirs(opt.result_path, exist_ok=True)
    out_path = os.path.join(opt.result_path, f'test_metrics_{opt.test_subset}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nMetrics written to {out_path}')


if __name__ == '__main__':
    run()

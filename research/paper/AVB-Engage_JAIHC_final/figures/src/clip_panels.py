"""The example clip of Figs. 1 and 2 and its data panels, shared by make_overview_drawio.py and make_architecture_drawio.py.

One EngageNet test clip (CLIP, true level 2, "engaged") as the seed-1 models of the reported runs see it: the log-mel
spectrogram the network reads (datasets/ravdess.py::get_mel on the clip's _croppad10s.wav), three of its 22 OpenFace
channels (order of models/behavior_features.py), the class probabilities of the three members (results/fullclip:
A_av_s1, segtf/T7_d64_L2_lr1e3_s1, gbm), their fixed-weight average 0.6/0.2/0.2 and the expected level cut at the
thresholds that scripts/fullclip/fuse_members.py fits on the validation split (seed-1 test top-1 70.17, the first of the
paper's 70.29 +- 0.14). The clip (chosen 2026-11-14) is classified correctly by all three seeds, its members disagree
(the Transformer alone says level 1), and its audio has no voiced frame (librosa.pyin), so the figures publish no voice;
the face is a drawing, not the participant. The spectrogram PNG repeats each mel bin 8x and each frame 2x (nearest
neighbour), so it prints at 700-1000 dpi without resampling the data. Needs the avtca environment (numpy, librosa,
Pillow, scikit-learn).
"""
import base64
import io
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts' / 'fullclip')]
from datasets.ravdess import get_mel, load_audio  # noqa: E402  (the network's audio front end)
from fuse_members import fit_thresholds, load      # noqa: E402  (the late fusion of the reported runs)

CLIP, SEED, WEIGHTS = 'subject_82_gofy9p13pj_vid_1_29', 1, (0.6, 0.2, 0.2)
MEMBERS = [ROOT / f'results/fullclip/A_av_s{SEED}/context/logits.npz',
           ROOT / f'results/fullclip/segtf/T7_d64_L2_lr1e3_s{SEED}/logits.npz',
           ROOT / 'results/fullclip/gbm/probs.npz']
LEVELS = ['not engaged', 'barely engaged', 'engaged', 'highly engaged']
TRACES = [('AU45', 16), ('gaze', 18), ('pitch', 19)]   # blink intensity, gaze_angle_y, pose_Rx

# ---------------------------------------------------------------------------------------------------------- data
ms = [load(str(p)) for p in MEMBERS]
lv = np.arange(4)
fuse = lambda split: sum(w * m[split] for w, m in zip(WEIGHTS, ms))   # noqa: E731
th = fit_thresholds(fuse('validation') @ lv, ms[0]['validation_y'])
pt, yt = fuse('testing'), ms[0]['testing_y']
acc = 100 * (np.digitize(pt @ lv, th) == yt).mean()
assert round(acc, 2) == 70.17, acc                     # seed-1 row of the paper's three-way fusion
z = np.load(MEMBERS[0])
keys = np.load(MEMBERS[2])['testing_keys']
assert (np.array([f'{s}_vid_{v}_{i}' for s, v, i in zip(z['testing_subj'], z['testing_vid'], z['testing_idx'])]) == keys).all()
assert (np.load(MEMBERS[1])['testing_keys'] == keys).all()
i = int(np.flatnonzero(keys == CLIP)[0])
probs = dict(zip(('AV', 'TF', 'GBM', 'p'), [m['testing'][i] for m in ms] + [pt[i]]))
mu = float(pt[i] @ lv)
level = int(np.digitize(mu, th))
assert level == yt[i] == 2, (level, yt[i])
y, sr = load_audio(str(ROOT / f'datasets/EngageNet/Test/{CLIP}_croppad10s.wav'), sr=22050)   # full clip, as trained
mel = get_mel(y, sr, n_mels=64)                        # (64, 431): dB relative to the clip maximum, floor -80
beh = np.load(ROOT / f'datasets/EngageNet/behavior/{CLIP}.npy')[:300]                       # (300, 22) at 30 fps
print(f'{CLIP}: level {yt[i]}, mu {mu:.3f}, thresholds {np.round(th, 2)}, mel {mel.shape}, behavior {beh.shape}')
for k, q in probs.items():
    print(f'  p_{k:<4} {np.round(q, 3)}')

# ------------------------------------------------------------------------------------------------ data panels
BLUES = ['#F7FBFF', '#DEEBF7', '#C6DBEF', '#9ECAE1', '#6BAED6', '#4292C6', '#2171B5', '#08519C', '#08306B']


def ramp(v, stops):
    """Sequential colour ramp (ColorBrewer Blues: lightness falls monotonically, so it also reads in grey)."""
    rgb = np.array([[int(s[k:k + 2], 16) for k in (1, 3, 5)] for s in stops], float)
    pos = np.clip(v, 0, 1) * (len(stops) - 1)
    lo = np.minimum(pos.astype(int), len(stops) - 2)
    f = (pos - lo)[..., None]
    return np.round(rgb[lo] * (1 - f) + rgb[lo + 1] * f).astype(np.uint8)


def png_uri(a):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, 'PNG', optimize=True)
    return 'data:image/png,' + base64.b64encode(buf.getvalue()).decode()   # draw.io style form (no ";base64")


def svg_uri(svg):
    return 'data:image/svg+xml,' + base64.b64encode(svg.encode()).decode()


FACE_34 = ('<ellipse cx="6.7" cy="19.6" rx="1.4" ry="2.3" fill="#FFFFFF" stroke-width="0.7"/>'
            '<ellipse cx="27.3" cy="19.6" rx="1.4" ry="2.3" fill="#FFFFFF" stroke-width="0.7"/>'
            '<ellipse cx="17" cy="18.6" rx="10.2" ry="12.6" fill="#FFFFFF" stroke-width="0.8"/>'
            '<path d="M7.1 15.4 C6.8 8.6 11.4 5.6 17 5.6 C22.6 5.6 27.2 8.6 26.9 15.4 C26.1 12.3 23.6 10.5 20.3 10.2 '
            'C17.2 10 14.3 11.1 12.1 13.1 C10.7 12.4 8.8 13.2 7.1 15.4 Z" fill="#6E6D68" stroke="none"/>'
            '<path d="M12 16.6 L15.1 16.1 M18.9 16.1 L22 16.6" fill="none" stroke-width="0.7"/>'
            '<circle cx="13.6" cy="18.9" r="1.1" fill="#3D3C39" stroke="none"/>'
            '<circle cx="20.4" cy="18.9" r="1.1" fill="#3D3C39" stroke="none"/>'
            '<path d="M17 20.1 L16.1 23.6 L17.8 23.7" fill="none" stroke-width="0.6"/>'
            '<path d="M14.2 26.5 Q17 28.3 19.8 26.5" fill="none" stroke-width="0.75"/>')   # a drawn face on a 34-unit grid


def face_g(x0, y0, scale):
    """The drawn face, placed at (x0, y0) and scaled from its 34-unit grid."""
    return (f'<g transform="translate({x0:.2f},{y0:.2f}) scale({scale:.4f})" stroke="#3D3C39" stroke-linecap="round" '
            'stroke-linejoin="round">' + FACE_34 + '</g>')


def face_stack_svg(w, h, dx=6.5, dy=5.5):
    """Three stacked face crops, offset by (dx, -dy); the front one shows a drawn face (no participant is shown)."""
    s = h - 2 * dy - 1.5                  # frame side; the face is drawn on a 34-unit grid and scaled to it
    x0, y0 = (w - s - 2 * dx) / 2, h - s - 0.75
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    for k, fill, stroke in ((2, '#F4F8FD', '#8DB5E8'), (1, '#F4F8FD', '#6A9EE0'), (0, '#EAF2FC', '#2A78D6')):
        out.append(f'<rect x="{x0 + k * dx:.2f}" y="{y0 - k * dy:.2f}" width="{s:.2f}" height="{s:.2f}" rx="3" '
                   f'fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>')
    out.append(face_g(x0, y0, s / 34) + '</svg>')
    return ''.join(out)


def face_row_svg(w, h, n=4, gap=2.5):
    """n face crops side by side (frames over time), each a drawn face (no participant is shown)."""
    fw = (w - (n - 1) * gap) / n
    fh = min(h - 1, fw * 1.15)
    y0 = (h - fh) / 2
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    for k in range(n):
        x0 = k * (fw + gap)
        out.append(f'<rect x="{x0 + 0.4:.2f}" y="{y0:.2f}" width="{fw - 0.8:.2f}" height="{fh:.2f}" rx="2" fill="#EAF2FC" '
                   'stroke="#2A78D6" stroke-width="0.8"/>')
        sc = (fw - 1) / 34
        out.append(face_g(x0 + 0.5, y0 + (fh - 34 * sc) / 2 + 0.6, sc))
    out.append('</svg>')
    return ''.join(out)


def traces_svg(w, h):
    """The clip's OpenFace channels over its 10 s, one band each, each scaled to its own range."""
    band = h / len(TRACES)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    for k, (_, col) in enumerate(TRACES):
        s = beh[:, col].astype(float)
        v = (s - s.min()) / max(s.max() - s.min(), 1e-9)
        top = k * band
        if k:
            out.append(f'<line x1="0" y1="{top:.2f}" x2="{w}" y2="{top:.2f}" stroke="#E2E1DD" stroke-width="0.5"/>')
        pts = ' '.join(f'{0.75 + t * (w - 1.5) / (len(s) - 1):.2f},{top + 1.8 + (1 - v[t]) * (band - 3.6):.2f}'
                       for t in range(len(s)))
        out.append(f'<polyline points="{pts}" fill="none" stroke="#3D3C39" stroke-width="0.75" '
                   'stroke-linejoin="round" stroke-linecap="round"/>')
    out.append('</svg>')
    return ''.join(out)


# low frequencies at the bottom; the clip's dB range (min..0) on the ramp, as librosa's specshow scales it; every mel
# bin 8 px tall and every frame 2 px wide
spec_png = png_uri(np.repeat(np.repeat(ramp((mel[::-1] - mel.min()) / -mel.min(), BLUES), 8, axis=0), 2, axis=1))

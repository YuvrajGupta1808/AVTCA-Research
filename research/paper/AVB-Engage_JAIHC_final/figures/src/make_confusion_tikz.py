"""Writes figures/fig_confusion.tex, the TikZ body of Fig. 3 (row-normalized test confusion matrices, pooled over the
three seeds), from the pooled counts in results/paper_springer/analysis.json (scripts/paper/springer_analysis.py).

Drawn at 84 x 45 mm (one Springer column) with 9-pt lettering, so it stays above 8 pt when the template scales it to its
76-mm review column. The two matrices share the row axis, which names the levels; cells use the ColorBrewer Blues ramp
of the Fig. 1 spectrogram (lightness falls monotonically, so the figure reads in grey) on a fixed 0-100 % scale, with
white numbers only where the cell is dark enough for a 4.5:1 contrast; the diagonal (correct level) is set in bold.
Panel titles take the colours of the network (blue) and of the fusion (orange) in Figs. 1 and 2.
Build: python3 make_confusion_tikz.py, then compile ../Fig3_source.tex with tectonic and copy the PDF to ../../Fig3.pdf.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OUT = HERE.parent / 'fig_confusion.tex'
conf = json.loads((ROOT / 'results/paper_springer/analysis.json').read_text())['pooled_confusion']
BLUES = ['#F7FBFF', '#DEEBF7', '#C6DBEF', '#9ECAE1', '#6BAED6', '#4292C6', '#2171B5', '#08519C', '#08306B']
NAMES = ['not', 'barely', 'engaged', 'highly']
PANELS = [('av', 'Audio-visual network', '1D5FAF'), ('fused', 'AVB-Engage', 'B4471A')]


def ramp(v):
    pos = min(max(v, 0.0), 1.0) * (len(BLUES) - 1)
    lo = min(int(pos), len(BLUES) - 2)
    f = pos - lo
    a, b = ([int(s[k:k + 2], 16) for k in (1, 3, 5)] for s in (BLUES[lo], BLUES[lo + 1]))
    return [round(x * (1 - f) + y * f) for x, y in zip(a, b)]


def luminance(rgb):
    c = [x / 255 for x in rgb]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


C, X1, GAP, Y0 = 0.74, 2.04, 0.44, -0.95       # cell (cm), left edge of the first matrix, gap, top of the matrices
out = ['% Fig. 3 -- written by figures/src/make_confusion_tikz.py from results/paper_springer/analysis.json (pooled counts',
       '% of the three seeds, 3 x 2,256 clips; row percentages rounded once). Do not edit by hand.',
       r'\begin{tikzpicture}[font=\small, cell/.style={minimum size=0.74cm, inner sep=0pt, draw=white, line width=0.8pt}]',
       r'\useasboundingbox (0,0) rectangle (8.4,-4.5);',
       r'\fill[white] (0,0) rectangle (8.4,-4.5);']   # white ground: the EPS bounding box then equals the page
for p, (key, title, colour) in enumerate(PANELS):
    x0 = X1 + p * (4 * C + GAP)
    counts = conf[key]['counts']
    out.append(rf'\node[font=\small\bfseries, text={{rgb,255:red,{int(colour[:2], 16)};green,{int(colour[2:4], 16)};'
               rf'blue,{int(colour[4:], 16)}}}] at ({x0 + 2 * C:.3f},-0.25) {{{title}}};')
    for j in range(4):
        out.append(rf'\node[text={{rgb,255:red,82;green,81;blue,78}}] at ({x0 + (j + 0.5) * C:.3f},-0.70) {{{j}}};')
    for i, row in enumerate(counts):
        y = Y0 - (i + 0.5) * C
        if p == 0:
            out.append(rf'\node[anchor=east, text={{rgb,255:red,82;green,81;blue,78}}] at (1.66,{y:.3f}) {{{NAMES[i]}}};')
            out.append(rf'\node at (1.84,{y:.3f}) {{{i}}};')
        for j, n in enumerate(row):
            v = round(100 * n / sum(row))
            rgb = ramp(v / 100)
            ink = 'white' if luminance(rgb) < 0.18 else '{rgb,255:red,11;green,11;blue,11}'
            num = rf'\textbf{{{v}}}' if i == j else str(v)
            out.append(rf'\node[cell, fill={{rgb,255:red,{rgb[0]};green,{rgb[1]};blue,{rgb[2]}}}, text={ink}] '
                       rf'at ({x0 + (j + 0.5) * C:.3f},{y:.3f}) {{{num}}};')
    out.append(rf'\node at ({x0 + 2 * C:.3f},-4.25) {{predicted level}};')
out.append(r'\node[anchor=east] at (1.98,-0.70) {true level};')          # row-axis title in the free corner
out.append(r'\end{tikzpicture}%')                                          # no trailing space in the box
OUT.write_text('\n'.join(out) + '\n')
print('wrote', OUT)

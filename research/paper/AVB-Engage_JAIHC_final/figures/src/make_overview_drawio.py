"""Writes figures/src/overview.drawio, the single-column overview (Fig. 1), editable in draw.io.

The data panels show one EngageNet test clip, loaded and checked in clip_panels.py (shared with Fig. 2): its log-mel
spectrogram, three of its OpenFace channels, the class probabilities of the three members and their average, and the
expected level cut at the validation thresholds of the seed-1 fusion. The face is a drawing, not the participant.

Drawn at its final size (Springer artwork rules, template manual Sec. 7.3): the canvas is 331 px wide, which draw.io's
PDF export prints as 84 mm (one column); text is 12.5 px (9 pt), small labels 11.5 px (8.3 pt). The spectrogram is the
only bitmap (lossless, 700-1000 dpi); everything else is vector. Needs the avtca environment (numpy, librosa, Pillow, scikit-learn).
Export (docs/plan.md Sec. 21.15): FONTCONFIG_FILE=<fonts.conf that maps Helvetica to FreeSans, the TrueType Nimbus Sans
clone> node export_drawio.js overview.drawio ../../Fig1.pdf (a CFF font such as TeX Gyre Heros would be embedded as
Type 3), then gs -sDEVICE=eps2write with Flate images for Fig1.eps.
"""
from pathlib import Path
from xml.sax.saxutils import escape

from clip_panels import LEVELS, TRACES, face_stack_svg, level, mu, probs, spec_png, svg_uri, th, traces_svg

OUT = Path(__file__).resolve().parent / 'overview.drawio'

# ---------------------------------------------------------------------------------------------- draw.io cells
FS, FS_S = 12.5, 11.5
INK, MUTED = '#0B0B0B', '#52514E'
C = {'net': ('#2A78D6', '#EAF2FC'), 'beh': ('#6E6D68', '#FFFFFF'), 'key': ('#EB6834', '#FDEDE6'), 'io': ('#6E6D68', '#FFFFFF')}
TITLE = {'net': '#1D5FAF', 'beh': '#5A5954', 'key': '#B4471A'}   # panel-title text, >= 4.5:1 on the panel fill
GROUP_FILL = {'net': '#F8FBFE', 'beh': '#FAFAF9', 'key': '#FFFBF9'}
BAR = {'AV': '#2A78D6', 'TF': '#6E6D68', 'GBM': '#6E6D68', 'p': '#EB6834'}
FONT = 'fontFamily=Helvetica;'
cells, n = [], [1]


def attr(s):
    return escape(s, {'"': '&quot;'})


def vertex(x, y, w, h, value, style):
    n[0] += 1
    cells.append(f'<mxCell id="o{n[0]}" value="{attr(value)}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/></mxCell>')
    return f'o{n[0]}'


def box(x, y, w, h, html, kind='net', bold=False):
    s, f = C[kind]
    return vertex(x, y, w, h, f'<b>{html}</b>' if bold else html,
                  f'rounded=1;arcSize=12;whiteSpace=wrap;html=1;spacing=2;strokeColor={s};fillColor={f};'
                  f'strokeWidth={2 if kind == "key" else 1};{FONT}fontSize={FS};fontColor={INK};')


def group(x, y, w, h, label, kind, valign='top'):
    s, _ = C[kind]
    vertex(x, y, w, h, label, f'rounded=1;arcSize=4;whiteSpace=wrap;html=1;dashed=1;strokeColor={s};'
           f'fillColor={GROUP_FILL[kind]};verticalAlign={valign};align=left;spacingLeft=6;spacingTop=1;spacingBottom=2;'
           f'{FONT}fontSize={FS};fontStyle=1;fontColor={TITLE[kind]};')


def text(x, y, w, h, html, size=FS, color=INK, align='center'):
    return vertex(x, y, w, h, html, f'text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;'
                  f'whiteSpace=nowrap;spacing=0;{FONT}fontSize={size};fontColor={color};')


def rect(x, y, w, h, fill, stroke='none', sw=1, value='', extra=''):
    return vertex(x, y, w, h, value, f'rounded=0;whiteSpace=wrap;html=1;spacing=0;fillColor={fill};strokeColor={stroke};'
                  f'strokeWidth={sw};{FONT}{extra}')


def image(x, y, w, h, uri):
    return vertex(x, y, w, h, '', f'shape=image;html=1;imageAspect=0;image={uri};')


def line(x1, y1, x2, y2, color=MUTED, sw=0.75):
    n[0] += 1
    cells.append(f'<mxCell id="o{n[0]}" style="endArrow=none;html=1;rounded=0;strokeColor={color};strokeWidth={sw};" '
                 f'edge="1" parent="1"><mxGeometry relative="1" as="geometry"><mxPoint x="{x1:.2f}" y="{y1:.2f}" '
                 f'as="sourcePoint"/><mxPoint x="{x2:.2f}" y="{y2:.2f}" as="targetPoint"/></mxGeometry></mxCell>')


def edge(a, b, exit_=None, entry=None, pts=()):
    st = 'edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endSize=5;strokeColor=#52514E;'
    if exit_:
        st += f'exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;'
    if entry:
        st += f'entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;'
    arr = ('<Array as="points">' + ''.join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in pts) + '</Array>') if pts else ''
    n[0] += 1
    cells.append(f'<mxCell id="o{n[0]}" style="{st}" edge="1" parent="1" source="{a}" target="{b}">'
                 f'<mxGeometry relative="1" as="geometry">{arr}</mxGeometry></mxCell>')


P = lambda s: f'<i>p</i><sub>{s}</sub>'   # noqa: E731

# (A) audio-visual network, (B) facial-behavior models, (C) late fusion and ordinal decoding. More room than the first
# 94-mm version (2026-11-14, Yuvraj: it looked cluttered; page 12 has space): 331 x 400 px = 84 x 102 mm. Keep the
# height at or below ~104 mm: from 106.5 mm on, "5 Results and discussion" no longer fits on page 6, the floats reflow
# and page 12 fills to 233 mm, eating the room the author block needs (height sweep, docs/plan.md Sec. 21.16).
HT, CY, HC = 232, 244, 156                # height of A and B; top and height of C
group(2, 2, 178, HT, '(A) Audio-visual network', 'net')
group(190, 2, 141, HT, '(B) Facial-behavior models', 'beh')
group(2, CY, 329, HC, '(C) Late fusion and ordinal decoding', 'key', valign='bottom')
IY, IH = 40, 50                           # the data panels of A and B share one row

# (A): inputs as data, then the network
C1, C2, CW = 8, 95, 79
text(C1, IY - 16, CW, 13, 'Face, 5 fps')
text(C2, IY - 16, CW, 13, 'Log-mel, 10 s')
face = image(C1, IY, CW, IH, svg_uri(face_stack_svg(CW, IH)))
spec = image(C2, IY, CW, IH, spec_png)
rect(C2, IY, CW, IH, 'none', '#6E6D68', 0.75)
eff = box(C1, 104, CW, 36, 'EfficientFace<br>+ Conv1D')
conv = box(C2, 104, CW, 36, 'Conv2D +<br>alignment')
xat = box(8, 154, 166, 36, 'Cross-modal attention,<br>three exchanges')
pool = box(8, 205, 166, 24, 'Attention pooling')
edge(face, eff)
edge(spec, conv)
edge(eff, xat, entry=(round((C1 + CW / 2 - 8) / 166, 3), 0))
edge(conv, xat, entry=(round((C2 + CW / 2 - 8) / 166, 3), 0))
edge(xat, pool)

# (B): OpenFace channels as data, then the two models on segment statistics
BX, BW, LW = 197, 127, 30
band = (IH - 2) / len(TRACES)
rect(BX, IY, BW, IH, '#FFFFFF', '#6E6D68', 0.75)
for k, (name, _) in enumerate(TRACES):
    text(BX + 2, IY + 1 + band * k, LW - 2, band, name, size=FS_S, align='right')
image(BX + LW + 3, IY + 1, BW - LW - 5, IH - 2, svg_uri(traces_svg(BW - LW - 5, IH - 2)))
of = text(BX, IY + IH + 2, BW, 13, 'OpenFace, 30 fps')
seg = box(BX, 115, BW, 36, '20 segments &#215;<br>(mean, std)', 'beh')
tf = box(BX, 160, 94, 36, 'Segment<br>Transformer', 'beh')
gb = box(BX, 205, BW, 24, 'Gradient boosting', 'beh')
edge(of, seg)
edge(seg, tf, exit_=(round(47 / BW, 3), 1), entry=(0.5, 0))
edge(seg, gb, exit_=(round(110 / BW, 3), 1), entry=(round(110 / BW, 3), 0))

# (C) row 1: 0.6 p_AV + 0.2 p_TF + 0.2 p_GBM = p, as the clip's probability bars for levels 0-3
BASE, BH, BAR_W, BAR_GAP = CY + 62, 36, 11, 5   # bar baseline, height of probability 1, bar width and gap
CENTER = {'AV': 37.5, 'TF': 122.5, 'GBM': 207.5, 'p': 292.5}
LABEL = {'AV': f'0.6 {P("AV")}', 'TF': f'0.2 {P("TF")}', 'GBM': f'0.2 {P("GBM")}', 'p': '<i>p</i>'}
half = (4 * BAR_W + 3 * BAR_GAP) / 2
lab = {}
for k, c in CENTER.items():
    lab[k] = text(c - 30, CY + 8, 60, 14, LABEL[k])
    for j in range(4):
        x = c - half + j * (BAR_W + BAR_GAP)
        h = BH * float(probs[k][j])
        rect(x, BASE - h, BAR_W, h, BAR[k])
        text(x + BAR_W / 2 - 7, BASE + 2, 14, 12, str(j), size=FS_S, color=MUTED)
    line(c - half - 2, BASE, c + half + 2, BASE)
for x, op in ((80, '+'), (165, '+'), (250, '=')):
    text(x - 9, BASE - BH / 2 - 9, 18, 18, op, size=16)
gap_y = 2 + HT + (CY - 2 - HT) / 2         # the free channel between the top panels and C
edge(pool, lab['AV'], exit_=(round((CENTER['AV'] - 8) / 166, 3), 1), entry=(0.5, 0))
edge(tf, lab['TF'], exit_=(0, 0.5), entry=(0.5, 0), pts=((185, 178), (185, gap_y), (CENTER['TF'], gap_y)))
edge(gb, lab['GBM'], exit_=(round((CENTER['GBM'] - BX) / BW, 3), 1), entry=(0.5, 0))

# (C) row 2: the expected level mu on the ordered scale, cut at the validation thresholds tau_1..tau_3
G0, G1, GY, GH = 8, 238, CY + 100, 16
X = lambda m: G0 + (G1 - G0) * m / 3     # noqa: E731
bounds = [0.0, *th, 3.0]
scale = [rect(X(bounds[j]), GY, X(bounds[j + 1]) - X(bounds[j]), GH, '#F6B79B' if j == level else '#FFFFFF', '#B4471A',
              1.5 if j == level else 0.75, value=str(j),
              extra=f'fontSize={FS_S};fontColor={INK};fontStyle={1 if j == level else 0};') for j in range(4)]
for m, t in enumerate(th, 1):
    line(X(t), GY + GH, X(t), GY + GH + 3)
    text(X(t) - 9, GY + GH + 3, 18, 13, f'<i>&#964;</i><sub>{m}</sub>', size=FS_S)
vertex(X(mu) - 4.5, GY - 8, 9, 7, '', 'triangle;whiteSpace=wrap;html=1;direction=south;fillColor=#B4471A;strokeColor=none;')
text(X(mu) - 92, GY - 17, 85, 13, 'Expected level', size=FS_S, color=MUTED, align='right')
text(X(mu) + 8, GY - 17, 60, 13, f'<i>&#956;</i> = {mu:.2f}', align='left')
out = box(250, GY - 8, 75, 32, f'Level {level},<br>{LEVELS[level]}', 'io', True)
edge(scale[-1], out, exit_=(1, 0.5), entry=(0, 0.5))

OUT.write_text('<mxfile host="drawio" agent="make_overview_drawio.py"><diagram id="overview" name="Overview"><mxGraphModel '
               'grid="1" gridSize="10" page="0" math="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + ''.join(cells) +
               '</root></mxGraphModel></diagram></mxfile>\n')
print('wrote', OUT, f'({OUT.stat().st_size / 1024:.0f} KB)')

"""Writes figures/src/architecture.drawio, the editable draw.io source of the architecture figure (Fig. 2).

Redesigned 2026-11-14 (docs/plan.md Sec. 21.18) after a survey of the architecture figures in papers/ (AVT-CA,
GLP-fusion, DeepAVFusion, MBT, CORE-Net, DSTC, CMOSE, FoleyGAN) and of published diagram guidance: the two streams run
as lanes, their token sequences are strips of squares (one square per eight steps, so the lengths show the 107 audio
steps pooled onto the 50 video steps and the 12 against 50 tokens after the second stage), encoders are trapezoids,
attention blocks share one colour and take q from their own stream and k, v from the other, residual additions are
(+) nodes, and the inputs and outputs are the thumbnails and probability bars of the example clip of Fig. 1.
Token shapes (length x width, batch omitted) are those of a forward pass of models/multimodal_cnn.py::MultiModalCNN
(fusion='it', it_fusion_mode='modern', num_heads=8, 4 classes) on one 431-frame mel clip and 50 face frames, i.e. the
configuration of the reported runs (results/fullclip/A_av_s*).

Drawn at its final size (Springer artwork rules, template manual Sec. 7.3): 686 x 502 px = 174 x 128 mm, text 12.5 px
(9 pt), annotations 11.5 px (8.3 pt). Keep the footprint and the caption length: the figure sits on page 4, and one
more caption line costs 50 mm on page 12 (docs/plan.md Sec. 21.17). Needs the avtca environment (clip_panels.py).
Export: FONTCONFIG_FILE=<fonts.conf that maps Helvetica to FreeSans> node export_drawio.js architecture.drawio
../../Fig2.pdf, then gs -sDEVICE=eps2write.
"""
from pathlib import Path
from xml.sax.saxutils import escape

from clip_panels import face_row_svg, probs, spec_png, svg_uri, th, traces_svg

OUT = Path(__file__).resolve().parent / 'architecture.drawio'
FS, FS_S = 12.5, 11.5
INK, MUTED, LINE = '#0B0B0B', '#52514E', '#52514E'
VID, VID_F = '#2A78D6', '#EAF2FC'         # video stream
AUD, AUD_F = '#1B9E77', '#E4F4EE'         # audio stream
BEH, BEH_F = '#6E6D68', '#F4F3F0'         # facial behavior
ATT, ATT_F = '#B8860B', '#FFF5D6'         # attention blocks
KEY, KEY_F = '#EB6834', '#FDEDE6'         # components evaluated in Section 5 (star)
TITLE = {'net': '#1D5FAF', 'beh': '#5A5954', 'key': '#B4471A'}   # panel titles, >= 4.5:1 on the panel fills
STAR = '<font face="DejaVu Sans">&#9733;</font>'                # FreeSans has no star
P = lambda s: f'<i>p</i><sub>{s}</sub>'   # noqa: E731
cells, _id = [], [1]


def attr(s):
    return escape(s, {'"': '&quot;'})


def nid():
    _id[0] += 1
    return f'n{_id[0]}'


def vertex(x, y, w, h, value, style):
    i = nid()
    cells.append(f'<mxCell id="{i}" value="{attr(value)}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/></mxCell>')
    return i


def font(size=FS, color=INK, bold=False):
    return f'fontFamily=Helvetica;fontSize={size};fontColor={color};fontStyle={1 if bold else 0};'


def box(x, y, w, h, html, stroke, fill, bold=False, sw=1, dashed=False, size=FS):
    return vertex(x, y, w, h, html, f'rounded=1;arcSize=14;whiteSpace=wrap;html=1;spacing=2;strokeColor={stroke};'
                  f'fillColor={fill};strokeWidth={sw};dashed={int(dashed)};' + font(size, INK, bold))


def trap(x, y, w, h, html, stroke, fill, inset=16):
    """Encoder: a trapezoid that narrows downwards (wide input, compact features)."""
    return vertex(x, y, w, h, html, f'shape=trapezoid;perimeter=trapezoidPerimeter;fixedSize=1;size={inset};flipV=1;'
                  f'whiteSpace=wrap;html=1;spacing=1;strokeColor={stroke};fillColor={fill};strokeWidth=1;' + font())


def text(x, y, w, h, html, size=FS, color=INK, align='center', bold=False, fill='none'):
    return vertex(x, y, w, h, html, f'text;html=1;strokeColor=none;fillColor={fill};align={align};verticalAlign=middle;'
                  'whiteSpace=nowrap;spacing=0;' + font(size, color, bold))


def group(x, y, w, h, label, kind, valign='top'):
    stroke, fill = {'net': (VID, '#F8FBFE'), 'beh': (BEH, '#FAFAF9'), 'key': (KEY, '#FFFBF9')}[kind]
    vertex(x, y, w, h, label, f'rounded=1;arcSize=3;whiteSpace=wrap;html=1;dashed=1;strokeColor={stroke};fillColor={fill};'
           f'verticalAlign={valign};align=left;spacingLeft=8;spacingTop=2;spacingBottom=3;' + font(FS, TITLE[kind], True))


def image(x, y, w, h, uri, frame=None):
    i = vertex(x, y, w, h, '', f'shape=image;html=1;imageAspect=0;image={uri};')
    if frame:
        vertex(x, y, w, h, '', f'rounded=0;html=1;fillColor=none;strokeColor={frame};strokeWidth=0.75;')
    return i


def plus(cx, cy, d=12):
    """Residual addition (+)."""
    return vertex(cx - d / 2, cy - d / 2, d, d, '', 'shape=orEllipse;perimeter=ellipsePerimeter;html=1;fillColor=#FFFFFF;'
                  f'strokeColor={LINE};strokeWidth=1;')


def anchor(x, y, w, h):
    """An invisible cell to attach arrows to (token strips, bar charts)."""
    return vertex(x, y, w, h, '', 'rounded=0;html=1;fillColor=none;strokeColor=none;')


def tokens(cx, y, steps, color, per=8, size=5.5, gap=1.0):
    """A token sequence of `steps` steps: one square per `per` steps, centred on cx; returns (anchor, x0, x1)."""
    n = max(1, round(steps / per))
    w = n * (size + gap) - gap
    x0 = cx - w / 2
    for k in range(n):
        vertex(x0 + k * (size + gap), y, size, size, '', f'rounded=0;html=1;fillColor={color};strokeColor=none;')
    return anchor(x0, y, w, size), x0, x0 + w


def bars(cx, base, p, color, bh=22, bw=8, gap=3):
    """The four class probabilities as bars on a baseline; returns (anchor, x0, x1)."""
    w = 4 * bw + 3 * gap
    x0 = cx - w / 2
    for j in range(4):
        h = bh * float(p[j])
        vertex(x0 + j * (bw + gap), base - h, bw, h, '', f'rounded=0;html=1;fillColor={color};strokeColor=none;')
    arrow(x0 - 2, base, x0 + w + 2, base, end=False)
    return anchor(x0, base - bh, w, bh), x0, x0 + w


def arrow(x1, y1, x2, y2, pts=(), dashed=False, end=True, color=LINE):
    """A free arrow (not attached to cells), optionally through waypoints."""
    st = (f'html=1;rounded=0;strokeColor={color};strokeWidth=1;endSize=5;endArrow={"block" if end else "none"};'
          f'dashed={int(dashed)};edgeStyle=none;')
    arr = ('<Array as="points">' + ''.join(f'<mxPoint x="{px:.2f}" y="{py:.2f}"/>' for px, py in pts) + '</Array>') if pts else ''
    cells.append(f'<mxCell id="{nid()}" style="{st}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                 f'<mxPoint x="{x1:.2f}" y="{y1:.2f}" as="sourcePoint"/><mxPoint x="{x2:.2f}" y="{y2:.2f}" as="targetPoint"/>'
                 f'{arr}</mxGeometry></mxCell>')


# ------------------------------------------------------------------------------------------------- panels
group(2, 2, 432, 500, '(A) Audio-visual cross-attention network', 'net')
group(442, 2, 244, 240, '(B) Facial-behavior models', 'beh')
group(442, 250, 244, 252, '(C) Late fusion and ordinal decoding', 'key', valign='bottom')
XV, XA = 112, 324                              # lane centres: video, audio
BW_A = 150                                     # attention-block width

# (A) inputs: thumbnails of the example clip, labelled above, shape on the outer side
text(XV - 98, 24, 196, 13, 'Face crops (MTCNN), 5 fps')
text(XA - 98, 24, 196, 13, 'Log-mel spectrogram')
text(XV - 98, 37, 196, 12, '50 &#215; 3 &#215; 224 &#215; 224', FS_S, MUTED)
text(XA - 98, 37, 196, 12, '64 mel bands &#215; 431 frames', FS_S, MUTED)
image(XV - 50, 52, 100, 32, svg_uri(face_row_svg(100, 32, n=4)))
image(XA - 50, 52, 100, 32, spec_png, frame='#6E6D68')
arrow(XV, 85, XV, 96)
arrow(XA, 85, XA, 96)

# encoders
trap(XV - 92, 96, 184, 30, 'EfficientFace per frame,<br>Conv1D &#215;2', VID, VID_F)
trap(XA - 92, 96, 184, 30, 'Conv2D &#215;2,<br>mean over frequency', AUD, AUD_F)

# audio: 107 steps, pooled onto the 50 video steps
a1, a1x0, a1x1 = tokens(XA, 134, 107, AUD)
text(a1x1 + 4, 130, 60, 13, '107 &#215; 128', FS_S, MUTED, 'left')
arrow(XA, 126, XA, 133)
box(XA - 86, 144, 172, 18, f'{STAR} Per-sample alignment', KEY, KEY_F, bold=True, sw=1.5)
arrow(XA, 140, XA, 144)

# modality dropout on both token sequences (training only); its label sits between the lanes
vertex(24, 170, 392, 46, '', f'rounded=1;arcSize=16;html=1;dashed=1;fillColor=none;strokeColor={KEY};strokeWidth=1.5;')
text(XV + 8, 174, XA - XV - 16, 26, f'<b>{STAR} Modality dropout</b><br><i>p</i> = 0.15, training only')
v1, v1x0, v1x1 = tokens(XV, 204, 50, VID)
a2, a2x0, a2x1 = tokens(XA, 204, 50, AUD)
text(v1x0 - 64, 200, 60, 13, '50 &#215; 64', FS_S, MUTED, 'right')
text(a2x1 + 4, 200, 60, 13, '50 &#215; 128', FS_S, MUTED, 'left')
arrow(XV, 126, XV, 203)                        # video encoder output straight into the frame
arrow(XA, 162, XA, 203)


def exchange(y, label, residual, kv_from=None):
    """One cross-attention exchange: a block per lane fed by its own stream (q) and, drawn for the first exchange, by
    the other stream's tokens (k, v, dashed); with a residual, a (+) node below each block and a skip line around it."""
    out = {}
    for lane, x in (('v', XV), ('a', XA)):
        box(x - BW_A / 2, y, BW_A, 20, label, ATT, ATT_F)
        if residual:
            plus(x, y + 34)
            arrow(x, y + 20, x, y + 28)
            side = -1 if lane == 'v' else 1
            sx = x + side * (BW_A / 2 + 9)
            arrow(x, y - 12, sx, y + 34, pts=((sx, y - 12),), end=False)    # branch above the block ...
            arrow(sx, y + 34, x + side * 6, y + 34)                         # ... into the (+) node
            out[lane] = y + 40
        else:
            out[lane] = y + 20
    if kv_from:
        (vx1, vy), (ax0, ay) = kv_from
        arrow(vx1 + 2, vy, XA - BW_A / 2 + 22, y, dashed=True)
        arrow(ax0 - 2, ay, XV + BW_A / 2 - 22, y, dashed=True)
        text(XV + 18, y - 14, 30, 12, '<i>k</i>, <i>v</i>', FS_S, MUTED)          # beside, not on, each dashed arrow
        text(XA - 44, y - 14, 30, 12, '<i>k</i>, <i>v</i>', FS_S, MUTED)
    return out


# exchange 1 on the aligned, dropped-out tokens
arrow(XV, 210, XV, 240)
arrow(XA, 210, XA, 240)
text(XV + 4, 214, 12, 12, '<i>q</i>', FS_S, MUTED, 'left')
text(XA - 16, 214, 12, 12, '<i>q</i>', FS_S, MUTED, 'right')
e1 = exchange(240, 'Cross-attention 1', True, kv_from=((v1x1, 210), (a2x0, 210)))

# stage 2: convolutions; audio max-pooled to 12 tokens
arrow(XV, e1['v'], XV, 292)
arrow(XA, e1['a'], XA, 292)
trap(XV - 70, 292, 140, 20, 'Conv1D &#215;2', VID, VID_F, inset=12)
trap(XA - 80, 292, 160, 20, 'Conv1D &#215;2, max-pool', AUD, AUD_F, inset=12)
v2, v2x0, v2x1 = tokens(XV, 320, 50, VID)
a3, a3x0, a3x1 = tokens(XA, 320, 12, AUD)
text(v2x0 - 64, 316, 60, 13, '50 &#215; 128', FS_S, MUTED, 'right')
text(a3x1 + 4, 316, 60, 13, '12 &#215; 128', FS_S, MUTED, 'left')
arrow(XV, 312, XV, 319)
arrow(XA, 312, XA, 319)

# exchanges 2 (with residual) and 3 (without)
arrow(XV, 326, XV, 344)
arrow(XA, 326, XA, 344)
e2 = exchange(344, 'Cross-attention 2', True)
arrow(XV, e2['v'], XV, 394)
arrow(XA, e2['a'], XA, 394)
e3 = exchange(394, 'Cross-attention 3', False)

# pooling and the network head
arrow(XV, e3['v'], XV, 424)
arrow(XA, e3['a'], XA, 424)
box(XV - 70, 424, 140, 18, 'Attention pooling', LINE, '#FFFFFF')
box(XA - 70, 424, 140, 18, 'Attention pooling', LINE, '#FFFFFF')
vertex(XV - 16, 450, 32, 5.5, '', f'rounded=0;html=1;fillColor={VID};strokeColor=none;')
vertex(XA - 16, 450, 32, 5.5, '', f'rounded=0;html=1;fillColor={AUD};strokeColor=none;')
text(XV - 82, 446, 60, 13, '128', FS_S, MUTED, 'right')
text(XA + 22, 446, 60, 13, '128', FS_S, MUTED, 'left')
arrow(XV, 442, XV, 449)
arrow(XA, 442, XA, 449)
arrow(XV, 456, XV, 466)
arrow(XA, 456, XA, 466)
box(XV - 56, 466, XA - XV + 112, 22, 'Concatenate (256), linear 256&#8594;4, softmax', LINE, '#FFFFFF')
pav, pav_x0, pav_x1 = bars(406, 486, probs['AV'], VID)
text(pav_x0 - 4, 448, 48, 13, P('AV'))
arrow(XA + 56, 477, pav_x0 - 3, 477)

# (B) OpenFace channels -> 20 segment tokens -> two models
image(452, 30, 86, 38, svg_uri(traces_svg(86, 38)), frame='#6E6D68')
text(544, 27, 136, 26, 'OpenFace 2.2:<br>17 AU, 2 gaze, 3 pose', FS, INK, 'left')
text(544, 55, 136, 13, 'per frame, <i>T</i> &#215; 22', FS_S, MUTED, 'left')
arrow(495, 69, 495, 80)
box(452, 80, 228, 20, '20 segments &#215; [mean, std]', BEH, BEH_F)
seg, sx0, sx1 = tokens(566, 108, 20, BEH, per=1)
text(sx1 + 4, 104, 50, 13, '20 &#215; 44', FS_S, MUTED, 'left')
arrow(566, 100, 566, 107)
arrow(540, 114, 506, 128)
arrow(592, 114, 626, 128)
box(452, 128, 108, 46, 'Segment<br>Transformer<br>2 layers, <i>d</i> = 64', BEH, '#FFFFFF')
box(572, 128, 108, 46, 'Gradient<br>boosting<br>3 fits, 924 inputs', BEH, '#FFFFFF')
ptf, ptf_x0, ptf_x1 = bars(506, 210, probs['TF'], BEH)
pgb, pgb_x0, pgb_x1 = bars(626, 210, probs['GBM'], BEH)
arrow(506, 174, 506, 186)
arrow(626, 174, 626, 186)
text(506 - 24, 212, 48, 14, P('TF'))
text(626 - 24, 212, 48, 14, P('GBM'))

# (C) late fusion, expected level, the scale cut at the validation thresholds, the level
fus = box(452, 262, 222, 34, f'<b>{STAR} Late fusion</b><br><i>p</i> = 0.6 {P("AV")} + 0.2 {P("TF")} + 0.2 {P("GBM")}',
          KEY, KEY_F, sw=2)
arrow(506, 226, 506, 262)
arrow(626, 226, 626, 262)
arrow(pav_x1 + 3, 477, 438, 279, pts=((438, 477),), end=False)
arrow(438, 279, 452, 279)
pfu, pfu_x0, pfu_x1 = bars(563, 334, probs['p'], KEY)
text(pfu_x1 + 6, 320, 20, 14, '<i>p</i>', FS, INK, 'left')
arrow(563, 296, 563, 309)
box(452, 344, 222, 34, f'<b>{STAR} Expected level</b> <i>&#956;</i> = &#931;<sub><i>k</i></sub> <i>k p</i><sub><i>k</i></sub><br>'
    'cut by <i>&#964;</i><sub>1</sub> &lt; <i>&#964;</i><sub>2</sub> &lt; <i>&#964;</i><sub>3</sub> (validation)',
    KEY, KEY_F, sw=2)
arrow(563, 334, 563, 344)
GX0, GX1, GY, GH = 452, 674, 392, 14
X = lambda m: GX0 + (GX1 - GX0) * m / 3       # noqa: E731
bounds = [0.0, *th, 3.0]
for j in range(4):
    vertex(X(bounds[j]), GY, X(bounds[j + 1]) - X(bounds[j]), GH, str(j), 'rounded=0;whiteSpace=wrap;html=1;spacing=0;'
           f'fillColor=#FFFFFF;strokeColor=#B4471A;strokeWidth=0.75;' + font(FS_S))
for m, t in enumerate(th, 1):
    arrow(X(t), GY + GH, X(t), GY + GH + 3, end=False)
    text(X(t) - 9, GY + GH + 3, 18, 13, f'<i>&#964;</i><sub>{m}</sub>', FS_S)
arrow(563, 378, 563, 391)
arrow(563, 406, 563, 432)
box(486, 432, 154, 22, f'<b>Level <i>&#375;</i> &#8712; {{0, 1, 2, 3}}</b>', LINE, '#FFFFFF')

OUT.write_text('<mxfile host="drawio" agent="make_architecture_drawio.py"><diagram id="arch" name="Architecture">'
               '<mxGraphModel grid="1" gridSize="10" page="0" math="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
               + ''.join(cells) + '</root></mxGraphModel></diagram></mxfile>\n')
print('wrote', OUT, f'({len(cells)} cells)')

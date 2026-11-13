"""Writes figures/src/architecture.drawio, the editable draw.io source of the architecture figure (Fig. 2).

Every tensor shape is the output of a forward pass of models/multimodal_cnn.py::MultiModalCNN (fusion='it',
it_fusion_mode='modern', num_heads=8, 4 classes) on one 431-frame mel clip and 50 face frames, i.e. the configuration
of the reported runs (results/fullclip/A_av_s*).

Drawn at its final size (Springer artwork rules, template manual Sec. 7.3): the canvas is 686 px wide, which draw.io's
PDF export prints as 174 mm (full page width), and all text is 12.5 px, which prints at 9 pt (8.3 pt in the template's
160-mm review layout). The legend lives in the caption, so no connector crosses text. Export: node export.js
architecture.drawio ../../Fig2.pdf (headless draw.io export, docs/plan.md Sec. 21.13), then gs -sDEVICE=eps2write.
"""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent / 'architecture.drawio'
FS = 12.5
BLUE, BLUE_FILL = '#2A78D6', '#EAF2FC'
ORANGE, ORANGE_FILL = '#EB6834', '#FDEDE6'
GREY = '#6E6D68'
TITLE = {BLUE: '#1D5FAF', ORANGE: '#B4471A', GREY: '#5A5954'}   # panel-title text, >= 4.5:1 on the panel fill
P = lambda s: f'<i>p</i><sub>{s}</sub>'
cells, _id = [], [1]


def attr(s):
    return escape(s, {'"': '&quot;'})


def nid():
    _id[0] += 1
    return f'n{_id[0]}'


def mono(s):
    return f'<font face="DejaVu Sans Mono" color="#52514E">{s}</font>'


def box(x, y, w, h, title, *lines, kind='net', bold=False):
    stroke, fill = {'net': (BLUE, BLUE_FILL), 'key': (ORANGE, ORANGE_FILL), 'beh': (GREY, '#FFFFFF'), 'io': (GREY, '#FFFFFF')}[kind]
    label = (f'<b>{title}</b>' if bold else title) + ''.join('<br>' + ln for ln in lines)
    i = nid()
    cells.append(f'<mxCell id="{i}" value="{attr(label)}" style="rounded=1;arcSize=10;whiteSpace=wrap;html=1;spacing=2;'
                 f'strokeColor={stroke};fillColor={fill};strokeWidth={2 if kind == "key" else 1};fontFamily=Helvetica;'
                 f'fontSize={FS};fontColor=#0B0B0B;" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i


def group(x, y, w, h, label, stroke, fill, valign='top'):
    cells.append(f'<mxCell id="{nid()}" value="{attr(label)}" style="rounded=1;arcSize=3;whiteSpace=wrap;html=1;dashed=1;'
                 f'strokeColor={stroke};fillColor={fill};verticalAlign={valign};align=left;spacingLeft=8;spacingTop=2;'
                 f'spacingBottom=3;fontFamily=Helvetica;fontSize={FS};fontStyle=1;fontColor={TITLE.get(stroke, stroke)};" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def edge(a, b, dashed=False, straight=False, exit_=None, entry=None, pts=()):
    style = ('edgeStyle=none;' if straight else 'edgeStyle=orthogonalEdgeStyle;') + \
        'rounded=0;html=1;endArrow=block;endSize=5;strokeColor=#52514E;'
    if dashed:
        style += 'dashed=1;'
    if exit_:
        style += f'exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;'
    if entry:
        style += f'entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;'
    geo = '<mxGeometry relative="1" as="geometry">'
    if pts:
        geo += '<Array as="points">' + ''.join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in pts) + '</Array>'
    cells.append(f'<mxCell id="{nid()}" style="{style}" edge="1" parent="1" source="{a}" target="{b}">{geo}</mxGeometry></mxCell>')


# ---------------------------------------------------------------- layout (px)
VX, AX, W, H = 10, 222, 204, 36          # video column, audio column, box width, two-line box height
ROW = {0: 30, 1: 76, 2: 122, 3: 168, 4: 228, 5: 274, 6: 320, 7: 366, 8: 412, 9: 458}   # extra gap 3-4 for the dashed X
BX, BW = 450, 222                         # behavior / fusion column

group(2, 2, 432, 500, '(A) Audio-visual cross-attention network', BLUE, '#F8FBFE')
group(442, 2, 242, 240, '(B) Facial-behavior models', GREY, '#FAFAF9')
group(442, 250, 242, 252, '(C) Late fusion and ordinal decoding', ORANGE, '#FFFBF9', valign='bottom')

# (A) two streams, every shape from a forward pass
v = [box(VX, ROW[0], W, H, 'Face crops, MTCNN, 5 fps', mono('(B, 50, 3, 224, 224)'), kind='io'),
     box(VX, ROW[1], W, H, 'EfficientFace per frame', '&#8594; ' + mono('(B, 50, 1024)')),
     box(VX, ROW[2], W, H, 'Stage-1 Conv1D &#215;2', mono('V&#8321; (B, 64, 50)')),
     box(VX, ROW[3], W, H, '&#9733; Modality dropout', '<i>p</i> = 0.15, training only', kind='key', bold=True),
     box(VX, ROW[4], W, H, 'va1: video queries audio', '+ residual &#8594; ' + mono('(B, 50, 64)')),
     box(VX, ROW[5], W, H, 'Stage-2 Conv1D &#215;2', mono('V&#8322; (B, 128, 50)')),
     box(VX, ROW[6], W, H, 'MHA: video queries audio', '+ residual &#8594; ' + mono('(B, 50, 128)')),
     box(VX, ROW[7], W, H, 'Final cross-attention', 'no residual &#8594; ' + mono('(B, 50, 128)')),
     box(VX, ROW[8], W, H, 'Masked attention pooling', mono('(B, 128)'))]
a = [box(AX, ROW[0], W, H, 'Log-mel spectrogram', mono('(B, 64, 431)'), kind='io'),
     box(AX, ROW[1], W, H, 'Conv2D &#215;2, mean over freq.', mono('A&#8321; (B, 128, 107)')),
     box(AX, ROW[2], W, H, '&#9733; Per-sample alignment', mono('&#195;&#8321; (B, 128, 50)'), kind='key', bold=True),
     box(AX, ROW[3], W, H, '&#9733; Modality dropout', '<i>p</i> = 0.15, training only', kind='key', bold=True),
     box(AX, ROW[4], W, H, 'av1: audio queries video', '+ residual &#8594; ' + mono('(B, 50, 128)')),
     box(AX, ROW[5], W, H, 'Stage-2 Conv1D &#215;2, MaxPool', mono('A&#8322; (B, 128, 12)')),
     box(AX, ROW[6], W, H, 'MHA: audio queries video', '+ residual &#8594; ' + mono('(B, 12, 128)')),
     box(AX, ROW[7], W, H, 'Final cross-attention', 'no residual &#8594; ' + mono('(B, 12, 128)')),
     box(AX, ROW[8], W, H, 'Masked attention pooling', mono('(B, 128)'))]
cat = box(VX, ROW[9], 416, H, 'Concatenate z, Linear 256&#8594;4, softmax', mono('z (B, 256)') + f' &#8594; {P("AV")} ' + mono('(B, 4)'))

# (B) behavior models on segment statistics
b0 = box(BX, 30, BW, 50, 'OpenFace 2.2, per frame', mono('(B, T, 22)'), '17 AU, 2 gaze, 3 pose', kind='io')
b1 = box(BX, 90, BW, H, '20 segments &#215; [mean, std]', mono('(B, 20, 44)'), kind='beh')
b2 = box(BX, 136, BW, H, 'Segment Transformer', f'2 layers, d = 64 &#8594; {P("TF")} ' + mono('(B, 4)'), kind='beh')
b3 = box(BX, 182, BW, 50, 'Gradient boosting, 3 fits', mono('(B, 924)') + ': 20 &#215; 44 + 44', f'&#8594; {P("GBM")} ' + mono('(B, 4)'), kind='beh')

# (C) late fusion and ordinal decoding
mix = box(BX, 270, BW, H, '&#9733; Late fusion', f'<i>p</i> = 0.6 {P("AV")} + 0.2 {P("TF")} + 0.2 {P("GBM")}', kind='key', bold=True)
dec = box(BX, 330, BW, 50, '&#9733; Expected level',
          '<i>&#956;</i> = &#931;<sub><i>k</i></sub> <i>k</i> <i>p</i><sub><i>k</i></sub>',
          'cut by <i>&#964;</i><sub>1</sub> &lt; <i>&#964;</i><sub>2</sub> &lt; <i>&#964;</i><sub>3</sub> (validation)', kind='key', bold=True)
out = box(BX, 392, BW, 24, 'Level <i>&#375;</i> &#8712; {0, 1, 2, 3}', kind='io', bold=True)

# edges
for col in (v, a):
    for s_, t_ in zip(col, col[1:]):
        edge(s_, t_)
edge(v[3], a[4], dashed=True, straight=True, exit_=(0.85, 1), entry=(0.15, 0))      # keys/values from the other stream
edge(a[3], v[4], dashed=True, straight=True, exit_=(0.15, 1), entry=(0.85, 0))
edge(v[8], cat, exit_=(0.5, 1), entry=(round((VX + W / 2 - VX) / 416, 3), 0))
edge(a[8], cat, exit_=(0.5, 1), entry=(round((AX + W / 2 - VX) / 416, 3), 0))
edge(cat, mix, exit_=(1, 0.5), entry=(0, 0.5), pts=((438, ROW[9] + H / 2), (438, 270 + H / 2)))
edge(b0, b1)
edge(b1, b2)
edge(b1, b3, exit_=(1, 0.5), entry=(1, 0.5), pts=((679, 90 + H / 2), (679, 182 + 25)))
edge(b2, mix, exit_=(0, 0.5), entry=(0.09, 0), pts=((446, 136 + H / 2), (446, 262), (BX + 0.09 * BW, 262)))
edge(b3, mix, exit_=(0.5, 1), entry=(0.5, 0))
edge(mix, dec)
edge(dec, out)

xml = ('<mxfile host="drawio" agent="make_architecture_drawio.py"><diagram id="arch" name="Architecture">'
       '<mxGraphModel grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
       'page="0" pageScale="1" math="0" shadow="0"><root>'
       '<mxCell id="0"/><mxCell id="1" parent="0"/>' + ''.join(cells) + '</root></mxGraphModel></diagram></mxfile>\n')
OUT.write_text(xml)
print('wrote', OUT)

"""Writes figures/src/overview.drawio, the single-column model overview (Fig. 1), editable in draw.io.

Drawn at its final size (Springer artwork rules, template manual Sec. 7.3): the canvas is 331 px wide, which
draw.io's PDF export prints as 84 mm (one column), and all text is 12.5 px, which prints at 9 pt (8.2 pt in the
template's 76-mm review column). Export: node export.js overview.drawio ../../Fig1.pdf (headless draw.io export,
docs/plan.md Sec. 21.13), then gs -sDEVICE=eps2write for Fig1.eps.
"""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent / 'overview.drawio'
FS = 12.5
C = {'net': ('#2A78D6', '#EAF2FC'), 'beh': ('#6E6D68', '#FFFFFF'), 'key': ('#EB6834', '#FDEDE6'), 'io': ('#6E6D68', '#FFFFFF')}
TITLE = {'net': '#1D5FAF', 'beh': '#5A5954', 'key': '#B4471A'}   # panel-title text, >= 4.5:1 on the panel fill
GROUP_FILL = {'net': '#F8FBFE', 'beh': '#FAFAF9', 'key': '#FFFBF9'}
P = lambda s: f'<i>p</i><sub>{s}</sub>'          # p_AV, p_TF, p_GBM with a real subscript
cells, n = [], [1]


def attr(s):
    return escape(s, {'"': '&quot;'})


def nid():
    n[0] += 1
    return f'o{n[0]}'


def box(x, y, w, h, html, kind='net', bold=False):
    s, f = C[kind]
    i = nid()
    v = f'<b>{html}</b>' if bold else html
    cells.append(f'<mxCell id="{i}" value="{attr(v)}" style="rounded=1;arcSize=12;whiteSpace=wrap;html=1;spacing=2;'
                 f'strokeColor={s};fillColor={f};strokeWidth={2 if kind == "key" else 1};fontFamily=Helvetica;'
                 f'fontSize={FS};fontColor=#0B0B0B;" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i


def group(x, y, w, h, label, kind, valign='top'):
    s, _ = C[kind]
    cells.append(f'<mxCell id="{nid()}" value="{attr(label)}" style="rounded=1;arcSize=4;whiteSpace=wrap;html=1;dashed=1;'
                 f'strokeColor={s};fillColor={GROUP_FILL[kind]};verticalAlign={valign};align=left;spacingLeft=6;spacingTop=1;'
                 f'spacingBottom=2;fontFamily=Helvetica;fontSize={FS};fontStyle=1;fontColor={TITLE[kind]};" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def edge(a, b, exit_=None, entry=None, pts=()):
    st = 'edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endSize=5;strokeColor=#52514E;'
    if exit_:
        st += f'exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;'
    if entry:
        st += f'entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;'
    arr = ('<Array as="points">' + ''.join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in pts) + '</Array>') if pts else ''
    cells.append(f'<mxCell id="{nid()}" style="{st}" edge="1" parent="1" source="{a}" target="{b}">'
                 f'<mxGeometry relative="1" as="geometry">{arr}</mxGeometry></mxCell>')


# (A) audio-visual network, (B) facial-behavior models, (C) late fusion and ordinal decoding
group(2, 2, 186, 236, '(A) Audio-visual network', 'net')
group(194, 2, 135, 236, '(B) Facial-behavior models', 'beh')
group(2, 246, 327, 104, '(C) Late fusion and ordinal decoding', 'key', valign='bottom')

face = box(8, 38, 86, 36, 'Face frames,<br>5 fps', 'io')
audio = box(98, 38, 86, 36, 'Audio,<br>log-mel', 'io')
eff = box(8, 84, 86, 50, 'EfficientFace<br>+ Conv1D')
conv = box(98, 84, 86, 50, 'Conv2D,<br>aligned to<br>video clock')
xat = box(8, 146, 176, 36, 'Cross-modal attention,<br>three exchanges')
pool = box(8, 194, 176, 24, f'Attention pooling &#8594; {P("AV")}')

of = box(202, 44, 116, 36, 'OpenFace AU,<br>gaze, head pose', 'io')
seg = box(202, 90, 116, 36, '20 segments &#215;<br>(mean, std)', 'beh')
tf = box(202, 136, 116, 50, f'Segment<br>Transformer<br>&#8594; {P("TF")}', 'beh')
gb = box(202, 196, 116, 36, f'Gradient<br>boosting &#8594; {P("GBM")}', 'beh')

wav = box(8, 256, 315, 24, f'Weighted average of {P("AV")}, {P("TF")}, {P("GBM")}', 'key', True)
exp = box(8, 292, 226, 36, 'Expected level, thresholds<br>fitted on validation', 'key', True)
out = box(242, 292, 81, 36, 'Level<br>0&#8211;3', 'io', True)

for s_, t_ in [(face, eff), (audio, conv), (xat, pool), (of, seg), (seg, tf), (exp, out)]:
    edge(s_, t_)
edge(eff, xat, entry=(0.25, 0))
edge(conv, xat, entry=(0.75, 0))
edge(pool, wav, exit_=(0.5, 1), entry=(round((96 - 8) / 315, 3), 0))
edge(seg, gb, exit_=(1, 0.5), entry=(1, 0.5), pts=((323.5, 108), (323.5, 214)))
edge(tf, wav, exit_=(0, 0.5), entry=(round((198 - 8) / 315, 3), 0), pts=((198, 161),))
edge(gb, wav, exit_=(0.5, 1), entry=(round((260 - 8) / 315, 3), 0))
edge(wav, exp, exit_=(round((121 - 8) / 315, 3), 1), entry=(0.5, 0))

OUT.write_text('<mxfile host="drawio" agent="make_overview_drawio.py"><diagram id="overview" name="Overview"><mxGraphModel '
               'grid="1" gridSize="10" page="0" math="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + ''.join(cells) +
               '</root></mxGraphModel></diagram></mxfile>\n')
print('wrote', OUT)

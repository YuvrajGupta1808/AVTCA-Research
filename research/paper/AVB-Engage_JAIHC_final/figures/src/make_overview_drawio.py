"""Writes figures/src/overview.drawio — the single-column model overview (Fig. 2), editable in draw.io."""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent / 'overview.drawio'
C = {'net': ('#2A78D6', '#EAF2FC'), 'beh': ('#6E6D68', '#FFFFFF'), 'key': ('#EB6834', '#FDEDE6'), 'io': ('#6E6D68', '#FFFFFF')}
cells, n = [], [1]


def attr(s):
    return escape(s, {'"': '&quot;'})


def nid():
    n[0] += 1
    return f'o{n[0]}'


def box(x, y, w, h, label, kind='net', bold=False):
    s, f = C[kind]
    i = nid()
    v = f'<b>{escape(label)}</b>' if bold else escape(label)
    cells.append(f'<mxCell id="{i}" value="{attr(v)}" style="rounded=1;arcSize=12;whiteSpace=wrap;html=1;strokeColor={s};fillColor={f};'
                 f'strokeWidth={2 if kind == "key" else 1};fontFamily=Helvetica;fontSize=11;fontColor=#0B0B0B;" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i


def group(x, y, w, h, label, kind):
    s, _ = C[kind]
    fill = {'net': '#F8FBFE', 'beh': '#FAFAF9', 'key': '#FFFBF9'}[kind]
    i = nid()
    cells.append(f'<mxCell id="{i}" value="{attr(label)}" style="rounded=1;arcSize=4;whiteSpace=wrap;html=1;dashed=1;strokeColor={s};'
                 f'fillColor={fill};verticalAlign=top;align=left;spacingLeft=6;fontFamily=Helvetica;fontSize=11;fontStyle=1;fontColor={s};"'
                 f' vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def edge(a, b, exit_=None, entry=None, pts=()):
    i = nid()
    st = 'edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endSize=5;strokeColor=#52514E;'
    if exit_:
        st += f'exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;'
    if entry:
        st += f'entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;'
    arr = ('<Array as="points">' + ''.join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in pts) + '</Array>') if pts else ''
    cells.append(f'<mxCell id="{i}" style="{st}" edge="1" parent="1" source="{a}" target="{b}"><mxGeometry relative="1" as="geometry">{arr}</mxGeometry></mxCell>')


group(10, 10, 280, 250, '(A) Audio-visual cross-attention network', 'net')
group(300, 10, 150, 250, '(B) Behavior statistics', 'beh')
group(10, 270, 440, 110, '(C) Late fusion and ordinal decoding', 'key')
v0 = box(20, 38, 125, 36, 'Face frames, 5 fps', 'io')
a0 = box(155, 38, 125, 36, 'Audio log-mel', 'io')
v1 = box(20, 88, 125, 36, 'EfficientFace + Conv1D')
a1 = box(155, 88, 125, 36, 'Conv2D + alignment to video clock')
x = box(20, 140, 260, 36, 'Cross-modal attention, three exchanges')
p = box(20, 192, 260, 36, 'Attention pooling → p_AV')
o0 = box(310, 38, 130, 36, 'OpenFace AU, gaze, head pose', 'io')
o1 = box(310, 88, 130, 36, '20 segments × [mean, std]', 'beh')
tf = box(310, 140, 130, 36, 'Segment Transformer → p_TF', 'beh')
gb = box(310, 192, 130, 36, 'Gradient boosting → p_GBM', 'beh')
mix = box(20, 300, 200, 40, 'Weighted average of p_AV, p_TF, p_GBM', 'key', True)
dec = box(240, 300, 200, 40, 'Expected level, thresholds fitted on validation', 'key', True)
out = box(240, 348, 200, 26, 'Engagement level 0–3', 'io', True)
for s_, t_ in [(v0, v1), (a0, a1), (o0, o1), (o1, tf), (mix, dec), (dec, out)]:
    edge(s_, t_)
edge(v1, x, entry=(0.25, 0))
edge(a1, x, entry=(0.75, 0))
edge(x, p)
edge(p, mix, exit_=(0.5, 1), entry=(0.35, 0))
edge(o1, gb, exit_=(1, 0.5), entry=(1, 0.5))
edge(tf, mix, exit_=(0, 0.5), entry=(0.75, 0), pts=((295, 158), (295, 285), (170, 285)))
edge(gb, mix, exit_=(0.5, 1), entry=(0.92, 0), pts=((375, 292), (204, 292)))
OUT.write_text('<mxfile host="drawio" agent="make_overview_drawio.py"><diagram id="overview" name="Overview"><mxGraphModel grid="1" gridSize="10" '
               'page="0" math="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + ''.join(cells) + '</root></mxGraphModel></diagram></mxfile>\n')
print('wrote', OUT)

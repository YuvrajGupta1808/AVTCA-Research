"""Writes figures/architecture.drawio — the editable draw.io source of the architecture figure.

Every tensor shape is the output of a forward pass of models/multimodal_cnn.py::MultiModalCNN
(fusion='it', it_fusion_mode='modern', num_heads=8, 4 classes) on one 431-frame mel clip and 50 face
frames, i.e. the configuration of the reported runs (results/fullclip/A_av_s*). Re-run this script
after changing the model, or edit architecture.drawio directly in draw.io and stop using the script.
"""
from pathlib import Path
from xml.sax.saxutils import escape


def attr(s):
    return escape(s, {'"': '&quot;'})

OUT = Path(__file__).resolve().parents[1] / 'architecture.drawio'

BLUE, BLUE_FILL = '#2A78D6', '#EAF2FC'
ORANGE, ORANGE_FILL = '#EB6834', '#FDEDE6'
GREY, GREY_FILL = '#6E6D68', '#F3F2EF'
FONT = 'fontFamily=Helvetica;fontSize=11;fontColor=#0B0B0B;'

cells, _id = [], [1]


def nid():
    _id[0] += 1
    return f'n{_id[0]}'


def box(x, y, w, h, title, shape='', kind='net', bold=False):
    stroke, fill = {'net': (BLUE, BLUE_FILL), 'key': (ORANGE, ORANGE_FILL), 'beh': (GREY, '#FFFFFF'),
                    'io': (GREY, '#FFFFFF')}[kind]
    width = 2 if kind == 'key' else 1
    label = f'<b>{escape(title)}</b>' if bold else escape(title)
    if shape:
        label += f'<br><font face="DejaVu Sans Mono" style="font-size:10px" color="#52514E">{escape(shape)}</font>'
    i = nid()
    cells.append(f'<mxCell id="{i}" value="{attr(label)}" style="rounded=1;arcSize=10;whiteSpace=wrap;html=1;'
                 f'strokeColor={stroke};fillColor={fill};strokeWidth={width};{FONT}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i


def group(x, y, w, h, label, stroke, fill, valign='top'):
    i = nid()
    cells.append(f'<mxCell id="{i}" value="{attr(label)}" style="rounded=1;arcSize=3;whiteSpace=wrap;html=1;dashed=1;'
                 f'strokeColor={stroke};fillColor={fill};verticalAlign={valign};align=left;spacingLeft=8;spacingTop=2;spacingBottom=4;'
                 f'fontFamily=Helvetica;fontSize=11;fontStyle=1;fontColor={stroke};" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def text(x, y, w, h, s, color='#52514E', size=10, align='left'):
    i = nid()
    cells.append(f'<mxCell id="{i}" value="{attr(s)}" style="text;html=1;whiteSpace=wrap;align={align};verticalAlign=middle;'
                 f'fontFamily=Helvetica;fontSize={size};fontColor={color};" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def edge(a, b, label='', dashed=False, exit_=None, entry=None, pts=()):
    i = nid()
    style = 'edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endSize=5;strokeColor=#52514E;'
    style += 'fontFamily=Helvetica;fontSize=9;fontColor=#52514E;labelBackgroundColor=#FFFFFF;'
    if dashed:
        style += 'dashed=1;'
    if exit_:
        style += f'exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;'
    if entry:
        style += f'entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;'
    geo = '<mxGeometry relative="1" as="geometry">'
    if pts:
        geo += '<Array as="points">' + ''.join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in pts) + '</Array>'
    geo += '</mxGeometry>'
    cells.append(f'<mxCell id="{i}" value="{attr(label)}" style="{style}" edge="1" parent="1" source="{a}" target="{b}">{geo}</mxCell>')


# ---------------------------------------------------------------- layout (px)
VX, AX, W = 40, 300, 230          # video column, audio column, box width
BX, BW = 610, 260                 # behavior column
H, R = 40, 56                     # box height, row pitch
y = lambda r: 40 + r * R

group(20, 18, 530, 598, '(A) Audio-visual cross-attention network', BLUE, '#F8FBFE')
group(590, 18, 300, 272, '(B) Facial-behavior statistics models', GREY, '#FAFAF9')
group(20, 628, 870, 148, '(A) ordinal head  ·  (C) late fusion and calibrated ordinal decoding', ORANGE, '#FFFBF9', valign='bottom')

# inputs
v0 = box(VX, y(0.3), W, H, 'Face crops, MTCNN, 5 fps', '(B, 50, 3, 224, 224)', 'io')
a0 = box(AX, y(0.3), W, H, 'Log-mel spectrogram, 64 bins', '(B, 64, 431)', 'io')
b0 = box(BX, y(0.3), BW, H, 'OpenFace 2.2 per source frame', '(B, T, 22): 17 AU, gaze 2, pose 3', 'io')

# (B) video encoder, (A) audio encoder
v1 = box(VX, y(1.3), W, H, 'EfficientFace, AffectNet-7 init.', 'per frame → (B, 50, 1024)')
v2 = box(VX, y(2.3), W, H, 'Stage-1 Conv1D ×2, 1024→64→64', 'V₁ (B, 64, 50)')
a1 = box(AX, y(1.3), W, H, 'Conv2D block ×2, MaxPool 2×2', '(B, 128, 16, 107)')
a2 = box(AX, y(2.3), W, H, 'Mean over frequency', 'A₁ (B, 128, 107)')
a3 = box(AX, y(3.3), W, H, '★ Per-sample adaptive alignment', 'Ã₁ (B, 128, 50)', 'key', bold=True)

# (C) fusion
mdv = box(VX, y(4.3), W, H - 6, '★ Modality dropout p = 0.15 (train)', '', 'key', bold=True)
mda = box(AX, y(4.3), W, H - 6, '★ Modality dropout p = 0.15 (train)', '', 'key', bold=True)
va1 = box(VX, y(5.2), W, H, 'va1: video queries audio, + residual', '(B, 50, 64)')
av1 = box(AX, y(5.2), W, H, 'av1: audio queries video, + residual', '(B, 50, 128)')
v3 = box(VX, y(6.2), W, H, 'Stage-2 Conv1D ×2, 64→128→128', 'V₂ (B, 128, 50)')
a4 = box(AX, y(6.2), W, H, 'Stage-2 Conv1D ×2, MaxPool ×2', 'A₂ (B, 128, 12)')
v4 = box(VX, y(7.2), W, H, 'Cross MHA: video queries audio', '+ residual → (B, 50, 128)')
a5 = box(AX, y(7.2), W, H, 'Cross MHA: audio queries video', '+ residual → (B, 12, 128)')
v5 = box(VX, y(8.2), W, H, 'Final cross-attention block V→A', '(B, 50, 128)')
a6 = box(AX, y(8.2), W, H, 'Final cross-attention block A→V', '(B, 12, 128)')
v6 = box(VX, y(9.2), W, H - 6, 'Masked attention pooling', '(B, 128)')
a7 = box(AX, y(9.2), W, H - 6, 'Masked attention pooling', '(B, 128)')

# (D) head + fusion + decoding
cat = box(VX, y(10.75), 260, H, 'Concatenate z → Linear 256→4, softmax', 'z (B, 256) → p_AV (B, 4)')
mix = box(330, y(10.75), 280, H, '★ Late fusion, fixed weights', 'p = 0.6 p_AV + 0.2 p_TF + 0.2 p_GBM', 'key', bold=True)
dec = box(640, y(10.75), 230, H, '★ Expected level μ = Σ k·p_k', 'cut by τ₁<τ₂<τ₃ (fit on val.)', 'key', bold=True)
out = box(640, y(11.75), 230, H - 8, 'Engagement level ŷ ∈ {0, 1, 2, 3}', '', 'io', bold=True)

# (E) behavior members
b1 = box(BX, y(1.3), BW, H, '20 uniform segments × [mean, std]', '(B, 20, 44)', 'beh')
b2 = box(BX, y(2.5), BW, H, 'Segment Transformer, 2 layers, d 64', '→ p_TF (B, 4)', 'beh')
b3 = box(BX, y(3.7), BW, H, 'Flatten + clip mean/std → (B, 924)', 'gradient boosting ×3 fits → p_GBM (B, 4)', 'beh')

# edges
for s_, t_ in [(v0, v1), (v1, v2), (a0, a1), (a1, a2), (a2, a3), (va1, v3), (av1, a4), (v3, v4), (a4, a5),
               (v4, v5), (a5, a6), (v5, v6), (a6, a7), (b0, b1), (mix, dec), (dec, out)]:
    edge(s_, t_)
edge(v2, mdv)
edge(a3, mda)
edge(mdv, va1)
edge(mda, av1)
edge(mdv, av1, dashed=True, exit_=(1, 0.5), entry=(0, 0.3), pts=((291, y(4.3) + 17), (291, y(5.2) + 12)))
edge(mda, va1, dashed=True, exit_=(0, 0.5), entry=(1, 0.7), pts=((279, y(4.3) + 17), (279, y(5.2) + 28)))
edge(v6, cat, exit_=(0.5, 1), entry=(0.3, 0))
edge(a7, cat, exit_=(0.5, 1), entry=(0.85, 0))
edge(cat, mix)
edge(b1, b2)
edge(b1, b3, exit_=(1, 0.5), entry=(1, 0.5), pts=((890 - 8, y(1.3) + 20), (890 - 8, y(3.7) + 20)))
edge(b2, mix, exit_=(0, 0.5), entry=(0.35, 0), pts=((570, y(2.5) + 20), (570, y(10.1)), (428, y(10.1))))
edge(b3, mix, exit_=(0.5, 1), entry=(0.75, 0), pts=((BX + BW / 2, y(10.3)), (540, y(10.3))))

text(606, 318, 280, 150,
     '★ orange: the components evaluated in Section VI.\n'
     'Shapes are traced from a forward pass of the trained configuration (B = batch). '
     'Audio: 22.05 kHz, full clip (≤ 10.06 s). Heads h = 8 in every attention block. '
     'Every attention block in (A) is cross-modal: the other stream supplies keys and values (dashed arrows, drawn for the first exchange).', size=10)

xml = ('<mxfile host="drawio" agent="make_architecture_drawio.py"><diagram id="arch" name="Architecture">'
       '<mxGraphModel dx="900" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
       'page="0" pageScale="1" pageWidth="910" pageHeight="920" math="0" shadow="0"><root>'
       '<mxCell id="0"/><mxCell id="1" parent="0"/>' + ''.join(cells) + '</root></mxGraphModel></diagram></mxfile>\n')
OUT.write_text(xml)
print('wrote', OUT)

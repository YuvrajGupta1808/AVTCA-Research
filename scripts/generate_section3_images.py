from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "generated" / "section3_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TITLE_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BODY_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

COLORS = {
    "bg": "#ffffff",
    "panel": "#ffffff",
    "ink": "#1f1f1f",
    "muted": "#666666",
    "line": "#363636",
    "navy": "#15304f",
    "teal": "#2f7d7a",
    "orange": "#ec8b35",
    "olive": "#96b684",
    "sand": "#f4dfb8",
    "lav": "#d9d4ee",
    "bluefill": "#d8edf3",
    "greenfill": "#dcebd3",
    "orangefill": "#f8eadc",
    "purplefill": "#ece8f8",
    "white": "#ffffff",
}


def rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def round_box(draw: ImageDraw.ImageDraw, box, fill, outline=COLORS["line"], radius=24, width=3):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def fit_lines(draw: ImageDraw.ImageDraw, text: str, max_width: int, text_font) -> list[str]:
    paragraphs = text.split("\n")
    lines: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if draw.textbbox((0, 0), trial, font=text_font)[2] <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def fit_text(
    draw: ImageDraw.ImageDraw,
    box,
    text: str,
    font_path: str,
    max_size: int,
    min_size: int = 10,
    fill=COLORS["ink"],
    align="center",
    line_gap=6,
    bold=False,
):
    x0, y0, x1, y1 = box
    inner_w = max(int(x1 - x0 - 20), 10)
    inner_h = max(int(y1 - y0 - 12), 10)
    chosen_font = font(font_path, min_size)
    chosen_lines = [text]

    for size in range(max_size, min_size - 1, -1):
        trial_font = font(font_path, size)
        lines = fit_lines(draw, text, inner_w, trial_font)
        heights = [draw.textbbox((0, 0), line, font=trial_font)[3] for line in lines]
        total_h = sum(heights) + max(0, len(lines) - 1) * line_gap
        widest = max(draw.textbbox((0, 0), line, font=trial_font)[2] for line in lines)
        if widest <= inner_w and total_h <= inner_h:
            chosen_font = trial_font
            chosen_lines = lines
            break
        chosen_font = trial_font
        chosen_lines = lines

    heights = [draw.textbbox((0, 0), line, font=chosen_font)[3] for line in chosen_lines]
    total_h = sum(heights) + max(0, len(chosen_lines) - 1) * line_gap
    y = y0 + (y1 - y0 - total_h) / 2
    for idx, line in enumerate(chosen_lines):
        line_w = draw.textbbox((0, 0), line, font=chosen_font)[2]
        if align == "left":
            x = x0 + 10
        elif align == "right":
            x = x1 - line_w - 10
        else:
            x = x0 + (x1 - x0 - line_w) / 2
        draw.text((x, y), line, font=chosen_font, fill=fill)
        y += heights[idx] + line_gap


def arrow(draw: ImageDraw.ImageDraw, start, end, color, width=8, head=18):
    x0, y0 = start
    x1, y1 = end
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    angle = math.atan2(y1 - y0, x1 - x0)
    left = (
        x1 - head * math.cos(angle) + 0.55 * head * math.sin(angle),
        y1 - head * math.sin(angle) - 0.55 * head * math.cos(angle),
    )
    right = (
        x1 - head * math.cos(angle) - 0.55 * head * math.sin(angle),
        y1 - head * math.sin(angle) + 0.55 * head * math.cos(angle),
    )
    draw.polygon([(x1, y1), left, right], fill=color)


def waveform(draw: ImageDraw.ImageDraw, box):
    round_box(draw, box, COLORS["navy"], outline=COLORS["navy"], radius=16, width=2)
    x0, y0, x1, y1 = box
    mid = (y0 + y1) / 2
    pts = []
    span = int(x1 - x0)
    for i in range(span):
        t = i / max(span - 1, 1)
        y = mid + (
            math.sin(t * 7 * math.pi) * 12
            + math.sin(t * 21 * math.pi) * 4
            + math.cos(t * 3 * math.pi) * 3
        )
        pts.append((x0 + i, y))
    draw.line(pts, fill="#7dd6ff", width=4)
    for frac in (0.22, 0.46, 0.72):
        x = x0 + span * frac
        draw.line((x, y0 + 10, x, y1 - 10), fill="#28496a", width=2)


def spectrogram_bars(draw: ImageDraw.ImageDraw, box, palette):
    x0, y0, x1, y1 = box
    count = 12
    gap = 5
    width = (x1 - x0 - gap * (count - 1)) / count
    base = y1
    for i in range(count):
        frac = i / max(count - 1, 1)
        height = (0.28 + 0.58 * math.sin(0.8 + frac * 2.8)) * (y1 - y0)
        bx0 = x0 + i * (width + gap)
        by0 = base - height
        color = palette[i % len(palette)]
        draw.rectangle((bx0, by0, bx0 + width, base), fill=color)


def face_tiles(draw: ImageDraw.ImageDraw, box):
    x0, y0, x1, y1 = box
    colors = ["#dbeafe", "#d8f7df", "#fde68a"]
    gap = 12
    tile_w = (x1 - x0 - gap * 2) / 3
    for i in range(3):
        tx0 = x0 + i * (tile_w + gap)
        tx1 = tx0 + tile_w
        ty0 = y0 + 8 + (i % 2) * 4
        ty1 = y1 - 8 - (i % 2) * 4
        round_box(draw, (tx0, ty0, tx1, ty1), colors[i], outline="#667085", radius=12, width=2)
        cx = (tx0 + tx1) / 2
        cy = ty0 + (ty1 - ty0) * 0.42
        draw.ellipse((cx - 18, cy - 20, cx + 18, cy + 16), outline="#475467", width=2)
        draw.rectangle((cx - 24, cy + 20, cx + 24, cy + 34), outline="#475467", width=2)
        draw.line((cx - 26, cy + 42, cx + 26, cy + 42), fill="#475467", width=2)


def stack_tokens(draw: ImageDraw.ImageDraw, box):
    x0, y0, x1, y1 = box
    cols = [
        COLORS["bluefill"],
        COLORS["greenfill"],
        COLORS["orangefill"],
        COLORS["purplefill"],
    ]
    group_w = (x1 - x0) / 4
    for c, fill in enumerate(cols):
        gx = x0 + c * group_w + 8
        for r in range(3):
            w = group_w - 28
            h = 52
            top = y0 + 12 + r * 78 - c * 4
            poly = [
                (gx, top),
                (gx + w, top),
                (gx + w - 14, top + h),
                (gx - 14, top + h),
            ]
            draw.polygon(poly, fill=fill, outline="#7c8698")
            fit_text(
                draw,
                (gx + 10, top + 10, gx + w - 14, top + h - 6),
                f"l{r}",
                BODY_FONT,
                20,
                min_size=16,
            )


def pill(draw: ImageDraw.ImageDraw, box, text, fill, outline, text_color=COLORS["ink"], size=20):
    round_box(draw, box, fill, outline=outline, radius=16, width=2)
    fit_text(draw, box, text, BODY_FONT, size, min_size=14, fill=text_color)


def make_architecture(path: Path):
    img = Image.new("RGB", (2600, 1500), rgb(COLORS["bg"]))
    draw = ImageDraw.Draw(img)

    top_y0, top_y1 = 50, 138
    band_gap = 26
    band_w = (2600 - 180 - band_gap * 3) / 4
    bands = [
        ("Multimodal Input", COLORS["navy"]),
        ("Unimodal Encoding", COLORS["teal"]),
        ("Cross-Modal Fusion", COLORS["orange"]),
        ("Prediction Head", "#8a5214"),
    ]
    for idx, (label, fill) in enumerate(bands):
        x0 = 90 + idx * (band_w + band_gap)
        x1 = x0 + band_w
        round_box(draw, (x0, top_y0, x1, top_y1), fill, outline=fill, radius=24, width=2)
        fit_text(draw, (x0 + 20, top_y0 + 10, x1 - 20, top_y1 - 10), label, TITLE_FONT, 30, min_size=20, fill=COLORS["white"])
        if idx < len(bands) - 1:
            arrow(draw, (x1 + 10, 204), (x1 + 40, 204), COLORS["line"], width=10, head=20)

    left = (70, 190, 620, 1320)
    middle = (670, 190, 1690, 1320)
    right = (1740, 190, 2530, 1320)
    for box in (left, middle, right):
        round_box(draw, box, COLORS["panel"], radius=28)

    fit_text(draw, (100, 212, 590, 256), "Input Acquisition", TITLE_FONT, 30, min_size=22)
    fit_text(draw, (105, 254, 585, 292), "Synchronized audio clip and face-frame sequence.", BODY_FONT, 20, min_size=16, fill=COLORS["muted"])

    round_box(draw, (110, 320, 580, 500), COLORS["bluefill"], outline="#6cb6d8", radius=22)
    fit_text(draw, (140, 336, 550, 374), "Audio Window", TITLE_FONT, 26, min_size=20, fill=COLORS["navy"])
    waveform(draw, (155, 390, 535, 450))
    fit_text(draw, (140, 454, 550, 484), "3.6 s waveform -> mel spectrogram", BODY_FONT, 18, min_size=15, fill=COLORS["muted"])

    round_box(draw, (110, 540, 580, 740), "#eef8ff", outline="#6cb6d8", radius=22)
    fit_text(draw, (140, 554, 550, 592), "Video Frames", TITLE_FONT, 26, min_size=20, fill=COLORS["teal"])
    face_tiles(draw, (160, 620, 530, 692))
    fit_text(draw, (140, 696, 550, 726), "15 aligned face crops", BODY_FONT, 18, min_size=15, fill=COLORS["muted"])

    round_box(draw, (110, 780, 580, 930), COLORS["greenfill"], outline="#9fc18e", radius=22)
    fit_text(draw, (140, 794, 550, 832), "Sequence Metadata", TITLE_FONT, 26, min_size=20, fill="#45663e")
    fit_text(draw, (135, 840, 555, 910), "video mask | audio mask | valid lengths | labels", BODY_FONT, 22, min_size=15)

    round_box(draw, (110, 970, 580, 1200), COLORS["sand"], outline="#d3a254", radius=22)
    fit_text(draw, (138, 984, 552, 1024), "Preprocessing Summary", TITLE_FONT, 28, min_size=20, fill="#7b5315")
    fit_text(
        draw,
        (140, 1040, 550, 1168),
        "Crop or pad audio, extract face-centered frames, then package both modalities under the same temporal sample index.",
        BODY_FONT,
        22,
        min_size=16,
        align="left",
    )

    fit_text(draw, (710, 212, 1650, 256), "Encoding and Fusion Core", TITLE_FONT, 30, min_size=22)

    upper = (705, 290, 1655, 690)
    lower = (705, 730, 1655, 1288)
    round_box(draw, upper, "#fcfbfe", outline="#4b5563", radius=24)
    round_box(draw, lower, "#fcfbfe", outline="#4b5563", radius=24)

    audio_box = (735, 330, 1140, 648)
    visual_box = (1210, 330, 1625, 648)
    round_box(draw, audio_box, COLORS["sand"], outline="#d7a14e", radius=22)
    round_box(draw, visual_box, COLORS["lav"], outline="#9187c8", radius=22)

    fit_text(draw, (760, 344, 1115, 384), "Audio Branch", TITLE_FONT, 28, min_size=22, fill="#8a5c19")
    spectrogram_bars(draw, (785, 420, 890, 490), ["#3777f6", "#43c7ff", "#f7db2a"])
    spectrogram_bars(draw, (785, 510, 890, 580), ["#10b5a6", "#56d364", "#fff27a"])
    arrow(draw, (910, 500), (980, 500), COLORS["orange"], width=8, head=18)
    pill(draw, (980, 392, 1105, 460), "Conv2D stage 1", "#fff8ef", "#d7a14e", size=20)
    pill(draw, (980, 472, 1105, 540), "Temporal align", "#fff8ef", "#d7a14e", size=20)
    pill(draw, (980, 552, 1105, 620), "Conv1D stage 2", "#fff8ef", "#d7a14e", size=20)

    fit_text(draw, (1235, 344, 1600, 384), "Visual Branch", TITLE_FONT, 28, min_size=22, fill="#5a5297")
    face_tiles(draw, (1245, 415, 1380, 590))
    arrow(draw, (1400, 500), (1465, 500), "#7f71c2", width=8, head=18)
    pill(draw, (1465, 392, 1585, 460), "EfficientFace", "#f6f3ff", "#9e94d7", size=20)
    pill(draw, (1465, 472, 1585, 540), "Temporal Conv1D", "#f6f3ff", "#9e94d7", size=17)
    pill(draw, (1465, 552, 1585, 620), "Visual stage 2", "#f6f3ff", "#9e94d7", size=20)

    fit_text(draw, (750, 752, 1610, 792), "Mask-Aware Temporal Fusion", TITLE_FONT, 30, min_size=22)

    align_box = (740, 820, 1020, 1220)
    stack_box = (1050, 820, 1325, 1220)
    head_box = (1360, 820, 1620, 1220)
    round_box(draw, align_box, COLORS["greenfill"], outline="#94b882", radius=22)
    round_box(draw, stack_box, "#eef5ff", outline="#84afe0", radius=22)
    round_box(draw, head_box, COLORS["orangefill"], outline="#d69c5b", radius=22)

    fit_text(draw, (760, 836, 1000, 900), "Alignment and Masking", TITLE_FONT, 26, min_size=18, fill="#45663e")
    fit_text(
        draw,
        (765, 910, 995, 1170),
        "AdaptiveAvgPool1d aligns audio to valid visual length. Attention masks preserve only valid time steps. Modality dropout improves robustness.",
        BODY_FONT,
        20,
        min_size=16,
        align="left",
    )

    fit_text(draw, (1070, 836, 1305, 900), "Bidirectional\nFusion", TITLE_FONT, 24, min_size=16, fill="#345d92")
    pill(draw, (1080, 930, 1295, 1008), "av1 / va1", COLORS["white"], "#9cc1eb", size=22)
    pill(draw, (1080, 1020, 1295, 1098), "audioAttention\nvisualAttention", COLORS["white"], "#9cc1eb", size=15)
    pill(draw, (1080, 1110, 1295, 1180), "audioCrossAttention\nvisualCrossAttention", COLORS["white"], "#9cc1eb", size=12)

    fit_text(draw, (1380, 836, 1600, 900), "Pooling and Head", TITLE_FONT, 26, min_size=18, fill="#8a5c19")
    fit_text(
        draw,
        (1390, 910, 1590, 1170),
        "Attention pooling selects useful time steps. Pooled audio and video features are concatenated into a shared 256-d embedding, then classified.",
        BODY_FONT,
        20,
        min_size=16,
        align="left",
    )

    fit_text(draw, (1775, 212, 2495, 256), "Prediction and Design Notes", TITLE_FONT, 30, min_size=22)

    out_box = (1785, 310, 2485, 630)
    note_box = (1785, 670, 2485, 920)
    summary_box = (1785, 960, 2485, 1190)
    round_box(draw, out_box, "#f8fbff", outline="#88afe1", radius=24)
    round_box(draw, note_box, "#edd7e2", outline="#c78faa", radius=24)
    round_box(draw, summary_box, "#edf5e7", outline="#97b985", radius=24)

    fit_text(draw, (1810, 326, 2460, 366), "Classifier Output", TITLE_FONT, 28, min_size=22, fill=COLORS["navy"])
    labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    for idx, label in enumerate(labels):
        col = idx // 4
        row = idx % 4
        bx0 = 1835 + col * 280
        by0 = 410 + row * 50
        pill(draw, (bx0, by0, bx0 + 220, by0 + 34), label, COLORS["white"], "#abc6ea", size=18)

    fit_text(draw, (1810, 686, 2460, 726), "Implementation Notes", TITLE_FONT, 28, min_size=22, fill="#7a5368")
    fit_text(
        draw,
        (1820, 750, 2450, 880),
        "True cross-modal attention, residual adds with dropout, optional audio channel gate, and learned pooling instead of max pooling.",
        BODY_FONT,
        22,
        min_size=18,
        align="left",
    )

    fit_text(draw, (1810, 976, 2460, 1016), "Section 3 Narrative", TITLE_FONT, 28, min_size=22, fill="#4f7147")
    fit_text(
        draw,
        (1820, 1040, 2450, 1150),
        "Audio and video are encoded independently first, fused through three bidirectional stages, and pooled into a final multimodal prediction.",
        BODY_FONT,
        22,
        min_size=18,
        align="left",
    )

    arrow(draw, (620, 645), (670, 645), COLORS["orange"], width=10, head=20)
    arrow(draw, (1690, 1010), (1740, 1010), COLORS["orange"], width=10, head=20)
    img.save(path)


def make_pipeline(path: Path):
    img = Image.new("RGB", (2400, 1300), rgb(COLORS["bg"]))
    draw = ImageDraw.Draw(img)

    cols = [
        (80, 70, 500, 1120, COLORS["bluefill"], "1. Input Tensors"),
        (560, 70, 1040, 1120, COLORS["sand"], "2. Unimodal Encoders"),
        (1100, 70, 1600, 1120, COLORS["lav"], "3. Bidirectional Fusion"),
        (1660, 70, 2320, 1120, COLORS["greenfill"], "4. Output Head"),
    ]
    for x0, y0, x1, y1, fill, title in cols:
        round_box(draw, (x0, y0, x1, y1), fill, radius=28)
        fit_text(draw, (x0 + 20, y0 + 16, x1 - 20, y0 + 64), title, TITLE_FONT, 34, min_size=22)

    waveform(draw, (130, 330, 450, 395))
    fit_text(draw, (120, 410, 460, 470), "Audio clip -> mel spectrogram", BODY_FONT, 24, min_size=16)
    face_tiles(draw, (145, 560, 435, 650))
    fit_text(draw, (120, 680, 460, 730), "15 aligned face frames", BODY_FONT, 24, min_size=16)
    round_box(draw, (120, 840, 460, 1000), COLORS["white"], outline="#9fc0d2", radius=18)
    fit_text(draw, (145, 865, 435, 975), "Masks and valid lengths move with the batch so attention never uses padded time steps.", BODY_FONT, 24, min_size=16)

    spectrogram_bars(draw, (630, 340, 750, 400), ["#3777f6", "#43c7ff", "#f7db2a"])
    spectrogram_bars(draw, (630, 460, 750, 520), ["#10b5a6", "#56d364", "#fff27a"])
    spectrogram_bars(draw, (630, 580, 750, 640), ["#9747ff", "#e879f9", "#fb923c"])
    pill(draw, (790, 320, 980, 400), "AudioCNNPool stage 1", "#fff8ef", "#d6a056", size=22)
    pill(draw, (790, 430, 980, 510), "Adaptive alignment", "#fff8ef", "#d6a056", size=22)
    pill(draw, (790, 540, 980, 620), "AudioCNNPool stage 2", "#fff8ef", "#d6a056", size=22)
    face_tiles(draw, (620, 760, 790, 845))
    pill(draw, (825, 740, 1000, 820), "EfficientFace", "#f6f3ff", "#9e94d7", size=22)
    pill(draw, (825, 850, 1000, 930), "Temporal Conv1D", "#f6f3ff", "#9e94d7", size=22)
    round_box(draw, (620, 980, 990, 1075), COLORS["white"], outline="#c7baa3", radius=18)
    fit_text(draw, (645, 994, 965, 1061), "Two modality-specific\nrepresentations are built\nbefore fusion.", BODY_FONT, 20, min_size=12)

    stack_tokens(draw, (1170, 320, 1510, 590))
    fit_text(draw, (1155, 615, 1545, 700), "Intermediate AV attention and later cross-modal MHA let each modality query the other.", BODY_FONT, 24, min_size=16)
    round_box(draw, (1170, 760, 1515, 905), COLORS["white"], outline="#9e94d7", radius=18)
    fit_text(draw, (1195, 785, 1490, 880), "Residual add\nAttention dropout\nMask-aware reasoning", TITLE_FONT, 24, min_size=16)
    round_box(draw, (1170, 965, 1515, 1075), COLORS["white"], outline="#9e94d7", radius=18)
    fit_text(draw, (1190, 985, 1495, 1055), "Final audioCrossAttention\n+ visualCrossAttention", BODY_FONT, 22, min_size=14)

    round_box(draw, (1730, 320, 2250, 450), COLORS["white"], outline="#9ab88e", radius=18)
    fit_text(draw, (1755, 345, 2225, 425), "Attention pooling selects the most informative time steps from each modality.", BODY_FONT, 24, min_size=16)
    round_box(draw, (1730, 520, 2250, 650), COLORS["white"], outline="#9ab88e", radius=18)
    fit_text(draw, (1755, 545, 2225, 625), "Pooled audio and pooled video features are concatenated into a 256-d representation.", BODY_FONT, 24, min_size=16)
    round_box(draw, (1730, 760, 2250, 1020), COLORS["white"], outline="#9ab88e", radius=18)
    fit_text(draw, (1760, 785, 2220, 845), "Emotion classes", TITLE_FONT, 34, min_size=22)
    fit_text(draw, (1760, 865, 2220, 955), "neutral  calm  happy  sad  angry  fearful  disgust  surprised", BODY_FONT, 30, min_size=18)

    for y in (362, 600, 840):
        arrow(draw, (500, y), (560, y), COLORS["orange"], width=10, head=20)
        arrow(draw, (1040, y), (1100, y), COLORS["orange"], width=10, head=20)
        arrow(draw, (1600, y), (1660, y), COLORS["orange"], width=10, head=20)

    img.save(path)


def main():
    make_architecture(OUT_DIR / "avtca_section3_architecture.png")
    make_pipeline(OUT_DIR / "avtca_section3_pipeline.png")
    print("Created image set in", OUT_DIR)


if __name__ == "__main__":
    main()

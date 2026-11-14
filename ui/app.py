"""Streamlit UI for AVT-CA engagement detection (EngageNet-trained).

Upload any recording of a person on camera; the app slices it into 10-second
windows — the clip length the EngageNet models were trained on — and reports an
engagement level per window plus a session-level rollup.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import (  # noqa: E402
    ENGAGEMENT_LABELS,
    WINDOW_SECONDS,
    analyze_video,
    available_models,
    load_model,
    mtcnn_available,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEVEL_COLORS = ["#d94f4f", "#e2a03f", "#5aa9e6", "#3fa860"]
LEVEL_HELP = {
    "Not Engaged": "Looking away, eyes closing, no response to the material.",
    "Barely Engaged": "Present but drifting — long blinks, drooping head, minimal reaction.",
    "Engaged": "Attending to the material, steady gaze, normal responsiveness.",
    "Highly Engaged": "Leaning in, expressive, actively tracking and responding.",
}


st.set_page_config(page_title="AVT-CA Engagement Detection", layout="wide")


@st.cache_resource(show_spinner=False)
def get_model(checkpoint_path, model_key, device):
    spec = dict(available_models(REPO_ROOT)[model_key])
    return load_model(spec, device)


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fmt_time(seconds):
    return f"{int(seconds) // 60:d}:{int(seconds) % 60:02d}"


st.title("Engagement Detection")
st.caption(
    "Audio-Video Transformer with Cross-Attention · trained on EngageNet · "
    "4-level engagement scale"
)

models = available_models(REPO_ROOT)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model")

    if not models:
        st.error(
            "No EngageNet checkpoints found. Expected at least\n\n"
            "`results/exp2026/E04_ft10s_ord_lr5e5_e8/ENGAGENET_multimodal_cnn_15_best.pth`"
        )
        st.stop()

    model_key = st.selectbox(
        "Checkpoint",
        list(models.keys()),
        format_func=lambda key: models[key]["label"],
    )
    spec = models[model_key]

    metrics = spec["metrics"]
    col_a, col_b = st.columns(2)
    col_a.metric("Test top-1", f"{metrics['test_top1']:.2f}%")
    col_b.metric("Adjacent", f"{metrics['test_adjacent']:.2f}%")
    st.caption(
        f"Validation top-1 {metrics['val_top1']:.2f}% · macro-F1 {metrics['test_f1_macro']:.2f}%. "
        f"{spec['notes']}"
    )
    st.caption(
        "Decoded with the calibrated expected-value thresholds "
        f"`{spec['thresholds']}` from this run's calibration sweep — the same rule "
        "that produced the accuracy above."
    )

    st.divider()
    st.header("Analysis")
    device_choice = st.selectbox("Device", ["auto", "cuda", "cpu", "mps"])
    device = default_device() if device_choice == "auto" else device_choice

    window_seconds = st.slider(
        "Window length (s)", 4.0, 20.0, float(WINDOW_SECONDS), 1.0,
        help="EngageNet clips are 10 s. Moving away from 10 s drifts from the training contract.",
    )
    overlap = st.checkbox(
        "Overlapping windows (50% hop)", value=False,
        help="Doubles the number of windows for a smoother timeline, and doubles runtime.",
    )
    hop_seconds = window_seconds / 2 if overlap else window_seconds

    redetect_every = st.select_slider(
        "Face detection frequency", options=[1, 2, 4],
        value=1,
        format_func=lambda n: "every sampled frame (faithful)" if n == 1 else f"every {n}th",
        help="Frames are sampled at 5 fps to match the training arrays. Detecting on every "
             "sampled frame reproduces the training crops exactly; skipping reuses the "
             "previous box and is faster on CPU.",
    )
    modality_split = st.checkbox(
        "Compute audio-only / video-only scores", value=False,
        help="Runs two extra forward passes per window to show what each modality "
             "contributes. Roughly triples inference time.",
    )

    st.divider()
    st.caption(f"Device: `{device}`")

if not mtcnn_available():
    st.error(
        "`facenet_pytorch` is not importable, so face cropping falls back to a Haar cascade. "
        "The training arrays were built with MTCNN, and the fallback produces visibly "
        "different crops — predictions will not be trustworthy. Run the app inside the "
        "`avtca` conda environment:\n\n"
        "```\nconda activate avtca && streamlit run ui/app.py\n```"
    )


# ── Main area ─────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a video of yourself (or a student) in a session",
    type=["mp4", "avi", "mov", "mkv", "webm"],
)

if not uploaded:
    st.info(
        "Upload a recording to get started. Best results come from footage that matches "
        "the EngageNet setup: a single person, webcam framing, face visible, with the "
        "audio track intact."
    )
    with st.expander("What the four levels mean"):
        for label in ENGAGEMENT_LABELS:
            st.markdown(f"**{label}** — {LEVEL_HELP[label]}")
    st.stop()

st.video(uploaded)

if st.button("Analyse engagement", type="primary"):
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded.name)[1] or ".mp4",
                                     delete=False) as tmp:
        uploaded.seek(0)
        tmp.write(uploaded.read())
        video_path = tmp.name

    try:
        with st.spinner("Loading model weights…"):
            model = get_model(spec["checkpoint_path"], model_key, device)

        progress = st.progress(0.0, text="Starting…")
        result = analyze_video(
            video_path,
            model,
            spec,
            device,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
            redetect_every=redetect_every,
            compute_modality_split=modality_split,
            progress_cb=lambda frac, msg: progress.progress(frac, text=msg),
        )
        progress.empty()
        st.session_state.result = result
        st.session_state.result_model = spec["label"]
    except FileNotFoundError:
        st.error("**ffmpeg** not found. Install it with:\n```\nsudo apt install ffmpeg\n```")
        st.stop()
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)

result = st.session_state.get("result")
if not result:
    st.stop()

summary = result["summary"]
windows = result["windows"]

st.divider()

# ── Headline ──────────────────────────────────────────────────────────────────
level = summary["overall_level"]
st.markdown(
    f"<h2 style='color:{LEVEL_COLORS[level]};margin-bottom:0'>{summary['overall_label']}</h2>"
    f"<p style='opacity:0.7;margin-top:4px'>{LEVEL_HELP[summary['overall_label']]}</p>",
    unsafe_allow_html=True,
)

cols = st.columns(4)
cols[0].metric("Mean engagement score", f"{summary['mean_score']:.2f} / 3")
cols[1].metric("Time engaged", f"{summary['engaged_fraction'] * 100:.0f}%",
               help="Share of analysed time at level Engaged or Highly Engaged.")
cols[2].metric("Analysed", f"{fmt_time(summary['total_seconds'])} "
                           f"({summary['n_windows']} windows)")
cols[3].metric("Variability", f"±{summary['score_std']:.2f}",
               help="Standard deviation of the window scores. Low means a flat session.")

if not result["has_audio"]:
    st.warning(
        "This file has no usable audio track — the audio branch received silence, so the "
        "prediction is effectively video-only."
    )

low_face = [w for w in windows if w["face_rate"] < 0.5]
if low_face:
    st.warning(
        f"A face was detected in fewer than half the sampled frames for "
        f"{len(low_face)} of {len(windows)} windows. Those windows fell back to the full "
        "frame and their scores are unreliable."
    )

# ── Timeline ──────────────────────────────────────────────────────────────────
st.subheader("Engagement over time")

timeline = pd.DataFrame({
    "minute": [w["start"] / 60.0 for w in windows],
    "Engagement score": [w["score"] for w in windows],
})
if any("audio_only" in w for w in windows):
    timeline["Audio only"] = [w["audio_only"]["score"] for w in windows]
    timeline["Video only"] = [w["video_only"]["score"] for w in windows]
st.line_chart(timeline.set_index("minute"), height=280)
st.caption(
    "Expected class index (0 = Not Engaged … 3 = Highly Engaged) per window, "
    f"x-axis in minutes. Level boundaries for this model: {result['thresholds']}."
)

# ── Distribution ──────────────────────────────────────────────────────────────
left, right = st.columns([1, 1])

with left:
    st.subheader("Time per level")
    dist = pd.DataFrame({
        "Seconds": [summary["time_per_level"][label] for label in ENGAGEMENT_LABELS],
    }, index=ENGAGEMENT_LABELS)
    st.bar_chart(dist, height=260)

with right:
    st.subheader("Mean class probability")
    mean_probs = np.mean([w["probs"] for w in windows], axis=0)
    st.bar_chart(pd.DataFrame({"Probability": mean_probs}, index=ENGAGEMENT_LABELS), height=260)

# ── Highlights ────────────────────────────────────────────────────────────────
st.subheader("Highlights")
high, low = summary["highest"], summary["lowest"]
h_col, l_col = st.columns(2)
h_col.success(
    f"**Most engaged** · {fmt_time(high['start'])}–{fmt_time(high['end'])} — "
    f"{high['label']} (score {high['score']:.2f})"
)
l_col.error(
    f"**Least engaged** · {fmt_time(low['start'])}–{fmt_time(low['end'])} — "
    f"{low['label']} (score {low['score']:.2f})"
)

# ── Per-window table ──────────────────────────────────────────────────────────
with st.expander("Per-window detail"):
    rows = []
    for w in windows:
        row = {
            "Window": f"{fmt_time(w['start'])}–{fmt_time(w['end'])}",
            "Level": w["label"],
            "Score": round(w["score"], 3),
            "Confidence": round(w["confidence"], 3),
            "Face rate": round(w["face_rate"], 2),
            "Audio RMS": round(w["audio_rms"], 4),
            "Frames": w["n_frames"],
        }
        for i, label in enumerate(ENGAGEMENT_LABELS):
            row[f"P({label})"] = round(float(w["probs"][i]), 3)
        if "audio_only" in w:
            row["Audio-only level"] = w["audio_only"]["label"]
            row["Video-only level"] = w["video_only"]["label"]
        rows.append(row)
    table = pd.DataFrame(rows)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="engagement_windows.csv",
        mime="text/csv",
    )

if any("audio_only" in w for w in windows):
    st.subheader("Modality contribution")
    agree_video = np.mean([w["level"] == w["video_only"]["level"] for w in windows])
    agree_audio = np.mean([w["level"] == w["audio_only"]["level"] for w in windows])
    m1, m2 = st.columns(2)
    m1.metric("Agreement with video-only", f"{agree_video * 100:.0f}%")
    m2.metric("Agreement with audio-only", f"{agree_audio * 100:.0f}%")
    st.caption(
        "On EngageNet, audio alone is close to the majority-class predictor — the fused "
        "model is video-dominant, and low audio agreement here is expected, not a bug."
    )

st.caption(
    f"Model: {st.session_state.get('result_model', '')} · face detector: "
    f"{result['face_backend']} · source {result['video_info']['fps']:.1f} fps, "
    f"{fmt_time(result['video_info']['duration'])} long."
)

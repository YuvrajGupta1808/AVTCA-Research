import os
import sys
import tempfile

import streamlit as st
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import discover_checkpoints, load_model, preprocess_audio, preprocess_video, predict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="AVTCA Emotion Recognizer", layout="wide")
st.title("AVTCA Emotion Recognition")
st.caption("Audio-Video Transformer with Cross Attention · 8-class emotion detection")

# ── Sidebar: model configuration ──────────────────────────────────────────────
with st.sidebar:
    st.header("Model")

    known = discover_checkpoints(REPO_ROOT)
    if known:
        choice = st.selectbox("Available checkpoints", ["— select —"] + list(known.keys()))
        auto_path = known.get(choice, "")
    else:
        st.info("No checkpoints found under results*/ directories.")
        auto_path = ""

    custom_path = st.text_input("Or enter custom .pth path")
    pth_path = custom_path.strip() or auto_path

    st.divider()
    num_heads = st.selectbox("num_heads", [1, 2, 4, 8])
    fusion    = st.selectbox("Fusion type", ["it", "ia", "lt"])

    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"

    device_choice = st.selectbox("Device", ["auto", "cuda", "cpu", "mps"])
    device = default_device if device_choice == "auto" else device_choice

    st.divider()
    if st.button("Load Model", type="primary"):
        if not pth_path:
            st.error("Select or enter a checkpoint path.")
        elif not os.path.isfile(pth_path):
            st.error(f"File not found:\n`{pth_path}`")
        else:
            with st.spinner("Loading model weights…"):
                try:
                    model = load_model(pth_path, num_heads, fusion, device)
                    st.session_state.model  = model
                    st.session_state.device = device
                    st.success(f"Model ready on **{device}**")
                except Exception as exc:
                    st.error(f"Load failed: {exc}")

    if "model" in st.session_state:
        st.caption(f"Active · device: `{st.session_state.device}`")


# ── Main area: upload + predict ───────────────────────────────────────────────
uploaded = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded:
    st.video(uploaded)

    if st.button("Predict Emotion", type="primary"):
        if "model" not in st.session_state:
            st.warning("Load a model in the sidebar first.")
        else:
            # Write uploaded bytes to a temp file so cv2 / ffmpeg can read it
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                uploaded.seek(0)
                tmp.write(uploaded.read())
                video_path = tmp.name

            try:
                with st.status("Analysing video…", expanded=True) as status:
                    st.write("Extracting audio and computing MFCCs…")
                    audio = preprocess_audio(video_path)

                    st.write("Detecting faces and sampling frames…")
                    video = preprocess_video(video_path)

                    st.write("Running the model…")
                    probs = predict(
                        st.session_state.model,
                        audio,
                        video,
                        st.session_state.device,
                    )
                    status.update(label="Done!", state="complete")

                top_emotion = max(probs, key=probs.get)
                st.markdown(f"## {top_emotion.upper()}  ({probs[top_emotion]:.1%})")

                df = pd.DataFrame({"Probability": probs})
                st.bar_chart(df)

            except FileNotFoundError:
                st.error(
                    "**ffmpeg** not found. Install it with:\n"
                    "```\nsudo apt install ffmpeg\n```"
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                raise
            finally:
                os.unlink(video_path)

# DAiSEE Preprocessing

This folder contains the first-pass DAiSEE pipeline for the current AVT-CA codebase.

## Official Download

DAiSEE is officially distributed by IIT Hyderabad after form approval. The official flow is:

1. Submit the DAiSEE registration form from the official project page.
2. Wait for the approval email.
3. Use the direct archive link provided in that approval email.

To download and extract it into `datasets/DAISEE`, run:

```bash
export DAISEE_URL='PASTE_THE_APPROVAL_EMAIL_URL_HERE'
python preprocessing/daisee/download_daisee.py
```

Or pass it directly:

```bash
python preprocessing/daisee/download_daisee.py --url 'PASTE_THE_APPROVAL_EMAIL_URL_HERE'
```

The script keeps the official archive at:

`datasets/DAISEE/DAiSEE.zip`

and extracts the contents directly under:

`datasets/DAISEE/`

## What the scripts produce

The scripts keep the whole clip by default: every frame (cap them with `--target_fps`, `--frame_stride`
or `--max_frames`) and the whole audio track (cap it with `--max_video_seconds`, which keeps the first
N seconds). The training loader handles variable-length clips with `--full_video_preprocessing`; without
that flag it reduces every clip to 15 frames and a centred 3.6 s audio window, the RAVDESS contract.
The pipeline produces:

- `<clip>_facecroppad.npy`
- `<clip>_croppad.wav`
- `preprocessing/daisee/annotations_engagement.txt`

This lets us try DAiSEE with the existing audio-video architecture before the full engagement-specific HDF5 pipeline is built.

## Current Label Choice

DAiSEE has four affective labels per clip:

- `Boredom`
- `Engagement`
- `Confusion`
- `Frustration`

The current model only supports a single classification target, so this first pass uses:

- `Engagement` only
- classes `0..3` exactly as stored in the DAiSEE CSV labels

That means this is a temporary 4-class engagement experiment, not the final multi-head engagement/confusion system described in `docs/plan.md`.

## Preprocessing Steps

```bash
# 1. Extract mono audio at 22,050 Hz (whole clip; --max_video_seconds N keeps the first N seconds).
python preprocessing/daisee/extract_audios.py --data_root datasets/DAISEE

# 2. Extract face crops (224 x 224, BGR) from every frame; e.g. --target_fps 5 or --max_frames 15 to cap them.
python preprocessing/daisee/extract_faces.py --data_root datasets/DAISEE

# 3. Build an annotation file for the current architecture.
python preprocessing/daisee/create_annotations.py --data_root datasets/DAISEE
```

Or run the one-command wrapper:

```bash
python preprocessing/daisee/prepare_daisee.py --data_root datasets/DAISEE
```

## Notes

- DAiSEE audio is weak compared with discussion-heavy classroom data, but we still preserve it so the current multimodal stack can be tested end-to-end.
- The official archive may contain a small number of damaged clips. If decoding fails on an individual sample, exclude that sample rather than discarding the dataset.
- The official manifests list `9068` clips, but `Labels/AllLabels.csv` only labels `8925` of them. The annotation builder skips the `143` unlabeled manifest entries automatically.
- `extract_faces.py` needs `facenet_pytorch` (MTCNN) and stops without it; `--allow_opencv_fallback` lets it use OpenCV Haar face detection instead. A frame without a detected face is resized whole.
- DAiSEE was explored but is not used by the EngageNet paper; its derived annotation file is not part of the repository (regenerate it with step 3).
- The optional bounding-box annotation archive is not required for the current AVT-CA pipeline, so it is not part of the official download script here.

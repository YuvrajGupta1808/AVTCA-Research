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

## Why This Preprocessing Looks Like RAVDESS

The current training path expects:

- `15` visual frames per sample
- one cropped/padded audio waveform per sample
- one annotation file with `video_path;audio_path;label;split`

So this bootstrap DAiSEE pipeline produces:

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
# 1. Extract centered 3.6 s mono audio clips.
python preprocessing/daisee/extract_audios.py --data_root datasets/DAISEE

# 2. Extract 15 face-focused frames per clip.
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
- If `facenet_pytorch` is unavailable in the current environment, `extract_faces.py` falls back to OpenCV Haar face detection and then to whole-frame resize.
- The optional bounding-box annotation archive is not required for the current AVT-CA pipeline, so it is not part of the official download script here.

# EngageNet Preprocessing

This folder contains the bootstrap EngageNet preprocessing path for the current AVT-CA codebase.

## What It Produces

To stay compatible with the current training pipeline, EngageNet preprocessing produces:

- `<clip>_croppad.wav`
- `<clip>_facecroppad.npy`
- `preprocessing/engagenet/annotations_engagement.txt`

The annotation file uses the same format as the existing datasets:

`video_path;audio_path;label;split`

## Label Mapping

EngageNet engagement labels are mapped to four classes:

- `Not-Engaged -> 0`
- `Barely-engaged -> 1`
- `Engaged -> 2`
- `Highly-Engaged -> 3`

The special label `SNP(Subject Not Present)` is skipped.

## Preprocessing Steps

If you need to re-download EngageNet, pass the OneDrive values at runtime instead of storing them in the repo:

```bash
export ENGAGENET_SHARE_URL='...'
export ENGAGENET_DRIVE_ID='...'
export ENGAGENET_ROOT_ITEM_ID='...'
python preprocessing/engagenet/download_engagenet.py --output-dir datasets/EngageNet
```

Or pass the same values directly as CLI flags.

```bash
# 1. Extract centered 3.6 s mono audio clips.
python preprocessing/engagenet/extract_audios.py --data_root datasets/EngageNet

# 2. Extract 15 face-focused frames from a centered 3.6 s window.
python preprocessing/engagenet/extract_faces.py --data_root datasets/EngageNet

# 3. Build the training annotation file.
python preprocessing/engagenet/create_annotations.py --data_root datasets/EngageNet
```

Or run the one-command wrapper:

```bash
python preprocessing/engagenet/prepare_engagenet.py --data_root datasets/EngageNet
```

## Notes

- The current AVT-CA stack is still a bootstrap 4-class engagement classifier, not the final multi-head engagement/confusion system in `docs/plan.md`.
- Face extraction uses `MTCNN` from `facenet_pytorch` by default and is meant to run inside the repo's `avtca` conda environment.
- EngageNet is intentionally reduced to the same temporal contract as the current RAVDESS-style model: a centered `3.6 s` window and `15` sampled frames.
- If face crops have not been extracted yet, the annotation builder falls back to raw `.mp4` paths for video, but audio preprocessing is still required.
- The training and validation label files are distributed as `.xlsx`, and this folder reads them directly without requiring `openpyxl`.

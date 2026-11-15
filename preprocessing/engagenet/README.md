# EngageNet preprocessing

These scripts turn the EngageNet release (Singh et al., ICMI 2023) into the inputs of the paper's three
models: face crops and full-clip audio for the audio-visual network, OpenFace series for the
gradient-boosting model and the segment Transformer, and the annotation files that list the clips.
EngageNet is distributed by its authors on request; nothing derived from it is part of this repository.

## 1. Data layout

Place the release under `datasets/EngageNet/`:

```
datasets/EngageNet/Train/*.mp4  Validation/*.mp4  Test/*.mp4
datasets/EngageNet/train_engagement_labels.xlsx  validation_engagement_labels.xlsx  test_engagement_labels.csv
```

If the authors give you their OneDrive share, `download_engagenet.py` fetches it; pass the share values at
run time rather than storing them in the repository:

```bash
export ENGAGENET_SHARE_URL='...' ENGAGENET_DRIVE_ID='...' ENGAGENET_ROOT_ITEM_ID='...'
python preprocessing/engagenet/download_engagenet.py --output-dir datasets/EngageNet
```

## 2. Steps used for the paper (run from the repository root, in this order)

```bash
# Audio: the whole clip, mono, 22,050 Hz -> <clip>_croppad10s.wav. A clip without a decodable audio
# stream gets 10 s of zeros and is listed in datasets/EngageNet/silent_fallback_croppad10s.txt.
python preprocessing/engagenet/extract_audios_full.py --data_root datasets/EngageNet

# Faces: 5 frames per second over the whole clip (default --target_fps 5), the first MTCNN face box
# cropped and resized to 224 x 224 (BGR uint8), the whole frame when no face is found
# -> <clip>_facecroppad.npy. Needs facenet_pytorch; --num_shards/--shard_index split the work.
python preprocessing/engagenet/extract_faces.py --data_root datasets/EngageNet

# OpenFace series for the behavior models (needs an OpenFace 2.2 build; FeatureExtraction is taken
# from OPENFACE_BIN or the PATH). 22 channels per frame: 17 action-unit intensities, gaze_angle_x/y,
# pose_Rx/Ry/Rz -> datasets/EngageNet/behavior/<clip>.npy, shape (frames, 22).
OPENFACE_BIN=/path/to/FeatureExtraction python preprocessing/engagenet/extract_behavior_parallel.py \
    --videos_glob 'datasets/EngageNet/*/*.mp4' --out_dir datasets/EngageNet/behavior
# Optional: the 49-column layout of the EngageNet baselines plus OpenFace's confidence and success
# flags (used for the 98-feature check of the paper and for the face-detection statistics).
OPENFACE_BIN=/path/to/FeatureExtraction python preprocessing/engagenet/extract_behavior_full.py \
    --videos_glob 'datasets/EngageNet/*/*.mp4' --out_dir datasets/EngageNet/behavior98

# Annotation files (absolute paths; full-clip audio; empty text column). The behavior models read the
# first file, the audio-visual networks the second (the same clips plus a participant-ID column).
python preprocessing/engagenet/create_annotations.py \
    --annotation_file preprocessing/engagenet/annotations_engagement_a10.txt
python preprocessing/engagenet/create_annotations.py --subject_column \
    --annotation_file preprocessing/engagenet/annotations_engagement_a10_subject.txt
```

`prepare_engagenet.py` runs the audio, face and annotation steps in one command (not OpenFace).
Regenerated on 2026-11-14, both annotation files matched the files of the paper's runs in every path,
label, split and participant ID; only their text column, which no reported model reads, is now empty.

Next steps (segment features, the two behavior models, the network fine-tuning and the fusion) are in
`scripts/fullclip/`; see the repository README.

## 3. What the annotation files contain

`video_path;audio_path;label;split[;text[;subject]]`, one row per labelled clip, splits in the order
training, validation, testing. Labels: `Not-Engaged -> 0`, `Barely-engaged -> 1`, `Engaged -> 2`,
`Highly-Engaged -> 3`. The 105 clips labelled `SNP(Subject Not Present)` (104 training, 1 test) are
skipped, which leaves 7,879 / 1,071 / 2,256 clips. The `.xlsx` label files are read without `openpyxl`.

## 4. Properties of the data worth knowing

`python scripts/paper/data_quirks.py` measures them (results/paper_jaihc/data_quirks.json):

- OpenFace finds a face in 96.3% of the frames of a clip on average (98.2% of all frames, a figure
  dominated by the clips with long series); 101 clips have no detected face at all, 100 of them labelled
  *not engaged*. Frames without a face are kept as OpenFace writes them (zero action units), so the
  segment statistics also encode face absence.
- Most OpenFace series have 300 rows for the 10 s clip, but 1,012 clips have exactly 10,000 rows and some
  have 150 or fewer; segment statistics split each series into equal parts whatever its length.
- 691 clips (454 training, 237 test, none in validation; seven participants) have a digitally silent
  audio track: 371 have no audio stream in the source file, 320 have a silent one.

## 5. Legacy and retracted options

- `extract_audios.py` writes the legacy centred 3.6 s window (`*_croppad.wav`); pass
  `--audio_suffix croppad` to `create_annotations.py` to point the annotations at it.
- `create_annotations.py --text_source chat` writes the v1 synthetic "student chat", sampled from the
  topic banks in `chat_banks/` **by the ground-truth label**: a label oracle, kept only to reproduce the
  retracted text experiments (docs/plan.md Sec. 16-17). `--text_source behavior` writes label-free
  captions from the OpenFace series (`behavior_caption.py`, thresholds from `compute_caption_stats.py`);
  `generate_chat.py` and `audit_text_leakage.py` belong to the same retracted text track.
- `compute_baselines.py` writes per-participant neutral action-unit baselines from training clips for
  the optional `--behavior_baselines` input; no reported model uses them.

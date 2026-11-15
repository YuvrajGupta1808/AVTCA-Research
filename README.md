# AVB-Engage: audio-visual and facial-behavior fusion for student engagement recognition

This repository holds the code behind the manuscript **"AVB-Engage: Audio-Visual and Facial-Behavior Fusion for
Ordinal Student Engagement Recognition"** by Yuvraj Gupta and Sanchita Ghose (San Francisco State University),
prepared for the *Journal of Ambient Intelligence and Humanized Computing*. The manuscript, its supplement
(Online Resource 1) and the figure sources are in
[`research/paper/AVB-Engage_JAIHC_final/`](research/paper/AVB-Engage_JAIHC_final/).

AVB-Engage reads a 10 s webcam clip of a student watching an educational video and predicts one of four ordered
engagement levels: not engaged, barely engaged, engaged, highly engaged. It combines three blocks:

- **A. Audio-visual network** ([`models/multimodal_cnn.py`](models/multimodal_cnn.py), `--fusion it`).
  - Video: EfficientFace embeddings of up to 50 face crops sampled at 5 fps.
  - Audio: a 64-bin log-mel spectrogram, whose time steps are pooled onto the video frames of the same clip.
  - Fusion: three cross-modal attention exchanges, following the intermediate fusion of AVT-CA.
  - Head: masked attention pooling and a linear layer.
- **B. Two facial-behavior models** ([`scripts/fullclip/`](scripts/fullclip/)), both reading segment statistics of a
  22-channel OpenFace series (17 action-unit intensities, 2 gaze angles, 3 head rotations; mean and standard
  deviation over 20 segments).
  - A histogram gradient-boosting classifier.
  - A small segment Transformer.
- **C. Late fusion** ([`scripts/fullclip/fuse_members.py`](scripts/fullclip/fuse_members.py)).
  - The three class-probability vectors are averaged with fixed weights 0.6 / 0.2 / 0.2.
  - The average is reduced to the expected level and cut at thresholds fitted on the validation split.

## Results

Official EngageNet test split, 2,256 clips. Mean ± standard deviation over three training seeds; the
gradient-boosting model is deterministic.

| Model | Top-1 (%) | Adjacent (%) | MAE (levels) | Macro-F1 |
|---|---:|---:|---:|---:|
| Majority-class predictor | 50.27 | 68.84 | 1.012 | 16.73 |
| Best published test-split result: Transformer on OpenFace features (Singh et al., ICMI 2023) | 67.61 | – | – | – |
| A. Audio-visual network | 66.27 ± 0.45 | 91.55 | 0.440 | 52.31 |
| B. Gradient boosting | 66.27 | 89.01 | 0.476 | 50.04 |
| B. Segment Transformer | 66.99 ± 0.93 | 87.90 | 0.487 | 49.45 |
| **AVB-Engage (A + B, fused)** | **70.29 ± 0.14** | 91.06 | **0.401** | **53.95** |

The comparison with the published baselines is not like-for-like. AVB-Engage averages three models and fits its
thresholds on the validation split. Part of its margin comes from the learner and the decoding rule rather than
from the audio-visual network (manuscript, Section 5.1).

## Repository layout

```
main.py                          training entry point (same as python -m src.cli.train)
src/
  cli/                           train.py, evaluate.py
  config/opts.py                 every training and evaluation option
  data/                          dataset registry, temporal sampling, padding and collation, transforms
  engine/                        training loop, evaluation, decoders (calibration.py), metrics, checkpoints
  models/factory.py              model construction
models/                          multimodal_cnn.py (block A), efficient_face.py, modulator.py, transformer.py;
                                 behavior_*.py and text_encoder.py: the in-network behavior/text branches of the ablation
datasets/                        loaders: engagenet.py, ravdess.py, daisee.py, cremad.py (the data is not tracked)
preprocessing/engagenet/         download, face/audio/OpenFace extraction, annotation files (see its README)
preprocessing/ravdess|daisee|cremad|mosei/   earlier datasets (legacy)
scripts/
  calibrate_engagement_logits.py validation-fitted decoders and the modality ablation for one network
  fullclip/                      reported fine-tunes, segment features, gradient boosting, segment Transformer, fusion
  paper/                         analyses behind the manuscript's tables and figures
  night/, exp2026_*.sh           the 14-run fine-tuning sweep (Online Resource 1, Table S2) and earlier drivers
  behavior_eval/                 historical evaluation of the in-network behavior/text branch
ui/                              Streamlit demo of block A
research/paper/AVB-Engage_JAIHC_final/       manuscript, supplement, figures, figure sources, submission files
pretrained/                      EfficientFace_Trained_on_AffectNet7.pth goes here (not tracked; see Installation)
tests/                           unit and smoke tests (CPU, synthetic data)
docs/                            research log: plan.md, memory.md, progress.md, architecture.md
```

## Installation

```bash
conda create -n avtca python=3.10
conda activate avtca
pip install -r requirements.txt
```

The visual backbone starts from the EfficientFace weights trained on AffectNet (7 classes). They are not in this
repository: download `EfficientFace_Trained_on_AffectNet7.pth` from the
[EfficientFace repository](https://github.com/zengqunzhao/EfficientFace) and save it as
`pretrained/EfficientFace_Trained_on_AffectNet7.pth`, the default of `--pretrain_path`.

For the exact versions behind the paper (PyTorch 2.5.1 with CUDA 12.1, librosa 0.8.1, numpy 1.23.5, scikit-learn
1.7.2), install from `requirements-lock.txt`; its header gives the commands.

The librosa version changes the audio features slightly:

- From librosa 0.10, the short-time Fourier transform pads the signal with zeros instead of reflecting it.
- On EngageNet clips, this changes the first two and the last of the 431 mel frames of a clip, by up to 6.4 dB.
  The other frames agree to within 0.001 dB (five test clips checked).
- The paper's features and networks used librosa 0.8.1.

External programs:

- **ffmpeg**: for audio extraction. The `imageio-ffmpeg` package bundles a binary.
- **OpenFace 2.x `FeatureExtraction`**: for the facial-behavior features. Set its path in the environment variable
  `OPENFACE_BIN`, or put it on `PATH`.
- **A CUDA GPU**: for MTCNN face detection and training. The reported fine-tunes need about 13 GB of GPU memory at
  50 frames per clip.

## Data: EngageNet

EngageNet (Singh et al., ICMI 2023) is available from its authors on request. This repository distributes none of
its videos, labels or derived files.

Place the dataset as follows:

```
datasets/EngageNet/
  Train/  Validation/  Test/          the .mp4 clips
  train_engagement_labels.xlsx
  validation_engagement_labels.xlsx
  test_engagement_labels.csv
```

If the authors give you their OneDrive share, `preprocessing/engagenet/download_engagenet.py` can fetch it. Pass the
share values through `ENGAGENET_SHARE_URL`, `ENGAGENET_DRIVE_ID` and `ENGAGENET_ROOT_ITEM_ID`.

Preprocessing runs in this order. [`preprocessing/engagenet/README.md`](preprocessing/engagenet/README.md) gives the
exact options.

1. **Face crops.** `extract_faces.py --target_fps 5` writes `<clip>_facecroppad.npy`: MTCNN crops of 224 × 224
   pixels at 5 fps, over the whole clip.
2. **Audio.** `extract_audios_full.py` writes `<clip>_croppad10s.wav`: 22,050 Hz mono, the whole clip.
   - `extract_audios.py` writes the 3.6 s `<clip>_croppad.wav` instead. Only the first two pre-training stages below
     used it.
3. **OpenFace features.**
   - `extract_behavior_parallel.py` writes `datasets/EngageNet/behavior/<clip>.npy`, with T × 22 values: blocks B and
     the behavior branch.
   - `extract_behavior_full.py` writes `datasets/EngageNet/behavior98/`, with T × 51 values: only the 98-feature
     ablation.
4. **Annotation files.** `create_annotations.py` writes one row per clip: `video;audio;label;split;text`.
   - It skips the 105 clips labelled "subject not present".
   - `annotations_engagement_a10.txt` points at the 10 s audio.
   - `annotations_engagement.txt` points at the 3.6 s audio (`--audio_suffix croppad`).
   - The full-clip scripts read `annotations_engagement_a10_subject.txt`: the same rows with the participant ID
     (`subject_<n>_<id>`) as a sixth column (`--subject_column`).
   - None of the reported networks reads the text column, so it is written empty (the default, `--text_source none`).

   ```bash
   python preprocessing/engagenet/create_annotations.py --data_root datasets/EngageNet \
     --annotation_file preprocessing/engagenet/annotations_engagement_a10.txt
   python preprocessing/engagenet/create_annotations.py --data_root datasets/EngageNet --subject_column \
     --annotation_file preprocessing/engagenet/annotations_engagement_a10_subject.txt
   ```

`preprocessing/engagenet/prepare_engagenet.py --subject_column` runs steps 1, 2 and 4 in one call; the OpenFace step
needs its own OpenFace build.

The annotation files and per-clip caches hold EngageNet labels and local absolute paths, so they are git-ignored.
Regenerate them for your copy of the data.

## Reproducing the paper

The repository state prepared for the JAIHC submission is tagged `jaihc-submission-2026-11-15`. All runs write to
`results/`, which is git-ignored. The commands below use these flags, which every EngageNet run
shares:

```bash
COMMON="--dataset ENGAGENET --n_classes 4 --model multimodal_cnn --fusion it --audio_features mel --num_heads 8 \
  --visual_backbone efficientface --data_root datasets/EngageNet --full_video_preprocessing \
  --frame_sampling uniform --max_audio_steps 0 --mask nodropout --no_late_text_fusion \
  --pretrain_path pretrained/EfficientFace_Trained_on_AffectNet7.pth"
```

### Block A: the audio-visual network

The reported networks are fine-tuned from a checkpoint built in three stages, starting from the AffectNet
EfficientFace weights. The commands below were reconstructed from the options stored with each run
(`results/*/opts*.json`).

**Stage 1.** Cross-entropy, learning rate 1e-2, batch size 2, 3.6 s audio, 30 epochs. The original run reached 30
epochs over three launches, resumed with `--resume_path` from its epoch-8 and epoch-11 checkpoints. The best
validation epoch was 24.

```bash
python main.py $COMMON --annotation_path preprocessing/engagenet/annotations_engagement.txt \
  --result_path results/engagenet_h8_uniform_fullaudio_v96 \
  --batch_size 2 --learning_rate 0.01 --n_epochs 30
```

**Stage 2.** Ordinal objective, learning rate 1e-4, batch size 2, resumed from the stage-1 best checkpoint (epoch
24). Two epochs were trained (epochs 24 and 25 of the continued count) before the run was stopped. Its `model.pth`
after epoch 25 is the stage-3 starting point. With the current code, `--resume_path` continues at epoch 25, so
`--n_epochs 26` trains the same two epochs.

```bash
python main.py $COMMON --annotation_path preprocessing/engagenet/annotations_engagement.txt \
  --result_path results/v12_05_finetune_uniform_best_ord_nosampler_lr0001_e30 \
  --resume_path results/engagenet_h8_uniform_fullaudio_v96/ENGAGENET_multimodal_cnn_15_best.pth \
  --batch_size 2 --learning_rate 0.0001 --loss ordinal_distance --ordinal_distance_weight 0.15 --n_epochs 26
```

**Stage 3 (E04).** Ordinal objective, learning rate 5e-5, batch size 8, 10 s audio, 8 epochs. The best validation
epoch was 3. When a run starts without `--resume_path`, it loads the `model.pth` it finds in its result directory
as initial weights, and records that file's path and SHA-256 in a `warm_start*.json` next to the run's options. This
stage was warm-started that way, from the stage-2 weights.

```bash
mkdir -p results/exp2026/E04_ft10s_ord_lr5e5_e8
cp results/v12_05_finetune_uniform_best_ord_nosampler_lr0001_e30/model.pth results/exp2026/E04_ft10s_ord_lr5e5_e8/
python main.py $COMMON --annotation_path preprocessing/engagenet/annotations_engagement_a10.txt \
  --result_path results/exp2026/E04_ft10s_ord_lr5e5_e8 \
  --batch_size 8 --learning_rate 0.00005 --loss ordinal_distance --ordinal_distance_weight 0.15 --n_epochs 8
```

**Stage 4: the reported fine-tunes.** These are three seeds of the recipe in `scripts/fullclip/common.sh`:

- E04's best checkpoint as warm start;
- at most 50 frames per clip;
- the inverse-square-root class-balanced sampler;
- learning rate 5e-5 and 6 epochs.

`run_job.sh` trains a seed, then calls `scripts/calibrate_engagement_logits.py`. That script fits the four decoders
of the paper on the validation split and applies them once to test: argmax, logit bias, expected-level thresholds and
refined thresholds. The paper reports the refined thresholds. `ablate` reruns the calibration with the audio or
the video tokens set to zero, on the same weights. `context_analysis.py` stores the validation and test logits that
the fusion reads.

```bash
for s in 1 2 3; do CUDA_VISIBLE_DEVICES=0 bash scripts/fullclip/run_job.sh train A_av_s$s A --manual_seed $s; done
for s in 1 2 3; do CUDA_VISIBLE_DEVICES=0 bash scripts/fullclip/run_job.sh ablate A_av_s$s; done
for s in 1 2 3; do python scripts/fullclip/context_analysis.py --run_dir results/fullclip/A_av_s$s; done
```

Arms `B` and `C` of `run_job.sh` are the behavior-branch ablation of Section 5.4.2. They need warm starts written
by `scripts/fullclip/seed_warmstart.py`. `python -m src.cli.evaluate --checkpoint <ckpt> --result_path <run dir>`
re-evaluates any checkpoint with the options stored in its run directory. It refuses a checkpoint that does not fill
every tensor of the model unless `--allow_partial_load` is given. Training runs on one GPU (`--device cuda:N`);
multi-GPU `DataParallel` is used only with `--data_parallel`.

### Block B: the two facial-behavior models

On first use, both scripts cache the segment statistics in `results/fullclip/segtf/features_S20.npz`, built by
`scripts/fullclip/segment_features.py`. Both train on the training split only.

```bash
python scripts/fullclip/gbm_member.py     # gradient boosting on the 924-d vector -> results/fullclip/gbm/probs.npz
python scripts/fullclip/sweep_segtf.py    # 8 Transformer configurations x 3 seeds, selected on validation -> results/fullclip/segtf/
```

The configuration selected on validation (`T7`), for one seed:

```bash
python scripts/fullclip/segment_transformer.py --out_dir results/fullclip/segtf/T7_d64_L2_lr1e3_s1 --seed 1 \
  --features 22 --n_segments 20 --d_model 64 --n_layers 2 --n_heads 8 --ffn 128 --lr 1e-3 --dropout 0.3 \
  --weight_decay 0.05 --ordinal_weight 0.15 --label_smoothing 0.1 --epochs 60 --batch_size 64
```

### Block C: late fusion

`fuse_members.py` aligns the members by clip key. It fits the expected-level thresholds on validation and reports
test once. `fuse_all.py` instead selects the weights on validation for every seed.

```bash
python scripts/fullclip/fuse_members.py --fixed 0.6,0.2,0.2 --names AV TF GBM \
  --members results/fullclip/A_av_s1/context/logits.npz results/fullclip/segtf/T7_d64_L2_lr1e3_s1/logits.npz \
            results/fullclip/gbm/probs.npz \
  --out results/fullclip/fusion_A_av_s1_fixed.json
python scripts/fullclip/fuse_all.py T7_d64_L2_lr1e3
```

### Analyses in the manuscript

Each script reads saved outputs only, except `significance_tests.py collect` and `complexity_benchmark.py`, which
run the networks.

| Script | Output |
|---|---|
| `scripts/paper/springer_analysis.py` | per-seed, per-level, confusion-matrix and fusion-weight results → `results/paper_springer/analysis.json` |
| `scripts/paper/segment_count_sweep.py` | gradient boosting for S ∈ {1, 5, 10, 20, 40} segments → `results/paper_springer/segment_count_sweep.json` |
| `scripts/paper/significance_tests.py collect` then `analyze` | exact McNemar tests (fusion against members; audio + video against video only) → `results/paper_jaihc/` |
| `scripts/paper/complexity_benchmark.py` | parameters, FLOPs and latency → `results/paper_springer/complexity.json`; with `--flops_only`, only the segment-Transformer FLOPs (CPU) → `results/paper_jaihc/segment_transformer_flops.json` |
| `scripts/paper/participant_bootstrap.py` | intervals that resample test participants instead of clips → `results/paper_jaihc/participant_bootstrap.json` |
| `scripts/paper/decoding_gain.py` | argmax against expected-level decoding for every network and member → `results/paper_jaihc/decoding_gain.json` |
| `scripts/paper/data_quirks.py` | OpenFace detection rates and silent audio tracks → `results/paper_jaihc/data_quirks.json` |
| `scripts/paper/equal_weights.py` | the equal-weight fusion → `results/paper_jaihc/equal_weights.json` |

The fourteen-run fine-tuning sweep of Online Resource 1, Table S2 was run from `scripts/night/`, with queue files
and `collect.py`.

## Streamlit demo

```bash
pip install streamlit pandas
streamlit run ui/app.py
```

The demo runs block A only: the audio-visual network decoded with its validation thresholds. It does not run the
OpenFace models or the fusion. It loads the stage-3 (E04) and stage-2 checkpoints from `results/`; set
`AVTCA_RESULTS_DIR` to read them from another directory. No weights are distributed, so train them first.

## What is not in this repository

- **EngageNet.** The videos, labels and every per-clip file derived from them are not here. That includes the
  annotation files, the OpenFace features and the audio caches.
- **Trained weights.** None of these are here:
  - the three reported networks and the warm-start stages;
  - the gradient-boosting and segment-Transformer models.
  The per-clip predictions are available from the corresponding author on reasonable request (manuscript, Data
  availability).
- **The EfficientFace starting weights.** Download them from the EfficientFace repository (Installation).
- **Run outputs.** `results/` and `logs/` are git-ignored.
- **The exact code of the pre-training stages.**
  - Stages 1–3 ran on uncommitted code between commits `9c82bc4` (2026-06-11) and `88f8652` (2026-09-12).
  - The reported fine-tunes ran on the tree of `88f8652`.
  - The current code reproduces the stored validation logits of seed 1 to within 3e-4 (24 clips checked,
    2026-11-14).
  - The four stages have not been re-run from the AffectNet initialization with the current code.

## Tests and CI

```bash
python -m pytest tests/ -q
```

The tests use synthetic data on the CPU. GitHub Actions runs an import check, a dataset-registry check and the test
suite on Python 3.10 and 3.11 with CPU PyTorch (`.github/workflows/ci.yml`).

## Legacy: RAVDESS emotion recognition

The repository began as a fork of the AVT-CA emotion-recognition code, for the eight emotion classes of RAVDESS.
That path still works. RAVDESS goes in `datasets/RAVDESS/ACTOR01` … `ACTOR24`.

```bash
python preprocessing/ravdess/extract_audios.py --data_root datasets/RAVDESS
python preprocessing/ravdess/extract_faces.py --data_root datasets/RAVDESS
python preprocessing/ravdess/create_annotations.py --data_root datasets/RAVDESS \
  --annotation_file preprocessing/ravdess/annotations.txt
python main.py --dataset RAVDESS --data_root datasets/RAVDESS --annotation_path preprocessing/ravdess/annotations.txt \
  --audio_features mel --num_heads 4 --learning_rate 0.01 --n_epochs 75 --mask nodropout \
  --result_path results/ravdess_mel_h4
```

These preprocessing steps crop the audio to 3.6 s and keep 15 face frames per video. The best recorded test
accuracy is 81.88%, in `results/v3_01_baseline_mel_h4`. The split inherited from upstream lists each of its 240 test
recordings twice, once with the audio-visual and once with the video-only face file.

DAiSEE (`preprocessing/daisee/`, `--dataset DAISEE`) and CREMA-D loaders also exist. They were bootstrap experiments
and are not used in the paper. The CMU-MOSEI utilities have no loader.

## Acknowledgements and licence

The network code descends, through the AVT-CA repository, from the audio-visual emotion-recognition code of
Chumachenko, Iosifidis and Gabbouj. This repository started as a fork of AVT-CA.

- **AVT-CA.** J. Dhanith P R, S. Venkatraman, V. Sharma and S. Malarvannan, "Multimodal emotion recognition using
  audio-video transformer fusion with cross attention", arXiv:2407.18552. Code:
  [shravan-18/AVTCA](https://github.com/shravan-18/AVTCA).
- **Intermediate fusion and modality dropout.** K. Chumachenko, A. Iosifidis and M. Gabbouj, "Self-attention fusion
  for audiovisual emotion recognition with incomplete data", Proc. ICPR 2022, pp. 2822–2828,
  doi:10.1109/ICPR56361.2022.9956592. Code:
  [katerynaCh/multimodal-emotion-recognition](https://github.com/katerynaCh/multimodal-emotion-recognition).
- **EfficientFace.** Z. Zhao, Q. Liu and F. Zhou, "Robust lightweight facial expression recognition network with
  label distribution training", AAAI 2021. Implementation and AffectNet-trained weights:
  [zengqunzhao/EfficientFace](https://github.com/zengqunzhao/EfficientFace).
- **Training pipeline.** Adapted from [okankop/Efficient-3DCNNs](https://github.com/okankop/Efficient-3DCNNs) (MIT).
- **Attention blocks.** Based on [timm](https://github.com/rwightman/pytorch-image-models) (Apache 2.0).
- **Face detection.** MTCNN from [facenet-pytorch](https://github.com/timesler/facenet-pytorch).
- **Facial-behavior features.** [OpenFace 2](https://github.com/TadasBaltrusaitis/OpenFace).
- **EngageNet and its baselines.** M. Singh, X. Hoque, D. Zeng, Y. Wang, K. Ikeda and A. Dhall, "Do I have your
  attention: A large scale engagement prediction dataset and baselines", ICMI 2023.

The code is released under the MIT licence in [`LICENSE`](LICENSE). It keeps the 2022 notice of the upstream code.

## Citation

```bibtex
@misc{gupta2026avbengage,
  title  = {{AVB-Engage}: Audio-Visual and Facial-Behavior Fusion for Ordinal Student Engagement Recognition},
  author = {Gupta, Yuvraj and Ghose, Sanchita},
  year   = {2026},
  note   = {Manuscript; code at https://github.com/YuvrajGupta1808/AVTCA-Research}
}
```

## Research log

`docs/` holds the project's working notes. `plan.md` covers plans, experiments and decisions; `memory.md` covers
non-obvious context; `progress.md` covers status; `architecture.md` covers the architecture diagrams. Section 22 of
`plan.md` records the full review of the paper and code (2026-11-14).

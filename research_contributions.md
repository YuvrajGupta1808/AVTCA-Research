# Major Research Contributions

**Project:** Audio-Visual Engagement Detection in Online Classrooms
**Author:** Yuvraj Gupta · Personal research repository
**Benchmark:** EngageNet — 11,206 clips, 4-level ordinal engagement, 2,257-clip test split
**Status as of 2026-08-14:** best test top-1 **66.36%**, beating four of the six published EngageNet baselines (best published: Transformer, 67.61%)

---

## 1. First audio-visual fusion result on EngageNet

**No published EngageNet work uses audio at all** — every baseline in the ICMI 2023 paper and subsequent literature is video-only. This project is the first to fuse audio with video on this benchmark, and it produces a positive result:

- **AV fusion 66.13% vs video-only 65.07%** (same weights, refined-expected-threshold decoder) — fusion wins on **all four decoders** for the 3.6 s-audio model.
- Best overall checkpoint: **66.36%** fusion, which beats the published CNN-LSTM (65.16%) and TCN (65.60%) baselines.
- The honest, narrow claim: **audio carries no standalone clip-level engagement signal (audio-only collapses to exactly the 50.27% majority predictor), yet adds ~1 point when fused with video** — a cross-modal interaction effect, consistent in sign across 12 of 13 matched models.

This contribution required first *reversing our own null result*: the original "audio adds nothing" finding was traced to three preprocessing defects (audio truncated to the first 3.6 s of 10 s clips, and a test split extracted at 15 frames vs 50-frame training clips). Fixing them flipped the conclusion — itself a methodological lesson documented below.

## 2. Ordinal calibration decoding — +2.03 top-1 at zero training cost

Engagement levels are ordered, but the field decodes them with plain argmax. We implemented four decoders (argmax, fitted logit bias, expected-value thresholds, refined expected thresholds) and showed:

- **Refined expected thresholds gains +2.03 top-1 over argmax** (66.13% vs 64.10%) on identical trained weights — larger than every architectural and optimizer change tested, combined.
- Decoder choice alone moves reported accuracy by ~1 point, so **no engagement accuracy figure is comparable across papers unless the decoder is reported**. This is a field-level measurement critique, not just a trick.

## 3. Per-sample audio→video temporal alignment

Audio exits the CNN at ~168 time steps against video's ~15 — an 11× granularity mismatch at cross-attention. We introduced `_adaptive_align_audio_to_video`: adaptive pooling of the full valid audio span onto the per-sample valid video length, before the first cross-attention block (unit-tested).

- Effect: **+0.80 adjacent accuracy, +1.86 macro-F1, −0.005 MAE**, at −0.44 top-1 (inside seed noise).
- The fix moves errors from off-by-two to off-by-one and separates minority levels — improvements that **top-1 accuracy is structurally blind to** on an ordinal task. This motivates the reporting standard in Contribution 6.

## 4. Inference-time modality ablation on identical weights

Either modality's token stream can be zeroed at inference on already-trained weights (`ablate_modality` in `models/multimodal_cnn.py`). Unlike the standard practice of training a separate unimodal model, this measures each modality's contribution on *identical parameters*, removing the confound of a different optimization run. This tooling is what made the audio-contribution claim in Contribution 1 measurable at all, and what exposed the earlier preprocessing artefact.

## 5. First matched, seed-controlled single-variable sweep on EngageNet — and a corpus-ceiling finding

14 finetunes warm-started from one shared checkpoint, each changing exactly one variable (loss type, ordinal weight, class weighting, balanced sampler, LR schedule, EMA, label smoothing, SpecAugment, frame cap, synced crop), plus 3 seed repeats of one configuration to establish a noise floor (sd 0.45 top-1).

- **Every isolated training-side comparison is null** — all deltas sit inside the seed band.
- Varying every hyperparameter produced about as much spread (sd 0.52) as changing the random seed alone (sd 0.45).
- Ensembling 14 checkpoints *underperforms* the best single model (shared warm start → correlated errors); augmentation actively hurts.
- Conclusion, reached by three independent routes: **the corpus is the ceiling, not the model.** On ~11k clips, remaining gains live in the data and label scheme, not the architecture — which directly motivates the purpose-built dataset (Contribution 7).

## 6. A reporting standard for ordinal engagement recognition

Distilled from the project's own retracted claims, each item costing nothing:

1. **Report the decoder** alongside every accuracy figure (decoder choice alone moves top-1 by ~1–2 points).
2. **Report adjacent accuracy and MAE next to top-1** on any ordinal task — two of our three genuine effects are invisible to top-1.
3. **Establish a seed-repeat noise floor** before claiming any sub-one-point improvement.
4. **Report macro-F1** — the majority predictor scores 50.27% top-1 at only 16.73 macro-F1 on this benchmark.

## 7. Purpose-built engagement dataset design (designed, pre-collection)

A complete design for a new audio-visual classroom engagement corpus, engineered around the failure modes found above:

- **5-level ordinal engagement scale + separate binary confusion flag** with behavioral anchors (confusion is deliberately orthogonal to engagement level: a Level 5 student can be productively confused).
- **Audio-visual parity by construction**: session scripts guarantee ~75% of clips contain active speech (vs 24% in CMOSE, whose audio path added only 0.41% overall), with a hard gate — audio-only accuracy must reach ≥65% of fusion accuracy before joint training proceeds.
- **W-curve session structure** (5 blocks per 75-minute session, two deliberate boredom blocks) using controllable difficulty ramps in ML/Statistics courses to elicit the full engagement range without deception.
- **Per-student baseline calibration**: all signals (blink rate, EAR, head pose, F0) relative to each student's first-5-minute neutral state — population thresholds rejected.
- **Hawthorne-effect control**: Session 1 data excluded from training by design.
- **Composite temporal scoring formulas** for boredom, confusion, and flow (not frame-level labels), and **speaker/listener role conditioning** — gaze and audio engagement signals invert with speaking role, so `is_speaking` conditions all scoring.
- Expected yield: ~10,000–12,000 labeled clips, single HDF5 with symmetric audio/video feature storage.

## 8. Supporting engineering contributions

- **Full engagement inference pipeline + Streamlit UI** serving the trained EngageNet models (MTCNN face extraction, 5 fps stride, calibrated thresholds), verified bit-exact against the training arrays.
- **DAiSEE and EngageNet preprocessing/training support** integrated into the original AVT-CA codebase, plus the earlier RAVDESS emotion-recognition phase (71.25% test, 8-class) that the architecture originated from.
- **Verified literature positioning**: EngageNet baselines PDF-verified from the ICMI 2023 tables after catching a propagated validation-vs-test citation error; known acronym collision with arXiv:2407.18552 ("AVT-CA") flagged for rename before submission.

---

## One-paragraph summary

This project delivers the first audio-visual fusion result on EngageNet — fusion beats video-only by ~1 point on all decoders, while showing audio has no standalone signal — alongside three methodological contributions that generalize beyond the benchmark: ordinal calibration decoding worth +2 top-1 points at zero training cost, per-sample audio-video temporal alignment that improves every ordinal metric, and the benchmark's first seed-controlled single-variable sweep, which shows hyperparameter variation is indistinguishable from seed noise and therefore that the corpus, not the model, is the ceiling. That finding motivates the project's final contribution: a fully specified, parity-by-construction classroom engagement dataset designed to remove the data limitations the experiments exposed.

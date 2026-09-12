# AVTCA Progress — Current Verified Results
**RAVDESS · 8-Class Emotion Recognition · 2026-05-28**

> ## Status at a glance — 2026-09-12
>
> **Audio-visual model (EfficientFace + mel, full 10 s clip, G04 config): 66.27 ± 0.45 test top-1**
> over 3 seeds (91.55 adjacent, 0.440 MAE, 52.31 macro-F1), bit-identical to the August G04 runs — the
> model was already using the whole clip. Audio fusion adds +0.74 over video-only (8/9 runs ≥ 0).
>
> **Behavior branch (OpenFace AUs, now on the 50-step video clock): no effect** — 66.08 ± 0.59 with it,
> 66.12 ± 0.59 with caption text added, equal to the AV model at every decoder. C7/C8/C9 closed.
>
> **New headline candidate: three-way late fusion AV + segment transformer + OpenFace GBM —
> 70.29 ± 0.14 test top-1** (3 AV seeds; 70.15 ± 0.25 over all 8 neural checkpoints), weights and
> thresholds chosen on validation, **+2.7 over the published best 67.61**. Two-way with the GBM alone:
> 69.86 ± 0.50. Our 22-feature segment transformer alone reproduces the published 67.6. Detail and the measured-null levers (neighbour
> smoothing, prior shift) in [`plan.md` §19.6](plan.md). Not yet in the paper or the UI.
>
> All numbers below this block are historical; the sections dated 2026-09-12 at the bottom are current.

> **Note (2026-08-08):** `docs/` was reduced to the four canonical files. The former external-facing
> documents — `evidence_tables.md`, `professor_progress_brief.md`, `latex.tex` and the paper-section
> drafts — were deleted and are **not recoverable** (they were never tracked by git). Their numeric
> content survives here and in `plan.md`; the raw experiment record survives in
> `results/exp2026/all_experiments.csv`, regenerable with `python scripts/compile_experiments.py`.

## Session 2026-08-08 — documentation cleanup and repository push

**No model, training, or experiment state changed.** Run state is exactly as recorded for 2026-08-07
(the F01/F02 stop-vs-finetune decision is still open — see the sections below).

Two things happened:

1. **A résumé-facing project overview was written and then deleted** with the rest of the non-core docs.
   Its substance is preserved in `memory.md` under the 2026-08-08 entry — in particular the constraint
   list that any external summary must respect: no SOTA framing (66.36% is 1.25 below the best published
   baseline), no "built from scratch" (this repo is a fork of the published AVT-CA implementation), no
   "proved audio helps" (the fusion gain is single-seed; E22 is outstanding), no DAiSEE 55.25% as an
   achievement, no text-fusion numbers as a finding about text, no "collected a dataset". The 66.36%
   always carries its validation-mismatch caveat.
2. **`docs/` was cut to the four canonical files** and the repository was committed and pushed, including
   source that had never been tracked — `src/engine/calibration.py` and `src/data/temporal.py` among
   them, i.e. the code behind the headline result.

⚠️ **Open item:** the supervisor's name was given in conversation as **"Professor Sanchita Goes"** and
appears nowhere in this repository. The spelling is unverified and must be confirmed before it is used in
any external document.

## EngageNet Verified Snapshot (2026-07-31)

| Run | Decoder | Test top-1 | Adjacent | MAE | Macro F1 | Artifact |
|---|---|---:|---:|---:|---:|---|
| V9 late-text pretrained | Argmax | 55.3191% | — | — | 45.0431 | text hurts |
| h8 stride full-video | Expected thresholds | 63.9628% | 85.1064% | 0.549202 | 45.5639 | prior cited target |
| V12 AV-only ordinal finetune | Refined expected | **64.1844%** | 87.4113% | 0.519504 | 46.5955 | **current best top-1** |
| V12 + align-fix recalib | Refined expected | 64.0514% | 86.6578% | 0.533245 | 44.5441 | `results/v13_alignfix_avonly_short/v12_recalib_alignfix/` |
| V13 short 3-ep finetune + calib | Refined expected | 63.7411% | **88.2092%** | **0.514184** | **48.4528** | `results/v13_alignfix_avonly_short/finetune_e3_randcrop/calibration/` |

**Professor takeaway:** AV architecture is solid; late text lowered accuracy (55.3%). Best top-1 remains V12 **64.18%** (beats 63.96%). Short V13 run improved adjacent/macro-F1 but not top-1.

## ~~Modality Ablation — 2026-08-07 (15-frame test set)~~ — SUPERSEDED

Measured before the train/test preprocessing mismatch was found; the test set was 100% 15-frame while
training was 50-frame. Its conclusion ("video-only matches fusion; no AV claim supported") **was an
artefact and has been reversed** — see CORRECTED RESULTS below. Retained only as the record of what the
mismatch did to the numbers: AV 64.05 / video-only 64.18 / audio-only 46.41.

## CORRECTED RESULTS (2026-08-07) — after fixing the train/test preprocessing mismatch

Test/Validation were re-extracted at `--target_fps 5`, going from 96.2%/100.0% of clips at 15 frames to
**0.0%** (median 50, matching Train). All headline numbers recomputed. Refined-expected decoding:

| Model | Condition | Provisional (15-frame) | **Corrected (50-frame)** | Δ |
|---|---|---:|---:|---:|
| V12 (3.6 s audio) | AV fusion | 64.05 | **66.13** | **+2.08** |
| V12 | Video-only | 64.18 | 65.07 | +0.89 |
| V12 | Audio-only | 46.41 | 50.27 | +3.86 |
| E04 (10 s audio) | **AV fusion** | 62.68 | **66.36** | **+3.68** |
| E04 | Video-only | 62.99 | 66.22 | +3.23 |
| E04 | Audio-only | 50.27 | 50.27 | 0.00 |

**New best: 66.36% top-1, 90.96% adjacent, 52.03 macro-F1** (E04 AV fusion). Best macro-F1 52.35,
best adjacent 91.00. This **beats CNN-LSTM (65.16) and TCN (65.60)** from the published EngageNet
baselines, trailing Transformer Fusion (66.50) by 0.14 and the best published result (67.61) by 1.25.
The frame-count mismatch, not the
architecture, accounted for the gap.

### The audio conclusion is REVERSED for the 3.6 s model

AV fusion beats video-only on **all four decodes**:

| Decode | V12 fusion | V12 video-only | Δ |
|---|---:|---:|---:|
| argmax | 64.10 | 62.90 | **+1.20** |
| logit bias | 65.74 | 64.32 | **+1.42** |
| expected thresholds | 65.65 | 65.03 | **+0.62** |
| refined expected | 66.13 | 65.07 | **+1.06** |

Macro-F1 agrees (52.32 vs 50.37). Consistency across four decodes is what makes it credible — a single
+1.06 would sit inside the ±1.95 binomial 95% CI on 2,257 clips. **The earlier "video-only ≥ fusion"
result was an artefact of degraded 15-frame video.**

Caveats that stand: the **E04 (10 s) model shows no fusion gain** (+0.13, one decode negative), and
**audio-only is exactly the majority predictor** (50.27 / 16.73) in both models. The honest claim is
narrow: *audio carries no standalone signal but adds ~1 point on top of video when fused.* Single seed —
needs a repeat before it goes in a paper (E22). Full analysis in plan.md Section 13.6.

## ~~DECISIVE RESULT — audio does not contribute~~ — WITHDRAWN 2026-08-07

This section concluded that video-only beat AV fusion at both audio spans and that no audio-visual claim
was available on EngageNet. **It was measured entirely on the 15-frame test set and is withdrawn.** On
the corrected 50-frame test set, AV fusion beats video-only on all four decodes for the 3.6 s model
(+0.62 to +1.42). See CORRECTED RESULTS above and plan.md Section 13.6.

What survives: **audio-only alone is exactly the majority-class predictor** (50.27 / 16.73) in both
models, and the E19 encoder-free probe still shows hand-crafted acoustics cannot beat that baseline
either. So audio has no standalone capability — but it does add ~1 point on top of video when fused.

## Remaining Train Defect (2026-08-07) — re-extraction in progress

The train split carried the same legacy defect: **1,822 of 7,983 clips (22.8%) were still at 15 frames**
despite having full 10 s / 300-frame sources (verified with OpenCV on the `.mp4` originals). A fifth of
the training set was learning from 3.3× less temporal evidence than the rest.

The 15-frame `.npy` files were deleted (sources untouched) and are being regenerated at `--target_fps 5`
across 6 shards. **After this all three splits are consistent for the first time.**

**Critical implication for every existing checkpoint:** V12, E04 and all others had their best epoch
selected against the broken 15-frame validation set (96.2% short clips). The training data was largely
correct but the *selection signal* measured the wrong distribution. The current 66.36% is therefore a
model chosen by a broken criterion and then evaluated properly — **nothing in this repo has been trained
under valid conditions.** Retraining on corrected splits is the primary next step. Full analysis and the
four improvement levers in plan.md Section 13.7.

## Documentation state — 2026-08-07 (final for this session)

**All five documentation artifacts** — `plan.md` (§13.0–13.10 complete), `memory.md`, `architecture.md`,
`progress.md` and `professor_progress_brief.md` — **are current and synchronised at session close.**

Session deliverables complete: preprocessing fixes, corrected results R01–R06, audio reversal,
PDF-verified literature, `professor_progress_brief.md`, and `evidence_tables.md`.

Exact run state at close: **F01 at epoch 9, F02 at epoch 11** of 18, both still running; nothing stopped
or launched. Best validation top-1 reached so far: **F01 53.6%, F02 54.0%** — both under the 55% pivot
threshold, and oscillating rather than trending (several epochs fall below the 50.27% majority baseline).
Neither will yield a usable model.

**DECISION PENDING — the only thing blocking progress.** Choose one:

- **Stop F01/F02 and launch G01/G02** — finetune the best checkpoint on corrected splits, ~45 min, both
  GPUs in parallel. This is what the pre-committed criterion (plan.md §13.10.2) calls for, and it isolates
  the preprocessing variable instead of confounding it with initialisation and training length.
- **Let F01/F02 run to completion** — ~50 min remaining; their numbers get recorded but are not expected
  to be usable (best 53.6% / 54.0%, oscillating below the 55% threshold and below the majority baseline at
  several epochs).

Nothing else blocks this. All documentation is complete and synchronised; every other decision is settled.


All four core docs (`plan.md`, `memory.md`, `architecture.md`, `progress.md`) plus
`professor_progress_brief.md` are **synchronised and represent the complete state as of 2026-08-07**.

Everything substantive from this period is recorded and version-controlled: the three preprocessing
defects and their fixes, the full re-extraction of all splits, the corrected result set (R01–R06), the
encoder-free audio probe, the temporal-augmentation verification, the pre-committed decision matrix, the
late text-fusion architecture and its caveat, and the rewritten discussion brief.

**One item outstanding:** F01/F02 are at epoch 3 of 18 (F01 49.9%, F02 51.3%), both recovering from the
epoch-2 dip but neither past the 55% pivot threshold. The epoch 5–6 decision point is live. Nothing else
is pending.

## PIVOT TRIGGERED — F01/F02 failed, switching to finetuning

The pre-committed criterion (plan.md §13.10.2: *"if neither crosses 55% top-1 by epoch 5–6, stop"*) is
**met**. Neither did, and both destabilised rather than converged:

| Run | ep 4 | ep 5 | ep 6 | ep 7 | ep 8 | ep 9 | ep 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F01 | 52.4 | 47.3 | 49.2 | 45.8 | 53.6 | 45.4 | — |
| F02 | 51.4 | 48.1 | 49.5 | 45.4 | 54.0 | 44.2 | 40.6 |

An 8–13 point oscillation with no trend; F02 declining. **18 epochs from the AffectNet face-recognition
pretrain is too short to learn this task** — the V12/E04 lineage reached 66% through many more epochs of
accumulated finetuning, not from a generic pretrain in 18.

**Decision: stop them and finetune the existing best checkpoint on corrected data (G01/G02, ~45 min).**
Better-designed too — it isolates correct preprocessing rather than confounding it with initialisation
and training length. The three questions below are unchanged and equally answerable from G01/G02.

> **Execution status: nothing has been changed yet.** F01/F02 are **still running** (they were not
> killed), and G01/G02 are **not launched**. The switch is a recommendation put to Yuvraj and is waiting
> on his answer. On approval: stop both F01/F02, then launch G01 and G02 in parallel across the two GPUs
> (~45 min), then run the modality ablation on whichever wins.
>
> If the runs are left alone instead, F01/F02 will simply finish their remaining epochs and calibrate;
> their numbers would be recorded but are not expected to be usable.

## Pending: what the corrected-data run decides

Both running (18 ep @ ~6 min = **~2 h wall-clock**, parallel on GPU 0/1, calibration runs automatically
after each). **These are the first models in the project trained end-to-end under valid conditions** —
correct audio span, all three splits at median 50 frames, and a validation set that finally matches the
test distribution so checkpoint selection scores the right thing.

⚠️ **The current 66.36% headline should not be assumed to survive.** That checkpoint was trained before
the corrections and had its best epoch picked against a 96.2%-15-frame validation set — see
[`architecture.md` → Provenance of the current 66.36% baseline](architecture.md). It is a model chosen by
a broken signal and evaluated correctly, so F01/F02 may land either side of it.

**Queued behind F01/F02:** the modality ablation on the winner (decides question 2 below), then **F03**
ensembling (~30 min, inference-only, typically +1–2 pts) and **E22** seed repeat (~2 h per seed,
**required before any written audio-visual claim**).

Three questions turn on them. **⏳ All three are PENDING — neither run has completed.** Record each answer when they land; outcomes are pre-committed in [`plan.md` §13.10.1](plan.md).

| Question <span title="pending">⏳ **ALL PENDING**</span> | Current state | What F01/F02 decide |
|---|---|---|
| **Headline accuracy** | 66.36%, i.e. 1.25 below the best published baseline (67.61) | Most credible route to closing the gap. If they don't beat 66.36%, broken checkpoint selection was *not* the limiting factor — fall back to F03 + the ordinal-metrics framing |
| **The audio-visual claim** | +1.06 fusion gain, measured on a defect-trained checkpoint | Re-run the modality ablation on the winner. **The claim only survives if fusion still beats video-only on a cleanly-trained model.** Difference between an AV paper and a video paper with an ablation appendix |
| **Does augmentation fix overfitting?** | E04 lineage peaks at epoch 3, then decays | Controlled: F01 vs F02 differ only in frame cap + crop flag. If F02 holds its peak later, the crop becomes default. If not, it's a dataset-size limit (7,983 clips) — an argument for the purpose-built corpus |

⚠️ **Early concern (epoch 2): both runs are declining and below the majority baseline.**

| Run | ep 1 top-1 | ep 2 top-1 | ep 2 MAE |
|---|---:|---:|---:|
| F01 | 48.46 | 46.41 | 1.021 |
| F02 | 49.21 | 47.15 | 0.979 |

Likely cause: both train from the EfficientFace AffectNet pretrain rather than an existing engagement
checkpoint, so they learn the task from scratch in 18 epochs while the V12/E04 lineage reached 66% via
many more epochs of accumulated finetuning. **Pivot criterion: if neither crosses 55% top-1 by epoch 5–6,
stop them and instead finetune the best existing checkpoint on the corrected data** (~45 min, and a
cleaner design — it isolates correct preprocessing instead of confounding it with training length).
See plan.md §13.10.2.

**Metric discipline:** judge them on top-1, adjacent, MAE *and* macro-F1 together. A majority-class
predictor scores 50.27% top-1 at 16.73 macro-F1, so top-1 can rise by collapsing onto class 3. The ~52
macro-F1 against ~66 top-1 is the real weakness and should narrow, not just the headline.

## Where the remaining gains are — priority decision

Four levers are available. They are not equal in cost or in confidence, and one of them is runnable
right now.

| Lever | Expected gain | Cost | Blocked by | Confidence |
|---|---|---|---|---|
| **F03 — ensemble existing checkpoints** | +1–2 pts (typical) | ~30 min, inference only | nothing — **runnable now** | Medium-high; standard technique, 8 checkpoints available |
| **F01/F02 — retrain on corrected splits** | Unknown, likely largest | ~2 h each, parallel across 2 GPUs | T01 | High that it helps; size unknown |
| **T01 — train-split consistency** | Prerequisite | ~25 min | in progress | Certain (removes a known defect) |
| **`max_video_frames` 96 → 40/50** | Unknown | free (a flag) | T01 | Medium; enables augmentation that has never run |

**Why retraining is the priority despite the unknown size.** Every current checkpoint had its best epoch
selected against a validation set that was 96.2% 15-frame clips — the selection criterion was scoring the
wrong distribution. The 66.36% headline is a model chosen by a broken signal and then evaluated properly.
**Nothing here has ever been trained end-to-end under valid conditions**, so this is the one lever
addressing a known-broken part of the pipeline rather than tuning a working one.

**Why ensembling should still go first.** It is inference-only, needs no corrected training data, and the
gap to the best published baseline (67.61) is only 1.25 points — within the range ensembling alone typically
delivers. It costs 30 minutes and cannot invalidate anything else.

**What not to spend time on:** E15 (WavLM/HuBERT encoder swap), SpecAugment, and OGM-GE audio
rebalancing. E19 showed a linear model and a tree ensemble hit the same ceiling as our neural audio
branch, so encoder capacity is not the limitation on this corpus.

## Experiment Tracker — exp2026 sweep

| ID | Question | Config | Status | Result |
|---|---|---|---|---|
| A0 | Reproduce best under current code | V12 best, 3.6 s audio | Done | 64.05% (15-frame test) → **R01: 66.13%** corrected |
| E02 | Audio-only capability | V12 best, ablate video | Done | 46.41% (15-frame) → **R02: 50.27%** = majority predictor exactly |
| E03 | Video-only capability | V12 best, ablate audio | Done | 64.18% (15-frame) → **R03: 65.07%**, now **below** fusion's 66.13 |
| E04 | Does full 10 s audio help on finetune? | From V12, 10 s audio, ordinal 0.15, lr 5e-5, 8 ep | **Done** | Val 65.27% but **test 62.68% (−1.37 vs baseline)**; macro-F1 +2.28. **E06 resolves the ambiguity: the lift was not audio** |
| ~~E05~~ | ~~Full 10 s audio from scratch~~ | ~~lr 0.01, `--lr_scheduler step`, 20 ep~~ | **ABORTED — misconfigured** | `lr_steps` defaults to `[40,55,…]`, so LR never decayed in a 20-epoch run. Trained at constant 0.01, degrading 63.77→59.38. Kept at `results/exp2026/E05a_fresh10s_ce_constantlr_ABORTED/` |
| E05b | Full 10 s audio from scratch, corrected schedule | Fresh pretrain, 10 s audio, CE+LS 0.1, lr 0.005, **warmup_cosine**, grad-clip 5.0, selection on `f1_macro`, 18 ep | **STOPPED at ep 8** | Killed when the preprocessing mismatch was found — was training against a 15-frame test/val. Needs relaunch on corrected data |
| E06 | Does fusion beat video-only *after* the audio fix? | Modality ablation on E04 best | **Done — but WITHDRAWN** | Measured on 15-frame test. Corrected (R04–R06): fusion 66.36 ≈ video-only 66.22 for the 10 s model; the 3.6 s model does show fusion > video-only |
| E07/E19 | Does EngageNet audio carry engagement signal at all? | Encoder-free probe on hand-crafted acoustics | **Done** | Probe ≈ our branch — ceiling is in the data |
| E09 | Does synced random crop fix E04's overfitting? | From V12, 10 s audio, ordinal 0.15, lr 5e-5, `--train_frame_sampling random`, 6 ep | **Done — NULL EXPERIMENT** | Crop never fires (clips ≤63 frames vs `--max_video_frames 96`); val log **bit-identical** to E04 |
| E11 | Does sqrt-inverse class weighting close the macro-F1 gap? | From V12, 10 s audio, ordinal 0.15, lr 5e-5, `--class_weighting sqrt_inverse`, 6 ep | **STOPPED** | Killed with E05b for the same reason; relaunch on corrected data |
| R01–R06 | Recompute all headline numbers on corrected 50-frame test | Calibration-only, both audio spans × {AV, audio-only, video-only} | **Done** | **New best 66.36%**; fusion > video-only on all 4 decodes for the 3.6 s model |
| T01 | Re-extract the 1,822 remaining 15-frame train clips | `--splits Train --target_fps 5`, 6 shards | **Done** | **All three splits consistent for the first time** — Train 7,983 @ 0.1% / median 50; Val 1,071 @ 0.0% / 50; Test 2,257 @ 0.0% / 50 |
| ~~F01~~ | ~~Retrain on corrected splits, no augmentation~~ | ~~EfficientFace pretrain, 18 ep~~ | **FAILED pivot criterion** | Never crossed 55%; oscillates 45.4–53.6 with no trend. 18 ep from a generic pretrain is too short |
| ~~F02~~ | ~~Retrain with augmentation~~ | ~~As F01 but `--max_video_frames 40 --train_frame_sampling random`~~ | **FAILED pivot criterion** | Same shape; declining by ep 10 (40.6). **Crop was confirmed working** (`visual=(8,40,…)`, `audio=(8,64,347)` vs 432) — that finding stands |
| G01 | Does correct preprocessing help, isolated? | **Finetune best checkpoint** on corrected splits, 10 s audio, `--max_video_frames 50`, ordinal 0.15, lr 5e-5, 6 ep | Proposed (~45 min) | Replaces F01 |
| G02 | Does augmentation help, isolated? | As G01 but `--max_video_frames 40 --train_frame_sampling random` | Proposed (~45 min) | Replaces F02 |
| F03 | Checkpoint ensemble | Logit-average top checkpoints, inference-only | Proposed — runnable now | ~30 min; highest gain-per-minute of anything queued |
| E11r | Class weighting on corrected splits | `--class_weighting sqrt_inverse` | Proposed — blocked on T01 | ~45 min (6 ep); targets macro-F1 52 vs top-1 66 gap |
| E22 | Seed repeat of the fusion gain | Confirm +1.06 (13.6) before any paper claim | Proposed — blocked on T01 | ~2 h per extra seed; **required before any written AV claim** |

**E09/E11 rationale.** Both target a weakness already visible in the numbers, so both pay off regardless
of E06's outcome. E09: E04 peaks at epoch 3 then decays to 63.96% by epoch 5 — with 7,879 training clips
that is an augmentation gap. E11: every run sits at 44–48 macro-F1 against ~64% top-1 while adjacent
accuracy is ~91%, meaning the ordinal structure is learned but minority classes (class 3 is 47% of train,
50% of test) are not separated. SpecAugment and the E15 encoder swap are **deliberately deferred** until
E06/E19 confirm the audio branch carries signal — both are audio-specific and would waste GPU if it does not.

Modality ablation is rerun on E04/E05 best checkpoints on completion.

### E04 validation trajectory (10 s audio vs the 3.6 s baseline)

| Epoch | Val top-1 | Adjacent | MAE |
|---:|---:|---:|---:|
| 1 | 65.08% | 90.76% | 0.4631 |
| 2 | 64.99% | 90.76% | 0.4641 |
| **3** | **65.27%** | **91.04%** | **0.4585** |
| 4 | 64.52% | 90.01% | 0.4790 |
| 5 | 63.96% | 90.38% | 0.4799 |

Reference: V12 (3.6 s audio) validated at 64.61% / 89.73% / 0.4809. Every E04 epoch through 3 beats it
on validation. Peak is epoch 3; epochs 4+ overfit, so the selected checkpoint is epoch 3.

### E04 test results — the validation gain did NOT transfer

| Decode | V12 baseline (3.6 s) | E04 (10 s) | Δ top-1 |
|---|---:|---:|---:|
| argmax | 61.9681 | 62.1897 | +0.22 |
| logit bias | 63.6968 | 62.7216 | −0.98 |
| expected thresholds | 63.8741 | 62.8989 | −0.97 |
| **refined expected** | **64.0514** | **62.6773** | **−1.37** |

Macro-F1 moves the other way: 44.5441 → **46.8214 (+2.28)**.

**Fixing the audio truncation did not improve test top-1.** The earlier interim framing that E04 "beats
the baseline" was validation-only and did not hold out of sample — do not repeat it. Two unresolved
confounds: (1) a 2.6-point val/test gap, wider than the baseline's, with checkpoint selection on val
top-1 over only 1,071 clips, so part of the val gain is selection noise; (2) the −1.37 top-1 / +2.28
macro-F1 trade is exactly what the E19 probe predicts if audio *started* contributing, since its signal
sits in the minority classes that top-1 penalises. E06's ablation on this same checkpoint is immune to
both and is the deciding test. Full analysis in plan.md Section 12.8.

**Resolved by E06.** The ablation on this same checkpoint shows video-only (62.99) still beats fusion
(62.68) and audio-only collapses to the majority predictor. So E04's macro-F1 gain was **not** audio
starting to contribute — reading #2 above is ruled out. The remaining explanation is #1, checkpoint
selection noise on a 1,071-clip val split, plus ordinary run-to-run variance. **E04 is a neutral-to-mild
regression, not evidence for audio.**

**Config lesson from E05:** `--lr_scheduler step` uses `--lr_steps`, which defaults to `[40,55,65,70,…]`.
Any run shorter than 40 epochs therefore trains at a **constant** learning rate. Use
`--lr_scheduler warmup_cosine`, or pass explicit `--lr_steps`, for every short run.

### E07 — encoder-free audio probe (new this session)

`scripts/audio_signal_probe.py` fits logistic regression and histogram gradient boosting directly on
hand-crafted acoustics (log-mel mean/std over 40 bins, RMS, ZCR, silence fraction and run-switch rate,
spectral centroid/rolloff/flatness), with no neural encoder involved. It isolates whether the *data*
carries signal from whether *our encoder* extracts it:

| Probe outcome | Interpretation | Next step |
|---|---|---|
| ≫ 50.27% while our audio-only branch stays at 46.41% | Audio carries signal; the mel-CNN encoder is the failure | E15 — replace the encoder (frozen WavLM/HuBERT or direct prosody features) |
| ≈ 50.27% | EngageNet audio carries little clip-level engagement signal | Reframe the paper around robustness, not a fusion accuracy gain |
| Beats the fusion model | Summary statistics outperform cross-attention | Finding in its own right; rethink the architecture |

`--with_f0` adds `librosa.yin` pitch statistics but is ~30x slower per clip and starves the training
dataloaders (20 cores total, load hit 139 when run at 20 workers alongside both GPU jobs). Default run
is F0-free at 4 workers; rerun with `--with_f0` once the GPUs are idle, since F0 range is central to the
project's own engagement scoring formulas.

**Result (2026-08-07, 94 features, no F0):**

| Method | Function class | Top-1 | Macro F1 | Adjacent |
|---|---|---:|---:|---:|
| Majority-class predictor | constant | **50.27** | 16.73 | — |
| Our mel-CNN audio branch | deep, cross-attention | 46.41 | 18.18 | 70.04 |
| Logistic regression | linear | 46.28 | 29.92 | 68.26 |
| Histogram gradient boosting | tree ensemble | 46.05 | **31.74** | 71.19 |

**The ceiling is in the data, not the encoder.** Three independent function classes on different
representations land within 0.4 points of each other, all below a constant predictor. If the mel-CNN were
the bottleneck the probe would have beaten it. **E15 (WavLM/HuBERT swap) is therefore deprioritised.**

**Top-1 is the wrong metric for audio claims here.** A majority-class predictor gets 50.27% top-1 at only
16.73 macro-F1; the probe reaches 31.74 — nearly double. Audio carries real but weak signal concentrated
in the minority classes, and top-1 punishes using it because guessing class 3 is free. Always report
macro-F1 and adjacent accuracy alongside top-1 for audio-facing claims.

**Our branch underuses even that weak signal** (18.18 vs GBM's 31.74 macro-F1) — evidence of modality
collapse under a dominant video stream, not of encoder incapacity. Points at gradient-blending / OGM-GE
style rebalancing rather than a bigger audio encoder. Full analysis in plan.md Section 12.7.

**Full experiment compilation:** `results/exp2026/all_experiments.csv` — 57 rows covering every
`calibration_results.json` / `evaluation_testing.json` in `results/`, regenerate with
`python scripts/compile_experiments.py`.

**Data change:** all 11,307 EngageNet clips re-extracted at full length to `*_croppad10s.wav`
(10,869 with audio, 438 genuinely silent); 3.6 s originals retained. Annotation file for the full-audio
variant: `preprocessing/engagenet/annotations_engagement_a10.txt`.

## Pending longer AV-only retrain

| Change | Files | Status |
|---|---|---|
| Per-sample audio→video alignment | `models/multimodal_cnn.py` | Recalibrated on V12 (64.05%); full retrain still useful |
| Train-only synced random A/V crop | `src/data/temporal.py` | Short 3-ep tried; needs ≥15–30 ep for fair top-1 claim |

## Current Run Results

<table>
  <thead>
    <tr>
      <th>Run</th>
      <th>Best Val Top-1</th>
      <th>Test Top-1</th>
      <th>Test Top-5</th>
      <th>F1-Weighted</th>
      <th>F1-Micro</th>
      <th>Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>baseline_mel_h4</td><td>82.9167% (ep 40)</td><td>81.8750%</td><td>99.7917%</td><td>81.7620%</td><td>81.8750%</td><td>0.866828</td></tr>
    <tr><td>specaugment_mel_h4</td><td>80.0000% (ep 24)</td><td>79.3750%</td><td>99.5833%</td><td>79.4753%</td><td>79.3750%</td><td>0.912555</td></tr>
    <tr><td>audio_channel_gate_mel_h4</td><td>81.6667% (ep 6)</td><td>76.8750%</td><td>99.1667%</td><td>76.6528%</td><td>76.8750%</td><td>1.063809</td></tr>
    <tr><td>specaugment_audio_gate_mel_h4</td><td>81.6667% (ep 10)</td><td>68.9583%</td><td>100.0000%</td><td>68.9105%</td><td>68.9583%</td><td>1.103064</td></tr>
  </tbody>
</table>

## Run Changes

<table>
  <thead>
    <tr><th>Run</th><th>Recorded Change</th><th>Test Top-1</th></tr>
  </thead>
  <tbody>
    <tr><td>baseline_mel_h4</td><td>4 heads; LR 0.010; SpecAugment=false; audio_channel_attention=false</td><td>81.8750%</td></tr>
    <tr><td>specaugment_mel_h4</td><td>4 heads; LR 0.010; SpecAugment=true; audio_channel_attention=false</td><td>79.3750%</td></tr>
    <tr><td>audio_channel_gate_mel_h4</td><td>4 heads; LR 0.010; SpecAugment=false; audio_channel_attention=true</td><td>76.8750%</td></tr>
    <tr><td>specaugment_audio_gate_mel_h4</td><td>4 heads; LR 0.010; SpecAugment=true; audio_channel_attention=true</td><td>68.9583%</td></tr>
  </tbody>
</table>

## Per-Class Test Accuracy

<table>
  <thead>
    <tr><th>Emotion</th><th>baseline_mel_h4</th><th>specaugment_mel_h4</th><th>audio_channel_gate_mel_h4</th><th>specaugment_audio_gate_mel_h4</th></tr>
  </thead>
  <tbody>
    <tr><td>neutral</td><td>65.62%</td><td>75.00%</td><td>37.50%</td><td>75.00%</td></tr>
    <tr><td>calm</td><td>71.88%</td><td>71.88%</td><td>53.12%</td><td>45.31%</td></tr>
    <tr><td>happy</td><td>89.06%</td><td>81.25%</td><td>85.94%</td><td>57.81%</td></tr>
    <tr><td>sad</td><td>71.88%</td><td>64.06%</td><td>93.75%</td><td>56.25%</td></tr>
    <tr><td>angry</td><td>100.00%</td><td>100.00%</td><td>90.62%</td><td>62.50%</td></tr>
    <tr><td>fearful</td><td>68.75%</td><td>68.75%</td><td>56.25%</td><td>70.31%</td></tr>
    <tr><td>disgust</td><td>100.00%</td><td>100.00%</td><td>100.00%</td><td>100.00%</td></tr>
    <tr><td>surprised</td><td>79.69%</td><td>71.88%</td><td>78.12%</td><td>87.50%</td></tr>
  </tbody>
</table>

## Per-Class Test F1

<table>
  <thead>
    <tr><th>Emotion</th><th>baseline_mel_h4</th><th>specaugment_mel_h4</th><th>audio_channel_gate_mel_h4</th><th>specaugment_audio_gate_mel_h4</th></tr>
  </thead>
  <tbody>
    <tr><td>neutral</td><td>68.85%</td><td>66.67%</td><td>48.00%</td><td>57.83%</td></tr>
    <tr><td>calm</td><td>83.64%</td><td>83.64%</td><td>69.39%</td><td>62.37%</td></tr>
    <tr><td>happy</td><td>92.68%</td><td>87.39%</td><td>92.44%</td><td>72.55%</td></tr>
    <tr><td>sad</td><td>70.23%</td><td>62.12%</td><td>65.22%</td><td>52.17%</td></tr>
    <tr><td>angry</td><td>94.81%</td><td>92.09%</td><td>82.86%</td><td>76.92%</td></tr>
    <tr><td>fearful</td><td>71.54%</td><td>69.84%</td><td>70.59%</td><td>69.23%</td></tr>
    <tr><td>disgust</td><td>91.43%</td><td>84.77%</td><td>87.07%</td><td>79.50%</td></tr>
    <tr><td>surprised</td><td>74.45%</td><td>82.88%</td><td>83.33%</td><td>75.17%</td></tr>
  </tbody>
</table>

## Current Best

<table>
  <thead>
    <tr><th>Run</th><th>Best Val Top-1</th><th>Test Top-1</th><th>Test Top-5</th><th>UAR</th><th>F1-Weighted</th><th>F1-Macro</th></tr>
  </thead>
  <tbody>
    <tr><td>baseline_mel_h4</td><td>82.9167% (ep 40)</td><td>81.8750%</td><td>99.7917%</td><td>80.8594%</td><td>81.7620%</td><td>80.9552%</td></tr>
  </tbody>
</table>

## Historical Run History

<table>
  <thead>
    <tr>
      <th>Run</th>
      <th>Test Top-1</th>
      <th>Test Top-5</th>
      <th>UAR</th>
      <th>F1-Weighted</th>
      <th>F1-Macro</th>
      <th>Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>spec_01_baseline</td><td>66.6667%</td><td>96.6667%</td><td>66.9922%</td><td>65.6612%</td><td>65.8484%</td><td>1.312542</td></tr>
    <tr><td>spec_02_retrain_h4_e70</td><td>60.0000%</td><td>98.5417%</td><td>60.3516%</td><td>58.8741%</td><td>59.5695%</td><td>1.293564</td></tr>
    <tr><td>mel_h1_lr001_e75</td><td>70.8333%</td><td>98.3333%</td><td>70.8984%</td><td>70.6221%</td><td>70.4993%</td><td>1.063559</td></tr>
    <tr><td>mel_h8_lr001_e75</td><td>71.2500%</td><td>99.5833%</td><td>72.0703%</td><td>70.2509%</td><td>69.8308%</td><td>0.810862</td></tr>
    <tr><td>v2_h4_lr001_rerun1</td><td>75.2083%</td><td>99.5833%</td><td>72.8516%</td><td>74.8032%</td><td>73.2530%</td><td>1.010606</td></tr>
    <tr><td>v2_h8_lr001_rerun1</td><td>78.5417%</td><td>100.0000%</td><td>76.3672%</td><td>77.9270%</td><td>76.4879%</td><td>0.891395</td></tr>
    <tr><td>v2_h8_lr005_rerun1</td><td>77.5000%</td><td>98.7500%</td><td>76.7578%</td><td>77.1586%</td><td>77.1089%</td><td>0.993458</td></tr>
    <tr><td>v2_h8_e100_rerun1</td><td>72.9167%</td><td>100.0000%</td><td>69.7266%</td><td>72.1371%</td><td>69.8160%</td><td>1.036233</td></tr>
  </tbody>
</table>

## Session 2026-08-12 — F01/F02 verdict, calibration defect, overnight G-sweep launched

**F01/F02 finished and failed.** Neither approached the 66.36% incumbent. Best validation top-1 was
53.59 (F01, epoch 8) and 53.97 (F02, epoch 8), against E04's 65.27. The §13.10.2 pivot criterion was
correct: 18 epochs from the AffectNet face-recognition pretrain is too short to learn this task.

**A fourth evaluation defect was found.** `scripts/exp2026_run.sh` hardcodes `--max_video_frames 96` in
`COMMON` and does not forward each run's own override to calibration. F01 trained at 50 frames and F02
at 40; both were tested at 96. **Their test numbers are a train/eval mismatch and must not be cited.**
Their validation curves are unaffected (validation runs inside training at the correct cap), and the val
curves alone support the failure verdict. Fixed in the new harness, which reads the cap back out of each
run's own `opts*.json`.

**Overnight G-sweep launched** 2026-08-12 04:11 UTC across both RTX 3090s, detached under `setsid nohup`
so it is independent of the SSH session or the laptop sleeping. Seven matched finetunes, all warm-started
from the E04 best checkpoint onto the corrected splits, with everything held fixed except one named
variable per run. Design and rationale in [`plan.md` §13.13](plan.md).

| Run | GPU | Epochs | Variable under test | Status |
|---|---|---:|---|---|
| G00 | 0 | 8 | control — E04's exact config on corrected data | running |
| G01 | 0 | 6 | `mvf 50` (no padding waste) | queued |
| G03 | 0 | 6 | `class_weighting sqrt_inverse` (E11r, macro-F1) | queued |
| G02 | 1 | 8 | `mvf 40` + random crop (augmentation) | running |
| G04 | 1 | 6 | `class_balance_sampler sqrt_inverse` | queued |
| G05 | 1 | 6 | `lr 1e-4` + warmup-cosine | queued |
| G06 | 1 | 6 | `spec_augment` | queued |

**Phase 2 generates itself** when all seven finish: modality ablation on the top two runs (§13.10
question b — does the AV claim survive clean training), two seed repeats of the winner (E22), and F03
checkpoint ensembling. Estimated phase-1 completion ~09:15 UTC, phase 2 ~10:45 UTC.

**New capability:** `--save_every_epoch` retains `epochs/epoch_NNN.pth`, so any epoch can be re-selected
post-hoc against a different `--selection_metric` without retraining, and every epoch is available as an
ensemble member.

Progress at any time:

```bash
python scripts/night/collect.py
```

## Session 2026-08-12 (cont.) — Streamlit UI repointed to engagement

The UI had been serving the RAVDESS 8-class emotion model. `ui/app.py` and `ui/inference.py`
were rewritten for EngageNet engagement; the RAVDESS path was removed.

| Item | Status |
|---|---|
| Curated checkpoint registry (E04 66.36%, V12-05 66.13%) with calibrated thresholds | ✅ Done |
| Long-video windowing (10 s windows, optional 50% overlap) + session rollup | ✅ Done |
| Preprocessing parity with training arrays | ✅ Verified — 0.0 mean abs frame diff, scores within 0.02 |
| Optional audio-only / video-only modality split via `ablate_modality` | ✅ Done |
| `facenet_pytorch` missing → blocking error instead of silent Haar fallback | ✅ Done |

**Defect found and fixed during this work:** the first implementation sampled 96 frames per
window, reading `--max_video_frames 96` as the frame budget. The stored clips are ~50 frames
because `extract_faces.py` ran at `--target_fps 5`; the 96 cap never binds. Under the wrong
stride only 6/12 test clips decoded to the same level as the training path. After the fix,
scores match to within 0.02. Recorded in [memory.md](memory.md).

**Run it:**

```bash
conda activate avtca && streamlit run ui/app.py
```

The `avtca` env is required — outside it `facenet_pytorch` is missing and face crops fall back
to a Haar cascade that does not reproduce the training crops.

**Follow-up:** when the overnight G-sweep produces a run above 66.36%, add it to
`MODEL_REGISTRY` in `ui/inference.py` with its own calibrated thresholds.

## Session 2026-08-12 (overnight) — G-sweep complete: 14 runs, 13 ablations, flat result

**All runs finished.** Fourteen matched warm-start finetunes across two RTX 3090s, plus a fusion /
video-only / audio-only ablation on 13 of them and a 14-member ensemble. Full analysis in
[`plan.md` §13.14](plan.md). All numbers use the fixed decoder `refined_expected_thresholds`.

**Headline: nothing beat the incumbent, and we now know why.**

| Source of variation | Top-1 sd | Spread |
|---|---:|---:|
| 14 different configurations | **0.52** | 1.95 |
| Same config, 3 seeds | **0.45** | 0.89 |

Varying every hyperparameter tested produces about as much variation as changing the random seed. The
corpus is the ceiling, not the model. This converges with the §12.7 encoder-free probe and is the
empirical case for the purpose-built dataset.

**The three pre-committed questions (§13.10.1), answered:**

| Question | Answer | Evidence |
|---|---|---|
| Beats 66.36% top-1? | **No** | Best 66.67 (+0.31, inside 0.45 seed sd). Control G00 = 66.09 vs E04 66.36 — the preprocessing/selection fix changed accuracy by −0.27 |
| Fusion still beats video-only? | **Top-1 yes, macro-F1 no** | Top-1 12/13 models positive, mean +0.67. macro-F1 4/13, mean −0.09. Audio-only never exceeds the majority baseline |
| Augmentation holds its peak? | **No** | G02 (crop active) 64.89 vs identical G01 65.82 — second-worst run |

**§13.6's macro-F1 fusion claim (+1.95) is withdrawn** — it does not reproduce on cleanly-trained
weights. The narrow top-1 claim survives.

**A mid-sweep finding was retracted within two hours.** G04 appeared to show that audio's minority-class
contribution required class-balanced training (macro-F1 +2.65 for fusion). Seed 2 of the identical config
gave −0.61. Seed noise. Recorded in §13.14.2 because it is exactly what E22 existed to catch.

**Every past single-run comparison in this project is now suspect.** Measured noise floor is 0.45 sd on
top-1 and ~1.1 sd on macro-F1; the old sweep tables quoted 0.5–2 point differences between single runs.
Future comparisons need ≥3 seeds.

| Status | Count |
|---|---|
| Training runs completed | 14 / 14 |
| Modality ablations | 13 |
| Failed runs | 0 |
| 14-member ensemble | running |

**New in the harness:** `scripts/night/` — memory-gated multi-worker queue driver, auto-generated phase 2,
cross-model ablation, probability-averaged ensembling. `--save_every_epoch` in `src/config/opts.py`.

**Ensembling (F03) refuted — final item closed.** 14-member ensemble scored 65.69 top-1; a top-5 variant
scored 65.60. Dropping the weak members changed nothing, so the failure is a lack of decorrelated error
among members that all warm-start from the same checkpoint, not dilution. Both ensembles are worse than
the best single model (66.67) on all four metrics. Detail in [`plan.md` §13.14.5](plan.md).

**All four levers from §13.7 are now closed and none produced a gain:** retraining on corrected splits
(§13.14.2a), train-split consistency, ensembling (§13.14.5), and the two wasted settings (§13.14.3).
The sweep is complete; the remaining contributions are the ordinal-evaluation framing, the negative audio
result, and the noise-floor finding.

**Label-granularity finding (§13.14.6).** Merging the two middle engagement levels lifts accuracy from
64.27 to 70.70 (+6.43, ~7x the seed noise) with no retraining; binary reaches 85.95. Per-class recall at
4 levels is 75.5 / 20.3 / 31.0 / 81.6 — the middle levels are not separable, and 28.1% of all predictions
are off-by-one. **This challenges the planned 5-level scale in the dataset design and should be resolved
before collection starts.**

## Collaborator branch evaluation — `feat/behavior-text-fusion` (2026-08-16)

First external contribution (Gakshith), evaluated in an isolated worktree. **Not merged.**

| Item | Status |
|---|---|
| Branch reviewed (6 commits, +1595/-23, 25 files) | done |
| Test suite on his branch (283 passed, 191 subtests) | done |
| Isolated worktree at `/home/922933190/AVTCA-collab-test` | done |
| Matched A/B training, 7 epochs, one arm per GPU | done |
| A_control result — best 67.69 (UAR 56.04, adj 93.74) | done |
| B_text_fusion result — 62.47, no `_best.pth` written | done, but **not interpretable** |
| Root cause of B collapse identified (3 silent defects) | done |
| C1 seed `classifier_fused` from `classifier_1`, re-run B | open |
| C2 fix `_facecroppad` behavior filename mismatch | open |
| C3 assert `present.mean() > 0` on absent modality | open |
| C4 install OpenFace + extract 11,311 clips | open |
| C5 decide transcripts vs AU captions for text fusion | open (author) |
| C6 fix `--n_epochs` / `begin_epoch` resume semantics | open |

**Headline:** the branch is well-engineered and nothing crashed, but on this machine both `--behavior` and
`--text_fusion` are untestable — the text path is downstream of OpenFace features that do not exist here, so
the text stream was a constant for all 11,206 clips. The 62.47 measures a randomly-initialised classifier
relearning from scratch, not his idea. Source videos are present (11,311 `.mp4`), so a real evaluation is
feasible once C2-C4 are done. Detail in [`plan.md` §15](plan.md).

**Also uncovered (pre-existing, ours):** `--n_epochs` is overridden by `begin_epoch` on resume, so
`--n_epochs 6` in the G-sweep trained 3 epochs — **past sweep epoch counts are overstated.**

## Session 2026-08-17 — text-modality v2: label-leakage fix, behavior captions, matched A/B launched

| Task | Status |
|---|---|
| Audit v1 chat text (`audit_text_leakage.py`) — 792 strings, 99.84% label-deterministic, BoW 97–98% held-out | done — v1 confirmed label oracle |
| Root-cause why oracle text still hurt: random `av_context` bottleneck at init (not hash collision, not weak correlation) | done — architecture.md corrected |
| Label-free behavior captions from OpenFace (300,22) (`behavior_caption.py`, train-only tertiles) | done — 100% coverage, 6,614 unique captions |
| `annotations_engagement_v2{,_a10}.txt` (cols 1–4 byte-identical to v1; v1 backed up to `backup_2026-08-17/`) | done |
| `LateTextFusionV2` zero-init residual + `--text_fusion_arch` flag; `--late_text_fusion` default flipped OFF | done — E04 loads 546/0-skipped/17-init |
| Calibration text defaults 48/8192 → 32/4096; `run_job.sh` calibrates at run's own annotation/text config | done |
| Tests (`tests/test_text_fusion_v2.py`) + full suite | done — 231 passed |
| T10 (AV control) vs T11 (text residual), 3 seeds each, warm-start E04, mvf 96 | **3/6 done** (2026-08-17 20:21) — T10s3, T11s2, T11s3 all 68.07 val top-1 (arms tied; +1.7 vs warm-start). GPU-1 queue drained; T11s1/T10s2/T10s1-rerun blocked on GPU 0 behind collab S3 chain (T10s1 first attempt OOM'd in memory-gate race, requeued) |
| Collect + decide (≥ +0.9 = claimable vs seed sd 0.45) | open — after runs finish |
| **Generated-chat corpus (v3)** — 20 Claude subagents, 6,614 captions × 3 label-blind variants; `annotations_engagement_v3_a10.txt` 39.6% coverage weighted to quiet clips (79/35/5% by RMS tertile); audit PASSED (BoW 51.0/49.0 vs baselines 49.3/47.5) | **done 2026-08-17** — `generate_chat.py`, `chat_generation_cache.json`, `audio_rms_cache.json` |
| AV vs AV+generated-chat A/B (3 seeds/arm, v3 annotations, LateTextFusionV2) | open — GPUs busy with T10/T11 + collab jobs |

## Behavior modality — real OpenFace evaluation (2026-08-18)

Continues the 2026-08-16 collaborator row. Branch **still unmerged**; all work in the worktree.

| Item | Status |
|---|---|
| OpenFace 2.x built from source (OpenBLAS + dlib-cpp fixes) | done |
| AU extraction, 11,311 clips | done — **0 errors, 100% resolved, 98.78% face-detected** |
| Filename `_facecroppad` fix (C2) | done |
| Loud-failure guard on absent modality (C3) | done |
| Calibration script behavior flags | done |
| Subject IDs from filenames (133 subjects) + leakage-free baselines | done |
| From-scratch A/B, 15 ep | done — B **+1.76** mean top1, **+1.79** UAR |
| Hyperparameter sweep, 10 runs | done — winner lr 3e-3 + cosine (60.10) |
| Long from-scratch, 60 ep | done — B **61.25** vs A 58.64; **B overfits after ep14** |
| Zero-init seeded warm start (C1) | done — verified **exact no-op**, max logit diff 0.0 |
| Warm A/B + calibrated test metrics | done — see below |
| C6 `--n_epochs`/`begin_epoch` resume semantics | open |
| C7 **re-run warm A/B at mvf 50** (match 66.36 conditions) | **done 2026-09-12** — no effect (B 66.08 ± 0.59 vs A 66.27 ± 0.45) |
| C8 seed-replicate A control | **done 2026-09-12** — 66.27 ± 0.45, reproduces G04 exactly |
| C9 ablate behavior vs text separately | **done 2026-09-12** — B 66.08, C 66.12, A 66.27: neither moves any metric |
| C10 report all fixes back to the author | open |

**Headline result — test set, calibrated:**

| | A control | B behavior+text |
|---|---|---|
| Best test top-1 | **65.47** | 65.25 |
| Best macro-F1 | 51.98 | **53.49** |
| Best adjacent | 91.93 | **92.64** |

**Verdict:** behavior+text does **not** improve top-1 under warm start (redundant with what the trained
model reads from pixels) but **does** improve minority-class and ordinal performance — the documented
largest quality gap. From scratch it is worth +1.76 to +2.61 top-1.

**Caveat blocking interpretation:** these ran at `--max_video_frames 96`; the 66.36 headline was measured
at 50. The A control's 65.47 (-0.89 vs 66.36) may be the frame cap, not a reproduction failure. Single
seed throughout. Resolve C7 before comparing any of these numbers to 66.36.

## Paper writing — plan after supervisor meeting 2026-09-08 (plan only, nothing implemented)

Supervisor: **Sanchita Ghose** (spelling confirmed from the Zoom meeting record; supersedes the "Goes" flag above).
Next review **Tue 2026-09-15, 7 PM**. Full plan in [`plan.md` §18](plan.md).

| # | Task | Owner | Status |
|---|---|---|---|
| P1 | C7 matched re-run (behavior vs AV, mvf 50) — prerequisite for the results table | Yuvraj | **done 2026-09-12** — behavior branch: no effect; the paper's behavior claim becomes the late-fusion result (plan.md §19.6) |
| P2 | Correct meeting misstatement: 65.47 is the AV control, behavior arm 65.25 (−0.22 top-1, +3.18 macro-F1) | Yuvraj | open |
| P3 | Related work 9 → 15–20 citations | Yuvraj | **done 2026-09-08** — 21 bib entries, all cited; 2 entries carry TODOs (author list, volume) |
| P4 | Preliminaries (~1 page) | Yuvraj | **done** — Yuvraj's section merged as §III; later sections reference its equations |
| P5 | Model Evaluation section, metric-grouped + ablations + hard-category discussion | Yuvraj | **drafted 2026-09-08** in `papers/research/paper.tex` (Sections III–V, 8 tables); behavior row awaits C7 |
| P6 | draw.io architecture diagram (non-AI) → `papers/research/figures/` + shared folder | Yuvraj | open |
| P7 | Call Akshit tonight: his visual-emphasis variant + section split | Yuvraj | open |
| P8 | Methodology section, 4 grouped components | Akshit | **first draft written 2026-09-08** (Section III, 4 subsections, 5 equations) — Akshit to revise; visual-emphasis variant commented out |
| P9 | Conceptual diagram for Introduction | Yuvraj | open |
| P10 | Feasibility note: retrain behavior stream on asynchronous-class recordings | Yuvraj | open |
| P11 | Send data-collection asks (per-participant audio, breakout rooms, chat export) before Sanchita's 09-09 meeting | Yuvraj | open |

## Session 2026-09-12 — full-clip audit and matched A/B/C at the video clock

**Request:** use the complete video and complete audio of every clip, mapped 0→end, then train the
EfficientFace AV model and the behavior model and report an absolute answer. Full detail in
[`plan.md` §19](plan.md).

| Item | Status |
|---|---|
| Probe all 11,311 source clips vs stored face arrays and wavs | **done** — max source duration **10.06 s**; 0 clips over 10.5 s; the "10 s" window is the whole clip; audio (431 mel frames) and video (50 frames @ 5 fps) already span 0→end and are pooled onto one clock |
| Find any stream not on the full-clip clock | **found one** — behavior (OpenFace) was resampled to **15 steps** (1.5 fps) against 50-frame video, in every §16 run |
| `--behavior_frames` (default = `--max_video_frames`), plumbed through opts / dataset / calibration / config identity | **done** (worktree) — 40 behavior tests + 3 new pass; step-to-frame offset ≤200 ms vs 667 ms before |
| Seeded warm starts for behavior-only (4×448) and behavior+text (4×576) | **done** — `seed_warmstart.py`, both verified **max abs logit diff 0.0** vs E04 on real batches |
| End-to-end smoke of all three arms through the runner (train → calibrate → JSON) | **done** |
| Arm A audio+video EfficientFace × 3 seeds (G04 config, mvf 50) | **done** — **66.27 ± 0.45** test top-1 / 91.55 adj / 0.440 MAE / 52.31 macro-F1; every epoch bit-identical to the August G04 seeds |
| Arm B + behavior (50-step clock) × 3 seeds | **done** — **66.08 ± 0.59** / 91.62 / 0.441 / 52.16: **B − A = −0.19, no effect on any metric** (C7 answered; §16.7's +3.18 macro-F1 was seed noise) |
| Arm C + behavior + caption-text × 3 seeds (C9) | **done** — **66.12 ± 0.59** / 91.67 / 0.439 / 52.26: equal to A and B at every decoder; the caption stream adds nothing |
| Results table, fixed decoder `refined_expected_thresholds`, 3-seed means ± sd | **done** — plan.md §19.5; `python scripts/fullclip/collect.py [--markdown]` |
| Modality ablations, all 9 runs | **done** — fusion − video-only **+0.74** mean, 8/9 ≥ 0; audio-only ≤ majority (50.27) on every run |
| Segment transformer (our implementation of the ICMI tokenisation; 8 configs × 3 seeds, val-selected T7 d64 L2) | **done** — alone **67.0** test (thresholds) / 67.6 (argmax) with 22 features, = published 67.61; noisy across seeds (±3) and overfits by epoch 2–12 |
| **Three-way fusion AV + transformer + GBM** (weights/thresholds on validation) | **done** — **70.29 ± 0.14** on the 3 A seeds, **70.15 ± 0.25** over all 8 neural checkpoints (min 69.68); weights 0.6/0.2/0.2 on 7 of 8; best and most stable number in the repo (plan.md §19.7) |
| Two-way AV + transformer | done — 68.69 ± 2.08: the transformer's validation score is a poor weight guide (seed 1 picked 0.85/0.15 → 66.31); keep the GBM in the fusion |
| Commit and push: development's pending work (5 commits) + merge of the behavior branch with the full-clip pipeline (`feat/behavior-fullclip`) | **done 2026-09-12** — 293 tests pass; A/B smokes identical post-merge; pushed to `origin/development` |
| A seed 1 calibrated test | **done** — 65.78 / 92.24 adj / 53.41 macro-F1, identical to G04 seed 1 (deterministic; proves worktree AV path == main tree) |
| **Top-1 levers measured** (plan.md §19.6) on the G04-s1 checkpoint | **done** — neighbour-clip smoothing: null (65.16 vs 65.69); EM prior shift: hurts (62.85); oracle test-fit thresholds 67.82 (bound only); confusion: 531/774 errors in classes 1–2 |
| OpenFace-statistics GBM alone (20 segments × mean/std of the 22-d series, CPU, no NN) | **done** — **67.15 test top-1** / 53.19 macro-F1, equals the pixel model and is 0.46 below the published best |
| **Late fusion AV + GBM** (probability averaging, weight + thresholds selected on val) | **done, seed 1** — **70.12 test top-1** / 90.69 adj / 0.408 MAE / 53.62 macro-F1; robust over w 0.4–0.8 (69.1–70.1); +4.4 over AV alone, **+2.5 over the published best 67.61**. **Seeds 2 and 3 replicate: 69.28, 70.17** → **3-seed mean 69.86 ± 0.50** vs AV alone 66.27 ± 0.45 (+3.59) and published best 67.61 (+2.25). Also holds with arm B as the neural member: `B_beh_s1` 66.80 → **69.77**, `B_beh_s2` 66.22 → **70.04**. Four members, mean **69.80**. |

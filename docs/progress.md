# AVTCA Progress — Current Verified Results
**RAVDESS · 8-Class Emotion Recognition · 2026-05-28**

> ## Status at a glance — 2026-08-07
>
> **All preprocessing is now correct.** Three defects were found and fixed this period: audio truncated
> to 3.6 s of each 10 s clip; test/validation extracted at 15 frames against 50-frame training; and 22.8%
> of the training split at 15 frames. All three splits now sit at median 50 frames, `--target_fps 5`.
>
> **F01 and F02 are running** (~2 h) — the first models in this project trained end-to-end under valid
> conditions. **Three substantive questions depend on their results:** whether the 66.36% headline
> improves, whether the audio-visual gain survives clean training, and whether temporal augmentation
> fixes the recurring epoch-3 overfitting. Decision matrix in
> [`plan.md` §13.10.1](plan.md).
>
> **Every number recorded below F01/F02 is defect-provenance** — kept for the record, not for citation.

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

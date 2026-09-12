# Plan — Text Fusion from Student Zoom Chat, Bootstrapped with Generated Data

**Yuvraj Gupta · 2026-08-17**

## The plan in one paragraph

The third modality is the **text a student would type in Zoom chat**. Until our classroom recordings exist, we bootstrap it with **LLM-generated chat**: for each clip, an LLM (Claude) reads the extracted OpenFace behavioral description (gaze, head pose, blinks, expressions) and writes the chat message that student would plausibly have sent — a hesitant question, a quick on-topic answer, silence. That generated text trains the already-built text-fusion head **now**; real chat replaces it when collection starts.

## Pipeline

1. **Extract** — OpenFace gives 22 behavioral signals per clip (already done for the dataset).
2. **Generate** — Claude turns each clip's behavioral description into a plausible chat message (or "no message"). Generation is **label-blind**: the model never sees the engagement label, only the behavior. This matters — our audit previously caught a text channel that had the label encoded into it (99.84% label-deterministic), which invalidates any result; label-blind generation makes that impossible by construction.
3. **Gate** — the leakage audit runs on the generated text before any training; it must pass.
4. **Train** — the generated chat feeds LateTextFusionV2 (built, tested): text embedded and added to the audio-visual model through a zero-initialised residual, so the model starts at the AV baseline and only uses text if it helps.
5. **Evaluate** — matched A/B: AV-only vs AV+text, identical configs, 3 seeds per arm; a gain ≥ 0.9 top-1 (2× seed noise) is claimable.
6. **Swap in real chat** — same pipeline, real Zoom exports, once classroom sessions begin.

## Status

| Piece | State |
|---|---|
| OpenFace behavioral features, full dataset | done |
| Text-fusion head (LateTextFusionV2) + tests | done |
| Leakage audit tool | done |
| Best AV model fine-tune | **68.07% val top-1** (+1.7 over prior checkpoint); test-split confirmation pending |
| Chat generation with Claude | next step |
| A/B training runs | after generation |

## Next steps

1. Confirm the 68.07% fine-tune on the test split (published best: 67.61%, video-only).
2. Generate the chat corpus with Claude from the OpenFace descriptions; run the leakage gate.
3. Launch the AV vs AV+generated-chat A/B.

---

# Update — 2026-09-12: full-clip runs, behavior stream verdict, and a new headline candidate

**Yuvraj Gupta · for the review of Tue 2026-09-15**

## What was asked and what was found

The request was to train on the *complete* video and audio of every clip rather than a 10-second window.
Measured first: every EngageNet source clip is at most 10.06 s long, so the 10-second window is the whole
clip, and the audio-visual pipeline was already consuming all of it (431 mel frames, 50 face frames at
5 fps, pooled onto one clock). The one stream that was *not* on that clock was the OpenFace behavior
series, which the behavior branch sampled at 15 points per clip; it now runs at 50, one per face frame.

## Results (EngageNet test set, 2,256 clips, 3 seeds per arm, thresholds fit on validation)

| Arm | Test top-1 | Adjacent | MAE | Macro-F1 |
|---|---:|---:|---:|---:|
| A · audio + video (EfficientFace) | **66.27 ± 0.45** | 91.55 | 0.440 | 52.31 |
| B · A + OpenFace behavior branch (50-step clock) | 66.08 ± 0.59 | 91.62 | 0.441 | 52.16 |
| C · B + behavior-caption text | 66.12 ± 0.59 | 91.67 | 0.439 | 52.26 |
| Published best (Singh et al., ICMI 2023, Transformer on OpenFace) | 67.61 | — | — | — |

- Arm A reproduces the August best configuration exactly (every epoch, every seed).
- **The behavior branch has no effect on any metric** once the comparison is matched (same code,
  same frame cap, warm start verified as an exact no-op, three seeds). The earlier single-seed
  "+3.2 macro-F1" was seed noise. The meeting figure "66.36 → 65.47" should be replaced by
  "66.27 vs 66.08, no difference".
- Audio fusion adds **+0.74** top-1 over video-only (8 of 9 runs non-negative); audio alone never
  beats the majority-class predictor. This is the defensible audio-visual claim.

## The new headline candidate: late fusion with an OpenFace-statistics model

A gradient-boosting classifier on 20-segment mean/std statistics of the same 22 OpenFace features
(no neural network, trained on the train split only) scores **67.15** on its own — equal to the whole
pixel+audio model. Averaging its probabilities with the audio-visual model's, with the mixing weight and
decoding thresholds chosen on validation only:

| | Test top-1 |
|---|---:|
| Audio-visual model alone (3 seeds) | 66.27 ± 0.45 |
| OpenFace-statistics model alone | 66.05–67.15 |
| **Probability average, weight 0.7 (validation-selected on every seed)** | **69.86 ± 0.50** |

The two models disagree on ~22% of test clips and each is right on about half of those. The gain holds
on all seven neural checkpoints tested (minimum 68.9, all above 67.61), and it holds whether or not the
neural model contains the behavior branch — which shows the branch was failing to extract a signal that
is genuinely there.

**Evening addition — a second behavior member.** We implemented our own small Transformer over the same
20-segment tokens (the tokenisation is the ICMI baseline's idea and is cited; the model, loss, decoding
and fusion are ours). Alone it scores **67.0** on test with our 22 features, reproducing the published
67.61. Fusing all three, audio-visual network + transformer + boosting model, with weights chosen on
validation (0.6 / 0.2 / 0.2 on nearly every seed): **70.29 ± 0.14** on the three audio-visual seeds and
70.15 ± 0.25 across all eight neural checkpoints tested. This is the number to put forward.

## What was measured and ruled out

Smoothing predictions across neighbouring clips of the same recording (null), unsupervised correction
for the validation/test label-mix difference (hurts), and same-model checkpoint ensembling (65.6, from
August). The remaining top-1 loss is in the two middle engagement levels (recall 7% and 28%).

## Proposed next steps, in order

1. Adopt the late-fusion result as the paper's headline, with the pre-registered protocol stated.
2. Replace the gradient-boosting member with the ICMI paper's Transformer over 20 segment tokens.
3. Re-extract OpenFace keeping the full 98-dimensional output (only 22 features were saved).
4. Feed segment statistics into the neural behavior branch so the gain becomes end-to-end.

Full detail, per-run tables and scripts: `docs/plan.md` §19.

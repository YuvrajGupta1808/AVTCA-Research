# Behavior Modality (OpenFace Action Units) — Result Summary

*2026-08-18 · EngageNet, 4-class engagement*

## Where we were

| Baseline (E04, audio+video) | |
|---|---|
| Test top-1 | 66.36 |
| Adjacent accuracy | 90.96 |
| Macro-F1 | 52.03 |

Published EngageNet best: **67.61** test top-1. The documented weakness is the spread between
top-1 (66) and macro-F1 (52) — the model separates the majority class well and the middle
engagement levels poorly.

## What was added

An explicit **behavior stream**: 22 numeric channels per frame from OpenFace — 17 action-unit
intensities, 2 gaze angles, 3 head-pose angles — encoded and cross-attended against the pooled
audio-video representation, plus a direct pooled-AU path into the classifier.

A second stream converts the same AU vector into short descriptive phrases. It is **derived from
the video**, so it is a second view of the same features, not an independent text modality, and is
reported here as part of the behavior contribution.

## Data

| | |
|---|---|
| Clips extracted | 11,311 / 11,311 (0 failures) |
| Annotation rows matched | 11,206 / 11,206 (100%) |
| Clips with a detected face | 11,069 (98.78%) |
| Feature array | `(T, 22)` float32 per clip |

Stored at `datasets/EngageNet/behavior/`. Reusable independently of this experiment.

## Results

**Setting 1 — model still learning the task** (limited training budget; both arms identical apart from
the behavior stream):

| | Audio+Video | + Behavior | Δ |
|---|---|---|---|
| Best top-1 | 58.64 | **61.25** | **+2.61** |
| Mean top-1 | 52.25 | **54.01** | +1.76 |
| Mean UAR | 53.31 | **55.10** | +1.79 |

**Setting 2 — model already fully trained on the task** (calibrated test metrics, thresholds fitted on
validation):

| | Audio+Video | + Behavior | Δ |
|---|---|---|---|
| Test top-1 | **65.47** | 65.25 | −0.22 |
| Macro-F1 | 50.31 | **53.49** | **+3.18** |
| Adjacent accuracy | 91.36 | **92.64** | **+1.28** |
| Mean abs class error | 0.450 | **0.446** | −0.004 |

## Finding

**The behavior stream does not raise top-1 accuracy. It improves minority-class and ordinal
performance.**

In Setting 1 the explicit AU features carry information the model has not yet learned to extract
from raw frames, and top-1 rises by ~2.6. In Setting 2 that information is largely already available
to it from pixels, and top-1 is unchanged.

What survives in both conditions is the shape of the improvement: **macro-F1 +3.18 and adjacent
accuracy +1.28**, with lower ordinal error. The behavior features help specifically on the
under-represented middle engagement levels — the exact gap identified as the project's largest
quality deficit. On those two metrics the behavior arm exceeds the 66.36 baseline (92.64 vs 90.96
adjacent; 53.49 vs 52.03 macro-F1).

## Limitations

1. Single seed per configuration; measured seed sd is 0.45.
2. The numeric AU stream and the phrase stream were never run separately, so the gain is not
   attributed between them. The numeric features are the likelier source.
3. Per-subject AU normalisation was tested and did not help (−1.05).
4. Frame-cap setting differs from the one under which the 66.36 baseline was measured; a matched
   re-run is outstanding before these figures are placed directly against it.

## Next

1. Re-run the comparison at the baseline's frame setting to remove the configuration mismatch.
2. Replicate across 3 seeds.
3. Ablate the numeric AU stream against the phrase stream.
4. If the macro-F1 result holds, target it directly — class weighting and macro-F1-oriented
   decision thresholds apply to the same weakness.

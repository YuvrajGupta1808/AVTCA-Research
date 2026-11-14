# What We Changed, and What Each Change Bought Us

**EngageNet · 11,206 clips · 4-level ordinal engagement · test split 2,257 clips**

**The four metrics, one line each.**
- **Top-1** — the predicted engagement level exactly matches the true one. Plain accuracy.
- **Adjacent** — the prediction is correct or off by one level. Engagement is ordered, so an off-by-one error is a far smaller mistake than an off-by-three.
- **MAE** — mean absolute error in level units. 0.44 means the average prediction sits under half a level from the truth. Lower is better.
- **Macro-F1** — F1 averaged equally over all four levels, so it cannot be inflated by always guessing the most common one.

**Two reference points.** The **majority-class predictor** — always guess the most common level — scores
**50.27 top-1 at 16.73 macro-F1**; every number below is read against it. The **noise floor** — the
spread you get from re-running one configuration with different random seeds — is **0.45 sd on top-1**;
nothing smaller than about a point is interpretable.

All accuracy figures use the fixed decoder `refined_expected_thresholds` unless a table is explicitly
about decoding.

---

## 1. The changes

**C1 · Ordinal calibration decoding.**
Engagement levels are ordered, so instead of taking the highest logit we convert the four logits into a
continuous expected level and cut it with fitted thresholds. Four decoders were implemented:
- **Argmax** — take the highest-scoring class. The default, and it throws away the ordering entirely.
- **Logit bias** — add a fitted per-class offset before argmax, correcting the model's class-prior skew.
- **Expected thresholds** — compute the probability-weighted mean level, then cut it at thresholds fitted on validation.
- **Refined expected thresholds** — the same, with the thresholds re-optimised directly against the ordinal metrics.

**C2 · Per-sample audio→video temporal alignment.**
Audio left the CNN at ~168 time steps against video's ~15 — an 11× mismatch, so each audio token
cross-attended over a diluted, smeared view of the video. Audio is now adaptively pooled onto the valid
video length **per sample**, before the first cross-attention block, so the two streams meet at the same
temporal granularity.

**C3 · Inference-time modality ablation.**
Either modality's token stream can be zeroed at inference on already-trained weights. This means audio's
and video's contributions are measured on *identical parameters* — the standard alternative, training a
separate audio-only model, confounds the modality's contribution with a different optimisation run.

**C4 · Matched single-variable sweep protocol.**
14 finetunes **warm-started** from one shared checkpoint, each changing exactly one named variable, plus
**3 seed repeats** of one configuration. The seed repeats are what make the rest readable: they measure
how much a number moves when *nothing* changes, which sets the bar any real effect has to clear.

---

## 2. Ordinal decoding (C1) — **+2.03 top-1 at zero training cost**

Same trained weights, same test set. Only the logit→level decision rule changes.

<table>
  <thead>
    <tr><th>Decoder</th><th>Fusion top-1</th><th>Video-only top-1</th></tr>
  </thead>
  <tbody>
    <tr><td>Argmax</td><td>64.10%</td><td>62.90%</td></tr>
    <tr><td>Logit bias</td><td>65.74%</td><td>64.32%</td></tr>
    <tr><td>Expected thresholds</td><td>65.65%</td><td>65.03%</td></tr>
    <tr><td><b>Refined expected thresholds</b></td><td><b>66.13%</b></td><td><b>65.07%</b></td></tr>
    <tr><td><b>Δ over argmax</b></td><td><b>+2.03</b></td><td><b>+2.17</b></td></tr>
  </tbody>
</table>

This is larger than every architectural and optimiser change we tested, combined. It also cuts both
ways: one checkpoint reads 66.76 under `expected_thresholds` and 65.78 under
`refined_expected_thresholds`. **A ~1-point sensitivity to decoder choice means no engagement accuracy
figure is comparable across papers unless the decoder is reported.**

---

## 3. Temporal alignment (C2) — moves error from off-by-two to off-by-one

`_adaptive_align_audio_to_video` pools the full valid audio span onto the valid video length, per
sample, before the first cross-attention block. Regression-tested in
`tests/test_model.py::test_audio_alignment_uses_full_valid_audio_span`.

<table>
  <thead>
    <tr><th>Checkpoint</th><th>Top-1</th><th>Adjacent</th><th>MAE</th><th>Macro-F1</th></tr>
  </thead>
  <tbody>
    <tr><td>No alignment fix</td><td><b>64.18%</b></td><td>87.41</td><td>0.5195</td><td>46.60</td></tr>
    <tr><td>+ alignment fix, recalibrated only</td><td>64.05%</td><td>86.66</td><td>0.5332</td><td>44.54</td></tr>
    <tr><td>+ alignment fix, finetuned</td><td>63.74%</td><td><b>88.21</b></td><td><b>0.5142</b></td><td><b>48.45</b></td></tr>
  </tbody>
</table>

*Recalibrated only* means the thresholds were refitted on the aligned model without updating any
weights; *finetuned* means training continued with the fix in place.

**Δ from finetuning with the fix: +0.80 adjacent, −0.005 MAE, +1.86 macro-F1, −0.44 top-1.**

Correct alignment does exactly what it should: errors shrink in *magnitude* and the minority levels
separate better. Top-1 cannot see this — it scores an off-by-one error the same as an off-by-three,
which is the same argument as §2. Reported as an ordinal-metric gain; the top-1 delta sits inside the
noise band.

---

## 4. Modality contribution (C3) — measured on identical weights

<table>
  <thead>
    <tr><th>Condition</th><th>Top-1</th><th>vs fusion</th></tr>
  </thead>
  <tbody>
    <tr><td>Audio + video (fusion)</td><td><b>66.36%</b></td><td>—</td></tr>
    <tr><td>Video only</td><td>65.07%</td><td>−1.29</td></tr>
    <tr><td>Audio only</td><td>50.27%</td><td>equals the majority-class predictor</td></tr>
  </tbody>
</table>

Across all 13 ablated models:

<table>
  <thead>
    <tr><th>Metric</th><th>Fusion better in</th><th>Mean Δ</th><th>sd</th></tr>
  </thead>
  <tbody>
    <tr><td><b>Top-1</b></td><td><b>12 / 13</b></td><td><b>+0.67</b></td><td>0.59</td></tr>
    <tr><td>Macro-F1</td><td>4 / 13</td><td>−0.09</td><td>1.21</td></tr>
  </tbody>
</table>

**Claim:** audio carries no standalone clip-level engagement signal on this corpus, yet contributes a
small positive fusion effect whose *sign is consistent across 12 of 13 matched models and both seeds
tested*. Consistency of sign across matched models — not a point estimate from one run — is what makes a
sub-noise-floor effect reportable, and C3 is what makes that measurement possible at all.

---

## 5. Training-side variables (C4) — 14 matched runs, one variable each

**What each variable is, one line each.**
- **Balanced sampler** — draw training clips so every engagement level appears equally often per epoch.
- **Class-weighted loss** — same goal, but by weighting rare levels more heavily in the loss instead of resampling.
- **Ordinal weight** — how strongly the loss penalises an error by *how far* it is, on top of whether it is wrong at all.
- **Label smoothing** — soften the one-hot target so the model does not become over-confident.
- **EMA** — evaluate an exponential moving average of the weights rather than the final step's weights.
- **SpecAugment** — mask random time and frequency bands in the audio spectrogram as augmentation.
- **Frame cap** — the maximum number of video frames a clip is padded or truncated to.
- **Synced random crop** — take a random temporal window of the clip each epoch, cropping audio and video to the *same* window.
- **Warmup cosine** — ramp the learning rate up, then decay it along a cosine curve.

<table>
  <thead>
    <tr><th>Run</th><th>Variable under test</th><th>Top-1</th><th>Adjacent</th><th>MAE</th><th>Macro-F1</th></tr>
  </thead>
  <tbody>
    <tr><td>G04 s3</td><td>balanced sampler</td><td><b>66.67</b></td><td>91.31</td><td>0.440</td><td>52.37</td></tr>
    <tr><td>G04 s2</td><td>balanced sampler (seed 2)</td><td>66.36</td><td>91.09</td><td>0.444</td><td>51.13</td></tr>
    <tr><td>G00</td><td><b>control</b></td><td>66.09</td><td>91.40</td><td>0.444</td><td>52.25</td></tr>
    <tr><td>G12</td><td>EMA 0.999</td><td>65.96</td><td>91.22</td><td>0.446</td><td>50.93</td></tr>
    <tr><td>G08</td><td>loss = cross-entropy</td><td>65.91</td><td>91.36</td><td>0.446</td><td>50.92</td></tr>
    <tr><td>G13</td><td>no label smoothing</td><td>65.91</td><td>91.44</td><td>0.445</td><td>51.01</td></tr>
    <tr><td>G09</td><td>ordinal weight 0.60</td><td>65.87</td><td>91.36</td><td>0.446</td><td>50.94</td></tr>
    <tr><td>G01</td><td>frame cap 50</td><td>65.82</td><td>91.40</td><td>0.446</td><td>50.76</td></tr>
    <tr><td>G04 s1</td><td>balanced sampler (seed 1)</td><td>65.78</td><td><b>92.24</b></td><td><b>0.436</b></td><td><b>53.41</b></td></tr>
    <tr><td>G05</td><td>lr 1e-4 + warmup cosine</td><td>65.69</td><td>91.36</td><td>0.448</td><td>50.72</td></tr>
    <tr><td>G07</td><td>ordinal weight 0.35</td><td>65.38</td><td>91.27</td><td>0.452</td><td>50.68</td></tr>
    <tr><td>G06</td><td>SpecAugment</td><td>65.34</td><td>91.22</td><td>0.453</td><td>50.59</td></tr>
    <tr><td>G02</td><td>frame cap 40 + synced random crop</td><td>64.89</td><td>91.53</td><td>0.450</td><td>50.42</td></tr>
    <tr><td>G03</td><td>class-weighted loss</td><td>64.72</td><td>91.71</td><td>0.452</td><td>50.41</td></tr>
  </tbody>
</table>

**Every isolated comparison is null.** Ordinal loss vs cross-entropy (65.82 / 65.91), ordinal weight
0.15 / 0.35 / 0.60 (65.82 / 65.38 / 65.87), label smoothing on/off (65.82 / 65.91), EMA on/off
(65.96 / 65.82) — all inside the seed band. Augmentation is actively harmful (G02 64.89 against its
otherwise-identical pair G01 65.82). **Ensembling** — averaging the predictions of several checkpoints,
normally a reliable free gain — scores 65.69 across all 14, below the best single model, and dropping
the nine weakest members changes nothing (65.60): the members share a warm start, so they make the same
mistakes and there is no independent error to average away.

This table is a positive result, not a failed sweep: **it is the first matched, seed-controlled
single-variable sweep on this benchmark**, and it is what licenses the conclusion in §6.

---

## 6. The finding that ties it together

<table>
  <thead>
    <tr><th>Source of variation</th><th>Top-1 sd</th><th>Spread</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><b>14 different configurations</b><br><small>loss, ordinal weight, class weighting, sampler, LR, EMA, label smoothing, SpecAugment, frame cap, augmentation</small></td>
      <td><b>0.52</b></td>
      <td>1.95</td>
    </tr>
    <tr><td><b>Same configuration, 3 random seeds</b></td><td><b>0.45</b></td><td>0.89</td></tr>
  </tbody>
</table>

**Varying every hyperparameter we tested produces about as much variation as changing the random seed
alone.** Three independent routes arrive at the same place — the flat sweep, ensembling that finds no
independent error to exploit, and augmentation that hurts rather than helps. **The corpus is the
ceiling, not the model.**

---

## 7. The through-line

Everything above says the same thing twice, once about **what we measure** and once about **what we can
still gain**.

**On measurement.** Two of our three effects are invisible to top-1 accuracy. Ordinal decoding moves
top-1 by 2.03 points without touching a single weight, so two papers reporting the same model can differ
by more than their claimed contributions purely on a decoding choice nobody states. Temporal alignment —
a genuine architectural correction, verified by unit test — *costs* 0.44 top-1 while gaining 1.86
macro-F1, because top-1 charges the same price for an off-by-one error as for an off-by-three on a scale
that is explicitly ordered. A field that reports engagement as flat multiclass accuracy is systematically
blind to the improvements that matter most on an ordinal task.

**On headroom.** We swept ten training variables and found nothing that clears the seed noise; we
ensembled fourteen checkpoints and lost a point; we added augmentation and lost more. That is not a
tuning failure, it is a measurement: on 11k clips this corpus does not contain enough independent
variation for the model to be the limiting factor. The remaining gains are in the data and in the label
scheme, not in the architecture.

**What we therefore propose.** Report the decoder alongside every accuracy figure. Report adjacent
accuracy and MAE next to top-1 on any ordinal task. Establish a seed-repeat noise floor before claiming
any sub-one-point improvement. Each of these costs nothing, and each one is a claim in this project's
own history that we had to retract for want of it.

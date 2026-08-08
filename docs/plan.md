# Dataset Collection Plan — Classroom Engagement Detection

**Current phase:** Design and collect a multimodal engagement dataset with equal audio-video importance.
**Status:** Pre-collection — session design locked, no recordings yet.

---

## What We Are Building

A dataset of online students in Zoom sessions, labeled for engagement level (1–5) and confusion (yes/no). Audio and video must each carry independent discriminative signal — neither is supplementary.

**Why not use an existing dataset:**
- DAiSEE (IIT Hyderabad): crowdsourced labels, poor quality on minority classes, video-only dominant
- CMOSE (CVPR 2024): 76% of clips have no speech → audio only adds 3.18% → not what we need
- EngageNet: not publicly available in full form; no per-participant audio tracks

**What we need that none of these provide:** per-student separate audio tracks + discussion-heavy sessions where speech varies meaningfully by engagement level.

---

## Key Resources That Informed These Decisions

| Paper / Resource | What it established | How it shaped our design |
|---|---|---|
| CMOSE — Wu et al., CVPR Workshop 2024 | 4-class engagement, ICC=0.84, showed audio adds only 3.18% when students are muted | Showed exactly what NOT to replicate; drove the discussion-only session constraint |
| CORE-Net / COLER — Tran et al., WACV 2026 | Ordinal-aware multimodal engagement for collaborative learning; context modelling + individual level | Directly validates our session format; ordinal supervision confirms our loss design |
| Gaze in Conversation — Maran et al., Applied Psychology 2021 | Listeners look at screen MORE than speakers; gaze behaviour is inverted by role | Speaker/listener role must be a conditioning variable — same gaze means opposite things |
| Gaze + Turn-Taking — arXiv 2025 | Gaze predicts turn-taking with AUC 0.71–0.78; back-channel vocalizations signal active listening | Back-channels ("mm-hmm", head nods) are the primary listener engagement audio signal |
| Group Size — Frontiers Psychology 2025, Medical Education 2022 | 4-member groups show highest engagement; groups of 5+ introduce social loafing | Breakout rooms fixed at 4 students, not 5 |
| Neural Computing & Applications, Springer 2025 | OpenFace AU features (82.9%) beat EfficientNet end-to-end (47.2%) on DAiSEE | Use OpenFace structured features, not raw CNN, until dataset exceeds 5,000 clips |
| Sümer et al., IEEE Trans. Affective Computing 2021 | Student-independent evaluation gives AUC 0.62–0.72 — the honest ceiling | Two separate cohorts required; Session 1 per cohort is Hawthorne-biased, exclude from training |
| MocoRank, CMOSE paper | Contrastive momentum ranking loss handles ordinal classes + class imbalance simultaneously | Loss function choice for training (instead of MSE or plain cross-entropy) |

---

## Session Design

### Subject Is Not Fixed

The subject does not determine engagement variation — the format does. Any course, any topic. What matters is that every activity is interactive: no extended instructor monologue, no individual silent reading, no passive watching. Engagement variation comes entirely from:

1. **Task difficulty** — a task that exceeds student capability produces confusion and disengagement without the instructor saying a word
2. **Who is in the room** — breakout room composition changes social dynamics and therefore engagement
3. **Stakes** — knowing you will be called on keeps waiting students at Level 2–3 rather than fully checked out
4. **Repetition fatigue** — a third round of the same format type will naturally produce lower engagement than the first, even with a different task

### Why No Monologues

Monologues produce disengagement but the audio track becomes uniformly silent — every student is silent during a monologue, regardless of whether they are at Level 1 or Level 3. A silent audio clip from a bored student is indistinguishable from a silent clip from an attentive one. Monologues destroy the audio signal. Every minute of the session must have some students speaking so that silence itself becomes informative (i.e., silence during an interactive task signals disengagement, not just compliance with the format).

### Session Format — 50 Minutes, Fully Interactive

Time budget is tight. No warm-up buffer, no lecture segments, no transitions longer than 90 seconds.

```
[00–03]  Instructor poses ONE question or problem. Max 2 minutes of speaking.
          Assigns breakout rooms. Students have NOT seen the question before.
          → Immediate mild confusion / orientation = good baseline reading

[03–18]  BREAKOUT ROUND 1 (groups of 3, Room A / B / C / D)
          Task: accessible version of the problem. Has a concrete answer.
          Room composition: random.
          → Expected: Level 3–5. Varies by student and group chemistry.
          → Audio: active discussion, overlapping speech, back-channelling

[18–22]  COLD-CALL RETURN
          Instructor calls ONE person from each group — not the group, a person.
          They answer for 60 seconds. Others listen.
          → Speaker: Level 5 (high stakes, no warning)
          → Waiting students: Level 2–3 (anticipation keeps them from fully dropping)
          → This is the sharpest engagement contrast in the session

[22–37]  BREAKOUT ROUND 2 (same rooms, harder task)
          Task: harder variant. Deliberately chosen to exceed what most groups
          can confidently solve. Produces productive struggle.
          → Expected: Level 2–4. More confusion, more silence within rooms,
             some students go quiet while one drives the conversation.
          → Audio: uneven — one or two voices per room, silence from others
          → This is the primary source of within-group engagement variation

[37–40]  COLD-CALL RETURN (same format)
          Different person called this time.
          → Same engagement contrast dynamic as [18–22]

[40–50]  BREAKOUT ROUND 3 (re-shuffled rooms)
          New room composition. Same difficulty as Round 2 or a debate variant
          (two students assigned opposite positions on a question, must argue).
          → Debate format reliably re-engages students who drifted in Round 2
             because it is personally directed — you must speak, not just listen.
          → Expected: Level 3–5 for debate participants; re-engagement spike visible
             in audio (F0 rises, speech rate increases from Round 2 baseline)

[50–53]  FINAL COLD-CALL
          Instructor asks each student individually: "One thing you're still
          unsure about." Forces every student to produce at least one utterance.
          → This single block guarantees at least one speech sample per student
             per session, which is required for per-student audio calibration.

[53–56]  SELF-REPORT SURVEY (typed into Zoom chat — 3 questions, 30 seconds each)
          Excluded from training clips.
```

### How Engagement Variation Is Generated Without Monologues

The Level 1–2 data now comes from three sources, all interaction-based:

| Source | Why it produces Level 1–2 | Audio signal preserved? |
|---|---|---|
| Waiting student during cold-call | Anticipation fades after 60s; student drifts | Yes — can hear silence vs. subtle sounds vs. whispering |
| Quiet student in Round 2 breakout | Hard task, one person dominates, others go passive | Yes — silence during active-discussion block is informative |
| Round 3 fatigue (third breakout, less novel) | Repetition of format, cumulative cognitive load | Yes — speech rate drops, energy drops, F0 range compresses |

None of these require a monologue. All of them preserve the audio signal.

### Breakout Room Composition Strategy

Room composition is a controlled variable, not random after Round 1.

**Round 1 — Random:**
Establishes each student's neutral participation rate and baseline behavior. No prior grouping bias.

**Round 2 — Deliberate difficulty mismatch:**
Put one student with strong prior knowledge alongside two students with weaker background. The strong student tends to dominate; the weaker students go quiet. This is the exact within-room engagement variation needed: one Level 4–5 (engaged, driving) alongside Level 2–3 (following or lost).

Do NOT put all strong students together — they will all be Level 5 and produce no Level 1–2 data. Do NOT put all weak students together — nobody drives the conversation and the whole room goes Level 1.

**Round 3 — Re-shuffle for debate:**
Assign two students per room who disagreed during the cold-calls. This near-guarantees they will engage in Round 3 even if they drifted in Round 2.

### Room Size

**4 students per room.** Literature finding (Frontiers 2025, Medical Education 2022): 4-member groups consistently produce the highest engagement levels and outperform triads on collaborative tasks. Groups of 5+ introduce social loafing — one student routinely goes quiet and their track becomes uninformative for training.

With 20 students: 5 rooms of 4. With 16 students: 4 rooms of 4.

---

## Speaker vs Listener — Critical Distinction

Speakers and listeners have inverted behavioral signals. The model must know which role a student is in before interpreting any feature. This is not optional metadata.

**Why gaze is opposite by role** (Maran et al. 2021, Applied Psychology):
- Speakers naturally avert gaze to hold the floor, organize thoughts, and signal they haven't finished. Eyes-away during speech is normal.
- Listeners maintain screen-directed gaze to signal attention and readiness to take a turn. Eyes-away during listening is disengagement.

**Feature interpretation table:**

| Signal | Speaking student | Listening student |
|---|---|---|
| Eyes on screen | Expected / neutral | Strong engagement signal |
| Eyes off screen | Normal (thinking) | Disengagement signal |
| Head nods | Turn-yielding | Back-channelling = engaged |
| Silence on audio | Off-task risk | Expected — neutral to positive |
| Short vocalization ("mm") | Filler pause | Back-channel = active listening |
| Low F0 range | Monotone / bored | Not applicable |

**How listener engagement is measured:**

1. **Gaze toward screen** — OpenFace gaze angle pointed toward the camera. Engaged listener maintains this. Disengaged listener's gaze drops or points away.
2. **Head orientation** — yaw/pitch aligned toward camera means attending to the speaker. Head turning away = disengaged.
3. **Back-channel vocalizations** — "mm-hmm", "yeah", "right" — voiced events under 500ms during another student's speaking turn. WebRTC VAD catches them; Whisper ASR identifies them. Most reliable audio signal for listener engagement.
4. **Micro head nods** — rhythmic Ry oscillation (~1 nod per 2–3s). Detectable via autocorrelation on OpenFace Ry. Present = active following; absent = passive or disengaged.
5. **Absence of off-task audio** — no background noise, shuffling, or side conversation during another student's speech.

**Implementation requirement:** Every clip in the HDF5 must carry a `speaking` flag (derived from per-student VAD). The model conditions on this flag. A listener clip and a speaker clip with identical visual features are NOT the same training example.

---

## Recording Setup

**Zoom settings (host must configure before every session):**
```
Settings → Recording → Record each participant separately: ON
Settings → Recording → Gallery view: ON
Settings → Audio → Record audio for each participant: ON
Settings → Video → HD video: ON (720p minimum)
```

**Output per session:**
```
session_XX/
├── gallery_view.mp4          ← face tiles for all students
├── audio_only/
│   ├── student_01.m4a        ← one file per student (mandatory)
│   └── ...
└── zoom_transcript.vtt       ← auto-transcript for ASR bootstrap
```

**Student requirements before each session:**
- Camera ON, eye level, face illuminated (not backlit)
- Headphones to prevent audio feedback
- Stable connection

**Pre-recording checklist:**
1. Confirm "Record each participant separately" is active
2. Brief students: "We're studying online learning experience." Do NOT say engagement is being measured — reduces Hawthorne bias.
3. Run 2-minute unrecorded warm-up before pressing Record.

---

## Dataset File Format

Single HDF5 file + CSV manifest. Symmetric storage — audio and video features occupy equal schema depth.

```
data/
├── engagement_dataset.h5
├── manifest.csv
└── raw/                    (not committed to git)
    └── session_XX/
```

**HDF5 structure:**
```
engagement_dataset.h5
├── video/{clip_id}/
│   ├── au_sequence     float32 (T × 17)   AU intensities, OpenFace
│   ├── head_pose       float32 (T × 6)    tx,ty,tz,Rx,Ry,Rz
│   ├── gaze            float32 (T × 6)    left + right eye gaze
│   └── ear             float32 (T × 2)    eye aspect ratio L+R
├── audio/{clip_id}/
│   ├── mel_spectrogram float32 (128 × T_a)
│   ├── f0_contour      float32 (T_a,)     0 = unvoiced
│   ├── rms_energy      float32 (T_a,)
│   ├── vad_flags       uint8   (T_a,)     1=voiced 0=silence
│   └── prosody_summary float32 (8,)       [mean_F0, F0_std, speech_rate,
│                                           pause_rate, mean_energy,
│                                           energy_std, speaking_frac, ZCR_mean]
└── labels/{clip_id}/
    ├── engagement_level uint8  scalar     1–5
    ├── confusion_flag   uint8  scalar     0 or 1
    ├── annotator_a      uint8  scalar
    └── annotator_b      uint8  scalar
```

**manifest.csv columns:**
```
clip_id, session_id, student_id, block, start_sec, end_sec,
engagement_level, confusion_flag, split,
is_speaking,           ← derived from per-student VAD; conditions model inference
has_back_channel,      ← 1 if back-channel vocalization detected during clip
openface_confidence_mean, audio_rms_mean, annotator_agreement
```

`has_speech` and `audio_rms_mean` enable speech-only subset filtering for audio ablations.

---

## How Engagement States Are Scored

All three states are temporal composites — a single frame is never enough.

### Boredom (developing over 2–5 minutes)
Cascade: blink rate rises → head stills → EAR drops → head pitches down → gaze drifts → phone look-down.

Key features:
- `AU45` blink rate excess above student's Block 1 baseline
- EAR rolling mean declining below (baseline − 0.04)
- Head pitch `Ry` drifting above +8° sustained
- Head landmark velocity dropping below 0.5 px/frame (frozen)
- Gaze off-screen events in interactive blocks

### Confusion (cognitively active but blocked — opposite of boredom's stillness)
Key features:
- `AU4` sustained > 1.5 intensity for > 2s (brow furrow)
- `AU23` lip tighten co-occurring with AU4
- Lateral head tilt `|Rz| > 10°` (universal confusion gesture)
- Filled pause rate ("um/uh/wait/no wait") > 2× student baseline
- Rising F0 at declarative utterance ends (uptalk on statements)

Confusion ≠ disengagement. A Level 5 student productively struggling shows high confusion + high engagement. The confusion flag is a separate output head precisely for this reason.

### Enhancement / Flow
Key features:
- Forward lean: `Ry < −5°` sustained
- `AU5` eye widening (upper lid raiser)
- Duchenne smile: `AU6 + AU12` co-active
- Head nodding: rhythmic Ry oscillation, > 3 nods/30s
- Speaking turn frequency above student baseline
- F0 range per utterance > 100 Hz (expressive, wide pitch)

---

## Annotation Protocol

Two annotators per clip, audio ON mandatory.

**Annotation form fields:**
- Engagement level: 1–5
- Confusion flag: yes/no
- Must check at least one VIDEO evidence box AND one AUDIO evidence box
- Confidence: 1–4

**Adjudication:** disagree by 1 level → mean. Disagree by 2+ → third annotator. Three-way split → majority vote.

**IRR target:** Cohen's κ ≥ 0.70. Pre-annotation calibration on 30 extreme clips. Replace annotator if κ < 0.65 after two calibration rounds.

**Self-report (secondary):** After each block, students type 3 Likert ratings (1–7) in Zoom chat:
- "How focused were you?" / "Did time pass quickly?" / "Were you bored?" (reverse-coded)
- Used to flag disagreements between behavioral labels and student experience, not as primary label.
- Final label weight: 0.7 × behavioral + 0.3 × self-report.

---

## Collection Calendar

```
Week 1   Annotator calibration (30 clips, target κ ≥ 0.65 before proceeding)
Week 2   Session 1, Cohort A — ML topic         [pipeline debug only, Hawthorne-biased]
Week 3   Session 2, Cohort A — Stats topic      [first usable data]
Week 4   Session 3, Cohort A — ML (new content) [full natural behavior]
         Begin annotating Sessions 2–3 in parallel
Week 5   Session 4, Cohort B (new students) — ML topic  [student-independent data starts]
Week 6   Session 5, Cohort B — Stats topic
Week 7   Annotate Sessions 4–5; compute κ; if < 0.70 hold calibration before continuing
Week 8   Session 6, Cohort B — ML (new content)
Week 9   Final annotation pass; build HDF5; train first model
```

Expected output: ~10,000–12,000 labeled clips, 2 cohorts, 8 sessions, 24 unique students.

---

## Open Tasks

| # | Task | Priority |
|---|---|---|
| E1 | Zoom session script (instructor-facing doc for each of the 5 blocks) | High |
| E2 | Annotation guide document with behavioral anchors and example clips | High |
| E3 | `preprocessing/zoom/extract_tiles.py` — crop per-student face tiles from gallery view | Critical |
| E4 | `preprocessing/zoom/extract_prosody.py` — F0, RMS, VAD, speech rate per clip | Critical |
| E5 | `preprocessing/zoom/build_hdf5.py` — assemble OpenFace CSVs + audio arrays into HDF5 | Critical |
| E6 | `datasets/engagement.py` — dataset loader returning (au_seq, mel, prosody, labels) | Critical |
| E7 | OpenFace 2.2 installation and CLI test on one sample clip | Critical |
| E8 | Implement `OpenFaceEncoder` in `models/multimodal_cnn.py` | High |
| E9 | Implement `ProsodyEncoder` as FiLM conditioning (not a sequence token — see architecture.md) | High |
| E10 | Implement dual output heads: engagement (CORN loss, 5-class ordinal) + confusion (BCE, binary) | High |
| E11 | Implement audio `AvgPool1D` temporal subsampling before cross-attention | Done — refined 2026-07-31: per-sample `_adaptive_align_audio_to_video` pools full valid audio span into valid video length (bug: previously treated video_lengths as audio lengths) |
| E12 | Implement modality dropout (p=0.15) in training forward pass | Done |
| E13 | Per-modality validation logging (audio-only, video-only, fusion) | Done (eval-time) — `--ablate_modality {none,audio_only,video_only}` in `scripts/calibrate_engagement_logits.py`; see Section 12 for results |
| E14 | Pilot session (Session 1, Cohort A) | High |
| — | Validate alignment + synced random crop on EngageNet/DAISEE (fresh `results/` run; compare top-1, adjacent acc, mean abs class error) | Partial — V13 short 3-ep done (`results/v13_alignfix_avonly_short/`); full ≥15–30 ep AV-only retrain still needed for a new top-1 SOTA |
| — | Implement role conditioning: `role_embedding(is_speaking)` added to video tokens before first AttentionBlock | Critical |
| — | Replace MaxPool aggregation with learned attention pooling | High |
| — | Add attention output dropout (p=0.1–0.2) before each residual add | High |

**ProsodyEncoder implementation note (E9):** Do NOT produce a single summary token for concatenation into the temporal sequence. A scalar token attends identically at every time step. Instead, condition the audio CNN features directly:
```python
gamma, beta = Linear(128, 128)(prosody_summary_8dim → 128).chunk(2, dim=-1)
audio_features = gamma * audio_features + beta
```

**Role conditioning implementation note:** Every forward pass that processes video tokens must receive the `is_speaking` tensor (B,) from the manifest. Before the first AttentionBlock:
```python
role_embed = self.role_embedding(is_speaking.long())  # B × 128
video_features = video_features + role_embed.unsqueeze(1)  # broadcast over T
```
Without this, listener gaze (screen-directed = engaged) and speaker gaze (screen-averted = normal) produce opposite gradients for the same label.

---

## Section 12 — EngageNet audio truncation bug and modality ablation (2026-08-07)

### 12.1 The audio truncation bug

EngageNet source clips are **10.00 s / 300 frames @ 30 fps** (verified directly with OpenCV on
`datasets/EngageNet/Train/*.mp4`). Video features (`*_facecroppad.npy`, 50 frames) span the whole clip.

`preprocessing/engagenet/extract_audios.py` was run with the legacy RAVDESS contract
`--max_video_seconds 3.6`, so every `*_croppad.wav` contains **only the first 3.6 s** — 36% of the clip.
Every EngageNet result produced before this date cross-attended audio from 0–3.6 s against video
from 0–10 s: the audio was both **truncated and temporally misaligned** with the video it attends to.

The per-sample `_adaptive_align_audio_to_video` fix (E11) does not address this. It correctly pools
the *available* audio span onto the valid video length, but if the available span only covers the
first 36% of the clip, pooling stretches that 3.6 s window across all 10 s of video.

**Fix:** `preprocessing/engagenet/extract_audios_full.py` re-extracts all clips uncapped to
`*_croppad10s.wav`, leaving the 3.6 s files in place so audio span is a clean ablation axis.

| | 3.6 s (`_croppad.wav`) | 10 s (`_croppad10s.wav`) |
|---|---|---|
| Annotation file | `annotations_engagement.txt` | `annotations_engagement_a10.txt` |
| Mel frames into the model | 156 | 432 |
| Clips extracted | 11,311 | 11,307 (10,869 with audio + 438 genuinely silent) |

### 12.2 Speech content of EngageNet (measured, n=400 random clips)

Frame-level 25 ms energy, silence threshold −50 dBFS:

| Statistic | 3.6 s audio | 10 s audio |
|---|---:|---:|
| Clips >90% silent frames | 27% | 28% |
| Clips <20% silent frames ("talkative") | 44% | 43% |
| Clips with no audio stream at all | — | 3.9% (438/11,307) |

**~57% of EngageNet clips contain meaningful speech.** This is the key contrast with CMOSE, where only
2,930/12,193 clips (24%) contain speech — the documented reason CMOSE's audio path added just +3.18%.
EngageNet can support an audio claim in a way CMOSE structurally could not.

### 12.3 Modality ablation on the current best checkpoint (3.6 s audio)

Eval-time ablation zeroes one stream at the same point in `forward_feature_3` where modality dropout
already operates, so the ablated stream matches a condition training has already seen.
Checkpoint: `results/v12_05_finetune_uniform_best_ord_nosampler_lr0001_e30/model.pth`.

| Condition | argmax top-1 | refined-expected top-1 | Adjacent | Macro F1 | Artifact |
|---|---:|---:|---:|---:|---|
| AV fusion | 61.97% | **64.05%** | 86.66% | 44.54 | `results/exp2026/A0_v12_baseline_recalib/` |
| Video-only | 60.64% | **64.18%** | 86.17% | 46.21 | `results/exp2026/E03_v12_video_only/` |
| Audio-only | 26.51% | 46.41% | 70.04% | 18.18 | `results/exp2026/E02_v12_audio_only/` |

**Verdict: the audio path currently contributes nothing.** Video-only equals or beats full fusion, and
audio-only lands *below* the 50.27% majority-class baseline (1134/2256 test clips are class 3).
The project's stated gate — "audio-only must reach ≥65% of fusion accuracy" — is met only on the ratio
(46.41/64.05 = 72%), but the absolute number is below chance-by-majority, so the gate as written is not
a meaningful test. **Restate the gate as: audio-only must beat the 50.27% majority-class baseline, and
AV fusion must beat video-only.** Neither currently holds.

This ablation was run on a model trained on truncated audio, so it does not yet distinguish
"audio is uninformative for engagement" from "we broke the audio". Section 12.4 is that test.

### 12.4 Experiments in flight

| ID | Question | Config | Status |
|---|---|---|---|
| E04 | Does full-length audio help when finetuning the existing best? | From V12 best, 10 s audio, ordinal 0.15, lr 5e-5, 8 ep, bs 8 | **Done — val 65.27% but test 62.68% (−1.37 vs baseline)**, see 12.8 |
| E05 | Does full-length audio help when trained on from the start? | Fresh from EfficientFace pretrain, 10 s audio, CE + LS 0.1, lr 0.01, 20 ep, bs 8 | Running — 63.77% @ ep 5, climbing |
| E06 | Does fusion beat video-only *after* the audio fix? | Modality ablation on E04 best checkpoint | Chained, pending E04 |
| E19 | Does EngageNet audio carry engagement signal independent of our encoder? | Hand-crafted acoustics + logreg/GBM, CPU | **Done — deprioritises E15** (see 12.7) |
| E09 | Does synced random temporal crop fix E04's overfitting? | From V12, 10 s audio, ordinal 0.15, lr 5e-5, `--train_frame_sampling random`, 6 ep | Queued behind E06 |
| E11 | Does sqrt-inverse class weighting close the macro-F1 gap? | From V12, 10 s audio, ordinal 0.15, lr 5e-5, `--class_weighting sqrt_inverse`, 6 ep | Queued behind E09 |

**Why E09 and E11 were chosen, and what was deliberately deferred.** Both target a weakness visible in
the existing numbers rather than a guess, so both are informative whichever way E06 goes:

- *E09* — E04 peaks at epoch 3 then decays to 63.96% by epoch 5. With only 7,879 training clips that is
  an augmentation problem, not a learning-rate one; the synced crop already exists in `src/data/temporal.py`.
- *E11* — every run sits at 44–48 macro-F1 against ~64% top-1, the signature of leaning on class 3
  (47% of train, 50% of test). Adjacent accuracy of ~91% shows the ordinal structure *is* being learned,
  so the deficit is minority-class separation, not ordering.

**Deferred until E06 + E19 report:** SpecAugment (an *audio* augmentation — worthless if the audio path
is still inert) and the E15 encoder replacement. Do not spend GPU on audio-specific tuning before the
ablation shows the audio branch carries signal.

**E04 validation beats the 3.6 s baseline on every metric** (65.27% vs 64.61% top-1, 91.04% vs 89.73%
adjacent, 0.4585 vs 0.4809 MAE), with epoch 1 alone already ahead. This is consistent with the
truncation being a real defect. It is **not** yet evidence that audio contributes: a better-regularised
video path could produce the same lift, and these are validation numbers. E06 is the test that
separates the two — until it lands, do not describe E04 as an audio-visual gain.

Both are ablated by modality on completion. **Decision rule:** if audio-only still fails to beat 50.27%
and fusion still fails to beat video-only after E04/E05, the audio encoder — not the audio data — is the
problem, and the next step is replacing the mel-CNN rather than tuning it further.

### 12.5 Experiment compilation

`scripts/compile_experiments.py` aggregates every `calibration_results.json` and
`evaluation_testing.json` under `results/` into `results/exp2026/all_experiments.csv`
(57 rows at time of writing) plus a ranked markdown table. Reruns are idempotent.

### 12.6 Open items added by this section

| # | Task | Priority |
|---|---|---|
| E15 | Decide audio encoder replacement (frozen WavLM/HuBERT vs prosody features) if E04/E05 confirm null audio | Critical — gates the AV claim |
| E16 | Rename the method — `AVT-CA` collides with arXiv:2407.18552 (see memory.md) | High — blocks submission |
| E17 | Re-extract DAiSEE audio; check it for the same 3.6 s truncation | High |
| E18 | Add a per-epoch per-modality validation log so modality collapse is visible during training, not after | Medium |
| E19 | Encoder-free audio probe (`scripts/audio_signal_probe.py`) — separates "audio carries no signal" from "our encoder fails to extract it" | Critical — decides whether E15 is worth doing |
| E20 | Rerun E19 with `--with_f0` once GPUs are idle; the F0-free run is a lower bound, and F0 range is central to the project's own engagement scoring | High |

### 12.7 E19 — encoder-free audio probe: the ceiling is in the data, not the encoder

`scripts/audio_signal_probe.py`, 94 hand-crafted features (40-bin log-mel mean/std, RMS, ZCR, silence
fraction and run-switch rate, spectral centroid/rolloff/flatness), no neural encoder. Train 7,879 /
test 2,256, 10 s audio.

| Method | Function class | Top-1 | Macro F1 | Adjacent |
|---|---|---:|---:|---:|
| Majority-class predictor | constant | **50.27** | 16.73 | — |
| Our mel-CNN audio branch | deep, cross-attention | 46.41 | 18.18 | 70.04 |
| Logistic regression | linear | 46.28 | 29.92 | 68.26 |
| Histogram gradient boosting | tree ensemble | 46.05 | **31.74** | 71.19 |

**Three independent function classes on different representations land within 0.4 points of each other,
all below a constant predictor.** If the mel-CNN were the bottleneck, the probe should have beaten it.
It did not. The ceiling is in the audio data at clip level, not in how we encode it.

**Consequence: E15 (frozen WavLM/HuBERT encoder swap) is deprioritised.** A stronger encoder extracting
the same absent signal will not help. Do not spend GPU on it on the strength of the SSL-beats-mel-CNN
literature alone — that literature is about emotion/paralinguistics corpora with dense speech, not
10-second lecture-watching clips that are ~28% near-silent.

**But "audio is useless" is too strong, and top-1 is the wrong metric here.** A majority-class predictor
scores 50.27% top-1 at just **16.73 macro-F1**. The probe reaches **31.74** — nearly double. Audio
carries real but weak signal, concentrated in the minority classes, and top-1 actively punishes using it
because guessing class 3 is worth 50% for free. Report macro-F1 and adjacent accuracy alongside top-1
for every audio-facing claim; a top-1-only table makes a genuinely informative audio branch look worthless.

This also reframes 12.3: our audio branch at 18.18 macro-F1 is **underusing** even the weak signal that
plain gradient boosting finds (31.74). The gap is not "audio has nothing" — it is that cross-attention
against a dominant video stream suppresses the audio branch. That is a modality-collapse problem
(gradient blending / OGM-GE territory), not an encoder-capacity problem.

**Caveat — this run has no F0.** `librosa.yin` was dropped for CPU contention (see E20). Pitch range is
central to the project's own engagement scoring formulas (Sections 11.5–11.7), so this table is a
**lower bound** on hand-crafted audio. E20 is now higher priority than originally rated; do not treat
the audio question as settled until it runs.

### 12.8 E04 — the validation gain did not transfer to test

Full-length (10 s) audio finetune from the V12 best checkpoint. Best epoch by val top-1 = epoch 3.

| Decode | V12 baseline (3.6 s) | E04 (10 s) | Δ top-1 |
|---|---:|---:|---:|
| argmax | 61.9681 | 62.1897 | +0.22 |
| logit bias | 63.6968 | 62.7216 | −0.98 |
| expected thresholds | 63.8741 | 62.8989 | −0.97 |
| **refined expected** | **64.0514** | **62.6773** | **−1.37** |

Macro-F1 on refined-expected moves the other way: **44.5441 → 46.8214 (+2.28)**.

**Honest summary: fixing the audio truncation did not improve test top-1.** On validation E04 beat the
baseline on all three metrics (65.27 / 91.04 / 0.4585 vs 64.61 / 89.73 / 0.4809); on test it is 1.37
points worse on the headline decode. Do not repeat the earlier interim framing that E04 "beats the
baseline" — that was a validation-only statement and it did not hold out of sample.

**Two confounds behind the reversal, both real:**

1. **Val/test gap.** 65.27 val vs 62.68 test is a 2.6-point spread, wider than the baseline's. With
   1,071 val clips and checkpoint selection on val top-1, part of the val gain is selection noise.
   Selecting on `f1_macro` or `mean_absolute_class_error` would likely pick a different epoch — worth
   testing before concluding the audio fix is neutral.
2. **Top-1 vs macro-F1 trade.** −1.37 top-1 with +2.28 macro-F1 is precisely the shape Section 12.7
   predicts if audio *started* contributing: its signal lives in the minority classes, and top-1
   rewards collapsing onto class 3. This is not obviously a regression — under the metric guidance in
   12.7 it may be a small improvement.

**Neither confound is settled by E04 alone.** The modality ablation on this same checkpoint (E06) is
immune to both — identical weights, identical test set, only the modality zeroed — so it is the test
that decides whether audio contributes. Record E06 before drawing any conclusion from 12.8.

### 12.9 E06 — DECISIVE: full-length audio does not rescue the audio path

Modality ablation on the E04 checkpoint (10 s audio, the truncation fix applied). Same weights, same
test set, only the zeroed modality differs — immune to both confounds in Section 12.8.

| Condition | argmax | refined expected | Adjacent | Macro F1 |
|---|---:|---:|---:|---:|
| AV fusion | 62.19 | 62.68 | 87.68 | 46.82 |
| **Video-only** | 61.92 | **62.99** | 87.37 | 46.77 |
| Audio-only | 20.26 | **50.27** | 68.84 | **16.73** |

**Audio-only lands on exactly 50.27% top-1 / 16.73 macro-F1 — bit-for-bit the majority-class predictor**
(class 3 = 1134/2256). The audio branch does not merely underperform; with video zeroed it collapses to
constant prediction. Its argmax score of 20.26% shows the raw logits carry no usable class structure and
only threshold calibration lifts it to the constant-predictor score.

**Video-only still beats AV fusion (62.99 vs 62.68).** This reproduces Section 12.3's result on the
3.6 s audio, so the finding is stable across both audio spans:

| Audio span | AV fusion | Video-only | Audio-only |
|---|---:|---:|---:|
| 3.6 s (truncated) | 64.05 | **64.18** | 46.41 |
| 10 s (full) | 62.68 | **62.99** | 50.27 (= majority) |

**Conclusion. The audio truncation was a real bug and worth fixing, but it was not the cause of the null
audio contribution. Fixing it did not make audio contribute.** Combined with E19 — where a linear model
and a tree ensemble hit the same ~46% ceiling as our neural branch — the evidence now points one way:
**EngageNet audio carries very little clip-level engagement signal, and no amount of encoder or
alignment work on this corpus will produce an audio-visual accuracy gain.**

**This closes the "first AV result on EngageNet" framing.** We cannot claim an audio-visual improvement
on this dataset. Do not write that claim. Remaining honest options:

1. **Report the negative result.** "Audio does not help engagement recognition on EngageNet, and here is
   the controlled evidence" — modality ablation across two audio spans, plus an encoder-free probe
   showing the ceiling is in the data. This is publishable as an ablation/analysis contribution and it
   is genuinely useful, because no prior EngageNet paper tested audio at all.
2. **Pivot the AV claim to the purpose-built corpus.** The project's own dataset (plan.md Sections 2–11)
   is explicitly designed so audio carries signal — discussion-heavy, per-student tracks, ~75% speech.
   EngageNet becomes the motivating negative result that justifies collecting it.
3. **Compete on video-only accuracy.** Our 64.18% is below the 65–68% video-only field, so this needs
   real architecture work and drops the AV angle entirely.

**Do not pursue** SpecAugment, WavLM/HuBERT swaps (E15), or audio-side gradient rebalancing on EngageNet.
E19 + E06 together show there is no signal there to recover. E20 (F0 probe) remains worth running as a
final confirmation for the paper's evidence table, not as a route to a gain.

### 12.10 What we actually have that is positive — and the strongest framing available

The negative audio result (12.9) has dominated this session, but the positive results are real and are
what a paper would be built on.

| | Top-1 | Adjacent | Macro F1 | MAE |
|---|---:|---:|---:|---:|
| Majority-class predictor | 50.27 | — | 16.73 | — |
| **Best model** | **64.32** | **89.63** | **50.13** | **0.51** |

1. **A working ordinal engagement model** — +14 points over the trivial baseline and **3× its macro-F1**,
   so it genuinely separates minority classes rather than collapsing onto class 3.
2. **Adjacent accuracy 89.63% / MAE 0.51** — nearly 9 in 10 predictions land within one engagement level.
   For an ordinal task this is the model's most defensible property and it is currently underused in how
   the work is framed.
3. **Expected-threshold calibration is a citable methodological win** — argmax 61.97 → 64.32,
   **+2.35 points with no retraining**, thresholds fit on validation and applied to test with no leakage,
   reproducing across every checkpoint tested (V11, V12, V13, E04).
4. **Two useful negatives with controlled evidence** — audio does not contribute (12.9), and late text
   fusion actively hurts (55.32 vs 62.90).

**Honest caveats.** The 64.32 figure comes from the *video-only* ablation — the model with audio switched
off — which is awkward for an AV-framed paper. Against the published field it beats only the weakest
original baseline (LSTM 61.84) and trails CNN-LSTM 65.16, TCN 65.60, Transformer Fusion 66.50 and the
best published result, Transformer G+HP+AU at 67.61 (verified against `papers/EngageNet.pdf`).

**Strongest available framing — compete on the metrics the field does not report.** Every published
EngageNet result reports top-1 accuracy only. **None report adjacent accuracy, MAE, or macro-F1** — the
metrics that actually matter for a 4-level ordinal task with 50% class imbalance, where top-1 rewards
collapsing onto the majority class (see 12.7). Our 89.63 adjacent / 50.13 macro-F1 / 0.51 MAE have no
published comparison point. Claiming rigorous ordinal evaluation plus a calibration method that delivers
+2.35 points for free is a defensible contribution and a better position than chasing 69% top-1.

**Recommended paper shape:** ordinal-evaluation and calibration contribution on EngageNet, with the audio
ablation (12.9) as a rigorous negative result motivating the purpose-built corpus — not an AV-gain paper.

---

## Section 13 — CRITICAL: train/test preprocessing mismatch (2026-08-07)

### 13.1 The defect

Video frame counts per split, measured exhaustively over every `*_facecroppad.npy`:

| Split | n | % at 15 frames | Median | Max |
|---|---:|---:|---:|---:|
| Train | 7,983 | 22.8% | **50** | 63 |
| Validation | 1,071 | 96.2% | **15** | 50 |
| **Test** | **2,257** | **100.0%** | **15** | **15** |

**The model trains predominantly on 50-frame clips and is evaluated entirely on 15-frame clips.**
Test and Validation were extracted under the legacy 15-frame RAVDESS contract; Train was later re-extracted
at `--target_fps 5` (300 source frames / stride 6 = 50). Source clips are 10 s / 300 frames for all splits,
so this is purely a preprocessing inconsistency, not a property of the corpus.

This is the same legacy-contract failure as the 3.6 s audio truncation (Section 12.1) — the RAVDESS
15-frame/3.6 s defaults silently applied to a corpus they do not fit.

### 13.2 Why this invalidates every test number in the repo

Every EngageNet test result recorded before 2026-08-07 — including the 64.18% "best" — was measured under
a severe covariate shift: ~70% of training clips carry 50 frames of temporal evidence, while 100% of test
clips carry 15. The model is evaluated on inputs 3.3× shorter than what it learned from.

This is a strong candidate for the gap to the published field (65–68% video-only, Section 12 / memory.md).
It also explains the persistent val/test spread: validation is 96.2% 15-frame, so it tracks *test*
preprocessing rather than train, and the val→test gap reflects sample size rather than distribution.

**Consequence for the audio conclusion (12.9):** the modality ablation compared fusion vs video-only vs
audio-only *within the same mismatched setting*, so the relative comparison stands — audio-only collapsing
to the majority predictor is not explained by frame count. But the **absolute** numbers, and the question
of whether audio might help once video is no longer degraded, must be re-checked after the fix.

### 13.3 The fix

`preprocessing/engagenet/extract_faces.py --splits Test Validation --target_fps 5 --force`, sharded 6 ways
across both GPUs (~45 min). This makes Test/Validation match Train's 50-frame extraction. Labels and split
membership are untouched — only our own preprocessing changes, so comparability with published baselines
is unaffected (they use their own gaze/head-pose/AU features over full clips).

### 13.4 What must be re-run afterwards

Every headline number needs recomputing on the corrected test set:

| Re-run | Why |
|---|---|
| A0 baseline calibration (V12) | Re-establish the reference; the 64.05/64.18 figures are not valid as-is |
| E02/E03 modality ablation (3.6 s) | Confirm video-only ≥ fusion still holds |
| E06 modality ablation (10 s) | Confirm the decisive audio result holds |
| E04 / E05b finetunes | Their test numbers were measured on 15-frame test |

**Do not report any EngageNet test number until this re-run completes.**

### 13.5 Also discovered: `--train_frame_sampling random` is a no-op on EngageNet

`_random_synced_audio_video_crop` (`src/data/temporal.py:69`) returns unchanged when
`video_length <= max_video_frames`. EngageNet clips are ≤63 frames and `--max_video_frames` is 96, so
**the crop never fires** — zero of 300 sampled clips exceed 96 frames.

E09 proved this empirically: launched with `--train_frame_sampling random`, its validation log is
**bit-identical to E04's** across all 6 epochs. The V13 run credited in progress.md with "train-only synced
random A/V crop" was likewise a null experiment — its differences came from the extra finetuning epochs,
not the crop.

To actually use the augmentation, set `--max_video_frames` below the clip length (e.g. 32–40 for
50-frame clips). Tracked as E21.

### 13.6 Revalidation results — the frame-count fix changes the headline AND reverses 12.9

All six runs recomputed on the corrected 50-frame test set (Validation/Test went from 96.2%/100.0% at
15 frames to **0.0%**, median 50, matching Train). Refined-expected decoding:

| Model | Condition | Provisional (15-frame) | **Corrected (50-frame)** | Δ |
|---|---|---:|---:|---:|
| V12 (3.6 s audio) | AV fusion | 64.05 | **66.13** | **+2.08** |
| V12 | Video-only | 64.18 | 65.07 | +0.89 |
| V12 | Audio-only | 46.41 | 50.27 | +3.86 |
| E04 (10 s audio) | AV fusion | 62.68 | **66.36** | **+3.68** |
| E04 | Video-only | 62.99 | 66.22 | +3.23 |
| E04 | Audio-only | 50.27 | 50.27 | 0.00 |

**(a) The frame-count penalty was 2–3.7 points.** It fully accounts for the gap to the mid-range of the
published field. New best is **66.36%** (E04 AV fusion, adjacent 90.96, macro-F1 52.03), which now
**beats LSTM (61.84), CNN-LSTM (65.16), MARLIN-Transformer (65.20) and TCN (65.60)**, and sits 0.14 below
Transformer Fusion (66.50) and 1.25 below the best published result, Transformer G+HP+AU (67.61).
Best macro-F1 is 52.35 and best adjacent is 91.00. The architecture was never the problem the numbers
suggested — the evaluation was.

**(b) Section 12.9's conclusion is REVERSED for the 3.6 s model.** AV fusion now beats video-only on
**every one of the four decodes**:

| Decode | V12 fusion | V12 video-only | Δ |
|---|---:|---:|---:|
| argmax | 64.10 | 62.90 | **+1.20** |
| logit bias | 65.74 | 64.32 | **+1.42** |
| expected thresholds | 65.65 | 65.03 | **+0.62** |
| refined expected | 66.13 | 65.07 | **+1.06** |

Macro-F1 moves the same way (52.32 vs 50.37, **+1.95**). Consistency across all four decodes is what
makes this credible — a single decode at +1.06 would be inside the ±1.95 binomial 95% CI on 2,257 clips.
**Audio does contribute, and the earlier "video-only ≥ fusion" finding was an artefact of the degraded
15-frame video.** With video crippled, the fusion model's extra parameters were pure overhead; with
video intact, the audio stream adds complementary signal.

**(c) The E04 (10 s) model shows no such gain** — +0.13 refined-expected, and one decode negative. So
full-length audio does **not** reproduce the effect; the 3.6 s model is the one showing fusion benefit.
Do not claim the 10 s re-extraction improved fusion — on this evidence it did not.

**(d) What survives from 12.9 unchanged: audio-only is useless alone.** Both models land on *exactly*
50.27 / 16.73 — the majority-class predictor. So the correct statement is narrow and specific:
**audio carries no standalone engagement signal, but adds ~1 point on top of video when fused.** That is
a cross-modal interaction effect, not an independent audio capability, and it is consistent with E19
(hand-crafted acoustics also could not beat the majority baseline alone).

**Consequences.** The "no AV claim available on EngageNet" verdict in 12.9 is withdrawn for the 3.6 s
configuration. A modest, honestly-scoped AV claim is now defensible — but it must be stated as
"+1.06 points over video-only, consistent across four decoding schemes, on a corpus where audio alone is
at chance", not as a large fusion gain. **Re-open E15/SpecAugment only if a repeat run confirms the
+1 point holds**; a single seed is not enough for a paper claim (E22).

### 13.7 Remaining train defect and the four improvement levers (2026-08-07)

**The train split had the same defect.** 1,822 of 7,983 train clips (22.8%) were still at 15 frames
despite having full 10 s / 300-frame sources — verified with OpenCV on their `.mp4` originals. So a fifth
of the training set carried 3.3× less temporal evidence than the rest. The 15-frame `.npy` files were
deleted (sources untouched; list saved to scratchpad) and are being regenerated at `--target_fps 5`,
6 shards. After this, **all three splits are consistent for the first time.**

**Lever 1 — retrain on corrected splits (largest expected gain, not yet exploited).** Every existing
checkpoint (V12, E04, all of them) had its best epoch chosen using the broken 15-frame validation set
(96.2% short clips). Training data was mostly correct, but the *selection signal* measured the wrong
distribution. The current 66.36% is a model picked by a broken criterion and then evaluated properly.
**Nothing in this repo has been trained under valid conditions.**

**Lever 2 — train-split consistency.** In progress (above).

**Lever 3 — checkpoint ensembling (cheapest reliable gain).** 8 usable checkpoints exist. Logit-averaging
is inference-only, ~30 min, and typically returns +1–2 points. The gap to the best published baseline
(67.61) is only 1.25 points, so this alone could close it.

**Lever 4 — two wasted settings.**
- `--max_video_frames 96` pads every 50-frame clip with 46 empty frames. Set to 50 to remove the waste.
- Setting it to ~40 **finally enables the random crop**, which has been a silent no-op (13.5, proven by
  E09's bit-identical logs). The project has never actually had temporal augmentation — precisely the
  remedy for E04's epoch-3 overfitting.

**Where the headroom is**

| Metric | Current best | Published field |
|---|---:|---|
| Top-1 | 66.36 | **67.61** best published (Transformer G+HP+AU) / 66.50 Transformer Fusion |
| Adjacent | 91.00 | **not reported by any EngageNet paper** |
| Macro-F1 | 52.35 | **not reported by any EngageNet paper** |

Top-1 needs +1.25 to beat the best published baseline — plausible from retraining + ensembling.
But **macro-F1 at 52 against top-1 at 66 is the real weakness**: minority classes remain poorly
separated. That is where E11 (class weighting, never completed) and macro-F1-targeted threshold
optimisation apply.

**Recommended next runs**

| ID | Run | Config |
|---|---|---|
| T01 | Train re-extraction | `--splits Train --target_fps 5`, 6 shards — **launched 2026-08-07, ~25 min total**. The 1,822 short `.npy` files were deleted (sources intact; list at `scratchpad/short_train.txt`) and regenerate without `--force`, so complete clips are skipped. Verify with the frame-count check in 13.1 before starting F01/F02. |
| F01 | Retrain, no augmentation | From EfficientFace pretrain, corrected splits, `--max_video_frames 50`, warmup_cosine, ordinal 0.15 |
| F02 | Retrain, with real augmentation | As F01 but `--max_video_frames 40 --train_frame_sampling random` — first run where the crop actually fires |
| F03 | Ensemble | Logit-average the top checkpoints, inference-only |
| E11r | Class weighting | Relaunch on corrected splits; targets the macro-F1 gap |
| E22 | Seed repeat | Confirm the +1.06 fusion gain (13.6) before any paper claim |

### 13.8 Post-T01 state and the single largest remaining gain

**When T01 completes, all three splits are consistent for the first time in the project's history** —
Train, Validation and Test all at `--target_fps 5`, median 50 frames, 0% at the legacy 15-frame length.
Every result produced before this point was measured under at least one preprocessing inconsistency
(3.6 s audio, 15-frame test/val, or 22.8% 15-frame train). Verify with the frame-count check in 13.1
before launching anything downstream.

**Retraining (F01/F02) is the single largest untapped gain before submission.** The reason is specific,
not general optimism: every existing checkpoint — V12, E04, and all of their ancestors — had its best
epoch selected against a validation set that was 96.2% 15-frame clips. The training data was largely
correct, but the criterion choosing *which epoch to keep* was scoring the wrong distribution. The current
66.36% is therefore a model picked by a broken selection signal and then evaluated properly.

**Nothing in this repository has ever been trained end-to-end under valid conditions.** That is the gap
F01/F02 close, and it is the most credible route to the +1.25 points needed to pass the best published
baseline (67.61) — ahead of any architectural change, and ahead of E15-style encoder work, which E19
already showed is not where the limitation lies.

Ordering: **F03 (ensembling) can run immediately** — inference-only on existing checkpoints, ~30 min,
typically +1–2 points, and it does not touch training data. F01/F02 run in parallel across the two GPUs
once T01 clears. E22 (seed repeat) gates any written audio-visual claim.

### 13.9 T01 complete — all splits consistent; F01/F02 running

Verified after re-extraction:

| Split | n | % at 15 frames | Median |
|---|---:|---:|---:|
| Train | 7,983 | 0.1% | 50 |
| Validation | 1,071 | 0.0% | 50 |
| Test | 2,257 | 0.0% | 50 |

The residual 0.1% is ~8 clips whose source videos are genuinely short. **This is the first point in the
project's history at which train, validation and test share the same preprocessing.**

F01 (no augmentation, `--max_video_frames 50`) and F02 (`--max_video_frames 40 --train_frame_sampling
random`) launched in parallel on GPU 0 and GPU 1, both 18 epochs, warmup-cosine at lr 5e-4, grad-clip
5.0, ordinal-distance 0.15, 10 s audio.

**The augmentation is confirmed working for the first time.** F02's first batch reports
`visual=(8, 40, 3, 224, 224)` and `audio=(8, 64, 347)` against 432 audio frames uncropped — the video is
cropped to the 40-frame cap and the audio window is cropped proportionally, i.e. the crop is both firing
and synced. Every prior run that claimed this augmentation (V13, E09) was a no-op (13.5).

F01 vs F02 is therefore a clean test of whether temporal augmentation addresses the epoch-3 overfitting
seen throughout the E04 lineage.

### 13.10 What F01/F02 decide — read before interpreting their results

**Status: both running, launched 2026-08-07, ~2 h wall-clock** (18 epochs each, ~6 min/epoch, F01 on
GPU 0 and F02 on GPU 1 in parallel), followed automatically by refined expected-threshold calibration.

**Their role:** these are the first models in this project trained end-to-end under valid conditions —
correct audio span (10 s), all three splits at median 50 frames, and a validation set that matches the
test distribution so checkpoint selection finally scores the right thing. Everything preceding them was
trained or selected under at least one preprocessing defect.


Three open questions turn on these two runs. Record the answer to each explicitly when they land, and do
not let a good top-1 number stand in for all three.

**(a) Headline accuracy and the gap to the field.** Current best is 66.36%, which is 1.25 below the best
published baseline (Transformer G+HP+AU, 67.61) and 0.14 below Transformer Fusion (66.50). F01/F02 are the first models trained with a
validation set that matches the test distribution, so this is the most credible route to closing that gap
— more so than any architectural change. **If they do not beat 66.36%, the conclusion is that broken
checkpoint selection was not the limiting factor**, and the remaining levers are ensembling (F03) and
accepting the ordinal-metrics framing rather than the accuracy race.

**(b) The audio-visual claim.** The +1.06 fusion gain (13.6) was measured on a checkpoint trained under
the defects. Re-run the modality ablation on whichever of F01/F02 wins. **The claim only survives if
fusion still beats video-only on a cleanly-trained model** — a gain that appears only in defective
training is not a result. This matters more than the headline number: it is the difference between an
audio-visual paper and a video paper with an ablation appendix.

**(c) Whether augmentation fixes the overfitting.** F01 vs F02 is controlled — identical except the frame
cap and crop flag. The E04 lineage peaks at epoch 3 and decays; if F02 holds its peak later, the synced
crop becomes default for short finetunes on this corpus. If it does not help, the overfitting is a
dataset-size limit (7,983 training clips) rather than an augmentation gap, which argues for the
purpose-built corpus rather than more tuning here.

**Metric discipline when reporting.** Report top-1, adjacent accuracy, MAE and macro-F1 together. Top-1
alone is misleading at 50% class imbalance (12.7): a majority-class predictor scores 50.27% top-1 at
16.73 macro-F1, so a model can gain top-1 by collapsing onto class 3 while getting worse at the task.
The current macro-F1 of ~52 against top-1 of ~66 is the real weakness, and F01/F02 should be judged on
whether they narrow that spread as much as on whether they raise top-1.

#### 13.10.1 Decision matrix — what each outcome means for next steps

> **PENDING — both runs still in progress as of this writing.** The matrix below is a *pre-commitment*,
> written before the results so the interpretation is not chosen after seeing numbers we might prefer.
> No row has an answer yet. Do not write a final interpretation, update the professor brief, or draft any
> claim until F01/F02 complete and the modality ablation has been run on the winner.


Write the answer to each row when F01/F02 land; do not let one good top-1 number stand in for all three.

| Question | If YES | If NO |
|---|---|---|
| **Beats 66.36% top-1?** | Broken checkpoint selection *was* a real limiter. Push further: F03 ensembling on top, then E22 seeds. The best published baseline (67.61) is a live target. | Selection was **not** the limiter. Stop chasing accuracy: fall back to F03 for whatever it gives, and commit to the **ordinal-evaluation + calibration** framing where adjacent (91.00) and macro-F1 (52.35) have no published comparison. |
| **Fusion still beats video-only on the winner?** | The audio-visual claim is real and survives clean training. Scope it exactly: "+~1 point over video-only across four decoders, on a corpus where audio alone is at chance." Run E22 to confirm across seeds, then it can be written. | **The AV claim dies.** The +1.06 was an artefact of defect-trained weights. Paper becomes a video model plus a rigorous negative audio ablation — still novel (no prior EngageNet paper tested audio), but not an AV-gain paper. Update §12.9/§13.6 accordingly. |
| **Does F02 hold its peak past epoch 3?** | Synced crop becomes default for all short finetunes on this corpus; re-run the best configs with it. | Overfitting is a **dataset-size limit** (7,983 training clips), not an augmentation gap. That is a direct argument for the purpose-built corpus (§2–11) rather than further tuning on EngageNet. |

Judge every row on top-1, adjacent, MAE **and** macro-F1 together (§12.7): a majority-class predictor
scores 50.27% top-1 at 16.73 macro-F1, so top-1 can rise while the model gets worse at the actual task.

#### 13.10.2 Early-run concern — F01/F02 may be misconceived, pivot criterion set

Through epoch 2 both runs are **declining, not climbing**, and both sit below the 50.27% majority-class
baseline:

| Run | ep 1 top-1 | ep 2 top-1 | ep 1 MAE | ep 2 MAE |
|---|---:|---:|---:|---:|
| F01 (no augmentation) | 48.46 | 46.41 | 0.926 | 1.021 |
| F02 (crop active) | 49.21 | 47.15 | 0.904 | 0.979 |

Two epochs is not conclusive — E05b dipped similarly before recovering, and early cosine warmup can look
like this. But the plausible cause is structural, not transient: **F01/F02 train from the EfficientFace
AffectNet face-recognition pretrain, not from an existing engagement checkpoint.** They are learning the
task from scratch in 18 epochs, while the V12/E04 lineage reached 66% through many more epochs of
accumulated finetuning. 18 epochs from that starting point may simply be too short.

**Pivot criterion: if neither run crosses 55% top-1 by epoch 5–6, stop them.** The better experiment is
then to **finetune the existing best checkpoint on the corrected data** rather than retrain from scratch.
That is also the cleaner design: it isolates the variable we actually care about — correct preprocessing
and a valid selection signal — instead of confounding it with training length and initialisation.

Note this does not change what §13.10 is testing, only how to get there. The three questions
(headline accuracy, audio-claim survival, augmentation efficacy) are equally answerable from a finetune
of the best checkpoint on corrected data, and that route is roughly 45 min rather than 2 h.

#### 13.10.3 The retraining decision is now a supervision question, not only a technical one

`docs/professor_progress_brief.md` (removed) frames six discussion points, two of
which bear directly on whether to continue F01/F02 or pivot per §13.10.2:

- *"Where to spend remaining compute"* — ensembling plus retraining could plausibly add 1–2 points of
  top-1, or the same effort could target the **macro-F1 gap** (52 against 66 top-1), which is the more
  honest weakness and the metric with no published competition. If the advisor prefers the
  ordinal-evaluation framing, the macro-F1 work outranks chasing the Transformer baseline and the
  retraining pivot matters less.
- *"How strongly to state the audio result"* — if the AV claim is to be load-bearing, E22 (seed repeat)
  and the modality ablation on a cleanly-trained checkpoint become mandatory, which makes finishing a
  *valid* training run a prerequisite rather than an optimisation.

So the §13.10.2 pivot should be decided together with the framing question, not purely on whether the
epoch 5–6 threshold is met.

#### 13.10.4 Pivot criterion still live — runs recovering at epoch 3

| Run | ep 1 | ep 2 | ep 3 |
|---|---:|---:|---:|
| F01 (no augmentation) | 48.5 | 46.4 | **49.9** |
| F02 (crop active) | 49.2 | 47.2 | **51.3** |

Both turned upward at epoch 3 after the epoch-2 dip — the recovery shape E05b showed, so the early
decline looks **transient rather than structural**. Neither has crossed the 55% threshold set in
§13.10.2, so the **epoch 5–6 decision point remains active**: continue the full 18 epochs, or stop and
finetune the best existing checkpoint on corrected data instead (~45 min, and a cleaner isolation of the
preprocessing variable). Decide alongside the framing question in §13.10.3, not on the threshold alone.

#### 13.10.5 PIVOT CRITERION TRIGGERED — F01/F02 have failed

The §13.10.2 criterion ("if neither run crosses 55% top-1 by epoch 5–6, stop them") is **met**. Neither
did, and both have since destabilised rather than converged:

| Run | ep 4 | ep 5 | ep 6 | ep 7 | ep 8 | ep 9 | ep 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F01 | 52.4 | 47.3 | 49.2 | 45.8 | **53.6** | 45.4 | — |
| F02 | 51.4 | 48.1 | 49.5 | 45.4 | **54.0** | 44.2 | 40.6 |

They oscillate across an 8–13 point band with no trend, and F02 is now declining. The diagnosis in
§13.10.2 holds: **18 epochs from the AffectNet face-recognition pretrain is too short to learn this
task.** The V12/E04 lineage reached 66% through many more epochs of accumulated finetuning, not from a
generic pretrain in 18.

**Decision: stop F01/F02 and finetune the existing best checkpoint on the corrected data instead**
(~45 min vs the ~50 min remaining on runs that are not converging).

> **AWAITING USER APPROVAL — not yet executed.** F01/F02 are still running and have not been stopped;
> G01/G02 have not been launched. The recommendation was put to Yuvraj and no answer has been received.
> Do not kill the runs or start the replacements without his go-ahead. This is also the better-designed
experiment — it isolates the variable of interest (correct preprocessing plus a valid selection signal)
rather than confounding it with initialisation and training length.

Successor runs to launch (naming: G01/G02):

| ID | Config |
|---|---|
| G01 | Finetune V12/E04 best on corrected splits, 10 s audio, `--max_video_frames 50`, ordinal 0.15, lr 5e-5, warmup_cosine, 6 ep |
| G02 | As G01 but `--max_video_frames 40 --train_frame_sampling random` — keeps the augmentation comparison that F01/F02 were meant to provide |

The three questions in §13.10 are unchanged and equally answerable from G01/G02.

**Lesson worth keeping:** when testing whether a *data* fix helps, finetune from the existing best rather
than retraining from a generic pretrain. Retraining changes two variables at once and needs far more
epochs before the comparison becomes meaningful — a pre-committed stopping criterion is what stopped this
from burning two full GPU-hours.

#### 13.10.6 Session close — what is recorded, what is gated

Everything substantive from this period is written down and version-controlled:

- **Three preprocessing defects** found and fixed — 3.6 s audio truncation (§12.1), 15-frame test/val
  against 50-frame train (§13.1), 22.8% of train also at 15 frames (§13.7). All splits now consistent
  (§13.9).
- **Corrected result set** R01–R06 (§13.6): new best 66.36%, and the audio null result reversed —
  fusion beats video-only across all four decoders for the 3.6 s model.
- **Encoder-free audio probe** (§12.7): three model families converge at the same ceiling, so the
  limitation is the data, not the encoder — E15 deprioritised.
- **Literature verified against the PDFs we hold**, not search: best published EngageNet test is
  **67.61%** (Transformer G+HP+AU), our gap is **1.25**, and TCCT-Net's 68.91% is excluded as unverified.
- **Temporal augmentation confirmed working** for the first time (§13.9), after being a silent no-op
  throughout the project's history (§13.5).
- **Decision matrix pre-committed** before results (§13.10.1), and the **stopping criterion triggered
  as designed** (§13.10.5), catching F01/F02's failure at epoch 10 rather than after two GPU-hours.
- **Professor brief rewritten** to evidence framing with a per-claim support rating.
- **Evidence consolidated** into `docs/evidence_tables.md` (§13.11): 60 run directories and 105
  evaluations across 3 trained datasets, 9 parameter sweeps, modality ablation and decoding ladder.

Sections 13.0–13.12 are complete and version-controlled (13.11 evidence consolidation, 13.12 evidence audit).

**Gated on one answer: whether to switch from F01/F02 to G01/G02.** Nothing else is outstanding.

*Session ends with the G01/G02 decision pending. F01/F02 were left running — F01 at epoch 9, F02 at
epoch 11 of 18, roughly 7–9 epochs each remaining (~50 min). They were not stopped because that call is
Yuvraj's; the §13.10.5 recommendation stands either way.*

### 13.11 Evidence consolidation for the advisor meeting (2026-08-07)

`docs/evidence_tables.md` (removed) created on request: a presentation-ready evidence summary
rather than a development narrative. Numbers only, HTML tables throughout, no model description (the
advisor knows the architecture) and no discussion/decision sections.

Consolidates **60 run directories and 105 recorded evaluations** into 9 sections. Content:

| § | Evidence |
|---|---|
| 2 | Multi-dataset: RAVDESS 81.88% (8-class, 2,880 clips), EngageNet 66.36% (4-level ordinal, 11,206), DAiSEE 55.25% (4-level ordinal, 8,925); CREMA-D and CMU-MOSEI preprocessed but untrained |
| 3.1–3.9 | Parameter sweeps: attention heads (1/4/8), LR (0.06→0.0001), epochs (30–100), loss (CE vs ordinal at 2 weights), class balancing, audio augmentation, visual backbone, temporal sampling, audio span |
| 4 | Modality ablation: 2 models × 3 conditions × 4 decoders, plus the fusion−video delta table |
| 5 | Encoder-free audio probe: 3 model families |
| 6 | Decoding ladder: 9 checkpoints × 4 schemes with gain-over-argmax |
| 7 | Corpus measurements: speech coverage, class distribution, clip/frame budgets |
| 8 | Published baselines, PDF-verified, validation *and* test columns |
| 9 | Late text fusion: 3 configurations + text-source properties |

**Two exclusions made rather than asserted**, applying the verification discipline from §13.6:
an MFCC-vs-mel row was removed because the early `spec_*` runs did not record `audio_features` and also
used lr 0.06 (feature type and LR confounded); the F02 random-crop row is marked incomplete; TCCT-Net's
68.91% is marked unverified.

### 13.12 Evidence audit — which comparisons are matched, which are confounded (2026-08-07)

Every sweep table in `docs/evidence_tables.md` was checked against the actual `opts*.json` of the runs it
cites. Tables are now labelled **matched** or **confounded** in the document itself.

| Section | Status | Detail |
|---|---|---|
| 3.3 Epoch budget | **Matched** | 75 vs 100 ep, same heads/LR/batch → 100 ep costs 5.62 points (overfitting) |
| 3.6 Audio augmentation | **Matched — cleanest in the document** | All four runs mel, h4, lr 0.01, 75 ep, bs 8; only the two flags differ |
| 3.8 Temporal sampling | **Matched** | Stride vs uniform, both h8 / 96 frames |
| 4 Modality ablation | **Matched — strongest evidence** | Same weights, same test set, only the zeroed input changes |
| 5 Encoder-free probe | **Independent** | No neural network involved; isolates data ceiling from encoder capacity |
| 6 Ordinal decoding | **Matched** | Same logits, four decoders |
| 3.1 Attention heads | **Partly confounded** | 1-head run used bs 2; 4- and 8-head used bs 8. The 4→8 comparison (+3.33) is matched |
| 3.7 Visual backbone | **Partly matched** | Headline EfficientFace-vs-attention-local pair matched at lr 0.01; the two scratch rows differ in LR |
| 3.2 Learning rate | **Range explored, not controlled** | Rows differ in heads and dataset. Matched pair: RAVDESS h8 0.01 (78.54) vs 0.005 (77.50) |
| 3.4 Loss function | **Confounded** | CE runs h4, ordinal runs h8, LRs differ. Direction consistent across 4 runs but not isolated |
| 3.5 Class balancing | **Confounded — not isolated** | v12_02 vs v12_05 also differ in LR (0.001/0.0001), ordinal weight (0.35/0.15) and batch size (8/2) |
| 3.9 Audio span | **Different checkpoints** | R01 vs R04 are separate models; all gaps inside ±1.95 CI → no measurable difference |
| 2 DAiSEE row | **Weak** | 55.25% is a **10-epoch** run; the 40-epoch run has no recorded test eval. Pipeline-portability evidence only (published DAiSEE ~69–73%) |

Also added: **§10 Glossary** — ~30 terms in plain language with our numbers attached (metrics, training
parameters, experiment types), so the evidence document is self-contained for a supervision discussion.

**Presentation guidance recorded in the document itself** rather than held separately: §3.4 and §3.5 are
to be described as trends, not isolated effects; the DAiSEE row as portability, not a result; §6 should be
quoted as **+1.47 on the best model**, not the +8.20 seen on weak checkpoints (calibration rescues poor
models more than good ones).

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
| TX1 | `preprocessing/zoom/extract_chat.py` — parse Zoom chat export, per-student per-clip text (Section 17) | High — unblocked now |
| TX3 | Chat elicitation prompts + display-name convention into E1 session script (Section 17.1) | High — merge with E1 |
| TX2, TX4–TX7 | Chat schema, text encoder, null token, text dropout, leakage gate + A/B (Section 17.7) | Blocked on pilot data |

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

### 13.13 F01/F02 final result, a calibration defect, and the overnight G-sweep (2026-08-12)

**F01/F02 completed.** They confirm the §13.10.5 diagnosis. Neither beat the 66.36% incumbent, and
neither approached it:

| Run | best val top-1 | best val epoch | test top-1 (argmax) | test top-1 (logit bias) |
|---|---:|---:|---:|---:|
| F01 (no augmentation) | 53.59 | 8 | 52.62 | 56.69 |
| F02 (crop active) | 53.97 | 8 | 51.91 | 55.63 |

Both sit **below the 50.27% majority-class baseline on some epochs** and oscillate without trend across
all 18. The §13.10.1 decision matrix row "beats 66.36% top-1?" is answered **NO for the retraining
route** — but the correct reading is narrower than "checkpoint selection was not the limiter". These runs
retrained from the AffectNet pretrain in 18 epochs, so they confound the data fix with initialisation and
training length. The matrix row is properly answered only by a finetune of the existing best checkpoint,
which is what the G-sweep below does.

#### 13.13.1 New defect: calibration ran at a frame cap the models never trained at

`scripts/exp2026_run.sh` hardcodes `--max_video_frames 96` in `COMMON`, and `run_train` calls
`run_calib` **without forwarding the run's own override**. F01 trained at `mvf 50` and F02 at `mvf 40`;
both were then calibrated and tested at `mvf 96`. Every F01/F02 **test** number above is therefore a
train/eval mismatch and should not be cited.

Their **validation** curves are unaffected — validation runs inside training at the correct cap — and the
val curves alone (peak 53.6 / 54.0 against E04's 65.3) already support the failure conclusion. So the
verdict stands; the test numbers do not.

This is the fourth preprocessing/evaluation mismatch found in this project (after §12.1, §13.1, §13.7).
The pattern is consistent: **a default in a shared config silently overriding a per-run setting.**
`scripts/night/run_job.sh` now reads `max_video_frames` and `frame_sampling` back out of each run's own
`opts*.json` and passes them to calibration, so the two can no longer diverge.

#### 13.13.2 Overnight G-sweep — design

Seven matched finetunes, all warm-started from the E04 best checkpoint (the 66.36% model) onto the
corrected splits, two GPUs in parallel. Everything is held fixed — `h8`, `mel`, `fusion it`, `bs 8`,
`ordinal_distance 0.15`, `label_smoothing 0.1`, `sgd`, `selection_metric top1_accuracy`, 10 epochs, AV
only — **except the one variable each run names.** This directly addresses the §13.12 audit, which found
the older class-balance and loss-function tables confounded across three or four parameters at once.

| ID | GPU | Variable under test | Why |
|---|---|---|---|
| G00 | 0 | none — E04's exact config (`mvf 96`, step lr 5e-5) | **Control.** Isolates the one variable §13.8 cares about: corrected data + a validation set that matches test, so selection finally scores the right distribution |
| G01 | 0 | `mvf 50` | Stops padding every 50-frame clip with 46 empty frames (§13.7 lever 4) |
| G03 | 0 | `class_weighting sqrt_inverse` | E11r, never completed. Targets the macro-F1 gap (52 against top-1 66) — the real weakness and the metric with no published EngageNet comparison |
| G02 | 1 | `mvf 40` + `train_frame_sampling random` | The only cap at which the temporal crop actually fires (§13.5). Tests whether augmentation fixes the E04 lineage's epoch-3 overfitting |
| G04 | 1 | `class_balance_sampler sqrt_inverse` | Same target as G03 by a different mechanism, and matched this time |
| G05 | 1 | `lr 1e-4` + warmup-cosine | E04 sat flat at ~65 for 8 epochs at 5e-5, suggesting it barely moved off the warm start |
| G06 | 1 | `spec_augment` | Bears on the AV claim: if audio's +1.06 is real, regularising the audio path should move it |

**Phase 2 is generated automatically** once all seven finish (`scripts/night/phase2.py`), and answers the
two remaining §13.10 questions:

- **Modality ablation** (`audio_only` / `video_only`) on the top two runs → does the AV claim survive
  clean training (§13.10 question b). This is the difference between an audio-visual paper and a video
  paper with a negative ablation appendix.
- **Two seed repeats of the winner** (E22) → is the fusion gain seed noise. No paper claim without this.
- **F03 checkpoint ensembling** (`scripts/night/ensemble.py`), inference-only. Averages **probabilities,
  not logits**, because members differ in LR and loss and would otherwise be weighted by logit magnitude.

**Calibration discipline is unchanged and was verified against the code:** logit bias and both
expected-threshold decoders are fit on **validation** logits and applied to **test**. Nothing is fit on
test.

#### 13.13.3 New capability: `--save_every_epoch`

Additive flag (default off, so no existing behaviour changes). Retains `epochs/epoch_NNN.pth` alongside
the usual best/last checkpoints. Two payoffs: any epoch can be re-selected post-hoc against a different
`--selection_metric` without retraining — which matters because the ordinal framing argues for selecting
on MAE rather than top-1 — and every epoch becomes an ensemble member for F03.

#### 13.13.4 Resilience

Both workers are launched under `setsid nohup`, so the sweep is independent of the SSH session, the
laptop sleeping, or Claude Code exiting. A failing job writes `FAILED` and the queue continues; a
completed job writes `DONE` and is skipped on relaunch, so the whole sweep is resumable by re-running
`scripts/night/worker.sh`.

---

## 14. Streamlit Engagement UI (2026-08-12)

The Streamlit app was written for the RAVDESS 8-class emotion model and had drifted out of
scope. It now serves the EngageNet engagement models instead. `ui/inference.py` and `ui/app.py`
were rewritten; the RAVDESS-specific asset lookups (`_find_precomputed_ravdess_asset`,
annotation-file audio resolution) were dropped rather than carried forward.

### 14.1 Which checkpoints the UI exposes

Only the two runs that beat the published EngageNet video-only baselines, decoded with the
calibrated `refined_expected_thresholds` from their own calibration sweep:

| Key | Checkpoint | Audio contract | Test top-1 | Adjacent | Calibration source |
|---|---|---|---:|---:|---|
| `e04_ft10s` (default) | `results/exp2026/E04_ft10s_ord_lr5e5_e8/ENGAGENET_multimodal_cnn_15_best.pth` | full 10 s | **66.36%** | 90.96% | `results/exp2026/R04_e04_av_fixed/` |
| `v12_05` | `results/v12_05_finetune_uniform_best_ord_nosampler_lr0001_e30/model.pth` | 3.6 s centre crop | 66.13% | 91.00% | `results/exp2026/R01_v12_av_fixed/` |

Everything else in `results/` is either RAVDESS/DAiSEE, a superseded run, or below these two.
The registry lives in `MODEL_REGISTRY` in `ui/inference.py`; when the G-sweep (§13.13) produces
a winner that beats 66.36%, add it there with its own thresholds and metrics.

### 14.2 Long-video handling

EngageNet clips are exactly 10 s. Arbitrary uploads are sliced into 10 s windows (optionally
50% overlapping), each window preprocessed and scored independently, then rolled up into a
session summary: mean expected score, share of time at level ≥ Engaged, time-per-level
distribution, most/least engaged window, and a per-window CSV export.

### 14.3 Preprocessing contract — verified bit-exact

The UI reproduces the training contract rather than approximating it. Verified by extracting
frames from a raw `.mp4` and diffing against the stored `*_facecroppad.npy`: **mean absolute
difference 0.0**, and window scores match the training path to within 0.02.

- **5 fps frame stride, not uniform-96.** `extract_faces.py` was run with `--target_fps 5`, so
  every stored clip is ~50 frames. Sampling 96 uniformly — the obvious reading of
  `--max_video_frames 96` — hands the visual stem a temporal resolution it never saw and
  measurably changes predictions. The 96 cap only applies after the 5 fps stride.
- **BGR channel order.** `extract_faces.py` saves frames straight from `cv2` without a
  BGR→RGB conversion, so the UI must not convert either.
- **MTCNN is mandatory.** With `facenet_pytorch` missing, the Haar fallback detected a face in
  only 32% of frames on a test clip and scores diverged wildly (2.50 → 0.08). The app now
  refuses quietly no longer — it shows a blocking error telling the user to run inside the
  `avtca` conda env.
- **Audio** is mono 22050 Hz, 64-bin mel in dB (`ref=np.max`), sliced per window and centre
  cropped to 3.6 s only for the `v12_05` contract.

### 14.4 Decoding

Softmax expected class index → `np.digitize` against the run's refined thresholds. This is the
decision rule the reported accuracy was measured with; plain argmax is 1.5–2 points worse on
test and is shown in the per-window table only as a secondary signal.

### 14.5 Optional modality split

A checkbox runs each window three times (fused / `audio_only` / `video_only`) using the model's
existing `ablate_modality` hook, and reports how often each single-modality decode agrees with
the fused one. Expect low audio agreement — that is the known EngageNet result, not a UI bug.

#### 13.13.5 Concurrency: what actually limits runs per GPU (measured 2026-08-12)

The sweep started at one run per GPU. Measurement showed that was leaving capacity unused, but not for the
reason a "24 GB card, small model" reading would suggest.

| Constraint | Measurement | Verdict |
|---|---|---|
| GPU memory | mvf 96 run = **19.3 GB of 24**; mvf 50 ≈ 11.7 GB; mvf 40 = 10.3 GB | Hard cap: **1 run at mvf 96, 2 at mvf 50** |
| Dataloader | Data wait **0.007 s of 0.46 s per batch (1.5%)** | Not the bottleneck |
| CPU | **0% idle, 0% iowait**, load 107 on 20 cores | Saturated |
| GPU utilisation | GPU0 ~60%, GPU1 ~45% | Headroom being wasted |
| Threads | **495 threads on 20 cores**, `OMP_NUM_THREADS` unset | The waste |

The model is small (2.4 M parameters) but the *input* is not: 40–96 frames of 224×224×3 per sample. Memory
scales with the frame cap, so **the frame cap sets the concurrency limit**, not parameter count.

The CPU saturation was largely self-inflicted. PyTorch defaults to one OMP thread per core *per process*,
and the sweep runs ~22 processes, giving 495 threads on 20 cores — load 107 and **48% of CPU time in
sys**. `OMP_NUM_THREADS=1` is correct here: the dataloader does no meaningful BLAS and the training step
runs on the GPU. Set in `scripts/night/common.sh`, with `--n_threads` reduced from 10 to 6.

Measured effect of going from 2 concurrent runs to 3:

| | 2 runs | 3 runs |
|---|---:|---:|
| GPU0 | 2.17 batch/s | 2.23 batch/s |
| GPU1 | 2.36 batch/s | **3.51** batch/s |
| Total | 4.53 batch/s | **5.74 batch/s (+27%)** |

Doubling up on one card returns **+49% on that card** — real, but well short of 2×, because CPU is still
the binding constraint (0.4% idle at three runs). **Six to seven runs per GPU is not achievable**, and
adding a fourth returns progressively less.

**Harness change:** `run_job.sh` now gates on free GPU memory before starting (waits up to 4 h for
`GPU_FREE_MIB`, default 13000), so several workers can share a card without racing into an OOM. A second
worker per GPU (`worker.sh <gpu> <queue> nophase2 <tag>`) drains an additional queue; only the primary
worker per GPU generates and drains the phase-2 queue, or they would duplicate every phase-2 job.

**Five runs added** with the freed capacity, chosen so that sweep tables §13.12 marked *confounded*
become citeable — G07/G01/G09 give a clean three-point ordinal-weight sweep (0.35 / 0.15 / 0.60), G08
isolates CE against ordinal loss, and G12/G13 isolate EMA and label smoothing, both of which have been
carried as unexamined defaults in every run in this project.

### 13.14 G-sweep results — the sweep is flat, and the noise floor explains why (2026-08-12)

Fourteen matched runs completed overnight (G00–G13 plus two seed repeats), all warm-started from the E04
best checkpoint onto the corrected splits, plus a full modality ablation on 13 of them and a 14-member
ensemble. **All numbers below use a single fixed decoder, `refined_expected_thresholds`**, so fusion and
video-only are compared like for like. This matters: G04 reads 66.76 under `expected_thresholds` and
65.78 under `refined_expected_thresholds` — the decoder alone moves a checkpoint by ~1 point, so no
number in this project should be quoted without its decoder.

#### 13.14.1 The decisive statistic

| Source of variation | Top-1 sd | Top-1 spread |
|---|---:|---:|
| **14 different configurations** (loss, ordinal weight, class weighting, balanced sampler, LR, EMA, label smoothing, SpecAugment, frame cap, augmentation) | **0.52** | 1.95 |
| **Same configuration, 3 random seeds** (G04) | **0.45** | 0.89 |

**Varying every hyperparameter we tested produces almost exactly as much variation as re-running one
configuration with a different seed.** For macro-F1 the picture is worse: across-run spread 3.00 against
a seed range of 2.29.

This is the quantitative form of the conclusion §12.7's encoder-free probe reached qualitatively — three
model families converging on the same ceiling. **The limitation is the corpus, not the model, the loss,
or the optimiser.** It is a stronger and more useful result than any 0.3-point accuracy claim, and it is
the empirical justification for the purpose-built dataset in §2–11.

**Practical consequence: no single-run comparison in this project is interpretable.** Every past sweep
table quoted differences of 0.5–2 points between single runs. The measured noise floor is 0.45 sd on
top-1 and ~1.1 sd on macro-F1. Those tables were reading noise. Future comparisons need ≥3 seeds or they
should not be made.

#### 13.14.2 The three pre-committed questions (§13.10.1)

**(a) Did anything beat 66.36% top-1? No.** Best of the night is 66.67 (G04 seed 3), **+0.31** — well
inside the 0.45 seed sd, and inside the ±2.75 binomial CI on 2,257 clips. The cleanest evidence is G00,
E04's exact configuration on corrected data, at **66.09 vs 66.36** — corrected preprocessing and a valid
selection signal changed test accuracy by *−0.27*. The §13.7 "Lever 1" hypothesis, that broken checkpoint
selection was the largest untapped gain, is **refuted**. The matrix's pre-committed response applies:
stop chasing accuracy, commit to the ordinal-evaluation and calibration framing.

**(b) Does fusion still beat video-only on cleanly-trained models? On top-1 yes, on macro-F1 no.**

| Metric | Fusion better in | Mean Δ | sd |
|---|---|---:|---:|
| Top-1 | **12 / 13** | **+0.67** | 0.59 |
| macro-F1 | 4 / 13 | −0.09 | 1.21 |

Audio-only never exceeds the majority-class baseline on any model (values 46.59–50.27 against 50.27).
So §13.6's finding survives **only in its narrow form**: audio carries no standalone engagement signal
but adds a small consistent top-1 gain when fused. §13.6's additional claim of a **+1.95 macro-F1 gain
does not reproduce** and must be withdrawn — it was an artefact of defect-trained weights.

The mean +0.67 is below the noise floor of any single comparison; what makes it credible is consistency
of sign across 13 models (12/13) and across the two seeds tested (+0.80, +1.99). State it as
"a small positive fusion effect, consistent in sign across 13 matched models", never as a point estimate
from one run.

**A retracted intermediate finding, recorded because it is instructive.** Mid-sweep, G04 showed a
macro-F1 fusion gain of +2.65 while every non-class-balanced run was negative, suggesting audio's
minority-class contribution required class-balanced training to surface. **Seed 2 of the same config gave
−0.61.** The effect was seed noise. It survived less than two hours and only because the seed repeats
were run — which is precisely what E22 was for.

**(c) Did augmentation hold its peak? No — it was the second-worst run.** G02 (mvf 40 + random crop,
the first run in this project where the crop actually fires) scored **64.89**, against 65.82 for the
otherwise-identical G01. The matrix's pre-committed reading applies: **overfitting is a dataset-size
limit** (7,983 training clips), not an augmentation gap — a direct argument for the purpose-built corpus
rather than further tuning on EngageNet.

#### 13.14.3 Full results, fixed decoder `refined_expected_thresholds`

| Run | Variable | Top-1 | Adj | MAE | macro-F1 | Δ vs video-only (top-1 / macro-F1) |
|---|---|---:|---:|---:|---:|---:|
| G04 seed 3 | balanced sampler | **66.67** | 91.31 | 0.440 | 52.37 | +1.99 / +1.01 |
| G04 seed 2 | balanced sampler | 66.36 | 91.09 | 0.444 | 51.13 | +0.80 / −0.61 |
| G00 | control (E04 config) | 66.09 | 91.40 | 0.444 | 52.25 | +1.24 / +0.05 |
| G12 | EMA 0.999 | 65.96 | 91.22 | 0.446 | 50.93 | +0.04 / −1.93 |
| G08 | loss = CE | 65.91 | 91.36 | 0.446 | 50.92 | +1.11 / −0.44 |
| G13 | no label smoothing | 65.91 | 91.44 | 0.445 | 51.01 | +0.66 / −0.81 |
| G09 | ordinal w = 0.60 | 65.87 | 91.36 | 0.446 | 50.94 | +0.71 / −0.47 |
| G01 | mvf 50 | 65.82 | 91.40 | 0.446 | 50.76 | +1.02 / −0.69 |
| G04 seed 1 | balanced sampler | 65.78 | **92.24** | **0.436** | **53.41** | +0.18 / +2.65 |
| G05 | lr 1e-4 cosine | 65.69 | 91.36 | 0.448 | 50.72 | +0.66 / −0.63 |
| G07 | ordinal w = 0.35 | 65.38 | 91.27 | 0.452 | 50.68 | +0.22 / −0.43 |
| G06 | SpecAugment | 65.34 | 91.22 | 0.453 | 50.59 | not ablated |
| G02 | mvf 40 + random crop | 64.89 | 91.53 | 0.450 | 50.42 | +0.27 / −0.59 |
| G03 | class-weighted loss | 64.72 | 91.71 | 0.452 | 50.41 | −0.18 / +1.74 |
| *E04 incumbent* | — | *66.36* | *90.96* | *0.458* | *52.03* | — |

**Isolated comparisons that are now clean but null.** CE vs ordinal loss (G08 65.91 vs G01 65.82),
ordinal weight 0.15/0.35/0.60 (65.82/65.38/65.87), label smoothing on/off (65.82/65.91), EMA
(65.96/65.82) — every one of these is inside the seed noise band. §13.12 marked the old loss and
class-balancing tables *confounded*; they are now **matched, and the answer is that none of these
variables has a measurable effect on this corpus.**

#### 13.14.4 Harness defect found and fixed during the sweep

`scripts/night/ensemble.py` completed all 14 members and every metric, then failed at
`json.dump` because `ordinal_metrics` and the threshold fits return numpy scalars. Fixed with
`default=float`. Worth recording only because the failure came *after* all compute was spent — the
result was recoverable by re-running, but a serialization guard belongs in any long inference job.

#### 13.14.5 F03 checkpoint ensembling is refuted — and it is not a dilution problem

§13.7 listed ensembling as "lever 3 — the cheapest reliable gain… typically returns +1–2 points", and
noted that since the gap to the best published baseline was only 1.25, "this alone could close it."
It does not. It costs about a point, and a second variant rules out the obvious excuse.

| Model | Top-1 (refined) | Best adjacent | MAE | macro-F1 |
|---|---:|---:|---:|---:|
| Best single model (G04 seed 3) | **66.67** | 91.31 | 0.440 | 52.37 |
| Best single adjacent (G04 seed 1, logit bias) | 65.78 | **92.55** | **0.436** | **53.41** |
| E04 incumbent | 66.36 | 90.96 | 0.458 | 52.03 |
| **F03 — 14-member ensemble** | 65.69 | 91.98 | 0.449 | 50.60 |
| **F03b — top-5 members only** | 65.60 | 92.07 | 0.447 | 51.32 |

**Dropping the nine weakest members changed nothing** (65.69 → 65.60, inside the 0.45 seed sd). So the
failure is not that weak members dragged the average down — **the members carry no decorrelated error to
average away.** All 14 warm-start from the same E04 checkpoint and diverge for only 6 epochs, so they are
near-copies of one model. Ensembling reduces variance only when members err independently; here there is
no independence to exploit.

This is the same conclusion the noise floor reached (§13.14.1) arriving by a second route: these
configurations are not meaningfully different models. **Both ensembles are worse than the best single
model on every one of the four metrics** — top-1, adjacent, MAE and macro-F1.

**Correction to an intermediate claim.** On first seeing the 14-member result it was noted that its
adjacent accuracy (91.98) was "the best of anything tonight". That is wrong: G04 seed 1 under logit-bias
decoding reaches **92.55**, above both ensembles. The ensembles do shift weight toward adjacent accuracy
relative to their own top-1, but they do not set the best adjacent figure.

**Consequence for the paper.** Of the four levers §13.7 identified — retraining on corrected splits,
train-split consistency, ensembling, and the two wasted settings — **none produced a measurable gain.**
Combined with §13.14.1, the remaining honest contributions are the ordinal evaluation and calibration
framing (where adjacent 92.55 and macro-F1 53.41 have no published EngageNet comparison), the negative
audio result, and the methodological finding that this corpus cannot resolve the differences prior work
in this repository claimed to measure.

#### 13.14.6 Label granularity: where the accuracy actually goes (2026-08-12)

Same checkpoint (G04 seed 3), same test set, argmax decoding — only the number of engagement levels
changes. Nothing is retrained.

| Levels | Accuracy | Majority baseline | **Gain over baseline** |
|---|---:|---:|---:|
| 4 (as trained) | 64.27 | 50.3 | +13.97 |
| 3 (merge the two middle levels) | 70.70 | 50.3 | **+20.40** |
| 2 (disengaged vs engaged) | 85.95 | 68.9 | +17.05 |

Per-class recall at 4 levels: class 0 **75.5%**, class 1 **20.3%**, class 2 **31.0%**, class 3 **81.6%**.
Error structure: exact 64.3%, **off-by-one 28.1%**, off-by-two-or-more 7.6%.

**The model separates the extremes and cannot separate the middle.** Nearly four fifths of all errors are
off-by-one, and they are concentrated in classes 1 and 2, which together are 29.5% of the test set and
are recognised at 20–31%. This is the same fact the 92% adjacent accuracy has been reporting all along,
stated in the form that matters for design.

**Merging the two middle levels is the single largest improvement found in this entire project** — +6.43
points, roughly seven times the measured seed noise (0.45 sd), and it costs no compute. It also gives the
largest gain over the majority baseline of any granularity, because it removes exactly the distinction
the data cannot support. The binary split scores higher in absolute terms (85.95) but its baseline is
also much higher, so it is less informative than the 3-level scheme.

**Direct consequence for the dataset design (§2, §11).** The planned corpus uses a **5-level** ordinal
scale. This evidence says the opposite direction is warranted: with 4 levels the two middle categories are
already not separable from audio-visual behaviour, and a 5-level scale subdivides precisely that
unresolvable middle. Before collection begins, either justify 5 levels against this result or re-scope to
3 levels plus the binary confusion flag. Annotator agreement on the middle levels should be measured in
the pilot, because if the model cannot separate them there is a real chance human raters cannot either —
in which case the labels, not the model, are the ceiling.

## 15. Collaborator branch evaluation — `feat/behavior-text-fusion` (2026-08-16)

First external contribution to the repo (author: Gakshith). Evaluated **without merging**, in a detached
git worktree at `/home/922933190/AVTCA-collab-test` on `f9cce71`. Six commits, +1595/-23 across 25 files,
branched from `development` at `3def305`.

### What the branch adds

| Feature | Flag | Mechanism |
|---|---|---|
| Numeric behavior modality | `--behavior` | 22 OpenFace channels (17 AU + 2 gaze + 3 pose), per-subject baseline subtraction, frozen `BehaviorFeatures` -> trainable `BehaviorEncoder`, cross-attention from the AV summary + a direct pooled-AU skip |
| Sentence text fusion | `--text_fusion` | `TextEncoder` (backends `hashing` / `minilm`) fused inside `it`, replacing the hashed late-text add-on |

Engineering quality is good: 283 tests pass (191 subtests, ~13 s), 9 new test files covering his own paths,
new dataset kwargs gated so RAVDESS/CREMAD loaders are unaffected, and the new flags registered in
`CONFIG_IDENTITY_KEYS` so checkpoint-identity checks stay honest.

### Matched A/B result (7 epochs, warm start from E04, one arm per 3090)

Both arms identical to `scripts/night/common.sh` except the treatment flag, `--max_video_frames 50`,
lr 5e-5, epochs 4-10.

| Epoch | A_control top1 | B_text_fusion top1 | B UAR |
|---|---|---|---|
| 4 | 67.13 | 53.13 | 25.0 |
| 5 | 67.04 | 53.13 | 25.0 |
| 6 | 67.60 | 53.13 | 25.0 |
| 7 | 67.41 | 55.00 | 28.79 |
| 8 | 67.23 | 60.22 | 39.39 |
| 9 | 67.41 | 61.62 | 42.23 |
| 10 | **67.69** | 62.47 | 43.94 |

`A_control` best 67.69 (UAR 56.04, adjacent 93.74) — above the 66.36 prior best but inside the +/-0.45
seed noise floor, so not a claimable gain. `B_text_fusion` never beat the 65.27 best score inherited from
the warm-start checkpoint and therefore **wrote no `_best.pth` at all**.

### Why B collapsed — three silent defects

**1. `classifier_fused` is randomly initialised and bypasses the warm-started head.** When `text_fusion` or
`behavior` is on, `forward_feature_3` returns `self.classifier_fused(...)` instead of `self.classifier_1(...)`.
13 tensors fail to warm-start, including the classifier itself:
`classifier_fused.{weight,bias}`, `textCrossAttention.*`, `text_av_proj.*`, `text_encoder.{proj,recency_emb,source_emb}`,
`text_missing`. At lr 5e-5 a from-scratch 4-way head cannot recover in 7 epochs — hence UAR pinned at exactly
25.0 (constant single-class prediction) for three epochs. Loss fell monotonically (1.245 -> 1.106) and UAR
climbed to 43.94, confirming a relearning head rather than a broken architecture.

**2. `--text_fusion` never reads the transcripts.** `_text_fusion(av_pair, behavior_feats, behavior_present)`
takes *behavior* features and calls `_captions_from_feats`, which captions the mean AU/gaze/pose vector with
`chat=""` hardcoded. The `text_tokens` / `text_mask` arguments carrying annotation column 5 are accepted and
ignored on this path. This is a design choice, not a bug, but it means text fusion is downstream of behavior
and cannot run without OpenFace.

**3. Behavior filename mismatch, failing silently.** `extract_behavior.py` names outputs from the video stem
(`subject_86_..._vid_0_4.npy`) while `ENGAGENET._behavior_for` derives the key from the annotation
`video_path`, which carries a `_facecroppad` suffix. Every lookup misses; a miss returns zeros with
`present=False` rather than raising. A full OpenFace extraction over ~11k clips would have been wasted before
anyone noticed.

The common thread: **nothing warns when the behavior stream is entirely absent.** A `present.mean() == 0`
assertion at dataset construction would surface all of this immediately.

### Feasibility of a real evaluation

OpenFace is **not installed** on this box, but the source videos **are** present: 11,311 `.mp4` across
Train/Validation/Test against 11,206 annotation rows. So extraction is possible — but fix defect 3 and add
the presence assertion first.

### Repo friction found along the way (ours, not his)

- `CLAUDE.md` documents `python -m src.main`; the real entry point is `main.py` at the repo root.
- `--save_every_epoch` is an **uncommitted local addition** to `src/config/opts.py`, absent from his branch.
- **`--n_epochs` is misleading on resume**: `opt.begin_epoch` is overwritten from the checkpoint and the loop
  is `range(begin_epoch, n_epochs+1)`. E04 is epoch 3 -> begin 4, so `--n_epochs 6` in `queue_gpu0.txt`
  trained **3** epochs, not 6. The G-sweep epoch counts in earlier notes are overstated.
- Adding parameters breaks `--resume_path`: SGD's saved param group no longer matches and
  `optimizer.load_state_dict` throws. Workaround used here: `warmstart_E04_noopt.pth`, an optimizer/scheduler-
  stripped copy, applied to **both** arms to keep the comparison matched.

### Open items from this evaluation

| # | Item | Status |
|---|---|---|
| C1 | Seed `classifier_fused` from `classifier_1` (copy trained weights into the first `e_dim*2` columns, zero the rest) and re-run B | open — cheap diagnostic, isolates how much of the gap is the head |
| C2 | Fix `_facecroppad` filename mismatch in `ENGAGENET._behavior_for` / `extract_behavior.py` | open — prerequisite for any behavior run |
| C3 | Add `present.mean() == 0` assertion so an absent behavior stream fails loudly | open |
| C4 | Install OpenFace, extract 11,311 clips, then evaluate `--behavior` and `--text_fusion` for real | open — only path to a genuine number |
| C5 | Decide whether text fusion should read transcripts or AU captions | **decided 2026-08-17: AU/behavior captions** — implemented in the main repo as text v2 (see §16); transcripts rejected because EngageNet clips are silent |
| C6 | Fix `--n_epochs` / `begin_epoch` semantics on resume, or document them | open — affects all past sweep records |

---

## 16. Text modality v2 — label-leakage fix and behavior captions (2026-08-17)

### 16.1 The defect

The v1 chat text (`preprocessing/engagenet/chat_text.py`) was generated by indexing hand-authored
phrase banks **by the ground-truth label** (`load_chat_bank(topic)[f"label_{label}"][bucket]`); even the
length bucket is label-conditioned. Audit (`audit_text_leakage.py`) on `annotations_engagement_a10.txt`:
4,480/11,206 rows with text, **792 unique strings, 99.84% label-deterministic, BoW logistic regression
text→label 98.1% val / 96.8% test**. A label oracle applied identically to all splits — every
`--late_text_fusion` run on a v1 file (V8/V9, `text_teacher_smoke`) is contaminated.

Why the oracle still *lowered* accuracy (V9 55.32 vs V7 62.90): legacy `LateTextFusion` wiring routes the
pooled AV vector through a randomly-initialized `av_context` Linear(256→128) before the classifier, so
enabling text destroys the warm-started AV representation at init. (Hash/padding collision ruled out.)
The old architecture.md explanation ("weakly label-correlated by construction") was factually wrong.

### 16.2 The fix

**Dataset**: `behavior_caption.py` + `compute_caption_stats.py` + `create_annotations.py --text_source
behavior` generate deterministic, label-free captions from the OpenFace `(300,22)` series in
`datasets/EngageNet/behavior/` (11 clip-level statistics → train-split-only tertiles →
phrase table; thresholds in `behavior_caption_stats.json`). Output: `annotations_engagement_v2.txt` and
`_v2_a10.txt` — columns 1–4 byte-identical to v1, 100% text coverage, **6,614 unique captions**, BoW
probe 61.3% val / 63.5% test vs 53.1/50.3 majority. The caption API takes no label anywhere. v1 files
untouched; physical backups in `preprocessing/engagenet/backup_2026-08-17/`.

**Model**: `LateTextFusionV2` (`--text_fusion_arch residual`; `--late_text_fusion` now defaults OFF).
Classifier input stays the untouched 256-d `cat(audio_pooled, video_pooled)`; text adds a
zero-initialized `Linear(128→256)` residual (embedding → biGRU → MHA, AV-projected query). Exact no-op at
init; E04 warm-starts with 546 restored / 0 skipped / 17 left-at-init (the zero text params).
Tests: `tests/test_text_fusion_v2.py` (bit-exact no-op, label-free API, full state-dict restore).
Calibration script text defaults fixed 48/8192 → 32/4096 to match training tokenization; `run_job.sh`
now calibrates each run at its own `annotation_path`/text flags read from the run's opts json.

### 16.3 Experiment in flight (queues `scripts/night/queue_textv2_gpu{0,1}.txt`)

| Arm | Runs | Config |
|---|---|---|
| T10 v2 control (AV-only) | seeds 1–3 | warm start E04, v2_a10 annotations, mvf 96, lr 5e-5 step, 8 epochs |
| T11 text residual | seeds 1–3 | same + `--late_text_fusion --text_fusion_arch residual` |

Decision rule (seed sd ≈ 0.45): mean(T11) − mean(T10) ≥ +0.9 → claimable gain; +0.4–0.9 → add 2 seeds;
≤ 0 → text stays interpretability-only and is reported honestly. Workers launched 2026-08-17 16:34 with
`GPU_FREE_MIB=19500`; they wait behind the collaborator S1/S2 sweeps currently on both GPUs. Collect via
`results/night/T1*/calibration/calibration_results.json`.

Honest-framing note for the paper: captions are derived from video (OpenFace), so the claim is a
structured-summary/longer-horizon gain, not an independent modality; the leakage audit is the
paper-facing before/after artifact.

---

## 17. Plan of action — late text fusion on real student Zoom chat (2026-08-17)

**Decision (2026-08-17):** the OpenFace behavior captions of Section 16 are **rejected as a text
modality** — they are computed from the video, so fusing them adds no information the model does not
already receive. `LateTextFusionV2` is kept as chat-ready architecture; the only text source we will
train on is **real per-student Zoom chat** from our own classroom collection. This section is the
end-to-end plan to get there.

### 17.1 Capture protocol (bakes into E1 session script — must be locked before Session 1)

1. **Zoom settings:** auto-save in-meeting chat ON (alongside the already-mandated per-participant
   audio). Zoom exports one `meeting_saved_chat.txt` per session: `HH:MM:SS From <display name>: <text>`.
2. **Identity mapping:** enforce a display-name convention at session start (`<student_id> - <first name>`)
   so chat lines map deterministically to the per-student audio/video tracks. Verified in the first
   5-minute calibration block.
3. **Elicitation:** chat is uselessly sparse unless prompted. Each of the 5 blocks includes ≥2 scripted
   check-for-understanding prompts answered **in chat** ("type your answer in the chat"), plus one
   open prompt per block. This is the text-modality analogue of the audio-parity design (Section 10):
   we engineer density instead of accepting a mostly-empty channel.
4. **Exclusion:** the self-report survey messages (minutes 53–56, Likert ratings) are labels-adjacent
   and are **excluded from training text**, exactly as those clips are already excluded from training.

### 17.2 Preprocessing — `preprocessing/zoom/extract_chat.py` (new, TX2)

- Parse `meeting_saved_chat.txt` → per-student, per-second message stream; drop instructor lines into a
  separate context channel.
- **Per-clip text assembly:** messages inside the clip's 10 s window, plus a trailing context window
  (default 120 s) because typing lags the stimulus. Store: raw text, `text_present` flag,
  `seconds_since_last_message`, and `latency_to_last_instructor_prompt`.
- Extend the HDF5 schema (Section "Dataset File Format") with a variable-length UTF-8 `chat_text`
  dataset per clip and the three scalar features; add the columns to `manifest.csv`.

### 17.3 Leakage and contamination gates (reuse `audit_text_leakage.py`)

- **Gate A — annotation blindness:** annotators never see the chat pane while labeling; labels must be
  behavioral only. (Otherwise chat→label correlation is annotator-induced, the v1 failure in disguise.)
- **Gate B — BoW probe:** before any training run, the Section-16 audit runs on the chat column.
  Hard fail if a bag-of-words logistic probe predicts labels anywhere near oracle levels
  (v1 was 98% — a genuine signal should be far weaker and must generalize across sessions, not memorize strings).

### 17.4 Model side (mostly done)

- `LateTextFusionV2` (zero-init residual, `--late_text_fusion --text_fusion_arch residual`) is built and
  tested. Remaining choices, deferred until real chat exists:
  - **TX4 — text encoder:** frozen sentence-embedding model (e.g. MiniLM) vs the current lightweight
    embedding. Decide on pilot data; frozen encoder favored at our dataset size.
  - **TX5 — empty-chat handling:** `text_present=0` clips contribute a learned null token; the zero-init
    residual already makes "no chat" a safe no-op at init.
  - **TX6 — modality dropout** extended to the text stream (p=0.15), matching the audio/video rule.

### 17.5 Evaluation protocol (fixed now, so results are pre-registered)

1. **Text-only probe:** chat features alone vs the majority predictor — establishes the channel carries
   signal at all (mirrors the audio-parity floor of Section 10).
2. **Matched A/B:** AV control vs AV+chat, identical warm start and config, ≥3 seeds per arm.
   Decision rule as in Section 16.3: mean gain ≥ +0.9 top-1 (2× seed sd) → claimable; +0.4–0.9 → add
   seeds; ≤ 0 → report honestly.
3. Report per-arm mean ± sd on the held-out session split (never same-session clips across splits).

### 17.6 What can be done before classroom data exists

- **TX1 (now):** write `extract_chat.py` and test it on a mock Zoom call among ourselves — one 15-minute
  call with scripted chat produces a real `meeting_saved_chat.txt` to develop against. No model training.
- Everything else (TX3–TX7) is blocked on pilot Session 1 (task E14).

### 17.7 Task list

| ID | Task | Depends on | Status |
|---|---|---|---|
| TX1 | `preprocessing/zoom/extract_chat.py` + mock-call fixture test | — | open (can start now) |
| TX2 | HDF5/manifest schema extension for chat text + 3 scalar features | TX1 | open |
| TX3 | Chat elicitation prompts written into the E1 session script; display-name convention | E1 | open |
| TX4 | Text encoder choice (frozen MiniLM vs learned embedding) on pilot data | E14 | open |
| TX5 | Null-token handling for `text_present=0` clips | TX4 | open |
| TX6 | Modality dropout on text stream (p=0.15) | TX4 | open |
| TX7 | Leakage Gate B run + text-only probe + matched A/B on cohort data | Sessions 3+ | open |

## 16. Behavior modality evaluated with real OpenFace features (2026-08-18)

Continuation of §15. The collaborator branch was untestable there because no OpenFace features existed.
They now do. Everything below ran in the worktree `/home/922933190/AVTCA-collab-test`; **still unmerged**.

### 16.1 OpenFace built from source (no root)

`TadasBaltrusaitis/OpenFace` -> `/home/922933190/openface_build/build/bin/FeatureExtraction`, conda env
`ofbuild`. Two dependency failures, both worth recording:

- **OpenBLAS**: `cmake/modules/FindOpenBLAS.cmake` searches a hardcoded path list with `NO_DEFAULT_PATH`
  (never sees the conda prefix) and keys on `f77blas.h`, which conda's openblas does not ship. Fix: point
  `-DOpenBLAS_INCLUDE_DIR` at OpenFace's own vendored `lib/3rdParty/OpenBLAS/include` (it has `f77blas.h`)
  and `-DOpenBLAS_LIB` at `$CONDA_PREFIX/lib/libopenblas.so`.
- **dlib**: conda-forge `dlib` is Python-only and installs **no** C++ headers or library. The C++ package
  is `dlib-cpp`; pinned **19.24.6** (not 20.x) since OpenFace targets 19.13.
- Patch-expert models: the Dropbox URLs in `download_models.sh` still work; the OneDrive mirrors are dead (403).

**OpenFace is CPU-only.** No CUDA option in CMake, zero GPU references in the C++ sources, OpenBLAS is the
only math backend. Not a build flag we missed — the code does not exist. GPU alternatives (LibreFace,
py-feat) drop AU45/blink, AU23 intensity and head roll, which §11 boredom/confusion scoring depends on.

### 16.2 Extraction

`preprocessing/engagenet/extract_behavior_parallel.py` (new, in the worktree): 16 concurrent OpenFace
processes, resumable, atomic writes, per-clip error log. ~2 h instead of ~30 h serial.

| | |
|---|---|
| Clips extracted | **11,311 / 11,311, 0 errors** |
| Annotation rows resolved | **11,206 / 11,206 (100%)** |
| Of those, face detected | **11,069 (98.78%)** |
| Frames/clip | min 29, median 300, max 10,000 |

Output: `datasets/EngageNet/behavior/*.npy`, `(T, 22)` float32. This is data, not code — reusable
regardless of what happens to the PR.

### 16.3 Fixes applied to the collaborator's code (worktree only)

1. **Filename mismatch (§15 defect 3)** — `ENGAGENET._behavior_for` now strips `_facecroppad` /
   `_croppad` / `_facecrop` before lookup. Without this, coverage is 0% and fails silently.
2. **`_assert_behavior_present`** — samples 200 clips at dataset construction and raises if none resolve,
   warns on partial. This is the guard that makes the whole class of failure visible.
3. **`scripts/calibrate_engagement_logits.py`** — its forward loop already handled `behavior_feats`, but
   the CLI never exposed `--behavior` / `--text_fusion` / `--behavior_dir`, so a behavior checkpoint could
   not be calibrated (model built without the modules -> state-dict mismatch). Flags added.

### 16.4 Subject IDs and leakage-free baselines

Annotations carry no subject column; subject IDs are recoverable from filenames
(`subject_\d+_[a-z0-9]+`): **133 subjects, all 11,206 rows, splits fully subject-disjoint**.

**His `compute_baselines` has a label-leakage problem.** It defines a subject's neutral reference as the
mean over their **label-0** clips. Splits are subject-disjoint, so baselining a *test* subject requires
knowing which of that subject's clips are label 0 — i.e. reading test labels at inference. Replaced with a
**label-agnostic per-subject mean** (`baselines_subject.json`, 133 subjects). Between-subject std on AU04
is 0.511, so the normalisation is not trivially redundant — but see 16.6: it did not help.

### 16.5 From-scratch A/B (15 epochs, matched)

First real test of the contribution.

| | A (audio+video) | B (+behavior+text) | Δ |
|---|---|---|---|
| Best top1 | 56.12 | **57.14** | +1.03 |
| Mean top1, ep 5–15 | 52.25 | **54.01** | **+1.76** |
| Mean UAR, ep 5–15 | 53.31 | **55.10** | **+1.79** |
| Epochs won (ep 5–15) | 2 | **9** | — |

B wins 9 of 11 settled epochs. Validation, single seed.

### 16.6 Hyperparameter sweep (staged, 10 runs)

Ranked by **mean of top-3 val epochs** (single best is noise-dominated: val swings ~8 points between
adjacent epochs).

| Config | Score |
|---|---|
| **S5 lr 3e-3 + cosine** | **60.10** |
| S2 cosine lr 1e-3 | 58.29 |
| S7 heads 4 | 57.89 |
| S8 per-subject baselines | 57.24 |
| S1 step lr 1e-3 | 57.05 |
| S3 cosine + EMA | 55.34 |
| S6 lr 5e-4 | 54.56 |
| S4 cosine + EMA + grad clip | 46.93 |

Findings: **learning rate dominates** (3e-3 best, 5e-4 costs -5.5); cosine beats step (+1.24);
**EMA hurts** (-2.95) and EMA+clipping is catastrophic (-11.4) — with val swinging 8 points, weight
averaging blends genuinely different models; 4 heads worse than 8; **per-subject baselines did not help**
(-1.05), contrary to the §11 design assumption.

Long from-scratch run at the winning config (60 epochs): **B best 61.25 (ep14), A best 58.64 (ep45)**.
**B overfits after ~epoch 14** and ends *below* A over the last 10 epochs — the extra capacity memorises.
B needs early stopping; A does not.

### 16.7 Warm-start A/B with a zero-initialised classifier (the §15 C1 fix, done properly)

Instead of patching the model, **checkpoint surgery**: `warmstart_E04_seeded.pth` sets
`classifier_fused.weight[:, :256]` = the trained `classifier_1.0.weight` and **zeros the 320 new
behavior/text columns**; bias copied. The new encoders stay at random init but feed only zeroed columns.

**Verified: `max abs logit difference = 0.000e+00`** against the AV-only model on identical inputs — the
seeded model is bit-identical to E04 at step 0. Load report: 548 restored / 0 skipped (vs 546 / 0 for the
control; the 2 extra are the seeded classifier). This is the same principle as the text-v2 zero-init
residual and is the correct general fix for adding a modality to a warm-started model.

Both arms, warm start from E04, mvf 96, lr 5e-5, 8 epochs:

| | WA control | WB behavior+text |
|---|---|---|
| Best **val** top1 | 67.6937 | 67.6937 (identical) |
| Mean val top1 | 67.28 | 67.49 (+0.21) |

**Calibrated TEST metrics** (thresholds fit on validation, applied to test):

| Decoder | Arm | Test top1 | Adjacent | Macro-F1 | MAE |
|---|---|---|---|---|---|
| argmax | A | 64.94 | 91.93 | 51.98 | 0.452 |
| argmax | B | 64.81 | 91.76 | 51.70 | 0.455 |
| logit_bias | A | 65.16 | 91.49 | 50.56 | 0.455 |
| logit_bias | B | **65.25** | 91.22 | 51.07 | 0.457 |
| expected_thr | A | **65.47** | 91.36 | 50.31 | 0.450 |
| expected_thr | B | 64.10 | **92.64** | **53.49** | **0.446** |
| refined_thr | A | **65.47** | 91.27 | 50.52 | 0.451 |
| refined_thr | B | 64.49 | 92.38 | 52.97 | 0.448 |

### 16.8 Conclusion

**Behavior+text helps from scratch and is redundant under warm start — on top-1.**

- From scratch: **+1.76 to +2.61**. The model has not learned to read facial behaviour from pixels, so
  explicit AUs add real information.
- Warm-started: **+0.21 val, -0.22 test on top-1**. The trained model already extracts this from pixels.
- **But B is consistently better on the ordinal/minority metrics**: +3.18 macro-F1 and +1.28 adjacent on
  the threshold decoders, with lower MAE. B's threshold decoder beats the 66.36 headline's own
  90.96 adjacent / 52.03 macro-F1 (92.64 / 53.49) while scoring 2.3 lower on top-1.

Since §13.14.6 and the memory notes both identify **macro-F1 52 vs top-1 66 (minority-class separation)**
as the largest quality gap, the behavior features are attacking the documented weakness — just not the
metric being tracked as the headline.

### 16.9 Caveats — read before using any number above

1. **Frame-cap mismatch.** These runs used `--max_video_frames 96`. The 66.36 headline was established on
   **50-frame** data, and memory.md explicitly flags 96 as wasting 46 padding frames per clip. The A
   control's 65.47 vs 66.36 may be this, not a failure to reproduce. Yesterday's warm control at mvf 50
   reached 67.69 val vs today's 67.69 at mvf 96 — untested on test. **A matched mvf 50 re-run is the
   single highest-value outstanding item.**
2. **Single seed everywhere.** Seed sd is 0.45; the A-control shortfall of 0.89 is ~2 sd — suggestive,
   not conclusive. Nothing here has been seed-replicated.
3. **Val-test gap ~3 points** (68.53 val -> 65.16 test), consistent with epoch selection fitting validation.
4. **Behavior and text were never ablated apart.** All B arms ran both flags.

### 16.10 §15 item C5 resolved — but NOT in his favour

§15 listed "decide whether text fusion should read transcripts or AU captions" as an open design question.
It is now resolved, and the answer is **neither**:

- **Not the v1 chat text** — it is label leakage (99.84% label-deterministic, BoW probe 97-98%). His code
  ignoring that column is correct.
- **Not AU captions either.** Yuvraj **rejected** the behavior-caption-as-text approach on 2026-08-17:
  captions are computed from the video, so fusing them adds no information the model does not already
  have. **Not a valid text modality — do not present `--text_fusion` as "text" in any report.**
  See [[project-text-v2-leakage-fix]].

**Consequence for how his contribution is described:** `--text_fusion` is a second view of the *same*
OpenFace features, not a text modality. It should be reported as part of the behavior contribution.
The 16.5–16.7 results are consistent with this: under warm start the combined arm added only +0.21 val,
and the from-scratch gain has never been attributed between `--behavior` and `--text_fusion` (C9). The
most likely reading is that the **numeric AU features do the work and the captions add little** — C9
would confirm it, and until it is run no claim should attribute any gain to the caption stream.

Real text fusion waits for genuine Zoom chat; `LateTextFusionV2` is the chat-ready architecture, and a
label-blind generated-chat proxy (`annotations_engagement_v3_a10.txt`) exists for interim testing.

### 16.11 Open items

| # | Item | Status |
|---|---|---|
| C1 | zero-init `classifier_fused` | **done** (16.7), verified exact no-op |
| C2 | `_facecroppad` filename fix | **done** (16.3) |
| C3 | loud failure on absent modality | **done** (16.3) |
| C4 | OpenFace install + extraction | **done** (16.1–16.2) |
| C5 | transcripts vs AU captions | **withdrawn** (16.10) |
| C6 | `--n_epochs` / `begin_epoch` resume semantics | open |
| C7 | **Re-run warm A/B at mvf 50** to match 66.36 conditions | **done 2026-09-12** — §19.5: B 66.08 ± 0.59 vs A 66.27 ± 0.45, no effect on any metric |
| C8 | Seed-replicate A control to confirm 66.36 reproduces | **done 2026-09-12** — 66.27 ± 0.45 (65.78 / 66.36 / 66.67), bit-identical to G04 |
| C9 | Ablate `--behavior` alone vs `--text_fusion` alone | **done 2026-09-12** — §19.5: B 66.08 ± 0.59, C 66.12 ± 0.59, A 66.27 ± 0.45 — neither stream moves any metric |
| C10 | Report filename fix, presence guard, calibration flags and baseline leakage to the author | open |

## 18. Paper plan after the 2026-09-08 supervisor meeting (Sanchita Ghose)

**Status: plan only — nothing below is implemented.** Next review: **Tuesday 2026-09-15, 7 PM** (Sanchita sends the invite).
Owners: Yuvraj = results/evaluation section + related-work citations + architecture diagram; Akshit = methodology section.

### 18.1 Correct the number reported in the meeting before it reaches the paper

The meeting summary records "top-1 improved from 66.36% to 65.47% by adding the behavior stream". Repo record
(`docs/behavior_modality_summary.md`, §16.7) says otherwise:

| Arm | Test top-1 | Macro-F1 | Adjacent | MAE |
|---|---:|---:|---:|---:|
| Audio+Video (warm A/B control, mvf differs from 66.36 run) | **65.47** | 50.31 | 91.36 | 0.450 |
| + Behavior (OpenFace AUs) | 65.25 | **53.49** | **92.64** | **0.446** |
| 66.36 baseline (different frame cap) | 66.36 | 52.03 | 90.96 | 0.444 |

So: 65.47 is the *control*, the behavior arm is −0.22 on top-1 (inside seed sd 0.45), and its gain is macro-F1 +3.18 /
adjacent +1.28. The paper claim is therefore "behavior stream improves minority-class and ordinal metrics, not top-1".
Yuvraj's own explanation in the meeting (more epochs) is the C7 confound — the matched re-run at mvf 50 (§16.11 C7) is
the single experiment that must finish before the results table is written.

### 18.2 Paper structure (journal target, IEEE template at `papers/research/paper.tex`)

Order agreed: Introduction → Related Work → **Preliminaries (~1 page, journal only)** → Methodology → Experimental
Results / Model Evaluation → Conclusion. Follow the format of Sanchita's previous paper (`papers/research/prof papeer.pdf`).

| Section | Owner | What changes vs current draft | Done when |
|---|---|---|---|
| Introduction | Yuvraj | Add a **conceptual diagram** (problem framing, not the architecture). Restate contributions as bullet points, mirroring `research_contributions.md` §1–§6 but respecting the claim limits in memory (no SOTA, no "proved audio helps" without E22, decoder always stated). | Diagram + bullet list in tex |
| Related Work | Yuvraj | Grow from 9 to **15–20 citations**. Mine the survey papers already in `papers/` (`qarbal2025review`, `s41019-025-00335-5.pdf`, `applsci-14-01190-v2.pdf`) and add recent multimodal audio-visual learning work (2023–2026). Keep the four subsections; add citations under "Multimodal Audio-Visual Learning" first. | ≥15 `\cite` keys, all in the bib |
| Preliminaries | Yuvraj | New ~1 page: ordinal engagement labels, decoder definitions (argmax / bias / expected / refined expected), the four metrics (top-1, adjacent, MAE, macro-F1) and why each matters for an ordinal task. | 1 page |
| Methodology | Akshit | Group the many blocks into **major components**: (1) audio encoder, (2) video encoder + OpenFace behavior stream, (3) temporal alignment + two-stage cross-attention fusion, (4) ordinal head + calibration. One paragraph per component, one equation where needed. Must describe Akshit's visual-emphasis variant honestly as a design choice, not assume it (see 18.4). | Draft with the diagram from 18.3 |
| Model Evaluation | Yuvraj | Sanchita's structure: datasets (EngageNet primary; DAiSEE and RAVDESS as secondary), training parameters, then results **grouped by metric** with a one-line explanation of what each metric measures and why it is significant; then ablation analysis (modality ablation on identical weights, decoder ablation, behavior-stream ablation); then discussion of the hard categories (the middle levels) and *why* the model fails there; then limitations/clarifications for reviewers. Human-survey subsection is N/A until classroom data exists — say so explicitly. Keep it concise. | Tables + text for the Tuesday review |

### 18.3 Architecture diagram — technical, hand-drawn in draw.io

- Produce in draw.io (or equivalent), **not** AI-generated. Export SVG + PDF into `papers/research/figures/`.
- Content must match `docs/architecture.md` Architecture 2 and the code: audio `(B,64,T)` → AudioCNNPool → `_adaptive_align_audio_to_video` → early cross-attention (av1/va1) → stage-2 conv → self-attention → final cross-attention → maxpool → concat (+ behavior tokens where the branch merges) → ordinal head → calibrated decoder.
- Mark tensor shapes at every boundary; mark the modality-dropout and alignment points, since those are two of the six contributions.
- Also upload it to the shared folder (Sanchita could not find one there).

### 18.4 Coordination with Akshit (call tonight, 2026-09-08)

1. Get the exact description of his "visual-emphasis" variant (what weighting, where in the fusion) and whether it has a measured result. Without a matched number it is a methodology *option*, not a paper result.
2. Reconcile with the project's hard constraint that audio and video contribute equally on the future classroom dataset (CLAUDE.md decision 1). Resolution for the paper: EngageNet is video-dominant by construction (most clips silent), so a visual-weighted variant is defensible **on EngageNet only**, and the paper should say the equal-contribution design is reserved for the speech-rich classroom data.
3. Agree section boundaries: he writes Methodology; Yuvraj supplies the component list in 18.2 and the diagram in 18.3 so terminology matches.

### 18.5 New data — psychology professor's asynchronous class recordings

Sanchita meets the professor 2026-09-09. What we need from that meeting, in order of importance:

1. **Breakout-room recordings**, with "Record each participant separately" ON so per-student audio exists (plan §10/§11 — mixed gallery audio is unusable).
2. **Zoom chat export** per session (plan §17 — the text modality is waiting on real chat; no substitute is acceptable).
3. Consent language covering audio, video and text; per-student IDs stable across sessions.
4. Session count and cadence — Session 1 of each cohort is Hawthorne-biased and is for pipeline debugging only (CLAUDE.md decision 5).

### 18.6 Retraining the behavior model on personalized recordings — feasibility

Question raised in the meeting: can the OpenFace behavior stream be retrained on the asynchronous-class recordings?
Answer to give Tuesday, with the reasoning:

- **Technically yes**: OpenFace extraction is already built (§16.1–16.2, CPU-only, ~real-time). Per-student baseline
  calibration (CLAUDE.md decision 4) is exactly what "personalized" means here — AU/EAR/head-pose relative to each
  student's first 5 minutes — and it is already in the plan (§11), not yet coded.
- **Blocked on labels**: retraining needs engagement labels on those recordings. Options, cheapest first: (a) fine-tune
  the EngageNet-trained model and evaluate only qualitatively until labels exist; (b) instructor/TA post-hoc labelling
  with the §2 anchors; (c) self-report per block. Decide after we know the professor's session count.
- **Minimum viable**: 2 sessions × ≥8 students × breakout rooms ≈ a few hundred clips — enough for calibration and a
  per-student sanity check, not enough for a new headline number.

### 18.7 Task list

| # | Task | Owner | Due | Depends on |
|---|---|---|---|---|
| P1 | Finish C7 matched re-run (behavior vs AV at mvf 50) so the results table is confound-free | Yuvraj | before P5 | **running 2026-09-12 (§19)** |
| P2 | Correct the 65.47/65.25 framing with Sanchita (email or Tuesday) | Yuvraj | 2026-09-15 | — |
| P3 | Related work → 15–20 citations, bib entries verified | Yuvraj | **done** (21 entries; verify `dan2026multimodal` authors and `li2025mersurvey` volume) | — |
| P4 | Preliminaries section | Yuvraj | **done** (Yuvraj wrote it; merged 2026-09-08 as §III with eq labels; Method/Eval now cite its equations instead of re-deriving) | — |
| P5 | Model Evaluation section: metric-grouped tables + ablation + hard-category discussion | Yuvraj | 2026-09-15 | P1 |
| P6 | draw.io architecture diagram, SVG/PDF in `papers/research/figures/`, copy to shared folder | Yuvraj | 2026-09-15 | — |
| P7 | Call Akshit; capture his variant + agree section split | Yuvraj | 2026-09-08 | — |
| P8 | Methodology section grouped into 4 components | Akshit | 2026-09-15 | P6, P7 |
| P9 | Conceptual diagram for the Introduction | Yuvraj | 2026-09-15 | — |
| P10 | Feasibility note on personalized retraining (18.6) ready to present | Yuvraj | 2026-09-15 | Sanchita's 09-09 meeting outcome |
| P11 | Data-collection asks list (18.5) sent to Sanchita before her 09-09 meeting | Yuvraj | 2026-09-09 AM | — |

### 18.8 Draft written 2026-09-08 (`papers/research/paper.tex`)

Sections III–V now exist, laid out exactly like Ghose & Prevost, *FoleyGAN*, IEEE TMM 2023:
III Proposed Research Method (A audio encoder, B visual encoder + behavior stream, C alignment + two-stage
cross-attention + modality dropout, D ordinal loss + calibrated decoding); IV Experimental Details (A dataset,
B protocols); V Model Evaluation (A–D one subsection per metric with its definition and why it matters,
E quantitative analysis vs the six ICMI-2023 baselines + decoder table, F ablation: modality / alignment /
behavior / 14-run sweep with noise floor, G error analysis on the middle levels + label-granularity table,
H human-evaluation protocol). Eight tables, five equations. Three bib entries added (EfficientFace, OpenFace,
RAVDESS) → 12 total; P3 still needs 15–20.

Decisions taken while drafting: the preprocessing-defect history is **not** in the paper (engineering
error, not a model property); the 3.6 s audio window is never mentioned; Akshit's visual-emphasis variant is
a commented-out subsection until he supplies a number. `\ref{fig_arch}` is unresolved until P6 lands.
TODO comments in the tex mark C7 (behavior row at mvf 96 vs 66.36 at mvf 50), C9, and the survey table.
No LaTeX toolchain on this machine — the file passed a static environment/ref/cite check only.

### 18.9 Master merge (2026-09-08, later)

Yuvraj's Overleaf version (which already had a Preliminaries section with 13 equations) is now the master
`papers/research/paper.tex`. Merged into it: expanded Related Work (21 cites), §IV Method, §V Experimental
Details, §VI Model Evaluation, full bibliography. Duplicate equations (MHA, loss, expected level, MAE) were
removed from §IV/§VI and replaced by `\ref` to the Preliminaries labels `eq:xattn_a/v`, `eq:argmax`, `eq:mu`,
`eq:thresh`, `eq:loss`. All hard-coded "Section III-D" style references replaced with `\ref`. Template
boilerplate comments stripped; `graphicx` enabled with `\graphicspath{{figures/}}`. My earlier standalone
draft is kept as `paper_claude_draft_2026-09-08.tex`. Static check: 8 tables, 9 equations, 2 align blocks, 21
bib entries all cited, only `fig_arch` unresolved (P6).

### 18.10 Paper-vs-code check on the behavior stream (2026-09-08)

Checked §IV-B/C of the paper against `AVTCA-collab-test/models/multimodal_cnn.py::_behavior_fusion` and
`behavior_features.py`. Four statements were wrong and are now fixed in the tex:

| Paper said | Code does |
|---|---|
| AU channels normalised by subject mean | absolute features; baseline subtraction tested in S8 and cost −1.05, so the reported run is un-normalised |
| encoder summary $h_b$ added via skip | skip = Linear(22→64)+ReLU on the **raw** clip-mean descriptor; the GRU summary is unused |
| behavior "refines" the AV summary | `[av_pair(256), ctx(128), skip(64)]` are **concatenated** → `classifier_fused(448→4)` |
| streams max-pooled over time | `AttentionPool` (learned, masked) in `it_fusion_mode='modern'`, which is what the 66.36 lineage uses |

Note for CLAUDE.md: its architecture summary still says "MaxPool each modality → concat → Linear(256,…)";
that describes the legacy path only. `docs/architecture.md` is already correct (attention pooling).

## 19. Full-clip audit and matched A/B/C at the video clock (2026-09-12)

**Request (Yuvraj):** stop using "ten seconds of video and ten seconds of audio linked separately"; use the
complete video and the complete audio of every clip, mapped onto each other from 0 s to the end, then train
(a) the EfficientFace audio-visual model and (b) the behavior-stream model, and report an absolute answer.

### 19.1 What "10 s" actually is on EngageNet — measured, not assumed

Every one of the 11,311 source `.mp4` files was probed with OpenCV/ffprobe and compared against the stored
face arrays and wavs (`scratchpad/duration_audit.json`):

| Quantity | Value |
|---|---|
| Source duration min / median / max | 1.03 s / **10.00 s** / **10.06 s** |
| Clips longer than 10.5 s | **0** |
| Clips shorter than 9.5 s | 249 (source-limited; the corpus ships them short) |
| Source fps (top values) | 30 (7,644) · 1000 (1,038, 10,000 frames) · 15 (325) · 29.97 (238) · 10 (165) |
| Stored `*_facecroppad.npy` frames | **50** for 10,306 clips; 41–63 for the rest (fps rounding), all at 5 fps over the whole clip |
| Stored `*_croppad10s.wav` duration | median **10.005 s**; 10,987 clips at 10.0 s; only 6 clips >0.25 s shorter than their video, and in every one the *source audio stream itself* is short (e.g. `subject_54_…_vid_1_1`: audio stream 3.38 s inside a 10.01 s video) |

**Conclusion: the 10 s window *is* the whole clip.** The `_croppad10s` suffix is a name, not a cap —
`extract_audios_full.py` runs ffmpeg with no `-t`, and `extract_faces.py --target_fps 5` walks every frame
to the end. There is nothing after 10 s to take. The only thing that ever *was* capped was the legacy
3.6 s audio (fixed 2026-08-07, §12.1) and the 15-frame test/val video (fixed 2026-08-07, §13).

Mapping is already synced end to end for audio and video: the loader keeps all ~431 mel frames
(`--max_audio_steps 0`) and all 50 face frames (`--max_video_frames 50`), and
`_adaptive_align_audio_to_video` average-pools the full valid audio span onto the valid video tokens
per sample, so audio token *i* covers exactly the same time window as video frame *i*. Verified from the
first training batch: `audio=(8, 64, 431)  visual=(8, 50, 3, 224, 224)`.

### 19.2 The one stream that was NOT on the full-clip clock: behavior

The collaborator branch's `BehaviorFeatures(num_frames=15)` resamples each clip's OpenFace `(T, 22)` series
to **15 steps** — the legacy RAVDESS frame count — while the video it was extracted from is fed at 50 frames.
It still spans 0→end (endpoint-preserving linspace) but at **1.5 fps against 5 fps video**: consecutive AU
samples are 667 ms apart. Every §16 behavior number was produced this way.

**Fix (worktree, `--behavior_frames`):** resample to `--max_video_frames` by default (50 here; 15 remains
reachable explicitly to reproduce §16). Alignment check against the frames `extract_faces.py` actually kept:

| Source fps / frames | Behavior step vs kept video frame, max offset | Old 15-step spacing |
|---|---|---|
| 30 / 300 | 5 src frames = 167 ms | 667 ms |
| 15 / 150 | 2 = 133 ms | 667 ms |
| 10 / 100 | 1 = 100 ms | 667 ms |
| 1000 / 10,000 | 199 = 199 ms | 667 ms |

So behavior step *i* now sits within one 5 fps interval of video frame *i* for every fps in the corpus.
Plumbed through `src/config/opts.py`, `src/data/dataset.py::resolve_behavior_frames`, the calibration
script and `CONFIG_IDENTITY_KEYS`; unit-tested (`tests/test_behavior_frames_option.py`, 40 behavior tests
pass). The behavior encoder (Conv1d + BiGRU + mean) is length-agnostic, so no architecture change.

### 19.3 Experiment — three arms, three seeds, one config, one code tree

All arms run from the collab worktree (the only tree with the behavior branch) so the comparison is
code-matched; results are written to the main repo at `results/fullclip/`. Scripts:
`AVTCA-collab-test/scripts/fullclip/{common.sh,run_job.sh,worker.sh,queue_gpu*.txt,seed_warmstart.py,collect.py}`.

| Arm | Model | Warm start (`model.pth`) | Extra flags |
|---|---|---|---|
| **A** | audio + video, EfficientFace backbone | E04 best (66.36) | — |
| **B** | A + OpenFace behavior stream | E04 with `classifier_fused` (4×448) seeded: AV columns copied, 192 new columns zeroed | `--behavior --behavior_frames 50` |
| **C** | A + behavior + behavior-caption "text" (plan §16.10: a second view of the AUs, not a text modality) | E04 seeded, 4×576, 320 columns zeroed | `--behavior --text_fusion --behavior_frames 50` |

Shared config = the G04 winner of the 2026-08-12 sweep, held fixed: `mvf 50, uniform, lr 5e-5 step,
6 epochs, sqrt-inverse balanced sampler, ordinal 0.15, label smoothing 0.1, bs 8, SGD, 8 heads, mel,
nodropout, full_video_preprocessing, max_audio_steps 0`. Seeds 1/2/3 per arm; **one job per GPU** — a
mvf-50 run holds ~12.9 GB of a 24 GB 3090, so two per card would OOM (the night sweep's 2-per-GPU gate
only ever admitted a second job once the first had finished). Queue order: A/B seeds first, C after.

`seed_warmstart.py` generalises the §16.7 surgery to any branch combination and **verified on real
validation batches that both seeded models produce logits identical to the AV model (max abs diff 0.0)**,
so B and C start exactly where A starts and can only diverge through gradients. Load reports: A 546
restored / 0 left at init; B 548 restored / 24 left at init (behavior encoder); C 548 / 35.

Calibration (thresholds fit on validation, applied to test) reads `max_video_frames`, `frame_sampling`,
`behavior`, `text_fusion`, `behavior_dir`, `behavior_frames` and `annotation_path` back out of each
run's own `opts*.json` — never from a shared default (memory.md, "shared-config defaults silently
override per-run settings").

### 19.4 Pre-committed reading rules (written before any result existed)

1. **Fixed decoder for every comparison: `refined_expected_thresholds`** (the 66.36 decoder). All four
   decoders are printed by `collect.py`; none is selected per arm.
2. **Seed noise on this corpus is sd 0.45 top-1 / ~2.3 macro-F1** (memory.md). A between-arm difference
   below ~1.0 top-1 or ~2.3 macro-F1 on 3-seed means is *not* an effect; report it as "within noise".
3. **top-1 alone is misleading at 50% class imbalance** — report adjacent, MAE and macro-F1 alongside.
4. A also answers C8 (does the 66.36 lineage reproduce at mvf 50 under valid selection?). B vs A answers
   C7. C vs B answers C9.

### 19.5 Results (test set, 2,256 clips, decoder fixed to `refined_expected_thresholds`, thresholds fit on validation)

**Arm A — audio + video (EfficientFace), the "full video + full audio" run: complete.**

| Run | Best val top-1 (epoch) | Test top-1 | Adjacent | MAE | Macro-F1 | Per-class acc 0/1/2/3 |
|---|---:|---:|---:|---:|---:|---|
| `A_av_s1` | 67.97 (ep 3) | 65.78 | 92.24 | 0.436 | 53.41 | 74.2 / 22.4 / 27.7 / 85.9 |
| `A_av_s2` | 67.88 (ep 4) | 66.36 | 91.09 | 0.444 | 51.13 | 79.2 / 12.2 / 26.7 / 87.6 |
| `A_av_s3` | 67.88 (ep 2) | 66.67 | 91.31 | 0.440 | 52.37 | 77.0 / 18.7 / 23.4 / 88.9 |
| **mean ± sd** | | **66.27 ± 0.45** | 91.55 | 0.440 | 52.31 ± 1.14 | |

Every epoch of every seed reproduces the 2026-08-12 G04 runs to the hundredth (`G04_*` val logs and
test numbers are identical), which (i) proves the collab worktree's AV path is bit-for-bit the main
tree's, and (ii) is the direct answer to the request: **the audio-visual model was already trained on the
full 10 s clip, so "use the complete video and audio" changes nothing for it** — 66.27 ± 0.45 is the
absolute number, 1.34 below the published best 67.61 and 16.0 above the majority predictor.

**Arm B — A + OpenFace behavior stream on the 50-step clock (C7 / C8 / the "behavior run"): complete.**

| Run | Best val top-1 (epoch) | Test top-1 | Adjacent | MAE | Macro-F1 | Per-class acc 0/1/2/3 |
|---|---:|---:|---:|---:|---:|---|
| `B_beh_s1` | 67.13 (ep 2) | 66.58 | 91.62 | 0.436 | 53.05 | 74.8 / 19.5 / 26.3 / 88.4 |
| `B_beh_s2` | 67.79 (ep 4) | 66.22 | 91.36 | 0.442 | 51.50 | 79.0 / 12.2 / 29.1 / 86.5 |
| `B_beh_s3` | 68.25 (ep 2) | 65.43 | 91.89 | 0.444 | 51.93 | 77.7 / 15.0 / 30.1 / 84.5 |
| **mean ± sd** | | **66.08 ± 0.59** | 91.62 | 0.441 | 52.16 ± 0.80 | |

**B − A = −0.19 top-1, +0.07 adjacent, +0.001 MAE, −0.15 macro-F1: zero effect on every metric**, all
well inside the seed sd. This is the confound-free answer to C7: with the frame cap matched (50), the
code matched (same tree, AV path verified bit-identical), the warm start verified as an exact no-op,
the behavior series on the video clock, and three seeds, **the end-to-end neural behavior branch adds
nothing on top of the pixel model** — on top-1 *or* on the ordinal metrics. The single-seed +3.18
macro-F1 / +1.28 adjacent reported in §16.7 was seed noise (macro-F1 seed sd here is 0.8–1.1). §18.1's
paper claim "behavior stream improves minority-class and ordinal metrics" must be withdrawn. The
behavior *signal* is real (§19.6: 67.15 alone as segment statistics, +3.6 in late fusion); the
*branch* is what fails to extract it.

**Arm C — B + behavior-caption "text" (§16.10: a second view of the same AUs), the C9 ablation: complete.**

| Run | Best val top-1 (epoch) | Test top-1 | Adjacent | MAE | Macro-F1 | Per-class acc 0/1/2/3 |
|---|---:|---:|---:|---:|---:|---|
| `C_behtext_s1` | 67.04 (ep 6) | 66.27 | 91.62 | 0.439 | 52.85 | 75.3 / 19.5 / 26.5 / 87.5 |
| `C_behtext_s2` | 67.60 (ep 3) | 66.62 | 91.71 | 0.433 | 53.07 | 76.8 / 16.3 / 31.3 / 86.5 |
| `C_behtext_s3` | 67.97 (ep 2) | 65.47 | 91.67 | 0.445 | 50.85 | 78.6 / 9.3 / 33.4 / 84.2 |
| **mean ± sd** | | **66.12 ± 0.59** | 91.67 | 0.439 | 52.26 ± 1.22 | |

**All three arms at every decoder (3 seeds each, test):**

| Decoder | A audio+video | B +behavior | C +behavior+text |
|---|---:|---:|---:|
| argmax | 64.27 ± 0.22 | 64.26 ± 0.23 | 64.42 ± 0.07 |
| logit bias | 65.06 ± 0.37 | 64.83 ± 0.23 | 64.52 ± 0.55 |
| expected thresholds | 66.34 ± 0.50 | 66.15 ± 0.73 | 66.52 ± 0.14 |
| **refined expected thresholds** | **66.27 ± 0.45** | **66.08 ± 0.59** | **66.12 ± 0.59** |
| adjacent / MAE / macro-F1 (refined) | 91.55 / 0.440 / 52.31 | 91.62 / 0.441 / 52.16 | 91.67 / 0.439 / 52.26 |

**C9 answered: neither the numeric-AU branch nor the caption stream moves any metric at any decoder**
(largest arm difference 0.54 top-1 at logit-bias, under the 0.9 two-sigma bar). The behavior *branch*
as designed in the collab branch is inert on top of a warm-started pixel model; the behavior *signal*
is not (§19.6).

**Modality ablations — same weights, one stream zeroed at evaluation (refined expected thresholds, test):**

| Run | Fusion | Video-only | Audio-only | Fusion − video-only | Fusion macro-F1 | Video-only macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| `A_av_s1` | 65.78 | 65.60 | 46.59 | +0.18 | 53.41 | 50.76 |
| `A_av_s2` | 66.36 | 65.56 | 50.27 | +0.80 | 51.13 | 51.74 |
| `A_av_s3` | 66.67 | 64.67 | 50.27 | +1.99 | 52.37 | 51.37 |
| `B_beh_s1` | 66.58 | 65.96 | 50.27 | +0.62 | 53.05 | 53.60 |
| `B_beh_s2` | 66.22 | 64.67 | 46.68 | +1.55 | 51.50 | 53.04 |
| `B_beh_s3` | 65.43 | 65.56 | 50.27 | −0.13 | 51.93 | 51.46 |
| `C_behtext_s1` | 66.27 | 65.65 | 46.59 | +0.62 | 52.85 | 51.35 |
| `C_behtext_s2` | 66.62 | 65.56 | 46.19 | +1.06 | 53.07 | 50.98 |
| `C_behtext_s3` | 65.47 | 65.47 | 50.27 | 0.00 | 50.85 | 50.55 |
| **mean** | 66.16 | 65.41 | 48.60 | **+0.74** (8/9 ≥ 0) | 52.24 | 51.65 |

The audio contribution measured in the August sweep (+0.67, 12/13 models) reproduces under clean
three-seed training: **+0.74 top-1 from fusing audio, sign-consistent on 8 of 9 runs**, while audio alone
is exactly the majority predictor (50.27) or below it (46.2–46.7 when the model collapses to a different
constant). Macro-F1 is not consistently helped (5/9). This is the defensible AV claim for the paper:
*audio has no standalone engagement signal on EngageNet but adds ~0.7 top-1 on top of video when fused.*

**Bottom line of §19.5.** "Full video + full audio" = the model that already existed: 66.27 ± 0.45.
The behavior branch, on the video clock, matched code, matched frame cap, verified no-op warm start,
three seeds: no effect (66.08 / 66.12). The way to move the number is §19.6.

### 19.6 How to raise top-1 — what was measured today (2026-09-12), before proposing anything

Yuvraj asked for a plan to raise top-1. Rather than list options, four levers were measured on the
G04 seed-1 checkpoint (identical weights to `A_av_s1`) with everything selected on validation and test
touched once. Scripts: `AVTCA-collab-test/scripts/fullclip/{context_analysis,behavior_only_probe,ensemble_probe}.py`;
outputs under each run's `context/`.

**Where top-1 is lost (test confusion matrix, expected-value thresholds):**

| true \ predicted | 0 | 1 | 2 | 3 | recall |
|---|---:|---:|---:|---:|---:|
| 0 | 374 | 21 | 35 | 27 | 81.8% |
| 1 | 79 | 18 | 80 | 69 | **7.3%** |
| 2 | 52 | 19 | 116 | 232 | **27.7%** |
| 3 | 12 | 9 | 139 | 974 | 85.9% |

531 of 774 errors are the two middle classes; 232 of 419 "engaged" clips are called "highly engaged".

**Lever 1 — session-context smoothing: NULL.** Clips are consecutive 10 s segments of one recording
(`_vid_<v>_<k>`), and consecutive test clips share a label 69.3% of the time (chance 34%). Smoothing the
expected engagement value over ±h neighbours of the same video, thresholds refit on validation:
val-selected config (triangular, h=1, centre weight 3) scores **65.16 test vs 65.69 unsmoothed**; no
window beats the baseline by more than noise and h ≥ 3 costs 3–8 points of adjacent accuracy. The
model's errors are as autocorrelated as the labels, so neighbours add no information. Do not pursue.

**Lever 2 — label-prior shift: real but not recoverable without labels.** Validation is 12.3% class 0 /
53.1% class 3; test is 20.3% / 50.3%. Thresholds fit on test itself would give **67.82** (an oracle bound,
not a method) vs 65.69 with validation-fit thresholds. Unsupervised EM prior re-estimation (Saerens 2002)
*hurts* (62.85): the probabilities are not calibrated enough and EM overshoots class 1 to 23%. The gap
is the cost of a small validation split with a different class mix; the only honest fix is a larger,
test-like validation set (e.g. cross-fitted thresholds over train+val by subject).

**Lever 3 — the OpenFace features alone are as strong as the whole pixel model.** Laid out the way the
EngageNet paper's best baseline does it (20 uniform segments × [mean, std] of the per-frame features
= 880-d + clip mean/std), a `HistGradientBoostingClassifier` on our 22-d series, trained on the train
split only, scores on **test**:

| Model | Test top-1 (argmax) | Adjacent | Macro-F1 |
|---|---:|---:|---:|
| Logistic regression (balanced) | 60.64 | 85.33 | 48.57 |
| HistGB lr 0.05, 300 it | 66.67 | 88.16 | 52.41 |
| HistGB lr 0.03, 600 it | **67.15** | 88.21 | **53.19** |
| our AV model (pixels + audio), same test set | 65.69 | 90.96 | 49.42 |
| published best (Transformer, OpenFace G+HP+AU, 98-d) | 67.61 | — | — |

No neural network, no GPU, 2 minutes on CPU. Our 22 features are a subset of the paper's 98 (we lack
gaze vectors, head *location*, and the 18 AU presence flags — the raw OpenFace CSVs were not kept, only
the 22-d `.npy`); re-extracting would take ~2 CPU-hours. The neural behavior encoder of the collab branch
(Conv1d + GRU, from-scratch val ≤ 61) is therefore badly under-using its own input.

**Lever 4 — late fusion of the pixel model and the OpenFace model: +4.4 top-1, above the published
best.** Probability averaging `w·p_AV + (1−w)·p_GBM`, weight **and** thresholds selected on validation
only, GBM = mean of 3 fits, test set touched once:

| w_AV | Val top-1 | **Test top-1 (E-thresholds)** | Adjacent | MAE | Macro-F1 |
|---:|---:|---:|---:|---:|---:|
| 1.0 (AV alone) | 67.97 | 65.69 | 90.96 | 0.451 | 49.42 |
| 0.0 (GBM alone) | 65.27 | 66.05 | 89.27 | 0.473 | 50.35 |
| 0.5 | 66.01 | 69.77 | 89.58 | 0.426 | 51.91 |
| 0.6 | 67.32 | 69.99 | 90.43 | 0.413 | 53.40 |
| **0.7 (selected on val)** | **68.72** | **70.12** | 90.69 | **0.408** | 53.62 |
| 0.8 | 68.25 | 69.15 | 90.91 | 0.417 | 52.42 |

Every weight in 0.4–0.8 lands at 69.1–70.1, so this is not a tuned point. Why it works: on test the AV
model is wrong on 35.9% of clips and the GBM on 33.5%, but **both are wrong on only 23.5%** — the AV
model is wrong-and-GBM-right on 12.4%, the reverse on 9.9%. Pixels and AU statistics make different
mistakes. A stacking logistic regression fit on validation reaches 70.83 argmax / 58.27 macro-F1, but
its validation number is then optimistic, so probability averaging is the number to quote.

**Replication on `A_av_s2` (same protocol, val-selected w_AV = 0.7 again):** AV alone 65.82 → ensemble
**69.28** test top-1 / 89.76 adj / 0.427 MAE (w 0.4–0.8: 68.48–70.30; complementarity 12.4% / 10.3%).
`A_av_s3`: AV alone 66.67 → **70.17** (w 0.4–0.8: 68.48–70.26). **Three AV seeds, all val-selected at
w = 0.7: 70.12 / 69.28 / 70.17 = 69.86 ± 0.50 test top-1**, against 66.27 ± 0.45 for the AV model alone
(**+3.59**) and 67.61 for the published best (**+2.25**). Adjacent 89.8–90.7, MAE 0.408–0.427,
macro-F1 51.6–53.6. This is the paper's headline candidate; the protocol (GBM on train only, weight and
thresholds on validation, test once) is pre-registered above.

**The neural behavior branch does not capture what the GBM captures.** Same protocol with `B_beh_s1`
(arm B: AV + OpenFace branch, 66.80 alone at E-thresholds) as the neural member: val-selected w = 0.7 →
**69.77** test (w 0.4–0.8: 68.35–70.08); `B_beh_s2` 66.22 → **70.04**; `B_beh_s3` 65.34 → **68.93**; `C_behtext_s2` 66.49 → **69.41**;
complementarity unchanged (neural member wrong & GBM right 12.0–12.7% on every member).

**Per-member summary (test top-1, expected-value thresholds; w = 0.7 selected on validation for every
single member; GBM member identical throughout):**

| Neural member | Alone | + GBM (w 0.7) | Gain | Range over w 0.4–0.8 |
|---|---:|---:|---:|---|
| `A_av_s1` | 65.69 | **70.12** | +4.43 | 69.06–70.12 |
| `A_av_s2` | 65.82 | **69.28** | +3.46 | 68.48–70.30 |
| `A_av_s3` | 66.67 | **70.17** | +3.50 | 68.48–70.26 |
| `B_beh_s1` | 66.80 | **69.77** | +2.97 | 68.35–70.08 |
| `B_beh_s2` | 66.22 | **70.04** | +3.82 | 67.91–70.04 |
| `B_beh_s3` | 65.34 | **68.93** | +3.59 | 68.40–70.21 |
| `C_behtext_s1` | 66.76 | **69.73** | +2.97 | 68.26–69.73 |
| `C_behtext_s2` | 66.49 | **69.41** | +2.92 | 68.04–69.68 |
| **mean of 8** | 66.22 | **69.68** | **+3.46** | |

Eight of eight members land above the published best (67.61) after fusion; the minimum over any
member and any weight in 0.4–0.8 is 67.91. The "alone" column uses this script's own threshold fitter
(step 0.02 grid) and differs from `collect.py`'s refined-threshold numbers by ≤ 0.2. Four neural members so far, all val-selected at w = 0.7: 70.12 / 69.28 / 69.77 / 70.04,
mean **69.80**, every one above the published 67.61. So the
+3 to +4 from the GBM is orthogonal to the end-to-end behavior branch — the branch's per-frame
Conv1d+GRU reads the AU series but not the segment statistics that carry the clip-level signal. This
is the strongest argument for item 4 below (feed segment statistics into the branch). Also to test once B/C finish: does the neural behavior branch (arm B) buy
the same gain end-to-end, or is the GBM's advantage the segment-statistics representation?

**Proposed order of work for top-1 (highest expected value first):**
1. Replicate lever 4 on three AV seeds; report mean ± sd. Zero training cost.
2. Replace the GBM with the paper's Transformer over 20 segment tokens (8 heads, head size 256,
   4 blocks, dropout 0.3) on the same 22-d features, then ensemble — the paper reaches 67.61 alone with
   98-d features, so ≥ 66 with 22-d is plausible and a stronger second member lifts the ensemble.
3. Re-extract OpenFace keeping the full CSV (98-d: gaze vectors, head location, AU presence), ~2 CPU-h,
   so the behavior model has the paper's full feature set.
4. Fold the segment-statistics representation into the neural behavior branch (arm B) so the gain is
   end-to-end rather than an ensemble — only if 1–3 show the branch itself is the weak part.
Not worth doing (measured): neighbour-clip smoothing, EM prior adaptation, checkpoint ensembling of
near-identical AV runs (65.6–65.7), more AV hyperparameter sweeps (noise floor).

### 19.7 Segment transformer member and three-way fusion (2026-09-12, evening)

Yuvraj asked to build the segment transformer and run it. Scripts (worktree `scripts/fullclip/`):
`segment_features.py` (cache, `results/fullclip/segtf/features_S20.npz`), `segment_transformer.py`
(trainer), `sweep_segtf.py` (8 configs × 3 seeds), `gbm_member.py` (saved boosting member,
`results/fullclip/gbm/probs.npz`), `fuse_members.py` (N-way fusion, weights + thresholds on validation),
`fuse_all.py` (seed-paired tables). Results: `results/fullclip/segtf/`.

**What is borrowed and what is ours.** The tokenisation (20 uniform segments × [mean, std] per clip) is
the EngageNet baseline's and is cited to Singh et al. Everything else is our own implementation: the
22-d feature set, a pre-norm Transformer encoder with a CLS token and learned positions, the repo's
`OrdinalDistanceCrossEntropy` (weight 0.15, label smoothing 0.1), AdamW + warmup-cosine, token dropout,
epoch selection on validation argmax top-1, expected-level thresholds fit on validation, and its role as
one member of a late fusion with our audio-visual network.

**Sweep (pre-registered selection: mean best-validation argmax over 3 seeds; test shown for all):**

| Config | Mean val | Test top-1, thresholds (s1 / s2 / s3) | Test argmax (s1 / s2 / s3) | Best epochs |
|---|---:|---|---|---|
| **T7 d64 L2 ffn128 lr 1e-3** ← selected | **65.36** | 67.02 / 67.91 / 66.05 = **67.0** | 67.69 / 68.26 / 66.80 = **67.6** | 3 / 7 / 7 |
| T3 d256 L4 ffn512 lr 5e-4 | 65.05 | 65.96 / 69.02 / 67.11 | 66.98 / 68.04 / 67.60 | 2 / 4 / 12 |
| T5 d128 L4 + noise 0.1 | 64.86 | 65.87 / 67.20 / 67.77 | 65.87 / 68.04 / 67.11 | 2 / 9 / 9 |
| T1 d128 L4 lr 1e-3 | 64.83 | 64.32 / 67.46 / 66.71 | 64.54 / 67.82 / 67.24 | 6 / 6 / 9 |
| T2 d128 L2 | 64.77 | 64.49 / 66.84 / 65.51 | 64.23 / 65.38 / 66.36 | 10 / 4 / 4 |
| T8 d128 L4 lr 3e-4 | 64.64 | 64.45 / 67.95 / 69.19 | 65.16 / 65.82 / 68.71 | 6 / 4 / 7 |
| T4 d128 L4 + sqrt-inverse sampler | 63.99 | 68.66 / 64.58 / 65.87 | 68.62 / 64.18 / 64.63 | 5 / 8 / 19 |
| T6 d128 L4 mean-pool, token dropout 0.2 | 63.96 | 66.76 / 63.79 / 65.51 | 66.80 / 64.10 / 65.07 | 2 / 1 / 9 |

**Alone, the selected transformer averages 67.0 (thresholds) / 67.6 (argmax) on test with 22 features**,
i.e. the published 67.61 baseline is reproduced with a fifth of its feature dimensionality. Two
properties matter for how it is used: it overfits within 2–12 epochs, and its seed-to-seed test spread
(~3 points) is much larger than the AV model's, with validation only weakly predicting test. The
smallest model won on validation, which is consistent with 7,879 training clips.

**Fusion (weights and thresholds on validation only; seed i paired with seed i; test once per row):**

| Neural member | Alone | + transformer (2-way) | + transformer + GBM (3-way) | 3-way weights (val-selected) |
|---|---:|---:|---:|---|
| `A_av_s1` | 65.69 | 66.31 (w 0.85/0.15) | **70.17** | 0.6 / 0.2 / 0.2 |
| `A_av_s2` | 66.40 | 69.59 | **70.26** | 0.6 / 0.2 / 0.2 |
| `A_av_s3` | 64.98 | 70.17 | **70.43** | 0.6 / 0.2 / 0.2 |
| `B_beh_s1` | 66.84 | — | 69.86 | 0.6 / 0.0 / 0.4 |
| `B_beh_s2` | 66.05 | — | 70.26 | 0.6 / 0.2 / 0.2 |
| `B_beh_s3` | 65.47 | — | 70.35 | 0.6 / 0.2 / 0.2 |
| `C_behtext_s1` | 66.40 | — | 69.68 | 0.6 / 0.2 / 0.2 |
| `C_behtext_s2` | 66.58 | — | 70.17 | 0.6 / 0.2 / 0.2 |
| **A seeds, mean ± sd** | 65.69 | 68.69 ± 2.08 | **70.29 ± 0.14** | |
| **all 8 members** | 66.04 | | **70.15 ± 0.25** (min 69.68) | |

Three-way adjacent 90.4–91.7, MAE 0.392–0.417, macro-F1 51.1–56.5. Reading:

- **The three-way fusion is the best and most stable number in the repository: 70.29 ± 0.14 on the
  three A seeds, 70.15 ± 0.25 over all eight neural checkpoints, +2.5 over the published best, and it
  chose the same weights (0.6 AV / 0.2 transformer / 0.2 GBM) on seven of eight members.** Versus the
  two-way GBM fusion (69.86 ± 0.50) the gain is +0.43 — inside noise on the mean, but the sd shrinks
  by 3.5× because the two behavior members average out each other's seed noise.
- **The transformer is a weaker *fusion member* than the GBM despite being a stronger *stand-alone
  model*.** Two-way A + transformer is 68.69 ± 2.08: on seed 1 validation chose w = 0.85/0.15 and
  scored 66.31 on test. Its validation score does not track test well enough to set a weight on its
  own; the boosting model's does. Use the transformer *with* the GBM, not instead of it.
- Paper framing: audio-visual cross-attention network + two OpenFace-statistics models (one neural,
  one boosting), fused at the probability level with a pre-registered validation protocol; the
  segment tokenisation is credited to Singh et al.; the AV network alone (66.27 ± 0.45) and each
  member alone are reported next to the fusion.

**Open after this:** (a) 98-d OpenFace re-extraction (gaze vectors, head location, AU presence) for
both behavior members; (b) stabilise the transformer (fewer epochs, stronger regularisation, or seed
averaging of its logits) so it can carry the fusion without the GBM; (c) wire the three-way fusion into
`ui/inference.py`; (d) the same protocol on DAiSEE as a second-corpus check.

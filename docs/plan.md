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
| E11 | Implement audio `AvgPool1D` temporal subsampling before cross-attention | Critical |
| E12 | Implement modality dropout (p=0.15) in training forward pass | High |
| E13 | Per-modality validation logging (audio-only, video-only, fusion) | High |
| E14 | Pilot session (Session 1, Cohort A) | High |
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

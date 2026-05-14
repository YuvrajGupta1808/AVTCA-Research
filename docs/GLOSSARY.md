# Technical Glossary — AVTCA Training Report

Alphabetical definitions of every technical term used in the training report.
Terms that have a direct counterpart in the codebase include the relevant file reference.

---

### AttentionBlock
(`models/transformer_timm.py`)

A ViT-style (Vision Transformer) cross-attention block that combines LayerNorm, multi-head
attention, a residual connection, a two-layer MLP, and another residual connection:

```
Output = Attention(LayerNorm(query), LayerNorm(key))
Output = query  + DropPath(Output)        ← first residual
Output = Output + DropPath(MLP(LayerNorm(Output)))  ← second residual
```

Used for cross-modal attention between audio and video branches. The query comes from one
modality and the key/value from the other.

---

### AverageMeter
(`utils.py`)

A helper class that tracks a running mean for a numeric quantity (e.g., loss or accuracy)
across all batches in an epoch. Stores `.val` (current value), `.avg` (running average),
`.sum`, and `.count`. Used to report per-epoch summaries without accumulating all batch values.

---

### Batch size
The number of training samples processed before the model parameters are updated. Set to 8
in all runs. With `softhard` modality dropout, each forward pass internally uses 4× the batch
size (8 → 32 effective samples per step) due to augmentation stacking.

---

### Classification report
A per-class summary table produced by `sklearn.metrics.classification_report`, showing:
- **Precision** per class
- **Recall** per class
- **F1-score** per class
- **Support** (number of true instances per class)
- Macro average and weighted average across classes

---

### Conv1D
A 1D convolution that slides a filter of width *k* along a single dimension (here: the
temporal dimension of audio or video features). Preserves the temporal structure while mixing
information across channels. In this model, audio uses Conv1D after frequency-dimension
collapse; video uses Conv1D across the 15 frame sequence.

---

### Conv2D
A 2D convolution applied to a spatial map (height × width) or a 2D time-frequency matrix.
In this model, audio MFCC is treated as a 2D input (10 frequency channels × ~172 time frames)
and processed by Conv2D before the frequency dimension is collapsed to a single vector.
Video uses Conv2D inside EfficientFace for spatial feature extraction.

---

### Cross-attention
An attention mechanism where the **query** (Q) comes from one source (modality A) and the
**key** (K) and **value** (V) come from another source (modality B). Lets modality A learn
what parts of modality B are most relevant to each of its own positions. Used twice in `it`
fusion mode: once after stage 1, once after stage 2 self-attention.

Contrast with **self-attention**, where Q, K, and V all come from the same sequence.

---

### Cross-entropy loss
The standard classification loss function. For a ground-truth class *y* and predicted
probability vector *p*:

```
Loss = -log(p[y])
```

Penalises confident wrong predictions more than uncertain correct ones. The optimizer
minimises the average cross-entropy over all samples in a batch.

---

### DataParallel
A PyTorch wrapper (`nn.DataParallel`) that replicates the model across multiple GPUs. Each
GPU processes a shard of the batch independently and gradients are averaged at the end of
each backward pass. Both runs were run with 2× RTX 3090 GPUs in DataParallel mode, though
effective training used both GPUs.

---

### Dampening
A modifier to the SGD momentum term. Set to 0.9 in all runs. Reduces the contribution of
the accumulated momentum gradient, acting as a form of velocity decay:

```
velocity = momentum * velocity − learning_rate * gradient * (1 − dampening)
```

---

### DropPath (Stochastic Depth)
(`models/transformer_timm.py`)

A regularisation technique that randomly drops entire residual branches during training
(replaces them with 0) with probability *p*. At test time, scales the branch by *(1−p)*.
Different from Dropout (which drops individual neurons). Used inside AttentionBlock.

---

### EfficientFace
(`models/efficientface.py`)

A lightweight face analysis CNN pretrained on the AffectNet-7 dataset (7-class expression
recognition, ~4M face images). Used here to initialise the video branch's convolutional
backbone. The pretrained weights provide general-purpose face feature extractors (facial
action units, texture, geometry) that transfer well across different subjects and recording
conditions. Controlled by the `--pretrain_path` CLI argument.

---

### F1 score
The harmonic mean of precision and recall. Ranges 0–1 (or 0–100%).

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Two aggregation strategies appear in this report:

- **F1 macro:** Simple unweighted average of per-class F1 scores. Gives equal weight to
  rare classes (like neutral, 32 samples) and common classes (64 samples).

- **F1 weighted:** Weighted average of per-class F1 scores, where each class is weighted
  by its support (number of true instances). More representative of overall performance
  when classes are imbalanced.

F1 is preferred over accuracy alone when classes have unequal support, or when false
positives and false negatives have different costs.

---

### Fusion strategy (`--fusion`)
Controls where and how audio and video branch representations are combined:

| Flag | Name | Description |
|------|------|-------------|
| `it` | Intermediate Transformer | Cross-attention after stage 1 of each branch (using AttentionBlock with MLP + residual). A second cross-attention after self-attention. Used in all reported runs. |
| `ia` | Intermediate Attention | Same cross-attention point, but uses only the attention weight matrix as a multiplicative gate on the original features — does not use the attended output. No secondary cross-attention. |
| `lt` | Late Transformer | Single cross-attention after both stages are complete (late fusion). |

---

### Global Average Pooling
Reduces a spatial feature map (H × W) or temporal sequence (T) to a single vector by taking
the mean over all positions. Used inside EfficientFace to collapse each frame's spatial
representation to a 1024-dim vector. Produces translation-invariant features.

---

### Gradient norm
The L2 norm of the concatenated parameter gradients from all layers, computed after the
backward pass. A sudden spike in gradient norm indicates training instability. Logged
per-batch to `train_runtime.log`. Normal values in these runs: 2–6.

---

### InvertedResidual
(`models/efficientface.py`)

The core building block of MobileNetV2-style networks. Uses a "bottleneck inverted" design:
first expands channels (×expansion factor), then applies depthwise convolution (one filter
per channel, cheap), then projects back down. A residual skip connection is added when
input and output channels match. Provides efficient local feature extraction within each
video frame.

---

### LocalFeatureExtractor
(`models/efficientface.py`)

Divides an input feature map into 4 equal spatial quadrants, applies a shared convolutional
block to each, then concatenates the outputs. Captures local spatial patterns (e.g.,
asymmetric facial expressions, mouth-only vs. eye-only movement) that might be averaged away
by global pooling. Applied in parallel with the main EfficientFace stream.

---

### Learning rate (LR)
The step size for parameter updates. Set to 0.06 initially for all runs. Decayed by ×0.1
at epoch 40 (→ 0.006). A high initial LR allows fast early progress; the decay stabilises
training. Controlled by `--learning_rate` and `--lr_steps`.

---

### LR decay / LR schedule
A policy for reducing the learning rate during training to refine the solution. In these
runs, a **step decay** is used: the LR is multiplied by 0.1 at predefined epoch milestones
(`lr_steps = [40, 55, 65, 70, 200, 250]`). Additional `ReduceLROnPlateau` scheduling was
also configured (patience=10) but the step decay triggers first.

---

### MaxPool
Takes the maximum value in each sliding window over a feature map. Retains the strongest
activation in each region, providing translation invariance. Used after temporal processing
in both audio and video branches to collapse the temporal dimension to a single vector
(e.g., 128-dim audio sequence → 128-dim vector).

---

### MFCC (Mel-Frequency Cepstral Coefficients)
A compact representation of audio frequency content. Computed by:
1. Taking the Short-Time Fourier Transform (STFT) of the waveform
2. Applying a Mel filterbank (non-linear frequency scale matching human auditory perception)
3. Taking the log of the Mel-scale energies
4. Applying the Discrete Cosine Transform (DCT)

This model uses **10 MFCC channels** × ~172 time frames per clip (3.6 s audio at standard
hop length). The 10 coefficients capture the spectral envelope; the temporal dimension
captures how it evolves through the utterance.

---

### Modulator
(`models/modulator.py`, used inside `EfficientFaceTemporal`)

A combined channel + spatial attention module applied to early video features:
- **Channel attention:** Global-average-pools the feature map spatially → small FC network
  → sigmoid gate applied to each channel. Emphasises informative channels (e.g., those
  responsive to mouth movement).
- **Spatial attention:** Pools across channels at each position → convolutional gate →
  sigmoid. Emphasises informative spatial regions (e.g., eyes, mouth corners).

Similar in spirit to CBAM (Convolutional Block Attention Module).

---

### Modality dropout (`--mask`, softhard mode)
A training-time augmentation strategy that makes the model robust to missing or degraded
modalities. In `softhard` mode, each mini-batch is augmented to 4× its size:

| Copy | Audio | Video |
|------|-------|-------|
| 1 | Clean | Clean |
| 2 | α × Clean | (1−α) × Clean  (α ~ Uniform[0,1]) |
| 3 | Zeros | Clean |
| 4 | Clean | Zeros |

The four copies are shuffled and fed together. The model learns to handle any combination
of clean, blended, or absent modalities.

---

### Momentum (SGD)
A technique that accumulates a velocity vector in the gradient direction across steps,
accelerating convergence and damping oscillations. Set to 0.9 (standard value). The update
rule is approximately:

```
velocity = 0.9 × velocity + gradient
parameters -= learning_rate × velocity
```

---

### MultiheadAttention (`nn.MultiheadAttention`)
PyTorch's built-in multi-head self-attention module. Splits the embedding dimension into
`num_heads` independent subspaces, computes attention in each, then concatenates and projects
the outputs. Used for **self-attention** within each modality (audio attends to itself, video
attends to itself) before the secondary cross-attention. Unlike `AttentionBlock` (which is
custom and used for cross-attention), this is applied with Q=K=V from the same source.

---

### num_heads
The number of parallel attention heads in the multi-head attention mechanisms. Each head
operates on a `(embedding_dim / num_heads)`-dimensional subspace. Multiple heads allow the
model to simultaneously attend to different aspects of the input (e.g., one head might
attend to rhythm cues in audio while another attends to pitch contours).

Run 1: `num_heads=1`. Run 2: `num_heads=4`.

---

### Precision
For class *c*:

```
Precision(c) = True Positives(c) / (True Positives(c) + False Positives(c))
```

Measures how many of the model's predictions for class *c* are actually correct.
High precision but low recall (as seen for `calm` in Run 2 test) means the model rarely
guesses `calm` but is correct when it does.

---

### prec@1 / Top-1 Accuracy
The fraction of samples where the model's **highest-scoring** class matches the true label.
The primary metric used for checkpoint selection in these runs. Equivalent to standard
classification accuracy.

---

### prec@5 / Top-5 Accuracy
The fraction of samples where the true label is among the model's **top 5** highest-scoring
classes. With only 8 classes total, Top-5 is trivially high (96–99%): the model must be
badly wrong on 4+ classes to miss the true label in the top 5. Not a meaningful metric for
8-class classification.

---

### RAVDESS
**Ryerson Audio-Visual Database of Emotional Speech and Song.** A publicly available
acted emotion dataset of 24 professional actors (12 male, 12 female) performing 8 emotional
states (neutral, calm, happy, sad, angry, fearful, disgust, surprised) at two intensities
(normal and strong) in 2,880 video clips. The dataset is balanced across actors, emotions,
and gender. Standard evaluation splits are by actor (leave-actors-out), preventing the model
from memorising actor-specific vocal or facial characteristics.

---

### Recall
For class *c*:

```
Recall(c) = True Positives(c) / (True Positives(c) + False Negatives(c))
```

Measures what fraction of all true instances of class *c* the model correctly identifies.
Low recall (as seen for `sad` in the validation set) means many `sad` clips are
misclassified as other emotions.

---

### Residual connection (skip connection)
Adding the input of a sub-block directly to its output:

```
Output = F(Input) + Input
```

Allows gradients to flow directly through the network without passing through potentially
saturated activation functions. Prevents vanishing gradients in deep networks. Used in
`AttentionBlock` (two residuals: around attention and around MLP) and `InvertedResidual`.

---

### Self-attention
An attention mechanism where query, key, and value all come from the **same** sequence.
Each position attends to all other positions in the same sequence, learning long-range
dependencies. Used here via `nn.MultiheadAttention` to process the temporal audio and
video features independently before cross-attention fusion.

---

### Softmax
A normalisation function that converts a vector of raw scores (logits) into a probability
distribution summing to 1:

```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

Applied to the model's 8-dimensional output before computing cross-entropy loss or
extracting the predicted class (argmax).

---

### Support
In a classification report, the number of ground-truth instances of a class in the
evaluation set. In this report, `neutral` has support 32 and all other classes have
support 64 (test set). Support is used as the weight in **weighted average** F1/precision/recall.

---

### Weight decay (L2 regularisation)
A penalty added to the loss function proportional to the sum of squared parameter values:

```
Total loss = Cross-entropy + λ × Σ w²
```

Set to λ = 1e-3 in all runs. Discourages very large weights, reducing overfitting. Applied
to all model parameters via the SGD `weight_decay` argument.

---

*This glossary covers all bold or specialised technical terms appearing in `TRAINING_REPORT.md`.*
*For questions about the codebase implementation, see `docs/CLAUDE.md`.*

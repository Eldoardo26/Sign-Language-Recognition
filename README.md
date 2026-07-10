# Cross-Modal Knowledge Distillation for Continuous Sign Language Recognition

Can a **skeleton-based teacher** teach an **appearance-based Transformer student** to
recognise continuous sign language better than it can on its own?

This repository contains the full experimental code for that question on
**RWTH-PHOENIX-Weather 2014T**: a skeleton CSLR teacher, a lightweight Signformer
student on I3D features, and two cross-modal distillation modules based on FD-CMKD
(*Distilling Cross-Modal Knowledge via Feature Disentanglement*).

**The short answer, on this setup, is no.** A controlled experiment with a
pre-registered decision margin finds the transfer statistically indistinguishable
from simply fine-tuning the student for longer. The results below report that
honestly, along with the error analysis that explains why.

---

## Results

Word Error Rate on PHOENIX-2014T (lower is better), recovered from the training logs
and saved notebook outputs in this repository.

| Model | Params | Dev WER | Test WER |
|---|---:|---:|---:|
| Teacher — skeleton, TSSI-75 + TCN/BiLSTM/CTC | 6.60 M | 47.81 | not evaluated |
| Student — Signformer on I3D, no distillation | 3.76 M | **39.54** | 41.53 |
| Student + Module 2 (full FD-CMKD) | 3.76 M | 39.81 | 40.90 |

The teacher is **1.76× larger than the student and 7.4 WER points worse.** This
inverts the usual distillation premise: there is no strong teacher here, only a
*different* one.

### The controlled experiment

`_archive/legacy_distillation_notebooks/run_optuna.ipynb` compares the feature-only
module against a **matched control**: same starting checkpoint (40.30 WER), same
budget, same early-stopping patience, same TPE sampler (seed 42), and a **decision
margin of 0.30 WER fixed before looking at the data**.

| Arm | Trials | Best | Median | Mean |
|---|---:|---:|---:|---:|
| Control (fine-tuning only, searches `lr`) | 6 | **38.86** | 38.92 | 39.07 |
| FD-CMKD (feature-only, searches `lr`, `λ`, `w_high`) | 15 | **38.70** | 39.02 | 39.07 |

Δ = **0.16 WER**, inside the margin. The notebook's own automatic verdict:

```
VERDETTO: distillazione NEUTRA vs controllo (delta +0.16 <= margine 0.3).
```

Three reasons why even that 0.16 is not a real effect:

- **The means are identical** (39.07 in both arms) and the distillation arm's
  **median is worse** (39.02 vs 38.92). The gap appears only in the minimum.
- **The minimum is not comparable across arms with different trial counts.**
  Resampling 6 trials from the distillation arm's own results yields a best-of-6 at
  least as good as the control's 38.86 in 95% of draws. The observed 0.16 is what one
  expects from minimising over 15 trials rather than 6.
- **A rank test finds no difference**: Mann–Whitney `U = 43.0`, `p = 0.45`.

Both arms improve by roughly 1.5 points over the starting checkpoint. That improvement
belongs to the fine-tuning, not to the teacher — and it is exactly what would have
been mistaken for a distillation gain had the control arm not been run.

### Why the transfer does not help

Distillation is not inert: it changes *which* errors the student makes, not *how many*.
On the test set (SUB = substitutions, DEL = deletions, INS = insertions):

| System | WER | SUB | DEL | INS |
|---|---:|---:|---:|---:|
| Baseline | 41.53 | 23.29 | 14.56 | 3.68 |
| + FD-CMKD | 40.90 | 23.45 | 13.53 | 3.92 |
| Δ | −0.63 | +0.16 | **−1.03** | +0.24 |

The teacher signal makes the student **less blank-biased**: it deletes fewer glosses.
But those recovered slots come back almost entirely as substitutions and insertions.
The teacher succeeds in telling the student **that** a sign is being articulated; it
fails to tell it **which**. At 47.81 WER it is not discriminative enough to teach
discrimination.

This contradicts the hypothesis that motivated the work — a teacher focused on hand
shape and motion should above all reduce **substitutions**. Substitutions went up.

---

## Layout

```
├── code/
│   ├── csrl_skeleton/     # Teacher: 75 MediaPipe keypoints → TSSI → TCN + attn + BiLSTM + CTC
│   │   ├── skeleton.py            # DFS joint ordering, TSSI image generation
│   │   ├── model.py               # PoseNetworkCTC
│   │   ├── losses.py              # CTC + entropy reg, greedy/beam decoding, WER
│   │   └── runner.ipynb           # entry point
│   ├── signformer/        # Student: I3D features → Transformer encoder + CTC gloss head
│   │   ├── main/                  # shared framework, reused by the distillation modules
│   │   └── runner.ipynb           # entry point
│   └── distillation/
│       ├── extract_skeleton_feats.py  # freeze the teacher, dump pre-classifier features
│       ├── distill.py                 # Module 1: feature-only FD-CMKD
│       ├── fd_cmkd.py                 # Module 2: full FD-CMKD + shared classifiers
│       ├── vocab_utils.py             # shared-vocabulary merge policy
│       └── run_feature_distill.ipynb  # entry point
├── distillation/optuna/   # the controlled study: ctrl_* vs dist_*
├── _archive/              # the controlled-experiment notebook
└── dataset/DATA.md        # how to rebuild the data and fetch the checkpoints
```

> An earlier, more modular standalone version of the skeleton pipeline
> (`decoding.py`, `ensemble.py`, `augmentation.py`, `metrics.py`) lives on the
> **`archive/csrl-standalone`** branch. It was not merged into this one.

---

## Method

Three stages; the teacher never runs during student training.

**1. Train the teacher.** The 75 keypoints per frame (`0–32` body, `33–53` left hand,
`54–74` right hand) are smoothed along time, normalised **per body part** so that hand
shape is described in its own local frame rather than being dominated by absolute
position, and reordered along a depth-first traversal of the skeletal graph so that
anatomically adjacent joints land in neighbouring rows. The result is a **Temporal
Skeleton Spatial Image** of shape `(3, 75, T)` — the channels are x, y, confidence.
`PoseNetworkCTC` consumes it with dilated temporal convolutions, a temporal
self-attention layer and a 3-layer BiLSTM, trained with CTC plus an entropy regulariser.

**2. Export teacher features.** The frozen teacher runs once over every video and its
pre-classifier feature sequence — `(T, 512)`, the concatenated BiLSTM directions — is
stored as float16, one pickle per split.

**3. Train the student with distillation.** Teacher sequences are linearly interpolated
in time onto the student's length; videos without a teacher entry drop out of the
distillation term for that batch.

### Module 1 — feature-only FD-CMKD

A linear head projects the student's 256-d encoder output into the teacher's 512-d
space. Both streams are standardised per frame (zero mean, unit ℓ₂ norm). A real DFT is
taken **along the feature axis**; a fixed binary mask splits the spectrum into a low and
a high band, and an inverse DFT maps each band back. The two bands are matched with
different strengths:

- **low band → MSE** (strong consistency; the semantics shared across modalities)
- **high band → logMSE** (weak consistency; modality-specific detail and noise)

where `σ(x) = log(1+x)` for `x ≥ 0` and `−log(1−x)` otherwise. The signed-log
compression damps the gradient from large high-frequency discrepancies, so the high
band is only loosely matched.

```
L = L_CTC + λ · (w_low · L_low + w_high · L_high)
```

### Module 2 — full FD-CMKD with shared classifiers

Adds the paper's third component: a **shared gloss decision space**. Teacher and student
were trained with different gloss vocabularies, so `vocab_utils.build_shared_vocab`
merges them with a deterministic policy — orthographically similar glosses
(`SequenceMatcher` ratio ≥ 0.85) collapse to the alphabetically first form, hyphenated
variants fold into their compact spelling, merge chains are resolved transitively, and
cycles are broken by taking the lexicographically smallest member.

Two linear classifiers `Φ_low`, `Φ_high` over that shared vocabulary score the low and
high bands of **both** modalities. The paper's cross-entropy is replaced by **CTC**,
since CSLR is sequence labelling with no frame alignment:

```
L_align = CTC(Φ_low(s_low)) + CTC(Φ_high(s_high)) + CTC(Φ_low(t_low)) + CTC(Φ_high(t_high))
L       = L_task + λ_feat · L_feat + λ_align · L_align
```

The teacher terms are computed only on videos that have a teacher entry. The teacher
stays frozen; the projection and the two shared classifiers are appended to the
student's optimiser.

---

## Setup

Python 3.10, PyTorch 2.6, CUDA 12.4. Dependencies are managed with
**[uv](https://github.com/astral-sh/uv)**.

```bash
git clone https://github.com/Eldoardo26/Sign-Language-Recognition.git
cd Sign-Language-Recognition
uv sync
```

Then, for the Signformer branch:

```bash
pip install sophia-opt   # then delete the trailing `from sophia.sophia import SophiaG`
                         # line in its __init__.py (an upstream Signformer quirk)
```

`requirements.txt` captures the environment used on the server but **does not list
`torch`** — install it separately.

For data and checkpoints, see **[dataset/DATA.md](dataset/DATA.md)**.

## Reproducing

```bash
jupyter notebook code/csrl_skeleton/runner.ipynb               # 1. teacher
python code/distillation/extract_skeleton_feats.py             # 2. teacher features
jupyter notebook code/signformer/runner.ipynb                  # 3. student baseline
jupyter notebook code/distillation/run_feature_distill.ipynb   # 4. distillation
```

`extract_skeleton_feats.py` honours `PHOENIX_ROOT`, `MSKA_DATA_DIR`, `TEACHER_CKPT` and
`SKELETON_FEATS_DIR` if your data lives elsewhere. The distillation notebooks prepend
`code/signformer/` to `sys.path` so that `import main.training` resolves to the shared
framework rather than duplicating it.

### Best configuration found

Module 2, from `code/distillation/active_fd_cmkd.yaml` (8 Optuna TPE trials over
`λ_feat`, `λ_align`, `w_high`, `lr`, at 30% of the training budget):

| | |
|---|---|
| `lambda_feat` | 0.1607 |
| `lambda_align` | 0.1498 |
| `w_low` / `w_high` | 1.0 / 0.2921 |
| optimiser | SophiaG, lr 8.18e-5, betas 0.95/0.998, wd 3e-3 |
| batch size | 32 |

> **Every training config sets `overwrite: true`.** `configs/sign.yaml` writes into
> `code/signformer/sign_sample/`, `sign_distill.yaml` and `active_fd_cmkd.yaml` into
> `dataset/checkpoints/`. Launching a notebook therefore erases the checkpoint and the
> `validations.txt` of the corresponding published run, without prompting. Point
> `model_dir` somewhere else before you experiment.

### What was verified

From a clean checkout, with the dataset rebuilt as described in
[dataset/DATA.md](dataset/DATA.md):

- the teacher loads `tssi75_cslr_best.pt` with `strict=True` and runs a forward pass
  (6.59 M parameters, 1022 gloss classes);
- `extract_skeleton_feats.py` exports 7096 / 519 / 642 teacher sequences;
- the Signformer baseline reproduces **39.54** dev WER at beam 1, **39.35** at beam 4,
  and **41.53** test WER with DEL 14.56 / INS 3.68 / SUB 23.29 — the error breakdown
  reported above;
- both distillation modules train, validate and checkpoint.

---

## Citation

The distillation method adapts, to CTC sequence labelling:

> Liu et al. *Distilling Cross-Modal Knowledge via Feature Disentanglement.* 2025.

The student is:

```bibtex
@article{eta2024signformer,
  title  = {Signformer is all you need: Towards Edge AI for Sign Language},
  author = {Eta Yang},
  year   = {2024},
  journal= {arXiv preprint arXiv:2411.12901}
}
```

PHOENIX-2014T and the `pami0` I3D features remain under their original licences and are
not redistributed here.

# Cross-Modal Knowledge Distillation for Continuous Sign Language Recognition

Can a **skeleton-based teacher** teach an **appearance-based Transformer student** to
recognise continuous sign language better than it can on its own?

This repository contains the full experimental code for that question on
**RWTH-PHOENIX-Weather 2014T**: a skeleton CSLR teacher on RTMW whole-body keypoints,
a lightweight Signformer student on I3D features, two cross-modal distillation
modules based on FD-CMKD (*Distilling Cross-Modal Knowledge via Feature
Disentanglement*), and the follow-up studies that explain the outcome.

**The short answer, on this setup, is no — and the reason is not the teacher.**
A controlled experiment with a pre-registered decision margin finds the transfer
statistically indistinguishable from simply fine-tuning the student for longer;
matched controls with a stronger whole-body teacher, a reversed (strong→weak)
direction, and three different loss geometries all agree. What finally works is
abandoning feature imitation altogether: training the two encoders **jointly** with
cross-attention fusion beats every single-stream system by a wide margin.

---

## Results

Word Error Rate on PHOENIX-2014T (lower is better). Every model is evaluated at its
best-on-dev checkpoint; dev figures use the beam width swept on dev (the baseline's
greedy figure is 39.54).

| Model | Params | Dev WER | Test WER |
|---|---:|---:|---:|
| Teacher — skeleton, RTMW-133 → 55-joint TSSI, GCN+TCN+attn+BiLSTM | 7.64 M | 41.76* | 42.61* |
| Student — Signformer on I3D, no distillation | 3.76 M | 39.35 | 41.53 |
| Student + FD-CMKD, warm-started | 3.76 M | 39.51 | 40.81 |
| Student + FD-CMKD, from scratch | 3.76 M | 39.49 | 40.81 |
| ↳ matched control, warm (teacher off, same schedule) | 3.76 M | 39.70 | 40.45 |
| ↳ matched control, from scratch (teacher off) | 3.76 M | 39.59 | 41.63 |
| **Two-stream end-to-end fusion (v2)** | 14.53 M | **33.30** | **33.56** |

\* greedy decoding with CTC prior correction; see [Decoding notes](#decoding-notes).

The teacher is **2.03× larger than the student and 2.22 dev-WER points worse.** This
inverts the usual distillation premise: there is no strong teacher here, only a
*different* one. Against its matched controls the distilled student wins some
comparisons and loses others, with all arms within 0.2 points on the selection
metric — the transfer is neutral. The two-stream model, which consumes the same two
modalities but fuses them with live cross-attention instead of imitating frozen
features, improves on the best single stream by **~8 test-WER points**.

### The controlled experiment

`_archive/legacy_distillation_notebooks/run_optuna.ipynb` (run in the earlier
75-keypoint era of this project; trial artefacts in `distillation/optuna/`) compares
the feature-only module against a **matched control**: same starting checkpoint,
same budget, same early-stopping patience, same TPE sampler (seed 42), and a
**decision margin of 0.30 WER fixed before looking at the data**.

| Arm | Trials | Best | Median | Mean |
|---|---:|---:|---:|---:|
| Control (fine-tuning only, searches `lr`) | 6 | **38.86** | 38.92 | 39.07 |
| FD-CMKD (feature-only, searches `lr`, `λ`, `w_high`) | 15 | **38.70** | 39.02 | 39.07 |

Δ = **0.16 WER**, inside the margin: the arms are indistinguishable. The means are
identical, the distillation arm's median is worse, resampling shows the best-of-15
vs best-of-6 gap is expected under the null, and a Mann–Whitney rank test gives
`U = 43.0`, `p = 0.45`. Both arms improve ~1.5 points over their common starting
checkpoint — that improvement belongs to the fine-tuning, not the teacher, and is
exactly what would have been mistaken for a distillation gain without the control.
The matched controls in the table above repeat this check against the current
(stronger) teacher and reach the same verdict.

### The loss form does not matter

Three geometries of the feature term, identical warm-start protocol:

| Feature loss | Dev WER | Test WER |
|---|---:|---:|
| FD-CMKD (frequency-decoupled DFT split) | 39.51 | 40.81 |
| Plain MSE (ablated decoupling) | 39.51 | 40.92 |
| Relational (similarity-preserving, Tung & Mori 2019) | 39.62 | 40.85 |

All three land within 0.11 test-WER points. The frequency decoupling — the method's
distinctive component — confers no measurable advantage over plain feature matching.

### Why the transfer does not help

Distillation is not inert: it changes *which* errors the student makes, not *how
many*. On the test set (warm-started arm):

| System | WER | SUB | DEL | INS |
|---|---:|---:|---:|---:|
| Baseline | 41.53 | 23.29 | 14.56 | 3.68 |
| + FD-CMKD | 40.81 | 23.33 | 13.48 | 3.99 |
| Δ | −0.72 | +0.04 | **−1.08** | +0.31 |

The teacher signal makes the student **less blank-biased**: it deletes fewer glosses,
and gives about a third of that ground back as insertions. Substitutions — the
failure a hand-shape-aware teacher was supposed to fix — barely move (+0.04 here,
−0.49 from scratch). The teacher succeeds in telling the student **that** a sign is
being articulated; it does not teach it **which**.

### The reverse direction: a stronger teacher

The forward experiments all have the weaker model as the teacher. Swapping the roles —
the **Signformer** teaching the **skeleton** model — is the conventional
strong-to-weak distillation, and tests whether the neutral result was just a matter
of teacher quality. This study was run in the earlier 75-keypoint configuration
(teacher 39.54 dev, skeleton student 47.81 dev, a ~6.9-point test-WER gap in the
teacher's favor); each regime is read against a matched no-teacher control under the
same 0.30 WER margin (development WER):

| Regime | Control | Distilled | Δ | |
|---|---:|---:|---:|---|
| From scratch | 48.51 | 48.99 | +0.48 | **hurts** |
| Warm-started | 47.39 | 47.79 | +0.40 | **hurts** |

In both regimes the transfer makes the student **worse** than its control, beyond the
margin. A teacher 6.9 test-WER points **stronger** than the student still fails to
help — so the binding constraint is not teacher quality but the **cross-modal transfer
itself**: the two modalities encode the sign along incompatible axes, and pulling one
representation towards the other is unhelpful weak-to-strong and harmful strong-to-weak.

This is produced by `run_reverse_{warm,scratch}_distill.ipynb`; each notebook trains
the distilled and the control arm and prints the verdict.

### From transfer to fusion

If imitation fails, is the skeleton information redundant? No. With both encoders
frozen (`experiments/02_fusion`), an **oracle** that picks the better stream per
utterance scores ~36.7 test WER against ~42.6/~44 for the single streams: the
modalities are **complementary**, but their independently-trained CTC spikes are
misaligned, so frame-level feature fusion collapses and only sequence-level routing
(`conf-route`, 42.4) survives. The remedy is to stop freezing
(`experiments/03_twostream_e2e`): train both encoders jointly with bidirectional
cross-attention and a shared CTC head, warm-starting the skeleton branch from the
teacher checkpoint. That model reaches **33.30 dev / 33.56 test** — below the
frozen-feature oracle, and ~8 points below the Signformer baseline.

> **On the optimizer.** The configs request SophiaG, but that package was not installed,
> so the Signformer runs fell back to plain Adam — every reported number is Adam's.
> The controlled comparisons hold the optimizer fixed across arms, so the verdicts are
> unaffected; only the absolute numbers might move under SophiaG.

---

## Layout

```
├── code/
│   ├── csrl_skeleton/     # Teacher: RTMW 133 kp → 55-joint TSSI → GCN + TCN + attn + BiLSTM + CTC
│   │   ├── extract_wholebody_133.py   # RTMW/DWPose inference over the raw frames
│   │   ├── prepare_phoenix_133.py     # pack per-video keypoints into split pickles
│   │   ├── skeleton_coco133.py        # 55-joint subset, DFS ordering, TSSI generation
│   │   ├── skeleton_mediapipe75.py    # legacy 75-kp layout (kept for the reverse runs)
│   │   ├── model.py                   # PoseNetworkCTC (optional spatial GCN front-end)
│   │   ├── losses.py                  # CTC, greedy/beam decoding (fixed), WER
│   │   ├── train_teacher.py           # headless two-phase training
│   │   └── runner_133.ipynb           # notebook entry point
│   ├── signformer/        # Student: I3D features → Transformer encoder + CTC gloss head
│   │   ├── main/                      # shared framework, reused by the distillation trainers
│   │   └── runner.ipynb               # entry point
│   └── distillation/
│       ├── extract_skeleton_feats.py     # skeleton teacher features (forward direction)
│       ├── extract_transformer_feats.py  # Signformer teacher features (reverse direction)
│       ├── distill.py / fd_cmkd.py       # Module 1 / Module 2 losses (+ rkd_sp, plain-MSE modes)
│       ├── fd_cmkd_trainer.py            # Signformer-student trainer (forward)
│       ├── reverse_distill.py            # skeleton-student trainer (reverse)
│       ├── vocab_utils.py                # shared-vocabulary merge policy
│       ├── run_{forward,reverse}_{warm,scratch}_distill.ipynb   # the four experiments
│       ├── future_works/                 # matched controls + plain-MSE ablation notebooks
│       └── experiments/
│           ├── 01_relational_kd/         # relational (similarity-preserving) loss arm
│           ├── 02_fusion/                # frozen-feature fusion, oracle, conf-route, probes
│           └── 03_twostream_e2e/         # end-to-end two-stream model (the best system)
├── distillation/optuna/   # the controlled study: ctrl_* vs dist_* trial validations
├── _archive/              # legacy notebooks, kept for provenance (see _archive/README.md)
└── dataset/DATA.md        # how to rebuild the data and fetch the checkpoints
```

### The four distillation notebooks

Each experiment writes to its own `runs/<name>/` directory, so it is always clear
which one produced a result. The direction says who teaches whom; the regime says how
the student is initialised.

| Notebook | Teacher → Student | Student init |
|---|---|---|
| `run_forward_warm_distill.ipynb` | skeleton → Signformer | warm-started from baseline |
| `run_forward_scratch_distill.ipynb` | skeleton → Signformer | from scratch |
| `run_reverse_warm_distill.ipynb` | Signformer → skeleton | warm-started |
| `run_reverse_scratch_distill.ipynb` | Signformer → skeleton | from scratch |

The forward notebooks store their full training logs as saved outputs, and the key
runs' `validations.txt` are tracked under `runs/` — the numbers in this README can be
re-derived from artefacts in the repository.

> An earlier, more modular standalone version of the skeleton pipeline lives on the
> **`archive/csrl-standalone`** branch. It was not merged into this one.

---

## Method

Three stages; the teacher never runs during student training.

**1. Train the teacher.** RTMW ("performance" mode) extracts 133 COCO-WholeBody
keypoints per frame; the 55 signing joints (upper body + both hands) are kept, and
coordinates are normalised in a body-centred frame (origin at the shoulder midpoint,
scale from the shoulder width). A depth-first traversal of the skeletal graph orders
the joints so anatomically adjacent ones land in neighbouring rows of a **Temporal
Skeleton Spatial Image** of shape `(3, 55, T)` (x, y, confidence). `PoseNetworkCTC`
consumes it with a spatial graph-convolution front-end followed by dilated temporal
convolutions, temporal self-attention and a 3-layer BiLSTM, trained with CTC in two
phases (frozen backbone, then full fine-tuning with differential learning rates).

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

### Two-stream end-to-end fusion

`experiments/03_twostream_e2e` drops distillation entirely: the appearance encoder
(3-layer Transformer on I3D, Signformer-style) and the skeleton encoder
(`PoseNetworkCTC.forward_feat`, warm-started from the teacher checkpoint) are trained
**jointly**. The skeleton stream is resampled onto the appearance time axis, N
bidirectional cross-attention blocks mix the two, and a single **shared CTC head**
emits one aligned spike train (per-stream auxiliary CTC heads provide deep
supervision, `loss = CTC_joint + α·(CTC_app + CTC_skel)`). Checkpoints are atomic and
resumable (`last.pt`), so an interrupted run continues with Run All.

---

## Setup

Python 3.10 to 3.12, PyTorch 2.6 on CUDA 12.4. Dependencies are managed with
**[uv](https://github.com/astral-sh/uv)**; `pyproject.toml` lists the direct ones and
`uv.lock` pins the rest.

```bash
git clone https://github.com/Eldoardo26/Sign-Language-Recognition.git
cd Sign-Language-Recognition
uv sync
```

`requirements.txt` is exported from `uv.lock` for people who prefer pip. It carries an
`--extra-index-url` line, because the pinned `torch==2.6.0+cu124` does not exist on
PyPI. For a CPU-only install, swap `cu124` for `cpu` in that line.

Two things are worth knowing before a training run.

**TensorFlow is a hard dependency of the student framework.** `main/model.py` calls
`tf.nn.ctc_beam_search_decoder` for the beam search, and hides the GPU from TensorFlow
at import, so it only ever runs on the CPU. `tensorflow-cpu` is a valid, much smaller
substitute.

**SophiaG is an optional extra, and its absence is silent.** The configs request it,
but without the package `main/builders.py` falls back to Adam and prints one line
saying so — and **every number reported here was produced under that Adam fallback**
(see the optimizer note above). To actually train with SophiaG:

```bash
uv sync --extra sophia
```

then delete the trailing `from sophia.sophia import SophiaG` line from the installed
package's `__init__.py`, an upstream defect that breaks the import. Setting
`optimizer: adam` in `configs/sign.yaml` makes the fallback explicit and reproduces
the reported figures.

For data and checkpoints, see **[dataset/DATA.md](dataset/DATA.md)**.

## Reproducing

```bash
python code/csrl_skeleton/extract_wholebody_133.py       # 1a. RTMW keypoints from raw frames
python code/csrl_skeleton/prepare_phoenix_133.py         # 1b. pack into split pickles
python code/csrl_skeleton/train_teacher.py               # 1c. teacher (or runner_133.ipynb)
jupyter notebook code/signformer/runner.ipynb            # 2.  student baseline
python code/distillation/extract_skeleton_feats.py       # 3a. skeleton teacher features
python code/distillation/extract_transformer_feats.py    # 3b. Signformer teacher features
jupyter notebook code/distillation/run_forward_warm_distill.ipynb   # 4. any experiment
jupyter notebook code/distillation/experiments/03_twostream_e2e/run_twostream.ipynb  # 5. best model
```

Step 3a feeds the forward notebooks, 3b the reverse ones. The matched controls and the
plain-MSE ablation live in `code/distillation/future_works/`, the relational arm in
`experiments/01_relational_kd/`. Skeleton-side training writes to `runs/` and aborts
rather than overwrite an existing directory.

`extract_skeleton_feats.py` honours `PHOENIX_ROOT`, `MSKA_DATA_DIR`, `TEACHER_CKPT` and
`SKELETON_FEATS_DIR` if your data lives elsewhere. The distillation notebooks prepend
`code/signformer/` to `sys.path` so that `import main.training` resolves to the shared
framework rather than duplicating it.

### Best distillation configuration found

Module 2, warm-started (8 Optuna TPE trials over `λ_feat`, `λ_align`, `w_high`, `lr`,
at 30% of the training budget):

| | |
|---|---|
| `lambda_feat` | 0.1607 |
| `lambda_align` | 0.1498 |
| `w_low` / `w_high` | 1.0 / 0.2921 |
| learning rate | 8.18e-5 |
| batch size | 32 |

> **The student framework's configs set `overwrite: true`.** `configs/sign.yaml`
> writes into `code/signformer/sign_sample/`, the distillation YAMLs into their
> `runs/<name>/` directory. Point `model_dir` somewhere fresh before you experiment.

## Decoding notes

Teacher figures use greedy decoding with a CTC prior correction (β tuned on dev).
An earlier version of `csrl_skeleton/losses.py` had a prefix-beam-search defect that
dropped the probability mass of label repeats, biasing the decoder towards insertions
and making beam search score worse than greedy; the current `beam_decode` folds that
mass back into the prefix (see the comment at `losses.py:74`). The student framework
decodes with TensorFlow's CTC beam search, width swept 1–10 on dev.

### What was verified

From this checkout:

- the teacher checkpoint (`runs/run133`) loads and counts **7,642,810 parameters**
  over 1,087 classes; its stored history reproduces the 41.76 best dev WER at epoch
  60 of 75, with the phase split at epoch 15;
- the forward warm/scratch runs and their matched controls reproduce the table above
  from their `validations.txt` and `train.log`;
- the two-stream v2 notebook's saved output contains the full 60-epoch log ending in
  `TEST beam 33.56%`, and `runs/twostream_v2/best.pt` stores `best = 0.33298`;
- the Signformer baseline reproduces **39.54** dev WER greedy, **39.35** at beam 4,
  and **41.53** test WER (DEL 14.56 / INS 3.68 / SUB 23.29).

---

## Citation

The distillation method adapts, to CTC sequence labelling:

> Liu et al. *Distilling Cross-Modal Knowledge via Feature Disentanglement.* 2025.

Other main ingredients:

```bibtex
@article{eta2024signformer,
  title  = {Signformer is all you need: Towards Edge AI for Sign Language},
  author = {Eta Yang},
  year   = {2024},
  journal= {arXiv preprint arXiv:2411.12901}
}
@article{jiang2024rtmw,
  title  = {RTMW: Real-Time Multi-Person 2D and 3D Whole-body Pose Estimation},
  author = {Jiang, Tao and Xie, Xinchen and Li, Yining},
  year   = {2024},
  journal= {arXiv preprint arXiv:2407.08634}
}
@inproceedings{chen2022twostream,
  title     = {Two-Stream Network for Sign Language Recognition and Translation},
  author    = {Chen, Yutong and Zuo, Ronglai and Wei, Fangyun and Wu, Yu and Liu, Shujie and Mak, Brian},
  year      = {2022},
  booktitle = {NeurIPS}
}
```

PHOENIX-2014T and the `pami0` I3D features remain under their original licences and are
not redistributed here.

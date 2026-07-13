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
| Teacher — skeleton, TSSI-75 + TCN/BiLSTM/CTC | 6.60 M | 47.81 | 48.41 |
| Student — Signformer on I3D, no distillation | 3.76 M | **39.54** | 41.53 |
| Student + Module 2 (full FD-CMKD) | 3.76 M | 39.81 | 40.90 |

The teacher is **1.76× larger than the student and 6.9 test-WER points worse.** This
inverts the usual distillation premise: there is no strong teacher here, only a
*different* one.

Teacher numbers are greedy decoding with prior correction (β = 0.324), checkpoint at
epoch 74. Beam search makes the teacher *worse*, not better — 55.25 test WER at width
25, against 48.41 greedy — and it does so by trading 299 deletions for 381 extra
insertions. That is not a tuning artefact; see [Known issues](#known-issues).

### The controlled experiment

`_archive/legacy_distillation_notebooks/run_optuna.ipynb` compares the feature-only
module against a **matched control**: same starting checkpoint (40.30 WER), same
budget, same early-stopping patience, same TPE sampler (seed 42), and a **decision
margin of 0.30 WER fixed before looking at the data**.

| Arm | Trials | Best | Median | Mean |
|---|---:|---:|---:|---:|
| Control (fine-tuning only, searches `lr`) | 6 | **38.86** | 38.92 | 39.07 |
| FD-CMKD (feature-only, searches `lr`, `λ`, `w_high`) | 15 | **38.70** | 39.02 | 39.07 |

Δ = **0.16 WER**, inside the margin. The notebook's own automatic verdict, quoted
verbatim from its stored output:

```
VERDETTO: distillazione NEUTRA vs controllo (delta +0.16 <= margine 0.3).
```

*Distillation is NEUTRAL against the control: delta +0.16 is within the 0.30 margin.*
The archived notebooks are kept in the language they ran in; see
[_archive/README.md](_archive/README.md).

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

### The reverse direction: a stronger teacher

The forward experiments all have the weaker model as the teacher. Swapping the roles —
the **Signformer** (39.54) teaching the **skeleton** model (47.81) — is the
conventional strong-to-weak distillation, and tests whether the neutral result was
just a matter of teacher quality. Each regime is run against a matched no-teacher
control under the same 0.30 WER margin (development WER):

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

> **On the optimizer.** The configs request SophiaG, but that package was not installed,
> so the Signformer runs fell back to plain Adam — every reported number is Adam's.
> The controlled comparisons hold the optimizer fixed across arms, so the verdicts are
> unaffected; only the absolute numbers might move under SophiaG.

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
│       ├── extract_skeleton_feats.py     # skeleton teacher features (forward direction)
│       ├── extract_transformer_feats.py  # Signformer teacher features (reverse direction)
│       ├── distill.py / fd_cmkd.py        # Module 1 / Module 2 losses
│       ├── fd_cmkd_trainer.py             # Signformer-student trainer (forward)
│       ├── reverse_distill.py             # skeleton-student trainer (reverse)
│       ├── vocab_utils.py                 # shared-vocabulary merge policy
│       └── run_{forward,reverse}_{warm,scratch}_distill.ipynb   # the four experiments
├── distillation/optuna/   # the controlled study: ctrl_* vs dist_*
├── _archive/              # legacy notebooks, kept for provenance
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
| `run_reverse_warm_distill.ipynb` | Signformer → skeleton | warm-started from baseline |
| `run_reverse_scratch_distill.ipynb` | Signformer → skeleton | from scratch |

The forward direction is the thesis's original, weak-teacher setup. The reverse
direction is conventional distillation — the stronger Signformer teaching the weaker
skeleton model — and each reverse notebook also runs a matched no-teacher control, so
its verdict is read against a control rather than against the baseline.

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

**TensorFlow is a hard dependency.** `main/model.py` calls
`tf.nn.ctc_beam_search_decoder` for the beam search, and hides the GPU from TensorFlow
at import, so it only ever runs on the CPU. `tensorflow-cpu` is a valid, much smaller
substitute. TensorFlow 2.13, which the original environment used, caps
`typing-extensions` below the version PyTorch 2.6 requires; the two cannot be resolved
together, so the pin here is 2.18.

**SophiaG is an optional extra, and its absence is silent.** The published results were
trained with it. Without it, `main/builders.py` falls back to Adam and prints one line
saying so, and the numbers will not reproduce.

```bash
uv sync --extra sophia
```

then delete the trailing `from sophia.sophia import SophiaG` line from the installed
package's `__init__.py`, an upstream defect that breaks the import. Setting
`optimizer: adam` in `configs/sign.yaml` is the honest alternative if you do not need
to match the reported figures.

For data and checkpoints, see **[dataset/DATA.md](dataset/DATA.md)**.

## Reproducing

```bash
jupyter notebook code/csrl_skeleton/runner.ipynb               # 1. teacher
jupyter notebook code/signformer/runner.ipynb                  # 2. student baseline
python code/distillation/extract_skeleton_feats.py             # 3a. skeleton teacher features
python code/distillation/extract_transformer_feats.py          # 3b. Signformer teacher features
jupyter notebook code/distillation/run_forward_warm_distill.ipynb    # 4. any of the four
```

Step 3a feeds the forward notebooks, 3b the reverse ones. Pick whichever of the four
distillation notebooks you want in step 4; each is self-contained.

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

## Known issues

### The teacher's beam search discards probability mass

`csrl_skeleton/losses.py:77`, inside `beam_decode`:

```python
elif c == last:
    continue          # the path is dropped, not merged into (pre, last)
```

In CTC a gloss is emitted over many consecutive frames, so *staying* on the current
label is the common case. A prefix beam search must fold that path back into the
existing prefix; here it is discarded. The only way a beam can stay put is through the
blank, while emitting a *new* label is always available — which biases the decoder
towards insertions. On test, beam(25) produces 447 insertions against greedy's 66.

Related, `key = (pre, last)` on a blank never resets `last`, so the decoder can never
emit the same gloss twice in a row. Empirically this costs little: only 5 of the 4264
test glosses are adjacent repeats.

The published teacher WERs use greedy decoding and are unaffected.

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

# Data and checkpoints

No data is redistributed in this repository. PHOENIX-2014T is subject to the RWTH
licence, and the derived features weigh about 1.8 GB. This page explains how to
rebuild `dataset/` from scratch.

## Expected layout

```
dataset/
├── i3d_features_rwth phoenix 2014t/   # raw .npy, (T, 1024) per video — download
│   ├── train/ (7096)  val/ (519)  test/ (642)
├── annotations/manual/                # PHOENIX-2014-T.{train,dev,test}.corpus.csv — download
├── keypoints-kaggle/                  # raw MediaPipe keypoints per video — download
├── pose/phoenix2014t_75kp/            # generated → teacher input
│   ├── Phoenix-2014T.{train,dev,test}
│   └── gloss2ids.pkl
├── features/
│   ├── i3d_pami0/                     # generated → student input
│   └── skeleton_feats/                # generated → distillation input
└── checkpoints/                       # downloaded from the Releases
```

Every path above is the default. `PHOENIX_ROOT`, `PHOENIX_DATASET_ROOT`,
`PHOENIX_KEYPOINTS`, `MSKA_DATA_DIR`, `TEACHER_CKPT` and `SKELETON_FEATS_DIR`
override the corresponding one if your data lives elsewhere.

### Verify what you have

```bash
python - <<'EOF'
import gzip, pickle
D = "dataset"
for s, n in [("train", 7095), ("dev", 519), ("test", 642)]:
    with gzip.open(f"{D}/features/i3d_pami0/phoenix14t.pami0.{s}", "rb") as f:
        got = len(pickle.load(f))
    print(f"i3d  {s:5s} {got:5d}  {'ok' if got == n else f'EXPECTED {n}'}")
for s, n in [("train", 7096), ("dev", 519), ("test", 642)]:
    with open(f"{D}/pose/phoenix2014t_75kp/Phoenix-2014T.{s}", "rb") as f:
        got = len(pickle.load(f))
    print(f"pose {s:5s} {got:5d}  {'ok' if got == n else f'EXPECTED {n}'}")
EOF
```

`i3d train` is 7095 and `pose train` is 7096 — one video in the CSV has no I3D
feature. That difference is expected, not corruption.

A truncated file raises `EOFError` or `UnpicklingError: pickle data was truncated`
here. Sizes that are exact multiples of 0.25 MiB are a reliable symptom of an
interrupted copy.

## 1. Original dataset

RWTH-PHOENIX-Weather 2014T, under its own licence:
<https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/>

You need the video frames (to extract keypoints) and the `manual/` annotations.

## 2. I3D features — student input

**These are not downloaded ready-made**: they are generated from the raw I3D `.npy`
files.

```bash
python code/signformer/prepare_phoenix_i3d.py
# -> dataset/features/i3d_pami0/phoenix14t.pami0.{train,dev,test}
```

Expected: `train` 7095 sequences (~1.5 GB), `dev` 519, `test` 642. The train count is
7095 rather than 7096 because one video in the CSV has no I3D feature; the script
reports it with `[WARN] missing/empty I3D`.

> **Careful.** The script skips splits that already exist and are non-empty
> (`already exists — skipped`). If a file is corrupt you must **delete it first**,
> otherwise it is never regenerated.

Integrity check:

```python
import gzip, pickle
for s in ["train", "dev", "test"]:
    with gzip.open(f"dataset/features/i3d_pami0/phoenix14t.pami0.{s}", "rb") as f:
        print(s, len(pickle.load(f)))
```

An `EOFError` means the file is truncated: delete it and regenerate.

## 3. MediaPipe keypoints — teacher input

The keypoints are **not** extracted from the video frames by this repository: they
are read from a pre-extracted per-video dump under `dataset/keypoints-kaggle/`,
one pickle per video with a `"keypoints"` array of shape `(T, 75, C)`.

```bash
python code/csrl_skeleton/prepare_phoenix_mediapipe.py
# -> dataset/pose/phoenix2014t_75kp/Phoenix-2014T.{train,dev,test} + gloss2ids.pkl
```

Joint layout: `0–32` body (MediaPipe Pose), `33–53` left hand, `54–74` right hand.

`C` is 2 (x, y) or 3 (x, y, confidence) and is passed through unchanged.
`skeleton.generate_tssi_75` accepts both, substituting a constant confidence
channel when `C = 2`. **The released teacher checkpoint was trained on 3-channel
keypoints**, so 2-channel input trains but will not reproduce the published WER.

Note that `gloss2ids.pkl` (1118 entries) is written for MSKA compatibility and is
*not* what the teacher trains on: `csrl_skeleton/vocab.py` rebuilds its own vocabulary
from the three splits, applying the gloss-merge policy, and lands on 1022 classes.

## 4. Teacher features — distillation input

Requires the teacher checkpoint (see below).

```bash
python code/distillation/extract_skeleton_feats.py
# -> dataset/features/skeleton_feats/{train,dev,test}.pkl
```

Each file is a dict `{video_name: ndarray float16 (T, 512)}`. The script honours the
environment variables `PHOENIX_ROOT`, `MSKA_DATA_DIR`, `TEACHER_CKPT` and
`SKELETON_FEATS_DIR`.

## 5. Trained checkpoints

Published as assets of this repository's **Releases** — not as tracked files, and not
through Git LFS (the free quota is 1 GB and runs out immediately).

<https://github.com/Eldoardo26/Sign-Language-Recognition/releases>

| Asset | Size | Model | Dev WER |
|---|---:|---|---:|
| `tssi75_cslr_best.pt` | 26 MB | skeleton teacher, epoch 74 | 47.81 |
| `sign_sample_best.ckpt` | 48 MB | Signformer student, baseline | 39.54 |
| `fd_cmkd_best.ckpt` | 58 MB | student + FD-CMKD, step 13400 | 39.81 |

Download them into `dataset/checkpoints/`:

```bash
gh release download v1.0 -D dataset/checkpoints/
```

**OneDrive mirror (UniBA):** the weights are also on the university SharePoint,
[at this address](https://unibari-my.sharepoint.com/:f:/g/personal/e_bufi5_studenti_uniba_it/IgCVGx3DNvfIRImcOzTHF0_KAYepcnkJaVrHIrufkML-awA?e=1uk882).
That link requires UniBA credentials, so the Releases remain the public route.

### Note on the FD-CMKD checkpoint

It was saved under NumPy ≥ 2.0 (module `numpy._core`). The locked environment installs
NumPy 2.0, so the checkpoint loads directly and nothing below applies.

If instead you are on NumPy 1.x (`numpy.core`) — an older conda environment, say —
`torch.load` fails with `ModuleNotFoundError: No module named 'numpy._core'`.

Do **not** alias `numpy._core` into `sys.modules`: it segfaults inside NumPy's C
reconstruct path. Use an unpickler that renames the module instead:

```python
import pickle, torch

class _Ren(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)

class _PM:
    Unpickler = _Ren
    load = staticmethod(pickle.load)

ck = torch.load(path, map_location="cpu", pickle_module=_PM, weights_only=False)
```

## Evaluating without the `train` file

`main.prediction.test()` discards `train_data` (`_, dev_data, test_data = load_data(...)`),
so evaluation works even when `train` is missing or corrupt — provided the vocabulary is
not built from it. Pass the saved vocabularies in the config:

```yaml
data:
  train: phoenix14t.pami0.dev     # placeholder, never read
  gls_vocab: path/to/gls.vocab    # 1087 entries, matches gloss_output_layer
  txt_vocab: path/to/txt.vocab
```

Sanity check that the substitution is correct: with beam size 1 the baseline must score
**39.54**, exactly the greedy validation figure in the training log.

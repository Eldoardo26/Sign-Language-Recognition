# Signformer on PHOENIX-2014T

The student branch: a lightweight Transformer encoder with a CTC gloss head, trained
on precomputed I3D features for continuous sign language **recognition**.

Because the visual features are frozen and precomputed, no large GPU is required.
The model has 3.76 M parameters; at `batch_size: 32` with `T_max = 400` and 1024-d
features it stays well under 3 GB of VRAM and trains on a laptop RTX 4050.

## 1. Dependencies

Install the environment from the repository root with `uv sync`. One optimiser needs
a manual step:

```bash
pip install sophia-opt
```

Then delete the trailing `from sophia.sophia import SophiaG` line from that package's
`__init__.py`, an upstream defect that makes the import fail. Alternatively set
`optimizer: adam` in `configs/sign.yaml`; `main/builders.py` also falls back to Adam
on its own when SophiaG cannot be imported.

## 2. Data

The raw features and the annotations are expected under `dataset/` at the repository
root:

```
dataset/i3d_features_rwth phoenix 2014t/{train,val,test}/*.npy
dataset/annotations/manual/PHOENIX-2014-T.{train,dev,test}.corpus.csv
```

Set `PHOENIX_ROOT` or `PHOENIX_DATASET_ROOT` if your copy lives elsewhere, then:

```bash
python prepare_phoenix_i3d.py
```

Expected output:

```
[train] 7095 samples written to dataset/features/i3d_pami0/phoenix14t.pami0.train
[dev]    519 samples written to dataset/features/i3d_pami0/phoenix14t.pami0.dev
[test]   642 samples written to dataset/features/i3d_pami0/phoenix14t.pami0.test
```

`train` holds 7095 rather than 7096 sequences because one video in the CSV has no
I3D feature; the script reports it. See [../../dataset/DATA.md](../../dataset/DATA.md)
for the complete data layout and for the released checkpoints.

## 3. Train and evaluate

`runner.ipynb` is the entry point. Its first cell decides where the run writes:
output goes to `runs/<RUN_NAME>/`, never into `dataset/checkpoints/`.

From the command line instead:

```bash
python -m main train configs/sign.yaml
python -m main test  configs/sign.yaml --ckpt ../../dataset/checkpoints/sign_sample/best.ckpt
```

## Expected numbers

The released baseline scores **39.54** dev WER with greedy decoding (beam 1), and
**39.35** dev / **41.53** test WER at beam 4. Reproducing 39.54 at beam 1 is the
quickest confirmation that the features, the vocabulary and the decoder all agree.

Translation metrics (BLEU, ROUGE, CHRF) print as `-1` throughout: this configuration
performs recognition only, and the translation head is disabled.

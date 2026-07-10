# Signformer — Phoenix-2014T Setup

## Cosa fa questo modello
Transformer encoder-decoder molto leggero (0.57M–few M params) per Sign Language **Translation** (segni → testo tedesco).
Feature input: I3D 1024-dim pre-estratte (non serve GPU potente per il preprocessing).

## Hardware consigliato
**Gira in locale sul portatile RTX 4050 6GB** — il modello è minuscolo (1 layer encoder, 1 layer decoder, hidden=256).
Batch 32 × T_max=400 × 1024 → ~500 MB VRAM. Con batch_size=16 stai sotto 3 GB.

## Step 1 — Installa dipendenze

```bash
pip install numpy portalocker PyYAML torch torchvision matplotlib seaborn
# torchtext 0.5.0 è vecchissimo; usa la versione compatibile col tuo PyTorch:
pip install torchtext==0.6.0   # per PyTorch 1.7–1.9
# oppure per PyTorch 2.x:
pip install torchtext          # versione recente
```

**Fix SophiaG optimizer** (richiesto dal config default):
```bash
pip install git+https://github.com/Liuhong99/Sophia.git
```
Se vuoi evitarlo, cambia `optimizer: sophiag` → `optimizer: adam` in `configs/sign.yaml`.

## Step 2 — Prepara i dati

Le i3d features e le annotations sono attese in `phoenix/dataset/`:
- `phoenix/dataset/i3d_features_rwth phoenix 2014t/{train,val,test}/*.npy`
- `phoenix/dataset/annotations/manual/PHOENIX-2014-T.{train,dev,test}.corpus.csv`

Se i dati sono altrove, esporta `PHOENIX_ROOT` (o `PHOENIX_DATASET_ROOT`) prima di lanciare lo script.

```bash
cd phoenix/code/signformer
python prepare_phoenix_i3d.py
```

Output atteso:
```
[train] 7096 samples written to data/PHOENIX2014T/phoenix14t.pami0.train
[dev]   519 samples written to data/PHOENIX2014T/phoenix14t.pami0.dev
[test]  642 samples written to data/PHOENIX2014T/phoenix14t.pami0.test
```

## Step 3 — Adatta il config

`configs/sign.yaml` punta già a `./data/PHOENIX2014T/...` e `feature_size: 1024`.
Opzionale: riduci `batch_size: 32` → `16` se hai poca VRAM.

## Step 4 — Avvia training

```bash
python -m main train configs/sign.yaml
```

## Step 5 — Valuta

```bash
python -m main test configs/sign.yaml --ckpt sign_sample/best.ckpt
```

## Metriche attese
Il paper riporta risultati competitivi su Phoenix-2014T con modello minimo.
Con I3D congelate aspettati BLEU4 ~20–25 (task SLT, non SLR puro).

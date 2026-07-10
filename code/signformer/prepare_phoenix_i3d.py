"""
Prepare Phoenix-2014T I3D features in the format expected by Signformer.

Input:
  - I3D .npy files: shape (T, 1024) per video
  - Annotations CSV: name|video|start|end|speaker|orth|translation

Output:
  - <phoenix>/dataset/features/i3d_pami0/phoenix14t.pami0.{train,dev,test}
    gzipped pickle, list of dicts:
      { name, signer, gloss, text, sign: torch.FloatTensor(T, 1024) }

Usage:
  python prepare_phoenix_i3d.py
"""

import os
import gzip
import pickle
import numpy as np
import torch
import pandas as pd
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
# Repo layout: phoenix/code/signformer/prepare_phoenix_i3d.py
# → PHOENIX_ROOT is the top-level "phoenix" folder (three levels up).
# Override any of these with an env var if your data lives elsewhere.
PHOENIX_ROOT = Path(os.environ.get("PHOENIX_ROOT", Path(__file__).resolve().parents[2]))
DATASET_ROOT = Path(os.environ.get("PHOENIX_DATASET_ROOT", PHOENIX_ROOT / "dataset"))
I3D_ROOT     = DATASET_ROOT / "i3d_features_rwth phoenix 2014t"
ANN_ROOT     = DATASET_ROOT / "annotations" / "manual"
# Canonical location for the packed features, shared by the standalone
# Signformer config (configs/sign.yaml) and the distillation notebook.
OUT_DIR      = DATASET_ROOT / "features" / "i3d_pami0"

SPLITS = {
    "train": ("train", "PHOENIX-2014-T.train.corpus.csv"),
    "dev":   ("val",   "PHOENIX-2014-T.dev.corpus.csv"),
    "test":  ("test",  "PHOENIX-2014-T.test.corpus.csv"),
}
# -----------------------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(split_name: str, i3d_split: str, ann_file: str):
    ann_path = ANN_ROOT / ann_file
    df = pd.read_csv(ann_path, sep="|")

    i3d_dir = I3D_ROOT / i3d_split
    samples = []
    missing = 0

    for _, row in df.iterrows():
        name   = row["name"]
        signer = row["speaker"]
        gloss  = row["orth"]
        text   = row["translation"]

        npy_path = i3d_dir / f"{name}.npy"
        if not npy_path.exists() or npy_path.stat().st_size == 0:
            print(f"  [WARN] missing/empty I3D for {name}")
            missing += 1
            continue

        try:
            feat = np.load(str(npy_path))      # (T, 1024)
        except Exception as e:
            print(f"  [WARN] cannot load {name}: {e}")
            missing += 1
            continue
        sign = torch.from_numpy(feat).float()  # FloatTensor

        samples.append({
            "name":   name,
            "signer": signer,
            "gloss":  gloss,
            "text":   text,
            "sign":   sign,
        })

    out_path = OUT_DIR / f"phoenix14t.pami0.{split_name}"
    tmp_path = out_path.with_suffix(".tmp")
    with gzip.open(str(tmp_path), "wb") as f:
        pickle.dump(samples, f)
    tmp_path.replace(out_path)   # atomic rename — mai file parziali

    print(f"[{split_name}] {len(samples)} samples written to {out_path}  (missing: {missing})")


if __name__ == "__main__":
    for split_name, (i3d_split, ann_file) in SPLITS.items():
        out_path = OUT_DIR / f"phoenix14t.pami0.{split_name}"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[{split_name}] {out_path} already exists — skipped")
            continue
        load_split(split_name, i3d_split, ann_file)
    print("Done — run: python -m main train configs/sign.yaml")

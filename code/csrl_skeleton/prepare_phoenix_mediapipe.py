"""
Prepare Phoenix-2014T MediaPipe keypoints in the format expected by MSKA-SLR.

Input:
  - keypoints-kaggle/<split>/<vid>.pkl  →  dict with key "keypoints":
      ndarray (T, 75, C) with C = 2 (x, y) or 3 (x, y, confidence)
      joints: 0-32 = body (MediaPipe Pose), 33-53 = left hand, 54-74 = right hand
  - Annotations CSV: name|video|start|end|speaker|orth|translation

Output:
  - dataset/pose/phoenix2014t_75kp/Phoenix-2014T.{train,dev,test}
    plain pickle, dict keyed by video name:
      { name, gloss, num_frames, keypoint: FloatTensor(T, 75, C), text }

  - dataset/pose/phoenix2014t_75kp/gloss2ids.pkl
    dict {gloss_token: int_id}  (with <blank>=0, <unk>=1)

The channel count is passed through unchanged. skeleton.generate_tssi_75 accepts
both: with C = 2 it substitutes a constant confidence channel. The released
teacher checkpoint was trained on C = 3, so 2-channel keypoints will train but
will not reproduce the published WER.

Usage:
  python prepare_phoenix_mediapipe.py
  python train.py --config configs/phoenix-2014t_mediapipe.yaml
"""

import os
import pickle
import numpy as np
import torch
import pandas as pd
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
# Repo layout: <root>/code/csrl_skeleton/prepare_phoenix_mediapipe.py
# PHOENIX_RAW_ROOT must hold the downloaded keypoints; everything else defaults
# inside the repo. The output lands where config.py and extract_skeleton_feats.py
# both expect it (dataset/pose/phoenix2014t_75kp), so no copying is needed after.
REPO_ROOT    = Path(os.environ.get("PHOENIX_ROOT", Path(__file__).resolve().parents[2]))
DATASET_ROOT = Path(os.environ.get("PHOENIX_RAW_ROOT", REPO_ROOT / "dataset"))
KP_ROOT      = Path(os.environ.get("PHOENIX_KEYPOINTS", DATASET_ROOT / "keypoints-kaggle"))
ANN_ROOT     = REPO_ROOT / "dataset" / "annotations" / "manual"
OUT_DIR      = Path(os.environ.get("MSKA_DATA_DIR",
                                   REPO_ROOT / "dataset" / "pose" / "phoenix2014t_75kp"))

SPLITS = {
    "train": ("train", "PHOENIX-2014-T.train.corpus.csv"),
    "dev":   ("dev",   "PHOENIX-2014-T.dev.corpus.csv"),   # keypoints folder is named "dev"
    "test":  ("test",  "PHOENIX-2014-T.test.corpus.csv"),
}
# -----------------------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(split_name: str, kp_split: str, ann_file: str):
    ann_path = ANN_ROOT / ann_file
    df = pd.read_csv(ann_path, sep="|")

    kp_dir   = KP_ROOT / kp_split
    samples  = {}
    missing  = 0

    for _, row in df.iterrows():
        name   = row["name"]
        gloss  = row["orth"]
        text   = row["translation"]

        pkl_path = kp_dir / f"{name}.pkl"
        if not pkl_path.exists():
            print(f"  [WARN] missing keypoints for {name}")
            missing += 1
            continue

        with open(str(pkl_path), "rb") as f:
            kp_data = pickle.load(f)

        kp = kp_data["keypoints"]          # (T, 75, 2)  float or int
        kp = np.array(kp, dtype=np.float32)

        # Normalise by frame dimensions if stored in pixel coords
        # (kaggle pkl stores raw pixel coords; w/h are in the dict when present)
        if "w" in kp_data and "h" in kp_data:
            w, h = float(kp_data["w"]), float(kp_data["h"])
            kp[..., 0] /= w   # x → [0,1]
            kp[..., 1] /= h   # y → [0,1]
        else:
            # Assume 210×260 Phoenix frame size
            kp[..., 0] /= 210.0
            kp[..., 1] /= 260.0

        keypoint = torch.from_numpy(kp)  # FloatTensor (T, 75, 2)

        samples[name] = {
            "name":       name,
            "gloss":      gloss,
            "num_frames": kp.shape[0],
            "keypoint":   keypoint,
            "text":       text,
        }

    out_path = OUT_DIR / f"Phoenix-2014T.{split_name}"
    with open(str(out_path), "wb") as f:
        pickle.dump(samples, f)

    print(f"[{split_name}] {len(samples)} samples written to {out_path}  (missing: {missing})")
    return samples


def build_gloss_vocab(all_samples: dict):
    """Build gloss→id vocab from all splits combined."""
    # <si> must be index 0 (silence/blank for CTC), <unk> index 1, <pad> index 2
    vocab = {"<si>": 0, "<unk>": 1, "<pad>": 2}
    for sample in all_samples.values():
        for tok in sample["gloss"].split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    out_path = OUT_DIR / "gloss2ids.pkl"
    with open(str(out_path), "wb") as f:
        pickle.dump(vocab, f)
    print(f"Vocab size: {len(vocab)}  →  {out_path}")


if __name__ == "__main__":
    all_samples = {}
    for split_name, (kp_split, ann_file) in SPLITS.items():
        samples = load_split(split_name, kp_split, ann_file)
        all_samples.update(samples)

    build_gloss_vocab(all_samples)
    print("\nDone — run: python train.py --config configs/phoenix-2014t_mediapipe.yaml")

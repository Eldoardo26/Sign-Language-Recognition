"""Pack the per-video 133-kp extraction into the split pickles the trainer reads.

Input : phoenix_dataset/PHOENIX-2014-T/keypoints-133-extracted/<split>/<name>.pkl
Output: <DATA_DIR>/Phoenix-2014T.{train,dev,test}
        dict keyed by video name -> { name, gloss, text, keypoint (T,133,3),
        num_frames }

DATA_DIR comes from config (phoenix2014t_133kp when KP_LAYOUT=coco133), so the
runner picks it up with no extra wiring. Videos not yet extracted are skipped
with a warning, so this can be run on partial data to sanity-check the format.

Usage:  python prepare_phoenix_133.py
"""

import os
import pickle

import numpy as np
import pandas as pd

from config import DATA_DIR  # phoenix2014t_133kp for the coco133 layout

PHOENIX = os.environ.get(
    "PHOENIX_ROOT",
    r"C:\Users\edoar\Downloads\phoenix_dataset\PHOENIX-2014-T",
)
KP_DIR = os.path.join(PHOENIX, "keypoints-133-extracted")
ANN = os.path.join(PHOENIX, "annotations", "manual")

SPLITS = {
    "train": "PHOENIX-2014-T.train.corpus.csv",
    "dev":   "PHOENIX-2014-T.dev.corpus.csv",
    "test":  "PHOENIX-2014-T.test.corpus.csv",
}


def build_split(split: str, csv: str) -> int:
    df = pd.read_csv(os.path.join(ANN, csv), sep="|")
    src = os.path.join(KP_DIR, split)
    samples, missing = {}, 0

    for _, row in df.iterrows():
        name = str(row["name"])
        p = os.path.join(src, name + ".pkl")
        if not os.path.exists(p):
            missing += 1
            continue
        with open(p, "rb") as f:
            rec = pickle.load(f)
        kp = np.asarray(rec["keypoints"], dtype=np.float32)   # (T,133,3)
        samples[name] = {
            "name": name,
            "gloss": str(row["orth"]),
            "text": str(row["translation"]),
            "keypoint": kp,
            "num_frames": int(kp.shape[0]),
        }

    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, f"Phoenix-2014T.{split}")
    with open(out, "wb") as f:
        pickle.dump(samples, f)
    print(f"[{split}] {len(samples)} samples -> {out}  (missing: {missing})")
    return len(samples)


if __name__ == "__main__":
    print("DATA_DIR:", DATA_DIR)
    for split, csv in SPLITS.items():
        build_split(split, csv)
    print("done")

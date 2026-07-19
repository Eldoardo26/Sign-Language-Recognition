"""Extract 133-keypoint COCO-WholeBody poses (RTMW / DWPose, "performance" mode)
from the raw PHOENIX-2014T frames, replacing the noisy MediaPipe 75-kp.

Why: all skeleton-CSLR SOTA (CoSign, MSKA, STARK) use HRNet/RTMPose whole-body
keypoints; MediaPipe hands are the accuracy ceiling of the current teacher.

Layout of the 133 COCO-WholeBody joints:
    0-16   body (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
    17-22  feet
    23-90  face (68)
    91-111 left hand (21)
    112-132 right hand (21)

The signer is centred in the 210x260 crop, so we skip per-frame person detection
and run the pose model directly on a full-frame bbox (quality is identical to the
detector path — median <1px — at ~3x the speed).

Output: keypoints-133-extracted/<split>/<name>.pkl
    dict{ name, gloss, text, speaker, keypoints float32 (T,133,3)=[x,y,score],
          w, h, n_frames }
Resumable: a video whose .pkl already exists is skipped.

Usage:
    python extract_wholebody_133.py                 # all splits
    python extract_wholebody_133.py --splits dev    # one split
    python extract_wholebody_133.py --limit 3       # smoke test (first N videos)
"""

import os
import sys
import glob
import time
import pickle
import argparse
from concurrent.futures import ThreadPoolExecutor

import onnxruntime as ort
ort.preload_dlls()  # load CUDA/cuDNN from torch so the CUDA EP is available
import cv2
import numpy as np
import pandas as pd
from rtmlib import Wholebody

# The pose call is CPU/IO-bound (PNG decode + rtmlib pre/post-processing) while the
# GPU idles; onnxruntime releases the GIL during inference, so a small thread pool
# overlaps decode+preprocess with inference and ~3.5x's throughput (bit-identical
# results). 5 workers is the sweet spot on the RTX 4050 laptop.
N_WORKERS = 5

BASE = os.environ.get(
    "PHOENIX_ROOT",
    r"C:\Users\edoar\Downloads\phoenix_dataset\PHOENIX-2014-T",
)
FRAMES = os.path.join(BASE, "features", "fullFrame-210x260px")
ANN = os.path.join(BASE, "annotations", "manual")
OUT = os.path.join(BASE, "keypoints-133-extracted")
LOG = os.path.join(OUT, "extract.log")

SPLITS = {
    "train": "PHOENIX-2014-T.train.corpus.csv",
    "dev":   "PHOENIX-2014-T.dev.corpus.csv",
    "test":  "PHOENIX-2014-T.test.corpus.csv",
}


def log(msg: str):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _infer_frame(wb, fp, bbox):
    img = cv2.imread(fp)
    if img is None:
        return None
    k, s = wb.pose_model(img, bboxes=bbox)
    k = np.asarray(k[0], dtype=np.float32)                  # (133,2)
    s = np.asarray(s[0], dtype=np.float32).reshape(-1, 1)   # (133,1)
    return np.concatenate([k, s], axis=1)                   # (133,3)


def process_split(wb, ex, split: str, csv: str, limit: int = 0):
    df = pd.read_csv(os.path.join(ANN, csv), sep="|")
    if limit:
        df = df.iloc[:limit]
    outdir = os.path.join(OUT, split)
    os.makedirs(outdir, exist_ok=True)
    fdir = os.path.join(FRAMES, split)

    total = len(df)
    done = skipped = missing = 0
    t0 = time.time()
    nf = 0
    log(f"[{split}] start — {total} videos")

    for _, row in df.iterrows():
        name = str(row["name"])
        outp = os.path.join(outdir, name + ".pkl")
        if os.path.exists(outp):
            skipped += 1
            done += 1
            continue

        frames = sorted(glob.glob(os.path.join(fdir, name, "images*.png")))
        if not frames:
            missing += 1
            log(f"[WARN] no frames: {split}/{name}")
            continue

        try:
            first = cv2.imread(frames[0])
            if first is None:
                raise RuntimeError(f"cannot read {frames[0]}")
            H, W = first.shape[:2]
            bbox = np.array([[0, 0, W, H]], dtype=np.float32)

            # threaded, order-preserving: decode+infer frames concurrently
            results = list(ex.map(lambda fp: _infer_frame(wb, fp, bbox), frames))
            kps = []
            for r in results:
                if r is None:
                    kps.append(kps[-1] if kps else np.zeros((133, 3), np.float32))
                else:
                    kps.append(r)
            arr = np.stack(kps).astype(np.float32)              # (T,133,3)
        except Exception as e:
            missing += 1
            log(f"[ERROR] {split}/{name}: {e}")
            continue

        rec = {
            "name": name,
            "gloss": str(row["orth"]),
            "text": str(row["translation"]),
            "speaker": str(row["speaker"]),
            "keypoints": arr,
            "w": int(W),
            "h": int(H),
            "n_frames": int(arr.shape[0]),
        }
        tmp = outp + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(rec, f)
        os.replace(tmp, outp)

        done += 1
        nf += arr.shape[0]
        if (done - skipped) % 25 == 0 and (done - skipped) > 0:
            el = time.time() - t0
            fps = nf / el if el > 0 else 0
            remain = (total - done) * (el / max(done - skipped, 1))
            log(f"[{split}] {done}/{total} | {fps:.1f} fps | "
                f"elapsed {el/60:.1f}m | eta {remain/60:.1f}m")

    log(f"[{split}] DONE — processed {done}/{total} "
        f"(skipped {skipped}, missing {missing})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=0, help="first N videos per split")
    args = ap.parse_args()

    log(f"providers: {ort.get_available_providers()}")
    wb = Wholebody(mode="performance", backend="onnxruntime", device="cuda")
    log(f"pose model ready (RTMW performance, full-frame bbox, {N_WORKERS} workers)")

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for split in args.splits:
            process_split(wb, ex, split, SPLITS[split], limit=args.limit)
    log("ALL DONE")


if __name__ == "__main__":
    main()

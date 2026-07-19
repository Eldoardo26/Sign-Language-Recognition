# -*- coding: utf-8 -*-
"""
extract_skeleton_feats.py — export the encoder features of the skeleton teacher,
for cross-modal distillation into the Signformer student.

Uses THE SAME modules as the trainer (csrl_skeleton is put on sys.path): the
keypoint layout comes from KP_LAYOUT (default "coco133" -> 55 joints, RTMW), and
the model is rebuilt from the cfg stored inside the checkpoint (spatial GCN
included). The previous version kept a duplicated copy of skeleton+model, which
silently drifted out of sync with the trainer — never again.

For every video, the teacher's pre-classifier feature sequence is saved.
Output: skeleton_feats_133/{train,dev,test}.pkl = {video_name: np.float16 (T, 512)}
(a new folder, so the old MediaPipe-teacher features stay untouched).

Env overrides:
    PHOENIX_ROOT       : top-level folder (default: two levels above this file)
    MSKA_DATA_DIR      : folder with Phoenix-2014T.{split} pickles (default 133kp)
    TEACHER_CKPT       : teacher checkpoint (default tssi133_gcn_best.pt)
    SKELETON_FEATS_DIR : output folder
    KP_LAYOUT          : "coco133" (default) or "mediapipe75"

Usage:
    python extract_skeleton_feats.py                # all splits
    python extract_skeleton_feats.py --splits dev --limit 2   # smoke test
"""
import os
import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import numpy as np
import torch
torch.backends.cudnn.enabled = False      # ← QUESTA riga
# make the trainer's modules importable (skeleton dispatcher, model)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "csrl_skeleton"))
from skeleton import generate_tssi_75, NUM_JOINTS, LAYOUT   # noqa: E402
from model import PoseNetworkCTC                            # noqa: E402

# ============================================================
# CONFIG
# ============================================================
PHOENIX_ROOT = Path(os.environ.get("PHOENIX_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = os.environ.get(
    "MSKA_DATA_DIR", str(PHOENIX_ROOT / "dataset" / "pose" / "phoenix2014t_133kp"))
CKPT_PATH = os.environ.get(
    "TEACHER_CKPT", str(PHOENIX_ROOT / "dataset" / "checkpoints" / "tssi133_gcn_best.pt"))
OUT_DIR = os.environ.get(
    "SKELETON_FEATS_DIR", str(PHOENIX_ROOT / "dataset" / "features" / "skeleton_feats_133"))
MAX_FRAMES = 400
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_teacher():
    ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    cfg = ck.get("cfg", {})

    num_classes = sd["fc.weight"].shape[0]
    hidden = sd["fc.weight"].shape[1] // 2
    use_gcn = any(k.startswith("gcn.") for k in sd)   # robust even without cfg

    n_joints = cfg.get("num_joints", NUM_JOINTS)
    if n_joints != NUM_JOINTS:
        raise SystemExit(
            f"checkpoint was trained with num_joints={n_joints} but the active "
            f"skeleton layout '{LAYOUT}' has NUM_JOINTS={NUM_JOINTS}. "
            f"Set KP_LAYOUT accordingly (coco133 <-> mediapipe75) and retry.")

    model = PoseNetworkCTC(
        num_classes=num_classes, in_channels=n_joints * 3,
        hidden_dim=cfg.get("hidden_dim", hidden),
        tcn_blocks=cfg.get("tcn_blocks", 3),
        lstm_layers=cfg.get("num_layers", 3),
        dropout=cfg.get("dropout", 0.3),
        drop_path_rate=cfg.get("drop_path_rate", 0.1),
        attn_heads=cfg.get("attn_heads", 4),
        use_gcn=cfg.get("use_gcn", use_gcn),
        gcn_channels=cfg.get("gcn_channels", 16),
    ).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()
    print(f"teacher loaded | layout={LAYOUT} joints={NUM_JOINTS} | "
          f"classes={num_classes} | feat_dim={hidden*2} | use_gcn={use_gcn} | {DEVICE}")
    return model


@torch.no_grad()
def extract_split(model, split: str, limit: int = 0):
    with open(os.path.join(DATA_DIR, f"Phoenix-2014T.{split}"), "rb") as f:
        raw = pickle.load(f)
    items = list(raw.items())
    if limit:
        items = items[:limit]

    feats = {}
    for i, (name, s) in enumerate(items):
        kp = np.asarray(s["keypoint"], dtype=np.float32)        # (T, J_full, C)
        T = kp.shape[0]
        if T > MAX_FRAMES:
            sel = np.linspace(0, T - 1, MAX_FRAMES).round().astype(int)
            kp = kp[sel]
        tssi = generate_tssi_75(kp)                             # (3, NUM_JOINTS, T)
        x = torch.from_numpy(tssi).unsqueeze(0).float().to(DEVICE)
        f = model.forward_feat(x)[0].cpu().numpy().astype(np.float16)   # (T, 512)
        feats[name] = f
        if (i + 1) % 500 == 0:
            print(f"  [{split}] {i+1}/{len(items)}")

    out = os.path.join(OUT_DIR, f"{split}.pkl")
    with open(out, "wb") as fp:
        pickle.dump(feats, fp)
    print(f"[{split}] wrote {len(feats)} feature sequences -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=0, help="first N videos per split")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"data: {DATA_DIR}\nckpt: {CKPT_PATH}\nout : {OUT_DIR}")
    model = load_teacher()
    for split in args.splits:
        extract_split(model, split, limit=args.limit)
    print("Done. Teacher features written to", OUT_DIR)

"""Global configuration, device setup, and reproducibility seed."""

import os
import random
from pathlib import Path

import numpy as np
import torch

from skeleton import NUM_JOINTS as SKELETON_NUM_JOINTS, LAYOUT as KP_LAYOUT

torch.backends.cudnn.enabled = False

def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(42)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fp16 autocast makes the GCN model diverge (NaN at the phase-2 unfreeze:
# tcn_in is 880 channels and overflows half precision). Keep AMP OFF for this
# model; export USE_AMP=1 only if you know what you are doing.
AMP = torch.cuda.is_available() and os.environ.get("USE_AMP", "0") == "1"

# Repo layout: <root>/code/csrl_skeleton/config.py -> root is three levels up.
# Override either path with an environment variable if your data lives elsewhere;
# the names match the ones read by distillation/extract_skeleton_feats.py, so the
# teacher and the feature exporter always agree on where the data is.
REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_DATA = "phoenix2014t_133kp" if KP_LAYOUT == "coco133" else "phoenix2014t_75kp"
DATA_DIR = os.environ.get(
    "MSKA_DATA_DIR", str(REPO_ROOT / "dataset" / "pose" / _DEFAULT_DATA)
)
CKPT_PATH = os.environ.get(
    "TEACHER_CKPT", str(REPO_ROOT / "dataset" / "checkpoints" / "tssi75_cslr_best.pt")
)
os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

CFG = {
    "num_joints": SKELETON_NUM_JOINTS,   # 55 (coco133) or 67 (mediapipe75) via KP_LAYOUT
    # architecture
    "hidden_dim": 256,
    "tcn_blocks": 3,
    "num_layers": 3,
    "attn_heads": 4,
    # D1: spatial graph-conv front-end over the joints (uses skeleton.ADJACENCY).
    # Set use_gcn=False to fall back to the flat-channel baseline. Lower
    # gcn_channels (e.g. 8) if VRAM is tight — tcn_in = gcn_channels * num_joints.
    "use_gcn": True,
    "gcn_channels": 16,
    "dropout": 0.317638942624273,
    "drop_path_rate": 0.1,
    # training
    "batch_size": 16,
    "grad_accum": 1,
    "num_epochs": 75,
    "early_stopping_patience": 20,
    "phase1_epochs": 15,
    "phase1_lr": 0.0001619145621568752,
    # Phase-2 LRs VALIDATED by A/B on the full run (coco133+GCN, AMP off):
    # head 2.5e-4 -> 41.76 dev; the old MediaPipe-tuned 5.04e-4 -> ~44.6 dev.
    # (The 28-epoch Optuna screening prefers the higher LR — and is wrong.)
    "phase2_lr_backbone": 1.0e-4,
    "phase2_lr_head": 2.5e-4,
    "weight_decay": 0.0002465667957406484,
    "grad_clip": 1.0,

    # ctc / decoding
    "ctc_smoothing": 0.10231111042825411,
    "prior_beta": 0.32404536014294083,
    "beam_width": 25,
    # input
    "max_frames": 400,
    "augment": True,
}

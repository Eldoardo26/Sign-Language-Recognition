"""Global configuration, device setup, and reproducibility seed."""

import os
import random
from pathlib import Path

import numpy as np
import torch

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
AMP = torch.cuda.is_available()

# Repo layout: <root>/code/csrl_skeleton/config.py -> root is three levels up.
# Override either path with an environment variable if your data lives elsewhere;
# the names match the ones read by distillation/extract_skeleton_feats.py, so the
# teacher and the feature exporter always agree on where the data is.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = os.environ.get(
    "MSKA_DATA_DIR", str(REPO_ROOT / "dataset" / "pose" / "phoenix2014t_75kp")
)
CKPT_PATH = os.environ.get(
    "TEACHER_CKPT", str(REPO_ROOT / "dataset" / "checkpoints" / "tssi75_cslr_best.pt")
)
os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

CFG = {
    "num_joints": 75,
    # architecture
    "hidden_dim": 256,
    "tcn_blocks": 3,
    "num_layers": 3,
    "attn_heads": 4,
    "dropout": 0.317638942624273,
    "drop_path_rate": 0.1,
    # training
    "batch_size": 16,
    "grad_accum": 1,
    "num_epochs": 75,
    "early_stopping_patience": 20,
    "phase1_epochs": 15,
    "phase1_lr": 0.0001619145621568752,
    "phase2_lr_backbone": 0.00021884694140182422,
    "phase2_lr_head": 0.0005038675815696706,
    "weight_decay": 0.0002465667957406484,
    "grad_clip": 3.3943208852931304,
    # ctc / decoding
    "ctc_smoothing": 0.10231111042825411,
    "prior_beta": 0.32404536014294083,
    "beam_width": 25,
    # input
    "max_frames": 400,
    "augment": True,
}

"""Global configuration, device setup, and reproducibility seed."""

import os
import random
import numpy as np
import torch
import yaml

torch.backends.cudnn.deterministic = True


def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


set_seed(42)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()

CFG_FILE = "configs/sign.yaml"
MODEL_DIR = None  # set from YAML at load time


def load_cfg(cfg_file: str = CFG_FILE) -> dict:
    """Load YAML config and return the full dict."""
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)
    global MODEL_DIR
    MODEL_DIR = cfg["training"].get("model_dir", "sign_sample")
    return cfg


def save_cfg(cfg: dict, cfg_file: str = CFG_FILE):
    """Write config dict back to YAML."""
    with open(cfg_file, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Config saved to {cfg_file}")


def print_cfg(cfg: dict):
    """Print key training/model parameters."""
    tr = cfg["training"]
    enc = cfg["model"]["encoder"]
    print(f"  batch_size      : {tr['batch_size']}")
    print(f"  optimizer       : {tr['optimizer']}")
    print(f"  learning_rate   : {tr.get('learning_rate', 'N/A')}")
    print(f"  weight_decay    : {tr.get('weight_decay', 'N/A')}")
    print(f"  epochs          : {tr['epochs']}")
    print(f"  early_stop      : {tr.get('early_stopping_patience', 'N/A')} epochs")
    print(f"  hidden_size     : {enc['hidden_size']}")
    print(f"  encoder_layers  : {enc['num_layers']}")
    print(f"  encoder_dropout : {enc.get('dropout', 'N/A')}")
    print(f"  eval_metric     : {tr.get('eval_metric', 'bleu')}")
    print(f"  recognition_wt  : {tr.get('recognition_loss_weight', 1.0)}")
    print(f"  translation_wt  : {tr.get('translation_loss_weight', 0.0)}")

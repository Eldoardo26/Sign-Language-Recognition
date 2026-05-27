"""
config.py — Configurazione centralizzata del progetto PHOENIX-2014-T CSLR.
Modifica qui percorsi e iperparametri prima di avviare il training.
"""

import os
import torch

# ============================================================
# PERCORSI
# ============================================================
BASE_PATH       = "/home/ebufi/phoenix"
ANNOTATIONS_DIR = os.path.join(BASE_PATH, "annotations", "annotations", "manual")
POSE_DIR        = os.path.join(BASE_PATH, "pose", "pose")
RESULTS_DIR     = os.path.join(BASE_PATH, "results")
TSSI_OUTPUT_DIR = os.path.join(BASE_PATH, "tssi_cache")

os.makedirs(RESULTS_DIR,     exist_ok=True)
os.makedirs(TSSI_OUTPUT_DIR, exist_ok=True)

# ============================================================
# IPERPARAMETRI
# ============================================================
CONFIG = {
    # Training generale
    "num_epochs":               150,
    "early_stopping_patience":  20,
    "checkpoint_every":         1,
    "keep_last_n_checkpoints":  10,

    # Batch e ottimizzazione
    "batch_size":                   16,
    "gradient_accumulation_steps":  1,
    "use_amp":                      True,

    # Architettura
    "hidden_dim":    256,
    "tcn_blocks":    3,
    "num_layers":    3,
    "num_joints":    48,
    "attn_heads":    4,
    "drop_path_rate": 0.1,

    # Regolarizzazione
    "dropout":       0.317638942624273,
    "weight_decay":  0.0002465667957406484,
    "grad_clip":     3.3943208852931304,

    # LR per le due fasi
    "phase1_epochs":          15,
    "phase1_lr":              0.0001619145621568752,
    "phase2_lr_backbone":     0.00021884694140182422,
    "phase2_lr_head":         0.0005038675815696706,

    # Prior scaling (Deep Sign Eq.13)
    "prior_beta":    0.32404536014294083,

    # CTC
    "ctc_smoothing": 0.10231111042825411,
    "beam_width":    25,

    # Input
    "frame_h":       240,

    # Gloss merging
    "use_gloss_merge": True,
    "merge_map_path":  "/home/ebufi/phoenix/working/gloss_merge_map.json",

    # Ensemble
    "ensemble_n":    3,

    # Anti-overfitting (Optuna objective)
    "overfitting_lambda": 1.1575402465259297,

    # Debug / reporting
    "debug_training":        False,
    "debug_every_n_batches":  28,
    "debug_preview_batches":  2,
    "debug_preview_samples":  3,
    "debug_topk_classes":     8,
    "debug_save_reports":     True,
}

# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Annotazioni: {ANNOTATIONS_DIR}")
    print(f"Keypoint:    {POSE_DIR}")
    print(f"Risultati:   {RESULTS_DIR}")
    print(f"Cache TSSI:  {TSSI_OUTPUT_DIR}")

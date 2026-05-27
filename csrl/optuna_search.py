"""
optuna_search.py — Ricerca iperparametri con Optuna (anti-overfitting).

Obiettivo composito:
    score = dev_wer + λ * max(0, dev_wer - train_wer - gap_tolerance)

Testa entrambe le fasi del two-phase finetuning su un subset del dataset.

Uso:
    from optuna_search import run_optuna_search
    best_params = run_optuna_search(train_ds, dev_ds, num_classes, n_trials=30)
"""

import gc
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import Subset, DataLoader

from config import CONFIG, DEVICE
from model import PoseNetworkCTC
from losses import CTCLossWithEntropy
from training import (
    freeze_backbone, unfreeze_backbone,
    make_optimizer_phase1, make_optimizer_phase2,
    make_scheduler, run_epoch,
)
from utils import cleanup_cuda

try:
    import optuna
except ImportError as exc:
    raise RuntimeError("Installa optuna con: pip install optuna") from exc


# ============================================================
# UTILITÀ
# ============================================================

def _make_subset(dataset, frac: float = 0.5, seed: int = 42) -> Subset:
    n   = len(dataset)
    k   = max(1, int(n * frac))
    idx = np.random.default_rng(seed).choice(n, size=k, replace=False)
    return Subset(dataset, idx)


def build_model_from_cfg(cfg: dict, num_classes: int) -> PoseNetworkCTC:
    return PoseNetworkCTC(
        num_classes=num_classes,
        num_joints=cfg["num_joints"],
        hidden_dim=cfg["hidden_dim"],
        tcn_blocks=cfg["tcn_blocks"],
        lstm_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(DEVICE)


# ============================================================
# OBJECTIVE
# ============================================================

def make_objective(train_ds, dev_ds, num_classes: int, i2g: dict, log_prior: torch.Tensor,
                   num_workers: int = 4):
    """Restituisce la funzione obiettivo compatibile con optuna.create_study."""

    def optuna_objective(trial):
        cleanup_cuda()
        cfg = dict(CONFIG)

        # Spazio di ricerca
        cfg["hidden_dim"]   = trial.suggest_categorical("hidden_dim", [128, 192, 256])
        cfg["tcn_blocks"]   = trial.suggest_int("tcn_blocks",   3, 5)
        cfg["num_layers"]   = trial.suggest_int("num_layers",   1, 3)
        cfg["dropout"]      = trial.suggest_float("dropout",    0.25, 0.55)
        cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-4, 5e-3, log=True)
        cfg["grad_clip"]    = trial.suggest_float("grad_clip",  1.0, 5.0)
        cfg["prior_beta"]   = trial.suggest_float("prior_beta", 0.1, 0.7)
        cfg["ctc_smoothing"]= trial.suggest_float("ctc_smoothing", 0.03, 0.15)
        cfg["phase1_lr"]    = trial.suggest_float("phase1_lr",  1e-4, 8e-4, log=True)
        cfg["phase2_lr_backbone"] = trial.suggest_float("phase2_lr_backbone", 1e-4, 1e-3, log=True)
        cfg["phase2_lr_head"]     = trial.suggest_float("phase2_lr_head",     1e-4, 1e-3, log=True)
        overfitting_lambda = trial.suggest_float("overfitting_lambda", 0.3, 1.5)

        gap_tolerance          = 0.08
        cfg["batch_size"]      = min(cfg["batch_size"], 8)
        cfg["gradient_accumulation_steps"] = 1
        PHASE1_EP, PHASE2_EP   = 15, 25

        try:
            model_trial     = build_model_from_cfg(cfg, num_classes)
            criterion_trial = CTCLossWithEntropy(blank=0, entropy_weight=cfg["ctc_smoothing"])
            scaler_trial    = GradScaler(enabled=cfg["use_amp"] and DEVICE.type == "cuda")

            train_sub = _make_subset(train_ds, frac=0.5, seed=trial.number + 7)
            dev_sub   = _make_subset(dev_ds,   frac=0.5, seed=trial.number + 11)
            from dataset import collate_fn_ctc
            train_dl  = DataLoader(train_sub, batch_size=cfg["batch_size"], shuffle=True,
                                   num_workers=num_workers, pin_memory=True,
                                   collate_fn=collate_fn_ctc, drop_last=True)
            dev_dl    = DataLoader(dev_sub, batch_size=cfg["batch_size"], shuffle=False,
                                   num_workers=num_workers, pin_memory=True,
                                   collate_fn=collate_fn_ctc)

            best_score   = float("inf")
            global_epoch = 0

            # Fase 1: backbone congelato
            freeze_backbone(model_trial)
            opt_p1 = make_optimizer_phase1(model_trial, lr=cfg["phase1_lr"],
                                           weight_decay=cfg["weight_decay"])
            sch_p1 = make_scheduler(opt_p1, PHASE1_EP, len(train_dl))

            for ep in range(PHASE1_EP):
                global_epoch += 1
                (train_loss, train_wer, *_) = run_epoch(
                    model_trial, train_dl, criterion_trial,
                    opt_p1, scaler_trial, sch_p1,
                    training=True, device=DEVICE, config=cfg,
                    log_prior=log_prior, i2g=i2g)
                (_, dev_wer, *_) = run_epoch(
                    model_trial, dev_dl, criterion_trial,
                    opt_p1, scaler_trial, sch_p1,
                    training=False, device=DEVICE, config=cfg,
                    log_prior=log_prior, i2g=i2g)

                overfit_gap = max(0.0, dev_wer - train_wer - gap_tolerance)
                score       = dev_wer + overfitting_lambda * overfit_gap
                best_score  = min(best_score, score)
                trial.report(score, global_epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # Fase 2: tutto sbloccato
            unfreeze_backbone(model_trial)
            opt_p2 = make_optimizer_phase2(
                model_trial,
                lr_backbone=cfg["phase2_lr_backbone"],
                lr_head=cfg["phase2_lr_head"],
                weight_decay=cfg["weight_decay"],
            )
            sch_p2 = make_scheduler(opt_p2, PHASE2_EP, len(train_dl))

            for ep in range(PHASE2_EP):
                global_epoch += 1
                (train_loss, train_wer, *_) = run_epoch(
                    model_trial, train_dl, criterion_trial,
                    opt_p2, scaler_trial, sch_p2,
                    training=True, device=DEVICE, config=cfg,
                    log_prior=log_prior, i2g=i2g)
                (_, dev_wer, *_) = run_epoch(
                    model_trial, dev_dl, criterion_trial,
                    opt_p2, scaler_trial, sch_p2,
                    training=False, device=DEVICE, config=cfg,
                    log_prior=log_prior, i2g=i2g)

                overfit_gap = max(0.0, dev_wer - train_wer - gap_tolerance)
                score       = dev_wer + overfitting_lambda * overfit_gap
                best_score  = min(best_score, score)
                trial.report(score, global_epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return best_score

        except torch.OutOfMemoryError:
            print(f"[Trial {trial.number}] OOM — scartato.")
            cleanup_cuda()
            return float("inf")
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            print(f"[Trial {trial.number}] Errore: {e}")
            cleanup_cuda()
            return float("inf")
        finally:
            cleanup_cuda()

    return optuna_objective


# ============================================================
# ENTRY POINT
# ============================================================

def run_optuna_search(
    train_ds, dev_ds, num_classes: int, i2g: dict,
    log_prior: torch.Tensor, n_trials: int = 30,
    num_workers: int = 4,
) -> dict:
    """
    Lancia la ricerca Optuna e stampa il trial migliore.

    Returns:
        dict — parametri del trial migliore
    """
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4),
    )
    objective = make_objective(train_ds, dev_ds, num_classes, i2g, log_prior, num_workers)

    print(f"Inizio ricerca ({n_trials} trial, two-phase, obiettivo anti-overfitting)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print("\n" + "=" * 55)
    print(f"Trial migliore: #{best.number}  |  Score: {best.value:.4f}")
    print("Parametri da copiare nel CONFIG:")
    for k, v in best.params.items():
        print(f"    '{k}': {v},")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("\nTop 5 trial per score:")
    for t in sorted(completed, key=lambda x: x.value)[:5]:
        print(f"  Trial #{t.number:3d}  score={t.value:.4f}  "
              f"dropout={t.params.get('dropout', '?'):.2f}  "
              f"wd={t.params.get('weight_decay', '?'):.1e}  "
              f"lambda={t.params.get('overfitting_lambda', '?'):.2f}")
    print("=" * 55)

    return best.params

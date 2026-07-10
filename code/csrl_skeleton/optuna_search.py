"""Optuna hyperparameter search with progress-file resume."""

import os
import gc
import json
import copy
import time
import torch
import torch.nn as nn

from model import PoseNetworkCTC
from losses import CTCLossWithEntropy, greedy_decode, compute_wer
from training import (freeze_backbone, unfreeze_backbone,
                      make_opt_phase1, make_opt_phase2, make_sched)


def run_optuna_search(cfg: dict, train_loader, dev_loader,
                      num_classes: int, log_prior, device, amp: bool,
                      n_trials: int = 4, trial_train_epochs: int = 28,
                      progress_file: str = "optuna_csrl_progress.json"):
    """Optuna search that screens configs on the FIRST `trial_train_epochs` of
    the real full-length run.

    The cosine LR schedule is built on the full horizon (cfg["num_epochs"]), so
    during those early epochs each trial follows the exact same LR trajectory it
    would have in the real run — then we stop early and rank by dev WER. This
    avoids the proxy mismatch of a short cosine (which would tune the LR for a
    schedule the real run never uses)."""

    try:
        import optuna
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "optuna"])
        import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from optuna.trial import create_trial
    from optuna.distributions import FloatDistribution

    # Full-run horizon (the real training length) — used to BUILD the cosine
    # LR schedule so each trial's early LR matches the real run.
    full_p1 = cfg["phase1_epochs"]
    full_p2 = cfg["num_epochs"] - cfg["phase1_epochs"]
    # Epochs actually TRAINED per trial: the first `trial_train_epochs` of the
    # full-horizon schedule (phase1 first, then phase2 with the leftover budget).
    p1_train = min(full_p1, trial_train_epochs)
    p2_train = max(0, min(full_p2, trial_train_epochs - full_p1))

    distributions = {
        "phase1_lr":      FloatDistribution(5e-5, 5e-4, log=True),
        "phase2_lr_head": FloatDistribution(1e-4, 8e-4, log=True),
        "dropout":        FloatDistribution(0.2, 0.45),
        "weight_decay":   FloatDistribution(1e-5, 1e-3, log=True),
    }

    optuna_log = []
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            optuna_log = json.load(f)
        print(f"[Optuna] RESUMED — {len(optuna_log)}/{n_trials} trials loaded from {progress_file}")
        for r in optuna_log:
            print(f"  trial {r['trial']} | WER {r['wer']:.2f}% | "
                  f"p1_lr={r['phase1_lr']:.6f} p2_lr_h={r['phase2_lr_head']:.6f} "
                  f"drop={r['dropout']:.4f} wd={r['weight_decay']:.6f}")
    else:
        print("[Optuna] CSRL hyperparameter search (fresh start)")

    print(f"  trials       : {n_trials} ({n_trials - len(optuna_log)} remaining)")
    print(f"  horizon      : {cfg['num_epochs']} epochs (phase1 {full_p1} + phase2 {full_p2}) — LR schedule built on this")
    print(f"  train/trial  : {trial_train_epochs} epochs (phase1 {p1_train} + phase2 {p2_train}) — early screening")
    print(f"  searching    : phase1_lr, phase2_lr_head, dropout, weight_decay")
    print(f"  progress     : {progress_file}")
    print("=" * 70)

    def _save_progress():
        with open(progress_file, "w") as f:
            json.dump(optuna_log, f, indent=2)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    for rec in optuna_log:
        study.add_trial(create_trial(
            params=rec["params"], distributions=distributions,
            values=[rec["value"]],
        ))

    def _trial_wer(trial):
        t0 = time.time()
        cfg_t = copy.deepcopy(cfg)
        cfg_t["phase1_lr"]          = trial.suggest_float("phase1_lr", 5e-5, 5e-4, log=True)
        cfg_t["phase2_lr_head"]     = trial.suggest_float("phase2_lr_head", 1e-4, 8e-4, log=True)
        cfg_t["phase2_lr_backbone"] = cfg_t["phase2_lr_head"] * 0.5
        cfg_t["dropout"]            = trial.suggest_float("dropout", 0.2, 0.45)
        cfg_t["weight_decay"]       = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        # keep cfg_t["num_epochs"] / ["phase1_epochs"] at the FULL horizon so the
        # schedule matches the real run; we just train fewer epochs (below).

        print(f"\n--- Trial {trial.number + 1}/{n_trials} ---")
        print(f"  phase1_lr      = {cfg_t['phase1_lr']:.6f}")
        print(f"  phase2_lr_head = {cfg_t['phase2_lr_head']:.6f}")
        print(f"  phase2_lr_back = {cfg_t['phase2_lr_backbone']:.6f}")
        print(f"  dropout        = {cfg_t['dropout']:.4f}")
        print(f"  weight_decay   = {cfg_t['weight_decay']:.6f}")

        _model = PoseNetworkCTC(
            num_classes=num_classes, in_channels=cfg_t["num_joints"] * 3,
            hidden_dim=cfg_t["hidden_dim"], tcn_blocks=cfg_t["tcn_blocks"],
            lstm_layers=cfg_t["num_layers"], dropout=cfg_t["dropout"],
            drop_path_rate=cfg_t["drop_path_rate"], attn_heads=cfg_t["attn_heads"],
        ).to(device)
        _crit = CTCLossWithEntropy(blank=0, entropy_weight=cfg_t["ctc_smoothing"]).to(device)
        _scaler = torch.amp.GradScaler("cuda", enabled=amp)

        def _train_one(opt, sched, m):
            m.train(); tot = 0.0; opt.zero_grad(set_to_none=True)
            for i, (x, tg, il, tl) in enumerate(train_loader):
                x = x.to(device)
                with torch.amp.autocast("cuda", enabled=amp):
                    main, aux = m(x)
                    lp = main.permute(1, 0, 2).float()
                    Tout = lp.shape[0]
                    il2 = (il.float() * (Tout / max(x.shape[3], 1))).long().clamp(1, Tout)
                    loss = (0.7 * _crit(lp, tg, il2, tl)
                            + 0.3 * _crit(aux.permute(1, 0, 2).float(), tg, il2, tl))
                _scaler.scale(loss / cfg_t["grad_accum"]).backward()
                if (i + 1) % cfg_t["grad_accum"] == 0 or (i + 1) == len(train_loader):
                    _scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(m.parameters(), cfg_t["grad_clip"])
                    _scaler.step(opt); _scaler.update(); opt.zero_grad(set_to_none=True)
                sched.step()
                tot += loss.item()
            return tot / len(train_loader)

        @torch.no_grad()
        def _eval(m):
            m.eval(); refs, hyps = [], []
            for x, tg, il, tl in dev_loader:
                x = x.to(device)
                lp = m(x).permute(1, 0, 2)
                dec = greedy_decode(lp, cfg_t["prior_beta"], log_prior)
                off = 0
                for b in range(x.size(0)):
                    n = tl[b].item()
                    refs.append(tg[off:off+n].tolist())
                    off += n
                hyps.extend(dec)
            return compute_wer(refs, hyps)[0]

        best_wer = 1e9
        freeze_backbone(_model)
        _o1 = make_opt_phase1(_model, cfg_t["phase1_lr"], cfg_t["weight_decay"])
        _s1 = make_sched(_o1, full_p1, len(train_loader))   # schedule for FULL phase1 horizon
        for ep in range(p1_train):
            loss = _train_one(_o1, _s1, _model)
            wer = _eval(_model)
            best_wer = min(best_wer, wer)
            print(f"    P1 ep {ep+1}/{p1_train} | loss {loss:.3f} | dev WER {wer*100:.2f}% | best {best_wer*100:.2f}%")

        _o2 = _s2 = None
        if p2_train > 0:
            unfreeze_backbone(_model)
            _o2 = make_opt_phase2(_model, cfg_t["phase2_lr_backbone"], cfg_t["phase2_lr_head"], cfg_t["weight_decay"])
            _s2 = make_sched(_o2, full_p2, len(train_loader))   # schedule for FULL phase2 horizon
            for ep in range(p2_train):
                loss = _train_one(_o2, _s2, _model)
                wer = _eval(_model)
                best_wer = min(best_wer, wer)
                print(f"    P2 ep {ep+1}/{p2_train} | loss {loss:.3f} | dev WER {wer*100:.2f}% | best {best_wer*100:.2f}%")

        elapsed = time.time() - t0
        optuna_log.append({
            "trial": trial.number + 1, "wer": best_wer * 100,
            "phase1_lr": cfg_t["phase1_lr"], "phase2_lr_head": cfg_t["phase2_lr_head"],
            "dropout": cfg_t["dropout"], "weight_decay": cfg_t["weight_decay"],
            "time": elapsed,
            "params": trial.params, "value": float(best_wer),
        })
        _save_progress()
        print(f"  => Trial {trial.number + 1} done | best dev WER = {best_wer*100:.2f}% | {elapsed:.0f}s | SAVED")

        del _model, _o1, _o2, _s1, _s2, _crit, _scaler
        gc.collect()
        torch.cuda.empty_cache()
        return float(best_wer)

    n_remaining = n_trials - len(optuna_log)
    if n_remaining > 0:
        study.optimize(_trial_wer, n_trials=n_remaining, show_progress_bar=False)
    else:
        print("All trials already completed.")

    print("\n" + "=" * 70)
    print("[Optuna] SEARCH COMPLETE — summary:")
    print(f"{'#':>3} {'WER%':>7} {'p1_lr':>10} {'p2_lr_h':>10} {'drop':>7} {'wd':>10} {'time':>6}")
    print("-" * 70)
    for r in sorted(optuna_log, key=lambda x: x["wer"]):
        print(f"{r['trial']:3d} {r['wer']:7.2f} {r['phase1_lr']:10.6f} {r['phase2_lr_head']:10.6f} "
              f"{r['dropout']:7.4f} {r['weight_decay']:10.6f} {r['time']:5.0f}s")
    print(f"\nbest dev WER : {study.best_value*100:.2f}%")
    print(f"best params  : {study.best_params}")

    return study.best_params

"""Optuna hyperparameter search — SQLite storage, per-experiment studies,
median pruning.

Good-practice setup:
- ONE STUDY PER EXPERIMENT: the study name defaults to the keypoint layout +
  model variant (e.g. "cslr_coco133_gcn1"), stored in a local SQLite DB.
  Resume is automatic (load_if_exists) and two experiments can never
  contaminate each other's trials.
- MEDIAN PRUNING: dev WER is reported after every epoch; a trial whose curve
  is worse than the median of previous trials at the same epoch is killed
  early, so the same GPU budget affords ~2-3x more trials. Pruning never
  fires before 2 epochs into phase 2 (phase 1 curves are uninformative).
- TPE sampler, seeded; the first `n_startup_trials` (5) are random
  exploration, after which TPE exploits. With fewer than ~8 trials this is
  effectively random search — raise n_trials if you want TPE to matter.

The screening proxy is unchanged: each trial trains the FIRST
`trial_train_epochs` of the real full-length run, with the cosine LR schedule
built on the FULL horizon (cfg["num_epochs"]), so early-epoch LR matches the
real run exactly.
"""

import gc
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
                      n_trials: int = 20, trial_train_epochs: int = 28,
                      storage: str = "sqlite:///optuna_cslr.db",
                      study_name: str | None = None,
                      search_space: str = "lr",
                      progress_file: str | None = None):
    """Run (or resume) the search and return the best params found.

    ``search_space``:
      "lr"   (default) — search ONLY phase1_lr and phase2_lr_head; dropout and
              weight_decay stay fixed at their cfg values. With a small budget
              (<20 trials) a 2-D space is the honest choice: 4-D would be blind
              random sampling.
      "full" — also search dropout and weight_decay (use with >=20 trials).

    If the study is empty, the CURRENT cfg values are enqueued as trial 0
    (incumbent seeding): the known-good config becomes the baseline the pruner
    compares against, and TPE starts from a real reference instead of nothing.

    ``n_trials`` counts attempts, pruned ones included — pruned trials are
    cheap, so be generous. ``progress_file`` is deprecated and ignored (the
    SQLite storage replaces it); it is kept so older notebook cells still run.
    """
    try:
        import optuna
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "optuna"])
        import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if progress_file is not None:
        print(f"[Optuna] NOTE: progress_file='{progress_file}' is deprecated and "
              f"ignored — trials now live in {storage} (one study per experiment).")

    if study_name is None:
        try:
            from skeleton import LAYOUT
        except ImportError:
            LAYOUT = "unknown"
        study_name = f"cslr_{LAYOUT}_gcn{int(cfg.get('use_gcn', False))}"

    # Full-run horizon (the real training length) — used to BUILD the cosine
    # LR schedule so each trial's early LR matches the real run.
    full_p1 = cfg["phase1_epochs"]
    full_p2 = cfg["num_epochs"] - cfg["phase1_epochs"]
    p1_train = min(full_p1, trial_train_epochs)
    p2_train = max(0, min(full_p2, trial_train_epochs - full_p1))

    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=4 if search_space == "lr" else 8,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,           # need 3 finished trials before pruning anyone
        n_warmup_steps=full_p1 + 2,   # never prune before 2 epochs into phase 2
    )
    study = optuna.create_study(
        storage=storage, study_name=study_name, load_if_exists=True,
        direction="minimize", sampler=sampler, pruner=pruner,
    )

    # Incumbent seeding: an empty study starts from the KNOWN-GOOD config, so
    # the pruner's median and TPE's reference are anchored to a real result.
    if len(study.trials) == 0:
        incumbent = {"phase1_lr": cfg["phase1_lr"],
                     "phase2_lr_head": cfg["phase2_lr_head"]}
        if search_space == "full":
            incumbent.update({"dropout": cfg["dropout"],
                              "weight_decay": cfg["weight_decay"]})
        study.enqueue_trial(incumbent)
        print(f"[Optuna] incumbent enqueued as trial 0: {incumbent}")

    searched = ("phase1_lr, phase2_lr_head (dropout/wd fixed from cfg)"
                if search_space == "lr"
                else "phase1_lr, phase2_lr_head, dropout, weight_decay")
    finished = [t for t in study.trials if t.state.is_finished()]
    print(f"[Optuna] study '{study_name}' @ {storage}")
    print(f"  finished trials : {len(finished)} | target {n_trials}")
    print(f"  horizon   : {cfg['num_epochs']} ep (P1 {full_p1} + P2 {full_p2}) — LR schedule built on this")
    print(f"  per trial : {trial_train_epochs} ep (P1 {p1_train} + P2 {p2_train}) + median pruning")
    print(f"  searching : {searched}")
    print("=" * 70)

    def _objective(trial):
        t0 = time.time()
        cfg_t = copy.deepcopy(cfg)
        cfg_t["phase1_lr"]          = trial.suggest_float("phase1_lr", 5e-5, 5e-4, log=True)
        cfg_t["phase2_lr_head"]     = trial.suggest_float("phase2_lr_head", 1e-4, 8e-4, log=True)
        cfg_t["phase2_lr_backbone"] = cfg_t["phase2_lr_head"] * 0.5
        if search_space == "full":
            cfg_t["dropout"]      = trial.suggest_float("dropout", 0.2, 0.45)
            cfg_t["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        # in "lr" mode dropout / weight_decay stay at their cfg values

        print(f"\n--- Trial {trial.number} ---")
        for k in ("phase1_lr", "phase2_lr_head", "phase2_lr_backbone",
                  "dropout", "weight_decay"):
            print(f"  {k:18s} = {cfg_t[k]:.6f}")

        _model = PoseNetworkCTC(
            num_classes=num_classes, in_channels=cfg_t["num_joints"] * 3,
            hidden_dim=cfg_t["hidden_dim"], tcn_blocks=cfg_t["tcn_blocks"],
            lstm_layers=cfg_t["num_layers"], dropout=cfg_t["dropout"],
            drop_path_rate=cfg_t["drop_path_rate"], attn_heads=cfg_t["attn_heads"],
            use_gcn=cfg_t.get("use_gcn", False),
            gcn_channels=cfg_t.get("gcn_channels", 16),
        ).to(device)
        _crit = CTCLossWithEntropy(blank=0, entropy_weight=cfg_t["ctc_smoothing"]).to(device)
        _scaler = torch.amp.GradScaler("cuda", enabled=amp)

        def _cleanup():
            nonlocal _model, _crit, _scaler
            del _model, _crit, _scaler
            gc.collect()
            torch.cuda.empty_cache()

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
        gep = 0

        freeze_backbone(_model)
        _o1 = make_opt_phase1(_model, cfg_t["phase1_lr"], cfg_t["weight_decay"])
        _s1 = make_sched(_o1, full_p1, len(train_loader))   # FULL phase-1 horizon
        for ep in range(p1_train):
            loss = _train_one(_o1, _s1, _model)
            wer = _eval(_model)
            best_wer = min(best_wer, wer)
            gep += 1
            trial.report(wer, gep)
            print(f"    P1 ep {ep+1}/{p1_train} | loss {loss:.3f} | dev WER {wer*100:.2f}% | best {best_wer*100:.2f}%")
            if trial.should_prune():
                print(f"  => Trial {trial.number} PRUNED @ ep {gep} ({time.time()-t0:.0f}s)")
                del _o1, _s1
                _cleanup()
                raise optuna.TrialPruned()

        if p2_train > 0:
            unfreeze_backbone(_model)
            _o2 = make_opt_phase2(_model, cfg_t["phase2_lr_backbone"],
                                  cfg_t["phase2_lr_head"], cfg_t["weight_decay"])
            _s2 = make_sched(_o2, full_p2, len(train_loader))   # FULL phase-2 horizon
            for ep in range(p2_train):
                loss = _train_one(_o2, _s2, _model)
                wer = _eval(_model)
                best_wer = min(best_wer, wer)
                gep += 1
                trial.report(wer, gep)
                print(f"    P2 ep {ep+1}/{p2_train} | loss {loss:.3f} | dev WER {wer*100:.2f}% | best {best_wer*100:.2f}%")
                if trial.should_prune():
                    print(f"  => Trial {trial.number} PRUNED @ ep {gep} ({time.time()-t0:.0f}s)")
                    del _o1, _s1, _o2, _s2
                    _cleanup()
                    raise optuna.TrialPruned()
            del _o2, _s2

        elapsed = time.time() - t0
        print(f"  => Trial {trial.number} done | best dev WER = {best_wer*100:.2f}% | {elapsed:.0f}s")
        del _o1, _s1
        _cleanup()
        return float(best_wer)

    remaining = max(0, n_trials - len(finished))
    if remaining > 0:
        study.optimize(_objective, n_trials=remaining, show_progress_bar=False)
    else:
        print("Target number of trials already reached — nothing to run.")

    # ── summary ──
    from optuna.trial import TrialState
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    print("\n" + "=" * 70)
    print(f"[Optuna] SEARCH COMPLETE — {len(completed)} completed, {len(pruned)} pruned")
    print(f"{'#':>4} {'WER%':>7} {'p1_lr':>10} {'p2_lr_h':>10} {'drop':>7} {'wd':>10}")
    print("-" * 70)
    for t in sorted(completed, key=lambda t: t.value):
        p = t.params
        print(f"{t.number:4d} {t.value*100:7.2f} {p['phase1_lr']:10.6f} "
              f"{p['phase2_lr_head']:10.6f} "
              f"{p.get('dropout', cfg['dropout']):7.4f} "
              f"{p.get('weight_decay', cfg['weight_decay']):10.6f}")

    if not completed:
        print("No completed trials — returning empty params.")
        return {}
    print(f"\nbest dev WER : {study.best_value*100:.2f}%")
    print(f"best params  : {study.best_params}")
    return study.best_params

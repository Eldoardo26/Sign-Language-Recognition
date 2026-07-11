# coding: utf-8
"""
reverse_distill.py -- reversed-direction cross-modal distillation.

In the forward direction the skeleton model teaches the Signformer. Here the roles
swap: the Signformer (the appearance/I3D model, 39.54 dev WER) is the TEACHER, and
the skeleton PoseNetworkCTC (47.81 dev WER) is the STUDENT. This is the conventional
distillation setup a strong model teaching a weaker one, which the thesis identifies
as the experiment its forward, weak-teacher result leaves untested.

The module is self-contained: it reuses the skeleton building blocks (model, dataset,
vocabulary, evaluation, and the exact two-phase schedule) and the feature-only
FD-CMKD loss from distill.py, but touches none of the existing files. It uses a
forward hook to read the student's pre-classifier feature, and interpolates the
teacher's stored features onto the student's temporal length.

Teacher features come from extract_transformer_feats.py
(dataset/features/transformer_feats/{train,dev,test}.pkl, dim 256).

The training schedule is the CSRL baseline's, reused here so the reverse student
follows the same recipe that produced 47.81: phase 1 freezes the backbone and trains
the head (and the distillation projection) for `phase1_epochs`; phase 2 unfreezes
everything with differential learning rates for the remaining epochs. The freeze /
optimizer / scheduler helpers are imported from csrl_skeleton.training; only the
per-epoch inner loop is reimplemented here, to carry video names and add the
distillation term. Run the matched control arm (`enabled=False`) for the verdict.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_SKEL = _HERE.parent / "csrl_skeleton"
for _p in (str(_SKEL), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as skel_cfg                       # noqa: E402
from model import PoseNetworkCTC                # noqa: E402
from vocab import build_vocab_from_raw          # noqa: E402
from dataset import load_pkl, TSSIDataset, collate_ctc  # noqa: E402
from losses import CTCLossWithEntropy           # noqa: E402
from training import (                          # noqa: E402
    evaluate as skel_evaluate, freeze_backbone, unfreeze_backbone,
    make_opt_phase1, make_opt_phase2, make_sched)
from distill import load_teacher_feats, DistillHead, batch_distill_loss  # noqa: E402

REPO_ROOT = _HERE.parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NamedTSSIDataset(TSSIDataset):
    """TSSIDataset that also carries the video name, needed to look up the
    teacher feature for each sample. Behaviour is otherwise identical."""

    def __init__(self, raw, gloss_to_ids, is_train, augment=False, max_frames=400):
        self.augment = augment
        self.max_frames = max_frames
        self.items = []
        self.names = []
        for s in raw.values():
            labels = gloss_to_ids(s["gloss"], is_train)
            if labels:
                self.items.append((s["keypoint"], labels))
                self.names.append(s["name"])
        print(f"  {len(self.items)} samples | augment={augment} | with names")

    def __getitem__(self, idx):
        d = super().__getitem__(idx)
        d["name"] = self.names[idx]
        return d


def collate_named(batch):
    x, tg, il, tl = collate_ctc(batch)
    names = [b["name"] for b in batch]
    return x, tg, il, tl, names


def _build_model(num_classes, cfg):
    return PoseNetworkCTC(
        num_classes=num_classes,
        in_channels=cfg.get("in_channels", 225),
        hidden_dim=cfg["hidden_dim"],
        tcn_blocks=cfg["tcn_blocks"],
        lstm_layers=cfg["num_layers"],
        attn_heads=cfg["attn_heads"],
        dropout=cfg.get("dropout", 0.3),
    ).to(DEVICE)


def train_reverse(run_dir,
                  warm_start,
                  enabled=True,
                  data_dir=None,
                  teacher_feats_dir=None,
                  warm_ckpt=None,
                  lambda_feat=0.5,
                  low_w=1.0,
                  high_w=0.25,
                  distill_warmup_steps=500,
                  epochs=None,
                  phase1_epochs=None,
                  subset=None,
                  seed=42):
    """Train the skeleton student, optionally distilling from the Signformer teacher.

    Uses the CSRL baseline's two-phase schedule: phase 1 freezes the backbone and
    trains the head (and the distillation projection); phase 2 unfreezes everything
    with differential learning rates. Returns (best_dev_wer, best_ckpt_path). Set
    enabled=False for the matched control arm (identical schedule, no teacher).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(run_dir, exist_ok=True)

    cfg = dict(skel_cfg.CFG)
    data_dir = data_dir or skel_cfg.DATA_DIR
    teacher_feats_dir = teacher_feats_dir or str(
        REPO_ROOT / "dataset" / "features" / "transformer_feats")
    epochs = int(epochs if epochs is not None else cfg["num_epochs"])
    p1_epochs = int(phase1_epochs if phase1_epochs is not None else cfg["phase1_epochs"])
    p1_epochs = min(p1_epochs, epochs)
    p2_epochs = max(epochs - p1_epochs, 0)
    bs = cfg["batch_size"]
    max_frames = cfg["max_frames"]

    # --- data + vocabulary (built exactly as the skeleton baseline does) ---
    train_raw = load_pkl(os.path.join(data_dir, "Phoenix-2014T.train"))
    dev_raw = load_pkl(os.path.join(data_dir, "Phoenix-2014T.dev"))
    test_raw = load_pkl(os.path.join(data_dir, "Phoenix-2014T.test"))

    # The vocabulary is always built from the full data, so num_classes matches the
    # released checkpoint (1022) even when `subset` shrinks the training set for a
    # quick run. Otherwise a subset would build a smaller vocab and break warm-start.
    V = build_vocab_from_raw(train_raw, dev_raw, test_raw)

    if subset:
        train_raw = {k: train_raw[k] for k in list(train_raw)[:subset]}
        dev_raw = {k: dev_raw[k] for k in list(dev_raw)[:max(subset // 4, 8)]}
    num_classes = V["num_classes"]
    log_prior = V["log_prior"].to(DEVICE)
    g2i = V["gloss_to_ids"]

    train_ds = NamedTSSIDataset(train_raw, g2i, is_train=True,
                                augment=cfg["augment"], max_frames=max_frames)
    dev_ds = TSSIDataset(dev_raw, g2i, is_train=False, augment=False,
                         max_frames=max_frames)
    train_loader = DataLoader(train_ds, bs, shuffle=True, collate_fn=collate_named,
                              num_workers=0, drop_last=True)
    dev_loader = DataLoader(dev_ds, bs, shuffle=False, collate_fn=collate_ctc,
                            num_workers=0)

    # --- model, optionally warm-started from the skeleton baseline ---
    model = _build_model(num_classes, cfg)
    if warm_start:
        ckpt_path = warm_ckpt or str(
            REPO_ROOT / "dataset" / "checkpoints" / "tssi75_cslr_best.pt")
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        sd = ck.get("model", ck.get("state_dict", ck))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"warm-start from {ckpt_path} "
              f"(missing {len(missing)}, unexpected {len(unexpected)})")

    # --- distillation extras ---
    teacher_feats, proj = None, None
    if enabled:
        teacher_feats = load_teacher_feats(
            os.path.join(teacher_feats_dir, "train.pkl"))
        teacher_dim = int(next(iter(teacher_feats.values())).shape[1])
        student_dim = cfg["hidden_dim"] * 2                # pre-classifier feature
        proj = DistillHead(student_dim, teacher_dim).to(DEVICE)
        print(f"reverse distillation ON | student_dim={student_dim} "
              f"teacher_dim={teacher_dim} | teacher feats={len(teacher_feats)}")

    # pre-classifier feature (B, T, hidden*2) via a forward hook on the LayerNorm
    captured = {}
    handle = model.norm.register_forward_hook(
        lambda m, i, o: captured.__setitem__("feat", o))

    criterion = CTCLossWithEntropy(blank=0,
                                   entropy_weight=cfg["ctc_smoothing"]).to(DEVICE)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    tag = "distill" if enabled else "control"

    # Mutable across the closures below (assigned in the phase loops).
    state = {"steps": 0}

    def _run_epoch(opt, sched):
        """One training epoch: CTC (0.7 main + 0.3 aux) plus, if enabled, the
        FD-CMKD feature loss. Returns the average batch loss."""
        model.train()
        tot, nbatch = 0.0, 0
        trainable = ([p for p in model.parameters() if p.requires_grad]
                     + (list(proj.parameters()) if proj is not None else []))
        for x, tg, il, tl, names in train_loader:
            x = x.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                main, aux = model(x)
                lp = main.permute(1, 0, 2).float()
                Tout = lp.shape[0]
                il2 = (il.float() * (Tout / max(x.shape[3], 1))).long().clamp(1, Tout)
                loss = (0.7 * criterion(lp, tg, il2, tl)
                        + 0.3 * criterion(aux.permute(1, 0, 2).float(), tg, il2, tl))
            if enabled:
                # FD-CMKD uses a real FFT, unreliable under AMP; compute in float32
                # outside autocast. The feature was captured by the hook above.
                ramp = (1.0 if distill_warmup_steps <= 0
                        else min(1.0, state["steps"] / distill_warmup_steps))
                dloss = batch_distill_loss(
                    captured["feat"].float(), il, names, teacher_feats, proj,
                    low_w=low_w, high_w=high_w)
                loss = loss + ramp * lambda_feat * dloss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step()
            state["steps"] += 1
            tot += float(loss.item()); nbatch += 1
        return tot / max(nbatch, 1)

    def _validate_and_log(ep_global, train_loss):
        wer, det, val_loss = skel_evaluate(model, dev_loader, cfg, DEVICE,
                                           log_prior, criterion=criterion)
        print(f"[{tag}] epoch {ep_global} | train loss {train_loss:.3f} | "
              f"val loss {val_loss:.3f} | dev WER {wer * 100:.2f}%")
        with open(os.path.join(run_dir, "validations.txt"), "a") as f:
            f.write(f"epoch {ep_global} Steps: {state['steps']} trainloss "
                    f"{train_loss:.4f} valloss {val_loss:.4f} WER {wer * 100:.2f}\n")
        return wer

    best_wer, best_path = 1e9, os.path.join(run_dir, "best.pt")
    ep_global, patience = 0, 0

    # ---- Phase 1: frozen backbone, head (+ distillation projection) only ----
    freeze_backbone(model)
    opt1 = make_opt_phase1(model, cfg["phase1_lr"], cfg["weight_decay"])
    if proj is not None:
        opt1.add_param_group({"params": list(proj.parameters()), "lr": cfg["phase1_lr"]})
    sch1 = make_sched(opt1, max(p1_epochs, 1), len(train_loader))
    print(f"[{tag}] PHASE 1: {p1_epochs} epochs, frozen backbone")
    for _ in range(p1_epochs):
        ep_global += 1
        tl_ = _run_epoch(opt1, sch1)
        wer = _validate_and_log(ep_global, tl_)
        if wer < best_wer:
            best_wer = wer
            torch.save({"epoch": ep_global, "model": model.state_dict(),
                        "wer": wer}, best_path)

    # ---- Phase 2: unfrozen, differential learning rates ----
    unfreeze_backbone(model)
    opt2 = make_opt_phase2(model, cfg["phase2_lr_backbone"],
                           cfg["phase2_lr_head"], cfg["weight_decay"])
    if proj is not None:
        opt2.add_param_group({"params": list(proj.parameters()),
                              "lr": cfg["phase2_lr_head"]})
    sch2 = make_sched(opt2, max(p2_epochs, 1), len(train_loader))
    print(f"[{tag}] PHASE 2: {p2_epochs} epochs, unfrozen + differential LR")
    for _ in range(p2_epochs):
        ep_global += 1
        tl_ = _run_epoch(opt2, sch2)
        wer = _validate_and_log(ep_global, tl_)
        if wer < best_wer:
            best_wer, patience = wer, 0
            torch.save({"epoch": ep_global, "model": model.state_dict(),
                        "wer": wer}, best_path)
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]:
                print(f"[{tag}] early stopping at epoch {ep_global}")
                break

    handle.remove()
    if not os.path.exists(best_path):   # phase 2 empty (tiny smoke run): save final
        torch.save({"epoch": ep_global, "model": model.state_dict(),
                    "wer": best_wer}, best_path)
    print(f"[{tag}] best dev WER {best_wer * 100:.2f}% -> {best_path}")
    return best_wer * 100, best_path

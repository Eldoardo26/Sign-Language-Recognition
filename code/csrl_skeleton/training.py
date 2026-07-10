"""Two-phase training: freeze/unfreeze helpers, optimizers, scheduler,
train/eval loops, and resumable checkpoint logic."""

import os
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from losses import CTCLossWithEntropy, greedy_decode, beam_decode, compute_wer


# ── Freeze / unfreeze backbone ──

def freeze_backbone(m):
    for mod in [m.tcn_first, m.tcn_second, m.temporal_attn]:
        for p in mod.parameters():
            p.requires_grad = False
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  backbone frozen — trainable: {n:,}")


def unfreeze_backbone(m):
    for p in m.parameters():
        p.requires_grad = True
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  backbone unfrozen — trainable: {n:,}")


# ── Optimizer factories ──

def make_opt_phase1(m, lr: float, wd: float):
    return torch.optim.AdamW(
        [p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=wd,
    )


def make_opt_phase2(m, lr_b: float, lr_h: float, wd: float):
    groups = [
        {"params": list(m.tcn_first.parameters()),     "lr": lr_b},
        {"params": list(m.tcn_second.parameters()),    "lr": lr_b},
        {"params": list(m.temporal_attn.parameters()), "lr": lr_b},
        {"params": list(m.bilstm.parameters()),        "lr": lr_h},
        {"params": list(m.norm.parameters()),          "lr": lr_h},
        {"params": list(m.fc.parameters()),            "lr": lr_h},
        {"params": list(m.aux_proj.parameters()),      "lr": lr_h},
    ]
    groups = [g for g in groups if len(g["params"]) > 0]
    return torch.optim.AdamW(groups, weight_decay=wd)


def make_sched(opt, num_epochs: int, steps_per_epoch: int):
    ws = 5 * steps_per_epoch
    ts = num_epochs * steps_per_epoch
    warm = LinearLR(opt, start_factor=0.1, total_iters=ws)
    cos = CosineAnnealingLR(opt, T_max=max(ts - ws, 1), eta_min=1e-6)
    return SequentialLR(opt, [warm, cos], milestones=[ws])


# ── Resumable checkpoint helpers ──

SAVE_EVERY = 5


def save_resume(path: str, phase: int, epoch_in_phase: int,
                gep: int, best: float, patience: int, hist: dict,
                model, opt, sched, scaler):
    torch.save({
        "phase": phase,
        "epoch_in_phase": epoch_in_phase,
        "gep": gep,
        "best": best,
        "patience": patience,
        "hist": hist,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "scaler": scaler.state_dict(),
    }, path)
    print(f"   [resume ckpt] saved @ phase={phase} gep={gep}")


def pick_best_checkpoint(resume_path: str, best_path: str, device):
    """Pick the most recent checkpoint between resume and best.
    Ignore best if its epoch >= 150 (training already finished)."""
    resume_ckpt, best_ckpt = None, None
    resume_mtime, best_mtime = 0, 0

    if os.path.exists(resume_path):
        resume_mtime = os.path.getmtime(resume_path)
        resume_ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        print(f"   [resume file] phase={resume_ckpt['phase']} gep={resume_ckpt['gep']} "
              f"best={resume_ckpt['best']*100:.2f}% | mtime={time.ctime(resume_mtime)}")

    if best_path and os.path.exists(best_path):
        best_mtime = os.path.getmtime(best_path)
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        best_ep = best_ckpt.get("epoch", 0)
        print(f"   [best file]   epoch={best_ep} wer={best_ckpt.get('wer', float('nan'))*100:.2f}% "
              f"| mtime={time.ctime(best_mtime)}")
        if best_ep >= 150:
            print("   [best file]   epoch=150, training already finished — skipping")
            best_ckpt = None

    if resume_ckpt and best_ckpt:
        if best_mtime > resume_mtime:
            print("   -> using best (more recent)")
            return "best", best_ckpt
        else:
            print("   -> using resume (more recent)")
            return "resume", resume_ckpt
    elif resume_ckpt:
        print("   -> using resume (only one available)")
        return "resume", resume_ckpt
    elif best_ckpt:
        print("   -> using best (only one available)")
        return "best", best_ckpt
    else:
        print("   -> no checkpoint found, starting from scratch")
        return None, None


# ── Training / evaluation loops ──

def train_one_epoch(model, train_loader, criterion, scaler, opt, sched, cfg, device, amp):
    model.train()
    tot = 0.0
    opt.zero_grad(set_to_none=True)
    for i, (x, tg, il, tl) in enumerate(train_loader):
        x = x.to(device)
        with torch.amp.autocast("cuda", enabled=amp):
            main, aux = model(x)
            lp = main.permute(1, 0, 2).float()
            Tout = lp.shape[0]
            il2 = (il.float() * (Tout / max(x.shape[3], 1))).long().clamp(1, Tout)
            loss = (0.7 * criterion(lp, tg, il2, tl)
                    + 0.3 * criterion(aux.permute(1, 0, 2).float(), tg, il2, tl))
        scaler.scale(loss / cfg["grad_accum"]).backward()
        if (i + 1) % cfg["grad_accum"] == 0 or (i + 1) == len(train_loader):
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        sched.step()
        tot += loss.item()
    return tot / len(train_loader)


@torch.no_grad()
def evaluate(model, loader, cfg, device, log_prior, use_beam=False, criterion=None):
    """Evaluate on a loader. Returns (wer, detail, val_loss).

    If ``criterion`` is given, the dev CTC loss (main head only — the model
    does not emit the aux head in eval mode) is averaged over the batches.
    Without a criterion val_loss is NaN, so the WER path stays unchanged.
    """
    model.eval()
    refs, hyps = [], []
    tot_loss, n_batches = 0.0, 0
    for x, tg, il, tl in loader:
        x = x.to(device)
        lp = model(x).permute(1, 0, 2)          # (T, B, C)
        if criterion is not None:
            Tout = lp.shape[0]
            il2 = (il.float() * (Tout / max(x.shape[3], 1))).long().clamp(1, Tout)
            tot_loss += criterion(lp.float(), tg, il2, tl).item()
            n_batches += 1
        dec = (beam_decode(lp, cfg["beam_width"], cfg["prior_beta"], log_prior)
               if use_beam
               else greedy_decode(lp, cfg["prior_beta"], log_prior))
        off = 0
        for b in range(x.size(0)):
            n = tl[b].item()
            refs.append(tg[off:off+n].tolist())
            off += n
        hyps.extend(dec)
    wer, det = compute_wer(refs, hyps)
    val_loss = tot_loss / n_batches if n_batches else float("nan")
    return wer, det, val_loss


# ── Full two-phase training with resume ──

def run_training(model, train_loader, dev_loader, cfg, device, amp,
                 log_prior, ckpt_path: str, best_fallback_path: str = "",
                 fresh_start: bool = False):
    resume_path = ckpt_path.replace(".pt", "_resume.pt")

    criterion = CTCLossWithEntropy(blank=0, entropy_weight=cfg["ctc_smoothing"]).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # attempt resume (skip if fresh_start requested)
    if fresh_start:
        print("   fresh_start=True — ignoring all checkpoints, training from epoch 0")
        ckpt_type, resume = None, None
    else:
        ckpt_type, resume = pick_best_checkpoint(resume_path, best_fallback_path, device)

    hist = {"ep": [], "phase": [], "loss": [], "val_loss": [], "wer": []}
    best = 1e9
    patience = 0
    gep = 0
    skip_phase1 = False
    start_epoch_p1 = 0
    start_epoch_p2 = 0

    if resume:
        model.load_state_dict(resume["model"] if "model" in resume else resume["model_state_dict"])
        if ckpt_type == "resume":
            hist = resume["hist"]
            best = resume["best"]
            patience = resume["patience"]
            gep = resume["gep"]
            if resume["phase"] == 1:
                start_epoch_p1 = resume["epoch_in_phase"]
            elif resume["phase"] == 2:
                skip_phase1 = True
                start_epoch_p2 = resume["epoch_in_phase"]
        elif ckpt_type == "best":
            ep = resume.get("epoch", 0)
            best = resume.get("wer", 1e9)
            gep = ep
            if ep < cfg["phase1_epochs"]:
                start_epoch_p1 = ep
            else:
                skip_phase1 = True
                start_epoch_p2 = ep - cfg["phase1_epochs"]

    # backward-compat: resume checkpoints saved before val_loss existed
    if "val_loss" not in hist:
        hist["val_loss"] = [float("nan")] * len(hist["ep"])

    # ---- PHASE 1: frozen backbone, head only ----
    if not skip_phase1:
        p1_total = cfg["phase1_epochs"]
        print("=" * 60 + f"\nPHASE 1: {p1_total} epochs, FROZEN BACKBONE"
              + (f" (resuming from ep {start_epoch_p1+1})" if start_epoch_p1 > 0 else "")
              + "\n" + "=" * 60)
        freeze_backbone(model)
        opt1 = make_opt_phase1(model, cfg["phase1_lr"], cfg["weight_decay"])
        sch1 = make_sched(opt1, p1_total, len(train_loader))

        if resume and ckpt_type == "resume" and resume["phase"] == 1:
            opt1.load_state_dict(resume["opt"])
            sch1.load_state_dict(resume["sched"])
            scaler.load_state_dict(resume["scaler"])
        elif resume and ckpt_type == "best" and start_epoch_p1 > 0:
            steps_to_skip = start_epoch_p1 * len(train_loader)
            for _ in range(steps_to_skip):
                sch1.step()
            print(f"   scheduler fast-forwarded {steps_to_skip} steps")

        for e in range(start_epoch_p1, p1_total):
            gep += 1
            t0 = time.time()
            tr = train_one_epoch(model, train_loader, criterion, scaler, opt1, sch1, cfg, device, amp)
            wer, det, vloss = evaluate(model, dev_loader, cfg, device, log_prior, criterion=criterion)
            hist["ep"].append(gep)
            hist["phase"].append(1)
            hist["loss"].append(tr)
            hist["val_loss"].append(vloss)
            hist["wer"].append(wer * 100)
            print(f"[F1] Ep {gep:3d} | loss {tr:.3f} | val {vloss:.3f} | dev WER {wer*100:.2f}% "
                  f"(S{det['S']} D{det['D']} I{det['I']}) | {time.time()-t0:.0f}s")
            if wer < best:
                best = wer
                torch.save({"epoch": gep, "model": model.state_dict(),
                            "wer": wer, "cfg": cfg}, ckpt_path)
                print(f"   -> best {wer*100:.2f}%")
            if (e + 1) % SAVE_EVERY == 0:
                save_resume(resume_path, 1, e + 1, gep, best, patience, hist,
                            model, opt1, sch1, scaler)

    # ---- PHASE 2: fully unfrozen, differential LR ----
    p2_total = cfg["num_epochs"] - cfg["phase1_epochs"]
    print("\n" + "=" * 60 + f"\nPHASE 2: {p2_total} epochs, UNFREEZE + differential LR"
          + (f" (resuming from ep {start_epoch_p2+1})" if start_epoch_p2 > 0 else "")
          + "\n" + "=" * 60)
    unfreeze_backbone(model)
    opt2 = make_opt_phase2(model, cfg["phase2_lr_backbone"], cfg["phase2_lr_head"], cfg["weight_decay"])
    sch2 = make_sched(opt2, p2_total, len(train_loader))

    if resume and ckpt_type == "resume" and resume["phase"] == 2:
        opt2.load_state_dict(resume["opt"])
        sch2.load_state_dict(resume["sched"])
        scaler.load_state_dict(resume["scaler"])
    elif resume and ckpt_type == "best" and start_epoch_p2 > 0:
        steps_to_skip = start_epoch_p2 * len(train_loader)
        for _ in range(steps_to_skip):
            sch2.step()
        print(f"   scheduler fast-forwarded {steps_to_skip} steps")

    for e in range(start_epoch_p2, p2_total):
        gep += 1
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, scaler, opt2, sch2, cfg, device, amp)
        wer, det, vloss = evaluate(model, dev_loader, cfg, device, log_prior, criterion=criterion)
        hist["ep"].append(gep)
        hist["phase"].append(2)
        hist["loss"].append(tr)
        hist["val_loss"].append(vloss)
        hist["wer"].append(wer * 100)
        print(f"[F2] Ep {gep:3d} | loss {tr:.3f} | val {vloss:.3f} | dev WER {wer*100:.2f}% "
              f"(S{det['S']} D{det['D']} I{det['I']}) | {time.time()-t0:.0f}s")
        if wer < best:
            best = wer
            patience = 0
            torch.save({"epoch": gep, "model": model.state_dict(),
                        "wer": wer, "cfg": cfg}, ckpt_path)
            print(f"   -> best {wer*100:.2f}%")
        else:
            patience += 1
            if patience >= cfg["early_stopping_patience"]:
                print(f"   early stop @ epoch {gep}")
                break
        if (e + 1) % SAVE_EVERY == 0:
            save_resume(resume_path, 2, e + 1, gep, best, patience, hist,
                        model, opt2, sch2, scaler)

    print(f"\nDONE. best dev WER = {best*100:.2f}%")
    return hist, best

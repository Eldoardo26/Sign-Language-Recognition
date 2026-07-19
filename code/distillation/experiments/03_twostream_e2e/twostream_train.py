# -*- coding: utf-8 -*-
"""
twostream_train.py — end-to-end two-stream trainer with RESUMABLE checkpoints.

Trains TwoStreamFusion (live appearance + skeleton encoders, cross-attention fusion,
shared CTC head + per-stream aux heads) with CTC. The skeleton encoder is warm-started
from the trained skeleton checkpoint; the appearance transformer trains from scratch.

Durability (the reason this file exists): every epoch writes an ATOMIC checkpoint
`last.pt` (model + optimizer + scheduler + epoch + best-dev + early-stop counter + RNG
states). `train(resume=True)` (the default) picks up exactly where a killed run stopped,
so a long job on a shared GPU survives interruries. `best.pt` holds the best-on-dev model.

Usage (notebook):
    from twostream_train import train
    train(run_dir="../../runs/twostream", epochs=60, resume=True)
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.backends.cudnn.enabled = False   # env cuDNN issue (Conv1d/LSTM NOT_INITIALIZED)

_HERE = Path(__file__).resolve().parent
_SKEL = _HERE.parents[1] / "csrl_skeleton"
for _p in (str(_HERE), str(_SKEL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vocab import build_vocab_from_raw               # noqa: E402
from losses import greedy_decode, beam_decode, compute_wer   # noqa: E402
import skeleton as _sk                                # noqa: E402  (NUM_JOINTS, ADJACENCY)
from twostream_model import TwoStreamFusion           # noqa: E402

REPO_ROOT = _HERE.parents[3]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_kp_dir():
    env = os.environ.get("MSKA_DATA_DIR")
    cands = ([env] if env else []) + [
        str(REPO_ROOT / "dataset" / "phoenix2014t_133kp"),
        str(REPO_ROOT / "dataset" / "pose" / "phoenix2014t_133kp"),
    ]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "Phoenix-2014T.train")):
            return c
    return str(REPO_ROOT / "dataset" / "phoenix2014t_133kp")


def _load_skeleton_ckpt(model, path):
    """Warm-start ONLY the skeleton backbone (model.skel_enc.net) from a trained ckpt."""
    if not path or not os.path.exists(path):
        print(f"[warm-start] skeleton ckpt not found ({path}); skeleton trains from scratch")
        return
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("model", "model_state", "state_dict"):
        if isinstance(ck, dict) and k in ck:
            ck = ck[k]; break
    miss, unexp = model.skel_enc.net.load_state_dict(ck, strict=False)
    print(f"[warm-start] skeleton loaded from {os.path.basename(path)} "
          f"(missing {len(miss)}, unexpected {len(unexp)})")


def _rng_state():
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(), "python": random.getstate()}


def _set_rng_state(s):
    if not s:
        return
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["numpy"]); random.setstate(s["python"])


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)          # atomic on POSIX/Windows: never a half-written last.pt


@torch.no_grad()
def _evaluate(model, loader, log_prior, beta=0.0, use_beam=False, beam_width=10):
    model.eval()
    refs, hyps = [], []
    for i3d, tssi, pad_mask, in_lens, skel_lens, tgt_lens, targets in loader:
        i3d, tssi, pad_mask = i3d.to(DEVICE), tssi.to(DEVICE), pad_mask.to(DEVICE)
        lp_all = model(i3d, tssi, pad_mask, skel_lens)["joint"]     # (B,T,C)
        off = 0
        for b in range(i3d.size(0)):
            # decode ONLY the valid frames -- padded positions carry arbitrary
            # logits and inject spurious insertions (BUGS.md #3)
            L = max(int(in_lens[b]), 1)
            lp = lp_all[b, :L].unsqueeze(1)                          # (L,1,C)
            dec = (beam_decode(lp, beam_width, beta, log_prior) if use_beam
                   else greedy_decode(lp, beta, log_prior))
            hyps.append(dec[0])
            n = int(tgt_lens[b]); refs.append(targets[off:off + n].tolist()); off += n
    return compute_wer(refs, hyps)


def train(run_dir,
          kp_dir=None, i3d_dir=None, skeleton_ckpt=None,
          epochs=60, batch_size=8, lr=1e-4, weight_decay=1e-4,
          d=256, app_layers=3, heads=8, ff=1024, fusion_blocks=2, dropout=0.1,
          alpha=0.4, max_frames=400, grad_clip=1.0, patience=15,
          prior_beta=0.0, seed=42, resume=True):
    os.makedirs(run_dir, exist_ok=True)
    last_path = os.path.join(run_dir, "last.pt")
    best_path = os.path.join(run_dir, "best.pt")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    kp_dir = kp_dir or _find_kp_dir()
    i3d_dir = i3d_dir or str(REPO_ROOT / "dataset" / "features" / "i3d_pami0")
    if skeleton_ckpt is not None and not os.path.exists(skeleton_ckpt):
        # An explicitly requested warm-start that silently falls back to scratch
        # wastes the whole run (BUGS.md #6) -- fail loudly instead.
        raise FileNotFoundError(
            f"skeleton_ckpt not found: {skeleton_ckpt}\n"
            "Point it at the trained skeleton checkpoint (the 41.76 teacher), or "
            "pass skeleton_ckpt=None explicitly to train the skeleton from scratch.")
    if skeleton_ckpt is None:
        print("[warm-start] skeleton_ckpt=None -> skeleton encoder trains FROM SCRATCH")

    # vocab first (needs raw keypoint splits), then loaders bound to gloss_to_ids
    from twostream_data import load_i3d, JointDataset, collate_joint
    from dataset import load_pkl
    from torch.utils.data import DataLoader
    kp_raw = {s: load_pkl(os.path.join(kp_dir, f"Phoenix-2014T.{s}"))
              for s in ("train", "dev", "test")}
    V = build_vocab_from_raw(kp_raw["train"], kp_raw["dev"], kp_raw["test"])
    g2i, num_classes = V["gloss_to_ids"], V["num_classes"]
    log_prior = V["log_prior"].to(DEVICE)

    i3d = {s: load_i3d(os.path.join(i3d_dir, f"phoenix14t.pami0.{s}"))
           for s in ("train", "dev", "test")}
    loaders = {}
    for s in ("train", "dev", "test"):
        ds = JointDataset(kp_raw[s], i3d[s], g2i, is_train=(s == "train"),
                          augment=(s == "train"), max_frames=max_frames)
        loaders[s] = DataLoader(ds, batch_size, shuffle=(s == "train"),
                                collate_fn=collate_joint, num_workers=0,
                                drop_last=(s == "train"))

    model = TwoStreamFusion(num_classes, _sk.NUM_JOINTS, adjacency=_sk.ADJACENCY,
                            d=d, app_layers=app_layers, heads=heads, ff=ff,
                            fusion_blocks=fusion_blocks, dropout=dropout).to(DEVICE)
    _load_skeleton_ckpt(model, skeleton_ckpt)
    ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M | "
          f"classes {num_classes} | joints {_sk.NUM_JOINTS}")

    start_epoch, best, wait = 1, 1e9, 0
    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best, wait = ck["epoch"] + 1, ck["best"], ck["wait"]
        _set_rng_state(ck.get("rng"))
        print(f"[resume] from epoch {ck['epoch']} (best dev {best*100:.2f}%) -> "
              f"continuing at {start_epoch}")

    def _ctc_on(head_lp, in_lens, tgt_lens, targets):
        return ctc(head_lp.permute(1, 0, 2), targets, in_lens, tgt_lens)

    for ep in range(start_epoch, epochs + 1):
        model.train(); tot = 0.0
        for i3d_b, tssi_b, pad_mask, in_lens, skel_lens, tgt_lens, targets in loaders["train"]:
            i3d_b, tssi_b, pad_mask = i3d_b.to(DEVICE), tssi_b.to(DEVICE), pad_mask.to(DEVICE)
            targets = targets.to(DEVICE)
            out = model(i3d_b, tssi_b, pad_mask, skel_lens)
            loss = (_ctc_on(out["joint"], in_lens, tgt_lens, targets)
                    + alpha * _ctc_on(out["app"], in_lens, tgt_lens, targets)
                    + alpha * _ctc_on(out["skel"], in_lens, tgt_lens, targets))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tot += float(loss.item())
        sched.step()

        wer, det = _evaluate(model, loaders["dev"], log_prior, prior_beta)
        improved = wer < best
        if improved:
            best, wait = wer, 0
        else:
            wait += 1
        print(f"[e2e] ep {ep:3d} | loss {tot/len(loaders['train']):.3f} | "
              f"dev {wer*100:.2f}% (S{det['S']} D{det['D']} I{det['I']})"
              f"{'  *best' if improved else ''}")

        ckpt = {"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "best": best, "wait": wait,
                "rng": _rng_state(),
                "cfg": dict(num_classes=num_classes, num_joints=_sk.NUM_JOINTS,
                            d=d, app_layers=app_layers, heads=heads, ff=ff,
                            fusion_blocks=fusion_blocks)}
        _atomic_save(ckpt, last_path)
        if improved:
            _atomic_save(ckpt, best_path)
        if wait >= patience:
            print(f"[e2e] early stop @ {ep}"); break

    # final test at best-on-dev
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE,
                                         weights_only=False)["model"])
    for use_beam in (False, True):
        wer, det = _evaluate(model, loaders["test"], log_prior, prior_beta,
                             use_beam=use_beam, beam_width=10)
        tag = "beam" if use_beam else "greedy"
        print(f"[e2e] TEST {tag:6} {wer*100:.2f}% (S{det['S']} D{det['D']} I{det['I']} N{det['N']})")
    print(f"[e2e] best dev {best*100:.2f}% -> {best_path}")
    return best * 100, best_path


if __name__ == "__main__":
    train(run_dir=str(REPO_ROOT / "runs" / "twostream"), epochs=60, resume=True)

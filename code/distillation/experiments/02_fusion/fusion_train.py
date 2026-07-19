# -*- coding: utf-8 -*-
"""
fusion_train.py — self-contained cross-attention two-stream FUSION trainer.

Fuses the two PRE-EXTRACTED feature streams and decodes from a joint CTC head:
    appearance : transformer_feats/{split}.pkl   (Signformer encoder, 256-d)
    skeleton   : skeleton_feats_133/{split}.pkl   (skeleton GCN encoder, 512-d)

This is the honest first cut of fusion: encoders are frozen (features precomputed),
so only the fusion module + heads are trained. It answers the core question -- does
FUSING the two complementary views beat DISTILLING one into the other (which was
neutral)? A follow-up with live encoders (bigger integration) is the natural next step.

Reuses the skeleton stack for the vocabulary, CTC decoding and WER (imported from
csrl_skeleton), and the fusion modules in this folder. Touches no existing files.

Compare the joint-head WER against: skeleton-alone 41.76 dev, appearance-alone
(Signformer) 39.54 dev / 41.53 test, and the distilled 40.81 test.

Usage (from a notebook, after adding this dir + csrl_skeleton to sys.path):
    from fusion_train import train_fusion
    train_fusion(run_dir="../../runs/fusion", epochs=60)
"""
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.enabled = False   # env cuDNN issue

_HERE = Path(__file__).resolve().parent
_SKEL = _HERE.parents[1] / "csrl_skeleton"          # code/csrl_skeleton
for _p in (str(_SKEL), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vocab import build_vocab_from_raw               # noqa: E402
from dataset import load_pkl                          # noqa: E402
from losses import greedy_decode, beam_decode, compute_wer   # noqa: E402
from fusion_model import CrossModalFusion             # noqa: E402
from fusion_loss import FusionLoss                    # noqa: E402

REPO_ROOT = _HERE.parents[3]   # 02_fusion/experiments/distillation/code -> repo root
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_data_dir():
    """The 133-kp split pickles: honour MSKA_DATA_DIR, else search the two known
    layouts (dataset/phoenix2014t_133kp or dataset/pose/phoenix2014t_133kp)."""
    env = os.environ.get("MSKA_DATA_DIR")
    cands = ([env] if env else []) + [
        str(REPO_ROOT / "dataset" / "phoenix2014t_133kp"),
        str(REPO_ROOT / "dataset" / "pose" / "phoenix2014t_133kp"),
    ]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "Phoenix-2014T.train")):
            return c
    return str(REPO_ROOT / "dataset" / "phoenix2014t_133kp")


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class FusionDataset(Dataset):
    """Per video: (appearance feats, skeleton feats, gloss-id labels). Only videos
    present in the raw split AND both feature pickles are kept."""

    def __init__(self, raw, app_feats, skel_feats, gloss_to_ids, is_train):
        self.items = []
        for name, s in raw.items():
            if name not in app_feats or name not in skel_feats:
                continue
            labels = gloss_to_ids(s["gloss"], is_train)
            if labels:
                self.items.append((np.asarray(app_feats[name], np.float32),
                                   np.asarray(skel_feats[name], np.float32),
                                   labels))
        print(f"  {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        a, s, y = self.items[i]
        return torch.from_numpy(a), torch.from_numpy(s), y


def _collate(batch):
    Ta = max(b[0].shape[0] for b in batch)
    Ts = max(b[1].shape[0] for b in batch)
    Da, Ds = batch[0][0].shape[1], batch[0][1].shape[1]
    app = torch.zeros(len(batch), Ta, Da)
    skel = torch.zeros(len(batch), Ts, Ds)
    mask = torch.zeros(len(batch), Ta, dtype=torch.bool)
    in_lens, tgt_lens, targets = [], [], []
    for i, (a, s, y) in enumerate(batch):
        app[i, :a.shape[0]] = a
        skel[i, :s.shape[0]] = s
        mask[i, :a.shape[0]] = True
        in_lens.append(a.shape[0]); tgt_lens.append(len(y)); targets += y
    return (app, skel, mask, torch.tensor(in_lens),
            torch.tensor(tgt_lens), torch.tensor(targets))


@torch.no_grad()
def _evaluate(model, loader, prior_beta, log_prior, use_beam=False, beam_width=10,
              which=("joint",)):
    """Decode one or more heads from the SAME forward pass (cheap) and return
    {head: (wer, det)}. `which` in {"joint","app","skel"} — per-stream WER is the
    diagnostic: if the appearance head recovers the Signformer's WER but the joint
    head does not, the fusion is the problem, not the features."""
    model.eval()
    refs, hyps = [], {h: [] for h in which}
    for app, skel, mask, in_lens, tgt_lens, targets in loader:
        app, skel, mask = app.to(DEVICE), skel.to(DEVICE), mask.to(DEVICE)
        out = model(app, skel, mask)
        for h in which:
            lp = out[h].permute(1, 0, 2)                    # (T, B, C)
            dec = (beam_decode(lp, beam_width, prior_beta, log_prior) if use_beam
                   else greedy_decode(lp, prior_beta, log_prior))
            hyps[h].extend(dec)
        off = 0
        for b in range(app.size(0)):
            n = int(tgt_lens[b].item())
            refs.append(targets[off:off + n].tolist()); off += n
    return {h: compute_wer(refs, hyps[h]) for h in which}


def train_fusion(run_dir,
                 data_dir=None,
                 app_feats_dir=None,
                 skel_feats_dir=None,
                 epochs=60, batch_size=16, lr=3e-4, weight_decay=1e-4,
                 d_model=256, heads=8, layers=2, dropout=0.1,
                 alpha=0.3, beta=0.5, prior_beta=0.3,
                 patience=15, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(run_dir, exist_ok=True)
    data_dir = data_dir or _find_data_dir()
    app_feats_dir = app_feats_dir or str(REPO_ROOT / "dataset" / "features" / "transformer_feats")
    skel_feats_dir = skel_feats_dir or str(REPO_ROOT / "dataset" / "features" / "skeleton_feats_133")

    raw = {s: load_pkl(os.path.join(data_dir, f"Phoenix-2014T.{s}"))
           for s in ("train", "dev", "test")}
    V = build_vocab_from_raw(raw["train"], raw["dev"], raw["test"])
    num_classes = V["num_classes"]
    log_prior = V["log_prior"].to(DEVICE)
    g2i = V["gloss_to_ids"]

    app = {s: _load(os.path.join(app_feats_dir, f"{s}.pkl")) for s in ("train", "dev", "test")}
    skel = {s: _load(os.path.join(skel_feats_dir, f"{s}.pkl")) for s in ("train", "dev", "test")}
    app_dim = int(next(iter(app["train"].values())).shape[1])
    skel_dim = int(next(iter(skel["train"].values())).shape[1])
    print(f"app_dim={app_dim} skel_dim={skel_dim} classes={num_classes}")

    loaders = {}
    for s in ("train", "dev", "test"):
        ds = FusionDataset(raw[s], app[s], skel[s], g2i, is_train=(s == "train"))
        loaders[s] = DataLoader(ds, batch_size, shuffle=(s == "train"),
                                collate_fn=_collate, num_workers=0,
                                drop_last=(s == "train"))

    model = CrossModalFusion(app_dim, skel_dim, num_classes, d_model=d_model,
                             heads=heads, layers=layers, dropout=dropout).to(DEVICE)
    crit = FusionLoss(blank=0, alpha=alpha, beta=beta).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    print(f"fusion params {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    best, best_path, wait = 1e9, os.path.join(run_dir, "best.pt"), 0
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for app_b, skel_b, mask, in_lens, tgt_lens, targets in loaders["train"]:
            app_b, skel_b, mask = app_b.to(DEVICE), skel_b.to(DEVICE), mask.to(DEVICE)
            out = model(app_b, skel_b, mask)
            loss, parts = crit(out, targets.to(DEVICE), in_lens, tgt_lens, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())
        sched.step()
        res = _evaluate(model, loaders["dev"], prior_beta, log_prior,
                        which=("joint", "app", "skel"))
        wer, det = res["joint"]
        print(f"[fusion] ep {ep:3d} | loss {tot/len(loaders['train']):.3f} | "
              f"dev joint {wer*100:.2f}% (S{det['S']} D{det['D']} I{det['I']}) "
              f"| app {res['app'][0]*100:.1f} | skel {res['skel'][0]*100:.1f}")
        if wer < best:
            best, wait = wer, 0
            torch.save({"epoch": ep, "model": model.state_dict(), "wer": wer,
                        "cfg": dict(app_dim=app_dim, skel_dim=skel_dim,
                                    num_classes=num_classes, d_model=d_model,
                                    heads=heads, layers=layers)}, best_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"[fusion] early stop @ {ep}"); break

    # final test at best-on-dev, greedy + beam
    ck = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model"])
    for use_beam in (False, True):
        res = _evaluate(model, loaders["test"], prior_beta, log_prior,
                        use_beam=use_beam, beam_width=10, which=("joint",))
        wer, det = res["joint"]
        tag = "beam" if use_beam else "greedy"
        print(f"[fusion] TEST {tag:6} WER {wer*100:.2f}% (S{det['S']} D{det['D']} I{det['I']} N{det['N']})")
    print(f"[fusion] best dev WER {best*100:.2f}% -> {best_path}")
    return best * 100, best_path


if __name__ == "__main__":
    train_fusion(run_dir=str(REPO_ROOT / "runs" / "fusion"))

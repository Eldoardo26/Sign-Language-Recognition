# -*- coding: utf-8 -*-
"""
late_fusion.py — light-weight late fusion of the two FROZEN streams + the decisive
complementarity diagnostic.

The heavy cross-attention fusion overfit (2.89M params corrupt appearance 44->52%).
The clean linear probes are: appearance 44.1%, skeleton 43.0% (same skeleton vocab).
The open question is now ONLY: are the two streams COMPLEMENTARY (different errors)?
If not, no fusion can beat ~43% -- same story as the neutral distillation.

This script trains the two linear heads (best-on-dev, so the skeleton head does not
overfit past its ~ep9 optimum), freezes them, and reports on dev + test:

    app          : appearance head alone
    skel         : skeleton head alone
    ORACLE       : per-sample best of the two   (upper bound of ANY routing)
    w-sum        : fixed log-linear  w*logp_app+(1-w)*logp_skel, w grid-searched on dev
    conf-gate    : per-frame routing to the more CONFIDENT stream (the "if the
                   transformer is unsure, defer to skeleton" idea, parameter-free)

Reading it:
    ORACLE << 43   -> real complementarity; build/learn the gate, a genuine win is on
                      the table (frozen-feature ceiling; live-encoder fusion is more).
    ORACLE ~ 43    -> streams are redundant; fusion is neutral for the SAME reason
                      distillation was -> a clean, unified thesis result.

Run on the server:
    cd ~/Sign-Language-Recognition/code/distillation/experiments/02_fusion
    python late_fusion.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = False   # env cuDNN issue

_HERE = Path(__file__).resolve().parent
_SKEL = _HERE.parents[1] / "csrl_skeleton"
for _p in (str(_SKEL), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vocab import build_vocab_from_raw               # noqa: E402
from dataset import load_pkl                          # noqa: E402
from losses import greedy_decode, compute_wer         # noqa: E402
from fusion_train import _find_data_dir, REPO_ROOT    # noqa: E402
from probe_appearance import ProbeDS, _collate, _eval, _load   # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_head(D, num_classes, loaders, log_prior, epochs=25, lr=1e-3, patience=6):
    """One linear CTC head, checkpointed at its best dev WER (kills the overfit)."""
    head = nn.Linear(D, num_classes).to(DEVICE)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, wait = 1e9, None, 0
    for ep in range(1, epochs + 1):
        head.train()
        for X, il, tl, tg in loaders["train"]:
            lp = torch.log_softmax(head(X.to(DEVICE)), -1).permute(1, 0, 2)
            loss = ctc(lp, tg.to(DEVICE), il, tl)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        w, _ = _eval(head, loaders["dev"], log_prior)
        if w < best:
            best = w
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    head.load_state_dict(best_state)
    return head, best


@torch.no_grad()
def _video_logps(head, feats, names):
    """name -> (T, C) log-probs on CPU."""
    head.eval()
    out = {}
    for n in names:
        x = torch.from_numpy(np.asarray(feats[n], np.float32)).to(DEVICE)
        out[n] = torch.log_softmax(head(x), -1).cpu()
    return out


def _resample(logp, T):
    """(Ts, C) -> (T, C) linear interpolation along time."""
    if logp.shape[0] == T:
        return logp
    x = logp.transpose(0, 1).unsqueeze(0)            # (1, C, Ts)
    x = F.interpolate(x, size=T, mode="linear", align_corners=False)
    return x.squeeze(0).transpose(0, 1)              # (T, C)


def _decode1(f, log_prior):
    lp = f.unsqueeze(1).to(DEVICE)                    # (T, 1, C)
    return greedy_decode(lp, 0.0, log_prior)[0]


def _fuse_wer(la, ls, refs, names, mode, w=0.5, log_prior=None):
    R, H = [], []
    for n in names:
        a = la[n]; s = _resample(ls[n], a.shape[0])
        if mode == "app":
            f = a
        elif mode == "skel":
            f = s
        elif mode == "wsum":
            f = w * a + (1.0 - w) * s
        elif mode == "conf":                          # route per-frame to the surer stream
            ca = a.exp().max(-1).values               # (T,)
            cs = s.exp().max(-1).values
            g = (ca / (ca + cs + 1e-8)).unsqueeze(-1)  # weight on appearance
            f = g * a + (1.0 - g) * s
        else:
            raise ValueError(mode)
        H.append(_decode1(f, log_prior)); R.append(refs[n])
    return compute_wer(R, H)


def _oracle_wer(la, ls, refs, names, log_prior):
    """Per-sample best of the two streams: the upper bound of any routing scheme."""
    tot_err = tot_N = 0
    for n in names:
        a = la[n]; s = _resample(ls[n], a.shape[0])
        ha = _decode1(a, log_prior); hs = _decode1(s, log_prior)
        _, da = compute_wer([refs[n]], [ha])
        _, ds = compute_wer([refs[n]], [hs])
        ea = da["S"] + da["D"] + da["I"]
        es = ds["S"] + ds["D"] + ds["I"]
        tot_err += min(ea, es); tot_N += da["N"]
    return tot_err / max(tot_N, 1)


def main(epochs=25):
    torch.manual_seed(42); np.random.seed(42)
    data_dir = _find_data_dir()
    app_dir = str(REPO_ROOT / "dataset" / "features" / "transformer_feats")
    skel_dir = str(REPO_ROOT / "dataset" / "features" / "skeleton_feats_133")

    raw = {s: load_pkl(os.path.join(data_dir, f"Phoenix-2014T.{s}"))
           for s in ("train", "dev", "test")}
    V = build_vocab_from_raw(raw["train"], raw["dev"], raw["test"])
    g2i, num_classes = V["gloss_to_ids"], V["num_classes"]
    log_prior = V["log_prior"].to(DEVICE)

    app = {s: _load(os.path.join(app_dir, f"{s}.pkl")) for s in ("train", "dev", "test")}
    skel = {s: _load(os.path.join(skel_dir, f"{s}.pkl")) for s in ("train", "dev", "test")}
    Da = int(next(iter(app["train"].values())).shape[1])
    Ds = int(next(iter(skel["train"].values())).shape[1])
    print(f"app_dim={Da} skel_dim={Ds} classes={num_classes}")

    def loaders_for(feats):
        out = {}
        for s in ("train", "dev"):
            ds = ProbeDS(raw[s], feats[s], g2i, is_train=(s == "train"))
            out[s] = DataLoader(ds, 16, shuffle=(s == "train"),
                                collate_fn=_collate, drop_last=(s == "train"))
        return out

    print("training appearance head..."); head_a, ba = train_head(Da, num_classes, loaders_for(app), log_prior, epochs)
    print(f"  app head best dev {ba*100:.2f}%")
    print("training skeleton head...");   head_s, bs = train_head(Ds, num_classes, loaders_for(skel), log_prior, epochs)
    print(f"  skel head best dev {bs*100:.2f}%")

    # videos present in BOTH streams, per split, with labels
    def common(split):
        return [n for n, s in raw[split].items()
                if n in app[split] and n in skel[split] and g2i(s["gloss"], False)]
    refs = {split: {n: g2i(raw[split][n]["gloss"], False) for n in common(split)}
            for split in ("dev", "test")}

    for split in ("dev", "test"):
        names = list(refs[split].keys())
        la = _video_logps(head_a, app[split], names)
        ls = _video_logps(head_s, skel[split], names)

        wer_a = _fuse_wer(la, ls, refs[split], names, "app",  log_prior=log_prior)[0]
        wer_s = _fuse_wer(la, ls, refs[split], names, "skel", log_prior=log_prior)[0]
        orc   = _oracle_wer(la, ls, refs[split], names, log_prior)

        # fixed log-linear weight grid (choose on dev, report same w on test)
        if split == "dev":
            grid = {}
            for w in [round(0.1 * i, 1) for i in range(0, 11)]:
                grid[w] = _fuse_wer(la, ls, refs[split], names, "wsum", w=w, log_prior=log_prior)[0]
            best_w = min(grid, key=grid.get)
            print("  w-grid dev:", {k: round(v * 100, 1) for k, v in grid.items()})
        wer_w, det_w = _fuse_wer(la, ls, refs[split], names, "wsum", w=best_w, log_prior=log_prior)
        wer_c, det_c = _fuse_wer(la, ls, refs[split], names, "conf", log_prior=log_prior)

        print(f"\n=== {split.upper()} (n={len(names)}) ===")
        print(f"  app                 {wer_a*100:6.2f}%")
        print(f"  skel                {wer_s*100:6.2f}%")
        print(f"  ORACLE (best/samp)  {orc*100:6.2f}%   <- upper bound of routing")
        print(f"  w-sum (w={best_w})     {wer_w*100:6.2f}%   (S{det_w['S']} D{det_w['D']} I{det_w['I']})")
        print(f"  conf-gate           {wer_c*100:6.2f}%   (S{det_c['S']} D{det_c['D']} I{det_c['I']})")


if __name__ == "__main__":
    main()

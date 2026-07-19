# -*- coding: utf-8 -*-
"""
probe_appearance.py — linear CTC probe on the FROZEN feature streams.

Diagnostic that explains why the fusion's `app` head stalls at ~52% WER instead
of the Signformer's ~39.5%. Train ONLY a single linear layer
    Linear(feat_dim -> num_classes)
on the raw features (no fusion, no cross-attention), with CTC, decoding in the
SAME skeleton vocab the fusion uses. The Signformer's own recognition head is a
linear CTC layer on exactly these 256-d encoder features, so this probe's ceiling
must be ~39-42% IF features and labels are aligned.

  * app probe ~39-42%  -> features + labels fine; the FUSION module (cross-attn on
                          7k frozen samples) overfits/corrupts the signal.
  * app probe ~52%     -> feature<->label MISALIGNMENT (vocab order or video keys),
                          a bug upstream of the fusion. Fix that first.

The skeleton probe is a sanity anchor: it should land near 42% (skeleton-alone).

Run on the server:
    cd ~/Sign-Language-Recognition/code/distillation/experiments/02_fusion
    python probe_appearance.py            # appearance stream (the key test)
    python probe_appearance.py skel       # skeleton stream (sanity ~42%)
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
_SKEL = _HERE.parents[1] / "csrl_skeleton"
for _p in (str(_SKEL), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vocab import build_vocab_from_raw               # noqa: E402
from dataset import load_pkl                          # noqa: E402
from losses import greedy_decode, beam_decode, compute_wer   # noqa: E402
from fusion_train import _find_data_dir, REPO_ROOT    # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


class ProbeDS(Dataset):
    def __init__(self, raw, feats, g2i, is_train):
        self.items = []
        for name, s in raw.items():
            if name not in feats:
                continue
            y = g2i(s["gloss"], is_train)
            if y:
                self.items.append((np.asarray(feats[name], np.float32), y))
        print(f"  {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y = self.items[i]
        return torch.from_numpy(x), y


def _collate(batch):
    T = max(b[0].shape[0] for b in batch)
    D = batch[0][0].shape[1]
    X = torch.zeros(len(batch), T, D)
    il, tl, tg = [], [], []
    for i, (x, y) in enumerate(batch):
        X[i, :x.shape[0]] = x
        il.append(x.shape[0]); tl.append(len(y)); tg += y
    return X, torch.tensor(il), torch.tensor(tl), torch.tensor(tg)


@torch.no_grad()
def _eval(head, loader, log_prior, use_beam=False):
    head.eval()
    refs, hyps = [], []
    for X, il, tl, tg in loader:
        lp = torch.log_softmax(head(X.to(DEVICE)), -1).permute(1, 0, 2)  # (T,B,C)
        dec = (beam_decode(lp, 10, 0.0, log_prior) if use_beam
               else greedy_decode(lp, 0.0, log_prior))
        off = 0
        for b in range(X.size(0)):
            n = int(tl[b]); refs.append(tg[off:off + n].tolist()); off += n
        hyps.extend(dec)
    return compute_wer(refs, hyps)


def main(stream="app", epochs=30, bs=16, lr=1e-3):
    torch.manual_seed(42); np.random.seed(42)
    data_dir = _find_data_dir()
    sub = "transformer_feats" if stream == "app" else "skeleton_feats_133"
    fdir = str(REPO_ROOT / "dataset" / "features" / sub)

    raw = {s: load_pkl(os.path.join(data_dir, f"Phoenix-2014T.{s}"))
           for s in ("train", "dev", "test")}
    V = build_vocab_from_raw(raw["train"], raw["dev"], raw["test"])
    log_prior = V["log_prior"].to(DEVICE)

    feats = {s: _load(os.path.join(fdir, f"{s}.pkl")) for s in ("train", "dev", "test")}
    D = int(next(iter(feats["train"].values())).shape[1])
    print(f"stream={stream} dim={D} classes={V['num_classes']} feats_dir={fdir}")

    loaders = {}
    for s in ("train", "dev", "test"):
        ds = ProbeDS(raw[s], feats[s], V["gloss_to_ids"], is_train=(s == "train"))
        loaders[s] = DataLoader(ds, bs, shuffle=(s == "train"),
                                collate_fn=_collate, drop_last=(s == "train"))

    head = nn.Linear(D, V["num_classes"]).to(DEVICE)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    best = 1e9
    for ep in range(1, epochs + 1):
        head.train(); tot = 0.0
        for X, il, tl, tg in loaders["train"]:
            lp = torch.log_softmax(head(X.to(DEVICE)), -1).permute(1, 0, 2)
            loss = ctc(lp, tg.to(DEVICE), il, tl)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss.item())
        wer, det = _eval(head, loaders["dev"], log_prior)
        best = min(best, wer)
        print(f"[probe:{stream}] ep {ep:2d} | loss {tot/len(loaders['train']):.3f} "
              f"| dev {wer*100:.2f}% (S{det['S']} D{det['D']} I{det['I']})")

    for ub in (False, True):
        wer, det = _eval(head, loaders["test"], log_prior, use_beam=ub)
        print(f"[probe:{stream}] TEST {'beam ' if ub else 'greedy'} "
              f"{wer*100:.2f}% (S{det['S']} D{det['D']} I{det['I']} N{det['N']})")
    print(f"[probe:{stream}] best dev {best*100:.2f}%")
    return best * 100


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "app"
    main(st)

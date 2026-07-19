# -*- coding: utf-8 -*-
"""
twostream_data.py — joint dataset that pairs, per video, the I3D appearance features
with the keypoint TSSI, so both encoders can be trained end-to-end.

  * I3D  : signformer's gzip-pickled split (list of dicts, keys name/sign/gloss),
           s["sign"] is a (T, 1024) tensor. Same format as code/signformer.
  * skel : csrl_skeleton's Phoenix-2014T.{split} pickle, s["keypoint"] -> TSSI via
           generate_tssi (reused verbatim, so the transform matches the trained model).

Only videos present in BOTH sources (and with non-empty labels) are kept. The two
modalities have different lengths; the model resamples the skeleton onto the appearance
time axis, and CTC uses the appearance length, so alignment is handled downstream.
"""
import gzip
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset, DataLoader

_HERE = Path(__file__).resolve().parent
_SKEL = _HERE.parents[1] / "csrl_skeleton"
if str(_SKEL) not in sys.path:
    sys.path.insert(0, str(_SKEL))

from skeleton import generate_tssi_75                 # noqa: E402  (layout-aware via KP_LAYOUT)
from dataset import load_pkl                          # noqa: E402


def _aug_joint(tssi: np.ndarray) -> np.ndarray:
    """Frame-local augmentations ONLY. The skeleton-alone pipeline also warps time,
    mirrors L/R and drops frames -- all of which desynchronise the TSSI from the
    UNTOUCHED appearance stream of the same video (BUGS.md #2). Here we keep the
    subset that leaves the temporal axis and handedness intact."""
    C, J, T = tssi.shape
    if np.random.rand() < 0.8:                        # gaussian coordinate jitter
        n = np.random.randn(2, J, T).astype(np.float32)
        n = gaussian_filter1d(n, 3, axis=2) * 0.008
        tssi[:2] = np.clip(tssi[:2] + n, 0, 1)
    if np.random.rand() < 0.6:                        # isotropic scaling
        tssi[:2] = np.clip(tssi[:2] * np.random.uniform(0.85, 1.15), 0, 1)
    if np.random.rand() < 0.4:                        # confidence dropout
        m = np.random.rand(J, T) < 0.08
        tssi[2, m] = 0.0
    return tssi


def load_i3d(path):
    """{name: (T,1024) float tensor} from a signformer gzip-pickle split."""
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    out = {}
    for s in data:
        sign = s["sign"]
        if not torch.is_tensor(sign):
            sign = torch.as_tensor(np.asarray(sign, dtype=np.float32))
        out[s["name"]] = sign.float()
    return out


class JointDataset(Dataset):
    def __init__(self, kp_raw, i3d, gloss_to_ids, is_train,
                 augment=False, max_frames=400):
        self.augment = augment
        self.max_frames = max_frames
        self.items = []
        for name, s in kp_raw.items():
            if name not in i3d:
                continue
            labels = gloss_to_ids(s["gloss"], is_train)
            if labels:
                self.items.append((s["keypoint"], i3d[name], labels))
        print(f"  {len(self.items)} joint samples | augment={augment}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        kp, i3d, labels = self.items[idx]
        kp = np.asarray(kp, dtype=np.float32)
        T = kp.shape[0]
        if T > self.max_frames:
            sel = np.linspace(0, T - 1, self.max_frames).round().astype(int)
            kp = kp[sel]
        tssi = generate_tssi_75(kp)                    # (3,J,T)
        if self.augment:
            tssi = _aug_joint(tssi)                    # joint-safe: no warp/flip/frame-drop
        i3d = i3d if torch.is_tensor(i3d) else torch.as_tensor(np.asarray(i3d, np.float32))
        if i3d.shape[0] > self.max_frames:             # cap appearance length too
            sel = torch.linspace(0, i3d.shape[0] - 1, self.max_frames).round().long()
            i3d = i3d[sel]
        return {
            "tssi": torch.from_numpy(np.ascontiguousarray(tssi)).float(),
            "i3d": i3d.float(),
            "labels": labels,
        }


def collate_joint(batch):
    Ta = max(b["i3d"].shape[0] for b in batch)
    Ts = max(b["tssi"].shape[2] for b in batch)
    B = len(batch)
    Di = batch[0]["i3d"].shape[1]
    C, J = batch[0]["tssi"].shape[0], batch[0]["tssi"].shape[1]

    i3d = torch.zeros(B, Ta, Di)
    tssi = torch.zeros(B, C, J, Ts)
    pad_mask = torch.ones(B, Ta, dtype=torch.bool)     # True = pad
    in_lens, skel_lens, tgt_lens, targets = [], [], [], []
    for i, b in enumerate(batch):
        ta = b["i3d"].shape[0]
        ts = b["tssi"].shape[2]
        i3d[i, :ta] = b["i3d"]
        tssi[i, :, :, :ts] = b["tssi"]
        pad_mask[i, :ta] = False
        in_lens.append(ta); skel_lens.append(ts)
        tgt_lens.append(len(b["labels"])); targets += b["labels"]
    # skel_lens: true per-sample TSSI lengths -- the model must resample each sample's
    # VALID skeleton span (not the padded axis) onto the appearance axis (BUGS.md #1).
    return (i3d, tssi, pad_mask, torch.tensor(in_lens), torch.tensor(skel_lens),
            torch.tensor(tgt_lens), torch.tensor(targets))


def make_joint_loaders(kp_dir, i3d_dir, gloss_to_ids, batch_size=8,
                       max_frames=400, augment=True, num_workers=0):
    """kp_dir has Phoenix-2014T.{train,dev,test}; i3d_dir has phoenix14t.pami0.{...}."""
    kp = {s: load_pkl(os.path.join(kp_dir, f"Phoenix-2014T.{s}"))
          for s in ("train", "dev", "test")}
    i3d = {s: load_i3d(os.path.join(i3d_dir, f"phoenix14t.pami0.{s}"))
           for s in ("train", "dev", "test")}
    loaders = {}
    for s in ("train", "dev", "test"):
        ds = JointDataset(kp[s], i3d[s], gloss_to_ids, is_train=(s == "train"),
                          augment=(augment and s == "train"), max_frames=max_frames)
        loaders[s] = DataLoader(ds, batch_size, shuffle=(s == "train"),
                                collate_fn=collate_joint, num_workers=num_workers,
                                drop_last=(s == "train"))
    return loaders, kp

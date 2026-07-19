"""TSSIDataset with six augmentations, collate function, and dataloader factory."""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d

from skeleton import generate_tssi_75, COL_SWAP, NUM_JOINTS
from vocab import PAD_IDX


def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


class TSSIDataset(Dataset):
    def __init__(self, raw: dict, gloss_to_ids, is_train: bool,
                 augment: bool = False, max_frames: int = 400):
        self.augment = augment
        self.max_frames = max_frames
        self.items = []
        for s in raw.values():
            labels = gloss_to_ids(s["gloss"], is_train)
            if labels:
                self.items.append((s["keypoint"], labels))
        print(f"  {len(self.items)} samples | augment={augment}")

    def __len__(self):
        return len(self.items)

    def _aug(self, tssi: np.ndarray) -> np.ndarray:
        C, J, T = tssi.shape

        # 1. temporal Gaussian jitter
        if np.random.rand() < 0.8:
            n = np.random.randn(2, J, T).astype(np.float32)
            n = gaussian_filter1d(n, 3, axis=2) * 0.008
            tssi[:2] = np.clip(tssi[:2] + n, 0, 1)

        # 2. temporal warping
        if np.random.rand() < 0.7:
            sp = np.random.uniform(0.7, 1.3, T).astype(np.float32)
            idx = np.cumsum(sp)
            idx = (idx - idx[0]) / (idx[-1] - idx[0]) * (T - 1)
            warped = np.zeros_like(tssi)
            for c in range(C):
                for j in range(J):
                    warped[c, j, :] = np.interp(np.arange(T), idx, tssi[c, j, :])
            tssi = warped

        # 3. L/R flip (mirror x + swap left/right hand columns)
        if np.random.rand() < 0.5:
            tssi = tssi.copy()
            tssi[0] = 1.0 - tssi[0]
            tssi = tssi[:, COL_SWAP, :]

        # 4. scaling
        if np.random.rand() < 0.6:
            tssi[:2] = np.clip(tssi[:2] * np.random.uniform(0.85, 1.15), 0, 1)

        # 5. frame dropout (hold previous)
        if np.random.rand() < 0.5:
            nd = np.random.randint(1, max(2, int(T * 0.05)))
            for i in sorted(np.random.choice(T, nd, replace=False)):
                if i > 0:
                    tssi[:, :, i] = tssi[:, :, i - 1]

        # 6. confidence dropout
        if np.random.rand() < 0.4:
            m = np.random.rand(J, T) < 0.08
            tssi[2, m] = 0.0

        return tssi

    def __getitem__(self, idx):
        kp, labels = self.items[idx]
        kp = np.asarray(kp, dtype=np.float32)
        T = kp.shape[0]
        if T > self.max_frames:
            sel = np.linspace(0, T - 1, self.max_frames).round().astype(int)
            kp = kp[sel]
        tssi = generate_tssi_75(kp)
        if self.augment:
            tssi = self._aug(tssi)
        return {
            "tssi": torch.from_numpy(tssi).float(),
            "labels": labels,
            "seq_len": tssi.shape[2],
        }


def collate_ctc(batch: list[dict]):
    Tmax = max(b["tssi"].shape[2] for b in batch)
    xs, il = [], []
    for b in batch:
        t = b["tssi"]
        xs.append(F.pad(t, (0, Tmax - t.shape[2])))
        il.append(t.shape[2])
    x = torch.stack(xs)
    targets = torch.cat([torch.LongTensor(b["labels"]) for b in batch])
    return x, targets, torch.LongTensor(il), torch.LongTensor([len(b["labels"]) for b in batch])


def make_dataloaders(data_dir: str, gloss_to_ids, cfg: dict):
    train_raw = load_pkl(os.path.join(data_dir, "Phoenix-2014T.train"))
    dev_raw = load_pkl(os.path.join(data_dir, "Phoenix-2014T.dev"))
    test_raw = load_pkl(os.path.join(data_dir, "Phoenix-2014T.test"))

    train_ds = TSSIDataset(train_raw, gloss_to_ids, is_train=True,
                           augment=cfg["augment"], max_frames=cfg["max_frames"])
    dev_ds = TSSIDataset(dev_raw, gloss_to_ids, is_train=False,
                         augment=False, max_frames=cfg["max_frames"])
    test_ds = TSSIDataset(test_raw, gloss_to_ids, is_train=False,
                          augment=False, max_frames=cfg["max_frames"])

    # Dataloader-bound: TSSI generation runs on the loader. Set NUM_WORKERS>0 on
    # Linux (fork) to parallelise it and keep the GPU fed. Keep 0 on Windows
    # (spawn can't pickle the gloss_to_ids closure).
    nw = int(os.environ.get("NUM_WORKERS", cfg.get("num_workers", 0)))
    pw = nw > 0
    train_loader = DataLoader(train_ds, cfg["batch_size"], shuffle=True,
                              collate_fn=collate_ctc, num_workers=nw,
                              persistent_workers=pw, drop_last=True)
    dev_loader = DataLoader(dev_ds, cfg["batch_size"], shuffle=False,
                            collate_fn=collate_ctc, num_workers=nw, persistent_workers=pw)
    test_loader = DataLoader(test_ds, cfg["batch_size"], shuffle=False,
                             collate_fn=collate_ctc, num_workers=nw, persistent_workers=pw)
    return train_loader, dev_loader, test_loader, train_raw, dev_raw, test_raw

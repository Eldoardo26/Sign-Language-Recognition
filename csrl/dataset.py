"""
dataset.py — Dataset e DataLoader per PHOENIX-2014-T CSLR.

Classi e funzioni principali:
    index_pose_files          — indicizza i file .npy dei keypoint per split
    PHOENIXDatasetContinuos   — Dataset CTC con TSSI precomputati o on-the-fly
    collate_fn_ctc            — collate con padding variabile per CTC
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import CONFIG, POSE_DIR, TSSI_OUTPUT_DIR
from skeleton import generate_tssi_optimized


# ============================================================
# INDICIZZAZIONE KEYPOINT
# ============================================================

def index_pose_files(split: str) -> dict:
    """
    Crea un dizionario {nome_video: percorso_file_npy} per lo split indicato.

    Args:
        split : 'train', 'dev' o 'test'

    Returns:
        dict {str: str}
    """
    split_dir = os.path.join(POSE_DIR, split)
    kp_dict   = {}
    if not os.path.exists(split_dir):
        print(f"  ATTENZIONE: {split_dir} non trovata")
        return kp_dict
    for fname in os.listdir(split_dir):
        if fname.endswith(".npy"):
            stem           = os.path.splitext(fname)[0]
            kp_dict[stem]  = os.path.join(split_dir, fname)
    return kp_dict


# ============================================================
# DATASET
# ============================================================

class PHOENIXDatasetContinuos(Dataset):
    """
    Dataset per continuous sign language recognition con CTC.

    Supporta due modalità di caricamento TSSI:
    - Da cache .npz precomputata (raccomandato per efficienza)
    - On-the-fly da file .npy di keypoint grezzi

    Augmentazioni disponibili (attive solo durante il training):
    1. Jitter gaussiano nel tempo
    2. Temporal warping
    3. Flip orizzontale (scambio mani)
    4. Scaling casuale
    5. Frame dropout
    6. Confidence dropout
    """

    def __init__(
        self,
        df,
        kp_dict: dict,
        g2i: dict,
        augment: bool = False,
        frame_h: int  = None,
        tssi_dir: str = None,
    ):
        """
        Args:
            df       : DataFrame con colonne 'name' e 'orth'
            kp_dict  : {nome_video: percorso_npy}
            g2i      : {gloss: indice}
            augment  : applica augmentazione se True
            frame_h  : resize temporale (None = lunghezza originale)
            tssi_dir : cartella con .npz precomputati per questo split
        """
        self.kp_dict  = kp_dict
        self.g2i      = g2i
        self.augment  = augment
        self.frame_h  = frame_h
        self.tssi_dir = tssi_dir
        self.samples  = []

        for _, row in df.iterrows():
            vid        = str(row["name"])
            if vid not in kp_dict:
                continue
            gloss_str  = str(row["orth"]).strip().upper()
            labels     = [g2i[g] for g in gloss_str.split() if g in g2i]
            if labels:
                self.samples.append({"vid": vid, "labels": labels})

        print(f"Dataset: {len(self.samples)} samples | augment={augment}")

    def __len__(self):
        return len(self.samples)

    # ---- Augmentazione interna ----

    def _augment_tssi(self, tssi: np.ndarray) -> np.ndarray:
        """Augmentazione ricca per il training."""
        from scipy.ndimage import gaussian_filter1d
        C, J, T = tssi.shape

        # 1. Jitter gaussiano nel tempo
        if np.random.rand() < 0.8:
            noise  = np.random.randn(2, J, T).astype(np.float32)
            noise  = gaussian_filter1d(noise, sigma=3, axis=2) * 0.008
            tssi[:2] = np.clip(tssi[:2] + noise, 0, 1)

        # 2. Temporal warping
        if np.random.rand() < 0.7:
            speeds = np.random.uniform(0.7, 1.3, T).astype(np.float32)
            idx    = np.cumsum(speeds)
            idx    = (idx - idx[0]) / (idx[-1] - idx[0]) * (T - 1)
            warped = np.zeros_like(tssi)
            for c in range(C):
                for j in range(J):
                    warped[c, j, :] = np.interp(np.arange(T), idx, tssi[c, j, :])
            tssi = warped

        # 3. Flip orizzontale (scambia mano sx ↔ dx)
        if np.random.rand() < 0.5:
            tssi     = tssi.copy()
            tssi[0]  = 1.0 - tssi[0]
            left, right = list(range(6, 27)), list(range(27, 48))
            if J >= 48:
                tssi[:, left + right, :] = tssi[:, right + left, :]

        # 4. Scaling casuale
        if np.random.rand() < 0.6:
            scale    = np.random.uniform(0.85, 1.15)
            tssi[:2] = np.clip(tssi[:2] * scale, 0, 1)

        # 5. Frame dropout
        if np.random.rand() < 0.5:
            n_drop   = np.random.randint(1, max(2, int(T * 0.05)))
            drop_idx = np.random.choice(T, n_drop, replace=False)
            for i in sorted(drop_idx):
                if i > 0:
                    tssi[:, :, i] = tssi[:, :, i - 1]

        # 6. Confidence dropout
        if np.random.rand() < 0.4:
            mask = np.random.rand(J, T) < 0.08
            tssi[2, mask] = 0.0

        return tssi

    # ---- __getitem__ ----

    def __getitem__(self, idx):
        s   = self.samples[idx]
        vid = s["vid"]

        # Carica da cache .npz se disponibile
        if self.tssi_dir is not None:
            npz_path = os.path.join(self.tssi_dir, f"{vid}.npz")
            if os.path.exists(npz_path):
                data    = np.load(npz_path, allow_pickle=False)
                tssi    = data["tssi"].astype(np.float32)
                seq_len = int(data["seq_len"])
            else:
                kp              = np.load(self.kp_dict[vid]).astype(np.float32)
                tssi, seq_len   = generate_tssi_optimized(kp, frame_h=None)
        else:
            kp              = np.load(self.kp_dict[vid]).astype(np.float32)
            tssi, seq_len   = generate_tssi_optimized(kp, frame_h=None)

        if self.augment:
            tssi = self._augment_tssi(tssi)

        return {
            "tssi":    torch.from_numpy(tssi).float(),
            "labels":  s["labels"],
            "seq_len": seq_len,
        }


# ============================================================
# COLLATE
# ============================================================

def collate_fn_ctc(batch):
    """
    Collate function per CTC: padding temporale a lunghezza massima nel batch.

    Returns:
        tssies         : Tensor (B, C, J, T_max)
        targets        : Tensor (sum_of_label_lengths,)
        input_lengths  : Tensor (B,)
        target_lengths : Tensor (B,)
    """
    max_h = max(b["tssi"].shape[2] for b in batch)

    tssies, input_lengths = [], []
    for b in batch:
        t   = b["tssi"]
        pad = max_h - t.shape[2]
        tssies.append(F.pad(t, (0, pad)))
        input_lengths.append(t.shape[2])

    tssies         = torch.stack(tssies)
    targets        = torch.cat([torch.LongTensor(b["labels"]) for b in batch])
    input_lengths  = torch.LongTensor(input_lengths)
    target_lengths = torch.LongTensor([len(b["labels"]) for b in batch])

    return tssies, targets, input_lengths, target_lengths

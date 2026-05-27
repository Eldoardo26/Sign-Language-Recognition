"""
augmentation.py — Augmentazione memory-efficient per sequenze TSSI (3, J, T).

Ogni tecnica è applicata in modo stocastico e indipendente,
con probabilità calibrate per non degradare la struttura gestuale.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def augment_tssi_fixed(tssi: np.ndarray, augment: bool = True) -> np.ndarray:
    """
    Applica augmentazione in-place su un array TSSI di shape (C, J, T).

    Tecniche applicate:
    1. Jitter gaussiano correlato nel tempo (p=0.8)
    2. Temporal warping vettorizzato (p=0.7)
    3. Spatial noise (p=0.5)
    4. Time masking — SpecAugment style (p=0.7)
    5. Joint masking (p=0.5)

    Args:
        tssi    : np.ndarray shape (C=3, J, T)  —  canali x, y, confidence
        augment : se False restituisce tssi invariato

    Returns:
        tssi augmentato (potrebbe essere una copia se temporal warping è attivo)
    """
    if not augment:
        return tssi

    C, J, T = tssi.shape

    # 1. Jitter gaussiano correlato nel tempo
    if np.random.rand() < 0.8:
        noise = gaussian_filter1d(
            np.random.randn(2, J, T).astype(np.float32), sigma=3, axis=2
        )
        tssi[:2] = np.clip(tssi[:2] + noise * 0.008, 0, 1)
        del noise

    # 2. Temporal warping — vettorizzato (no loop su joint)
    if np.random.rand() < 0.7:
        speeds  = np.random.uniform(0.7, 1.3, T).astype(np.float32)
        indices = np.cumsum(speeds)
        indices = (indices / indices[-1] * (T - 1)).astype(np.float32)
        src     = np.arange(T, dtype=np.float32)
        tssi = np.stack([
            np.clip(
                np.array([np.interp(indices, src, tssi[c, j]) for j in range(J)]),
                0, 1
            )
            for c in range(C)
        ])
        del speeds, indices

    # 3. Spatial noise
    if np.random.rand() < 0.5:
        spatial_noise = np.random.randn(2, J, T).astype(np.float32) * 0.005
        tssi[:2] = np.clip(tssi[:2] + spatial_noise, 0, 1)
        del spatial_noise

    # 4. Time masking (SpecAugment)
    if np.random.rand() < 0.7:
        for _ in range(2):
            t_len   = np.random.randint(1, max(2, int(T * 0.15)))
            t_start = np.random.randint(0, max(1, T - t_len))
            tssi[:, :, t_start:t_start + t_len] = 0.0

    # 5. Joint masking
    if np.random.rand() < 0.5:
        mask = np.random.rand(J) < 0.15
        tssi[:, mask, :] = 0.0

    return tssi

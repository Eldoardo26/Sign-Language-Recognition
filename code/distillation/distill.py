# coding: utf-8
"""
distill.py — Cross-modal feature distillation (FD-CMKD, Liu et al. 2025).

Teacher: the skeleton model, whose features are precomputed by
extract_skeleton_feats.py. Student: Signformer on I3D features.

The features are decoupled in frequency along the feature axis:
  - low band  -> MSE     (semantics shared across modalities, strong consistency)
  - high band -> logMSE  (modality-specific detail and noise, weak consistency)

Both streams are standardised per frame (zero mean, unit L2 norm) so that the two
modalities' feature scales do not dominate the loss.

This is the feature-only variant: it needs no shared gloss vocabulary. See
fd_cmkd.py for the full method, which adds the shared-classifier term.
"""
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_teacher_feats(path):
    """Load {video_name: ndarray (T, D_teacher)}."""
    with open(path, "rb") as f:
        return pickle.load(f)


class DistillHead(nn.Module):
    """Project the student's encoder features into the teacher's feature space."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, x):
        return self.proj(x)


def _standardize(x, eps=1e-6):
    x = x - x.mean(dim=-1, keepdim=True)
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _sigma(x):
    """Signed log compression: log(1+x) for x >= 0, -log(1-x) otherwise.

    Damps the gradient contributed by large high-frequency discrepancies, which are
    mostly modality-specific detail and noise.
    """
    return torch.where(x >= 0, torch.log1p(x.clamp(min=0)), -torch.log1p((-x).clamp(min=0)))


def fd_cmkd_loss(s, t, low_w=1.0, high_w=1.0):
    """Frequency-decoupled feature loss for one sequence.

    Args:
        s, t: (L, D) — the projected student features and the teacher features,
              already matched in length and dimensionality.

    A real DFT along the feature axis D splits the spectrum into a low and a high
    band; the low band is matched with MSE, the high band with logMSE.
    """
    s = _standardize(s)
    t = _standardize(t)
    D = s.shape[-1]
    Sf = torch.fft.rfft(s, dim=-1)
    Tf = torch.fft.rfft(t, dim=-1)
    Fbins = Sf.shape[-1]
    half = max(Fbins // 2, 1)
    mask_low = torch.zeros(Fbins, device=s.device, dtype=s.dtype)
    mask_low[:half] = 1.0
    mask_high = 1.0 - mask_low

    s_low = torch.fft.irfft(Sf * mask_low, n=D, dim=-1)
    s_high = torch.fft.irfft(Sf * mask_high, n=D, dim=-1)
    t_low = torch.fft.irfft(Tf * mask_low, n=D, dim=-1)
    t_high = torch.fft.irfft(Tf * mask_high, n=D, dim=-1)

    l_low = ((s_low - t_low) ** 2).mean()
    l_high = ((_sigma(s_high) - _sigma(t_high)) ** 2).mean()
    return low_w * l_low + high_w * l_high


def batch_distill_loss(student_feat, sgn_lengths, names, teacher_feats, proj,
                       low_w=1.0, high_w=1.0):
    """Average the frequency-decoupled loss over the videos in one batch.

    Args:
        student_feat : (N, T_s, student_dim) — student encoder output.
        sgn_lengths  : (N,) valid temporal lengths.
        names        : video names (batch.sequence).
        teacher_feats: dict {name: ndarray (T_t, teacher_dim)}.
        proj         : DistillHead, student_dim -> teacher_dim.

    Each teacher sequence is resampled in time onto the student's length. Videos
    with no teacher entry are skipped.
    """
    s_proj = proj(student_feat)                 # (N, T_s, teacher_dim)
    device = s_proj.device
    total = s_proj.new_zeros(())
    cnt = 0
    for i, name in enumerate(names):
        tf = teacher_feats.get(name)
        if tf is None:
            continue
        ls = int(sgn_lengths[i].item())
        if ls < 2:
            continue
        s = s_proj[i, :ls]                       # (ls, Dt)
        t = torch.as_tensor(tf, dtype=torch.float32, device=device)  # (T_t, Dt)
        t = t.transpose(0, 1).unsqueeze(0)       # (1, Dt, T_t)
        t = F.interpolate(t, size=ls, mode="linear", align_corners=False)
        t = t.squeeze(0).transpose(0, 1)         # (ls, Dt)
        total = total + fd_cmkd_loss(s, t, low_w, high_w)
        cnt += 1
    return total / max(cnt, 1)

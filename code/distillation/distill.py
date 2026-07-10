# coding: utf-8
"""
distill.py — Cross-modal feature distillation (FD-CMKD, Liu et al. 2025).

Teacher = skeleton/pose (feature precalcolate con extract_skeleton_feats.py),
student = Signformer (I3D). Decoupling in frequenza della feature:
  - basse frequenze -> MSE  (semantica condivisa fra modalita', consistenza forte)
  - alte frequenze  -> logMSE (dettagli specifici + rumore, consistenza debole)
Standardizzazione (media zero + L2) per gestire le differenze di scala.
Feature-only: NON richiede vocabolario condiviso.
"""
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_teacher_feats(path):
    """Carica {nome_video: ndarray (T, D_teacher)}."""
    with open(path, "rb") as f:
        return pickle.load(f)


class DistillHead(nn.Module):
    """Proietta la feature encoder dello studente nello spazio del teacher."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)

    def forward(self, x):
        return self.proj(x)


def _standardize(x, eps=1e-6):
    x = x - x.mean(dim=-1, keepdim=True)
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _sigma(x):
    """σ(x)=log(1+x) se x>=0, -log(1-x) altrimenti: gradienti smorzati sul rumore (alte freq)."""
    return torch.where(x >= 0, torch.log1p(x.clamp(min=0)), -torch.log1p((-x).clamp(min=0)))


def fd_cmkd_loss(s, t, low_w=1.0, high_w=1.0):
    """
    s, t: (L, D) — feature studente (proiettata) e teacher, stessa lunghezza e dimensione.
    DFT lungo la dimensione feature D; MSE sulle basse freq, logMSE sulle alte.
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
    """
    student_feat : (N, T_s, student_dim) — uscita encoder studente.
    sgn_lengths  : (N,) lunghezze valide.
    names        : lista nomi video (batch.sequence).
    teacher_feats: dict {nome: ndarray (T_t, teacher_dim)}.
    proj         : DistillHead (student_dim -> teacher_dim).
    Il teacher viene ricampionato temporalmente sulla lunghezza dello studente.
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

# -*- coding: utf-8 -*-
"""
rkd_loss.py — Relational / similarity-preserving cross-modal distillation.

Motivation (from the neutral-transfer diagnosis): FD-CMKD matches feature *values*,
which forces the appearance student into the skeleton teacher's geometry and destroys
its own information -> neutral transfer. Relational KD instead matches the *structure*
of the representation (which frames the teacher considers similar), which is
modality-agnostic: two modalities can be complementary in absolute terms yet agree on
"these two frames belong to the same sign". This is the honest thing to transfer.

Two losses, both drop-in replacements for distill.batch_distill_loss:
  - similarity_preserving  (Tung & Mori, ICCV 2019): match row-normalised Gram
    matrices of frame features.
  - rkd_distance           (Park et al., CVPR 2019): match the pairwise-distance
    structure between frames (scale-invariant).

Usage in fd_cmkd_trainer (replacing / adding to the feature term):
    from experiments... .rkd_loss import batch_rkd_loss
    fd_feat = batch_rkd_loss(student_feat, sgn_lengths, names, teacher_feats,
                             proj, mode="sp", rkd_w=1.0)
`proj` is the same DistillHead(student_dim -> teacher_dim) already used; here it only
needs to bring the two into a comparable space (relations are computed after it).
"""
import torch
import torch.nn.functional as F


def _standardize(x, eps=1e-6):
    x = x - x.mean(dim=-1, keepdim=True)
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def similarity_preserving(s, t, eps=1e-8):
    """s, t: (L, D). Match row-normalised frame-frame Gram matrices (L x L)."""
    Gs = s @ s.t()                                   # (L, L)
    Gt = t @ t.t()
    Gs = F.normalize(Gs, p=2, dim=1, eps=eps)
    Gt = F.normalize(Gt, p=2, dim=1, eps=eps)
    L = s.shape[0]
    return ((Gs - Gt) ** 2).sum() / (L * L)


def rkd_distance(s, t, eps=1e-8):
    """Distance-wise RKD: match the (mean-normalised) pairwise distance matrices."""
    def _pdist(x):
        d = torch.cdist(x, x, p=2)                   # (L, L)
        mean = d[d > 0].mean() if (d > 0).any() else d.mean() + eps
        return d / (mean + eps)
    return F.smooth_l1_loss(_pdist(s), _pdist(t))


def batch_rkd_loss(student_feat, sgn_lengths, names, teacher_feats, proj,
                   mode: str = "sp", rkd_w: float = 1.0):
    """Average a relational loss over the videos in one batch.

    Args mirror distill.batch_distill_loss exactly, so this is a drop-in swap.
        student_feat : (N, T_s, student_dim)
        teacher_feats: dict {name: ndarray (T_t, teacher_dim)}
        proj         : DistillHead student_dim -> teacher_dim
        mode         : "sp" (similarity-preserving) or "rkd" (distance-wise)
    """
    s_proj = proj(student_feat)                      # (N, T_s, Dt)
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
        s = _standardize(s_proj[i, :ls])             # (ls, Dt)
        t = torch.as_tensor(tf, dtype=torch.float32, device=device)   # (T_t, Dt)
        t = t.transpose(0, 1).unsqueeze(0)
        t = F.interpolate(t, size=ls, mode="linear", align_corners=False)
        t = _standardize(t.squeeze(0).transpose(0, 1))                # (ls, Dt)
        loss = similarity_preserving(s, t) if mode == "sp" else rkd_distance(s, t)
        total = total + loss
        cnt += 1
    return rkd_w * total / max(cnt, 1)

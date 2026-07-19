# coding: utf-8
"""
fd_cmkd.py — Full FD-CMKD cross-modal knowledge distillation
(Liu et al., 2025: "Distilling Cross-Modal Knowledge via Feature Disentanglement").

Implements the three components of the paper, adapted to continuous sign
language recognition (sequence labelling with CTC):

1. Frequency-decoupled feature distillation (Eq. 1-3):
   features are transformed with a DFT along the feature dimension, split into
   low-/high-frequency bands with fixed binary masks, and reconstructed with
   the inverse DFT.

2. Feature scale alignment (Eq. 7-9): per-frame standardisation
   Std(X) = (X - mean(X)) / ||X - mean(X)||_2, followed by
   - MSE on the low-frequency band  ("strong consistency", Eq. 8);
   - logMSE on the high-frequency band ("weak consistency", Eq. 9), where
     sigma(x) = log(1+x) for x >= 0 and -log(1-x) otherwise (Eq. 6).

3. Feature space alignment via shared classifiers (Eq. 10): two classifiers
   (Phi_low, Phi_high) are shared between the teacher and the student branch
   and trained with the task loss on both. For CSLR the cross-entropy of the
   paper is replaced by CTC over the gloss sequence, expressed in a single
   SHARED VOCABULARY so that both modalities live in the same decision space.

Teacher = skeleton model (precomputed per-video features);
student  = Signformer encoder (features exposed during training).
"""
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared vocabulary / gloss-merge policy lives in vocab_utils (single source of
# truth). Support both top-level (notebook adds this dir to sys.path) and
# package-style imports.
try:
    from vocab_utils import build_shared_vocab
except ImportError:                      # pragma: no cover - package context
    from .vocab_utils import build_shared_vocab


# ---------------------------------------------------------------------------
# Teacher features
# ---------------------------------------------------------------------------
def load_teacher_feats(path):
    """Load {video_name: ndarray (T, D_teacher)} produced by the extractor."""
    with open(path, "rb") as f:
        return pickle.load(f)


def build_teacher_batch(student_feat, sgn_lengths, names, teacher_feats):
    """Interpolate each teacher sequence to the student's temporal length and
    pad it into a dense batch tensor.

    Returns:
        t_batch : (N, T, D_teacher) float tensor on the student's device
        has_t   : (N,) bool tensor, False where the video has no teacher entry
    """
    device = student_feat.device
    N, T, _ = student_feat.shape
    d_t = next(iter(teacher_feats.values())).shape[1]
    t_batch = student_feat.new_zeros((N, T, d_t))
    has_t = torch.zeros(N, dtype=torch.bool, device=device)
    for i, name in enumerate(names):
        tf = teacher_feats.get(name)
        ls = int(sgn_lengths[i].item())
        if tf is None or ls < 2:
            continue
        t = torch.as_tensor(tf, dtype=torch.float32, device=device)  # (T_t, D)
        t = t.transpose(0, 1).unsqueeze(0)                            # (1, D, T_t)
        t = F.interpolate(t, size=ls, mode="linear", align_corners=False)
        t_batch[i, :ls] = t.squeeze(0).transpose(0, 1)
        has_t[i] = True
    return t_batch, has_t


# Shared vocabulary (Eq. 10) is built by vocab_utils.build_shared_vocab,
# imported above.


# ---------------------------------------------------------------------------
# FD-CMKD loss module
# ---------------------------------------------------------------------------
def _standardize(x, eps=1e-6):
    """Eq. 7 — zero-mean, unit L2 norm per frame (over the feature dim)."""
    x = x - x.mean(dim=-1, keepdim=True)
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _sigma(x):
    """Eq. 6 — signed logarithmic compression used by the logMSE loss."""
    return torch.where(x >= 0, torch.log1p(x.clamp(min=0)),
                       -torch.log1p((-x).clamp(min=0)))


def _freq_decouple(x):
    """Eq. 1-3 — DFT along the feature dim, fixed half-band binary masks,
    inverse DFT. Returns (x_low, x_high) with the same shape as x."""
    D = x.shape[-1]
    Xf = torch.fft.rfft(x, dim=-1)
    n_bins = Xf.shape[-1]
    half = max(n_bins // 2, 1)
    mask_low = torch.zeros(n_bins, device=x.device, dtype=x.dtype)
    mask_low[:half] = 1.0
    x_low = torch.fft.irfft(Xf * mask_low, n=D, dim=-1)
    x_high = torch.fft.irfft(Xf * (1.0 - mask_low), n=D, dim=-1)
    return x_low, x_high


def _similarity_preserving(s, t, eps=1e-8):
    """Relational KD (Tung & Mori, ICCV 2019): match the row-normalised
    frame-frame Gram matrices of one (L, D) student/teacher pair. Transfers the
    representation's *structure* (which frames are similar), not its values, so
    complementary modalities need not share a geometry -- the fix for the neutral
    value-matching transfer."""
    Gs = F.normalize(s @ s.t(), p=2, dim=1, eps=eps)
    Gt = F.normalize(t @ t.t(), p=2, dim=1, eps=eps)
    return ((Gs - Gt) ** 2).sum() / (s.shape[0] ** 2)


class FDCMKDModule(nn.Module):
    """Trainable components of FD-CMKD: the student-to-teacher projection and
    the two shared classifiers (Phi_low, Phi_high) over the shared vocabulary.
    """

    def __init__(self, student_dim, teacher_dim, n_shared_classes):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)
        self.phi_low = nn.Linear(teacher_dim, n_shared_classes)
        self.phi_high = nn.Linear(teacher_dim, n_shared_classes)

    def forward(self, s_feat, t_feat, has_t, lengths, targets_shared,
                tgt_lengths, ctc, low_w=1.0, high_w=0.25, disable_dft=False,
                feat_mode="fd"):
        """Compute the FD-CMKD terms for one batch.

        Args:
            s_feat        : (N, T, D_s) student encoder output (padded)
            t_feat        : (N, T, D_t) teacher features aligned to student time
            has_t         : (N,) bool — samples with a valid teacher entry
            lengths       : (N,) valid temporal lengths
            targets_shared: (N, S) gloss ids in the shared vocabulary
            tgt_lengths   : (N,) gloss sequence lengths
            ctc           : nn.CTCLoss(blank=0, zero_infinity=True)

        Returns dict with 'feat' (Eq. 8 + Eq. 9) and 'align' (Eq. 10, CTC).
        """
        N, T, _ = s_feat.shape
        device = s_feat.device
        s_proj = self.proj(s_feat)                             # (N, T, D_t)

        s_std = _standardize(s_proj)
        t_std = _standardize(t_feat)
        use_rkd = (feat_mode == "rkd_sp")
        no_split = disable_dft or use_rkd
        if no_split:
            # RQ2 ablation (plain MSE) OR relational mode — no frequency split.
            # The high band is zeroed, so l_high and the phi_high CTC terms vanish
            # (handled below). Default FD path is unchanged.
            s_low, s_high = s_std, torch.zeros_like(s_std)
            t_low, t_high = t_std, torch.zeros_like(t_std)
        else:
            s_low, s_high = _freq_decouple(s_std)
            t_low, t_high = _freq_decouple(t_std)

        # valid-frame mask (padding excluded); teacher terms only where has_t
        frame_mask = (torch.arange(T, device=device).unsqueeze(0)
                      < lengths.to(device).unsqueeze(1))       # (N, T)
        fm = (frame_mask & has_t.unsqueeze(1)).unsqueeze(-1).float()
        denom = fm.sum().clamp(min=1.0) * s_low.shape[-1]

        if use_rkd:
            # relational (similarity-preserving) feature loss, per sequence
            tot, cnt = s_feat.new_zeros(()), 0
            for i in range(N):
                if not bool(has_t[i]):
                    continue
                L = int(lengths[i].item())
                if L < 2:
                    continue
                tot = tot + _similarity_preserving(s_std[i, :L], t_std[i, :L])
                cnt += 1
            feat_loss = tot / max(cnt, 1)
            l_low = l_high = feat_loss.detach()
        else:
            l_low = (((s_low - t_low) ** 2) * fm).sum() / denom              # Eq. 8
            l_high = (((_sigma(s_high) - _sigma(t_high)) ** 2) * fm).sum() / denom  # Eq. 9
            feat_loss = low_w * l_low + high_w * l_high

        # --- shared-classifier alignment (Eq. 10), CTC-adapted --------------
        in_lens = lengths.long().clamp(max=T).to(device)
        tgt_lens = tgt_lengths.long().to(device)

        def _ctc_branch(feats, subset=None):
            logits = feats                                     # (N, T, C)
            lp = logits.log_softmax(-1).permute(1, 0, 2)       # (T, N, C)
            if subset is None:
                return ctc(lp, targets_shared, in_lens, tgt_lens)
            idx = subset.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                return s_feat.new_zeros(())
            return ctc(lp[:, idx], targets_shared[idx],
                       in_lens[idx], tgt_lens[idx])

        align = (_ctc_branch(self.phi_low(s_low))
                 + _ctc_branch(self.phi_low(t_low), subset=has_t))
        if not no_split:
            align = (align
                     + _ctc_branch(self.phi_high(s_high))
                     + _ctc_branch(self.phi_high(t_high), subset=has_t))

        return {"feat": feat_loss, "align": align,
                "low": l_low.detach(), "high": l_high.detach()}

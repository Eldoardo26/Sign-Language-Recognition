# -*- coding: utf-8 -*-
"""
fusion_model.py — Cross-Attention Two-Stream fusion for CSLR.

Instead of distilling the skeleton teacher into the appearance student (which
collapses one modality into the other -> neutral), FUSE both live streams and
decode from the joint representation. This is the family that reaches ~20% test WER
(TwoStream-SLR, Chen et al. NeurIPS 2022; MSKA). It preserves both axes:
appearance = handshape texture, skeleton = geometry/trajectory.

This module is architecture-agnostic: it takes the two encoders' *output sequences*
(appearance and skeleton features, already produced by SignFormer and the skeleton
GCN encoder) and returns three CTC log-prob heads. Wire your encoders around it.

    app_feat : (B, Ta, Da)   e.g. SignFormer encoder output (Da=256)
    skel_feat: (B, Ts, Ds)   e.g. PoseNetworkCTC pre-classifier (Ds=512)
    -> logits: {"joint": (B, Ta, C), "app": (B, Ta, C), "skel": (B, Ta, C)}

The skeleton stream is temporally resampled to the appearance length Ta (the CTC
target length), so both align to one time axis for the joint head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalCrossAttention(nn.Module):
    """One layer: each stream attends to the other (lateral connection)."""

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.a2b = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.b2a = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.na, self.nb = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffa, self.ffb = self._ff(dim, dropout), self._ff(dim, dropout)
        self.nfa, self.nfb = nn.LayerNorm(dim), nn.LayerNorm(dim)

    @staticmethod
    def _ff(dim, dropout):
        return nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                             nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, a, b, key_padding_mask=None):
        # a queries b (appearance pulls in skeleton), b queries a (and vice versa)
        a2, _ = self.a2b(a, b, b, key_padding_mask=key_padding_mask)
        a = self.na(a + a2)
        b2, _ = self.b2a(b, a, a, key_padding_mask=key_padding_mask)
        b = self.nb(b + b2)
        a = self.nfa(a + self.ffa(a))
        b = self.nfb(b + self.ffb(b))
        return a, b


class CrossModalFusion(nn.Module):
    def __init__(self, app_dim: int, skel_dim: int, num_classes: int,
                 d_model: int = 256, heads: int = 8, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.proj_a = nn.Linear(app_dim, d_model)
        self.proj_s = nn.Linear(skel_dim, d_model)
        self.blocks = nn.ModuleList(
            [BidirectionalCrossAttention(d_model, heads, dropout) for _ in range(layers)])
        self.head_app = nn.Linear(d_model, num_classes)
        self.head_skel = nn.Linear(d_model, num_classes)
        self.head_joint = nn.Linear(d_model * 2, num_classes)

    def forward(self, app_feat, skel_feat, mask=None):
        """mask: (B, Ta) bool, True = valid frame (appearance time axis)."""
        B, Ta, _ = app_feat.shape
        a = self.proj_a(app_feat)                                # (B, Ta, d)
        # resample skeleton onto the appearance time axis
        s = skel_feat.transpose(1, 2)                            # (B, Ds, Ts)
        s = F.interpolate(s, size=Ta, mode="linear", align_corners=False)
        s = self.proj_s(s.transpose(1, 2))                       # (B, Ta, d)

        kpm = (~mask) if mask is not None else None              # True = pad for MHA
        for blk in self.blocks:
            a, s = blk(a, s, key_padding_mask=kpm)

        joint = torch.cat([a, s], dim=-1)                        # (B, Ta, 2d)
        return {
            "joint": self.head_joint(joint).log_softmax(-1),
            "app":   self.head_app(a).log_softmax(-1),
            "skel":  self.head_skel(s).log_softmax(-1),
        }

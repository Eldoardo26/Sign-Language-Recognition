# -*- coding: utf-8 -*-
"""
twostream_model.py — end-to-end two-stream CSLR model.

Both encoders are LIVE (trained jointly), which is the whole point: the frozen-feature
study (02_fusion) showed the two streams are complementary (oracle -7.5pt) but their
independently-trained CTC spikes are misaligned, so frame-level fusion collapsed.
Training the encoders together lets their representations and timing become compatible.

    appearance : I3D features (T,1024) -> linear proj -> 3-layer Transformer encoder
                 (matches the Signformer baseline: d=256, 8 heads, ff=1024)
    skeleton   : keypoint TSSI (3,J,T) -> PoseNetworkCTC.forward_feat (GCN+TCN+attn+BiLSTM,
                 512-d) -> linear proj to d
    fusion     : skeleton resampled to the appearance time axis, then N bidirectional
                 cross-attention blocks (lateral connections), a shared CTC head on the
                 fused stream, and per-stream aux CTC heads for deep supervision.

The shared head means the joint decode has a single, well-defined emission timing --
exactly what frozen late fusion could not provide.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_SKEL = _HERE.parents[1] / "csrl_skeleton"          # code/csrl_skeleton
if str(_SKEL) not in sys.path:
    sys.path.insert(0, str(_SKEL))

from model import PoseNetworkCTC                     # noqa: E402  (skeleton encoder)


class PositionalEncoding(nn.Module):
    def __init__(self, d, dropout=0.1, max_len=5000):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x):                             # (B, T, d)
        return self.drop(x + self.pe[:, :x.size(1)])


class AppearanceEncoder(nn.Module):
    """Signformer-style transformer encoder over I3D features (kept trainable)."""
    def __init__(self, in_dim=1024, d=256, layers=3, heads=8, ff=1024, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d)
        self.pe = PositionalEncoding(d, dropout)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout,
                                           batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, x, pad_mask):                   # x (B,Ta,1024), pad_mask (B,Ta) True=pad
        h = self.pe(self.proj(x))
        return self.enc(h, src_key_padding_mask=pad_mask)   # (B,Ta,d)


class SkeletonEncoder(nn.Module):
    """The trained skeleton backbone, used up to its pre-classifier features (512-d)."""
    def __init__(self, num_joints, num_classes, d=256, hidden_dim=256,
                 use_gcn=True, gcn_channels=16, adjacency=None, dropout=0.3):
        super().__init__()
        self.net = PoseNetworkCTC(num_classes=num_classes, in_channels=num_joints * 3,
                                  hidden_dim=hidden_dim, use_gcn=use_gcn,
                                  gcn_channels=gcn_channels, adjacency=adjacency,
                                  dropout=dropout)
        self.proj = nn.Linear(hidden_dim * 2, d)

    def forward(self, tssi):                          # (B,3,J,T)
        feat = self.net.forward_feat(tssi)            # (B,T,512)
        return self.proj(feat)                        # (B,T,d)


class BiCrossBlock(nn.Module):
    """One bidirectional cross-attention block: each stream attends to the other,
    with residual + FFN. Both streams share the appearance time axis and pad mask."""
    def __init__(self, d, heads=8, ff=1024, dropout=0.1):
        super().__init__()
        self.a2s = nn.MultiheadAttention(d, heads, dropout, batch_first=True)
        self.s2a = nn.MultiheadAttention(d, heads, dropout, batch_first=True)
        self.na1, self.ns1 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffa = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.ffs = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.na2, self.ns2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, a, s, pad_mask):
        a2, _ = self.a2s(a, s, s, key_padding_mask=pad_mask)
        a = self.na1(a + a2)
        s2, _ = self.s2a(s, a, a, key_padding_mask=pad_mask)
        s = self.ns1(s + s2)
        a = self.na2(a + self.ffa(a))
        s = self.ns2(s + self.ffs(s))
        return a, s


class TwoStreamFusion(nn.Module):
    def __init__(self, num_classes, num_joints, adjacency=None,
                 d=256, app_layers=3, heads=8, ff=1024, fusion_blocks=2,
                 dropout=0.1):
        super().__init__()
        self.app_enc = AppearanceEncoder(1024, d, app_layers, heads, ff, dropout)
        self.skel_enc = SkeletonEncoder(num_joints, num_classes, d=d,
                                        adjacency=adjacency, dropout=max(dropout, 0.3))
        self.blocks = nn.ModuleList([BiCrossBlock(d, heads, ff, dropout)
                                     for _ in range(fusion_blocks)])
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d), nn.GELU(),
                                  nn.Dropout(dropout))
        self.head_joint = nn.Linear(d, num_classes)
        self.head_app = nn.Linear(d, num_classes)
        self.head_skel = nn.Linear(d, num_classes)

    def forward(self, i3d, tssi, pad_mask, skel_lens=None):
        # i3d (B,Ta,1024), tssi (B,3,J,Ts), pad_mask (B,Ta) True=pad,
        # skel_lens (B,) true per-sample TSSI lengths (pre-padding)
        a = self.app_enc(i3d, pad_mask)               # (B,Ta,d)
        s = self.skel_enc(tssi)                        # (B,Ts,d)
        Ta = a.size(1)
        if skel_lens is not None:
            # Per-sample resample of the VALID skeleton span onto the valid
            # appearance span. Resampling the padded axis instead compresses the
            # real data into a fraction of the time axis for every sample shorter
            # than the batch max, desynchronising the two streams (BUGS.md #1).
            app_lens = (~pad_mask).sum(1)              # (B,)
            out = s.new_zeros(a.size(0), Ta, s.size(-1))
            for i in range(a.size(0)):
                ts = max(int(skel_lens[i]), 1)
                ta = max(int(app_lens[i]), 1)
                si = s[i, :ts].transpose(0, 1).unsqueeze(0)      # (1,d,ts)
                si = F.interpolate(si, size=ta, mode="linear",
                                   align_corners=False)
                out[i, :ta] = si.squeeze(0).transpose(0, 1)
            s = out
        elif s.size(1) != Ta:                          # legacy path (no lengths given)
            s = F.interpolate(s.transpose(1, 2), size=Ta, mode="linear",
                              align_corners=False).transpose(1, 2)
        for blk in self.blocks:
            a, s = blk(a, s, pad_mask)
        fused = self.fuse(torch.cat([a, s], dim=-1))
        return {
            "joint": F.log_softmax(self.head_joint(fused), dim=-1),
            "app":   F.log_softmax(self.head_app(a), dim=-1),
            "skel":  F.log_softmax(self.head_skel(s), dim=-1),
        }

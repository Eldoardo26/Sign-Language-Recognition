"""PoseNetworkCTC: (optional spatial GCN) + dilated TCN + Temporal Attention
+ BiLSTM + CTC/SR-CTC heads."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGraphConv(nn.Module):
    """One ST-GCN-style spatial graph convolution over the joint axis.

    Input/output are (B, C, J, T). The 1x1 conv mixes channels per joint, then the
    normalised adjacency aggregates each joint with its skeletal neighbours, giving
    the temporal backbone features that already encode who-is-connected-to-whom
    instead of a flat bag of joint coordinates.
    """

    def __init__(self, cin: int, cout: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)                       # (J, J) normalised
        self.theta = nn.Conv2d(cin, cout, kernel_size=1)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(True)

    def forward(self, x):                                   # (B, Cin, J, T)
        x = self.theta(x)                                  # (B, Cout, J, T)
        x = torch.einsum("bcjt,jk->bckt", x, self.A)       # aggregate over joints
        return self.relu(self.bn(x))


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        m = (torch.rand(shape, dtype=x.dtype, device=x.device) < keep).float()
        return x * m / keep


class TemporalAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        return self.norm(x + self.drop(a))


class TCNBlock(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3,
                 dilation: int = 1, dropout: float = 0.2, drop_path: float = 0.0):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=pad, dilation=dilation),
            nn.BatchNorm1d(cout),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(cout, cout, k, padding=pad, dilation=dilation),
            nn.BatchNorm1d(cout),
        )
        self.skip = (
            nn.Sequential(nn.Conv1d(cin, cout, 1), nn.BatchNorm1d(cout))
            if cin != cout else nn.Identity()
        )
        self.dp = DropPath(drop_path)

    def forward(self, x):
        return F.relu(self.dp(self.net(x)) + self.skip(x))


class PoseNetworkCTC(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 225,
                 hidden_dim: int = 256, tcn_blocks: int = 3,
                 lstm_layers: int = 3, dropout: float = 0.3,
                 drop_path_rate: float = 0.1, attn_heads: int = 4,
                 use_gcn: bool = False, gcn_channels: int = 16,
                 adjacency=None):
        super().__init__()
        mid = max(1, tcn_blocks // 2)
        dp = [drop_path_rate * i / max(tcn_blocks - 1, 1) for i in range(tcn_blocks)]

        # Optional spatial graph front-end (D1). It maps the 3 raw channels
        # (x, y, conf) per joint into gcn_channels that already mix skeletal
        # neighbours, then flattens to gcn_channels*J for the temporal backbone.
        if use_gcn:
            if adjacency is None:
                from skeleton import ADJACENCY as adjacency
            A = torch.as_tensor(np.asarray(adjacency), dtype=torch.float32)
            n_joints = in_channels // 3
            self.gcn = nn.Sequential(
                SpatialGraphConv(3, gcn_channels, A),
                SpatialGraphConv(gcn_channels, gcn_channels, A),
            )
            tcn_in = gcn_channels * n_joints
        else:
            self.gcn = None
            tcn_in = in_channels

        first = [TCNBlock(tcn_in, hidden_dim, 5, 1, dropout, dp[0])]
        for i in range(1, mid):
            first.append(TCNBlock(hidden_dim, hidden_dim, 3, min(2**(i-1), 8), dropout, dp[i]))
        self.tcn_first = nn.Sequential(*first)

        second = [
            TCNBlock(hidden_dim, hidden_dim, 3, min(2**(i-1), 8), dropout, dp[i])
            for i in range(mid, tcn_blocks)
        ]
        self.tcn_second = nn.Sequential(*second) if second else nn.Identity()

        self.temporal_attn = TemporalAttention(hidden_dim, attn_heads, dropout * 0.5)
        self.bilstm = nn.LSTM(
            hidden_dim, hidden_dim, lstm_layers, batch_first=True,
            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.aux_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout * 0.5),
        )

    def forward_feat(self, x):
        """Pre-classifier features (B, T, hidden*2) — what gets distilled.
        Same path as forward() up to the LayerNorm, classifier excluded."""
        B, C, J, T = x.shape
        if self.gcn is not None:
            x = self.gcn(x)
            xf = x.reshape(B, -1, T)
        else:
            xf = x.reshape(B, C * J, T)
        fm = self.tcn_first(xf).permute(0, 2, 1)
        feat = self.tcn_second(fm.permute(0, 2, 1)).permute(0, 2, 1)
        feat = self.temporal_attn(feat)
        feat, _ = self.bilstm(feat)
        return self.norm(feat)

    def forward(self, x):
        B, C, J, T = x.shape
        if self.gcn is not None:
            x = self.gcn(x)               # (B, gcn_channels, J, T)
            xf = x.reshape(B, -1, T)      # (B, gcn_channels*J, T)
        else:
            xf = x.reshape(B, C * J, T)
        fm = self.tcn_first(xf).permute(0, 2, 1)
        feat = self.tcn_second(fm.permute(0, 2, 1)).permute(0, 2, 1)
        feat = self.temporal_attn(feat)

        if self.training:
            aux = torch.log_softmax(self.fc(self.aux_proj(fm)), dim=-1)

        feat, _ = self.bilstm(feat)
        feat = self.drop(self.norm(feat))
        main = torch.log_softmax(self.fc(feat), dim=-1)
        return (main, aux) if self.training else main

"""PoseNetworkCTC: dilated TCN + Temporal Attention + BiLSTM + CTC/SR-CTC heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 drop_path_rate: float = 0.1, attn_heads: int = 4):
        super().__init__()
        mid = max(1, tcn_blocks // 2)
        dp = [drop_path_rate * i / max(tcn_blocks - 1, 1) for i in range(tcn_blocks)]

        first = [TCNBlock(in_channels, hidden_dim, 5, 1, dropout, dp[0])]
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

    def forward(self, x):
        B, C, J, T = x.shape
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

"""
model.py — Architettura TCN-BiLSTM-CTC per CSLR su PHOENIX-2014-T.

Componenti:
    DropPath          — Stochastic Depth per regolarizzazione inter-blocco
    TemporalAttention — Multi-head self-attention temporale (residual + LN)
    TCNBlock          — Residual TCN con dilated conv + DropPath
    PoseNetworkTCN    — TCN puro (backbone estrattore di feature)
    PoseNetworkCTC    — Modello completo: SR-CTC + TCN + Attention + BiLSTM
"""

from skeleton import DFS_UNIQUE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# DROP PATH (STOCHASTIC DEPTH)
# ============================================================

class DropPath(nn.Module):
    """
    Stochastic Depth: azzera l'intero ramo conv per un sottoinsieme
    casuale di sample nel batch durante il training (no-op a inference).
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape     = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask      = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        return x * mask.float() / keep_prob


# ============================================================
# TEMPORAL ATTENTION
# ============================================================

class TemporalAttention(nn.Module):
    """
    Multi-head self-attention temporale con residual + LayerNorm.
    Cattura dipendenze a lungo raggio non coperte dalle conv dilatate.
    Input/Output shape: (B, T, D).
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads,
            dropout=dropout, batch_first=True,
        )
        self.norm    = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out))


# ============================================================
# TCN BLOCK
# ============================================================

class TCNBlock(nn.Module):
    """
    Residual TCN block con dilated convolution e DropPath.
    DropPath viene applicato al ramo conv; il gradiente scorre sempre
    attraverso il residual anche quando il blocco è "saltato".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int    = 1,
        dropout: float   = 0.2,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        return F.relu(self.drop_path(self.net(x)) + residual)


# ============================================================
# POSE NETWORK TCN (backbone standalone)
# ============================================================

class PoseNetworkTCN(nn.Module):
    """TCN puro con DropPath lineare crescente sui blocchi."""

    def __init__(
        self,
        num_joints: int  = 48,
        hidden_dim: int  = 256,
        num_blocks: int  = 4,
        dropout: float   = 0.3,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        from skeleton import DFS_UNIQUE
        self.in_channels = len(DFS_UNIQUE) * 3    # 144

        dp_rates = [
            drop_path_rate * i / max(num_blocks - 1, 1)
            for i in range(num_blocks)
        ]
        layers = [
            TCNBlock(self.in_channels, hidden_dim,
                     kernel_size=5, dilation=1,
                     dropout=dropout, drop_path_rate=dp_rates[0]),
        ]
        for i in range(1, num_blocks):
            dilation = min(2 ** (i - 1), 8)
            layers.append(
                TCNBlock(hidden_dim, hidden_dim,
                         kernel_size=3, dilation=dilation,
                         dropout=dropout, drop_path_rate=dp_rates[i])
            )
        self.tcn     = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """x_flat: (B, 144, T) → (B, T, hidden_dim)"""
        return self.tcn(x_flat).permute(0, 2, 1)


# ============================================================
# POSE NETWORK CTC (modello completo)
# ============================================================

class PoseNetworkCTC(nn.Module):
    """
    Architettura completa per CSLR:
        TCN-first → SR-CTC ausiliario
        TCN-second → TemporalAttention → BiLSTM → CTC principale

    SR-CTC (classifier-sharing): il punto a metà TCN produce una
    supervisione ausiliaria condividendo il classificatore fc con
    l'uscita principale (aux_proj allinea le dimensioni).

    In training:  restituisce (main_log_prob, aux_log_prob)
    In inference: restituisce  main_log_prob
    """

    def __init__(
        self,
        num_classes: int,
        num_joints: int  = 48,
        hidden_dim: int  = 256,
        tcn_blocks: int  = 4,
        lstm_layers: int = 2,
        dropout: float   = 0.3,
        drop_path_rate: float = 0.1,
        attn_heads: int  = 4,
    ):
        super().__init__()

        mid      = max(1, tcn_blocks // 2)
        dp_rates = [
            drop_path_rate * i / max(tcn_blocks - 1, 1)
            for i in range(tcn_blocks)
        ]

        # -- Prima metà TCN --
        # Con:
        from skeleton import DFS_UNIQUE
        in_ch = len(DFS_UNIQUE) * 3   # canali reali: n_unique_joints × 3

        first = [
            TCNBlock(in_ch, hidden_dim,
                kernel_size=5, dilation=1,
                dropout=dropout, drop_path_rate=dp_rates[0])
        ]
        for i in range(1, mid):
            first.append(
                TCNBlock(hidden_dim, hidden_dim,
                         kernel_size=3, dilation=min(2 ** (i - 1), 8),
                         dropout=dropout, drop_path_rate=dp_rates[i])
            )
        self.tcn_first = nn.Sequential(*first)

        # -- Seconda metà TCN --
        second = []
        for i in range(mid, tcn_blocks):
            second.append(
                TCNBlock(hidden_dim, hidden_dim,
                         kernel_size=3, dilation=min(2 ** (i - 1), 8),
                         dropout=dropout, drop_path_rate=dp_rates[i])
            )
        self.tcn_second = nn.Sequential(*second) if second else nn.Identity()

        # -- Temporal Attention --
        self.temporal_attn = TemporalAttention(
            dim=hidden_dim, heads=attn_heads, dropout=dropout * 0.5
        )

        # -- BiLSTM --
        self.bilstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.norm    = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)

        # -- Classificatore principale (condiviso con SR-CTC) --
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

        # -- SR-CTC: proiezione per allineare mid-TCN → dimensione fc --
        self.aux_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout * 0.5),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : Tensor (B, C=3, J, T)

        Returns (training) : (main_log_prob, aux_log_prob)  shape (B, T, num_classes) each
        Returns (inference): main_log_prob                  shape (B, T, num_classes)
        """
        B, C, J, T = x.shape
        x_flat = x.reshape(B, C * J, T)               # (B, 144, T)

        # Prima metà TCN → supervisione SR-CTC
        feat_mid = self.tcn_first(x_flat)              # (B, hidden_dim, T)
        feat_mid = feat_mid.permute(0, 2, 1)           # (B, T, hidden_dim)

        # Seconda metà TCN
        feat = self.tcn_second(
            feat_mid.permute(0, 2, 1)
        ).permute(0, 2, 1)                             # (B, T, hidden_dim)

        # Temporal attention
        feat = self.temporal_attn(feat)                # (B, T, hidden_dim)

        # SR-CTC ausiliario (solo training)
        if self.training:
            aux_feat     = self.aux_proj(feat_mid)            # (B, T, hidden_dim*2)
            aux_log_prob = torch.log_softmax(self.fc(aux_feat), dim=-1)

        # BiLSTM → output principale
        feat, _       = self.bilstm(feat)              # (B, T, hidden_dim*2)
        feat          = self.norm(feat)
        feat          = self.dropout(feat)
        main_log_prob = torch.log_softmax(self.fc(feat), dim=-1)

        if self.training:
            return main_log_prob, aux_log_prob
        return main_log_prob

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """Residual TCN block with dilated convolution.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Kernel size.
        dilation: Dilation rate.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=dilation * (kernel_size - 1) // 2,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=dilation * (kernel_size - 1) // 2,
                dilation=dilation,
            ),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor with shape (B, C, T).

        Returns:
            Tensor with shape (B, out_channels, T).
        """
        residual = self.skip(x)
        return F.relu(self.net(x) + residual)


class PoseNetworkTCN(nn.Module):
    """Temporal Convolutional Network for pose keypoints.

    Args:
        num_joints: Number of joints.
        hidden_dim: Hidden dimension.
        num_blocks: Number of TCN blocks.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_joints: int = 48,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = num_joints * 3

        layers = [
            TCNBlock(self.in_channels, hidden_dim, kernel_size=5, dilation=1, dropout=dropout),
        ]
        for i in range(num_blocks - 1):
            dilation = min(2**i, 8)
            layers.append(
                TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=dilation, dropout=dropout)
            )

        self.tcn = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x_flat: Tensor with shape (B, 3 * J, T).

        Returns:
            Tensor with shape (B, T, hidden_dim).
        """
        tcn_out = self.tcn(x_flat)
        return tcn_out.permute(0, 2, 1)


class PoseNetworkCTC(nn.Module):
    """TCN + BiLSTM + CTC model for pose sequences.

    Args:
        num_classes: Number of output classes.
        num_joints: Number of joints.
        hidden_dim: Hidden dimension.
        tcn_blocks: TCN blocks.
        lstm_layers: LSTM layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_classes: int,
        num_joints: int = 48,
        hidden_dim: int = 256,
        tcn_blocks: int = 4,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.pose_network = PoseNetworkTCN(
            num_joints=num_joints,
            hidden_dim=hidden_dim,
            num_blocks=tcn_blocks,
            dropout=dropout,
        )

        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor with shape (B, 3, J, T).

        Returns:
            Log probabilities with shape (B, T, C).
        """
        b, c, j, t = x.shape
        x_flat = x.reshape(b, c * j, t)
        feat = self.pose_network(x_flat)
        feat, _ = self.bilstm(feat)
        feat = self.norm(feat)
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return torch.log_softmax(logits, dim=-1)


def build_model(
    num_classes: int,
    num_joints: int,
    hidden_dim: int,
    tcn_blocks: int,
    lstm_layers: int,
    dropout: float,
) -> PoseNetworkCTC:
    """Factory for PoseNetworkCTC.

    Args:
        num_classes: Number of output classes.
        num_joints: Number of joints.
        hidden_dim: Hidden dimension.
        tcn_blocks: TCN blocks.
        lstm_layers: LSTM layers.
        dropout: Dropout rate.

    Returns:
        PoseNetworkCTC instance.
    """
    return PoseNetworkCTC(
        num_classes=num_classes,
        num_joints=num_joints,
        hidden_dim=hidden_dim,
        tcn_blocks=tcn_blocks,
        lstm_layers=lstm_layers,
        dropout=dropout,
    )

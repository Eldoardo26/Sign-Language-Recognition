"""Attention modules for the Signformer encoder/decoder.

Provides:
- MultiHeadAttention: standard scaled dot-product MHA (returns output, attn_weights)
- DeformableMultiHeadedAttention: deformable temporal attention with optional CoPE
- RelPosMultiHeadSelfAttention: relative positional self-attention (returns x, attn, None)
- ContextualMultiHeadAttention: contextual position encoding (CoPE) attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with separate Q/K/V projections.
    Returns (output, attention_weights)."""

    def __init__(self, dim_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim_model % num_heads == 0
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads

        self.q_proj = nn.Linear(dim_model, dim_model)
        self.k_proj = nn.Linear(dim_model, dim_model)
        self.v_proj = nn.Linear(dim_model, dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                mask: Optional[Tensor] = None):
        B = q.size(0)
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(q).view(B, -1, H, D).transpose(1, 2)
        k = self.k_proj(k).view(B, -1, H, D).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, H, D).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.dim_model)
        return self.out_proj(out), attn


class CoPE(nn.Module):
    """Contextual Position Encoding — computes position-dependent bias
    from attention gates, used inside DeformableMultiHeadedAttention."""

    def __init__(self, npos_max: int, head_dim: int):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))

    def forward(self, query: Tensor, attn_logits: Tensor) -> Tensor:
        gates = torch.sigmoid(attn_logits)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor.float()
        return logits_ceil * w + logits_floor * (1 - w)


class DeformableMultiHeadedAttention(nn.Module):
    """Deformable temporal attention for sign language sequences.

    Each head learns a set of reference offsets (query_nb points) and attends
    only to those sampled positions via bilinear interpolation."""

    def __init__(self, query_type: str = "attention", size: int = 256,
                 query_nb: int = 7, num_heads: int = 8, cope: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        assert size % num_heads == 0
        self.size = size
        self.num_heads = num_heads
        self.head_dim = size // num_heads
        self.query_nb = query_nb

        self.q_proj = nn.Linear(size, size)
        self.k_proj = nn.Linear(size, size)
        self.v_proj = nn.Linear(size, size)
        self.out_proj = nn.Linear(size, size)

        self.offset_net = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.GELU(),
            nn.Linear(self.head_dim, query_nb),
        )
        self.attn_net = nn.Linear(self.head_dim, query_nb)
        self.dropout = nn.Dropout(dropout)

        self.cope = CoPE(npos_max=2048, head_dim=self.head_dim) if cope else None

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        B, T, _ = q.shape
        H, D, K = self.num_heads, self.head_dim, self.query_nb

        q_h = self.q_proj(q).view(B, T, H, D).permute(0, 2, 1, 3)  # (B, H, T, D)
        v_h = self.v_proj(v).view(B, T, H, D).permute(0, 2, 1, 3)

        offsets = self.offset_net(q_h) * (T / 4.0)  # (B, H, T, K)
        ref = torch.arange(T, device=q.device, dtype=q.dtype).view(1, 1, T, 1)
        sample_pos = (ref + offsets).clamp(0, T - 1)

        pos_floor = sample_pos.long().clamp(0, T - 1)
        pos_ceil = (pos_floor + 1).clamp(0, T - 1)
        w = sample_pos - pos_floor.float()

        v_exp = v_h.unsqueeze(3).expand(-1, -1, -1, K, -1)
        v_floor = torch.gather(v_exp, 2, pos_floor.unsqueeze(-1).expand(-1, -1, -1, -1, D))
        v_ceil = torch.gather(v_exp, 2, pos_ceil.unsqueeze(-1).expand(-1, -1, -1, -1, D))
        sampled_v = v_floor * (1 - w.unsqueeze(-1)) + v_ceil * w.unsqueeze(-1)

        attn_logits = self.attn_net(q_h)  # (B, H, T, K)

        if self.cope is not None:
            attn_logits = attn_logits + self.cope(q_h, attn_logits)

        attn_weights = self.dropout(F.softmax(attn_logits, dim=-1))
        out = (sampled_v * attn_weights.unsqueeze(-1)).sum(dim=3)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.size)
        return self.out_proj(out)


class RelPosMultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with relative positional encoding.
    Returns (output, attention_weights, None)."""

    def __init__(self, dim_model: int, num_heads: int,
                 causal: bool = False, max_pos: int = 5000,
                 dropout: float = 0.1):
        super().__init__()
        assert dim_model % num_heads == 0
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        self.causal = causal

        self.q_proj = nn.Linear(dim_model, dim_model)
        self.k_proj = nn.Linear(dim_model, dim_model)
        self.v_proj = nn.Linear(dim_model, dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)

        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_pos - 1, num_heads))
        self.max_pos = max_pos
        self.dropout = nn.Dropout(dropout)

    def _get_rel_pos_bias(self, T: int):
        positions = torch.arange(T, device=self.rel_pos_bias.device)
        rel = positions.unsqueeze(0) - positions.unsqueeze(1) + self.max_pos - 1
        rel = rel.clamp(0, 2 * self.max_pos - 2)
        return self.rel_pos_bias[rel].permute(2, 0, 1).unsqueeze(0)

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                mask: Optional[Tensor] = None):
        B, T, _ = q.shape
        H, D = self.num_heads, self.head_dim

        q_h = self.q_proj(q).view(B, T, H, D).transpose(1, 2)
        k_h = self.k_proj(k).view(B, T, H, D).transpose(1, 2)
        v_h = self.v_proj(v).view(B, T, H, D).transpose(1, 2)

        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(D)
        scores = scores + self._get_rel_pos_bias(T)

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(T, T, device=q.device, dtype=torch.bool), 1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v_h)
        out = out.transpose(1, 2).contiguous().view(B, T, self.dim_model)
        return self.out_proj(out), attn, None


class ContextualMultiHeadAttention(nn.Module):
    """Multi-head attention with Contextual Position Encoding (CoPE).
    Returns a single tensor."""

    def __init__(self, dim_model: int, num_heads: int, max_pos: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        assert dim_model % num_heads == 0
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads

        self.q_proj = nn.Linear(dim_model, dim_model)
        self.k_proj = nn.Linear(dim_model, dim_model)
        self.v_proj = nn.Linear(dim_model, dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)

        self.cope = CoPE(npos_max=max_pos, head_dim=self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        B = q.size(0)
        H, D = self.num_heads, self.head_dim

        q_h = self.q_proj(q).view(B, -1, H, D).transpose(1, 2)
        k_h = self.k_proj(k).view(B, -1, H, D).transpose(1, 2)
        v_h = self.v_proj(v).view(B, -1, H, D).transpose(1, 2)

        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(D)
        scores = scores + self.cope(q_h, scores)

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v_h)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.dim_model)
        return self.out_proj(out)

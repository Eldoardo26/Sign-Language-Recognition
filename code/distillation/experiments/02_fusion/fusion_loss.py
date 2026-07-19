# -*- coding: utf-8 -*-
"""
fusion_loss.py — multi-objective loss for the cross-attention two-stream fusion.

    L = L_joint(CTC)                          # main, on the fused head
      + alpha * (L_app + L_skel)              # per-stream CTC, keeps both honest
      + beta  * L_crossKD                     # bidirectional KL between the two
                                              #   stream posteriors (mutual learning)

L_crossKD is the key term: it aligns the two streams' *decisions* (not their raw
features, which is what made distillation neutral). Each stream teaches the other
"which gloss", symmetrically -- so complementary evidence flows both ways without
forcing either representation to imitate the other.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionLoss(nn.Module):
    def __init__(self, blank: int = 0, alpha: float = 0.3, beta: float = 0.5,
                 kd_tau: float = 2.0):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=True)
        self.alpha, self.beta, self.tau = alpha, beta, kd_tau

    def _ctc(self, logp_BTC, targets, in_lens, tgt_lens):
        return self.ctc(logp_BTC.permute(1, 0, 2), targets, in_lens, tgt_lens)

    def _cross_kd(self, lp_a, lp_b, mask):
        """Symmetric KL between two (B,T,C) log-prob streams over valid frames."""
        m = mask.unsqueeze(-1).float()
        pa, pb = lp_a.exp(), lp_b.exp()
        # detach each side as the target of the other (mutual learning)
        kl_ab = F.kl_div(lp_a, pb.detach(), reduction="none").sum(-1, keepdim=True)
        kl_ba = F.kl_div(lp_b, pa.detach(), reduction="none").sum(-1, keepdim=True)
        denom = m.sum().clamp(min=1.0)
        return ((kl_ab + kl_ba) * m).sum() / denom

    def forward(self, out, targets, in_lens, tgt_lens, mask):
        """out: dict from CrossModalFusion. mask: (B,T) bool, True=valid."""
        l_joint = self._ctc(out["joint"], targets, in_lens, tgt_lens)
        l_app = self._ctc(out["app"], targets, in_lens, tgt_lens)
        l_skel = self._ctc(out["skel"], targets, in_lens, tgt_lens)
        l_kd = self._cross_kd(out["app"], out["skel"], mask)
        total = l_joint + self.alpha * (l_app + l_skel) + self.beta * l_kd
        return total, {"joint": float(l_joint), "app": float(l_app),
                       "skel": float(l_skel), "crossKD": float(l_kd)}

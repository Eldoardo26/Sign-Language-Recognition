"""
losses.py — Funzioni di loss per il training CTC.

Classi:
    CTCLossWithEntropy   — CTC + entropy regularization per stabilità numerica
    CTCLossWithSmoothing — Wrapper per compatibilità (delega a CTCLossWithEntropy)
"""

import numpy as np
import torch
import torch.nn as nn


class CTCLossWithEntropy(nn.Module):
    """
    CTC Loss con regularizzazione entropica.

    Aggiunge un termine di entropia normalizzato per:
    - Evitare distribuzioni troppo peaked (collasso su blank)
    - Migliorare la stabilità durante le prime epoche

    Formula:
        loss = CTC(log_probs) + entropy_weight * H(probs) / log(C)

    dove H(probs) = -sum(p * log(p)) e C = numero di classi.
    """

    def __init__(self, blank: int = 0, entropy_weight: float = 0.05):
        super().__init__()
        self.blank          = blank
        self.entropy_weight = entropy_weight
        self.ctc            = nn.CTCLoss(blank=blank, reduction="mean",
                                         zero_infinity=True)

    def forward(
        self,
        log_probs_TBC: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        T, B, C = log_probs_TBC.shape

        ctc_loss = self.ctc(log_probs_TBC, targets, input_lengths, target_lengths)

        if self.entropy_weight > 0:
            probs       = torch.exp(log_probs_TBC).clamp(min=1e-10)
            entropy     = -(probs * torch.log(probs)).sum(dim=-1).mean()
            max_entropy = np.log(C)
            entropy_reg = entropy / max(max_entropy, 1e-8)
            return ctc_loss + self.entropy_weight * entropy_reg

        return ctc_loss


class CTCLossWithSmoothing(nn.Module):
    """
    Wrapper per compatibilità — delega a CTCLossWithEntropy.
    Il parametro 'smoothing' corrisponde a entropy_weight.
    """

    def __init__(self, blank: int = 0, smoothing: float = 0.05):
        super().__init__()
        self.ctc = CTCLossWithEntropy(blank=blank, entropy_weight=smoothing)

    def forward(
        self,
        log_probs_TBC: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self.ctc(log_probs_TBC, targets, input_lengths, target_lengths)

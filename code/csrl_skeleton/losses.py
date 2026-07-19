"""CTC loss with entropy regularisation, greedy/beam decoding, corpus-level WER."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLossWithEntropy(nn.Module):
    def __init__(self, blank: int = 0, entropy_weight: float = 0.05):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=True)
        self.ew = entropy_weight

    def forward(self, log_probs_TBC, targets, in_lens, tgt_lens):
        loss = self.ctc(log_probs_TBC, targets, in_lens, tgt_lens)
        if self.ew > 0:
            p = torch.exp(log_probs_TBC).clamp(min=1e-10)
            H = -(p * torch.log(p)).sum(-1).mean() / max(math.log(log_probs_TBC.size(-1)), 1e-8)
            loss = loss + self.ew * H
        return loss


def collapse_ctc(ids: list[int]) -> list[int]:
    out, prev = [], None
    for v in ids:
        v = int(v)
        if v != 0 and v != prev:
            out.append(v)
        prev = v
    return out


@torch.no_grad()
def greedy_decode(log_probs_TBC: torch.Tensor,
                  beta: float = 0.0,
                  log_prior: torch.Tensor | None = None) -> list[list[int]]:
    T, B, C = log_probs_TBC.shape
    res = []
    for b in range(B):
        lp = log_probs_TBC[:, b, :]
        if beta > 0 and log_prior is not None:
            lp = lp - beta * log_prior.unsqueeze(0)
        res.append(collapse_ctc(lp.argmax(-1).tolist()))
    return res


@torch.no_grad()
def beam_decode(log_probs_TBC: torch.Tensor,
                beam_width: int = 25, beta: float = 0.3,
                log_prior: torch.Tensor | None = None,
                topk: int | None = None) -> list[list[int]]:
    T, B, C = log_probs_TBC.shape
    topk = min(topk or beam_width, C)
    res = []
    for b in range(B):
        lp = log_probs_TBC[:, b, :]
        if beta > 0 and log_prior is not None:
            lp = lp - beta * log_prior.unsqueeze(0)
            lp = lp - torch.logsumexp(lp, dim=1, keepdim=True)
        lp = lp.cpu().numpy()

        beams = {((), None): 0.0}
        for t in range(T):
            cand = np.argpartition(lp[t], -topk)[-topk:]
            if 0 not in cand:
                cand = np.append(cand, 0)
            nb = {}
            for (pre, last), sc in beams.items():
                for c in cand:
                    c = int(c)
                    nlp = sc + lp[t, c]
                    if c == 0 or c == last:
                        # Blank, or a repeat of the current label: under the CTC
                        # collapse both leave the emitted prefix unchanged. Fold the
                        # probability mass back into that prefix instead of dropping
                        # it. The previous `continue` discarded the repeat mass, which
                        # biased the decoder towards emitting new labels and made beam
                        # search score several points worse than greedy.
                        key = (pre, last)
                    else:
                        key = (pre + (c,), c)
                    if key not in nb or nb[key] < nlp:
                        nb[key] = nlp
            beams = dict(sorted(nb.items(), key=lambda kv: -kv[1])[:beam_width])

        best = max(beams, key=lambda k: beams[k] / max(len(k[0]), 1) ** 0.7)
        res.append(list(best[0]))
    return res


def compute_wer(refs: list[list[int]],
                hyps: list[list[int]]) -> tuple[float, dict]:
    tS = tD = tI = tN = 0
    for ref, hyp in zip(refs, hyps):
        r, h = len(ref), len(hyp)
        tN += r
        if r == 0:
            tI += h
            continue
        dp = [[0] * (h + 1) for _ in range(r + 1)]
        for i in range(1, r + 1):
            dp[i][0] = i
        for j in range(1, h + 1):
            dp[0][j] = j
        for i in range(1, r + 1):
            for j in range(1, h + 1):
                dp[i][j] = (
                    dp[i-1][j-1] if ref[i-1] == hyp[j-1]
                    else 1 + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
                )
        # backtrace for S/D/I counts
        i, j = r, h
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
                i -= 1; j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                tS += 1; i -= 1; j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                tD += 1; i -= 1
            else:
                tI += 1; j -= 1

    N = max(tN, 1)
    return (tS + tD + tI) / N, {"S": tS, "D": tD, "I": tI, "N": N}

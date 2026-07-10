"""Thin wrappers around sacrebleu for BLEU and chrF scoring.

Falls back to a simple BLEU implementation if sacrebleu is not installed.
"""

from collections import namedtuple

try:
    from sacrebleu import corpus_chrf as _corpus_chrf
    from sacrebleu import raw_corpus_bleu as _raw_corpus_bleu

    def corpus_chrf(hypotheses, references):
        return _corpus_chrf(hypotheses, [references])

    def raw_corpus_bleu(sys_stream, ref_streams):
        return _raw_corpus_bleu(sys_stream, ref_streams)

except ImportError:
    import math
    from collections import Counter

    _Score = namedtuple("_Score", ["score", "scores"])

    def _ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def raw_corpus_bleu(sys_stream, ref_streams):
        refs = ref_streams[0]
        max_n = 4
        total_correct = [0] * max_n
        total_total = [0] * max_n
        total_hyp_len = 0
        total_ref_len = 0

        for hyp, ref in zip(sys_stream, refs):
            hyp_tok = hyp.strip().split()
            ref_tok = ref.strip().split()
            total_hyp_len += len(hyp_tok)
            total_ref_len += len(ref_tok)
            for n in range(1, max_n + 1):
                hyp_ng = Counter(_ngrams(hyp_tok, n))
                ref_ng = Counter(_ngrams(ref_tok, n))
                clipped = {ng: min(c, ref_ng[ng]) for ng, c in hyp_ng.items()}
                total_correct[n - 1] += sum(clipped.values())
                total_total[n - 1] += max(len(hyp_tok) - n + 1, 0)

        scores = []
        log_bleu = 0.0
        for n in range(max_n):
            if total_total[n] == 0 or total_correct[n] == 0:
                scores.append(0.0)
                log_bleu = float("-inf")
            else:
                p = total_correct[n] / total_total[n]
                scores.append(p * 100)
                if log_bleu > float("-inf"):
                    log_bleu += math.log(p) / max_n

        bp = min(1.0, math.exp(1 - total_ref_len / max(total_hyp_len, 1)))
        bleu = bp * math.exp(log_bleu) * 100 if log_bleu > float("-inf") else 0.0
        return _Score(score=bleu, scores=scores)

    _ChrfScore = namedtuple("_ChrfScore", ["score"])

    def corpus_chrf(hypotheses, references):
        total = 0.0
        n = len(hypotheses)
        for hyp, ref in zip(hypotheses, references):
            hyp_chars = set(hyp)
            ref_chars = set(ref)
            if not ref_chars:
                continue
            p = len(hyp_chars & ref_chars) / max(len(hyp_chars), 1)
            r = len(hyp_chars & ref_chars) / len(ref_chars)
            if p + r > 0:
                total += 2 * p * r / (p + r)
        return _ChrfScore(score=total / max(n, 1))

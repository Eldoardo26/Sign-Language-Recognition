"""ROUGE-L score computation (MS-COCO style).

Computes the longest common subsequence based ROUGE-L score between
hypothesis and reference sentences, following the MS-COCO evaluation.
"""


def _lcs_length(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def calc_score(hypotheses, references):
    """Compute ROUGE-L F1 score between hypothesis and reference lists.

    :param hypotheses: list of hypothesis strings
    :param references: list of reference strings
    :return: average ROUGE-L F1 score (0..1)
    """
    total = 0.0
    count = 0
    for hyp, ref in zip(hypotheses, references):
        hyp_tok = hyp.strip().split()
        ref_tok = ref.strip().split()
        if not ref_tok:
            continue
        lcs = _lcs_length(hyp_tok, ref_tok)
        prec = lcs / max(len(hyp_tok), 1)
        rec = lcs / len(ref_tok)
        if prec + rec > 0:
            total += 2 * prec * rec / (prec + rec)
        count += 1
    return total / max(count, 1)

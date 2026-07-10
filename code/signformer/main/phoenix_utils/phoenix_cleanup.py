"""PHOENIX-2014 / PHOENIX-2014T gloss string cleanup functions.

These normalize gloss sequences before WER evaluation to match
the official evaluation protocol.
"""

import re


def clean_phoenix_2014(gloss_str: str) -> str:
    """Clean gloss string for PHOENIX-2014 (recognition only).

    Removes location markers (locN, loc-N), punctuation, and normalizes whitespace.
    """
    gloss_str = gloss_str.strip()
    gloss_str = re.sub(r"\bloc-?\d+\b", "", gloss_str, flags=re.IGNORECASE)
    gloss_str = re.sub(r"__[A-Z]+__", "", gloss_str)
    gloss_str = re.sub(r"[.,!?;:\-\"]", "", gloss_str)
    gloss_str = re.sub(r"\s+", " ", gloss_str).strip()
    return gloss_str


def clean_phoenix_2014_trans(gloss_str: str) -> str:
    """Clean gloss string for PHOENIX-2014T (recognition + translation).

    Same cleanup as PHOENIX-2014, but also removes special markers and
    normalizes compound glosses.
    """
    gloss_str = gloss_str.strip()
    gloss_str = re.sub(r"\bloc-?\d+\b", "", gloss_str, flags=re.IGNORECASE)
    gloss_str = re.sub(r"__[A-Z]+__", "", gloss_str)
    gloss_str = re.sub(r"[.,!?;:\-\"]", "", gloss_str)
    gloss_str = re.sub(r"\s+", " ", gloss_str).strip()
    return gloss_str

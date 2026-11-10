"""Text normalization applied to both reference and hypothesis before scoring.

This is a self-contained English normalizer in the spirit of the Whisper
English normalizer used by the Hugging Face Open ASR Leaderboard, but smaller
and simpler. It does not replicate every rule, so scores will not match the
leaderboard to the decimal. What matters here is that reference and hypothesis
go through the same steps. See docs/metrics.md for the rule list.
"""

from __future__ import annotations

import re
import unicodedata

_CONTRACTIONS = [
    (r"\bwon't\b", "will not"),
    (r"\bcan't\b", "can not"),
    (r"\bain't\b", "is not"),
    (r"\blet's\b", "let us"),
    (r"n't\b", " not"),
    (r"'re\b", " are"),
    (r"'ve\b", " have"),
    (r"'ll\b", " will"),
    (r"'m\b", " am"),
    (r"'d\b", " would"),
]

_FILLERS = {"um", "uh", "umm", "uhh", "hmm", "mm", "mhm", "ah", "er", "erm"}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**9, "billion"), (10**6, "million"), (10**3, "thousand")]


def number_to_words(n: int) -> str:
    """Spell out a non-negative integer in plain American English (no 'and')."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = "" if n % 100 == 0 else " " + number_to_words(n % 100)
        return _ONES[n // 100] + " hundred" + rest
    for size, name in _SCALES:
        if n >= size:
            rest = n % size
            return number_to_words(n // size) + " " + name + ("" if rest == 0 else " " + number_to_words(rest))
    return str(n)  # pragma: no cover


def _digits_to_words(text: str) -> str:
    def repl(m: re.Match) -> str:
        raw = m.group(0).replace(",", "")
        try:
            return " " + number_to_words(int(raw)) + " "
        except (ValueError, IndexError):
            return m.group(0)

    text = re.sub(r"(?<=\d)\.(?=\d)", " point ", text)
    return re.sub(r"\d[\d,]*", repl, text)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and diacritics, expand contractions, spell digits, drop fillers."""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = re.sub(r"[\[\(][^\]\)]*[\]\)]", " ", t)  # bracketed asides such as [noise]
    t = t.replace("%", " percent ").replace("&", " and ").replace("$", " dollars ")
    t = _digits_to_words(t)
    for pat, rep in _CONTRACTIONS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"'s\b", "", t)  # possessive and "is" are both dropped
    t = t.replace("'", "")
    t = re.sub(r"[^a-z0-9\s]", " ", t)  # hyphens and all other punctuation become spaces
    words = [w for w in t.split() if w not in _FILLERS]
    return " ".join(words)

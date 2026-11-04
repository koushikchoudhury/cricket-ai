"""Numeric faithfulness: does every number in an answer come from its evidence?

Small models answer with the right idea and the wrong detail: "120/2" when the tool said
100/2, "102 balls" when it said 12. A correct-looking answer with an invented number is
worse than no answer, so evaluations check that each number in the answer appears in the
question or in what the tools / retrieved sources returned. Rounding is allowed (83.9 in
the evidence supports "84" in the answer).
"""

from __future__ import annotations

import re

NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")


def numbers(text: str) -> list[str]:
    return [m.group(0).replace(",", "") for m in NUMBER.finditer(text or "")]


def _supported(value: str, evidence: list[float]) -> bool:
    a = float(value)
    decimals = len(value.split(".")[1]) if "." in value else 0
    return any(abs(round(m, decimals) - a) < 1e-9 or abs(m - a) < 1e-9 for m in evidence)


def unsupported_numbers(answer: str, *evidence: str) -> list[str]:
    """Numbers in ``answer`` that appear nowhere in the evidence (after rounding)."""
    pool = [float(n) for text in evidence for n in numbers(text)]
    return [n for n in numbers(answer) if not _supported(n, pool)]


def is_faithful(answer: str, *evidence: str) -> bool:
    return not unsupported_numbers(answer, *evidence)

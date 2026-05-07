"""Deterministic regex slot extractor for property facts.

Pulls numeric facts out of a listing description so that downstream rewrites
can be checked for fact preservation. Numeric facts ONLY — qualitative claims
("luxurious", "sun-drenched") are out of scope by design.

Schema: beds, baths, sqft, year_built, lot_sqft, parking, stories. Missing
slots are returned as None. Tolerates the common Redfin formatting quirks
(written-out numbers up to ten, "two-car garage", "1,400 square feet", etc.).
"""
from __future__ import annotations

import re
from typing import Optional

_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_NUM = r"(\d+(?:,\d{3})*(?:\.\d+)?)"
_INT = r"(\d+)"


def _w2n(s: str) -> Optional[float]:
    s = s.strip().lower().replace(",", "")
    if s in _WORD_NUM:
        return float(_WORD_NUM[s])
    try:
        return float(s)
    except ValueError:
        return None


def _first(text: str, patterns: list[str]) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            v = _w2n(m.group(1))
            if v is not None:
                return v
    return None


def extract_slots(text: str) -> dict[str, Optional[float]]:
    """Return canonical numeric slots from a description. Always returns the
    full schema with None for missing values."""
    if not isinstance(text, str):
        text = ""
    t = text.replace("–", "-").replace("—", "-")
    word = "(?:one|two|three|four|five|six|seven|eight|nine|ten)"

    beds = _first(t, [
        rf"{_INT}\s*[-\s]?bed(?:room)?s?\b",
        rf"({word})\s*[-\s]?bed(?:room)?s?\b",
        rf"\b{_INT}\s*br\b",
    ])
    baths = _first(t, [
        rf"{_NUM}\s*[-\s]?bath(?:room)?s?\b",
        rf"({word})\s*[-\s]?bath(?:room)?s?\b",
        rf"\b{_NUM}\s*ba\b",
    ])
    sqft = _first(t, [
        rf"{_NUM}\s*(?:sq\.?\s*ft\.?|square\s*feet|square\s*foot|sf)\b",
        rf"approximately\s*{_NUM}\s*(?:sq|square)",
    ])
    year = _first(t, [
        r"built\s+in\s+(\d{4})\b",
        r"\bcirca\s+(\d{4})\b",
        r"originally\s+(?:built|crafted|constructed)\s+in\s+(\d{4})",
        r"\b(\d{4})\s*(?:-built|construction)\b",
    ])
    lot = _first(t, [
        rf"lot\s*(?:size|area)?\s*(?:of|:)?\s*{_NUM}\s*(?:sq|square)",
        rf"{_NUM}\s*(?:sq\.?\s*ft\.?|square\s*feet)\s*lot\b",
    ])
    parking = _first(t, [
        rf"{_INT}\s*[-\s]?car\s*garage",
        rf"({word})\s*[-\s]?car\s*garage",
        rf"parking\s*for\s*{_INT}\b",
        rf"{_INT}\s*parking\s*spaces?",
    ])
    stories = _first(t, [
        rf"{_INT}[-\s]?stor(?:y|ies|ey|eys)\b",
        rf"({word})[-\s]?stor(?:y|ies|ey|eys)\b",
        rf"{_INT}[-\s]?level\b",
    ])

    return {
        "beds": beds,
        "baths": baths,
        "sqft": sqft,
        "year_built": year,
        "lot_sqft": lot,
        "parking": parking,
        "stories": stories,
    }


def slots_match(a: dict[str, Optional[float]], b: dict[str, Optional[float]]) -> dict[str, bool]:
    """Per-slot equality check tolerating None on either side as match-by-default
    (we only fail when BOTH sides extract a value AND they disagree)."""
    out: dict[str, bool] = {}
    for k in a:
        va, vb = a[k], b.get(k)
        if va is None or vb is None:
            out[k] = True
        else:
            out[k] = abs(float(va) - float(vb)) < 1e-6
    return out

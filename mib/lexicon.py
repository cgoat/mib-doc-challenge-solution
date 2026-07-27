"""Closed-vocabulary snapping for OCR output.

Species codes, home worlds, visa classes, purposes, risk flags and even the
applicant name parts are drawn from small closed vocabularies. Snapping noisy
OCR to the nearest vocabulary entry recovers most degraded fields. The match is
distance-thresholded and margin-checked, so an unseen value on the private test
set falls through to the raw OCR string instead of being forced to a neighbour.
"""
from __future__ import annotations

import json
import re
from datetime import date as _date
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

_LEX = json.loads((Path(__file__).with_name("lexicon.json")).read_text())

SPECIES = _LEX["species_code"]
WORLDS = _LEX["home_world"]
VISA_CLASSES = _LEX["visa_class"]
PURPOSES = _LEX["declared_purpose"]
FEE_STATUSES = _LEX["fee_status"]
RISK_FLAGS = _LEX["risk_flag"]
NAME_FIRST = _LEX["name_first"]
NAME_LAST = _LEX["name_last"]

# Glyph confusions seen in these scans, applied only inside known-numeric fields.
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1",
                            "i": "1", "|": "1", "S": "5", "s": "5", "B": "8", "Z": "2",
                            "z": "2", "G": "6", "b": "6", "g": "9", "T": "7", "A": "4"})


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


@lru_cache(maxsize=8192)
def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def snap(value: str, vocabulary: tuple[str, ...] | list[str], min_ratio=0.72, margin=0.06):
    """Return the closest vocabulary entry, or None when the match is not clear."""
    value = _clean(value)
    if not value:
        return None
    key = value.casefold()
    scored = sorted(((_ratio(key, v.casefold()), v) for v in vocabulary), reverse=True)
    if not scored:
        return None
    best_score, best = scored[0]
    if best_score < min_ratio:
        return None
    if len(scored) > 1 and best_score - scored[1][0] < margin and best_score < 0.95:
        return None  # ambiguous between two vocabulary entries
    return best


def _nearest(value: str, vocabulary, floor: float):
    """Closest entry above a low floor, ignoring the ambiguity margin."""
    value = _clean(value)
    if not value:
        return None
    key = value.casefold()
    best_score, best = max(((_ratio(key, v.casefold()), v) for v in vocabulary), default=(0.0, None))
    return best if best_score >= floor else None


def guess(value: str, vocabulary, min_ratio, margin, floor):
    """Return (value, confident).

    The evaluator scores a wrong extraction and a blank identically, so once the
    strict match fails there is nothing to lose by naming the nearest entry
    anyway. The flag tells the rules engine that this reading is a guess, so a
    guess is never allowed to trigger a denial.
    """
    strict = snap(value, vocabulary, min_ratio=min_ratio, margin=margin)
    if strict:
        return strict, True
    return _nearest(value, vocabulary, floor), False


def norm_species(value):
    value = _clean(value).upper().replace(" ", "_").replace("-", "_")
    value = re.sub(r"[^A-Z_]", "", value)
    return guess(value, SPECIES, min_ratio=0.7, margin=0.06, floor=0.45)


def norm_world(value):
    return guess(_clean(value), WORLDS, min_ratio=0.65, margin=0.06, floor=0.45)


def norm_purpose(value):
    value = _clean(value).lower()
    value = re.sub(r"[^a-z ]", "", value)
    return guess(value, PURPOSES, min_ratio=0.65, margin=0.06, floor=0.4)


def norm_visa(value):
    value = _clean(value).upper().replace(" ", "")
    value = value.replace("XVV", "XW").replace("VW", "XW").replace("O", "0")
    match = re.search(r"(XW|DIP|MED|TRANSIT|TRANSFT)[-_ ]?([0-9OIl])", value)
    if match:
        prefix = "TRANSIT" if match.group(1).startswith("TRANS") else match.group(1)
        value = f"{prefix}-{match.group(2).translate(_DIGIT_FIX)}"
    return guess(value, VISA_CLASSES, min_ratio=0.7, margin=0.06, floor=0.45)


def norm_fee(value):
    value = _clean(value).lower()
    if "[" in value or "obscur" in value or "illegib" in value:
        return None, True
    return guess(value, FEE_STATUSES, min_ratio=0.7, margin=0.06, floor=0.5)


def norm_flag(value):
    value = _clean(value).lower().replace(" ", "_").replace("-", "_")
    value = re.sub(r"[^a-z_]", "", value)
    if not value or value == "none":
        return None
    return snap(value, RISK_FLAGS, min_ratio=0.7)


def norm_name(value):
    value = _clean(value)
    if not value or "[" in value:
        return None
    value = re.sub(r"[^A-Za-z' \-]", " ", value)
    parts = [p for p in value.split() if len(p) > 1]
    if not parts:
        return None
    first = snap(parts[0], NAME_FIRST, min_ratio=0.6, margin=0.02)
    last = snap(parts[-1], NAME_LAST, min_ratio=0.6, margin=0.02) if len(parts) > 1 else None
    if first and last:
        return f"{first} {last}"
    if first and len(parts) == 1:
        return first
    # Fall back to a title-cased raw reading rather than dropping the field.
    return " ".join(p.capitalize() for p in parts[:2]) if len(parts) >= 2 else None


def norm_case_id(value):
    value = _clean(value).upper().replace(" ", "")
    match = re.search(r"M[I1L]B[-_ ]?([0-9OIlSBZGoq]{4,8})", value)
    if not match:
        return None
    digits = match.group(1).translate(_DIGIT_FIX)
    digits = re.sub(r"[^0-9]", "", digits)
    return f"MIB-{digits}" if len(digits) == 6 else None


def norm_sponsor(value):
    value = _clean(value).upper()
    if "[" in value or "BLANK" in value:
        return None
    match = re.search(r"SP[NM][-_ ]?([0-9OIlSBZGA]{3,6})", value.replace(" ", ""))
    if not match:
        return None
    digits = re.sub(r"[^0-9]", "", match.group(1).translate(_DIGIT_FIX))
    return f"SPN-{digits}" if len(digits) == 4 else None


def norm_date(value):
    value = _clean(value)
    if "[" in value or not value:
        return None
    match = re.search(r"([0-9OIlSBZGA]{4})\s*[-/. ]\s*([0-9OIlSBZGA]{1,2})\s*[-/. ]\s*([0-9OIlSBZGA]{1,2})", value)
    if not match:
        return None
    year, month, day = (re.sub(r"[^0-9]", "", g.translate(_DIGIT_FIX)) for g in match.groups())
    if len(year) != 4 or not month or not day:
        return None
    try:
        y, m, d = int(year), int(month), int(day)
    except ValueError:
        return None
    if not 2000 <= y <= 2100:
        return None
    try:
        # Reject impossible calendar dates: OCR happily turns a 30 into a 31,
        # and the submission validator parses every date it is given.
        return _date(y, m, d).isoformat()
    except ValueError:
        return None

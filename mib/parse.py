"""Turn page lines (native text layer or OCR) into typed field candidates."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import lexicon as lx

# Damage markers the generator prints in place of destroyed evidence.
DAMAGE_PAT = re.compile(r"\[[A-Z /]{3,40}\]")

_LABELS = {
    "case_id": ("case id", "casei d", "cese id", "case 1d", "case d"),
    "applicant_name": ("applicant", "registry name", "applicant name", "name"),
    "species_code": ("species code", "species match", "species"),
    "home_world": ("home world", "homeworld", "home wortd"),
    "visa_class": ("visa class", "class"),
    "sponsor_id": ("sponsor id", "sponsor"),
    "arrival_date": ("arrival date", "arival date", "date of arrival"),
    "declared_purpose": ("declared purpose", "purpose"),
    "fee_status": ("fee status", "fee"),
    "waiver_code": ("waiver code", "waiver"),
    "registry_status": ("registry status",),
    "risk_flags": ("observed flags", "observed nags", "flags", "risk flags"),
    "biometric_confidence": ("biometric confidence", "biometric conf"),
}

_ADJ_WORDS = {"APPROVED": ("approved",), "DENIED": ("denied",), "NEEDS_REVIEW": ("needs_review", "needs review")}


@dataclass
class PageFields:
    kind: str
    source: str
    values: dict[str, str] = field(default_factory=dict)
    # Fields whose value is the nearest vocabulary match rather than a clear
    # one. Reported, but never allowed to drive a denial.
    uncertain: set[str] = field(default_factory=set)
    damaged: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    adjudication: str | None = None
    finding_reason: str = ""


# Decorative overlay stamps the scans print on top of the form body.
_JUNK = ("passport image", "scan image", "registry image", "mib eyes only",
         "primary intake record", "synthetic hiring challenge document",
         "scan tab", "casework", "copy artifact", "mib archive", "archive copy")


def _score_label(key: str) -> tuple[str | None, float]:
    key = re.sub(r"[^a-z ]", " ", key.lower())
    key = re.sub(r"\s+", " ", key).strip()
    if not key:
        return None, 0.0
    best, best_score = None, 0.0
    for name, variants in _LABELS.items():
        for variant in variants:
            score = lx._ratio(key, variant)
            if score > best_score:
                best, best_score = name, score
    return best, best_score


def _label_of(text: str) -> str | None:
    name, score = _score_label(text)
    return name if score >= 0.82 else None


def _find_label_span(line: str):
    """Locate a label anywhere in a line and return (label, before, after).

    Native pages lay out forms in two columns, so a single visual row holds both
    the value and its label, in either order and sometimes with graphic
    placeholder text appended. OCR'd pages instead render 'Label: value'.
    """
    tokens = line.split()
    if not tokens:
        return None
    best = None
    for size in (1, 2, 3):
        for start in range(0, len(tokens) - size + 1):
            window_tokens = tokens[start:start + size]
            # A label is words only; a window that swallows the value would be
            # scored on its alphabetic remainder and win spuriously.
            if any(re.search(r"\d", t) for t in window_tokens):
                continue
            name, score = _score_label(" ".join(window_tokens))
            # Longest span wins: "Fee" alone would otherwise beat "Fee Status"
            # and leave "Status" looking like the value.
            if score >= 0.85 and (best is None or (size, score) > (best[3] - best[2], best[0])):
                best = (score, name, start, start + size)
    if best is None:
        return None
    _score, name, start, end = best
    before = " ".join(tokens[:start]).strip(" :•|-")
    after = " ".join(tokens[end:]).strip(" :•|-")
    return name, before, after


def _strip_junk(value: str) -> str:
    low = value.lower()
    for junk in _JUNK:
        index = low.find(junk)
        if index >= 0:
            value = value[:index] + value[index + len(junk):]
            low = value.lower()
    return re.sub(r"\s+", " ", value).strip(" :|-")


_TITLE_PAT = re.compile(r"form [ib]-|fee receipt|registry extract|attestation letter|adjudicator note", re.I)


def _split_pairs(lines: list[str]) -> list[tuple[str, list[str], bool]]:
    """Yield (label, candidate values, recovered_by_value) for each labelled row.

    Both sides of the label are returned as candidates because the two-column
    native layout puts the value to the left while OCR'd scans put it to the
    right; the caller picks whichever side normalizes to a valid value. The
    third element marks rows identified by their value rather than their label.
    """
    pairs: list[tuple[str, list[str], bool]] = []
    pending: str | None = None
    for raw in lines:
        line = raw.strip()
        if not line or _TITLE_PAT.search(line):
            continue
        found = _find_label_span(line)
        if found:
            name, before, after = found
            candidates = [c for c in (_strip_junk(after), _strip_junk(before)) if c]
            if candidates:
                pairs.append((name, candidates, False))
                pending = None
            else:
                pending = name
            continue
        if pending:
            value = _strip_junk(line)
            if value:
                pairs.append((pending, [value], False))
            pending = None
            continue
        recovered = _recover_by_value(line)
        if recovered:
            pairs.append((recovered[0], [line], True))
    return pairs


_NORMALIZERS = {
    "case_id": lx.norm_case_id,
    "applicant_name": lx.norm_name,
    "species_code": lx.norm_species,
    "home_world": lx.norm_world,
    "visa_class": lx.norm_visa,
    "sponsor_id": lx.norm_sponsor,
    "arrival_date": lx.norm_date,
    "declared_purpose": lx.norm_purpose,
    "fee_status": lx.norm_fee,
}


# Value-first recovery. When a scan mangles a label past the matching threshold
# ("Antval Date", "mnsor ID") the value beside it often survives intact. If a
# line carries a value that validates strictly for exactly one field, a much
# weaker resemblance to that field's label is enough to assign it - the value
# itself is doing the identifying.
_STRICT_VALUE = (
    ("arrival_date", lambda s: lx.norm_date(s)),
    ("sponsor_id", lambda s: lx.norm_sponsor(s)),
    ("species_code", lambda s: lx.snap(re.sub(r"[^A-Z_]", "", s.upper().replace(" ", "_")),
                                       lx.SPECIES, min_ratio=0.75, margin=0.08)),
    ("home_world", lambda s: lx.snap(s, lx.WORLDS, min_ratio=0.75, margin=0.08)),
)
_WEAK_LABEL_RATIO = 0.6


def _resembles_label(line: str, field: str) -> bool:
    variants = _LABELS.get(field, ())
    tokens = [t for t in line.split() if not re.search(r"\d", t)]
    for size in (1, 2):
        for start in range(0, max(len(tokens) - size + 1, 0)):
            window = re.sub(r"[^a-z ]", " ", " ".join(tokens[start:start + size]).lower())
            window = re.sub(r"\s+", " ", window).strip()
            if not window:
                continue
            if any(lx._ratio(window, v) >= _WEAK_LABEL_RATIO for v in variants):
                return True
    return False


def _recover_by_value(line: str):
    """Return (field, value) when a line identifies itself by its value."""
    hits = []
    for field, validator in _STRICT_VALUE:
        try:
            value = validator(line)
        except Exception:
            value = None
        if value:
            hits.append((field, value))
    # Ambiguous lines are left alone; the point is to be sure, not greedy.
    if len(hits) != 1:
        return None
    field, value = hits[0]
    return (field, value) if _resembles_label(line, field) else None


def _apply(normalizer, text):
    """Normalizers for closed vocabularies return (value, confident); the
    regex-backed ones return just a value."""
    result = normalizer(text)
    return result if isinstance(result, tuple) else (result, True)


def parse_page(kind: str, lines: list[str], source: str) -> PageFields:
    out = PageFields(kind=kind, source=source)
    blob = "\n".join(lines)

    for label, candidates, recovered in _split_pairs(lines):
        if any(DAMAGE_PAT.search(c) for c in candidates):
            out.damaged.add(label)
            continue
        if label == "risk_flags":
            for candidate in candidates:
                flags = [lx.norm_flag(p) for p in re.split(r"[,|;/]| and ", candidate)]
                flags = sorted({f for f in flags if f})
                if flags or "none" in candidate.lower():
                    out.values.setdefault("risk_flags", "|".join(flags) if flags else "none")
                    break
            continue
        if label in ("waiver_code", "registry_status", "biometric_confidence"):
            out.values.setdefault(label, candidates[0])
            continue
        normalizer = _NORMALIZERS.get(label)
        if not normalizer:
            continue
        # Try every candidate for a confident reading before settling for a
        # guess: relaxing the match must not let junk on one side of the label
        # pre-empt the real value on the other.
        fallback = None
        for candidate in candidates:
            normalized, confident = _apply(normalizer, candidate)
            if normalized and confident:
                fallback = (normalized, True)
                break
            if normalized and fallback is None:
                fallback = (normalized, False)
        if fallback and label not in out.values:
            out.values[label] = fallback[0]
            # A value-identified row had no readable label, so the assignment
            # itself is a guess even when the value parsed cleanly.
            if not fallback[1] or recovered:
                out.uncertain.add(label)

    for match in DAMAGE_PAT.finditer(blob):
        token = match.group(0).lower()
        for key, words in (("applicant_name", ("name",)), ("species_code", ("species",)),
                           ("home_world", ("registry", "world")), ("visa_class", ("visa",)),
                           ("sponsor_id", ("sponsor",)), ("arrival_date", ("date",)),
                           ("declared_purpose", ("purpose",)), ("fee_status", ("fee",)),
                           ("risk_flags", ("risk",))):
            if any(w in token for w in words):
                out.damaged.add(key)

    if kind == "sponsor":
        _parse_sponsor_letter(blob, out)
    if kind == "note":
        _parse_note(lines, out)
    return out


def _parse_sponsor_letter(blob: str, out: PageFields) -> None:
    sponsor = lx.norm_sponsor(blob)
    if sponsor:
        out.values.setdefault("sponsor_id", sponsor)
    match = re.search(r"attests? that (.+?) is expected", blob, re.I | re.S)
    if match:
        name = lx.norm_name(match.group(1).replace("\n", " "))
        if name:
            out.values.setdefault("applicant_name", name)
    match = re.search(r"expected on Earth for (.+?)[.\n]", blob, re.I | re.S)
    if match:
        purpose, confident = _apply(lx.norm_purpose, match.group(1).replace("\n", " "))
        if purpose and "declared_purpose" not in out.values:
            out.values["declared_purpose"] = purpose
            if not confident:
                out.uncertain.add("declared_purpose")
    match = re.search(r"class ([A-Z]+-?\d)", blob)
    if match:
        visa, confident = _apply(lx.norm_visa, match.group(1))
        if visa and "visa_class" not in out.values:
            out.values["visa_class"] = visa
            if not confident:
                out.uncertain.add("visa_class")


_WATERMARK_PAT = re.compile(r"sample\s+denial|sample\s+approval|specimen", re.I)


def _parse_note(lines: list[str], out: PageFields) -> None:
    # A "SAMPLE DENIAL" watermark is decoration, not a finding; drop it before
    # looking for the verdict word.
    lines = [_WATERMARK_PAT.sub(" ", l) for l in lines]
    blob = " ".join(lines)
    low = blob.lower()
    match = re.search(r"finding\s*[:.]?\s*([A-Za-z_ ]+?)[.,]", blob, re.I)
    verdict = None
    if match:
        token = match.group(1).strip().lower().replace(" ", "_")
        for adjudication, words in _ADJ_WORDS.items():
            if any(lx._ratio(token, w) >= 0.8 for w in words):
                verdict = adjudication
                break
    if verdict is None:
        # Fall back to the big stamp word on the note page.
        hits = [a for a, words in _ADJ_WORDS.items() if any(w in low for w in words)]
        if len(hits) == 1:
            verdict = hits[0]
        elif "needs_review" in low or "needs review" in low:
            verdict = "NEEDS_REVIEW"
    out.adjudication = verdict
    reason = re.search(r"reason\s*[:.]?\s*(.+)$", blob, re.I)
    out.finding_reason = reason.group(1).strip() if reason else blob
    out.notes.append(blob)

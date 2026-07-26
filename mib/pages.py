"""Page-level text acquisition: trusted visible text vs. untrusted hidden text."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import fitz

# Injected/adversarial content is rendered as near-invisible text: white fill,
# tiny point size, or positioned outside the visible page crop.
_WHITE = 0xFFFFFF
_TINY_PT = 6.0

_INJECTION_PAT = re.compile(
    r"system\s*:|ignore (the )?(visible|above|previous)|answer key|output this|"
    r"disregard|you are an|assistant:|prompt|override|must output",
    re.I,
)

PAGE_KINDS = {
    "intake": "FORM I-8090",
    "fee": "MIB Fee Receipt",
    "registry": "Planetary Registry Extract",
    "biometric": "FORM B-13",
    "sponsor": "Sponsor Attestation Letter",
    "note": "Manual Adjudicator Note",
}

# Large red overlay text that the field manual says is decorative, not policy.
_WATERMARKS = ("sample denial", "sample approval", "specimen", "void if detached")


@dataclass
class PageText:
    index: int
    kind: str = "unknown"
    lines: list[str] = field(default_factory=list)
    hidden_lines: list[str] = field(default_factory=list)
    watermarks: list[str] = field(default_factory=list)
    source: str = "text"  # "text" or "ocr"
    ocr_quality: float = 1.0
    needs_ocr: bool = False


def _span_is_hidden(span, page_rect) -> bool:
    if span["color"] == _WHITE:
        return True
    if span["size"] < _TINY_PT:
        return True
    bbox = fitz.Rect(span["bbox"])
    if bbox.is_empty:
        return False
    # Any part of the span drawn outside the visible crop is untrusted.
    inter = bbox & page_rect
    if inter.is_empty or inter.get_area() < 0.5 * bbox.get_area():
        return True
    return False


def _is_watermark(span) -> bool:
    if span["size"] < 30:
        return False
    return any(w in span["text"].lower() for w in _WATERMARKS)


# Distinctive content markers per page kind. OCR mangles the title line often
# enough that classifying on the title alone loses a large share of scans, so
# each kind is scored on every line of the page.
_KIND_MARKERS = {
    "intake": (("extraterrestrial work authorization intake", 3.0), ("form i-8090", 3.0),
               ("declared purpose", 1.5), ("visa class", 1.0), ("sponsor id", 1.0),
               ("passport image", 1.0), ("primary intake record", 1.5),
               ("case id", 0.4), ("applicant", 0.4), ("species code", 0.4)),
    "biometric": (("biometric scan slip", 3.0), ("form b-13", 3.0), ("observed flags", 2.0),
                  ("biometric confidence", 2.0), ("species match", 1.5), ("scan image", 1.4)),
    "fee": (("mib fee receipt", 3.0), ("fee status", 2.0), ("waiver code", 1.5), ("amount", 0.8)),
    "registry": (("planetary registry extract", 3.0), ("registry status", 2.0),
                 ("registry name", 2.0), ("registry image", 1.6), ("home world", 0.4)),
    "sponsor": (("sponsor attestation letter", 3.0), ("to mib intake", 2.0), ("attests that", 2.0),
                ("acknowledges responsibility", 1.5), ("this attestation is valid", 1.5)),
    "note": (("manual adjudicator note", 3.0), ("finding", 1.5), ("reason", 1.0),
             ("adjudicator", 1.5), ("rescinded", 1.0)),
}

_MIN_KIND_SCORE = 1.3


def _fuzzy_contains(haystack: str, needle: str) -> float:
    """Best similarity of `needle` against any equal-length window of text."""
    if needle in haystack:
        return 1.0
    size = len(needle)
    if len(haystack) < size:
        return 0.0
    best = 0.0
    step = max(1, size // 4)
    for start in range(0, len(haystack) - size + 1, step):
        best = max(best, SequenceMatcher(None, haystack[start:start + size], needle).ratio())
        if best > 0.95:
            break
    return best


def classify_kind(lines: list[str]) -> str:
    blob = " \n".join(lines)
    for kind, marker in PAGE_KINDS.items():
        if marker in blob:
            return kind

    low = re.sub(r"\s+", " ", blob.lower())
    scores: dict[str, float] = {}
    for kind, markers in _KIND_MARKERS.items():
        total = 0.0
        for marker, weight in markers:
            similarity = _fuzzy_contains(low, marker)
            if similarity >= 0.78:
                total += weight * similarity
        scores[kind] = total
    if not scores:
        return "unknown"
    kind, score = max(scores.items(), key=lambda kv: kv[1])
    return kind if score >= _MIN_KIND_SCORE else "unknown"


def read_page(page: fitz.Page, index: int) -> PageText:
    """Split a page's text layer into trusted visible lines and untrusted lines."""
    out = PageText(index=index)
    rect = page.rect
    visible_spans: list[tuple[float, float, str]] = []
    hidden: list[str] = []

    data = page.get_text("dict")
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text.strip():
                    continue
                if _is_watermark(span):
                    out.watermarks.append(text.strip())
                    continue
                if _span_is_hidden(span, rect):
                    hidden.append(text.strip())
                    continue
                bbox = span["bbox"]
                visible_spans.append((round(bbox[1], 1), bbox[0], text))

    # Reassemble visible spans into reading-order lines.
    lines: list[str] = []
    visible_spans.sort(key=lambda s: (s[0], s[1]))
    current_y = None
    buf: list[str] = []
    for y, _x, text in visible_spans:
        if current_y is None or abs(y - current_y) > 2.0:
            if buf:
                lines.append(" ".join(buf).strip())
            buf = [text]
            current_y = y
        else:
            buf.append(text)
    if buf:
        lines.append(" ".join(buf).strip())

    out.lines = [l for l in (x.strip() for x in lines) if l]
    out.hidden_lines = hidden
    out.kind = classify_kind(out.lines)
    # A page whose text layer carries only the footer is a scan: it must be OCR'd.
    out.needs_ocr = out.kind == "unknown"
    return out


def has_injection(pages: list[PageText]) -> bool:
    for p in pages:
        for line in p.hidden_lines:
            if _INJECTION_PAT.search(line):
                return True
    return False

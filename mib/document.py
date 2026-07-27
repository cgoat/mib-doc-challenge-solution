"""Packet-level pipeline: read pages, OCR scans, merge evidence by precedence."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import fitz

from . import ocr, pages as pagemod
from .parse import PageFields, parse_page

# Field-manual evidence precedence (higher wins).
KIND_WEIGHT = {"note": 6, "intake": 5, "biometric": 4, "sponsor": 3, "registry": 2, "fee": 4, "unknown": 1}
# A clean text layer is more trustworthy than OCR of a degraded scan.
SOURCE_BONUS = {"text": 3.0, "ocr": 0.0}

FOOTER_PAT = re.compile(r"Packet\s+(MIB[-\s]?\d{6})\s*/\s*page", re.I)
HEADER_PAT = re.compile(r"(MIB-\d{6})\s*\|", re.I)

# Which labels a page yielded identifies it more reliably than its title does:
# a scan can lose its heading to noise and still parse every field cleanly.
# Ordered most to least distinctive - only the intake form carries a visa class,
# sponsor id or declared purpose, and only the biometric slip a flag panel.
_LABEL_SIGNATURE = (
    ("biometric", ("risk_flags", "biometric_confidence")),
    ("fee", ("fee_status", "waiver_code")),
    ("registry", ("registry_status",)),
    ("intake", ("visa_class", "sponsor_id", "declared_purpose")),
)


def infer_kind_from_labels(values: dict) -> str:
    for kind, keys in _LABEL_SIGNATURE:
        if any(key in values for key in keys):
            return kind
    # A registry extract that lost its status line still has exactly the
    # registry field set: identity and origin, but nothing about the visa.
    if {"home_world", "species_code", "arrival_date"} <= set(values):
        return "registry"
    return "unknown"

OUTPUT_FIELDS = ("applicant_name", "species_code", "home_world", "visa_class",
                 "sponsor_id", "arrival_date", "declared_purpose", "risk_flags", "fee_status")


@dataclass
class Packet:
    case_id: str | None = None
    fields: dict[str, str] = field(default_factory=dict)
    agreement: dict[str, float] = field(default_factory=dict)
    damaged: set[str] = field(default_factory=set)
    kinds: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    note_adjudication: str | None = None
    note_reason: str = ""
    notes: list[str] = field(default_factory=list)
    watermarks: list[str] = field(default_factory=list)
    injection: bool = False
    registry_status: str | None = None
    waiver_codes: list[str] = field(default_factory=list)
    page_count: int = 0
    ocr_pages: int = 0
    multi_applicant: bool = False
    error: str | None = None


def _case_id_from_pages(page_texts: list[pagemod.PageText]) -> tuple[str | None, bool]:
    """The footer stamps the active case id on every page, scanned or not."""
    counts: dict[str, int] = defaultdict(int)
    for page in page_texts:
        blob = "\n".join(page.lines)
        for match in FOOTER_PAT.finditer(blob):
            counts[match.group(1).upper().replace(" ", "-")] += 3
        for match in HEADER_PAT.finditer(blob):
            counts[match.group(1).upper()] += 1
    if not counts:
        return None, False
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return ranked[0][0], len(ranked) > 1


def read_packet(path, text_only: bool = False) -> Packet:
    """Extract one packet. `text_only` skips OCR when the run is out of time."""
    packet = Packet()
    try:
        doc = fitz.open(path)
    except Exception as exc:  # pragma: no cover - defensive
        packet.error = f"open_failed: {exc}"
        return packet

    page_texts: list[pagemod.PageText] = []
    parsed: list[PageFields] = []
    try:
        for index, page in enumerate(doc):
            text = pagemod.read_page(page, index)
            if text.needs_ocr and not text_only:
                try:
                    image = ocr.native_image(doc, page)
                    ocr_lines = ocr.ocr_page(image)
                except Exception:
                    ocr_lines = []
                if ocr_lines:
                    text.source = "ocr"
                    text.kind = pagemod.classify_kind(ocr_lines)
                    # Footer text lives in the real text layer; keep both.
                    text.lines = ocr_lines + text.lines
                    packet.ocr_pages += 1
            fields = parse_page(text.kind, text.lines, text.source)
            if text.kind == "unknown":
                inferred = infer_kind_from_labels(fields.values)
                if inferred != "unknown":
                    text.kind = inferred
                    # Re-parse so the sponsor-letter and adjudicator-note
                    # readers, which only run for their own page kind, get a
                    # chance at a page the title lookup had written off.
                    fields = parse_page(inferred, text.lines, text.source)
            page_texts.append(text)
            parsed.append(fields)
    finally:
        doc.close()

    packet.page_count = len(page_texts)
    packet.kinds = [p.kind for p in page_texts]
    packet.sources = [p.source for p in page_texts]
    packet.injection = pagemod.has_injection(page_texts)
    packet.watermarks = [w for p in page_texts for w in p.watermarks]
    packet.case_id, packet.multi_applicant = _case_id_from_pages(page_texts)

    votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for page_fields in parsed:
        weight = KIND_WEIGHT.get(page_fields.kind, 1) + SOURCE_BONUS.get(page_fields.source, 0.0)
        for name, value in page_fields.values.items():
            if name in ("waiver_code", "registry_status", "biometric_confidence"):
                continue
            votes[name][value] += weight
        packet.damaged |= page_fields.damaged
        if page_fields.adjudication and packet.note_adjudication is None:
            packet.note_adjudication = page_fields.adjudication
            packet.note_reason = page_fields.finding_reason
        packet.notes.extend(page_fields.notes)
        status = page_fields.values.get("registry_status")
        if status:
            packet.registry_status = status
        waiver = page_fields.values.get("waiver_code")
        if waiver:
            packet.waiver_codes.append(waiver)

    for name, tally in votes.items():
        ranked = sorted(tally.items(), key=lambda kv: -kv[1])
        packet.fields[name] = ranked[0][0]
        total = sum(tally.values())
        packet.agreement[name] = ranked[0][1] / total if total else 0.0

    return packet

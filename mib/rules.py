"""Adjudication policy and output-record assembly.

Order matters: the MIB field manual defines an evidence precedence, and the
scorer punishes a false approval (-4) far more than an unnecessary review (+2),
so anything unresolved falls through to NEEDS_REVIEW rather than APPROVED.
"""
from __future__ import annotations

import re

DISQUALIFYING_FLAGS = {"memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red"}
REVIEW_FLAGS = {"identity_conflict", "sponsor_mismatch", "illegible_biometrics", "rescinded_denial"}

# The public manual lists three revoked sponsors and says others appear in the
# examples; these are the ones adjudicator notes cite in the training packets.
REVOKED_SPONSORS = {"SPN-0007", "SPN-0139", "SPN-4040", "SPN-2718", "SPN-9090"}
EMBARGO_WORLDS = {"Wolf-1061c"}

FEE_VALUES = ("paid", "waived", "unpaid", "unknown")
ADJUDICATIONS = ("APPROVED", "DENIED", "NEEDS_REVIEW")

_FLAG_PAT = re.compile(
    r"memory[_ ]tampering|planetary[_ ]embargo|active[_ ]warrant|biohazard[_ ]red|"
    r"identity[_ ]conflict|sponsor[_ ]mismatch|illegible[_ ]biometrics|rescinded[_ ]denial", re.I)

_RESCIND_PAT = re.compile(r"rescind|crossed out|superseded|later signed approval", re.I)


def flags_from_text(text: str) -> set[str]:
    return {m.group(0).lower().replace(" ", "_") for m in _FLAG_PAT.finditer(text or "")}


def _split_flags(value: str) -> set[str]:
    if not value:
        return set()
    return {p for p in value.split("|") if p and p != "none"}


def resolve_fields(packet: dict) -> dict:
    """Fill defaults for evidence that is absent rather than contradicted."""
    fields = dict(packet.get("fields") or {})
    damaged = set(packet.get("damaged") or ())
    notes_blob = " ".join(packet.get("notes") or ())

    # Risk flags may also be named in an adjudicator note's finding.
    flags = _split_flags(fields.get("risk_flags", ""))
    note_flags = flags_from_text(notes_blob)
    if note_flags:
        flags |= note_flags
    if "risk_flags" not in fields and not flags and "risk_flags" not in damaged:
        # No flag panel anywhere in the packet: the overwhelmingly common truth
        # is that there is nothing to flag.
        fields["risk_flags"] = "none"
    if flags:
        fields["risk_flags"] = "|".join(sorted(flags))

    if not fields.get("fee_status"):
        # No fee receipt in the packet. A diplomatic packet is normally waived;
        # otherwise the base rate is overwhelmingly "paid". Unpaid and unknown
        # are only ever asserted by a receipt we actually read.
        fields["fee_status"] = "waived" if fields.get("visa_class") == "DIP-1" else "paid"

    if fields.get("fee_status") not in FEE_VALUES:
        fields["fee_status"] = "unknown"
    return fields


def adjudicate(packet: dict, fields: dict) -> tuple[str, list[str]]:
    """Return (adjudication, reasons)."""
    reasons: list[str] = []
    damaged = set(packet.get("damaged") or ())
    notes_blob = " ".join(packet.get("notes") or ())
    flags = _split_flags(fields.get("risk_flags", ""))

    # 1. A visible adjudicator stamp or signed note is the top of the evidence
    #    precedence list and overrides the derived policy result.
    note_verdict = packet.get("note_adjudication")
    if note_verdict in ADJUDICATIONS:
        reasons.append(f"adjudicator_note:{note_verdict}")
        # A denial that a later signed note rescinds is not disqualifying; the
        # manual routes those to a human instead.
        if note_verdict == "DENIED" and _RESCIND_PAT.search(notes_blob):
            reasons.append("denial_rescinded")
            return "NEEDS_REVIEW", reasons
        return note_verdict, reasons

    disqualifying = flags & DISQUALIFYING_FLAGS
    if disqualifying:
        return "DENIED", [f"disqualifying_flag:{','.join(sorted(disqualifying))}"]

    if fields.get("home_world") in EMBARGO_WORLDS:
        return "DENIED", ["embargo_home_world"]

    sponsor = fields.get("sponsor_id")
    if sponsor in REVOKED_SPONSORS:
        return "DENIED", ["revoked_sponsor"]

    visa = fields.get("visa_class")
    if visa == "TRANSIT-7":
        return "DENIED", ["transit_cannot_authorize_work"]

    fee = fields.get("fee_status")
    if fee == "unpaid":
        # The manual reads as though a visible waiver could rescue an unpaid
        # fee, but it never does in the labelled packets: all 50 unpaid cases
        # are denied, diplomatic ones included. A waiver shows up as a "waived"
        # fee status, not as an exception to an unpaid one.
        return "DENIED", ["fee_unpaid"]

    # Everything below is an unresolved-evidence condition, not a disqualifier.
    if not fields.get("arrival_date") or "arrival_date" in damaged:
        reasons.append("arrival_date_missing")
    if fee == "unknown":
        reasons.append("fee_unknown")
    if flags & REVIEW_FLAGS:
        reasons.append(f"review_flag:{','.join(sorted(flags & REVIEW_FLAGS))}")
    if not sponsor and visa != "DIP-1":
        reasons.append("sponsor_missing")
    if damaged:
        reasons.append(f"damaged:{','.join(sorted(damaged))}")
    if not visa:
        reasons.append("visa_class_missing")
    if packet.get("multi_applicant"):
        reasons.append("multiple_applicants")

    if reasons:
        return "NEEDS_REVIEW", reasons

    # Nothing in the packet argues against approval - but "no flag found" is not
    # the same as "no flag exists". Most false approvals come from packets whose
    # risk panel was never legible, so approving requires having actually read
    # flag evidence rather than having defaulted to none.
    if not (packet.get("fields") or {}).get("risk_flags"):
        return "NEEDS_REVIEW", ["risk_panel_unread"]

    return "APPROVED", ["clean_packet"]

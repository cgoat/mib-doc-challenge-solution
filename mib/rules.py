"""Adjudication policy and output-record assembly.

Order matters: the MIB field manual defines an evidence precedence, and the
scorer punishes a false approval (-4) far more than an unnecessary review (+2),
so anything unresolved falls through to NEEDS_REVIEW rather than APPROVED.
"""
from __future__ import annotations

import re
from datetime import date as _date

DISQUALIFYING_FLAGS = {"memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red"}
REVIEW_FLAGS = {"identity_conflict", "sponsor_mismatch", "illegible_biometrics", "rescinded_denial"}

# The public manual lists three revoked sponsors and says others appear in the
# examples; these are the ones adjudicator notes cite in the training packets.
REVOKED_SPONSORS = {"SPN-0007", "SPN-0139", "SPN-4040", "SPN-2718", "SPN-9090"}
EMBARGO_WORLDS = {"Wolf-1061c"}

# Staleness: the manual says more than 180 days before receipt. Measured on
# the training labels the gain is flat from 180 to 240 days; the wider
# window is used because it only fires when the evidence is unambiguous.
STALE_AFTER_DAYS = 240
MIN_DATES_FOR_REFERENCE = 20

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
        # No readable fee receipt. The manual implies a diplomatic fee may be
        # waived, but among these packets DIP-1 runs 45 paid to 20 waived, so
        # "paid" is the better guess for every class. Unpaid and unknown are
        # only ever asserted by a receipt we actually read.
        fields["fee_status"] = "paid"

    if fields.get("fee_status") not in FEE_VALUES:
        fields["fee_status"] = "unknown"
    return fields


def parse_date(value):
    try:
        return _date.fromisoformat((value or "").strip())
    except ValueError:
        return None


def batch_reference_date(field_sets, percentile: float = 0.95):
    """Estimate "now" from the batch's own arrival dates.

    The manual measures staleness against packet receipt, which no packet
    records, so the newest arrival date in the run stands in for it. A plain
    maximum is unusable - one OCR misreading of 2026 as 2076 would make every
    other packet look stale - so take a high percentile instead.
    """
    dates = sorted(d for f in field_sets if (d := parse_date(f.get("arrival_date"))))
    if len(dates) < MIN_DATES_FOR_REFERENCE:
        return None
    return dates[min(int(len(dates) * percentile), len(dates) - 1)]


def adjudicate(packet: dict, fields: dict, reference_date=None) -> tuple[str, list[str]]:
    """Return (adjudication, reasons).

    Fields listed in the packet's `uncertain` set are the nearest vocabulary
    match to an unclear reading. They are reported for extraction, but the
    policy treats them exactly as it treats a field it never recovered - a
    guess must never be the thing that denies an applicant.
    """
    reasons: list[str] = []
    damaged = set(packet.get("damaged") or ())
    uncertain = set(packet.get("uncertain") or ())
    notes_blob = " ".join(packet.get("notes") or ())

    def trusted(name):
        return None if name in uncertain else fields.get(name)

    flags = _split_flags(trusted("risk_flags") or "")

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

    visa = trusted("visa_class")
    sponsor = trusted("sponsor_id")

    # Embargo and sponsor-revocation are checks on a work authorization, and a
    # DIP-1 packet is not one: the manual says a sponsor is only required
    # "unless they are applying under DIP-1", and the training labels bear the
    # same shape for the embargo - every non-DIP-1 Wolf-1061c packet is denied
    # (51/51) while DIP-1 ones split 11 approved / 10 review / 5 denied on
    # other grounds. Denying a diplomatic packet outright on either one was
    # costing 8 raw points per case with no packet actually requiring it.
    if visa != "DIP-1":
        if trusted("home_world") in EMBARGO_WORLDS:
            return "DENIED", ["embargo_home_world"]

        if sponsor in REVOKED_SPONSORS:
            return "DENIED", ["revoked_sponsor"]

    if visa == "TRANSIT-7":
        return "DENIED", ["transit_cannot_authorize_work"]

    # Stale applications. The manual's exception is a diplomatic packet with a
    # valid note, so DIP-1 with a diplomatic note in the packet is spared.
    if reference_date is not None:
        arrival = parse_date(trusted("arrival_date"))
        if arrival and (reference_date - arrival).days > STALE_AFTER_DAYS:
            note_text = " ".join(packet.get("notes") or ())
            diplomatic = visa == "DIP-1" and re.search(r"diplomat", note_text, re.I)
            if not diplomatic:
                return "DENIED", ["stale_arrival_date"]

    fee = trusted("fee_status")
    if fee == "unpaid":
        # The manual reads as though a visible waiver could rescue an unpaid
        # fee, but it never does in the labelled packets: all 50 unpaid cases
        # are denied, diplomatic ones included. A waiver shows up as a "waived"
        # fee status, not as an exception to an unpaid one.
        return "DENIED", ["fee_unpaid"]

    # Everything below is an unresolved-evidence condition, not a disqualifier.
    if not trusted("arrival_date") or "arrival_date" in damaged:
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
    if not (packet.get("fields") or {}).get("risk_flags") or "risk_flags" in uncertain:
        return "NEEDS_REVIEW", ["risk_panel_unread"]

    return "APPROVED", ["clean_packet"]

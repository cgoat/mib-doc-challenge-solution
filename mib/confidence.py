"""Confidence estimation.

Calibration is scored as Brier error against "was this adjudication correct",
so the target is an honest probability rather than a high number. The strongest
predictor by far is *why* the decision was made - a signed adjudicator note is
right ~99% of the time while an unreadable risk panel is right ~25% - so the
decision reason, the amount of OCR the packet needed, and how much evidence was
actually recovered are fed to a small logistic model fitted on the training
packets by tools/fit_calibration.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REASON_HEADS = (
    "adjudicator_note", "disqualifying_flag", "embargo_home_world", "revoked_sponsor",
    "transit_cannot_authorize_work", "fee_unpaid", "arrival_date_missing", "fee_unknown",
    "review_flag", "sponsor_missing", "damaged", "visa_class_missing", "multiple_applicants",
    "risk_panel_unread", "clean_packet",
)

CORE_FIELDS = ("applicant_name", "species_code", "home_world", "visa_class", "arrival_date")

_MODEL_PATH = Path(__file__).with_name("calibration.json")
_MODEL = json.loads(_MODEL_PATH.read_text()) if _MODEL_PATH.exists() else None


def features(packet: dict, fields: dict, adjudication: str, reasons: list[str]) -> dict[str, float]:
    head = reasons[0].split(":")[0] if reasons else ""
    pages = max(int(packet.get("page_count") or 1), 1)
    raw = packet.get("fields") or {}
    agreement = packet.get("agreement") or {}

    values = {"bias": 1.0}
    for name in REASON_HEADS:
        values[f"reason={name}"] = 1.0 if head == name else 0.0
    values["ocr_fraction"] = float(packet.get("ocr_pages") or 0) / pages
    values["unknown_pages"] = sum(1 for k in (packet.get("kinds") or ()) if k == "unknown") / pages
    values["flags_observed"] = 1.0 if raw.get("risk_flags") else 0.0
    values["fee_observed"] = 1.0 if raw.get("fee_status") else 0.0
    values["core_recovered"] = sum(1 for f in CORE_FIELDS if fields.get(f)) / len(CORE_FIELDS)
    values["agreement"] = (sum(agreement.values()) / len(agreement)) if agreement else 0.0
    values["damaged"] = 1.0 if packet.get("damaged") else 0.0
    values["injection"] = 1.0 if packet.get("injection") else 0.0
    values["multi_applicant"] = 1.0 if packet.get("multi_applicant") else 0.0
    values["extra_reasons"] = min(len(reasons) - 1, 3) / 3.0
    return values


def confidence_for(packet: dict, fields: dict, adjudication: str, reasons: list[str]) -> float:
    values = features(packet, fields, adjudication, reasons)
    if _MODEL:
        weights = _MODEL["weights"]
        z = sum(weight * values.get(name, 0.0) for name, weight in weights.items())
        probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        lo, hi = _MODEL.get("clip", [0.03, 0.97])
    else:
        # Untrained fallback: lean on the reason alone.
        head = reasons[0].split(":")[0] if reasons else ""
        probability = {"adjudicator_note": 0.95, "disqualifying_flag": 0.93, "fee_unknown": 0.9,
                       "review_flag": 0.88, "transit_cannot_authorize_work": 0.85,
                       "clean_packet": 0.72}.get(head, 0.4)
        lo, hi = 0.05, 0.95
    return round(min(hi, max(lo, probability)), 4)

"""Unit tests for the parts of the pipeline that are easy to regress."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mib import lexicon as lx
from mib.pages import classify_kind
from mib.parse import parse_page
from mib.rules import adjudicate, resolve_fields


# --- layout handling -------------------------------------------------------

def test_native_two_column_row_puts_value_left_of_label():
    lines = ["MIB-000001 Case ID", "Ixodane Luzarn Applicant", "DIP-1 Visa Class",
             "2026-02-10 Arrival Date", "SPN-4732 Sponsor ID PASSPORT IMAGE"]
    values = parse_page("intake", lines, "text").values
    assert values["case_id"] == "MIB-000001"
    assert values["applicant_name"] == "Ixodane Luzarn"
    assert values["visa_class"] == "DIP-1"
    assert values["arrival_date"] == "2026-02-10"
    assert values["sponsor_id"] == "SPN-4732"


def test_ocr_style_row_puts_value_right_of_label():
    lines = ["Case ID: MIB-000003", "Applicant: Solix Qorquell", "Fee Status: paid"]
    values = parse_page("fee", lines, "ocr").values
    assert values["case_id"] == "MIB-000003"
    assert values["fee_status"] == "paid"


def test_generic_label_does_not_beat_the_full_label():
    # "Fee" alone must not win over "Fee Status" and leave "Status" as the value.
    assert parse_page("fee", ["paid Fee Status"], "text").values["fee_status"] == "paid"


def test_damage_marker_records_the_field_as_unrecoverable():
    out = parse_page("intake", ["Applicant: [NAME CUT OUT]", "Visa Class: [VISA CLASS TORN]"], "ocr")
    assert "applicant_name" in out.damaged
    assert "applicant_name" not in out.values


# --- OCR normalisation -----------------------------------------------------

@pytest.mark.parametrize("noisy,expected", [
    ("LUNA_SECURID", "LUNA_SECURID"),
    ("LUNA SECURID", "LUNA_SECURID"),
    ("LUNA_SECUR1D", "LUNA_SECURID"),
])
def test_species_snaps_to_vocabulary(noisy, expected):
    assert lx.norm_species(noisy) == expected


def test_unknown_species_is_not_forced_onto_a_neighbour():
    assert lx.norm_species("ZZQQXX_NOTREAL") is None


def test_date_and_sponsor_digit_confusions():
    assert lx.norm_date("2O26-O3-15") == "2026-03-15"
    assert lx.norm_sponsor("SPN-67S9") == "SPN-6759"
    assert lx.norm_sponsor("[SPONSOR ID BLANK]") is None


# --- adversarial content ---------------------------------------------------

def test_sample_denial_watermark_is_not_a_finding():
    out = parse_page("note", ["Manual Adjudicator Note", "SAMPLE DENIAL"], "ocr")
    assert out.adjudication is None


def test_hidden_answer_key_never_reaches_the_fields():
    # Hidden spans are dropped in pages.read_page, so the parser only ever sees
    # visible lines; a packet with an injected key still resolves from evidence.
    packet = {"fields": {"visa_class": "XW-1", "risk_flags": "none", "fee_status": "paid",
                         "arrival_date": "2026-03-15", "sponsor_id": "SPN-6799",
                         "applicant_name": "Solix Qorquell"},
              "injection": True, "kinds": ["intake"], "notes": [], "damaged": []}
    fields = resolve_fields(packet)
    assert adjudicate(packet, fields)[0] == "APPROVED"


# --- policy ----------------------------------------------------------------

def _packet(**fields):
    base = {"fields": dict(fields), "kinds": ["intake", "biometric"], "notes": [],
            "damaged": [], "page_count": 2, "ocr_pages": 0}
    return base


def test_disqualifying_flag_denies():
    p = _packet(risk_flags="biohazard_red", fee_status="paid", visa_class="MED-3",
                arrival_date="2026-01-01", sponsor_id="SPN-1234")
    assert adjudicate(p, resolve_fields(p))[0] == "DENIED"


def test_review_only_flag_routes_to_review():
    p = _packet(risk_flags="illegible_biometrics", fee_status="paid", visa_class="XW-2",
                arrival_date="2026-01-01", sponsor_id="SPN-1234")
    assert adjudicate(p, resolve_fields(p))[0] == "NEEDS_REVIEW"


def test_transit_class_cannot_authorize_work():
    p = _packet(risk_flags="none", fee_status="paid", visa_class="TRANSIT-7",
                arrival_date="2026-01-01", sponsor_id="SPN-1234")
    assert adjudicate(p, resolve_fields(p))[0] == "DENIED"


def test_adjudicator_note_overrides_derived_policy():
    p = _packet(risk_flags="none", fee_status="paid", visa_class="XW-2",
                arrival_date="2026-01-01", sponsor_id="SPN-1234")
    p["note_adjudication"] = "DENIED"
    p["notes"] = ["Finding: DENIED. Reason: Ambiguous packet."]
    assert adjudicate(p, resolve_fields(p))[0] == "DENIED"


def test_rescinded_denial_note_goes_to_human_review():
    p = _packet(risk_flags="none", fee_status="paid", visa_class="XW-2",
                arrival_date="2026-01-01", sponsor_id="SPN-1234")
    p["note_adjudication"] = "DENIED"
    p["notes"] = ["Prior denial stamp rescinded. Route to human review."]
    assert adjudicate(p, resolve_fields(p))[0] == "NEEDS_REVIEW"


def test_unread_risk_panel_blocks_approval():
    # No flag evidence anywhere: "no flag found" is not "no flag exists".
    p = _packet(fee_status="paid", visa_class="XW-2", arrival_date="2026-01-01",
                sponsor_id="SPN-1234")
    adjudication, reasons = adjudicate(p, resolve_fields(p))
    assert adjudication == "NEEDS_REVIEW"
    assert reasons[0] == "risk_panel_unread"


def test_missing_fee_receipt_defaults_by_visa_class():
    assert resolve_fields(_packet(visa_class="DIP-1"))["fee_status"] == "waived"
    assert resolve_fields(_packet(visa_class="XW-2"))["fee_status"] == "paid"


# --- page classification ---------------------------------------------------

def test_classifies_a_badly_ocrd_intake_title():
    assert classify_kind(["RM | 8090 Extraterrestrial Work Authc on Intake",
                          "Visa Class MED-2", "Applicant: x"]) == "intake"


def test_unreadable_page_stays_unknown():
    assert classify_kind(["SCAN TAB", "CASEWORK"]) == "unknown"

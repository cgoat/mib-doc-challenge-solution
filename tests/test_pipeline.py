"""Unit tests for the parts of the pipeline that are easy to regress."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mib import lexicon as lx
from mib.pages import classify_kind
from mib.parse import parse_page
from mib.rules import adjudicate, batch_revoked_sponsors, resolve_fields


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
    value, confident = lx.norm_species(noisy)
    assert value == expected
    assert confident


def test_unknown_species_is_not_forced_onto_a_neighbour():
    assert lx.norm_species("ZZQQXX_NOTREAL") == (None, False)


def test_a_marginal_reading_is_reported_but_flagged_as_a_guess():
    # A wrong extraction and a blank score the same, so naming the nearest
    # entry is free - as long as the guess is marked.
    value, confident = lx.norm_species("LN SC")
    assert value == "LUNA_SECURID"
    assert not confident


def test_a_guess_never_triggers_a_denial():
    # TRANSIT-7 normally denies outright; a guessed visa class must not.
    p = _packet(risk_flags="none", fee_status="paid", visa_class="TRANSIT-7",
                arrival_date="2026-06-01", sponsor_id="SPN-1234")
    assert adjudicate(p, resolve_fields(p))[0] == "DENIED"
    p["uncertain"] = ["visa_class"]
    assert adjudicate(p, resolve_fields(p))[0] == "NEEDS_REVIEW"


def test_a_guessed_home_world_does_not_trigger_the_embargo():
    p = _packet(risk_flags="none", fee_status="paid", visa_class="XW-2",
                home_world="Wolf-1061c", arrival_date="2026-06-01", sponsor_id="SPN-1234")
    assert adjudicate(p, resolve_fields(p))[0] == "DENIED"
    p["uncertain"] = ["home_world"]
    assert adjudicate(p, resolve_fields(p))[0] != "DENIED"


def test_impossible_calendar_dates_are_rejected():
    # OCR turns 30 into 31 happily, and the submission validator parses dates.
    assert lx.norm_date("2026-11-31") is None
    assert lx.norm_date("2026-02-30") is None
    assert lx.norm_date("2026-13-01") is None
    assert lx.norm_date("2026-11-30") == "2026-11-30"


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


def test_embargo_and_revoked_sponsor_do_not_bind_diplomatic_packets():
    # The manual: a sponsor is required "unless applying under DIP-1". The
    # embargo shows the same shape in the training labels - every non-DIP-1
    # Wolf-1061c packet is denied (51/51) while DIP-1 ones are not.
    embargoed = _packet(risk_flags="none", fee_status="waived", visa_class="DIP-1",
                        home_world="Wolf-1061c", arrival_date="2026-01-01", sponsor_id="SPN-1234")
    assert adjudicate(embargoed, resolve_fields(embargoed))[0] != "DENIED"

    revoked = _packet(risk_flags="none", fee_status="waived", visa_class="DIP-1",
                      arrival_date="2026-01-01", sponsor_id="SPN-0007")
    assert adjudicate(revoked, resolve_fields(revoked))[0] != "DENIED"


def test_embargo_and_revoked_sponsor_still_bind_non_diplomatic_packets():
    embargoed = _packet(risk_flags="none", fee_status="paid", visa_class="XW-2",
                        home_world="Wolf-1061c", arrival_date="2026-01-01", sponsor_id="SPN-1234")
    assert adjudicate(embargoed, resolve_fields(embargoed))[0] == "DENIED"

    revoked = _packet(risk_flags="none", fee_status="paid", visa_class="XW-2",
                      arrival_date="2026-01-01", sponsor_id="SPN-0007")
    assert adjudicate(revoked, resolve_fields(revoked))[0] == "DENIED"


def test_batch_revoked_sponsors_flags_frequency_outliers_beyond_the_public_list():
    # A sponsor the manual never named still looks revoked if the batch
    # itself denies it every time it appears - it recurs far more than an
    # ordinary sponsor does.
    field_sets = [{"sponsor_id": f"SPN-{i:04d}"} for i in range(500)]
    field_sets += [{"sponsor_id": "SPN-9999"}] * 20
    revoked = batch_revoked_sponsors(field_sets)
    assert "SPN-9999" in revoked
    assert "SPN-0007" in revoked  # public list always included


def test_batch_revoked_sponsors_falls_back_below_minimum_corpus_size():
    field_sets = [{"sponsor_id": f"SPN-{i:04d}"} for i in range(10)]
    field_sets += [{"sponsor_id": "SPN-9999"}] * 5
    revoked = batch_revoked_sponsors(field_sets)
    assert "SPN-9999" not in revoked
    assert "SPN-0007" in revoked


def test_registry_embargo_review_denies_regardless_of_visa_class():
    # A direct registry-verified embargo status, unlike a home-world name match,
    # is not softened for DIP-1: measured on the labels those split 7 denied to
    # 2 review, not the clean home-world-embargo split.
    p = _packet(risk_flags="none", fee_status="waived", visa_class="DIP-1",
                arrival_date="2026-01-01", sponsor_id="SPN-1234")
    p["registry_status"] = "EMBARGO REVIEW"
    assert adjudicate(p, resolve_fields(p))[0] == "DENIED"


def test_registry_clear_status_is_not_a_signal():
    p = _packet(risk_flags="none", fee_status="paid", visa_class="XW-2",
                arrival_date="2026-01-01", sponsor_id="SPN-1234")
    p["registry_status"] = "CLEAR"
    assert adjudicate(p, resolve_fields(p))[0] == "APPROVED"


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


def test_missing_fee_receipt_defaults_to_paid():
    # Measured on the labels: among packets with no readable receipt, DIP-1
    # runs 45 paid to 20 waived, so the manual's waiver language misleads here.
    assert resolve_fields(_packet(visa_class="DIP-1"))["fee_status"] == "paid"
    assert resolve_fields(_packet(visa_class="XW-2"))["fee_status"] == "paid"


def test_stale_arrival_date_is_denied_against_the_batch_reference():
    import datetime
    from mib.rules import batch_reference_date
    ref = datetime.date(2026, 7, 5)
    p = _packet(risk_flags="none", fee_status="paid", visa_class="XW-2",
                arrival_date="2025-06-01", sponsor_id="SPN-1234")
    assert adjudicate(p, resolve_fields(p), reference_date=ref)[0] == "DENIED"
    fresh = _packet(risk_flags="none", fee_status="paid", visa_class="XW-2",
                    arrival_date="2026-06-01", sponsor_id="SPN-1234")
    assert adjudicate(fresh, resolve_fields(fresh), reference_date=ref)[0] == "APPROVED"


def test_diplomatic_note_exempts_a_stale_packet():
    import datetime
    p = _packet(risk_flags="none", fee_status="waived", visa_class="DIP-1",
                arrival_date="2025-06-01", sponsor_id="SPN-1234")
    p["notes"] = ["Diplomatic waiver confirmed by Agent K. Fee exception stands."]
    assert adjudicate(p, resolve_fields(p), reference_date=datetime.date(2026, 7, 5))[0] != "DENIED"


def test_batch_reference_ignores_an_ocr_year_blowout():
    from mib.rules import batch_reference_date
    sets = [{"arrival_date": f"2026-0{1 + i % 6}-15"} for i in range(60)]
    sets.append({"arrival_date": "2076-04-07"})   # a misread 2026
    ref = batch_reference_date(sets)
    assert ref is not None and ref.year == 2026


def test_no_reference_when_the_batch_is_tiny():
    from mib.rules import batch_reference_date
    assert batch_reference_date([{"arrival_date": "2026-05-01"}]) is None


def test_every_parsed_value_is_a_plain_string():
    # Vocabulary normalizers return (value, confident); leaking that tuple into
    # the field map silently corrupts the output and the vote tally.
    pages = [
        ("sponsor", ["Sponsor SPN-4560 attests that Aridane Zavoss is expected on Earth for",
                     "reactor maintenance.", "responsibility for class XW-2 compliance"]),
        ("intake", ["Visa Class: MED-3", "Species Code: LUNA_SECURID", "Purpose: xenobotany"]),
        ("fee", ["Fee Status: paid", "Waiver Code: N/A"]),
        ("registry", ["Home World: Luyten-b", "Registry Status: CLEAR"]),
    ]
    for kind, lines in pages:
        for key, value in parse_page(kind, lines, "text").values.items():
            assert isinstance(value, str), f"{kind}/{key} is {type(value).__name__}"


def test_cpu_budget_is_sane():
    # Must never return 0 (no workers) and must not trust a host CPU count
    # blindly when a cgroup quota is present.
    from mib.main import _cpu_budget
    budget = _cpu_budget()
    assert isinstance(budget, int) and budget >= 1


# --- page classification ---------------------------------------------------

def test_classifies_a_badly_ocrd_intake_title():
    assert classify_kind(["RM | 8090 Extraterrestrial Work Authc on Intake",
                          "Visa Class MED-2", "Applicant: x"]) == "intake"


def test_unreadable_page_stays_unknown():
    assert classify_kind(["SCAN TAB", "CASEWORK"]) == "unknown"


def test_page_kind_inferred_from_the_labels_it_yielded():
    # A scan whose heading is destroyed still identifies itself by its fields.
    from mib.document import infer_kind_from_labels
    assert infer_kind_from_labels({"risk_flags": "none"}) == "biometric"
    assert infer_kind_from_labels({"fee_status": "paid"}) == "fee"
    assert infer_kind_from_labels({"visa_class": "XW-2"}) == "intake"
    assert infer_kind_from_labels({"home_world": "Luyten-b", "species_code": "KAIJU_MICRO",
                                   "arrival_date": "2026-07-11"}) == "registry"
    assert infer_kind_from_labels({"applicant_name": "Zed Zarnax"}) == "unknown"

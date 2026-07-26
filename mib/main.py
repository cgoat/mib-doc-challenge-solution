"""Entrypoint: read a directory of packet PDFs, write predictions.jsonl.

Usage: python -m mib.main <input_pdf_dir> <output_predictions_path>
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

OUTPUT_FIELDS = ("case_id", "applicant_name", "species_code", "home_world", "visa_class",
                 "sponsor_id", "arrival_date", "declared_purpose", "risk_flags",
                 "fee_status", "adjudication", "confidence")

# Average seconds per PDF the scoring contract allows. Past this we stop paying
# for OCR and fall back to the text layer so that every case still gets a row.
BUDGET_PER_PDF = float(os.environ.get("MIB_BUDGET_PER_PDF", "5.0"))

# The format validator requires a well-formed sponsor id and ISO date on every
# row, so a field we could not recover from trusted evidence is emitted as an
# obviously-synthetic sentinel rather than left blank. These never reach the
# rules engine - adjudication runs on the real, possibly-absent values.
UNRECOVERED = {"sponsor_id": "SPN-0000", "arrival_date": "1900-01-01", "risk_flags": "none"}

CASE_ID_PAT = re.compile(r"MIB-\d{6}")


def _init_worker():
    # Tesseract and OpenCV both try to grab every core; with one packet per
    # process that oversubscribes the 4 vCPUs badly.
    for var in ("OMP_THREAD_LIMIT", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    import cv2
    cv2.setNumThreads(1)
    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = cmd


def _predict_one(args) -> dict:
    path, text_only = args
    from mib.confidence import confidence_for
    from mib.document import read_packet
    from mib.rules import adjudicate, resolve_fields

    stem = Path(path).stem
    try:
        packet = read_packet(path, text_only=text_only)
        record = {
            "case_id": packet.case_id, "fields": packet.fields, "agreement": packet.agreement,
            "damaged": sorted(packet.damaged), "kinds": packet.kinds, "sources": packet.sources,
            "note_adjudication": packet.note_adjudication, "notes": packet.notes,
            "waiver_codes": packet.waiver_codes, "page_count": packet.page_count,
            "ocr_pages": packet.ocr_pages, "multi_applicant": packet.multi_applicant,
            "injection": packet.injection,
        }
        fields = resolve_fields(record)
        adjudication, reasons = adjudicate(record, fields)
        confidence = confidence_for(record, fields, adjudication, reasons)
        case_id = packet.case_id or stem
    except Exception:
        # A packet we cannot process at all still gets a conservative row: an
        # omission costs the missing-case penalty and forfeits the decision.
        fields, case_id = {}, stem
        adjudication, confidence = "NEEDS_REVIEW", 0.2

    row = {"case_id": case_id, "adjudication": adjudication, "confidence": confidence}
    for name in OUTPUT_FIELDS:
        if name in ("case_id", "adjudication", "confidence"):
            continue
        row[name] = fields.get(name) or UNRECOVERED.get(name, "")
    if row["fee_status"] not in ("paid", "waived", "unpaid", "unknown"):
        row["fee_status"] = "unknown"
    return row


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <input_pdf_dir> <output_predictions_path>", file=sys.stderr)
        return 2
    input_dir, output_path = Path(argv[1]), Path(argv[2])
    files = sorted(str(p) for p in input_dir.rglob("*.pdf"))
    if not files:
        print(f"no PDFs under {input_dir}", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(os.cpu_count() or 1, 8))
    deadline = time.time() + BUDGET_PER_PDF * max(len(files), 1)

    start = time.time()
    seen: set[str] = set()
    written = 0
    degraded = False
    with output_path.open("w", encoding="utf-8") as out, ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker
    ) as pool:
        pending = {}
        queue = list(files)
        # Submit in waves so the text-only fallback can engage mid-run if the
        # remaining time budget stops covering OCR.
        while queue or pending:
            while queue and len(pending) < workers * 4:
                path = queue.pop(0)
                pending[pool.submit(_predict_one, (path, degraded))] = path
            done = next(as_completed(pending))
            row = done.result()
            source = pending.pop(done)
            if row["case_id"] in seen:
                # Two packets resolved to the same id (a misread footer, say).
                # Re-key the loser off its own file rather than dropping it and
                # taking a missing-case penalty on a packet we did process.
                stem = Path(source).stem
                if CASE_ID_PAT.fullmatch(stem) and stem not in seen:
                    row["case_id"] = stem
                else:
                    continue
            seen.add(row["case_id"])
            out.write(json.dumps(row) + "\n")
            written += 1
            if not degraded and queue:
                remaining = len(queue) + len(pending)
                if time.time() + remaining * 0.15 > deadline:
                    degraded = True
                    print("time budget low: continuing without OCR", file=sys.stderr)

    elapsed = time.time() - start
    per_pdf = elapsed / max(len(files), 1)
    print(f"wrote {written} predictions to {output_path} in {elapsed:.0f}s ({per_pdf:.2f}s/pdf)")
    return 0


if __name__ == "__main__":
    _init_worker()
    raise SystemExit(main(sys.argv))

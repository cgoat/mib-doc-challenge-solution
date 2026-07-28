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
from concurrent.futures.process import BrokenProcessPool
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

# Recycle workers periodically so the OCR runtime's memory arenas are returned
# instead of growing across thousands of packets.
TASKS_PER_WORKER = 250
MAX_POOL_RESTARTS = 5


def _cpu_budget() -> int:
    """CPUs this container may actually use.

    os.cpu_count() reports the host's CPUs, not the cgroup quota - inside a
    `--cpus 4` container on a 32-core host it returns 32. Believing it starts
    twice as many OCR workers as there are CPUs to run them, and the extra
    copies of the model runtime are what exhausted an 8 GiB limit part-way
    through a 5,000-packet run.
    """
    for quota_path, period_path in (
        ("/sys/fs/cgroup/cpu.max", None),
        ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    ):
        try:
            raw = Path(quota_path).read_text().split()
            if period_path is None:
                quota, period = raw[0], raw[1]
                if quota == "max":
                    continue
            else:
                quota, period = raw[0], Path(period_path).read_text().strip()
            cpus = int(int(quota) / int(period))
            if cpus >= 1:
                return cpus
        except Exception:
            continue
    return os.cpu_count() or 1


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


def _decide(record: dict, fields: dict, case_id: str, reference_date=None) -> dict:
    from mib.confidence import confidence_for
    from mib.rules import adjudicate

    adjudication, reasons = adjudicate(record, fields, reference_date=reference_date)
    confidence = confidence_for(record, fields, adjudication, reasons)
    row = {"case_id": case_id, "adjudication": adjudication, "confidence": confidence}
    for name in OUTPUT_FIELDS:
        if name in ("case_id", "adjudication", "confidence"):
            continue
        row[name] = fields.get(name) or UNRECOVERED.get(name, "")
    if row["fee_status"] not in ("paid", "waived", "unpaid", "unknown"):
        row["fee_status"] = "unknown"
    return row


def _predict_one(args) -> dict:
    path, text_only = args
    from mib.document import read_packet
    from mib.rules import resolve_fields

    stem = Path(path).stem
    try:
        packet = read_packet(path, text_only=text_only)
        record = {
            "case_id": packet.case_id, "fields": packet.fields, "agreement": packet.agreement,
            "damaged": sorted(packet.damaged), "uncertain": sorted(packet.uncertain),
            "kinds": packet.kinds, "sources": packet.sources,
            "note_adjudication": packet.note_adjudication, "notes": packet.notes,
            "waiver_codes": packet.waiver_codes, "registry_status": packet.registry_status,
            "page_count": packet.page_count,
            "ocr_pages": packet.ocr_pages, "multi_applicant": packet.multi_applicant,
            "injection": packet.injection,
        }
        fields = resolve_fields(record)
        case_id = packet.case_id or stem
    except Exception:
        # A packet we cannot process at all still gets a conservative row: an
        # omission costs the missing-case penalty and forfeits the decision.
        record, fields, case_id = {}, {}, stem

    # Decided provisionally here so a killed container still leaves a usable
    # file; main() revisits every decision once the batch reference date exists.
    return {"row": _decide(record, fields, case_id), "record": record,
            "fields": fields, "case_id": case_id}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <input_pdf_dir> <output_predictions_path>", file=sys.stderr)
        return 2
    input_dir, output_path = Path(argv[1]), Path(argv[2])
    files = sorted(str(p) for p in input_dir.rglob("*.pdf"))
    if not files:
        print(f"no PDFs under {input_dir}", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(_cpu_budget(), 8))
    deadline = time.time() + BUDGET_PER_PDF * max(len(files), 1)

    start = time.time()
    seen: set[str] = set()
    results: list[dict] = []
    degraded = False
    queue = list(files)
    restarts = 0

    with output_path.open("w", encoding="utf-8") as out:
        while queue and restarts <= MAX_POOL_RESTARTS:
            pending: dict = {}
            try:
                with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                         max_tasks_per_child=TASKS_PER_WORKER) as pool:
                    # Submit in waves so the text-only fallback can engage
                    # mid-run if the remaining time budget stops covering OCR.
                    while queue or pending:
                        while queue and len(pending) < workers * 4:
                            path = queue.pop(0)
                            pending[pool.submit(_predict_one, (path, degraded))] = path
                        done = next(as_completed(pending))
                        result = done.result()
                        source = pending.pop(done)
                        case_id = result["case_id"]
                        if case_id in seen:
                            # Two packets resolved to the same id (a misread
                            # footer, say). Re-key the loser off its own file
                            # rather than dropping it and taking a missing-case
                            # penalty on a packet we did process.
                            stem = Path(source).stem
                            if CASE_ID_PAT.fullmatch(stem) and stem not in seen:
                                case_id = result["case_id"] = result["row"]["case_id"] = stem
                            else:
                                continue
                        seen.add(case_id)
                        results.append(result)
                        out.write(json.dumps(result["row"]) + "\n")
                        if not degraded and queue:
                            remaining = len(queue) + len(pending)
                            if time.time() + remaining * 0.15 > deadline:
                                degraded = True
                                print("time budget low: continuing without OCR", file=sys.stderr)
            except BrokenProcessPool:
                # A worker died - out of memory, or a library segfaulting on one
                # packet. Losing the whole run over it would forfeit thousands of
                # cases, so put the in-flight packets back and start a fresh
                # pool with fewer workers.
                restarts += 1
                queue = [p for p in pending.values()] + queue
                workers = max(1, workers // 2)
                print(f"worker pool died; restarting ({restarts}) with {workers} workers, "
                      f"{len(queue)} packets left", file=sys.stderr)
                out.flush()

    # Second pass. Staleness is measured against packet receipt, which no packet
    # states, so it can only be judged once the whole batch has been read.
    from mib.rules import batch_reference_date

    reference = batch_reference_date([r["fields"] for r in results])
    if reference is not None:
        for result in results:
            result["row"] = _decide(result["record"], result["fields"],
                                    result["case_id"], reference_date=reference)
        with output_path.open("w", encoding="utf-8") as out:
            for result in results:
                out.write(json.dumps(result["row"]) + "\n")
        print(f"batch reference date {reference}: re-decided {len(results)} cases")

    elapsed = time.time() - start
    per_pdf = elapsed / max(len(files), 1)
    print(f"wrote {len(results)} predictions to {output_path} in {elapsed:.0f}s ({per_pdf:.2f}s/pdf)")
    return 0


if __name__ == "__main__":
    _init_worker()
    raise SystemExit(main(sys.argv))

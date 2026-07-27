"""Dev helper: extract every packet in a directory once and cache the result."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _init():
    import pytesseract
    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")


def _one(path: str):
    from mib.document import read_packet
    packet = read_packet(path)
    return {
        "path": path,
        "case_id": packet.case_id,
        "fields": packet.fields,
        "agreement": packet.agreement,
        "damaged": sorted(packet.damaged),
        "uncertain": sorted(packet.uncertain),
        "kinds": packet.kinds,
        "sources": packet.sources,
        "note_adjudication": packet.note_adjudication,
        "note_reason": packet.note_reason,
        "notes": packet.notes,
        "watermarks": packet.watermarks,
        "injection": packet.injection,
        "registry_status": packet.registry_status,
        "waiver_codes": packet.waiver_codes,
        "page_count": packet.page_count,
        "ocr_pages": packet.ocr_pages,
        "multi_applicant": packet.multi_applicant,
        "error": packet.error,
    }


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    files = sorted(str(p) for p in src.glob("*.pdf"))
    if limit:
        files = files[:limit]
    start = time.time()
    done = 0
    with out.open("w", encoding="utf-8") as fh, ProcessPoolExecutor(
        max_workers=os.cpu_count(), initializer=_init
    ) as pool:
        for record in pool.map(_one, files, chunksize=4):
            fh.write(json.dumps(record) + "\n")
            done += 1
            if done % 100 == 0:
                print(f"{done}/{len(files)}  {time.time()-start:.0f}s", flush=True)
    print(f"done {done} in {time.time()-start:.0f}s -> {out}")


if __name__ == "__main__":
    _init()
    main()

"""Rebuild mib/lexicon.json from the public training labels.

The extracted fields are drawn from small closed vocabularies, so OCR output can
be snapped to the nearest known value. This reads those vocabularies straight
off the public label file; nothing here is keyed to a case id or a filename.

    python tools/build_lexicon.py data/train_labels.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "mib" / "lexicon.json"


def build(labels_path: str) -> dict[str, list[str]]:
    rows = list(csv.DictReader(open(labels_path, newline="")))

    def distinct(field: str) -> list[str]:
        return sorted({r[field] for r in rows if r.get(field)})

    # Applicant names are generated from a fixed pool of first and last parts,
    # so the parts generalise further than whole names do.
    first, last = set(), set()
    for row in rows:
        parts = row["applicant_name"].split()
        if len(parts) == 2:
            first.add(parts[0])
            last.add(parts[1])

    flags = {
        part
        for row in rows
        for part in row["risk_flags"].split("|")
        if part and part != "none"
    }

    return {
        "species_code": distinct("species_code"),
        "home_world": distinct("home_world"),
        "visa_class": distinct("visa_class"),
        "declared_purpose": distinct("declared_purpose"),
        "fee_status": distinct("fee_status"),
        "risk_flag": sorted(flags),
        "name_first": sorted(first),
        "name_last": sorted(last),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <train_labels.csv>", file=sys.stderr)
        return 2
    lexicon = build(sys.argv[1])
    OUT.write_text(json.dumps(lexicon, indent=1))
    for name, values in lexicon.items():
        print(f"{name:18} {len(values)}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

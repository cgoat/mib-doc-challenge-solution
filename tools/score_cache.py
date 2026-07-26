"""Score cached extractions with the current rules, mirroring scripts/evaluate.py."""
from __future__ import annotations

import csv, json, sys, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mib.rules import adjudicate, resolve_fields
from mib.confidence import confidence_for

FIELDS = ["applicant_name","species_code","home_world","visa_class","sponsor_id",
          "arrival_date","declared_purpose","risk_flags","fee_status"]
W = {"applicant_name":5,"species_code":6,"home_world":5,"visa_class":5,"sponsor_id":5,
     "arrival_date":4,"declared_purpose":3,"risk_flags":8,"fee_status":4}


def norm(v):
    return " ".join(str(v or "").strip().split()).casefold()


def normflags(v):
    r = norm(v)
    if r in ("", "none", "null", "unknown"):
        return "none"
    return "|".join(sorted(p.strip() for p in r.split("|") if p.strip()))


def main():
    truth = {r["case_id"]: r for r in csv.DictReader(open(sys.argv[1]))}
    recs = [json.loads(l) for l in open(sys.argv[2])]

    ext_raw = ext_max = cls_raw = 0.0
    briers = []
    confusion = collections.Counter()
    reason_wrong = collections.Counter()
    cat = 0
    for rec in recs:
        t = truth.get(rec["case_id"])
        if not t:
            continue
        fields = resolve_fields(rec)
        adj, reasons = adjudicate(rec, fields)
        conf = confidence_for(rec, fields, adj, reasons)

        for f in FIELDS:
            ext_max += W[f]
            p = fields.get(f, "")
            ok = normflags(t[f]) == normflags(p) if f == "risk_flags" else norm(t[f]) == norm(p)
            ext_raw += W[f] if ok else 0

        truth_adj = t["adjudication"]
        confusion[(truth_adj, adj)] += 1
        if truth_adj == adj:
            cls_raw += 8
        elif truth_adj == "DENIED" and adj == "APPROVED":
            cls_raw -= 4; cat += 1
        elif adj == "NEEDS_REVIEW":
            cls_raw += 2
        elif truth_adj == "NEEDS_REVIEW":
            cls_raw += 1
        if truth_adj != adj:
            reason_wrong[(truth_adj, adj, reasons[0] if reasons else "-")] += 1
        briers.append((conf - (1.0 if truth_adj == adj else 0.0)) ** 2)

    n = len(recs)
    extraction = 50 * ext_raw / ext_max
    classification = 80 * cls_raw / (8 * n)
    mb = sum(briers) / len(briers)
    calibration = 20 * max(0.0, 1 - 2 * mb)
    print(f"extraction     {extraction:6.2f} / 50")
    print(f"classification {classification:6.2f} / 80")
    print(f"calibration    {calibration:6.2f} / 20   (mean brier {mb:.4f})")
    print(f"TOTAL          {extraction+classification+calibration:6.2f} / 150")
    print(f"accuracy {sum(v for (a,b),v in confusion.items() if a==b)/n:.1%}   catastrophic false approvals: {cat}")
    print("\nconfusion (truth -> pred):")
    for k, v in sorted(confusion.items()):
        print(f"   {k[0]:12} -> {k[1]:12} {v}")
    print("\ntop wrong-decision drivers:")
    for k, v in reason_wrong.most_common(18):
        print(f"   {v:4d}  {k[0]:12} -> {k[1]:12}  {k[2]}")


if __name__ == "__main__":
    main()

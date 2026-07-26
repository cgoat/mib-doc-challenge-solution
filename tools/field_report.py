import csv, json, sys, collections

FIELDS = ["applicant_name","species_code","home_world","visa_class","sponsor_id",
          "arrival_date","declared_purpose","risk_flags","fee_status"]

def norm(v):
    return " ".join(str(v or "").strip().split()).casefold()

def normflags(v):
    r = norm(v)
    if r in ("", "none", "null", "unknown"): return "none"
    return "|".join(sorted(p.strip() for p in r.split("|") if p.strip()))

truth = {r["case_id"]: r for r in csv.DictReader(open(sys.argv[1]))}
recs = [json.loads(l) for l in open(sys.argv[2])]

hit = collections.Counter(); tot = collections.Counter(); miss = collections.defaultdict(list)
blank = collections.Counter()
scanned_hit = collections.Counter(); scanned_tot = collections.Counter()
for rec in recs:
    t = truth.get(rec["case_id"])
    if not t:
        print("UNMATCHED", rec["path"], rec["case_id"]); continue
    is_scan = rec["ocr_pages"] > 0
    for f in FIELDS:
        p = rec["fields"].get(f, "")
        ok = normflags(t[f]) == normflags(p) if f == "risk_flags" else norm(t[f]) == norm(p)
        tot[f] += 1; hit[f] += ok
        if is_scan: scanned_tot[f] += 1; scanned_hit[f] += ok
        if not p: blank[f] += 1
        if not ok and len(miss[f]) < 6: miss[f].append((rec["case_id"], t[f], p))

print(f"{'field':18} {'all':>7} {'scanned':>9} {'blank':>6}")
for f in FIELDS:
    a = hit[f]/tot[f]; s = scanned_hit[f]/max(scanned_tot[f],1)
    print(f"{f:18} {a:7.1%} {s:9.1%} {blank[f]:6d}")
print(f"{'OVERALL':18} {sum(hit.values())/sum(tot.values()):7.1%}")
print("\nnote adjudication present:", sum(1 for r in recs if r["note_adjudication"]))
print("injection flagged:", sum(1 for r in recs if r["injection"]))
print("cases with no case_id:", sum(1 for r in recs if not r["case_id"]))
if len(sys.argv) > 3:
    for f in FIELDS:
        print(f"\n-- {f}")
        for m in miss[f]: print("   ", m)

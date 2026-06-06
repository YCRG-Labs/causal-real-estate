"""Inventory of EVERY result file under results/ (full landscape, not curated).

Usage:
    python3 data/scripts/replications/inventory_results.py
        -> manifest: every csv/json grouped by directory, with a one-line
           content summary (row count + columns for tables; top-level keys
           for json).

    python3 data/scripts/replications/inventory_results.py <substring>
        -> full dump: print the complete contents of every file whose path
           contains <substring> (e.g. "counterfactual", "moderator", "sim").

Read-only.
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "results"
filt = sys.argv[1].lower() if len(sys.argv) > 1 else None


def summarize_csv(p):
    try:
        with open(p) as fh:
            r = list(csv.reader(fh))
        if not r:
            return "empty"
        cols = r[0]
        shown = ",".join(cols[:8]) + ("..." if len(cols) > 8 else "")
        return f"{len(r) - 1:>4} rows x {len(cols):>2} cols  [{shown}]"
    except Exception as e:
        return f"(unreadable: {e})"


def summarize_json(p):
    try:
        d = json.load(open(p))
        if isinstance(d, dict):
            parts = []
            for k, v in list(d.items())[:8]:
                if isinstance(v, (dict, list)):
                    parts.append(f"{k}({len(v)})")
                else:
                    parts.append(k)
            return "keys: " + ", ".join(parts) + ("..." if len(d) > 8 else "")
        if isinstance(d, list):
            return f"list[{len(d)}]"
        return str(d)[:60]
    except Exception as e:
        return f"(unreadable: {e})"


def dump(p):
    print("\n" + "=" * 78)
    print(p.relative_to(REPO))
    print("=" * 78)
    if p.suffix == ".csv":
        with open(p) as fh:
            for line in fh:
                print("  " + line.rstrip())
    else:
        print(json.dumps(json.load(open(p)), indent=1, default=str))


files = sorted(q for q in RES.rglob("*") if q.suffix in (".csv", ".json"))

if filt:
    hits = [q for q in files if filt in str(q).lower()]
    print(f"FULL DUMP of {len(hits)} file(s) matching '{filt}':")
    for q in hits:
        dump(q)
    sys.exit(0)

total = len(files)
size = sum(q.stat().st_size for q in files)
print("=" * 78)
print(f"  RESULTS INVENTORY  --  {total} csv/json files, {size/1e6:.1f} MB")
print(f"  (drill in with:  python3 <thisfile> <substring>)")
print("=" * 78)

groups = {}
for q in files:
    groups.setdefault(q.parent.relative_to(RES), []).append(q)

for d in sorted(groups, key=str):
    qs = groups[d]
    print(f"\n[{d}/]  ({len(qs)} files)")
    for q in qs:
        s = summarize_csv(q) if q.suffix == ".csv" else summarize_json(q)
        print(f"  {q.name:<42} {s}")

print("\n" + "=" * 78)

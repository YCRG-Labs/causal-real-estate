"""Print every number in every results/*.json, flattened to dotted keys.

Walks a results directory (default: results/), loads each .json, and prints its
scalar leaves as `dotted.key = value`. Large numeric arrays are summarized
(count, min, mean, max) rather than dumped; long string fields are truncated.
This is a read-only inventory so you can eyeball every canonical number on the
Brev box without opening files one at a time.

Usage:
  python3 data/scripts/show_all_jsons.py                 # all of results/
  python3 data/scripts/show_all_jsons.py results/replications
  python3 data/scripts/show_all_jsons.py --grep theta    # only keys matching 'theta'
  python3 data/scripts/show_all_jsons.py --max-array 0    # never summarize arrays, just count
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _fmt_scalar(x):
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:+.6g}"
    if isinstance(x, int):
        return f"{x:+d}" if x else "0"
    s = str(x).replace("\n", " ")
    return s if len(s) <= 80 else s[:77] + "..."


def _summarize_array(arr):
    nums = [v for v in arr if _is_number(v)]
    if nums and len(nums) == len(arr):
        return (f"[{len(arr)} nums  min={min(nums):+.4g}  "
                f"mean={sum(nums)/len(nums):+.4g}  max={max(nums):+.4g}]")
    return f"[{len(arr)} items]"


def flatten(obj, prefix, out, max_array):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out, max_array)
    elif isinstance(obj, list):
        if max_array >= 0 and len(obj) > max_array and not any(
            isinstance(e, (dict, list)) for e in obj
        ):
            out.append((prefix, _summarize_array(obj)))
        elif all(not isinstance(e, (dict, list)) for e in obj) and len(obj) <= max(max_array, 0):
            out.append((prefix, "[" + ", ".join(_fmt_scalar(e) for e in obj) + "]"))
        else:
            for i, e in enumerate(obj):
                flatten(e, f"{prefix}[{i}]", out, max_array)
    else:
        out.append((prefix, _fmt_scalar(obj)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results",
                    help="directory to walk (default: results)")
    ap.add_argument("--grep", default=None,
                    help="only print keys whose dotted path contains this substring")
    ap.add_argument("--max-array", type=int, default=12,
                    help="arrays longer than this are summarized (default 12; 0 = always summarize)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"no such directory: {root}", file=sys.stderr)
        return 1

    files = sorted(root.rglob("*.json"))
    print(f"# {len(files)} json files under {root}\n")
    n_vals = 0
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"== {f} ==\n  [unreadable: {e}]\n")
            continue
        rows = []
        flatten(data, "", rows, args.max_array)
        if args.grep:
            rows = [(k, v) for k, v in rows if args.grep.lower() in k.lower()]
        if not rows:
            continue
        print(f"== {f}  ({len(rows)} values) ==")
        width = min(max((len(k) for k, _ in rows), default=0), 48)
        for k, v in rows:
            print(f"  {k:<{width}}  {v}")
            n_vals += 1
        print()
    print(f"# {n_vals} values printed across {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

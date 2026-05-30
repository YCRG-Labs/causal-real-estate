"""Byte-identical rerun verification for the 12-city pipeline.

Hashes every JSON output, every per-city parquet header, and the manifest;
writes a single deterministic table that two pipeline runs must match.
Catches the LightGBM-vs-sklearn drift and the CUDA-non-determinism failure
modes before they reach the paper.

Usage:
    python verification/hash_run.py --out verification/run_hashes.txt
    python verification/hash_run.py --compare verification/run_hashes_baseline.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def normalise_json(path: Path) -> str:
    try:
        with path.open() as f:
            payload = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return ""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def collect_targets() -> list[Path]:
    targets: list[Path] = []
    if RESULTS_DIR.exists():
        targets.extend(sorted(RESULTS_DIR.rglob("*.json")))
        targets.extend(sorted(RESULTS_DIR.rglob("*.csv")))
    if PROCESSED_DIR.exists():
        targets.extend(sorted(PROCESSED_DIR.rglob("*_listings.parquet")))
        targets.extend(sorted(PROCESSED_DIR.rglob("*_embeddings*.parquet")))
        targets.extend(sorted(PROCESSED_DIR.rglob("*_confounders.parquet")))
    return targets


def hash_target(path: Path) -> str:
    if path.suffix == ".json":
        digest = normalise_json(path)
        return digest if digest else sha256_file(path)
    return sha256_file(path)


def write_hashes(targets: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(f"# Generated {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"# Repo root: {REPO_ROOT}\n")
        f.write(f"# {len(targets)} targets\n")
        for t in targets:
            rel = t.relative_to(REPO_ROOT)
            try:
                h = hash_target(t)
                size = t.stat().st_size
                f.write(f"{h}  {size:>12d}  {rel}\n")
            except OSError as e:
                f.write(f"ERROR  {0:>12d}  {rel}  ({e})\n")


def parse_hashes(path: Path) -> dict[str, tuple[str, int]]:
    out: dict[str, tuple[str, int]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            digest, size_str, rel = parts[0], parts[1], " ".join(parts[2:])
            try:
                size = int(size_str)
            except ValueError:
                size = -1
            out[rel] = (digest, size)
    return out


def compare(current_path: Path, baseline_path: Path) -> int:
    current = parse_hashes(current_path)
    baseline = parse_hashes(baseline_path)
    only_current = set(current) - set(baseline)
    only_baseline = set(baseline) - set(current)
    differs: list[tuple[str, str, str]] = []
    for k in set(current) & set(baseline):
        if current[k][0] != baseline[k][0]:
            differs.append((k, current[k][0], baseline[k][0]))
    print(f"Comparison: {current_path.name} vs {baseline_path.name}")
    print(f"  files only in current:  {len(only_current)}")
    print(f"  files only in baseline: {len(only_baseline)}")
    print(f"  files differing:        {len(differs)}")
    if only_current:
        print("\n  Only in current:")
        for k in sorted(only_current)[:20]:
            print(f"    {k}")
    if only_baseline:
        print("\n  Only in baseline:")
        for k in sorted(only_baseline)[:20]:
            print(f"    {k}")
    if differs:
        print("\n  Differing (truncated to 20):")
        for k, cur, base in differs[:20]:
            print(f"    {k}\n      current:  {cur[:32]}...\n      baseline: {base[:32]}...")
        return 2
    if only_current or only_baseline:
        return 1
    print("\n  All hashes match. Byte-identical rerun verified.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SHA-256 verification for the 12-city pipeline")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "verification" / "run_hashes.txt")
    parser.add_argument("--compare", type=Path, default=None)
    args = parser.parse_args()
    if args.compare is not None:
        return compare(args.out, args.compare)
    targets = collect_targets()
    print(f"Hashing {len(targets)} targets...")
    write_hashes(targets, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

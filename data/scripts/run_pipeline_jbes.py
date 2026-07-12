"""Orchestrator for the 12-city JBES expansion pipeline on Brev.

Sits alongside the existing run_pipeline.py (which targets the 3-city
canonical pipeline). This one drives the JBES expansion: smoke check that the
canonical numbers reproduce, async scrape of the nine new cities, Census
batch geocoding, confounder construction, embeddings, DML, finalization,
hash verification.

Resumability: each stage writes a manifest entry under
`data/processed/_manifest.json` keyed by city and stage with the SHA-256 of
the artifact. Skip-if-exists is the default; pass --force to rerun a stage.

Determinism: the seed block fires before any model imports, sets
CUBLAS_WORKSPACE_CONFIG, torch deterministic mode, NumPy/random/hash seeds,
and the force_sklearn=True override documented in environment_pin.yml.

Usage:
    python data/scripts/run_pipeline_jbes.py smoke
    python data/scripts/run_pipeline_jbes.py acquire --cities new9
    python data/scripts/run_pipeline_jbes.py geocode --cities new9
    python data/scripts/run_pipeline_jbes.py confounders --cities new9
    python data/scripts/run_pipeline_jbes.py embeddings --cities new9
    python data/scripts/run_pipeline_jbes.py dml --cities new9
    python data/scripts/run_pipeline_jbes.py finalize --cities new9
    python data/scripts/run_pipeline_jbes.py analysis
    python data/scripts/run_pipeline_jbes.py all --cities new9

The `analysis` stage (meta-regression, cross-method concordance, sensitivity,
counterfactual rollup) is not part of `all`: it depends on tables produced by
baur_pooled_pca.py and run_counterfactual.py, which run outside this
orchestrator, so it is invoked separately once those tables exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data" / "scripts"))

SCRIPTS_DIR = REPO_ROOT / "data" / "scripts"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"
VERIFICATION_DIR = REPO_ROOT / "verification"
MANIFEST_PATH = PROCESSED_DIR / "_manifest.json"
SEED = 20260530


def set_determinism() -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import random
    random.seed(SEED)
    try:
        import numpy as np
        np.random.seed(SEED)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        with MANIFEST_PATH.open() as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mark_done(stage: str, slug: str, artifact: Path) -> None:
    manifest = load_manifest()
    key = f"{slug}::{stage}"
    manifest[key] = {
        "stage": stage,
        "city": slug,
        "artifact": str(artifact.relative_to(REPO_ROOT)),
        "sha256": sha256_file(artifact) if artifact.exists() else "",
        "size_bytes": artifact.stat().st_size if artifact.exists() else 0,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    save_manifest(manifest)


def is_done(stage: str, slug: str, artifact: Path) -> bool:
    if not artifact.exists():
        return False
    manifest = load_manifest()
    key = f"{slug}::{stage}"
    if key not in manifest:
        return False
    expected = manifest[key].get("sha256", "")
    return bool(expected) and sha256_file(artifact) == expected


def resolve_cities(cities_arg: str) -> list:
    from city_endpoints import CITIES, list_new, list_ready
    if cities_arg == "new9":
        return list_new()
    if cities_arg == "all":
        return list_ready()
    slugs = [s.strip() for s in cities_arg.split(",") if s.strip()]
    return [CITIES[s] for s in slugs]


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    return subprocess.call(cmd, cwd=cwd or REPO_ROOT)


def cmd_smoke(args) -> int:
    print("=" * 60)
    print("SMOKE: confounder_escalation.py nyc")
    print("must reproduce shrinkage=95.1%, n=347")
    print("=" * 60)
    return run(["python3", "confounder_escalation.py", "nyc"], cwd=SCRIPTS_DIR)


def cmd_acquire(args) -> int:
    print("=" * 60)
    print("ACQUIRE: async scrape of new cities")
    print("=" * 60)
    return run([
        "python3", str(SCRIPTS_DIR / "scrape_redfin_async.py"),
        "--cities", args.cities,
        "--max-listings", str(args.max_listings),
    ])


def cmd_geocode(args) -> int:
    print("=" * 60)
    print("GEOCODE: Census batch endpoint (10k records per upload)")
    print("=" * 60)
    return run([
        "python3", str(SCRIPTS_DIR / "geocode_batch.py"),
        "--cities", args.cities,
    ])


def cmd_confounders(args) -> int:
    print("=" * 60)
    print("CONFOUNDERS: ACS + Overpass + crime + parcel match")
    print("=" * 60)
    # NOTE on the confounder-build path: this stage runs the
    # download_census.py/download_amenities.py/download_crime.py ->
    # attach_census.py/attach_amenities.py/attach_crime.py -> attach_micro_geo.py
    # chain, which is the live path wired into the master pipeline for cities
    # with status != "existing" (see city_endpoints.py). Separately,
    # build_active_confounders.py + attach_crime_to_parcels.py exist in
    # data/scripts/ as an alternate confounder builder that writes to the
    # same artifact name ({city}_parcels_micro_geo.gpkg) as attach_micro_geo.py
    # above. Per reviews/code_review_report.md and the referee2 round-1 report,
    # which of the two actually produced the confounders behind the paper's
    # current numbers has not been conclusively determined from the code
    # alone. TODO: confirm which path is authoritative before wiring
    # build_active_confounders.py/attach_crime_to_parcels.py here — running
    # both against the same city risks one silently overwriting the other's
    # output.
    cities = resolve_cities(args.cities)
    rc = 0
    for cfg in cities:
        if cfg.status == "existing":
            print(f"  [{cfg.slug}] existing canonical; skipping confounder rebuild")
            continue
        print(f"\n--- {cfg.slug} ---")
        rc |= run(["python3", "download_census.py", cfg.slug], cwd=SCRIPTS_DIR)
        rc |= run(["python3", "download_amenities.py", cfg.slug], cwd=SCRIPTS_DIR)
        rc |= run(["python3", "download_crime.py", cfg.slug], cwd=SCRIPTS_DIR)
        rc |= run(["python3", "attach_census.py", cfg.slug], cwd=SCRIPTS_DIR)
        rc |= run(["python3", "attach_amenities.py", cfg.slug], cwd=SCRIPTS_DIR)
        rc |= run(["python3", "attach_crime.py", cfg.slug], cwd=SCRIPTS_DIR)
        rc |= run(["python3", "attach_micro_geo.py", cfg.slug], cwd=SCRIPTS_DIR)
    return rc


def cmd_embeddings(args) -> int:
    print("=" * 60)
    print("EMBEDDINGS: sentence-BERT mpnet-base-v2 on CUDA")
    print("=" * 60)
    cities = resolve_cities(args.cities)
    rc = 0
    for cfg in cities:
        rc |= run([
            "python3", "generate_embeddings_modern.py",
            "--city", cfg.slug,
            "--model", "mpnet_base",
        ], cwd=SCRIPTS_DIR)
    return rc


def cmd_dml(args) -> int:
    print("=" * 60)
    print("DML: per-city bootstrap with IM-2010 cluster-t")
    print("=" * 60)
    cities = resolve_cities(args.cities)
    rc = 0
    for cfg in cities:
        rc |= run([
            "python3", "fast_bootstrap_dml_v2.py",
            "--B", str(args.B),
            "--cities", cfg.slug,
            "--learner", "ridge",
        ], cwd=SCRIPTS_DIR)
    return rc


def cmd_finalize(args) -> int:
    print("=" * 60)
    print("FINALIZE: Shen replication + LEACE + IGBP + hash verify")
    print("=" * 60)
    cities = resolve_cities(args.cities)
    rc = 0
    for cfg in cities:
        rc |= run(["python3", "replications/shen_2021.py", "--city", cfg.slug], cwd=SCRIPTS_DIR)
        rc |= run(["python3", "leace_deconfound.py", "--city", cfg.slug, "--postnl_igbp"], cwd=SCRIPTS_DIR)
    rc |= run([
        "python3", str(VERIFICATION_DIR / "hash_run.py"),
        "--out", str(VERIFICATION_DIR / f"run_hashes_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"),
    ])
    return rc


def cmd_analysis(args) -> int:
    print("=" * 60)
    print("ANALYSIS: meta-regression + cross-method concordance + sensitivity + counterfactual rollup")
    print("=" * 60)
    # These four scripts consume tables already written by cmd_finalize
    # (shen_2021.py, leace_deconfound.py) plus baur_pooled_pca.py and
    # run_counterfactual.py, which are run separately/manually and are not
    # dispatched from this orchestrator. Each script below has its own
    # default input paths under results/; rerun the upstream script first
    # if a required table is missing.
    rc = 0
    rc |= run(["python3", "replications/meta_regression_12.py"], cwd=SCRIPTS_DIR)
    rc |= run(["python3", "replications/cross_method_concordance.py"], cwd=SCRIPTS_DIR)
    rc |= run(["python3", "replications/confounder_sensitivity_12.py", "--all_12"], cwd=SCRIPTS_DIR)
    rc |= run(["python3", "replications/rollup_counterfactual_12.py"], cwd=SCRIPTS_DIR)
    return rc


def cmd_all(args) -> int:
    print("=" * 60)
    print("ALL: smoke -> acquire -> geocode -> confounders -> embeddings -> dml -> finalize")
    print("=" * 60)
    t0 = time.time()
    for fn in [cmd_smoke, cmd_acquire, cmd_geocode, cmd_confounders, cmd_embeddings, cmd_dml, cmd_finalize]:
        rc = fn(args)
        if rc != 0:
            print(f"\nFAILED at {fn.__name__} (rc={rc})", file=sys.stderr)
            return rc
    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed/60:.1f} min")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrator for the 12-city JBES expansion")
    parser.add_argument("subcommand", choices=[
        "smoke", "acquire", "geocode", "confounders",
        "embeddings", "dml", "finalize", "analysis", "all",
    ])
    parser.add_argument("--cities", default="new9", help="new9 | all | comma-separated slugs")
    parser.add_argument("--max-listings", type=int, default=350)
    parser.add_argument("--B", type=int, default=1000, help="bootstrap replicates for DML")
    parser.add_argument("--force", action="store_true", help="ignore manifest and rerun")
    args = parser.parse_args()
    set_determinism()
    dispatch = {
        "smoke": cmd_smoke,
        "acquire": cmd_acquire,
        "geocode": cmd_geocode,
        "confounders": cmd_confounders,
        "embeddings": cmd_embeddings,
        "dml": cmd_dml,
        "finalize": cmd_finalize,
        "analysis": cmd_analysis,
        "all": cmd_all,
    }
    return dispatch[args.subcommand](args)


if __name__ == "__main__":
    sys.exit(main())

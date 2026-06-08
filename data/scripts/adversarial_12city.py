"""GRL adversarial deconfounding + frozen-encoder probe across the 12-city panel.

Foil experiment for the LEACE deconfounding story: run gradient-reversal
adversarial erasure of location on each market's listing-text embedding, then
hit the frozen encoder with a fresh post-hoc probe. The point is to show that
"live discriminator at chance" (the adversarial training appears to succeed)
does NOT imply location is gone -- the frozen probe recovers it -- which is the
methodological pitfall that motivates the closed-form LEACE guardedness used in
the headline analysis.

Reuses the exact 12-city loader the LEACE pipeline uses
(load_analysis_data -> get_features_and_target) and the adversarial pipeline the
original 3-city analysis used (adversarial_deconfounding, which calls the frozen
probe internally). GPU-bound: run on Brev.

  python3 data/scripts/adversarial_12city.py                # all 12, epochs=150
  python3 data/scripts/adversarial_12city.py sf boston      # subset
  ADV_EPOCHS=20 python3 data/scripts/adversarial_12city.py sf   # quick smoke test
"""
import os
import sys
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "data" / "scripts"))

from causal_inference import (  # noqa: E402
    load_analysis_data,
    get_features_and_target,
    adversarial_deconfounding,
)

CITIES_12 = [
    "boston", "nyc", "sf", "dc", "philadelphia", "chicago",
    "seattle", "denver", "atlanta", "portland", "phoenix", "dallas",
]

OUT_DIR = REPO / "results" / "adversarial_12city"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EPOCHS = int(os.environ.get("ADV_EPOCHS", "150"))

SUMMARY_COLS = [
    "city", "n", "pred_r2",
    "zip_acc", "zip_random", "zip_probe_acc",
    "geo_r2", "geo_probe_r2",
    "inc_acc", "income_random", "income_probe_acc",
    "live_random", "probe_random",
]


def _ratio(num, den):
    if num is None or den in (None, 0):
        return float("nan")
    return float(num) / float(den)


def run_city(city):
    loaded = load_analysis_data(city)
    if loaded is None:
        print(f"[{city}] no analysis data; skipping")
        return None
    emb_df, parcels = loaded
    data = get_features_and_target(emb_df, parcels)
    if data is None:
        print(f"[{city}] no features/target; skipping")
        return None
    T, _conf, Y, meta = data
    T = np.asarray(T, dtype=np.float32)
    Y = np.asarray(Y).ravel()
    pred_r2, out = adversarial_deconfounding(T, Y, meta, epochs=EPOCHS)
    out["city"] = city
    out["n"] = int(len(Y))
    out["pred_r2"] = float(pred_r2)
    (OUT_DIR / f"{city}_adversarial.json").write_text(
        json.dumps(out, indent=2, default=float)
    )
    zr = _ratio(out.get("zip_probe_acc"), out.get("zip_random"))
    fooled = bool(out.get("live_random")) and not bool(out.get("probe_random"))
    print(f"[{city}] n={out['n']:>6}  live zip_acc={out.get('zip_acc', float('nan')):.3f}"
          f"  frozen zip_probe={out.get('zip_probe_acc', float('nan')):.3f}"
          f" ({zr:.1f}x)  geo_probe_R2={out.get('geo_probe_r2', float('nan')):.3f}"
          f"  {'FOOLED (live~chance, probe recovers)' if fooled else ''}")
    return out


def main():
    cities = sys.argv[1:] or CITIES_12
    rows = []
    for city in cities:
        try:
            r = run_city(city)
        except Exception as exc:
            print(f"[{city}] FAILED: {type(exc).__name__}: {exc}")
            r = None
        if r is not None:
            rows.append({k: r.get(k) for k in SUMMARY_COLS})
    if rows:
        with open(OUT_DIR / "adversarial_12city_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
            w.writeheader()
            w.writerows(rows)
        n_fooled = sum(1 for r in rows if r["live_random"] and not r["probe_random"])
        print(f"\n{len(rows)} markets done; {n_fooled} show the live-random / "
              f"frozen-probe-recovers pattern.")
        print(f"Summary -> {OUT_DIR / 'adversarial_12city_summary.csv'}")


if __name__ == "__main__":
    main()

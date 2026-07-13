"""Twin-matched within-building FE-DML: the most stringent rung of the ladder.

The within-building design (within_building_dml.py) differences out location but not
unit quality. This restricts the comparison to near-twin units, matching within
building on bedrooms, bathrooms, and a square-footage bucket, so the residual text
variation is between nearly identical units and coarse quality is held fixed by
matching rather than assumed. Fixed effect = (building x beds x baths x sqft-bucket);
building-key = lat/lon rounded to ~11 m; SE clustered at the matched group.
"""
from __future__ import annotations
import glob, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np, pandas as pd
from numpy.linalg import svd
sys.path.insert(0, "data/scripts")
from replications.within_building_dml import fe_dml, winsor, STRUCT

OUT = "results/within_building"


def main():
    os.makedirs(OUT, exist_ok=True)
    f = sorted([x for x in glob.glob("data/processed/*_embeddings.parquet") if "all_MiniLM" not in x])
    emb = [c for c in pd.read_parquet(f[0]).columns if c.startswith("emb_")]
    frames = {}
    for x in f:
        df = pd.read_parquet(x)
        df = df[(df.price > 1e4) & (df.price < 1e8)].dropna(subset=["latitude", "longitude"]).copy()
        df["bldg"] = df.latitude.round(4).astype(str) + "_" + df.longitude.round(4).astype(str)
        df["sqftb"] = (pd.to_numeric(df["sqft"], errors="coerce") // 100).fillna(-1)
        df["twin"] = df.bldg + "|" + df.beds.astype(str) + "|" + df.baths.astype(str) + "|" + df.sqftb.astype(str)
        frames[os.path.basename(x).split("_")[0]] = df.reset_index(drop=True)

    blocks = [fr[emb].to_numpy(np.float64) - fr[emb].to_numpy(np.float64).mean(0) for fr in frames.values()]
    Ec = np.vstack(blocks); _, _, Vt = svd(Ec - Ec.mean(0), full_matrices=False); v = Vt[0]

    rows = []
    for mkt, fr in frames.items():
        tw = fr.groupby("twin").filter(lambda g: len(g) >= 2)
        if len(tw) < 150:
            print(f"  {mkt}: too few twin rows ({len(tw)})", flush=True); continue
        E = tw[emb].to_numpy(np.float64); E = E - E.mean(0); T = E @ v; T = (T - T.mean()) / (T.std() or 1)
        Y = np.log(tw.price.to_numpy(float)); X = winsor(tw[STRUCT].to_numpy(float))
        r = fe_dml(Y, T, X, tw["twin"].to_numpy())
        if r is None:
            print(f"  {mkt}: failed", flush=True); continue
        th, si, sc, ne, nb = r
        rows.append({"market": mkt, "n_twin": len(tw), "n_groups": nb,
                     "theta_fe": th, "se_clustered": sc, "t_clustered": th / sc,
                     "sig": bool(abs(th / sc) > 1.96)})
        print(f"  {mkt:12} n={len(tw):5} groups={nb:4} theta={th:+.4f} t={th/sc:+.1f} "
              f"{'SIG' if rows[-1]['sig'] else 'ns'}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/twin_matched_table.csv", index=False)
    th = df.theta_fe.to_numpy(float); se = df.se_clustered.to_numpy(float); w = 1 / se**2
    pth = float((th * w).sum() / w.sum()); pse = float(np.sqrt(1 / w.sum()))
    summ = {"pooled_theta": pth, "pooled_se": pse, "pooled_t": pth / pse,
            "sig_markets": int(df.sig.sum()), "n_markets": len(df), "all_positive": bool((df.theta_fe > 0).all())}
    (Path(OUT) / "twin_matched_summary.json").write_text(json.dumps(summ, indent=2))
    print(f"\nPOOLED twin-matched: theta={pth:+.4f} se={pse:.4f} (t={pth/pse:+.1f})  "
          f"sig {int(df.sig.sum())}/{len(df)}")
    print(f"wrote {OUT}/twin_matched_table.csv")


if __name__ == "__main__":
    main()

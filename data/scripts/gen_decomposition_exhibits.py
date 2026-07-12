"""Emit the LaTeX for the spatial-confounding decomposition exhibits from the
current decomposition CSVs, so the appendix table, the TPRS table, and the TPRS
figure cannot drift from the numbers the pipeline actually produced.

Run: .venv/bin/python3 data/scripts/gen_decomposition_exhibits.py
Sources: results/replications/leace_price_decomposition_quad.csv (+ _tprs30.csv)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "replications"
DISPLAY = {
    "boston": "Boston", "sf": "San Francisco", "nyc": "New York",
    "chicago": "Chicago", "philadelphia": "Philadelphia", "atlanta": "Atlanta",
    "dallas": "Dallas", "denver": "Denver", "phoenix": "Phoenix",
    "dc": "Washington", "portland": "Portland", "seattle": "Seattle",
}


def s3(x):
    return f"${x:+.3f}$"


def main():
    q = pd.read_csv(RES / "leace_price_decomposition_quad.csv").set_index("city")
    t = pd.read_csv(RES / "leace_price_decomposition_tprs30.csv").set_index("city")
    order = q.sort_values("naive_theta", ascending=False).index.tolist()

    print("% ===== tab:decomposition-12 body =====")
    for c in order:
        r = q.loc[c]
        d = r.naive_theta - r.geo_theta
        print(f"{DISPLAY[c]:<13} & {s3(r.naive_theta)} & {s3(r.geo_theta)} & "
              f"{s3(d)} & ${r.pc1_geo_r2:.3f}$ \\\\")

    print("\n% ===== tab:decomposition-tprs body =====")
    for c in order:
        rq, rt = q.loc[c], t.loc[c]
        d = rt.geo_theta - rq.geo_theta
        print(f"{DISPLAY[c]:<13} & {s3(rq.geo_theta)} & {s3(rt.geo_theta)} & "
              f"{s3(d)} & ${rq.pc1_geo_r2:.3f}$ & ${rt.pc1_geo_r2:.3f}$ \\\\")

    print("\n% ===== fig:decomposition-tprs (order + coords) =====")
    forder = q.sort_values("geo_theta", ascending=True).index.tolist()
    ycoords = ",".join(DISPLAY[c] for c in forder)
    print(f"symbolic y coords={{{ycoords}}},")
    print("% connector lines:")
    for c in forder:
        print(f"\\draw[thin,gray] (axis cs:{q.loc[c].geo_theta:.3f},{DISPLAY[c]})"
              f"--(axis cs:{t.loc[c].geo_theta:.3f},{DISPLAY[c]});")
    print("% quad markers:")
    print("  " + " ".join(f"({q.loc[c].geo_theta:.3f},{DISPLAY[c]})" for c in forder))
    print("% tprs markers:")
    print("  " + " ".join(f"({t.loc[c].geo_theta:.3f},{DISPLAY[c]})" for c in forder))

    # summary numbers for the prose
    dq = (q.naive_theta - q.geo_theta).abs()
    nyc_d = q.loc["nyc"].naive_theta - q.loc["nyc"].geo_theta
    ex_nyc = dq.drop("nyc")
    print("\n% ===== prose numbers =====")
    print(f"% max |Delta| overall = {dq.max():.3f} ({dq.idxmax()})")
    print(f"% NYC Delta = {nyc_d:+.3f}")
    print(f"% max |Delta| excluding NYC = {ex_nyc.max():.3f} ({ex_nyc.idxmax()})")
    dse = ((q.naive_theta - q.geo_theta).abs() / q.geo_se)
    print(f"% |Delta|/geo_se: NYC={dse['nyc']:.2f}; max ex-NYC={dse.drop('nyc').max():.2f} ({dse.drop('nyc').idxmax()})")
    print(f"% TPRS: NYC R2 quad {q.loc['nyc'].pc1_geo_r2:.2f} -> tprs {t.loc['nyc'].pc1_geo_r2:.2f}")
    dt = (t.geo_theta - q.geo_theta).abs()
    print(f"% TPRS max |Delta| = {dt.max():.3f} ({dt.idxmax()}); max ex-NYC = {dt.drop('nyc').max():.3f} ({dt.drop('nyc').idxmax()})")
    print(f"% mean R2 quad {q.pc1_geo_r2.mean():.3f} -> tprs {t.pc1_geo_r2.mean():.3f}")


if __name__ == "__main__":
    main()

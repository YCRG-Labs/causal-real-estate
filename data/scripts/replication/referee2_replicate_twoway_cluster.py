"""Referee 2 clean-room replication of the metro x quarter two-way cluster SE
for the pooled sale-price DML (results/soldprice/soldprice_twoway_cluster.json).

Per the audit instructions, the DML psi_i score itself is reused from the
author's data-loading + cross-fitted-residual code (load_pooled,
_oof_residuals in data/scripts/soldprice/soldprice_twoway_cluster.py) -- that
part is the same 192k-row, 10-metro panel construction as the headline
result, not what is being audited here. What IS independently re-implemented
from scratch, without looking at (or calling) the author's `_cluster_V` /
`two_way_se` functions, is the CGM (2011) two-way cluster-robust variance
arithmetic itself:

    V(g)      = sum over clusters g of (sum_{i in g} psi_i)^2
    scale(G)  = G/(G-1) * (N-1)/(N-K) / N^2
    V_metro   = scale(G_metro)   * V(metro)
    V_quarter = scale(G_quarter) * V(quarter)
    V_cell    = scale(G_cell)    * V(metro x quarter cell)
    V_twoway  = V_metro + V_quarter - V_cell   (fallback to max(V_metro,V_quarter) if <=0)
    dof       = min(G_metro, G_quarter) - 1

Implemented here via pandas groupby (independent code path), and checked
against results/soldprice/soldprice_twoway_cluster.json (committed).

Run:
  .venv/bin/python data/scripts/replication/referee2_replicate_twoway_cluster.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "data" / "scripts" / "soldprice"))

# Reused: only the data assembly + cross-fitted nuisance residualization
# (panel construction, RidgeCV OOF residuals). This produces psi_i, theta,
# metro_i, quarter_i exactly as the headline run does -- it is not the
# object under audit in this half of the task.
from soldprice_twoway_cluster import load_pooled, _oof_residuals  # noqa: E402

AUTHOR_JSON = REPO / "results" / "soldprice" / "soldprice_twoway_cluster.json"


def independent_cluster_variance(psi: np.ndarray, group_labels: np.ndarray) -> tuple[float, int]:
    """sum_g (sum_{i in g} psi_i)^2 via a clean groupby -- no reuse of author code."""
    df = pd.DataFrame({"psi": psi, "grp": group_labels})
    cluster_sums = df.groupby("grp")["psi"].sum()
    V = float((cluster_sums ** 2).sum())
    G = int(cluster_sums.shape[0])
    return V, G


def independent_two_way_se(psi: np.ndarray, metro: np.ndarray, quarter: np.ndarray, K: int):
    N = len(psi)

    V_m_raw, G_m = independent_cluster_variance(psi, metro)
    V_q_raw, G_q = independent_cluster_variance(psi, quarter)
    cell = pd.Series(metro).astype(str) + "||" + pd.Series(quarter).astype(str)
    V_c_raw, G_c = independent_cluster_variance(psi, cell.to_numpy())

    def scale(G):
        return (G / (G - 1)) * ((N - 1) / (N - K)) / (N ** 2)

    V_metro = scale(G_m) * V_m_raw
    V_quarter = scale(G_q) * V_q_raw
    V_cell = scale(G_c) * V_c_raw

    V_twoway = V_metro + V_quarter - V_cell
    neg_var_triggered = V_twoway <= 0
    if neg_var_triggered:
        V_twoway = max(V_metro, V_quarter)

    return {
        "N": N, "K": K,
        "G_metro": G_m, "G_quarter": G_q, "G_cell": G_c,
        "V_metro_raw": V_m_raw, "V_quarter_raw": V_q_raw, "V_cell_raw": V_c_raw,
        "se_metro": float(np.sqrt(V_metro)),
        "se_quarter": float(np.sqrt(V_quarter)),
        "se_twoway": float(np.sqrt(V_twoway)),
        "dof": min(G_m, G_q) - 1,
        "neg_var_fallback": neg_var_triggered,
        "V_twoway_before_fallback": V_metro + V_quarter - V_cell,
    }


def main():
    print("loading pooled sale-price panel + recomputing DML psi "
          "(reused loader/nuisance code; not the object under audit)...", flush=True)
    pool = load_pooled()
    metro = pool["_metro"].to_numpy()
    quarter = pool["_quarter"].to_numpy()
    Y = pool["_Y"].to_numpy(float)
    T = pool["_T"].to_numpy(float)
    Xbase = pool.drop(columns=["_Y", "_T", "_metro", "_quarter"]).to_numpy(float)

    # winsorize + median-impute, same preprocessing the headline script does
    # to the base features before adding FE dummies (independent re-typing,
    # not a functional import)
    lo = np.nanquantile(Xbase, 0.005, axis=0)
    hi = np.nanquantile(Xbase, 0.995, axis=0)
    Xbase = np.clip(Xbase, lo, hi)
    med = np.nanmedian(Xbase, axis=0)
    ix = np.where(~np.isfinite(Xbase))
    Xbase[ix] = np.take(np.nan_to_num(med), ix[1])

    metro_fe = pd.get_dummies(pool["_metro"], drop_first=True).to_numpy(float)
    quarter_fe = pd.get_dummies(pool["_quarter"], drop_first=True).to_numpy(float)
    X = np.column_stack([Xbase, metro_fe, quarter_fe])
    n = len(Y)
    K = X.shape[1] + 1
    print(f"pooled n={n:,}  metros={pool['_metro'].nunique()}  "
          f"quarters={pool['_quarter'].nunique()}  X dim={X.shape[1]}  K={K}", flush=True)

    Yr, Tr = _oof_residuals(T, X, Y, seed=0)
    denom = float(np.mean(Tr * Tr))
    theta = float(np.mean(Tr * Yr)) / denom
    psi = (Yr - theta * Tr) * Tr / denom
    se_if = float(np.sqrt(np.var(psi, ddof=1) / n))

    result = independent_two_way_se(psi, metro, quarter, K=K)

    author = json.loads(AUTHOR_JSON.read_text())

    print(f"\nreferee2 theta       = {theta:+.6f}   author theta       = {author['theta']:+.6f}")
    print(f"referee2 se_if       = {se_if:.6f}      author se_if       = {author['se_if']:.6f}")
    print(f"referee2 se_metro    = {result['se_metro']:.6f}      author se_metro    = {author['se_metro']:.6f}")
    print(f"referee2 se_quarter  = {result['se_quarter']:.6f}      author se_quarter  = {author['se_quarter']:.6f}")
    print(f"referee2 se_twoway   = {result['se_twoway']:.6f}      author se_twoway   = {author['se_twoway']:.6f}")
    print(f"referee2 G_metro/G_quarter/G_cell = {result['G_metro']}/{result['G_quarter']}/{result['G_cell']}"
          f"   author = {author['G_metro']}/{author['G_quarter']}/{author['G_cell']}")
    print(f"referee2 dof = {result['dof']}   author dof = {author['dof']}")
    print(f"referee2 neg_var_fallback = {result['neg_var_fallback']}"
          f"   author neg_var_fallback = {author['neg_var_fallback']}")
    print(f"referee2 V_twoway before fallback check = {result['V_twoway_before_fallback']:.6e}")

    # sanity check: does the 1/N^2 normalization make sense for an
    # IF-based (already-averaged) estimator?
    # psi as defined here already has the 1/n implicit in theta's estimating
    # equation form: theta solves mean(psi)=0, and Var(theta_hat) ~= Var(mean(psi))
    # = Var(sum psi)/N^2 under independence. Under cluster dependence the
    # sandwich generalizes sum(psi_i^2) -> sum_g (sum_{i in g} psi_i)^2, and
    # the whole thing still gets divided by N^2 (not N, not 1) because we are
    # estimating Var(theta_hat) = Var((1/N) sum psi_i), and (1/N)^2 factors
    # out of the sum-of-squares. So the N^2 division is the correct analogue
    # of the usual robust-variance N^2 in Var(mean) = Var(sum)/N^2; it is not
    # an ad hoc choice.
    check_1 = abs(theta - author["theta"]) < 1e-3
    check_2 = abs(result["se_metro"] - author["se_metro"]) < 5e-4
    check_3 = abs(result["se_twoway"] - author["se_twoway"]) < 5e-4
    check_4 = result["dof"] == author["dof"]
    check_5 = result["neg_var_fallback"] == author["neg_var_fallback"]

    table = pd.DataFrame([
        {"quantity": "theta", "author": author["theta"], "referee2": theta,
         "match": check_1},
        {"quantity": "se_if", "author": author["se_if"], "referee2": se_if,
         "match": abs(se_if - author["se_if"]) < 5e-5},
        {"quantity": "se_metro", "author": author["se_metro"], "referee2": result["se_metro"],
         "match": check_2},
        {"quantity": "se_quarter", "author": author["se_quarter"], "referee2": result["se_quarter"],
         "match": abs(result["se_quarter"] - author["se_quarter"]) < 5e-4},
        {"quantity": "se_twoway", "author": author["se_twoway"], "referee2": result["se_twoway"],
         "match": check_3},
        {"quantity": "dof", "author": author["dof"], "referee2": result["dof"], "match": check_4},
        {"quantity": "neg_var_fallback", "author": author["neg_var_fallback"],
         "referee2": result["neg_var_fallback"], "match": check_5},
        {"quantity": "G_metro", "author": author["G_metro"], "referee2": result["G_metro"],
         "match": author["G_metro"] == result["G_metro"]},
        {"quantity": "G_quarter", "author": author["G_quarter"], "referee2": result["G_quarter"],
         "match": author["G_quarter"] == result["G_quarter"]},
    ])
    out_csv = REPO / "data" / "scripts" / "replication" / "referee2_twoway_cluster_comparison.csv"
    table.to_csv(out_csv, index=False)
    print("\n=== Comparison table ===")
    print(table.to_string(index=False))
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()

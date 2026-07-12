"""Pooled cross-metro sale-price DML with a metro x sale-quarter two-way
cluster-robust standard error (Cameron, Gelbach and Miller 2011, JBES 29(2)).

The per-market sale-price panel of soldprice_dml.py reports tract cluster-bootstrap,
Conley HAC and Ibragimov-Muller intervals. This script adds a pooled estimate whose
standard error accounts for dependence in both the metro and the calendar-quarter
dimension at once. The DML score is the partialling-out influence function
psi_i = (Yr_i - theta Tr_i) Tr_i / mean(Tr^2); the two-way cluster-robust variance is

    V(two-way) = V(metro) + V(quarter) - V(metro x quarter cell),

each term a sum over clusters of the squared within-cluster sum of psi, with the
Stata small-sample scaling c_G = G/(G-1) * (N-1)/(N-K) applied per component and the
negative-variance fallback V := max(V_metro, V_quarter) if the combination is
negative. Reference distribution is t with min(G_metro, G_quarter) - 1 degrees of
freedom.

Honest scope: with only ten metros this is a robustness column, not primary
inference. CGM's two-way estimator is justified as min(G, H) grows, which the metro
dimension (G=10) does not satisfy on its own, so the metro-clustered leg is read
under the Cameron-Miller (2015) few-clusters caveat. This is a post-estimation
clustering adjustment on standard single-way cross-fitted DML scores, weaker than
the fully fold-consistent multiway estimator of Chiang, Kato, Ma and Sasaki (2022),
which the asking-price panel found unstable at this cluster count.

    python data/scripts/soldprice/soldprice_twoway_cluster.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
PANEL = REPO / "results" / "soldprice"
EMB = PANEL / "emb"
CONF = PANEL / "confounders"
OUT = PANEL / "soldprice_twoway_cluster.json"

CITIES = ["boston", "sf", "dc", "philadelphia", "chicago", "seattle",
          "denver", "atlanta", "portland", "phoenix"]
DEMO = ["median_household_income", "median_home_value", "median_gross_rent",
        "pct_white", "pct_black", "pct_asian", "pct_hispanic", "pct_bachelors",
        "labor_force_participation", "pct_under_25", "pct_over_60"]
ALPHAS = np.logspace(-3, 3, 25)
LIST_PROP = ["list_beds", "list_baths", "list_sqft", "list_year",
             "list_beds_missing", "list_sqft_missing", "list_year_missing"]


def _winsor(X, p=0.005):
    lo = np.nanquantile(X, p, axis=0)
    hi = np.nanquantile(X, 1 - p, axis=0)
    return np.clip(X, lo, hi)


def load_pooled():
    parts = []
    for c in CITIES:
        panel = pd.read_parquet(PANEL / f"{c}_panel.parquet",
                                columns=["log_price", "is_single_unit", "sold_quarter"])
        T = pd.read_parquet(EMB / f"{c}_treatment.parquet")["treatment"].to_numpy(float)
        L = pd.read_parquet(EMB / f"{c}_emb.parquet", columns=["log_len"])["log_len"].to_numpy(float)
        conf = pd.read_parquet(CONF / f"{c}_conf.parquet")
        n = len(panel)
        if not (len(T) == len(L) == len(conf) == n):
            raise SystemExit(f"{c} misaligned")
        lat, lon = conf["lat"].to_numpy(float), conf["lon"].to_numpy(float)
        spatial = np.column_stack([lat, lon, lat ** 2, lon ** 2, lat * lon])
        demo = np.column_stack([conf[[f"demo_{d}" for d in DEMO]].to_numpy(float),
                                conf["demo_missing"].to_numpy(float)])
        prop = conf[LIST_PROP].to_numpy(float)
        base = np.column_stack([spatial, demo, prop, L, L ** 2])
        Y = panel["log_price"].to_numpy(float)
        fin = np.isfinite(Y) & panel["is_single_unit"].to_numpy(bool)
        df = pd.DataFrame(base[fin])
        df["_Y"] = Y[fin]
        df["_T"] = T[fin]
        df["_metro"] = c
        df["_quarter"] = panel["sold_quarter"].to_numpy()[fin]
        parts.append(df)
    pool = pd.concat(parts, ignore_index=True)
    return pool


def _oof_residuals(T, X, Y, seed=0, k=5):
    n = len(Y)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    Yr, Tr = np.empty(n), np.empty(n)
    for tr, te in kf.split(np.arange(n)):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        Yr[te] = Y[te] - RidgeCV(alphas=ALPHAS).fit(Xtr, Y[tr]).predict(Xte)
        Tr[te] = T[te] - RidgeCV(alphas=ALPHAS).fit(Xtr, T[tr]).predict(Xte)
    return Yr, Tr


def _cluster_V(psi, labels):
    s = pd.Series(psi).groupby(np.asarray(labels)).sum().to_numpy()
    return float(np.sum(s ** 2)), int(len(s))


def two_way_se(psi, metro, quarter, K):
    N = len(psi)
    Vm, Gm = _cluster_V(psi, metro)
    Vq, Gq = _cluster_V(psi, quarter)
    cell = np.char.add(np.char.add(np.asarray(metro, str), "|"), np.asarray(quarter, str))
    Vc, Gc = _cluster_V(psi, cell)

    def cf(G):
        return (G / (G - 1)) * ((N - 1) / (N - K)) / (N ** 2)

    V2 = cf(Gm) * Vm + cf(Gq) * Vq - cf(Gc) * Vc
    fallback = False
    if V2 <= 0:
        V2 = max(cf(Gm) * Vm, cf(Gq) * Vq)
        fallback = True
    return {
        "se_metro": float(np.sqrt(cf(Gm) * Vm)),
        "se_quarter": float(np.sqrt(cf(Gq) * Vq)),
        "se_twoway": float(np.sqrt(V2)),
        "G_metro": Gm, "G_quarter": Gq, "G_cell": Gc,
        "dof": min(Gm, Gq) - 1, "neg_var_fallback": fallback,
    }


def main():
    from scipy import stats
    print("loading pooled sale-price panel...", flush=True)
    pool = load_pooled()
    metro = pool["_metro"].to_numpy()
    quarter = pool["_quarter"].to_numpy()
    Y = pool["_Y"].to_numpy(float)
    T = pool["_T"].to_numpy(float)
    Xbase = pool.drop(columns=["_Y", "_T", "_metro", "_quarter"]).to_numpy(float)
    Xbase = _winsor(Xbase)
    med = np.nanmedian(Xbase, axis=0)
    ix = np.where(~np.isfinite(Xbase))
    Xbase[ix] = np.take(np.nan_to_num(med), ix[1])
    metro_fe = pd.get_dummies(pool["_metro"], drop_first=True).to_numpy(float)
    quarter_fe = pd.get_dummies(pool["_quarter"], drop_first=True).to_numpy(float)
    X = np.column_stack([Xbase, metro_fe, quarter_fe])
    n = len(Y)
    print(f"pooled n={n:,}  metros={pool['_metro'].nunique()}  "
          f"quarters={pool['_quarter'].nunique()}  X dim={X.shape[1]}", flush=True)

    Yr, Tr = _oof_residuals(T, X, Y, seed=0)
    denom = float(np.mean(Tr * Tr))
    theta = float(np.mean(Tr * Yr)) / denom
    psi = (Yr - theta * Tr) * Tr / denom
    se_if = float(np.sqrt(np.var(psi, ddof=1) / n))

    tw = two_way_se(psi, metro, quarter, K=X.shape[1] + 1)
    t_crit = float(stats.t.ppf(0.975, df=tw["dof"]))
    ci = [theta - t_crit * tw["se_twoway"], theta + t_crit * tw["se_twoway"]]

    out = {"n": n, "theta": theta, "se_if": se_if, **tw,
           "t_crit": t_crit, "ci_twoway": ci,
           "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
    OUT.write_text(json.dumps(out, indent=2))

    print(f"\npooled DML theta = {theta:+.4f}")
    print(f"  IF SE (no clustering)      {se_if:.4f}")
    print(f"  metro-clustered SE  (G={tw['G_metro']})   {tw['se_metro']:.4f}")
    print(f"  quarter-clustered SE (G={tw['G_quarter']})  {tw['se_quarter']:.4f}")
    print(f"  metro x quarter two-way SE  {tw['se_twoway']:.4f}"
          f"{'  [neg-var fallback]' if tw['neg_var_fallback'] else ''}")
    print(f"  95% CI t_{{{tw['dof']}}}={t_crit:.3f}: [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
          f"{'EXCLUDES 0' if out['excludes_zero'] else 'covers 0'}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

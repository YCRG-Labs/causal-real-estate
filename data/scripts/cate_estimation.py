"""Conditional Average Treatment Effect (CATE) estimation for the SF DML.

Continuous-treatment partially-linear DML on PC1 of text embedding.
Stratifies SF by:
  (1) predicted-price quartile from a confounders-only baseline model
      (pre-treatment proxy; avoids the post-treatment-stratification bias
      of using Y itself);
  (2) description-length quartile (pre-treatment, intrinsic to T).

Reports stratum-specific θ̂ with influence-function-based standard errors,
plus a Best-Linear-Predictor (BLP) heterogeneity test on the quartile
indicator basis (Chernozhukov, Demirer, Duflo, Fernandez-Val 2018).

Implementation reuses the same residualized-DML pipeline as
causal_inference.dml_continuous_treatment to keep the headline number
identical and the stratum estimates apples-to-apples.

Refs:
  Kennedy 2023 EJS — DR-Learner; arXiv:2004.14497
  Kennedy, Ma, McHugh, Small 2017 JRSS-B — continuous-T DR
  Semenova & Chernozhukov 2021 EJ — debiased ML for CATE
  Chernozhukov, Demirer, Duflo, Fernandez-Val 2018 NBER WP 24678 — BLP/GATES

Usage:
  python cate_estimation.py --city sf --n_seeds 50 --out results/cate/sf.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor  # noqa: F401
from booster import make_regressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_inference import get_features_and_target, load_analysis_data
from config import CITIES


@dataclass
class CellResult:
    stratifier: str
    quartile: int
    n: int
    theta: float
    se: float
    ci_low: float
    ci_high: float
    contains_zero: bool


def _gbm(seed: int, n_jobs: int = -1):
    return make_regressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=seed,
        n_jobs=n_jobs,
    )


def cross_fit_residuals(
    pc1: np.ndarray, W: np.ndarray, Y: np.ndarray, seed: int = 42, k_folds: int = 5,
    inner_n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-fit GBM nuisances; return (Y_resid, T_resid, Y_oof_pred).

    Y_oof_pred is the out-of-fold predicted Y from confounders alone — used
    as the pre-treatment proxy for price segment when stratifying.

    Pass inner_n_jobs=1 when calling from within joblib.Parallel to avoid
    nested over-subscription (e.g. stability_check's 50-seed sweep).
    """
    n = len(Y)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    Y_resid = np.zeros(n)
    T_resid = np.zeros(n)
    Y_oof_pred = np.zeros(n)
    for tr, te in kf.split(np.arange(n)):
        m_y = _gbm(seed, n_jobs=inner_n_jobs)
        m_y.fit(W[tr], Y[tr])
        Y_oof_pred[te] = m_y.predict(W[te])
        Y_resid[te] = Y[te] - Y_oof_pred[te]
        m_t = _gbm(seed, n_jobs=inner_n_jobs)
        m_t.fit(W[tr], pc1[tr])
        T_resid[te] = pc1[te] - m_t.predict(W[te])
    return Y_resid, T_resid, Y_oof_pred


def dml_theta_and_psi(Y_resid: np.ndarray, T_resid: np.ndarray) -> tuple[float, np.ndarray]:
    """Partialling-out estimator + per-row Neyman-orthogonal score."""
    denom = float(np.mean(T_resid ** 2))
    if denom < 1e-12:
        return float("nan"), np.zeros_like(Y_resid)
    theta = float(np.mean(T_resid * Y_resid)) / denom
    psi = (Y_resid - theta * T_resid) * T_resid / denom
    return theta, psi


def stratum_ate(psi: np.ndarray, mask: np.ndarray, theta: float) -> tuple[float, float]:
    """GATES-style per-stratum CATE estimator (Chernozhukov-Demirer-Duflo-
    Fernandez-Val 2018). The IF score psi has E[psi] = 0 by the FOC of theta;
    the per-stratum CATE level is theta + E[psi | i in stratum]. The variance
    is Var(psi | i in stratum) / n_q (same as before — shift is constant).
    """
    n_q = int(mask.sum())
    if n_q < 5:
        return float("nan"), float("nan")
    s = psi[mask]
    return float(theta + s.mean()), float(s.std(ddof=1) / np.sqrt(n_q))


def quartile_cells(
    psi: np.ndarray, stratifier: np.ndarray, label: str, theta: float,
) -> list[CellResult]:
    valid = ~np.isnan(stratifier)
    if valid.sum() < 100:
        return []
    quartiles = pd.qcut(stratifier[valid], q=4, labels=False, duplicates="drop")
    full_q = np.full(len(stratifier), -1, dtype=int)
    full_q[valid] = quartiles

    cells: list[CellResult] = []
    for q in sorted(set(full_q[full_q >= 0])):
        mask = full_q == q
        ate, se = stratum_ate(psi, mask, theta)
        if not np.isfinite(ate):
            continue
        ci_lo = ate - 1.96 * se
        ci_hi = ate + 1.96 * se
        cells.append(CellResult(
            stratifier=label,
            quartile=int(q),
            n=int(mask.sum()),
            theta=float(ate),
            se=float(se),
            ci_low=float(ci_lo),
            ci_high=float(ci_hi),
            contains_zero=bool(ci_lo <= 0 <= ci_hi),
        ))
    return cells


def blp_heterogeneity_test(
    psi: np.ndarray, stratifier: np.ndarray, theta: float,
) -> dict:
    """Joint Wald test of equal stratum CATE levels (CDDF 2018 BLP/GATES).

    H0: theta_q = theta for all q (no heterogeneity across quartiles).
    Statistic: Σ (theta_q − theta)² / Var(theta_q), distributed χ²_(K-1).
    Because theta_q = theta + E[psi | q] and Var(theta_q) = Var(psi | q) / n_q,
    the statistic reduces to Σ (E[psi | q])² / Var(theta_q) (theta cancels).
    """
    valid = ~np.isnan(stratifier)
    psi_v = psi[valid]
    quartiles = pd.qcut(stratifier[valid], q=4, labels=False, duplicates="drop")
    cells = sorted(np.unique(quartiles))
    if len(cells) < 2:
        return {"error": "fewer than 2 quartiles"}
    cell_thetas = {
        int(c): float(theta + psi_v[quartiles == c].mean()) for c in cells
    }
    vars_ = {
        int(c): float(psi_v[quartiles == c].var(ddof=1) / max((quartiles == c).sum(), 1))
        for c in cells
    }
    wald = sum(
        (cell_thetas[c] - theta) ** 2 / vars_[c]
        for c in cells if vars_[c] > 0
    )
    df = len(cells) - 1
    return {
        "stat_wald": float(wald),
        "df": int(df),
        "p_value": float(1 - chi2.cdf(wald, df=df)),
        "cell_thetas": cell_thetas,
    }


def _stability_one_seed(s: int, pc1: np.ndarray, W: np.ndarray, Y: np.ndarray) -> float:
    # inner_n_jobs=1: outer joblib already parallelizes over seeds.
    Y_r, T_r, _ = cross_fit_residuals(pc1, W, Y, seed=s, inner_n_jobs=1)
    theta, _ = dml_theta_and_psi(Y_r, T_r)
    return float(theta)


def stability_check(
    pc1: np.ndarray, W: np.ndarray, Y: np.ndarray, n_seeds: int = 50,
    n_jobs: int = -1,
) -> dict:
    """Run n_seeds independent DML refits in parallel. Each seed gives its own
    KFold split; the spread across seeds is the headline stability metric."""
    from joblib import Parallel, delayed
    thetas = Parallel(n_jobs=n_jobs)(
        delayed(_stability_one_seed)(s, pc1, W, Y) for s in range(n_seeds)
    )
    arr = np.array(thetas, dtype=float)
    return {
        "n_seeds": int(n_seeds),
        "median_theta": float(np.median(arr)),
        "mean_theta": float(arr.mean()),
        "sd_theta": float(arr.std(ddof=1)),
        "min_theta": float(arr.min()),
        "max_theta": float(arr.max()),
    }


def run_cate(city: str, n_seeds: int = 50) -> dict:
    print(f"\n=== CATE estimation: {city} ===")
    loaded = load_analysis_data(city)
    if loaded is None:
        return {"city": city, "error": "no data"}
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    if feats is None:
        return {"city": city, "error": "no features"}
    T, W, Y, meta = feats
    print(f"  N={len(Y):,}, embedding dim={T.shape[1]}, confounders={W.shape[1]}")

    if not meta["has_rich_confounders"]:
        return {"city": city, "error": "rich confounders required"}

    # Project T → PC1 (z-scored); standardize confounders
    pca = PCA(n_components=min(50, T.shape[1], T.shape[0] - 1), random_state=42)
    T_pca = pca.fit_transform(T)
    pc1 = T_pca[:, 0]
    pc1 = (pc1 - pc1.mean()) / (pc1.std() if pc1.std() > 0 else 1.0)
    Ws = StandardScaler().fit_transform(W)

    # Cross-fit residuals once at seed=42 (for headline + stratification)
    Y_resid, T_resid, Y_oof_pred = cross_fit_residuals(pc1, Ws, Y, seed=42)
    theta, psi = dml_theta_and_psi(Y_resid, T_resid)
    se = float(np.sqrt(np.var(psi, ddof=1) / len(Y)))
    print(f"  overall θ = {theta:+.4f}, SE = {se:.4f}, "
          f"95% CI = [{theta-1.96*se:+.4f}, {theta+1.96*se:+.4f}]")

    # Stratifier 1: predicted-price quartile (pre-treatment proxy)
    print("\n  CATE by predicted-price quartile (OOF μ̂(W)):")
    cells_pp = quartile_cells(psi, Y_oof_pred, "predicted_price_q", theta)
    for c in cells_pp:
        flag = "contains 0" if c.contains_zero else "EXCLUDES 0"
        print(f"    Q{c.quartile+1}: n={c.n:>4}  θ={c.theta:+.4f}  "
              f"se={c.se:.4f}  CI=[{c.ci_low:+.4f}, {c.ci_high:+.4f}]  {flag}")
    blp_pp = blp_heterogeneity_test(psi, Y_oof_pred, theta)
    if "p_value" in blp_pp:
        print(f"    BLP heterogeneity Wald = {blp_pp['stat_wald']:.2f} "
              f"(df={blp_pp['df']}), p = {blp_pp['p_value']:.3f}")

    # Stratifier 2: description-length quartile
    desc_len = None
    if "description" in emb_df.columns:
        d = emb_df["description"].fillna("").astype(str).str.len().values
        if len(d) >= len(Y):
            desc_len = d[: len(Y)].astype(float)
    if desc_len is None:
        # fallback: embedding L2 norm (correlates with description complexity)
        desc_len = np.linalg.norm(T, axis=1).astype(float)

    print("\n  CATE by description-length quartile (pre-treatment proxy):")
    cells_dl = quartile_cells(psi, desc_len, "desc_length_q", theta)
    for c in cells_dl:
        flag = "contains 0" if c.contains_zero else "EXCLUDES 0"
        print(f"    Q{c.quartile+1}: n={c.n:>4}  θ={c.theta:+.4f}  "
              f"se={c.se:.4f}  CI=[{c.ci_low:+.4f}, {c.ci_high:+.4f}]  {flag}")
    blp_dl = blp_heterogeneity_test(psi, desc_len, theta)
    if "p_value" in blp_dl:
        print(f"    BLP heterogeneity Wald = {blp_dl['stat_wald']:.2f} "
              f"(df={blp_dl['df']}), p = {blp_dl['p_value']:.3f}")

    print(f"\n  Stability across {n_seeds} seeds:")
    stab = stability_check(pc1, Ws, Y, n_seeds=n_seeds)
    print(f"    overall θ: median = {stab['median_theta']:+.4f}, "
          f"sd = {stab['sd_theta']:.4f}, "
          f"range = [{stab['min_theta']:+.4f}, {stab['max_theta']:+.4f}]")

    return {
        "city": city,
        "n": int(len(Y)),
        "n_confounders": int(W.shape[1]),
        "overall_theta": float(theta),
        "overall_se": float(se),
        "cate_predicted_price_quartile": [asdict(c) for c in cells_pp],
        "blp_predicted_price": blp_pp,
        "cate_description_length_quartile": [asdict(c) for c in cells_dl],
        "blp_description_length": blp_dl,
        "stability": stab,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=CITIES, required=True)
    ap.add_argument("--n_seeds", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = run_cate(args.city, n_seeds=args.n_seeds)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

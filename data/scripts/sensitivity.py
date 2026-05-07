"""Sensitivity bounds & E-values for the SF DML estimate.

Four complementary diagnostics, all targeting "how strong would unmeasured
confounding need to be to overturn our null?":

  (1) OVB-DML Robustness Value (RV) and RVa — Chernozhukov, Cinelli, Newey,
      Sharma, Syrgkanis 2022, arXiv:2112.13398. Generalizes the Cinelli-
      Hazlett (2020) OLS RV to ML-fitted causal estimands. RV is the partial
      R² of an unobserved confounder (with both treatment and outcome) that
      would shrink |θ̂| to zero; RVa is the value that pushes the 95% CI to
      include zero.

  (2) E-values — VanderWeele & Ding 2017, AnnIntMed 167(4):268-274.
      Convert the DML coefficient (per-σ of PC1) to an approximate risk
      ratio via the Chinn / VanderWeele transformation, then report the
      minimum joint association strength of an unobserved confounder
      with both T and Y to fully explain the effect. Easy to communicate;
      cross-check on the RV.

  (3) Bayesian sensitivity analysis — McCandless & Gustafson 2017, SiM
      36(18):2887-2901. Place Beta(2,8) priors on (η_Y, η_D) (each
      partial-R² ~ 0.2 in expectation), compute the posterior probability
      that the bias-adjusted |θ| exceeds 0.05.

  (4) Manski-style plausibility bounds — Manski 1990 AER P&P 80(2):319-323,
      operationalized via percentile bounds of the BSA-adjusted θ
      distribution. Reports the implied bound on θ under bounded-strength
      unmeasured confounding.

Refs (full dossier in research/sensitivity/research_notes.md).

Usage:
  python sensitivity.py --city sf
  python sensitivity.py --city sf --n_mc 50000 --out results/sensitivity/sf.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cate_estimation import cross_fit_residuals, dml_theta_and_psi
from causal_inference import get_features_and_target, load_analysis_data
from config import CITIES


def ovb_dml_scaling(
    Y_resid: np.ndarray, T_resid: np.ndarray, theta: float
) -> float:
    """The S-statistic from CCNSS 2022, Theorem 2.4 / Cor 2.5.

    For a partially-linear model the bias of the long-vs-short estimator is
    bounded by |bias| <= S * cf(eta_Y, eta_D) where
      S^2 = Var(Y - theta*T - g(X)) / Var(T - m(X))
            = Var(long-regression residual) / Var(treatment residual)
      cf(eta_Y, eta_D) = eta_Y * eta_D / sqrt(1 - eta_D^2)

    Using cross-fitted residuals: Y_resid = Y - g_hat(X); T_resid = T - m_hat(X);
    the long-regression residual at the fitted theta is (Y_resid - theta*T_resid).
    """
    long_var = float(np.var(Y_resid - theta * T_resid, ddof=1))
    t_var = float(np.var(T_resid, ddof=1))
    if t_var <= 0:
        return float("inf")
    return float(np.sqrt(long_var / t_var))


def confounding_factor(eta_y: np.ndarray, eta_d: np.ndarray) -> np.ndarray:
    """cf(eta_Y, eta_D) = eta_Y * eta_D / sqrt(1 - eta_D^2).

    Defined for eta_d < 1; returns inf at the boundary.
    """
    denom = np.sqrt(np.clip(1.0 - eta_d ** 2, 1e-12, None))
    return eta_y * eta_d / denom


def robustness_value_point(theta: float, S: float) -> float:
    """RV: solves |theta| = S * RV^2 / sqrt(1 - RV^2) for RV in [0, 1).

    Setting eta_Y = eta_D = RV (the diagonal at which the confounder must lie
    for the bias to exactly equal |theta|). Closed form via quadratic on RV^2.
    """
    if S <= 0 or not np.isfinite(theta):
        return float("nan")
    a = S ** 2
    b = theta ** 2
    if b == 0:
        return 0.0
    # S^2 * x^2 + b * x - b = 0  where x = RV^2
    disc = b ** 2 + 4 * a * b
    rv2 = (-b + np.sqrt(disc)) / (2 * a)
    rv2 = max(0.0, min(rv2, 1.0))
    return float(np.sqrt(rv2))


def robustness_value_alpha(theta: float, se: float, S: float, alpha: float = 0.05) -> float:
    """RVa: same as RV but uses |theta| - z_{alpha/2} * SE in place of |theta|.

    The partial R² needed to make the (1-alpha) CI include zero rather than
    just the point estimate. RVa <= RV.
    """
    z = float(stats.norm.ppf(1 - alpha / 2))
    margin = max(0.0, abs(theta) - z * se)
    return robustness_value_point(margin, S)


def evalue_point(theta: float, sd_Y: float) -> float:
    """E-value via Chinn (2000) / VanderWeele (2017) for continuous T.

    Approximates RR by exp(0.91 * theta / sd_Y) and applies the standard
    E-value formula. theta is per-sigma of T (here PC1 z-scored).
    """
    rr = float(np.exp(0.91 * abs(theta) / sd_Y))
    if rr < 1:
        rr = 1 / rr  # standard inversion for protective effects
    return float(rr + np.sqrt(rr * (rr - 1)))


def evalue_ci(theta: float, se: float, sd_Y: float, alpha: float = 0.05) -> float:
    """E-value of the CI bound nearer the null (returns 1.0 if CI crosses null)."""
    z = float(stats.norm.ppf(1 - alpha / 2))
    bound_lo = theta - z * se
    bound_hi = theta + z * se
    if bound_lo <= 0 <= bound_hi:
        return 1.0
    bound = bound_lo if theta > 0 else bound_hi
    rr = float(np.exp(0.91 * abs(bound) / sd_Y))
    if rr < 1:
        rr = 1 / rr
    return float(rr + np.sqrt(rr * (rr - 1)))


def bayesian_sensitivity(
    theta: float, S: float,
    threshold: float = 0.05, n_mc: int = 50000,
    a: float = 2.0, b: float = 8.0, seed: int = 42,
) -> dict:
    """Posterior P(|theta_adj| > threshold | (eta_Y, eta_D) ~ Beta(a, b) iid).

    Bias is signed conservatively against the focal estimate (i.e., the
    adjusted theta moves toward zero by the bias magnitude). The default
    Beta(2, 8) has mean 0.2 and 95% mass below ~0.5, encoding "unmeasured
    confounder is probably moderate to small."
    """
    rng = np.random.default_rng(seed)
    eta_y = rng.beta(a, b, size=n_mc)
    eta_d = rng.beta(a, b, size=n_mc)
    bias = S * confounding_factor(eta_y, eta_d)
    sign = 1.0 if theta >= 0 else -1.0
    theta_adj = theta - sign * bias
    return {
        "prior_mean": a / (a + b),
        "prior_a": a,
        "prior_b": b,
        "n_mc": int(n_mc),
        "p_exceeds_threshold": float(np.mean(np.abs(theta_adj) > threshold)),
        "theta_adj_quantiles": {
            "q025": float(np.quantile(theta_adj, 0.025)),
            "q250": float(np.quantile(theta_adj, 0.25)),
            "q500": float(np.quantile(theta_adj, 0.5)),
            "q750": float(np.quantile(theta_adj, 0.75)),
            "q975": float(np.quantile(theta_adj, 0.975)),
        },
        "manski_plausibility_95": [
            float(np.quantile(theta_adj, 0.025)),
            float(np.quantile(theta_adj, 0.975)),
        ],
        "manski_plausibility_50": [
            float(np.quantile(theta_adj, 0.25)),
            float(np.quantile(theta_adj, 0.75)),
        ],
        "threshold": float(threshold),
    }


def benchmark_against_observed(
    Y_resid: np.ndarray, T_resid: np.ndarray, W: np.ndarray, k_top: int = 5
) -> list[dict]:
    """For each observed confounder, report partial R² with Y_resid and T_resid.

    The strongest observed covariate sets the natural plausibility yardstick
    for what an unmeasured confounder might look like.
    """
    out = []
    for j in range(W.shape[1]):
        wj = W[:, j]
        wj = wj - wj.mean()
        denom = float(np.sum(wj ** 2))
        if denom <= 0:
            continue
        # R² = 1 - SSR/SST for the simple OLS of resid on wj
        beta_y = float(np.sum(wj * Y_resid) / denom)
        ssr_y = float(np.sum((Y_resid - beta_y * wj) ** 2))
        sst_y = float(np.sum(Y_resid ** 2))
        r2_y = max(0.0, 1.0 - ssr_y / max(sst_y, 1e-12))
        beta_t = float(np.sum(wj * T_resid) / denom)
        ssr_t = float(np.sum((T_resid - beta_t * wj) ** 2))
        sst_t = float(np.sum(T_resid ** 2))
        r2_t = max(0.0, 1.0 - ssr_t / max(sst_t, 1e-12))
        out.append({"covariate_index": int(j), "partial_r2_Y": r2_y, "partial_r2_T": r2_t})
    out.sort(key=lambda d: -(d["partial_r2_Y"] * d["partial_r2_T"]))
    return out[:k_top]


def run_sensitivity(city: str, n_mc: int = 50000, threshold: float = 0.05) -> dict:
    print(f"\n=== Sensitivity analysis: {city} ===")
    loaded = load_analysis_data(city)
    if loaded is None:
        return {"city": city, "error": "no data"}
    emb_df, parcels = loaded
    feats = get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
    if feats is None:
        return {"city": city, "error": "no features"}
    T, W, Y, meta = feats
    if not meta["has_rich_confounders"]:
        return {"city": city, "error": "rich confounders required"}
    print(f"  N={len(Y):,}, embedding dim={T.shape[1]}, confounders={W.shape[1]}")

    # Reproduce the headline DML estimate
    pca = PCA(n_components=min(50, T.shape[1], T.shape[0] - 1), random_state=42)
    T_pca = pca.fit_transform(T)
    pc1 = T_pca[:, 0]
    pc1 = (pc1 - pc1.mean()) / (pc1.std() if pc1.std() > 0 else 1.0)
    Ws = StandardScaler().fit_transform(W)
    Y_resid, T_resid, _ = cross_fit_residuals(pc1, Ws, Y, seed=42)
    theta, psi = dml_theta_and_psi(Y_resid, T_resid)
    se = float(np.sqrt(np.var(psi, ddof=1) / len(Y)))
    sd_Y = float(np.std(Y, ddof=1))
    print(f"  θ̂ = {theta:+.4f}, SE = {se:.4f}, sd(Y) = {sd_Y:.4f}")
    print(f"  95% CI = [{theta-1.96*se:+.4f}, {theta+1.96*se:+.4f}]")

    # OVB-DML scaling
    S = ovb_dml_scaling(Y_resid, T_resid, theta)
    print(f"\n  [1] OVB-DML scaling (CCNSS 2022): S = {S:.4f}")
    rv = robustness_value_point(theta, S)
    rva = robustness_value_alpha(theta, se, S, alpha=0.05)
    print(f"      Robustness Value RV (point):  {rv:.3f}  "
          f"({'ATTAINABLE' if rv < 0.5 else 'large'})")
    print(f"      Robustness Value RVa (CI95):  {rva:.3f}")
    print(f"      Interpretation: an unmeasured confounder with partial R² >= {rv:.2f}"
          f" with BOTH treatment and outcome would be needed to drive |θ̂| to 0.")

    # E-values
    ev_pt = evalue_point(theta, sd_Y)
    ev_ci = evalue_ci(theta, se, sd_Y, alpha=0.05)
    print(f"\n  [2] E-values (continuous T via Chinn-VanderWeele):")
    print(f"      E-value (point estimate): {ev_pt:.3f}")
    print(f"      E-value (95% CI bound):   {ev_ci:.3f}")
    if ev_pt <= 1.05:
        print(f"      → effect size is essentially null on the RR scale; E-value collapses to 1")

    # Benchmark against observed covariates
    print(f"\n  [3] Plausibility benchmarks (top observed covariates by joint partial R²):")
    bench = benchmark_against_observed(Y_resid, T_resid, Ws, k_top=5)
    for b in bench:
        print(f"      W[{b['covariate_index']:>2}]: η²_Y = {b['partial_r2_Y']:.4f}, "
              f"η²_T = {b['partial_r2_T']:.4f}")
    if bench:
        max_obs = max(b["partial_r2_Y"] for b in bench)
        max_obt = max(b["partial_r2_T"] for b in bench)
        print(f"      Max observed η²_Y = {max_obs:.3f}, max η²_T = {max_obt:.3f}")
        print(f"      For RV={rv:.2f} to be attainable, an unmeasured confounder would need to "
              f"be {'comparable to' if rv < max_obs else 'STRONGER than'} the strongest observed "
              f"covariate on both sides.")

    # Bayesian sensitivity
    bsa = bayesian_sensitivity(theta, S, threshold=threshold, n_mc=n_mc)
    print(f"\n  [4] Bayesian sensitivity (Beta({bsa['prior_a']},{bsa['prior_b']}) priors, "
          f"prior mean η = {bsa['prior_mean']:.2f}):")
    print(f"      P(|θ_adj| > {threshold}): {bsa['p_exceeds_threshold']:.4f}")
    q = bsa["theta_adj_quantiles"]
    print(f"      θ_adj posterior quantiles: "
          f"2.5% = {q['q025']:+.4f}, median = {q['q500']:+.4f}, 97.5% = {q['q975']:+.4f}")
    print(f"      Manski plausibility-bounded 95% interval: "
          f"[{bsa['manski_plausibility_95'][0]:+.4f}, "
          f"{bsa['manski_plausibility_95'][1]:+.4f}]")

    return {
        "city": city,
        "n": int(len(Y)),
        "n_confounders": int(W.shape[1]),
        "theta_hat": float(theta),
        "se": float(se),
        "sd_Y": float(sd_Y),
        "ci_95": [float(theta - 1.96 * se), float(theta + 1.96 * se)],
        "ovb_dml": {
            "S": float(S),
            "RV_point": float(rv),
            "RVa_95ci": float(rva),
        },
        "evalue_point": float(ev_pt),
        "evalue_ci": float(ev_ci),
        "benchmarks": bench,
        "bayesian": bsa,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=CITIES, required=True)
    ap.add_argument("--n_mc", type=int, default=50000)
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="practical-significance threshold for BSA")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = run_sensitivity(args.city, n_mc=args.n_mc, threshold=args.threshold)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()

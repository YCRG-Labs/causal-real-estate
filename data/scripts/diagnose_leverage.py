"""
Leverage / influential-observation diagnostics for the PLR DML estimator,
plus a Huber-loss "trimmed-DML" refit and a Dirichlet-bootstrap sensitivity
curve.  The whole diagnostic is computed on the residuals already produced
by ``fast_bootstrap_dml._dml_core_fast`` (line 164), so this script imports
the core estimator and reuses its cross-fit, never re-implementing the DML
loop.

Per-observation influence-function value for the partialling-out score
(Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins 2018,
eq. (4.3); DoubleML "PLR-IRM partialling-out" docs):

    psi_i(theta) = (Y_tilde_i - theta * T_tilde_i) * T_tilde_i,
    IC_i         = psi_i(theta_hat) / E_n[T_tilde^2].

The asymptotic standard error reduces to

    se_IF = sqrt( Var_i[ IC_i ] / n ).

Auxiliary leverage statistics:

    h_i              = T_tilde_i^2 / sum_j T_tilde_j^2  (pure-treatment leverage)
    IC_share_scaled_i = IC_i^2 / mean_j(IC_j^2)         (scaled influence share;
        NOT Cook's distance D_i = e_i^2 h_i / (p MSE (1-h_i)^2) per Cook 1977)

Sensitivity curve theta_{-k}: refit the partialling-out estimator after
dropping the k rows with largest |IC_i|, for k in {0,1,2,5,10,20}.  This
curve is the standard "robustness of theta to the top-k leveraged rows"
diagnostic (Coppejans (Saco) 2026 "Ill-Conditioned Orthogonal Scores in
Double Machine Learning", arXiv:2512.07083, Section 4): when the score
denominator E[T_tilde^2] is small a small handful of rows drives theta
through the influence channel.

Trimmed-DML refit (this script's robust-headline number).  After
cross-fitting the partialling-out residuals once, we re-estimate theta by
*Huber-robust* regression of Y_tilde on T_tilde with no intercept and
default tuning constant c = 1.345 (95% Gaussian efficiency; Huber 1964;
Maronna, Martin, Yohai 2006, p. 105).  The Huber estimator solves

    theta_HUB = argmin_theta sum_i rho_c( (Y_tilde_i - theta T_tilde_i) / sigma_hat ),
    rho_c(u)  = 0.5 u^2                          if |u| <= c
                c |u| - 0.5 c^2                  if |u|  > c,

so the score is bounded and the top-|IC| rows lose their influence at
exactly the rate one would expect from a 1.345-sigma Winsorisation.  An
M-estimator standard error is reported from the asymptotic formula

    Var(theta_HUB) = sigma^2_hat * E[psi'(u)^2] / ( E[psi'(u)] )^2 / sum T_tilde^2,

with psi = rho' and sigma_hat the median-absolute-deviation scale on the
OLS residuals.

CLI
---
    python diagnose_leverage.py nyc
    python diagnose_leverage.py sf nyc boston
    python diagnose_leverage.py nyc --plot
    python diagnose_leverage.py --self_test

References
----------
Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, Robins (2018)
  "Double/debiased machine learning for treatment and structural
  parameters." The Econometrics Journal 21(1):C1-C68.  arXiv:1608.00060.
Saco (2026, formerly Coppejans) "Ill-Conditioned Orthogonal Scores in
  Double Machine Learning." arXiv:2512.07083.
Huber (1964) "Robust estimation of a location parameter." Ann. Math.
  Stat. 35:73-101.  (Tuning constant c=1.345 for 95% efficiency.)
Maronna, Martin & Yohai (2006) "Robust Statistics: Theory and Methods,"
  Wiley, Section 4.4.
Rubin (1981) "The Bayesian bootstrap." Annals of Statistics 9(1):130-134.
  (Dirichlet weights for the bootstrap sensitivity bands.)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import _silence
except Exception:
    pass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_inference import load_analysis_data, get_features_and_target
from fast_bootstrap_dml_v2 import _make_lgbm, _pc1_randomized, _dml_core_fast



def _residualize(T, conf_s, Y, pc1, k_folds: int, seed: int) -> tuple:
    n = len(Y)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    Y_resid = np.zeros(n)
    T_resid = np.zeros(n)
    for tr, te in kf.split(np.arange(n)):
        m_y = _make_lgbm(); m_t = _make_lgbm()
        m_y.fit(conf_s[tr], Y[tr])
        m_t.fit(conf_s[tr], pc1[tr])
        Y_resid[te] = Y[te] - m_y.predict(conf_s[te])
        T_resid[te] = pc1[te] - m_t.predict(conf_s[te])
    return Y_resid, T_resid



def compute_influence_and_leverage(T_resid: np.ndarray, Y_resid: np.ndarray,
                                   top_k: int = 10) -> dict:
    """Compute per-observation IC, h, and scaled IC-share from cross-fit
    residuals.

    Returns a dict with theta_hat, se_IF, denom_T_resid_sq, IC, h,
    IC_share_scaled, top_idx, summary DataFrame.  The IC_share_scaled
    column was previously labeled `cook` / `cook_std`, which was a
    misnomer; the textbook Cook's distance is D_i = e_i^2 h_i /
    (p MSE (1 - h_i)^2) (Cook 1977 Technometrics 19:15), a different
    quantity.  We retain `cook`/`cook_std` keys as deprecated aliases so
    callers in existing notebooks continue to work.
    """
    T = np.asarray(T_resid, dtype=np.float64)
    Y = np.asarray(Y_resid, dtype=np.float64)
    n = T.size
    if n == 0 or Y.size != n:
        raise ValueError("T_resid and Y_resid must be 1-D arrays of equal length.")
    denom = float(np.mean(T ** 2))
    if denom < 1e-12:
        raise RuntimeError(
            "E[T_resid^2] ~ 0; estimator ill-conditioned "
            "(Saco 2026, kappa_DML diverges)."
        )
    theta = float(np.mean(T * Y)) / denom
    psi = (Y - theta * T) * T
    IC = psi / denom
    se_if = float(np.sqrt(np.var(IC, ddof=1) / n))

    sumT2 = float((T ** 2).sum())
    h = (T ** 2) / sumT2 if sumT2 > 0 else np.full(n, 1.0 / n)
    meanIC2 = float((IC ** 2).mean())
    ic_share_scaled = (IC ** 2) / meanIC2 if meanIC2 > 0 else np.zeros(n)

    order = np.argsort(-np.abs(IC))
    top_idx = order[: min(top_k, n)]
    summary = pd.DataFrame({
        "row": top_idx,
        "Y_resid": Y[top_idx],
        "T_resid": T[top_idx],
        "IC": IC[top_idx],
        "h_leverage": h[top_idx],
        "IC_share_scaled": ic_share_scaled[top_idx],
        "IC_share_of_var": (IC[top_idx] ** 2) / (IC ** 2).sum(),
    })

    return {
        "theta_hat": theta, "se_IF": se_if, "denom_T_resid_sq": denom,
        "IC": IC, "h": h,
        "IC_share_scaled": ic_share_scaled,
        "cook": ic_share_scaled,
        "top_idx": top_idx, "summary": summary, "n": n,
    }



def theta_minus_top_k(T: np.ndarray, conf: np.ndarray, Y: np.ndarray,
                      k_list=(1, 2, 5, 10, 20), k_folds: int = 5,
                      seed: int = 0) -> pd.DataFrame:
    """Refit DML PLR after dropping the top-k highest |IC| rows.

    Cross-fits the residuals once on the full sample, ranks by |IC|, then
    for each k re-PC1s on the kept subset, re-residualises on the kept
    confounders, and re-estimates theta.  Returns a DataFrame indexed by k
    with columns {theta, se_IF, n_kept, drop_indices, denom_T_resid_sq}.

    Notes: a refit (rather than a "leave-out-and-recompute-from-the-full-fit"
    shortcut) is the conservative thing to do because the cross-fit residuals
    depend on the kept rows; this is what Saco (2026) §4 recommends.
    """
    T = np.asarray(T, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    pc1 = _pc1_randomized(T, seed=seed)
    conf_s = StandardScaler().fit_transform(conf)
    Y_resid, T_resid = _residualize(T, conf_s, Y, pc1, k_folds, seed)
    base = compute_influence_and_leverage(T_resid, Y_resid, top_k=max(k_list))
    order = np.argsort(-np.abs(base["IC"]))

    rows = [{
        "k": 0, "theta": base["theta_hat"], "se_IF": base["se_IF"],
        "n_kept": base["n"], "drop_indices": [],
        "denom_T_resid_sq": base["denom_T_resid_sq"],
    }]

    for k in k_list:
        drop = set(order[:k].tolist())
        keep = np.array([i for i in range(len(Y)) if i not in drop])
        T_k, conf_k, Y_k = T[keep], conf[keep], Y[keep]
        try:
            pc1_k = _pc1_randomized(T_k, seed=seed)
            conf_s_k = StandardScaler().fit_transform(conf_k)
            Y_r, T_r = _residualize(T_k, conf_s_k, Y_k, pc1_k, k_folds, seed)
            sub = compute_influence_and_leverage(T_r, Y_r, top_k=1)
            rows.append({
                "k": k, "theta": sub["theta_hat"], "se_IF": sub["se_IF"],
                "n_kept": len(keep), "drop_indices": sorted(drop),
                "denom_T_resid_sq": sub["denom_T_resid_sq"],
            })
        except Exception as e:
            warnings.warn(f"theta_minus_top_k(k={k}) failed: {e}")
            rows.append({"k": k, "theta": float("nan"), "se_IF": float("nan"),
                         "n_kept": len(keep), "drop_indices": sorted(drop),
                         "denom_T_resid_sq": float("nan")})

    return pd.DataFrame(rows)



def _huber_psi(u: np.ndarray, c: float) -> np.ndarray:
    """Huber psi function: rho'(u) = u if |u|<=c else c sign(u)."""
    return np.clip(u, -c, c)


def trimmed_dml_huber(T: np.ndarray, conf: np.ndarray, Y: np.ndarray,
                      c: float = 1.345, k_folds: int = 5, seed: int = 0,
                      max_iter: int = 200, tol: float = 1e-9) -> dict:
    """Huber-robust residual-on-residual regression with tuning c=1.345.

    Cross-fits Y_tilde, T_tilde via the standard DML PLR loop, then solves
    the no-intercept Huber regression

        theta_HUB = argmin_theta sum_i rho_c( (Y_tilde_i - theta T_tilde_i) / sigma ),

    via iteratively-reweighted least squares (IRLS).  Scale sigma is fixed
    to MAD(Y_tilde - theta_OLS T_tilde) / 0.6745 (Maronna et al. 2006 eq.
    2.27), giving sigma a robust estimate even when the residuals have
    leptokurtic tails.  The IRLS update is

        theta_new = sum_i w_i T_tilde_i Y_tilde_i / sum_i w_i T_tilde_i^2,
        w_i       = psi(u_i / sigma) / (u_i / sigma)  with the limit at 0
                    handled by w_i = 1 (consistent with Huber's definition).

    Returns dict with theta_huber, se_huber, theta_ols (for comparison),
    iters, sigma_hat, n_clipped (rows with |residual|/sigma > c, i.e. the
    rows whose Huber weights are < 1).
    """
    T = np.asarray(T, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = Y.size

    pc1 = _pc1_randomized(T, seed=seed)
    conf_s = StandardScaler().fit_transform(conf)
    Y_resid, T_resid = _residualize(T, conf_s, Y, pc1, k_folds, seed)

    denom = float(np.mean(T_resid ** 2))
    if denom < 1e-12:
        raise RuntimeError("E[T_resid^2] ~ 0; estimator ill-conditioned.")
    theta_ols = float(np.mean(T_resid * Y_resid)) / denom

    resid_ols = Y_resid - theta_ols * T_resid
    mad = float(np.median(np.abs(resid_ols - np.median(resid_ols))))
    sigma_hat = mad / 0.6745 if mad > 0 else float(np.std(resid_ols, ddof=1))
    if sigma_hat <= 0:
        sigma_hat = 1.0

    theta = theta_ols
    for it in range(max_iter):
        u = (Y_resid - theta * T_resid) / sigma_hat
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(np.abs(u) <= c, 1.0, c / np.maximum(np.abs(u), 1e-12))
        num = float(np.sum(w * T_resid * Y_resid))
        den = float(np.sum(w * T_resid ** 2))
        if den < 1e-300:
            warnings.warn("Huber IRLS denominator collapsed; returning OLS.")
            return {
                "theta_huber": theta_ols, "se_huber": float("nan"),
                "theta_ols": theta_ols, "iters": it, "sigma_hat": sigma_hat,
                "n_clipped": 0, "c": c, "converged": False,
            }
        theta_new = num / den
        if abs(theta_new - theta) < tol * max(1.0, abs(theta)):
            theta = theta_new
            break
        theta = theta_new

    u = (Y_resid - theta * T_resid) / sigma_hat
    psi = _huber_psi(u, c)
    E_psi_sq = float(np.mean(psi ** 2))
    E_psi_prime = float(np.mean(np.abs(u) <= c))
    if E_psi_prime <= 0:
        warnings.warn("All Huber observations clipped; falling back to MAD SE.")
        se_huber = float(np.sqrt(np.var(psi * T_resid, ddof=1) / max(1, n)) *
                         sigma_hat / max(np.mean(T_resid ** 2), 1e-12))
    else:
        sumT2 = float(np.sum(T_resid ** 2))
        se_huber = float(np.sqrt(
            sigma_hat ** 2 * E_psi_sq / (E_psi_prime ** 2) / max(sumT2, 1e-300)
        ))

    n_clipped = int(np.sum(np.abs(u) > c))
    return {
        "theta_huber": float(theta), "se_huber": se_huber,
        "theta_ols": float(theta_ols), "iters": int(it + 1),
        "sigma_hat": float(sigma_hat), "n_clipped": n_clipped,
        "c": float(c), "converged": bool(it + 1 < max_iter),
    }



def dirichlet_bootstrap_bands(T: np.ndarray, conf: np.ndarray, Y: np.ndarray,
                              k_list=(1, 2, 5, 10, 20), B: int = 50,
                              k_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Per-k Dirichlet-bootstrap (Rubin 1981) sensitivity bands.

    For each bootstrap replicate b (Dirichlet(1,...,1) * n weights), we
    drop the rows whose Dirichlet-weighted |IC| is largest (mimicking the
    bootstrap distribution under weighted resampling) and recompute theta.
    Returns a DataFrame with columns k, mean, sd, lower_95, upper_95.
    """
    T = np.asarray(T, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = Y.size

    boots = {k: [] for k in (0,) + tuple(k_list)}
    rng = np.random.default_rng(seed)

    pc1 = _pc1_randomized(T, seed=seed)
    conf_s = StandardScaler().fit_transform(conf)

    for b in range(B):
        w = rng.dirichlet(np.ones(n)) * n
        res = _dml_core_fast(T, conf, Y, k_folds=k_folds, seed=seed + 1 + b,
                             cross_fit_pca=False, weights=w)
        if res is None:
            continue
        theta_b, _ = res
        boots[0].append(theta_b)
        Y_resid_w = np.zeros(n); T_resid_w = np.zeros(n)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed + 1 + b)
        for tr, te in kf.split(np.arange(n)):
            m_y = _make_lgbm(); m_t = _make_lgbm()
            m_y.fit(conf_s[tr], Y[tr], sample_weight=w[tr])
            m_t.fit(conf_s[tr], pc1[tr], sample_weight=w[tr])
            Y_resid_w[te] = Y[te] - m_y.predict(conf_s[te])
            T_resid_w[te] = pc1[te] - m_t.predict(conf_s[te])
        denom_w = float(np.average(T_resid_w ** 2, weights=w))
        if denom_w < 1e-12:
            continue
        theta_w = float(np.average(T_resid_w * Y_resid_w, weights=w)) / denom_w
        IC_w = ((Y_resid_w - theta_w * T_resid_w) * T_resid_w) / denom_w
        order = np.argsort(-np.abs(IC_w) * w)
        for k in k_list:
            drop = set(order[:k].tolist())
            keep = np.array([i for i in range(n) if i not in drop])
            if keep.size < 5:
                continue
            T_r = T_resid_w[keep]; Y_r = Y_resid_w[keep]; w_r = w[keep]
            denom_k = float(np.average(T_r ** 2, weights=w_r))
            if denom_k < 1e-12:
                continue
            theta_k = float(np.average(T_r * Y_r, weights=w_r)) / denom_k
            boots[k].append(theta_k)

    out_rows = []
    for k in (0,) + tuple(k_list):
        arr = np.array(boots[k]) if boots[k] else np.array([np.nan])
        out_rows.append({
            "k": k, "n_boots": int(np.sum(~np.isnan(arr))),
            "boot_mean": float(np.nanmean(arr)),
            "boot_sd": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else float("nan"),
            "boot_q025": float(np.nanquantile(arr, 0.025)) if arr.size > 1 else float("nan"),
            "boot_q975": float(np.nanquantile(arr, 0.975)) if arr.size > 1 else float("nan"),
        })
    return pd.DataFrame(out_rows)



def plot_leverage(T, conf, Y, out_path: Path, city: str = "city",
                  k_list=(1, 2, 5, 10, 20), B_boots: int = 0,
                  k_folds: int = 5, seed: int = 0) -> Path:
    """Two-panel figure: residual scatter + theta_{-k} curve.

    Panel A: T_tilde vs Y_tilde scatter, top-10 |IC| in red, OLS slope line.
    Panel B: theta_{-k} curve with IF-SE bands, plus Dirichlet-bootstrap
             bands if B_boots > 0.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = np.asarray(T, dtype=np.float64)
    conf = np.asarray(conf, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    pc1 = _pc1_randomized(T, seed=seed)
    conf_s = StandardScaler().fit_transform(conf)
    Y_resid, T_resid = _residualize(T, conf_s, Y, pc1, k_folds, seed)
    base = compute_influence_and_leverage(T_resid, Y_resid, top_k=10)
    sens = theta_minus_top_k(T, conf, Y, k_list=k_list, k_folds=k_folds, seed=seed)
    boots_df = (dirichlet_bootstrap_bands(T, conf, Y, k_list=k_list, B=B_boots,
                                          k_folds=k_folds, seed=seed)
                if B_boots > 0 else None)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].scatter(T_resid, Y_resid, s=12, alpha=0.45, color="steelblue",
                  label="all obs")
    top = base["top_idx"]
    ax[0].scatter(T_resid[top], Y_resid[top], s=46, color="crimson",
                  edgecolor="black", linewidth=0.5, label="top-10 |IC|", zorder=5)
    xs = np.linspace(T_resid.min(), T_resid.max(), 50)
    ax[0].plot(xs, base["theta_hat"] * xs, color="black", lw=1,
               label=fr"$\hat\theta_{{OLS}} = {base['theta_hat']:+.3f}$")
    ax[0].axhline(0, color="grey", lw=0.4)
    ax[0].axvline(0, color="grey", lw=0.4)
    ax[0].set_xlabel(r"$\widetilde T_i$ (PC1 residual)")
    ax[0].set_ylabel(r"$\widetilde Y_i$ (log-price residual)")
    ax[0].set_title(f"{city}: residual-on-residual, n={base['n']}")
    ax[0].legend(loc="best", fontsize=8)

    ax[1].plot(sens["k"], sens["theta"], marker="o", color="black",
               label=r"$\hat\theta_{-k}$ (IF SE band)")
    ax[1].fill_between(sens["k"],
                       sens["theta"] - 1.96 * sens["se_IF"],
                       sens["theta"] + 1.96 * sens["se_IF"],
                       alpha=0.18, color="steelblue")
    if boots_df is not None:
        ax[1].fill_between(boots_df["k"], boots_df["boot_q025"],
                           boots_df["boot_q975"], alpha=0.20, color="orange",
                           label=f"Dirichlet bootstrap (B={B_boots})")
    ax[1].axhline(0, color="grey", lw=0.5)
    ax[1].set_xlabel(r"$k$  (top-$|IC|$ rows dropped)")
    ax[1].set_ylabel(r"$\hat\theta$")
    ax[1].set_title(f"{city}: leverage sensitivity curve")
    ax[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path



def _run_city(city: str, plot: bool = False, B_boots: int = 0,
              huber_c: float = 1.345, k_folds: int = 5, seed: int = 0) -> dict | None:
    res = load_analysis_data(city)
    if res is None:
        print(f"[{city}] no data"); return None
    emb_df, parcels = res
    data = get_features_and_target(emb_df, parcels)
    if data is None:
        print(f"[{city}] no price target"); return None
    T, conf, Y, meta = data
    n = len(Y)
    print(f"\n[{city}] n={n} text_dim={T.shape[1]} confounders={conf.shape[1]}")

    pc1 = _pc1_randomized(T, seed=seed)
    conf_s = StandardScaler().fit_transform(conf)
    Y_resid, T_resid = _residualize(T, conf_s, Y, pc1, k_folds, seed)
    diag = compute_influence_and_leverage(T_resid, Y_resid, top_k=10)
    print(f"  theta_hat = {diag['theta_hat']:+.4f}   se_IF = {diag['se_IF']:.4f}")
    print(f"  E[T_resid^2] = {diag['denom_T_resid_sq']:.4f}   "
          f"kappa_DML ~ 1/E[T_resid^2] = {1/max(diag['denom_T_resid_sq'],1e-12):.2f}")
    print(f"  max h_leverage = {diag['h'].max():.4f}  (uniform would be {1/n:.4f}; "
          f"top-10 share = {np.sort(diag['h'])[-10:].sum():.3f})")
    print("\n  Top-10 rows by |IC|:")
    print(diag["summary"].to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    sens = theta_minus_top_k(T, conf, Y, k_folds=k_folds, seed=seed)
    print("\n  Sensitivity curve theta_{-k}:")
    print(sens[["k", "n_kept", "theta", "se_IF"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    huber = trimmed_dml_huber(T, conf, Y, c=huber_c, k_folds=k_folds, seed=seed)
    print(f"\n  Trimmed-DML (Huber c={huber['c']}): "
          f"theta_HUB={huber['theta_huber']:+.4f}  se={huber['se_huber']:.4f}  "
          f"(theta_OLS={huber['theta_ols']:+.4f}, "
          f"{huber['n_clipped']}/{n} rows clipped, "
          f"sigma_hat={huber['sigma_hat']:.3f}, iters={huber['iters']})")

    out = {
        "city": city, "n": int(n),
        "theta_hat": float(diag["theta_hat"]),
        "se_IF": float(diag["se_IF"]),
        "denom_T_resid_sq": float(diag["denom_T_resid_sq"]),
        "kappa_DML_proxy": float(1 / max(diag["denom_T_resid_sq"], 1e-12)),
        "max_h_leverage": float(diag["h"].max()),
        "top10_h_share": float(np.sort(diag["h"])[-10:].sum()),
        "top_idx": diag["top_idx"].tolist(),
        "summary_top10": diag["summary"].to_dict(orient="records"),
        "sensitivity": sens.to_dict(orient="records"),
        "huber_trimmed_dml": huber,
    }

    if B_boots > 0:
        boots_df = dirichlet_bootstrap_bands(T, conf, Y, B=B_boots,
                                             k_folds=k_folds, seed=seed)
        print("\n  Dirichlet-bootstrap sensitivity bands:")
        print(boots_df.to_string(index=False,
              float_format=lambda x: f"{x:8.4f}"))
        out["dirichlet_bands"] = boots_df.to_dict(orient="records")

    if plot:
        png_path = Path(f"results/leverage_{city}.png")
        plot_leverage(T, conf, Y, png_path, city=city, B_boots=B_boots,
                      k_folds=k_folds, seed=seed)
        print(f"  plot -> {png_path}")
        out["plot"] = str(png_path)
    return out



def _self_test(seed: int = 0) -> None:
    print("== diagnose_leverage self-test ==")
    rng = np.random.default_rng(seed)
    n = 500; theta_true = 0.4
    T = rng.normal(size=n)
    Y = theta_true * T + rng.normal(scale=0.5, size=n)
    denom = float(np.mean(T ** 2))
    theta_ols = float(np.mean(T * Y) / denom)
    diag = compute_influence_and_leverage(T, Y, top_k=5)
    assert abs(diag["theta_hat"] - theta_ols) < 1e-12
    se_check = float(np.sqrt(np.var(diag["IC"], ddof=1) / n))
    assert abs(diag["se_IF"] - se_check) < 1e-12
    print(f"  IC identity:   theta = E_n[T Y] / E_n[T^2] = {diag['theta_hat']:.4f}  OK")
    print(f"  se_IF identity: sqrt(Var(IC)/n) = {diag['se_IF']:.4f}  OK")

    Y_dirty = Y.copy()
    bad_idx = rng.choice(n, 25, replace=False)
    Y_dirty[bad_idx] += 10.0
    theta_ols_dirty = float(np.mean(T * Y_dirty) / np.mean(T ** 2))
    huber = trimmed_dml_huber(
        T=T.reshape(-1, 1), conf=np.column_stack([T, T ** 2]),
        Y=Y_dirty, c=1.345, k_folds=5, seed=seed,
    )
    assert huber["converged"], huber
    assert huber["n_clipped"] >= 0 and huber["sigma_hat"] > 0
    print(f"  Huber IRLS converged in {huber['iters']} iters, "
          f"clipped {huber['n_clipped']}/{n} obs.")

    T2 = rng.normal(size=n); Y2 = rng.normal(size=n)
    sens = theta_minus_top_k(
        T=T2.reshape(-1, 1), conf=rng.normal(size=(n, 3)), Y=Y2,
        k_list=(2, 5, 10), seed=seed,
    )
    assert (~sens["theta"].isna()).all(), sens
    print(f"  sensitivity curve runs on null data: "
          f"{sens[['k', 'theta', 'se_IF']].to_dict(orient='records')}")
    print("== self-test PASS ==")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cities", nargs="*",
                    help="subset of sf nyc boston; default = all three")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--B_boots", type=int, default=0,
                    help="if > 0, also fit Dirichlet-bootstrap sensitivity bands")
    ap.add_argument("--huber_c", type=float, default=1.345)
    ap.add_argument("--k_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", type=Path,
                    default=Path("results/leverage_diagnostics.json"))
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test(); return

    cities = args.cities or ["sf", "nyc", "boston"]
    rows = []
    for c in cities:
        r = _run_city(c, plot=args.plot, B_boots=args.B_boots,
                      huber_c=args.huber_c, k_folds=args.k_folds,
                      seed=args.seed)
        if r is not None:
            rows.append(r)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()

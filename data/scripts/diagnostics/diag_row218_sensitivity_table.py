"""Builds the JBES-appendix sensitivity table for NYC row 218 fragility.

Columns: theta_hat, SE_IF, SE_Salerno (q in {.05,.10,.15,.20}), CI95.
Rows:
  - Headline DML (ridge, k=5, cross-fit)
  - Drop top-1 |IC| (row 218)
  - Drop top-5 |IC|
  - Bonferroni IC trim (cutoff = sd(IC) * sqrt(2 log n))
  - Hampel redescending psi IRLS (a=1.7, b=3.4, c=8.5)
  - OLS of Y_resid on T_resid (plug-in, no robustness)
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _silence
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fast_bootstrap_dml_v2 import _build_features, _LEARNERS, _dml_core
from spatial_se import salerno_jackknife_hac

OUT = Path("results/diagnostics"); OUT.mkdir(parents=True, exist_ok=True)


def salerno_max(Yr, Tr, fold_ids, coords, theta):
    denom = float(np.mean(Tr * Tr))
    scores = (Yr - theta * Tr) * Tr / denom
    ses = []
    for q in (0.05, 0.10, 0.15, 0.20):
        try:
            r = salerno_jackknife_hac(scores, fold_ids, coords, bandwidth_quantile=q)
            ses.append(r['se'])
        except Exception:
            pass
    return float(max(ses)) if ses else float('nan')


def hampel_irls(Tr, Yr, breakdowns=(1.7, 3.4, 8.5), n_iter=30):
    theta = float(np.sum(Tr * Yr) / np.sum(Tr * Tr))
    a_, b_, c_ = breakdowns
    for _ in range(n_iter):
        u = Yr - theta * Tr
        sig = max(float(np.median(np.abs(u - np.median(u))) / 0.6745), 1e-6)
        r = u / sig
        absr = np.abs(r)
        w = np.ones_like(r)
        m2 = (absr > a_) & (absr <= b_)
        m3 = (absr > b_) & (absr <= c_)
        m4 = absr > c_
        w[m2] = a_ / absr[m2]
        w[m3] = a_ * (c_ - absr[m3]) / ((c_ - b_) * absr[m3])
        w[m4] = 0.0
        num = float(np.sum(w * Tr * Yr))
        den = float(np.sum(w * Tr * Tr))
        new = num / den
        if abs(new - theta) < 1e-10:
            theta = new; break
        theta = new
    u = Yr - theta * Tr
    sig = max(float(np.median(np.abs(u - np.median(u))) / 0.6745), 1e-6)
    absr = np.abs(u / sig)
    m1 = absr <= a_
    m2 = (absr > a_) & (absr <= b_)
    m3 = (absr > b_) & (absr <= c_)
    psi = np.zeros_like(u)
    psi[m1] = u[m1]
    psi[m2] = a_ * np.sign(u[m2]) * sig
    psi[m3] = a_ * (c_ - absr[m3]) / (c_ - b_) * np.sign(u[m3]) * sig
    psip = np.zeros_like(u)
    psip[m1] = 1.0
    psip[m3] = -a_ / (c_ - b_)
    e_psi2_T = float(np.mean((psi * Tr) ** 2))
    e_psip_T2 = float(np.mean(psip * Tr * Tr))
    n_local = len(u)
    se = float(np.sqrt(max(e_psi2_T / max(e_psip_T2 ** 2, 1e-12) / n_local, 0.0)))
    return theta, se, int((absr > c_).sum())


def main():
    data = _build_features("nyc")
    T, conf, Y, coords, meta = data
    n = len(Y)
    conf_s = StandardScaler().fit_transform(conf).astype(np.float64)
    fit = _LEARNERS["ridge"]

    theta, se_if, Yr, Tr, fold_ids = _dml_core(
        T, conf_s, Y, fit, seed=0, k_folds=5, return_residuals=True
    )
    denom = float(np.mean(Tr * Tr))
    IC = (Yr - theta * Tr) * Tr / denom
    order = np.argsort(-np.abs(IC))
    sd_IC = float(np.std(IC))
    bonf_cut = sd_IC * float(np.sqrt(2 * np.log(n)))

    rows = []

    def add(name, th, se_if_, n_kept, se_sal=None):
        rows.append({
            "spec": name,
            "n_kept": int(n_kept),
            "theta": float(th),
            "se_IF": float(se_if_),
            "se_Salerno_max": (float(se_sal) if se_sal is not None and np.isfinite(se_sal) else None),
            "ci_low_IF": float(th - 1.96 * se_if_),
            "ci_high_IF": float(th + 1.96 * se_if_),
        })

    sal_full = salerno_max(Yr, Tr, fold_ids, coords, theta)
    add("(1) headline DML, ridge, k=5", theta, se_if, n, sal_full)

    th_ols = float(np.sum(Tr * Yr) / np.sum(Tr * Tr))
    res_ols = Yr - th_ols * Tr
    se_ols = float(np.sqrt(np.sum(res_ols ** 2 * Tr ** 2)) / np.sum(Tr ** 2))
    add("(2) OLS Yr ~ Tr (plug-in)", th_ols, se_ols, n)

    for k in (1, 5):
        drop = set(order[:k].tolist())
        keep = np.array([i for i in range(n) if i not in drop])
        T_k, conf_k, Y_k = T[keep], conf_s[keep], Y[keep]
        res = _dml_core(T_k, conf_k, Y_k, fit, seed=0, k_folds=5, return_residuals=True)
        th_, se_, Yr_, Tr_, fid_ = res
        sal_ = salerno_max(Yr_, Tr_, fid_, coords[keep], th_)
        add(f"(3) drop top-{k} |IC|", th_, se_, len(keep), sal_)

    keep_b = np.abs(IC) <= bonf_cut
    T_b, conf_b, Y_b = T[keep_b], conf_s[keep_b], Y[keep_b]
    res = _dml_core(T_b, conf_b, Y_b, fit, seed=0, k_folds=5, return_residuals=True)
    th_, se_, Yr_, Tr_, fid_ = res
    sal_ = salerno_max(Yr_, Tr_, fid_, coords[keep_b], th_)
    add(f"(4) Bonferroni IC trim, cutoff={bonf_cut:.3f}", th_, se_, int(keep_b.sum()), sal_)

    th_h, se_h, n_clip = hampel_irls(Tr, Yr)
    add(f"(5) Hampel redescending psi (a,b,c=1.7,3.4,8.5), n_clipped={n_clip}",
        th_h, se_h, n)

    out = {
        "city": "nyc", "n": n, "bonferroni_cutoff": bonf_cut,
        "sd_IC": sd_IC, "denom_T_resid_sq": denom,
        "top10_IC_rows": [int(i) for i in order[:10]],
        "table": rows,
    }
    (OUT / "row218_sensitivity_table.json").write_text(json.dumps(out, indent=2))
    print("WROTE", OUT / "row218_sensitivity_table.json")
    fmt = "{:<54} {:>7} {:>9} {:>9} {:>10} {:>20}"
    print(fmt.format("spec", "n", "theta", "SE_IF", "SE_Sal_max", "CI95_IF"))
    for r in rows:
        ci = f"[{r['ci_low_IF']:+.3f}, {r['ci_high_IF']:+.3f}]"
        sal = f"{r['se_Salerno_max']:.4f}" if r['se_Salerno_max'] is not None else "  -- "
        print(fmt.format(r["spec"][:54], r["n_kept"],
                         f"{r['theta']:+.4f}", f"{r['se_IF']:.4f}", sal, ci))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

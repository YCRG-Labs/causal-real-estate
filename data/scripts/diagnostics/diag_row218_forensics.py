"""NYC top-IC row forensics: who is row 218, jackknife sensitivity, Hampel trim,
leave-one-fold-out. Output: results/diagnostics/row218_forensics.json + plots.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _silence  # noqa: F401
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fast_bootstrap_dml_v2 import (
    _build_features, _LEARNERS, _dml_core,
)
from sklearn.preprocessing import StandardScaler

OUT = Path("results/diagnostics"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    rng = np.random.default_rng(0)
    data = _build_features("nyc")
    T, conf, Y, coords, meta = data
    n = len(Y)
    scaler = StandardScaler().fit(conf)
    conf_s = scaler.transform(conf).astype(np.float64)

    fit = _LEARNERS["ridge"]
    theta, se_if, Yr, Tr, fold_ids = _dml_core(
        T, conf_s, Y, fit, seed=0, k_folds=5, return_residuals=True
    )
    denom = float(np.mean(Tr * Tr))
    psi = (Yr - theta * Tr) * Tr / denom
    IC = psi.copy()
    order = np.argsort(-np.abs(IC))
    top10 = [int(i) for i in order[:10]]
    print(f"theta_hat={theta:+.4f}  denom={denom:.4f}  top-10 row indices: {top10}")

    # 1) forensic dump on row 218 + the top-10 by |IC|
    emb_df = pd.read_parquet("data/processed/nyc_embeddings.parquet")
    # _build_features filters rows; we need to align indices. fast path: just dump
    # the SAME rows from the parquet by row index (best-effort; if v2 drops rows we
    # report what the parquet says at that integer position).
    fields = [c for c in ["url", "address", "zip", "price", "latitude", "longitude",
                          "bedrooms", "bldg_area_sqft", "year_built", "description"]
              if c in emb_df.columns]
    rows_dump = []
    for r in [218] + [int(i) for i in top10 if int(i) != 218]:
        if r >= len(emb_df):
            continue
        rd = emb_df.iloc[r][fields].to_dict()
        rd["row_index"] = int(r)
        rd["Y_resid"] = float(Yr[r]) if r < len(Yr) else None
        rd["T_resid"] = float(Tr[r]) if r < len(Tr) else None
        rd["IC"] = float(IC[r]) if r < len(IC) else None
        rd["IC_share_pct"] = float(IC[r] ** 2 / (IC ** 2).sum() * 100)
        if "description" in rd and rd["description"]:
            rd["description"] = rd["description"][:400]
        rows_dump.append(rd)

    # 2) leave-one-row-out theta sweep
    loo = []
    for k in [0, 1, 2, 3, 5, 10, 20, 30]:
        drop = set(order[:k].tolist())
        keep = np.array([i for i in range(n) if i not in drop])
        T_k, conf_k, Y_k = T[keep], conf_s[keep], Y[keep]
        res = _dml_core(T_k, conf_k, Y_k, fit, seed=0, k_folds=5,
                        return_residuals=True)
        if res is None:
            continue
        th, se, _, _, _ = res
        loo.append({"k": int(k), "n_kept": int(len(keep)),
                    "theta": float(th), "se_IF": float(se),
                    "ci_low": float(th - 1.96 * se),
                    "ci_high": float(th + 1.96 * se)})

    # 3) per-fold contribution: drop the fold containing row 218
    if 218 < n:
        fold_of_218 = int(fold_ids[218])
        keep = np.array([i for i in range(n) if fold_ids[i] != fold_of_218])
        T_k, conf_k, Y_k = T[keep], conf_s[keep], Y[keep]
        res = _dml_core(T_k, conf_k, Y_k, fit, seed=0, k_folds=4,
                        return_residuals=True)
        loo_fold = None
        if res is not None:
            th, se, _, _, _ = res
            loo_fold = {"dropped_fold": fold_of_218, "n_kept": int(len(keep)),
                        "theta": float(th), "se_IF": float(se)}
    else:
        loo_fold = None

    # 4) Hampel redescending psi-function trimmed DML
    # Hampel rho': psi(u) = u for |u|<=a; a*sgn(u) for a<|u|<=b;
    # a*(c-|u|)/(c-b)*sgn(u) for b<|u|<=c; 0 for |u|>c. Standard {a=1.7,b=3.4,c=8.5}.
    def hampel_irls(Tr, Yr, breakdowns=(1.7, 3.4, 8.5), n_iter=20):
        # iteratively reweight: theta solves sum w_i T_i (Y_i - T_i theta) = 0
        theta_h = float(np.sum(Tr * Yr) / np.sum(Tr * Tr))
        for _ in range(n_iter):
            u = Yr - theta_h * Tr
            sig = float(np.median(np.abs(u - np.median(u)))) / 0.6745
            sig = max(sig, 1e-6)
            r = u / sig
            a, b, c = breakdowns
            w = np.ones_like(r)
            absr = np.abs(r)
            w[(absr > a) & (absr <= b)] = a / absr[(absr > a) & (absr <= b)]
            mask3 = (absr > b) & (absr <= c)
            w[mask3] = (a * (c - absr[mask3]) / ((c - b) * absr[mask3]))
            w[absr > c] = 0.0
            num = float(np.sum(w * Tr * Yr))
            den = float(np.sum(w * Tr * Tr))
            if den <= 0:
                break
            new_theta = num / den
            if abs(new_theta - theta_h) < 1e-8:
                theta_h = new_theta; break
            theta_h = new_theta
        # asymptotic SE under Hampel psi (HRRS86 eq 4.2.9, p.105):
        # V(theta) = E[psi^2] / (E[psi'])^2 * 1 / E[Tr^2]; n in denom
        u = Yr - theta_h * Tr
        sig = max(float(np.median(np.abs(u - np.median(u))) / 0.6745), 1e-6)
        r = u / sig
        absr = np.abs(r)
        a_, b_, c_ = breakdowns
        m1 = absr <= a_
        m2 = (absr > a_) & (absr <= b_)
        m3 = (absr > b_) & (absr <= c_)
        m4 = absr > c_
        psi_u = np.zeros_like(u)
        psi_u[m1] = u[m1]
        psi_u[m2] = a_ * np.sign(u[m2]) * sig
        psi_u[m3] = a_ * (c_ - absr[m3]) / (c_ - b_) * np.sign(u[m3]) * sig
        psi_u[m4] = 0.0
        e_psi2 = float(np.mean((psi_u * Tr) ** 2))
        psip = np.zeros_like(u)
        psip[m1] = 1.0
        psip[m3] = -a_ / (c_ - b_)
        e_psip = float(np.mean(psip * Tr * Tr))
        n_local = len(u)
        var_theta = e_psi2 / max(e_psip ** 2, 1e-12) / n_local
        se_h = float(np.sqrt(max(var_theta, 0.0)))
        n_clipped = int(np.sum(absr > c_))
        return theta_h, se_h, n_clipped, sig
    th_h, se_h, n_clip, sig_h = hampel_irls(Tr, Yr)
    hampel = {"theta": th_h, "se": se_h, "n_clipped": n_clip, "sigma_hat": sig_h}

    # 5) "Bonferroni IC trim": drop rows with |IC| > c * (sigma_IC * sqrt(2 log n))
    sd_IC = float(np.std(IC))
    bonf_cutoff = sd_IC * float(np.sqrt(2 * np.log(n)))
    keep = np.abs(IC) <= bonf_cutoff
    T_b, conf_b, Y_b = T[keep], conf_s[keep], Y[keep]
    res = _dml_core(T_b, conf_b, Y_b, fit, seed=0, k_folds=5, return_residuals=True)
    if res is not None:
        th_b, se_b, _, _, _ = res
        bonf = {"cutoff": bonf_cutoff, "n_kept": int(keep.sum()),
                "theta": float(th_b), "se_IF": float(se_b)}
    else:
        bonf = None

    # 6) residual scatter plot
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(Tr, Yr, s=8, alpha=0.4)
    for r in top10:
        if r < n:
            ax[0].scatter(Tr[r], Yr[r], s=70, c='red',
                          edgecolors='black', linewidths=0.6,
                          label='top-10 |IC|' if r == top10[0] else None)
            ax[0].annotate(str(r), (Tr[r], Yr[r]), fontsize=7, xytext=(3, 3),
                           textcoords='offset points')
    ax[0].axhline(0, color='gray', lw=0.5)
    ax[0].axvline(0, color='gray', lw=0.5)
    ax[0].set_xlabel("T_resid (PC1 partialled out)")
    ax[0].set_ylabel("Y_resid (log-price partialled out)")
    ax[0].set_title("NYC residual scatter: top-10 |IC| highlighted")
    ax[0].legend(loc='lower right', fontsize=8)
    ks = [l["k"] for l in loo]; thetas = [l["theta"] for l in loo]
    ses = [l["se_IF"] for l in loo]
    ax[1].errorbar(ks, thetas, yerr=[1.96 * s for s in ses], fmt='o-',
                   capsize=4, color='steelblue', label='theta_{-k} +/- 1.96 SE')
    ax[1].axhline(0, color='gray', lw=0.5)
    ax[1].set_xlabel("k = # of top-|IC| rows dropped")
    ax[1].set_ylabel("theta_hat")
    ax[1].set_title("NYC sensitivity curve (deeper sweep)")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(OUT / "row218_forensics.png", dpi=140)

    out = {
        "city": "nyc", "n": int(n),
        "theta_hat_full": float(theta), "se_IF_full": float(se_if),
        "denom_T_resid_sq": float(denom),
        "top10_IC_rows": top10,
        "row_dump_top10_plus_218": rows_dump,
        "leave_top_k_out": loo,
        "leave_fold_of_218_out": loo_fold,
        "hampel_redescending": hampel,
        "bonferroni_IC_trim": bonf,
        "notes": [
            "Hampel breakdowns a=1.7, b=3.4, c=8.5 (Hampel-Ronchetti-Rousseeuw-Stahel 1986)",
            "Bonferroni cutoff = sd(IC) * sqrt(2 log n)",
            "leave_top_k_out shows how theta moves as the top-|IC| rows are dropped",
        ],
    }
    (OUT / "row218_forensics.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"WROTE {OUT/'row218_forensics.json'}")
    print(f"WROTE {OUT/'row218_forensics.png'}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

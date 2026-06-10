"""Random-effects pooling for the 12-market panel: Paule-Mandel tau^2 with the
Hartung-Knapp-Sidik-Jonkman (HKSJ) interval, the modified Knapp-Hartung (mKH)
correction recommended for few studies of unequal precision, and the
listing-weighted (fixed-effect and sample-size-weighted) complements.

The panel has K=12 with study sizes varying ~15-fold (n 987..15240), which is
exactly the regime where plain HKSJ can be anticonservative; mKH floors the
variance multiplier at 1 so the corrected SE never falls below the
inverse-variance SE (Röver, Knapp & Friede 2015). The RE mean answers "the
average market's effect"; the FE / size-weighted means answer "the average
listing's effect" and are reported as complements because one market (NY) is
~22% of total n.

Matches metafor: re_meta == rma(yi, sei, method="PM", test="knha"); the mKH
fields == rma(..., test="adhoc").

References: Paule & Mandel (1982); DerSimonian & Kacker (2007); Hartung & Knapp
(2001); Sidik & Jonkman (2002); IntHout, Ioannidis & Borm (2014); Röver, Knapp &
Friede (2015).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def pm_tau2(y, v, tol: float = 1e-7, maxit: int = 500) -> float:
    """Paule-Mandel tau^2: the root of sum_i w_i (y_i - ybar)^2 = K-1 with
    w_i = 1/(v_i + tau2). Newton step (DerSimonian-Kacker), floored at 0."""
    y = np.asarray(y, float)
    v = np.asarray(v, float)
    k = len(y)
    tau2 = 0.0
    for _ in range(maxit):
        w = 1.0 / (v + tau2)
        ybar = np.sum(w * y) / np.sum(w)
        F = np.sum(w * (y - ybar) ** 2) - (k - 1)
        dF = -np.sum(w ** 2 * (y - ybar) ** 2)
        if dF == 0:
            break
        new = tau2 - F / dF
        if new < 0.0:
            return 0.0
        if abs(new - tau2) < tol:
            return float(new)
        tau2 = new
    return float(tau2)


def re_meta(y, v, n=None, alpha: float = 0.05) -> dict:
    """RE pool with PM tau^2; returns RE-z, HKSJ, mKH intervals and the FE /
    size-weighted complements."""
    y = np.asarray(y, float)
    v = np.asarray(v, float)
    k = len(y)
    tau2 = pm_tau2(y, v)
    w = 1.0 / (v + tau2)
    mu = np.sum(w * y) / np.sum(w)
    se_iv = np.sqrt(1.0 / np.sum(w))
    q = np.sum(w * (y - mu) ** 2) / (k - 1)
    qstar = max(1.0, q)
    tcr = stats.t.ppf(1 - alpha / 2, k - 1)
    zcr = stats.norm.ppf(1 - alpha / 2)
    se_hksj = np.sqrt(q) * se_iv
    se_mkh = np.sqrt(qstar) * se_iv

    out = {
        "k": k, "tau2": tau2, "mu": float(mu), "I2": float(_i2(y, v, tau2)),
        "se_RE": float(se_iv), "ci_RE": (float(mu - zcr * se_iv), float(mu + zcr * se_iv)),
        "q": float(q), "qstar": float(qstar),
        "se_HKSJ": float(se_hksj), "ci_HKSJ": (float(mu - tcr * se_hksj), float(mu + tcr * se_hksj)),
        "se_mKH": float(se_mkh), "ci_mKH": (float(mu - tcr * se_mkh), float(mu + tcr * se_mkh)),
        "mkh_binds": bool(qstar > q),
    }
    wf = 1.0 / v
    out["FE_mu"] = float(np.sum(wf * y) / np.sum(wf))
    out["FE_se"] = float(np.sqrt(1.0 / np.sum(wf)))
    if n is not None:
        n = np.asarray(n, float)
        out["N_mu"] = float(np.sum(n * y) / np.sum(n))
        out["N_se"] = float(np.sqrt(np.sum(n ** 2 * v)) / np.sum(n))
    return out


def _i2(y, v, tau2) -> float:
    k = len(y)
    wf = 1.0 / np.asarray(v, float)
    muf = np.sum(wf * y) / np.sum(wf)
    Q = np.sum(wf * (np.asarray(y, float) - muf) ** 2)
    if Q <= 0:
        return 0.0
    return max(0.0, (Q - (k - 1)) / Q) * 100.0


if __name__ == "__main__":
    # Unit test against the hand-computed worked example (equal SEs => q==1, so
    # HKSJ == mKH, and both use t_{K-1}).
    y = [0.0, 0.2, 0.4, 0.6]
    se = [0.2, 0.2, 0.2, 0.2]
    n = [2000, 1000, 1000, 1000]
    r = re_meta(y, np.array(se) ** 2, n=n)
    ok = []
    ok.append(("tau2≈0.026667", abs(r["tau2"] - 0.0266667) < 1e-4))
    ok.append(("mu==0.30", abs(r["mu"] - 0.30) < 1e-9))
    ok.append(("se_RE≈0.129099", abs(r["se_RE"] - 0.1290994) < 1e-5))
    ok.append(("q==1", abs(r["q"] - 1.0) < 1e-6))
    ok.append(("HKSJ low≈-0.110843", abs(r["ci_HKSJ"][0] + 0.110843) < 1e-4))
    ok.append(("HKSJ high≈0.710843", abs(r["ci_HKSJ"][1] - 0.710843) < 1e-4))
    ok.append(("mKH==HKSJ here", r["ci_mKH"] == r["ci_HKSJ"]))
    ok.append(("FE_se==0.10", abs(r["FE_se"] - 0.10) < 1e-9))
    ok.append(("N_mu==0.24", abs(r["N_mu"] - 0.24) < 1e-9))
    for name, passed in ok:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    raise SystemExit(0 if all(p for _, p in ok) else 1)

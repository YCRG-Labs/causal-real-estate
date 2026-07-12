"""Monte Carlo calibration of the Composition-and-Conditioning Audit (CoCA).

CoCA screens a partially-linear DML estimate before theta is inspected, in two
steps with pre-committed thresholds:

  Step 1 (conditioning): flag if cond(X) > 1e6 or any confounder column has zero
          out-of-sample variance. This catches a design-matrix degeneracy, e.g. a
          confounder block that a silent pipeline bug zero-filled to a constant,
          which inflates the nuisance variance without any omitted-confounding story.
  Step 2 (composition):  flag if an omitted categorical C is chance-predictable from
          the controls X, AUC(C|X) < 0.60, yet strongly predictable from the
          treatment, AUC(C|T) > 0.75. This catches an omitted category (e.g. property
          type) the design cannot see but the text-derived treatment encodes.

The claim under test is that these two steps DISCRIMINATE two distinct failure
modes: Step 1 fires on the degeneracy and not the composition confound, Step 2 fires
on the composition confound and not the degeneracy, with low false-positive and
false-negative rates across a grid of category shares and confounder strengths.

Three data-generating processes, common structure Y = theta T + X beta_Y + eps,
T = X beta_T + nu, X ~ N(0, I_p):

  clean         no omitted category, no degenerate column        (both steps should pass)
  composition   omitted binary C orthogonal to X (C independent of X) with C -> T and
                C -> Y net of T; no degenerate column             (Step 2 should fire, Step 1 not)
  degeneracy    a confounder column overwritten by a constant;
                no omitted category                               (Step 1 should fire, Step 2 not)

    python data/scripts/coca_montecarlo.py --n_rep 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "coca_montecarlo.json"
OUT_BYSTRENGTH = REPO / "results" / "coca_montecarlo_bystrength.json"
ALPHAS = np.logspace(-3, 3, 13)
COND_THRESH = 1e6
AUC_X_THRESH = 0.60
AUC_T_THRESH = 0.75


def _oof(X, y, seed):
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        pred[te] = RidgeCV(alphas=ALPHAS).fit(X[tr], y[tr]).predict(X[te])
    return pred


def _r2(y, pred):
    ss = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - pred) ** 2).sum() / ss if ss > 0 else 0.0


def _auc(Z, c):
    if c.sum() < 3 or (1 - c).sum() < 3:
        return float("nan")
    p = cross_val_predict(LogisticRegression(max_iter=2000), Z, c, cv=5,
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(c, p))


def make_data(mode, n, p, share, a, b, theta, rng):
    X = rng.standard_normal((n, p))
    beta_T = rng.standard_normal(p) / np.sqrt(p)
    beta_Y = rng.standard_normal(p) / np.sqrt(p)
    C = (rng.random(n) < share).astype(float)
    Cc = C - C.mean()
    if mode == "composition":
        T = X @ beta_T + a * Cc + rng.standard_normal(n)
        Y = theta * T + X @ beta_Y + b * Cc + rng.standard_normal(n)
    else:
        T = X @ beta_T + rng.standard_normal(n)
        Y = theta * T + X @ beta_Y + rng.standard_normal(n)
    if mode == "degeneracy":
        X = X.copy()
        X[:, rng.integers(0, p)] = 3.14159  # a silently constant column
    return X, T, Y, C


def audit(X, T, C, seed=0):
    sd = X.std(0)
    zero_var = int((sd < 1e-9).sum())
    Xs = (X - X.mean(0)) / np.where(sd < 1e-12, 1.0, sd)
    cond_X = float(np.linalg.cond(Xs))
    r2_t = _r2(T, _oof(Xs, T, seed))
    kappa = 1.0 / max(1e-12, 1.0 - max(0.0, r2_t))
    step1 = bool(cond_X > COND_THRESH or zero_var > 0)

    auc_x = _auc(Xs, C.astype(int))
    auc_t = _auc(T.reshape(-1, 1), C.astype(int))
    step2 = bool(np.isfinite(auc_x) and np.isfinite(auc_t)
                 and auc_x < AUC_X_THRESH and auc_t > AUC_T_THRESH)
    return dict(cond_X=cond_X, zero_var=zero_var, kappa=kappa,
                auc_x=auc_x, auc_t=auc_t, step1=step1, step2=step2)


def run(n_rep, n=1500, p=10, theta=0.3):
    shares = [0.02, 0.05, 0.10, 0.20]
    strengths = [(1.0, 1.0), (2.0, 2.0)]
    cells = []
    for mode in ("clean", "composition", "degeneracy"):
        grid = [(s, a, b) for s in shares for (a, b) in strengths] if mode == "composition" \
            else [(0.05, 0.0, 0.0)]
        for (share, a, b) in grid:
            s1 = s2 = 0
            aucx = []
            auct = []
            conds = []
            for r in range(n_rep):
                rng = np.random.default_rng(10_000 * len(cells) + r)
                X, T, Y, C = make_data(mode, n, p, share, a, b, theta, rng)
                res = audit(X, T, C, seed=r)
                s1 += res["step1"]
                s2 += res["step2"]
                aucx.append(res["auc_x"])
                auct.append(res["auc_t"])
                conds.append(res["cond_X"])
            cells.append(dict(
                mode=mode, share=share, a=a, b=b, n_rep=n_rep,
                step1_rate=s1 / n_rep, step2_rate=s2 / n_rep,
                mean_auc_x=float(np.nanmean(aucx)), mean_auc_t=float(np.nanmean(auct)),
                median_cond=float(np.nanmedian(conds))))
            c = cells[-1]
            print(f"{mode:12s} share={share:.2f} a={a:.1f} | "
                  f"Step1 fires {c['step1_rate']*100:5.1f}%  Step2 fires {c['step2_rate']*100:5.1f}%  "
                  f"AUC(C|X)={c['mean_auc_x']:.2f} AUC(C|T)={c['mean_auc_t']:.2f}  "
                  f"cond~{c['median_cond']:.1e}", flush=True)
    return cells


def summarize(cells):
    comp = [c for c in cells if c["mode"] == "composition"]
    deg = [c for c in cells if c["mode"] == "degeneracy"]
    clean = [c for c in cells if c["mode"] == "clean"]
    return {
        "composition_step2_detection": float(np.mean([c["step2_rate"] for c in comp])),
        "composition_step1_falsepos": float(np.mean([c["step1_rate"] for c in comp])),
        "degeneracy_step1_detection": float(np.mean([c["step1_rate"] for c in deg])),
        "degeneracy_step2_falsepos": float(np.mean([c["step2_rate"] for c in deg])),
        "clean_step1_falsepos": float(np.mean([c["step1_rate"] for c in clean])),
        "clean_step2_falsepos": float(np.mean([c["step2_rate"] for c in clean])),
    }


def summarize_by_strength(cells):
    """Step-2 detection broken out by composition strength (a) and category share.

    A single pooled composition_step2_detection average is misleading because
    the grid mixes a weak-signal regime (a=1.0, share small -> Step 2 barely
    fires) with a strong-signal regime (a=2.0 -> Step 2 fires almost always);
    the pooled mean lands in between and represents neither regime.
    """
    comp = [c for c in cells if c["mode"] == "composition"]
    strengths = sorted({c["a"] for c in comp})
    by_strength = {}
    for a in strengths:
        at_a = sorted([c for c in comp if c["a"] == a], key=lambda c: c["share"])
        per_share = {
            f"share={c['share']:.2f}": {
                "step2_rate": c["step2_rate"], "mean_auc_x": c["mean_auc_x"],
                "mean_auc_t": c["mean_auc_t"], "n_rep": c["n_rep"],
            }
            for c in at_a
        }
        mean_rate = float(np.mean([c["step2_rate"] for c in at_a]))
        by_strength[f"a={a:.1f}"] = {"per_share": per_share, "mean_step2_rate": mean_rate}
    return {"by_strength": by_strength, "n_rep": comp[0]["n_rep"] if comp else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rep", type=int, default=200)
    args = ap.parse_args()
    cells = run(args.n_rep)
    summary = summarize(cells)
    OUT.write_text(json.dumps({"cells": cells, "summary": summary}, indent=2))
    print("\n=== summary (detection = true positive, falsepos = wrong-step firing) ===")
    for k, v in summary.items():
        print(f"  {k:34s} {v*100:5.1f}%")
    print(f"\nwrote {OUT}")

    by_strength = summarize_by_strength(cells)
    OUT_BYSTRENGTH.write_text(json.dumps({"cells": cells, "by_strength": by_strength}, indent=2))
    print(f"\n=== Step-2 detection by composition strength and category share (n_rep={by_strength['n_rep']}) ===")
    print(f"the pooled {summary['composition_step2_detection']*100:.1f}% above averages a bimodal result:")
    for a_key, block in by_strength["by_strength"].items():
        print(f"\n  {a_key}  (mean across shares: {block['mean_step2_rate']*100:5.1f}%)")
        for share_key, r in block["per_share"].items():
            print(f"    {share_key:12s} Step2 fires {r['step2_rate']*100:5.1f}%  "
                  f"AUC(C|X)={r['mean_auc_x']:.2f} AUC(C|T)={r['mean_auc_t']:.2f}")
    print(f"\nwrote {OUT_BYSTRENGTH}")


if __name__ == "__main__":
    main()

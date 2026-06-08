"""Per-city DML conditioning / overlap diagnostic — explains the Portland degeneracy.

Computes, per market, the quantities that distinguish an ill-conditioned orthogonal
score (Saco 2025 arXiv:2512.07083; Chernozhukov et al. 2018; Gui & Veitch 2023):

  R2_oof(Y|X)   outcome-nuisance out-of-fold fit   (~1 => outcome leak / bad control)
  R2_oof(T|X)   treatment-nuisance out-of-fold fit  (~1 => apparent-overlap collapse)
  kappa = 1/(1 - R2_oof(T|X))   Saco conditioning number  (>100 flags it)
  E[Ttilde^2]   partialling-out denominator E[Var(T|X)]  (->0 = degenerate)
  final_R2 = 1 - Var(eps)/Var(Ytilde), eps = Ytilde - theta*Ttilde
            (->1 => residuals near-collinear => the spuriously tiny SE)
  theta_hat = E[Ttilde*Ytilde]/E[Ttilde^2]   the partialling-out ratio

and localizes the leaking confounder by |corr| with log-price and with the text score.
Treatment = within-city PC1 of the BERT embedding (the degeneracy is representation-
agnostic, per the cross-method evidence, so PC1 is a faithful self-contained proxy).

Run on Brev (the 9 expanded cities' confounders live there):
    python3 data/scripts/replications/portland_conditioning_diag.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

import causal_inference as ci

try:
    import canonical_confounders as cc
    NAMES_BASE = ["lat", "lon"] + list(cc.PROPERTY) + list(cc.CONTEXTUAL)
except Exception:
    NAMES_BASE = None

CITIES = ["sf", "boston", "nyc", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]
ALPHAS = np.logspace(-3, 3, 13)


def oof_predict(X, y, seed=42):
    pred = np.zeros(len(y), float)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        pred[te] = RidgeCV(alphas=ALPHAS).fit(X[tr], y[tr]).predict(X[te])
    return pred


def r2(y, pred):
    sstot = ((y - y.mean()) ** 2).sum()
    return 1 - ((y - pred) ** 2).sum() / sstot if sstot > 0 else float("nan")


def safe_corr(x, y):
    return abs(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0


rows = []
for c in CITIES:
    try:
        loaded = ci.load_analysis_data(c)
        if loaded is None:
            print(f"{c:<13} no data (skip)")
            continue
        emb_df, parcels = loaded
        feats = ci.get_features_and_target(emb_df, parcels, drop_mismatched_crime=True)
        if feats is None:
            print(f"{c:<13} get_features_and_target None (skip)")
            continue
        T_emb, X, Y, meta = feats
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        ok = np.isfinite(Y) & (Y > 0) & np.all(np.isfinite(X), axis=1)
        T_emb, X, Y = np.asarray(T_emb, float)[ok], X[ok], Y[ok]
        logY = np.log(Y)

        sd = X.std(0)
        name0 = (lambda j: NAMES_BASE[j] if (NAMES_BASE and j < len(NAMES_BASE)) else f"col{j}")
        nz = np.array([np.minimum((X[:, j] != X[:, j].min()).sum(),
                                  (X[:, j] != X[:, j].max()).sum()) for j in range(X.shape[1])])
        degen_cols = [(name0(j), float(sd[j]), int(nz[j]))
                      for j in range(X.shape[1]) if sd[j] < 1e-6 or nz[j] <= 5]
        near_const = len(degen_cols)
        Xs = (X - X.mean(0)) / np.where(sd < 1e-12, 1.0, sd)
        cond_X = float(np.linalg.cond(Xs))

        E = T_emb - T_emb.mean(0)
        pc1 = PCA(1, random_state=0).fit_transform(E).ravel()
        t = (pc1 - pc1.mean()) / pc1.std()

        absz = np.abs(Xs)
        maxz = float(absz.max())
        ri, cj = np.unravel_index(int(absz.argmax()), Xs.shape)
        max_lev = (name0(int(cj)), round(maxz, 1), int(ri),
                   round(float(X[ri, cj]), 3))
        Xw = np.clip(Xs, -5, 5)
        r2y_w = r2(logY, oof_predict(Xw, logY))

        yhat, that = oof_predict(Xs, logY), oof_predict(Xs, t)
        r2y, r2t = r2(logY, yhat), r2(t, that)
        Yt, Tt = logY - yhat, t - that
        denom = float((Tt ** 2).mean())
        theta = float((Tt * Yt).sum() / (Tt ** 2).sum())
        eps = Yt - theta * Tt
        final_r2 = float(1 - eps.var() / Yt.var()) if Yt.var() > 0 else float("nan")
        kappa = 1.0 / max(1e-12, 1.0 - max(0.0, r2t))

        cy = np.array([safe_corr(X[:, j], logY) for j in range(X.shape[1])])
        ct = np.array([safe_corr(X[:, j], t) for j in range(X.shape[1])])
        name = (lambda j: NAMES_BASE[j] if (NAMES_BASE and j < len(NAMES_BASE)) else f"col{j}")
        topy = [(name(j), round(float(cy[j]), 3)) for j in np.argsort(-cy)[:3]]
        topt = [(name(j), round(float(ct[j]), 3)) for j in np.argsort(-ct)[:3]]

        rows.append(dict(city=c, n=len(Y), n_conf=X.shape[1], r2_y_X=r2y, r2_t_X=r2t,
                         kappa=kappa, denom_E_Tt2=denom, final_r2=final_r2,
                         theta=theta, cond_X=cond_X, degen_cols=degen_cols,
                         top_corr_Y=topy, top_corr_T=topt))
        rows[-1]["max_leverage_col"] = max_lev
        rows[-1]["r2_y_X_winsor5"] = r2y_w
        print(f"{c:<13} n={len(Y):>6}  R2(Y|X)={r2y:+9.3f}  R2(Y|X,clip5)={r2y_w:+.3f}  "
              f"finalR2={final_r2:+.3f}  theta={theta:+.3f}  "
              f"maxz={max_lev[1]:>7} @row{max_lev[2]} col={max_lev[0]}")
    except Exception as e:
        print(f"{c:<13} ERROR {type(e).__name__}: {e}")

print("\n=== degenerate / high-leverage confounder columns (name, sd, "
      "#rows differing from the constant value) ===")
for r in rows:
    flag = "  <== ILL-CONDITIONED" if (r["cond_X"] > 1e6 or r["final_r2"] > 0.99) else ""
    dc = r["degen_cols"] if r["degen_cols"] else "(none)"
    print(f"{r['city']:<13} cond(X)={r['cond_X']:.2e}  degenerate_cols={dc}{flag}")

print("\n=== leak localization (top confounders by |corr|) ===")
for r in rows:
    print(f"{r['city']:<13} |corr w/ logprice|: {r['top_corr_Y']}   "
          f"|corr w/ text-PC1|: {r['top_corr_T']}")

out = Path(__file__).resolve().parents[3] / "results/replications/portland_conditioning_diag.json"
out.write_text(json.dumps(rows, indent=1, default=str))
print(f"\nJSON -> {out}")

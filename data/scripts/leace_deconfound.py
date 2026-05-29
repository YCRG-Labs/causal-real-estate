"""
LEACE-based location erasure on text embeddings, with SPLINCE as covariance-
preserving variant. Replaces multi-head gradient reversal as the deconfounding
method. The frozen-encoder probe is repurposed as a *validation* of residual
nonlinear leakage past provable linear erasure rather than as a post-mortem
on a method known to fail.

References:
  Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman (NeurIPS 2023)
    "LEACE: Perfect Linear Concept Erasure in Closed Form"  arXiv:2306.03819
  Holstege, Ravfogel, Wouters (NeurIPS 2025) "SPLINCE"      arXiv:2506.10703
  Fan, Tian, Ravfogel, Sachan, Ash, Hoyle (EMNLP 2025)
    "The Medium Is Not the Message"                         arXiv:2507.01234
  Kumar, Tan, Sharma (NeurIPS 2022) "Probing Classifiers Are Unreliable for
    Concept Removal and Detection"

Install: pip install "concept-erasure>=0.2.4"

Usage:
    python leace_deconfound.py --city sf [--variant leace|splince] [--all]
"""
from __future__ import annotations
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import _silence  # noqa: F401
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_inference import load_analysis_data, get_features_and_target


def _ensure_concept_erasure():
    try:
        from concept_erasure import LeaceFitter, LeaceEraser  # noqa: F401
        return True
    except ImportError:
        print("install with: pip install 'concept-erasure>=0.2.4'", file=sys.stderr)
        return False


def leace_erase_location(T, Z_zip, Z_latlon, Z_income,
                         T_holdout=None,
                         Z_zip_holdout=None, Z_latlon_holdout=None,
                         Z_income_holdout=None,
                         shrinkage=True, dtype=torch.float64):
    """Joint LEACE on concatenated [one-hot zip | std lat/lon | std income].

    Joint Z is the right idiom: the kernel of the single projection equals
    col-span(Sigma_{x,Z_joint}), which is the union of all three concept
    subspaces, with the minimum-distortion objective solved once.
    """
    from concept_erasure import LeaceFitter
    T = T.to(dtype)

    enc_zip = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    Z_zip_oh = enc_zip.fit_transform(np.asarray(Z_zip).reshape(-1, 1))
    scaler_ll = StandardScaler().fit(np.asarray(Z_latlon))
    scaler_inc = StandardScaler().fit(np.asarray(Z_income).reshape(-1, 1))
    Z_ll = scaler_ll.transform(np.asarray(Z_latlon))
    Z_inc = scaler_inc.transform(np.asarray(Z_income).reshape(-1, 1))
    Z = np.concatenate([Z_zip_oh, Z_ll, Z_inc], axis=1)
    Z_t = torch.from_numpy(Z).to(dtype)

    fitter = LeaceFitter(
        x_dim=T.shape[1], z_dim=Z_t.shape[1],
        affine=True, constrain_cov_trace=True,
        shrinkage=shrinkage, svd_tol=0.01, dtype=dtype,
    )
    fitter.update(T, Z_t)
    eraser = fitter.eraser

    # Linear-guardedness sanity check (Belrose 2023 Theorem 1):
    # P @ Sigma_xz should be ~zero up to svd_tol
    P = eraser.P
    Sigma_xz = fitter.sigma_xz
    residual = (P @ Sigma_xz).abs().max().item()
    if residual > 1e-4:
        print(f"  warning: LEACE residual {residual:.2e} > 1e-4")

    T_erased = eraser(T)
    T_holdout_erased = None
    if T_holdout is not None:
        T_holdout_erased = eraser(T_holdout.to(dtype))
    return T_erased, T_holdout_erased, eraser


def _whitening_matrix(Sigma_xx, eps=1e-8):
    eigvals, eigvecs = np.linalg.eigh(Sigma_xx)
    eigvals = np.clip(eigvals, 0.0, None)
    nz = eigvals > eps * eigvals.max()
    inv_sqrt = np.zeros_like(eigvals)
    sqrt = np.zeros_like(eigvals)
    inv_sqrt[nz] = 1.0 / np.sqrt(eigvals[nz])
    sqrt[nz] = np.sqrt(eigvals[nz])
    W = (eigvecs * inv_sqrt) @ eigvecs.T
    W_pinv = (eigvecs * sqrt) @ eigvecs.T
    return W, W_pinv


def _orth_basis(M, tol=1e-10):
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    rank = int((s > tol * s.max()).sum()) if s.size else 0
    return U[:, :rank]


def splince_erase_location(T, Z_zip, Z_latlon, Z_income, Y_logprice,
                           T_holdout=None, shrinkage_eps=1e-3):
    """SPLINCE: erase location while exactly preserving Cov(T, Y_logprice).

    Implements P* = W^+ V (U^T V)^-1 U^T W (Holstege et al. NeurIPS 2025 eq. 4)
    with the numerically stable pseudoinverse from their proj.py.
    """
    X = T.detach().cpu().numpy().astype(np.float64)
    n, d = X.shape

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    Z_zip_oh = enc.fit_transform(np.asarray(Z_zip).reshape(-1, 1))
    Z = np.concatenate([
        Z_zip_oh[:, :-1],
        StandardScaler().fit_transform(np.asarray(Z_latlon)),
        StandardScaler().fit_transform(np.asarray(Z_income).reshape(-1, 1)),
    ], axis=1)
    Y = np.asarray(Y_logprice).reshape(n, -1).astype(np.float64)
    Y = Y - Y.mean(axis=0, keepdims=True)

    x_mean = X.mean(axis=0)
    Xc = X - x_mean
    Zc = Z - Z.mean(axis=0, keepdims=True)

    S_xx = (Xc.T @ Xc) / (n - 1)
    S_xx += shrinkage_eps * np.trace(S_xx) / d * np.eye(d)

    W, W_pinv = _whitening_matrix(S_xx)
    XW = Xc @ W
    Sigma_xz = (XW.T @ Zc) / (n - 1)
    Sigma_xy = (XW.T @ Y) / (n - 1)

    V = _orth_basis(Sigma_xz)
    U = _orth_basis(Sigma_xy)
    UTV = U.T @ V
    UTV_inv = np.linalg.pinv(UTV)

    P_prime = V @ UTV_inv @ U.T
    P_star = np.eye(d) - W_pinv @ P_prime @ W
    b_star = x_mean - P_star @ x_mean

    def apply(M):
        Xa = M.detach().cpu().numpy().astype(np.float64)
        return torch.from_numpy(Xa @ P_star.T + b_star)

    T_erased = apply(T)
    T_holdout_erased = apply(T_holdout) if T_holdout is not None else None
    return T_erased, T_holdout_erased, P_star


class _MLPProbe(nn.Module):
    def __init__(self, d_in, n_classes, hidden=256, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def linear_guardedness_test(T_tr, T_te, Z_tr, Z_te):
    Xtr = T_tr.detach().cpu().numpy()
    Xte = T_te.detach().cpu().numpy()
    Xtr = Xtr - Xtr.mean(axis=0)
    Xte = Xte - Xtr.mean(axis=0)
    lr = LogisticRegression(penalty=None, tol=0.0, max_iter=5000).fit(Xtr, Z_tr)
    chance = float(np.bincount(Z_te).max() / len(Z_te))
    return {
        "linear_acc": float(lr.score(Xte, Z_te)),
        "chance_acc": chance,
        "max_coef": float(np.abs(lr.coef_).max()),
    }


def mlp_probe(T_tr, T_te, Z_tr, Z_te, n_epochs=80, lr=1e-3,
              weight_decay=1e-4, hidden=256, seed=0):
    torch.manual_seed(seed)
    n_classes = int(np.max(Z_tr)) + 1
    device = T_tr.device
    model = _MLPProbe(T_tr.shape[1], n_classes, hidden=hidden).to(device).double()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    Ztr_t = torch.from_numpy(Z_tr).long().to(device)
    Zte_t = torch.from_numpy(Z_te).long().to(device)
    for _ in range(n_epochs):
        model.train()
        logits = model(T_tr.double())
        loss = loss_fn(logits, Ztr_t)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(T_te.double()).argmax(-1)
        probe_acc = float((preds == Zte_t).float().mean().item())
    baseline_acc = float(np.bincount(Z_te).max() / len(Z_te))
    return {
        "probe_acc": probe_acc,
        "baseline_acc": baseline_acc,
        "leakage": probe_acc - baseline_acc,
    }


def run_leace(city, variant="leace", seed=42):
    if not _ensure_concept_erasure():
        return None
    print(f"\n[{city}] {variant.upper()} deconfounding")
    res = load_analysis_data(city)
    if res is None:
        return None
    emb_df, parcels = res
    data = get_features_and_target(emb_df, parcels)
    if data is None:
        return None
    T_np, conf, Y, meta = data
    Y = np.asarray(Y).ravel()
    n = len(Y)

    # get_features_and_target may have dropped rows for invalid Y; trim emb_df
    # to match (rows correspond 1:1 from the top, per causal_inference loader)
    if len(emb_df) != n:
        emb_df = emb_df.iloc[:n].reset_index(drop=True)

    zip_codes = emb_df["zip"].astype(str).values
    z_lab = np.unique(zip_codes, return_inverse=True)[1]
    latlon = emb_df[["latitude", "longitude"]].values.astype(np.float64)
    inc_col = next((c for c in ["median_household_income", "median_income"]
                    if c in emb_df.columns), None)
    if inc_col is None:
        income = np.zeros(n)
    else:
        income = emb_df[inc_col].fillna(emb_df[inc_col].median()).values.astype(np.float64)

    # drop rows with non-finite values in T or any Z column (LEACE SVD will
    # crash on NaN/Inf; concept-erasure's whitening assumes finite covariances)
    T_arr = np.asarray(T_np, dtype=np.float64)
    finite_T = np.isfinite(T_arr).all(axis=1)
    finite_ll = np.isfinite(latlon).all(axis=1)
    finite_inc = np.isfinite(income)
    finite_Y = np.isfinite(Y)
    mask = finite_T & finite_ll & finite_inc & finite_Y
    n_drop = int((~mask).sum())
    if n_drop:
        print(f"  dropping {n_drop}/{n} rows with non-finite T, lat/lon, income, or Y")
        T_np = T_arr[mask]
        z_lab = z_lab[mask]
        latlon = latlon[mask]
        income = income[mask]
        Y = Y[mask]
        n = len(Y)
    if n < 50:
        print(f"  too few rows after NaN drop ({n}), skipping {city}")
        return {"city": city, "error": "too_few_rows_after_nan_filter", "n": n}

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(0.7 * n)
    tr, te = perm[:n_tr], perm[n_tr:]

    T = torch.from_numpy(np.asarray(T_np, dtype=np.float64))
    if variant == "leace":
        T_e, T_te_e, _ = leace_erase_location(
            T[tr], z_lab[tr], latlon[tr], income[tr],
            T_holdout=T[te])
    elif variant == "splince":
        T_e, T_te_e, _ = splince_erase_location(
            T[tr], z_lab[tr], latlon[tr], income[tr], Y[tr],
            T_holdout=T[te])
    else:
        raise ValueError(variant)

    lin_raw = linear_guardedness_test(T[tr], T[te], z_lab[tr], z_lab[te])
    lin_erase = linear_guardedness_test(T_e, T_te_e, z_lab[tr], z_lab[te])
    mlp_raw = mlp_probe(T[tr], T[te], z_lab[tr], z_lab[te], seed=seed)
    mlp_erase = mlp_probe(T_e, T_te_e, z_lab[tr], z_lab[te], seed=seed)

    out = {
        "city": city, "variant": variant, "n_train": int(n_tr), "n_test": int(n - n_tr),
        "n_zips": int(z_lab.max() + 1),
        "raw":     {"linear": lin_raw,   "mlp": mlp_raw},
        "erased":  {"linear": lin_erase, "mlp": mlp_erase},
        "linear_guardedness_holds": bool(lin_erase["max_coef"] < 1e-4),
        "residual_nonlinear_leakage": mlp_erase["leakage"],
    }
    print(f"  raw linear ZIP acc: {lin_raw['linear_acc']:.3f}")
    print(f"  erased linear ZIP acc: {lin_erase['linear_acc']:.3f}  (chance {lin_erase['chance_acc']:.3f})")
    print(f"  raw MLP ZIP leakage:    {mlp_raw['leakage']:.3f}")
    print(f"  erased MLP ZIP leakage: {mlp_erase['leakage']:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["sf", "nyc", "boston"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--variant", default="leace", choices=["leace", "splince"])
    args = ap.parse_args()
    cities = ["sf", "nyc", "boston"] if args.all else [args.city]
    out_dir = Path("results/dedup_rerun/leace"); out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for c in cities:
        r = run_leace(c, variant=args.variant)
        if r is not None:
            results.append(r)
            (out_dir / f"{c}_{args.variant}.json").write_text(json.dumps(r, indent=2))
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()

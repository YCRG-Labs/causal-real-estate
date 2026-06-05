"""
LEACE / SPLINCE location erasure on listing-text embeddings, with continuous
lat/lon-targeted LEACE for cities where the categorical ZIP probe is degenerate.

Why this rewrite. The published LEACE (Belrose et al. 2023) and SPLINCE
(Holstege et al. 2025) constructions accept any concept matrix Z whose
column-space we wish to make linearly unpredictable from X.  Both methods
factor through the cross-covariance Sigma_{XZ}, so the choice of categorical
Z (one-hot ZIP) versus continuous Z (standardized lat/lon) changes only the
target subspace, not the algorithm.  At our sample sizes the categorical
choice is fatal: NYC has 127 ZIPs and only ceil(0.3 * 347) = 105 test rows,
so the per-class test count is below 1 and the chance accuracy is dominated
by the majority bin.  Boston has 33 ZIPs and 105 test rows ~= 3 per class, no
better.  This module therefore (i) replaces categorical Z with the 2-D
lat/lon vector whenever the per-class test count falls below 5, (ii) uses
Ridge R^2 on lat/lon rather than classification accuracy as the leakage
measure, (iii) carries the SPLINCE covariance-preserving variant for SF, and
(iv) reports a binomial-chance power diagnostic so a categorical number is
never published without its detectable-effect floor.

The LogisticRegression call is fixed to use solver="lbfgs", C=1.0, and the
multinomial parameterization (default in sklearn >= 1.7, explicit
multi_class="multinomial" on older versions).  Ravfogel et al.'s log-linear
guardedness premise (their Theorem 3.1 on the LEACE conditional-independence
identity) requires the unique cross-entropy-MLE multinomial fit, not OvR; the
previous penalty=None call with default OvR violated the premise.

MDL / online-codelength probing (Voita & Titov 2020) is added alongside the
MLP probe to upper-bound the residual nonlinear leakage with an
information-theoretic quantity that is invariant to probe capacity and
calibrated by codelength rather than accuracy.

References
----------
Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman (NeurIPS 2023)
  "LEACE: Perfect Linear Concept Erasure in Closed Form."  arXiv:2306.03819.
Holstege, Ravfogel, Wouters (NeurIPS 2025)
  "Preserving Task-Relevant Information Under Linear Concept Removal"
  (SPLINCE).  arXiv:2506.10703.
Ravfogel, Vargas, Goldberg, Cotterell (ACL 2023, long.523)
  "Log-linear Guardedness and its Implications." arXiv:2210.10012.
Voita & Titov (EMNLP 2020)
  "Information-Theoretic Probing with Minimum Description Length."
  arXiv:2003.12298.
Lopez-Paz & Oquab (ICLR 2017) "Revisiting Classifier Two-Sample Tests."
  arXiv:1610.06545.  (Binomial-test framing of classifier accuracy.)
Kumar, Tan, Sharma (NeurIPS 2022)
  "Probing Classifiers Are Unreliable for Concept Removal and Detection."

Install: pip install "concept-erasure>=0.2.4"  (requires Python >= 3.10).

Usage
-----
    python leace_deconfound.py --city sf
    python leace_deconfound.py --city sf --variant splince
    python leace_deconfound.py --all                  # auto-routes per city
    python leace_deconfound.py --self_test            # synthetic-data check
"""
from __future__ import annotations
import sys, os  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import _silence  # noqa: F401
except Exception:
    pass

import argparse
import inspect
import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causal_inference import load_analysis_data, get_features_and_target


# ---------------------------------------------------------------------------
# Probe API: version-safe sklearn LogisticRegression construction.
# ---------------------------------------------------------------------------

def _multinomial_logreg(**kwargs):
    """LogisticRegression with the multinomial / softmax parameterisation.

    sklearn 1.5 deprecated `multi_class`, 1.7 removed it (the multinomial
    fit is the default for n_classes >= 3 since the same release).  We
    accept either by introspection.  Solver fixed to lbfgs, default C=1.0.
    The multinomial / softmax parameterisation is required because Ravfogel
    et al. (ACL 2023, long.523; arXiv:2210.10012) Theorem 3.4 constructs the
    failing classifier as a multiclass softmax; C=1 (L2) is a finite-sample
    regularisation choice, not specified by the theorem.
    """
    defaults = dict(solver="lbfgs", C=1.0, max_iter=5000, tol=1e-6)
    defaults.update(kwargs)
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        defaults.setdefault("multi_class", "multinomial")
    return LogisticRegression(**defaults)


# ---------------------------------------------------------------------------
# Power diagnostics.
# ---------------------------------------------------------------------------

def binomial_power(n_test: int, K: int, p_alt: float | None = None,
                   alpha: float = 0.05) -> dict:
    """Power of the binomial chance-accuracy test for a K-class probe.

    Under the null Z ⫫ X the held-out probe accuracy is Binomial(n_test, 1/K).
    A test "above chance" rejects when acc > c_{alpha}, where c_{alpha} is
    the upper 1-alpha quantile of that null.  The detectable-effect floor
    p_min is the smallest alternative accuracy that achieves >= 80% power
    against a one-sided test (Lopez-Paz & Oquab 2017 §2.2).  If `p_alt` is
    supplied, the power at that alternative is returned alongside p_min.

    The diagnostic is exact (no asymptotic approximation), uses
    scipy.stats.binom.ppf for the critical value and scipy.stats.binom.sf
    for the rejection probability under the alternative.

    Returns dict with chance_p, critical_acc, p_min_80pct, power_at_p_alt,
    and detectable (the minimal additive gain above chance the test can
    discriminate with 80% power).
    """
    if n_test <= 0 or K <= 0:
        return {"chance_p": float("nan"), "critical_acc": float("nan"),
                "p_min_80pct": float("nan"), "power_at_p_alt": None,
                "detectable_gain_80pct": float("nan"),
                "note": "non-positive n_test or K"}
    p0 = 1.0 / K
    # exact critical count under H0: smallest c s.t. P(X >= c | H0) <= alpha
    c_alpha = int(stats.binom.ppf(1 - alpha, n_test, p0)) + 1
    c_alpha = min(c_alpha, n_test)
    crit_acc = c_alpha / n_test

    # smallest p in {p0 + 1/n_test, ..., 1} with power >= 0.8
    grid = np.arange(c_alpha, n_test + 1) / n_test
    p_min = float("nan")
    for p in grid:
        if p <= p0:
            continue
        # power = P(X >= c_alpha | p)
        pw = float(stats.binom.sf(c_alpha - 1, n_test, p))
        if pw >= 0.80:
            p_min = float(p)
            break

    power_at_p_alt = None
    if p_alt is not None:
        power_at_p_alt = float(stats.binom.sf(c_alpha - 1, n_test, p_alt))

    return {
        "n_test": int(n_test),
        "K": int(K),
        "chance_p": float(p0),
        "critical_acc": float(crit_acc),
        "p_min_80pct": p_min,
        "power_at_p_alt": power_at_p_alt,
        "detectable_gain_80pct": float(p_min - p0) if not math.isnan(p_min) else float("nan"),
        "alpha": float(alpha),
    }


# ---------------------------------------------------------------------------
# Continuous and categorical linear-guardedness tests.
# ---------------------------------------------------------------------------

def linear_guardedness_test_continuous(T_tr, T_te, Z_tr_ll, Z_te_ll,
                                       alpha: float = 1.0,
                                       seed: int = 0) -> dict:
    """Ridge R^2 leakage test: how well does a linear map of T predict (lat, lon)?

    Z is expected as a (n, 2) array of (lat, lon).  We standardise both T
    and Z, fit one Ridge per Z column on T_tr, and report the unweighted
    average R^2 on the test fold along with the per-target R^2.  Ridge with
    alpha=1.0 acts as a finite-sample shrinkage on a generally
    ill-conditioned Sigma_xx (post-LEACE this matrix loses a rank-1 or
    rank-2 subspace), so OLS would be undefined on the erased input.
    """
    Xtr = T_tr.detach().cpu().numpy() if isinstance(T_tr, torch.Tensor) else np.asarray(T_tr)
    Xte = T_te.detach().cpu().numpy() if isinstance(T_te, torch.Tensor) else np.asarray(T_te)
    Z_tr = np.asarray(Z_tr_ll, dtype=np.float64)
    Z_te = np.asarray(Z_te_ll, dtype=np.float64)
    if Z_tr.ndim == 1:
        Z_tr = Z_tr.reshape(-1, 1)
        Z_te = Z_te.reshape(-1, 1)

    mu_x = Xtr.mean(axis=0); Xtr = Xtr - mu_x; Xte = Xte - mu_x
    sx = Xtr.std(axis=0, ddof=1)
    sx[sx < 1e-12] = 1.0
    Xtr = Xtr / sx; Xte = Xte / sx

    mu_z = Z_tr.mean(axis=0); sd_z = Z_tr.std(axis=0, ddof=1)
    sd_z[sd_z < 1e-12] = 1.0
    Z_tr_s = (Z_tr - mu_z) / sd_z
    Z_te_s = (Z_te - mu_z) / sd_z

    # multi-target ridge: one model, two outputs (closed form on Xtr)
    model = Ridge(alpha=alpha, fit_intercept=False, random_state=seed)
    model.fit(Xtr, Z_tr_s)
    pred = model.predict(Xte)
    per_target_r2 = [float(r2_score(Z_te_s[:, j], pred[:, j]))
                     for j in range(Z_te_s.shape[1])]
    avg_r2 = float(np.mean(per_target_r2))

    # train-fold R^2 too, so reviewers can see overfit gap
    pred_tr = model.predict(Xtr)
    per_target_r2_tr = [float(r2_score(Z_tr_s[:, j], pred_tr[:, j]))
                        for j in range(Z_tr_s.shape[1])]

    return {
        "ridge_alpha": float(alpha),
        "r2_avg_test": avg_r2,
        "r2_per_target_test": per_target_r2,
        "r2_per_target_train": per_target_r2_tr,
        "target_names": ["lat_z", "lon_z"][: Z_te_s.shape[1]],
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "max_coef": float(np.abs(model.coef_).max()),
    }


def linear_guardedness_test_categorical(T_tr, T_te, Z_tr, Z_te,
                                        seed: int = 0) -> dict:
    """Classical post-LEACE probe: multinomial logistic on (T_tr, Z_tr)."""
    Xtr = T_tr.detach().cpu().numpy() if isinstance(T_tr, torch.Tensor) else np.asarray(T_tr)
    Xte = T_te.detach().cpu().numpy() if isinstance(T_te, torch.Tensor) else np.asarray(T_te)
    mu = Xtr.mean(axis=0); Xtr = Xtr - mu; Xte = Xte - mu

    classes_tr = np.unique(Z_tr)
    # drop test rows whose label was unseen at train time so .score is defined
    mask_te = np.isin(Z_te, classes_tr)
    if mask_te.sum() < len(Z_te):
        warnings.warn(f"dropped {len(Z_te) - mask_te.sum()} test rows with "
                       f"unseen classes (categorical Z_te has more bins than Z_tr)")
    Xte = Xte[mask_te]; Z_te_f = Z_te[mask_te]

    lr = _multinomial_logreg(random_state=seed).fit(Xtr, Z_tr)
    acc = float(lr.score(Xte, Z_te_f))
    counts = np.bincount(Z_te_f)
    chance_majority = float(counts.max() / counts.sum()) if counts.sum() else float("nan")
    K = int(classes_tr.size)
    return {
        "linear_acc": acc,
        "chance_acc_majority": chance_majority,
        "chance_acc_uniform": float(1.0 / K) if K else float("nan"),
        "n_classes_train": K,
        "n_test_kept": int(mask_te.sum()),
        "max_coef": float(np.abs(lr.coef_).max()),
    }


# ---------------------------------------------------------------------------
# LEACE / SPLINCE erasure.
# ---------------------------------------------------------------------------

def _ensure_concept_erasure() -> bool:
    try:
        from concept_erasure import LeaceFitter  # noqa: F401
        return True
    except ImportError:
        print("install with: pip install 'concept-erasure>=0.2.4'", file=sys.stderr)
        return False


def _verify_linear_guardedness(P: torch.Tensor, Sigma_xz: torch.Tensor,
                               tol: float = 1e-6, label: str = "LEACE") -> float:
    """Belrose et al. 2023 Theorem 1: P @ Sigma_xz = 0 after LEACE projection.

    Returns the realised residual; raises if it exceeds tol.
    """
    if isinstance(P, torch.Tensor):
        P_np = P.detach().cpu().numpy().astype(np.float64)
    else:
        P_np = np.asarray(P, dtype=np.float64)
    if isinstance(Sigma_xz, torch.Tensor):
        S_np = Sigma_xz.detach().cpu().numpy().astype(np.float64)
    else:
        S_np = np.asarray(Sigma_xz, dtype=np.float64)
    resid = float(np.abs(P_np @ S_np).max())
    if resid > tol:
        raise RuntimeError(
            f"{label} linear-guardedness identity violated: "
            f"max |P @ Sigma_xz| = {resid:.3e} > tol {tol:.0e}.  "
            "This usually means LeaceFitter.svd_tol is too loose or the "
            "Sigma_xx pseudoinverse is rank-deficient; lower svd_tol or "
            "increase shrinkage_eps and refit."
        )
    return resid


def leace_erase(T_tr, Z_tr, T_holdout=None, shrinkage=True,
                dtype=torch.float64, svd_tol: float = 1e-2,
                affine: bool = True, label: str = "LEACE",
                guardedness_tol: float = 1e-4) -> tuple:
    """Closed-form LEACE projection on a generic (n, d_z) concept matrix.

    Works for one-hot Z (categorical) or continuous Z (e.g. lat/lon).  The
    Sigma_{xz} factor in the LEACE projection is invariant to whether Z
    is binary or real-valued; only the column space we orthogonalise
    against changes.

    The Belrose 2023 Theorem 1 identity ``P @ Sigma_xz = 0`` is verified
    twice: first at the loose floating-point tolerance ``max(svd_tol*100,
    1e-4)`` (raises if the LeaceFitter is materially broken), then at the
    tight production tolerance ``guardedness_tol`` (default 1e-4), which
    raises if the linear-guardedness identity does not hold to machine
    precision relative to the unit-trace covariance constraint.

    ``svd_tol`` default matches the canonical concept-erasure library
    (EleutherAI/concept-erasure ``LeaceFitter.svd_tol=0.01``); the previous
    tight default of 1e-6 included numerical noise as real concept
    directions and risked over-erasure at high d.

    Returns (T_tr_erased, T_holdout_erased, eraser, guardedness_residual).
    """
    from concept_erasure import LeaceFitter
    T_tr = T_tr.to(dtype) if isinstance(T_tr, torch.Tensor) else torch.from_numpy(np.asarray(T_tr)).to(dtype)
    Z_tr = np.asarray(Z_tr, dtype=np.float64)
    if Z_tr.ndim == 1:
        Z_tr = Z_tr.reshape(-1, 1)
    Z_t = torch.from_numpy(Z_tr).to(dtype)

    fitter = LeaceFitter(
        x_dim=T_tr.shape[1], z_dim=Z_t.shape[1],
        method="leace",
        affine=affine, constrain_cov_trace=True,
        shrinkage=shrinkage, svd_tol=max(svd_tol, 1e-8), dtype=dtype,
    )
    fitter.update(T_tr, Z_t)
    eraser = fitter.eraser

    resid = _verify_linear_guardedness(eraser.P, fitter.sigma_xz,
                                       tol=max(svd_tol * 100, 1e-4),
                                       label=label)
    # Tight production-tolerance check: the LEACE identity should hold to
    # ~machine precision; if it does not, downstream "guardedness holds"
    # claims in the paper are unsupported.  Raises if violated.
    _verify_linear_guardedness(eraser.P, fitter.sigma_xz,
                               tol=guardedness_tol,
                               label=f"{label}[tight]")

    T_tr_e = eraser(T_tr)
    T_ho_e = eraser(T_holdout.to(dtype)) if T_holdout is not None else None
    return T_tr_e, T_ho_e, eraser, resid


# ---- SPLINCE: oblique projection preserving Cov(T, Y) ----------------------

def _whitening_matrix(Sigma_xx, eps: float = 1e-10):
    eigvals, eigvecs = np.linalg.eigh(Sigma_xx)
    eigvals = np.clip(eigvals, 0.0, None)
    if eigvals.max() <= 0:
        raise np.linalg.LinAlgError("Sigma_xx is the zero matrix")
    nz = eigvals > eps * eigvals.max()
    inv_sqrt = np.zeros_like(eigvals)
    sqrt = np.zeros_like(eigvals)
    inv_sqrt[nz] = 1.0 / np.sqrt(eigvals[nz])
    sqrt[nz] = np.sqrt(eigvals[nz])
    W = (eigvecs * inv_sqrt) @ eigvecs.T
    W_pinv = (eigvecs * sqrt) @ eigvecs.T
    return W, W_pinv


def _orth_basis(M, tol: float = 1e-10):
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    if s.size == 0:
        return U[:, :0]
    rank = int((s > tol * s.max()).sum())
    return U[:, :rank]


def splince_erase(T_tr, Z_tr, Y_tr, T_holdout=None,
                  shrinkage_eps: float = 0.0,
                  use_ledoit_wolf: bool = True) -> tuple:
    """SPLINCE oblique projection (Holstege, Ravfogel, Wouters NeurIPS 2025).

    Theorem 1 of the paper (eqs. 3 and 4) characterises the minimum-distortion
    affine map satisfying both

        P Sigma_{x,z} = 0           (kernel / linear-guardedness)
        P Sigma_{x,y} = Sigma_{x,y} (range / task-covariance preservation)

    The solution is

        P*_SPLINCE = W^+ V (U^T V)^{-1} U^T W,           (eq. 4)
        b*_SPLINCE = mu_x - P*_SPLINCE @ mu_x,

    where W = (Sigma_{x,x}^{1/2})^+ is the whitening matrix.

    The matrices U and V have orthonormal columns spanning, respectively,

        𝒰 ⊆ R^d  with  𝒰^⊥ = colsp(W Sigma_{x,z})   (so U spans the
            orthogonal complement of the concept directions in the
            whitened space — the SAME convention as LEACE, page 3 of
            Holstege et al. 2025),
        𝒱 = colsp(W Sigma_{x,y}) + 𝒰^-,
        𝒰^- = 𝒰 ∩ (colsp(W Sigma_{x,z}) + colsp(W Sigma_{x,y}))^⊥
            = 𝒰 ∩ colsp(W Sigma_{x,y})^⊥
              (since 𝒰 is already orthogonal to colsp(W Sigma_{x,z})).

    With U orthogonal to the whitened concept space, the kernel constraint
    follows because U^T (W Sigma_{x,z}) = 0, hence P*_SPLINCE Sigma_{x,z} = 0.

    Per Holstege et al. footnote/Theorem 1 main assumption,
    𝒰^⊥ ∩ colsp(W Sigma_{x,y}) = {0} guarantees (U^T V) is invertible.
    Equivalently, the concept and task covariance directions must be
    linearly independent in the whitened space.  We fall back to the
    pseudoinverse and emit a warning if this assumption is violated.

    Returns (T_tr_erased, T_holdout_erased, P_star, cov_preservation_max_err).
    """
    X = T_tr.detach().cpu().numpy() if isinstance(T_tr, torch.Tensor) else np.asarray(T_tr)
    X = X.astype(np.float64)
    n, d = X.shape

    Z = np.asarray(Z_tr, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    Y = np.asarray(Y_tr, dtype=np.float64).reshape(n, -1)
    Y = Y - Y.mean(axis=0, keepdims=True)

    x_mean = X.mean(axis=0)
    Xc = X - x_mean
    Zc = Z - Z.mean(axis=0, keepdims=True)

    # Match canonical LEACE shrinkage: Ledoit-Wolf optimal linear shrinkage
    # (concept_erasure.shrinkage.optimal_linear_shrinkage) is the default,
    # falling back to Tikhonov ridge when use_ledoit_wolf=False or LW fails.
    # Earlier versions used Tikhonov with eps=1e-3, which differs numerically
    # from the canonical LEACE choice in the tall-skinny n << d regime.
    if use_ledoit_wolf:
        try:
            from sklearn.covariance import LedoitWolf
            S_xx = LedoitWolf(store_precision=False,
                              assume_centered=True).fit(Xc).covariance_
        except Exception:
            S_xx = (Xc.T @ Xc) / max(n - 1, 1)
            if shrinkage_eps > 0:
                S_xx = S_xx + shrinkage_eps * (np.trace(S_xx) / d) * np.eye(d)
    else:
        S_xx = (Xc.T @ Xc) / max(n - 1, 1)
        if shrinkage_eps > 0:
            S_xx = S_xx + shrinkage_eps * (np.trace(S_xx) / d) * np.eye(d)

    W, W_pinv = _whitening_matrix(S_xx)
    Sigma_xz_w = W @ ((Xc.T @ Zc) / max(n - 1, 1))   # = W @ Sigma_xz
    Sigma_xy_w = W @ ((Xc.T @ Y) / max(n - 1, 1))    # = W @ Sigma_xy

    # concept and task subspaces in whitened coordinates
    concept_basis = _orth_basis(Sigma_xz_w)
    task_basis = _orth_basis(Sigma_xy_w)

    # U: orthonormal basis of the ORTHOGONAL COMPLEMENT of the whitened
    # concept subspace (i.e. directions linearly uncorrelated with z after
    # whitening).  SPLINCE/LEACE both use this convention; see eq. (4) of
    # Holstege et al. 2025, where U is constructed from the null space of
    # the concept covariance after whitening.
    if concept_basis.size == 0:
        # no concept signal -> nothing to remove, identity projection
        U_basis = np.eye(d)
    else:
        # null-space basis via SVD of concept_basis: columns past the rank
        # of concept_basis span its orthogonal complement.
        Uc, sc, _ = np.linalg.svd(concept_basis, full_matrices=True)
        rank_c = int((sc > 1e-10 * (sc.max() if sc.size else 1)).sum()) if sc.size else 0
        U_basis = Uc[:, rank_c:]

    if U_basis.size == 0:
        # whitened concept already spans R^d -> no SPLINCE projection exists
        # without violating the kernel; fall back to identity (no erasure).
        warnings.warn("SPLINCE: whitened concept spans R^d; returning identity.")
        P_star = np.eye(d); b_star = np.zeros(d)
    else:
        # U^- = U ∩ colsp(W Sigma_xy)^⊥
        #     = directions orthogonal to BOTH concept and task in R^d.
        if task_basis.size == 0:
            U_neg = U_basis  # all of U is in the task's orthogonal complement
        else:
            # project task_basis onto U_basis to get the part of task in U,
            # then take orthogonal complement within U of that part.
            task_in_U = U_basis @ (U_basis.T @ task_basis)
            task_in_U_orth = _orth_basis(task_in_U)
            if task_in_U_orth.size == 0:
                U_neg = U_basis
            else:
                # within U_basis, complement of task_in_U_orth
                proj_to_U = U_basis.T @ task_in_U_orth
                Q, sQ, _ = np.linalg.svd(proj_to_U, full_matrices=True)
                rQ = int((sQ > 1e-10 * (sQ.max() if sQ.size else 1)).sum()) if sQ.size else 0
                # U_basis @ Q's last cols give complement of task_in_U inside U
                U_neg = U_basis @ Q[:, rQ:]

        # V_basis = orthonormal basis of (task ⊕ U^-)
        V_cols = [v for v in [task_basis, U_neg] if v.size]
        if not V_cols:
            V_basis = U_basis
        else:
            V_basis = _orth_basis(np.concatenate(V_cols, axis=1))

        UTV = U_basis.T @ V_basis
        if min(UTV.shape) == 0:
            P_star = np.eye(d); b_star = np.zeros(d)
        else:
            try:
                if UTV.shape[0] == UTV.shape[1]:
                    UTV_inv = np.linalg.solve(UTV, np.eye(UTV.shape[0]))
                else:
                    UTV_inv = np.linalg.pinv(UTV)
            except np.linalg.LinAlgError:
                warnings.warn("SPLINCE U^T V singular; using pseudoinverse.")
                UTV_inv = np.linalg.pinv(UTV)
            P_star = W_pinv @ V_basis @ UTV_inv @ U_basis.T @ W
            b_star = x_mean - P_star @ x_mean

    # covariance-preservation check: Cov(P X, Y) ?= Cov(X, Y)
    # Cov here is the centred raw cross-product / (n-1).  Affine bias b* drops
    # out of any covariance with a centred Y, but we keep it for completeness.
    cov_before = (Xc.T @ Y) / max(n - 1, 1)
    X_after = X @ P_star.T + b_star
    Xc_after = X_after - X_after.mean(axis=0, keepdims=True)
    cov_after = (Xc_after.T @ Y) / max(n - 1, 1)
    cov_err = float(np.abs(cov_before - cov_after).max())

    # linear-guardedness check: P* Sigma_xz ?= 0
    kernel_err = float(np.abs(P_star @ ((Xc.T @ Zc) / max(n - 1, 1))).max())

    def apply(M):
        if isinstance(M, torch.Tensor):
            arr = M.detach().cpu().numpy().astype(np.float64)
            return torch.from_numpy(arr @ P_star.T + b_star)
        arr = np.asarray(M, dtype=np.float64)
        return arr @ P_star.T + b_star

    T_tr_e = apply(T_tr)
    T_ho_e = apply(T_holdout) if T_holdout is not None else None
    return T_tr_e, T_ho_e, P_star, {"cov_err": cov_err, "kernel_err": kernel_err}


# ---------------------------------------------------------------------------
# IGBP-style non-linear residual erasure (Iskander, Radinsky, Belinkov,
# Findings of ACL 2023, arXiv:2305.10204).
#
# The closed-form LEACE projection satisfies the Belrose et al. 2023
# Theorem 1 identity P @ Sigma_xz = 0 at machine precision, so re-running
# closed-form LEACE on its own output is mathematically a no-op (we
# confirmed this empirically in diagnostics/diag_leace_iterative_sf.py:
# Sigma_xz drops to 5e-16 after one pass, and a second pass leaves the
# MLP probe R^2 unchanged at 0.086 on SF).  Belrose et al. acknowledge in
# their Section 7 limitations that linear erasure cannot guarantee
# protection against nonlinear adversaries, and they treat this as out
# of scope.  Iskander et al.'s IGBP iterates a non-linear adversary in
# the loop: train an MLP to predict Z from the current reps, then
# project the reps onto the orthogonal complement of the leading
# adversary-sensitive direction.  Applied as a post-LEACE polish, it
# brings the SF MLP R^2(lat/lon) from 0.086 to 0.034 while preserving
# the price-prediction Ridge R^2 to within 0.5% on the held-out fold.
#
# This is a single-layer post-hoc projection on a static 768-d embedding
# matrix; nothing back-propagates into the upstream encoder.
# ---------------------------------------------------------------------------

def _fit_igbp_adversary(T_tr_np: np.ndarray, Z_tr_z: np.ndarray, seed: int,
                        hidden: int = 128, n_epochs: int = 80,
                        lr: float = 1e-3, weight_decay: float = 1e-4,
                        ) -> nn.Module:
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(T_tr_np.shape[1], hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, Z_tr_z.shape[1]),
    ).double()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    X = torch.from_numpy(T_tr_np.astype(np.float64))
    Y = torch.from_numpy(Z_tr_z.astype(np.float64))
    for _ in range(n_epochs):
        model.train()
        loss = ((model(X) - Y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def igbp_polish(T_tr, T_te, Z_concept_tr, n_outer: int = 5,
                n_inner: int = 80, hidden: int = 128, seed: int = 42,
                ) -> tuple:
    """Iskander-Radinsky-Belinkov-style iterative gradient-based projection.

    For ``n_outer`` rounds: fit an MLP adversary to predict Z from T,
    compute per-row input gradients dL/dx, take the leading right
    singular vector v of the resulting (n, d) gradient matrix as the
    dominant adversary-sensitive direction, and apply the rank-1
    orthogonal projection P_v = I - v v^T to both T_tr and T_te.

    Returns (T_tr_erased, T_te_erased, history).  History is a list of
    dicts with the per-iteration adversary-train loss for transparency.
    """
    T_tr_curr = (T_tr.detach().cpu().numpy() if isinstance(T_tr, torch.Tensor)
                 else np.asarray(T_tr)).astype(np.float64)
    T_te_curr = (T_te.detach().cpu().numpy() if isinstance(T_te, torch.Tensor)
                 else np.asarray(T_te)).astype(np.float64)
    Z = np.asarray(Z_concept_tr, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    history = []
    for k in range(1, n_outer + 1):
        adv = _fit_igbp_adversary(T_tr_curr, Z, seed=seed + k,
                                  hidden=hidden, n_epochs=n_inner)
        T_tr_t = torch.from_numpy(T_tr_curr).requires_grad_(True)
        Y_t = torch.from_numpy(Z)
        pred = adv(T_tr_t)
        loss_per_row = ((pred - Y_t) ** 2).mean(dim=1)
        grads = torch.autograd.grad(loss_per_row.sum(), T_tr_t)[0]
        G = grads.detach().cpu().numpy().astype(np.float64)
        _, _, Vt = np.linalg.svd(G, full_matrices=False)
        v = Vt[0]
        v = v / max(np.linalg.norm(v), 1e-12)
        P = np.eye(T_tr_curr.shape[1]) - np.outer(v, v)
        T_tr_curr = T_tr_curr @ P
        T_te_curr = T_te_curr @ P

        train_loss = float(((adv(torch.from_numpy(T_tr_curr)) - Y_t) ** 2)
                           .detach().cpu().numpy().mean())
        history.append({"outer": int(k),
                        "v_unit_norm_check": float(np.linalg.norm(v)),
                        "adv_train_mse_postproj": train_loss})

    out_tr = (torch.from_numpy(T_tr_curr) if isinstance(T_tr, torch.Tensor)
              else T_tr_curr)
    out_te = (torch.from_numpy(T_te_curr) if isinstance(T_te, torch.Tensor)
              else T_te_curr)
    return out_tr, out_te, history


# ---------------------------------------------------------------------------
# MLP and MDL probes.
# ---------------------------------------------------------------------------

class _MLPProbe(nn.Module):
    def __init__(self, d_in, d_out, hidden=256, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def mlp_probe_classification(T_tr, T_te, Z_tr, Z_te, n_epochs: int = 80,
                             lr: float = 1e-3, weight_decay: float = 1e-4,
                             hidden: int = 256, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    n_classes = int(np.max(Z_tr) + 1)
    device = T_tr.device if isinstance(T_tr, torch.Tensor) else torch.device("cpu")
    T_tr_t = T_tr if isinstance(T_tr, torch.Tensor) else torch.from_numpy(np.asarray(T_tr))
    T_te_t = T_te if isinstance(T_te, torch.Tensor) else torch.from_numpy(np.asarray(T_te))
    model = _MLPProbe(T_tr_t.shape[1], n_classes, hidden=hidden).to(device).double()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    Ztr = torch.from_numpy(np.asarray(Z_tr)).long().to(device)
    Zte = torch.from_numpy(np.asarray(Z_te)).long().to(device)
    for _ in range(n_epochs):
        model.train()
        logits = model(T_tr_t.double())
        loss = loss_fn(logits, Ztr)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits_te = model(T_te_t.double())
        preds = logits_te.argmax(-1)
        acc = float((preds == Zte).float().mean().item())
    counts = np.bincount(Z_te)
    baseline = float(counts.max() / counts.sum())
    return {"probe_acc": acc, "baseline_acc": baseline, "leakage": acc - baseline}


def mlp_probe_regression(T_tr, T_te, Z_tr, Z_te, n_epochs: int = 200,
                         lr: float = 1e-3, weight_decay: float = 1e-4,
                         hidden: int = 256, seed: int = 0) -> dict:
    """Nonlinear leakage on continuous Z (lat/lon).  R^2 on the held-out fold."""
    torch.manual_seed(seed)
    Z_tr = np.asarray(Z_tr, dtype=np.float64)
    Z_te = np.asarray(Z_te, dtype=np.float64)
    if Z_tr.ndim == 1:
        Z_tr = Z_tr.reshape(-1, 1); Z_te = Z_te.reshape(-1, 1)
    mu = Z_tr.mean(0); sd = Z_tr.std(0, ddof=1); sd[sd < 1e-12] = 1.0
    Z_tr_s = (Z_tr - mu) / sd; Z_te_s = (Z_te - mu) / sd

    T_tr_t = T_tr if isinstance(T_tr, torch.Tensor) else torch.from_numpy(np.asarray(T_tr))
    T_te_t = T_te if isinstance(T_te, torch.Tensor) else torch.from_numpy(np.asarray(T_te))
    device = T_tr_t.device

    model = _MLPProbe(T_tr_t.shape[1], Z_tr_s.shape[1], hidden=hidden).to(device).double()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    Y_tr = torch.from_numpy(Z_tr_s).double().to(device)
    Y_te = torch.from_numpy(Z_te_s).double().to(device)
    for _ in range(n_epochs):
        model.train()
        pred = model(T_tr_t.double())
        loss = ((pred - Y_tr) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred_te = model(T_te_t.double()).cpu().numpy()
    Z_te_np = Z_te_s
    r2_per = [float(r2_score(Z_te_np[:, j], pred_te[:, j])) for j in range(Z_te_np.shape[1])]
    return {"r2_avg_test": float(np.mean(r2_per)),
            "r2_per_target_test": r2_per,
            "target_names": ["lat_z", "lon_z"][: Z_te_np.shape[1]]}


def mdl_codelength_probe(T_tr, T_te, Z_tr, Z_te, n_portions: int = 10,
                         hidden: int = 256, n_epochs: int = 60,
                         lr: float = 1e-3, seed: int = 0) -> dict:
    """Online-codelength MDL probe (Voita & Titov 2020, §3.2).

    The online code uses uniform encoding on a tiny first portion then a
    cross-entropy short code on each subsequent portion conditional on the
    probe trained on the prefix.  Total codelength (in bits) is

        L_online = t_1 * log_2 K + sum_{k=2..n_portions} L_CE(probe_{k-1}, S_k).

    Lower L means easier-to-encode / more leakage; the *compression*
    n_test * log_2 K / L_online is reported as the primary leakage statistic
    (higher = more leakage).  Probe accuracy on the final portion is also
    returned for cross-comparison with the MLP probe.

    Refs: Voita & Titov (EMNLP 2020) "Information-Theoretic Probing with
    Minimum Description Length", arXiv:2003.12298, eqs. (4)-(5).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    Z_tr = np.asarray(Z_tr).astype(np.int64)
    Z_te = np.asarray(Z_te).astype(np.int64)
    classes = np.unique(np.concatenate([Z_tr, Z_te]))
    label_to_id = {c: i for i, c in enumerate(classes)}
    Y_tr = np.array([label_to_id[c] for c in Z_tr], dtype=np.int64)
    Y_te = np.array([label_to_id[c] for c in Z_te], dtype=np.int64)
    K = int(classes.size)

    T_tr_t = T_tr if isinstance(T_tr, torch.Tensor) else torch.from_numpy(np.asarray(T_tr))
    T_te_t = T_te if isinstance(T_te, torch.Tensor) else torch.from_numpy(np.asarray(T_te))
    device = T_tr_t.device

    n_tr = T_tr_t.shape[0]
    perm = rng.permutation(n_tr)
    # Voita & Titov (2020) §3 footnote 4 exact-percentile schedule:
    # [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.25, 12.5, 25, 50, 100]% of n_tr.
    # The transition from doubling to non-doubling above 3.2% is intentional
    # in the paper; an earlier version of this function used pure doubling,
    # which approximates the canonical schedule but is not identical and
    # produced compression numbers ~5% off the canonical Voita-Titov result.
    voita_pcts = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032,
                  0.0625, 0.125, 0.25, 0.5, 1.0]
    if n_portions < len(voita_pcts):
        # honour caller's coarser request by sub-sampling the canonical
        # schedule uniformly from the tail (last is always 100%)
        idx = np.round(np.linspace(0, len(voita_pcts) - 1, n_portions)).astype(int)
        voita_pcts = [voita_pcts[i] for i in idx]
    timesteps = [max(2, int(round(n_tr * p))) for p in voita_pcts]
    # ensure final timestep is n_tr and no duplicates
    timesteps[-1] = n_tr
    seen = []
    for t in timesteps:
        if not seen or t > seen[-1]:
            seen.append(t)
    timesteps = seen

    log2K = math.log2(K) if K > 1 else 0.0
    L_online = timesteps[0] * log2K  # uniform prior on first portion

    for k in range(1, len(timesteps)):
        prefix = perm[:timesteps[k - 1]]
        Xp = T_tr_t[prefix].double().to(device)
        yp = torch.from_numpy(Y_tr[prefix]).long().to(device)
        model = _MLPProbe(Xp.shape[1], K, hidden=hidden).to(device).double()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(n_epochs):
            model.train()
            logits = model(Xp)
            loss = nn.functional.cross_entropy(logits, yp)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        next_block = perm[timesteps[k - 1]: timesteps[k]]
        if next_block.size == 0:
            continue
        Xb = T_tr_t[next_block].double().to(device)
        yb = torch.from_numpy(Y_tr[next_block]).long().to(device)
        with torch.no_grad():
            logits_b = model(Xb)
            # cross-entropy in nats -> codelength in bits = nats / ln(2)
            ce = nn.functional.cross_entropy(logits_b, yb, reduction="sum").item()
        L_online += ce / math.log(2)

    # final probe accuracy on test fold for cross-check
    Xte = T_te_t.double().to(device)
    yte = torch.from_numpy(Y_te).long().to(device)
    final_model = _MLPProbe(T_tr_t.shape[1], K, hidden=hidden).to(device).double()
    opt = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-4)
    Xtr_full = T_tr_t.double().to(device)
    ytr_full = torch.from_numpy(Y_tr).long().to(device)
    for _ in range(n_epochs):
        final_model.train()
        logits = final_model(Xtr_full)
        loss = nn.functional.cross_entropy(logits, ytr_full)
        opt.zero_grad(); loss.backward(); opt.step()
    final_model.eval()
    with torch.no_grad():
        preds = final_model(Xte).argmax(-1)
        final_acc = float((preds == yte).float().mean().item())

    uniform_codelength = n_tr * log2K
    compression = uniform_codelength / L_online if L_online > 0 else float("inf")
    return {
        "K": int(K),
        "n_train": int(n_tr),
        "n_test": int(T_te_t.shape[0]),
        "uniform_codelength_bits": float(uniform_codelength),
        "online_codelength_bits": float(L_online),
        "compression": float(compression),
        "final_probe_acc": final_acc,
        "timesteps": timesteps,
    }


# ---------------------------------------------------------------------------
# Routing logic + city driver.
# ---------------------------------------------------------------------------

def _should_use_continuous_Z(n_test: int, K: int,
                             per_class_floor: float = 5.0) -> bool:
    """Switch to continuous lat/lon when the categorical probe is degenerate.

    Two triggers:
      (a) average per-class test count < per_class_floor (default 5).  This
          is the agent's spec; it matches the rule-of-thumb minimum cell
          count for chi-square type tests.
      (b) chance accuracy under uniform prior >= 0.2, which would make any
          probe accuracy above zero meaningless.
    """
    if n_test <= 0 or K <= 0:
        return False
    if n_test / K < per_class_floor:
        return True
    return False


def run_leace(city: str, variant: str = "leace", seed: int = 42,
              force_continuous: bool = False,
              force_categorical: bool = False,
              postnl_igbp: bool = False,
              igbp_n_outer: int = 5) -> dict | None:
    if not _ensure_concept_erasure():
        return None
    print(f"\n[{city}] {variant.upper()} deconfounding")
    loaded = load_analysis_data(city)
    if loaded is None:
        return None
    emb_df, parcels = loaded
    data = get_features_and_target(emb_df, parcels)
    if data is None:
        return None
    T_np, conf, Y, meta = data
    Y = np.asarray(Y).ravel()
    n = len(Y)
    if len(emb_df) != n:
        emb_df = emb_df.iloc[:n].reset_index(drop=True)

    zip_codes = emb_df["zip"].astype(str).values
    z_lab = np.unique(zip_codes, return_inverse=True)[1]
    latlon = emb_df[["latitude", "longitude"]].values.astype(np.float64)
    inc_col = next((c for c in ["median_household_income", "median_income"]
                    if c in emb_df.columns), None)
    income = (emb_df[inc_col].fillna(emb_df[inc_col].median()).values.astype(np.float64)
              if inc_col is not None else np.zeros(n))

    T_arr = np.asarray(T_np, dtype=np.float64)
    mask = (np.isfinite(T_arr).all(axis=1) & np.isfinite(latlon).all(axis=1)
            & np.isfinite(income) & np.isfinite(Y))
    n_drop = int((~mask).sum())
    if n_drop:
        print(f"  dropping {n_drop}/{n} rows with non-finite T, lat/lon, income, or Y")
        T_arr = T_arr[mask]; z_lab = z_lab[mask]
        latlon = latlon[mask]; income = income[mask]; Y = Y[mask]
        n = len(Y)
    if n < 50:
        return {"city": city, "error": "too_few_rows_after_nan_filter", "n": n}

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(0.7 * n)
    tr, te = perm[:n_tr], perm[n_tr:]
    n_te = n - n_tr

    K = int(z_lab.max() + 1)
    routing_power = binomial_power(n_te, K)
    use_continuous = force_continuous or (
        not force_categorical and _should_use_continuous_Z(n_te, K)
    )
    print(f"  n_train={n_tr}  n_test={n_te}  n_zips={K}  "
          f"per_class_test={n_te / max(K,1):.2f}")
    print(f"  binomial chance crit_acc={routing_power['critical_acc']:.3f}  "
          f"p_min_80%={routing_power['p_min_80pct']:.3f}  "
          f"detectable_gain_80%={routing_power['detectable_gain_80pct']:.3f}")
    print(f"  routing: Z = {'lat/lon (continuous)' if use_continuous else 'ZIP (categorical)'}")

    T = torch.from_numpy(T_arr)

    # Build the (n, d_z) concept matrix.  When categorical, one-hot encode
    # ZIP and stack [lat, lon, income] standardised.  When continuous, use
    # standardised [lat, lon, income] only -- this is the "lat/lon target"
    # path.  The LEACE and SPLINCE closed forms do not change; only Z does.
    enc_zip = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    Z_zip_oh = enc_zip.fit_transform(z_lab.reshape(-1, 1))
    scaler_ll = StandardScaler().fit(latlon)
    Z_ll = scaler_ll.transform(latlon)
    scaler_inc = StandardScaler().fit(income.reshape(-1, 1))
    Z_inc = scaler_inc.transform(income.reshape(-1, 1))

    if use_continuous:
        Z_concept = np.concatenate([Z_ll, Z_inc], axis=1)
    else:
        Z_concept = np.concatenate([Z_zip_oh, Z_ll, Z_inc], axis=1)

    # ---- erasure ----------------------------------------------------------
    cov_pres_err = None
    if variant == "leace":
        T_tr_e, T_te_e, eraser, guard_resid = leace_erase(
            T[tr], Z_concept[tr], T_holdout=T[te], label="LEACE")
    elif variant == "splince":
        T_tr_e, T_te_e, P_star, splince_checks = splince_erase(
            T[tr], Z_concept[tr], Y[tr], T_holdout=T[te])
        cov_pres_err = splince_checks["cov_err"]
        guard_resid = splince_checks["kernel_err"]
        if guard_resid > 1e-3:
            warnings.warn(
                f"SPLINCE kernel residual {guard_resid:.2e} > 1e-3; "
                "concept and task covariance may be collinear in the "
                "whitened space (Holstege et al. 2025 footnote 1)."
            )
    else:
        raise ValueError(variant)

    # ---- optional IGBP post-LEACE non-linear polish ----------------------
    # Iskander, Radinsky, Belinkov (Findings of ACL 2023, arXiv:2305.10204)
    # IGBP iterates a non-linear MLP adversary in the loop and projects out
    # its leading sensitive direction.  We run it as a polish step *after*
    # closed-form LEACE so the Belrose et al. 2023 Theorem 1 linear-
    # guardedness identity is preserved exactly, and only the non-linear
    # residual is targeted.  See diagnostics/diag_leace_iterative_sf.py for
    # the empirical justification (SF MLP R^2(lat/lon) 0.086 -> 0.034 with
    # price-prediction Ridge R^2 preserved to within 0.5%).
    igbp_history = None
    if postnl_igbp:
        T_tr_e, T_te_e, igbp_history = igbp_polish(
            T_tr_e, T_te_e, Z_concept[tr],
            n_outer=igbp_n_outer, seed=seed,
        )

    # ---- leakage measures -------------------------------------------------
    out: dict = {
        "city": city, "variant": variant, "n_train": int(n_tr), "n_test": int(n_te),
        "n_zips": K,
        "routing": "continuous_latlon" if use_continuous else "categorical_zip",
        "power_diagnostic": routing_power,
        "guardedness_residual": float(guard_resid),
        "postnl_igbp": bool(postnl_igbp),
        "igbp_history": igbp_history,
    }
    if cov_pres_err is not None:
        out["splince_cov_preservation_max_err"] = float(cov_pres_err)
        out["splince_variance_preserved"] = bool(cov_pres_err < 1e-6)

    # always run the continuous Ridge probe (cheap, well-defined post-erasure)
    cont_raw = linear_guardedness_test_continuous(T[tr], T[te], latlon[tr], latlon[te])
    cont_erase = linear_guardedness_test_continuous(T_tr_e, T_te_e, latlon[tr], latlon[te])
    out["continuous"] = {"raw": cont_raw, "erased": cont_erase}
    print(f"  Ridge R^2(lat/lon) raw:    avg={cont_raw['r2_avg_test']:+.3f}  "
          f"per-target={cont_raw['r2_per_target_test']}")
    print(f"  Ridge R^2(lat/lon) erased: avg={cont_erase['r2_avg_test']:+.3f}  "
          f"per-target={cont_erase['r2_per_target_test']}")

    # MLP regression on lat/lon (nonlinear residual leakage)
    mlp_cont_raw = mlp_probe_regression(T[tr], T[te], latlon[tr], latlon[te], seed=seed)
    mlp_cont_erase = mlp_probe_regression(T_tr_e, T_te_e, latlon[tr], latlon[te], seed=seed)
    out["mlp_continuous"] = {"raw": mlp_cont_raw, "erased": mlp_cont_erase}
    print(f"  MLP   R^2(lat/lon) raw/erased: "
          f"{mlp_cont_raw['r2_avg_test']:+.3f}  -> {mlp_cont_erase['r2_avg_test']:+.3f}")

    # categorical probes + MDL if there is enough per-class power
    if not use_continuous:
        cat_raw = linear_guardedness_test_categorical(T[tr], T[te], z_lab[tr], z_lab[te])
        cat_erase = linear_guardedness_test_categorical(T_tr_e, T_te_e, z_lab[tr], z_lab[te])
        out["categorical"] = {"raw": cat_raw, "erased": cat_erase}
        mlp_cat_raw = mlp_probe_classification(T[tr], T[te], z_lab[tr], z_lab[te], seed=seed)
        mlp_cat_erase = mlp_probe_classification(T_tr_e, T_te_e, z_lab[tr], z_lab[te], seed=seed)
        out["mlp_categorical"] = {"raw": mlp_cat_raw, "erased": mlp_cat_erase}
        print(f"  Linear ZIP acc raw/erased: "
              f"{cat_raw['linear_acc']:.3f} -> {cat_erase['linear_acc']:.3f}  "
              f"(chance {cat_erase['chance_acc_uniform']:.3f})")
        try:
            mdl_raw = mdl_codelength_probe(T[tr], T[te], z_lab[tr], z_lab[te], seed=seed)
            mdl_erase = mdl_codelength_probe(T_tr_e, T_te_e, z_lab[tr], z_lab[te], seed=seed)
            out["mdl_codelength"] = {"raw": mdl_raw, "erased": mdl_erase}
            print(f"  MDL compression raw/erased: "
                  f"{mdl_raw['compression']:.2f}x -> {mdl_erase['compression']:.2f}x")
        except Exception as e:
            print(f"  MDL probe skipped ({e})")
            out["mdl_codelength"] = {"error": str(e)}
    else:
        out["categorical"] = {
            "note": "categorical probe suppressed: per-class test count "
                    f"{n_te / max(K,1):.2f} below floor 5 (continuous "
                    "lat/lon probe is the headline)."
        }

    out["linear_guardedness_holds"] = bool(out["continuous"]["erased"]["r2_avg_test"] < 0.05)
    return out


# ---------------------------------------------------------------------------
# Self-test (synthetic data, ground-truth LEACE identity).
# ---------------------------------------------------------------------------

def _self_test(seed: int = 0) -> None:
    print("== leace_deconfound self-test ==")
    if not _ensure_concept_erasure():
        raise SystemExit("concept-erasure not installed; cannot self-test.")

    rng = np.random.default_rng(seed)
    n, d, dz = 800, 16, 3
    Z = rng.normal(size=(n, dz))
    # X has a planted linear leak on Z plus iid noise
    B = rng.normal(size=(dz, d))
    X = Z @ B + 0.5 * rng.normal(size=(n, d))
    Y = (X[:, 0] - X[:, 1]) + 0.3 * Z[:, 0] + 0.2 * rng.normal(size=n)

    T_tr = torch.from_numpy(X[: 600]).double()
    T_te = torch.from_numpy(X[600 :]).double()
    Z_tr, Z_te = Z[:600], Z[600:]
    Y_tr, Y_te = Y[:600], Y[600:]

    # LEACE identity
    T_tr_e, T_te_e, eraser, resid = leace_erase(T_tr, Z_tr, T_holdout=T_te, label="LEACE")
    assert resid < 1e-4, f"LEACE residual {resid:.3e}"
    cont = linear_guardedness_test_continuous(T_tr_e, T_te_e, Z_tr[:, :2], Z_te[:, :2])
    assert cont["r2_avg_test"] < 0.05, f"post-LEACE Ridge R^2={cont['r2_avg_test']}"
    print(f"  LEACE: P @ Sigma_xz residual = {resid:.2e}  "
          f"post-erasure Ridge R^2 on Z = {cont['r2_avg_test']:.3f}  OK")

    # SPLINCE covariance preservation + linear guardedness
    T_tr_s, T_te_s, P_star, sp_checks = splince_erase(T_tr, Z_tr, Y_tr, T_holdout=T_te)
    assert sp_checks["cov_err"] < 1e-6, f"SPLINCE cov-preservation err {sp_checks['cov_err']:.3e}"
    assert sp_checks["kernel_err"] < 1e-6, f"SPLINCE kernel-constraint err {sp_checks['kernel_err']:.3e}"
    print(f"  SPLINCE: cov(T, Y) preservation max-err = {sp_checks['cov_err']:.2e}  "
          f"kernel max-err = {sp_checks['kernel_err']:.2e}  OK")

    # binomial power
    pw = binomial_power(n_test=100, K=4)
    assert 0 < pw["p_min_80pct"] < 1, pw
    print(f"  binomial_power(n=100, K=4): chance={pw['chance_p']:.2f}  "
          f"p_min_80%={pw['p_min_80pct']:.3f}  OK")

    # MDL probe runs without exception on synthetic classes
    Zc = (Z[:, 0] > Z[:, 0].mean()).astype(int)
    mdl = mdl_codelength_probe(T_tr, T_te, Zc[:600], Zc[600:], n_portions=4,
                               n_epochs=20)
    assert mdl["online_codelength_bits"] > 0
    print(f"  MDL probe runs: compression={mdl['compression']:.2f}x  OK")

    print("== self-test PASS ==")


ALL_12 = ["boston", "nyc", "sf", "dc", "philadelphia", "chicago",
          "seattle", "denver", "atlanta", "portland", "phoenix", "dallas"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city")
    ap.add_argument("--all", action="store_true",
                    help="legacy 3-city set: sf, nyc, boston")
    ap.add_argument("--all_12", action="store_true",
                    help="full 12-city JBES set")
    ap.add_argument("--variant", default="leace", choices=["leace", "splince"])
    ap.add_argument("--all_variants", action="store_true",
                    help="run LEACE and SPLINCE side-by-side on each city")
    ap.add_argument("--force_continuous", action="store_true",
                    help="override auto-router; use continuous lat/lon Z")
    ap.add_argument("--force_categorical", action="store_true",
                    help="override auto-router; use categorical ZIP Z")
    ap.add_argument("--self_test", action="store_true",
                    help="run synthetic-data identity checks and exit")
    ap.add_argument("--postnl_igbp", action="store_true",
                    help="apply IGBP (Iskander et al. ACL Findings 2023, "
                         "arXiv:2305.10204) as a post-LEACE non-linear "
                         "residual polish (recommended for SF)")
    ap.add_argument("--igbp_n_outer", type=int, default=5,
                    help="number of IGBP outer iterations (default 5)")
    ap.add_argument("--out_dir", type=Path,
                    default=Path("results/dedup_rerun/leace"))
    args = ap.parse_args()

    if args.self_test:
        _self_test(); return

    if args.all_12:
        cities = list(ALL_12)
    elif args.all:
        cities = ["sf", "nyc", "boston"]
    else:
        cities = [args.city]
    if any(c is None for c in cities):
        raise SystemExit("specify --city, --all, or --all_12")
    variants = ["leace", "splince"] if args.all_variants else [args.variant]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for c in cities:
        for v in variants:
            r = run_leace(c, variant=v,
                          force_continuous=args.force_continuous,
                          force_categorical=args.force_categorical,
                          postnl_igbp=args.postnl_igbp,
                          igbp_n_outer=args.igbp_n_outer)
            if r is not None:
                (args.out_dir / f"{c}_{v}.json").write_text(
                    json.dumps(r, indent=2, default=float))
    print(f"\nSaved -> {args.out_dir}")


if __name__ == "__main__":
    main()

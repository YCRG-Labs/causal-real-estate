"""
Short-stacked DML following Ahrens, Hansen, Schaffer, Wiemann (JAE 2025),
"Model Averaging and Double Machine Learning" (arXiv 2401.01645).

Short-stacking reuses the outer K-fold cross-fitting OOF predictions to fit a
constrained-least-squares meta-learner (weights non-negative, sum to one) over
J base learners. Total cost: K x J fits per nuisance, vs K x V x J for
conventional stacking.

This is the Ahrens et al. `ensemble_type='nnls1'` with `shortstack=TRUE` setting
from the R `ddml` package, transcribed to scikit-learn.

Drop-in for the single-LightGBM nuisance regressor in `_dml_core_fast`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _make_lgbm_base(seed: int = 42):
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=200, num_leaves=8, min_data_in_leaf=30, max_bin=63,
        learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, force_col_wise=True, n_jobs=1, verbose=-1,
        random_state=seed,
    )


def _make_ridge_loocv():
    return RidgeCV(alphas=np.logspace(-3, 3, 25), cv=None)


def _make_mlp(seed: int = 42):
    return MLPRegressor(
        hidden_layer_sizes=(32, 32), activation="relu", solver="adam",
        alpha=1e-3, batch_size=64, learning_rate_init=1e-3, max_iter=200,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=10,
        random_state=seed,
    )


def _make_kernel_ridge_rbf(X_ref: np.ndarray | None = None):
    gamma = None
    if X_ref is not None and len(X_ref) > 1:
        from scipy.spatial.distance import pdist
        d = pdist(X_ref[: min(len(X_ref), 200)])
        med = float(np.median(d)) if d.size else 1.0
        gamma = 1.0 / (2.0 * max(med, 1e-6) ** 2)
    return KernelRidge(alpha=1.0, kernel="rbf", gamma=gamma)


def make_default_stack(seed: int = 42, X_ref: np.ndarray | None = None):
    """Recommended base learners for n~350, d~35.

    Returns a list of (name, estimator) tuples ready for short_stack_oof.
    """
    return [
        ("lgbm", _make_lgbm_base(seed)),
        ("ridge", _make_ridge_loocv()),
        ("mlp", _make_mlp(seed)),
        ("krr_rbf", _make_kernel_ridge_rbf(X_ref)),
    ]


def _solve_nnls1(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Constrained least squares: w >= 0, sum(w) = 1.

    Equivalent to Ahrens et al. 'nnls1' meta-learner.
    """
    n, J = Z.shape
    if J == 1:
        return np.ones(1)

    def obj(w):
        r = y - Z @ w
        return float(r @ r)

    def grad(w):
        return -2.0 * Z.T @ (y - Z @ w)

    w0 = np.full(J, 1.0 / J)
    cons = ({"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},)
    bnds = [(0.0, 1.0)] * J
    res = minimize(obj, w0, jac=grad, method="SLSQP",
                   bounds=bnds, constraints=cons,
                   options={"ftol": 1e-10, "maxiter": 200})
    w = res.x
    w = np.maximum(w, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.full(J, 1.0 / J)


def short_stack_oof(
    X: np.ndarray, y: np.ndarray, base_learners,
    k_folds: int = 5, seed: int = 0,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Outer K-fold cross-fit producing OOF predictions for one nuisance.

    Returns (oof_pred, weights) where oof_pred is the n-vector of stacked
    held-out predictions (residualisation-ready) and weights is the J-vector
    of CLS meta-weights fit on the OOF prediction matrix.
    """
    n = len(y)
    J = len(base_learners)
    Z = np.zeros((n, J))
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_id = np.empty(n, dtype=np.int64)
    for k, (tr, te) in enumerate(kf.split(np.arange(n))):
        fold_id[te] = k
        for j, (_name, est) in enumerate(base_learners):
            est_j = _clone_safely(est)
            try:
                if sample_weight is not None:
                    est_j.fit(X[tr], y[tr], sample_weight=sample_weight[tr])
                else:
                    est_j.fit(X[tr], y[tr])
            except TypeError:
                est_j.fit(X[tr], y[tr])
            Z[te, j] = est_j.predict(X[te])

    if sample_weight is None:
        w = _solve_nnls1(Z, y)
    else:
        sw = np.sqrt(sample_weight)
        w = _solve_nnls1(Z * sw[:, None], sw * y)
    oof = Z @ w
    return oof, w


def _clone_safely(est):
    from sklearn.base import clone
    try:
        return clone(est)
    except Exception:
        return est


def short_stacked_dml(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    base_learners=None, k_folds: int = 5, seed: int = 42,
    standardize: bool = True,
) -> dict:
    """Short-stacked DML for a scalar treatment T and outcome Y.

    Fits m_Y(X) -> Y and m_T(X) -> T as CLS combinations of base learners over
    a single outer K-fold cross-fit. Returns point estimate, IF-based SE, and
    95% normal CI. For honest CI at n~350, prefer the cheap-bootstrap wrapper
    in fast_bootstrap_dml.py with this function swapped in.

    Parameters
    ----------
    X : (n, d) confounders.
    T : (n,) scalar treatment (e.g. PC1 of text embeddings).
    Y : (n,) outcome.
    base_learners : list of (name, sklearn_estimator) or None.
        None -> make_default_stack(seed, X).
    k_folds : outer cross-fit folds, default 5.
    seed : random state for the outer K-fold split.
    standardize : if True, fit a StandardScaler on X (recommended for ridge,
        MLP, KRR; LightGBM is invariant).
    """
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64).ravel()
    n = len(Y)

    if standardize:
        X = StandardScaler().fit_transform(X)

    if base_learners is None:
        base_learners = make_default_stack(seed=seed, X_ref=X)

    Y_hat, w_y = short_stack_oof(X, Y, base_learners, k_folds, seed)
    T_hat, w_t = short_stack_oof(X, T, base_learners, k_folds, seed)

    Y_resid = Y - Y_hat
    T_resid = T - T_hat

    denom = float(np.mean(T_resid ** 2))
    if denom < 1e-12:
        raise ValueError("Treatment fully explained by confounders.")
    theta = float(np.mean(T_resid * Y_resid)) / denom
    psi = (Y_resid - theta * T_resid) * T_resid / denom
    se = float(np.sqrt(np.var(psi, ddof=1) / n))
    ci = (theta - 1.96 * se, theta + 1.96 * se)

    return {
        "theta": theta,
        "se": se,
        "ci": ci,
        "weights_Y": dict(zip([n_ for n_, _ in base_learners], w_y)),
        "weights_T": dict(zip([n_ for n_, _ in base_learners], w_t)),
        "n": n,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 350, 35
    X = rng.standard_normal((n, d))
    T = X[:, 0] + 0.5 * X[:, 1] ** 2 + rng.standard_normal(n)
    Y = 0.3 * T + X[:, 0] - 0.2 * X[:, 2] + rng.standard_normal(n)
    out = short_stacked_dml(X, T, Y, k_folds=5, seed=0)
    print(out)

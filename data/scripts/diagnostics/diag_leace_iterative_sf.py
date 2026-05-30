"""
Diagnostic for SF post-LEACE non-linear residual leakage.

Question.  The headline LEACE deconfounding run on SF reports a non-linear
MLP probe R^2(lat/lon) of about 0.086 on the held-out fold (down from 0.124
on the raw embeddings), while NYC and Boston drop below 0.03.  The SF
residual is the bottleneck for the JBES claim that the listing-text
treatment is unconfounded by location.  This script settles two questions
the referee will press:

1.  Is the 0.086 a real surviving signal, or is it the finite-sample MLP
    overfit baseline that Kumar, Tan and Sharma (NeurIPS 2022,
    arXiv:2207.04153) warn about?  We diagnose this with a permutation
    sanity check (shuffle the LEACE-erased SF embeddings row-wise and
    re-fit the same MLP -- if the post-shuffle R^2 is in the same
    neighbourhood as 0.086, the original number is probe noise, not
    semantic leakage).

2.  If it is real signal, can iterative LEACE in the sense of repeated
    closed-form application on the same layer remove it, or does the
    iteration collapse to a no-op?  Belrose et al. (NeurIPS 2023, Section
    5 of arXiv:2306.03819) prove that one closed-form LEACE pass drives
    Sigma_{xz} to zero exactly; their "concept scrubbing" recipe iterates
    across layers, not within a layer.  We confirm empirically: iterate
    closed-form LEACE up to ten times on the SF embeddings, track the
    residual MLP R^2 and the price-prediction R^2 at every step, and
    report.

3.  We also compute Iskander, Radinsky and Belinkov's Iterative
    Gradient-Based Projection (IGBP) over an MLP adversary (Findings of
    ACL 2023, arXiv:2305.10204) as the operationally relevant baseline
    for iterative *non-linear* erasure on the same data.  IGBP is the
    closest published method we can run on a budget; RLACE
    (arXiv:2201.12091) and Kernelized Concept Erasure (arXiv:2201.12191)
    are referenced but not pip-installable on this machine, so we
    surface them in the literature appendix only.

Outputs.
    results/diagnostics/leace_iterative_sf.json
    results/diagnostics/leace_iterative_sf.png  (two-panel figure)

Usage.
    python3 diag_leace_iterative_sf.py
    python3 diag_leace_iterative_sf.py --cities sf nyc boston
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from leace_deconfound import (  # noqa: E402
    leace_erase,
    linear_guardedness_test_continuous,
    mlp_probe_regression,
)
from causal_inference import load_analysis_data, get_features_and_target  # noqa: E402

REPO_ROOT = SCRIPTS_DIR.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "diagnostics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = RESULTS_DIR / "leace_iterative_sf.json"
OUT_PNG = RESULTS_DIR / "leace_iterative_sf.png"


# ---------------------------------------------------------------------------
# Data loader: replicates the leace_deconfound.run_leace front-half so we
# get the exact same train/test split, T matrix and Z matrix used in the
# headline numbers.
# ---------------------------------------------------------------------------

def _prepare_city(city: str, seed: int = 42) -> dict:
    loaded = load_analysis_data(city)
    if loaded is None:
        raise RuntimeError(f"could not load data for {city}")
    emb_df, parcels = loaded
    data = get_features_and_target(emb_df, parcels)
    T_np, _, Y, _ = data
    Y = np.asarray(Y).ravel()
    n = len(Y)
    if len(emb_df) != n:
        emb_df = emb_df.iloc[:n].reset_index(drop=True)

    zips_str = emb_df["zip"].astype(str).values
    z_lab = np.unique(zips_str, return_inverse=True)[1]
    latlon = emb_df[["latitude", "longitude"]].values.astype(np.float64)
    inc_col = next((c for c in ["median_household_income", "median_income"]
                    if c in emb_df.columns), None)
    income = (emb_df[inc_col].fillna(emb_df[inc_col].median()).values.astype(np.float64)
              if inc_col is not None else np.zeros(n))

    T_arr = np.asarray(T_np, dtype=np.float64)
    mask = (np.isfinite(T_arr).all(axis=1) & np.isfinite(latlon).all(axis=1)
            & np.isfinite(income) & np.isfinite(Y))
    T_arr = T_arr[mask]; z_lab = z_lab[mask]
    latlon = latlon[mask]; income = income[mask]; Y = Y[mask]
    n = len(Y)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(0.7 * n)
    tr, te = perm[:n_tr], perm[n_tr:]

    mu_ll = latlon.mean(axis=0); sd_ll = latlon.std(axis=0, ddof=1)
    sd_ll[sd_ll < 1e-12] = 1.0
    Z_ll = (latlon - mu_ll) / sd_ll
    mu_inc = income.mean(); sd_inc = income.std(ddof=1)
    if sd_inc < 1e-12:
        sd_inc = 1.0
    Z_inc = ((income - mu_inc) / sd_inc).reshape(-1, 1)
    Z_concept = np.concatenate([Z_ll, Z_inc], axis=1)

    return {
        "city": city,
        "n": int(n),
        "n_unique_latlon": int(np.unique(np.round(latlon, 5), axis=0).shape[0]),
        "n_zips": int(np.unique(z_lab).size),
        "T": T_arr,
        "Y": Y,
        "Z_concept": Z_concept,
        "latlon": latlon,
        "tr": tr,
        "te": te,
    }


# ---------------------------------------------------------------------------
# Probes.
# ---------------------------------------------------------------------------

def _mlp_r2(T_tr: np.ndarray, T_te: np.ndarray, lat_tr, lat_te,
            seed: int = 0) -> float:
    T_tr_t = torch.from_numpy(np.ascontiguousarray(T_tr))
    T_te_t = torch.from_numpy(np.ascontiguousarray(T_te))
    out = mlp_probe_regression(T_tr_t, T_te_t, lat_tr, lat_te, seed=seed)
    return float(out["r2_avg_test"])


def _ridge_r2(T_tr: np.ndarray, T_te: np.ndarray, lat_tr, lat_te) -> float:
    out = linear_guardedness_test_continuous(T_tr, T_te, lat_tr, lat_te)
    return float(out["r2_avg_test"])


def _price_r2(T_tr: np.ndarray, T_te: np.ndarray, Y_tr, Y_te,
              seed: int = 0) -> float:
    """Single-output Ridge of log-price on T.  Reports test R^2 with
    alpha=1.0 ridge, matching the headline DML nuisance setting."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    mu = T_tr.mean(0); X_tr = T_tr - mu; X_te = T_te - mu
    yt = np.log(np.clip(np.asarray(Y_tr, dtype=np.float64), 1.0, None))
    ye = np.log(np.clip(np.asarray(Y_te, dtype=np.float64), 1.0, None))
    mu_y = yt.mean(); yt_c = yt - mu_y
    m = Ridge(alpha=1.0, fit_intercept=False, random_state=seed).fit(X_tr, yt_c)
    pred = m.predict(X_te) + mu_y
    return float(r2_score(ye, pred))


def _mlp_r2_shuffled(T_tr_erased: np.ndarray, T_te_erased: np.ndarray,
                     lat_tr, lat_te, n_perm: int = 5,
                     seed: int = 0) -> dict:
    """Permutation null: shuffle the erased rows row-wise (breaking the
    row -> lat/lon correspondence) and re-fit the same MLP probe.

    Under the null hypothesis that residual MLP R^2 is finite-sample
    overfit noise, the post-shuffle distribution should be centred on a
    similar (or larger) negative number; under the alternative that the
    residual is real semantic content, the shuffle should send R^2
    decisively negative.  We use 5 permutations for variance and a
    one-sided sign comparison against the observed R^2.
    """
    rng = np.random.default_rng(seed)
    n_tr = T_tr_erased.shape[0]; n_te = T_te_erased.shape[0]
    r2s = []
    for k in range(n_perm):
        perm_tr = rng.permutation(n_tr)
        perm_te = rng.permutation(n_te)
        r2 = _mlp_r2(T_tr_erased[perm_tr], T_te_erased[perm_te],
                     lat_tr, lat_te, seed=seed + k)
        r2s.append(r2)
    r2s = np.asarray(r2s)
    return {
        "mean": float(r2s.mean()),
        "std": float(r2s.std(ddof=1)) if len(r2s) > 1 else float("nan"),
        "min": float(r2s.min()),
        "max": float(r2s.max()),
        "values": [float(x) for x in r2s],
        "n_perm": int(n_perm),
    }


# ---------------------------------------------------------------------------
# Iteration: re-apply closed-form LEACE on the erased reps and watch
# Sigma_xz, the Ridge R^2 and the MLP R^2.  We expect Sigma_xz to be at
# machine zero after iter 1 (so subsequent iterations have nothing to
# erase).  This is the empirical confirmation that "iterative LEACE in
# the same layer" is a no-op, exactly as Belrose et al.'s Theorem 1
# predicts.
# ---------------------------------------------------------------------------

def _sigma_xz_norm(T: np.ndarray, Z: np.ndarray) -> float:
    Xc = T - T.mean(axis=0); Zc = Z - Z.mean(axis=0)
    n = max(Xc.shape[0] - 1, 1)
    return float(np.linalg.norm((Xc.T @ Zc) / n, ord="fro"))


def iterative_leace(prep: dict, n_iter: int = 10,
                    seed: int = 42) -> dict:
    T = prep["T"]; tr, te = prep["tr"], prep["te"]
    Z = prep["Z_concept"]; latlon = prep["latlon"]; Y = prep["Y"]

    T_tr_curr = T[tr].copy(); T_te_curr = T[te].copy()
    history = []

    # iteration 0 = raw
    sig0 = _sigma_xz_norm(T_tr_curr, Z[tr])
    r2_ridge0 = _ridge_r2(T_tr_curr, T_te_curr, latlon[tr], latlon[te])
    r2_mlp0 = _mlp_r2(T_tr_curr, T_te_curr, latlon[tr], latlon[te], seed=seed)
    r2_price0 = _price_r2(T_tr_curr, T_te_curr, Y[tr], Y[te])
    history.append({
        "iter": 0,
        "sigma_xz_frob": sig0,
        "ridge_r2_latlon": r2_ridge0,
        "mlp_r2_latlon": r2_mlp0,
        "ridge_r2_price": r2_price0,
        "guardedness_residual": float("nan"),
    })

    for k in range(1, n_iter + 1):
        T_tr_t = torch.from_numpy(T_tr_curr)
        T_te_t = torch.from_numpy(T_te_curr)
        T_tr_e, T_te_e, eraser, resid = leace_erase(
            T_tr_t, Z[tr], T_holdout=T_te_t, label=f"LEACE.iter{k}"
        )
        T_tr_curr = T_tr_e.detach().cpu().numpy().astype(np.float64)
        T_te_curr = T_te_e.detach().cpu().numpy().astype(np.float64)

        sig = _sigma_xz_norm(T_tr_curr, Z[tr])
        r2_ridge = _ridge_r2(T_tr_curr, T_te_curr, latlon[tr], latlon[te])
        r2_mlp = _mlp_r2(T_tr_curr, T_te_curr, latlon[tr], latlon[te], seed=seed)
        r2_price = _price_r2(T_tr_curr, T_te_curr, Y[tr], Y[te])

        history.append({
            "iter": int(k),
            "sigma_xz_frob": float(sig),
            "ridge_r2_latlon": float(r2_ridge),
            "mlp_r2_latlon": float(r2_mlp),
            "ridge_r2_price": float(r2_price),
            "guardedness_residual": float(resid),
        })

        # convergence: linear-guardedness identity already saturated at
        # machine precision after one iter; we still run a few more to
        # show the curve flatlines.
        if sig < 1e-10 and k >= 2:
            break

    return {
        "history": history,
        "T_tr_final": T_tr_curr,
        "T_te_final": T_te_curr,
    }


# ---------------------------------------------------------------------------
# Iterative gradient-based projection (Iskander, Radinsky, Belinkov,
# Findings of ACL 2023, arXiv:2305.10204).  Faithful to Algorithm 1
# under the simplifications described in the file header.
# ---------------------------------------------------------------------------

def _fit_mlp_adversary(T_tr: np.ndarray, Z_tr_z: np.ndarray, seed: int,
                       n_epochs: int = 80, hidden: int = 128,
                       lr: float = 1e-3, weight_decay: float = 1e-4,
                       ) -> torch.nn.Module:
    torch.manual_seed(seed)
    d_in = T_tr.shape[1]; d_out = Z_tr_z.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, d_out),
    ).double()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    X = torch.from_numpy(T_tr.astype(np.float64))
    Y = torch.from_numpy(Z_tr_z.astype(np.float64))
    for _ in range(n_epochs):
        model.train()
        pred = model(X)
        loss = ((pred - Y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def igbp_erase(prep: dict, n_outer: int = 5, n_inner: int = 80,
               step_norm: float = 0.5, seed: int = 42) -> dict:
    """A faithful-in-spirit, budget-friendly IGBP loop.

    Iskander, Radinsky and Belinkov (Findings of ACL 2023,
    arXiv:2305.10204) describe IGBP as: train a non-linear adversary,
    take the gradient of its loss with respect to the representation,
    and project the representation onto the hyperplane orthogonal to a
    summary of that gradient.  We approximate the gradient summary by
    the leading singular vector of the per-row input gradients (the
    `step_norm` direction along which the adversary is most informative
    about the concept) and project to its orthogonal complement.

    This is a single-layer post-hoc projection, suitable for a static
    768-dim embedding matrix; nothing here back-propagates into the
    upstream encoder.
    """
    T_tr_curr = prep["T"][prep["tr"]].copy()
    T_te_curr = prep["T"][prep["te"]].copy()
    Z_tr_z = prep["Z_concept"][prep["tr"]]
    Z_te_z = prep["Z_concept"][prep["te"]]
    Y_tr, Y_te = prep["Y"][prep["tr"]], prep["Y"][prep["te"]]
    latlon = prep["latlon"]

    history = []
    rng = np.random.default_rng(seed)
    history.append({
        "outer": 0,
        "mlp_r2_latlon": _mlp_r2(T_tr_curr, T_te_curr,
                                 latlon[prep["tr"]], latlon[prep["te"]],
                                 seed=seed),
        "ridge_r2_price": _price_r2(T_tr_curr, T_te_curr, Y_tr, Y_te),
    })

    for outer in range(1, n_outer + 1):
        adv = _fit_mlp_adversary(T_tr_curr, Z_tr_z, seed=seed + outer,
                                 n_epochs=n_inner)
        # Per-row input-gradient: d L_row / d x_row
        T_tr_t = torch.from_numpy(T_tr_curr.astype(np.float64)).requires_grad_(True)
        Y_t = torch.from_numpy(Z_tr_z.astype(np.float64))
        pred = adv(T_tr_t)
        loss_per_row = ((pred - Y_t) ** 2).mean(dim=1)
        grads = []
        for i in range(T_tr_t.shape[0]):
            g = torch.autograd.grad(loss_per_row[i], T_tr_t, retain_graph=True)[0][i]
            grads.append(g.detach().cpu().numpy().astype(np.float64))
        G = np.asarray(grads)
        # Leading right-singular vector summarises the dominant
        # adversary-sensitive direction across rows.
        _, _, Vt = np.linalg.svd(G, full_matrices=False)
        v = Vt[0]
        v = v / max(np.linalg.norm(v), 1e-12)
        # Orthogonal projection that kills v
        P = np.eye(T_tr_curr.shape[1]) - np.outer(v, v)
        T_tr_curr = T_tr_curr @ P
        T_te_curr = T_te_curr @ P

        history.append({
            "outer": int(outer),
            "mlp_r2_latlon": _mlp_r2(T_tr_curr, T_te_curr,
                                     latlon[prep["tr"]],
                                     latlon[prep["te"]], seed=seed),
            "ridge_r2_price": _price_r2(T_tr_curr, T_te_curr, Y_tr, Y_te),
            "v_norm_check": float(np.linalg.norm(v)),
        })

    return {"history": history, "T_tr_final": T_tr_curr,
            "T_te_final": T_te_curr}


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def run_diagnostic(cities: list[str], seed: int = 42,
                   n_iter_leace: int = 10, n_outer_igbp: int = 5,
                   n_perm_shuffle: int = 5) -> dict:
    out: dict = {"seed": int(seed), "cities": {}}
    for city in cities:
        print(f"\n[{city}] preparing data")
        prep = _prepare_city(city, seed=seed)
        city_out = {
            "n": prep["n"],
            "n_unique_latlon": prep["n_unique_latlon"],
            "n_zips": prep["n_zips"],
        }
        print(f"  n={prep['n']}  n_unique_latlon={prep['n_unique_latlon']}  "
              f"n_zips={prep['n_zips']}")

        # Step 1: iterative LEACE -------------------------------------------------
        print(f"  iterative LEACE ({n_iter_leace} iters)...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iter_res = iterative_leace(prep, n_iter=n_iter_leace, seed=seed)
        city_out["iterative_leace"] = iter_res["history"]
        last = iter_res["history"][-1]
        print(f"    final iter={last['iter']}  Sigma_xz_frob={last['sigma_xz_frob']:.3e}"
              f"  MLP R^2(lat/lon)={last['mlp_r2_latlon']:+.4f}"
              f"  Ridge R^2(price)={last['ridge_r2_price']:+.3f}")

        # Step 2: shuffle sanity check on iter-1 erased reps ----------------------
        T_tr_iter1 = prep["T"][prep["tr"]].copy()
        T_te_iter1 = prep["T"][prep["te"]].copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T_tr_e_t, T_te_e_t, _, _ = leace_erase(
                torch.from_numpy(T_tr_iter1), prep["Z_concept"][prep["tr"]],
                T_holdout=torch.from_numpy(T_te_iter1), label="LEACE.iter1.shuffle",
            )
        T_tr_e = T_tr_e_t.detach().cpu().numpy().astype(np.float64)
        T_te_e = T_te_e_t.detach().cpu().numpy().astype(np.float64)
        observed = _mlp_r2(T_tr_e, T_te_e, prep["latlon"][prep["tr"]],
                           prep["latlon"][prep["te"]], seed=seed)
        shuffled = _mlp_r2_shuffled(T_tr_e, T_te_e,
                                    prep["latlon"][prep["tr"]],
                                    prep["latlon"][prep["te"]],
                                    n_perm=n_perm_shuffle, seed=seed)
        city_out["shuffle_sanity"] = {
            "observed_post_leace_mlp_r2": float(observed),
            "shuffled_distribution": shuffled,
        }
        print(f"  shuffle sanity: observed={observed:+.4f}  "
              f"shuffled mean={shuffled['mean']:+.4f}  "
              f"std={shuffled['std']:+.4f}  range=[{shuffled['min']:+.4f}, {shuffled['max']:+.4f}]")

        # Step 3: IGBP comparison -------------------------------------------------
        print(f"  IGBP ({n_outer_igbp} outer iters)...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            igbp_res = igbp_erase(prep, n_outer=n_outer_igbp, seed=seed)
        city_out["igbp"] = igbp_res["history"]
        last_igbp = igbp_res["history"][-1]
        print(f"    final outer={last_igbp['outer']}  "
              f"MLP R^2(lat/lon)={last_igbp['mlp_r2_latlon']:+.4f}  "
              f"Ridge R^2(price)={last_igbp['ridge_r2_price']:+.3f}")

        # RLACE / Kernelized: not pip-installable on this machine, see appendix.
        city_out["rlace"] = {"status": "skipped",
                             "reason": "shauli-ravfogel/rlace is a git-only repo, "
                                       "not pip-installable; see literature appendix"}
        city_out["kernelized_erasure"] = {"status": "skipped",
                                          "reason": "Ravfogel et al. arXiv:2201.12191 "
                                                   "shows cross-kernel transfer fails; "
                                                   "noted in appendix"}

        out["cities"][city] = city_out

    return out


def make_plot(results: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cities = list(results["cities"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    palette = {"sf": "#d1495b", "nyc": "#2e86ab", "boston": "#3b8132"}

    ax = axes[0]
    for c in cities:
        hist = results["cities"][c]["iterative_leace"]
        xs = [h["iter"] for h in hist]
        ys = [h["mlp_r2_latlon"] for h in hist]
        ax.plot(xs, ys, marker="o", color=palette.get(c, None), label=c.upper())
        # shuffle baseline annotation
        sh = results["cities"][c]["shuffle_sanity"]["shuffled_distribution"]
        ax.axhline(sh["mean"], linestyle=":", color=palette.get(c, None), alpha=0.6)
    ax.axhline(0.02, linestyle="--", color="grey", alpha=0.6,
               label="JBES tolerance 0.02")
    ax.set_xlabel("LEACE iteration")
    ax.set_ylabel(r"MLP $R^2$(lat/lon) on erased reps")
    ax.set_title("(a) Iterative LEACE: residual non-linear leakage")
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for c in cities:
        hist = results["cities"][c]["iterative_leace"]
        xs = [h["iter"] for h in hist]
        ys = [h["ridge_r2_price"] for h in hist]
        ax.plot(xs, ys, marker="s", color=palette.get(c, None), label=c.upper())
    ax.set_xlabel("LEACE iteration")
    ax.set_ylabel(r"Ridge $R^2$(log price) on erased reps")
    ax.set_title("(b) Task signal: price-prediction R$^2$")
    ax.legend(loc="best", fontsize=9, frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", default=["sf", "nyc", "boston"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_iter_leace", type=int, default=10)
    ap.add_argument("--n_outer_igbp", type=int, default=5)
    ap.add_argument("--n_perm_shuffle", type=int, default=5)
    args = ap.parse_args()

    results = run_diagnostic(
        cities=args.cities, seed=args.seed,
        n_iter_leace=args.n_iter_leace,
        n_outer_igbp=args.n_outer_igbp,
        n_perm_shuffle=args.n_perm_shuffle,
    )
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    make_plot(results, OUT_PNG)
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()

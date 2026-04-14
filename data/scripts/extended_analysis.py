import sys
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from config import PROCESSED_DIR, EMBEDDING_DIM


def load_city(city, with_rich_confounders=True):
    """
    Loads embeddings and price for a city.
    Returns: T (text), L_zip (zip-only one-hot, legacy), Y (log price), df (full).
    If with_rich_confounders=True, also attaches the 35-feature confounder
    matrix used by the DR estimator (lat/lon + property + census + crime + amenity + micro-geo)
    via a spatial join — accessible as df.attrs["rich_confounders"].
    """
    path = PROCESSED_DIR / f"{city}_embeddings.parquet"
    df = pd.read_parquet(path)
    emb_cols = [f"emb_{i}" for i in range(EMBEDDING_DIM)]
    available = [c for c in emb_cols if c in df.columns]

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price", "zip"])
    df = df[df["price"] > 0]

    T = df[available].values
    Y = np.log(df["price"].values.astype(float))

    zips = df["zip"].astype(float).astype(int).astype(str)
    le = LabelEncoder()
    zip_enc = le.fit_transform(zips)
    n_bins = min(len(le.classes_), 20)
    L = np.zeros((len(T), n_bins))
    for i, z in enumerate(zip_enc):
        L[i, z % n_bins] = 1.0

    if with_rich_confounders:
        try:
            from causal_inference import (
                load_analysis_data, get_features_and_target,
            )
            data = load_analysis_data(city)
            if data is not None:
                emb_df_full, parcels = data
                feat = get_features_and_target(emb_df_full, parcels)
                if feat is not None:
                    _, rich_conf, Y_rich, meta = feat
                    if len(rich_conf) == len(T):
                        df.attrs["rich_confounders"] = rich_conf
                        df.attrs["n_rich_features"] = rich_conf.shape[1]
                    else:
                        df.attrs["rich_confounders"] = None
                        df.attrs["n_rich_features"] = 0
        except Exception as e:
            print(f"  Note: rich confounders unavailable ({e})")
            df.attrs["rich_confounders"] = None
            df.attrs["n_rich_features"] = 0

    return T, L, Y, df


def _ci_partial_corr_test(T_s, Y, conf_s, n_permutations=500, label=""):
    """
    Tests Y indep T | conf via partial Spearman correlations on
    nonparametrically residualized Y and each T component.

    Residualization uses Ridge regression rather than OLS so the test is
    well-defined when conf is high-dimensional relative to n.
    """
    ridge_y = Ridge(alpha=1.0).fit(conf_s, Y)
    resid_Y = Y - ridge_y.predict(conf_s)

    resid_T = np.zeros_like(T_s)
    for j in range(T_s.shape[1]):
        ridge_t = Ridge(alpha=1.0).fit(conf_s, T_s[:, j])
        resid_T[:, j] = T_s[:, j] - ridge_t.predict(conf_s)

    partial_corrs = []
    for j in range(resid_T.shape[1]):
        r, _ = spearmanr(resid_Y, resid_T[:, j])
        partial_corrs.append(abs(r) if not np.isnan(r) else 0.0)

    mean_partial = float(np.mean(partial_corrs))
    max_partial = float(np.max(partial_corrs))

    n_components_used = min(10, resid_T.shape[1])
    null_means = []
    for p in range(n_permutations):
        perm = np.random.RandomState(p).permutation(len(resid_Y))
        perm_corrs = []
        for j in range(n_components_used):
            r, _ = spearmanr(resid_Y[perm], resid_T[:, j])
            perm_corrs.append(abs(r) if not np.isnan(r) else 0.0)
        null_means.append(np.mean(perm_corrs))

    null_means = np.array(null_means)
    p_value = float((np.sum(null_means >= mean_partial) + 1) / (n_permutations + 1))

    print(f"  [{label}]  conf_dim={conf_s.shape[1]:>3d}  "
          f"mean|ρ|={mean_partial:.4f}  max|ρ|={max_partial:.4f}  "
          f"p={p_value:.4f}  "
          f"{'REJECT' if p_value < 0.05 else 'consistent with CI'}")

    return {
        "label": label,
        "conf_dim": int(conf_s.shape[1]),
        "mean_partial": mean_partial,
        "max_partial": max_partial,
        "p_value": p_value,
    }


def test_conditional_independence(T, L, Y, rich_confounders=None):
    """
    Tests Y indep T | conf at two levels of adjustment:
      (1) conf = zip-only one-hot (the legacy spec the paper currently uses)
      (2) conf = rich 35-feature DR adjustment set, if provided

    The headline result is whether the rejection at level (1) survives
    finer-grained adjustment at level (2). Under SCM_0, the rejection
    should weaken as confounders improve.
    """
    print("\n" + "="*60)
    print("1. CONDITIONAL INDEPENDENCE TESTS")
    print("="*60)
    print("  H0: Y ⫫ T | conf  (text has no information about price beyond conf)")

    n_pca = min(30, T.shape[1], len(T) - 2)
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)

    scaler_t = StandardScaler()
    T_s = scaler_t.fit_transform(T_pca)
    scaler_l = StandardScaler()
    L_s = scaler_l.fit_transform(L)

    print()
    zip_result = _ci_partial_corr_test(T_s, Y, L_s, label="zip-only ")

    rich_result = None
    if rich_confounders is not None and len(rich_confounders) == len(Y):
        scaler_r = StandardScaler()
        rich_s = scaler_r.fit_transform(rich_confounders)
        rich_result = _ci_partial_corr_test(T_s, Y, rich_s, label="rich 35d")

        if zip_result["p_value"] < 0.05 and rich_result["p_value"] >= 0.05:
            print("\n  → Zip-only test REJECTED but rich-confounder test does NOT.")
            print("    The earlier rejection was incomplete adjustment, not real signal.")
            print("    This is the SCM_0 prediction: refuting confounding shrinks the residual.")
        elif zip_result["p_value"] < 0.05 and rich_result["p_value"] < 0.05:
            print("\n  → Both tests reject. Either confounding remains at finer scales,")
            print("    or there is a small genuine direct effect of T on Y.")
        else:
            print("\n  → Neither test rejects (or only the rich one does).")
    else:
        print("\n  Note: rich confounders unavailable; only the zip-only test was run.")
        print("        Re-run with df.attrs['rich_confounders'] populated to get the")
        print("        full 35-feature adjustment that matches the DR estimator.")

    return {"zip_only": zip_result, "rich": rich_result}


def _shared_pca_basis(cities_data, n_components):
    """
    Fits a single PCA basis on the union of all cities' text embeddings,
    so train- and test-city projections share the same axes (avoids the
    sign/rotation arbitrariness of fitting per-city PCA bases).
    """
    all_T = np.vstack([cities_data[c][0] for c in cities_data])
    n_components = min(n_components, all_T.shape[1], all_T.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(all_T)
    return pca


def _cv_r2(model_fn, X, Y, n_folds=5, seed=42):
    """Held-out 5-fold CV R², not train-set R²."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = []
    for tr, te in kf.split(Y):
        m = model_fn()
        m.fit(X[tr], Y[tr])
        scores.append(m.score(X[te], Y[te]))
    return float(np.mean(scores))


def cross_market_transfer(cities_data):
    """
    Tests whether text-only price models are portable across cities.

    AUDIT CHANGES vs prior version:
      1. Self R² is now 5-fold CV, not train-set R² (the old SF self R²=1.00
         was just GBR overfitting ~1000 samples).
      2. PCA basis is shared across cities (fit on union), so per-city
         component ordering does not introduce alignment artifacts.
      3. Reports three transfer metrics:
           - raw R² (with city-level price gap intact)
           - mean-centered R² (price level removed; tests *within-city ranking*)
           - Spearman ρ (scale-free rank correlation)
         These distinguish "different price levels" from "text doesn't transfer."
      4. Adds a "predict test mean" baseline so R² < 0 is interpretable.
    """
    print("\n" + "="*60)
    print("2. CROSS-MARKET TRANSFER TEST  (audited)")
    print("="*60)

    city_names = list(cities_data.keys())
    if len(city_names) < 2:
        print("  Need >= 2 cities for transfer test, skipping")
        return []

    n_pca = min(20, min(cities_data[c][0].shape[1] for c in city_names),
                min(len(cities_data[c][2]) for c in city_names) - 2)
    shared_pca = _shared_pca_basis(cities_data, n_pca)
    print(f"  Shared PCA basis: {n_pca} components fit on union of "
          f"{sum(len(cities_data[c][2]) for c in city_names)} embeddings")

    def gbr():
        return GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42,
        )

    results = []
    for train_city in city_names:
        T_tr, _, Y_tr, _ = cities_data[train_city]
        T_tr_pca = shared_pca.transform(T_tr)
        Y_tr_centered = Y_tr - Y_tr.mean()

        r2_self_cv = _cv_r2(gbr, T_tr_pca, Y_tr)
        print(f"\n  Train: {train_city} (n={len(Y_tr)})  "
              f"self CV R²={r2_self_cv:.4f}  "
              f"mean log-price={Y_tr.mean():.3f}")

        model_raw = gbr()
        model_raw.fit(T_tr_pca, Y_tr)

        model_centered = gbr()
        model_centered.fit(T_tr_pca, Y_tr_centered)

        for test_city in city_names:
            if test_city == train_city:
                continue
            T_te, _, Y_te, _ = cities_data[test_city]
            T_te_pca = shared_pca.transform(T_te)
            Y_te_centered = Y_te - Y_te.mean()

            preds_raw = model_raw.predict(T_te_pca)
            r2_raw = r2_score(Y_te, preds_raw)

            preds_centered = model_centered.predict(T_te_pca)
            r2_centered = r2_score(Y_te_centered, preds_centered)

            rank_rho, _ = spearmanr(Y_te, preds_raw)
            if np.isnan(rank_rho):
                rank_rho = 0.0

            r2_baseline = r2_score(Y_te, np.full_like(Y_te, Y_te.mean()))
            r2_train_mean_pred = r2_score(Y_te, np.full_like(Y_te, Y_tr.mean()))

            print(f"    -> {test_city} (n={len(Y_te)}, mean log-price={Y_te.mean():.3f}):")
            print(f"         raw R²:           {r2_raw:>8.4f}")
            print(f"         centered R²:      {r2_centered:>8.4f}  (price-level removed)")
            print(f"         Spearman ρ:       {rank_rho:>8.4f}  (scale-free)")
            print(f"         R² of test-mean:  {r2_baseline:>8.4f}  (definitionally 0)")
            print(f"         R² of train-mean: {r2_train_mean_pred:>8.4f}  (price-level baseline)")

            verdict = "fails"
            if rank_rho > 0.3 and r2_centered > 0.1:
                verdict = "partial transfer"
            elif rank_rho > 0.5:
                verdict = "rank transfer only"

            results.append({
                "train": train_city, "test": test_city,
                "self_cv_r2": r2_self_cv,
                "transfer_raw_r2": r2_raw,
                "transfer_centered_r2": r2_centered,
                "transfer_rank_rho": rank_rho,
                "verdict": verdict,
            })

    print("\n  Transfer audit summary:")
    print(f"  {'train→test':<20} {'self CV R²':>10} {'raw R²':>10} {'centered R²':>12} {'rank ρ':>8}  verdict")
    for r in results:
        pair = f"{r['train']}→{r['test']}"
        print(f"  {pair:<20} {r['self_cv_r2']:>10.4f} {r['transfer_raw_r2']:>10.4f} "
              f"{r['transfer_centered_r2']:>12.4f} {r['transfer_rank_rho']:>8.4f}  {r['verdict']}")

    mean_centered = np.mean([r["transfer_centered_r2"] for r in results])
    mean_rank = np.mean([r["transfer_rank_rho"] for r in results])
    print(f"\n  Mean centered R²:  {mean_centered:.4f}")
    print(f"  Mean Spearman ρ:   {mean_rank:.4f}")

    if mean_centered < 0 and mean_rank < 0.1:
        print("  → Text models fail even after removing price-level differences:")
        print("    text encodes city-specific patterns, not portable property semantics")
    elif mean_rank > 0.3:
        print("  → Within-city ranking partially transfers: residual semantic signal")
    else:
        print("  → Text transfer is weak: dominantly location-specific encoding")

    return results


def embedding_ablation(df_with_text, T_orig, L, Y):
    print("\n" + "="*60)
    print("3. EMBEDDING ABLATION (location words removed)")
    print("="*60)

    location_patterns = [
        r'\b\d{5}\b',
        r'\bmanhattan\b', r'\bbrooklyn\b', r'\bbronx\b', r'\bqueens\b', r'\bstaten island\b',
        r'\bupper east\b', r'\bupper west\b', r'\blower east\b', r'\bchelsea\b', r'\bsoho\b',
        r'\btribeca\b', r'\bwilliamsburg\b', r'\bbushwick\b', r'\bpark slope\b', r'\bdumbo\b',
        r'\bsouth end\b', r'\bback bay\b', r'\bbeacon hill\b', r'\bjamaica plain\b',
        r'\bmission\b', r'\bnoe valley\b', r'\bpacific heights\b', r'\bsunset\b', r'\brichmond\b',
        r'\bnorth beach\b', r'\bhaight\b', r'\bcast[r]o\b', r'\bsoma\b',
        r'\bdowntown\b', r'\bmidtown\b', r'\buptown\b', r'\bvillage\b',
        r'\bheights\b', r'\bhill\b', r'\bpark\b', r'\bave\b', r'\bstreet\b', r'\bst\b',
    ]

    descs = df_with_text["clean_description"].values
    ablated = []
    words_removed = 0
    for desc in descs:
        d = str(desc).lower()
        orig_len = len(d.split())
        for pat in location_patterns:
            d = re.sub(pat, "", d)
        d = re.sub(r"\s+", " ", d).strip()
        words_removed += orig_len - len(d.split())
        ablated.append(d)

    print(f"  Avg words removed per description: {words_removed / len(descs):.1f}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")
    T_ablated = model.encode(ablated, batch_size=64, show_progress_bar=False)

    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score

    zips = df_with_text["zip"].fillna(0).astype(float).astype(int).astype(str).values
    le = LabelEncoder()
    zip_labels = le.fit_transform(zips)

    n_clusters = min(50, len(T_orig))
    km_orig = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(T_orig)
    nmi_orig = normalized_mutual_info_score(zip_labels, km_orig)

    km_ablated = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(T_ablated)
    nmi_ablated = normalized_mutual_info_score(zip_labels, km_ablated)

    n_pca = min(30, T_orig.shape[1], len(T_orig) - 2)
    pca_o = PCA(n_components=n_pca, random_state=42)
    T_o_pca = pca_o.fit_transform(T_orig)
    pca_a = PCA(n_components=n_pca, random_state=42)
    T_a_pca = pca_a.fit_transform(T_ablated)

    valid_classes = np.bincount(zip_labels)
    valid = np.where(valid_classes >= 3)[0]
    if len(valid) >= 2:
        mask = np.isin(zip_labels, valid)
        le2 = LabelEncoder()
        labels = le2.fit_transform(zip_labels[mask])

        clf_orig = LogisticRegression(max_iter=1000, random_state=42)
        acc_orig = cross_val_score(clf_orig, T_o_pca[mask], labels, cv=min(5, len(valid)), scoring="accuracy").mean()

        clf_ablated = LogisticRegression(max_iter=1000, random_state=42)
        acc_ablated = cross_val_score(clf_ablated, T_a_pca[mask], labels, cv=min(5, len(valid)), scoring="accuracy").mean()
    else:
        acc_orig = acc_ablated = float("nan")

    print(f"\n  NMI (original):  {nmi_orig:.4f}")
    print(f"  NMI (ablated):   {nmi_ablated:.4f}")
    print(f"  NMI drop:        {nmi_orig - nmi_ablated:.4f} ({(nmi_orig - nmi_ablated) / nmi_orig * 100:.1f}%)")
    print(f"\n  Classifier acc (original):  {acc_orig:.4f}")
    print(f"  Classifier acc (ablated):   {acc_ablated:.4f}")
    print(f"  Accuracy drop:              {acc_orig - acc_ablated:.4f}")

    if (nmi_orig - nmi_ablated) / nmi_orig < 0.2:
        print("\n  -> Location encoding persists after removing explicit place names")
        print("     Confounding operates through SUBTLE linguistic patterns")
    else:
        print("\n  -> Explicit place names account for significant location encoding")

    return nmi_orig, nmi_ablated, acc_orig, acc_ablated


def cinelli_hazlett_sensitivity(T, L, Y):
    print("\n" + "="*60)
    print("4. CINELLI-HAZLETT SENSITIVITY ANALYSIS")
    print("="*60)

    n_pca = min(20, T.shape[1], len(T) - 2)
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)

    scaler = StandardScaler()
    T_s = scaler.fit_transform(T_pca)

    T_norm = np.linalg.norm(T_s, axis=1)
    D = (T_norm > np.median(T_norm)).astype(float)

    X_full = np.column_stack([D, L])
    X_restricted = L.copy()

    reg_full = LinearRegression().fit(X_full, Y)
    reg_restricted = LinearRegression().fit(X_restricted, Y)

    tau_hat = reg_full.coef_[0]
    Y_hat_full = reg_full.predict(X_full)
    residuals = Y - Y_hat_full
    se_tau = np.sqrt(np.sum(residuals**2) / (len(Y) - X_full.shape[1])) / np.sqrt(np.sum((D - D.mean())**2))

    r2_full = r2_score(Y, Y_hat_full)
    r2_restricted = r2_score(Y, reg_restricted.predict(X_restricted))

    partial_r2_D = (r2_full - r2_restricted) / (1 - r2_restricted)

    if abs(tau_hat) < 1e-10:
        rv = 0.0
    else:
        rv = abs(tau_hat) / (abs(tau_hat) + 2 * se_tau)

    print(f"  Treatment effect (tau): {tau_hat:.4f}")
    print(f"  Standard error:         {se_tau:.4f}")
    print(f"  t-statistic:            {tau_hat / se_tau:.4f}")
    print(f"  R2 (full model):        {r2_full:.4f}")
    print(f"  R2 (without treatment): {r2_restricted:.4f}")
    print(f"  Partial R2 of D:        {partial_r2_D:.6f}")
    print(f"  Robustness Value (RV):  {rv:.4f}")

    print(f"\n  Interpretation:")
    print(f"    An unobserved confounder explaining just {rv*100:.1f}% of residual")
    print(f"    variation in both treatment and outcome would suffice to")
    print(f"    reduce the estimated effect to zero.")

    if rv < 0.05:
        print(f"    -> VERY LOW robustness: even tiny confounders overturn the effect")
    elif rv < 0.15:
        print(f"    -> LOW robustness: moderate confounders suffice")
    else:
        print(f"    -> MODERATE robustness")

    return tau_hat, se_tau, partial_r2_D, rv


def partial_r2_decomposition(T, L, Y):
    print("\n" + "="*60)
    print("5. PARTIAL R-SQUARED DECOMPOSITION")
    print("="*60)

    n_pca = min(20, T.shape[1], len(T) - 2)
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)

    scaler_t = StandardScaler()
    T_s = scaler_t.fit_transform(T_pca)
    scaler_l = StandardScaler()
    L_s = scaler_l.fit_transform(L)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def cv_r2(X, Y):
        scores = []
        for tr, te in kf.split(Y):
            m = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            m.fit(X[tr], Y[tr])
            scores.append(m.score(X[te], Y[te]))
        return np.mean(scores)

    r2_L = cv_r2(L_s, Y)
    r2_T = cv_r2(T_s, Y)
    r2_LT = cv_r2(np.hstack([L_s, T_s]), Y)

    unique_T = r2_LT - r2_L
    unique_L = r2_LT - r2_T
    shared = r2_L + r2_T - r2_LT
    if shared < 0:
        shared = 0

    print(f"  R2 (location only):       {r2_L:.4f}")
    print(f"  R2 (text only):           {r2_T:.4f}")
    print(f"  R2 (location + text):     {r2_LT:.4f}")
    print(f"\n  Variance decomposition:")
    print(f"    Unique to location:     {unique_L:.4f}")
    print(f"    Unique to text:         {unique_T:.4f}")
    print(f"    Shared (redundant):     {shared:.4f}")

    if r2_T > 0:
        pct_shared = shared / r2_T * 100
        print(f"\n  {pct_shared:.1f}% of text's predictive variance is shared with location")
    if unique_T < 0.01:
        print("  -> Text contributes near-zero UNIQUE variance beyond location")
    else:
        print(f"  -> Text contributes {unique_T:.4f} unique variance")

    return r2_L, r2_T, r2_LT, unique_L, unique_T, shared


def competing_scm_test(cities_data):
    print("\n" + "="*60)
    print("7. COMPETING SCM TEST (SCM_0 vs SCM_1)")
    print("="*60)
    print("  SCM_0: T has NO direct effect on Y (text encodes location)")
    print("  SCM_1: T has a direct effect on Y (text carries semantic signal)")
    print()

    print("  Test 1: Cross-market portability")
    print("  Under SCM_1, semantic content is portable across cities.")
    print("  Under SCM_0, text-price associations are location-specific.")
    print()

    city_names = list(cities_data.keys())
    if len(city_names) < 2:
        print("  Need >= 2 cities for cross-market test, skipping")
        return {"verdict": "insufficient_cities"}

    n_pca_shared = min(20, min(cities_data[c][0].shape[1] for c in city_names),
                       min(len(cities_data[c][2]) for c in city_names) - 2)
    shared_pca = _shared_pca_basis(cities_data, n_pca_shared)

    def gbr():
        return GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42,
        )

    transfer_r2s = []
    self_r2s = []
    for train_city in city_names:
        T_tr, L_tr, Y_tr, _ = cities_data[train_city]
        T_tr_pca = shared_pca.transform(T_tr)
        self_r2s.append(_cv_r2(gbr, T_tr_pca, Y_tr))

        model = gbr()
        model.fit(T_tr_pca, Y_tr - Y_tr.mean())

        for test_city in city_names:
            if test_city == train_city:
                continue
            T_te, _, Y_te, _ = cities_data[test_city]
            T_te_pca = shared_pca.transform(T_te)
            preds = model.predict(T_te_pca)
            transfer_r2s.append(r2_score(Y_te - Y_te.mean(), preds))

    mean_self = np.mean(self_r2s)
    mean_transfer = np.mean(transfer_r2s)
    print(f"  Within-city R2:  {mean_self:.4f}")
    print(f"  Cross-city R2:   {mean_transfer:.4f}")

    if mean_transfer < 0:
        print(f"  Cross-city R2 is NEGATIVE (worse than predicting the mean).")
        print(f"  This is inconsistent with SCM_1: if text encodes universal")
        print(f"  property semantics, cross-city transfer should work.")
        portability_verdict = "rejects_SCM1"
    elif mean_transfer < mean_self * 0.3:
        print(f"  Cross-city R2 is <30% of within-city.")
        print(f"  Weak evidence for SCM_1.")
        portability_verdict = "weak_SCM1"
    else:
        print(f"  Cross-city transfer partially succeeds.")
        print(f"  Consistent with SCM_1.")
        portability_verdict = "supports_SCM1"

    print()
    print("  Test 2: Effect stability under richer confounders")
    print("  Under SCM_0, the estimated effect shrinks toward zero as")
    print("  confounders improve (bias is being removed).")
    print("  Under SCM_1, the effect stays stable (it's real, not bias).")
    print()

    primary_city = city_names[0]
    T_p, L_p, Y_p, _ = cities_data[primary_city]

    n_pca = min(20, T_p.shape[1], len(T_p) - 2)
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T_p)
    T_norm = np.linalg.norm(StandardScaler().fit_transform(T_pca), axis=1)
    treatment = (T_norm > np.median(T_norm)).astype(float)

    def quick_dr(D, Y, conf):
        if conf.shape[1] == 0:
            conf = np.zeros((len(Y), 1))
        sc = StandardScaler()
        C = sc.fit_transform(conf)
        out = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        out.fit(np.hstack([D.reshape(-1, 1), C]), Y)
        prop = LogisticRegression(max_iter=1000, random_state=42)
        prop.fit(C, D)
        e = np.clip(prop.predict_proba(C)[:, 1], 0.05, 0.95)
        m1 = out.predict(np.hstack([np.ones((len(Y), 1)), C]))
        m0 = out.predict(np.hstack([np.zeros((len(Y), 1)), C]))
        return float(np.mean(m1 - m0 + D*(Y-m1)/e - (1-D)*(Y-m0)/(1-e)))

    ate_no_conf = quick_dr(treatment, Y_p, np.zeros((len(Y_p), 0)))
    ate_with_conf = quick_dr(treatment, Y_p, StandardScaler().fit_transform(L_p))

    print(f"  ATE with no confounders:  {ate_no_conf:.4f}")
    print(f"  ATE with location conf:   {ate_with_conf:.4f}")

    if abs(ate_no_conf) > 0.001:
        shrinkage = 1 - abs(ate_with_conf) / abs(ate_no_conf)
        print(f"  Shrinkage: {shrinkage*100:.1f}%")
        if shrinkage > 0.5:
            stability_verdict = "rejects_SCM1"
            print(f"  Effect shrinks >50% under confounders: consistent with SCM_0")
        elif shrinkage > 0.2:
            stability_verdict = "ambiguous"
            print(f"  Moderate shrinkage: ambiguous")
        else:
            stability_verdict = "supports_SCM1"
            print(f"  Effect is stable: consistent with SCM_1")
    else:
        stability_verdict = "near_zero"
        print(f"  Effect already near zero without confounders")

    print()
    print("  OVERALL VERDICT:")
    if portability_verdict == "rejects_SCM1" and stability_verdict in ("rejects_SCM1", "near_zero"):
        print("  Both tests reject SCM_1. Data are consistent with SCM_0:")
        print("  text has no direct causal effect on price.")
        overall = "SCM_0"
    elif portability_verdict == "supports_SCM1" and stability_verdict == "supports_SCM1":
        print("  Both tests support SCM_1. Evidence for a direct T->Y effect.")
        overall = "SCM_1"
    else:
        print("  Mixed evidence. Neither model is clearly favored.")
        overall = "ambiguous"

    return {
        "portability": portability_verdict,
        "stability": stability_verdict,
        "overall": overall,
        "mean_self_r2": mean_self,
        "mean_transfer_r2": mean_transfer,
        "ate_no_conf": ate_no_conf,
        "ate_with_conf": ate_with_conf,
    }


def cate_by_property_type(T, L, Y, df):
    print("\n" + "="*60)
    print("6. CATE BY PROPERTY TYPE / PRICE SEGMENT")
    print("="*60)

    n_pca = min(20, T.shape[1], len(T) - 2)
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)

    scaler_t = StandardScaler()
    T_s = scaler_t.fit_transform(T_pca)
    scaler_l = StandardScaler()
    L_s = scaler_l.fit_transform(L)

    T_norm = np.linalg.norm(T_s, axis=1)
    treatment = (T_norm > np.median(T_norm)).astype(float)

    n_quantiles = 4
    quantile_edges = np.percentile(Y, np.linspace(0, 100, n_quantiles + 1))
    quantile_labels = np.digitize(Y, quantile_edges[1:-1])

    segment_names = ["Budget", "Mid-Low", "Mid-High", "Luxury"]

    results = []
    for q in range(n_quantiles):
        mask = quantile_labels == q
        n_q = mask.sum()
        if n_q < 20:
            print(f"  {segment_names[q]}: n={n_q} (too few)")
            results.append({"segment": segment_names[q], "n": n_q, "ate": np.nan})
            continue

        Y_q, D_q, C_q = Y[mask], treatment[mask], L_s[mask]

        if len(np.unique(D_q)) < 2:
            print(f"  {segment_names[q]}: n={n_q}, no treatment variation")
            results.append({"segment": segment_names[q], "n": n_q, "ate": np.nan})
            continue

        outcome = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42,
        )
        outcome.fit(np.hstack([D_q.reshape(-1, 1), C_q]), Y_q)

        propensity = LogisticRegression(max_iter=1000, random_state=42)
        propensity.fit(C_q, D_q)
        e_q = np.clip(propensity.predict_proba(C_q)[:, 1], 0.05, 0.95)

        mu1 = outcome.predict(np.hstack([np.ones((n_q, 1)), C_q]))
        mu0 = outcome.predict(np.hstack([np.zeros((n_q, 1)), C_q]))

        ate = np.mean(
            mu1 - mu0
            + D_q * (Y_q - mu1) / e_q
            - (1 - D_q) * (Y_q - mu0) / (1 - e_q)
        )

        from scipy.stats import bootstrap as bs
        def stat(idx, d=D_q, y=Y_q, m1=mu1, m0=mu0, ps=e_q):
            i = idx[0]
            return np.mean(m1[i] - m0[i] + d[i]*(y[i]-m1[i])/ps[i] - (1-d[i])*(y[i]-m0[i])/(1-ps[i]))

        rng = np.random.default_rng(42 + q)
        ci = bs((np.arange(n_q),), stat, n_resamples=500, random_state=rng, method="percentile")
        contains_zero = ci.confidence_interval.low <= 0 <= ci.confidence_interval.high

        price_lo = np.exp(quantile_edges[q])
        price_hi = np.exp(quantile_edges[q + 1])
        print(f"  {segment_names[q]} (${price_lo:,.0f}-${price_hi:,.0f}): n={n_q}, "
              f"ATE={ate:.4f} [{ci.confidence_interval.low:.4f}, {ci.confidence_interval.high:.4f}] "
              f"{'(zero in CI)' if contains_zero else '*** SIGNIFICANT ***'}")

        results.append({
            "segment": segment_names[q], "n": n_q, "ate": ate,
            "ci_low": ci.confidence_interval.low, "ci_high": ci.confidence_interval.high,
        })

    any_sig = any(
        not np.isnan(r.get("ate", np.nan))
        and not (r.get("ci_low", -1) <= 0 <= r.get("ci_high", 1))
        for r in results
    )
    if any_sig:
        print("\n  -> HETEROGENEOUS EFFECTS DETECTED: text may matter in some segments")
    else:
        print("\n  -> No significant effects in any price segment")

    return results


def main():
    cities = sys.argv[1:] if len(sys.argv) > 1 else ["nyc", "sf"]
    cities_data = {}

    for city in cities:
        try:
            T, L, Y, df = load_city(city)
            cities_data[city] = (T, L, Y, df)
            print(f"Loaded {city}: n={len(Y)}")
        except Exception as e:
            print(f"Could not load {city}: {e}")

    if not cities_data:
        print("No data loaded")
        return

    primary = list(cities_data.keys())[0]
    T, L, Y, df = cities_data[primary]

    print(f"\n{'#'*60}")
    print(f"EXTENDED ANALYSIS: {primary.upper()}")
    print(f"{'#'*60}")

    rich_conf = df.attrs.get("rich_confounders", None) if hasattr(df, "attrs") else None
    test_conditional_independence(T, L, Y, rich_confounders=rich_conf)
    partial_r2_decomposition(T, L, Y)
    cinelli_hazlett_sensitivity(T, L, Y)
    cate_by_property_type(T, L, Y, df)

    if len(cities_data) >= 2:
        cross_market_transfer(cities_data)
        competing_scm_test(cities_data)

    if "clean_description" in df.columns:
        embedding_ablation(df, T, L, Y)

    print(f"\n{'#'*60}")
    print("EXTENDED ANALYSIS COMPLETE")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()

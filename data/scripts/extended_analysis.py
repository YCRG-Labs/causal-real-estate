import sys
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from config import PROCESSED_DIR, EMBEDDING_DIM


def load_city(city):
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

    return T, L, Y, df


def test_conditional_independence(T, L, Y):
    print("\n" + "="*60)
    print("1. CONDITIONAL INDEPENDENCE TESTS")
    print("="*60)

    n_pca = min(30, T.shape[1], len(T) - 2)
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)

    scaler_t = StandardScaler()
    T_s = scaler_t.fit_transform(T_pca)
    scaler_l = StandardScaler()
    L_s = scaler_l.fit_transform(L)

    reg_Y_L = LinearRegression().fit(L_s, Y)
    resid_Y = Y - reg_Y_L.predict(L_s)

    reg_T_L = LinearRegression().fit(L_s, T_s)
    resid_T = T_s - reg_T_L.predict(L_s)

    partial_corrs = []
    for j in range(resid_T.shape[1]):
        r, _ = spearmanr(resid_Y, resid_T[:, j])
        partial_corrs.append(abs(r))

    mean_partial = np.mean(partial_corrs)
    max_partial = np.max(partial_corrs)

    print(f"  Partial correlation (Y, T | L):")
    print(f"    Mean |rho|: {mean_partial:.4f}")
    print(f"    Max |rho|:  {max_partial:.4f}")

    n_permutations = 500
    null_means = []
    for p in range(n_permutations):
        perm = np.random.RandomState(p).permutation(len(resid_Y))
        perm_corrs = [abs(spearmanr(resid_Y[perm], resid_T[:, j])[0]) for j in range(min(10, resid_T.shape[1]))]
        null_means.append(np.mean(perm_corrs))

    p_value = np.mean(np.array(null_means) >= mean_partial)
    print(f"    Permutation p-value: {p_value:.4f}")

    if p_value > 0.05:
        print("    -> CONSISTENT with conditional independence (Y indep T | L)")
    else:
        print("    -> REJECTS conditional independence at alpha=0.05")

    return mean_partial, max_partial, p_value


def cross_market_transfer(cities_data):
    print("\n" + "="*60)
    print("2. CROSS-MARKET TRANSFER TEST")
    print("="*60)

    city_names = list(cities_data.keys())
    results = []

    for train_city in city_names:
        T_tr, L_tr, Y_tr, _ = cities_data[train_city]

        n_pca = min(20, T_tr.shape[1], len(T_tr) - 2)
        pca = PCA(n_components=n_pca, random_state=42)
        T_tr_pca = pca.fit_transform(T_tr)

        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        model.fit(T_tr_pca, Y_tr)

        r2_self = model.score(T_tr_pca, Y_tr)
        print(f"\n  Train: {train_city} (n={len(Y_tr)}, self R2={r2_self:.4f})")

        for test_city in city_names:
            if test_city == train_city:
                continue
            T_te, L_te, Y_te, _ = cities_data[test_city]
            T_te_pca = pca.transform(T_te)
            r2_transfer = model.score(T_te_pca, Y_te)
            print(f"    -> {test_city}: R2={r2_transfer:.4f}")
            results.append((train_city, test_city, r2_self, r2_transfer))

    print("\n  Transfer summary:")
    for train_c, test_c, r2_s, r2_t in results:
        drop = r2_s - r2_t
        print(f"    {train_c} -> {test_c}: self={r2_s:.4f}, transfer={r2_t:.4f}, drop={drop:.4f}")

    mean_transfer = np.mean([r[3] for r in results])
    mean_self = np.mean([r[2] for r in results])
    print(f"\n  Mean self R2: {mean_self:.4f}")
    print(f"  Mean transfer R2: {mean_transfer:.4f}")

    if mean_transfer < 0.1:
        print("  -> Text models FAIL to transfer: consistent with location encoding")
    elif mean_transfer < mean_self * 0.5:
        print("  -> Substantial transfer degradation: mostly location-specific signal")
    else:
        print("  -> Transfer partially succeeds: some location-independent signal")

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

    test_conditional_independence(T, L, Y)
    partial_r2_decomposition(T, L, Y)
    cinelli_hazlett_sensitivity(T, L, Y)

    if len(cities_data) >= 2:
        cross_market_transfer(cities_data)

    if "clean_description" in df.columns:
        embedding_ablation(df, T, L, Y)

    print(f"\n{'#'*60}")
    print("EXTENDED ANALYSIS COMPLETE")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()

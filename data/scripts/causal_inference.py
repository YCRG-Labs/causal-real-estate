import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from scipy.stats import bootstrap
from config import PROCESSED_DIR, CITIES, EMBEDDING_DIM

CENSUS_COLS = [
    "median_household_income", "pct_bachelors", "pct_white", "pct_black",
    "pct_asian", "pct_hispanic", "median_home_value", "median_gross_rent",
    "labor_force_participation", "pct_under_25", "pct_over_60",
]
CRIME_COLS = ["crime_violent", "crime_property", "crime_quality_of_life", "crime_total"]
AMENITY_COLS = [
    "amenity_food_dining", "amenity_retail", "amenity_services",
    "amenity_recreation", "amenity_transportation", "amenity_education",
    "amenity_total", "amenity_diversity",
]
MICRO_GEO_COLS = [
    "dist_park_m", "dist_transit_m", "dist_school_m",
    "dist_restaurant_m", "dist_retail_m", "dist_medical_m",
]
PROPERTY_COLS = ["bedrooms", "bldg_area_sqft", "lot_area_sqft", "year_built"]
CONTEXTUAL_COLS = CENSUS_COLS + CRIME_COLS + AMENITY_COLS + MICRO_GEO_COLS


def load_analysis_data(city, embedding_model=None):
    if embedding_model:
        safe_name = embedding_model.replace("/", "_").replace("-", "_")
        emb_path = PROCESSED_DIR / f"{city}_embeddings_{safe_name}.parquet"
    else:
        emb_path = PROCESSED_DIR / f"{city}_embeddings.parquet"

    micro_geo_path = PROCESSED_DIR / f"{city}_parcels_micro_geo.gpkg"
    amenities_path = PROCESSED_DIR / f"{city}_parcels_amenities.gpkg"

    if not emb_path.exists():
        return None

    import geopandas as gpd
    emb_df = pd.read_parquet(emb_path)

    if micro_geo_path.exists():
        parcels = gpd.read_file(micro_geo_path, layer=city)
    elif amenities_path.exists():
        parcels = gpd.read_file(amenities_path, layer=city)
    else:
        parcels = None

    return emb_df, parcels


def _spatial_join_parcels(emb_df, parcels):
    lat = pd.to_numeric(emb_df.get("latitude", pd.Series(dtype=float)), errors="coerce").values
    lon = pd.to_numeric(emb_df.get("longitude", pd.Series(dtype=float)), errors="coerce").values
    valid_emb = ~(np.isnan(lat) | np.isnan(lon))

    if hasattr(parcels, "geometry") and parcels.geometry is not None:
        if parcels.crs and not parcels.crs.is_projected:
            centroids = parcels.geometry.centroid
        else:
            centroids = parcels.geometry.centroid
        p_lat = centroids.y.values.astype(float)
        p_lon = centroids.x.values.astype(float)
    elif "latitude" in parcels.columns and "longitude" in parcels.columns:
        p_lat = pd.to_numeric(parcels["latitude"], errors="coerce").values
        p_lon = pd.to_numeric(parcels["longitude"], errors="coerce").values
    else:
        return None

    valid_parcels = ~(np.isnan(p_lat) | np.isnan(p_lon))
    if valid_parcels.sum() == 0 or valid_emb.sum() == 0:
        return None

    tree = cKDTree(np.column_stack([p_lat[valid_parcels], p_lon[valid_parcels]]))
    _, nn_idx = tree.query(np.column_stack([lat[valid_emb], lon[valid_emb]]))
    parcel_valid_idx = np.where(valid_parcels)[0]
    matched_idx = parcel_valid_idx[nn_idx]

    n = len(emb_df)
    result = {}

    for col_group, cols in [("property", PROPERTY_COLS), ("contextual", CONTEXTUAL_COLS)]:
        available = [c for c in cols if c in parcels.columns]
        if not available:
            continue
        arr = np.full((n, len(available)), np.nan)
        for j, col in enumerate(available):
            vals = pd.to_numeric(parcels[col], errors="coerce").values
            arr[np.where(valid_emb)[0], j] = vals[matched_idx]
        result[col_group] = arr
        result[f"{col_group}_cols"] = available

    if "median_household_income" in parcels.columns:
        income = np.full(n, np.nan)
        vals = pd.to_numeric(parcels["median_household_income"], errors="coerce").values
        income[np.where(valid_emb)[0]] = vals[matched_idx]
        result["income"] = income

    return result


def get_features_and_target(emb_df, parcels, drop_mismatched_crime=False):
    """
    Builds the (T, confounders, Y, meta) tuple for the causal estimators.

    drop_mismatched_crime:
      When True, crime_* columns are excluded from the confounder set if the
      parcels file flags `crime_temporal_match=False` for the majority of
      rows. This is the conservative spec for cities (e.g., NYC at the time
      of writing) where the crime extract does not overlap the sale period.
    """
    emb_cols = [f"emb_{i}" for i in range(EMBEDDING_DIM)]
    available_emb = [c for c in emb_cols if c in emb_df.columns]
    T = emb_df[available_emb].values

    if "price" in emb_df.columns:
        Y = pd.to_numeric(emb_df["price"], errors="coerce").values
    elif parcels is not None and "sale_price" in parcels.columns:
        Y = parcels["sale_price"].values[:len(T)].astype(float)
    else:
        return None

    crime_temporal_ok = True
    if parcels is not None and "crime_temporal_match" in parcels.columns:
        match_rate = pd.to_numeric(parcels["crime_temporal_match"], errors="coerce").mean()
        crime_temporal_ok = bool(match_rate >= 0.5)
        if not crime_temporal_ok:
            print(f"  ⚠ Crime data temporal mismatch ({match_rate*100:.0f}% of parcels in window)")
            if drop_mismatched_crime:
                print(f"  → dropping crime_* features from confounder set")

    lat = pd.to_numeric(emb_df.get("latitude", pd.Series(dtype=float)), errors="coerce").values
    lon = pd.to_numeric(emb_df.get("longitude", pd.Series(dtype=float)), errors="coerce").values

    if "zip" in emb_df.columns:
        zips = emb_df["zip"].fillna(0).astype(float).astype(int).astype(str)
        le = LabelEncoder()
        zip_labels = le.fit_transform(zips)
    else:
        zip_labels = np.zeros(len(T), dtype=int)

    joined = None
    if parcels is not None:
        try:
            joined = _spatial_join_parcels(emb_df, parcels)
        except Exception as e:
            print(f"    Warning: spatial join failed ({e}), falling back to zip-only")

    if joined and drop_mismatched_crime and not crime_temporal_ok and "contextual" in joined:
        ctx_cols = joined["contextual_cols"]
        keep_idx = [i for i, c in enumerate(ctx_cols) if not c.startswith("crime_")]
        joined["contextual"] = joined["contextual"][:, keep_idx]
        joined["contextual_cols"] = [ctx_cols[i] for i in keep_idx]

    confounder_parts = []

    if not np.all(np.isnan(lat)) and not np.all(np.isnan(lon)):
        confounder_parts.append(np.column_stack([lat, lon]))

    if joined and "property" in joined:
        confounder_parts.append(joined["property"])

    if joined and "contextual" in joined:
        confounder_parts.append(joined["contextual"])

    if not confounder_parts:
        n_bins = min(len(np.unique(zip_labels)), 20)
        L_onehot = np.zeros((len(T), n_bins))
        for i, z in enumerate(zip_labels):
            L_onehot[i, z % n_bins] = 1.0
        confounder_parts.append(L_onehot)

    confounders = np.hstack(confounder_parts)

    income = joined["income"] if joined and "income" in joined else np.zeros(len(T))

    valid = ~(np.isnan(Y) | np.isinf(Y) | (Y <= 0))
    nan_conf = np.isnan(confounders)
    col_nan_rate = nan_conf.mean(axis=0)
    good_cols = col_nan_rate < 0.5
    confounders = confounders[:, good_cols]
    np.nan_to_num(confounders, copy=False, nan=0.0)

    valid &= ~np.all(confounders == 0, axis=1) | True

    T = T[valid]
    confounders = confounders[valid]
    Y = np.log(Y[valid])
    zip_labels = zip_labels[valid]
    lat_v = lat[valid]
    lon_v = lon[valid]
    income_v = income[valid]

    meta = {
        "zip_labels": zip_labels,
        "lat": lat_v,
        "lon": lon_v,
        "income": income_v,
        "n_confounders": confounders.shape[1],
        "has_rich_confounders": joined is not None and "contextual" in joined,
        "crime_temporal_ok": crime_temporal_ok,
        "crime_dropped": (drop_mismatched_crime and not crime_temporal_ok),
    }

    return T, confounders, Y, meta


def backdoor_adjustment(T, confounders, Y, n_pca=50):
    print("\n  [1] Backdoor Adjustment")
    n_pca = min(n_pca, T.shape[1], T.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)

    scaler_c = StandardScaler()
    conf_s = scaler_c.fit_transform(confounders)

    scaler_t = StandardScaler()
    T_s = scaler_t.fit_transform(T_pca)

    full_features = np.hstack([T_s, conf_s])
    conf_only = conf_s

    max_features = min(0.5, 10.0 / full_features.shape[1]) if full_features.shape[1] > 20 else 0.8

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    r2_full, r2_conf = [], []
    for train_idx, test_idx in kf.split(Y):
        model_full = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, max_features=max_features, random_state=42,
        )
        model_full.fit(full_features[train_idx], Y[train_idx])
        r2_full.append(model_full.score(full_features[test_idx], Y[test_idx]))

        model_conf = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        model_conf.fit(conf_only[train_idx], Y[train_idx])
        r2_conf.append(model_conf.score(conf_only[test_idx], Y[test_idx]))

    r2_full_mean = np.mean(r2_full)
    r2_conf_mean = np.mean(r2_conf)
    delta_r2 = r2_full_mean - r2_conf_mean

    print(f"    R² (confounders only):     {r2_conf_mean:.4f}")
    print(f"    R² (confounders + text):   {r2_full_mean:.4f}")
    print(f"    ΔR² (text contribution):   {delta_r2:.4f}")

    return delta_r2


def doubly_robust_estimation(T, confounders, Y, n_pca=50):
    """
    Doubly-robust ATE for a binarized text treatment.

    Reports BOTH bootstrap CI (which is wide and well-known to be conservative
    in moderate samples) AND influence-function (IF) SE plus minimum detectable
    effect at 80% power. The MDE makes "CI contains zero" interpretable by
    saying what effect sizes the test could and could not have detected.
    """
    print("\n  [2] Doubly-Robust Estimation (binarized treatment)")
    n_pca = min(n_pca, T.shape[1], T.shape[0])
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)

    scaler = StandardScaler()
    conf_s = scaler.fit_transform(confounders)

    T_norm = np.linalg.norm(T_pca, axis=1)
    T_median = np.median(T_norm)
    treatment = (T_norm > T_median).astype(float)

    outcome_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42,
    )
    outcome_model.fit(
        np.hstack([treatment.reshape(-1, 1), conf_s]),
        Y,
    )

    propensity_model = LogisticRegression(max_iter=1000, random_state=42)
    propensity_model.fit(conf_s, treatment)
    e = propensity_model.predict_proba(conf_s)[:, 1]
    e = np.clip(e, 0.05, 0.95)

    mu1 = outcome_model.predict(np.hstack([np.ones((len(Y), 1)), conf_s]))
    mu0 = outcome_model.predict(np.hstack([np.zeros((len(Y), 1)), conf_s]))

    psi = (
        mu1 - mu0
        + treatment * (Y - mu1) / e
        - (1 - treatment) * (Y - mu0) / (1 - e)
    )
    dr_effect = float(np.mean(psi))

    n = len(Y)
    if_var = float(np.var(psi - dr_effect, ddof=1)) / n
    if_se = float(np.sqrt(if_var))
    if_ci_low = dr_effect - 1.96 * if_se
    if_ci_high = dr_effect + 1.96 * if_se

    z_alpha = 1.96
    z_beta = 0.84
    mde = (z_alpha + z_beta) * if_se

    def dr_statistic(indices):
        idx = indices[0]
        t, y, m1, m0, ps = treatment[idx], Y[idx], mu1[idx], mu0[idx], e[idx]
        return np.mean(
            m1 - m0 + t * (y - m1) / ps - (1 - t) * (y - m0) / (1 - ps)
        )

    rng = np.random.default_rng(42)
    ci = bootstrap(
        (np.arange(len(Y)),),
        dr_statistic,
        n_resamples=1000,
        random_state=rng,
        method="percentile",
    )
    boot_low = float(ci.confidence_interval.low)
    boot_high = float(ci.confidence_interval.high)

    print(f"    DR estimate (ATE): {dr_effect:.4f}")
    print(f"    Influence-function SE: {if_se:.4f}")
    print(f"    95% CI (IF):       [{if_ci_low:.4f}, {if_ci_high:.4f}]")
    print(f"    95% CI (bootstrap): [{boot_low:.4f}, {boot_high:.4f}]  ← wider, percentile method")
    print(f"    Min detectable effect (80% power, two-sided 5%): ±{mde:.4f}")
    print(f"      i.e., this test cannot rule out true effects with |τ| < {mde:.3f} log-points")
    print(f"      ({100*(np.exp(mde)-1):.1f}% in price terms)")

    if if_ci_low <= 0 <= if_ci_high:
        print("    → IF CI contains zero: no significant causal effect of text")
    else:
        print("    → IF CI excludes zero: significant effect detected")

    return dr_effect, (boot_low, boot_high), {
        "if_se": if_se,
        "if_ci": (if_ci_low, if_ci_high),
        "mde": mde,
    }


def dml_continuous_treatment(T, confounders, Y, n_pca=50, k_folds=5):
    """
    Double Machine Learning estimate of the partial effect of the first text
    principal component on log-price, after partialling out confounders via
    cross-fitted gradient boosting.

    This is a "continuous treatment" alternative to the binarized DR and
    avoids the awkward "median PCA norm" treatment definition. The estimand
    is: per-unit-σ change in PC1 of the text embedding → expected change in
    log-price, holding confounders fixed.

    Returns: theta (effect per σ of PC1), SE (Neyman-orthogonal IF SE).
    """
    print("\n  [2b] DML Continuous Treatment (PC1 of text)")
    n_pca = min(n_pca, T.shape[1], T.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)

    pc1 = T_pca[:, 0]
    pc1 = (pc1 - pc1.mean()) / (pc1.std() if pc1.std() > 0 else 1.0)

    scaler = StandardScaler()
    conf_s = scaler.fit_transform(confounders)

    n = len(Y)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    Y_resid = np.zeros(n)
    T_resid = np.zeros(n)

    for tr, te in kf.split(np.arange(n)):
        m_y = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
        )
        m_y.fit(conf_s[tr], Y[tr])
        Y_resid[te] = Y[te] - m_y.predict(conf_s[te])

        m_t = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
        )
        m_t.fit(conf_s[tr], pc1[tr])
        T_resid[te] = pc1[te] - m_t.predict(conf_s[te])

    denom = float(np.mean(T_resid ** 2))
    if denom < 1e-12:
        print("    Treatment fully explained by confounders; effect undefined")
        return None

    theta = float(np.mean(T_resid * Y_resid)) / denom

    psi = (Y_resid - theta * T_resid) * T_resid / denom
    var_theta = float(np.var(psi, ddof=1)) / n
    se = float(np.sqrt(var_theta))
    ci_low = theta - 1.96 * se
    ci_high = theta + 1.96 * se

    z_alpha = 1.96
    z_beta = 0.84
    mde = (z_alpha + z_beta) * se

    print(f"    DML θ (per σ of PC1):    {theta:.4f}")
    print(f"    Neyman-orthogonal SE:    {se:.4f}")
    print(f"    95% CI:                  [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"    Min detectable effect:   ±{mde:.4f} ({100*(np.exp(mde)-1):.1f}% in price)")

    if ci_low <= 0 <= ci_high:
        print("    → CI contains zero: no significant continuous effect")
    else:
        print("    → CI excludes zero: significant continuous effect")

    return {
        "theta": theta,
        "se": se,
        "ci": (ci_low, ci_high),
        "mde": mde,
    }


def cate_by_price_quantile(T, confounders, Y, n_quantiles=4, n_pca=50):
    print(f"\n  [5] CATE by Price Quantile (Q={n_quantiles})")
    n_pca = min(n_pca, T.shape[1], T.shape[0])
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)

    scaler = StandardScaler()
    conf_s = scaler.fit_transform(confounders)

    T_norm = np.linalg.norm(T_pca, axis=1)
    T_median = np.median(T_norm)
    treatment = (T_norm > T_median).astype(float)

    quantile_edges = np.percentile(Y, np.linspace(0, 100, n_quantiles + 1))
    quantile_labels = np.digitize(Y, quantile_edges[1:-1])

    results = []
    for q in range(n_quantiles):
        mask = quantile_labels == q
        n_q = mask.sum()
        if n_q < 30:
            print(f"    Q{q+1}: n={n_q} (too few, skipping)")
            results.append({"quantile": q + 1, "n": n_q, "ate": np.nan, "ci_low": np.nan, "ci_high": np.nan})
            continue

        Y_q = Y[mask]
        treat_q = treatment[mask]
        conf_q = conf_s[mask]

        outcome_q = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42,
        )
        outcome_q.fit(np.hstack([treat_q.reshape(-1, 1), conf_q]), Y_q)

        prop_q = LogisticRegression(max_iter=1000, random_state=42)
        if len(np.unique(treat_q)) < 2:
            print(f"    Q{q+1}: n={n_q}, no treatment variation, skipping")
            results.append({"quantile": q + 1, "n": n_q, "ate": np.nan, "ci_low": np.nan, "ci_high": np.nan})
            continue

        prop_q.fit(conf_q, treat_q)
        e_q = np.clip(prop_q.predict_proba(conf_q)[:, 1], 0.05, 0.95)

        mu1_q = outcome_q.predict(np.hstack([np.ones((n_q, 1)), conf_q]))
        mu0_q = outcome_q.predict(np.hstack([np.zeros((n_q, 1)), conf_q]))

        ate_q = np.mean(
            mu1_q - mu0_q
            + treat_q * (Y_q - mu1_q) / e_q
            - (1 - treat_q) * (Y_q - mu0_q) / (1 - e_q)
        )

        def dr_stat_q(indices, t=treat_q, y=Y_q, m1=mu1_q, m0=mu0_q, ps=e_q):
            idx = indices[0]
            return np.mean(
                m1[idx] - m0[idx]
                + t[idx] * (y[idx] - m1[idx]) / ps[idx]
                - (1 - t[idx]) * (y[idx] - m0[idx]) / (1 - ps[idx])
            )

        rng = np.random.default_rng(42 + q)
        ci = bootstrap(
            (np.arange(n_q),), dr_stat_q,
            n_resamples=500, random_state=rng, method="percentile",
        )

        price_lo = np.exp(quantile_edges[q])
        price_hi = np.exp(quantile_edges[q + 1])
        contains_zero = ci.confidence_interval.low <= 0 <= ci.confidence_interval.high

        print(f"    Q{q+1} (${price_lo:,.0f}-${price_hi:,.0f}): n={n_q}, ATE={ate_q:.4f} "
              f"[{ci.confidence_interval.low:.4f}, {ci.confidence_interval.high:.4f}] "
              f"{'(zero in CI)' if contains_zero else '(SIGNIFICANT)'}")

        results.append({
            "quantile": q + 1,
            "n": n_q,
            "ate": ate_q,
            "ci_low": ci.confidence_interval.low,
            "ci_high": ci.confidence_interval.high,
            "price_range": (price_lo, price_hi),
        })

    return results


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MultiHeadDiscriminator(nn.Module):
    def __init__(self, input_dim=64, n_zips=10, n_income_bins=5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.zip_head = nn.Linear(64, n_zips)
        self.geo_head = nn.Linear(64, 2)
        self.income_head = nn.Linear(64, n_income_bins)

    def forward(self, x):
        h = self.shared(x)
        return self.zip_head(h), self.geo_head(h), self.income_head(h)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def _frozen_encoder_probe(encoder, T_tensor, lat, lon, zip_labels, income):
    """
    After adversarial training, freeze the encoder and train a fresh,
    independent probe classifier on the deconfounded representation.

    The discriminator's accuracy at training time can be misleading: gradient
    reversal can drive it to random *during* training while the underlying
    representation still contains location information that a fresh probe
    can recover. This probe is the cleanest test of whether location was
    actually removed.

    Returns a dict of probe scores. If they exceed the in-training
    discriminator accuracy, the adversarial training "fooled" the live
    discriminator but the location signal is still in the representation.
    """
    encoder.eval()
    with torch.no_grad():
        z = encoder(T_tensor).numpy()

    n = len(z)
    split = int(n * 0.7)
    perm = np.random.RandomState(123).permutation(n)
    tr, te = perm[:split], perm[split:]

    probes = {}

    le_zip = LabelEncoder()
    zip_enc = le_zip.fit_transform(zip_labels)
    if len(np.unique(zip_enc[tr])) >= 2:
        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(z[tr], zip_enc[tr])
        probes["zip_probe_acc"] = float(clf.score(z[te], zip_enc[te]))
        probes["zip_random"] = 1.0 / len(np.unique(zip_enc))

    inc_clean = np.nan_to_num(income, nan=np.nanmedian(income) if np.any(~np.isnan(income)) else 0)
    if np.std(inc_clean) > 0:
        try:
            inc_q = pd.qcut(inc_clean, q=5, labels=False, duplicates="drop")
            if len(np.unique(inc_q[tr])) >= 2:
                clf = LogisticRegression(max_iter=2000, random_state=42)
                clf.fit(z[tr], inc_q[tr])
                probes["income_probe_acc"] = float(clf.score(z[te], inc_q[te]))
                probes["income_random"] = 1.0 / len(np.unique(inc_q))
        except Exception:
            pass

    coords = np.column_stack([
        np.nan_to_num(lat, nan=0.0),
        np.nan_to_num(lon, nan=0.0),
    ])
    coord_scaler = StandardScaler()
    coords_s = coord_scaler.fit_transform(coords)
    if coords_s[tr].shape[0] >= 10:
        from sklearn.linear_model import Ridge as _Ridge
        ridge_lat = _Ridge(alpha=1.0).fit(z[tr], coords_s[tr, 0])
        ridge_lon = _Ridge(alpha=1.0).fit(z[tr], coords_s[tr, 1])
        pred = np.column_stack([
            ridge_lat.predict(z[te]),
            ridge_lon.predict(z[te]),
        ])
        ss_res = float(np.sum((coords_s[te] - pred) ** 2))
        ss_tot = float(np.sum((coords_s[te] - coords_s[te].mean(axis=0)) ** 2))
        probes["geo_probe_r2"] = float(1 - ss_res / max(ss_tot, 1e-8))

    return probes


def adversarial_deconfounding(T, Y, meta, n_pca=50, epochs=150, lr=1e-3):
    print("\n  [3] Adversarial Deconfounding (Multi-Head + Frozen Probe)")

    n_pca = min(n_pca, T.shape[1], T.shape[0])
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)

    zip_labels = meta["zip_labels"]
    lat = meta["lat"]
    lon = meta["lon"]
    income = meta["income"]

    le_zip = LabelEncoder()
    zip_enc = le_zip.fit_transform(zip_labels)
    n_zips = len(le_zip.classes_)

    n_income_bins = 5
    income_clean = np.nan_to_num(income, nan=np.nanmedian(income) if np.any(~np.isnan(income)) else 0)
    if np.std(income_clean) > 0:
        income_quantiles = pd.qcut(income_clean, q=n_income_bins, labels=False, duplicates="drop")
    else:
        income_quantiles = np.zeros(len(income_clean), dtype=int)
    n_income_bins = len(np.unique(income_quantiles))

    geo_targets = np.column_stack([
        np.nan_to_num(lat, nan=0.0),
        np.nan_to_num(lon, nan=0.0),
    ])
    geo_scaler = StandardScaler()
    geo_targets = geo_scaler.fit_transform(geo_targets)

    scaler_t = StandardScaler()
    T_s = scaler_t.fit_transform(T_pca)
    scaler_y = StandardScaler()
    Y_s = scaler_y.fit_transform(Y.reshape(-1, 1)).ravel()

    n = len(T_s)
    split = int(n * 0.7)
    perm = np.random.RandomState(42).permutation(n)
    tr, te = perm[:split], perm[split:]

    T_train = torch.FloatTensor(T_s[tr])
    Y_train = torch.FloatTensor(Y_s[tr])
    zip_train = torch.LongTensor(zip_enc[tr])
    geo_train = torch.FloatTensor(geo_targets[tr])
    inc_train = torch.LongTensor(income_quantiles[tr])

    T_test = torch.FloatTensor(T_s[te])
    Y_test = torch.FloatTensor(Y_s[te])
    zip_test = torch.LongTensor(zip_enc[te])
    geo_test = torch.FloatTensor(geo_targets[te])
    inc_test = torch.LongTensor(income_quantiles[te])

    encoder = Encoder(n_pca, 128, 64)
    predictor = Predictor(64)
    discriminator = MultiHeadDiscriminator(64, n_zips, n_income_bins)

    opt_enc_pred = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=1e-4)

    pred_loss_fn = nn.MSELoss()
    zip_loss_fn = nn.CrossEntropyLoss()
    geo_loss_fn = nn.MSELoss()
    inc_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        if epoch <= 10:
            lam = 0.1
        elif epoch <= 50:
            lam = 0.1 + 0.9 * (epoch - 10) / 40
        else:
            lam = 1.0

        encoder.train(); predictor.train(); discriminator.train()

        z = encoder(T_train)
        y_pred = predictor(z)
        pred_loss = pred_loss_fn(y_pred, Y_train)

        z_rev = GradientReversal.apply(z, lam)
        zip_logits, geo_pred, inc_logits = discriminator(z_rev)
        disc_loss = (
            zip_loss_fn(zip_logits, zip_train)
            + geo_loss_fn(geo_pred, geo_train)
            + inc_loss_fn(inc_logits, inc_train)
        ) / 3.0

        total_loss = pred_loss + disc_loss
        opt_enc_pred.zero_grad()
        total_loss.backward()
        opt_enc_pred.step()

        z_det = encoder(T_train).detach()
        zip_l2, geo_l2, inc_l2 = discriminator(z_det)
        disc_loss2 = (
            zip_loss_fn(zip_l2, zip_train)
            + geo_loss_fn(geo_l2, geo_train)
            + inc_loss_fn(inc_l2, inc_train)
        ) / 3.0
        opt_disc.zero_grad()
        disc_loss2.backward()
        opt_disc.step()

    encoder.eval(); predictor.eval(); discriminator.eval()

    with torch.no_grad():
        z = encoder(T_test)
        y_pred = predictor(z)
        pred_r2 = 1 - torch.mean((Y_test - y_pred) ** 2).item() / torch.var(Y_test).item()

        zip_logits, geo_pred, inc_logits = discriminator(z)
        zip_acc = (zip_logits.argmax(dim=1) == zip_test).float().mean().item()
        geo_r2 = 1 - torch.mean((geo_pred - geo_test) ** 2).item() / max(torch.var(geo_test).item(), 1e-8)
        inc_acc = (inc_logits.argmax(dim=1) == inc_test).float().mean().item()

        zip_random = 1.0 / max(n_zips, 1)
        inc_random = 1.0 / max(n_income_bins, 1)

    print(f"    Predictor R² (deconfounded): {pred_r2:.4f}")
    print(f"    Live discriminator (during training):")
    print(f"      Zip head accuracy:    {zip_acc:.4f} (random: {zip_random:.4f})")
    print(f"      Geo head R²:          {geo_r2:.4f}")
    print(f"      Income head accuracy: {inc_acc:.4f} (random: {inc_random:.4f})")

    print(f"\n    Frozen-encoder probe (independent fresh classifier):")
    T_all = torch.FloatTensor(T_s)
    probes = _frozen_encoder_probe(encoder, T_all, lat, lon, zip_labels, income)
    if "zip_probe_acc" in probes:
        ratio = probes["zip_probe_acc"] / max(probes["zip_random"], 1e-8)
        print(f"      Zip probe acc:    {probes['zip_probe_acc']:.4f} "
              f"(random: {probes['zip_random']:.4f}, ratio: {ratio:.1f}x)")
    if "income_probe_acc" in probes:
        ratio = probes["income_probe_acc"] / max(probes["income_random"], 1e-8)
        print(f"      Income probe acc: {probes['income_probe_acc']:.4f} "
              f"(random: {probes['income_random']:.4f}, ratio: {ratio:.1f}x)")
    if "geo_probe_r2" in probes:
        print(f"      Geo probe R²:     {probes['geo_probe_r2']:.4f}")

    live_random = (zip_acc < zip_random * 1.5 and geo_r2 < 0.1 and inc_acc < inc_random * 1.5)
    probe_random = True
    if "zip_probe_acc" in probes and probes["zip_probe_acc"] > probes["zip_random"] * 1.5:
        probe_random = False
    if "geo_probe_r2" in probes and probes["geo_probe_r2"] > 0.1:
        probe_random = False
    if "income_probe_acc" in probes and probes["income_probe_acc"] > probes["income_random"] * 1.5:
        probe_random = False

    print()
    if live_random and probe_random:
        print("    → Location successfully removed (live discriminator AND frozen probe random).")
        print("      Residual predictor R² is location-INDEPENDENT — small SCM_1 refinement plausible.")
    elif live_random and not probe_random:
        print("    → ⚠ Live discriminator looks random but frozen probe recovers location.")
        print("      Residual predictor R² is location-DEPENDENT — adversarial training was fooled.")
        print("      The R²=0.898 in SF is more consistent with subtle residual location encoding")
        print("      than with a genuine semantic effect.")
    else:
        print("    → Some location signal remains (live discriminator above random).")

    out = {"zip_acc": zip_acc, "geo_r2": geo_r2, "inc_acc": inc_acc}
    out.update(probes)
    out["live_random"] = live_random
    out["probe_random"] = probe_random
    return pred_r2, out


def randomization_test(T, confounders, Y, n_permutations=100, n_pca=50):
    print("\n  [4] Randomization Intervention")

    n_pca = min(n_pca, T.shape[1], T.shape[0])
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)

    scaler = StandardScaler()
    T_s = scaler.fit_transform(T_pca)

    features_orig = np.hstack([T_s, confounders])

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )

    n = len(Y)
    train_n = int(n * 0.7)
    idx = np.random.RandomState(42).permutation(n)
    train_idx, test_idx = idx[:train_n], idx[train_n:]

    model.fit(features_orig[train_idx], Y[train_idx])
    r2_original = model.score(features_orig[test_idx], Y[test_idx])
    print(f"    Original R²: {r2_original:.4f}")

    r2_permuted = []
    for p in range(n_permutations):
        perm = np.random.RandomState(p).permutation(n)
        conf_perm = confounders[perm]

        features_perm = np.hstack([T_s, conf_perm])

        model_p = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=p,
        )
        model_p.fit(features_perm[train_idx], Y[train_idx])
        r2_permuted.append(model_p.score(features_perm[test_idx], Y[test_idx]))

        if (p + 1) % 20 == 0:
            print(f"\r    Permutation {p + 1}/{n_permutations}", end="", flush=True)

    print()

    r2_permuted = np.array(r2_permuted)
    delta_r2 = r2_original - np.mean(r2_permuted)
    p_value = np.mean(r2_permuted >= r2_original)

    print(f"    Permuted R² (mean): {np.mean(r2_permuted):.4f} ± {np.std(r2_permuted):.4f}")
    print(f"    ΔR²: {delta_r2:.4f}")
    print(f"    p-value: {p_value:.4f}")

    return r2_original, np.mean(r2_permuted), p_value


def run_causal_analysis(city):
    result = load_analysis_data(city)
    if result is None:
        print(f"{city}: no data found, skipping")
        return

    emb_df, parcels = result
    data = get_features_and_target(emb_df, parcels)
    if data is None:
        print(f"{city}: no price target available, skipping")
        return

    T, confounders, Y, meta = data
    print(f"\n{'='*60}")
    print(f"CAUSAL ANALYSIS: {city}")
    print(f"{'='*60}")
    print(f"  n={len(Y)}, text_dim={T.shape[1]}, confounders={confounders.shape[1]}")
    print(f"  Rich confounders: {meta['has_rich_confounders']}")

    delta_r2 = backdoor_adjustment(T, confounders, Y)
    dr_effect, dr_ci, dr_extras = doubly_robust_estimation(T, confounders, Y)
    dml_result = dml_continuous_treatment(T, confounders, Y)
    pred_r2, disc_metrics = adversarial_deconfounding(T, Y, meta)
    r2_orig, r2_perm, p_val = randomization_test(T, confounders, Y)
    cate_results = cate_by_price_quantile(T, confounders, Y)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {city}")
    print(f"{'='*60}")
    print(f"  Backdoor ΔR²:              {delta_r2:.4f}")
    print(f"  DR causal effect (ATE):    {dr_effect:.4f} [{dr_ci[0]:.4f}, {dr_ci[1]:.4f}] (boot)")
    print(f"    IF SE: {dr_extras['if_se']:.4f}, MDE: ±{dr_extras['mde']:.4f}")
    if dml_result is not None:
        print(f"  DML continuous θ:          {dml_result['theta']:.4f} "
              f"[{dml_result['ci'][0]:.4f}, {dml_result['ci'][1]:.4f}] "
              f"(MDE: ±{dml_result['mde']:.4f})")
    print(f"  Adversarial predictor R²:  {pred_r2:.4f}")
    print(f"    Zip disc acc:            {disc_metrics['zip_acc']:.4f}")
    print(f"    Geo disc R²:             {disc_metrics['geo_r2']:.4f}")
    print(f"    Income disc acc:         {disc_metrics['inc_acc']:.4f}")
    print(f"  Randomization ΔR²:         {r2_orig - r2_perm:.4f} (p={p_val:.4f})")
    print(f"  CATE by quantile:")
    for r in cate_results:
        if not np.isnan(r["ate"]):
            print(f"    Q{r['quantile']}: ATE={r['ate']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")


def main():
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for city in cities:
        run_causal_analysis(city)


if __name__ == "__main__":
    main()

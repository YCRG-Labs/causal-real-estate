import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import bootstrap
from config import PROCESSED_DIR, CITIES, EMBEDDING_DIM


def load_analysis_data(city):
    emb_path = PROCESSED_DIR / f"{city}_embeddings.parquet"
    parcels_path = PROCESSED_DIR / f"{city}_parcels_amenities.gpkg"

    if not emb_path.exists():
        return None

    import geopandas as gpd
    emb_df = pd.read_parquet(emb_path)
    if parcels_path.exists():
        parcels = gpd.read_file(parcels_path, layer=city)
    else:
        parcels = None

    return emb_df, parcels


def get_features_and_target(emb_df, parcels):
    emb_cols = [f"emb_{i}" for i in range(EMBEDDING_DIM)]
    available_emb = [c for c in emb_cols if c in emb_df.columns]
    T = emb_df[available_emb].values

    if "price" in emb_df.columns:
        Y = pd.to_numeric(emb_df["price"], errors="coerce").values
    elif parcels is not None and "sale_price" in parcels.columns:
        Y = parcels["sale_price"].values[:len(T)].astype(float)
    else:
        return None

    if "zip" in emb_df.columns:
        from sklearn.preprocessing import LabelEncoder
        zips = emb_df["zip"].fillna(0).astype(float).astype(int).astype(str)
        le = LabelEncoder()
        zip_encoded = le.fit_transform(zips)
        n_bins = min(len(le.classes_), 20)
        L = np.zeros((len(T), n_bins))
        for i, z in enumerate(zip_encoded):
            L[i, z % n_bins] = 1.0
    elif "latitude" in emb_df.columns and "longitude" in emb_df.columns:
        lat = pd.to_numeric(emb_df["latitude"], errors="coerce").values
        lon = pd.to_numeric(emb_df["longitude"], errors="coerce").values
        L = np.column_stack([lat, lon])
    else:
        L = np.zeros((len(T), 1))

    X = np.zeros((len(T), 0))

    valid = ~(np.isnan(Y) | np.isinf(Y) | (Y <= 0))
    if L.shape[1] > 0:
        valid &= ~np.any(np.isnan(L), axis=1)

    T, L, X, Y = T[valid], L[valid], X[valid], Y[valid]
    Y = np.log(Y)

    return T, L, X, Y


def backdoor_adjustment(T, L, X, Y, n_pca=50):
    print("\n  [1] Backdoor Adjustment")
    n_pca = min(n_pca, T.shape[1], T.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)

    confounders = np.hstack([L, X]) if X.shape[1] > 0 else L

    scaler_c = StandardScaler()
    confounders_s = scaler_c.fit_transform(confounders)

    scaler_t = StandardScaler()
    T_s = scaler_t.fit_transform(T_pca)

    full_features = np.hstack([T_s, confounders_s])
    conf_only = confounders_s

    max_features = min(0.5, 10.0 / full_features.shape[1]) if full_features.shape[1] > 20 else 0.8

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    r2_full = []
    r2_conf = []
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

    if delta_r2 < 0.01:
        print("    → Text adds negligible predictive power after controlling for confounders")
    elif delta_r2 < 0.05:
        print("    → Text adds small but measurable signal")
    else:
        print("    → Text adds substantial signal beyond confounders")

    return delta_r2


def doubly_robust_estimation(T, L, X, Y, n_pca=50):
    print("\n  [2] Doubly-Robust Estimation")
    n_pca = min(n_pca, T.shape[1], T.shape[0])
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)

    confounders = np.hstack([L, X]) if X.shape[1] > 0 else L
    scaler = StandardScaler()
    confounders_s = scaler.fit_transform(confounders)

    T_norm = np.linalg.norm(T_pca, axis=1)
    T_median = np.median(T_norm)
    treatment = (T_norm > T_median).astype(float)

    outcome_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42,
    )
    outcome_model.fit(
        np.hstack([treatment.reshape(-1, 1), confounders_s]),
        Y,
    )

    from sklearn.linear_model import LogisticRegression
    propensity_model = LogisticRegression(max_iter=1000, random_state=42)
    propensity_model.fit(confounders_s, treatment)
    e = propensity_model.predict_proba(confounders_s)[:, 1]
    e = np.clip(e, 0.05, 0.95)

    mu1 = outcome_model.predict(np.hstack([np.ones((len(Y), 1)), confounders_s]))
    mu0 = outcome_model.predict(np.hstack([np.zeros((len(Y), 1)), confounders_s]))

    dr_effect = np.mean(
        mu1 - mu0
        + treatment * (Y - mu1) / e
        - (1 - treatment) * (Y - mu0) / (1 - e)
    )

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

    print(f"    DR estimate (ATE): {dr_effect:.4f}")
    print(f"    95% CI: [{ci.confidence_interval.low:.4f}, {ci.confidence_interval.high:.4f}]")

    if ci.confidence_interval.low <= 0 <= ci.confidence_interval.high:
        print("    → CI contains zero: no significant causal effect of text")
    else:
        print("    → CI excludes zero: significant effect detected")

    return dr_effect, (ci.confidence_interval.low, ci.confidence_interval.high)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
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
    def __init__(self, input_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Discriminator(nn.Module):
    def __init__(self, input_dim=128, n_locations=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_locations),
        )

    def forward(self, x):
        return self.net(x)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def adversarial_deconfounding(T, L, Y, n_pca=50, epochs=100, lr=1e-3, n_loc_bins=20):
    print("\n  [3] Adversarial Deconfounding")

    n_pca = min(n_pca, T.shape[1], T.shape[0])
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)

    lat_bins = pd.cut(L[:, 0], bins=n_loc_bins, labels=False)
    lon_bins = pd.cut(L[:, 1], bins=n_loc_bins, labels=False)
    loc_labels = lat_bins * n_loc_bins + lon_bins

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    loc_labels = le.fit_transform(loc_labels)
    n_locations = len(le.classes_)

    scaler_t = StandardScaler()
    T_s = scaler_t.fit_transform(T_pca)

    scaler_y = StandardScaler()
    Y_s = scaler_y.fit_transform(Y.reshape(-1, 1)).ravel()

    n = len(T_s)
    split = int(n * 0.7)
    perm = np.random.RandomState(42).permutation(n)
    train_perm, test_perm = perm[:split], perm[split:]

    T_train = torch.FloatTensor(T_s[train_perm])
    Y_train = torch.FloatTensor(Y_s[train_perm])
    L_train = torch.LongTensor(loc_labels[train_perm])
    T_test = torch.FloatTensor(T_s[test_perm])
    Y_test = torch.FloatTensor(Y_s[test_perm])
    L_test = torch.LongTensor(loc_labels[test_perm])

    encoder = Encoder(n_pca, 128, 64)
    predictor = Predictor(64)
    discriminator = Discriminator(64, n_locations)

    opt_enc_pred = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=1e-4)

    pred_loss_fn = nn.MSELoss()
    disc_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        if epoch <= 10:
            lam = 0.1
        elif epoch <= 50:
            lam = 0.1 + 0.9 * (epoch - 10) / 40
        else:
            lam = 1.0

        encoder.train()
        predictor.train()
        discriminator.train()

        z = encoder(T_train)
        y_pred = predictor(z)
        pred_loss = pred_loss_fn(y_pred, Y_train)

        z_rev = GradientReversal.apply(z, lam)
        loc_pred = discriminator(z_rev)
        disc_loss = disc_loss_fn(loc_pred, L_train)

        total_loss = pred_loss + disc_loss
        opt_enc_pred.zero_grad()
        total_loss.backward()
        opt_enc_pred.step()

        z_detached = encoder(T_train).detach()
        loc_pred2 = discriminator(z_detached)
        disc_loss2 = disc_loss_fn(loc_pred2, L_train)
        opt_disc.zero_grad()
        disc_loss2.backward()
        opt_disc.step()

    encoder.eval()
    predictor.eval()
    discriminator.eval()

    with torch.no_grad():
        z = encoder(T_test)
        y_pred = predictor(z)
        pred_r2 = 1 - torch.mean((Y_test - y_pred) ** 2).item() / torch.var(Y_test).item()

        loc_pred = discriminator(z)
        disc_acc = (loc_pred.argmax(dim=1) == L_test).float().mean().item()
        random_acc = 1.0 / n_locations

    print(f"    Predictor R² (deconfounded): {pred_r2:.4f}")
    print(f"    Discriminator accuracy:       {disc_acc:.4f} (random: {random_acc:.4f})")

    if disc_acc < random_acc * 1.5:
        print("    → Location successfully removed from embeddings")
    else:
        print("    → Some location signal remains")

    if pred_r2 < 0.05:
        print("    → After deconfounding, text has near-zero predictive power")
    else:
        print(f"    → Text retains {pred_r2:.4f} R² after location removal")

    return pred_r2, disc_acc


def randomization_test(T, L, X, Y, n_permutations=100, n_pca=50):
    print("\n  [4] Randomization Intervention")

    n_pca = min(n_pca, T.shape[1], T.shape[0])
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)

    scaler = StandardScaler()
    T_s = scaler.fit_transform(T_pca)

    features_orig = np.hstack([T_s, L])
    if X.shape[1] > 0:
        features_orig = np.hstack([features_orig, X])

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
        L_perm = L[perm]

        features_perm = np.hstack([T_s, L_perm])
        if X.shape[1] > 0:
            features_perm = np.hstack([features_perm, X])

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

    if delta_r2 > 0.05:
        print("    → Large drop when randomizing locations: text encodes location")
    elif delta_r2 > 0.01:
        print("    → Moderate drop: some location dependence in text")
    else:
        print("    → Minimal drop: text may contain location-independent signal")

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

    T, L, X, Y = data
    print(f"\n{'='*60}")
    print(f"CAUSAL ANALYSIS: {city}")
    print(f"{'='*60}")
    print(f"  n={len(Y)}, text_dim={T.shape[1]}, confounders={X.shape[1]}")

    delta_r2 = backdoor_adjustment(T, L, X, Y)
    dr_effect, dr_ci = doubly_robust_estimation(T, L, X, Y)
    pred_r2, disc_acc = adversarial_deconfounding(T, L, Y)
    r2_orig, r2_perm, p_val = randomization_test(T, L, X, Y)

    print(f"\n{'='*60}")
    print(f"SUMMARY: {city}")
    print(f"{'='*60}")
    print(f"  Backdoor ΔR²:              {delta_r2:.4f}")
    print(f"  DR causal effect (ATE):    {dr_effect:.4f} [{dr_ci[0]:.4f}, {dr_ci[1]:.4f}]")
    print(f"  Adversarial predictor R²:  {pred_r2:.4f}")
    print(f"  Randomization ΔR²:         {r2_orig - r2_perm:.4f} (p={p_val:.4f})")


def main():
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for city in cities:
        run_causal_analysis(city)


if __name__ == "__main__":
    main()

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from config import PROCESSED_DIR, EMBEDDING_DIM


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


class Discriminator(nn.Module):
    def __init__(self, input_dim=64, n_locations=10):
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


def run_and_plot(city, epochs=150, n_pca=50, n_loc_bins=20, lr=1e-3):
    emb_df = pd.read_parquet(PROCESSED_DIR / f"{city}_embeddings.parquet")
    emb_cols = [f"emb_{i}" for i in range(EMBEDDING_DIM)]
    available = [c for c in emb_cols if c in emb_df.columns]

    emb_df = emb_df.dropna(subset=["price"])
    emb_df = emb_df[emb_df["price"] > 0]

    T = emb_df[available].values
    Y = np.log(emb_df["price"].values.astype(float))

    zips = emb_df["zip"].fillna(0).astype(float).astype(int).astype(str)
    le = LabelEncoder()
    loc_labels = le.fit_transform(zips)
    n_locations = len(le.classes_)

    n_pca = min(n_pca, T.shape[1], len(T) - 2)
    pca = PCA(n_components=n_pca, random_state=42)
    T_pca = pca.fit_transform(T)

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

    history = {
        "epoch": [], "pred_loss_train": [], "pred_loss_test": [],
        "disc_acc_train": [], "disc_acc_test": [], "lambda": [],
    }

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
            z_tr = encoder(T_train)
            z_te = encoder(T_test)

            pl_tr = pred_loss_fn(predictor(z_tr), Y_train).item()
            pl_te = pred_loss_fn(predictor(z_te), Y_test).item()

            da_tr = (discriminator(z_tr).argmax(dim=1) == L_train).float().mean().item()
            da_te = (discriminator(z_te).argmax(dim=1) == L_test).float().mean().item()

        history["epoch"].append(epoch)
        history["pred_loss_train"].append(pl_tr)
        history["pred_loss_test"].append(pl_te)
        history["disc_acc_train"].append(da_tr)
        history["disc_acc_test"].append(da_te)
        history["lambda"].append(lam)

    h = pd.DataFrame(history)

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.5))

    c_train = "#2c7bb6"
    c_test = "#d7191c"

    ax1.plot(h["epoch"], h["pred_loss_train"], color=c_train, linewidth=1.2, label="Train")
    ax1.plot(h["epoch"], h["pred_loss_test"], color=c_test, linewidth=1.2, label="Test")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Predictor MSE Loss")
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_title("(a) Predictor Loss", fontweight="normal")

    ax2.plot(h["epoch"], h["disc_acc_train"], color=c_train, linewidth=1.2, label="Train")
    ax2.plot(h["epoch"], h["disc_acc_test"], color=c_test, linewidth=1.2, label="Test")
    random_baseline = 1.0 / n_locations
    ax2.axhline(y=random_baseline, color="#888888", linestyle=":", linewidth=0.8, label=f"Random ({random_baseline:.2f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Discriminator Accuracy")
    ax2.legend(frameon=False, fontsize=9)
    ax2.set_title("(b) Location Discriminator", fontweight="normal")
    ax2.set_ylim(-0.05, 1.05)

    ax3.fill_between(h["epoch"], 0, h["lambda"], color="#abd9e9", alpha=0.6)
    ax3.plot(h["epoch"], h["lambda"], color="#2c7bb6", linewidth=1.2)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel(r"$\lambda$ (adversarial weight)")
    ax3.set_title(r"(c) $\lambda$ Annealing Schedule", fontweight="normal")
    ax3.set_ylim(-0.05, 1.15)

    plt.tight_layout(w_pad=2.5)

    fig.savefig(PROCESSED_DIR / f"{city}_training_curves.png", dpi=600)
    fig.savefig(PROCESSED_DIR / f"{city}_training_curves.pdf")
    print(f"Saved training curves to {PROCESSED_DIR / f'{city}_training_curves.png'}")

    print(f"\nFinal metrics (test set):")
    print(f"  Predictor loss: {h['pred_loss_test'].iloc[-1]:.4f}")
    print(f"  Discriminator accuracy: {h['disc_acc_test'].iloc[-1]:.4f}")
    print(f"  Random baseline: {random_baseline:.4f}")


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "sf"
    run_and_plot(city)


if __name__ == "__main__":
    main()

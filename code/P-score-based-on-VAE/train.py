# train_vae.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

from VAE import VAE


def load_control_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # 选出特征列：排除 Metadata_ 开头, ObjectNumber, ImageNumber
    drop_cols = [c for c in df.columns if c.startswith("Metadata_")]
    drop_cols += ["ObjectNumber", "ImageNumber"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()

    # 处理缺失值：用列中位数填充
    X = X.fillna(X.median())

    return df, X.values.astype("float32"), feature_cols


def _plot_training_curves(
    epochs,
    train_total,
    train_recon,
    train_kl,
    val_total=None,
    val_recon=None,
    val_kl=None,
    out_prefix: str = "vae_training",
):
    """绘制 Nature 风格的训练/验证损失曲线"""
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    fig, ax = plt.subplots(figsize=(3.5, 3))  # 单栏图大小 ~9 cm

    # 训练集
    ax.plot(epochs, train_total, label="Train total", linewidth=1.2)
    ax.plot(epochs, train_recon, label="Train recon", linewidth=1.0, linestyle="--")
    ax.plot(epochs, train_kl, label="Train KL (×β)", linewidth=1.0, linestyle=":")

    # 验证集（如果提供）
    if val_total is not None:
        ax.plot(epochs, val_total, label="Val total", linewidth=1.2, color="tab:red")
    if val_recon is not None:
        ax.plot(
            epochs,
            val_recon,
            label="Val recon",
            linewidth=1.0,
            linestyle="--",
            color="tab:red",
        )
    if val_kl is not None:
        ax.plot(
            epochs,
            val_kl,
            label="Val KL (×β)",
            linewidth=1.0,
            linestyle=":",
            color="tab:red",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()

    png_path = f"{out_prefix}.png"
    pdf_path = f"{out_prefix}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    # fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Training curves saved to: {png_path} and {pdf_path}")


def train_vae(
    csv_path: str = "qrm_A549_control_Mit.csv",
    model_out: str = "vae_model.pt",
    scaler_out: str = "vae_scaler.pkl",
    batch_size: int = 128,
    latent_dim: int = 10,
    num_epochs: int = 100,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: str | None = None,
    plot_prefix: str = "vae_training",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 读取对照数据
    df, X, feature_cols = load_control_data(csv_path)
    input_dim = X.shape[1]
    print(f"Loaded control data: {X.shape[0]} cells, {input_dim} features")

    # 2. train / val 拆分（8:2）
    X_train, X_val = train_test_split(
        X, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"Train cells: {X_train.shape[0]}, Val cells: {X_val.shape[0]}")

    # 3. 标准化：仅用训练集拟合 scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_all_scaled = scaler.transform(X)  # 用于后面计算整体 loss 分布

    # 4. 构建 DataLoader
    train_dataset = TensorDataset(torch.from_numpy(X_train_scaled))
    val_dataset = TensorDataset(torch.from_numpy(X_val_scaled))
    all_dataset = TensorDataset(torch.from_numpy(X_all_scaled))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    all_loader = DataLoader(
        all_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # 5. 初始化模型
    model = VAE(input_dim=input_dim, latent_dim=latent_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ====== 记录每个 epoch 的 train / val loss ======
    epoch_indices = []
    train_total_losses = []
    train_recon_losses = []
    train_kl_losses = []

    val_total_losses = []
    val_recon_losses = []
    val_kl_losses = []

    # 6. 训练
    model.train()
    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        n_train = 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            batch_size_actual = batch_x.size(0)
            n_train += batch_size_actual

            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x)

            loss, recon_l, kl_l, _ = VAE.loss_function(
                recon_x, batch_x, mu, logvar, beta=beta
            )
            loss.backward()
            optimizer.step()

            total_loss_sum += loss.item() * batch_size_actual
            recon_loss_sum += recon_l.item() * batch_size_actual
            kl_loss_sum += (kl_l.item() * beta) * batch_size_actual

        avg_total_train = total_loss_sum / n_train
        avg_recon_train = recon_loss_sum / n_train
        avg_kl_train = kl_loss_sum / n_train

        # ---- Validation ----
        model.eval()
        val_total_sum = 0.0
        val_recon_sum = 0.0
        val_kl_sum = 0.0
        n_val = 0

        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                bs = batch_x.size(0)
                n_val += bs

                recon_x, mu, logvar = model(batch_x)
                loss, recon_l, kl_l, _ = VAE.loss_function(
                    recon_x, batch_x, mu, logvar, beta=beta
                )

                val_total_sum += loss.item() * bs
                val_recon_sum += recon_l.item() * bs
                val_kl_sum += (kl_l.item() * beta) * bs

        avg_total_val = val_total_sum / n_val
        avg_recon_val = val_recon_sum / n_val
        avg_kl_val = val_kl_sum / n_val

        # ---- 记录 ----
        epoch_indices.append(epoch)

        train_total_losses.append(avg_total_train)
        train_recon_losses.append(avg_recon_train)
        train_kl_losses.append(avg_kl_train)

        val_total_losses.append(avg_total_val)
        val_recon_losses.append(avg_recon_val)
        val_kl_losses.append(avg_kl_val)

        model.train()  # 下一 epoch 前切回 train 模式

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch}/{num_epochs}]  "
                f"Train total: {avg_total_train:.4f}  recon: {avg_recon_train:.4f}  KL(×β): {avg_kl_train:.4f} | "
                f"Val total: {avg_total_val:.4f}  recon: {avg_recon_val:.4f}  KL(×β): {avg_kl_val:.4f}"
            )

    # 7. 用训练好的模型在全部对照数据上计算 per-cell loss，得到 mean/std
    model.eval()
    with torch.no_grad():
        all_losses = []
        for (batch_x,) in all_loader:
            batch_x = batch_x.to(device)
            recon_x, mu, logvar = model(batch_x)
            _, _, _, total_loss_per_sample = VAE.loss_function(
                recon_x, batch_x, mu, logvar, beta=beta
            )
            all_losses.append(total_loss_per_sample.cpu().numpy())
        all_losses = np.concatenate(all_losses, axis=0)

    loss_mean = float(all_losses.mean())
    loss_std = float(all_losses.std())
    print(f"Control loss mean: {loss_mean:.6f}, std: {loss_std:.6f}")

    # 8. 保存模型 + 统计量
    save_dict = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "beta": beta,
        "feature_cols": feature_cols,
        "loss_mean": loss_mean,
        "loss_std": loss_std,
    }
    torch.save(save_dict, model_out)
    joblib.dump(scaler, scaler_out)

    print(f"Model saved to {model_out}")
    print(f"Scaler saved to {scaler_out}")

    # 9. 绘制训练/验证曲线（Nature 风格）
    _plot_training_curves(
        epochs=epoch_indices,
        train_total=train_total_losses,
        train_recon=train_recon_losses,
        train_kl=train_kl_losses,
        val_total=val_total_losses,
        val_recon=val_recon_losses,
        val_kl=val_kl_losses,
        out_prefix=plot_prefix,
    )

    return {
        "epochs": epoch_indices,
        "train_total": train_total_losses,
        "train_recon": train_recon_losses,
        "train_kl": train_kl_losses,
        "val_total": val_total_losses,
        "val_recon": val_recon_losses,
        "val_kl": val_kl_losses,
        "loss_mean": loss_mean,
        "loss_std": loss_std,
    }


if __name__ == "__main__":
    train_vae(
        csv_path="../data/MB_qrm_A549_control_BF_1.csv",
        model_out="qrm_A549_bf_vae_model.pt",
        scaler_out="qrm_A549_bf_vae_scaler.pkl",
        batch_size=64,
        latent_dim=20,
        num_epochs=300,
        lr=1e-3,
        beta=0.01,
        plot_prefix="vae_training_bf",
    )

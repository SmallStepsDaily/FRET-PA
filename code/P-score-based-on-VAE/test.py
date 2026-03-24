import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib

from VAE import VAE

def load_features(csv_path: str, feature_cols=None):
    df = pd.read_csv(csv_path)

    # 分离元数据列（Metadata_开头）、ObjectNumber、ImageNumber
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    id_cols = ["ObjectNumber", "ImageNumber"]
    # 确保id列存在（如果不存在会报错，可根据需要调整）
    for col in id_cols:
        if col not in df.columns:
            raise ValueError(f"数据中缺少必要的列: {col}")
    all_keep_cols = meta_cols + id_cols

    if feature_cols is None:
        # 和训练时一样自动选列（排除元数据和id列）
        feature_cols = [c for c in df.columns if c not in all_keep_cols]

    # 提取特征列数据
    X = df[feature_cols].copy()
    X = X.fillna(X.median())

    # 返回：元数据+id的DataFrame、特征数据、特征列名、元数据+id列名
    return df[all_keep_cols].copy(), X.values.astype("float32"), feature_cols, all_keep_cols

def compute_P(
    model_path: str = "qrm_A549_vae_model.pt",
    scaler_path: str = "qrm_A549_vae_scaler.pkl",
    csv_path: str = "../data/MNB_qrm_A549_Mit.csv",
    batch_size: int = 256,
    device: str = None,
    out_csv: str = "qrm_Mit_with_P.csv",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 加载模型和统计量
    checkpoint = torch.load(model_path, map_location=device)
    input_dim = checkpoint["input_dim"]
    latent_dim = checkpoint["latent_dim"]
    beta = checkpoint["beta"]
    feature_cols_trained = checkpoint["feature_cols"]
    loss_mean = checkpoint["loss_mean"]
    loss_std = checkpoint["loss_std"]

    scaler = joblib.load(scaler_path)

    # 2. 读取待测数据（只保留元数据+id列）
    df_meta_id, X, feature_cols, keep_cols = load_features(csv_path, feature_cols=feature_cols_trained)
    assert feature_cols == feature_cols_trained, "特征列不一致，注意列顺序和名字要与训练时一致"

    # 3. 标准化
    X_scaled = scaler.transform(X)
    tensor_X = torch.from_numpy(X_scaled)
    dataset = TensorDataset(tensor_X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 4. 初始化 VAE 并加载权重
    model = VAE(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 5. 计算每个细胞的 loss
    all_losses = []
    with torch.no_grad():
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            recon_x, mu, logvar = model(batch_x)
            _, _, _, total_loss_per_sample = VAE.loss_function(
                recon_x, batch_x, mu, logvar, beta=beta
            )
            all_losses.append(total_loss_per_sample.cpu().numpy())
    all_losses = np.concatenate(all_losses, axis=0)

    # 6. 基于对照组的 mean/std 计算 P 值
    # z = max(0, (loss - mean) / std)
    # 再缩放到 [0,1]，例如除以 3 (3σ 以外视为接近 1)
    eps = 1e-8
    z = np.maximum(0.0, (all_losses - loss_mean) / (loss_std + eps))
    P = np.clip(z / 3.0, 0.0, 1.0)

    # 7. 拼接元数据+id列 和 P值、VAE_loss
    df_out = df_meta_id.copy()
    df_out["VAE_loss"] = all_losses
    df_out["P"] = P

    # 8. 保存结果
    df_out.to_csv(out_csv, index=False)
    print(f"Saved results with P to {out_csv}")
    print("\n前5行结果：")
    print(df_out.head())
    print(f"\n输出列名：{df_out.columns.tolist()}")
    print(f"\n数据总行数：{len(df_out)}")

if __name__ == "__main__":
    compute_P(
        model_path="qrm_A549_m_vae_model.pt",
        scaler_path="qrm_A549_m_vae_scaler.pkl",
        csv_path="../data/MB_qrm_A549_FRET_Mit.csv",
        batch_size=64,
        out_csv="P_m.csv",
    )
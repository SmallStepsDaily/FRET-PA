# VAE.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    一个用于表型特征的简单 VAE：
    - 输入: 313 维特征
    - 编码器: 313 -> 256 -> 128 -> latent(mu, logvar)
    - 解码器: latent -> 128 -> 256 -> 300
    """
    def __init__(self, input_dim: int, latent_dim: int = 10, hidden_dims=None):
        super(VAE, self).__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ----- Encoder -----
        encoder_layers = []
        last_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, h))
            encoder_layers.append(nn.ReLU())
            last_dim = h
        self.encoder = nn.Sequential(*encoder_layers)

        # 均值和 logvar
        self.fc_mu = nn.Linear(last_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_dim, latent_dim)

        # ----- Decoder -----
        decoder_layers = []
        last_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(last_dim, h))
            decoder_layers.append(nn.ReLU())
            last_dim = h
        decoder_layers.append(nn.Linear(last_dim, input_dim))
        # 不加激活，回归到原始特征（配合 MSE）
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # 采样 z = mu + eps * sigma
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta: float = 1.0):
        """
        返回:
        - total_loss_mean: 标量，用于反向传播
        - recon_loss_mean: 标量
        - kl_loss_mean: 标量
        - total_loss_per_sample: (batch,) 每个样本的总损失，用于后续算 P
        """
        # 重建损失: 对每个样本求 MSE 的均值 (feature 维度平均)
        recon_loss_per_sample = F.mse_loss(recon_x, x, reduction='none').mean(dim=1)

        # KL loss per sample: KL(N(mu, sigma^2) || N(0, I))
        kl_per_sample = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        )

        total_loss_per_sample = recon_loss_per_sample + beta * kl_per_sample

        # 取 batch 平均用于训练
        total_loss_mean = total_loss_per_sample.mean()
        recon_loss_mean = recon_loss_per_sample.mean()
        kl_loss_mean = kl_per_sample.mean()

        return total_loss_mean, recon_loss_mean, kl_loss_mean, total_loss_per_sample

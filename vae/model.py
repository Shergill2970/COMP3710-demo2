"""
Variational Autoencoder for OASIS brain MRI slices (128x128 grayscale).

Architecture: convolutional encoder -> (mu, logvar) -> reparameterization ->
convolutional decoder. Images are expected normalized to [-1, 1] (matching
data_loader.py's OASISImageDataset), so the decoder's final activation is
tanh, and reconstruction loss is MSE rather than BCE.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim, img_size=128, base_channels=32):
        super().__init__()
        c = base_channels
        # 128 -> 64 -> 32 -> 16 -> 8
        self.conv = nn.Sequential(
            nn.Conv2d(1, c, 4, stride=2, padding=1), nn.BatchNorm2d(c), nn.LeakyReLU(0.2),
            nn.Conv2d(c, c * 2, 4, stride=2, padding=1), nn.BatchNorm2d(c * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1), nn.BatchNorm2d(c * 4), nn.LeakyReLU(0.2),
            nn.Conv2d(c * 4, c * 8, 4, stride=2, padding=1), nn.BatchNorm2d(c * 8), nn.LeakyReLU(0.2),
        )
        self.feat_size = img_size // 16          # 128 // 16 = 8
        self.flat_dim = c * 8 * self.feat_size * self.feat_size

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, img_size=128, base_channels=32):
        super().__init__()
        c = base_channels
        self.feat_size = img_size // 16
        self.c8 = c * 8
        self.fc = nn.Linear(latent_dim, self.c8 * self.feat_size * self.feat_size)

        # 8 -> 16 -> 32 -> 64 -> 128
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, 4, stride=2, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(),
            nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(),
            nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.ConvTranspose2d(c, 1, 4, stride=2, padding=1),
            nn.Tanh(),  # matches [-1, 1] normalized input images
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), self.c8, self.feat_size, self.feat_size)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, latent_dim=32, img_size=128, base_channels=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, img_size, base_channels)
        self.decoder = Decoder(latent_dim, img_size, base_channels)

    def reparameterize(self, mu, logvar):
        # z = mu + std * epsilon, epsilon ~ N(0, I)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, kl_weight=1.0):
    """
    ELBO loss = reconstruction loss + KL divergence to the standard normal prior.

    kl_weight < 1.0 implements "KL annealing" / beta-VAE weighting - useful if
    training collapses to ignoring the latent code (posterior collapse) or if
    reconstructions are too blurry early in training.
    """
    recon_loss = nn.functional.mse_loss(recon, x, reduction="sum") / x.size(0)

    # KL(N(mu, sigma^2) || N(0, 1)) in closed form
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    total = recon_loss + kl_weight * kl_div
    return total, recon_loss, kl_div

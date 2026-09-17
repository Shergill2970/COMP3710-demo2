"""
DCGAN-style Generator and Discriminator for OASIS brain MRI slices.

Uses 64x64 resolution (not 128, unlike the VAE/UNet) - GANs are notoriously
unstable to train, and working at a smaller resolution significantly improves
training stability and speed, which matters given the assignment's warning
about chaotic convergence. Pass img_size=64 to OASISImageDataset (from
data_loader.py) to match.

Stability choices baked in, per common DCGAN/GAN-training best practices:
  - BatchNorm in the generator (helps gradient flow, stabilises training)
  - No BatchNorm on the discriminator's very first layer (standard DCGAN advice)
  - LeakyReLU in the discriminator (avoids dead gradients from plain ReLU)
  - Tanh output on the generator, matching [-1, 1] normalized real images
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=128, base_channels=64):
        super().__init__()
        c = base_channels
        # Project z -> 4x4 feature map, then upsample 4->8->16->32->64
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, c * 8, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c * 8), nn.ReLU(True),                          # 4x4

            nn.ConvTranspose2d(c * 8, c * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4), nn.ReLU(True),                          # 8x8

            nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2), nn.ReLU(True),                          # 16x16

            nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(True),                              # 32x32

            nn.ConvTranspose2d(c, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),                                                     # 64x64, matches [-1,1] data
        )

    def forward(self, z):
        # z: [B, latent_dim] -> reshape to [B, latent_dim, 1, 1] for conv input
        z = z.view(z.size(0), -1, 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),                               # 32x32, no BatchNorm here (standard DCGAN advice)

            nn.Conv2d(c, c * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2), nn.LeakyReLU(0.2, inplace=True),        # 16x16

            nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4), nn.LeakyReLU(0.2, inplace=True),        # 8x8

            nn.Conv2d(c * 4, c * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 8), nn.LeakyReLU(0.2, inplace=True),        # 4x4

            nn.Conv2d(c * 8, 1, 4, stride=1, padding=0, bias=False),       # 1x1, single real/fake score
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, 1).squeeze(1)  # raw logit, BCEWithLogitsLoss applies sigmoid internally


def weights_init(m):
    """DCGAN paper's recommended weight init - helps training stability."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

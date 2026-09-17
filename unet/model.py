"""
UNet for OASIS brain MRI segmentation (4 classes, per data_loader.NUM_SEG_CLASSES).

Standard UNet: contracting path (conv blocks + maxpool) paired with an
expanding path (upsampling + conv blocks), with skip connections concatenating
matching-resolution feature maps from the contracting path into the expanding
path. Skip connections let the decoder recover fine spatial detail that would
otherwise be lost through the pooling operations - critical for segmentation,
where exact pixel boundaries matter.

Output: NUM_CLASSES channels of raw logits (categorical/one-hot style output,
one channel per class) - softmax + argmax is applied at inference/evaluation
time, not inside the model, so the model can be trained directly with
CrossEntropyLoss (which applies log-softmax internally).
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3x3 convs, each followed by BatchNorm + ReLU - the basic UNet building block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base_channels=64):
        super().__init__()
        c = base_channels

        # ---- Contracting path (encoder) ----
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        # ---- Bottleneck ----
        self.bottleneck = DoubleConv(c * 8, c * 16)

        # ---- Expanding path (decoder) ----
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 8)  # c*16 in-channels: c*8 upsampled + c*8 skip

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)

        # ---- Output: one channel per class (categorical / one-hot style) ----
        self.out_conv = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder, keeping intermediate feature maps for skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        # Decoder, concatenating skip connections at each matching resolution
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)  # [B, num_classes, H, W] raw logits

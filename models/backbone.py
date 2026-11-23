"""Image and LiDAR convolutional feature encoders."""

from torch import nn


def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(),
    )


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            conv(width, width), nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
        )

    def forward(self, x):
        return (x + self.layers(x)).relu()


def encoder(token_grid):
    return nn.Sequential(
        conv(1, 32), nn.MaxPool2d(2), conv(32, 64), nn.MaxPool2d(2),
        conv(64, 128), Residual(128), nn.MaxPool2d(2),
        conv(128, 256), nn.AdaptiveAvgPool2d((token_grid, token_grid)),
    )

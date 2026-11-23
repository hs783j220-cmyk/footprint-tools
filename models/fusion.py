"""Cross-attention fusion and candidate confidence head."""

import torch
from torch import nn

from .backbone import encoder


class CandidateNetwork(nn.Module):
    def __init__(self, width, dropout, token_grid):
        super().__init__()
        self.image_encoder = encoder(token_grid)
        self.lidar_encoder = encoder(token_grid)
        self.image_tokens = nn.Linear(256, width)
        self.lidar_tokens = nn.Linear(256, width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.attention_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 2 * width), nn.ReLU(), nn.Linear(2 * width, width))
        self.ffn_norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))
        self.scale = width ** -0.5

    def forward(self, masks, occupancy):
        image = self.image_tokens(self.image_encoder(masks).flatten(2).transpose(1, 2))
        lidar = self.lidar_tokens(self.lidar_encoder(occupancy).flatten(2).transpose(1, 2))
        weights = torch.softmax((self.query(image) @ self.key(lidar).transpose(1, 2)) * self.scale, dim=-1)
        tokens = self.attention_norm(image + weights @ self.value(lidar))
        tokens = self.ffn_norm(tokens + self.ffn(tokens))
        return self.head(tokens.mean(dim=1)).squeeze(-1)

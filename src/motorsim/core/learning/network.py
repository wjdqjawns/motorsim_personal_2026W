"""1-D CNN that maps a recorded experiment (channels x time) to motor parameters."""
from __future__ import annotations

import torch
from torch import nn

class ParamNet(nn.Module):
    def __init__(self, c_in: int, n_out: int, width: int = 64):
        super().__init__()
        layers, c = [], c_in
        for w in (width // 2, width, width * 2, width * 2, width * 2):
            layers += [nn.Conv1d(c, w, 7, stride=2, padding=3), nn.BatchNorm1d(w), nn.GELU()]
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(2 * c, 128), nn.GELU(), nn.Linear(128, n_out))

    def forward(self, x):
        h = self.features(x)
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.head(h)
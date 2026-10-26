"""CRT study: the documented 3D-CNN backbone, copied verbatim from
experiments/run_oasis3_cnn_backbone.py (class CNN3D) so this package is
self-contained. tests/test_crt.py asserts state-dict shape equality against a
fresh import of the original to guarantee the copy never drifts.

GroupNorm everywhere: no batch statistics, so train/eval modes agree and the
deterministic objective used for derivatives equals the evaluation objective.
No dropout, no augmentation anywhere in the recipe.
"""
from __future__ import annotations

import torch.nn as nn


class CNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [1, 8, 16, 32, 64]
        layers = []
        for i in range(4):
            layers += [nn.Conv3d(ch[i], ch[i + 1], 3, padding=1),
                       nn.GroupNorm(min(8, ch[i + 1]), ch[i + 1]), nn.ReLU(),
                       nn.MaxPool3d(2)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        return self.head(self.body(x).mean(dim=(2, 3, 4)))


def n_params(model):
    return sum(p.numel() for p in model.parameters())

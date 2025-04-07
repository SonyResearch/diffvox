from torch import nn
from typing import List


class SumLosses(nn.Module):
    def __init__(self, weights: List[float] = [], loss_fns: List[nn.Module] = []):
        super().__init__()

        self.losses = nn.ModuleList(loss_fns)
        self.weights = weights

    def forward(self, pred, target):
        losses = [fn(pred, target) for fn in self.losses]
        return sum(map(lambda w, x: w * x, self.weights, losses)), [
            x.item() for x in losses
        ]

import torch
from torch import nn
from torch.nn import functional as F
from torchcomp import ms2coef

from modules.functional import avg
from modules.fx import hadamard
from modules.utils import chain_functions


class LDRLoss(nn.Module):
    def __init__(self, sr: int, short_tau=50, long_tau=3000):
        super().__init__()
        self.register_buffer(
            "short_tau",
            ms2coef(torch.tensor(short_tau, dtype=torch.float32), sr),
            persistent=False,
        )
        self.register_buffer(
            "long_tau",
            ms2coef(torch.tensor(long_tau, dtype=torch.float32), sr),
            persistent=False,
        )
        self.align_shift = int(sr * (long_tau - short_tau) * 0.0005)

    def forward(self, pred_sq: torch.Tensor, target_sq: torch.Tensor):
        f = chain_functions(
            lambda x: x.reshape(-1, x.shape[-1]),
            lambda x: torch.log(avg(x, self.short_tau))
            - torch.log(avg(x, self.long_tau).roll(-self.align_shift)),
        )
        return F.l1_loss(f(pred_sq), f(target_sq))


class MLDRLoss(nn.Module):
    def __init__(
        self,
        *args,
        s_taus: list,
        l_taus: list,
        mid_side: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.losses = nn.ModuleList(
            [
                LDRLoss(*args, short_tau=s, long_tau=l, **kwargs)
                for s, l in zip(s_taus, l_taus)
            ]
        )
        self.mid_side = mid_side

    def forward(self, x_pred: torch.Tensor, x_true: torch.Tensor):
        if self.mid_side:
            x_pred = hadamard(x_pred)
            x_true = hadamard(x_true)

        x_pred_sq = x_pred.square().clamp_min(1e-8)
        x_true_sq = x_true.square().clamp_min(1e-8)

        return sum(loss(x_pred_sq, x_true_sq) for loss in self.losses)

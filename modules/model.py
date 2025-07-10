import torch
from torch import nn
import torch.nn.functional as F
from functools import partial, reduce
from typing import Optional, List
from torchaudio.transforms import MelSpectrogram, MFCC


class LogMelSpectrogram(MelSpectrogram):
    def forward(self, waveform):
        return super().forward(waveform).add(1e-8).log()


class LogMFCC(MFCC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, log_mels=True, **kwargs)


class LightningSequential(nn.Sequential):
    def __init__(self, modules: List[nn.Module]):
        super().__init__(*modules)

    def forward(self, *args):
        return reduce(lambda x, f: f(*x) if isinstance(x, tuple) else f(x), self, args)


class ResidualWrapper(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x):
        return x + self.m(x)

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial, reduce
from typing import Optional, List

from .utils import chain_functions


class LogSpectralCentroid(nn.Module):
    def forward(self, spec):
        # assume spec is of shape (..., freq, time)
        freqs = torch.linspace(0, 1, spec.size(-2), device=spec.device)
        spec_T = spec.transpose(-1, -2)
        normalised_spec = spec_T / spec_T.sum(-1, keepdim=True).clamp_min(1e-8)
        return torch.log(normalised_spec @ freqs + 1e-8).unsqueeze(-2)


class LogSpectralFlatness(nn.Module):
    def forward(self, spec):
        # assume spec is of shape (..., freq, time)
        spec_pow = spec.clamp(1e-8).square()
        log_gmean = spec_pow.log().mean(-2, keepdim=True)
        log_amean = spec_pow.mean(-2, keepdim=True).log()
        return log_gmean - log_amean


class LogSpectralBandwidth(nn.Module):
    def __init__(self):
        super().__init__()
        self.centroid = LogSpectralCentroid()

    def forward(self, spec):
        # assume spec is of shape (..., freq, time)
        freqs = torch.linspace(0, 1, spec.size(-2), device=spec.device)
        centroid = self.centroid(spec).exp()
        normalised_spec = spec / spec.sum(-2, keepdim=True).clamp_min(1e-8)
        return (
            torch.log(
                (normalised_spec * (freqs[:, None] - centroid).square()).sum(
                    -2, keepdim=True
                )
                + 1e-8
            )
            * 0.5
        )


class LogRMS(nn.Module):
    def forward(self, frame):
        return torch.log(frame.square().mean(-2, keepdim=True).sqrt() + 1e-8)


class LogCrest(nn.Module):
    def __init__(self):
        super().__init__()
        self.rms = LogRMS()

    def forward(self, frame):
        log_rms = self.rms(frame)
        return frame.abs().amax(-2, keepdim=True).add(1e-8).log() - log_rms


class LogSpread(nn.Module):
    def __init__(self):
        super().__init__()
        self.rms = LogRMS()

    def forward(self, frame):
        log_rms = self.rms(frame)
        return (frame.abs().add(1e-8).log() - log_rms).mean(-2, keepdim=True)


class MapAndMerge(nn.Module):
    def __init__(self, funcs: List[nn.Module], dim=-1):
        super().__init__()
        self.funcs = nn.ModuleList(funcs)
        self.dim = dim

    def forward(self, frame):
        return torch.cat([f(frame) for f in self.funcs], dim=self.dim)


class Frame(nn.Module):
    def __init__(self, frame_length, hop_length, center=False):
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.center = center

    def forward(self, waveform):
        if self.center:
            waveform = F.pad(waveform, (self.frame_length // 2, self.frame_length // 2))
        return waveform.unfold(-1, self.frame_length, self.hop_length).transpose(-1, -2)


class StatisticReduction(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        mu = x.mean(self.dim, keepdim=True)
        diffs = x - mu
        std = diffs.square().mean(self.dim, keepdim=True).sqrt()
        zscores = diffs / std.clamp_min(1e-8)
        skews = zscores.pow(3).mean(self.dim, keepdim=True)
        kurts = zscores.pow(4).mean(self.dim, keepdim=True) - 3
        return torch.cat([mu, std, skews, kurts], dim=self.dim)

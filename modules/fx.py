import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization
from torchcomp import ms2coef, coef2ms, db2amp
from torchaudio.transforms import Spectrogram, InverseSpectrogram

from typing import List, Tuple, Union, Any, Optional, Callable
import math
from torch_fftconv import fft_conv1d
from functools import reduce

from .functional import (
    compressor_expander,
    lowpass_biquad,
    highpass_biquad,
    equalizer_biquad,
    lowshelf_biquad,
    highshelf_biquad,
    lowpass_biquad_coef,
    highpass_biquad_coef,
    highshelf_biquad_coef,
    lowshelf_biquad_coef,
    equalizer_biquad_coef,
)
from .utils import chain_functions


class Clip(nn.Module):
    def __init__(self, max: Optional[float] = None, min: Optional[float] = None):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        if self.min is not None:
            x = torch.clip(x, min=self.min)
        if self.max is not None:
            x = torch.clip(x, max=self.max)
        return x


def clip_delay_eq_Q(m: nn.Module, Q: float):
    if isinstance(m, Delay) and isinstance(m.eq, LowPass):
        register_parametrization(m.eq.params, "Q", Clip(max=Q))
    return m


float2param = lambda x: nn.Parameter(
    torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
)

STEREO_NORM = math.sqrt(2)


def broadcast2stereo(m, args):
    x, *_ = args
    return x.expand(-1, 2, -1) if x.shape[1] == 1 else x


hadamard = lambda x: torch.stack([x.sum(1), x[:, 0] - x[:, 1]], 1) / STEREO_NORM


class Hadamard(nn.Module):
    def forward(self, x):
        return hadamard(x)


class FX(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.params = nn.ParameterDict({k: float2param(v) for k, v in kwargs.items()})


class SmoothingCoef(nn.Module):
    def forward(self, x):
        return x.sigmoid()

    def right_inverse(self, y):
        return (y / (1 - y)).log()


class CompRatio(nn.Module):
    def forward(self, x):
        return x.exp() + 1

    def right_inverse(self, y):
        return torch.log(y - 1)


class MinMax(nn.Module):
    def __init__(self, min=0.0, max: Union[float, torch.Tensor] = 1.0):
        super().__init__()
        if isinstance(min, torch.Tensor):
            self.register_buffer("min", min, persistent=False)
        else:
            self.min = min

        if isinstance(max, torch.Tensor):
            self.register_buffer("max", max, persistent=False)
        else:
            self.max = max

        self._m = SmoothingCoef()

    def forward(self, x):
        return self._m(x) * (self.max - self.min) + self.min

    def right_inverse(self, y):
        return self._m.right_inverse((y - self.min) / (self.max - self.min))


class WrappedPositive(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period

    def forward(self, x):
        return x.abs() % self.period

    def right_inverse(self, y):
        return y


class CompressorExpander(FX):
    cmp_ratio_min: float = 1
    cmp_ratio_max: float = 20

    def __init__(
        self,
        sr: int,
        cmp_ratio: float = 2.0,
        exp_ratio: float = 0.5,
        at_ms: float = 50.0,
        rt_ms: float = 50.0,
        avg_coef: float = 0.3,
        cmp_th: float = -18.0,
        exp_th: float = -54.0,
        make_up: float = 0.0,
        delay: int = 0,
        lookahead: bool = False,
        max_lookahead: float = 15.0,
    ):
        super().__init__(
            cmp_th=cmp_th,
            exp_th=exp_th,
            make_up=make_up,
            avg_coef=avg_coef,
            cmp_ratio=cmp_ratio,
            exp_ratio=exp_ratio,
        )
        # deprecated, please use lookahead instead
        self.delay = delay
        self.sr = sr

        self.params["at"] = nn.Parameter(ms2coef(torch.tensor(at_ms), sr))
        self.params["rt"] = nn.Parameter(ms2coef(torch.tensor(rt_ms), sr))

        if lookahead:
            self.params["lookahead"] = nn.Parameter(torch.ones(1) / sr * 1000)
            register_parametrization(
                self.params, "lookahead", WrappedPositive(max_lookahead)
            )
            sinc_length = int(sr * (max_lookahead + 1) * 0.001) + 1
            left_pad_size = int(sr * 0.001)
            self._pad_size = (left_pad_size, sinc_length - left_pad_size - 1)
            self.register_buffer(
                "_arange",
                torch.arange(sinc_length) - left_pad_size,
                persistent=False,
            )
        self.lookahead = lookahead

        register_parametrization(self.params, "at", SmoothingCoef())
        register_parametrization(self.params, "rt", SmoothingCoef())
        register_parametrization(self.params, "avg_coef", SmoothingCoef())
        register_parametrization(
            self.params, "cmp_ratio", MinMax(self.cmp_ratio_min, self.cmp_ratio_max)
        )
        register_parametrization(self.params, "exp_ratio", SmoothingCoef())

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = (
                f"attack: {coef2ms(self.params.at, self.sr).item()} (ms)\n"
                f"release: {coef2ms(self.params.rt, self.sr).item()} (ms)\n"
                f"avg_coef: {self.params.avg_coef.item()}\n"
                f"compressor_ratio: {self.params.cmp_ratio.item()}\n"
                f"expander_ratio: {self.params.exp_ratio.item()}\n"
                f"compressor_threshold: {self.params.cmp_th.item()} (dB)\n"
                f"expander_threshold: {self.params.exp_th.item()} (dB)\n"
                f"make_up: {self.params.make_up.item()} (dB)"
            )
            if self.lookahead:
                s += f"\nlookahead: {self.params.lookahead.item()} (ms)"
        return s

    def forward(self, x):
        if self.lookahead:
            lookahead_in_samples = self.params.lookahead * 0.001 * self.sr
            sinc_filter = torch.sinc(self._arange - lookahead_in_samples)
            lookahead_func = lambda gain: F.conv1d(
                F.pad(
                    gain.view(-1, 1, gain.size(-1)), self._pad_size, mode="replicate"
                ),
                sinc_filter[None, None, :],
            ).view(*gain.shape)
        else:
            lookahead_func = lambda x: x

        return compressor_expander(
            x.reshape(-1, x.shape[-1]),
            lookahead_func=lookahead_func,
            **{k: v for k, v in self.params.items() if k != "lookahead"},
        ).view(*x.shape)


class Panning(FX):
    def __init__(self, pan: float = 0.0):
        assert pan <= 100 and pan >= -100
        super().__init__(pan=(pan + 100) / 200)

        register_parametrization(self.params, "pan", SmoothingCoef())

        self.register_forward_pre_hook(broadcast2stereo)

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = f"pan: {self.params.pan.item() * 200 - 100}"
        return s

    def forward(self, x: torch.Tensor):
        angle = self.params.pan.view(1) * torch.pi * 0.5
        amp = torch.concat([angle.cos(), angle.sin()]).view(2, 1) * STEREO_NORM
        return x * amp


class StereoWidth(Panning):
    def forward(self, x: torch.Tensor):
        return chain_functions(hadamard, super().forward, hadamard)(x)


class ImpulseResponse(nn.Module):
    def forward(self, h):
        return torch.cat([torch.ones_like(h[..., :1]), h], dim=-1)


class FIR(FX):
    def __init__(
        self,
        length: int,
        channels: int = 2,
        conv_method: str = "direct",
    ):
        super().__init__(kernel=torch.zeros(channels, length - 1))
        self._padding = length - 1
        self.channels = channels

        match conv_method:
            case "direct":
                self.conv_func = F.conv1d
            case "fft":
                self.conv_func = fft_conv1d
            case _:
                raise ValueError(f"Unknown conv_method: {conv_method}")

        if channels == 2:
            self.register_forward_pre_hook(broadcast2stereo)

    def forward(self, x: torch.Tensor):
        zero_padded = F.pad(x[..., :-1], (self._padding, 0), "constant", 0)
        return x + self.conv_func(
            zero_padded, self.params.kernel.flip(1).unsqueeze(1), groups=self.channels
        )


class QFactor(nn.Module):
    def forward(self, x):
        return x.exp()

    def right_inverse(self, y):
        return y.log()


class LowPass(FX):
    def __init__(
        self,
        sr: int,
        freq: float = 17500.0,
        Q: float = 0.707,
        min_freq: float = 200.0,
        max_freq: float = 18000,
        min_Q: float = 0.5,
        max_Q: float = 10.0,
    ):
        super().__init__(freq=freq, Q=Q)

        self.sr = sr
        register_parametrization(self.params, "freq", MinMax(min_freq, max_freq))
        register_parametrization(self.params, "Q", MinMax(min_Q, max_Q))

    def forward(self, x):
        return lowpass_biquad(
            x, sample_rate=self.sr, cutoff_freq=self.params.freq, Q=self.params.Q
        )

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = f"freq: {self.params.freq.item():.4f}, Q: {self.params.Q.item():.4f}"
        return s


class HighPass(LowPass):
    def __init__(
        self,
        *args,
        freq: float = 200.0,
        min_freq: float = 16.0,
        max_freq: float = 5300.0,
        **kwargs,
    ):
        super().__init__(
            *args, freq=freq, min_freq=min_freq, max_freq=max_freq, **kwargs
        )

    def forward(self, x):
        return highpass_biquad(
            x, sample_rate=self.sr, cutoff_freq=self.params.freq, Q=self.params.Q
        )


class Peak(FX):
    def __init__(
        self,
        sr: int,
        gain: float = 0.0,
        freq: float = 2000.0,
        Q: float = 0.707,
        min_freq: float = 33.0,
        max_freq: float = 17500.0,
        min_Q: float = 0.2,
        max_Q: float = 20,
    ):
        super().__init__(freq=freq, Q=Q, gain=gain)

        self.sr = sr

        register_parametrization(self.params, "freq", MinMax(min_freq, max_freq))
        register_parametrization(self.params, "Q", MinMax(min_Q, max_Q))

    def forward(self, x):
        return equalizer_biquad(
            x,
            sample_rate=self.sr,
            center_freq=self.params.freq,
            Q=self.params.Q,
            gain=self.params.gain,
        )

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = f"freq: {self.params.freq.item():.4f}, gain: {self.params.gain.item():.4f}, Q: {self.params.Q.item():.4f}"
        return s


class LowShelf(FX):
    def __init__(
        self,
        sr: int,
        gain: float = 0.0,
        freq: float = 115.0,
        min_freq: float = 30,
        max_freq: float = 200,
    ):
        super().__init__(freq=freq, gain=gain)

        self.sr = sr
        register_parametrization(self.params, "freq", MinMax(min_freq, max_freq))

        self.register_buffer("Q", torch.tensor(0.707), persistent=False)

    def forward(self, x):
        return lowshelf_biquad(
            x,
            sample_rate=self.sr,
            cutoff_freq=self.params.freq,
            gain=self.params.gain,
            Q=self.Q,
        )

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = f"freq: {self.params.freq.item():.4f}, gain: {self.params.gain.item():.4f}"
        return s


class HighShelf(LowShelf):
    def __init__(
        self,
        *args,
        freq: float = 4525,
        min_freq: float = 750,
        max_freq: float = 8300,
        **kwargs,
    ):
        super().__init__(
            *args, freq=freq, min_freq=min_freq, max_freq=max_freq, **kwargs
        )

    def forward(self, x):
        return highshelf_biquad(
            x,
            sample_rate=self.sr,
            cutoff_freq=self.params.freq,
            gain=self.params.gain,
            Q=self.Q,
        )


def module2coeffs(
    m: Union[LowPass, HighPass, Peak, LowShelf, HighShelf],
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    match m:
        case LowPass():
            return lowpass_biquad_coef(m.sr, m.params.freq, m.params.Q)
        case HighPass():
            return highpass_biquad_coef(m.sr, m.params.freq, m.params.Q)
        case Peak():
            return equalizer_biquad_coef(m.sr, m.params.freq, m.params.Q, m.params.gain)
        case LowShelf():
            return lowshelf_biquad_coef(m.sr, m.params.freq, m.params.gain, m.Q)
        case HighShelf():
            return highshelf_biquad_coef(m.sr, m.params.freq, m.params.gain, m.Q)
        case _:
            raise ValueError(f"Unknown module: {m}")


class AlwaysNegative(nn.Module):
    def forward(self, x):
        return -F.softplus(x)

    def right_inverse(self, y):
        return torch.log(y.neg().exp() - 1)


class Reverb(FX):
    def __init__(self, ir_len=60000, n_fft=384, hop_length=192, downsample_factor=1):
        super().__init__(
            log_mag=torch.full((2, n_fft // downsample_factor // 2 + 1), -1.0),
            log_mag_delta=torch.full((2, n_fft // downsample_factor // 2 + 1), -5.0),
        )

        self.steps = (ir_len - n_fft + hop_length - 1) // hop_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.downsample_factor = downsample_factor

        self._noise_angle = nn.Parameter(
            torch.rand(2, n_fft // 2 + 1, self.steps) * 2 * torch.pi
        )

        self.register_buffer(
            "_arange", torch.arange(self.steps, dtype=torch.float32), persistent=False
        )
        self.spec_forward = Spectrogram(n_fft, hop_length=hop_length, power=None)
        self.spec_inverse = InverseSpectrogram(
            n_fft,
            hop_length=hop_length,
        )

        register_parametrization(self.params, "log_mag", AlwaysNegative())
        register_parametrization(self.params, "log_mag_delta", AlwaysNegative())

        self.register_forward_pre_hook(broadcast2stereo)

    def forward(self, x):
        h = x
        H = self.spec_forward(h)

        log_mag = self.params.log_mag
        log_mag_delta = self.params.log_mag_delta

        if self.downsample_factor > 1:
            log_mag = F.interpolate(
                log_mag.unsqueeze(0),
                size=self._noise_angle.size(1),
                align_corners=True,
                mode="linear",
            ).squeeze(0)
            log_mag_delta = F.interpolate(
                log_mag_delta.unsqueeze(0),
                size=self._noise_angle.size(1),
                align_corners=True,
                mode="linear",
            ).squeeze(0)

        ir_2d = torch.exp(
            log_mag.unsqueeze(-1)
            + log_mag_delta.unsqueeze(-1) * self._arange
            + self._noise_angle * 1j
        )

        padded_H = F.pad(H.flatten(1, 2), (ir_2d.shape[-1] - 1, 0))

        H = F.conv1d(
            padded_H,
            hadamard(ir_2d.unsqueeze(0)).flatten(1, 2).flip(-1).transpose(0, 1),
            groups=H.shape[2] * 2,
        ).view(*H.shape)

        h = self.spec_inverse(H)
        return h


class Delay(FX):
    min_delay: float = 100
    max_delay: float = 1000

    def __init__(
        self,
        sr: int,
        delay=200.0,
        feedback=0.1,
        gain=0.1,
        ir_duration: float = 2,
        eq: Optional[nn.Module] = None,
        recursive_eq=False,
    ):
        super().__init__(
            delay=delay,
            feedback=feedback,
            gain=gain,
        )
        self.sr = sr
        self.ir_length = int(sr * max(ir_duration, self.max_delay * 0.002))

        register_parametrization(
            self.params, "delay", MinMax(self.min_delay, self.max_delay)
        )
        register_parametrization(self.params, "feedback", SmoothingCoef())
        register_parametrization(self.params, "gain", SmoothingCoef())

        self.eq = eq
        self.recursive_eq = recursive_eq

        self.register_buffer(
            "_arange", torch.arange(self.ir_length, dtype=torch.float32)
        )

        self.odd_pan = Panning(0)
        self.even_pan = Panning(0)

    def forward(self, x):
        assert x.size(1) == 1, x.size()
        delay_in_samples = self.sr * self.params.delay * 0.001
        num_delays = self.ir_length // int(delay_in_samples.item() + 1)
        series = torch.arange(1, num_delays + 1, device=x.device)
        decays = self.params.feedback ** (series - 1)

        if self.recursive_eq and self.eq is not None:
            sinc_index = self._arange - delay_in_samples
            single_sinc_filter = torch.sinc(sinc_index)
            eq_sinc_filter = self.eq(single_sinc_filter)
            H = torch.fft.rfft(eq_sinc_filter)
            H_powered = torch.polar(
                H.abs() ** series.unsqueeze(-1), H.angle() * series.unsqueeze(-1)
            )
            sinc_filters = torch.fft.irfft(H_powered, n=self.ir_length)
        else:
            delays_in_samples = delay_in_samples * series
            sinc_indexes = self._arange - delays_in_samples.unsqueeze(-1)
            sinc_filters = torch.sinc(sinc_indexes)

        decayed_sinc_filters = sinc_filters * decays.unsqueeze(-1)
        return self._filter(x, decayed_sinc_filters)

    def _filter(self, x: torch.Tensor, decayed_sinc_filters: torch.Tensor):
        odd_delay_filters = torch.sum(decayed_sinc_filters[::2], 0)
        even_delay_filters = torch.sum(decayed_sinc_filters[1::2], 0)
        stacked_filters = torch.stack([odd_delay_filters, even_delay_filters])

        if self.eq is not None and not self.recursive_eq:
            stacked_filters = self.eq(stacked_filters)

        gained_odd_even_filters = stacked_filters * self.params.gain
        padded_x = F.pad(x, (gained_odd_even_filters.size(-1) - 1, 0))
        conv1d = F.conv1d if x.size(-1) > 44100 * 20 else fft_conv1d
        return sum(
            [
                panner(s)
                for panner, s in zip(
                    [self.odd_pan, self.even_pan],
                    # fft_conv1d(
                    conv1d(
                        padded_x,
                        gained_odd_even_filters.flip(-1).unsqueeze(1),
                    ).chunk(2, 1),
                )
            ]
        )

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = (
                f"delay: {self.sr * self.params.delay.item() * 0.001} (samples)\n"
                f"feedback: {self.params.feedback.item()}\n"
                f"gain: {self.params.gain.item()}"
            )
        return s


class SurrogateDelay(Delay):
    def __init__(self, *args, dropout=0.5, straight_through=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.dropout = dropout
        self.straight_through = straight_through
        self.log_damp = nn.Parameter(torch.ones(1) * -0.01)
        register_parametrization(self, "log_damp", AlwaysNegative())

    def forward(self, x):
        assert x.size(1) == 1, x.size()
        if not self.training:
            return super().forward(x)

        log_damp = self.log_damp
        delay_in_samples = self.sr * self.params.delay * 0.001
        num_delays = self.ir_length // int(delay_in_samples.item() + 1)
        series = torch.arange(1, num_delays + 1, device=x.device)
        decays = self.params.feedback ** (series - 1)

        if self.recursive_eq and self.eq is not None:
            exp_factor = self._arange[: self.ir_length // 2 + 1]
            damped_exp = torch.exp(
                log_damp * exp_factor
                - 1j * delay_in_samples / self.ir_length * 2 * torch.pi * exp_factor
            )
            sinc_filter = torch.fft.irfft(damped_exp, n=self.ir_length)
            if self.straight_through:
                sinc_index = self._arange - delay_in_samples
                hard_sinc_filter = torch.sinc(sinc_index)
                sinc_filter = sinc_filter + (hard_sinc_filter - sinc_filter).detach()

            eq_sinc_filter = self.eq(sinc_filter)
            H = torch.fft.rfft(eq_sinc_filter)

            # use polar form to avoid NaN
            H_powered = torch.polar(
                H.abs() ** series.unsqueeze(-1), H.angle() * series.unsqueeze(-1)
            )
            sinc_filters = torch.fft.irfft(H_powered, n=self.ir_length)
        else:
            exp_factors = series.unsqueeze(-1) * self._arange[: self.ir_length // 2 + 1]
            damped_exps = torch.exp(
                log_damp * exp_factors
                - 1j * delay_in_samples / self.ir_length * 2 * torch.pi * exp_factors
            )
            sinc_filters = torch.fft.irfft(damped_exps, n=self.ir_length)
            if self.straight_through:
                delays_in_samples = delay_in_samples * series
                sinc_indexes = self._arange - delays_in_samples.unsqueeze(-1)
                hard_sinc_filters = torch.sinc(sinc_indexes)
                sinc_filters = (
                    sinc_filters + (hard_sinc_filters - sinc_filters).detach()
                )

        decayed_sinc_filters = sinc_filters * decays.unsqueeze(-1)

        dropout_mask = torch.rand(x.size(0), device=x.device) < self.dropout
        if not torch.any(dropout_mask):
            return self._filter(x, decayed_sinc_filters)
        elif torch.all(dropout_mask):
            return super().forward(x)

        out = torch.zeros((x.size(0), 2, x.size(2)), device=x.device)
        out[~dropout_mask] = self._filter(x[~dropout_mask], decayed_sinc_filters)
        out[dropout_mask] = super().forward(x[dropout_mask])
        return out

    def extra_repr(self) -> str:
        with torch.no_grad():
            return super().extra_repr() + f"\ndamp: {self.log_damp.exp().item()}"


class FSDelay(FX):
    def __init__(
        self,
        sr: int,
        delay=200.0,
        feedback=0.1,
        gain=0.1,
        ir_duration: float = 6,
        eq: Optional[LowPass] = None,
        recursive_eq=False,
    ):
        super().__init__(
            delay=delay,
            feedback=feedback,
            gain=gain,
        )
        self.sr = sr
        self.ir_length = int(sr * max(ir_duration, Delay.max_delay * 0.002))

        register_parametrization(
            self.params, "delay", MinMax(Delay.min_delay, Delay.max_delay)
        )
        register_parametrization(self.params, "gain", SmoothingCoef())

        T_60 = ir_duration * 0.75
        max_delay_in_samples = sr * Delay.max_delay * 0.001
        maximum_decay = db2amp(torch.tensor(-60 / sr / T_60 * max_delay_in_samples))
        register_parametrization(self.params, "feedback", MinMax(0, maximum_decay))

        self.eq = eq
        self.recursive_eq = recursive_eq

        self.odd_pan = Panning(0)
        self.even_pan = Panning(0)

        self.register_buffer(
            "_arange", torch.arange(self.ir_length, dtype=torch.float32)
        )

    def _get_h(self):
        freqs = self._arange[: self.ir_length // 2 + 1] / self.ir_length * 2 * torch.pi
        delay_in_samples = self.sr * self.params.delay * 0.001

        # construct it like a fdn
        Dinv = torch.exp(1j * freqs * delay_in_samples)
        Dinv2 = torch.exp(2j * freqs * delay_in_samples)
        if self.recursive_eq and self.eq is not None:
            b0, b1, b2, a0, a1, a2 = module2coeffs(self.eq)
            z_inv = torch.exp(-1j * freqs)
            z_inv2 = torch.exp(-2j * freqs)
            eq_H = (b0 + b1 * z_inv + b2 * z_inv2) / (a0 + a1 * z_inv + a2 * z_inv2)
            damp = eq_H * self.params.feedback
            det = Dinv2 - damp * damp
        else:
            damp = torch.full_like(Dinv, self.params.feedback) + 0j
            det = Dinv2 - self.params.feedback.square()
        inv_Dinv_m_A = torch.stack([Dinv, damp], 0) / det
        h = torch.fft.irfft(inv_Dinv_m_A, n=self.ir_length) * self.params.gain

        if self.eq is not None and not self.recursive_eq:
            h = self.eq(h)
        return h

    def forward(self, x):
        assert x.size(1) == 1, x.size()
        h = self._get_h()

        padded_x = F.pad(x, (h.size(-1) - 1, 0))
        conv1d = F.conv1d if x.size(-1) > 44100 * 20 else fft_conv1d
        return sum(
            [
                panner(s)
                for panner, s in zip(
                    [self.odd_pan, self.even_pan],
                    conv1d(
                        padded_x,
                        h.flip(-1).unsqueeze(1),
                    ).chunk(2, 1),
                )
            ]
        )

    def extra_repr(self) -> str:
        with torch.no_grad():
            s = (
                f"delay: {self.sr * self.params.delay.item() * 0.001} (samples)\n"
                f"feedback: {self.params.feedback.item()}\n"
                f"gain: {self.params.gain.item()}"
            )
        return s


class FSSurrogateDelay(FSDelay):
    def __init__(self, *args, straight_through=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.straight_through = straight_through
        self.log_damp = nn.Parameter(torch.ones(1) * -0.0001)
        register_parametrization(self, "log_damp", AlwaysNegative())

    def _get_h(self):
        if not self.training:
            return super()._get_h()

        log_damp = self.log_damp
        delay_in_samples = self.sr * self.params.delay * 0.001

        exp_factor = self._arange[: self.ir_length // 2 + 1]
        freqs = exp_factor / self.ir_length * 2 * torch.pi
        D = torch.exp(log_damp * exp_factor - 1j * delay_in_samples * freqs)
        D2 = torch.exp(log_damp * exp_factor * 2 - 2j * delay_in_samples * freqs)

        if self.straight_through:
            D_orig = torch.exp(-1j * delay_in_samples * freqs)
            D2_orig = torch.exp(-2j * delay_in_samples * freqs)
            D = torch.stack([D, D_orig], 0)
            D2 = torch.stack([D2, D2_orig], 0)

        if self.recursive_eq and self.eq is not None:
            b0, b1, b2, a0, a1, a2 = module2coeffs(self.eq)
            z_inv = torch.exp(-1j * freqs)
            z_inv2 = torch.exp(-2j * freqs)
            eq_H = (b0 + b1 * z_inv + b2 * z_inv2) / (a0 + a1 * z_inv + a2 * z_inv2)
            damp = eq_H * self.params.feedback
            odd_H = D / (1 - damp * damp * D2)
            even_H = odd_H * D * damp
        else:
            damp = torch.full_like(D, self.params.feedback) + 0j
            odd_H = D / (1 - self.params.feedback.square() * D2)
            even_H = odd_H * D * self.params.feedback

        inv_Dinv_m_A = torch.stack([odd_H, even_H], 0)
        h = torch.fft.irfft(inv_Dinv_m_A, n=self.ir_length)

        if self.straight_through:
            damped_h, orig_h = h.unbind(1)
            h = damped_h + (orig_h - damped_h).detach()

        if self.eq is not None and not self.recursive_eq:
            h = self.eq(h)
        return h * self.params.gain

    def extra_repr(self) -> str:
        with torch.no_grad():
            return super().extra_repr() + f"\ndamp: {self.log_damp.exp().item()}"


class SendFXsAndSum(FX):
    def __init__(self, *args, cross_send=True, pan_direct=False):
        super().__init__(
            **(
                {
                    f"sends_{i}": torch.full([len(args) - i - 1], 0.01)
                    for i in range(len(args) - 1)
                }
                if cross_send
                else {}
            )
        )
        self.effects = nn.ModuleList(args)
        if pan_direct:
            self.pan = Panning()

        if cross_send:
            for i in range(len(args) - 1):
                register_parametrization(self.params, f"sends_{i}", SmoothingCoef())

    def forward(self, x):
        if hasattr(self, "pan"):
            di = self.pan(x)
        else:
            di = x

        if len(self.params) == 0:
            return reduce(
                lambda x, y: x[..., : y.shape[-1]] + y[..., : x.shape[-1]],
                map(lambda f: f(x), self.effects),
                di,
            )

        def f(states, ps):
            x, cum_sends = states
            m, send_gains = ps
            h = m(cum_sends[0])
            return (
                x[..., : h.shape[-1]] + h[..., : x.shape[-1]],
                (
                    None
                    if cum_sends.size(0) == 1
                    else cum_sends[1:, ..., : h.shape[-1]]
                    + send_gains[:, None, None, None] * h[..., : cum_sends.shape[-1]]
                ),
            )

        return reduce(
            f,
            zip(
                self.effects,
                [self.params[f"sends_{i}"] for i in range(len(self.effects) - 1)]
                + [None],
            ),
            (di, x.unsqueeze(0).expand(len(self.effects), -1, -1, -1)),
        )[0]


class UniLossLess(nn.Module):
    def forward(self, x):
        tri = x.triu(1)
        return torch.linalg.matrix_exp(tri - tri.T)


class FDN(FX):
    max_delay = 100

    def __init__(
        self,
        sr: int,
        ir_duration: float = 1.0,
        delays=(997, 1153, 1327, 1559, 1801, 2099),
        trainable_delay=False,
        num_decay_freq=1,
        delay_independent_decay=False,
        eq: Optional[nn.Module] = None,
    ):
        # beta = torch.distributions.Beta(1.1, 6)
        num_delays = len(delays)
        super().__init__(
            b=torch.ones(num_delays, 2) / num_delays,
            c=torch.zeros(2, num_delays),
            U=torch.randn(num_delays, num_delays) / num_delays**0.5,
            gamma=torch.rand(
                num_decay_freq, num_delays if not delay_independent_decay else 1
            )
            * 0.2
            + 0.4,
            # delays=beta.sample((num_delays,)) * 64,
        )

        self.sr = sr
        self.ir_length = int(sr * ir_duration)

        # ir_duration = T_60
        T_60 = ir_duration * 0.75
        delays = torch.tensor(delays)
        if delay_independent_decay:
            gamma_max = db2amp(-60 / sr / T_60 * delays.min())
        else:
            gamma_max = db2amp(-60 / sr / T_60 * delays)

        register_parametrization(self.params, "gamma", MinMax(0, gamma_max))
        register_parametrization(self.params, "U", UniLossLess())

        if not trainable_delay:
            self.register_buffer(
                "delays",
                delays,
            )
        else:
            self.params["delays"] = nn.Parameter(delays / sr * 1000)
            register_parametrization(self.params, "delays", MinMax(0, self.max_delay))

        self.register_forward_pre_hook(broadcast2stereo)

        self.eq = eq

    def forward(self, x):
        conv1d = F.conv1d if x.size(-1) > 44100 * 20 else fft_conv1d

        c = self.params.c + 0j
        b = self.params.b + 0j

        gamma = self.params.gamma
        delays = self.delays if hasattr(self, "delays") else self.params.delays

        if gamma.size(0) > 1:
            gamma = F.interpolate(
                gamma.T.unsqueeze(1),
                size=self.ir_length // 2 + 1,
                align_corners=True,
                mode="linear",
            ).transpose(0, 2)

        if gamma.size(2) == 1:
            gamma = gamma ** (delays / delays.min())

        A = self.params.U * gamma

        freqs = (
            torch.arange(self.ir_length // 2 + 1, device=x.device)
            / self.ir_length
            * 2
            * torch.pi
        )
        invD = torch.exp(1j * freqs[:, None] * delays)
        # H = c @ torch.linalg.inv(torch.diag_embed(invD) - A) @ b
        H = c @ torch.linalg.solve(torch.diag_embed(invD) - A, b)

        h = torch.fft.irfft(H.permute(1, 2, 0), n=self.ir_length)

        if self.eq is not None:
            h = self.eq(h)

        # return fft_conv1d(
        return conv1d(
            F.pad(x, (self.ir_length - 1, 0)),
            h.flip(-1),
        )

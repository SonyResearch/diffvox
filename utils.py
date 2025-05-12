import torch
import numpy as np
from scipy.signal import freqz
from typing import Iterable

from modules import fx
from modules.functional import (
    highpass_biquad_coef,
    lowpass_biquad_coef,
    highshelf_biquad_coef,
    lowshelf_biquad_coef,
    equalizer_biquad_coef,
)


@torch.no_grad()
def get_log_mags_from_eq(eq: Iterable, worN=1024, sr=44100):
    get_ba = lambda xs: torch.cat([x.view(1) for x in xs]).view(2, 3)

    def f(biquad):
        params = biquad.params
        match type(biquad):
            case fx.HighPass:
                coeffs = highpass_biquad_coef(sr, params.freq, params.Q)
            case fx.LowPass:
                coeffs = lowpass_biquad_coef(sr, params.freq, params.Q)
            case fx.HighShelf:
                coeffs = highshelf_biquad_coef(sr, params.freq, params.gain, biquad.Q)
            case fx.LowShelf:
                coeffs = lowshelf_biquad_coef(sr, params.freq, params.gain, biquad.Q)
            case fx.Peak:
                coeffs = equalizer_biquad_coef(sr, params.freq, params.gain, params.Q)
            case _:
                raise ValueError(biquad)

        b, a = get_ba(coeffs)
        w, h = freqz(b.numpy(), a.numpy(), worN, fs=sr)
        log_h = 20 * np.log10(np.abs(h))
        return w, log_h

    log_mags = list(map(f, eq))
    return log_mags[0][0], [x for _, x in log_mags]

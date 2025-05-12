import torch
import numpy as np
from scipy.signal import freqz
from scipy.stats import norm, chi2, kurtosis, shapiro
from typing import Iterable

from modules import fx
from modules.functional import (
    highpass_biquad_coef,
    lowpass_biquad_coef,
    highshelf_biquad_coef,
    lowshelf_biquad_coef,
    equalizer_biquad_coef,
)


def Roystest(X, alpha=0.05):
    """
    Royston's Multivariate Normality Test.

    Parameters:
    X : array-like, shape (n_samples, n_features)
        Data matrix.
    alpha : float, optional (default=0.05)
        Significance level.

    Returns:
    None
    """
    n, p = X.shape

    if n <= 3:
        raise ValueError("n is too small.")
    elif 4 <= n <= 11:
        x = n
        g = -2.273 + 0.459 * x
        m = 0.5440 - 0.39978 * x + 0.025054 * x**2 - 0.0006714 * x**3
        s = np.exp(1.3822 - 0.77857 * x + 0.062767 * x**2 - 0.0020322 * x**3)
        Z = [(-np.log(g - (np.log(1 - ShaWilstat(X[:, j])))) - m) / s for j in range(p)]
    elif 12 <= n <= 2000:
        x = np.log(n)
        g = 0
        m = -1.5861 - 0.31082 * x - 0.083751 * x**2 + 0.0038915 * x**3
        s = np.exp(-0.4803 - 0.082676 * x + 0.0030302 * x**2)
        Z = [(np.log(1 - ShaWilstat(X[:, j])) + g - m) / s for j in range(p)]
    else:
        raise ValueError("n is not in the proper size range.")

    R = [(norm.ppf(norm.cdf(-z) / 2)) ** 2 for z in Z]

    u = 0.715
    v = 0.21364 + 0.015124 * (np.log(n)) ** 2 - 0.0018034 * (np.log(n)) ** 3
    l = 5
    C = np.corrcoef(X, rowvar=False)
    NC = (C**l) * (1 - (u * (1 - C) ** u) / v)
    T = np.sum(NC) - p
    mC = T / (p**2 - p)
    e = p / (1 + (p - 1) * mC)
    H = (e * np.sum(R)) / p
    P = 1 - chi2.cdf(H, e)

    # print(" ")
    # print("Royston's Multivariate Normality Test")
    # print("-------------------------------------------------------------------")
    # print(f"Number of variables: {p}")
    # print(f"Sample size: {n}")
    # print("-------------------------------------------------------------------")
    # print(f"Royston's statistic: {H:.6f}")
    # print(f"Equivalent degrees of freedom: {e:.6f}")
    # print(f"P-value associated to the Royston's statistic: {P:.6f}")
    # print(f"With a given significance = {alpha:.3f}")
    # if P >= alpha:
    #     print("Data analyzed have a normal distribution.")
    # else:
    #     print("Data analyzed do not have a normal distribution.")
    # print("-------------------------------------------------------------------")
    return H, P


def ShaWilstat(x):
    """
    Shapiro-Wilk's W statistic for assessing a sample normality.

    Parameters:
    x : array-like, shape (n_samples,)
        Data vector.

    Returns:
    W : float
        Shapiro-Wilk's W statistic.
    """
    return shapiro(x)[0]
    x = np.sort(x)
    n = len(x)

    if n < 3:
        raise ValueError("Sample vector must have at least 3 valid observations.")
    if n > 5000:
        print(
            "Warning: Shapiro-Wilk statistic might be inaccurate due to large sample size ( > 5000)."
        )

    m = norm.ppf((np.arange(1, n + 1) - 3 / 8) / (n + 0.25))
    w = np.zeros(n)

    if kurtosis(x) > 3:
        w = 1 / np.sqrt(np.sum(m**2)) * m
        W = (np.sum(w * x)) ** 2 / np.sum((x - np.mean(x)) ** 2)
    else:
        c = 1 / np.sqrt(np.sum(m**2)) * m
        u = 1 / np.sqrt(n)
        p1 = [-2.706056, 4.434685, -2.071190, -0.147981, 0.221157, c[-1]]
        p2 = [-3.582633, 5.682633, -1.752461, -0.293762, 0.042981, c[-2]]

        w[-1] = np.polyval(p1, u)
        w[0] = -w[-1]

        if n == 3:
            w[0] = 0.707106781
            w[-1] = -w[0]

        if n >= 6:
            w[-2] = np.polyval(p2, u)
            w[1] = -w[-2]

            ct = 3
            phi = (np.sum(m**2) - 2 * m[-1] ** 2 - 2 * m[-2] ** 2) / (
                1 - 2 * w[-1] ** 2 - 2 * w[-2] ** 2
            )
        else:
            ct = 2
            phi = (np.sum(m**2) - 2 * m[-1] ** 2) / (1 - 2 * w[-1] ** 2)

        w[ct - 1 : n - ct + 1] = m[ct - 1 : n - ct + 1] / np.sqrt(phi)

        W = (np.sum(w * x)) ** 2 / np.sum((x - np.mean(x)) ** 2)

    return W


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

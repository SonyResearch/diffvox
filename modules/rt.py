import numpy as np
from numba import njit, prange
from scipy.signal import firwin2
import torch

from .fx import Delay, FDN, module2coeffs


@njit
def rt_fdn(
    x: np.ndarray,
    delay_steps: np.ndarray,
    firs: np.ndarray,
    U: np.ndarray,
):
    _, T = x.shape
    M = delay_steps.shape[0]
    order = firs.shape[1]
    y = np.zeros_like(x)
    buf_size = delay_steps.max() + order
    delay_buf = np.zeros((M, buf_size), dtype=x.dtype)
    read_pointer = 0

    for t in range(T):
        # out = delay_buf[(range(M), read_pointers)]
        # for i in prange(M):
        #     out[i] = delay_buf[i, read_pointers[i]]
        out = delay_buf[:, read_pointer]
        y[:, t] = out

        s = out * firs[:, 0]
        # indexes = (read_pointers[:, None] - np.arange(1, order)) % buf_sizes[:, None]
        # reg = np.take_along_axis(delay_buf, indexes, axis=1)
        # s += firs[:, 1:] @ reg.T
        # for j in prange(M):
        #     s[j] += firs[j, 1:] @ delay_buf[j, indexes[j]]
        for i in prange(M):
            for j in prange(1, order):
                s[i] += firs[i, j] * delay_buf[i, (read_pointer - j) % buf_size]
        # for i in prange(1, order):
        #     s += firs[:, i] * delay_buf[:, (read_pointer - i) % buf_size]

        feedback = U @ s + x[:, t]
        w_pointers = (read_pointer + delay_steps) % buf_size
        # delay_buf[(range(M), w_pointers)] = s + B @ x[:, t]
        for i in prange(M):
            delay_buf[i, w_pointers[i]] = feedback[i]
        read_pointer = (read_pointer + 1) % buf_size

    return y


@njit
def rt_delay(
    x: np.ndarray,
    delay_step: int,
    b0: float,
    b1: float,
    b2: float,
    a1: float,
    a2: float,
):
    T = x.shape[0]
    y = np.zeros((2, T), dtype=x.dtype)
    buf_size = delay_step + 1
    read_pointer = 0
    delay_buf = np.zeros((2, buf_size), dtype=x.dtype)
    bq_buf = np.zeros((2, 2), dtype=x.dtype)

    for t in range(T):
        out = delay_buf[:, read_pointer]
        y[:, t] = out

        s = bq_buf[:, 0] + b0 * out
        bq_buf[:, 0] = bq_buf[:, 1] + b1 * out - a1 * s
        bq_buf[:, 1] = b2 * out - a2 * s

        w_pointer = (read_pointer + delay_step) % buf_size
        # cross feeding because of ping-pong delay
        delay_buf[0, w_pointer] = s[1] + x[t]
        delay_buf[1, w_pointer] = s[0]

        read_pointer = (read_pointer + 1) % buf_size

    return y


class RealTimeDelay(Delay):
    def forward(self, x):
        assert x.size(1) == 1, x.size()
        assert x.size(0) == 1, x.size()
        with torch.no_grad():
            delay_in_samples = round(self.sr * self.params.delay.item() * 0.001)
            feedback = self.params.feedback.item()

            if self.recursive_eq and self.eq is not None:
                b0, b1, b2, a0, a1, a2 = [p.item() for p in module2coeffs(self.eq)]
                b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
            else:
                b0, b1, b2, a1, a2 = 1.0, 0.0, 0.0, 0.0, 0.0

            b0 = b0 * feedback
            b1 = b1 * feedback
            b2 = b2 * feedback
            x_numpy = x.squeeze().cpu().numpy()
            y_numpy = rt_delay(x_numpy, delay_in_samples, b0, b1, b2, a1, a2)
        y = torch.from_numpy(y_numpy).unsqueeze(0).to(x.device) * self.params.gain
        return self.odd_pan(y[:, :1]) + self.even_pan(y[:, 1:])


class RealTimeFDN(FDN):
    def forward(self, x):
        assert x.size(1) == 2, x.size()
        assert x.size(0) == 1, x.size()
        with torch.no_grad():
            delays = self.delays if hasattr(self, "delays") else self.params.delays

            c = self.params.c
            b = self.params.b
            gamma = self.params.gamma.clone()

            if gamma.size(1) == 1:
                gamma = gamma ** (delays / delays.min())

            freqs = np.linspace(0, 1, gamma.size(0))
            firs = np.apply_along_axis(
                lambda x: firwin2(gamma.size(0) * 2 - 1, freqs, x, fs=2),
                1,
                gamma.cpu().numpy().T,
            ).astype(np.float32)
            shifted_delays = delays - firs.shape[1] // 2

            U = self.params.U

            x = b @ x.squeeze()

            y_numpy = rt_fdn(
                x.cpu().numpy(),
                # delays.cpu().numpy(),
                shifted_delays.cpu().numpy(),
                # firs.cpu().numpy(),
                firs,
                U.cpu().numpy(),
            )
            y = c @ torch.from_numpy(y_numpy).to(x.device)
            y = y.unsqueeze(0)

        if self.eq is not None:
            y = self.eq(y)
        return y

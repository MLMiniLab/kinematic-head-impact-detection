import numpy as np

from src.data import FS


def fft_amplitude(x, fs=FS):
    n = x.shape[-1]
    Y = np.fft.rfft(x, axis=-1) / n
    Y[..., 1:-1] *= 2
    f = np.fft.rfftfreq(n, d=1 / fs)
    return f, np.abs(Y)

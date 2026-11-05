"""Order tracking from a once-per-revolution tachometer: shaft angle, speed and the complex phasor of each order.

The tachometer gives one pulse per revolution. The shaft angle is ``2 pi i`` at the ``i``-th rising edge (edge times
refined by linear interpolation between samples) and is interpolated linearly in between, so the angle follows the
small speed fluctuations of the run. The phasor of order ``k`` on a channel ``x(t)`` is the synchronous demodulation

    X_k = (2 / N) sum_n x(t_n) exp(-i k theta(t_n))

over the samples between the first and the last pulse (a whole number of revolutions, so a component that is
periodic in the shaft angle at order ``k`` is captured without leakage, whatever the speed). Its phase is referred to
the tachometer pulse, which is what lets responses of different runs be added and subtracted as vectors (the
influence-coefficient method of rotor balancing).

Resolution: a record of ``R`` revolutions separates orders that differ by more than about ``1 / R``. For five
seconds at 30 Hz (150 revolutions) that is 0.007 orders, so a bearing order of 2.998 cannot be told apart from 3X
by a phasor or an FFT of the record; :func:`order_resolution` reports it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ShaftAngle:
    """Shaft angle of every sample inside the revolutions spanned by the pulses."""
    theta: np.ndarray          # rad, one value per sample in [first, last)
    first: int                 # index of the first sample used
    last: int                  # index one past the last sample used
    pulse_times_s: np.ndarray
    speed_hz: float            # mean shaft speed over the whole revolutions
    speed_cv: float            # coefficient of variation of the revolution periods
    revolutions: int

    @property
    def reliable(self) -> bool:
        """The pulses look like one per revolution: revolution periods within 5 % of each other (coefficient of
        variation). A missing or extra pulse doubles or halves a period and fails this test."""
        return self.speed_cv <= 0.05


def _fundamental_hz(tach: np.ndarray, fs: float, f_min: float = 2.0, f_max: float = 400.0) -> float:
    """Repetition rate of a pulse train: the strongest line of its spectrum in [f_min, f_max] that is not a harmonic
    of a stronger lower line (spikes are broadband, the pulses put their energy at the rate and its harmonics)."""
    x = tach - tach.mean()
    S = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    band = (f >= f_min) & (f <= f_max)
    k = np.flatnonzero(band)[np.argmax(S[band])]
    f0 = f[k]
    for div in (2, 3, 4):                         # a sub-harmonic nearly as strong is the true rate
        j = np.argmin(np.abs(f - f0 / div))
        lo, hi = max(j - 3, 0), j + 4
        if f0 / div >= f_min and S[lo:hi].max() > 0.5 * S[k]:
            f0 = f[lo + np.argmax(S[lo:hi])]
    return float(f0)


def pulse_times(tach: np.ndarray, fs: float, threshold: float | None = None, refractory: float = 0.6,
                min_width: float = 0.3) -> np.ndarray:
    """Rising-edge times (s) of a once-per-revolution pulse tachometer, refined by linear interpolation. Edges cross
    ``threshold`` (default: the middle of the signal's range); a pulse must stay above it for at least ``min_width``
    x the median width of the pulses, and an edge closer than ``refractory`` x the period of the train (its spectral
    fundamental) to the previous kept edge is dropped -- spikes between pulses are rejected either way."""
    tach = np.asarray(tach, float)
    th = 0.5 * (np.max(tach) + np.min(tach)) if threshold is None else float(threshold)
    above = tach >= th
    up = np.flatnonzero(~above[:-1] & above[1:])
    dn = np.flatnonzero(above[:-1] & ~above[1:])
    if len(up) == 0:
        return np.empty(0)
    ends = np.searchsorted(dn, up)
    width = np.where(ends < len(dn), dn[np.minimum(ends, len(dn) - 1)] - up, len(tach) - up)
    wide = width >= min_width * np.median(width[width >= np.median(width)])
    up = up[wide]
    frac = (th - tach[up]) / np.maximum(tach[up + 1] - tach[up], 1e-300)
    tp = (up + frac) / fs
    if len(tp) > 2:
        period = 1.0 / _fundamental_hz(tach, fs)
        out = [tp[0]]
        for x in tp[1:]:
            if x - out[-1] >= refractory * period:
                out.append(x)
        tp = np.asarray(out)
    return tp


def shaft_angle(tach: np.ndarray, fs: float, threshold: float | None = None) -> ShaftAngle:
    """Shaft angle from a once-per-revolution tachometer (see the module docstring)."""
    tp = pulse_times(tach, fs, threshold)
    if len(tp) < 3:
        raise ValueError(f"only {len(tp)} tachometer pulses: cannot track the shaft angle")
    periods = np.diff(tp)
    t = np.arange(len(tach)) / fs
    first = int(np.ceil(tp[0] * fs))
    last = int(np.floor(tp[-1] * fs)) + 1
    theta = np.interp(t[first:last], tp, 2 * np.pi * np.arange(len(tp)))
    return ShaftAngle(theta=theta, first=first, last=last, pulse_times_s=tp, speed_hz=float(1.0 / periods.mean()),
                      speed_cv=float(periods.std() / periods.mean()), revolutions=len(tp) - 1)


def order_phasors(x: np.ndarray, angle: ShaftAngle, orders=(1, 2, 3)) -> np.ndarray:
    """Complex phasors (amplitude in the channel's unit, phase against the pulse) of each order on each channel:
    ``x`` (n_samples,) or (n_samples, n_channels); returns (len(orders), n_channels)."""
    x = np.asarray(x, float)
    x = x[:, None] if x.ndim == 1 else x
    seg = x[angle.first:angle.last]
    seg = seg - seg.mean(0)
    out = np.empty((len(orders), x.shape[1]), complex)
    for j, k in enumerate(orders):
        out[j] = 2.0 / len(seg) * (np.exp(-1j * k * angle.theta) @ seg)
    return out


def order_resolution(angle: ShaftAngle) -> float:
    """Smallest order separation a record of this length resolves (about 1 / revolutions)."""
    return 1.0 / angle.revolutions


def angle_resample(x: np.ndarray, angle: ShaftAngle, fs: float, samples_per_rev: int = 64) -> np.ndarray:
    """Computed order tracking: ``x`` (n_samples,) or (n_samples, n_channels) low-passed below the new Nyquist order
    (0.8 x samples_per_rev / 2 at the run's mean speed) and resampled at ``samples_per_rev`` equal angle steps over the
    whole revolutions; returns (revolutions x samples_per_rev, n_channels)."""
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(x, float)
    x = x[:, None] if x.ndim == 1 else x
    cut = 0.4 * samples_per_rev * angle.speed_hz
    if cut < 0.45 * fs:
        x = sosfiltfilt(butter(6, cut, fs=fs, output="sos"), x, axis=0)
    seg = x[angle.first:angle.last]
    t = (angle.first + np.arange(len(seg))) / fs
    th_u = np.arange(angle.revolutions * samples_per_rev) * (2 * np.pi / samples_per_rev) + angle.theta[0]
    t_u = np.interp(th_u, angle.theta, t)
    return np.stack([np.interp(t_u, t, seg[:, c]) for c in range(seg.shape[1])], 1)


def order_spectrum(x: np.ndarray, angle: ShaftAngle, fs: float, samples_per_rev: int = 64, envelope_band=None,
                   max_order: float = 12.0, step: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude order spectrum (or envelope order spectrum when ``envelope_band = (f_lo, f_hi)`` Hz: band-pass,
    Hilbert envelope, then order tracking) on a fixed grid of orders ``0, step, ..., max_order``: the largest line of
    the record inside each grid cell, so peaks survive whatever the record's resolution. Returns (grid, (n_grid,
    n_channels))."""
    from scipy.signal import butter, hilbert, sosfiltfilt
    x = np.asarray(x, float)
    x = x[:, None] if x.ndim == 1 else x
    if envelope_band is not None:
        x = sosfiltfilt(butter(4, envelope_band, btype="bandpass", fs=fs, output="sos"), x, axis=0)
        x = np.abs(hilbert(x, axis=0))
    y = angle_resample(x, angle, fs, samples_per_rev)
    y = y - y.mean(0)
    n = len(y)
    spec = 2.0 / n * np.abs(np.fft.rfft(y, axis=0))
    orders = np.fft.rfftfreq(n, d=1.0 / samples_per_rev)
    grid = np.arange(0.0, max_order + step / 2, step)
    keep = orders <= max_order + step / 2
    cell = np.clip(np.round(orders[keep] / step).astype(int), 0, len(grid) - 1)
    out = np.zeros((len(grid), x.shape[1]))
    np.maximum.at(out, cell, spec[keep])
    return grid, out

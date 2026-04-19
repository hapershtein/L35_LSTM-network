"""Synthetic dataset: four-tone mixture (1/4 scaled), multi-carrier noise, one-hot context."""

from __future__ import annotations

import numpy as np

SIGNAL_FREQS_HZ = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float64)
# Incommensurate carrier set (irrational-ish ratios; not aligned with 1/3/5/7 Hz).
NOISE_CARRIER_FREQS_HZ = np.array(
    [np.sqrt(2.0), np.pi, np.e, np.sqrt(7.0), np.sqrt(11.0)], dtype=np.float64
)


def _piecewise_constant_schedule(
    n_samples: int,
    seg_len_min: int,
    seg_len_max: int,
    n_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Integer labels 0..n_classes-1, piecewise constant with random segment lengths."""
    labels = np.zeros(n_samples, dtype=np.int32)
    t = 0
    while t < n_samples:
        seg = int(rng.integers(seg_len_min, seg_len_max + 1))
        cls = int(rng.integers(0, n_classes))
        end = min(t + seg, n_samples)
        labels[t:end] = cls
        t = end
    return labels


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    oh = np.zeros((labels.size, n_classes), dtype=np.float64)
    oh[np.arange(labels.size), labels] = 1.0
    return oh


def noise_carrier_waveform(
    t: np.ndarray,
    freqs_hz: np.ndarray,
    phases_rad: np.ndarray,
) -> np.ndarray:
    """
    Normalized sum of sinusoids at incommensurate frequencies; phases fixed in time.
    Shape of phases must match freqs.
    """
    w = 2.0 * np.pi * freqs_hz[:, None] * t[None, :]
    return np.mean(np.sin(w + phases_rad[:, None]), axis=0)


def generate_series(
    fs_hz: float = 1000.0,
    duration_s: float = 10.0,
    signal_phases_rad: np.ndarray | None = None,
    noise_carrier_phases_rad: np.ndarray | None = None,
    a_min: float = 0.0,
    a_max: float = 0.3,
    context_seg_len_min: int = 200,
    context_seg_len_max: int = 800,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """
    Returns per-sample arrays length N:
      t, components (4,N) raw tones, x_mix_scaled, x_obs, noise_carrier, noise_term (=a*n_carrier),
      a_t, context_labels, one_hot C (N,4), y_target
    Mixture uses (1/4) * sum_i sin(2*pi*f_i*t + phi_i). Noise: a(t) * n_carrier(t), a ~ U[a_min,a_max] per step.
    Target: sin(2*pi*f_sel(t)*t) with f_sel from context at time t.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = int(round(fs_hz * duration_s))
    t = np.arange(n, dtype=np.float64) / fs_hz

    if signal_phases_rad is None:
        signal_phases_rad = rng.uniform(-np.pi, np.pi, size=SIGNAL_FREQS_HZ.size)
    if noise_carrier_phases_rad is None:
        noise_carrier_phases_rad = rng.uniform(-np.pi, np.pi, size=NOISE_CARRIER_FREQS_HZ.size)

    w_sig = 2.0 * np.pi * SIGNAL_FREQS_HZ[:, None] * t[None, :]
    components = np.sin(w_sig + signal_phases_rad[:, None])
    mixture = np.sum(components, axis=0)
    x_mix_scaled = 0.25 * mixture

    n_carrier = noise_carrier_waveform(t, NOISE_CARRIER_FREQS_HZ, noise_carrier_phases_rad)
    a_t = rng.uniform(a_min, a_max, size=n)
    noise_term = a_t * n_carrier
    x_obs = x_mix_scaled + noise_term

    labels = _piecewise_constant_schedule(
        n, context_seg_len_min, context_seg_len_max, SIGNAL_FREQS_HZ.size, rng
    )
    c = one_hot(labels, SIGNAL_FREQS_HZ.size)
    f_sel = SIGNAL_FREQS_HZ[labels]
    y_target = np.sin(2.0 * np.pi * f_sel * t)

    return {
        "t": t,
        "components": components.astype(np.float64, copy=False),
        "x_mix_scaled": x_mix_scaled,
        "x_obs": x_obs,
        "noise_carrier": n_carrier,
        "noise_term": noise_term,
        "a_t": a_t,
        "context_labels": labels,
        "C": c,
        "y_target": y_target,
    }


def build_windowed_dataset(
    series: dict[str, np.ndarray],
    window_len: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    X: (num_samples, window_len, 5) = [x_obs, c1..c4] per lag.
    y: (num_samples,) target at end index t of each window.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    x_obs = series["x_obs"]
    c = series["C"]
    y = series["y_target"]
    n = x_obs.shape[0]
    if window_len < 1 or window_len > n:
        raise ValueError("Invalid window_len")
    xw = sliding_window_view(x_obs, window_len)
    c_stacked = np.stack(
        [sliding_window_view(c[:, k], window_len) for k in range(c.shape[1])],
        axis=-1,
    )
    X = np.concatenate([xw[..., np.newaxis], c_stacked], axis=-1)
    y_out = y[window_len - 1 :].astype(np.float64, copy=False)
    return X.astype(np.float64, copy=False), y_out


def replace_context_with_constant_one_hot(X: np.ndarray, class_idx: int) -> np.ndarray:
    """Copy window tensor X (…, 5) and set channels 1:5 to a fixed one-hot for class_idx ∈ {0,1,2,3}."""
    if class_idx < 0 or class_idx >= SIGNAL_FREQS_HZ.size:
        raise ValueError("class_idx must be 0..3")
    Xf = np.array(X, dtype=np.float64, copy=True)
    oh = np.zeros(4, dtype=np.float64)
    oh[class_idx] = 1.0
    Xf[:, :, 1:5] = oh
    return Xf


def ideal_sine_at_window_ends(t: np.ndarray, window_len: int, class_idx: int) -> np.ndarray:
    """y(t_end) = sin(2π f t) for fixed frequency index, aligned with build_windowed_dataset targets."""
    f = float(SIGNAL_FREQS_HZ[class_idx])
    t_end = t[window_len - 1 :]
    return np.sin(2.0 * np.pi * f * t_end)


def ideal_sine_full_grid(t: np.ndarray, class_idx: int) -> np.ndarray:
    """sin(2π f t) on the same time grid as *t*."""
    f = float(SIGNAL_FREQS_HZ[class_idx])
    return np.sin(2.0 * np.pi * f * t)


def time_train_val_split(
    X: np.ndarray, y: np.ndarray, val_fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Last fraction of *samples* (time-ordered windows) for validation."""
    n = X.shape[0]
    split = int(round(n * (1.0 - val_fraction)))
    return X[:split], y[:split], X[split:], y[split:]

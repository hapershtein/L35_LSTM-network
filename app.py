"""
Streamlit UI: LSTM learns to regress y(t)=sin(2*pi*f_sel*t) from noisy mixture + context trajectory.

Run: streamlit run app.py
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

from lstm_signal.data import (
    NOISE_CARRIER_FREQS_HZ,
    SIGNAL_FREQS_HZ,
    build_windowed_dataset,
    generate_series,
    ideal_sine_at_window_ends,
    ideal_sine_full_grid,
    replace_context_with_constant_one_hot,
    time_train_val_split,
)
from lstm_signal.model import build_lstm_regressor

WINDOW_LEN = 50

st.set_page_config(
    page_title="LSTM frequency selection",
    layout="wide",
    initial_sidebar_state="expanded",
)


def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_time_overlay(
    t: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_obs: np.ndarray,
    t_start: float,
    t_span: float,
) -> plt.Figure:
    mask = (t >= t_start) & (t < t_start + t_span)
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    ax[0].plot(t[mask], x_obs[mask], color="#555", lw=0.8, alpha=0.9, label="x_obs (mixture + noise)")
    ax[0].set_ylabel("Input mixture")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(t[mask], y_true[mask], color="#0a6", lw=1.2, label="Target sin(2πf t)")
    ax[1].plot(t[mask], y_pred[mask], color="#c30", lw=1.0, alpha=0.85, label="LSTM prediction")
    ax[1].set_ylabel("Regression")
    ax[1].set_xlabel("Time (s)")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].grid(True, alpha=0.3)
    fig.suptitle(f"Zoom: {t_start:.2f}s … {t_start + t_span:.2f}s", fontsize=11)
    fig.tight_layout()
    return fig


def plot_loss(history: tf.keras.callbacks.History) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(history.history["loss"], label="train MSE")
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], label="val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_component_signals(
    t: np.ndarray,
    components: np.ndarray,
    t_zoom_end: float,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2), sharex=True)
    axes = axes.ravel()
    for i in range(4):
        ax = axes[i]
        m = t <= t_zoom_end
        ax.plot(t[m], components[i, m], lw=0.7, color=f"C{i}")
        ax.set_ylabel(f"{int(SIGNAL_FREQS_HZ[i])} Hz")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Input tone {i + 1}: sin(2π·{int(SIGNAL_FREQS_HZ[i])}·t + φᵢ)", fontsize=9)
    axes[2].set_xlabel("t (s)")
    axes[3].set_xlabel("t (s)")
    fig.suptitle(f"Generated component signals (0 … {t_zoom_end:.2f} s)", fontsize=11)
    fig.tight_layout()
    return fig


def plot_mixture_noise_obs(t: np.ndarray, series: dict, t_zoom_end: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.4))
    m = t <= t_zoom_end
    ax.plot(t[m], series["noise_term"][m], lw=0.6, color="#a60", alpha=0.9, label="Noise: a(t)·n_carrier(t)")
    ax.plot(t[m], series["x_mix_scaled"][m], lw=0.7, color="#06a", alpha=0.85, label="Mixture (÷4 on Σ tones)")
    ax.plot(t[m], series["x_obs"][m], lw=0.55, color="#333", alpha=0.9, label="x_obs (mixture + noise)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Noise term, scaled mixture, and observed input (0 … {t_zoom_end:.2f} s)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_filtered_output(
    t: np.ndarray,
    y_ideal: np.ndarray,
    t_end: np.ndarray | None,
    y_lstm: np.ndarray | None,
    f_hz: float,
    t_start: float,
    t_span: float,
    title_suffix: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    mask_full = (t >= t_start) & (t < t_start + t_span)
    ax.plot(
        t[mask_full],
        y_ideal[mask_full],
        color="#0a6",
        lw=1.3,
        label=f"Ideal filtered output: sin(2π·{f_hz:.0f}·t)",
    )
    if t_end is not None and y_lstm is not None:
        mw = (t_end >= t_start) & (t_end < t_start + t_span)
        ax.plot(
            t_end[mw],
            y_lstm[mw],
            color="#c30",
            lw=0.9,
            alpha=0.85,
            marker=".",
            ms=3,
            label="LSTM output (constant one-hot in window)",
        )
    ax.set_xlabel("t (s)")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Filtered output — {title_suffix}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_spectrum(t: np.ndarray, sig: np.ndarray, title: str) -> plt.Figure:
    dt = float(t[1] - t[0])
    n = sig.size
    freqs = np.fft.rfftfreq(n, d=dt)
    spec = np.abs(np.fft.rfft(sig * np.hanning(n))) / max(n, 1)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(freqs, spec, lw=0.9)
    for f in SIGNAL_FREQS_HZ:
        ax.axvline(f, color="#080", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlim(0, 15)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|FFT| (windowed)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def init_state() -> None:
    if "trained_model" not in st.session_state:
        st.session_state.trained_model = None
    if "train_history" not in st.session_state:
        st.session_state.train_history = None
    if "last_series" not in st.session_state:
        st.session_state.last_series = None
    if "last_Xy" not in st.session_state:
        st.session_state.last_Xy = None
    if "last_data_key" not in st.session_state:
        st.session_state.last_data_key = None
    if "data_nonce" not in st.session_state:
        st.session_state.data_nonce = 0


init_state()

st.title("LSTM frequency selection demo")
st.caption(
    "Mixture: 1/4 · Σ sin(2πfᵢt+φᵢ) for fᵢ ∈ {1,3,5,7} Hz. "
    "Noise: a(t)·n_carrier(t) with per-step random a(t) and n_carrier = mean_k sin(2πνₖt+ψₖ) "
    f"at incommensurate ν (Hz): {', '.join(f'{v:.4f}' for v in NOISE_CARRIER_FREQS_HZ)}. "
    f"Window L={WINDOW_LEN}. Target: sin(2πf_sel·t) from piecewise-constant one-hot context."
)

with st.sidebar:
    st.header("Data generation")
    data_seed = st.slider("Dataset RNG seed", 0, 9999, 42, step=1)
    a_min = st.slider("Noise amplitude a(t) — min", 0.0, 1.0, 0.05, 0.01)
    a_max = st.slider("Noise amplitude a(t) — max", 0.0, 1.0, 0.40, 0.01)
    if a_max < a_min:
        st.warning("a_max < a_min; values will be swapped for generation.")
    seg_min = st.slider("Context segment min (samples)", 50, 2000, 200, 10)
    seg_max = st.slider("Context segment max (samples)", 50, 3000, 800, 10)
    if seg_max < seg_min:
        st.warning("Segment max < min; they will be swapped.")

    st.subheader("Signal phases φᵢ (rad)")
    phi_cols = st.columns(2)
    signal_phases = np.zeros(4, dtype=np.float64)
    for i in range(4):
        with phi_cols[i % 2]:
            signal_phases[i] = st.slider(
                f"φ{i+1} ({int(SIGNAL_FREQS_HZ[i])} Hz)",
                -float(np.pi),
                float(np.pi),
                float((-1) ** i * 0.4),
                0.05,
                key=f"phi_{i}",
            )

    st.subheader("Noise carrier phases ψₖ (constant)")
    psi_cols = st.columns(2)
    noise_phases = np.zeros(NOISE_CARRIER_FREQS_HZ.size, dtype=np.float64)
    for k in range(NOISE_CARRIER_FREQS_HZ.size):
        with psi_cols[k % 2]:
            noise_phases[k] = st.slider(
                f"ψ{k+1}",
                -float(np.pi),
                float(np.pi),
                float(0.25 * k),
                0.05,
                key=f"psi_{k}",
            )

    st.header("Model & training")
    lstm_units = st.slider("LSTM hidden units", 16, 256, 64, 8)
    num_layers = st.slider("LSTM layers", 1, 3, 1, 1)
    dropout = st.slider("Dropout (after last LSTM)", 0.0, 0.5, 0.0, 0.05)
    lr = st.select_slider(
        "Learning rate",
        options=[1e-4, 3e-4, 1e-3, 3e-3],
        value=1e-3,
    )
    batch_size = st.slider("Batch size", 16, 256, 64, 16)
    epochs = st.slider("Epochs", 5, 200, 50, 5)
    val_frac = st.slider("Validation fraction (tail)", 0.05, 0.4, 0.2, 0.05)
    tf_seed = st.number_input("TensorFlow seed", min_value=0, max_value=2**31 - 1, value=123, step=1)

    regen = st.button("Regenerate data (new a(t) & context)", use_container_width=True)
    train_btn = st.button("Train LSTM", type="primary", use_container_width=True)

    with st.expander("Noise carrier frequencies νₖ (Hz, fixed)"):
        st.table(
            {
                "k": list(range(1, NOISE_CARRIER_FREQS_HZ.size + 1)),
                "νₖ (Hz)": [float(v) for v in NOISE_CARRIER_FREQS_HZ],
            }
        )

a_lo, a_hi = (a_min, a_max) if a_min <= a_max else (a_max, a_min)
s_lo, s_hi = (seg_min, seg_max) if seg_min <= seg_max else (seg_max, seg_min)

data_key = (
    int(data_seed),
    float(a_lo),
    float(a_hi),
    int(s_lo),
    int(s_hi),
    tuple(float(x) for x in signal_phases),
    tuple(float(x) for x in noise_phases),
)
if data_key != st.session_state.last_data_key:
    st.session_state.trained_model = None
    st.session_state.train_history = None
    st.session_state.last_data_key = data_key

if regen:
    st.session_state.data_nonce = int(st.session_state.data_nonce) + 1
    st.session_state.trained_model = None
    st.session_state.train_history = None

rng = np.random.default_rng((int(data_seed), int(st.session_state.data_nonce)))

series = generate_series(
    rng=rng,
    signal_phases_rad=signal_phases,
    noise_carrier_phases_rad=noise_phases,
    a_min=a_lo,
    a_max=a_hi,
    context_seg_len_min=int(s_lo),
    context_seg_len_max=int(s_hi),
)
st.session_state.last_series = series
X, y = build_windowed_dataset(series, window_len=WINDOW_LEN)
st.session_state.last_Xy = (X, y)

series = st.session_state.last_series
X, y = st.session_state.last_Xy

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Samples", f"{series['t'].size}")
with col_b:
    st.metric("Windowed examples", f"{X.shape[0]}")
with col_c:
    st.metric("Input channels / step", "5 (x_obs + 4× one-hot)")

tab_data, tab_onehot, tab_train, tab_viz = st.tabs(
    ["Signals & context", "One-hot & filtered output", "Training", "Evaluation & spectra"]
)

with tab_data:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Observed mixture + noise")
        f1, ax = plt.subplots(figsize=(10, 2.8))
        t = series["t"]
        ax.plot(t, series["x_obs"], lw=0.6, color="#333")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("x_obs")
        ax.set_xlim(0, min(2.0, float(t[-1])))
        ax.grid(True, alpha=0.3)
        st.pyplot(f1)
        plt.close(f1)
    with c2:
        st.subheader("Scaled mixture only (÷4 on sum of tones)")
        f2, ax = plt.subplots(figsize=(10, 2.8))
        ax.plot(t, series["x_mix_scaled"], lw=0.7, color="#06a")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("x_mix_scaled")
        ax.set_xlim(0, min(2.0, float(t[-1])))
        ax.grid(True, alpha=0.3)
        st.pyplot(f2)
        plt.close(f2)

    st.subheader("Piecewise context (selected frequency index)")
    f3, ax = plt.subplots(figsize=(10, 2.2))
    ax.step(t, series["context_labels"], where="post", lw=0.9)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("class (0..3)")
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(True, alpha=0.3)
    st.pyplot(f3)
    plt.close(f3)

with tab_onehot:
    st.subheader("4-bit one-hot selection vector (context)")
    freq_options = [f"{int(f)} Hz — one-hot {['[1,0,0,0]', '[0,1,0,0]', '[0,0,1,0]', '[0,0,0,1]'][i]}" for i, f in enumerate(SIGNAL_FREQS_HZ)]
    sel_idx = st.radio(
        "Select frequency to extract (fixes **c** = one-hot over the whole past window for the plots below)",
        range(4),
        format_func=lambda i: freq_options[i],
        horizontal=True,
        key="onehot_sel_idx",
    )
    bit_cols = st.columns(4)
    bits = [1 if i == sel_idx else 0 for i in range(4)]
    for i in range(4):
        with bit_cols[i]:
            st.metric(label=f"b{i + 1} ({int(SIGNAL_FREQS_HZ[i])} Hz)", value=str(bits[i]))

    zt = st.slider("Plot zoom (seconds from start)", 0.2, 5.0, 1.5, 0.1, key="onehot_zoom")

    st.markdown("**Generated input signals** (underlying tones, noise, mixture, observation)")
    comp = series["components"]
    st.pyplot(plot_component_signals(series["t"], comp, zt))
    st.pyplot(plot_mixture_noise_obs(series["t"], series, zt))

    f_sel = float(SIGNAL_FREQS_HZ[sel_idx])
    y_ideal_full = ideal_sine_full_grid(series["t"], sel_idx)
    st.markdown(
        f"**Filtered output (ideal)** for selection **f = {int(f_sel)} Hz**: "
        r"$y(t)=\sin(2\pi f t)$ on the full grid."
    )

    t_end = series["t"][WINDOW_LEN - 1 : WINDOW_LEN - 1 + X.shape[0]]
    y_ideal_w = ideal_sine_at_window_ends(series["t"], WINDOW_LEN, sel_idx)

    model_oh = st.session_state.trained_model
    if model_oh is not None:
        X_force = replace_context_with_constant_one_hot(X, int(sel_idx))
        y_lstm_force = model_oh.predict(X_force, batch_size=512, verbose=0).reshape(-1)
        rmse_f = float(np.sqrt(np.mean((y_lstm_force - y_ideal_w) ** 2)))
        st.caption(
            "Orange: LSTM regression when each window’s context channels are forced to this one-hot "
            "(scalar input **x_obs** still the real mixture+noise)."
        )
        st.metric("RMSE (forced one-hot vs ideal at window ends)", f"{rmse_f:.5f}")
        t_s = st.slider("Filtered plot zoom start (s)", 0.0, float(series["t"][-1]) - 0.05, 0.0, 0.05, key="fo_t0")
        t_sp = st.slider("Filtered plot zoom span (s)", 0.05, 3.0, 0.6, 0.05, key="fo_span")
        st.pyplot(
            plot_filtered_output(
                series["t"],
                y_ideal_full,
                t_end,
                y_lstm_force,
                f_sel,
                float(t_s),
                float(t_sp),
                f"{int(f_sel)} Hz + trained LSTM",
            )
        )
    else:
        st.info("Train the LSTM in the **Training** tab to overlay **learned** filtered output for this one-hot.")
        t_s = st.slider("Filtered plot zoom start (s)", 0.0, float(series["t"][-1]) - 0.05, 0.0, 0.05, key="fo_t0_nt")
        t_sp = st.slider("Filtered plot zoom span (s)", 0.05, 3.0, 0.6, 0.05, key="fo_span_nt")
        st.pyplot(
            plot_filtered_output(
                series["t"],
                y_ideal_full,
                None,
                None,
                f_sel,
                float(t_s),
                float(t_sp),
                f"{int(f_sel)} Hz (ideal only)",
            )
        )

with tab_train:
    st.write(
        "Training uses the **current sidebar** data parameters. "
        "Click **Train LSTM** after changing sliders to refit."
    )
    if train_btn:
        tf.keras.utils.set_random_seed(int(tf_seed))
        X_tr, y_tr, X_va, y_va = time_train_val_split(X, y, val_fraction=float(val_frac))
        model = build_lstm_regressor(
            WINDOW_LEN,
            lstm_units=int(lstm_units),
            num_lstm_layers=int(num_layers),
            dropout=float(dropout),
        )
        opt = tf.keras.optimizers.Adam(learning_rate=float(lr))
        model.compile(optimizer=opt, loss="mse", metrics=["mae"])
        prog = st.progress(0.0, text="Training…")
        class _cb(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                prog.progress((epoch + 1) / float(epochs), text=f"Epoch {epoch + 1}/{epochs}")

        hist = model.fit(
            X_tr,
            y_tr,
            validation_data=(X_va, y_va),
            epochs=int(epochs),
            batch_size=int(batch_size),
            verbose=0,
            callbacks=[_cb()],
        )
        prog.empty()
        st.session_state.trained_model = model
        st.session_state.train_history = hist
        st.success("Training finished.")

    if st.session_state.train_history is not None:
        st.image(fig_to_bytes(plot_loss(st.session_state.train_history)), use_container_width=True)

with tab_viz:
    model = st.session_state.trained_model
    if model is None:
        st.info("Train a model in the **Training** tab to see predictions and spectra.")
    else:
        y_hat = model.predict(X, batch_size=512, verbose=0).reshape(-1)
        # Align predictions with time indices (window ends at L-1 + i)
        t = series["t"]
        L = WINDOW_LEN
        t_end = t[L - 1 : L - 1 + y_hat.size]
        y_true_aligned = series["y_target"][L - 1 : L - 1 + y_hat.size]
        x_obs_aligned = series["x_obs"][L - 1 : L - 1 + y_hat.size]

        err = y_hat - y_true_aligned
        st.metric("RMSE (all windows)", f"{float(np.sqrt(np.mean(err**2))):.5f}")

        z1, z2 = st.columns(2)
        with z1:
            t_start = st.slider("Zoom start (s)", 0.0, float(t[-1]) - 0.05, 1.0, 0.05)
        with z2:
            t_span = st.slider("Zoom span (s)", 0.05, 2.0, 0.5, 0.05)

        st.image(
            fig_to_bytes(
                plot_time_overlay(
                    t_end,
                    y_true_aligned,
                    y_hat,
                    x_obs_aligned,
                    t_start=float(t_start),
                    t_span=float(t_span),
                )
            ),
            use_container_width=True,
        )

        s1, s2 = st.columns(2)
        with s1:
            st.image(
                fig_to_bytes(plot_spectrum(t_end, y_true_aligned, "FFT: target (windowed)")),
                use_container_width=True,
            )
        with s2:
            st.image(
                fig_to_bytes(plot_spectrum(t_end, y_hat, "FFT: prediction (windowed)")),
                use_container_width=True,
            )

        st.subheader("Per-class RMSE (frequency at window end)")
        labels = series["context_labels"][L - 1 : L - 1 + y_hat.size]
        cols = st.columns(4)
        for fi in range(4):
            m = labels == fi
            if np.any(m):
                r = float(np.sqrt(np.mean(err[m] ** 2)))
            else:
                r = float("nan")
            with cols[fi]:
                st.metric(f"{int(SIGNAL_FREQS_HZ[fi])} Hz", f"{r:.5f}" if np.isfinite(r) else "—")

"""Optional CLI training (same logic as the Streamlit app)."""

from __future__ import annotations

import argparse

import numpy as np

from lstm_signal.data import (
    SIGNAL_FREQS_HZ,
    build_windowed_dataset,
    generate_series,
    time_train_val_split,
)
from lstm_signal.model import build_lstm_regressor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lstm-units", type=int, default=64)
    p.add_argument("--window", type=int, default=50)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    series = generate_series(
        rng=rng,
        a_min=0.05,
        a_max=0.35,
        context_seg_len_min=200,
        context_seg_len_max=800,
    )
    X, y = build_windowed_dataset(series, window_len=args.window)
    X_tr, y_tr, X_va, y_va = time_train_val_split(X, y, val_fraction=0.2)

    model = build_lstm_regressor(args.window, lstm_units=args.lstm_units)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=1,
    )
    print("Signal frequencies (Hz):", SIGNAL_FREQS_HZ)


if __name__ == "__main__":
    main()

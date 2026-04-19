# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- Standardized changelog structure to Keep a Changelog format with release sections.

## [0.1.0] - 2026-04-19

### Added

- TensorFlow LSTM regression pipeline for frequency-selective sine reconstruction.
- Synthetic data generation with:
  - 4 base tones at 1, 3, 5, and 7 Hz
  - 10 s duration at 1000 Hz sampling
  - Mixture normalization using `1/4` on summed tones only
  - Incommensurate multi-carrier noise
  - Per-step random amplitude `a(t)`
  - Constant carrier phase configuration
  - Piecewise-constant one-hot context schedule
- Windowed dataset builder with window length `L=50`.
- Streamlit UI (`app.py`) with:
  - Data generation sliders
  - Model/training controls
  - Signal and context plots
  - One-hot selector tab (4-bit context)
  - Generated input signal displays
  - Filtered output display for selected one-hot vector
  - Evaluation metrics and spectra
- CLI training script (`train.py`).
- Project documentation:
  - `Docs/README.md`
  - `Docs/PRD.md`
  - `Docs/TASKS.md`

### Notes

- Current implementation is intended for educational/demo usage.
- TensorFlow runs on CPU in native Windows setup for standard installs.

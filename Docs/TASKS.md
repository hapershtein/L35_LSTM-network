# Tasks

This checklist tracks implementation and documentation status for the LSTM frequency-selection project.

## Phase 1 — Foundations

- [x] Define core signal spec (frequencies, sampling, duration).
- [x] Define target equation \(y(t)=\sin(2\pi f_{\text{sel}} t)\).
- [x] Choose framework: TensorFlow.
- [x] Set window length \(L=50\).

## Phase 2 — Data pipeline

- [x] Implement 4-tone mixture generation.
- [x] Apply `1/4` normalization to mixture sum only.
- [x] Implement incommensurate multi-carrier noise.
- [x] Implement per-step random amplitude \(a(t)\).
- [x] Keep noise carrier phases constant over time.
- [x] Implement piecewise-constant one-hot context schedule.
- [x] Implement windowed dataset builder `(Nw, 50, 5)`.
- [x] Implement helper utilities for fixed one-hot override.

## Phase 3 — Model and training

- [x] Implement TensorFlow LSTM regressor.
- [x] Add configurable layers/units/dropout.
- [x] Add training loop with validation split.
- [x] Add quick CLI training entry point (`train.py`).

## Phase 4 — Interactive UI

- [x] Build Streamlit app shell and sidebar controls.
- [x] Add data generation controls (seed, amplitudes, phases, context segments).
- [x] Add model/training controls.
- [x] Add signals/context visualization tab.
- [x] Add one-hot selector UI (4-bit context).
- [x] Display generated input signals (components + mixture/noise/observed).
- [x] Display filtered output according to selected one-hot.
- [x] Add evaluation tab with metrics and spectra.

## Phase 5 — Validation

- [x] Smoke-test data generation and window shapes.
- [x] Smoke-test short training run after dependency install.
- [x] Run lint diagnostics for edited files.

## Phase 6 — Documentation

- [x] Add `Docs/README.md`.
- [x] Add `Docs/PRD.md`.
- [x] Add `Docs/TASKS.md`.

## Optional next tasks

- [ ] Add model save/load controls in UI.
- [ ] Add export of plots/results to `Docs/artifacts/`.
- [ ] Add ablation toggles (no-context / no-noise / no-scaling).
- [ ] Add unit tests for data utilities.
- [ ] Add CI workflow for lint + smoke training.

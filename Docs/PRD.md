# Product Requirements Document (PRD)

## Product name

LSTM Frequency Selection Demo

## Purpose

Build an educational project that explains how an LSTM can use a context vector to isolate one target frequency from a noisy mixed signal and regress a mathematically pure sine output.

## Goals

- Demonstrate sequence regression with an LSTM on synthetic signals.
- Show the role of a **one-hot selection vector** in conditioning output frequency.
- Provide a rich, interactive UI for data generation, model training, and visualization.
- Keep the pipeline reproducible and configurable.

## Non-goals

- Production-grade deployment
- Real-time low-latency inference service
- Advanced hyperparameter search automation

## User stories

- As a learner, I want to change signal/noise parameters and see how training changes.
- As a learner, I want to choose a one-hot context and inspect filtered output behavior.
- As an instructor, I want plots that clearly show input components, mixed signal, and prediction.

## Functional requirements

### FR1 — Signal generation

- Generate four sinusoidal components at frequencies:
  - 1 Hz, 3 Hz, 5 Hz, 7 Hz
- Duration: 10 seconds
- Sample rate: 1000 Hz

### FR2 — Mixture normalization

- Mixed signal must use:
  - \(x_{\text{mix}}(t)=\frac{1}{4}\sum_{i=1}^{4}\sin(2\pi f_i t + \phi_i)\)
- `1/4` scaling is applied only to the component sum.

### FR3 — Noise model

- Use sum of incommensurate carriers:
  - \(n_{\text{carrier}}(t)=\frac{1}{K}\sum_{k}\sin(2\pi \nu_k t+\psi_k)\)
- Carrier phases \(\psi_k\) are constant over time.
- Noise amplitude \(a(t)\) is randomized per time step.
- Observed input:
  - \(x_{\text{obs}}(t)=x_{\text{mix}}(t)+a(t)\,n_{\text{carrier}}(t)\)

### FR4 — Context schedule

- Context is a 4-bit one-hot vector.
- Default dataset uses piecewise-constant context segments.
- UI provides explicit selector for fixed one-hot filtering exploration.

### FR5 — Regression target

- Target must be pure sine:
  - \(y(t)=\sin(2\pi f_{\text{sel}} t)\)
- \(f_{\text{sel}}\) is determined by one-hot context.

### FR6 — Model

- Framework: TensorFlow/Keras.
- Sequence model: LSTM-based regressor.
- Input window length: 50.
- Input channels per time step: 5 (`x_obs + one-hot(4)`).
- Output: scalar regression at window end.

### FR7 — UI (Streamlit)

- Controls for data generation and model/training params.
- Tabs for:
  - Signals/context
  - One-hot selector & filtered output
  - Training
  - Evaluation/spectra
- Display generated components, mixture/noise/observed signal, and filtered output overlays.

## Non-functional requirements

- Reproducibility with user-defined seeds.
- Responsive interaction for typical CPU laptop usage.
- Clear visualizations suitable for teaching/demo.

## Acceptance criteria

- App runs with:
  - `streamlit run app.py`
- User can:
  - Regenerate data with controls
  - Train LSTM from UI
  - Select one-hot context and inspect filtered output
- Plots include:
  - Underlying component signals
  - Mixture/noise/observed input
  - Target vs prediction overlays
- CLI training works:
  - `python train.py ...`

## Risks and mitigations

- **Risk**: Underfitting/overfitting for some parameter settings  
  - **Mitigation**: expose units/layers/dropout/lr/batch/epochs sliders.
- **Risk**: Training variability due to random signals/noise  
  - **Mitigation**: fixed seeds and deterministic data keying.
- **Risk**: Misinterpretation of one-hot effect  
  - **Mitigation**: dedicated one-hot tab with explicit fixed-context plots.

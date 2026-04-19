# LSTM Frequency Selection Demo

This project demonstrates how an LSTM can learn to recover a **target pure sine wave** from a noisy mixed signal, using a **4-bit one-hot context vector** as a frequency selector.

## What this project shows

- Mixed input signal with 4 components at **1, 3, 5, 7 Hz**
- Additive structured noise built from **incommensurate carriers**
- Per-time-step randomized noise amplitude \(a(t)\)
- Piecewise-constant one-hot context (4-bit) to select desired frequency
- Regression target:
  - \(y(t) = \sin(2\pi f_{\text{sel}} t)\)

## Signal model

- **Sampling rate**: 1000 Hz  
- **Duration**: 10 s  
- **Samples**: 10,000  
- **Mixture normalization**:
  - \(x_{\text{mix}}(t) = \frac{1}{4}\sum_{i=1}^{4}\sin(2\pi f_i t + \phi_i)\)
- **Noise**:
  - \(n_{\text{carrier}}(t)=\frac{1}{K}\sum_{k=1}^{K}\sin(2\pi \nu_k t + \psi_k)\)
  - \(\psi_k\) are constant in time
  - \(a(t)\sim U(a_{\min}, a_{\max})\), randomized per step
  - \(x_{\text{obs}}(t)=x_{\text{mix}}(t)+a(t)\,n_{\text{carrier}}(t)\)

## Model input and target

- Window length: **L = 50**
- Per-step feature vector inside a window: `[x_obs, c1, c2, c3, c4]`
- Input tensor shape: `(num_samples, 50, 5)`
- Target aligned to window end index.

## Tech stack

- TensorFlow / Keras
- NumPy
- Matplotlib
- Streamlit

## Project structure

- `app.py` — Streamlit UI (data controls, one-hot selector, plots, training)
- `train.py` — CLI training entry point
- `lstm_signal/data.py` — signal generation and dataset utilities
- `lstm_signal/model.py` — LSTM regressor builder
- `requirements.txt` — dependencies
- `Docs/` — project docs (this folder)

## Setup

```powershell
cd "c:\25D\L35_LSTM network"
pip install -r requirements.txt
```

## Run Streamlit UI

```powershell
streamlit run app.py
```

## Run CLI training

```powershell
python train.py --seed 42 --epochs 40 --batch 64 --lstm-units 64 --window 50
```

## UI highlights

- Sliders for:
  - Noise amplitude range `a_min`, `a_max`
  - Signal phases \(\phi_i\)
  - Noise phases \(\psi_k\)
  - Context segment lengths
  - LSTM/training hyperparameters
- One-hot selector tab:
  - Choose a fixed 4-bit context
  - View generated component signals
  - View ideal filtered output and model output overlay

## Notes

- Normalization factor `1/4` is applied to the **mixture only**.
- Noise phases are constant in time; only amplitude \(a(t)\) is randomized per step.

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

## Setup and run from WSL

Use this if you want to run TensorFlow/Streamlit inside WSL (recommended on Windows).

1. Open Ubuntu (WSL) and go to the project mounted under `/mnt`:

```bash
cd "/mnt/c/25D/L35_LSTM network"
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Run Streamlit:

```bash
streamlit run app.py
```

4. Open the URL shown in terminal from Windows (typically):

- `http://localhost:8501`

Optional: run with explicit host/port if needed:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

5. Run CLI training from WSL:

```bash
python train.py --seed 42 --epochs 40 --batch 64 --lstm-units 64 --window 50
```

### WSL troubleshooting

- `python3 -m venv .venv` fails with `ensurepip`/`venv` error:
  - Install venv package, then retry:
  ```bash
  sudo apt update
  sudo apt install -y python3-venv
  ```

- `streamlit: command not found`:
  - Ensure venv is active and reinstall:
  ```bash
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

- Browser cannot reach app on `localhost:8501`:
  - Confirm Streamlit is running and bound correctly:
  ```bash
  streamlit run app.py --server.address 0.0.0.0 --server.port 8501
  ```
  - Check if another process already uses port 8501:
  ```bash
  ss -ltnp | rg 8501
  ```
  - If needed, switch to another port (for example 8502):
  ```bash
  streamlit run app.py --server.address 0.0.0.0 --server.port 8502
  ```

- TensorFlow is slow or no GPU in Windows:
  - Native Windows TensorFlow commonly runs on CPU for this setup.
  - WSL2 + Linux CUDA stack is the typical route for GPU acceleration.

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

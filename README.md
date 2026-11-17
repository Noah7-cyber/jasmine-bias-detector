# Jasmine AI Bias Detector

A Streamlit application for reviewing binary prediction differences across demographic groups. It supports ProPublica COMPAS, held-out UCI Adult baseline predictions, reproducible synthetic scenarios, and session-only CSV uploads.

## Run locally

See [docs/APP_GUIDE.md](docs/APP_GUIDE.md) for environment setup and data preparation. In brief:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/prepare_public_data.py
.\.venv\Scripts\python.exe scripts/generate_synthetic.py
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

## Project documents

- [Implementation plan](docs/IMPLEMENTATION_PLAN.md)
- [Metric and evaluation contract](docs/METRICS_AND_EVALUATION.md)
- [Run guide](docs/APP_GUIDE.md)
- [Visual direction](docs/DESIGN.md)

This tool presents fairness metrics as indicators for investigation; it does not make legal findings or determine that a system is fair or unfair.

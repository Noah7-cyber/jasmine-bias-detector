# Jasmine AI Bias Detector

A Python application for reviewing binary prediction differences across demographic groups. It supports ProPublica COMPAS, held-out UCI Adult baseline predictions, reproducible synthetic scenarios, and CSV uploads processed in memory. Vercel uses a Flask web interface; the proposal's original Streamlit research interface is included separately. Both use the same Python metric engine and ReportLab PDF generator.

## Deploy on Vercel

Import this repository at its root. The tracked `vercel.json` selects Flask; `app.py` exports a real WSGI `app`, and `pyproject.toml` explicitly selects `app:app` and Python 3.12. Leave Build Command and Output Directory unset. No environment variables, database or build-time dataset download is required. The sanitised public samples are bundled.

The hosted interface uses a 3 MiB CSV limit to leave room below Vercel's request-body limit. Data are read in each request and held in the browser's memory between requests; refreshing or clearing removes that workspace. Neither CSVs nor reports are stored by the application. The local Streamlit interface retains its 25 MiB limit.

## Run the Vercel interface locally

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m flask --app app run --host 127.0.0.1 --port 8501
```

## Run the Streamlit research interface

See [docs/APP_GUIDE.md](docs/APP_GUIDE.md) for environment setup and data preparation. In brief:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\.venv\Scripts\python.exe scripts/prepare_public_data.py
.\.venv\Scripts\python.exe scripts/generate_synthetic.py
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

## Project documents

- [Implementation plan](docs/IMPLEMENTATION_PLAN.md)
- [Metric and evaluation contract](docs/METRICS_AND_EVALUATION.md)
- [Run guide](docs/APP_GUIDE.md)
- [Visual direction](docs/DESIGN.md)

This tool presents fairness metrics as indicators for investigation; it does not make legal findings or determine that a system is fair or unfair.

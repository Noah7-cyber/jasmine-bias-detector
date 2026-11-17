# Jasmine AI Bias Detector — Run guide

## Requirements

Python 3.12 and a modern browser. Create a dedicated environment in this directory; do not use the coursework environment. For the hosted Flask interface, install `requirements.txt` and run `python -m flask --app app run`. Vercel loads the exported WSGI `app` automatically. For the Streamlit research interface and evaluation tools, use the following commands.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\.venv\Scripts\python.exe scripts/prepare_public_data.py
.\.venv\Scripts\python.exe scripts/generate_synthetic.py
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

The preparation command downloads COMPAS and UCI Adult data in memory, removes COMPAS direct identifiers before writing the offline sample, fits an untuned Adult logistic regression on the training partition, and saves only held-out predictions with selected demographic fields. Raw source downloads and uploaded rows are not written to disk. Sample preparation requires a network connection; the launched app can run offline after preparation.

Stop the app with Ctrl+C. User uploads are temporary to the current Streamlit session. Use **Clear current dataset and analysis** before leaving a shared browser session. Do not upload identifiable data.

## Analysis inputs

Choose one demographic attribute, a binary prediction column or numerical score, optional binary truth, and mappings for labels. Score prediction is positive at or above the selected threshold. COMPAS defaults to risk score >=5; positive is a higher risk flag. Screening defaults are DPD >=0.10, DIR <0.80, minimum group size 30, and two-sided p<0.05.

Undefined rates remain unavailable with an explanation. Sparse expected z-test cells prompt a caution. Results are indicators for investigation and do not establish legal discrimination.

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/evaluate.py
```

The evaluation command should publish numeric outputs only under `evaluation/`. Review [`METRICS_AND_EVALUATION.md`](METRICS_AND_EVALUATION.md) for targets and unresolved research interpretations.

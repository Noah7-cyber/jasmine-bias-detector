"""Stateless WSGI entrypoint for Vercel. No uploaded data is stored or cached."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import plotly
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
from io import BytesIO

from bias_detector.data_loader import read_uploaded_csv
from bias_detector.metrics import AuditSettings, calculate_audit
from bias_detector.reporting import build_pdf

ROOT = Path(__file__).resolve().parent
SAMPLES = {
    "compas": ("compas.csv", "COMPAS recidivism risk", "https://github.com/propublica/compas-analysis", "higher recidivism risk (adverse flag)", True),
    "adult": ("adult_holdout.csv", "Adult held-out predictions", "https://archive.ics.uci.edu/dataset/2/adult", "income above $50,000", False),
    **{f"s{i}": (f"synthetic_s{i}.csv", label, "Fixed-seed synthetic reference", "positive prediction", False)
       for i, label in enumerate(["S1: Equal rates", "S2: Below threshold", "S3: Moderate disparity", "S4: Large disparity", "S5: Small group", "S6: Error-rate disparity"], 1)},
}

app = Flask(__name__)
# Leave room below Vercel's request limit for JSON encoding and configuration.
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024


@app.after_request
def response_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.errorhandler(ValueError)
def invalid_input(error):
    return jsonify(error=str(error)), 400


@app.errorhandler(BadRequest)
def invalid_json(error):
    return jsonify(error="Send a valid JSON request with a dataset and analysis settings."), 400


@app.errorhandler(RequestEntityTooLarge)
def oversized_input(error):
    return jsonify(error="The request is too large. Use a CSV smaller than 3 MiB."), 413


def payload():
    data = request.get_json()
    if not isinstance(data, dict):
        raise ValueError("Provide a dataset and analysis settings.")
    return data


def load_dataset(data):
    if "sample" in data:
        if not isinstance(data["sample"], str) or data["sample"] not in SAMPLES:
            raise ValueError("Choose a recognised built-in sample.")
        filename, name, source, positive, adverse = SAMPLES[data["sample"]]
        frame = pd.read_csv(ROOT / "data" / "samples" / filename)
    else:
        csv = data.get("csv")
        if not isinstance(csv, str) or not csv.strip():
            raise ValueError("Choose a sample or upload a CSV containing data rows.")
        raw = csv.encode("utf-8")
        if len(raw) > 3 * 1024 * 1024:
            raise ValueError("Use a CSV smaller than 3 MiB for this hosted version.")
        try:
            frame, duplicates = read_uploaded_csv(raw)
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError("Could not parse the CSV. Check the header, delimiters and consistent column counts.") from exc
        if duplicates:
            raise ValueError("Rename duplicate column headers before uploading.")
        name, source, positive, adverse = "Uploaded CSV", "User-provided CSV (memory only)", "configured positive class", False
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("Include at least two columns and one data row: a demographic attribute and a prediction.")
    if len(frame) > 500_000 or len(frame.columns) > 200:
        raise ValueError("Use at most 500,000 rows and 200 columns.")
    return frame, dict(name=name, source=source, positive=positive, adverse=adverse)


def settings_for(data, frame):
    raw = data.get("settings")
    if not isinstance(raw, dict):
        raise ValueError("Choose the analysis columns first.")
    try:
        settings = AuditSettings(**raw)
    except TypeError as exc:
        raise ValueError("The analysis settings are incomplete or invalid.") from exc
    columns = [settings.group_column, settings.prediction_column]
    if settings.true_column is not None:
        columns.append(settings.true_column)
    if any(not isinstance(c, str) or c not in frame.columns for c in columns):
        raise ValueError("Choose existing demographic, prediction and optional outcome columns.")
    if len(columns) != len(set(columns)):
        raise ValueError("Choose different columns for demographics, predictions and true outcomes.")
    if not isinstance(settings.score_mode, bool):
        raise ValueError("Prediction mode must be binary labels or numerical scores.")
    for number in (settings.score_threshold, settings.dpd_threshold, settings.dir_threshold):
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError("Enter finite numerical thresholds.")
    if type(settings.minimum_group_size) is not int or settings.minimum_group_size < 1:
        raise ValueError("Minimum group size must be a positive whole number.")
    if frame[settings.group_column].nunique() > 100:
        raise ValueError("Choose a demographic field with at most 100 groups, rather than an identifier.")
    for a, b in ((settings.positive_label, settings.negative_label), (settings.true_positive_label or "1", settings.true_negative_label or "0")):
        if not isinstance(a, str) or not isinstance(b, str) or a == b:
            raise ValueError("Choose distinct positive and negative labels.")
    return settings


@app.get("/")
def home():
    return render_template("index.html", samples=SAMPLES)


@app.get("/api/health")
def health():
    return jsonify(status="ok")


@app.get("/assets/plotly.min.js")
def plotly_script():
    return send_file(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", mimetype="text/javascript", max_age=86400)


@app.post("/api/inspect")
def inspect():
    frame, metadata = load_dataset(payload())
    columns = []
    for column in frame:
        values = frame[column].dropna().astype(str).str.strip().unique()
        columns.append(dict(name=str(column), missing=int(frame[column].isna().sum()), values=sorted(values.tolist()) if len(values) <= 100 else [], unique=len(values)))
    return jsonify(metadata=metadata, rows=len(frame), columns=columns)


@app.post("/api/analyze")
def analyze():
    data = payload()
    frame, metadata = load_dataset(data)
    result = calculate_audit(frame, settings_for(data, frame), positive_is_adverse=metadata["adverse"])
    return jsonify(result=result, metadata=metadata)


@app.post("/api/report")
def report():
    data = payload()
    frame, metadata = load_dataset(data)
    result = calculate_audit(frame, settings_for(data, frame), positive_is_adverse=metadata["adverse"])
    pdf = build_pdf(result, "AI Bias Detector - Summary Report", metadata["name"], metadata["source"], metadata["positive"])
    return send_file(BytesIO(pdf), mimetype="application/pdf", as_attachment=True, download_name="bias_audit_summary.pdf")

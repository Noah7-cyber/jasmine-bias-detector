"""Download, prepare and document offline public sample datasets."""

from __future__ import annotations

import hashlib
from io import BytesIO, StringIO
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
SAMPLES.mkdir(parents=True, exist_ok=True)

COMPAS_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
ADULT_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    retrieved = datetime.now(timezone.utc).isoformat()
    compas_bytes_response = requests.get(COMPAS_URL, timeout=90)
    compas_bytes_response.raise_for_status()
    compas_bytes = compas_bytes_response.content
    compas_raw = pd.read_csv(BytesIO(compas_bytes))
    required = ["days_b_screening_arrest", "is_recid", "c_charge_degree", "score_text"]
    compas = compas_raw.loc[
        compas_raw["days_b_screening_arrest"].between(-30, 30, inclusive="both")
        & compas_raw["is_recid"].ne(-1)
        & compas_raw["c_charge_degree"].ne("O")
        & compas_raw["score_text"].notna()
        & compas_raw["score_text"].ne("N/A")
    ].copy()
    if compas.empty or not set(["sex", "race", "age_cat", "decile_score", "two_year_recid"]).issubset(compas.columns):
        raise ValueError("COMPAS file did not match expected schema or filtering returned no rows.")
    compas_sample = compas[["sex", "race", "age_cat", "decile_score", "two_year_recid"]].rename(columns={"decile_score": "risk_score", "two_year_recid": "recidivism_2yr"})
    compas_sample.to_csv(SAMPLES / "compas.csv", index=False)

    train_response = requests.get(ADULT_BASE + "adult.data", timeout=90)
    train_response.raise_for_status()
    test_response = requests.get(ADULT_BASE + "adult.test", timeout=90)
    test_response.raise_for_status()
    train_bytes, test_bytes = train_response.content, test_response.content
    adult_train = pd.read_csv(StringIO(train_bytes.decode("utf-8")), names=ADULT_COLUMNS, skipinitialspace=True, na_values="?")
    adult_test = pd.read_csv(StringIO(test_bytes.decode("utf-8")), names=ADULT_COLUMNS, skiprows=1, skipinitialspace=True, na_values="?")
    for frame in (adult_train, adult_test):
        for col in frame.select_dtypes(include="object").columns:
            frame[col] = frame[col].str.strip().str.rstrip(".")

    feature_cols = [c for c in ADULT_COLUMNS if c != "income"]
    train_clean = adult_train.dropna(subset=feature_cols + ["income"]).copy()
    test_clean = adult_test.dropna(subset=feature_cols + ["income"]).copy()
    numeric = train_clean[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical = [c for c in feature_cols if c not in numeric]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    baseline = Pipeline([("prep", pre), ("classifier", LogisticRegression(max_iter=1200, random_state=42))])
    y_train = train_clean["income"].eq(">50K").astype(int)
    baseline.fit(train_clean[feature_cols], y_train)
    output = test_clean[["sex", "race", "age"]].copy()
    output["age_group"] = pd.cut(output["age"], bins=[0, 24, 44, 64, float("inf")], labels=["<25", "25–44", "45–64", "65+"]).astype(str)
    output["income_true"] = test_clean["income"].eq(">50K").astype(int)
    output["income_prediction"] = baseline.predict(test_clean[feature_cols]).astype(int)
    output = output.rename(columns={"age_group": "age_band"})
    output.to_csv(SAMPLES / "adult_holdout.csv", index=False)

    manifest = {
        "retrieved_utc": retrieved,
        "adult_source": "https://archive.ics.uci.edu/dataset/2/adult",
        "compas_source": COMPAS_URL,
        "adult_train_sha256": checksum(train_bytes),
        "adult_test_sha256": checksum(test_bytes),
        "compas_sha256": checksum(compas_bytes),
        "compas_raw_shape": list(compas_raw.shape),
        "compas_prepared_shape": list(compas_sample.shape),
        "compas_filter_counts": {"raw": len(compas_raw), "screening_window": int(compas_raw["days_b_screening_arrest"].between(-30,30,inclusive="both").sum()), "not_recidivist_missing": int((compas_raw["days_b_screening_arrest"].between(-30,30,inclusive="both") & compas_raw["is_recid"].ne(-1)).sum()), "final": len(compas_sample)},
        "adult_train_rows_raw": len(adult_train),
        "adult_train_rows_complete_for_baseline": len(train_clean),
        "adult_test_rows_raw": len(adult_test),
        "adult_holdout_rows_complete": len(output),
        "adult_baseline": {"estimator": "untuned LogisticRegression", "random_state": 42, "max_iter": 1200, "features": feature_cols, "train_only_fit": True, "heldout_only_predictions": True},
        "positive_meaning": {"compas": "higher recidivism risk; adverse flag", "adult": "income over $50,000"},
    }
    (SAMPLES / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

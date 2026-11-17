"""Reproducible synthetic fairness scenarios with exact seeded row counts."""

from __future__ import annotations

import numpy as np
import pandas as pd


SCENARIOS = {
    "S1 — Equal rates": "Two large groups with identical selection and error rates.",
    "S2 — Below threshold": "Selection-rate gap of 0.05, below the 0.10 screen.",
    "S3 — Moderate disparity": "Selection-rate gap of 0.15 in large groups.",
    "S4 — Large disparity": "Selection-rate gap of 0.30 in large groups.",
    "S5 — Small group": "Selection-rate gap of 0.30; one group has fewer than 30 people.",
    "S6 — Error-rate disparity": "Equal selection rates with a 0.20 true-positive-rate gap.",
}


def _confusion_rows(group: str, tp: int, fn: int, fp: int, tn: int, rng: np.random.Generator) -> list[dict]:
    records = ([{"group": group, "truth": 1, "prediction": 1}] * tp
               + [{"group": group, "truth": 1, "prediction": 0}] * fn
               + [{"group": group, "truth": 0, "prediction": 1}] * fp
               + [{"group": group, "truth": 0, "prediction": 0}] * tn)
    rng.shuffle(records)
    return records


def generate_scenario(name: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if name.startswith("S1"):
        specs = [("Group A", 800, 200, 400, 600), ("Group B", 800, 200, 400, 600)]
    elif name.startswith("S2"):
        specs = [("Group A", 800, 200, 400, 600), ("Group B", 800, 200, 300, 700)]
    elif name.startswith("S3"):
        specs = [("Group A", 800, 200, 400, 600), ("Group B", 700, 300, 200, 800)]
    elif name.startswith("S4"):
        specs = [("Group A", 800, 200, 600, 400), ("Group B", 500, 500, 300, 700)]
    elif name.startswith("S5"):
        specs = [("Group A", 800, 200, 600, 400), ("Group B", 4, 6, 4, 6)]
    elif name.startswith("S6"):
        specs = [("Group A", 800, 200, 200, 800), ("Group B", 600, 400, 400, 600)]
    else:
        raise ValueError(f"Unknown scenario: {name}")
    records: list[dict] = []
    for group, tp, fn, fp, tn in specs:
        records.extend(_confusion_rows(group, tp, fn, fp, tn, rng))
    return pd.DataFrame.from_records(records)


def scenario_expected_status(name: str) -> str:
    if name.startswith("S1") or name.startswith("S2"):
        return "no_dp_flag"
    if name.startswith("S3") or name.startswith("S4"):
        return "supported_dp_flag"
    if name.startswith("S5"):
        return "insufficient_data"
    if name.startswith("S6"):
        return "equal_opportunity_flag_only"
    raise ValueError(name)

"""Pure metric calculations; independent from UI and Fairlearn references."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class AuditSettings:
    group_column: str
    prediction_column: str
    score_threshold: float = 5.0
    score_mode: bool = False
    true_column: str | None = None
    positive_label: str = "1"
    negative_label: str = "0"
    true_positive_label: str | None = None
    true_negative_label: str | None = None
    dpd_threshold: float = 0.10
    dir_threshold: float = 0.80
    minimum_group_size: int = 30


def _extreme_span(values: dict[Any, float]) -> tuple[float, Any, Any]:
    ordered = sorted(values.items(), key=lambda item: (item[1], str(item[0])))
    return float(ordered[-1][1] - ordered[0][1]), ordered[-1][0], ordered[0][0]


def _to_binary(series: pd.Series, positive: str, negative: str, label: str) -> np.ndarray:
    values = series.astype("string").str.strip()
    positives, negatives = str(positive).strip(), str(negative).strip()
    invalid = values.isna() | ~values.isin([positives, negatives])
    if invalid.any():
        raise ValueError(f"{label} has {int(invalid.sum())} missing or invalid value(s); map exactly two binary labels.")
    return values.eq(positives).to_numpy(dtype=np.int8)


def calculate_audit(
    frame: pd.DataFrame,
    settings: AuditSettings,
    *,
    positive_is_adverse: bool = False,
) -> dict[str, Any]:
    """Calculate group metrics and the proposal's thresholds for one attribute."""
    missing = [c for c in (settings.group_column, settings.prediction_column) if c not in frame.columns]
    if settings.true_column and settings.true_column not in frame.columns:
        missing.append(settings.true_column)
    if missing:
        raise ValueError(f"Selected column(s) missing: {', '.join(missing)}")
    if not 0 <= settings.dpd_threshold <= 1 or not 0 <= settings.dir_threshold <= 1:
        raise ValueError("DPD and DIR thresholds must be between 0 and 1.")
    if settings.minimum_group_size < 1:
        raise ValueError("Minimum group size must be at least 1.")

    selected = [settings.group_column, settings.prediction_column]
    if settings.true_column:
        selected.append(settings.true_column)
    data = frame[selected].copy()
    if data[settings.group_column].isna().any():
        raise ValueError(f"Selected demographic column has {int(data[settings.group_column].isna().sum())} missing value(s).")
    if data.empty:
        raise ValueError("The selected data contains no rows.")

    if settings.score_mode:
        scores = pd.to_numeric(data[settings.prediction_column], errors="coerce")
        if scores.isna().any() or not np.isfinite(scores.to_numpy(dtype=float)).all():
            raise ValueError("Score column contains missing, non-numeric, or non-finite values.")
        predicted = scores.ge(settings.score_threshold).to_numpy(dtype=np.int8)
    else:
        predicted = _to_binary(data[settings.prediction_column], settings.positive_label, settings.negative_label, "Prediction")

    groups = data[settings.group_column].astype("string").to_numpy()
    names = sorted(pd.unique(groups).tolist(), key=str)
    if len(names) < 2:
        raise ValueError("Choose a demographic column with at least two nonempty groups.")

    y_true = None
    if settings.true_column:
        y_true = _to_binary(
            data[settings.true_column],
            settings.true_positive_label or settings.positive_label,
            settings.true_negative_label or settings.negative_label,
            "True-outcome column",
        )

    rows = []
    rates: dict[str, float] = {}
    exact_rates: dict[str, Fraction] = {}
    tprs: dict[str, float] = {}
    fprs: dict[str, float] = {}
    undefined: list[str] = []
    for group in names:
        mask = groups == group
        yp = predicted[mask]
        n = int(mask.sum())
        sr = float(yp.mean())
        rates[str(group)] = sr
        exact_rates[str(group)] = Fraction(int(yp.sum()), n)
        row: dict[str, Any] = {"group": str(group), "n": n, "selection_rate": sr}
        if y_true is not None:
            yt = y_true[mask]
            tp = int(((yt == 1) & (yp == 1)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            tn = int(((yt == 0) & (yp == 0)).sum())
            pos_den, neg_den = tp + fn, fp + tn
            row.update(tp=tp, fn=fn, fp=fp, tn=tn)
            if pos_den:
                row.update(tpr=tp / pos_den, fnr=fn / pos_den)
                tprs[str(group)] = tp / pos_den
            else:
                row.update(tpr=None, fnr=None)
                undefined.append(f"TPR/FNR undefined for {group}: no true positive-class rows.")
            if neg_den:
                row["fpr"] = fp / neg_den
                fprs[str(group)] = fp / neg_den
            else:
                row["fpr"] = None
                undefined.append(f"FPR undefined for {group}: no true negative-class rows.")
        rows.append(row)

    selection_gap, high_group, low_group = _extreme_span(rates)
    max_rate, min_rate = max(rates.values()), min(rates.values())
    dir_value = min_rate / max_rate if max_rate > 0 else None
    rate_order = sorted(exact_rates.items(), key=lambda item: (item[1], str(item[0])))
    exact_min, exact_max = rate_order[0][1], rate_order[-1][1]
    dpd_flag = (exact_max - exact_min) >= Fraction(str(settings.dpd_threshold))
    dir_flag = exact_max > 0 and (exact_min / exact_max) < Fraction(str(settings.dir_threshold))

    n_a = next(r["n"] for r in rows if r["group"] == high_group)
    n_b = next(r["n"] for r in rows if r["group"] == low_group)
    rate_a, rate_b = rates[high_group], rates[low_group]
    pooled = (n_a * rate_a + n_b * rate_b) / (n_a + n_b)
    se = float(np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))) if 0 < pooled < 1 else 0.0
    z_score = (rate_a - rate_b) / se if se > 0 else None
    p_value = float(2 * norm.sf(abs(z_score))) if z_score is not None else None
    sparse_expected = min(pooled * n_a, (1-pooled)*n_a, pooled*n_b, (1-pooled)*n_b) < 5
    reliable = all(r["n"] >= settings.minimum_group_size for r in rows)

    def message(flag: bool | None) -> str:
        if not reliable:
            return "Insufficient data: the difference cannot be reliably assessed."
        if flag is None:
            return "Statistical support is unavailable for this comparison."
        if not flag:
            return "No disparity above this screening threshold was found. This is not proof of fairness."
        if p_value is None:
            return "Potential disparity above the threshold; statistical support is unavailable."
        if sparse_expected:
            return "Potential disparity above the threshold; expected counts are sparse, so the normal-approximation test is unreliable. Further investigation is recommended."
        if p_value >= .05:
            return "Potential disparity above the threshold, but it is not statistically supported and may be due to chance."
        return "Potential disparity above the threshold and statistically supported; further investigation is recommended. This is not proof of discrimination."

    tpr_gap = _extreme_span(tprs) if len(tprs) == len(rows) else None
    fpr_gap = _extreme_span(fprs) if len(fprs) == len(rows) else None
    equal_odds = max(tpr_gap[0], fpr_gap[0]) if tpr_gap and fpr_gap else None
    metrics: dict[str, Any] = {
        "selection_rate": rates,
        "demographic_parity_difference": selection_gap,
        "disparate_impact_ratio": dir_value,
        "demographic_parity_flag": dpd_flag,
        "disparate_impact_flag": dir_flag,
        "selection_rate_comparison": {"high_group": high_group, "low_group": low_group, "n_high": n_a, "n_low": n_b, "z": z_score, "p_value_two_sided": p_value, "pooled_rate": pooled, "standard_error": se, "sparse_expected_cell": sparse_expected},
        "reliable": reliable,
        "decision_dpd": message(dpd_flag),
        "decision_dir": message(dir_flag),
        "true_positive_rate": tprs if y_true is not None else None,
        "false_positive_rate": fprs if y_true is not None else None,
        "equal_opportunity_difference": tpr_gap[0] if tpr_gap else None,
        "equalised_odds_difference": equal_odds,
        "equal_opportunity_groups": [tpr_gap[1], tpr_gap[2]] if tpr_gap else None,
        "false_positive_rate_gap": fpr_gap[0] if fpr_gap else None,
        "undefined": undefined,
        "positive_is_adverse": bool(positive_is_adverse),
        "rows_analysed": len(data),
        "settings": asdict(settings),
    }
    # Keep the original all-group metrics for research reproducibility.
    eligible = [r for r in rows if r["n"] >= settings.minimum_group_size]
    for row in rows:
        row["sample_status"] = "Eligible" if row["n"] >= settings.minimum_group_size else "Insufficient group data"
    excluded = [r["group"] for r in rows if r["n"] < settings.minimum_group_size]
    screening = {"excluded_groups": excluded, "eligible_groups": [r["group"] for r in eligible],
                 "dpd": None, "dir": None, "high_group": None, "low_group": None,
                 "pairs": [], "message": "Insufficient data: fewer than two eligible groups."}
    if len(eligible) >= 2:
        ordered = sorted(eligible, key=lambda r: (exact_rates[r["group"]], r["group"]))
        lo, hi = ordered[0], ordered[-1]
        low, high = exact_rates[lo["group"]], exact_rates[hi["group"]]
        screening.update(dpd=float(high-low), dir=float(low/high) if high else None,
                         high_group=hi["group"], low_group=lo["group"])
        # Bonferroni across every eligible pair, including pairs with unavailable tests.
        comparisons = len(eligible)*(len(eligible)-1)//2
        for i, a in enumerate(ordered):
            for b in ordered[i+1:]:
                pa, pb = exact_rates[a["group"]], exact_rates[b["group"]]
                x, y = int(pa*a["n"]), int(pb*b["n"])
                pooled_pair = (x+y)/(a["n"]+b["n"])
                sparse = min(pooled_pair*a["n"], (1-pooled_pair)*a["n"],
                             pooled_pair*b["n"], (1-pooled_pair)*b["n"]) < 5
                test = two_proportion_test(y,b["n"],x,a["n"])
                raw = test["p_value"] if not sparse else None
                adjusted = min(1.0, raw*comparisons) if raw is not None else None
                flag = (pb-pa >= Fraction(str(settings.dpd_threshold)) or
                        (pb > 0 and pa/pb < Fraction(str(settings.dir_threshold))))
                status = ("Below screening thresholds" if not flag else
                          "Potential disparity; statistical support unavailable" if adjusted is None else
                          "Further investigation recommended" if adjusted < .05 else
                          "Potential disparity; insufficient statistical evidence")
                screening["pairs"].append({"high_group": b["group"], "low_group": a["group"],
                    "dpd": float(pb-pa), "dir": float(pa/pb) if pb else None,
                    "z": test["z"] if not sparse else None, "p": raw, "adjusted_p": adjusted,
                    "status": status})
        screening["message"] = ("Eligible-group screening uses Bonferroni-adjusted two-sided tests; "
                                "excluded groups remain visible but cannot be reliably assessed. "
                                "Findings are not proof of discrimination.")
    metrics["eligible_screening"] = screening
    return {"groups": rows, "metrics": metrics}


def two_proportion_test(success_a: int, n_a: int, success_b: int, n_b: int) -> dict[str, float | None]:
    """Reusable exact-count z-test for independently verified fixtures."""
    if not (0 <= success_a <= n_a and 0 <= success_b <= n_b and n_a > 0 and n_b > 0):
        raise ValueError("Each group must have positive size and valid success counts.")
    p_a, p_b = success_a / n_a, success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    se = float(np.sqrt(pooled * (1-pooled) * (1/n_a + 1/n_b))) if 0 < pooled < 1 else 0.0
    z = (p_a - p_b) / se if se > 0 else None
    return {"z": z, "p_value": float(2*norm.sf(abs(z))) if z is not None else None, "pooled_rate": pooled, "standard_error": se}

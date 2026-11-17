---
title: Bias Detector — Formula and Evaluation Contract
date: 2026-11-17
status: proposed
source_sections: 3.6–3.8
---

# Formula and evaluation contract

This document transcribes the proposal's definitions into implementation-ready notation. Complements [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Equation numbers refer to Chapter 3 of the original DOCX. Choices for otherwise unspecified edge cases are identified explicitly.

## Inputs and calculations

For each row i, let a_i be its selected demographic group, y_i its optional true binary outcome, s_i its prediction/score, and yhat_i its normalised predicted label. Unused feature columns are not inputs to the metric engine (3.1). Label 1 denotes the **positive class as configured**, which need not be favourable.

| Equation | Exact calculation |
| --- | --- |
| 3.2 | Score mode: yhat_i = 1 if s_i >= tau, else 0. Binary-label mode skips thresholding. |
| 3.3 | n_g = sum_i I(a_i = g). |
| 3.4 | SR_g = sum_i I(a_i = g) * yhat_i / n_g. |
| 3.5 | DPD = max_g SR_g - min_g SR_g. |
| 3.6 | DIR = min_g SR_g / max_g SR_g. Unavailable if max SR = 0. |
| 3.7 | TPR_g = TP_g / (TP_g + FN_g). |
| 3.8 | FPR_g = FP_g / (FP_g + TN_g). |
| 3.9 | FNR_g = FN_g / (TP_g + FN_g) = 1 - TPR_g when defined. |
| 3.10 | Equal opportunity difference = delta_TPR = max_g TPR_g - min_g TPR_g. |
| 3.11 | delta_FPR = max_g FPR_g - min_g FPR_g. |
| 3.12 | Equalised odds difference = max(delta_TPR, delta_FPR). |
| 3.13 | For the highest/lowest SR groups a,b: pooled_p = (n_a SR_a + n_b SR_b) / (n_a + n_b). |
| 3.14 | SE = sqrt(pooled_p * (1-pooled_p) * (1/n_a + 1/n_b)); z = (SR_a - SR_b) / SE. |
| §3.6.9 | Two-sided p = 2 * scipy.stats.norm.sf(abs(z)). |
| 3.15 | Flag_DPD = I(DPD >= delta), default delta = 0.10. |
| 3.16 | Flag_DIR = I(DIR < theta), default theta = 0.80. |
| 3.17 | Reliable_g = I(n_g >= n_min), default n_min = 30. |

TP/FN/FP/TN are counted against the configured positive class on exactly the same rows used for selection rates. All supplied groups enter max/min calculations. No reference-group selector changes these definitions. For multiple groups, TPR and FPR extrema may be different pairs; compute their spans independently.

Use full precision through calculations and screening. Validate configurable delta/theta within [0,1], and n_min as a positive integer. For exact screening boundaries, compare count-derived rational rates with the decimal threshold using rational cross-multiplication, so that binary floating-point subtraction cannot turn an exact 0.10 gap into a value just below the threshold. Do not use a broad numerical tolerance to redefine the screening inequalities. Alpha is fixed at 0.05 in the proposed first version; the proposal explicitly makes screening thresholds adjustable, not necessarily the significance level.

The DIR definition corresponds to Fairlearn's between-group demographic parity ratio. Equalised odds corresponds to the worst of its TPR/FPR spans. Tests must pass explicit between-group/worst-case options supported by the pinned reference version rather than assume defaults. [Fairlearn ratio](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.demographic_parity_ratio.html), [equalised odds](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.equalized_odds_difference.html). SciPy's normal survival function provides the p-value calculation. [SciPy normal distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html).

## Decision messages (Table 3.7)

Keep descriptive metrics, raw threshold flags and interpretive status separate. For each DPD/DIR comparison, apply this priority:

1. Any group included in that comparison is below n_min: **Insufficient data: the difference cannot be reliably assessed.** Retain descriptive values and flag booleans; do not imply no disparity.
2. No threshold flag for the metric: **No disparity above the screening thresholds was found for this metric.** This is not a claim of overall fairness.
3. Flag with p >= 0.05: **Potential disparity above the threshold, but it is not statistically supported and may be due to chance.**
4. Flag with p < 0.05: **Potential disparity above the threshold and statistically supported; further investigation is recommended. This is not proof of discrimination.**

For a summary covering all groups, conservatively warn if any displayed group is undersized. Record the exact extrema pair separately so the tested comparison remains clear. Do not attach selection-rate significance claims to error-rate gaps.

## Defined edge-case policy

These details complete cases the proposal does not fully specify:

- No selected truth: selection metrics remain available; confusion/error metrics are marked “requires true outcomes”.
- TP+FN=0: TPR/FNR unavailable for that group; FP+TN=0: FPR unavailable. Mark dependent all-group spans unavailable if any required group rate is undefined; do not silently reduce the compared group set.
- All predictions zero: DPD=0, DIR unavailable with reason. All predictions one: DPD=0, DIR=1. In both cases the pooled SE is zero; show z/p unavailable, never NaN as a number or a fabricated p-value. An unusual user threshold raising a flag here yields an unavailable significance status.
- Equal nondegenerate rates: z=0 and p=1. Choose two distinct groups in stable group-label order if extrema are tied; record tie policy.
- Fewer than two nonempty groups: comparison blocked with a clear message. Single-group descriptive counts may be previewed.
- Sparse expected selected/unselected counts under the pooled null: warn when any expected cell is below 5; retain the specified z-test numerically when defined but suppress conclusive support language. This is an added interpretation safeguard, not a replacement exact test.
- Undefined p for a flagged comparison: “Potential disparity; statistical support unavailable”, with the cause. Do not force it into a supported/unsupported Table 3.7 branch.
- Malformed, missing or non-finite selected inputs: validation blocks calculation with counts. No silent imputation, row dropping, invented truth, or automatically inferred positive outcome.
- DIR below 0.80 is a configurable screen; higher COMPAS positive rates describe adverse risk flags, so the UI must explain direction rather than apply employment-benefit language.

## Mandatory worked example (Table 3.8)

| Quantity | A | B |
| --- | ---: | ---: |
| n | 200 | 150 |
| TP | 80 | 30 |
| FN | 20 | 20 |
| FP | 40 | 30 |
| TN | 60 | 70 |
| SR | 0.60 | 0.40 |
| TPR | 0.80 | 0.60 |
| FPR | 0.40 | 0.30 |
| FNR | 0.20 | 0.40 |

Expected DPD=0.20; DIR=2/3; opportunity difference=0.20; FPR span=0.10; equalised odds difference=0.20. The pooled rate is 180/350; SE=sqrt((180/350)*(170/350)*(1/200+1/150)). The proposal rounds z to 3.70 and p to 0.0002. Compare against independently calculated full-precision values, not those rounded illustrations, at tolerance 1e-6. Both screens flag, groups meet the size rule, and the difference is statistically supported.

## Independent verification

Keep expected count fixtures and mathematical reference outputs separate from application code. A reference must not call the implementation being tested.

- Verify SR and group error rates with manual counts and Fairlearn MetricFrame; verify confusion counts independently with scikit-learn using explicit labels [0,1].
- Verify DPD with demographic_parity_difference; DIR with demographic_parity_ratio; opportunity with the between-group TPR span; odds with equalized_odds_difference using the worst-case aggregation.
- Test undefined denominators as semantic cases; reference-library zero-division defaults must not override the declared policy.
- Compare multi-group, reversed-label, prediction-only, tied, all-zero/all-one, exact-threshold and minimum-size fixtures.
- Verify threshold boundary values using rational count fixtures and explicit numerical comparison policy; display rounding must never change a flag.
- Independently check z/p for a few hand-calculable proportion tables, including the worked example. Test score exactly equal to tau.

## Evaluation equations and targets

Let K count comparisons with defined numerical values on both sides; record excluded/unavailable cases and test their statuses separately. Never drop a failed defined comparison from K. Use absolute tolerance epsilon=0.000001.

| Proposal equation | Measure | Target |
| --- | --- | --- |
| 3.18 | Metric accuracy = 100 * sum_k I(abs(app_k-ref_k) <= epsilon) / K | 100% |
| 3.19 | MAE = sum_k abs(app_k-ref_k) / K | <= 0.000001 |
| 3.20 | Detection accuracy = 100 * (TP+TN)/(TP+TN+FP+FN) | >=95% on the declared binary detection population |
| 3.21 | Detection precision = 100 * TP/(TP+FP) | Report; proposal gives no separate target |
| 3.22 | Detection recall = 100 * TP/(TP+FN) | Report; proposal gives no separate target |
| 3.23 | Functional pass rate = 100 * passed/executed | 100% essential tests; >=95% all tests |
| 3.24 | Mean response time = sum_r time_r / R | <=5 seconds |
| 3.25 | Optional SUS = 2.5 * [sum_odd(r_j-1) + sum_even(5-r_j)] | Optional; no required target |

For detection formulas, TP/FP/FN/TN refer to **scenario classifications**, not individual prediction confusion counts. If a precision/recall denominator is zero, report unavailable with its denominator, not an invented perfect score.

Generate all six scenarios for 100 seeds each. Report 600-case expected-status accuracy, the 400-case S1–S4 DP binary confusion matrix and scores, 100 S5 abstention checks, and 100 S6 equal-opportunity checks separately. This resolves the proposal's mixed expected outcomes transparently; the researcher must settle the denominator interpretation before final dissertation claims. Include any failures rather than tuning the generator after seeing app outputs. Freeze expected counts and outcomes independently first.

Benchmark each bundled dataset for 10 repeated measured runs after one warmup, recording every duration, machine and package versions, cohort size and configuration. Also report a cold local run. Measure validation, normalisation, metrics, statistics and chart preparation; measure request-to-render time separately in browser walkthroughs. PDF generation gets a separate timing. No download or Adult fitting occurs during these runs. Benchmark the real held-out Adult sample and a labelled synthetic 48,842-row stress fixture; disclose why the latter differs from the proposal's full-Adult timing description.

Functional tests trace to FR1–FR11. Privacy tests cover absence of upload writes/logging/global caches, Clear analysis, changing datasets and isolated simultaneous sessions. PDF tests compare reported metrics/settings with the same result object; inspect pagination and long category labels manually. UI checks cover no-truth mode, invalid mappings, threshold changes, warnings, and accessibility. Test Chrome and Edge and record versions. Complete a heuristic inspection with no unresolved major issue. Human usability testing/SUS remains optional and, under §3.8.7/§3.9, requires institutional approval and consent before participants are recruited.

Write measured outputs to `evaluation/` using only public/synthetic fixtures. Put the accompanying narrative, formula explanations, dataset datasheets and limitations under `docs/`. No evaluation artifact may contain a user's uploaded rows or identify COMPAS individuals.

---
title: Bias Detector — Proposal-Aligned Implementation Plan
author: Planning handoff for Luna
date: 2026-11-17
status: local-implementation-complete
source: C:/Users/HP USER/Downloads/MIT_PROJECT_edit_revised.docx
---

# Implementation plan for Luna

## Purpose and boundaries

Build **Development of a Web-Based Application for Detecting Bias in AI Predictions**, the separate Master's project by Emakhu Jasmine Isaac, MIVA Open University.

Project root: `C:\Users\HP USER\Downloads\jasmine-bias-detector`.

The authority is the supplied proposal, particularly Chapters 1–3, supplemented by the user's explicit instructions. This plan supersedes `jasmine/docs/LUNA_IMPLEMENTATION_PLAN.md`. Do not implement Jasmine Credit Risk Lab. Do not edit, import from, retrain, or reorganise the completed German Credit coursework in `jasmine`. Create a separate environment and dependencies in this project.

The application analyses **existing binary predictions or thresholded scores by one demographic attribute at a time**. Its contribution is a verifiable metric engine, reliability assessment, accessible visual explanation, and reporting. It does not diagnose the cause of disparities or produce legal conclusions.

Use Python, Streamlit, pandas, NumPy, SciPy, Plotly, ReportLab, and pytest. Fairlearn is an independent metric oracle; scikit-learn supplies reference confusion matrices and the untuned Adult preparation classifier. Do not implement general ML training, hyperparameter tuning, mitigation, causal analysis, individual fairness, intersectional analysis, accounts, a database, monitoring, non-tabular inputs, or multiclass predictions.

**The local application implementation is complete.** The evaluation results and known research ambiguities are recorded in `docs/IMPLEMENTATION_REPORT.md`. A full request-to-render timing study, structured heuristic inspection, and public demonstration deployment remain follow-up research tasks.

## Proposal inspection and traceability

Chapters 1 and 2 establish scope and interpretation limits; Chapter 3 defines the executable requirements. Word equations were inspected through their Office Math XML, including fraction structure and absolute-value delimiters, rather than relying on plain paragraph text alone.

Source SHA-256: `f3c1132a71b2645442a35a302ee6078ee6efe4b91e55f04da031f7205cbffb0f`.

| Source | Implementation obligation | Verification |
| --- | --- | --- |
| §1.7; §2.9 | Audit supplied predictions, one attribute at a time; no model correction | Scope inspection |
| FR1 | Bundled COMPAS, Adult, synthetic samples; CSV upload | Offline samples and upload walkthrough |
| FR2, FR11 | Report absent columns, missing/invalid values, nonbinary labels, empty input, and undersized groups without crashes | Validation and UI tests |
| FR3 | Choose attribute, binary prediction or score, score threshold, optional truth | Mapping and threshold tests |
| FR4 | Selection rates, DPD, DIR | Manual and independent Fairlearn comparisons |
| FR5 | TPR, FPR, FNR, equal opportunity and equalised odds gaps when truth exists | Confusion counts and reference comparisons |
| FR6 | Highest/lowest selection-rate comparison, two-proportion z-test, minimum group size | Analytical fixtures and boundary tests |
| FR7, FR8 | Tables, interactive charts, explanations and limitations | Walkthrough and content inspection |
| FR9 | Downloadable summary including groups, metrics, flags, warnings | Extracted PDF content and visual inspection |
| FR10 | Adjustable screening thresholds with visible defaults | Exact threshold boundary tests |
| NFR1 | Absolute numerical tolerance 0.000001 | 100% of defined reference comparisons within tolerance |
| NFR2 | Invalid or unusual inputs do not crash application | Validation and UI regression cases |
| NFR3 | Results displayed within five seconds at about 50,000 rows | Local analysis/chart-construction timings are measured; full request-to-render timing remains pending |
| NFR4, NFR5 | Accessible navigation; every result defined and qualified | Definitions and limitations are present; structured heuristic inspection remains pending |
| NFR6 | Uploads remain in temporary session memory | Code inspection confirms session-only upload state; a separate session-isolation test remains pending |
| NFR7 | Standard browser portability | Chrome and Edge walkthroughs, recording versions |
| §3.5 | Documented original sources, checksums, preparation, offline samples | Datasheets and preparation manifest |
| §3.7.4 | Demonstration deployment and public reproducibility repository | Deployment checklist; do not publish during planning |
| §3.8 | Calculation, detection, functionality, speed and usability evidence | Reproducible evaluation report with denominators |

## Architecture

Use the proposal's three layers and six logic modules. Streamlit is the presentation layer; no separate API or JavaScript frontend is required.

```text
Browser: Dataset → Analysis → Results → Report
                     |
Application: loader → validator/normaliser → metric engine
                                              |
                              reliability and flagging
                                              |
                              charts/explanations + PDF
                     |
Data: bundled public samples + seeded synthetic fixtures
      session-only uploaded table + independent reference fixtures
```

| Implemented file/module | Responsibility |
| --- | --- |
| `app.py`, `server.py` | Vercel WSGI entrypoint and stateless Python web endpoints |
| `streamlit_app.py` | Original four-stage Streamlit research workflow |
| `templates/`, `static/` | Hosted research workbench UI with local Plotly charts |
| `bias_detector/data_loader.py` | Read uploaded CSV bytes in memory and detect duplicate raw headers |
| `bias_detector/validation.py` | Inspect missing values and validate binary mappings |
| `bias_detector/metrics.py` | Pure group counts, formulas, z-test, reliability checks, thresholds and decision messages |
| `bias_detector/synthetic.py` | Six fixed-seed known-answer synthetic scenarios |
| `bias_detector/reporting.py` | ReportLab PDF generated into in-memory bytes |
| `scripts/prepare_public_data.py` | Reproducible public-data preparation outside the running app |
| `scripts/generate_synthetic.py` | Write sample CSVs and scenario manifest |
| `scripts/evaluate.py` | Fairlearn comparisons, synthetic detection results and timing evidence |
| `tests/` | Metric, reference, Streamlit workflow and PDF tests |
| `data/samples/` | Sanitised offline public samples and reproducible synthetic examples |
| `evaluation/` | Aggregate research evaluation outputs from public/synthetic data only |
| `docs/` | Run guide, design, metric definitions, implementation plan and results |

No upload folder, database, account service, or training page. No dependency on another prototype. Pin the package versions actually tested in the new environment; do not assume development-documentation versions are release versions.

### Data and result contracts

An analysis configuration records source kind, selected attribute, prediction mode, prediction/score column, optional truth column, positive-label mapping, score threshold, DPD threshold, DIR threshold and minimum group size. A result records input/analysed/excluded row counts, group counts, confusion counts when available, full-precision metrics, unavailable-value reasons, the exact extrema pair, z/p, raw threshold flags, reliability status, messages, settings and preparation provenance.

The UI, charts and PDF consume the **same result object**, never recalculate metrics independently. Editing the dataset or any configuration invalidates old results and downloads. Store uploads and derived results only in that user's Streamlit session; provide Clear analysis. Do not cache uploads or their results globally, persist them to disk, or include rows in logs or exceptions. Generate PDF bytes in memory. Cache only immutable, bundled public samples. Streamlit's default caches can be shared across users, so session ownership must be explicit. [Streamlit caching](https://docs.streamlit.io/develop/concepts/architecture/caching), [Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state).

## Data preparation

### COMPAS (§3.5.2–3.5.4)

Use the proposal's [original repository](https://github.com/propublica/compas-analysis), the `compas-scores-two-years.csv` file, and original analysis code. Record source revision, retrieval date, checksum and terms of use. Expected raw shape is 7,214 rows and 53 columns. The documented analysis filters should yield 6,172 rows; investigate discrepancies rather than silently forcing that count.

Reproduce the original filters: inclusive `days_b_screening_arrest` range −30 to 30, `is_recid != -1`, `c_charge_degree != 'O'`, and `score_text != 'N/A'` (normalise parser-missing score labels consistently). Maintain a filter audit with row counts. Default prediction is `decile_score >= 5`; truth is `two_year_recid`. Available attributes include race, sex and `age_cat`, analysed separately. Do not narrow the full cohort to two races implicitly.

Persist a sanitised snapshot containing only necessary analysis columns. Remove names, first/last names, birth dates, case numbers and other direct identifiers before display or storage. Default positive means **higher recidivism risk**, an adverse flag; charts must never call it a favourable outcome.

### Adult (§1.7.2; §3.5)

Use the [UCI Adult source](https://archive.ics.uci.edu/dataset/2/adult), recording retrieval metadata, attribution, checksum, raw shape and missingness. Normalise whitespace, `?` missing markers and terminal punctuation in test income labels. The original data have 48,842 records and 14 features; processed/held-out row counts will differ and must be shown accurately.

Proposed reproducible preparation choice: use the published train/test partition, remove incomplete feature rows only for this baseline, fit a single untuned logistic regression on the training partition, and predict the held-out partition. Fit encoding/scaling solely on training features; never include income as an input. Record solver, fixed seed 42, sufficient iteration limit, convergence status, feature list and retained/excluded counts. A larger iteration limit enables convergence and is not a hyperparameter search. Do not tune for fairness or predictive performance.

Bundle the held-out prediction table with demographic attributes and true income label. Positive is income above US$50,000. Use sex/race and a documented preparation-only age-band choice: `<25`, `25–44`, `45–64`, `65+`. These bands and seed are implementation choices, not proposal formulas. Do not train when the app starts or the user analyses a file.

### Synthetic cases (§3.8, Table 3.12)

Construct known confusion counts first, then seed-shuffle rows; this preserves exact selection/error rates while supplying 100 reproducible realisations per scenario (seeds 0–99). Publish the generator and count manifests. Clearly describe these as controlled count fixtures, rather than 600 independent statistical experiments. A separate stochastic robustness exercise is optional and must not replace the specified evaluation.

| Scenario | Concrete preparation choice | Expected result |
| --- | --- | --- |
| S1 | Two groups of 2,000, identical selection/error rates, SR 0.60 each | No DPD/DIR flag |
| S2 | Two groups of 2,000, SR 0.60 and 0.55 | No flag: DPD 0.05; DIR above 0.80 |
| S3 | Two groups of 2,000, SR 0.60 and 0.45 | Flag, statistically supported |
| S4 | Two groups of 2,000, SR 0.70 and 0.40 | Flag, statistically supported |
| S5 | Groups of 2,000 and 20, SR 0.70 and 0.40 | Insufficient data, while showing descriptive gap |
| S6 | Two groups of 2,000 with true base rate 0.50; TPR 0.80/0.60 and FPR 0.20/0.40 | Equal SR 0.50; no DP flag; equal opportunity gap 0.20 reported |

Also include separate small fixtures for zero denominators, one group, three or more groups, tied extrema and exact screening boundaries.

### Uploaded CSVs

Parse uploaded bytes in memory. Validate empty/malformed files, duplicate column names before parser renaming, encoding issues and usable mappings. Validate only the columns selected for analysis; unrelated missing features do not invalidate the file. Require nonmissing groups and finite predictions/scores, with explicit two-label mapping to 0/1 when needed. If truth is selected, require valid nonmissing binary truth on the same cohort; users can deselect truth to run prediction-only analysis. Do not silently drop or impute analysis rows. Report counts of invalid/missing values and block analysis until corrected. An absent true class in a group makes its dependent rates unavailable, rather than invalidating descriptive selection-rate analysis.

Require at least two nonempty groups for comparisons. Numeric age must use explicit documented bands, not an automatic group per individual age. Undersized groups remain visible with warnings. Never remove them silently to obtain more favourable results. A proposed 25 MiB upload cap is an operational limit, not a proposal requirement; show it before upload.

## Exact calculation specification

The canonical formulas and decision rules are in [METRICS_AND_EVALUATION.md](METRICS_AND_EVALUATION.md). Implement them in the application's own pure functions; use Fairlearn only in reference tests. In particular:

- Score conversion uses `>=`, not `>`; default COMPAS score threshold is 5.
- DPD is maximum minus minimum selection rate across all selected groups; DIR is minimum divided by maximum.
- DPD flag is `DPD >= 0.10`; DIR flag is `DIR < 0.80`. Both defaults are adjustable and prominently recorded.
- Default minimum group size is 30; equality at 30 meets this rule. Two-sided significance uses `p < 0.05`.
- Error-rate significance is not established by a selection-rate z-test. Report equal opportunity and equalised odds gaps descriptively; do not invent extra significance tests or screening thresholds for them.
- Undefined values remain unavailable with reasons. Never coerce them to zero or a passing status.

## Application workflow

1. **Dataset:** introduction, public sample selector or CSV upload, provenance, privacy note, data-quality counts and sanitised preview. Explain whether positive is a benefit or an adverse flag.
2. **Analysis:** choose one attribute, binary prediction/score, optional truth and mappings. Score mode exposes its threshold. Screening controls show defaults 0.10, 0.80 and 30. Analyse only after valid settings are submitted.
3. **Results:** group counts and selection-rate chart; DPD/DIR with thresholds; exact high/low groups, p-value and reliability message. When truth exists, show confusion counts, TPR/FPR/FNR charts and opportunity/odds gaps. Include numeric tables so colour is not the only indication. Explain unavailable metrics. Avoid a single “fair/unfair” verdict.
4. **Report:** download a ReportLab PDF containing source description, cohort/preparation counts, selected columns/mappings, positive meaning, thresholds, group tables, metrics, flags, z/p and pair, warnings, definitions and limitations. No individual rows or direct identifiers. Use ReportLab-native summary charts if charts are included; interactive Plotly remains in-browser. Include run/package versions for reproducibility, using full-precision values for decisions and rounded values for display.

Use visible processing/validation feedback, clear error recovery, readable chart labels and keyboard-accessible standard widgets. A changed configuration must not leave a previous PDF downloadable as if current.

## Ordered implementation increments

| Increment | Concrete deliverable | Required exit evidence |
| --- | --- | --- |
| 0. Isolate project | New environment, tested dependency versions, package/test folders, source manifest | Coursework paths untouched; no imports from `jasmine` |
| 1. Data module | Public-data preparation, offline samples, seeded fixtures, session loader and validation | Provenance/counts, exact filter tests, invalid-file and privacy checks |
| 2. Metric engine | Counts, formulas, statistics, flags and typed results | Worked example, manual/Fairlearn parity, boundary and undefined-value tests |
| 3. Interface | Dataset/Analysis/Results/Report navigation and state handling | FR1–FR6/FR11 walkthroughs; stale results and two-session isolation checked |
| 4. Explanation | Plotly comparisons, definitions and decision messages | FR7/FR8/NFR5 inspection, context labels correct for both datasets |
| 5. Report/hardening | In-memory PDF, configurable thresholds, response-time checks, deployment-ready configuration | FR9/FR10, PDF text/visual checks, privacy tests, browser walkthroughs |
| 6. Research evaluation | 600-case scenario report, calculation/oracle report, functional report, timings, usability findings | Targets and exclusions reported honestly; remaining issues documented |

Run the full new-project regression suite after each modified increment. Fix discrepancies before advancing; record findings and fixes in `docs/IMPLEMENTATION_REPORT.md`. Never fabricate successful evaluation results in advance. Prepare public repository/deployment instructions and checklists; actual publishing is a later action, outside this planning deliverable.

## Interpretations requiring visibility before implementation

1. **Positive versus favourable:** the proposal sometimes equates these, but COMPAS positive is adverse. Preserve its specified high-risk coding and explicitly label the meaning; do not invert COMPAS silently.
2. **Raw snapshots versus identifier removal:** §3.5.3 asks for original snapshots, while §3.9 prohibits storing direct identifiers. Proposed resolution: hash raw COMPAS bytes transiently, store only a sanitised snapshot with a separate checksum, and document source revision and removed fields. Do not commit identifiable raw COMPAS data.
3. **600 scenarios versus binary accuracy:** S5 is an abstention, and S6 reports a different metric rather than a DP flag. Report exact expected-status accuracy for all 600 cases, DP binary accuracy/precision/recall for S1–S4 (400), S5 abstention correctness (100), and S6 opportunity-gap correctness (100). This is a transparent evaluation interpretation, not an assertion that the proposal explicitly specifies those denominators. Resolve with the researcher before claiming that a single binary score covers all 600.
4. **Adult size versus held-out analysis:** the proposal quotes the full 48,842 rows for timing while requiring held-out predictions. Benchmark the actual Adult held-out table, plus a clearly labelled 48,842-row synthetic performance fixture. Do not describe duplicated/test-generated rows as original Adult individuals or present training predictions as held-out data.
5. **Extrema selection and normal approximation:** the required test selects maximum/minimum rates from the data. For many groups, describe p-values as exploratory and unadjusted, with no multiple-testing guarantee. A minimum total n=30 does not guarantee adequate positive/negative counts. Warn when the normal approximation has sparse expected cells; suppress a “statistically supported” interpretation in that case. These safeguards qualify interpretation without replacing the specified equations.

The first two are scope/privacy clarifications. The third and fourth affect evaluation claims and must remain explicitly documented if adopted; do not silently rewrite the proposal to match the implementation.

## Delivery notes

> The implemented application and verification record live in this separate project. The coursework directory was left untouched. For setup and local launch, use `docs/APP_GUIDE.md`; measured outcomes and remaining research tasks are in `docs/IMPLEMENTATION_REPORT.md`. Do not publish a hosted app or public repository without the researcher's later deployment decision.

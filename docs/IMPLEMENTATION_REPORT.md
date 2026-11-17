# Implementation report

## Delivered

The proposal-aligned Streamlit app is implemented in this separate project. It supports COMPAS, UCI Adult held-out predictions, six fixed-seed synthetic scenarios, and session-only CSV uploads. It includes input mapping and validation, score thresholding, pure metric calculations, two-proportion screening, minimum-group warnings, Plotly group charts, explanations, and in-memory ReportLab reports.

Adult classifier fitting is explicit preparation-time work only. The UI never trains models. COMPAS raw data is parsed in memory, then only five analysis columns are retained; the preparation manifest records original file checksums and row counts. User-uploaded records are not shown in a preview, globally cached, written to disk, or included in reports.

## Measured checks

- Pinned project environment installed from `requirements.txt` (Fairlearn 0.13.0 and compatible SciPy 1.15.3).
- Python syntax compilation completed for the app, package, scripts and tests.
- Automated suite: **20 passed**. This includes worked-example formulas, score threshold inclusivity, exact DPD/DIR threshold boundaries, malformed binary mappings, duplicate CSV headers, reproducible scenarios, PDF content, Fairlearn group/aggregate parity, the upload-screen regression, and Streamlit navigation/clearing with stale-result invalidation.
- Worked example: DPD 0.20, DIR 0.667, equal opportunity gap 0.20, equalised odds gap 0.20, matching proposal values.
- Real-sample Fairlearn comparisons: **39/39 numerical comparisons within 1e-6; 100% accuracy, MAE 0**. COMPAS: 21 comparisons across per-group rates and aggregate DPD/DIR/equalised odds. Adult held-out: 18 such comparisons.
- Synthetic suite: **600/600 expected statuses correct**. For the binary DPD scenarios S1–S4 (400 cases), TP=200, TN=200, FP=0, FN=0; accuracy, precision and recall each 100%. S5 abstentions and S6 equal-opportunity gaps were separately checked.
- Local analysis/chart-construction timings (10 runs after warmup): COMPAS 6,172 rows mean 0.103 s; Adult held-out 15,060 rows mean 0.191 s. A 48,842-row repeated-row stress fixture took mean 0.184 s for metric calculation. This stress fixture is explicitly not independent Adult observations, and these measurements exclude full browser rendering and PDF generation.
- App HTTP smoke check on `127.0.0.1:8765`: HTTP 200. Headless AppTest rendered without exceptions. The local server was bound to loopback during the smoke check.
- Chrome 157.0.8103.51 and Edge 157.0.4341.60 both loaded the interface over loopback and completed the sample-to-analysis-to-results flow without page errors. Screenshot review led to a primary-button contrast fix and the Plotly configuration was updated to remove a deprecation warning. The final code changes passed the Streamlit AppTest suite; the browser flow was not repeated after those two presentation/API adjustments.
- Impeccable mechanical detector reported no findings after the final UI changes.

Raw evaluation detail: [`../evaluation/synthetic_scenarios.json`](../evaluation/synthetic_scenarios.json). Dataset source hashes and preparation counts: [`../data/samples/manifest.json`](../data/samples/manifest.json).

## Research boundaries and unresolved interpretation

The six scenario labels do not all describe the same binary DP decision: S5 expects insufficient-data abstention, and S6 expects an equal-opportunity gap without a DP flag. The evaluator therefore reports 600-case expected-status accuracy and a separate 400-case binary DP confusion matrix. This is transparent about denominators but still needs to be stated as the study's evaluation interpretation in the dissertation.

The proposal also asks for timing on about 48,000 Adult records while requiring held-out classifier predictions. The app's actual Adult prediction sample is a 15,060-row held-out cohort. The 48,842-row benchmark repeats those rows for a labelled stress test; it must not be reported as a fairness analysis of 48,842 independent records. The displayed preparation counts are 30,162 complete training rows and 15,060 complete held-out rows.

The automated browser flow and Streamlit AppTest cover local Chrome/Edge and interaction behavior. A full browser request-to-render timing study, user heuristic inspection, public hosting, and GitHub publication remain to be completed during the research demonstration.

## UI and upload fix

The unsupported per-widget `max_upload_size` argument was removed for Streamlit 1.50.0. The 25 MiB server limit remains in `.streamlit/config.toml`, with an explicit size check before CSV parsing. The upload page has a CSV example; navigation has a visible four-stage workflow and next-step actions; dataset summaries show row, column and missing-value counts. Upload fingerprints preserve results when revisiting the dataset page. Clear resets the uploader, and changing datasets resets old column mappings.

A real CSV upload was exercised in Edge through Analysis and Results with zero rendered Streamlit exceptions. Desktop (1440 px) and mobile (390 px) captures were inspected. The UI regression suite covers the upload mode, next-step navigation, and dataset clearing. Captions were darkened for legibility and duplicate decision messages were removed.

## Local launch

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Dataset preparation and test commands are in [`APP_GUIDE.md`](APP_GUIDE.md).

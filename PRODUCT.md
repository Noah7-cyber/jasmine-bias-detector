# Product: Jasmine AI Bias Detector

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Python and Streamlit, with pandas, NumPy, SciPy, Plotly, ReportLab, Fairlearn, scikit-learn, and pytest. Chosen in the user's project brief.

## Purpose

A web application for students, auditors, and decision makers to inspect whether binary AI predictions vary across demographic groups. The app calculates descriptive group fairness metrics, checks selected differences against screening and reliability rules, explains limitations, and creates a summary report.

## Users and task

The primary user is a non-programmer with an existing table of binary predictions or model scores. They select a built-in dataset or upload a compatible CSV, choose one demographic field and prediction field, optionally choose ground truth, and review metrics and warnings.

## Product boundaries

The application audits predictions; it does not train or tune models apart from a fixed, untuned Adult baseline used only to generate held-out predictions. It does not mitigate bias, decide whether an outcome is unlawful or unfair, perform causal or individual fairness analysis, or compare intersecting groups. Uploaded data and derived artifacts stay in the user's temporary session and are not persisted.

## Success

Metric outputs agree with manual and Fairlearn references to 1e-6. Six fixed-seed synthetic scenarios can be evaluated reproducibly. A user can inspect group metrics and caveats, recover from invalid data, and download an accurate PDF summary. The complete analysis meets the proposed five-second target at the stated performance size.

## Assumptions

- Desktop-first browser interface, with standard responsive layout on narrower screens.
- Confirmed visual direction: a research workbench, clear and designed for students and auditors.
- Positive-class meaning is explicitly dataset-specific; COMPAS positive means higher risk and is adverse.
- Sample data are local/offline and sourced/cited in the accompanying docs.

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from bias_detector.metrics import AuditSettings, calculate_audit
from bias_detector.data_loader import read_uploaded_csv
from bias_detector.reporting import build_pdf
from bias_detector.synthetic import SCENARIOS, generate_scenario
from bias_detector.validation import binary_values, inspect_frame, selected_missing_counts


ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "data" / "samples"

st.set_page_config(page_title="AI Bias Detector", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --ink:#18352f; --muted:#566a63; --forest:#176b55; --forest-soft:#e7f1ec; --paper:#f7f8f5; --line:#dbe4de; --warning:#8c5415; }
html, body, [class*="css"] { font-family:ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); }
.stApp { background:var(--paper); }
[data-testid="stMainBlockContainer"] { max-width:1200px; padding-top:2.5rem; padding-bottom:3rem; }
[data-testid="stSidebar"] h2 { font-size:1.25rem !important; }
[data-testid="stSidebar"] [role="radiogroup"] { gap:.6rem; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding:.5rem .65rem; border-radius:8px; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:#dce9df; }
[data-testid="stFileUploader"] { background:white; border:1px solid var(--line); border-radius:12px; padding:1rem; }
[data-testid="stButton"] button { min-height:44px; border-radius:8px; }
[data-testid="stDataFrame"] { border-radius:8px; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:var(--muted); opacity:1; }
.workflow { display:flex; gap:0; border-bottom:1px solid var(--line); margin:0 0 1.5rem; padding:0; list-style:none; }
.workflow li { flex:1; padding:.75rem .3rem; color:var(--muted); font-size:.9rem; }
.workflow li[aria-current="step"] { color:var(--forest); border-bottom:2px solid var(--forest); font-weight:650; }
.dataset-facts { display:flex; flex-wrap:wrap; gap:1.5rem; padding:.85rem 0; border-top:1px solid var(--line); border-bottom:1px solid var(--line); margin:.8rem 0 1rem; }
.dataset-facts span { color:var(--muted); font-size:.9rem; }
.dataset-facts strong { color:var(--ink); font-variant-numeric:tabular-nums; }
::selection { background:#cce4d7; color:#18352f; }
input, textarea { caret-color:var(--forest); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#edf2ed; border-right:1px solid var(--line); }
h1,h2,h3 { color:var(--ink); letter-spacing:-.025em; }
h1 { font-size:2.05rem !important; font-weight:700; }
h2 { font-size:1.35rem !important; font-weight:650; }
p, label, li { color:#344c44; }
[data-testid="stMetric"] { background:white; border:1px solid var(--line); padding:15px 17px; border-radius:10px; }
[data-testid="stMetricValue"] { color:var(--ink); font-family:ui-monospace, "Cascadia Code", Consolas, monospace; }
.rule { border:0; border-top:1px solid var(--line); margin:1.1rem 0; }
.source-note { font-size:.87rem; color:var(--muted); }
.warning-box { background:#fff5e4; padding:14px 16px; border-radius:9px; color:#56380f; }
button[kind="primary"] { background:#176b55; border-color:#176b55; color:#ffffff !important; }
button[kind="primary"] p { color:#ffffff !important; }
:focus-visible { outline:3px solid #176b55 !important; outline-offset:2px; }
@media (max-width: 760px) { h1 {font-size:1.7rem !important;} [data-testid="stMetric"] {padding:10px;} [data-testid="stMainBlockContainer"] { padding-top:1.25rem; padding-left:1rem; padding-right:1rem; } .workflow li {font-size:.8rem;} }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_public_sample(filename: str) -> pd.DataFrame:
    """Cache only bundled public samples; uploaded data never enters this cache."""
    return pd.read_csv(SAMPLES / filename)


def sample_options() -> dict[str, tuple[str, str, str, bool]]:
    choices: dict[str, tuple[str, str, str, bool]] = {}
    if (SAMPLES / "compas.csv").exists():
        choices["COMPAS · recidivism risk"] = ("compas.csv", "https://github.com/propublica/compas-analysis", "higher recidivism risk (adverse flag)", True)
    if (SAMPLES / "adult_holdout.csv").exists():
        choices["Adult · held-out baseline predictions"] = ("adult_holdout.csv", "https://archive.ics.uci.edu/dataset/2/adult", "income above $50,000", False)
    if (SAMPLES / "synthetic_s1.csv").exists():
        for name in SCENARIOS:
            index = list(SCENARIOS).index(name) + 1
            choices[name] = (f"synthetic_s{index}.csv", "Seeded synthetic reference data", "positive prediction (scenario-defined)", False)
    return choices


def _set_dataset(frame: pd.DataFrame, name: str, source: str, positive: str, adverse: bool) -> None:
    for key in list(st.session_state):
        if key.startswith("cfg_"):
            del st.session_state[key]
    st.session_state.audit_frame = frame
    st.session_state.dataset_name = name
    st.session_state.dataset_source = source
    st.session_state.positive_description = positive
    st.session_state.positive_is_adverse = adverse
    st.session_state.audit_result = None


def _format(value) -> str:
    return "Unavailable" if value is None else f"{float(value):.4f}"


def _navigate(page: str) -> None:
    st.session_state.page = page


def _clear_dataset() -> None:
    for key in ("audit_frame", "dataset_name", "dataset_source", "positive_description", "positive_is_adverse", "audit_result", "audit_config", "audit_upload", "upload_fingerprint"):
        st.session_state.pop(key, None)


st.title("AI Bias Detector")
st.caption("A practical workbench for examining prediction differences across demographic groups.")
pages = ["Dataset", "Analysis", "Results", "Report"]
st.sidebar.header("Research workspace")
current_page = st.sidebar.radio("Workspace", pages, index=pages.index(st.session_state.get("page", "Dataset")), key="page")
st.sidebar.markdown("---")
st.sidebar.markdown("**Analysis status**")
st.sidebar.caption(st.session_state.get("dataset_name", "No dataset selected"))
st.sidebar.caption("Results ready" if st.session_state.get("audit_result") else "Ready to configure" if st.session_state.get("audit_frame") is not None else "Choose data to begin")
st.sidebar.caption("Upload → compare → investigate")
st.markdown('<ol class="workflow" aria-label="Analysis workflow">' + ''.join(f'<li aria-current="{"step" if p == current_page else "false"}">{i + 1}. {p}</li>' for i, p in enumerate(pages)) + '</ol>', unsafe_allow_html=True)

if current_page == "Dataset":
    st.header("Choose a prediction dataset")
    st.write("Explore a documented sample or bring your own predictions. Choose a demographic attribute, compare group outcomes, and export your findings.")
    options = sample_options()
    source_mode = st.radio("Data source", ["Built-in sample", "Upload a CSV"], horizontal=True, key="data_source_mode")
    if st.session_state.get("last_data_source_mode") != source_mode:
        for key in ("audit_frame", "dataset_name", "dataset_source", "positive_description", "positive_is_adverse", "audit_result", "audit_config", "upload_fingerprint"):
            st.session_state.pop(key, None)
        st.session_state.last_data_source_mode = source_mode
    if source_mode == "Built-in sample" and options:
        col, action = st.columns([3, 1], vertical_alignment="bottom")
        choice = col.selectbox("Built-in samples", list(options), key="sample_choice")
        if action.button("Load sample", type="primary", use_container_width=True):
            filename, source, positive, adverse = options[choice]
            _set_dataset(load_public_sample(filename).copy(), choice, source, positive, adverse)
            st.success(f"Loaded {choice}.")
    elif source_mode == "Built-in sample":
        st.info("Built-in data are not prepared yet. Run `python scripts/prepare_public_data.py` and `python scripts/generate_synthetic.py`.")

    if source_mode == "Upload a CSV":
        st.markdown("#### Upload a CSV")
        st.caption("Files are read from memory for this session only. Keep identifiers out of uploaded files. Maximum file size: 25 MiB.")
        with st.expander("CSV format and example"):
            st.write("Include a demographic column and a prediction column. Actual outcomes are optional. Use two distinct labels for binary predictions, or provide a numerical score.")
            st.code("group,prediction,truth\nA,1,1\nA,0,1\nB,1,0\nB,0,0", language="csv")
    uploaded = st.file_uploader("Choose a prediction CSV", type=["csv"], key="audit_upload") if source_mode == "Upload a CSV" else None
    if source_mode == "Upload a CSV" and uploaded is None and st.session_state.get("dataset_name") != "Uploaded CSV (session only)":
        st.session_state.audit_result = None
    upload_fingerprint = hashlib.sha256(uploaded.getbuffer()).hexdigest() if uploaded is not None else None
    if uploaded is not None and upload_fingerprint != st.session_state.get("upload_fingerprint"):
        for key in ("audit_frame", "dataset_name", "dataset_source", "positive_description", "positive_is_adverse", "audit_result"):
            st.session_state.pop(key, None)
        try:
            if uploaded.size > 25 * 1024 * 1024:
                raise ValueError("The file exceeds 25 MiB. Upload a smaller CSV.")
            user_frame, duplicate_columns = read_uploaded_csv(uploaded.getvalue())
            if duplicate_columns:
                st.error("Duplicate column names need to be renamed before analysis: " + ", ".join(duplicate_columns))
            elif user_frame.empty:
                st.error("This CSV contains no data rows.")
            elif len(user_frame.columns) < 2:
                st.error("Include at least two columns: a demographic attribute and a prediction.")
            elif len(user_frame) > 500_000:
                st.error("Uploads are limited to 500,000 rows.")
            else:
                _set_dataset(user_frame, "Uploaded CSV (session only)", "User-provided CSV", "configured positive class", False)
                st.session_state.upload_fingerprint = upload_fingerprint
                st.success(f"Loaded {len(user_frame):,} rows in session memory.")
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            st.error(f"Could not read this CSV: {exc}")
    if st.session_state.get("audit_frame") is not None:
        frame = st.session_state.audit_frame
        report = inspect_frame(frame)
        st.markdown("---")
        st.subheader(st.session_state.dataset_name)
        st.caption(f"Source: {st.session_state.dataset_source}")
        st.markdown(f'<div class="dataset-facts"><span><strong>{report.rows:,}</strong> rows</span><span><strong>{report.columns}</strong> columns</span><span><strong>{sum(report.missing_by_column.values()):,}</strong> missing values</span></div>', unsafe_allow_html=True)
        if st.session_state.dataset_name.startswith("COMPAS"):
            st.warning("Positive means a higher recidivism risk score, an adverse flag. It is not a favourable selection.")
        missing = frame.isna().sum()
        missing = missing[missing.gt(0)]
        if not missing.empty:
            st.warning("Missing values by column: " + ", ".join(f"{k} ({v:,})" for k, v in missing.items()))
        if st.session_state.dataset_name == "Uploaded CSV (session only)":
            st.caption("For privacy, uploaded record values are not previewed. Choose fields on the Analysis page; only selected columns are processed.")
        else:
            st.dataframe(frame.head(12), width="stretch", hide_index=True)
        st.button("Continue to analysis", type="primary", on_click=_navigate, args=("Analysis",))
        st.button("Clear current dataset and analysis", on_click=_clear_dataset)

elif current_page == "Analysis":
    st.header("Set up the group comparison")
    frame = st.session_state.get("audit_frame")
    if frame is None:
        st.info("Choose a built-in sample or upload a CSV on the Dataset page first.")
    else:
        st.caption(st.session_state.dataset_name + " · Select the fields to compare, then review the screening defaults.")
        columns = frame.columns.astype(str).tolist()
        first_col, second_col = st.columns(2)
        pred_col = second_col.selectbox("Prediction or score column", columns, index=columns.index("risk_score") if "risk_score" in columns else columns.index("income_prediction") if "income_prediction" in columns else 0, key="cfg_prediction")
        group_columns=[c for c in columns if c != pred_col]
        group_col = first_col.selectbox("Demographic attribute", group_columns, index=group_columns.index("race") if "race" in group_columns else 0, help="Analyze one attribute at a time. Combining attributes is outside this study.", key="cfg_group")
        available_truth = [None] + [c for c in columns if c not in (group_col, pred_col)]
        true_col = st.selectbox("True-outcome column (optional)", available_truth, format_func=lambda v: "No true outcome" if v is None else v, index=(available_truth.index("recidivism_2yr") if "recidivism_2yr" in available_truth else available_truth.index("income_true") if "income_true" in available_truth else 0), key="cfg_truth")
        score_mode = st.checkbox("Treat prediction column as a numerical score and apply a threshold", value=pred_col in ("risk_score", "decile_score"), key="cfg_score_mode")
        threshold = st.number_input("Positive prediction when score is at least", value=5.0, step=1.0, disabled=not score_mode, key="cfg_score_threshold")
        vals = binary_values(frame[pred_col]) if pred_col in frame.columns else []
        positive_default = "1" if "1" in vals else (">50K" if ">50K" in vals else (vals[-1] if vals else "1"))
        negative_default = "0" if "0" in vals else (vals[0] if vals and vals[0] != positive_default else "0")
        if score_mode:
            positive_label, negative_label = "1", "0"
            st.caption("Binary label mapping applies only to the true-outcome column when score mode is selected.")
        else:
            map1, map2 = st.columns(2)
            positive_label = map1.selectbox("Positive label", vals or ["1"], index=(vals.index(positive_default) if positive_default in vals else 0), key="cfg_pred_positive")
            neg_choices = [v for v in (vals or ["0"]) if v != positive_label]
            negative_label = map2.selectbox("Negative label", neg_choices or ["0"], index=(neg_choices.index(negative_default) if negative_default in neg_choices else 0), key="cfg_pred_negative")
        if true_col:
            truth_vals = binary_values(frame[true_col])
            tpos = ">50K" if ">50K" in truth_vals else ("1" if "1" in truth_vals else (truth_vals[-1] if truth_vals else "1"))
            truth_map1, truth_map2 = st.columns(2)
            true_positive = truth_map1.selectbox("True positive label", truth_vals or ["1"], index=(truth_vals.index(tpos) if tpos in truth_vals else 0), key="cfg_true_positive")
            truth_negative_choices = [v for v in (truth_vals or ["0"]) if v != true_positive]
            true_negative = truth_map2.selectbox("True negative label", truth_negative_choices or ["0"], key="cfg_true_negative")
        else:
            true_positive, true_negative = "1", "0"
        st.markdown("#### Screening and reliability")
        a, b, c = st.columns(3)
        dpd_threshold = a.number_input("DPD screen", 0.0, 1.0, 0.10, 0.01, help="Default 0.10; flag when the gap is at least this value.", key="cfg_dpd")
        dir_threshold = b.number_input("DIR screen", 0.0, 1.0, 0.80, 0.01, help="Default 0.80; flag when the lowest/highest ratio is below this value.", key="cfg_dir")
        min_group = c.number_input("Minimum group size", 1, 1_000_000, 30, 1, key="cfg_min_group")
        settings = AuditSettings(group_column=group_col, prediction_column=pred_col, score_threshold=threshold, score_mode=score_mode, true_column=true_col,
                                 positive_label=positive_label, negative_label=negative_label,
                                 true_positive_label=true_positive if true_col else None, true_negative_label=true_negative if true_col else None,
                                 dpd_threshold=dpd_threshold, dir_threshold=dir_threshold, minimum_group_size=int(min_group))
        if st.session_state.get("audit_result") and st.session_state.get("audit_config") != settings.__dict__:
            st.session_state.audit_result = None
            st.info("Settings changed. Run the analysis again to refresh results and the report.")
        submitted = st.button("Run analysis", type="primary")
        if submitted:
            try:
                result = calculate_audit(frame, settings, positive_is_adverse=bool(st.session_state.get("positive_is_adverse", False)))
                st.session_state.audit_result = result
                st.session_state.audit_config = settings.__dict__
                st.success("Analysis complete. Review the group results and limitations.")
            except ValueError as exc:
                st.session_state.audit_result = None
                st.error(str(exc))
        miss = selected_missing_counts(frame, [group_col, pred_col] + ([true_col] if true_col else []))
        miss = {k:v for k,v in miss.items() if v}
        if miss:
            st.warning("Selected columns have missing values: " + ", ".join(f"{k} ({v:,})" for k,v in miss.items()) + ". Resolve these values or deselect the true-outcome column before analysis.")
        if st.session_state.get("audit_result"):
            st.button("View results", type="primary", on_click=_navigate, args=("Results",))

elif current_page == "Results":
    st.header("Group comparison results")
    result = st.session_state.get("audit_result")
    if not result:
        st.info("Run an analysis first.")
    else:
        m = result["metrics"]
        if m["positive_is_adverse"]:
            st.warning("Positive predictions mean a higher risk flag (an adverse outcome) for this dataset.")
        else:
            st.caption("Positive class: " + str(st.session_state.get("positive_description", "the configured positive outcome")))
        one, two, three, four = st.columns(4)
        one.metric("Rows analysed", f"{m['rows_analysed']:,}")
        two.metric("DP difference", _format(m["demographic_parity_difference"]))
        three.metric("Impact ratio", _format(m["disparate_impact_ratio"]))
        four.metric("Minimum group", str(m["settings"]["minimum_group_size"]))
        st.write(m["decision_dpd"])
        if m["decision_dir"] != m["decision_dpd"]:
            st.write(m["decision_dir"])
        if m["selection_rate_comparison"]["sparse_expected_cell"]:
            st.warning("At least one expected cell in the selection-rate test is below 5. Interpret the normal-approximation p-value cautiously.")
        st.subheader("Positive prediction rate by group")
        group_df = pd.DataFrame(result["groups"])
        fig = px.bar(group_df, x="group", y="selection_rate", text=group_df["selection_rate"].map(lambda v:f"{v:.1%}"),
                     labels={"group": "Demographic group", "selection_rate": "Selection rate"}, color_discrete_sequence=["#176b55"])
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0,1], height=360, margin=dict(t=20,b=20), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, config={"responsive": True})
        st.dataframe(group_df, width="stretch", hide_index=True)
        comp = m["selection_rate_comparison"]
        st.caption(f"Two-sided selection-rate test compares {comp['high_group']} (highest rate) and {comp['low_group']} (lowest rate): z = {_format(comp['z'])}, p = {_format(comp['p_value_two_sided'])}. A z-test is not calculated for error-rate gaps.")
        if m["true_positive_rate"] is not None:
            st.subheader("Error rates by group")
            rate_rows=[]
            for row in result["groups"]:
                for key, label in (("tpr","True positive rate"),("fpr","False positive rate"),("fnr","False negative rate")):
                    rate_rows.append({"group":row["group"],"metric":label,"rate":row.get(key)})
            rate_df=pd.DataFrame(rate_rows).dropna()
            if not rate_df.empty:
                rate_fig=px.bar(rate_df,x="group",y="rate",color="metric",barmode="group",labels={"group":"Demographic group","rate":"Rate","metric":"Error measure"},color_discrete_sequence=["#176b55","#cc774a","#879b91"])
                rate_fig.update_layout(yaxis_tickformat=".0%",yaxis_range=[0,1],height=360,plot_bgcolor="white",paper_bgcolor="white")
                st.plotly_chart(rate_fig,config={"responsive":True})
            left,right=st.columns(2)
            left.metric("Equal opportunity difference",_format(m["equal_opportunity_difference"]))
            right.metric("Equalised odds difference",_format(m["equalised_odds_difference"]))
        with st.expander("What these measures can and cannot tell you"):
            st.markdown("**Selection rate** is the share of each group receiving a positive prediction. **Demographic parity difference** is the largest minus the smallest group rate. **Disparate impact ratio** divides the smallest rate by the largest; 0.80 is a configurable screening rule of thumb. **Equal opportunity difference** is the TPR gap. **Equalised odds difference** is the larger TPR/FPR gap. These measures capture different definitions and may disagree. They are indicators for further investigation, not legal findings or a verdict that a system is fair or unfair.")
            if m["undefined"]:
                st.warning("Some measures are unavailable: " + " ".join(m["undefined"]))
        st.button("Prepare PDF report", type="primary", on_click=_navigate, args=("Report",))
        if st.button("Clear results"):
            st.session_state.audit_result = None
            st.rerun()

elif current_page == "Report":
    st.header("Download an audit summary")
    result=st.session_state.get("audit_result")
    if not result:
        st.info("Run an analysis first to prepare a report.")
    else:
        try:
            pdf=build_pdf(result,"AI Bias Detector — Summary Report",st.session_state.dataset_name,st.session_state.dataset_source,st.session_state.positive_description)
            st.write("The PDF contains dataset provenance, the selected configuration, group metrics, thresholds, statistical checks and limitations. Uploaded records are not included.")
            st.download_button("Download PDF report",data=pdf,file_name="bias_audit_summary.pdf",mime="application/pdf",type="primary")
        except Exception as exc:
            st.error(f"Could not prepare the report: {exc}")

st.markdown("---")
st.caption("Results are screening indicators for further investigation. Differences alone do not establish discrimination; context, label quality and domain expertise matter.")

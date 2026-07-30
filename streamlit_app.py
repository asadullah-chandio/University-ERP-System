import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.predict import ModelLoadError, load_model_bundle, run_all_modules
from src.schema import MODULE_LABELS
from src.validation import UnsupportedFileTypeError, clean_and_coerce, read_uploaded_file, validate_upload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "erp_models.pkl"
SAMPLE_TEMPLATE_PATH = BASE_DIR / "sample_data" / "sample_students_template.xlsx"
SAMPLE_TEMPLATE_CSV_PATH = BASE_DIR / "sample_data" / "sample_students_template.csv"

st.set_page_config(page_title="University ERP Predictive Analytics", page_icon="🎓", layout="wide")

CUSTOM_CSS = """
<style>
    :root { --accent: #FF1E27; }
    .stApp { background-color: #0E0E10; color: #F2F2F2; }
    h1, h2, h3, h4 { color: #FFFFFF !important; }
    .app-title { color: var(--accent) !important; }
    .stButton > button, .stDownloadButton > button {
        background-color: var(--accent);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
        transition: background-color 0.2s ease-in-out;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #cc1820;
        color: #FFFFFF;
    }
    .module-card {
        background-color: #1A1A1D;
        border: 1px solid #2A2A2E;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .module-card b { color: var(--accent); }
    .footer {
        text-align: center;
        color: #888888;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #2A2A2E;
        font-size: 0.9rem;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #1A1A1D;
        border-color: #2A2A2E;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#FF1E27"
RISK_COLORS = {"Low": "#2ECC71", "Medium": "#F5A623", "High": "#FF1E27"}


@st.cache_resource
def get_model_bundle():
    return load_model_bundle(MODEL_PATH)


try:
    bundle = get_model_bundle()
    model_load_error = None
except ModelLoadError as exc:
    bundle = None
    model_load_error = str(exc)

st.markdown('<h1 class="app-title">🎓 University ERP — Predictive Analytics Demo</h1>', unsafe_allow_html=True)

if model_load_error:
    st.error(f"The prediction models could not be loaded: {model_load_error}")
    st.stop()

st.markdown(
    """
This is a portfolio demo of a predictive analytics layer for a university ERP system,
covering **6 modules** in one place. Upload your own student data (any university's
own Excel/CSV export) or try the sample dataset below — the same 6 models run on
whichever columns are present.
"""
)

with st.container():
    cols = st.columns(3)
    descriptions = [
        ("🚨 Student Risk", "Flags students likely to fail this term, from attendance, CGPA, assignment completion, and backlogs."),
        ("📉 Dropout Prediction", "Estimates dropout risk from attendance, CGPA, fee payment status, and LMS activity."),
        ("💸 Fee Default", "Predicts the chance a student defaults on tuition, from demographics, department, and CGPA."),
        ("📈 GPA Prediction", "Forecasts next-term GPA from attendance, quiz/assignment/exam performance, and current CGPA."),
        ("🎯 Recommendations", "Flags which academic area (quizzes, assignments, exams, labs) a student is weakest in."),
        ("🏛️ Enrollment Forecast", "Projects future enrollment from historical year-over-year enrollment counts."),
    ]
    for i, (title, desc) in enumerate(descriptions):
        with cols[i % 3]:
            st.markdown(f'<div class="module-card"><b>{title}</b><br/><span style="color:#AAAAAA; font-size: 0.9rem;">{desc}</span></div>', unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Data input: upload or sample dataset
# ---------------------------------------------------------------------------
st.subheader("1. Provide student data")

col_upload, col_sample = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload your own student data (.xlsx or .csv)", type=["xlsx", "xls", "csv"]
    )

with col_sample:
    st.write("No data handy?")
    use_sample = st.button("▶ Try the sample dataset", width="stretch")
    if SAMPLE_TEMPLATE_PATH.exists():
        with open(SAMPLE_TEMPLATE_PATH, "rb") as f:
            st.download_button(
                "⬇ Download sample template",
                data=f.read(),
                file_name="sample_students_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

raw_df = None
source_label = None

if use_sample:
    if SAMPLE_TEMPLATE_PATH.exists():
        with open(SAMPLE_TEMPLATE_PATH, "rb") as f:
            raw_df = read_uploaded_file(f.read(), SAMPLE_TEMPLATE_PATH.name)
        source_label = "sample dataset"
    else:
        st.error("Sample dataset not found.")
elif uploaded_file is not None:
    try:
        raw_df = read_uploaded_file(uploaded_file.getvalue(), uploaded_file.name)
        source_label = uploaded_file.name
    except UnsupportedFileTypeError as exc:
        st.error(str(exc))
    except ValueError as exc:
        st.error(f"Could not process file: {exc}")

if raw_df is not None:
    with st.spinner(f"Validating and processing {source_label}..."):
        report = validate_upload(raw_df)

    if not report.is_valid:
        st.error("This file can't be processed:")
        for err in report.errors:
            st.error(f"• {err}")
    else:
        st.success(f"Loaded {len(raw_df)} student record(s) from {source_label}.")
        if report.warnings:
            with st.expander(f"⚠ {len(report.warnings)} validation warning(s)", expanded=False):
                for warning in report.warnings:
                    st.warning(warning)

        with st.spinner("Running predictions across all available modules..."):
            try:
                cleaned = clean_and_coerce(raw_df)
                results = run_all_modules(cleaned, bundle, report.available_modules)
            except Exception:
                logger.exception("Prediction pipeline failed")
                results = {}
                st.error("Something went wrong while generating predictions. Please check your file and try again.")

        if results:
            st.divider()
            st.subheader("2. Results")

            tab_labels = [MODULE_LABELS[m] for m in results.keys()]
            tabs = st.tabs(tab_labels)

            for tab, module_name in zip(tabs, results.keys()):
                with tab:
                    result = results[module_name]

                    if module_name == "student_risk":
                        st.dataframe(result, hide_index=True, width="stretch")
                        fig = px.pie(result, names="risk_level", title="Risk level distribution",
                                     color="risk_level", color_discrete_map=RISK_COLORS, template=PLOTLY_TEMPLATE)
                        st.plotly_chart(fig, width="stretch")

                    elif module_name == "dropout":
                        st.dataframe(result, hide_index=True, width="stretch")
                        fig = px.pie(result, names="risk_level", title="Dropout risk distribution",
                                     color="risk_level", color_discrete_map=RISK_COLORS, template=PLOTLY_TEMPLATE)
                        st.plotly_chart(fig, width="stretch")

                    elif module_name == "fee_default":
                        st.dataframe(result, hide_index=True, width="stretch")
                        fig = px.pie(result, names="risk_level", title="Fee default risk distribution",
                                     color="risk_level", color_discrete_map=RISK_COLORS, template=PLOTLY_TEMPLATE)
                        st.plotly_chart(fig, width="stretch")

                    elif module_name == "gpa":
                        st.dataframe(result, hide_index=True, width="stretch")
                        fig = px.histogram(result, x="predicted_gpa", nbins=20, title="Predicted GPA distribution",
                                            color_discrete_sequence=[ACCENT], template=PLOTLY_TEMPLATE)
                        st.plotly_chart(fig, width="stretch")

                    elif module_name == "recommendation":
                        st.dataframe(result, hide_index=True, width="stretch")
                        fig = px.bar(result["status"].value_counts().reset_index(), x="status", y="count",
                                     title="Students needing attention", color_discrete_sequence=[ACCENT],
                                     template=PLOTLY_TEMPLATE)
                        st.plotly_chart(fig, width="stretch")

                    elif module_name == "enrollment_forecast":
                        hist = result["historical"]
                        fcast = result["forecast"]
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=list(hist.keys()), y=list(hist.values()),
                                                  mode="lines+markers", name="Historical", line=dict(color="#5AC8FA")))
                        fig.add_trace(go.Scatter(x=list(fcast.keys()), y=list(fcast.values()),
                                                  mode="lines+markers", name="Forecast", line=dict(color=ACCENT, dash="dash")))
                        fig.update_layout(title=f"Enrollment: historical + forecast (trend R²={result['trend_r2']:.2f})",
                                           template=PLOTLY_TEMPLATE, xaxis_title="Year", yaxis_title="Students enrolled")
                        st.plotly_chart(fig, width="stretch")
                        st.caption(
                            "Forecast is a simple linear trend on real historical enrollment counts "
                            "(lightweight by design — see README for why Prophet wasn't used)."
                        )
        elif report.is_valid:
            st.info("No modules could run with the columns available in this file.")

st.divider()

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------
with st.expander("📊 How it works — model performance & methodology"):
    st.markdown(f"Models trained on **{bundle.get('n_training_students', '?')} students** from real attendance and exam records.")

    rows = []
    for module in ["student_risk", "dropout", "fee_default"]:
        m = bundle[module]["test_metrics"]
        rows.append({
            "Module": MODULE_LABELS[module], "Model": bundle[module]["model_name"],
            "Accuracy": f"{m['accuracy']:.2f}", "Precision": f"{m['precision']:.2f}",
            "Recall": f"{m['recall']:.2f}", "F1": f"{m['f1']:.2f}",
        })
    st.write("**Classification modules (held-out test set):**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    gpa_metrics = bundle["gpa"]["test_metrics"]
    st.write("**GPA regression module (held-out test set):**")
    st.dataframe(pd.DataFrame([{
        "Model": bundle["gpa"]["model_name"], "R²": f"{gpa_metrics['r2']:.3f}",
        "RMSE": f"{gpa_metrics['rmse']:.3f}", "MAE": f"{gpa_metrics['mae']:.3f}",
    }]), hide_index=True, width="stretch")

    st.markdown(
        """
**Honest notes on methodology:**
- Student Risk and GPA targets are derived from real attendance/exam data.
- Dropout and Fee Default labels are documented, simulated rules (the source
  dataset has no real dropout outcomes or fee-transaction records) — see
  README "Model methodology" for exactly how and why.
- All 4 ML modules use 5-fold cross-validation for model selection and a
  held-out test set (never seen during training or tuning) for the metrics above.
"""
    )

st.markdown('<div class="footer">Built by Vexanex Digital Solutions</div>', unsafe_allow_html=True)

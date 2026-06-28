import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(".")
DATAMART = ROOT / "datamart"
MODEL_STORE = ROOT / "model_store"
REPORTS = ROOT / "reports"

st.set_page_config(
    page_title="Loan Default MLOps Dashboard",
    page_icon="🏦",
    layout="wide",
)

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.metric-card {
    background: linear-gradient(135deg, #1f77b4, #0b3d91);
    padding: 22px;
    border-radius: 18px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.metric-card h3 {
    font-size: 18px;
    margin-bottom: 8px;
}
.metric-card h1 {
    font-size: 34px;
    margin: 0;
}
.status-red {
    background: linear-gradient(135deg, #ff4b4b, #a60000);
    padding: 22px;
    border-radius: 18px;
    color: white;
    text-align: center;
}
.status-green {
    background: linear-gradient(135deg, #2ecc71, #087f23);
    padding: 22px;
    border-radius: 18px;
    color: white;
    text-align: center;
}
.section-box {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

st.title("🏦 Loan Default Prediction - MLOps Dashboard")
st.caption("Model training, batch inference, monitoring, drift detection and governance overview")

metadata_path = MODEL_STORE / "model_metadata.json"
metadata = {}
if metadata_path.exists():
    with open(metadata_path) as f:
        metadata = json.load(f)

best_model = metadata.get("best_model", "N/A")
best_auc = metadata.get("roc_auc", None)

pred_path = DATAMART / "gold_predictions.csv"
pred = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()

metrics_path = DATAMART / "gold_model_training_metrics.csv"
metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()

monitoring_path = DATAMART / "gold_model_monitoring.csv"
mon = pd.read_csv(monitoring_path) if monitoring_path.exists() else pd.DataFrame()

latest_auc = None
latest_psi = None

if not mon.empty:
    if "roc_auc" in mon.columns:
        latest_auc = mon["roc_auc"].dropna().iloc[-1]
    if "score_psi_vs_baseline" in mon.columns:
        latest_psi = mon["score_psi_vs_baseline"].dropna().iloc[-1]

retrain_required = False
if latest_psi is not None and latest_psi > 0.25:
    retrain_required = True
if latest_auc is not None and latest_auc < 0.60:
    retrain_required = True

# KPI Cards
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🏆 Best Model</h3>
        <h1>{best_model}</h1>
    </div>
    """, unsafe_allow_html=True)

with c2:
    auc_text = f"{best_auc:.3f}" if best_auc else "N/A"
    st.markdown(f"""
    <div class="metric-card">
        <h3>📈 Best ROC-AUC</h3>
        <h1>{auc_text}</h1>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>📊 Prediction Rows</h3>
        <h1>{len(pred):,}</h1>
    </div>
    """, unsafe_allow_html=True)

with c4:
    status_class = "status-red" if retrain_required else "status-green"
    status_text = "Retraining Required" if retrain_required else "Model Stable"
    st.markdown(f"""
    <div class="{status_class}">
        <h3>🚦 Status</h3>
        <h1>{status_text}</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Overview",
    "🤖 Model Comparison",
    "📈 Monitoring",
    "📜 Governance"
])

with tab1:
    st.subheader("Pipeline Summary")
    st.markdown("""
    <div class="section-box">
    This dashboard visualises the outputs of the Airflow-based MLOps pipeline.
    It summarises the selected model, prediction volume, model performance, population drift,
    retraining decision and governance policy.
    </div>
    """, unsafe_allow_html=True)

    if not pred.empty:
        st.subheader("Prediction Sample")
        st.dataframe(pred.head(20), use_container_width=True)

with tab2:
    st.subheader("Model Comparison")

    if not metrics.empty:
        st.dataframe(metrics, use_container_width=True)

        chart_metrics = metrics.copy()
        chart_metrics["model_name"] = chart_metrics["model_name"].str.replace("_", " ").str.title()

        fig = px.bar(
            chart_metrics,
            x="model_name",
            y=["roc_auc", "average_precision", "accuracy"],
            barmode="group",
            title="Model Performance Comparison",
            labels={
                "model_name": "Model",
                "value": "Score",
                "variable": "Metric"
            },
            text_auto=".3f",
        )
        fig.update_layout(
            height=500,
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend_title_text="Metric",
        )
        st.plotly_chart(fig, use_container_width=True)

        best_row = metrics.sort_values("roc_auc", ascending=False).iloc[0]
        st.success(
            f"Best model selected: {best_row['model_name']} "
            f"with ROC-AUC = {best_row['roc_auc']:.3f}"
        )
    else:
        st.warning("Model metrics not found. Run the Airflow DAG first.")

with tab3:
    st.subheader("Model Monitoring Over Time")

    if not mon.empty:
        st.dataframe(mon, use_container_width=True)

        if "prediction_month" in mon.columns:
            trend_cols = []
            if "roc_auc" in mon.columns:
                trend_cols.append("roc_auc")
            if "default_rate" in mon.columns:
                trend_cols.append("default_rate")
            if "avg_score" in mon.columns:
                trend_cols.append("avg_score")

            if trend_cols:
                fig_auc = px.line(
                    mon,
                    x="prediction_month",
                    y=trend_cols,
                    markers=True,
                    title="Performance Metrics Over Time",
                    labels={
                        "prediction_month": "Prediction Month",
                        "value": "Metric Value",
                        "variable": "Metric"
                    },
                )
                fig_auc.update_layout(height=500, plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_auc, use_container_width=True)

            if "score_psi_vs_baseline" in mon.columns:
                fig_psi = px.line(
                    mon,
                    x="prediction_month",
                    y="score_psi_vs_baseline",
                    markers=True,
                    title="Population Stability Index Over Time",
                    labels={
                        "prediction_month": "Prediction Month",
                        "score_psi_vs_baseline": "PSI"
                    },
                )
                fig_psi.add_hline(
                    y=0.25,
                    line_dash="dash",
                    annotation_text="Drift Threshold = 0.25",
                    annotation_position="top left"
                )
                fig_psi.update_layout(height=500, plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig_psi, use_container_width=True)

        st.subheader("Retraining Decision")

        if retrain_required:
            st.error(
                f"Retraining Required: PSI = {latest_psi:.3f}, ROC-AUC = {latest_auc:.3f}. "
                "Population drift exceeded the threshold or model performance dropped below the minimum level."
            )
        else:
            st.success(
                f"No Retraining Required: PSI = {latest_psi:.3f}, ROC-AUC = {latest_auc:.3f}. "
                "Model remains within governance thresholds."
            )
    else:
        st.warning("Monitoring file not found. Run the Airflow DAG first.")

with tab4:
    st.subheader("Governance Report")

    sop_path = REPORTS / "governance_and_retraining_sop.txt"
    if sop_path.exists():
        st.text(sop_path.read_text())
    else:
        st.warning("Governance report not found.")

    st.subheader("Governance Rules")
    st.markdown("""
    - Retraining is required if **PSI > 0.25**
    - Retraining is required if **ROC-AUC < 0.60**
    - Model artefacts are stored in `model_store/`
    - Predictions and monitoring outputs are stored in `datamart/`
    - Governance reports are stored in `reports/`
    """)
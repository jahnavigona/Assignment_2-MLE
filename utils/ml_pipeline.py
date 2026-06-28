import json
import os
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow"))
DATA = ROOT / "data"
DATAMART = ROOT / "datamart"
MODEL_STORE = ROOT / "model_store"
REPORTS = ROOT / "reports"
for p in [DATAMART, MODEL_STORE, REPORTS]:
    p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def _clean_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace("_", "", regex=False), errors="coerce")


def _credit_age_months(x):
    if pd.isna(x):
        return np.nan
    text = str(x)
    years = re.search(r"(\d+)\s*Years?", text)
    months = re.search(r"(\d+)\s*Months?", text)
    return (int(years.group(1)) if years else 0) * 12 + (int(months.group(1)) if months else 0)


def build_training_table():
    loans = pd.read_csv(DATA / "lms_loan_daily.csv", parse_dates=["loan_start_date", "snapshot_date"])
    fin = pd.read_csv(DATA / "features_financials.csv", parse_dates=["snapshot_date"])
    attr = pd.read_csv(DATA / "features_attributes.csv", parse_dates=["snapshot_date"])
    click = pd.read_csv(DATA / "feature_clickstream.csv", parse_dates=["snapshot_date"])

    # Label at loan level: default if max overdue after origination is more than 0.
    # This target is computed from future loan repayment history and is NOT used as a feature.
    labels = (
        loans.groupby(["loan_id", "Customer_ID", "loan_start_date", "loan_amt", "tenure"], as_index=False)
        .agg(max_overdue_amt=("overdue_amt", "max"), final_balance=("balance", "last"))
    )
    labels["target_default"] = (labels["max_overdue_amt"] > 0).astype(int)

    # Use only feature snapshots available on or before loan application date to avoid temporal leakage.
    base = labels.sort_values("loan_start_date")
    fin = fin.sort_values("snapshot_date")
    attr = attr.sort_values("snapshot_date")
    click = click.sort_values("snapshot_date")

    for c in ["Annual_Income"]:
        if c in fin.columns:
            fin[c] = _clean_numeric(fin[c])
    if "Credit_History_Age" in fin.columns:
        fin["Credit_History_Age_Months"] = fin["Credit_History_Age"].apply(_credit_age_months)
        fin = fin.drop(columns=["Credit_History_Age"])

    # As-of joins by customer and date.
    out = pd.merge_asof(base, fin, left_on="loan_start_date", right_on="snapshot_date", by="Customer_ID", direction="backward")
    out = pd.merge_asof(out.sort_values("loan_start_date"), attr, left_on="loan_start_date", right_on="snapshot_date", by="Customer_ID", direction="backward", suffixes=("", "_attr"))
    out = pd.merge_asof(out.sort_values("loan_start_date"), click, left_on="loan_start_date", right_on="snapshot_date", by="Customer_ID", direction="backward", suffixes=("", "_click"))

    leakage_cols = ["max_overdue_amt", "final_balance", "snapshot_date", "snapshot_date_attr", "snapshot_date_click", "Name", "SSN"]
    out = out.drop(columns=[c for c in leakage_cols if c in out.columns])
    out = out.dropna(subset=["target_default"])
    out.to_csv(DATAMART / "silver_training_table.csv", index=False)
    return str(DATAMART / "silver_training_table.csv")


def train_and_select_model():
    df = pd.read_csv(DATAMART / "silver_training_table.csv", parse_dates=["loan_start_date"])
    y = df["target_default"]
    X = df.drop(columns=["target_default", "loan_id", "Customer_ID", "loan_start_date"], errors="ignore")

    # Keep loan_amt and tenure because they are known at application time.
    date_cols = X.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    X = X.drop(columns=date_cols, errors="ignore")
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=40,
            max_depth=8,
            min_samples_leaf=30,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE
        ),

        "xgboost": XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=1
        ),
    }
    rows = []
    best_auc, best_name, best_pipe = -1, None, None
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba)
        rows.append({
            "model_name": name,
            "roc_auc": auc,
            "average_precision": average_precision_score(y_test, proba),
            "accuracy": accuracy_score(y_test, pred),
        })
        if auc > best_auc:
            best_auc, best_name, best_pipe = auc, name, pipe

    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics.to_csv(DATAMART / "gold_model_training_metrics.csv", index=False)
    joblib.dump(best_pipe, MODEL_STORE / "best_model.joblib")
    with open(MODEL_STORE / "model_metadata.json", "w") as f:
        json.dump({"best_model": best_name, "selection_metric": "roc_auc", "roc_auc": float(best_auc)}, f, indent=2)
    return best_name


def batch_inference():
    df = pd.read_csv(DATAMART / "silver_training_table.csv", parse_dates=["loan_start_date"])
    model = joblib.load(MODEL_STORE / "best_model.joblib")
    ids = df[["loan_id", "Customer_ID", "loan_start_date", "target_default"]].copy()
    X = df.drop(columns=["target_default", "loan_id", "Customer_ID", "loan_start_date"], errors="ignore")
    date_cols = X.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    X = X.drop(columns=date_cols, errors="ignore")
    ids["prediction_score"] = model.predict_proba(X)[:, 1]
    ids["prediction_label"] = (ids["prediction_score"] >= 0.5).astype(int)
    ids["prediction_month"] = pd.to_datetime(ids["loan_start_date"]).dt.to_period("M").astype(str)
    ids.to_csv(DATAMART / "gold_predictions.csv", index=False)
    return str(DATAMART / "gold_predictions.csv")


def _psi(expected, actual, buckets=10):
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    if expected.empty or actual.empty:
        return np.nan
    breaks = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(breaks) < 3:
        return 0.0
    e_counts = pd.cut(expected, bins=breaks, include_lowest=True).value_counts(normalize=True).sort_index()
    a_counts = pd.cut(actual, bins=breaks, include_lowest=True).value_counts(normalize=True).sort_index()
    e, a = e_counts.align(a_counts, fill_value=0)
    e = e.replace(0, 0.0001)
    a = a.replace(0, 0.0001)
    return float(((a - e) * np.log(a / e)).sum())


def monitor_model():
    pred = pd.read_csv(DATAMART / "gold_predictions.csv", parse_dates=["loan_start_date"])
    months = sorted(pred["prediction_month"].dropna().unique())
    baseline = pred[pred["prediction_month"] == months[0]]["prediction_score"] if months else pred["prediction_score"]
    rows = []
    for m, g in pred.groupby("prediction_month"):
        if g["target_default"].nunique() > 1:
            auc = roc_auc_score(g["target_default"], g["prediction_score"])
        else:
            auc = np.nan
        rows.append({
            "prediction_month": m,
            "records": len(g),
            "default_rate": g["target_default"].mean(),
            "avg_score": g["prediction_score"].mean(),
            "roc_auc": auc,
            "score_psi_vs_baseline": _psi(baseline, g["prediction_score"]),
        })
    mon = pd.DataFrame(rows).sort_values("prediction_month")
    mon["performance_status"] = np.where(mon["roc_auc"] < 0.60, "review", "ok")
    mon["stability_status"] = np.where(mon["score_psi_vs_baseline"] > 0.25, "drift_alert", "ok")
    mon.to_csv(DATAMART / "gold_model_monitoring.csv", index=False)

    plt.figure(figsize=(9, 4.8))
    plt.plot(mon["prediction_month"], mon["roc_auc"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("Model ROC-AUC over time")
    plt.tight_layout()
    plt.savefig(REPORTS / "model_performance_auc.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    plt.plot(mon["prediction_month"], mon["score_psi_vs_baseline"], marker="o")
    plt.axhline(0.25, linestyle="--")
    plt.xticks(rotation=45, ha="right")
    plt.title("Prediction score drift: PSI vs baseline")
    plt.tight_layout()
    plt.savefig(REPORTS / "model_stability_psi.png", dpi=160)
    plt.close()
    return str(DATAMART / "gold_model_monitoring.csv")


def write_governance_note():
    note = """Model governance and retraining SOP\n\nPrediction point: loan application date. Only customer features with snapshot_date <= loan_start_date are used.\nModel store: model_store/best_model.joblib with metadata in model_metadata.json.\nPerformance monitor: monthly ROC-AUC. Review if ROC-AUC < 0.60 or falls materially from training benchmark.\nStability monitor: monthly PSI on prediction scores against baseline month. Drift alert if PSI > 0.25.\nRetraining SOP: retrain monthly or when performance_status=review / stability_status=drift_alert for two consecutive months.\nDeployment option: batch Airflow scoring for daily/monthly loan applications; promote to API service only if real-time application scoring is required.\nApproval: model owner reviews metrics, data leakage checks, feature documentation, and rollback model before production refresh.\n"""
    (REPORTS / "governance_and_retraining_sop.txt").write_text(note)
    return str(REPORTS / "governance_and_retraining_sop.txt")


if __name__ == "__main__":
    build_training_table()
    train_and_select_model()
    batch_inference()
    monitor_model()
    write_governance_note()

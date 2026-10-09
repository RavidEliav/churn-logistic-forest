import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import train

BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_PATH = BASE_DIR / "churn_modelling.csv"

st.set_page_config(page_title="Churn Model Comparison", layout="wide")


@st.cache_data
def load_artifacts():
    metrics = json.loads((ARTIFACTS_DIR / "metrics.json").read_text())
    predictions = pd.read_csv(ARTIFACTS_DIR / "test_predictions.csv")
    weights = pd.read_csv(ARTIFACTS_DIR / "feature_importance.csv")
    return metrics, predictions, weights


@st.cache_resource
def load_model(file_name: str):
    return joblib.load(ARTIFACTS_DIR / file_name)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_tuning():
    return train.saved_params(), pd.read_csv(train.CV_RESULTS_PATH)


def artifacts_are_current() -> bool:
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    tuned = train.saved_params()
    if not metrics_path.exists() or tuned is None:
        return False
    saved = json.loads(metrics_path.read_text())
    return set(saved) == set(tuned) and all(
        saved[name].get("params") == tuned[name]["params"] for name in tuned
    )


# Artifacts are not committed (RF models are large); tuned params in tuning/ are, so deploys
# refit the best models quickly instead of re-running the grid search
if not artifacts_are_current():
    with st.spinner("Training models..."):
        train.main(tune=train.saved_params() is None)
    st.cache_data.clear()
    st.cache_resource.clear()

metrics, predictions, weights = load_artifacts()
model_names = list(metrics)
y_true = predictions["y_true"]

st.sidebar.header("Settings")
selected = st.sidebar.multiselect("Models", model_names, default=model_names)
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)
st.sidebar.caption(
    f"Test set: {len(y_true):,} customers, churn rate {y_true.mean():.1%}"
)

st.title("Customer Churn: Logistic Regression vs Random Forest vs Gradient Boosting")

if not selected:
    st.warning("Select at least one model in the sidebar.")
    st.stop()


def threshold_metrics(name: str) -> dict:
    proba = predictions[name]
    pred = (proba >= threshold).astype(int)
    return {
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred),
        "F1 Score": f1_score(y_true, pred),
        "ROC AUC": roc_auc_score(y_true, proba),
    }


summary = pd.DataFrame({name: threshold_metrics(name) for name in selected}).T

tabs = st.tabs(
    [
        "Metrics Overview",
        "ROC Curves",
        "Precision-Recall",
        "Confusion Matrices",
        "Feature Importance",
        "Cross-Validation",
        "Predict Customer",
    ]
)

with tabs[0]:
    st.subheader(f"Metrics at threshold {threshold:.2f}")
    cols = st.columns(len(summary.columns))
    for col, metric in zip(cols, summary.columns):
        best = summary[metric].idxmax()
        col.metric(f"Best {metric}", f"{summary.loc[best, metric]:.3f}", best,
                   delta_color="off")

    st.dataframe(
        summary.style.format("{:.4f}").highlight_max(axis=0, color="#2e7d32"),
        width="stretch",
    )

    long = summary.reset_index(names="Model").melt(
        id_vars="Model", var_name="Metric", value_name="Score"
    )
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group",
                 text_auto=".3f", range_y=[0, 1])
    st.plotly_chart(fig, width="stretch")
    st.caption("ROC AUC is threshold-independent; the other metrics change with the slider.")

with tabs[1]:
    fig = go.Figure()
    for name in selected:
        fpr, tpr, _ = roc_curve(y_true, predictions[name])
        auc = roc_auc_score(y_true, predictions[name])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"{name} (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(dash="dash", color="gray")))
    fig.update_layout(xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate", height=550)
    st.plotly_chart(fig, width="stretch")

with tabs[2]:
    fig = go.Figure()
    for name in selected:
        prec, rec, _ = precision_recall_curve(y_true, predictions[name])
        ap = average_precision_score(y_true, predictions[name])
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                 name=f"{name} (AP={ap:.3f})"))
    fig.add_hline(y=y_true.mean(), line_dash="dash", line_color="gray",
                  annotation_text="Baseline (churn rate)")
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=550)
    st.plotly_chart(fig, width="stretch")

with tabs[3]:
    st.subheader(f"Confusion matrices at threshold {threshold:.2f}")
    cols = st.columns(2)
    labels = ["Stayed (0)", "Churned (1)"]
    for i, name in enumerate(selected):
        pred = (predictions[name] >= threshold).astype(int)
        cm = confusion_matrix(y_true, pred)
        fig = px.imshow(cm, x=labels, y=labels, text_auto=True,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual", color="Count"))
        fig.update_layout(title=name, coloraxis_showscale=False, height=380)
        cols[i % 2].plotly_chart(fig, width="stretch")

with tabs[4]:
    top_n = st.slider("Top features", 5, 13, 13)
    cols = st.columns(2)
    for i, name in enumerate(selected):
        w = weights[weights["model"] == name].copy()
        kind = w["kind"].iloc[0]
        w["feature"] = w["feature"].str.replace(r"^(num|cat)__", "", regex=True)
        w = w.reindex(w["value"].abs().sort_values().index).tail(top_n)
        fig = px.bar(w, x="value", y="feature", orientation="h",
                     color="value", color_continuous_scale="RdBu",
                     color_continuous_midpoint=0 if kind == "coefficient" else None)
        fig.update_layout(title=f"{name} - {kind}", height=450,
                          coloraxis_showscale=False, yaxis_title=None)
        cols[i % 2].plotly_chart(fig, width="stretch")
    st.caption("Logistic Regression coefficients are on standardized features; "
               "positive values increase churn probability. "
               "Random Forest and Gradient Boosting show impurity-based importances.")

with tabs[5]:
    tuned, cv_results = load_tuning()
    st.subheader(f"GridSearchCV with {train.CV_FOLDS}-fold stratified cross-validation")
    st.caption(f"Run on the 80% training set only; best parameters chosen by mean CV "
               f"{train.REFIT_METRIC.upper().replace('_', ' ')}, then refit on the full "
               f"training set and evaluated on the held-out test set.")

    metric_labels = {"precision": "Precision", "recall": "Recall", "f1": "F1 Score",
                     "roc_auc": "ROC AUC"}
    rows = []
    for name in selected:
        row = {"Model": name,
               "Best parameters": ", ".join(
                   f"{k.removeprefix('clf__')}={v}"
                   for k, v in tuned[name]["params"].items()),
               "Combinations": int((cv_results["model"] == name).sum())}
        for key, label in metric_labels.items():
            cv = tuned[name]["cv"][key]
            row[f"CV {label}"] = f"{cv['mean']:.3f} ± {cv['std']:.3f}"
        row["Test ROC AUC"] = f"{metrics[name]['roc_auc']:.3f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    cv_long = pd.DataFrame(
        [{"Model": name, "Metric": label, "Mean": tuned[name]["cv"][key]["mean"],
          "Std": tuned[name]["cv"][key]["std"]}
         for name in selected for key, label in metric_labels.items()]
    )
    fig = px.bar(cv_long, x="Metric", y="Mean", color="Model", barmode="group",
                 error_y="Std", range_y=[0, 1],
                 title="Mean CV score ± std (best parameters, threshold 0.5)")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    fold_metric = c1.selectbox("Per-fold metric", list(metric_labels),
                               format_func=metric_labels.get,
                               index=list(metric_labels).index(train.REFIT_METRIC))
    folds = pd.DataFrame(
        [{"Model": name, "Fold": k + 1, "Score": score}
         for name in selected
         for k, score in enumerate(tuned[name]["cv"][fold_metric]["folds"])]
    )
    fig = px.box(folds, x="Model", y="Score", color="Model", points="all",
                 title=f"{metric_labels[fold_metric]} per fold")
    fig.update_layout(showlegend=False, xaxis_title=None)
    c1.plotly_chart(fig, width="stretch")

    gap = pd.DataFrame(
        [{"Model": name, "Set": s, "ROC AUC": v}
         for name in selected
         for s, v in [("CV (train)", tuned[name]["cv"]["roc_auc"]["mean"]),
                      ("Test", metrics[name]["roc_auc"])]]
    )
    fig = px.bar(gap, x="Model", y="ROC AUC", color="Set", barmode="group",
                 text_auto=".3f", range_y=[0.5, 1], title="CV vs test ROC AUC")
    fig.update_layout(xaxis_title=None)
    c2.plotly_chart(fig, width="stretch")

    with st.expander("All grid search combinations"):
        grid_model = st.selectbox("Model", selected, key="grid_model")
        grid = cv_results[cv_results["model"] == grid_model].sort_values("rank")
        grid = grid.assign(params=grid["params"].str.replace("clf__", ""))
        cols = ["rank", "params"] + [f"mean_{m}" for m in metric_labels] + ["fit_time"]
        st.dataframe(grid[cols].style.format(precision=4), width="stretch",
                     hide_index=True)

with tabs[6]:
    data = load_data()
    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        customer = {
            "CreditScore": c1.number_input("Credit score", 300, 900, 650),
            "Geography": c1.selectbox("Geography", sorted(data["Geography"].unique())),
            "Gender": c1.selectbox("Gender", sorted(data["Gender"].unique())),
            "Age": c2.number_input("Age", 18, 100, 40),
            "Tenure": c2.number_input("Tenure (years)", 0, 10, 5),
            "Balance": c2.number_input("Balance", 0.0, 300000.0, 75000.0, 1000.0),
            "NumOfProducts": c3.number_input("Number of products", 1, 4, 1),
            "HasCrCard": int(c3.checkbox("Has credit card", True)),
            "IsActiveMember": int(c3.checkbox("Active member", True)),
            "EstimatedSalary": c3.number_input("Estimated salary", 0.0, 250000.0,
                                               100000.0, 1000.0),
        }
        submitted = st.form_submit_button("Predict churn")

    if submitted:
        row = pd.DataFrame([customer])
        results = pd.DataFrame(
            {
                "Model": selected,
                "Churn probability": [
                    load_model(metrics[n]["file"]).predict_proba(row)[0, 1]
                    for n in selected
                ],
            }
        )
        results["Prediction"] = results["Churn probability"].map(
            lambda p: "Churn" if p >= threshold else "Stay"
        )
        fig = px.bar(results, x="Model", y="Churn probability", color="Prediction",
                     text_auto=".1%", range_y=[0, 1],
                     color_discrete_map={"Churn": "#d62728", "Stay": "#2ca02c"})
        fig.add_hline(y=threshold, line_dash="dash",
                      annotation_text=f"Threshold {threshold:.2f}")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(results.style.format({"Churn probability": "{:.1%}"}),
                     width="stretch", hide_index=True)

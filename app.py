import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import train

st.set_page_config(page_title="Churn Model Comparison", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem;padding-bottom:1rem}</style>",
            unsafe_allow_html=True)

STAGE_LABELS = {"baseline": "Before tuning (defaults)", "tuned": "After tuning (GridSearchCV)"}
METRICS = {"precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "AUC"}
COLORS = dict(zip(train.PARAM_GRIDS, px.colors.qualitative.Plotly))


def artifacts_are_current() -> bool:
    tuned = train.saved_params()
    if tuned is None or not train.METRICS_PATH.exists():
        return False
    saved = json.loads(train.METRICS_PATH.read_text())
    return set(saved) == set(train.STAGES) and all(
        saved["tuned"].get(n, {}).get("params") == tuned[n]["params"] for n in tuned
    )


@st.cache_data
def load():
    return (json.loads(train.METRICS_PATH.read_text()),
            pd.read_csv(train.PREDICTIONS_PATH), train.saved_params())


# Results aren't in git but tuned params are, so deploys refit without re-running the search
if not artifacts_are_current():
    with st.spinner("Training models..."):
        train.main(tune=train.saved_params() is None)
    st.cache_data.clear()

metrics, predictions, best = load()
y_true = predictions["y_true"]
models = list(train.PARAM_GRIDS)

head, ctrl1, ctrl2 = st.columns([3, 2, 1.3], vertical_alignment="bottom")
head.markdown("### Customer churn: balanced models")
head.caption(f"Train 80% / test 20% ({len(y_true):,} customers, churn rate "
             f"{y_true.mean():.1%}). All models use balanced class weights.")
stage = ctrl1.segmented_control("Hyperparameters", list(STAGE_LABELS),
                                format_func=STAGE_LABELS.get, default="tuned") or "tuned"
other = "baseline" if stage == "tuned" else "tuned"
threshold = ctrl2.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)


def scores(stage_name: str, name: str) -> dict:
    proba = predictions[f"{stage_name}|{name}"]
    pred = (proba >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
    }


current = {n: scores(stage, n) for n in models}
previous = {n: scores(other, n) for n in models}

for col, name in zip(st.columns(3), models):
    with col.container(border=True):
        st.markdown(f"**{name}**")
        for c, (key, label) in zip(st.columns(4), METRICS.items()):
            delta = current[name][key] - previous[name][key]
            c.metric(label, f"{current[name][key]:.3f}",
                     f"{delta:+.3f}" if stage == "tuned" else None)

left, right = st.columns(2)
fig = go.Figure()
for name in models:
    fpr, tpr, _ = roc_curve(y_true, predictions[f"{stage}|{name}"])
    fig.add_scatter(x=fpr, y=tpr, name=f"{name} ({current[name]['roc_auc']:.3f})",
                    line=dict(color=COLORS[name]))
fig.add_scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color="gray"))
fig.update_layout(title="ROC curves (AUC)", height=330, margin=dict(t=40, b=10, l=10, r=10),
                  xaxis_title="False positive rate", yaxis_title="True positive rate",
                  legend=dict(x=0.45, y=0.05))
left.plotly_chart(fig, width="stretch")

bars = pd.DataFrame([{"Model": n, "Metric": METRICS[k], "Score": v}
                     for n in models for k, v in current[n].items()])
fig = px.bar(bars, x="Metric", y="Score", color="Model", barmode="group", text_auto=".2f",
             range_y=[0, 1], color_discrete_map=COLORS, title=f"Metrics at threshold {threshold:.2f}")
fig.update_layout(height=330, margin=dict(t=40, b=10, l=10, r=10), legend_title=None,
                  xaxis_title=None, legend=dict(orientation="h", y=-0.15))
right.plotly_chart(fig, width="stretch")

cm_cols = st.columns([1, 1, 1, 2.2])
labels = ["Stay", "Churn"]
for col, name in zip(cm_cols, models):
    cm = confusion_matrix(y_true, (predictions[f"{stage}|{name}"] >= threshold).astype(int))
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual"))
    fig.update_layout(title=name, height=250, coloraxis_showscale=False,
                      margin=dict(t=35, b=5, l=5, r=5))
    col.plotly_chart(fig, width="stretch")

rows = []
for name in models:
    params = metrics[stage][name]["params"]
    row = {"Model": name,
           "Hyperparameters": ", ".join(f"{k.removeprefix('clf__')}={v}"
                                        for k, v in params.items())}
    if stage == "tuned":
        cv = best[name]["cv"]
        row[f"CV AUC ({train.CV_FOLDS}-fold)"] = (
            f"{cv['roc_auc']['mean']:.3f} ± {cv['roc_auc']['std']:.3f}")
        row["CV F1"] = f"{cv['f1']['mean']:.3f} ± {cv['f1']['std']:.3f}"
    rows.append(row)
with cm_cols[3]:
    st.markdown(f"**Hyperparameters: {STAGE_LABELS[stage].lower()}**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"Tuned with GridSearchCV + {train.CV_FOLDS}-fold stratified CV on the training set, "
        f"best by mean CV AUC. Deltas on the cards compare against the defaults."
        if stage == "tuned" else
        "scikit-learn defaults for the tuned hyperparameters. Switch to "
        "*After tuning* to see the grid search results and improvements.")

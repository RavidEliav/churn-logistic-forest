import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "churn_modelling.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

TARGET = "Exited"
CATEGORICAL = ["Geography", "Gender"]
NUMERIC = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
RANDOM_STATE = 42
# GradientBoostingClassifier has no class_weight, so these are balanced via fit sample weights
SAMPLE_WEIGHTED = {"Gradient Boosting (balanced)"}


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        remainder="drop",  # discards CustomerId and Surname
    )


def build_models() -> dict[str, Pipeline]:
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Logistic Regression (balanced)": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Gradient Boosting (balanced)": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }
    return {
        name: Pipeline([("prep", build_preprocessor()), ("clf", est)])
        for name, est in estimators.items()
    }


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def feature_weights(name: str, pipe: Pipeline) -> pd.DataFrame:
    features = pipe.named_steps["prep"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    if isinstance(clf, LogisticRegression):
        values, kind = clf.coef_[0], "coefficient"
    else:
        values, kind = clf.feature_importances_, "importance"
    return pd.DataFrame(
        {"model": name, "feature": features, "value": values, "kind": kind}
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    # Split raw data first; all preparation is fitted inside the pipeline on the 80% train set only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows\n")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    predictions = pd.DataFrame({"y_true": y_test.to_numpy()})
    metrics, weights = {}, []

    for name, pipe in build_models().items():
        fit_params = {}
        if name in SAMPLE_WEIGHTED:
            fit_params["clf__sample_weight"] = compute_sample_weight("balanced", y_train)
        pipe.fit(X_train, y_train, **fit_params)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        predictions[name] = proba
        metrics[name] = {
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "file": f"{slugify(name)}.joblib",
        }
        weights.append(feature_weights(name, pipe))
        joblib.dump(pipe, ARTIFACTS_DIR / metrics[name]["file"])

    predictions.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)
    pd.concat(weights).to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    summary = pd.DataFrame(metrics).T[["precision", "recall", "f1", "roc_auc"]]
    print(summary.astype(float).round(4).to_string())
    print(f"\nArtifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()

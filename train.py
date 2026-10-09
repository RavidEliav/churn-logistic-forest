import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "churn_modelling.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"
TUNING_DIR = BASE_DIR / "tuning"
BEST_PARAMS_PATH = TUNING_DIR / "best_params.json"
CV_RESULTS_PATH = TUNING_DIR / "cv_results.csv"

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
STAGES = ("baseline", "tuned")

CV_FOLDS = 10
REFIT_METRIC = "roc_auc"
SCORING = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

PARAM_GRIDS = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__l1_ratio": [0, 1],  # 0 = L2, 1 = L1
    },
    "Random Forest": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 10],
        "clf__min_samples_leaf": [1, 5],
        "clf__max_features": ["sqrt", 0.5],
    },
    "Gradient Boosting": {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [3, 4],
        "clf__subsample": [0.8, 1.0],
    },
}


def build_models() -> dict[str, Pipeline]:
    """All models are balanced; GradientBoostingClassifier has no class_weight, see fit_params()."""
    estimators = {
        "Logistic Regression": LogisticRegression(
            solver="liblinear", max_iter=1000, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        remainder="drop",  # discards CustomerId and Surname
    )
    return {
        name: Pipeline([("prep", clone(preprocessor)), ("clf", est)])
        for name, est in estimators.items()
    }


def fit_params(name: str, y) -> dict:
    if name == "Gradient Boosting":
        return {"clf__sample_weight": compute_sample_weight("balanced", y)}
    return {}


def saved_params() -> dict | None:
    """Best params from a previous grid search, or None if missing or out of date."""
    if not BEST_PARAMS_PATH.exists():
        return None
    saved = json.loads(BEST_PARAMS_PATH.read_text())
    if set(saved) != set(PARAM_GRIDS) or any(
        saved[name]["grid"] != PARAM_GRIDS[name] for name in saved
    ):
        return None
    return saved


def grid_search(name: str, pipe: Pipeline, X, y) -> tuple[dict, pd.DataFrame]:
    search = GridSearchCV(
        pipe,
        PARAM_GRIDS[name],
        scoring=SCORING,
        refit=False,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    ).fit(X, y, **fit_params(name, y))

    res = search.cv_results_
    best = int(res[f"rank_test_{REFIT_METRIC}"].argmin())
    cv = {
        m: {
            "mean": float(res[f"mean_test_{m}"][best]),
            "std": float(res[f"std_test_{m}"][best]),
            "folds": [float(res[f"split{k}_test_{m}"][best]) for k in range(CV_FOLDS)],
        }
        for m in SCORING
    }
    table = pd.DataFrame(
        {
            "model": name,
            "params": [json.dumps(p) for p in res["params"]],
            **{f"mean_{m}": res[f"mean_test_{m}"] for m in SCORING},
            "rank": res[f"rank_test_{REFIT_METRIC}"],
        }
    )
    return {"params": res["params"][best], "grid": PARAM_GRIDS[name], "cv": cv}, table


def evaluate(pipe: Pipeline, name: str, X_test, y_test) -> tuple[dict, list]:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    clf_params = pipe.named_steps["clf"].get_params()
    return {
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "params": {k: clf_params[k.removeprefix("clf__")] for k in PARAM_GRIDS[name]},
    }, proba.tolist()


def main(tune: bool = True) -> None:
    df = pd.read_csv(DATA_PATH)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    # Split raw data first; all preparation is fitted inside the pipeline on the 80% train set only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    best_params = None if tune else saved_params()
    cv_tables = []
    if best_params is None:
        print(f"Grid search, {CV_FOLDS}-fold stratified CV on the training set "
              f"(best by {REFIT_METRIC})")
        best_params = {}

    metrics = {stage: {} for stage in STAGES}
    predictions = pd.DataFrame({"y_true": y_test.to_numpy()})

    for name, pipe in build_models().items():
        if name not in best_params:
            start = time.perf_counter()
            best_params[name], table = grid_search(name, pipe, X_train, y_train)
            cv_tables.append(table)
            print(f"  {name}: CV {REFIT_METRIC}="
                  f"{best_params[name]['cv'][REFIT_METRIC]['mean']:.4f} "
                  f"({time.perf_counter() - start:.0f}s)")

        tuned = clone(pipe).set_params(**best_params[name]["params"])
        for stage, model in zip(STAGES, (clone(pipe), tuned)):
            model.fit(X_train, y_train, **fit_params(name, y_train))
            metrics[stage][name], predictions[f"{stage}|{name}"] = evaluate(
                model, name, X_test, y_test
            )

    if cv_tables:
        TUNING_DIR.mkdir(exist_ok=True)
        BEST_PARAMS_PATH.write_text(json.dumps(best_params, indent=2))
        pd.concat(cv_tables).to_csv(CV_RESULTS_PATH, index=False)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    for stage in STAGES:
        summary = pd.DataFrame(metrics[stage]).T[["precision", "recall", "f1", "roc_auc"]]
        print(f"\nTest set, {stage} (threshold 0.5):")
        print(summary.astype(float).round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tune", action="store_true",
                        help="reuse tuning/best_params.json instead of running the grid search")
    main(tune=not parser.parse_args().no_tune)

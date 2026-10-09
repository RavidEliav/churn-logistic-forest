import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd
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
# GradientBoostingClassifier has no class_weight, so these are balanced via fit sample weights
SAMPLE_WEIGHTED = {"Gradient Boosting (balanced)"}

CV_FOLDS = 5
REFIT_METRIC = "roc_auc"
SCORING = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

LR_GRID = {
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__l1_ratio": [0, 1],  # 0 = L2, 1 = L1
}
RF_GRID = {
    "clf__n_estimators": [200, 400],
    "clf__max_depth": [None, 10],
    "clf__min_samples_leaf": [1, 5],
    "clf__max_features": ["sqrt", 0.5],
}
GB_GRID = {
    "clf__n_estimators": [100, 200],
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_depth": [3, 4],
    "clf__subsample": [0.8, 1.0],
}
PARAM_GRIDS = {
    "Logistic Regression": LR_GRID,
    "Logistic Regression (balanced)": LR_GRID,
    "Random Forest": RF_GRID,
    "Random Forest (balanced)": RF_GRID,
    "Gradient Boosting": GB_GRID,
    "Gradient Boosting (balanced)": GB_GRID,
}


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
        "Logistic Regression": LogisticRegression(solver="liblinear", max_iter=1000),
        "Logistic Regression (balanced)": LogisticRegression(
            solver="liblinear", max_iter=1000, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE
        ),
        "Random Forest (balanced)": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
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


def saved_params() -> dict | None:
    """Best params from a previous grid search, or None if missing or out of date."""
    if not BEST_PARAMS_PATH.exists():
        return None
    saved = json.loads(BEST_PARAMS_PATH.read_text())
    if set(saved) != set(build_models()) or any(
        saved[name]["grid"] != PARAM_GRIDS[name] for name in saved
    ):
        return None
    return saved


def grid_search(name: str, pipe: Pipeline, X, y, fit_params: dict) -> GridSearchCV:
    search = GridSearchCV(
        pipe,
        PARAM_GRIDS[name],
        scoring=SCORING,
        refit=REFIT_METRIC,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    return search.fit(X, y, **fit_params)


def summarize_search(name: str, search: GridSearchCV) -> tuple[dict, pd.DataFrame]:
    res = search.cv_results_
    best = search.best_index_
    cv = {
        metric: {
            "mean": float(res[f"mean_test_{metric}"][best]),
            "std": float(res[f"std_test_{metric}"][best]),
            "folds": [float(res[f"split{k}_test_{metric}"][best]) for k in range(CV_FOLDS)],
        }
        for metric in SCORING
    }
    summary = {"params": search.best_params_, "grid": PARAM_GRIDS[name], "cv": cv}

    table = pd.DataFrame(
        {
            "model": name,
            "params": [json.dumps(p) for p in res["params"]],
            **{f"mean_{m}": res[f"mean_test_{m}"] for m in SCORING},
            **{f"std_{m}": res[f"std_test_{m}"] for m in SCORING},
            "rank": res[f"rank_test_{REFIT_METRIC}"],
            "fit_time": res["mean_fit_time"],
        }
    )
    return summary, table


def main(tune: bool = True) -> None:
    df = pd.read_csv(DATA_PATH)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    # Split raw data first; all preparation is fitted inside the pipeline on the 80% train set only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows\n")

    best_params = None if tune else saved_params()
    cv_tables = []
    if best_params is None:
        print(f"Grid search with {CV_FOLDS}-fold stratified CV on the training set "
              f"(refit on {REFIT_METRIC})...")
        best_params = {}

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    predictions = pd.DataFrame({"y_true": y_test.to_numpy()})
    metrics, weights = {}, []

    for name, pipe in build_models().items():
        fit_params = {}
        if name in SAMPLE_WEIGHTED:
            fit_params["clf__sample_weight"] = compute_sample_weight("balanced", y_train)

        if name in best_params:
            pipe.set_params(**best_params[name]["params"])
            pipe.fit(X_train, y_train, **fit_params)
        else:
            start = time.perf_counter()
            search = grid_search(name, pipe, X_train, y_train, fit_params)
            best_params[name], table = summarize_search(name, search)
            cv_tables.append(table)
            pipe = search.best_estimator_
            print(f"  {name}: CV {REFIT_METRIC}={search.best_score_:.4f} "
                  f"({time.perf_counter() - start:.0f}s) {search.best_params_}")

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        predictions[name] = proba
        metrics[name] = {
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "params": best_params[name]["params"],
            "file": f"{slugify(name)}.joblib",
        }
        weights.append(feature_weights(name, pipe))
        joblib.dump(pipe, ARTIFACTS_DIR / metrics[name]["file"])

    if cv_tables:
        TUNING_DIR.mkdir(exist_ok=True)
        BEST_PARAMS_PATH.write_text(json.dumps(best_params, indent=2))
        pd.concat(cv_tables).to_csv(CV_RESULTS_PATH, index=False)

    predictions.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)
    pd.concat(weights).to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    summary = pd.DataFrame(metrics).T[["precision", "recall", "f1", "roc_auc"]]
    print("\nTest set (threshold 0.5):")
    print(summary.astype(float).round(4).to_string())
    print(f"\nArtifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tune", action="store_true",
                        help="reuse tuning/best_params.json instead of running the grid search")
    main(tune=not parser.parse_args().no_tune)

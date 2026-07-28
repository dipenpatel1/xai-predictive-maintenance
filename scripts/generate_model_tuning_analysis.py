from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from predictive_maintenance.config import RANDOM_STATE
from predictive_maintenance.data_loader import load_ai4i_dataset


OUTPUT_DIR = Path("outputs/model_tuning")

INPUT_FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

NUMERIC_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

CATEGORICAL_FEATURES = ["Type"]


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_tree(max_depth=4, class_weight="balanced"):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    class_weight=class_weight,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_predictions(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "true_negatives": cm[0][0],
        "false_positives": cm[0][1],
        "false_negatives": cm[1][0],
        "true_positives": cm[1][1],
    }


def threshold_analysis(model, x_test, y_test):
    probabilities = model.predict_proba(x_test)[:, 1]
    rows = []

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        predictions = (probabilities >= threshold).astype(int)
        row = {"threshold": threshold}
        row.update(evaluate_predictions(y_test, predictions))
        rows.append(row)

    return pd.DataFrame(rows)


def depth_analysis(x_train, x_test, y_train, y_test):
    rows = []

    for depth in [2, 3, 4, 5, 6, 8, None]:
        model = build_tree(max_depth=depth, class_weight="balanced")
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        row = {"max_depth": "None" if depth is None else depth}
        row.update(evaluate_predictions(y_test, predictions))
        rows.append(row)

    return pd.DataFrame(rows)


def class_weight_analysis(x_train, x_test, y_train, y_test):
    class_weights = [
        None,
        "balanced",
        {0: 1, 1: 3},
        {0: 1, 1: 5},
        {0: 1, 1: 10},
        {0: 1, 1: 20},
    ]
    labels = [
        "none",
        "balanced",
        "failure_weight_3",
        "failure_weight_5",
        "failure_weight_10",
        "failure_weight_20",
    ]
    rows = []

    for label, weight in zip(labels, class_weights):
        model = build_tree(max_depth=4, class_weight=weight)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        row = {"class_weight": label}
        row.update(evaluate_predictions(y_test, predictions))
        rows.append(row)

    return pd.DataFrame(rows)


def anomaly_contamination_analysis(x_train, x_test, y_train, y_test):
    preprocessor = build_preprocessor()
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)
    normal_mask = y_train == 0
    x_train_normal = x_train_processed[normal_mask]

    rows = []
    for contamination in [0.02, 0.03, 0.05, 0.08, 0.10]:
        model = IsolationForest(
            contamination=contamination,
            random_state=RANDOM_STATE,
        )
        model.fit(x_train_normal)
        raw_predictions = model.predict(x_test_processed)
        anomaly_flags = (raw_predictions == -1).astype(int)

        row = {"contamination": contamination}
        row.update(evaluate_predictions(y_test, anomaly_flags))
        row["records_flagged_as_anomalous"] = int(anomaly_flags.sum())
        rows.append(row)

    return pd.DataFrame(rows)


def plot_metric_table(frame, x_col, filename, title):
    frame = frame.copy()
    if frame[x_col].dtype == "object":
        frame[x_col] = frame[x_col].astype(str)

    plot_frame = frame.melt(
        id_vars=[x_col],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="value",
    )

    plt.figure(figsize=(8, 5))
    sns.pointplot(data=plot_frame, x=x_col, y="value", hue="metric")
    plt.title(title)
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    save_figure(OUTPUT_DIR / filename)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_ai4i_dataset()
    x = data[INPUT_FEATURES].copy()
    y = data["Machine failure"].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    base_model = build_tree(max_depth=4, class_weight="balanced")
    base_model.fit(x_train, y_train)

    threshold_frame = threshold_analysis(base_model, x_test, y_test)
    threshold_frame.to_csv(OUTPUT_DIR / "threshold_tuning.csv", index=False)
    plot_metric_table(
        threshold_frame,
        "threshold",
        "threshold_tuning_precision_recall_f1.png",
        "Threshold Tuning: Precision, Recall, and F1",
    )

    depth_frame = depth_analysis(x_train, x_test, y_train, y_test)
    depth_frame.to_csv(OUTPUT_DIR / "decision_tree_depth_tuning.csv", index=False)
    plot_metric_table(
        depth_frame,
        "max_depth",
        "decision_tree_depth_tuning.png",
        "Decision Tree Depth Tuning",
    )

    class_weight_frame = class_weight_analysis(x_train, x_test, y_train, y_test)
    class_weight_frame.to_csv(OUTPUT_DIR / "class_weight_tuning.csv", index=False)
    plot_metric_table(
        class_weight_frame,
        "class_weight",
        "class_weight_tuning.png",
        "Class Weight Tuning",
    )

    anomaly_frame = anomaly_contamination_analysis(x_train, x_test, y_train, y_test)
    anomaly_frame.to_csv(OUTPUT_DIR / "anomaly_contamination_tuning.csv", index=False)
    plot_metric_table(
        anomaly_frame,
        "contamination",
        "anomaly_contamination_tuning.png",
        "Anomaly Contamination Tuning",
    )

    print(f"Model tuning outputs saved to {OUTPUT_DIR}")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"- {path}")


if __name__ == "__main__":
    main()

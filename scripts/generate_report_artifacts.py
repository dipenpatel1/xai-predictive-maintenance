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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path("data/raw/ai4i2020.csv")
OUTPUT_DIR = Path("outputs/report_artifacts")
RANDOM_STATE = 42

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


def build_classifier():
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    classifier = DecisionTreeClassifier(
        max_depth=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def generate_classification_artifacts():
    df = pd.read_csv(DATA_PATH)
    x = df[INPUT_FEATURES].copy()
    y = df["Machine failure"].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_classifier()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_probability = model.predict_proba(x_test)[:, 1]

    metrics = pd.DataFrame(
        [
            {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }
        ]
    )
    metrics.to_csv(OUTPUT_DIR / "classification_metrics.csv", index=False)

    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
    report.to_csv(OUTPUT_DIR / "classification_report.csv")

    cm = confusion_matrix(y_test, y_pred)
    cm_frame = pd.DataFrame(
        cm,
        index=["actual_no_failure", "actual_failure"],
        columns=["predicted_no_failure", "predicted_failure"],
    )
    cm_frame.to_csv(OUTPUT_DIR / "classification_confusion_matrix.csv")

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_frame, annot=True, fmt="d", cmap="Blues")
    plt.title("Failure Classification Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    save_figure(OUTPUT_DIR / "classification_confusion_matrix.png")

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(OUTPUT_DIR / "classification_feature_importance.csv", index=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=feature_importance, x="importance", y="feature")
    plt.title("Failure Classification Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    save_figure(OUTPUT_DIR / "classification_feature_importance.png")

    precision, recall, thresholds = precision_recall_curve(y_test, y_probability)
    pr_frame = pd.DataFrame(
        {
            "precision": precision[:-1],
            "recall": recall[:-1],
            "threshold": thresholds,
        }
    )
    pr_frame.to_csv(OUTPUT_DIR / "classification_precision_recall_thresholds.csv", index=False)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, alpha=0.3)
    save_figure(OUTPUT_DIR / "classification_precision_recall_curve.png")

    predictions = x_test.copy()
    predictions["actual_machine_failure"] = y_test.values
    predictions["predicted_machine_failure"] = y_pred
    predictions["predicted_failure_probability"] = y_probability
    predictions.to_csv(OUTPUT_DIR / "classification_predictions.csv", index=False)


def generate_anomaly_artifacts():
    anomaly_path = Path("outputs/anomaly_detection_test_results.csv")
    explanation_path = Path("outputs/top_anomaly_explanations.csv")

    if not anomaly_path.exists():
        return

    results = pd.read_csv(anomaly_path)

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=results,
        x="anomaly_score",
        hue="Machine failure",
        bins=40,
        kde=True,
    )
    plt.title("Anomaly Score Distribution by Failure Label")
    plt.xlabel("Anomaly score (lower = more anomalous)")
    plt.ylabel("Count")
    save_figure(OUTPUT_DIR / "anomaly_score_distribution.png")

    sensor_means = (
        results.groupby("anomaly_flag")[NUMERIC_FEATURES]
        .mean()
        .round(3)
        .rename(index={0: "normal_flag", 1: "anomaly_flag"})
    )
    sensor_means.to_csv(OUTPUT_DIR / "anomaly_sensor_means_by_flag.csv")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=results,
        x="Rotational speed",
        y="Torque",
        hue="anomaly_flag",
        style="Machine failure",
        alpha=0.75,
    )
    plt.title("Rotational Speed vs Torque by Anomaly Flag")
    plt.xlabel("Rotational speed")
    plt.ylabel("Torque")
    save_figure(OUTPUT_DIR / "anomaly_speed_torque_scatter.png")

    if explanation_path.exists():
        explanations = pd.read_csv(explanation_path)
        explanations.to_csv(OUTPUT_DIR / "top_anomaly_explanations_for_report.csv", index=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_classification_artifacts()
    generate_anomaly_artifacts()

    print(f"Report artifacts saved to {OUTPUT_DIR}")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"- {path}")


if __name__ == "__main__":
    main()

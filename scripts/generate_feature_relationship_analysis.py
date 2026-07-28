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


OUTPUT_DIR = Path("outputs/feature_relationships")

BASE_FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

NUMERIC_BASE_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

ENGINEERED_FEATURES = [
    "Temperature difference",
    "Power proxy",
    "Wear torque interaction",
    "Speed torque ratio",
]


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def add_engineered_features(data):
    """Add relationship-based features derived from the sensor columns."""
    engineered = data.copy()

    engineered["Temperature difference"] = (
        engineered["Process temperature"] - engineered["Air temperature"]
    )
    engineered["Power proxy"] = engineered["Torque"] * engineered["Rotational speed"]
    engineered["Wear torque interaction"] = engineered["Tool wear"] * engineered["Torque"]
    engineered["Speed torque ratio"] = engineered["Rotational speed"] / (
        engineered["Torque"] + 1e-6
    )

    return engineered


def build_model(numeric_features, categorical_features=None, max_depth=4):
    if categorical_features is None:
        categorical_features = ["Type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_model(name, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    cm = confusion_matrix(y_test, predictions)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "true_negatives": cm[0][0],
        "false_positives": cm[0][1],
        "false_negatives": cm[1][0],
        "true_positives": cm[1][1],
    }


def save_feature_importance(model, path):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_importance = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    feature_importance.to_csv(path, index=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=feature_importance, x="importance", y="feature")
    plt.title("Feature Importance With Engineered Relationship Features")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    save_figure(path.with_suffix(".png"))

    return feature_importance


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_ai4i_dataset()
    data = add_engineered_features(data)

    relationship_features = NUMERIC_BASE_FEATURES + ENGINEERED_FEATURES
    modeling_features = ["Type"] + relationship_features

    correlation = data[relationship_features + ["Machine failure"]].corr().round(4)
    correlation.to_csv(OUTPUT_DIR / "feature_correlation_matrix.csv")

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix With Engineered Features")
    save_figure(OUTPUT_DIR / "feature_correlation_matrix.png")

    target_correlation = (
        correlation["Machine failure"]
        .drop("Machine failure")
        .sort_values(key=lambda series: series.abs(), ascending=False)
        .rename("correlation_with_machine_failure")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    target_correlation.to_csv(
        OUTPUT_DIR / "correlation_with_machine_failure.csv", index=False
    )

    pair_plots = [
        ("Rotational speed", "Torque", "speed_torque_relationship.png"),
        ("Tool wear", "Torque", "tool_wear_torque_relationship.png"),
        ("Temperature difference", "Rotational speed", "temperature_difference_speed_relationship.png"),
        ("Power proxy", "Wear torque interaction", "power_wear_interaction_relationship.png"),
    ]

    for x_feature, y_feature, filename in pair_plots:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=data.sample(n=2000, random_state=RANDOM_STATE),
            x=x_feature,
            y=y_feature,
            hue="Machine failure",
            alpha=0.75,
        )
        plt.title(f"{x_feature} vs {y_feature} by Failure Label")
        save_figure(OUTPUT_DIR / filename)

    x_base = data[BASE_FEATURES].copy()
    x_engineered = data[modeling_features].copy()
    y = data["Machine failure"].copy()

    x_base_train, x_base_test, y_train, y_test = train_test_split(
        x_base,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    x_eng_train, x_eng_test, _, _ = train_test_split(
        x_engineered,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    base_model = build_model(NUMERIC_BASE_FEATURES)
    engineered_model = build_model(relationship_features)

    metrics = [
        evaluate_model(
            "base_features",
            base_model,
            x_base_train,
            x_base_test,
            y_train,
            y_test,
        ),
        evaluate_model(
            "base_plus_engineered_features",
            engineered_model,
            x_eng_train,
            x_eng_test,
            y_train,
            y_test,
        ),
    ]

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(OUTPUT_DIR / "base_vs_engineered_model_metrics.csv", index=False)

    save_feature_importance(
        engineered_model,
        OUTPUT_DIR / "engineered_model_feature_importance.csv",
    )

    engineered_preview = data[
        BASE_FEATURES + ENGINEERED_FEATURES + ["Machine failure"]
    ].head(20)
    engineered_preview.to_csv(OUTPUT_DIR / "engineered_feature_examples.csv", index=False)

    print(f"Feature relationship outputs saved to {OUTPUT_DIR}")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"- {path}")


if __name__ == "__main__":
    main()

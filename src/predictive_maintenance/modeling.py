import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from predictive_maintenance.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)


def build_preprocessor():
    """Create preprocessing for numeric and categorical AI4I features."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_decision_tree_pipeline():
    """Build the main interpretable classifier."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=4,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_logistic_regression_pipeline():
    """Build a simple baseline classifier."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_anomaly_detector(x_train):
    """Fit Isolation Forest on preprocessed training features."""
    preprocessor = build_preprocessor()
    x_train_preprocessed = preprocessor.fit_transform(x_train)

    detector = IsolationForest(
        contamination=0.05,
        random_state=RANDOM_STATE,
    )
    detector.fit(x_train_preprocessed)

    return preprocessor, detector


def predict_anomalies(preprocessor, detector, x):
    """Return 1 for anomalous records and 0 for normal records."""
    x_preprocessed = preprocessor.transform(x)
    raw_predictions = detector.predict(x_preprocessed)
    return (raw_predictions == -1).astype(int)


def evaluate_classifier(model, x_test, y_test):
    """Return common classification metrics."""
    predictions = model.predict(x_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def metrics_to_frame(metrics_by_model):
    """Convert nested metric dictionaries into a readable table."""
    rows = []

    for model_name, metrics in metrics_by_model.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
        )

    return pd.DataFrame(rows)


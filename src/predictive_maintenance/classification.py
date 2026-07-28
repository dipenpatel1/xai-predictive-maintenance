import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from predictive_maintenance.config import RANDOM_STATE
from predictive_maintenance.preprocessing import MaintenancePreprocessor


class FailureClassificationExperiment:
    """Train, evaluate, and explain a failure classifier."""

    def __init__(
        self,
        classifier=None,
        max_depth=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ):
        self.random_state = random_state
        self.preprocessor = MaintenancePreprocessor(random_state=random_state)
        self.classifier = classifier or DecisionTreeClassifier(
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
        )
        self.model = None
        self.split_data = None
        self.predictions = None
        self.probabilities = None

    def fit(self, data):
        """Fit preprocessing and classifier as a single pipeline."""
        self.split_data = self.preprocessor.train_test_split(data)
        self.model = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor.build_transformer()),
                ("classifier", self.classifier),
            ]
        )
        self.model.fit(self.split_data.x_train, self.split_data.y_train)
        return self

    def predict(self, threshold=0.5):
        """Predict failure labels at a configurable probability threshold."""
        probabilities = self.predict_proba()
        self.predictions = (probabilities >= threshold).astype(int)
        return self.predictions

    def predict_proba(self):
        """Return predicted failure probabilities for the test set."""
        if self.probabilities is None:
            self.probabilities = self.model.predict_proba(self.split_data.x_test)[:, 1]
        return self.probabilities

    def metrics(self, threshold=0.5):
        """Return common classification metrics at a selected threshold."""
        y_pred = self.predict(threshold=threshold)
        y_test = self.split_data.y_test

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    def metrics_frame(self, threshold=0.5):
        """Return metrics as a one-row DataFrame."""
        metrics = self.metrics(threshold=threshold).copy()
        metrics["threshold"] = threshold
        return pd.DataFrame([metrics])

    def classification_report(self, threshold=0.5):
        """Return sklearn classification report text."""
        y_pred = self.predict(threshold=threshold)
        return classification_report(
            self.split_data.y_test,
            y_pred,
            target_names=["No failure", "Failure"],
            zero_division=0,
        )

    def predictions_frame(self, threshold=0.5):
        """Return test records with actual labels, predictions, and probabilities."""
        probabilities = self.predict_proba()
        predictions = self.predict(threshold=threshold)

        frame = self.split_data.x_test.copy()
        frame["actual_machine_failure"] = self.split_data.y_test.values
        frame["predicted_machine_failure"] = predictions
        frame["predicted_failure_probability"] = probabilities
        return frame

    def feature_importance(self):
        """Return global feature importance for tree-based classifiers."""
        preprocessor = self.model.named_steps["preprocessor"]
        classifier = self.model.named_steps["classifier"]

        if not hasattr(classifier, "feature_importances_"):
            raise TypeError("Classifier does not expose feature_importances_.")

        return pd.DataFrame(
            {
                "feature": preprocessor.get_feature_names_out(),
                "importance": classifier.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

    def threshold_table(self, thresholds=None):
        """Evaluate precision/recall tradeoff across thresholds."""
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        rows = []
        for threshold in thresholds:
            metrics = self.metrics(threshold=threshold)
            cm = metrics["confusion_matrix"]
            rows.append(
                {
                    "threshold": threshold,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "false_positives": cm[0][1],
                    "false_negatives": cm[1][0],
                    "true_positives": cm[1][1],
                    "true_negatives": cm[0][0],
                }
            )

        return pd.DataFrame(rows)

    def precision_recall_thresholds(self):
        """Return precision-recall values across learned probability thresholds."""
        precision, recall, thresholds = precision_recall_curve(
            self.split_data.y_test,
            self.predict_proba(),
        )
        return pd.DataFrame(
            {
                "precision": precision[:-1],
                "recall": recall[:-1],
                "threshold": thresholds,
            }
        )

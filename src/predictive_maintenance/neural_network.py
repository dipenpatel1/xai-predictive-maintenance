import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from predictive_maintenance.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)
from predictive_maintenance.preprocessing import MaintenancePreprocessor


class NeuralNetworkExperiment:
    """Train and evaluate a small feedforward network on AI4I data."""

    def __init__(
        self,
        hidden_layer_sizes=(16, 8),
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=RANDOM_STATE,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.preprocessor = MaintenancePreprocessor(random_state=random_state)
        self.model = None
        self.split_data = None
        self.probabilities = None
        self.selected_threshold = None
        self.threshold_selection_table = None

    def build_pipeline(self):
        """Build dense preprocessing followed by a regularized MLP."""
        transformer = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), NUMERIC_FEATURES),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
        classifier = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=self.random_state,
        )
        return Pipeline(
            steps=[
                ("preprocessor", transformer),
                ("classifier", classifier),
            ]
        )

    def fit(self, data):
        """Fit the network with balanced sample weights."""
        self.split_data = self.preprocessor.train_test_split(data)
        self.model = self.build_pipeline()
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=self.split_data.y_train,
        )
        self.model.fit(
            self.split_data.x_train,
            self.split_data.y_train,
            classifier__sample_weight=sample_weights,
        )
        self.probabilities = None
        return self

    def predict_proba(self):
        """Return failure probabilities for the held-out test set."""
        if self.probabilities is None:
            self.probabilities = self.model.predict_proba(
                self.split_data.x_test
            )[:, 1]
        return self.probabilities

    def predict(self, threshold=0.5):
        """Convert failure probabilities into labels at a chosen threshold."""
        return (self.predict_proba() >= threshold).astype(int)

    def metrics(self, threshold=0.5):
        """Return classification and ranking metrics."""
        y_test = self.split_data.y_test
        probabilities = self.predict_proba()
        predictions = (probabilities >= threshold).astype(int)
        matrix = confusion_matrix(y_test, predictions)

        return {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "pr_auc": average_precision_score(y_test, probabilities),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "false_positives": int(matrix[0, 1]),
            "false_negatives": int(matrix[1, 0]),
            "true_positives": int(matrix[1, 1]),
            "true_negatives": int(matrix[0, 0]),
            "threshold": threshold,
        }

    def metrics_frame(self, threshold=0.5):
        """Return test metrics as a one-row table."""
        return pd.DataFrame([self.metrics(threshold=threshold)])

    def classification_report(self, threshold=0.5):
        """Return sklearn's class-level report."""
        return classification_report(
            self.split_data.y_test,
            self.predict(threshold=threshold),
            target_names=["No failure", "Failure"],
            zero_division=0,
        )

    def predictions_frame(self, threshold=0.5):
        """Return test records with actual labels and network outputs."""
        frame = self.split_data.x_test.copy()
        frame["actual_machine_failure"] = self.split_data.y_test.values
        frame["predicted_machine_failure"] = self.predict(threshold=threshold)
        frame["predicted_failure_probability"] = self.predict_proba()
        return frame

    def training_history(self):
        """Return training loss and available validation scores."""
        classifier = self.model.named_steps["classifier"]
        history = pd.DataFrame(
            {
                "iteration": np.arange(1, len(classifier.loss_curve_) + 1),
                "training_loss": classifier.loss_curve_,
            }
        )
        if hasattr(classifier, "validation_scores_"):
            history["validation_accuracy"] = classifier.validation_scores_
        return history

    def threshold_table(self, thresholds=None):
        """Evaluate held-out metrics across descriptive thresholds."""
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        return pd.DataFrame(
            [self.metrics(threshold=float(value)) for value in thresholds]
        )

    def select_threshold_from_training(self, folds=5, thresholds=None):
        """Select an F1-maximizing threshold using training folds only."""
        if thresholds is None:
            thresholds = np.linspace(0.05, 0.95, 181)
        if self.split_data is None:
            raise RuntimeError("Call fit(data) before selecting a threshold.")

        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=self.split_data.y_train,
        )
        cross_validation = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=self.random_state,
        )
        out_of_fold_probabilities = cross_val_predict(
            self.build_pipeline(),
            self.split_data.x_train,
            self.split_data.y_train,
            cv=cross_validation,
            method="predict_proba",
            params={"classifier__sample_weight": sample_weights},
        )[:, 1]

        rows = []
        for threshold in thresholds:
            predictions = (out_of_fold_probabilities >= threshold).astype(int)
            rows.append(
                {
                    "threshold": float(threshold),
                    "precision": precision_score(
                        self.split_data.y_train,
                        predictions,
                        zero_division=0,
                    ),
                    "recall": recall_score(
                        self.split_data.y_train,
                        predictions,
                        zero_division=0,
                    ),
                    "f1": f1_score(
                        self.split_data.y_train,
                        predictions,
                        zero_division=0,
                    ),
                }
            )

        self.threshold_selection_table = pd.DataFrame(rows)
        best_row = self.threshold_selection_table.loc[
            self.threshold_selection_table["f1"].idxmax()
        ]
        self.selected_threshold = float(best_row["threshold"])
        return best_row.copy()

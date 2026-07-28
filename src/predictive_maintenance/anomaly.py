import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

from predictive_maintenance.config import NUMERIC_FEATURES, RANDOM_STATE
from predictive_maintenance.preprocessing import MaintenancePreprocessor


class AnomalyDetectionExperiment:
    """Train, evaluate, and explain an anomaly detector."""

    def __init__(
        self,
        contamination=0.05,
        random_state=RANDOM_STATE,
        train_on_normal_only=True,
    ):
        self.contamination = contamination
        self.random_state = random_state
        self.train_on_normal_only = train_on_normal_only
        self.preprocessor = MaintenancePreprocessor(random_state=random_state)
        self.transformer = None
        self.model = None
        self.split_data = None
        self.x_train_processed = None
        self.x_test_processed = None
        self.anomaly_flags = None
        self.anomaly_scores = None

    def fit(self, data):
        """Fit preprocessing and anomaly model."""
        self.split_data = self.preprocessor.train_test_split(data)
        (
            self.transformer,
            self.x_train_processed,
            self.x_test_processed,
        ) = self.preprocessor.fit_transform_train_test(self.split_data)

        if self.train_on_normal_only:
            normal_mask = self.split_data.y_train == 0
            training_matrix = self.x_train_processed[normal_mask]
        else:
            training_matrix = self.x_train_processed

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.model.fit(training_matrix)
        return self

    def predict(self):
        """Predict anomaly flags for the test set."""
        raw_predictions = self.model.predict(self.x_test_processed)
        self.anomaly_flags = (raw_predictions == -1).astype(int)
        self.anomaly_scores = self.model.decision_function(self.x_test_processed)
        return self.anomaly_flags

    def results_frame(self):
        """Return test records with anomaly flags and scores."""
        if self.anomaly_flags is None:
            self.predict()

        results = self.split_data.x_test.copy()
        results["Machine failure"] = self.split_data.y_test.values
        results["anomaly_flag"] = self.anomaly_flags
        results["anomaly_score"] = self.anomaly_scores
        return results

    def summary_metrics(self):
        """Return key anomaly detection summary metrics."""
        if self.anomaly_flags is None:
            self.predict()

        y_test = self.split_data.y_test
        flags = self.anomaly_flags

        return pd.DataFrame(
            {
                "metric": [
                    "total_test_records",
                    "actual_failures_in_test_set",
                    "records_flagged_as_anomalous",
                    "anomaly_rate",
                    "failures_flagged_as_anomalous",
                    "normal_records_flagged_as_anomalous",
                    "failures_not_flagged_as_anomalous",
                ],
                "value": [
                    len(y_test),
                    int((y_test == 1).sum()),
                    int(flags.sum()),
                    flags.mean(),
                    int(((y_test == 1) & (flags == 1)).sum()),
                    int(((y_test == 0) & (flags == 1)).sum()),
                    int(((y_test == 1) & (flags == 0)).sum()),
                ],
            }
        )

    def confusion_matrix(self):
        """Compare actual failures with anomaly flags."""
        if self.anomaly_flags is None:
            self.predict()
        return confusion_matrix(self.split_data.y_test, self.anomaly_flags)

    def classification_report(self):
        """Return sklearn classification report comparing failure vs anomaly flag."""
        if self.anomaly_flags is None:
            self.predict()
        return classification_report(
            self.split_data.y_test,
            self.anomaly_flags,
            target_names=["Normal", "Failure/Anomaly"],
            zero_division=0,
        )

    def top_anomalies(self, n=10):
        """Return the n most anomalous test records."""
        return self.results_frame().sort_values("anomaly_score").head(n)

    def explain_record(self, row, top_n=3):
        """Explain one record by deviation from normal training behavior."""
        normal_mask = self.split_data.y_train == 0
        normal_train = self.split_data.x_train.loc[normal_mask, NUMERIC_FEATURES]
        normal_means = normal_train.mean()
        normal_stds = normal_train.std()

        z_scores = ((row[NUMERIC_FEATURES] - normal_means) / normal_stds).abs()
        top_features = z_scores.sort_values(ascending=False).head(top_n)

        return ", ".join(
            f"{feature}: {value:.2f} std away"
            for feature, value in top_features.items()
        )

    def top_anomaly_explanations(self, n=10):
        """Return top anomalies with simple explanation text."""
        top = self.top_anomalies(n=n).copy()
        top["simple_explanation"] = top.apply(self.explain_record, axis=1)
        return top

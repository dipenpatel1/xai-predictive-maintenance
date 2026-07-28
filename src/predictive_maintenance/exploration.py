import pandas as pd

from predictive_maintenance.config import (
    FAILURE_MODE_COLUMNS,
    INPUT_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


class DatasetExplorer:
    """Reusable dataset exploration helpers for the AI4I dataset."""

    def __init__(self, data):
        self.data = data

    def overview(self):
        """Return basic dataset shape and columns."""
        return {
            "rows": len(self.data),
            "columns": len(self.data.columns),
            "column_names": list(self.data.columns),
        }

    def missing_values(self):
        """Return missing value counts per column."""
        return self.data.isna().sum().rename("missing_values").reset_index(
            names="column"
        )

    def target_balance(self):
        """Return class counts and failure rate for the target label."""
        counts = self.data[TARGET_COLUMN].value_counts().sort_index()
        return {
            "counts": counts,
            "failure_rate": self.data[TARGET_COLUMN].mean(),
        }

    def sensor_summary(self):
        """Return summary statistics for numeric sensor features."""
        return self.data[NUMERIC_FEATURES].describe().round(3)

    def normal_vs_failure_means(self):
        """Compare average sensor values for normal and failure records."""
        return self.data.groupby(TARGET_COLUMN)[NUMERIC_FEATURES].mean().round(3)

    def feature_roles(self):
        """Return the intended role of each important column group."""
        return {
            "input_features": INPUT_FEATURES,
            "target_column": TARGET_COLUMN,
            "failure_mode_columns_for_evaluation_only": FAILURE_MODE_COLUMNS,
        }

    def report_tables(self):
        """Return common exploration tables in one dictionary."""
        return {
            "missing_values": self.missing_values(),
            "sensor_summary": self.sensor_summary(),
            "normal_vs_failure_means": self.normal_vs_failure_means(),
            "target_counts": pd.DataFrame(
                self.target_balance()["counts"]
            ).reset_index(),
        }

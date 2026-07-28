from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictive_maintenance.config import (
    CATEGORICAL_FEATURES,
    FAILURE_MODE_COLUMNS,
    INPUT_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
)


@dataclass
class SplitData:
    """Container for train/test split outputs."""

    x_train: object
    x_test: object
    y_train: object
    y_test: object
    failure_modes_train: object
    failure_modes_test: object


class MaintenancePreprocessor:
    """Prepare AI4I features and labels without data leakage."""

    def __init__(
        self,
        input_features=None,
        target_column=TARGET_COLUMN,
        failure_mode_columns=None,
        numeric_features=None,
        categorical_features=None,
        random_state=RANDOM_STATE,
    ):
        self.input_features = input_features or INPUT_FEATURES
        self.target_column = target_column
        self.failure_mode_columns = failure_mode_columns or FAILURE_MODE_COLUMNS
        self.numeric_features = numeric_features or NUMERIC_FEATURES
        self.categorical_features = categorical_features or CATEGORICAL_FEATURES
        self.random_state = random_state

    def split_features_target(self, data):
        """Create X, y, and failure-mode labels for evaluation only."""
        x = data[self.input_features].copy()
        y = data[self.target_column].copy()
        failure_modes = data[self.failure_mode_columns].copy()
        return x, y, failure_modes

    def train_test_split(self, data, test_size=0.2):
        """Split X/y while preserving the rare failure ratio."""
        x, y, failure_modes = self.split_features_target(data)

        x_train, x_test, y_train, y_test, modes_train, modes_test = train_test_split(
            x,
            y,
            failure_modes,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )

        return SplitData(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            failure_modes_train=modes_train,
            failure_modes_test=modes_test,
        )

    def build_transformer(self):
        """Build a reusable preprocessing transformer for sklearn pipelines."""
        return ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), self.numeric_features),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    self.categorical_features,
                ),
            ]
        )

    def fit_transform_train_test(self, split_data):
        """Fit preprocessing on train data and transform train/test data."""
        transformer = self.build_transformer()
        x_train_processed = transformer.fit_transform(split_data.x_train)
        x_test_processed = transformer.transform(split_data.x_test)
        return transformer, x_train_processed, x_test_processed

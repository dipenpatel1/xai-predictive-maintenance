import pandas as pd

from predictive_maintenance.config import (
    CATEGORICAL_FEATURES,
    DEFAULT_DATASET_PATH,
    DROP_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def load_ai4i_dataset(path=DEFAULT_DATASET_PATH):
    """Load the AI4I dataset from CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download the AI4I 2020 CSV and place it at data/raw/ai4i2020.csv."
        )

    data = pd.read_csv(path)
    validate_columns(data)
    return data


def validate_columns(data):
    """Fail early if the expected AI4I columns are not present."""
    required_columns = (
        DROP_COLUMNS + CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET_COLUMN]
    )
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        raise ValueError(
            "Dataset is missing expected columns: " + ", ".join(missing_columns)
        )


def split_features_and_target(data):
    """Return feature matrix X and target vector y."""
    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    x = data[feature_columns].copy()
    y = data[TARGET_COLUMN].copy()
    return x, y


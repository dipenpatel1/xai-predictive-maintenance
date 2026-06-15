from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_DATASET_PATH = DATA_RAW_DIR / "ai4i2020.csv"
RANDOM_STATE = 42

TARGET_COLUMN = "Machine failure"
DROP_COLUMNS = ["UDI", "Product ID"]

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_FEATURES = ["Type"]


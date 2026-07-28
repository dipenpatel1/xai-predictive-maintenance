from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_DATASET_PATH = DATA_RAW_DIR / "ai4i2020.csv"
RANDOM_STATE = 42

TARGET_COLUMN = "Machine failure"
ID_COLUMNS = ["UID", "Product ID"]
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
DROP_COLUMNS = ID_COLUMNS

NUMERIC_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

CATEGORICAL_FEATURES = ["Type"]
INPUT_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

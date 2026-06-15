from ucimlrepo import fetch_ucirepo

from predictive_maintenance.config import DEFAULT_DATASET_PATH


def main():
    dataset = fetch_ucirepo(id=601)
    data = dataset.data.original

    DEFAULT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(DEFAULT_DATASET_PATH, index=False)

    print(f"Saved AI4I 2020 dataset to {DEFAULT_DATASET_PATH}")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")


if __name__ == "__main__":
    main()

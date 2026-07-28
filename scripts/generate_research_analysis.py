from pathlib import Path

import pandas as pd


ARTIFACT_DIR = Path("outputs/report_artifacts")
RESEARCH_DIR = Path("outputs/research_analysis")


def assign_error_type(row):
    actual = row["actual_machine_failure"]
    predicted = row["predicted_machine_failure"]

    if actual == 1 and predicted == 1:
        return "true_positive"
    if actual == 0 and predicted == 0:
        return "true_negative"
    if actual == 0 and predicted == 1:
        return "false_positive"
    if actual == 1 and predicted == 0:
        return "false_negative"

    return "unknown"


def main():
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(ARTIFACT_DIR / "classification_predictions.csv")
    anomaly_results = pd.read_csv("outputs/anomaly_detection_test_results.csv")

    predictions["error_type"] = predictions.apply(assign_error_type, axis=1)

    error_counts = (
        predictions["error_type"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="count")
    )
    error_counts.to_csv(RESEARCH_DIR / "classification_error_type_counts.csv", index=False)

    numeric_features = [
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
    ]

    error_group_summary = (
        predictions.groupby("error_type")[numeric_features + ["predicted_failure_probability"]]
        .mean()
        .round(3)
    )
    error_group_summary.to_csv(RESEARCH_DIR / "classification_error_group_sensor_means.csv")

    for error_type in [
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    ]:
        subset = predictions[predictions["error_type"] == error_type].copy()
        if error_type in ["true_positive", "false_positive"]:
            subset = subset.sort_values("predicted_failure_probability", ascending=False)
        else:
            subset = subset.sort_values("predicted_failure_probability", ascending=True)

        subset.head(20).to_csv(
            RESEARCH_DIR / f"classification_examples_{error_type}.csv",
            index=False,
        )

    comparison = predictions.copy()
    comparison["anomaly_flag"] = anomaly_results["anomaly_flag"].values
    comparison["anomaly_score"] = anomaly_results["anomaly_score"].values
    comparison["signal_group"] = (
        "anomaly_"
        + comparison["anomaly_flag"].astype(str)
        + "_classification_"
        + comparison["predicted_machine_failure"].astype(str)
    )

    signal_counts = (
        comparison["signal_group"]
        .value_counts()
        .rename_axis("signal_group")
        .reset_index(name="count")
    )
    signal_counts.to_csv(RESEARCH_DIR / "anomaly_classification_signal_counts.csv", index=False)

    signal_summary = (
        comparison.groupby("signal_group")[
            numeric_features
            + [
                "actual_machine_failure",
                "predicted_failure_probability",
                "anomaly_score",
            ]
        ]
        .mean()
        .round(3)
    )
    signal_summary.to_csv(RESEARCH_DIR / "anomaly_classification_signal_summary.csv")

    comparison.to_csv(RESEARCH_DIR / "combined_anomaly_classification_results.csv", index=False)

    print(f"Research analysis outputs saved to {RESEARCH_DIR}")
    for path in sorted(RESEARCH_DIR.iterdir()):
        print(f"- {path}")


if __name__ == "__main__":
    main()

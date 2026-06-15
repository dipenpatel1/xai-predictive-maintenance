from sklearn.model_selection import train_test_split

from predictive_maintenance.config import OUTPUTS_DIR, RANDOM_STATE
from predictive_maintenance.data_loader import load_ai4i_dataset, split_features_and_target
from predictive_maintenance.explainability import (
    explain_decision_tree_global,
    explain_logistic_regression_global,
)
from predictive_maintenance.feedback import (
    add_feedback_to_training_data,
    select_cases_for_review,
)
from predictive_maintenance.modeling import (
    build_decision_tree_pipeline,
    build_logistic_regression_pipeline,
    evaluate_classifier,
    metrics_to_frame,
    predict_anomalies,
    train_anomaly_detector,
)


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_ai4i_dataset()
    x, y = split_features_and_target(data)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    anomaly_preprocessor, anomaly_detector = train_anomaly_detector(x_train)
    anomaly_flags = predict_anomalies(anomaly_preprocessor, anomaly_detector, x_test)

    decision_tree = build_decision_tree_pipeline()
    logistic_regression = build_logistic_regression_pipeline()

    decision_tree.fit(x_train, y_train)
    logistic_regression.fit(x_train, y_train)

    reviewed_cases = select_cases_for_review(
        x_test=x_test,
        y_test=y_test,
        classifier=decision_tree,
        anomaly_flags=anomaly_flags,
    )

    updated_x_train, updated_y_train = add_feedback_to_training_data(
        x_train=x_train,
        y_train=y_train,
        reviewed_cases=reviewed_cases,
    )

    decision_tree_after_feedback = build_decision_tree_pipeline()
    decision_tree_after_feedback.fit(updated_x_train, updated_y_train)

    metrics = {
        "decision_tree_before_feedback": evaluate_classifier(decision_tree, x_test, y_test),
        "decision_tree_after_feedback": evaluate_classifier(
            decision_tree_after_feedback, x_test, y_test
        ),
        "logistic_regression_baseline": evaluate_classifier(
            logistic_regression, x_test, y_test
        ),
    }

    metrics_frame = metrics_to_frame(metrics)
    tree_explanation = explain_decision_tree_global(decision_tree)
    logistic_explanation = explain_logistic_regression_global(logistic_regression)

    metrics_frame.to_csv(OUTPUTS_DIR / "classification_metrics.csv", index=False)
    reviewed_cases.to_csv(OUTPUTS_DIR / "simulated_human_review_cases.csv", index=False)
    tree_explanation.to_csv(OUTPUTS_DIR / "decision_tree_feature_importance.csv", index=False)
    logistic_explanation.to_csv(
        OUTPUTS_DIR / "logistic_regression_coefficients.csv", index=False
    )

    print("Baseline pipeline completed.")
    print()
    print("Classification metrics:")
    print(metrics_frame.to_string(index=False))
    print()
    print(f"Simulated human review cases selected: {len(reviewed_cases)}")
    print()
    print("Top decision tree explanation features:")
    print(tree_explanation.to_string(index=False))


if __name__ == "__main__":
    main()

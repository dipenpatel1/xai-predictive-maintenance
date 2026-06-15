import pandas as pd


def select_cases_for_review(
    x_test,
    y_test,
    classifier,
    anomaly_flags,
    max_cases=50,
    low_confidence_threshold=0.65,
):
    """
    Simulate the routing step for human review.

    A record is reviewed when it is anomalous, predicted as failure,
    or the classifier is not confident.
    """
    probabilities = classifier.predict_proba(x_test)
    predicted_labels = classifier.predict(x_test)
    confidence = probabilities.max(axis=1)

    review_mask = (
        (anomaly_flags == 1)
        | (predicted_labels == 1)
        | (confidence < low_confidence_threshold)
    )

    review_cases = x_test.loc[review_mask].copy()
    review_cases["true_label_from_simulated_expert"] = y_test.loc[review_mask].values
    review_cases["model_prediction"] = predicted_labels[review_mask]
    review_cases["model_confidence"] = confidence[review_mask]
    review_cases["anomaly_flag"] = anomaly_flags[review_mask]

    return review_cases.head(max_cases)


def add_feedback_to_training_data(x_train, y_train, reviewed_cases):
    """Add simulated expert corrections back into the training set."""
    if reviewed_cases.empty:
        return x_train.copy(), y_train.copy()

    corrected_y = reviewed_cases["true_label_from_simulated_expert"]
    corrected_x = reviewed_cases.drop(
        columns=[
            "true_label_from_simulated_expert",
            "model_prediction",
            "model_confidence",
            "anomaly_flag",
        ]
    )

    updated_x_train = pd.concat([x_train, corrected_x], axis=0)
    updated_y_train = pd.concat([y_train, corrected_y], axis=0)

    return updated_x_train, updated_y_train


import numpy as np
import pandas as pd


def _to_dense(matrix):
    """Convert sparse sklearn output to a dense array when an XAI library needs it."""
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def get_feature_names(preprocessor):
    """Return transformed feature names from a fitted ColumnTransformer."""
    return list(preprocessor.get_feature_names_out())


def get_transformed_train_test(model, split_data):
    """Return transformed train/test matrices and transformed feature names."""
    preprocessor = model.named_steps["preprocessor"]
    x_train_transformed = preprocessor.transform(split_data.x_train)
    x_test_transformed = preprocessor.transform(split_data.x_test)

    return {
        "x_train": _to_dense(x_train_transformed),
        "x_test": _to_dense(x_test_transformed),
        "feature_names": get_feature_names(preprocessor),
    }


def explain_decision_tree_global(model, top_n=10):
    """Explain the decision tree using global feature importance."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = get_feature_names(preprocessor)
    importances = classifier.feature_importances_

    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    return explanation.head(top_n)


def explain_logistic_regression_global(model, top_n=10):
    """Explain logistic regression using largest absolute coefficients."""
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = get_feature_names(preprocessor)
    coefficients = classifier.coef_[0]

    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)

    return explanation.head(top_n)


def shap_global_importance(model, split_data, top_n=None):
    """Compute global SHAP importance for the failure class.

    This is most suitable for tree-based models such as DecisionTreeClassifier.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Install project requirements with `pip install -r requirements.txt`."
        ) from exc

    transformed = get_transformed_train_test(model, split_data)
    classifier = model.named_steps["classifier"]

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed["x_test"])

    if isinstance(shap_values, list):
        failure_shap_values = shap_values[1]
    elif np.asarray(shap_values).ndim == 3:
        failure_shap_values = shap_values[:, :, 1]
    else:
        failure_shap_values = shap_values

    importance = pd.DataFrame(
        {
            "feature": transformed["feature_names"],
            "mean_absolute_shap_value": np.abs(failure_shap_values).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap_value", ascending=False)

    if top_n is not None:
        return importance.head(top_n)
    return importance


def shap_local_explanation(model, split_data, row_position, top_n=8):
    """Explain one test-set prediction with SHAP values for the failure class."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Install project requirements with `pip install -r requirements.txt`."
        ) from exc

    transformed = get_transformed_train_test(model, split_data)
    classifier = model.named_steps["classifier"]
    x_test = transformed["x_test"]

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(x_test)

    if isinstance(shap_values, list):
        failure_shap_values = shap_values[1]
    elif np.asarray(shap_values).ndim == 3:
        failure_shap_values = shap_values[:, :, 1]
    else:
        failure_shap_values = shap_values

    feature_values = x_test[row_position]
    local_values = failure_shap_values[row_position]

    explanation = pd.DataFrame(
        {
            "feature": transformed["feature_names"],
            "transformed_feature_value": feature_values,
            "shap_value_for_failure": local_values,
            "absolute_shap_value": np.abs(local_values),
        }
    ).sort_values("absolute_shap_value", ascending=False)

    return explanation.head(top_n)


def lime_local_explanation(model, split_data, row_position, top_n=8):
    """Explain one test-set prediction with LIME."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise ImportError(
            "LIME is not installed. Install project requirements with `pip install -r requirements.txt`."
        ) from exc

    transformed = get_transformed_train_test(model, split_data)
    classifier = model.named_steps["classifier"]

    explainer = LimeTabularExplainer(
        training_data=transformed["x_train"],
        feature_names=transformed["feature_names"],
        class_names=["No failure", "Failure"],
        mode="classification",
        random_state=42,
    )

    explanation = explainer.explain_instance(
        data_row=transformed["x_test"][row_position],
        predict_fn=classifier.predict_proba,
        num_features=top_n,
        labels=[1],
    )

    return pd.DataFrame(
        explanation.as_list(label=1),
        columns=["lime_condition", "lime_weight_for_failure"],
    )

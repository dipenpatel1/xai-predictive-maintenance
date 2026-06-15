import pandas as pd


def get_feature_names(preprocessor):
    """Return transformed feature names from a fitted ColumnTransformer."""
    return list(preprocessor.get_feature_names_out())


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


# Explainable AI Implementation Guide

## Why We Need Explainability

The classification model predicts whether a machine record is likely to represent failure or no failure. However, a prediction alone is not enough for a Responsible AI project. In predictive maintenance, a maintenance engineer or decision maker should also understand which sensor readings influenced the model.

Explainable AI helps answer questions such as:

- Which features usually influence the model most?
- Why was this specific record predicted as failure?
- Why did the model create a false alarm?
- Why did the model miss a failure?

## SHAP

SHAP explains model predictions by assigning each feature a contribution value. A positive SHAP value pushes the prediction toward the failure class, while a negative SHAP value pushes it away from the failure class.

In this project, SHAP is useful for both global and local explanation:

- Global SHAP importance shows which features influence the model most across the test set.
- Local SHAP explanation shows why one selected machine record was predicted as failure or no failure.

## LIME

LIME explains one prediction at a time. It slightly changes the input record many times, observes how the model prediction changes, and then fits a simple local explanation model around that one record.

In this project, LIME is useful because it can produce a human-readable explanation for individual cases such as true positives, false positives, and false negatives.

## How This Fits the Project

The project now has three model interpretation levels:

1. Basic decision tree feature importance explains the model globally.
2. SHAP explains both global feature influence and individual predictions.
3. LIME explains individual predictions using a local surrogate model.

This supports the Responsible AI goals of transparency, interpretability, accountability, and human-centered decision support.

## Files Added

- `src/07_explainable_ai/lime_shap_explainability.ipynb`
- `outputs/explainable_ai/shap_global_importance.csv`
- `outputs/explainable_ai/shap_local_*.csv`
- `outputs/explainable_ai/lime_local_*.csv`
- `outputs/explainable_ai/xai_local_example_index.csv`
- `outputs/explainable_ai/xai_case_comparison.csv`
- `outputs/explainable_ai/figures/*.png`

The notebook is organized as a learning exercise. It trains the existing
classifier, labels test predictions as true positive, false positive, false
negative, or true negative, and then compares SHAP and LIME explanations for
one example of each outcome.

## Suggested Report Interpretation

The SHAP and LIME explanations should be compared with the earlier feature relationship analysis. If features such as torque, rotational speed, tool wear, or engineered interaction features appear repeatedly in the explanations, this strengthens the interpretation that the model is learning meaningful machine behavior patterns rather than arbitrary noise.

Explanations describe the model's learned behavior; they do not prove that a
sensor feature physically caused a machine failure. Differences between SHAP
and LIME should also be reported because the two methods use different
explanation strategies.

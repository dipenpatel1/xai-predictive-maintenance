# Human-in-the-Loop Anomaly Detection and Explainable Failure Classification

This repository is being built for a Responsible and Explainable AI course project.

The project goal is to develop a small predictive maintenance prototype that:

- detects abnormal machine behavior using anomaly detection
- classifies machine records as failure or no failure
- explains model outcomes using interpretable feature information
- routes uncertain, anomalous, or high-risk cases to simulated human review
- retrains the classifier after corrected human feedback
- demonstrates Responsible AI principles such as transparency, accountability, robustness, and human oversight

## Project Structure

```text
data/
  raw/                  # Place the AI4I 2020 CSV here
  processed/            # Reserved for processed data artifacts
docs/
  project_proposal.md
notes/
  linear_regression_from_scratch.md
outputs/                # Pipeline reports are written here
src/
  predictive_maintenance/
    config.py
    data_loader.py
    explainability.py
    feedback.py
    modeling.py
  linear_regression_from_scratch.py
  run_baseline_pipeline.py
```

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the AI4I 2020 Predictive Maintenance Dataset and place the CSV at:

```text
data/raw/ai4i2020.csv
```

The expected target column is:

```text
Machine failure
```

## Run the Baseline Project Pipeline

```bash
PYTHONPATH=src python3 src/run_baseline_pipeline.py
```

The pipeline will:

- load and preprocess the AI4I dataset
- train an Isolation Forest anomaly detector
- train a Decision Tree classifier
- train a Logistic Regression baseline
- explain global model behavior
- simulate human review for selected cases
- retrain the Decision Tree after feedback
- write outputs to `outputs/`

Generated outputs:

```text
outputs/classification_metrics.csv
outputs/simulated_human_review_cases.csv
outputs/decision_tree_feature_importance.csv
outputs/logistic_regression_coefficients.csv
```

## Learning Track

Before building the full predictive maintenance pipeline, we are implementing core ML ideas from first principles.

Current module:

- `src/linear_regression_from_scratch.py`

This script implements simple linear regression using pure Python:

- prediction: `y_hat = wx + b`
- residuals/errors
- mean squared error
- gradients with respect to `w` and `b`
- gradient descent
- convergence history

Run it with:

```bash
python3 src/linear_regression_from_scratch.py
```
# predictive_maintenance_with_explainable_ai

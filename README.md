# Explainable Predictive Maintenance

This repository is being built for a Responsible and Explainable AI course project.

The project goal is to develop a small predictive maintenance prototype that:

- detects abnormal machine behavior using anomaly detection
- classifies machine records as failure or no failure
- compares interpretable decision trees with a small neural network
- studies correlations and engineered sensor relationships
- explains model outcomes using feature importance, SHAP, and LIME
- evaluates explanation faithfulness and stability
- demonstrates transparency, reliability, error tradeoffs, and appropriate-use limits

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
    anomaly.py
    classification.py
    config.py
    data_loader.py
    explainability.py
    neural_network.py
    preprocessing.py
  01_data_exploration/
  02_data_preprocessing/
  03_anomaly_detection/
  04_failure_classification/
  05_neural_network/
  06_feature_relationship_analysis/
  07_explainable_ai/
  08_integrated_research_analysis/
  linear_regression_from_scratch.py
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

## Run the Project

Run notebooks `01` through `08` in order. Each notebook explains its theory,
implementation, executed findings, and report-ready artifacts. See
`docs/reproducibility_guide.md` for the output-generation workflow.

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

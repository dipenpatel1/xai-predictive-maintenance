# Reusable Code Guide

The notebooks are useful for learning and reporting, but reusable project logic now lives in Python classes under:

```text
src/predictive_maintenance/
```

This makes future experiments easier because preprocessing, anomaly detection, classification, and explanation logic can be reused without copy-pasting notebook cells.

## 1. Dataset Exploration

Class:

```python
from predictive_maintenance.exploration import DatasetExplorer
```

Example:

```python
from predictive_maintenance.data_loader import load_ai4i_dataset
from predictive_maintenance.exploration import DatasetExplorer

data = load_ai4i_dataset()
explorer = DatasetExplorer(data)

overview = explorer.overview()
target_balance = explorer.target_balance()
sensor_summary = explorer.sensor_summary()
normal_vs_failure = explorer.normal_vs_failure_means()
```

Use this for:

- dataset shape
- missing values
- target balance
- sensor summaries
- normal vs failure comparisons

## 2. Data Preprocessing

Class:

```python
from predictive_maintenance.preprocessing import MaintenancePreprocessor
```

Example:

```python
from predictive_maintenance.data_loader import load_ai4i_dataset
from predictive_maintenance.preprocessing import MaintenancePreprocessor

data = load_ai4i_dataset()
preprocessor = MaintenancePreprocessor()

split_data = preprocessor.train_test_split(data)
transformer, x_train_processed, x_test_processed = (
    preprocessor.fit_transform_train_test(split_data)
)
```

Use this for:

- selecting safe input features
- separating target labels
- preserving failure-mode labels for evaluation only
- train/test splitting
- scaling numeric features
- encoding categorical features

Important:

The reusable preprocessor excludes failure-mode columns from model inputs to avoid data leakage.

## 3. Anomaly Detection

Class:

```python
from predictive_maintenance.anomaly import AnomalyDetectionExperiment
```

Example:

```python
from predictive_maintenance.data_loader import load_ai4i_dataset
from predictive_maintenance.anomaly import AnomalyDetectionExperiment

data = load_ai4i_dataset()

experiment = AnomalyDetectionExperiment(contamination=0.05)
experiment.fit(data)

summary = experiment.summary_metrics()
results = experiment.results_frame()
top_anomalies = experiment.top_anomalies(n=10)
explanations = experiment.top_anomaly_explanations(n=10)
```

Use this for:

- training anomaly detection on normal behavior
- generating anomaly flags
- generating anomaly scores
- finding the most anomalous records
- explaining anomalies using feature deviations from normal behavior

## 4. Failure Classification

Class:

```python
from predictive_maintenance.classification import FailureClassificationExperiment
```

Example:

```python
from predictive_maintenance.data_loader import load_ai4i_dataset
from predictive_maintenance.classification import FailureClassificationExperiment

data = load_ai4i_dataset()

experiment = FailureClassificationExperiment()
experiment.fit(data)

metrics = experiment.metrics_frame(threshold=0.5)
feature_importance = experiment.feature_importance()
predictions = experiment.predictions_frame(threshold=0.5)
threshold_table = experiment.threshold_table([0.3, 0.5, 0.7])
```

Use this for:

- training a failure classifier
- evaluating precision, recall, F1, and confusion matrix
- changing classification thresholds
- generating prediction probabilities
- extracting feature importance

## 5. Neural Network Comparison

Class:

```python
from predictive_maintenance.neural_network import NeuralNetworkExperiment
```

Example:

```python
from predictive_maintenance.data_loader import load_ai4i_dataset
from predictive_maintenance.neural_network import NeuralNetworkExperiment

data = load_ai4i_dataset()

experiment = NeuralNetworkExperiment(hidden_layer_sizes=(16, 8)).fit(data)
default_metrics = experiment.metrics_frame(threshold=0.5)
selected = experiment.select_threshold_from_training(folds=5)
selected_metrics = experiment.metrics_frame(
    threshold=experiment.selected_threshold
)
training_history = experiment.training_history()
```

Use this for:

- training a small weighted feedforward neural network
- inspecting training and validation history
- evaluating PR-AUC and ROC-AUC as well as threshold metrics
- selecting an alternative threshold from training folds only
- comparing model complexity with the decision tree

## 6. Why This Refactor Matters

This structure supports research-style iteration:

```text
load data
    ->
explore data
    ->
preprocess safely
    ->
run anomaly experiment
    ->
run classification experiment
    ->
compare and explain results
    ->
tune model settings
```

Instead of rewriting the same code in each notebook, future experiments can import the reusable classes and focus on research questions.

## 7. Example End-to-End Usage

Run this from the project root:

```bash
PYTHONPATH=src .venv/bin/python
```

Then:

```python
from predictive_maintenance.data_loader import load_ai4i_dataset
from predictive_maintenance.anomaly import AnomalyDetectionExperiment
from predictive_maintenance.classification import FailureClassificationExperiment

data = load_ai4i_dataset()

anomaly_experiment = AnomalyDetectionExperiment(contamination=0.05).fit(data)
print(anomaly_experiment.summary_metrics())

classification_experiment = FailureClassificationExperiment().fit(data)
print(classification_experiment.metrics_frame())
print(classification_experiment.threshold_table([0.3, 0.5, 0.7]))
```

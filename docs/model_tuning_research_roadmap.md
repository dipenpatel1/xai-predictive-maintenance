# Model Tuning Research Roadmap

## Purpose

This note explains how to move beyond basic model training and begin tuning the predictive maintenance system like an AI researcher. The goal is not only to improve performance, but to understand the tradeoffs between failure detection, false alarms, explainability, and responsible use.

In this project, a more productive model means:

```text
catch most real failures
while reducing unnecessary false alarms
```

The current failure classification model has:

| Metric | Value |
|---|---:|
| Precision | 0.3058 |
| Recall | 0.9265 |

This means the model catches most actual failures, but many predicted failures are false alarms.

## 1. Tune the Classification Threshold

Most classifiers use a default threshold:

```text
if failure_probability >= 0.5:
    predict failure
else:
    predict no failure
```

This threshold can be changed.

If the threshold is increased:

```text
fewer records are predicted as failure
precision usually increases
recall usually decreases
```

If the threshold is decreased:

```text
more records are predicted as failure
recall usually increases
precision usually decreases
```

Research question:

```text
What threshold gives a better balance between catching failures and reducing false alarms?
```

Recommended experiment:

Create a table like:

| Threshold | Precision | Recall | F1-score | False Positives | False Negatives |
|---:|---:|---:|---:|---:|---:|
| 0.30 |  |  |  |  |  |
| 0.40 |  |  |  |  |  |
| 0.50 |  |  |  |  |  |
| 0.60 |  |  |  |  |  |
| 0.70 |  |  |  |  |  |
| 0.80 |  |  |  |  |  |

Why this matters:

Threshold tuning directly connects model behavior to real-world risk. A safety-focused system may prefer high recall, while a cost-sensitive system may prefer fewer false alarms.

## 2. Tune Decision Tree Depth

The current classifier uses:

```python
max_depth=4
```

This controls how complex the decision tree can become.

Possible values to test:

```python
max_depth = 2, 3, 4, 5, 6, 8, None
```

Smaller tree:

- simpler
- easier to explain
- may underfit

Deeper tree:

- can learn more complex patterns
- may improve performance
- may overfit
- becomes harder to explain

Research question:

```text
How much model complexity improves performance before the model becomes too complex or less explainable?
```

Why this matters:

This experiment studies the tradeoff between performance and interpretability.

## 3. Tune Class Weight

The current classifier uses:

```python
class_weight="balanced"
```

This makes the model pay more attention to the minority failure class.

Manual class weights can also be tested:

```python
class_weight={0: 1, 1: 3}
class_weight={0: 1, 1: 5}
class_weight={0: 1, 1: 10}
class_weight={0: 1, 1: 20}
```

Higher failure weight:

```text
more sensitive to failures
recall may increase
false alarms may increase
```

Lower failure weight:

```text
fewer false alarms
but more failures may be missed
```

Research question:

```text
What class weight gives the best safety-cost tradeoff?
```

Why this matters:

Class weighting changes how the model treats rare but important failure cases.

## 4. Compare Another Simple Model

A useful research project should not rely on only one model without comparison.

Possible comparison models:

- Decision Tree: main explainable model
- Logistic Regression: simple baseline
- Random Forest: stronger but less directly interpretable model

Important question:

```text
Does a better-performing model become harder to explain?
```

Research question:

```text
What is the tradeoff between predictive performance and interpretability?
```

Why this matters:

Responsible and Explainable AI is not only about getting the highest score. It is also about whether the model can be understood and trusted.

## 5. Tune Anomaly Detection Contamination

The anomaly detection model uses a contamination parameter, such as:

```python
contamination=0.05
```

This tells the model approximately what proportion of records should be treated as anomalous.

Possible values to test:

```python
contamination = 0.02, 0.03, 0.05, 0.08, 0.10
```

Lower contamination:

```text
fewer records flagged as anomalous
may reduce false alarms
may miss abnormal failure-related behavior
```

Higher contamination:

```text
more records flagged as anomalous
may catch more abnormal behavior
may create more false alarms
```

Research question:

```text
What anomaly rate best captures failure-related abnormal behavior?
```

Why this matters:

This helps tune anomaly detection as an early-warning system.

## 6. Combine Anomaly Detection and Classification Signals

The project already found that anomaly detection and classification provide different signals:

- anomaly detection identifies unusual behavior
- classification predicts known failure patterns

These can be combined into risk categories:

| Anomaly Detection | Classification | Suggested Risk Level |
|---:|---:|---|
| 0 | 0 | Low risk |
| 1 | 0 | Monitor |
| 0 | 1 | Warning |
| 1 | 1 | High risk |

Interpretation:

- `Low risk`: normal behavior and no predicted failure
- `Monitor`: unusual behavior but not predicted failure
- `Warning`: predicted failure but not anomalous
- `High risk`: both unusual and predicted failure

Research question:

```text
Can combining anomaly detection and classification produce more meaningful risk levels?
```

Why this matters:

This is more useful than a single binary prediction. It also supports Responsible AI because it communicates uncertainty and risk level more clearly.

## 7. Recommended Implementation Order

The recommended next steps are:

1. Threshold tuning
2. Decision tree depth tuning
3. Class weight tuning
4. Anomaly contamination tuning
5. Combined risk category design
6. Local explanation examples

Start with threshold tuning because it directly addresses the current issue:

```text
high recall but low precision
```

## 8. Expected Research Contribution

These tuning experiments can make the project more meaningful by showing:

- how model performance changes under different settings
- how false alarms and missed failures trade off
- how model complexity affects explainability
- how anomaly detection and classification can work together
- how Responsible AI requires interpreting errors, uncertainty, and risk

The final project should not simply say:

```text
The model achieved 92.6% accuracy.
```

It should say:

```text
The system was evaluated under different risk tradeoffs. The classifier achieved high recall but low precision, so threshold tuning and combined anomaly-classification risk categories were explored to make the model output more useful and responsible for predictive maintenance decision support.
```


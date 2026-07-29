# Presentation Outline

## Slide 1: Title

**Explainable Predictive Maintenance Using Anomaly Detection, Failure Classification, and Explanation Evaluation**

Presenter: Bijay Pandey  
Course: Responsible and Explainable AI

Speaker note:

This project studies how machine learning can support predictive maintenance while making model behavior understandable and responsible.

## Slide 2: Problem Motivation

Main points:

- Industrial machine failures can cause downtime, cost, and safety concerns.
- Predictive maintenance uses sensor data to identify abnormal behavior and possible failure.
- A useful system should not only predict, but also explain why a machine is risky.

The project asks more than:
Can a model predict machine failure?

It asks:
How are anomalies different from failures?
Which features influence predictions?
Where does the classifier fail?
Are explanations faithful and stable?
How do error consequences affect model selection?

Speaker note:

The focus is not just model accuracy. The project asks how to make predictive maintenance outputs understandable and useful for decision support.

## Slide 3: Dataset

Dataset: AI4I 2020 Predictive Maintenance Dataset

Key details:

- 10,000 machine records
- Failure rate: about 3.39%
- Input features:
  - machine type
  - air temperature
  - process temperature
  - rotational speed
  - torque
  - tool wear

Important data decision:

- Failure-mode columns were excluded from inputs to avoid data leakage.

Speaker note:

The dataset is imbalanced, which means accuracy alone can be misleading. I avoided using failure-mode labels as input features because that would leak the answer into the model.

## Slide 4: Project Workflow

Show workflow:

```text
Data Exploration
      ->
Preprocessing
      ->
Anomaly Detection
      ->
Failure Classification
      ->
Feature Relationship Analysis
      ->
SHAP and LIME Explanation
      ->
Validation and Responsible AI Interpretation
      ->
Neural Network Comparison
```

Speaker note:

The work was developed in stages. First I studied abnormal behavior, then supervised failure prediction, then feature relationships and model tradeoffs.

## Slide 5: Anomaly Detection

Question:

```text
Does this machine behavior look unusual?
```

Results:

- Test records: 2000
- Records flagged as anomalous: 113
- Actual failures in test set: 68
- Failures flagged as anomalous: 16

Recommended visual:

```text
outputs/report_artifacts/anomaly_score_distribution.png
```

Speaker note:

Anomaly detection identified some failure-related abnormal behavior, but it did not catch all failures. This shows anomaly detection is useful for abnormal-behavior detection, not as a complete failure classifier.

## Slide 6: Anomaly and Classification Are Different

Same-record comparison:

| Signal group | Records | Failures | Failure rate |
|---|---:|---:|---:|
| Neither | 1716 | 4 | 0.23% |
| Classifier only | 171 | 48 | 28.07% |
| Anomaly only | 78 | 1 | 1.28% |
| Both | 35 | 15 | 42.86% |

Recommended visual:

```text
outputs/integrated_research_analysis/figures/signal_agreement_matrix.png
```

Speaker note:

The methods answer different questions. Agreement identifies a group with a higher observed failure rate, but many known failures are detected only by the classifier. Anomaly does not mean failure, and no flag does not guarantee safety.

## Slide 7: Failure Classification

Question:

```text
Does this record indicate failure or no failure?
```

Results:

| Metric | Value |
|---|---:|
| Accuracy | 0.9260 |
| Precision | 0.3058 |
| Recall | 0.9265 |
| F1-score | 0.4599 |

Recommended visual:

```text
outputs/report_artifacts/classification_confusion_matrix.png
```

Speaker note:

The classifier caught most actual failures, but it produced many false alarms. This creates a useful Responsible AI discussion about recall versus precision.

## Slide 8: SHAP and LIME Explanations

Global SHAP ranking:

1. Rotational speed
2. Torque
3. Tool wear

Local cases studied:

- true positive
- false positive
- false negative
- true negative

Recommended visual:

```text
outputs/explainable_ai/figures/shap_global_importance.png
```

Speaker note:

SHAP explains how each feature moves a prediction away from the model baseline. LIME fits an approximate local model. The methods were compared rather than treated as interchangeable.

## Slide 9: Feature Relationships

Study relationships between features, not only individual features.

Engineered features:

- temperature difference
- power proxy
- wear-torque interaction
- speed-torque ratio

Recommended visual:

```text
outputs/feature_relationships/feature_correlation_matrix.png
```

Speaker note:

This step asks whether machine failure is related to combinations of features. For example, torque and rotational speed together approximate a power-like relationship.

```text
Temperature difference = Process temperature - Air temperature
Power proxy = Torque x Rotational speed
Wear-load interaction = Tool wear x Torque
Speed-load ratio = Rotational speed / (Torque + epsilon)
```

These are interpretable candidate relationships, not discovered physical equations. The speed-torque ratio had only 0.063 correlation with failure but became the engineered tree's most-used feature, demonstrating that low linear correlation does not rule out useful nonlinear thresholds.

## Slide 10: Five-Fold Feature Results

Model comparison:

| Model | Mean precision | Mean recall | Mean F1 |
|---|---:|---:|---:|
| Base features | 0.2256 | 0.9204 | 0.3621 |
| Base + engineered | 0.3552 | 0.9203 | 0.5119 |

Recommended visual:

```text
outputs/integrated_research_analysis/figures/five_fold_model_comparison.png
```

Speaker note:

The engineered features improved mean precision and F1 while maintaining almost identical mean recall across five data splits. This is stronger evidence than reporting only the best holdout result.

## Slide 11: Neural Network Comparison

Five-fold mean results:

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Base tree | 0.2256 | 0.9204 | 0.3621 | 0.4584 | 0.9250 |
| Engineered tree | 0.3552 | 0.9203 | 0.5119 | 0.7885 | 0.9439 |
| Neural network | 0.2279 | 0.8525 | 0.3593 | 0.5648 | 0.9501 |

Recommended visual:

```text
outputs/neural_network_comparison/figures/model_precision_recall_comparison.png
```

Speaker note:

The neural network achieved the highest ROC-AUC, but the engineered tree remained stronger on precision, recall, F1, and PR-AUC. The additional complexity therefore did not justify replacing the more interpretable tree. The network remains useful as a nonlinear benchmark and post-hoc explanation case study.

## Slide 12: Responsible AI Tradeoff

Main tradeoff:

```text
Catch more failures
      vs
Reduce false alarms
```

Cost-sensitivity result:

- Missed-failure cost ratio 1-5: failure weight 5
- Ratio 6-14: threshold 0.8
- Ratio 15 or greater: high-recall threshold 0.5

Recommended visual:

```text
outputs/integrated_research_analysis/figures/cost_sensitivity.png
```

Speaker note:

There is no universally responsible threshold. The preferred setting changes with the assumed consequences of missed failures and false alarms. These ratios are sensitivity assumptions, not real industrial cost estimates.

## Slide 13: Conclusion and Next Steps

Conclusion:

- Anomaly detection identifies abnormal behavior.
- Classification detects known failures more effectively.
- Feature relationships improve interpretability.
- Engineered features improve five-fold F1.
- The neural network does not automatically outperform the interpretable tree.
- SHAP is faithful to the fitted tree.
- Selected LIME explanations have weak local surrogate fit.
- Model tuning reveals cost-dependent tradeoffs.
- The system should be used as decision support, not autonomous decision-making.

Future work:

- probability calibration
- testing on real industrial data
- explanation evaluation with maintenance experts
- temporal validation using machine histories
- broader neural-network architecture evaluation

Speaker note:

The main takeaway is that responsible predictive maintenance requires more than accuracy or a readable explanation. It requires error analysis, explanation evaluation, validation across splits, and clear communication of limitations.

## Optional Backup Slide: Key Numbers

Anomaly detection:

- 113 anomalies flagged
- 16 of 68 failures flagged

Classification:

- 63 of 68 failures detected
- 143 false alarms

Engineered model:

- false alarms reduced from 143 to 111
- missed failures increased from 5 to 8

Class weight 5:

- false alarms reduced to 78
- detected 58 of 68 failures

Explanation audit:

- SHAP mean reconstruction error: approximately 3.11 x 10^-14
- LIME mean surrogate R2: 0.1115 for FP and 0.0970 for FN

Neural-network comparison:

- default threshold: precision 0.2711, recall 0.8971, F1 0.4164
- training-selected threshold 0.86: precision 0.5281, recall 0.6912, F1 0.5987
- selected threshold reduces false alarms but increases missed failures to 21

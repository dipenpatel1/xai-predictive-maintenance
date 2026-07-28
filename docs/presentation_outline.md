# Presentation Outline

## Slide 1: Title

**Explainable Predictive Maintenance Using Anomaly Detection and Failure Classification**

Presenter: Bijay Pandey  
Course: Responsible and Explainable AI

Speaker note:

This project studies how machine learning can support predictive maintenance while making model behavior understandable and responsible.

## Slide 2: Problem Motivation

Main points:

- Industrial machine failures can cause downtime, cost, and safety concerns.
- Predictive maintenance uses sensor data to identify abnormal behavior and possible failure.
- A useful system should not only predict, but also explain why a machine is risky.

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
Model Tuning and Responsible AI Interpretation
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

## Slide 6: Anomaly Explanation

Main finding:

- Top anomalies were often explained by:
  - high rotational speed
  - low torque
  - high tool wear
  - unusual temperature values

Recommended visual:

```text
outputs/report_artifacts/anomaly_speed_torque_scatter.png
```

Speaker note:

The explainability step compared anomalous records against normal machine behavior. This helped explain why a record looked unusual instead of only reporting an anomaly flag.

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

## Slide 8: Classification Explainability

Most important features:

1. Rotational speed
2. Torque
3. Tool wear

Recommended visual:

```text
outputs/report_artifacts/classification_feature_importance.png
```

Speaker note:

The classifier's explanations aligned with anomaly detection findings. Both methods pointed to rotational speed, torque, and tool wear as important.

## Slide 9: Feature Relationships

Professor-guided research step:

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

## Slide 10: Engineered Feature Results

Model comparison:

| Model | Precision | Recall | F1 | False Positives | False Negatives |
|---|---:|---:|---:|---:|---:|
| Base features | 0.3058 | 0.9265 | 0.4599 | 143 | 5 |
| Base + engineered | 0.3509 | 0.8824 | 0.5021 | 111 | 8 |

Recommended visual:

```text
outputs/feature_relationships/engineered_model_feature_importance.png
```

Speaker note:

Engineered features reduced false alarms and improved F1, but missed slightly more failures. This is a meaningful tradeoff.

## Slide 11: Responsible AI Tradeoff

Main tradeoff:

```text
Catch more failures
      vs
Reduce false alarms
```

Examples:

- Balanced model: catches 63 failures, 143 false alarms
- Class weight 5: catches 58 failures, 78 false alarms

Recommended visual:

```text
outputs/model_tuning/class_weight_tuning.png
```

Speaker note:

The model setting depends on the maintenance context. If failures are very costly, high recall is preferred. If inspections are expensive, fewer false alarms may be preferred.

## Slide 12: Conclusion and Next Steps

Conclusion:

- Anomaly detection identifies abnormal behavior.
- Classification detects known failures more effectively.
- Feature relationships improve interpretability.
- Model tuning reveals responsible tradeoffs.
- The system should be used as decision support, not autonomous decision-making.

Future work:

- local explanations
- cost-sensitive evaluation
- probability calibration
- testing on real industrial data
- combined risk scoring

Speaker note:

The main takeaway is that responsible predictive maintenance requires more than accuracy. It requires explanations, error analysis, and clear communication of tradeoffs.

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


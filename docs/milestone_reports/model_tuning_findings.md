# Model Tuning Findings

## 1. Objective

The goal of model tuning is to understand how model settings affect the tradeoff between detecting actual machine failures and reducing false alarms. This is important because predictive maintenance is a risk-sensitive domain: missing a real failure may be costly or unsafe, while too many false alarms may reduce trust and cause unnecessary maintenance work.

The tuning experiments focused on:

- classification threshold
- decision tree depth
- class weighting
- anomaly detection contamination

## 2. Threshold Tuning

The classifier produces a predicted probability of failure. A threshold converts that probability into a final class prediction:

```text
if predicted failure probability >= threshold:
    predict failure
else:
    predict no failure
```

The default threshold is usually 0.5, but changing the threshold changes the precision-recall tradeoff.

| Threshold | Accuracy | Precision | Recall | F1-score | False Positives | False Negatives | True Positives |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.9115 | 0.2700 | 0.9412 | 0.4197 | 173 | 4 | 64 |
| 0.4 | 0.9260 | 0.3058 | 0.9265 | 0.4599 | 143 | 5 | 63 |
| 0.7 | 0.9275 | 0.3103 | 0.9265 | 0.4649 | 140 | 5 | 63 |
| 0.8 | 0.9535 | 0.4126 | 0.8676 | 0.5592 | 84 | 9 | 59 |
| 0.9 | 0.9650 | 0.4889 | 0.6471 | 0.5570 | 46 | 24 | 44 |

Increasing the threshold reduced false positives and improved precision, but it also reduced recall at higher thresholds. For example, threshold 0.8 reduced false positives from 143 to 84, but increased missed failures from 5 to 9.

This shows a clear Responsible AI tradeoff:

```text
Lower threshold: catches more failures but creates more false alarms.
Higher threshold: reduces false alarms but misses more failures.
```

## 3. Decision Tree Depth Tuning

The decision tree depth controls model complexity. A shallow tree is easier to explain but may underfit. A deeper tree may improve performance but can become harder to interpret.

| Max Depth | Accuracy | Precision | Recall | F1-score | False Positives | False Negatives | True Positives |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.8240 | 0.1450 | 0.8529 | 0.2479 | 342 | 10 | 58 |
| 3 | 0.9080 | 0.2339 | 0.7500 | 0.3566 | 167 | 17 | 51 |
| 4 | 0.9260 | 0.3058 | 0.9265 | 0.4599 | 143 | 5 | 63 |
| 5 | 0.9300 | 0.3125 | 0.8824 | 0.4615 | 132 | 8 | 60 |
| 6 | 0.9365 | 0.3314 | 0.8529 | 0.4774 | 117 | 10 | 58 |
| 8 | 0.9535 | 0.4088 | 0.8235 | 0.5463 | 81 | 12 | 56 |
| No limit | 0.9790 | 0.7321 | 0.6029 | 0.6613 | 15 | 27 | 41 |

The unlimited-depth tree achieved high precision but much lower recall, missing 27 failures. This may not be acceptable for predictive maintenance if the goal is to catch most failures.

Depth 4 provides a strong safety-focused setting because it catches 63 of 68 failures while remaining relatively simple and interpretable.

## 4. Class Weight Tuning

Class weights control how much the model prioritizes the rare failure class.

| Class Weight | Accuracy | Precision | Recall | F1-score | False Positives | False Negatives | True Positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| None | 0.9710 | 0.8125 | 0.1912 | 0.3095 | 3 | 55 | 13 |
| Balanced | 0.9260 | 0.3058 | 0.9265 | 0.4599 | 143 | 5 | 63 |
| Failure weight 3 | 0.9655 | 0.4935 | 0.5588 | 0.5241 | 39 | 30 | 38 |
| Failure weight 5 | 0.9560 | 0.4265 | 0.8529 | 0.5686 | 78 | 10 | 58 |
| Failure weight 10 | 0.9540 | 0.4143 | 0.8529 | 0.5577 | 82 | 10 | 58 |
| Failure weight 20 | 0.9260 | 0.3058 | 0.9265 | 0.4599 | 143 | 5 | 63 |

Without class weighting, the model had high precision but detected only 13 of 68 failures. This is risky because the model missed 55 actual failures.

The balanced setting caught 63 of 68 failures but generated more false alarms. The manual failure weight of 5 achieved a more balanced result, catching 58 failures while reducing false positives from 143 to 78.

This suggests that `class_weight={0: 1, 1: 5}` may be a useful compromise setting.

## 5. Anomaly Contamination Tuning

The anomaly contamination parameter controls how many records are expected to be anomalous.

| Contamination | Accuracy | Precision | Recall | F1-score | False Positives | False Negatives | True Positives | Records Flagged |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.9535 | 0.2093 | 0.1324 | 0.1622 | 34 | 59 | 9 | 43 |
| 0.03 | 0.9430 | 0.1618 | 0.1618 | 0.1618 | 57 | 57 | 11 | 68 |
| 0.05 | 0.9225 | 0.1081 | 0.1765 | 0.1341 | 99 | 56 | 12 | 111 |
| 0.08 | 0.8935 | 0.0857 | 0.2206 | 0.1235 | 160 | 53 | 15 | 175 |
| 0.10 | 0.8735 | 0.0814 | 0.2647 | 0.1246 | 203 | 50 | 18 | 221 |

Increasing contamination flagged more records as anomalous and caught slightly more failures, but precision stayed low. This confirms that anomaly detection is useful for identifying unusual behavior, but it should not be treated as the main failure classifier.

## 6. Recommended Model Settings

For a safety-focused predictive maintenance setting, the original balanced depth-4 decision tree is useful because it detects most failures:

```text
max_depth = 4
class_weight = "balanced"
threshold = 0.5
```

This setting catches 63 of 68 failures but produces 143 false alarms.

For a more balanced setting that reduces false alarms while still catching many failures, the class weight 5 model is promising:

```text
max_depth = 4
class_weight = {0: 1, 1: 5}
```

This catches 58 of 68 failures and reduces false positives to 78.

## 7. Responsible and Explainable AI Interpretation

The tuning experiments show why model configuration is not only a technical choice but also a Responsible AI decision. Different settings create different consequences:

- high recall reduces missed failures
- high precision reduces unnecessary maintenance alerts
- shallow trees are easier to explain
- deeper trees may improve precision but can miss more failures
- anomaly detection helps identify unusual behavior but does not replace classification

The final system should clearly communicate this tradeoff rather than presenting one model score as universally best.

## 8. Generated Artifacts

The following tuning artifacts were generated:

```text
outputs/model_tuning/threshold_tuning.csv
outputs/model_tuning/threshold_tuning_precision_recall_f1.png
outputs/model_tuning/decision_tree_depth_tuning.csv
outputs/model_tuning/decision_tree_depth_tuning.png
outputs/model_tuning/class_weight_tuning.csv
outputs/model_tuning/class_weight_tuning.png
outputs/model_tuning/anomaly_contamination_tuning.csv
outputs/model_tuning/anomaly_contamination_tuning.png
```

## 9. Conclusion

Model tuning showed that there is no single best configuration without considering the maintenance context. If missed failures are more costly than false alarms, a high-recall model is preferred. If unnecessary maintenance is very expensive, a setting with fewer false positives may be preferred.

This tuning analysis strengthens the project because it connects machine learning performance to real-world risk tradeoffs, which is central to Responsible AI.


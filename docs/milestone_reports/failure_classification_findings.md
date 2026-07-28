# Failure Classification Findings

## 1. Objective

After completing anomaly detection, the next step was to train a supervised failure classification model. The purpose of this section is to predict whether a machine record represents `failure` or `no failure` using the safe machine condition features from the AI4I 2020 dataset.

This step answers a different question from anomaly detection:

- Anomaly detection asks: **Does this machine behavior look unusual?**
- Failure classification asks: **Does this machine record indicate failure or no failure?**

The classification model uses the `Machine failure` label during training, so this is a supervised learning task.

## 2. Input Features and Target

The model used the following input features:

- `Type`
- `Air temperature`
- `Process temperature`
- `Rotational speed`
- `Torque`
- `Tool wear`

The target variable was:

- `Machine failure`

The failure mode columns `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` were not used as input features because they contain failure-related label information. Excluding these columns prevents data leakage.

## 3. Model Evaluation Results

The model was evaluated on 2,000 test records. The overall classification metrics were:

| Metric | Value |
|---|---:|
| Accuracy | 0.9260 |
| Precision | 0.3058 |
| Recall | 0.9265 |
| F1-score | 0.4599 |

The detailed classification report showed:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No failure | 1.00 | 0.93 | 0.96 | 1932 |
| Failure | 0.31 | 0.93 | 0.46 | 68 |

The model achieved high overall accuracy and very high recall for the failure class. The failure-class recall was 0.9265, meaning the model detected most of the actual failure cases in the test set.

However, the precision for the failure class was only 0.3058. This means that many records predicted as failures were actually normal records. In predictive maintenance, this reflects an important tradeoff: the model is good at catching failures, but it also produces false alarms.

## 4. Interpretation of Precision and Recall

The high recall result is useful for predictive maintenance because missing an actual failure can be costly or unsafe. A model with high recall can help identify most machines that may require attention.

The low precision result means the model may recommend attention for some machines that are not actually failure cases. This could lead to unnecessary inspections or maintenance checks. However, in many maintenance settings, false alarms may be more acceptable than missed failures, depending on the cost and safety risk of equipment failure.

Therefore, the model should be interpreted as a high-sensitivity failure detection model. It is useful for catching most failures, but its predictions should still be reviewed carefully because not every predicted failure is a true failure.

## 5. Feature Importance

The classifier identified the following features as most important:

| Feature | Importance |
|---|---:|
| `Rotational speed` | 0.354252 |
| `Torque` | 0.345959 |
| `Tool wear` | 0.267254 |
| `Air temperature` | 0.029370 |
| `Process temperature` | 0.003165 |
| `Type_H` | 0.000000 |
| `Type_L` | 0.000000 |
| `Type_M` | 0.000000 |

The most important features were rotational speed, torque, and tool wear. Together, these three features explained most of the model's decision-making behavior.

This finding aligns with the anomaly detection results. In the anomaly detection section, the most anomalous records were often explained by unusual rotational speed, low torque, and high tool wear. The classification model also found these same features to be the most important for predicting failure.

## 6. Explainable AI Contribution

This classification step contributes to Explainable AI by showing which sensor features influenced the model globally. Instead of only reporting whether the model predicted failure or no failure, the feature importance table shows which machine conditions were most important to the prediction process.

The result is understandable from a predictive maintenance perspective:

- unusual rotational speed may indicate abnormal machine operation
- torque reflects mechanical load
- tool wear reflects equipment degradation over time

These explanations make the model output more transparent and easier to interpret.

## 7. Comparison With Anomaly Detection

The anomaly detection model flagged 16 out of 68 actual failures as anomalous. The classification model detected approximately 63 out of 68 failures based on the reported recall of 0.9265.

This shows that supervised classification is better at detecting known failure cases because it learns directly from the `Machine failure` label. However, anomaly detection is still useful because it identifies unusual machine behavior, including unusual records that may not be labeled as failures.

The two methods therefore provide complementary information:

- anomaly detection highlights unusual operating behavior
- classification predicts known failure patterns
- explainability helps users understand the sensor features behind both outputs

## 8. Limitations

The classification model has several limitations:

- Failure cases are rare, so the dataset is highly imbalanced.
- The model has low precision for the failure class, meaning it produces false alarms.
- The result depends on a simple model and may change with different model settings.
- Feature importance provides a global explanation, but it does not fully explain every individual prediction.
- The dataset is synthetic, so performance may not fully represent real industrial systems.

These limitations are important for Responsible and Explainable AI because they show that the model should be interpreted carefully rather than treated as a perfect decision-making system.

## 9. Next Steps

The next steps for this part of the project are:

1. Tune the classification threshold to study the tradeoff between precision and recall. This can help decide whether the system should prioritize catching more failures or reducing false alarms.
2. Add local explanations for individual predictions, not only global feature importance. This would show why a specific machine record was predicted as failure or no failure.
3. Compare classification results with anomaly detection results on the same test records. This will show when both methods agree and when they produce different signals.
4. Save classification predictions, probabilities, and feature explanations into output files so they can be used in the final report.
5. Explore whether a second simple model improves the precision-recall balance while remaining explainable.

## 10. Conclusion

The failure classification model successfully detected most actual machine failures in the test set, achieving high recall for the failure class. This is valuable in predictive maintenance because missed failures can be costly or unsafe.

However, the model also produced many false alarms, shown by its lower precision. This means the model is useful as a warning or decision-support tool, but its predictions should not be treated as final decisions without further analysis.

The feature importance results showed that rotational speed, torque, and tool wear were the most important predictors of machine failure. This supports the Explainable AI goal of the project by connecting model predictions to understandable machine sensor behavior.

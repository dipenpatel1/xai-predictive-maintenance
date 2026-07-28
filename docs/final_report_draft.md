# Final Report Draft

## Title

**Explainable Predictive Maintenance Using Anomaly Detection, Failure Classification, and Explanation Evaluation**

## Abstract

This project develops an Explainable and Responsible AI workflow for predictive maintenance using the AI4I 2020 Predictive Maintenance Dataset. The work combines anomaly detection, supervised failure classification, feature relationship analysis, and local and global explanation. The main findings show that anomaly detection identifies unusual operating behavior but does not capture all failures, while supervised classification detects most known failures but produces false alarms. Engineered features improve mean five-fold F1 from 0.3621 to 0.5119 while maintaining similar mean recall. SHAP faithfully reconstructs the fitted tree, whereas repeated LIME explanations show weak local surrogate fit. The project demonstrates that predictive maintenance models should be evaluated through predictive performance, explanation quality, error consequences, and clearly communicated limitations.

## 1. Introduction

Industrial equipment failure can lead to downtime, production delays, safety concerns, and unnecessary maintenance costs. Predictive maintenance uses sensor data to detect abnormal machine behavior and predict possible failure before serious problems occur.

However, predictive maintenance systems should not only provide predictions. In practical maintenance settings, users need to understand why a machine is flagged, which sensor conditions contributed to the output, and what the limitations of the model are. This makes the problem suitable for Explainable AI and Responsible AI analysis.

This project uses the AI4I 2020 Predictive Maintenance Dataset introduced by Matzka [1] and is motivated by the broader literature on explainable predictive maintenance [2]. The project focuses on a simple but meaningful workflow: anomaly detection, failure classification, feature relationship analysis, and model tuning.

## 2. Research Questions

The project is guided by the following research questions:

1. Can anomaly detection identify failure-related abnormal machine behavior?
2. Can supervised classification detect known machine failures more effectively than anomaly detection?
3. Which sensor features and feature relationships are most important for explaining model outcomes?
4. How do model tuning choices affect the tradeoff between missed failures and false alarms?
5. How can Explainable and Responsible AI principles improve interpretation of predictive maintenance models?
6. How stable and faithful are the generated explanations?

## 3. Dataset

The AI4I 2020 dataset contains 10,000 machine records. The input features used in this project are:

- `Type`
- `Air temperature`
- `Process temperature`
- `Rotational speed`
- `Torque`
- `Tool wear`

The target label is:

- `Machine failure`

The dataset also contains failure-mode columns:

- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

These failure-mode columns were not used as input features because they contain failure-related label information and would cause data leakage.

Machine failures are rare in the dataset. Only about 3.39% of records represent failures, making this an imbalanced classification problem.

## 4. Methodology

The project followed five main stages:

```text
Data Exploration
      ->
Data Preprocessing
      ->
Anomaly Detection
      ->
Failure Classification
      ->
Feature Relationship and Tuning Analysis
      ->
Local, Global, and Integrated Explanation Analysis
```

### 4.1 Data Preprocessing

The dataset was split into training and test sets using stratification to preserve the rare failure ratio in both sets. The categorical `Type` column was encoded, and numeric sensor features were scaled. The preprocessing was designed to avoid leakage by fitting transformations only on the training data.

### 4.2 Anomaly Detection

Anomaly detection was used to identify machine records that looked unusual compared with normal operating behavior. Isolation Forest [3] was trained using normal machine records and then evaluated on the test set. The goal was to determine whether unusual behavior overlaps with actual failure cases.

### 4.3 Failure Classification

A supervised classifier was trained to predict whether a machine record represents failure or no failure. Unlike anomaly detection, classification uses the `Machine failure` label during training. The classifier was evaluated using accuracy, precision, recall, F1-score, and confusion matrix.

### 4.4 Explainability

Explainability was added through:

- anomaly explanations based on feature deviation from normal behavior
- classifier feature importance
- feature correlation analysis
- engineered relationship features
- error analysis and model tuning interpretation
- local and global SHAP feature contributions [5]
- local LIME surrogate explanations [4]
- SHAP faithfulness and LIME stability checks [6], [7]

### 4.5 Feature Relationship Analysis

Additional engineered features were created to capture relationships between sensor variables:

| Engineered Feature | Formula | Interpretation |
|---|---|---|
| Temperature difference | `Process temperature - Air temperature` | Thermal behavior |
| Power proxy | `Torque * Rotational speed` | Approximate mechanical load/power relationship |
| Wear torque interaction | `Tool wear * Torque` | Degradation under mechanical load |
| Speed torque ratio | `Rotational speed / Torque` | Speed-load relationship |

These features were used to study whether feature interactions provide better explanatory value.

## 5. Results

### 5.1 Anomaly Detection Results

The anomaly detection model was evaluated on 2,000 test records.

| Metric | Value |
|---|---:|
| Total test records | 2000 |
| Actual failures in test set | 68 |
| Records flagged as anomalous | 113 |
| Anomaly rate | 5.65% |
| Failures flagged as anomalous | 16 |
| Normal records flagged as anomalous | 97 |
| Failures not flagged as anomalous | 52 |

The anomaly detector identified some failure-related abnormal behavior, but it did not capture all failures. This confirms that anomaly detection is useful for identifying unusual operating conditions but should not be treated as a complete failure classifier.

### 5.2 Failure Classification Results

The failure classification model produced the following results:

| Metric | Value |
|---|---:|
| Accuracy | 0.9260 |
| Precision | 0.3058 |
| Recall | 0.9265 |
| F1-score | 0.4599 |

The confusion matrix showed:

|  | Predicted No Failure | Predicted Failure |
|---|---:|---:|
| Actual No Failure | 1789 | 143 |
| Actual Failure | 5 | 63 |

The classifier detected 63 out of 68 failures, showing high recall. However, it also produced 143 false alarms, resulting in lower precision.

### 5.3 Feature Importance

The most important classifier features were:

| Feature | Importance |
|---|---:|
| Rotational speed | 0.3543 |
| Torque | 0.3460 |
| Tool wear | 0.2673 |
| Air temperature | 0.0294 |
| Process temperature | 0.0032 |

This suggests that rotational speed, torque, and tool wear are the main features used by the classifier.

### 5.4 Feature Relationship Results

The feature relationship analysis showed that engineered features improved model interpretability and changed the model's decision behavior.

| Model | Accuracy | Precision | Recall | F1-score | False Positives | False Negatives |
|---|---:|---:|---:|---:|---:|---:|
| Base features | 0.9260 | 0.3058 | 0.9265 | 0.4599 | 143 | 5 |
| Base + engineered features | 0.9405 | 0.3509 | 0.8824 | 0.5021 | 111 | 8 |

Adding engineered features improved precision and F1-score and reduced false positives. However, recall decreased slightly, meaning the engineered model missed more actual failures.

The engineered model identified `Speed torque ratio`, `Power proxy`, `Tool wear`, and `Temperature difference` as important. This suggests that failure behavior depends not only on individual sensor features but also on relationships between features.

### 5.5 Model Tuning Results

Model tuning showed that there is no universally best setting. Different settings create different tradeoffs.

For example, increasing the classification threshold from 0.5 to 0.8 reduced false positives from 143 to 84, but increased false negatives from 5 to 9.

Class weight tuning showed that a manual failure weight of 5 may provide a more balanced model:

| Setting | Precision | Recall | False Positives | False Negatives |
|---|---:|---:|---:|---:|
| Balanced | 0.3058 | 0.9265 | 143 | 5 |
| Failure weight 5 | 0.4265 | 0.8529 | 78 | 10 |

This shows a clear Responsible AI tradeoff between catching failures and reducing false alarms.

### 5.6 Integrated Explanation and Validation Results

Combining the anomaly and classifier outputs on the same test records produced four groups:

| Signal group | Records | Failures | Observed failure rate |
|---|---:|---:|---:|
| Neither | 1716 | 4 | 0.23% |
| Classifier only | 171 | 48 | 28.07% |
| Anomaly only | 78 | 1 | 1.28% |
| Both | 35 | 15 | 42.86% |

The highest observed failure rate occurred when both methods flagged a record, but most detected failures appeared in the classifier-only group. This confirms that anomaly and failure are not equivalent.

Group-level SHAP analysis showed that rotational speed and torque pushed all five false negatives away from the failure class. SHAP reconstructed the fitted tree probabilities with a mean absolute error of approximately 3.11 x 10^-14. In contrast, repeated LIME explanations produced mean local surrogate R2 values of 0.1115 for the selected false positive and 0.0970 for the selected false negative. The LIME results were readable but only weak approximations of the tree near those records.

Five-fold validation strengthened the engineered-feature result:

| Model | Mean precision | Mean recall | Mean F1 |
|---|---:|---:|---:|
| Base features | 0.2256 | 0.9204 | 0.3621 |
| Engineered features | 0.3552 | 0.9203 | 0.5119 |

The engineered model improved precision and F1 without reducing mean recall across the five folds.

## 6. Discussion

The results show that anomaly detection and classification play different roles. Anomaly detection is useful for identifying unusual behavior, even when a record is not labeled as a failure. Classification is better for detecting known failure labels because it learns directly from labeled examples.

The feature relationship analysis strengthened the project by showing that important behavior may exist in feature combinations. In particular, speed-torque ratio and power proxy became highly important when engineered features were added. This suggests that system-level relationships can improve interpretability.

The tuning analysis shows that model performance should not be summarized with one accuracy number. The model must be interpreted through precision, recall, false positives, false negatives, and the practical cost of each error type.

## 7. Responsible and Explainable AI Analysis

This project contributes to Explainable AI by connecting model outputs to understandable sensor behavior. Instead of only reporting anomaly or failure predictions, the project explains outcomes using:

- feature deviations from normal behavior
- feature importance
- feature correlations
- engineered feature relationships
- error analysis

The Responsible AI contribution comes from recognizing model limitations and tradeoffs. A high-recall model may catch most failures but produce many false alarms. A higher-precision model may reduce false alarms but miss more failures. Explanation methods also have limitations: a faithful explanation can describe the fitted model without proving that the model is physically correct, and an unstable or weak surrogate can produce a persuasive but incomplete account. These issues support the NIST AI RMF emphasis on validity, reliability, transparency, explainability, and documented risk management [8]. They are also consistent with machine-learning safety research that distinguishes test performance from system-level safety [9].

The system should be treated as decision support, not as an autonomous maintenance decision-maker.

## 8. Limitations

This project has several limitations:

- The dataset is synthetic.
- The system was not tested in a real industrial environment.
- The models are simple and intended for interpretability.
- LIME explanations had low local surrogate R2 for the two audited cases.
- SHAP explains the fitted model but does not establish physical causality.
- No maintenance engineer evaluated the usefulness of the explanations.
- Anomaly detection does not directly predict failure.
- The cost analysis used hypothetical ratios rather than industrial cost data.

## 9. Future Work

Future work may include:

- comparison with additional explainable models
- calibration of predicted probabilities
- testing on a real industrial predictive maintenance dataset
- evaluation of explanations with maintenance experts
- temporal validation using real machine histories

## 10. Conclusion

This project developed an Explainable and Responsible AI workflow for predictive maintenance. Anomaly detection identified unusual machine behavior, while supervised classification detected most known failures. Feature relationship analysis showed that engineered system-level features such as speed-torque ratio and power proxy provide meaningful explanatory value.

The main finding is that predictive maintenance modeling requires more than basic accuracy. A responsible system must analyze false alarms, missed failures, feature relationships, explanation quality, and uncertainty across data splits. The project demonstrates how simple machine learning models can be made more meaningful through explanation evaluation, error analysis, and explicit cost assumptions.

## References

[1] S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," in *2020 Third International Conference on Artificial Intelligence for Industries (AI4I)*, 2020, pp. 69-74, doi: 10.1109/AI4I49448.2020.00023.

[2] L. Cummins, A. Sommers, S. B. Ramezani, S. Mittal, J. E. Jabour, M. Seale, and S. Rahimi, "Explainable Predictive Maintenance: A Survey of Current Methods, Challenges and Opportunities," *IEEE Access*, vol. 12, pp. 57574-57602, 2024, doi: 10.1109/ACCESS.2024.3391130.

[3] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation Forest," in *Proc. 8th IEEE Int. Conf. Data Mining*, 2008, pp. 413-422, doi: 10.1109/ICDM.2008.17.

[4] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why Should I Trust You?': Explaining the Predictions of Any Classifier," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 1135-1144, doi: 10.1145/2939672.2939778.

[5] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Advances in Neural Information Processing Systems 30*, 2017, pp. 4765-4774.

[6] D. Alvarez-Melis and T. S. Jaakkola, "On the Robustness of Interpretability Methods," in *Proc. ICML Workshop on Human Interpretability in Machine Learning*, 2018.

[7] F. Doshi-Velez and B. Kim, "Towards a Rigorous Science of Interpretable Machine Learning," arXiv:1702.08608, 2017.

[8] E. Tabassi, "Artificial Intelligence Risk Management Framework (AI RMF 1.0)," National Institute of Standards and Technology, NIST AI 100-1, 2023, doi: 10.6028/NIST.AI.100-1.

[9] S. Mohseni, H. Wang, C. Xiao, Z. Yu, Z. Wang, and J. Yadawa, "Taxonomy of Machine Learning Safety: A Survey and Primer," *ACM Comput. Surv.*, vol. 55, no. 8, Art. no. 157, pp. 1-38, 2023, doi: 10.1145/3551385.

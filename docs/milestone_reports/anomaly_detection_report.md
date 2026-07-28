# Milestone 1 Report

## Title

**Anomaly Detection and Explainable Analysis for Predictive Maintenance**

## 1. Objective

The objective of this milestone is to build the first working part of the predictive maintenance project: detecting abnormal machine behavior and explaining why certain machine records are considered unusual. This milestone focuses on anomaly detection and explainability using the AI4I 2020 Predictive Maintenance Dataset.

The goal is not to build a complete failure prediction system yet. Instead, this milestone answers the following questions:

- Which machine records look unusual compared with normal operating behavior?
- Do anomalous records overlap with actual machine failures?
- Which sensor features help explain why a record was flagged as anomalous?

## 2. Dataset

This milestone uses the AI4I 2020 Predictive Maintenance Dataset. The dataset contains 10,000 machine records with operating condition features such as machine type, air temperature, process temperature, rotational speed, torque, and tool wear.

The dataset also includes failure-related labels, including `Machine failure` and specific failure mode columns such as `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`. These failure labels were not used as input features for anomaly detection. They were kept only for evaluation and interpretation.

The target label distribution shows that machine failures are rare. In the full dataset, only about 3.39% of records are failure cases. This confirms that the dataset is imbalanced, which is common in predictive maintenance because most machines operate normally most of the time.

## 3. Preprocessing

The preprocessing step selected only safe machine condition features as model inputs:

- `Type`
- `Air temperature`
- `Process temperature`
- `Rotational speed`
- `Torque`
- `Tool wear`

The following columns were excluded from the input features because they contain failure label information:

- `Machine failure`
- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

This separation was important to avoid data leakage. If failure-mode columns were used as inputs, the model would indirectly receive the answer during training.

The dataset was split into training and test sets using stratification so that the rare failure class was represented in both sets. The categorical `Type` feature was encoded, and numeric sensor features were standardized using a scaler fitted only on the training data. This avoided leakage from the test set into the preprocessing step.

## 4. Anomaly Detection

The anomaly detection model was trained to identify records that differ from normal machine behavior. For this milestone, the model was trained using normal training records so that it could learn typical machine operating patterns.

The model then assigned anomaly flags and anomaly scores to the test records. A lower anomaly score indicates that the record is more unusual. The output was saved in:

```text
outputs/anomaly_detection_test_results.csv
```

## 5. Results

The anomaly detection model was evaluated on 2,000 test records. The summary results were:

| Metric | Value |
|---|---:|
| Total test records | 2000 |
| Actual failures in test set | 68 |
| Records flagged as anomalous | 113 |
| Anomaly rate | 5.65% |
| Failures flagged as anomalous | 16 |
| Normal records flagged as anomalous | 97 |
| Failures not flagged as anomalous | 52 |

The model flagged 113 out of 2,000 test records as anomalous, giving an anomaly rate of 5.65%. Among the 68 actual machine failures in the test set, 16 were flagged as anomalous.

These results show that anomaly detection can identify some failure-related abnormal behavior, but it does not capture all failures. This is expected because anomaly detection is not the same as failure classification. Anomaly detection identifies unusual behavior, while failure classification predicts whether a record belongs to a failure class.

## 6. Explainability

To explain the anomaly results, the most anomalous records were compared against the average behavior of normal training records. For each anomalous record, the explanation identified which numeric sensor features were farthest from normal behavior in terms of standard deviations.

The top anomaly explanations were saved in:

```text
outputs/top_anomaly_explanations.csv
```

The explanations showed that the most anomalous records were commonly influenced by:

- unusually high rotational speed
- unusually low torque
- high tool wear
- unusual temperature values

For example, one actual failure case was flagged as anomalous because its rotational speed was 7.60 standard deviations away from normal behavior and its torque was 3.56 standard deviations away. This type of explanation is useful because it does not only report that a record is anomalous; it also gives a human-understandable reason for the anomaly.

## 7. Interpretation

The results suggest that anomaly detection can be useful as an early warning tool in predictive maintenance. It can highlight machine records that look different from typical operating behavior. However, the results also show that not every anomaly is a failure, and not every failure is detected as an anomaly.

This means anomaly detection should be interpreted carefully. It is useful for identifying unusual machine behavior, but it should not be treated as a complete failure prediction system. The explainability step helps make the anomaly results more meaningful by showing which sensor values caused a record to appear unusual.

## 8. Limitations

This milestone has several limitations:

- The dataset is synthetic, so it may not fully represent real industrial environments.
- The anomaly detector does not directly learn the difference between failure and no failure.
- Some actual failures were not flagged as anomalous.
- Some normal records were flagged as anomalous.
- The explanation method is simple and based on feature deviation from normal averages.

These limitations are acceptable for the first milestone because the purpose was to build and understand the anomaly detection and explanation workflow.

## 9. Next Steps

The next project steps are:

1. Create visualizations for anomaly detection results.
2. Analyze how anomaly scores differ between normal and failure records.
3. Improve the explanation section by summarizing which features most often contribute to anomalies.
4. Later, add a simple failure classification model and compare it with anomaly detection.

## 10. Conclusion

Milestone 1 successfully implemented an anomaly detection workflow for predictive maintenance and added a simple explainability method. The system detected unusual machine records, compared anomaly flags with actual failure labels, and explained anomalous records using sensor feature deviations from normal behavior.

The main finding is that anomaly detection can identify some failure-related abnormal behavior, but it should be understood as a method for detecting unusual operating conditions rather than as a complete failure classifier. Explainability improves the usefulness of the results by showing which sensor features made each record unusual.


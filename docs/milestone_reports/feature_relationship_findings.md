# Feature Relationship and Engineered Feature Findings

## 1. Objective

After the initial anomaly detection and failure classification experiments, the next research step was to study relationships between input features. This was motivated by the idea that machine failure may not depend only on individual sensor values, but also on combinations of sensor features.

This section investigates:

- correlation between sensor features
- correlation between engineered features and machine failure
- whether relationship-based engineered features improve model performance
- whether engineered features make the model explanation more meaningful

## 2. Why Feature Relationships Matter

The first classification model showed that rotational speed, torque, and tool wear were the most important features. However, these features may interact with each other. For example, torque and rotational speed together can represent a proxy for mechanical power, while tool wear and torque together can represent load on a worn tool.

Therefore, this analysis moves beyond asking:

```text
Which single feature is important?
```

and asks:

```text
Do combinations of features reveal more meaningful system behavior?
```

## 3. Engineered Relationship Features

Four engineered features were created from the original sensor variables:

| Engineered Feature | Formula | Interpretation |
|---|---|---|
| Temperature difference | `Process temperature - Air temperature` | Difference between process and ambient machine temperature |
| Power proxy | `Torque * Rotational speed` | Approximate mechanical load/power relationship |
| Wear torque interaction | `Tool wear * Torque` | Interaction between degradation and mechanical load |
| Speed torque ratio | `Rotational speed / Torque` | Relationship between speed and load |

These features were added to test whether system-level relationships provide better explanatory value than raw sensor features alone.

## 4. Correlation With Machine Failure

The strongest correlations with `Machine failure` were:

| Feature | Correlation With Machine Failure |
|---|---:|
| Torque | 0.1913 |
| Wear torque interaction | 0.1904 |
| Power proxy | 0.1760 |
| Temperature difference | -0.1117 |
| Tool wear | 0.1054 |
| Air temperature | 0.0826 |
| Speed torque ratio | 0.0633 |
| Rotational speed | -0.0442 |
| Process temperature | 0.0359 |

The correlations are not extremely high, which suggests that machine failure is not explained by one simple linear relationship. However, torque, wear-torque interaction, power proxy, and temperature difference show meaningful relationships with the failure label.

This supports the professor's suggestion that relationships between features should be considered, not only individual feature importance.

## 5. Model Comparison

A model using only the original features was compared with a model using both original and engineered relationship features.

| Model | Accuracy | Precision | Recall | F1-score | False Positives | False Negatives | True Positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base features | 0.9260 | 0.3058 | 0.9265 | 0.4599 | 143 | 5 | 63 |
| Base + engineered features | 0.9405 | 0.3509 | 0.8824 | 0.5021 | 111 | 8 | 60 |

Adding engineered relationship features improved accuracy, precision, and F1-score. The number of false positives decreased from 143 to 111, which means the model produced fewer false alarms. However, recall decreased from 0.9265 to 0.8824, and false negatives increased from 5 to 8.

This creates an important research tradeoff:

```text
The engineered model reduces false alarms but misses slightly more actual failures.
```

In predictive maintenance, the better choice depends on the practical cost of false alarms compared with missed failures.

## 6. Feature Importance With Engineered Features

The engineered model identified the following features as most important:

| Feature | Importance |
|---|---:|
| Speed torque ratio | 0.3413 |
| Power proxy | 0.3172 |
| Tool wear | 0.1519 |
| Temperature difference | 0.1075 |
| Wear torque interaction | 0.0499 |
| Rotational speed | 0.0319 |
| Type_L | 0.0003 |
| Air temperature | 0.0000 |
| Process temperature | 0.0000 |
| Torque | 0.0000 |
| Type_H | 0.0000 |
| Type_M | 0.0000 |

This result is important because the model shifted from using raw features such as rotational speed and torque separately to using relationship-based features such as speed-torque ratio and power proxy. This suggests that combinations of features may better represent the underlying machine behavior.

## 7. Explainable AI Interpretation

The engineered features improve explainability because they represent meaningful relationships:

- `Power proxy` connects torque and rotational speed.
- `Wear torque interaction` connects tool degradation with mechanical load.
- `Temperature difference` captures thermal behavior.
- `Speed torque ratio` captures unusual speed-load combinations.

Instead of saying only:

```text
Torque is important.
```

the project can now say:

```text
The relationship between rotational speed and torque is important, suggesting that failure risk may depend on combined operating conditions rather than one sensor alone.
```

This is a stronger research interpretation because it connects model behavior to system behavior.

## 8. Responsible AI Interpretation

This analysis also supports Responsible AI because it makes the model's behavior more transparent. By studying feature relationships, the project avoids treating feature importance as a black-box output. Instead, it asks whether the important features make sense in the maintenance domain.

The model comparison also shows an important decision tradeoff:

- the base model catches more failures but creates more false alarms
- the engineered model creates fewer false alarms but misses slightly more failures

This tradeoff should be communicated clearly because different maintenance settings may prefer different risk levels.

## 9. Generated Artifacts

The following report artifacts were generated:

```text
outputs/feature_relationships/feature_correlation_matrix.csv
outputs/feature_relationships/feature_correlation_matrix.png
outputs/feature_relationships/correlation_with_machine_failure.csv
outputs/feature_relationships/base_vs_engineered_model_metrics.csv
outputs/feature_relationships/engineered_model_feature_importance.csv
outputs/feature_relationships/engineered_model_feature_importance.png
outputs/feature_relationships/speed_torque_relationship.png
outputs/feature_relationships/tool_wear_torque_relationship.png
outputs/feature_relationships/temperature_difference_speed_relationship.png
outputs/feature_relationships/power_wear_interaction_relationship.png
```

## 10. Conclusion

The feature relationship analysis showed that predictive maintenance behavior is better understood by studying relationships between features, not only individual sensor values. Engineered features such as speed-torque ratio and power proxy became highly important in the classification model, suggesting that nonlinear or interaction-based patterns exist in the data.

Adding engineered features improved precision and F1-score while reducing false positives, but it slightly reduced recall. This creates a meaningful research and Responsible AI tradeoff between reducing false alarms and catching as many failures as possible.


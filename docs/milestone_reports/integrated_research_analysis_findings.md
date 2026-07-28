# Integrated Research Analysis Findings

## 1. Purpose

This analysis connects anomaly detection, failure classification, feature
relationships, SHAP, and LIME on the same predictive-maintenance experiment.
The purpose is to move beyond showing that separate models run successfully
and ask research-oriented questions about agreement, errors, explanation
quality, result stability, and decision costs.

The complete reproducible analysis is in:

`src/07_integrated_research_analysis/integrated_research_analysis.ipynb`

## 2. Anomaly and Classification Agreement

The anomaly detector and classifier were applied to the same 2,000 test
records.

| Signal group | Records | Actual failures | Observed failure rate |
|---|---:|---:|---:|
| Neither | 1,716 | 4 | 0.23% |
| Classifier only | 171 | 48 | 28.07% |
| Anomaly only | 78 | 1 | 1.28% |
| Both | 35 | 15 | 42.86% |

The records flagged by both models had the highest observed failure rate.
However, the classifier-only group contained 48 failures, showing that many
known failure patterns were not globally unusual enough for Isolation Forest.
The anomaly-only group contained mainly non-failure records, confirming that
anomalous behavior and labeled failure are related but different concepts.

Four failures received neither signal. Therefore, the absence of both flags
must not be interpreted as a guarantee of safe operation.

## 3. Error-Group Analysis

The classifier produced 63 true positives, 143 false positives, 5 false
negatives, and 1,789 true negatives.

| Outcome | Rotational speed | Torque | Tool wear | Mean failure probability |
|---|---:|---:|---:|---:|
| True positive | 1,463.79 | 51.71 | 149.86 | 0.935 |
| False positive | 1,487.15 | 44.97 | 161.59 | 0.853 |
| False negative | 1,678.40 | 30.56 | 197.00 | 0.091 |
| True negative | 1,542.56 | 39.13 | 103.13 | 0.033 |

The five false negatives shared a recognizable pattern: higher speed, lower
torque, and greater tool wear than the true-positive group. This supports the
individual false-negative explanation in Notebook 06 and suggests a repeated
model blind spot.

False positives were not merely borderline cases. Their mean predicted
failure probability was 0.853. This indicates that the fitted tree often made
confident false alarms and motivates inspecting learned decision rules rather
than changing only the threshold.

## 4. Group-Level SHAP Findings

Mean absolute SHAP values showed that the main influential features varied by
outcome:

- True positives were driven mainly by rotational speed, torque, and tool wear.
- False positives received their largest average contribution from tool wear.
- False negatives received strong contributions from rotational speed and
  torque, but the signed values pushed predictions away from failure.
- Machine `Type` had zero SHAP contribution in the fitted depth-4 tree.

For false negatives, mean signed SHAP contributions from rotational speed and
torque were -0.2197 and -0.1688. These negative contributions explain why the
model assigned the missed failures a low mean probability.

This is a model-behavior explanation. It does not prove that a sensor feature
physically caused failure.

## 5. Explanation Evaluation

### 5.1 SHAP Faithfulness

For all 2,000 test records, the SHAP baseline plus feature contributions was
compared with the tree's failure probability.

| Check | Value |
|---|---:|
| SHAP baseline | 0.5000 |
| Mean absolute reconstruction error | 3.11 x 10^-14 |
| Maximum absolute reconstruction error | 3.24 x 10^-14 |

The near-zero errors show that Tree SHAP faithfully reconstructs this fitted
model's output. Faithfulness to the model does not establish model correctness,
causality, or usefulness to a maintenance expert.

### 5.2 LIME Stability

LIME was repeated across five random seeds for one false positive and one
false negative.

| Case | Mean surrogate R2 | Mean top-3 Jaccard | Minimum top-3 Jaccard |
|---|---:|---:|---:|
| False positive | 0.1115 | 1.00 | 1.00 |
| False negative | 0.0970 | 0.70 | 0.50 |

The false-positive feature set was stable, while the false-negative feature
set changed between runs. More importantly, both local surrogate R2 values
were low. The LIME explanations should therefore be presented as approximate
supporting evidence, not as exact descriptions of the tree.

This is an important Responsible AI result: explanation methods must be
evaluated instead of trusted only because they generate readable charts.

## 6. Five-Fold Validation

Stratified five-fold cross-validation tested whether the engineered-feature
improvement depended on one train-test split.

| Model | Accuracy mean (SD) | Precision mean (SD) | Recall mean (SD) | F1 mean (SD) |
|---|---:|---:|---:|---:|
| Base | 0.8897 (0.0094) | 0.2256 (0.0153) | 0.9204 (0.0130) | 0.3621 (0.0200) |
| Engineered | 0.9402 (0.0074) | 0.3552 (0.0257) | 0.9203 (0.0339) | 0.5119 (0.0264) |

The engineered model improved mean precision and F1 while maintaining almost
the same mean recall. This provides stronger evidence than the earlier
single-split comparison.

The cross-validation mean for the base model differs from the original
holdout score because each fold contains a different training and validation
partition. The holdout result remains valid for its test set, while the
five-fold result gives a better estimate of variability.

## 7. Cost-Sensitivity Analysis

The analysis assigned a relative cost of 1 to each false positive and varied
the cost of a false negative from 1 to 50:

`Relative cost = FP + missed-failure cost ratio x FN`

| Configuration | False positives | False negatives | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Balanced, threshold 0.5 | 143 | 5 | 0.3058 | 0.9265 | 0.4599 |
| Balanced, threshold 0.8 | 84 | 9 | 0.4126 | 0.8676 | 0.5592 |
| Failure weight 5 | 78 | 10 | 0.4265 | 0.8529 | 0.5686 |
| Engineered, threshold 0.5 | 111 | 8 | 0.3509 | 0.8824 | 0.5021 |

The preferred configuration changed with the assumed error costs:

- Ratios 1-5: failure-weight-5 model
- Ratios 6-14: balanced model with threshold 0.8
- Ratios 15-50: balanced model with threshold 0.5

These are sensitivity results, not estimated industrial costs. A real
deployment would require cost values and acceptable error rates supplied or
validated by domain experts.

## 8. Main Research Conclusions

1. Anomaly detection and failure classification provide different information.
2. Agreement between the methods identifies a higher-risk subgroup, but it is
   not a calibrated risk probability.
3. Group analysis reveals a repeated high-speed, low-torque false-negative
   pattern.
4. Engineered sensor relationships improve F1 across multiple data splits.
5. SHAP is faithful to the fitted tree, while selected LIME explanations have
   weak local surrogate fit.
6. Responsible model selection depends on explicit error consequences, not
   accuracy alone.

## 9. Limitations

- AI4I is a synthetic dataset.
- The five false negatives form a very small group.
- Failure rates for the four signal groups come from one held-out test set.
- The cost ratios are hypothetical.
- SHAP and LIME explain model behavior, not physical causality.
- No maintenance engineer evaluated whether the explanations are useful.
- Results do not establish production readiness or industrial safety.

## 10. Presentation-Ready Evidence

The following figures are generated under
`outputs/integrated_research_analysis/figures/`:

- `signal_agreement_matrix.png`: shows why anomaly and failure are different.
- `error_group_sensor_means.png`: highlights the false-negative pattern.
- `group_shap_heatmaps.png`: compares explanations across TP, FP, FN, and TN.
- `five_fold_model_comparison.png`: supports the engineered-feature result.
- `cost_sensitivity.png`: demonstrates responsible model-selection tradeoffs.

The signal-agreement matrix, SHAP heatmaps, and five-fold comparison are the
strongest candidates for the main presentation. The remaining figures can be
used as backup slides.

## 11. Related References

The consolidated IEEE bibliography and guidance for citing these methods is
available in:

`docs/references/research_references_ieee.md`

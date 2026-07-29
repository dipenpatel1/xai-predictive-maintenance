# Neural Network Comparison Findings

## 1. Research Purpose

The neural network was added as a comparison model, not with the assumption
that it would outperform the decision tree. It addresses this question:

> Does a less directly interpretable nonlinear model improve failure
> prediction enough to justify its additional complexity and post-hoc
> explanation requirements?

The experiment is implemented in:

`src/05_neural_network/neural_network_comparison.ipynb`

## 2. Model Design

The model is a small feedforward multilayer perceptron:

```text
8 transformed inputs -> 16 ReLU neurons -> 8 ReLU neurons -> 1 output
```

It has 289 trainable parameters. Numeric sensors are standardized and machine
`Type` is one-hot encoded. The model uses:

- Adam optimization
- balanced sample weights
- L2 regularization with `alpha=0.001`
- early stopping
- a fixed random seed

Balanced sample weights increase the contribution of rare failure records to
the loss. Early stopping ended training after 162 iterations.

## 3. Held-Out Test Results

| Configuration | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base tree | 0.50 | 0.3058 | 0.9265 | 0.4599 | 0.5246 | 0.9538 | 143 | 5 |
| Engineered tree | 0.50 | 0.3509 | 0.8824 | 0.5021 | 0.7918 | 0.9255 | 111 | 8 |
| Neural network | 0.50 | 0.2711 | 0.8971 | 0.4164 | 0.6185 | 0.9618 | 164 | 7 |
| Neural network, selected threshold | 0.86 | 0.5281 | 0.6912 | 0.5987 | 0.6185 | 0.9618 | 42 | 21 |

At the common threshold of 0.5, the neural network did not outperform the
trees on precision or F1. It detected 61 of 68 failures but generated 164
false alarms.

The network had the highest ROC-AUC and higher PR-AUC than the base tree,
showing useful probability ranking. The engineered tree nevertheless had the
strongest PR-AUC.

## 4. Training-Only Threshold Selection

An alternative threshold was selected by maximizing F1 using five-fold
out-of-fold predictions from the training set. Final test labels were not
used during selection.

The selected threshold was 0.86. On the test set it improved precision and F1
but increased missed failures from 7 to 21.

This result must be interpreted as an operating-policy comparison:

- the network architecture determines probabilities;
- the threshold determines when an alarm is raised;
- threshold improvement must not be described as architecture improvement.

## 5. Five-Fold Validation

| Model | Precision mean (SD) | Recall mean (SD) | F1 mean (SD) | PR-AUC mean (SD) | ROC-AUC mean (SD) |
|---|---:|---:|---:|---:|---:|
| Base tree | 0.2256 (0.0153) | 0.9204 (0.0130) | 0.3621 (0.0200) | 0.4584 (0.0733) | 0.9250 (0.0184) |
| Engineered tree | 0.3552 (0.0257) | 0.9203 (0.0339) | 0.5119 (0.0264) | 0.7885 (0.0472) | 0.9439 (0.0157) |
| Neural network | 0.2279 (0.0170) | 0.8525 (0.0142) | 0.3593 (0.0207) | 0.5648 (0.0400) | 0.9501 (0.0038) |

The network achieved the highest mean ROC-AUC and the lowest ROC-AUC
variability. However, the engineered tree remained stronger on mean
precision, recall, F1, and PR-AUC.

For rare-failure detection, PR-AUC is particularly informative because it
focuses on positive-class precision and recall. The many true negatives can
make ROC-AUC appear strong even when the selected alarm threshold creates
many false positives.

## 6. Global Explanation

Permutation importance measured the decrease in test PR-AUC after shuffling
one original feature.

| Feature | Mean PR-AUC decrease |
|---|---:|
| Torque | 0.4099 |
| Air temperature | 0.4074 |
| Rotational speed | 0.3417 |
| Process temperature | 0.2679 |
| Tool wear | 0.1784 |
| Type | 0.0051 |

The neural network used temperature features more strongly than the base
decision tree. Machine `Type` had little measured influence.

Approximate model-agnostic SHAP on 200 test records ranked air temperature,
tool wear, torque, process temperature, and rotational speed as the main
contributors. Permutation importance and SHAP need not have identical
rankings because they measure different properties.

## 7. Local Explanation

One true positive, false positive, false negative, and true negative was
explained using model-agnostic SHAP and LIME.

- The selected true positive was mainly pushed toward failure by air
  temperature.
- The selected false positive was dominated by a strong positive torque
  contribution.
- In the selected false negative, torque pushed toward failure, but process
  temperature and other contributions suppressed the final output.
- The selected false negative received failure probability 0.0134.

The maximum SHAP reconstruction error for the four cases was approximately
`5.55 x 10^-16`, demonstrating numerical additivity for those explanations.

LIME local surrogate R2 values ranged from 0.1164 to 0.3557. The explanations
were readable but only weak-to-moderate approximations of the network near
the selected records.

## 8. Explainability Tradeoff

The decision tree is intrinsically easier to inspect through:

- decision paths
- split thresholds
- feature importance
- exact Tree SHAP

The neural network requires post-hoc methods:

- permutation importance for global sensitivity
- model-agnostic SHAP for feature contributions
- LIME for local approximation

These methods add computational cost and interpretation assumptions. A neural
network should therefore be preferred only when its predictive improvement is
meaningful enough to justify that cost.

## 9. Responsible AI Interpretation

The experiment strengthens the Responsible AI analysis in several ways:

- It reports that the more complex model did not automatically perform better.
- It separates probability ranking from threshold-dependent decisions.
- It selects the alternative threshold without using final test labels.
- It evaluates both positive-class performance and explanation quality.
- It communicates the additional missed failures caused by threshold 0.86.
- It does not treat post-hoc explanations as physical causality.

## 10. Recommendation

Retain the engineered depth-4 decision tree as the primary project model.
Use the neural network as a nonlinear comparison and Explainable AI case
study.

This conclusion is supported by the engineered tree's stronger five-fold
precision, recall, F1, and PR-AUC, together with its simpler interpretation.
The neural network contributes useful evidence about ranking quality,
threshold selection, and the cost of explaining a less transparent model.

## 11. Limitations

- Only one deliberately small architecture was evaluated.
- Hyperparameters were not extensively optimized.
- The AI4I dataset is synthetic and tabular.
- Predicted probabilities were not calibrated.
- Global neural SHAP used a representative sample of 200 test records.
- Local SHAP and LIME examples do not establish physical causality.
- No maintenance expert evaluated explanation usefulness.

## 12. Presentation-Ready Evidence

The strongest figures under `outputs/neural_network_comparison/figures/` are:

- `model_precision_recall_comparison.png`
- `five_fold_neural_comparison.png`
- `neural_network_confusion_matrices.png`
- `neural_permutation_importance.png`

The precision-recall comparison is the best main-slide figure. The learning
curve and explanation figures are appropriate for backup slides.

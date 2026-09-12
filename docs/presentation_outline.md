# Simple Final Presentation Outline (12 Slides)

Suggested duration: 10-12 minutes, followed by questions.

The main presentation focuses on the central project story. More advanced
experiments, such as the neural-network comparison and detailed explanation
evaluation, can be kept as backup material.

## Slide 1: Title

### Put on the slide

**Explainable Predictive Maintenance Using Machine Learning**

- Bijay Pandey - Student ID: `[add ID]`
- `[Add other group members and IDs, or remove this line]`
- Responsible and Explainable AI
- University of Regina

### What to say

Good morning. My project studies how machine learning can identify unusual
machine behavior and possible failures. It also explores how explanations can
help users understand the model's output.

## Slide 2: Introduction

### Put on the slide

### Problem

- Machine failure can cause downtime, cost, and safety concerns.
- Sensor data can help identify problems before failure occurs.
- A prediction is more useful when users can understand why it was made.

### Motivation

Build a small predictive-maintenance prototype that is understandable and
supports human decision-making.

### Simple visual

```text
Machine sensors -> Machine-learning model -> Prediction + explanation
```

### What to say

Predictive maintenance uses information such as temperature, rotational
speed, torque, and tool wear. My motivation was to study both prediction and
explanation instead of considering accuracy alone.

## Slide 3: Project Objectives

### Put on the slide

1. Explore and prepare industrial sensor data.
2. Detect machine records with unusual behavior.
3. Classify records as failure or no failure.
4. Study relationships between sensor features.
5. Explain important features and individual predictions.
6. Discuss the results using Responsible AI principles.

### What to say

The project has two related machine-learning tasks. Anomaly detection asks
whether a record looks unusual. Classification asks whether a record matches
known failure examples. I then use explainability to understand the results.

## Slide 4: Brief Literature Review

### Put on the slide

```text
Predictive maintenance      Explainable AI       This project
        |                         |                    |
Detect problems early  +  Explain predictions  ->  Responsible prototype
```

- Matzka introduced the AI4I dataset for explainable predictive maintenance
  [1].
- Cummins et al. showed the importance of understandable outputs in
  predictive maintenance [2].
- Isolation Forest identifies unusual observations [3].
- SHAP and LIME explain how features influence model predictions [4], [5].

### What to say

The literature shows that predictive-maintenance models should not only
produce outputs. Operators should also be able to understand the main reasons
behind those outputs. These studies guided my choice of dataset, anomaly
detection, and explanation methods.

## Slide 5: Dataset and Methodology

### Put on the slide

### AI4I 2020 dataset

- 10,000 machine records
- 339 failure records, or 3.39%
- Features: machine type, temperatures, speed, torque, and tool wear
- Target: machine failure

```text
Explore data -> Prepare data -> Detect anomalies -> Classify failures
             -> Study feature relationships -> Explain results
```

**Tools:** Python, Jupyter Notebook, pandas, scikit-learn, Matplotlib,
SHAP, and LIME.

### What to say

I removed identifier and failure-mode columns from the model inputs. The
failure-mode columns were excluded because they already contain information
about the target and could cause data leakage. I split the data into training
and testing sets while preserving the small failure percentage.

## Slide 6: Anomaly Detection Results

### Put on the slide

**Question:** Does this machine record look unusual?

| Result on 2,000 test records | Count |
|---|---:|
| Records flagged as unusual | 113 |
| Actual failures | 68 |
| Failures also flagged as unusual | 16 |

### Visual

Use:

`outputs/report_artifacts/anomaly_score_distribution.png`

### Main finding

**Unusual behavior is not always failure, and failure is not always unusual.**

### What to say

I used Isolation Forest to identify records that were different from normal
machine behavior. It found some failures, but most failures were not flagged
as anomalies. It also flagged normal records. Therefore, anomaly detection is
useful for finding unusual conditions, but it should not be treated as a
complete failure predictor.

## Slide 7: Failure Classification Results

### Put on the slide

**Question:** Does this record match a known failure pattern?

| Metric | Result |
|---|---:|
| Accuracy | 92.60% |
| Precision | 30.58% |
| Recall | 92.65% |
| F1-score | 45.99% |

- Detected 63 of 68 failures
- Missed 5 failures
- Produced 143 false alarms

### Visual

Use:

`outputs/report_artifacts/classification_confusion_matrix.png`

### What to say

Recall tells us how many real failures were detected. The model detected about
93% of them, which is useful. Precision tells us how many failure alarms were
correct. Precision was lower because the model produced many false alarms.
This shows why accuracy alone is not enough for an imbalanced dataset.

## Slide 8: Neural Network Experiment

### Put on the slide

A small feedforward neural network was added as a nonlinear benchmark:

```text
8 inputs -> 16 ReLU -> 8 ReLU -> 1 sigmoid
```

It used balanced sample weights, Adam optimization, regularization, and early
stopping. Three supervised classification models were then compared:

- **Base decision tree:** original sensor features
- **Engineered decision tree:** original features plus sensor relationships
- **Neural network:** small nonlinear comparison model

| Five-fold mean | Precision | Recall | F1 |
|---|---:|---:|---:|
| Base decision tree | 0.2256 | **0.9204** | 0.3621 |
| Engineered decision tree | **0.3552** | 0.9203 | **0.5119** |
| Neural network | 0.2279 | 0.8525 | 0.3593 |

**Selected model:** engineered decision tree

### Visual

Use:

`outputs/neural_network_comparison/figures/five_fold_neural_comparison.png`

### What to say

I compared two decision trees and a small feedforward neural network using the
same five-fold evaluation. The neural network was included to test whether a
more complex nonlinear model would improve failure prediction. The engineered
tree achieved the strongest average precision and F1 while maintaining almost
the same recall as the base tree. The neural network did not improve these
failure-focused metrics. Therefore, I retained the engineered tree because it
provided the best performance and was also easier to explain.

## Slide 9: Feature Relationships

### Put on the slide

Important original features:

- rotational speed
- torque
- tool wear

Simple relationships studied:

- temperature difference
- speed and torque together
- tool wear and torque together

**Finding:** Adding understandable feature relationships improved average F1
from **0.36 to 0.51** across five validation runs.

### Visual

Use:

`outputs/feature_relationships/feature_correlation_matrix.png`

### What to say

My professor suggested studying relationships between features instead of
looking at each feature separately. For example, the combination of speed and
torque may describe machine load better than either feature alone. Some useful
relationships had low simple correlation because the classifier could learn
nonlinear decision rules. I repeated the experiment on five data splits and
used the average result rather than depending on one split.

## Slide 10: Global Explanation With SHAP

### Put on the slide

### Explainable AI

- **SHAP:** shows how each feature pushes a prediction toward or away from
  failure.
- Main influential features: rotational speed, torque, and tool wear.
- The global chart summarizes average absolute influence across test records.

### Visual

Use:

`outputs/explainable_ai/figures/shap_global_importance.png`

### What to say

SHAP helped me understand overall feature influence and individual
predictions. The chart uses absolute contribution size, so it shows the amount
of influence but not its direction. SHAP explains what the model learned, but
it does not prove that a feature physically caused a failure or guarantee that
the prediction is correct.

## Slide 11: Local Explanation With LIME

### Put on the slide

- LIME creates nearby artificial records around one selected case.
- It asks the classifier for predictions and fits a weighted local surrogate.
- Positive and negative weights show which conditions support or oppose the
  local prediction.
- The displayed case is a false negative: an actual failure predicted as
  no failure.

**Fidelity finding:** repeated false-negative explanations had mean local
surrogate \(R^2 = 0.097\) and mean top-three feature overlap of 0.70.

### Visual

Use:

`outputs/explainable_ai/figures/local_false_negative_shap_lime.png`

### What to say

LIME provides a readable local explanation, but it is only an approximation of
the classifier around one record. I evaluated this approximation instead of
assuming it was reliable. Its low mean local R-squared for false-negative cases
shows weak fidelity, so I treat LIME as supporting evidence rather than ground
truth. This is itself a Responsible AI result: explanations must be evaluated,
not merely displayed.

## Slide 12: Threshold Tuning and Responsible AI

### Put on the slide

**The threshold decides when a failure alarm is raised.**

| Threshold | Correctly detected failures | False alarms | Missed failures |
|---|---:|---:|---:|
| 0.5 | 63 of 68 | 143 | 5 |
| 0.8 | 59 of 68 | 84 | 9 |

```text
Lower threshold -> catch more failures, but create more false alarms
Higher threshold -> fewer false alarms, but miss more failures
```

### Responsible AI connection

- Do not accept the default threshold without examining its consequences.
- Report both false alarms and missed failures.
- Choose the threshold according to the real maintenance context.
- Keep the system as decision support rather than an autonomous decision-maker.

### Visual

Use:

`outputs/model_tuning/threshold_tuning_precision_recall_f1.png`

### What to say

The classifier produces a score, and the threshold converts that score into a
failure or no-failure decision. At the default threshold of 0.5, the model
caught more failures but produced 143 false alarms. At 0.8, false alarms fell
to 84, but missed failures increased from 5 to 9. Threshold 0.8 is not always
better. A responsible choice depends on whether missing a failure or ordering
an unnecessary inspection has the more serious consequence. In a real system,
maintenance experts should help define that choice.

## Slide 13: Conclusion, Limitations, and Future Work

### Put on the slide

### Conclusion

- Anomaly detection and failure classification answer different questions.
- Classification detected most known failures but produced false alarms.
- The engineered tree performed better than the neural network on the main
  failure-focused metrics.
- Feature relationships improved the classification result.
- SHAP and LIME made model behavior easier to examine.
- Threshold tuning showed the tradeoff between false alarms and missed failures.
- Responsible evaluation requires more than accuracy.

### Limitations

- The dataset is synthetic.
- There is no time history for individual machines.
- No maintenance expert evaluated the explanations.

### Future work

- test on real machine data
- include time-series information
- obtain expert feedback on explanations

**Thank you. Questions?**

### What to say

The main conclusion is that unusual behavior and machine failure are not the
same problem. Combining classification with understandable feature analysis
provides more useful information. However, this remains a research prototype,
not a system ready for real industrial deployment.

## Simple Q&A Preparation

### Why did you use both anomaly detection and classification?

They answer different questions. Anomaly detection finds unusual records,
while classification finds patterns similar to labeled failures.

### Why is accuracy not enough?

Only 3.39% of records are failures. Accuracy is strongly influenced by the
large number of normal records, so recall, precision, and F1 are also needed.

### Why did the model produce many false alarms?

Failures are rare, and the model was designed to prioritize detecting them.
Catching more failures generally increases the number of normal records that
are incorrectly flagged.

### What is the difference between SHAP and LIME?

SHAP assigns feature contributions relative to a model baseline. LIME builds
a simple approximate model near one selected prediction.

### Does SHAP identify the physical cause of failure?

No. It explains how the trained model used the features. Physical causality
would require domain knowledge and additional experiments.

### What makes the project Responsible AI?

It reports important errors, prevents data leakage, explains predictions,
tests results on multiple splits, examines the consequences of changing the
decision threshold, communicates limitations, and does not recommend
autonomous use.

### Which threshold is the best?

There is no universally best threshold. A lower threshold prioritizes catching
failures, while a higher threshold reduces false alarms. The choice should be
based on maintenance costs, safety requirements, and expert judgment.

### Why did you include a neural network?

The neural network was an additional nonlinear comparison based on concepts
covered in the course. It tested whether additional complexity improved the
result. The engineered decision tree had better five-fold precision, recall,
and F1 and was easier to explain, so it remained the main model.

## Optional Backup Material

Keep these outside the main 13-slide presentation and show them only if a
question requires more detail:

1. Detailed neural-network architecture and learning curve.
2. ROC-AUC and PR-AUC comparison table.
3. True-positive, false-positive, false-negative, and true-negative SHAP/LIME
   examples.
4. Detailed explanation faithfulness and stability results.
5. Detailed error-cost analysis.

## Presentation Style

- Use one main image and no more than four or five bullets per slide.
- Explain results in ordinary language before mentioning technical terms.
- Do not show code or notebook screenshots in the main presentation.
- Do not memorize every number. Remember the meaning of the main results.
- It is acceptable to say that an advanced analysis was exploratory and that
  it would need more study before deployment.
- Aim for about 45-60 seconds per slide.

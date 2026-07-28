# Research Analysis Plan

## Purpose

This document explains how to move from basic model training to research-style analysis for the predictive maintenance project.

So far, the project has implemented:

- anomaly detection to identify unusual machine behavior
- failure classification to predict `Machine failure`
- simple explanations using anomaly feature deviation and classifier feature importance

The next phase is to interpret these results like an AI researcher: not just asking whether the model works, but asking what the model is learning, where it fails, what tradeoffs it creates, and how explainability and responsible AI can improve interpretation.

## 1. Main Research Questions

The project can now be organized around the following research questions:

### RQ1: Can anomaly detection identify failure-related abnormal behavior?

Evidence already available:

- anomaly detector flagged 113 of 2000 test records
- 16 of 68 actual failures were flagged as anomalous
- top anomalies were often explained by unusual rotational speed, torque, and tool wear

Research interpretation:

Anomaly detection captures some failure-related behavior, but it is not a complete failure detector. It is better understood as an early-warning or abnormal-behavior detection method.

### RQ2: Does supervised classification detect known failures more effectively?

Evidence already available:

- classifier accuracy: 0.9260
- failure precision: 0.3058
- failure recall: 0.9265
- confusion matrix: 63 failures detected, 5 failures missed, 143 false alarms

Research interpretation:

Classification is much better than anomaly detection at detecting known failure labels, but it creates many false alarms. This creates a precision-recall tradeoff.

### RQ3: Which sensor features are most important for abnormal or failure-related behavior?

Evidence already available:

- anomaly explanations often identify rotational speed, torque, and tool wear
- classifier feature importance also ranks rotational speed, torque, and tool wear highest

Research interpretation:

Both methods point toward similar machine behavior patterns. This consistency strengthens the explanation that rotational speed, torque, and tool wear are important indicators in this dataset.

### RQ4: What are the Responsible AI risks of using this system?

Risks:

- false negatives may miss actual failures
- false positives may cause unnecessary inspections
- synthetic data may not represent real industrial settings
- feature importance gives global explanation, but not full local explanation
- users may over-trust model outputs if limitations are not clearly communicated

Research interpretation:

The model should be framed as decision support, not as an autonomous maintenance decision-maker.

## 2. Research-Style Analysis Tasks

### Task 1: Error Analysis

Goal:

Understand where the classifier is wrong.

Analyze:

- false positives: predicted failure, actual no failure
- false negatives: predicted no failure, actual failure
- true positives: predicted failure, actual failure

Questions:

- What sensor patterns appear in false positives?
- What sensor patterns appear in false negatives?
- Are missed failures close to normal operating behavior?
- Are false alarms still unusual or potentially risky?

Why this matters:

Responsible AI requires knowing not only the average performance, but also the kinds of mistakes the system makes.

### Task 2: Threshold Analysis

Goal:

Study how changing the classification threshold affects precision and recall.

Default classification usually uses:

```text
failure probability >= 0.5 -> predicted failure
```

But this threshold can be changed.

Questions:

- What happens if the threshold is increased?
- Does precision improve?
- How much recall is lost?
- Which threshold is reasonable for predictive maintenance?

Why this matters:

This connects model design to real-world risk. A safety-focused system may prefer high recall, while a cost-sensitive system may prefer fewer false alarms.

### Task 3: Local Explanation

Goal:

Explain individual predictions, not only global feature importance.

Analyze a few examples:

- one true positive
- one false positive
- one false negative
- one true negative

Questions:

- Which features pushed the prediction toward failure?
- Does the explanation make sense?
- Are false positives understandable?
- Are false negatives concerning?

Why this matters:

Explainable AI is most useful when users can understand individual decisions.

### Task 4: Compare Anomaly Detection and Classification

Goal:

Understand how the two methods agree or disagree.

Create four groups:

- anomaly yes, classification failure
- anomaly yes, classification no failure
- anomaly no, classification failure
- anomaly no, classification no failure

Questions:

- When both methods agree, are predictions more trustworthy?
- When they disagree, what does that reveal?
- Do false positives from the classifier also look anomalous?
- Do missed failures have low anomaly scores?

Why this matters:

This helps explain the complementary roles of anomaly detection and classification.

### Task 5: Responsible AI Discussion

Goal:

Connect technical results to responsible deployment.

Discuss:

- transparency: explanations show important sensor features
- accountability: model outputs should support human decisions, not replace them
- robustness: false positives and false negatives must be analyzed
- uncertainty: threshold and probability outputs should be interpreted carefully
- limitation: synthetic data limits real-world generalization

Why this matters:

The project is not only about model performance. It is about responsible and explainable use of AI in maintenance decisions.

## 3. Recommended Next Implementation Order

1. Generate error analysis tables.
2. Add threshold analysis using predicted probabilities.
3. Add local explanation examples for selected records.
4. Compare anomaly detection and classification outputs.
5. Write a Responsible AI interpretation section.

## 4. Expected Final Research Claims

The final project should support claims like:

1. Anomaly detection can identify some abnormal operating behavior, but it should not be treated as a complete failure classifier.
2. Supervised classification detects most known failures but produces false alarms, creating a precision-recall tradeoff.
3. Rotational speed, torque, and tool wear are consistently important across anomaly detection and classification.
4. Explainability improves the usefulness of model outputs by connecting predictions to understandable sensor behavior.
5. Responsible use requires communicating limitations, analyzing errors, and treating the system as decision support rather than autonomous authority.


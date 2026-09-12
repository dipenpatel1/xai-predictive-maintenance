# Explainable Predictive Maintenance

## A Responsible AI Study of Anomaly Detection, Failure Classification, Feature Relationships, and Explanation Quality

**Final Course Project Report**  
**Course:** ENSE 861 - Responsible and Explainable Artificial Intelligence  
**Student:** Bijay Pandey  
**Student ID:** [Add Student ID]  
**Instructor:** [Add Instructor Name]  
**Institution:** University of Regina  
**Date:** August 2026

---

## Abstract

Predictive maintenance can help organizations identify equipment problems before they lead to costly downtime or unsafe operating conditions. A prediction alone, however, is not enough for responsible decision support. Maintenance users also need to know what the model recognized, why it raised an alert, what kinds of errors it makes, and how dependable its explanations are. This project develops a compact Responsible and Explainable AI workflow using the AI4I 2020 Predictive Maintenance Dataset. The workflow combines anomaly detection, supervised failure classification, feature-relationship analysis, a neural-network comparison, and local and global explanation.

The experiments show that unusual behavior and labeled failure are related but different concepts. Isolation Forest flagged 113 of 2,000 test records as anomalous, yet only 16 of the 68 actual failures were included. A balanced depth-4 decision tree detected 63 of 68 failures, producing recall of 0.9265, but also generated 143 false alarms and precision of 0.3058. Interpretable relationship features improved mean five-fold F1 from 0.3621 to 0.5119 while maintaining almost the same mean recall. A small feedforward neural network achieved the highest mean ROC-AUC, 0.9501, but did not outperform the engineered decision tree on precision, recall, F1, or PR-AUC.

SHAP and LIME were then used to inspect correct predictions, false alarms, and missed failures. Tree SHAP reconstructed model outputs with near-zero numerical error, while repeated LIME explanations had weak local surrogate fit for the audited error cases. The final result is not a deployment-ready maintenance system. It is a course-level research prototype showing that responsible predictive maintenance requires more than accuracy: it requires error analysis, understandable evidence, explanation evaluation, explicit decision tradeoffs, and honest communication of limitations.

**Keywords:** predictive maintenance, Responsible AI, Explainable AI, anomaly detection, failure classification, SHAP, LIME, neural network

---

## 1. Introduction

Industrial equipment does not usually fail at a convenient time. A breakdown can interrupt production, damage other components, increase maintenance cost, or create safety concerns. Predictive maintenance attempts to reduce this uncertainty by using operational data to identify machines that may need attention before a serious failure occurs.

The technical task appears simple: collect sensor readings and predict failure. The practical task is more demanding. A maintenance team must decide whether an alert justifies an inspection, whether repeated false alarms are wasting resources, and whether a missed failure creates unacceptable risk. If the model cannot communicate the basis of its output, users may either ignore a useful warning or place too much confidence in an unreliable one.

This project studies that problem using the AI4I 2020 dataset introduced by Matzka [1]. The dataset was designed as a synthetic but accessible representation of industrial predictive maintenance and has been used to connect failure prediction with explainability. The broader explainable predictive-maintenance literature similarly argues that useful systems should support both predictive performance and human understanding [2].

The project began with a narrow goal: detect abnormal machine behavior, classify failure, and explain the result. It later developed into a more research-oriented investigation. Instead of asking only whether a model could run successfully, the work examined whether anomaly and failure meant the same thing, whether sensor relationships improved repeated performance, whether a neural network justified its additional complexity, how false positives and false negatives were formed, and whether the explanation methods faithfully represented the model.

### 1.1 Research Problem

The central research problem is:

> How can a simple predictive-maintenance prototype combine anomaly detection, failure classification, feature relationships, and explanation in a workflow that is both technically useful and responsible about its limitations?

### 1.2 Research Questions

The investigation was guided by the following questions:

1. To what extent do anomalous machine records overlap with labeled failures?
2. Can supervised classification recognize known failure patterns more effectively than anomaly detection?
3. Which sensor measurements and sensor relationships most influence the classifier?
4. Does a small neural network provide enough improvement to justify its added complexity?
5. What can explanations reveal about correct alerts, false alarms, and missed failures?
6. How faithful and stable are the generated explanations?
7. How do threshold and error-cost assumptions change the preferred decision rule?

### 1.3 Project Contributions

This project does not propose a new machine-learning algorithm. Its contribution is an integrated and reproducible course-level study that:

- distinguishes unusual behavior from labeled failure using the same held-out records;
- compares a base decision tree, an engineered decision tree, and a neural network under common validation folds;
- evaluates performance using rare-class measures rather than accuracy alone;
- explains both correct and incorrect predictions using SHAP and LIME;
- evaluates explanation faithfulness and stability instead of treating explanation graphics as automatically trustworthy; and
- frames model selection as a decision-support problem involving false alarms, missed failures, interpretability, and documented limitations.

## 2. Related Work

Predictive maintenance combines condition monitoring with data-driven estimates of degradation or failure. Matzka's AI4I work provides the dataset and the immediate foundation for this project [1]. Cummins *et al.* survey explainable predictive maintenance and identify the continuing need for explanations that are technically sound and meaningful to maintenance users [2]. These sources motivated the use of a transparent tabular workflow rather than treating model accuracy as the only project outcome.

For abnormal-behavior detection, this project uses Isolation Forest. The method isolates observations through randomized tree partitions; records that require shorter paths to isolate are considered more unusual [3]. This approach is useful when the goal is to identify rare operating patterns, but it does not learn the meaning of a failure label.

The local explanation methods were selected from established Explainable AI research. LIME creates perturbed samples around one record and fits a simple weighted surrogate to approximate the model in that local neighborhood [4]. SHAP represents a model output as a baseline plus additive feature contributions [5]. Because explanation methods can vary under small changes in their inputs or sampling process, robustness is itself an important evaluation property [6]. More generally, interpretability should be evaluated against a defined purpose rather than assumed from the existence of an explanation [7].

The Responsible AI framing follows the NIST AI Risk Management Framework, particularly its emphasis on validity, reliability, transparency, explainability, accountability, and risk management across the AI lifecycle [8]. Machine-learning safety research also warns that test-set performance is not the same as system-level safety, especially when a model encounters conditions outside its development environment [9]. In high-consequence settings, Rudin argues that interpretable models should be preferred when they perform adequately, rather than automatically replacing them with black boxes and post-hoc explanations [10]. This informed the decision to retain a shallow tree when the neural network did not provide a clear performance advantage. Guidotti *et al.* provide broader terminology for local, global, model-specific, and model-agnostic explanation methods [11].

The neural-network experiment was included as a nonlinear comparison. Its learning process follows backpropagation, in which output error is propagated through earlier layers to update model weights [12]. Adam was used as the optimizer, providing adaptive gradient-based updates during training [13].

## 3. Dataset and Data Preparation

### 3.1 AI4I 2020 Dataset

The AI4I 2020 dataset contains 10,000 independent machine records and 14 columns [1]. It is synthetic, meaning that it was designed to resemble industrial conditions rather than collected from a live factory. This makes it appropriate for a controlled course project, but it also limits how confidently the results can be transferred to real equipment.

The six model inputs are shown in Table 1.

**Table 1. Model input features**

| Feature | Type | Role in the study |
|---|---|---|
| Machine type | Categorical | Product-quality variant (`L`, `M`, or `H`) |
| Air temperature | Numerical | Ambient thermal condition |
| Process temperature | Numerical | Process thermal condition |
| Rotational speed | Numerical | Machine operating speed |
| Torque | Numerical | Mechanical load indicator |
| Tool wear | Numerical | Accumulated wear measure |

The target is `Machine failure`, where 1 indicates failure and 0 indicates no failure. The dataset also contains five specific failure-mode columns: tool-wear failure, heat-dissipation failure, power failure, overstrain failure, and random failure. These columns were retained only for descriptive analysis. They were excluded from model inputs because they contain outcome-related information and would create target leakage.

### 3.2 Data Quality and Class Imbalance

No missing values were found. The target distribution was strongly imbalanced: 9,661 records represented normal operation and 339 represented failure. The failure rate was therefore 3.39%.

This imbalance is central to the evaluation. A model predicting “no failure” for every record would achieve 96.61% accuracy while detecting no failures. Accuracy was reported for completeness, but precision, recall, F1, PR-AUC, and the confusion matrix were treated as more informative measures of failure-class performance.

### 3.3 Leakage-Safe Split and Transformation

The data was divided into an 80% training set and a 20% test set using a fixed random seed of 42. Stratification preserved the failure ratio: the training set contained 8,000 records with a 3.39% failure rate, while the test set contained 2,000 records with a 3.40% failure rate, including 68 failures.

Machine type was one-hot encoded. Numerical features were standardized where required. All transformations were fitted on training data and then applied to the test set. The test labels were not used for model fitting, preprocessing, or neural-network threshold selection.

## 4. Methodology

### 4.1 Experimental Workflow

The implementation followed a sequential workflow:

```text
Data exploration
      ->
Leakage-safe preprocessing
      ->
Anomaly detection
      ->
Failure classification
      ->
Feature-relationship analysis
      ->
Neural-network comparison
      ->
SHAP and LIME explanation
      ->
Explanation and decision-tradeoff evaluation
```

Each stage answered a different question. Anomaly detection examined unusualness, supervised models learned labeled failure, feature engineering tested sensor interactions, and explanation analysis investigated model behavior at both global and individual levels.

### 4.2 Anomaly Detection

Isolation Forest [3] was fitted using only the 7,729 normal records in the training set. The 271 training failures were excluded so that the anomaly detector learned a representation of normal operation. The contamination parameter was set to 0.05, and the random seed was fixed at 42.

For each test record, the model produced an anomaly score and a binary flag. Lower scores indicated greater abnormality. The failure labels were used only afterward to measure overlap; they were not used to train the anomaly detector.

A simple deviation explanation was also created for highly anomalous records. Each numerical value was compared with the mean and standard deviation of normal training records. The three largest absolute standardized deviations were reported. This explanation describes why a record differs from normal data; it does not explain a failure mechanism.

### 4.3 Baseline Failure Classifier

The baseline supervised model was a decision tree with maximum depth 4, balanced class weights, and random seed 42. Balanced weighting increased the training influence of rare failure examples. The shallow depth was selected to limit overfitting and retain inspectable decision behavior.

The tree was fitted inside a preprocessing pipeline. At the default threshold of 0.5, its failure scores were converted to failure or no-failure labels. Performance was measured through accuracy, precision, recall, F1, PR-AUC, ROC-AUC, and the confusion matrix.

### 4.4 Feature Relationships and Engineered Tree

Pairwise correlation was calculated to study linear associations. Correlation alone was not treated as a complete test of usefulness because a decision tree can learn thresholds and conditional relationships that have weak linear correlation.

Four interpretable relationship features were created:

**Table 2. Engineered relationship features**

| Feature | Definition | Intended interpretation |
|---|---|---|
| Temperature difference | Process temperature - air temperature | Difference between process and ambient thermal conditions |
| Power proxy | Torque x rotational speed | Approximate combined speed-load condition |
| Wear-torque interaction | Tool wear x torque | Wear under mechanical load |
| Speed-torque ratio | Rotational speed / (torque + epsilon) | Relative speed-load condition |

The power feature is intentionally called a proxy. The dataset does not provide the units and physical context needed to claim an exact mechanical-power equation. The base and engineered trees used the same depth, class weighting, split, and preprocessing pattern so that the comparison focused on the added relationships.

### 4.5 Neural-Network Comparison

A small feedforward neural network was implemented with transformed input features, hidden layers of 16 and 8 ReLU units, and one binary output. The model contained 289 trainable parameters. It used balanced sample weights, L2 regularization (`alpha = 0.001`), an initial learning rate of 0.001, Adam optimization [13], early stopping, a maximum of 500 iterations, and random seed 42. Training stopped after 162 iterations.

The network was evaluated at threshold 0.5. A second threshold, 0.86, was selected by maximizing F1 on out-of-fold predictions generated from the training data only. The selected-threshold result was kept separate because changing the threshold changes the decision rule, not the learned network.

For a fair model comparison, the base tree, engineered tree, and neural network were evaluated under the same stratified five-fold partitions. Mean and standard deviation were recorded for accuracy, precision, recall, F1, PR-AUC, and ROC-AUC.

### 4.6 Explainability

Global tree importance was first calculated from impurity reduction. This shows how strongly the fitted tree relied on features overall, but not the direction of influence for an individual prediction.

Tree SHAP [5] was then used to calculate additive contributions for every test record. Mean absolute SHAP values provided global influence rankings, while signed local values showed whether each feature pushed one prediction toward failure or no failure.

LIME [4] generated a second local explanation by sampling around each selected record and fitting a weighted linear surrogate. One example was selected from each confusion-matrix outcome:

- true positive: a correctly detected failure;
- false positive: a false maintenance alarm;
- false negative: a missed failure; and
- true negative: a correctly recognized normal record.

### 4.7 Explanation Evaluation

SHAP faithfulness was measured by reconstructing the classifier output from the SHAP baseline and feature contributions for all 2,000 test records. Reconstruction error measures whether the explanation correctly decomposes the fitted model; it does not measure whether the model itself is correct.

LIME was repeated across five random seeds for a selected false positive and false negative. Two properties were measured:

1. local surrogate R2, indicating how closely LIME reproduced the classifier in the sampled neighborhood; and
2. top-three Jaccard overlap, indicating whether repeated explanations selected the same important features.

These checks reflect the need to evaluate explanation robustness and purpose rather than assuming that a readable explanation is reliable [6], [7].

### 4.8 Responsible Decision Analysis

Threshold sensitivity was examined from 0.1 to 0.9. In addition, false positives were assigned a relative cost of 1 while false-negative cost was varied from 1 to 50. The analysis used:

`Relative cost = false positives + cost ratio x false negatives`

These ratios were hypothetical sensitivity values, not measured industrial costs. Their purpose was to show how operational priorities can change the preferred model configuration. A real deployment would require maintenance and safety experts to define acceptable errors.

## 5. Results

### 5.1 Anomaly Detection

The anomaly detector flagged 113 of the 2,000 test records. Table 3 compares these flags with the known failure labels.

**Table 3. Anomaly detection results**

| Test outcome | Count |
|---|---:|
| Actual failures | 68 |
| Records flagged as anomalous | 113 |
| Failures also flagged as anomalous | 16 |
| Normal records flagged as anomalous | 97 |
| Failures not flagged as anomalous | 52 |

Only 16 of 68 failures were anomalous, while 97 normal records were flagged. Figure 1 shows substantial overlap between normal and failure anomaly-score distributions. The result demonstrates that unusual behavior is neither necessary nor sufficient for a labeled failure.

![Figure 1. Distribution of anomaly scores for normal and failure records. Lower scores indicate more unusual records.](../outputs/report_artifacts/anomaly_score_distribution.png)

### 5.2 Baseline Classification

The balanced decision tree detected most failures but produced many false alarms.

**Table 4. Baseline decision-tree performance**

| Metric | Value |
|---|---:|
| Accuracy | 0.9260 |
| Precision | 0.3058 |
| Recall | 0.9265 |
| F1 | 0.4599 |
| PR-AUC | 0.5246 |
| ROC-AUC | 0.9538 |

The confusion matrix in Figure 2 contains 1,789 true negatives, 143 false positives, 5 false negatives, and 63 true positives. Recall of 0.9265 means that 63 of the 68 real failures were detected. Precision of 0.3058 means that fewer than one in three failure alerts was correct.

![Figure 2. Confusion matrix for the baseline decision tree at threshold 0.5.](../outputs/report_artifacts/classification_confusion_matrix.png)

### 5.3 Baseline Feature Importance

The tree relied mainly on rotational speed, torque, and tool wear.

**Table 5. Baseline decision-tree feature importance**

| Feature | Importance |
|---|---:|
| Rotational speed | 0.3543 |
| Torque | 0.3460 |
| Tool wear | 0.2673 |
| Air temperature | 0.0294 |
| Process temperature | 0.0032 |

Machine-type indicators had zero importance in this fitted depth-4 tree. This does not prove that machine type is universally irrelevant; it means the specific fitted tree did not use those indicators in its splits.

### 5.4 Feature Relationships

The holdout comparison is shown in Table 6.

**Table 6. Base and engineered-tree holdout results**

| Model | Precision | Recall | F1 | False positives | False negatives |
|---|---:|---:|---:|---:|---:|
| Base tree | 0.3058 | 0.9265 | 0.4599 | 143 | 5 |
| Engineered tree | 0.3509 | 0.8824 | 0.5021 | 111 | 8 |

On this split, relationship features reduced false positives and increased F1, but missed three additional failures. Repeated validation provided stronger evidence. Across five folds, the engineered tree increased mean precision from 0.2256 to 0.3552 and mean F1 from 0.3621 to 0.5119 while mean recall remained nearly unchanged at approximately 0.92.

The engineered tree relied most on speed-torque ratio (0.3413), power proxy (0.3172), tool wear (0.1519), and temperature difference (0.1075). Figure 3 summarizes the repeated model comparison.

![Figure 3. Five-fold comparison of the base tree, engineered tree, and neural network.](../outputs/neural_network_comparison/figures/five_fold_neural_comparison.png)

### 5.5 Neural-Network Comparison

Table 7 reports held-out results at the common threshold and at the neural threshold selected from training-only out-of-fold predictions.

**Table 7. Held-out supervised-model comparison**

| Configuration | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base tree | 0.50 | 0.3058 | 0.9265 | 0.4599 | 0.5246 | 0.9538 | 143 | 5 |
| Engineered tree | 0.50 | 0.3509 | 0.8824 | 0.5021 | 0.7918 | 0.9255 | 111 | 8 |
| Neural network | 0.50 | 0.2711 | 0.8971 | 0.4164 | 0.6185 | 0.9618 | 164 | 7 |
| Neural network, selected threshold | 0.86 | 0.5281 | 0.6912 | 0.5987 | 0.6185 | 0.9618 | 42 | 21 |

At threshold 0.5, the neural network did not outperform the engineered tree on the failure-focused balance. Raising its threshold to 0.86 improved precision and F1 but increased missed failures from 7 to 21. This is a different operational choice rather than evidence that the network learned a better representation.

Across five folds, the neural network achieved mean precision 0.2279, recall 0.8525, F1 0.3593, PR-AUC 0.5648, and ROC-AUC 0.9501. Although it had the highest mean ROC-AUC, the engineered tree was stronger on precision, recall, F1, and PR-AUC. The engineered tree was therefore retained as the main model.

### 5.6 Agreement Between Anomaly and Classification Signals

Applying both methods to the same test records produced the groups in Table 8.

**Table 8. Agreement between anomaly and classification signals**

| Signal group | Records | Failures | Observed failure rate |
|---|---:|---:|---:|
| Neither | 1,716 | 4 | 0.23% |
| Classifier only | 171 | 48 | 28.07% |
| Anomaly only | 78 | 1 | 1.28% |
| Both | 35 | 15 | 42.86% |

Records flagged by both methods had the highest observed failure rate. However, the classifier-only group contained 48 failures, confirming that many known failure patterns were not globally unusual. Four failures received neither signal, so the absence of flags cannot be interpreted as guaranteed safe operation.

### 5.7 Global and Local Explanations

Global SHAP importance again identified rotational speed, torque, and tool wear as the main influences, with mean absolute contributions of 0.1736, 0.1606, and 0.1472, respectively.

![Figure 4. Global SHAP importance for the baseline failure classifier.](../outputs/explainable_ai/figures/shap_global_importance.png)

Local explanations made individual outcomes more concrete:

- **True positive:** Dataset row 4,851 was correctly classified as failure with score 0.9661. Rotational speed contributed +0.4132 toward failure, while torque contributed -0.0922 in the opposite direction.
- **False positive:** Dataset row 2,997 received a failure score of 0.8637 but did not fail. Torque (+0.3708) and rotational speed (+0.1350) strongly supported the false alert.
- **False negative:** Dataset row 9,174 failed but received score 0.0308. Rotational speed (-0.2597) and torque (-0.1295) pushed the model toward no failure.
- **True negative:** Dataset row 4,871 was correctly classified as no failure with score 0.0308.

Figure 5 shows the missed-failure explanation. This error is operationally important because a high-confidence no-failure output could prevent a needed inspection.

![Figure 5. SHAP and LIME explanations for a false negative. Positive values support failure; negative values support no failure.](../outputs/explainable_ai/figures/local_false_negative_shap_lime.png)

### 5.8 Explanation Quality

Tree SHAP reconstructed all 2,000 model outputs with mean absolute error `3.11 x 10^-14` and maximum absolute error `3.24 x 10^-14`. The near-zero error shows that the contributions faithfully decompose this fitted tree's score.

The selected LIME explanations were less dependable.

**Table 9. Repeated LIME explanation audit**

| Error case | Mean local surrogate R2 | Mean top-3 overlap | Minimum top-3 overlap |
|---|---:|---:|---:|
| False positive | 0.1115 | 1.00 | 1.00 |
| False negative | 0.0970 | 0.70 | 0.50 |

The false-positive feature set was stable, but both surrogate-fit values were low. The false-negative feature set also changed across runs. LIME was therefore treated as an approximate second view rather than an exact explanation of the tree.

### 5.9 Threshold and Cost Sensitivity

At threshold 0.5, the baseline tree detected 63 failures, produced 143 false positives, and missed 5 failures. At threshold 0.8, it detected 59 failures, produced 84 false positives, and missed 9 failures. Raising the threshold removed 59 false alarms but lost 4 correctly detected failures.

The hypothetical cost analysis showed that the preferred configuration changed with the assumed relative cost of a missed failure:

- ratios 1-5 favored the classifier with failure weight 5;
- ratios 6-14 favored the balanced classifier at threshold 0.8; and
- ratios 15-50 favored the balanced classifier at threshold 0.5.

No threshold is universally responsible. A lower threshold may be appropriate when missed failures carry severe consequences, while a higher threshold may be preferred when unnecessary shutdowns or inspections are costly. Domain experts must define these priorities.

## 6. Discussion

### 6.1 Anomaly Is Not Failure

The first research question produced one of the clearest findings. Anomaly detection and classification should not be treated as interchangeable. Isolation Forest identified rare operating patterns, while the classifier recognized patterns associated with known labels. Only 16 of 68 failures were anomalous, and 48 failures appeared in the classifier-only group.

Using both methods remains meaningful because they provide different evidence. Agreement identified a smaller group with a higher observed failure rate, while anomaly-only records may still deserve investigation for unfamiliar behavior. The anomaly flag should therefore be presented as supporting context, not as a failure diagnosis.

### 6.2 Rare-Class Performance Matters More Than Accuracy

The baseline classifier's 92.6% accuracy appears strong until it is considered alongside 30.58% precision. The model found most failures, but most alerts were false. In a real maintenance environment, this could consume inspection resources and eventually encourage users to dismiss warnings.

At the same time, the five missed failures cannot be ignored. A model may be useful for screening while still being unsafe as an autonomous decision-maker. Reporting both error types makes the operational tradeoff visible.

### 6.3 Sensor Relationships Added Useful Structure

The engineered tree's improvement across five folds suggests that relationships between measurements contain useful predictive structure. The repeated result is stronger evidence than the single holdout comparison because it is less dependent on one partition.

The interpretation must remain modest. Speed-torque ratio and power proxy are useful representations for this model; they are not newly discovered laws of equipment failure. Establishing a physical equation would require units, machine specifications, temporal measurements, and domain validation.

### 6.4 Greater Complexity Did Not Guarantee a Better Model

The neural network ranked records well but did not provide the best rare-class operating performance. The engineered tree was also easier to inspect through shallow rules and Tree SHAP. Consistent with the argument for interpretable models in consequential settings [10], the neural network's additional complexity was not justified by a clear practical gain.

This result should not be generalized to all neural networks. AI4I is small, tabular, synthetic, and non-temporal, and only one deliberately compact architecture was tested. The value of the experiment is that model choice was based on common validation evidence rather than an assumption that deep learning must be superior.

### 6.5 Explanation Is Evidence About the Model, Not the Machine

Local explanation was most useful when applied to errors. The false positive showed which learned conditions created a confident but unnecessary alert. The false negative revealed a repeated high-speed, low-torque pattern that pushed missed failures away from the failure class.

SHAP was faithful to the tree calculation, but this does not prove that the tree's reasoning is physically correct. LIME generated readable local weights, but its low surrogate fit showed that readability and faithfulness are different qualities. A responsible report should communicate both the explanation and the strength of evidence behind it.

## 7. Responsible and Explainable AI Analysis

### 7.1 Transparency and Explainability

The system reports global feature reliance, signed local contributions, example error explanations, and the limitations of each explanation method. A human-readable explanation is structured around five questions:

1. What did the model predict?
2. What evidence supported the prediction?
3. What evidence opposed it?
4. What human action is appropriate?
5. What does the explanation not establish?

This structure makes the output more useful than a score alone without presenting the explanation as causal truth.

### 7.2 Validity, Reliability, and Robustness

Validity was supported through a held-out test set, stratified five-fold validation, multiple rare-class metrics, and explicit leakage prevention. Robustness was explored through variation across folds, threshold sensitivity, and repeated LIME explanations. These checks align with the NIST AI RMF view that trustworthy AI requires context-aware measurement and ongoing risk management rather than a single performance value [8].

### 7.3 Accountability and Human Oversight

The prototype is framed as decision support. A qualified maintenance professional should retain authority over inspections, shutdowns, and repairs. The report documents data assumptions, model settings, error counts, explanation limitations, and hypothetical cost assumptions so that model behavior can be reviewed rather than accepted silently.

### 7.4 Consequence-Aware Evaluation

False positives and false negatives have different consequences. False positives may create unnecessary inspections, downtime, and alert fatigue. False negatives may leave a failing machine in operation. Threshold and cost-sensitivity analyses make this value judgment explicit instead of hiding it behind the default threshold.

### 7.5 Fairness Scope

The dataset contains machine measurements rather than people or protected demographic attributes. Demographic fairness metrics are therefore not meaningful for this experiment. Responsible evaluation instead focuses on operational error distribution, transparency, reliability, and safety. A future system affecting workers, teams, or resource allocation would require a broader sociotechnical fairness assessment.

### 7.6 Governance Boundary

The prototype demonstrates selected Responsible AI engineering practices, but it does not establish industrial compliance or production readiness. No maintenance expert approved the explanations, no formal hazard analysis was completed, and no monitoring process was deployed. Test performance and model faithfulness should not be confused with system safety [9].

## 8. Limitations

The evidence in this report is bounded by several limitations:

1. **Synthetic data:** AI4I represents simulated conditions rather than measurements from deployed industrial equipment [1].
2. **No machine history:** Records are independent and do not provide temporal degradation trajectories.
3. **Limited model search:** One shallow tree design and one deliberately small neural architecture were emphasized.
4. **No probability calibration:** Model scores should not be interpreted as verified real-world failure probabilities.
5. **Small failure sample:** The test set contains 68 failures and only 5 false negatives, limiting conclusions about error subgroups.
6. **Post-hoc explanation limits:** SHAP explains the fitted model, not physical causality. LIME showed weak local fit for audited cases.
7. **No expert evaluation:** Maintenance engineers did not evaluate whether the explanations were understandable or actionable.
8. **Hypothetical costs:** The cost ratios were sensitivity assumptions, not measured industrial values.
9. **No deployment monitoring:** Data drift, sensor failure, cybersecurity, runtime anomaly handling, and retraining governance were outside the project scope.

These limitations do not remove the value of the findings. They define what the experiment can support: a reproducible course-level analysis of model behavior, not a claim of industrial safety.

## 9. Future Work

The most valuable next steps are:

- evaluate the workflow on real temporal machine data, such as run-to-failure histories;
- calibrate model scores before communicating them as probabilities;
- validate thresholds using maintenance costs and safety priorities supplied by domain experts;
- compare the shallow tree with additional interpretable models and carefully tuned nonlinear baselines;
- conduct a user study with maintenance professionals to evaluate explanation clarity and usefulness;
- monitor feature distributions, model errors, and explanation behavior over time; and
- define a formal human-review and audit process for high-risk or conflicting cases.

## 10. Conclusion

This project developed an end-to-end Responsible and Explainable AI study for predictive maintenance. The work began with a simple question about abnormal machine behavior and developed into a broader investigation of failure prediction, model comparison, sensor relationships, error consequences, and explanation quality.

The experiments showed that anomaly detection and failure classification provide different information. Isolation Forest identified unusual operation but missed most labeled failures. The decision tree detected 63 of 68 failures, although its 143 false alarms exposed an important precision problem. Interpretable sensor relationships improved repeated F1 without reducing mean recall. The neural network offered strong ranking performance but did not outperform the engineered tree on the main failure-focused measures.

The explanation analysis added a second layer of accountability. SHAP made the tree's global and local behavior inspectable and reconstructed its outputs faithfully. LIME provided an understandable alternative view, but its weak local fit demonstrated why explanation methods must be evaluated rather than displayed uncritically. Examining false alarms and missed failures made the model's limitations more concrete than aggregate metrics alone.

The main lesson is that responsible predictive maintenance is not achieved by adding an explanation chart after training. It comes from the whole workflow: leakage-safe data preparation, suitable metrics, repeated validation, interpretable model comparison, explicit error tradeoffs, explanation audits, human decision authority, and honest limits. The resulting prototype is not ready to control industrial maintenance, but it demonstrates a defensible way to study how predictive performance and human understanding can be developed together.

## References

[1] S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," in *2020 Third International Conference on Artificial Intelligence for Industries (AI4I)*, 2020, pp. 69-74, doi: 10.1109/AI4I49448.2020.00023.

[2] L. Cummins, A. Sommers, S. B. Ramezani, S. Mittal, J. E. Jabour, M. Seale, and S. Rahimi, "Explainable Predictive Maintenance: A Survey of Current Methods, Challenges and Opportunities," *IEEE Access*, vol. 12, pp. 57574-57602, 2024, doi: 10.1109/ACCESS.2024.3391130.

[3] F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation Forest," in *Proc. 8th IEEE Int. Conf. Data Mining (ICDM)*, Pisa, Italy, 2008, pp. 413-422, doi: 10.1109/ICDM.2008.17.

[4] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why Should I Trust You?': Explaining the Predictions of Any Classifier," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, San Francisco, CA, USA, 2016, pp. 1135-1144, doi: 10.1145/2939672.2939778.

[5] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Advances in Neural Information Processing Systems 30*, Long Beach, CA, USA, 2017, pp. 4765-4774.

[6] D. Alvarez-Melis and T. S. Jaakkola, "On the Robustness of Interpretability Methods," in *Proc. ICML Workshop on Human Interpretability in Machine Learning*, Stockholm, Sweden, 2018. [Online]. Available: https://arxiv.org/abs/1806.08049.

[7] F. Doshi-Velez and B. Kim, "Towards a Rigorous Science of Interpretable Machine Learning," arXiv:1702.08608, 2017. [Online]. Available: https://arxiv.org/abs/1702.08608.

[8] E. Tabassi, *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*. Gaithersburg, MD, USA: National Institute of Standards and Technology, NIST AI 100-1, Jan. 2023, doi: 10.6028/NIST.AI.100-1.

[9] S. Mohseni, H. Wang, C. Xiao, Z. Yu, Z. Wang, and J. Yadawa, "Taxonomy of Machine Learning Safety: A Survey and Primer," *ACM Comput. Surv.*, vol. 55, no. 8, Art. no. 157, pp. 1-38, 2023, doi: 10.1145/3551385.

[10] C. Rudin, "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead," *Nature Machine Intelligence*, vol. 1, pp. 206-215, 2019, doi: 10.1038/s42256-019-0048-x.

[11] R. Guidotti, A. Monreale, S. Ruggieri, F. Turini, F. Giannotti, and D. Pedreschi, "A Survey of Methods for Explaining Black Box Models," *ACM Comput. Surv.*, vol. 51, no. 5, Art. no. 93, pp. 1-42, 2018, doi: 10.1145/3236009.

[12] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning Representations by Back-Propagating Errors," *Nature*, vol. 323, pp. 533-536, 1986, doi: 10.1038/323533a0.

[13] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *Proc. 3rd Int. Conf. Learning Representations*, San Diego, CA, USA, 2015. [Online]. Available: https://arxiv.org/abs/1412.6980.

---

## Appendix A. Reproducibility and Project Artifacts

The analysis is organized in the following notebooks:

1. `src/01_data_exploration/data_exploration.ipynb`
2. `src/02_data_preprocessing/data_preprocessing.ipynb`
3. `src/03_anomaly_detection/anomaly_detection.ipynb`
4. `src/04_failure_classification/failure_classification.ipynb`
5. `src/05_neural_network/neural_network_comparison.ipynb`
6. `src/06_feature_relationship_analysis/feature_relationship_analysis.ipynb`
7. `src/07_explainable_ai/lime_shap_explainability.ipynb`
8. `src/08_integrated_research_analysis/integrated_research_analysis.ipynb`

The project uses Python, pandas, NumPy, scikit-learn, Matplotlib, seaborn, SHAP, and LIME. Generated tables and figures are stored under `outputs/`. A complete reproducibility guide is available in `docs/reproducibility_guide.md`.


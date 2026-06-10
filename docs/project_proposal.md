# Project Proposal

## Title

**Human-in-the-Loop Anomaly Detection and Explainable Failure Classification for Predictive Maintenance**

## 1. Abstract

This project proposes a small Responsible AI prototype for predictive maintenance using the AI4I 2020 Predictive Maintenance Dataset. The system will detect abnormal machine behavior, classify whether a machine record indicates failure or no failure, and explain the main sensor features that influenced the result. A simple human-in-the-loop process will also be included, where selected uncertain, abnormal, or high-risk cases are reviewed by a simulated maintenance expert. The feedback from this review will be added back to the training data to retrain the model and compare performance before and after feedback. The goal is to demonstrate how explainability, human oversight, accountability, and continuous improvement can be included in a practical machine learning workflow.

## 2. Introduction and Problem Statement

Industrial equipment failure can lead to downtime, safety concerns, production delays, and unnecessary maintenance costs. Predictive maintenance uses machine sensor data to identify abnormal behavior and predict equipment failure before serious problems occur, so that maintenance decisions can be made earlier and more effectively [2].

This project narrows predictive maintenance to a simple and understandable application: detecting abnormal machine behavior and classifying whether a machine is likely to fail using a tabular industrial sensor dataset. The system will not be designed as a fully autonomous maintenance solution. Instead, it will act as a decision-support tool that provides predictions, explanations, and a human feedback mechanism. This follows the idea that explainable predictive maintenance should support human understanding and trust, especially when predictions may influence practical maintenance decisions [2].

The project is based on the AI4I 2020 Predictive Maintenance Dataset introduced by Matzka [1], which connects predictive maintenance with explainable artificial intelligence. It is also supported by Cummins et al. [2], who survey explainable predictive maintenance and emphasize that human operators need understandable and trustworthy outputs when predictive systems are used in serious applications.

Many machine learning systems focus mainly on predictive accuracy, but Responsible AI systems should also support transparency, human oversight, and accountability. In a maintenance setting, users need to know whether a machine's behavior is unusual, why the model made a prediction, and when a human expert should review the result. This need for explanation is consistent with explainable predictive maintenance research, which argues that interpretability can help users trust and evaluate predictive systems [1], [2].

This project asks:

> How can a simple predictive maintenance prototype combine anomaly detection, failure classification, explanation, and human feedback in a Responsible AI workflow?

## 3. Project Objectives

The main objectives of this project are:

- to use a simple predictive maintenance dataset for a fixed application
- to detect abnormal machine behavior using anomaly detection
- to classify machine records as failure or no failure using a simple classification model
- to explain model outcomes using interpretable feature information
- to simulate human review for uncertain, abnormal, or high-risk cases
- to retrain the classifier using corrected human feedback
- to compare model performance before and after feedback

## 4. Dataset

The project will use the **AI4I 2020 Predictive Maintenance Dataset**. This dataset is suitable because it is tabular, understandable, and directly related to predictive maintenance [1]. It includes machine operating features such as machine type, air temperature, process temperature, rotational speed, torque, tool wear, and a machine failure label.

The main classification target will be `Machine failure`. The sensor features will be used for both anomaly detection and failure classification.

## 5. Proposed Methodology

The system will follow this workflow:

```text
AI4I Sensor Data
      ↓
Data Preprocessing
      ↓
Anomaly Detection
      ↓
Failure Classification
      ↓
Explanation of Outcome
      ↓
Human Review and Feedback
      ↓
Model Retraining
```

First, the dataset will be cleaned and prepared for machine learning. Numerical sensor features will be selected, categorical features such as machine type may be encoded, and the data will be split into training and testing sets. Class imbalance will also be checked because machine failures are usually less common than normal operation.

Second, anomaly detection will be performed using a simple model such as Isolation Forest. This step will identify records that appear unusual compared with normal machine behavior. These anomaly flags can help decide which cases should be reviewed by a human.

Third, a simple classifier such as Logistic Regression or Decision Tree Classifier will be trained to predict failure or no failure. These models are appropriate for this project because they are easier to interpret than complex deep learning models.

Fourth, the system will explain its outcomes. For Logistic Regression, explanations can be based on feature coefficients and feature values. For a Decision Tree, explanations can be based on the decision path and feature importance. The explanation should show, for example, whether high tool wear, high torque, unusual rotational speed, or temperature values influenced the prediction. This explanation step is included because previous work on explainable predictive maintenance highlights the importance of making model outputs understandable to human users [1], [2].

Finally, selected cases will be sent to a simulated human expert. Cases may be escalated when the record is anomalous, the classifier predicts failure, the confidence is low, or the anomaly result and classifier result disagree. The simulated expert may confirm or correct the prediction. Corrected labels will then be added back to the training data, and the classifier will be retrained.

## 6. Responsible and Explainable AI Contribution

This project demonstrates both Responsible AI and Explainable AI in a practical way. The Explainable AI contribution comes from providing understandable reasons for model outcomes, such as showing which sensor features influenced a failure prediction or anomaly flag. For example, the system may explain that high tool wear, high torque, or unusual rotational speed contributed to a risky prediction.

The Responsible AI contribution comes from how the system is designed and evaluated. Transparency is supported through explanations, while human oversight is included by routing selected anomalous, uncertain, or high-risk cases to simulated expert review instead of accepting every model output automatically. Accountability is represented through the feedback step, where the simulated human reviewer can confirm or correct the system output. Robustness is considered by evaluating false positives and false negatives, since both types of errors have practical consequences in predictive maintenance. Continuous improvement is shown by retraining the classifier after corrected feedback is added back to the training data.

## 7. Evaluation Plan

The classification model will be evaluated using accuracy, precision, recall, F1-score, and a confusion matrix. Recall is important because missing a real failure may be costly or unsafe, while precision is important because too many false alarms may reduce trust in the system.

The anomaly detection component will be evaluated by checking whether anomaly flags correspond to unusual sensor values and whether they overlap with failure cases. The human-in-the-loop component will be evaluated by measuring how many cases are escalated, how many corrected labels are added, and whether retraining improves performance after feedback.

## 8. Expected Outcomes

The expected outcome is a working prototype that can load and preprocess the AI4I 2020 dataset, detect anomalous machine records, classify records as failure or no failure, and explain the main reasons for each prediction. The system is also expected to simulate human review for selected cases, use corrected feedback for retraining, and compare model performance before and after feedback. Overall, the project should show how a simple machine learning system can be made more responsible through explanation, human oversight, and feedback-based improvement.

## 9. Tools and Technologies

The project may use Python, pandas, NumPy, scikit-learn, matplotlib or seaborn, and possibly SHAP as an optional explainability extension. The demonstration may be implemented as a notebook, command-line program, or simple interface depending on the final project requirements.

## 10. Scope and Limitations

This project is a course-level prototype, not a production-ready industrial system. The AI4I dataset is synthetic, and the human feedback process will be simulated rather than collected from real maintenance engineers. The project will focus on simple and explainable models rather than complex deep learning methods. These limitations keep the project manageable while still demonstrating the main Responsible AI ideas.

## 11. References

[1] S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," in *2020 Third International Conference on Artificial Intelligence for Industries (AI4I)*, 2020, pp. 69-74, doi: 10.1109/AI4I49448.2020.00023.

[2] L. Cummins, A. Sommers, S. B. Ramezani, S. Mittal, J. E. Jabour, M. Seale, and S. Rahimi, "Explainable Predictive Maintenance: A Survey of Current Methods, Challenges and Opportunities," *IEEE Access*, vol. 12, pp. 57574-57602, 2024, doi: 10.1109/ACCESS.2024.3391130.

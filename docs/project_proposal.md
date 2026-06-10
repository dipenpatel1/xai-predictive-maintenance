# Project Proposal

## Title

**Anomaly Detection and Explainable Failure Classification for Predictive Maintenance**

## 1. Abstract

This project proposes a small Explainable AI prototype for predictive maintenance using the AI4I 2020 Predictive Maintenance Dataset. The system will detect abnormal machine behavior, classify whether a machine record indicates failure or no failure, and explain the main sensor features that influenced the result. The goal is to demonstrate how anomaly detection, classification, and explainability can be combined in a simple machine learning workflow.

## 2. Introduction and Problem Statement

Industrial equipment failure can lead to downtime, safety concerns, production delays, and unnecessary maintenance costs. Predictive maintenance uses machine sensor data to identify abnormal behavior and predict equipment failure before serious problems occur, so that maintenance decisions can be made earlier and more effectively [2].

This project narrows predictive maintenance to a simple and understandable application: detecting abnormal machine behavior and classifying whether a machine is likely to fail using a tabular industrial sensor dataset. The system will not be designed as a fully autonomous maintenance solution. Instead, it will act as a decision-support tool that provides predictions and explanations. This follows the idea that explainable predictive maintenance should support user understanding and trust, especially when predictions may influence practical maintenance decisions [2].

The project is based on the AI4I 2020 Predictive Maintenance Dataset introduced by Matzka [1], which connects predictive maintenance with explainable artificial intelligence. It is also supported by Cummins et al. [2], who survey explainable predictive maintenance and emphasize that users need understandable and trustworthy outputs when predictive systems are used in serious applications.

Many machine learning systems focus mainly on predictive accuracy, but Explainable AI systems should also support transparency and interpretability. In a maintenance setting, users need to know whether a machine's behavior is unusual and why the model made a prediction. This need for explanation is consistent with explainable predictive maintenance research, which argues that interpretability can help users trust and evaluate predictive systems [1], [2].

This project asks:

> How can a simple predictive maintenance prototype combine anomaly detection, failure classification, and explanation in an Explainable AI workflow?

## 3. Project Objectives

The main objectives of this project are:

- to detect abnormal machine behavior
- to predict whether a machine record shows failure or no failure
- to explain the main reasons behind model outcomes

## 4. Dataset

The project will use the **AI4I 2020 Predictive Maintenance Dataset**. This dataset is suitable because it is tabular, understandable, and directly related to predictive maintenance [1]. It includes machine operating features such as machine type, air temperature, process temperature, rotational speed, torque, tool wear, and a machine failure label.

The main classification target will be `Machine failure`. The sensor features will be used for both anomaly detection and failure classification.

## 5. Proposed Methodology

The system will follow this workflow:

```text
AI4I Sensor Data
      ->
Data Preprocessing
      ->
Anomaly Detection
      ->
Failure Classification
      ->
Explanation of Outcome
```

First, the dataset will be cleaned and prepared for machine learning. Numerical sensor features will be selected, categorical features such as machine type may be encoded, and the data will be split into training and testing sets. Class imbalance will also be checked because machine failures are usually less common than normal operation.

Second, anomaly detection will be performed to identify records that appear unusual compared with normal machine behavior. These anomaly flags will help show which records differ from typical machine operating patterns.

Third, a simple classification model will be trained to predict failure or no failure. The model will be selected to keep the project understandable and suitable for explanation.

Fourth, the system will explain its outcomes using the most important input features and their relationship to the prediction. The explanation should show, for example, whether high tool wear, high torque, unusual rotational speed, or temperature values influenced the prediction. This explanation step is included because previous work on explainable predictive maintenance highlights the importance of making model outputs understandable to users [1], [2].

## 6. Explainable AI Contribution

This project demonstrates Explainable AI in a practical predictive maintenance setting. The explainability contribution comes from providing understandable reasons for model outcomes, such as showing which sensor features influenced a failure prediction or anomaly flag. For example, the system may explain that high tool wear, high torque, or unusual rotational speed contributed to a risky prediction.

The project also supports transparency by connecting predictions with feature-based explanations. Instead of only reporting a failure or no-failure result, the prototype will show the important machine conditions behind that result. This can make the model output easier to understand and evaluate.

## 7. Evaluation Plan

The classification model will be evaluated using accuracy, precision, recall, F1-score, and a confusion matrix. Recall is important because missing a real failure may be costly or unsafe, while precision is important because too many false alarms may reduce trust in the system.

The anomaly detection component will be evaluated by checking whether anomaly flags correspond to unusual sensor values and whether they overlap with failure cases. The explanation component will be evaluated by checking whether the reported important features are understandable and relevant to the prediction.

## 8. Expected Outcomes

The expected outcome is a working prototype that can load and preprocess the AI4I 2020 dataset, detect anomalous machine records, classify records as failure or no failure, and explain the main reasons for each prediction. Overall, the project should show how a simple machine learning system can become more understandable through explanation.

## 9. Tools and Technologies

The project may use Python and common data analysis, machine learning, visualization, and explainability tools. The demonstration may be implemented as a notebook, command-line program, or simple interface depending on the final project requirements.

## 10. Scope and Limitations

This project is a course-level prototype, not a production-ready industrial system. The AI4I dataset is synthetic, so the results may not fully represent real industrial maintenance conditions. The project will focus on simple and explainable models rather than complex deep learning methods. These limitations keep the project manageable while still demonstrating the main Explainable AI ideas.

## 11. References

[1] S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," in *2020 Third International Conference on Artificial Intelligence for Industries (AI4I)*, 2020, pp. 69-74, doi: 10.1109/AI4I49448.2020.00023.

[2] L. Cummins, A. Sommers, S. B. Ramezani, S. Mittal, J. E. Jabour, M. Seale, and S. Rahimi, "Explainable Predictive Maintenance: A Survey of Current Methods, Challenges and Opportunities," *IEEE Access*, vol. 12, pp. 57574-57602, 2024, doi: 10.1109/ACCESS.2024.3391130.


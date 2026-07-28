# Milestone 1 Plan: Anomaly Detection + Explainability

Target date: July 7-8, 2026

## Goal

Complete the first project milestone: understand the AI4I dataset, prepare clean features, build a first anomaly detection workflow, and explain why selected records appear abnormal.

## Current Status

- Dataset is available at `data/raw/ai4i2020.csv`.
- Data exploration notebook exists at `src/data_exploration/data_exploration.ipynb`.
- Data preprocessing notebook exists at `src/data_preprocessing/data_preprocessing.ipynb`.
- Proposal has been narrowed to anomaly detection and explainability.

## Important Correction

Do not use these columns as input features:

- `Machine failure`
- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

Reason: these are target/failure label columns. Using them as model inputs would leak the answer into the model.

Use these as input features:

- `Type`
- `Air temperature`
- `Process temperature`
- `Rotational speed`
- `Torque`
- `Tool wear`

Use these only for evaluation and interpretation:

- `Machine failure`
- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

## Step-by-Step Plan

### Day 1: Clean Data Exploration

Understand the dataset:

- shape of the dataset
- column meanings
- missing values
- target imbalance
- distribution of sensor features
- normal vs failure counts

Expected output:

- a clean exploration notebook
- short written observations about the dataset

### Day 2: Correct Preprocessing

Prepare clean model inputs:

- drop ID columns
- drop target/failure label columns from features
- encode `Type`
- scale numeric sensor features
- separate features from labels

Expected output:

- clean `X` feature matrix
- target labels saved separately for evaluation
- no data leakage

### Day 3: First Anomaly Detection

Build a first anomaly detection workflow:

- train anomaly detection on sensor features
- produce anomaly scores or anomaly flags
- inspect the top anomalous records
- compare anomaly flags with `Machine failure`

Expected output:

- anomaly flags for records
- table of most anomalous examples
- initial comparison with failure labels

### Day 4: Explain Anomalies

Explain why records appear anomalous:

- compare anomalous records against normal feature averages
- identify which features are unusually high or low
- write simple text explanations

Expected output:

- explanation table for selected anomalous records
- examples such as: high torque, high tool wear, unusual speed

### Day 5: Evaluation and Milestone Report

Summarize the first milestone:

- what dataset was used
- what preprocessing was done
- how anomaly detection was applied
- how explanations were generated
- what early results were observed
- what limitations remain

Expected output:

- milestone notebook or report section
- saved plots/tables
- clear next step for failure classification later

## First Milestone Deliverables

- Clean EDA notebook
- Clean preprocessing notebook
- Initial anomaly detection results
- Explanation examples for anomalous records
- Short milestone summary


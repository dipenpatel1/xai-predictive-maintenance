# Reproducibility Guide

This guide explains how to regenerate the main outputs used in the report and presentation.

Run all commands from the project root:

```bash
cd "/Users/bijaypandey/Documents/Uni of Regina/ENSE_861/responsible_ai_project"
```

## 1. Activate Environment

```bash
source .venv/bin/activate
```

If dependencies are missing:

```bash
pip install -r requirements.txt
```

## 2. Confirm Dataset Exists

The dataset should be located at:

```text
data/raw/ai4i2020.csv
```

If needed, download it with:

```bash
PYTHONPATH=src python scripts/download_ai4i_dataset.py
```

## 3. Generate Report Artifacts

This creates classification and anomaly detection plots/tables:

```bash
PYTHONPATH=src python scripts/generate_report_artifacts.py
```

Outputs are saved to:

```text
outputs/report_artifacts/
```

Important files:

```text
classification_confusion_matrix.png
classification_feature_importance.png
classification_precision_recall_curve.png
anomaly_score_distribution.png
anomaly_speed_torque_scatter.png
```

## 4. Generate Research Error Analysis

This creates false positive, false negative, true positive, and combined anomaly-classification analysis:

```bash
PYTHONPATH=src python scripts/generate_research_analysis.py
```

Outputs are saved to:

```text
outputs/research_analysis/
```

Important files:

```text
classification_error_type_counts.csv
classification_error_group_sensor_means.csv
anomaly_classification_signal_counts.csv
anomaly_classification_signal_summary.csv
```

## 5. Generate Feature Relationship Analysis

This creates correlation, engineered feature, and relationship plots:

```bash
PYTHONPATH=src python scripts/generate_feature_relationship_analysis.py
```

Outputs are saved to:

```text
outputs/feature_relationships/
```

Important files:

```text
feature_correlation_matrix.png
correlation_with_machine_failure.csv
base_vs_engineered_model_metrics.csv
engineered_model_feature_importance.png
speed_torque_relationship.png
tool_wear_torque_relationship.png
```

## 6. Generate Model Tuning Analysis

This creates threshold, tree-depth, class-weight, and anomaly-contamination tuning results:

```bash
PYTHONPATH=src python scripts/generate_model_tuning_analysis.py
```

Outputs are saved to:

```text
outputs/model_tuning/
```

Important files:

```text
threshold_tuning.csv
threshold_tuning_precision_recall_f1.png
decision_tree_depth_tuning.csv
class_weight_tuning.csv
class_weight_tuning.png
anomaly_contamination_tuning.csv
```

## 7. Documentation Files

Main report and presentation documents:

```text
docs/final_report_draft.md
docs/presentation_outline.md
docs/project_proposal.md
```

Supporting milestone/research documents:

```text
docs/milestone_reports/anomaly_detection_report.md
docs/milestone_reports/failure_classification_findings.md
docs/milestone_reports/feature_relationship_findings.md
docs/milestone_reports/model_tuning_findings.md
docs/milestone_reports/research_analysis_plan.md
docs/model_tuning_research_roadmap.md
docs/reusable_code_guide.md
```


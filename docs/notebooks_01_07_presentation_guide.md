# Notebooks 01-07 Technical Presentation Guide

This guide accompanies
`presentation/notebooks_01_07_code_output_walkthrough.pptx`.

## Main Story

The implementation moves through four levels of understanding:

```text
Data -> Model output -> Feature evidence -> Responsible interpretation
```

For every technical slide, present the result using this order:

1. **Question:** What did this notebook try to learn?
2. **Code:** Which core operation produced the result?
3. **Evidence:** Which number or graph supports the finding?
4. **Meaning:** Why does it matter for predictive maintenance?
5. **Limit:** What does the result not prove?

## Slide Sequence

| Slides | Notebook | Main message |
|---|---|---|
| 1-2 | Overview | The notebooks form one sequential research workflow. |
| 3 | 01 Exploration | Failures are rare, so accuracy alone is misleading. |
| 4 | 02 Preprocessing | Split-first preprocessing prevents leakage. |
| 5-7 | 03 Anomaly detection | Unusual behavior and failure are different concepts. |
| 8-10 | 04 Classification | High recall comes with many false alarms; importance is global only. |
| 11-12 | 05 Neural network | Added complexity did not improve the main rare-class metrics. |
| 13-14 | 06 Relationships | Interpretable interactions improved repeated model performance. |
| 15-17 | 07 SHAP/LIME | Explanations reveal both correct reasoning and model errors. |
| 18 | Explanation quality | Verbosity must include evidence, fidelity, context, and limits. |
| 19 | Conclusion | Responsible AI combines performance, explanation, errors, and limits. |

## Numbers to Remember

- Dataset: 10,000 records, 339 failures, 3.39% failure rate.
- Anomaly detection: 113 flags, but only 16 of 68 failures overlapped.
- Base classifier: precision 0.3058, recall 0.9265, F1 0.4599.
- Five-fold engineered tree: precision 0.3552, recall 0.9203, F1 0.5119.
- Five-fold neural network: precision 0.2279, recall 0.8525, F1 0.3593.
- Main SHAP features: rotational speed, torque, and tool wear.
- LIME false-negative fidelity: mean local surrogate R-squared 0.097.

## Final Explanation Message

The final verbose explanation is not a second prediction model. It is an
explanation layer that organizes the existing output into:

1. predicted outcome and score
2. evidence supporting the outcome
3. evidence opposing the outcome
4. decision context or suggested review
5. explanation limitations

More text does not automatically mean a better explanation. The explanation
must remain faithful to the model, understandable to the intended user, and
clear about uncertainty and non-causality.


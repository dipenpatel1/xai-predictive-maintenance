# Research References and Citation Guide

This guide consolidates the proposal references, the papers needed for the
implemented methods, and relevant material from Quizzes 2-4. The final report
should cite a source only where it supports a specific statement.

## Recommended Core References

### [1] AI4I Dataset and Project Foundation

S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance
Applications," in *2020 Third International Conference on Artificial
Intelligence for Industries (AI4I)*, 2020, pp. 69-74, doi:
[10.1109/AI4I49448.2020.00023](https://doi.org/10.1109/AI4I49448.2020.00023).

Use for:

- origin and design of the AI4I 2020 dataset
- explainable predictive maintenance motivation
- the project's dataset limitations and synthetic nature

### [2] Explainable Predictive Maintenance Survey

L. Cummins, A. Sommers, S. B. Ramezani, S. Mittal, J. E. Jabour, M. Seale,
and S. Rahimi, "Explainable Predictive Maintenance: A Survey of Current
Methods, Challenges and Opportunities," *IEEE Access*, vol. 12,
pp. 57574-57602, 2024, doi:
[10.1109/ACCESS.2024.3391130](https://doi.org/10.1109/ACCESS.2024.3391130).

Use for:

- the need for understandable predictive-maintenance outputs
- the roles of local and global explanation
- open challenges and limitations in explainable predictive maintenance

### [3] Isolation Forest

F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation Forest," in *Proc. 8th
IEEE Int. Conf. Data Mining (ICDM)*, Pisa, Italy, 2008, pp. 413-422, doi:
[10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17).

Use for:

- why anomalies can be identified through short isolation paths
- the anomaly-detection method's theoretical basis
- distinguishing anomaly detection from supervised failure classification

### [4] LIME

M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why Should I Trust You?':
Explaining the Predictions of Any Classifier," in *Proc. 22nd ACM SIGKDD
Int. Conf. Knowledge Discovery and Data Mining*, San Francisco, CA, USA,
2016, pp. 1135-1144, doi:
[10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778).

Use for:

- LIME as a local, model-agnostic surrogate explanation
- perturbing a neighborhood around one record
- using individual explanations to inspect model behavior

### [5] SHAP

S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model
Predictions," in *Advances in Neural Information Processing Systems 30*,
Long Beach, CA, USA, 2017, pp. 4765-4774. [Online]. Available:
[NeurIPS paper page](https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).

Use for:

- SHAP as an additive feature-attribution framework
- the baseline-plus-contributions interpretation
- local explanations and aggregation into global importance

### [6] Explanation Robustness

D. Alvarez-Melis and T. S. Jaakkola, "On the Robustness of Interpretability
Methods," in *Proc. ICML Workshop on Human Interpretability in Machine
Learning*, Stockholm, Sweden, 2018. [Online]. Available:
[arXiv:1806.08049](https://arxiv.org/abs/1806.08049).

Use for:

- explanation stability as an evaluation property
- motivation for repeating LIME with multiple random seeds
- avoiding the assumption that one explanation is automatically reliable

### [7] Rigorous Evaluation of Interpretability

F. Doshi-Velez and B. Kim, "Towards a Rigorous Science of Interpretable
Machine Learning," arXiv:1702.08608, 2017. [Online]. Available:
[arXiv:1702.08608](https://arxiv.org/abs/1702.08608).

Use for:

- the need to define and evaluate interpretability
- separating explanation usefulness from predictive performance
- framing the SHAP faithfulness and LIME stability checks

### [8] Responsible AI Risk Management

E. Tabassi, "Artificial Intelligence Risk Management Framework (AI RMF
1.0)," National Institute of Standards and Technology, Gaithersburg, MD,
USA, NIST AI 100-1, Jan. 2023, doi:
[10.6028/NIST.AI.100-1](https://doi.org/10.6028/NIST.AI.100-1).

Use for:

- validity, reliability, transparency, explainability, and accountability
- documenting limitations, assumptions, and error consequences
- treating the prototype as decision support rather than an autonomous system

### [9] Machine-Learning Safety

S. Mohseni, H. Wang, C. Xiao, Z. Yu, Z. Wang, and J. Yadawa, "Taxonomy of
Machine Learning Safety: A Survey and Primer," *ACM Comput. Surv.*, vol. 55,
no. 8, Art. no. 157, pp. 1-38, 2023, doi:
[10.1145/3551385](https://doi.org/10.1145/3551385).

Use for:

- generalization error and open-world limitations
- runtime error detection and model dependability
- explaining why test performance alone does not establish system safety

### [10] Interpretable Models in High-Stakes Settings

C. Rudin, "Stop Explaining Black Box Machine Learning Models for High Stakes
Decisions and Use Interpretable Models Instead," *Nature Machine
Intelligence*, vol. 1, pp. 206-215, 2019, doi:
[10.1038/s42256-019-0048-x](https://doi.org/10.1038/s42256-019-0048-x).

Use for:

- justification for retaining a small decision tree as the main classifier
- the distinction between an interpretable model and post-hoc explanation
- discussion of performance-versus-interpretability choices

### [11] General XAI Survey

R. Guidotti, A. Monreale, S. Ruggieri, F. Turini, F. Giannotti, and
D. Pedreschi, "A Survey of Methods for Explaining Black Box Models,"
*ACM Comput. Surv.*, vol. 51, no. 5, Art. no. 93, pp. 1-42, 2018, doi:
[10.1145/3236009](https://doi.org/10.1145/3236009).

Use for:

- terminology for black-box, local, global, and model-agnostic explanations
- positioning LIME and SHAP within the wider XAI literature

## Suggested Citation Placement

| Report section | Suggested references |
|---|---|
| Introduction and dataset | [1], [2] |
| Anomaly detection method | [3] |
| Explainability method | [4], [5], [11] |
| Explanation evaluation | [6], [7] |
| Responsible AI discussion | [8], [9], [10] |
| Limitations and future work | [1], [2], [6], [8], [9] |

For a concise course report, references [1]-[9] are sufficient. Add [10] when
discussing the choice of a simple tree, and [11] when a broader XAI taxonomy
is useful.

## Audit of Quiz 2-4 Sources

### Quiz 2

The local Quiz 2 folder contains:

- R. Pascanu, T. Mikolov, and Y. Bengio, "On the Difficulty of Training
  Recurrent Neural Networks," in *Proc. 30th Int. Conf. Machine Learning*,
  2013, pp. 1310-1318. The local copy is arXiv:1211.5063.
- Y. Hu, A. Huber, J. Anumula, and S.-C. Liu, "Overcoming the Vanishing
  Gradient Problem in Plain Recurrent Networks," arXiv:1801.06105, 2018.

These papers concern recurrent-network optimization. They are useful course
background but are not directly relevant to the present tree-based,
tabular predictive-maintenance workflow. They should not be included in the
final report unless a neural or temporal extension is actually implemented.

### Quiz 3

The submitted Quiz 3 PDF links to NIST AI RMF 1.0 and the machine-learning
safety taxonomy. These are references [8] and [9] above and are directly
useful for the Responsible AI analysis.

### Quiz 4

Quiz 4 implemented industrial fault classification and discussed LIME,
SHAP, local versus global explanation, and LIME instability. The primary
research sources supporting those concepts are the original LIME paper [4],
the original SHAP paper [5], and the robustness paper [6]. Cite the primary
papers in the project report rather than citing the quiz report itself.

## Citation Rules for the Final Report

1. Use IEEE numbers in order of first appearance.
2. Cite methods when they are first introduced, not only in the references.
3. Cite claims from literature; do not cite tables containing this project's
   own experimental results.
4. Keep the same number for a source every time it is reused.
5. Do not claim that SHAP proves causality or that explanation guarantees
   trust. The implemented checks establish model faithfulness or stability
   only within the stated experiment.

# Modeling results

Training: 1014 matches (2013–2017). Test: 279 matches (2017–2018).

Hyperparameters were chosen using three expanding-season validation folds inside the training period. Scaling is fitted within each fold. The final season is used for evaluation only.

| Model | Features | Test accuracy | Test log loss |
| --- | --- | ---: | ---: |
| benchmark | always_home | 47.31% | not scored |
| benchmark | train_class_prior | 47.31% | 1.0599 |
| logistic | baseline | 50.18% | 1.0098 |
| random_forest | baseline | 49.46% | 1.0182 |

The deterministic always-home rule is an accuracy benchmark. Its probabilities assign zero to other results, so its theoretical log loss is infinite and is not used as a probability-quality benchmark. The smoothed training class prior supplies the probability benchmark.

The logistic baseline correctly classified 140/279 cases, eight more than always home (132/279). It predicted no draws among the 71 actual draws; the accuracy gain alone should not be interpreted as reliable improvement.

The LLM-only, combined, and feature-ablation models have not been run because no extracted LLM features are available. These baseline results do not answer whether LLM features help.

Limitations: one historical league and one held-out season; small sample; previews cover a selected subset of fixtures. The test season has now been inspected for this fixed design and must not be used for further model or prompt tuning.

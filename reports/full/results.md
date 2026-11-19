# Modeling results

Training: 1014 matches (2013–2017). Test: 279 matches (2017–2018).

Adding Codex features gave logistic regression three more correct predictions (143 versus 140) and a small decrease in log loss. The confidence interval spans zero, and the combined random forest had slightly higher log loss than its baseline. Both combined models still predicted zero draws. Codex also recovered all 279 held-out outcomes from team names and dates in a separate diagnostic. This is a serious historical-outcome leakage risk, so the Codex-feature results are exploratory. [Extraction quality and leakage findings](../quality/full/findings.md)

Hyperparameters were chosen using three expanding-season validation folds inside the training period. Scaling is fitted within each fold. The final season is used for evaluation only.

| Model | Features | Test accuracy | Test log loss |
| --- | --- | ---: | ---: |
| benchmark | always_home | 47.31% | not scored |
| benchmark | train_class_prior | 47.31% | 1.0599 |
| logistic | baseline | 50.18% | 1.0098 |
| logistic | llm_only | 48.75% | 1.0477 |
| logistic | combined | 51.25% | 1.0074 |
| logistic | baseline_plus_sentiment | 49.82% | 1.0096 |
| logistic | baseline_plus_context | 52.69% | 1.0092 |
| random_forest | baseline | 49.46% | 1.0182 |
| random_forest | llm_only | 47.31% | 1.0487 |
| random_forest | combined | 50.18% | 1.0185 |

The deterministic always-home rule is an accuracy benchmark. Its probabilities assign zero to other results, so its theoretical log loss is infinite and is not used as a probability-quality benchmark. The smoothed training class prior supplies the probability benchmark.

The logistic baseline correctly classified 140/279 cases. It predicted 0 draws among 71 actual draws. Small accuracy differences alone should not be interpreted as reliable improvement.

The main hypothesis compares combined versus baseline logistic log loss. See paired_comparisons.json for week-block bootstrap intervals. Compare baseline_plus_sentiment and baseline_plus_context to assess H3; this is an exploratory ablation, not a causal test.

Review masking limitations, repeat-extraction agreement, the exclusion log, and the metadata-only leakage diagnostic alongside these results. Model fitting alone does not establish extraction quality or absence of contamination.

Observed hypothesis comparisons (logistic regression):

- H1: LLM-only accuracy minus always-home accuracy = +1.43%.
- H2: Combined minus baseline log loss = -0.0025; exploratory 95% week-block interval [-0.0145, +0.0090]. Negative values favor combined.
- H3: Baseline-plus-sentiment minus baseline-plus-context log loss = +0.0003. Negative values favor the sentiment addition. This ablation does not establish general or causal feature importance.

Independent human reference annotation was omitted by the team, departing from the 50-preview hand-labeled pilot proposed in Milestone 1. The classifier targets are actual match outcomes, so model training does not require human feature labels. Automated schema/quotation checks and repeatability do not establish the semantic accuracy of the extracted features.

Limitations: one historical league and one held-out season; small sample; previews cover a selected subset of fixtures. The test season has now been inspected for this fixed design and must not be used for further model or prompt tuning.

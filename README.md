# EPL match-preview modeling

This BUS244A final project tests whether features extracted from Guardian pre-match previews improve Premier League home-win, draw, and away-win predictions beyond recent match results.

Codex extracted features from all **1,409 matched previews**. The baseline, LLM-only, and combined models are trained and evaluated. Independent human annotation was omitted from this run, a change from Milestone 1. The full run used `gpt-6.1-sol` at low reasoning effort in batches of 25. The initial high-effort pilot remains documented separately. [Full run protocol](docs/FULL_RUN_PROTOCOL.md) · [Pilot report](reports/quality/pilot_status.md)

## Held-out results

The source data reproduce the Milestone 1 counts: 1,506 raw previews, 1,409 joined previews, and 1,293 cases with complete three-match history. Training uses 1,014 matches from 2013–2017; evaluation uses 279 matches from 2017–2018.

| Model | Test accuracy | Test log loss |
| --- | ---: | ---: |
| Always home | 47.31% | Not scored |
| Smoothed training class prior | 47.31% | 1.0599 |
| Logistic regression, recent results | 50.18% | 1.0098 |
| Logistic regression, Codex features | 48.75% | 1.0477 |
| Logistic regression, combined | 51.25% | 1.0074 |
| Random forest, recent results | 49.46% | 1.0182 |
| Random forest, Codex features | 47.31% | 1.0487 |
| Random forest, combined | 50.18% | 1.0185 |

The combined logistic model got three more matches right than the recent-results baseline (143 versus 140). Its log-loss change was −0.0025, with an exploratory 95% interval from −0.0145 to +0.0090. The gain is small and uncertain. The combined forest had slightly higher log loss than its baseline, and the combined logistic model predicted zero draws. [Full results and feature ablations](reports/full/results.md)

**Historical-outcome leakage risk.** In a separate diagnostic, Codex recovered all 279 held-out outcomes from team names and match dates alone. The request hashes match prompts reconstructed from those fields, and the recorded responses contain zero tool events. Masked preview text can still contain identifying clues, so treat the results from Codex features as exploratory. [Extraction quality and leakage findings](reports/quality/full/findings.md)

Regularization and forest depth were chosen using pooled log loss from three expanding-season folds: validate on 2014–2015 after training on earlier data, then 2015–2016, then 2016–2017. Test scores do not select parameters. Numeric scaling is fitted only on each training fold. No team identifiers, dates, target-match goals, or results are predictors. During the test season, prior test-season results may update the next fixture's rolling history; this is rolling pre-match prediction, not a single preseason forecast.

The final-season outcomes have now been inspected for this fixed design. Do not change the prompt, feature definitions, or hyperparameters in response to test performance. Prompt development uses only the training-season pilot.

## Files to open

- [Full results and hypothesis comparisons](reports/full/results.md)
- [Test metrics](reports/full/metrics.csv) and [per-match predictions](reports/full/test_predictions.csv)
- [Derived feature values](data/llm_feature_values.csv) and [provenance](data/llm_feature_values.manifest.json)
- [Extraction quality and leakage findings](reports/quality/full/findings.md)
- `reports/baseline/results.md`: baseline findings.
- `reports/baseline/metrics.csv`: numeric results.
- `reports/baseline/model_comparison.png`: results figure.
- `reports/baseline/test_predictions.csv`: per-fixture predictions and probabilities.
- `reports/baseline/forward_validation.csv`: every tuning candidate and validation fold.
- `reports/baseline/confusion_matrices.csv`: classification counts, including draws.
- `models/baseline/`: fitted logistic and forest pipelines.
- `data/pilot_labels.csv`: **blank** 50-preview human annotation sheet, sampled only from training data.
- `data/processed/previews_with_results.csv`: the joined analysis data.
- `data/llm_inputs.jsonl`: 1,409 masked previews, without outcomes, dates, or team-name metadata.
- `prompts/extraction.txt`: fixed initial extraction prompt and value definitions.
- `reports/data_validation.json`: cohort, chronology, and independently recomputed rolling-history checks.
- `data/source_manifest.json`: source URLs, retrieval times, and hashes.
- `FILES.md`: index of the repository's publishable files.

Original Milestone 1 scripts and summary files were copied from https://github.com/bhschwartz619/epl-preview-llm into `upstream/`; the downloaded commit is in `upstream/SOURCE.json`. The working copy of `build_dataset.py` normalizes nonbreaking spaces before parsing publication dates, fixing a pandas 3 string-backend compatibility problem. Original upstream files are preserved.

## Run locally

The workflow was tested with Python 3.14.7. Exact installed package versions are in `requirements.txt`. From the repository directory, use a Python environment with these dependencies installed:

```sh
PY=python3
$PY scripts/download_data.py
$PY scripts/build_dataset.py data/raw/GuardianReportsAll.csv data/raw/raw_results data/processed
$PY scripts/prepare.py
$PY scripts/model.py --features data/llm_feature_values.csv --human-validation omitted
$PY -m unittest discover -s tests -v
```

The downloaded source files are reused when their hashes match. The exported CSV reproduces the model comparison without another Codex run. Guardian text and raw Codex responses are excluded from Git.

## Original Claude API route (optional)

1. Read `prompts/extraction.txt` and manually label `data/pilot_labels.csv` **before looking at model labels**. Fill all ten value columns, the evidence columns, and `annotator`. Use `unknown` with empty evidence when unsupported. Supporting evidence for other labels must be a contiguous quote from that row's masked text. Do not fill human labels with AI output and report them as human work.
2. Configure `ANTHROPIC_API_KEY` in your local shell or credential manager, outside this project. Do not paste credentials into chat. Choose an explicit Claude model identifier supporting `temperature=0`, preferably a dated snapshot. The script uses Anthropic's Messages API and records the requested/returned model, API version, prompt/input hashes, usage, and raw responses. The default API version is `2023-06-01`; real account/model compatibility remains untested. If temperature zero is unsupported, select a compatible model or document a deliberate change to the Milestone 1 protocol.
3. Run the pilot and compare against the human labels. Review disagreement cases and evidence; Milestone 1 specifies no numerical agreement threshold. If the prompt changes, use a fresh output filename so old and new versions cannot mix. Freeze the prompt before processing the rest of the data.

```sh
CLAUDE_MODEL='YOUR_EXPLICIT_MODEL_ID'
$PY scripts/extract.py --model "$CLAUDE_MODEL" --scope pilot --max-requests 50
$PY scripts/quality.py --human-labels data/pilot_labels.csv
```

4. With the final prompt, complete extraction and repeat 20 of the pilot inputs in a separate file. Reusing the same features file skips completed records; successful pilot records count toward the full 1,409. There is one model call per preview except bounded retries for transient HTTP errors. There are no extra LLM calls to recheck outputs: JSON values and quoted evidence are validated locally. Failures are saved with their raw responses for review; corrections must be documented rather than silently fabricated.

```sh
$PY scripts/extract.py --model "$CLAUDE_MODEL" --scope all --max-requests 1409
$PY scripts/extract.py --model "$CLAUDE_MODEL" --scope repeat --max-requests 20
$PY scripts/quality.py --human-labels data/pilot_labels.csv --repeat data/llm_repeat.jsonl
$PY scripts/model.py --features data/llm_features.jsonl
```

The full model command requires all 1,293 eligible matches to have features; it never drops missing-feature cases to improve results. It fits logistic and forest models on baseline, LLM-only, and combined features. Logistic ablations add sentiment alone or contextual features alone to the baseline. These address H1–H3, with accuracy and log loss plus paired week-block bootstrap intervals for the combined-versus-baseline difference. Outputs go to `reports/full/` and `models/full/`, leaving the baseline results separate. Full results must still be accompanied by the extraction-quality evidence.

5. Run the metadata-only diagnostic with the same model. This uses the fixed 279-case holdout; `--max-requests` allows a smaller resumable batch. It is a separate diagnostic and never supplies modeling features.

```sh
$PY scripts/extract.py --model "$CLAUDE_MODEL" --kind metadata --scope all --max-requests 279
$PY scripts/quality.py --metadata data/metadata_diagnostic.jsonl
```

Better metadata-only accuracy is **not proof of memorized outcomes**: team identities carry legitimate strength information. Likewise, a weak score cannot rule out contamination. Report this diagnostic together with the masking limitations.

## Extraction choices and limits

Sentiment is numeric -2 through +2, with a separate unknown indicator. Unknown sentiment is coded as zero plus that indicator, preserving its distinction from explicitly neutral sentiment. The eight contextual variables use predefined one-hot categories, including unknown. For the legacy availability category `none_mentioned`, this implementation requires an explicit statement of no availability concerns; silence is unknown. This avoids treating missing discussion as evidence of a healthy squad.

Masking handles known team aliases, some nicknames, year/date strings, and a heuristic trailing byline. It cannot guarantee anonymity. Ambiguous aliases are logged rather than replaced; players, managers, venues, and indirect historical clues may remain. `data/masking_log.csv` identifies automated changes and missing HOME/AWAY tokens for review. Review of excluded previews and semantic evidence quality is still a human task; the code does not claim it has been completed.

The full Codex extraction and model comparison are complete. The original Anthropic API route remains untested. Independent human agreement was omitted, so the accuracy of Codex's interpretation of individual previews remains unmeasured. Automated checks cover the feature schema, the presence of supporting quotes, and the model's consistency on a repeated batch. [Full run protocol](docs/FULL_RUN_PROTOCOL.md)

Codex assisted with the modeling code, extraction prompt, and documentation and generated the preview features.

Technical references: [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages/create), [scikit-learn temporal validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split), and [log loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).

# Full extraction and modeling without human reference labels

The team chose to omit the independent human feature-annotation step proposed in Milestone 1. Human labels are not required to fit the prediction models: their supervised targets are the actual home-win/draw/away-win match outcomes. No human agreement or semantic feature accuracy will be reported as measured.

## Frozen full-run settings

- Model identifier: `gpt-6.1-sol`, through Codex CLI and the existing ChatGPT sign-in.
- Reasoning effort: **low**, chosen before full-model evaluation to reduce annotation overhead. The original 50-preview pilot used high reasoning effort and is retained separately.
- Batch size: 25. All full-run inputs, including the pilot previews, are extracted under this one protocol into a separate file. Original pilot labels are not mixed into the full model input.
- Transport: compact arrays of ten values and ten quotes, expanded into the standard HOME/AWAY feature schema. Feature definitions remain the same as the original extraction prompt.
- Evidence policy: unsupported non-unknown values whose quotes do not occur in the supplied text become `unknown`, with the original value and quote logged as an automated correction. This is conservative filtering, not human correction or a claim of semantic accuracy.
- Controls: fresh ephemeral sessions in empty temporary directories; shell, patching, image viewing, web search, multi-agent tools, apps, and configured MCP servers disabled. No match outcomes are sent to feature extraction.
- Temperature: not exposed by the CLI; temperature zero was not set.

The requested model identifier, CLI version, wrapper, schema template, settings hash, preview hashes, actual batch membership, raw responses, usage, and corrections are retained. An immutable backend model snapshot is not returned by the CLI. The unchanged feature definitions are in `prompts/extraction.txt`.

## Commands

From the repository directory, with the required Python packages and Codex CLI available:

```sh
python3 scripts/extract_codex.py --model gpt-6.1-sol --reasoning-effort low --scope all --compact --evidence-policy unknown --batch-size 25 --max-previews 1409 --timeout 180 --output data/codex_features_full.jsonl
python3 scripts/quality.py --features data/codex_features_full.jsonl --summary-only --output-dir reports/quality/full
```

The full extraction covers all 1,409 matched articles. The modeling cohort remains the same 1,293 complete-history matches: 1,014 training cases and 279 held-out cases. Missing extraction records cannot be silently dropped from model comparisons. Successful extraction records are checkpointed and reused when the command resumes with identical settings.

Repeat one complete training-season batch, selected using seed 244, in its original order. This preserves the other previews seen with it, so context changes do not masquerade as sampling inconsistency:

```sh
python3 scripts/extract_codex.py --model gpt-6.1-sol --reasoning-effort low --scope repeat --repeat-source data/codex_features_full.jsonl --compact --evidence-policy unknown --batch-size 25 --max-previews 25 --timeout 180 --output data/codex_full_repeat.jsonl
python3 scripts/quality.py --features data/codex_features_full.jsonl --repeat data/codex_full_repeat.jsonl --output-dir reports/quality/full
python3 scripts/diagnose_codex.py --model gpt-6.1-sol --reasoning-effort low --batch-size 100 --timeout 180
python3 scripts/quality.py --metadata data/codex_metadata_diagnostic.jsonl --output-dir reports/quality/full
python3 scripts/export_feature_values.py --features data/codex_features_full.jsonl
python3 scripts/model.py --features data/llm_feature_values.csv --human-validation omitted
```

The logistic and forest families compare baseline, LLM-only, and combined features, with the planned logistic sentiment/context ablations. Hyperparameters are selected by the existing forward-season training folds. Test outcomes do not select prompts, features, thresholds, or hyperparameters. Accuracy and log loss are evaluated against actual held-out match outcomes; the combined-versus-baseline comparison also has exploratory calendar-week bootstrap intervals.

The shareable `data/llm_feature_values.csv` contains only IDs and derived feature values. Its manifest records hashes and model provenance. It includes no preview text or supporting quotations. After downloading and preparing the source data, a teammate can reproduce model fitting from this CSV without rerunning Codex. The loader verifies its schema, categories, row count, file hash, and analysis-dataset hash.

## Reporting limits

Supporting-quote occurrence, valid JSON, and repeatability do not establish correct interpretation or team attribution. High repeatability can reflect consistent errors and many unknown labels. Masking does not remove all historical clues, and model memory of match outcomes remains a contamination risk. Metadata-only prediction is a separate diagnostic; team names carry genuine strength information, so diagnostic accuracy alone cannot prove memorization or its absence.

Results should be presented as a model comparison using AI-derived features without independent human extraction validation. Any departure from Milestone 1, including provider, sampling controls, batch transport, and omission of human labels, must remain visible in the final write-up.

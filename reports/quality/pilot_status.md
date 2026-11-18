# Codex pilot extraction

The 50-preview pilot has been extracted using Codex CLI and the existing ChatGPT sign-in. The LLM-only and combined prediction models have not been trained.

| Item | Status |
| --- | --- |
| Pilot previews | 50/50 |
| Recorded feature values | 500 |
| Codex requests | 10 batches of 5 |
| Model identifier | gpt-6.1-sol |
| Reasoning effort | high |
| Temperature | Not exposed by Codex CLI; temperature zero was not set |
| Output IDs, schema, allowed values, quote occurrence | Passed local checks |
| Tool activity | No tool events accepted by the extraction runner |
| Pilot isolation | All 50 IDs belong to training seasons |
| Human rows with an annotator | 0/50 |
| Human agreement | Not computed |
| Repeatability and metadata-only diagnostic | Not run |
| LLM-only and combined models | Not fitted |

The shared prompt defines ten features: five for HOME and five for AWAY. The Codex wrapper preserves these definitions and requests independent judgments within each batch. Model settings and request/input hashes are saved with each record. The recorded model identifier is the explicit CLI selection; an immutable backend snapshot was not returned.

The six Codex/summary tests passed, as did the targeted existing feature-provenance test. No baseline model was retrained. Local structural checks and quote occurrence do not establish correct interpretation, correct team attribution, or absence of historical-outcome contamination.

## Local artifacts

- `data/codex_features.jsonl`: AI labels and supporting quotes; Git-ignored.
- `data/codex_features_responses/`: effective prompt and raw CLI responses; Git-ignored.
- `data/codex_features.manifest.json`: model and protocol metadata.
- `data/pilot_labels.csv`: independent human annotation sheet; Git-ignored.
- `data/processed/pilot_reading_packet.md`: readable human pilot packet; Git-ignored.
- `reports/quality/extraction_summary.json`: counts, distributions, and batch usage.
- `reports/quality/feature_coverage.csv`: unknown share for each feature.

## Next step

A human should complete all 50 reference annotations without consulting the AI label file. Compare the human sheet with Codex using:

```sh
python3 scripts/quality.py --features data/codex_features.jsonl --human-labels data/pilot_labels.csv
```

Review the disagreements, finalize the prompt using only training-season pilot feedback, then run the remaining extraction and the LLM-only/combined models. [The Codex method](../../docs/CODEX_METHOD.md) gives the commands.

# Verification

- Reproduced the documented 1,409 matched previews and 1,293 complete-history cases; exact 1,014/279 chronological split.
- Independently recomputed every eligible team's points and goal difference from three strictly prior fixtures within the same season.
- Eleven targeted unittest checks passed, including synthetic full-feature logistic/forest pipelines, training-only scaling, exclusion of target columns, pilot isolation, schema/quote validation, and a mocked HTTP request.
- Inspected the exported comparison figure.
- Live Anthropic API and real LLM-derived models remain untested. Human pilot labels, agreement, repeatability, metadata-only diagnostic, and exclusion review remain unfinished.
- Reloaded both fitted model files and reproduced every exported holdout probability within 1e-12.
- After strengthening model/version provenance checks, the targeted provenance test passed again.

# Leakage study: run guide

**Question.** Can structured masking stop the LLM from recognising historical matches and recalling their results, and do preview features still add predictive value once it does?

The existing names-and-date diagnostic (279/279 results recalled) is Level 0. This guide adds:

1. Five masking levels of every preview (`scripts/mask_previews.py`, method in [MASKING.md](MASKING.md)).
2. A leakage probe on the 279 held-out matches at each level (`scripts/leakage_probe.py`): the model guesses the teams and season and gives the result with probabilities.
3. Scoring and the headline chart (`scripts/score_leakage.py`).
4. Feature extraction and the model comparison on fully masked (Level 5) text, using the existing pipeline.

Run everything from the repository root.

## 1. Install the masking dependency (once)

```sh
pip install spacy==3.8.18
python -m spacy download en_core_web_lg
```

## 2. Mask the previews and audit the result (about 1 minute)

```sh
python scripts/mask_previews.py data/processed/previews_with_results.csv data/masked
python scripts/audit_masking.py data/masked/masked_previews.csv reports/masking
python -m unittest tests.test_masking
```

Check `reports/masking/residual_counts.csv`: at Level 4, every row except "weekday or month" should be 0.0. Weekdays and months are intentionally removed at Level 5, where every row should be 0.0. `data/masked/` holds Guardian text and stays out of Git.

## 3. Commit the per-match evidence for the 279/279 diagnostic

`.gitignore` now allows this one file. It contains IDs, hashes and H/D/A answers, no article text.

```sh
git add data/codex_metadata_diagnostic.jsonl
```

## 4. Leakage probe (about 70 Codex calls)

Try one batch first and look at the answers:

```sh
python scripts/leakage_probe.py --model gpt-6.1-sol --levels L5 --max-batches 1
head -3 reports/leakage/answers_L5.jsonl
```

Then run all levels. The script resumes, so the trial batch is not repeated, and an interrupted run continues where it stopped.

```sh
python scripts/leakage_probe.py --model gpt-6.1-sol --levels L1 L2 L3 L4 L5
```

Settings: low reasoning effort (same as the full extraction), batches of 20, random four-digit IDs instead of MatchIDs, and one seeded random batch order shared by every level, so only the masking differs between levels. Use the same `--model` as the earlier diagnostic so L0 and L1–L5 are comparable.

## 5. Score

```sh
python scripts/score_leakage.py
```

Writes `reports/leakage/summary.csv`, `reports/leakage/per_match.csv` and `reports/leakage/leakage_by_level.png`. If it prints team guesses that matched no club, check whether any is a real club spelled unusually; if so add it to `ALIASES` in the script and rerun.

How to read it:
- **Both teams identified / exact match identified** (teams and season): direct evidence the model recognises the fixture.
- **Result correct**: compare with the recent-results model (50.2%) and always-home (47.3%) on the same 279 matches. Results far above about 50% while the teams are masked suggest recall, not forecasting.
- **Result correct when identified vs not identified**: if accuracy is much higher on matches the model identified, identification is driving the recall.
- Accuracy cannot prove memorisation on its own. Report it together with identification.

## 6. Features and model comparison on fully masked text (57 Codex calls)

Same protocol as the original full run, with only the input file changed through the `LLM_INPUTS` variable:

```sh
export LLM_INPUTS=data/masked/llm_inputs_L5.jsonl
python scripts/extract_codex.py --model gpt-6.1-sol --reasoning-effort low --scope all --compact --evidence-policy unknown --batch-size 25 --max-previews 1409 --timeout 180 --output data/codex_features_L5.jsonl
python scripts/export_feature_values.py --features data/codex_features_L5.jsonl --output data/llm_feature_values_L5.csv
unset LLM_INPUTS
python scripts/model.py --features data/llm_feature_values_L5.csv --human-validation omitted --tag L5
```

Results go to `reports/full_L5/`; the original `reports/full/` is untouched. Compare the combined-versus-baseline difference with the original run. If the small gain disappears at Level 5, it probably came from recognised fixtures, not the text.

Optional, for a like-for-like comparison: repeat step 6 with `llm_inputs_L3.jsonl` and `--tag L3`. The original features used the earlier masking, which is close to Level 3 but catches fewer nicknames.

## 7. Blind human check

Someone who has not read the original previews should read about 30 Level 5 test-season previews and write down any match they can identify. This command writes a sheet without answers:

```sh
python -c "import pandas as pd; d=pd.read_csv('data/masked/masked_previews.csv'); d[d.Season=='2017-18'].sample(30, random_state=1)[['MatchID','L5']].assign(guess='').to_csv('data/masked/blind_check.csv', index=False)"
```

Score their guesses against `data/processed/previews_with_results.csv` afterwards. The sheet contains Guardian text and stays local.

## 8. What to commit

Commit: `scripts/`, `tests/test_masking.py`, `docs/`, `reports/masking/`, `reports/leakage/` (answers, summary, per-match scores, chart, manifest), `reports/full_L5/`, `data/llm_feature_values_L5.csv` and its manifest, `data/codex_features_L5.manifest.json`, `data/codex_metadata_diagnostic.jsonl`.

Never commit: `data/masked/`, `data/leakage/`, `data/codex_features_L5.jsonl` (all ignored already).

## Limits to report

- Masking removes names, not situations; distinctive situations (a farewell, a final-day relegation) can still identify a match.
- The model may have memorised the exact wording of these public articles; masking does not change wording.
- Probe answers come from batches of 20; other previews in a batch could influence an answer. Batches are identical across levels.
- The Codex CLI does not expose temperature, so answers are not guaranteed to repeat exactly.

# Masking the Guardian match previews

`scripts/mask_previews.py` produces five versions of every preview, each removing more
identifying information than the last. Each level contains everything the level below it masks.

| Level | What is masked (cumulative) | Placeholders |
| --- | --- | --- |
| L1 | Nothing (whitespace normalized only) | – |
| L2 | Home and away team names and nicknames ("Spurs", "the Saints", "United" when only one team is a United) | `[HOME]`, `[AWAY]` |
| L3 | + journalist byline, years, calendar dates | byline deleted, `[YEAR]`, `[DATE]` |
| L4 | + other clubs, people, stadiums and stands, places, nationalities | `[CLUB_n]`, `[PERSON_n]`, `[VENUE]`, `[PLACE]`, `[NATIONALITY]`, `[NAME_n]` |
| L5 | + weekdays, months, holidays, seasons of the year, "opening day" / "final day" type phrases, dated tournaments; sponsored cup names become "League Cup" | `[TIME]`, `[EVENT]` |

Numbered placeholders are consistent within a preview and across levels: "Arsène Wenger" and a later "Wenger" are both `[PERSON_2]`.
Competitions that carry stakes information (Premier League, Champions League, Europa League, FA Cup, relegation, Championship) are never masked.

## How names are found

1. **Curated lists** (`scripts/masking_lists.py`): all 28 clubs in the sample with their nicknames, 190 other clubs, 111 stadiums and stands, 70 regions and cities, 82 nationality nouns, and time words. Lists are matched as whole, case-sensitive words; the longest match wins ("Villa Park" before "Villa").
2. **spaCy name recognition** (`en_core_web_lg`): finds people, places, venues and organisations the lists miss. spaCy often mislabels clubs (it tags "Burnley" as a person), so list matches always take precedence.
3. **Person lexicon**: every name spaCy recognises as a person anywhere in the 1,409 previews is reused in all previews, and the surnames of people named in a preview are reused within it, which catches mentions spaCy misses in a given sentence. A capitalised common word at the start of a sentence ("Given that...") is not treated as a name.
4. **Shared nicknames** ("City", "United", "Blues", "Reds", "Albion", "Town", "Rangers") are assigned to HOME or AWAY only when exactly one of the two teams uses them. Otherwise they become `[TEAM]` (ambiguous between the two teams) or `[CLUB_n]` (a third club).

Overlaps are resolved so that lower levels win: a level-2 team name is never swallowed by a level-4 match. As a result "Villa Park" in an Aston Villa home match becomes "[HOME] Park" rather than `[VENUE]`.

## Audit

`scripts/audit_masking.py` checks the output in two independent ways and writes `audit/residual_counts.csv` and `audit/leftover_names_L4.csv` / `_L5.csv`.

Share of the 1,409 previews in which each kind of information is still present:

| Still present | L1 | L2 | L3 | L4 | L5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Home/away team name | 98.9% | 0.0% | 0.0% | 0.0% | 0.0% |
| Known person | 99.9% | 99.9% | 96.3% | 0.0% | 0.0% |
| Other club | 60.3% | 60.3% | 60.3% | 0.0% | 0.0% |
| Stadium | 40.4% | 39.8% | 39.8% | 0.0% | 0.0% |
| Place | 37.4% | 26.5% | 26.5% | 0.0% | 0.0% |
| Weekday or month | 39.7% | 39.7% | 37.8% | 37.6% | 0.0% |
| Year | 10.0% | 10.0% | 0.0% | 0.0% | 0.0% |
| Calendar date | 3.2% | 3.2% | 0.0% | 0.0% | 0.0% |

Running spaCy again on the fully masked (L5) text finds 11 remaining name mentions across all previews, mostly kept competition names ("Premier League era") and one-off fragments. A random sample of 12 fully masked test-season previews was also read by hand: every name was masked, and the features (form, injuries, stakes) remained readable.

## What masking cannot remove

Masking removes names, not situations. A fully masked preview can still describe a match that only happened once, for example a manager's farewell after 22 years, two relegated teams on the final day, or a team "already guaranteed second place". These clues are part of the text the features are extracted from, so they cannot be removed without removing the information the study needs. Whether an LLM can still identify such matches is what the re-identification probe measures.

Known imperfections, all of which still mask the text but under a less precise label:
- A small number of mentions are masked with the wrong placeholder type (for example a stadium nickname, "the Lane", as `[PERSON_n]`, and a few players as `[NAME_n]`).
- A few ordinary capitalised words are masked as people (for example "David and Goliath").
- Two typos in the source text ("Manchesterchester United", "West BromwichAlbion") are handled by adding them to the alias list.

The lists were built and extended while auditing this corpus. Report the masking as a corpus-specific procedure, not a general-purpose anonymiser.

## Outputs

| File | Contents | In Git? |
| --- | --- | --- |
| `data/masked/llm_inputs_L1.jsonl` ... `L5.jsonl` | One line per preview: `MatchID`, `text`, `input_sha256`. Same format and order as `data/llm_inputs.jsonl`. | No (Guardian text) |
| `data/masked/masked_previews.csv` | All five levels side by side, for reading and spot checks. | No |
| `data/masked/masking_log.csv` | Every replacement: MatchID, level, category, original text, replacement. | No |
| `data/masked/person_lexicon.csv` | Names reused as people across the corpus. | No |
| `reports/masking/` | Residual counts and leftover names from the audit. | Yes |

`data/masked/` is listed in `.gitignore`. Do not force-add it.

## Reproduce

```
pip install spacy==3.8.18
python -m spacy download en_core_web_lg
python scripts/mask_previews.py data/processed/previews_with_results.csv data/masked
python scripts/audit_masking.py data/masked/masked_previews.csv reports/masking
python -m unittest tests.test_masking
```

Tested with spaCy 3.8.18 and en_core_web_lg 3.8.0. Masking takes about 20 seconds. To feed a level to any script that reads `data/llm_inputs.jsonl`, set the environment variable `LLM_INPUTS`, for example `LLM_INPUTS=data/masked/llm_inputs_L5.jsonl`. See [LEAKAGE_STUDY.md](LEAKAGE_STUDY.md).

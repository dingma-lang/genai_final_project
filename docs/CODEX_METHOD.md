# Codex feature extraction protocol

The team chose to replace the Claude API extraction route proposed in Milestone 1 with Codex CLI extraction using an existing ChatGPT sign-in. This changes the extraction provider and sampling controls; it does not change the research question, feature definitions, fixed 1,014/279 modeling cohort, evaluation metrics, or baseline models.

The extractor uses `gpt-6.1-sol` with high reasoning effort, matching the configured local CLI model and effort when this route was set up. The selected model identifier, CLI version, exact prompt, output schema hash, settings hash, input hashes, actual batch membership, timestamps, usage, and raw CLI responses are recorded. The CLI does not return an immutable provider snapshot identifier, so the selected model name should not be described as an immutable model version. It also does not expose a temperature control: **temperature zero was not set**, unlike the original Claude plan.

Each invocation starts a fresh, ephemeral Codex session in an empty temporary working directory. User configuration is ignored for the extraction worker while cached authentication is reused. Shell execution, patching, image viewing, web search, multi-agent tools, apps, and configured MCP servers are disabled by explicit settings. The worker receives only the shared labeling instructions and masked preview text/IDs. The wrapper requires separate judgments for each preview. The runner rejects any CLI event indicating tool activity and locally validates record coverage, allowed feature values, and supporting quotations.

Five previews are processed per request. A preview can be influenced by the other previews in its batch; exact batch membership and request hashes are retained. All outputs are labeled as AI annotations. None is written to the independent human annotation sheet.

## Pilot

```sh
python3 scripts/extract_codex.py --model gpt-6.1-sol --reasoning-effort high --scope pilot --batch-size 5 --max-previews 50
python3 scripts/quality.py --features data/codex_features.jsonl --summary-only
```

Successful records are appended to `data/codex_features.jsonl`. Running the command again skips completed IDs if the model, prompt, CLI version, input text, and settings match. Local raw responses and the effective prompt are in `data/codex_features_responses/`. A protocol manifest is saved beside the feature file. Article text and raw responses remain Git-ignored.

The 50 pilot previews come only from training seasons. Human annotators should label `data/pilot_labels.csv` without consulting the AI feature file. The reading packet is available locally at `data/processed/pilot_reading_packet.md`. Then compare labels:

```sh
python3 scripts/quality.py --features data/codex_features.jsonl --human-labels data/pilot_labels.csv
```

The summary-only command reports feature coverage, value distributions, and usage counted once per batch. It does not compute or claim human agreement. Schema-valid output and quotations establish output structure and that a quote occurs in the supplied preview. They do not establish that the quote supports the assigned interpretation, that the team attribution is correct, or that the model used no memorized match information. Human agreement remains a separate, unfinished check until real labels are supplied.

## After pilot review

Freeze the final prompt using training-only pilot feedback. If it changes, write all revised features to a new output path rather than mixing old and new labels. With the final prompt, process the remaining previews and run a separate repeatability sample:

```sh
python3 scripts/extract_codex.py --model gpt-6.1-sol --reasoning-effort high --scope all --batch-size 5 --max-previews 1409
python3 scripts/extract_codex.py --model gpt-6.1-sol --reasoning-effort high --scope repeat --batch-size 5 --max-previews 20
python3 scripts/quality.py --features data/codex_features.jsonl --human-labels data/pilot_labels.csv --repeat data/codex_repeat.jsonl
python3 scripts/model.py --features data/codex_features.jsonl
```

The real LLM-only and combined models must not be described as trained until this final command has completed on the complete eligible cohort. The original metadata-only diagnostic currently remains implemented for Anthropic in `scripts/extract.py`; running it for Codex requires a separate Codex diagnostic adapter and is not established by successful feature extraction.

Technical references: [Codex authentication](https://learn.chatgpt.com/docs/auth), [non-interactive structured outputs](https://learn.chatgpt.com/docs/non-interactive-mode), and [tool/configuration controls](https://learn.chatgpt.com/docs/config-file/config-reference).

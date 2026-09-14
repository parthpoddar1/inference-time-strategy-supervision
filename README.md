# Inference-Time Strategy Supervision for Mathematical Reasoning

This repository contains the code, frozen artifacts, and results for a study of whether high-level strategies from teacher language models improve mathematical reasoning by smaller student models at inference time.

## Experimental design

The primary experiment uses a frozen 1,050-problem MATH subset, balanced across seven domains and difficulty levels 3–5. Two student models are evaluated in six conditions (Normal plus five teacher-strategy conditions), across three experimental runs. This yields 36 response files and 37,800 student responses.

Students:

- Llama 3.1 8B Instruct
- Qwen 2.5 7B Instruct

Teachers:

- Qwen 2.5 72B Instruct
- DeepSeek V4 Flash 0423
- Llama 3.3 70B Instruct
- Gemma 3 27B IT
- Mistral Small 3.2 24B Instruct

Correctness was independently audited three times with `gpt-5.6-luna`. These are model-audit replicates, not independent reruns of student inference.

## Layout

- `data/main/` — frozen 1,050-question manifest and the original audit input ZIP.
- `data/teacher_strategies/` — five frozen teacher-strategy files, one per teacher.
- `results/main/student_responses/` — 36 primary-experiment response files. Filenames are canonical: `run{1..3}_{llama|qwen}_{condition}.json`.
- `results/main/audits/` — three completed GPT-5.6 Luna correctness-label outputs.
- `results/main/summary/` — completed summary and an analysis-generated summary.
- `results/leakage/` — final answer-leakage verdicts and leaked-case correctness results.
- `data/pilot/`, `results/pilot/`, and `scripts/run_pilot_t0_t4.py` — the controlled 420-question T0–T4 pilot.
- `scripts/` — audit, leakage, pilot, and analysis scripts.

The `source_file` fields and response filenames—not some embedded `run`, `student`, or `condition` fields—are the canonical condition identifiers. This avoids historical metadata inconsistencies in a subset of the raw response files.

## Reproduce the reported main accuracy table

No API access is needed to recompute the main summary:

```bash
python scripts/analyze_main_results.py
```

This reads the three frozen audit outputs and writes `results/main/summary/recomputed_accuracy_summary.json`.

The exact paired McNemar calculation for Qwen 2.5 7B, Normal versus Mistral, is also retained:

```bash
python scripts/mcnemar_qwen_mistral.py
```

To launch fresh model-audit replicates, first set an OpenAI API key in your environment, then run:

```bash
python scripts/audit_responses.py --replicates 3 --workers 40
```

Fresh outputs go to `results/main/audits/reruns/` by default. They should not overwrite the frozen published audit outputs.

## Leakage analysis

The teacher-strategy leakage audit counts strategies that provide the requested final answer, including explicit, spelled-out, and mathematically equivalent forms. The retained JSON results are the final detailed verdicts and summaries. The scripts make API calls and require `OPENAI_API_KEY`.

## Notes on reproducibility

The repository preserves the frozen question order, teacher strategies, student outputs, and correctness-audit outputs used for the reported analyses. Hosted model outputs may vary over time. API keys are intentionally not included.

The primary student-generation script and standalone main-experiment prompt files were not included in the supplied materials; this repository does not reconstruct or claim to provide them. The pilot generation script and its prompts are retained separately.

## License and citation

Released under the MIT License. Citation information can be added here once the associated paper is finalized.

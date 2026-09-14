"""Recompute main condition accuracies from the three frozen audit outputs.

The source filename is canonical because historical embedded metadata differs
in some run-1/run-2 records.  No API calls are made by this script.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
AUDITS = ROOT / "results" / "main" / "audits"
OUT = ROOT / "results" / "main" / "summary" / "recomputed_accuracy_summary.json"

STUDENTS = {"llama": "Llama 3.1 8B Instruct", "qwen": "Qwen 2.5 7B Instruct"}
CONDITIONS = {
    "normal": "Normal",
    "qwen": "Qwen 2.5 72B Instruct",
    "deepseek": "DeepSeek V4 Flash 0423",
    "llama": "Llama 3.3 70B Instruct",
    "gemma": "Gemma 3 27B IT",
    "mistral": "Mistral Small 3.2 24B Instruct",
}
PATTERN = re.compile(r"run(?P<run>[123])_(?P<student>llama|qwen)_(?P<condition>normal|qwen|deepseek|llama|gemma|mistral)\\.json")


def parse_source(name: str) -> tuple[int, str, str]:
    match = PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"Unexpected source filename: {name}")
    return int(match["run"]), match["student"], match["condition"]


def accuracy(rows: list[dict]) -> float:
    return sum(row["correctness"] == "CORRECT" for row in rows) / len(rows)


def main() -> None:
    audit_paths = sorted(AUDITS.glob("correctness_audit_gpt_5_6_luna_replicate_*.json"))
    if len(audit_paths) != 3:
        raise RuntimeError(f"Expected exactly three audit outputs in {AUDITS}; found {len(audit_paths)}.")

    per_replicate: list[dict[str, float]] = []
    overall = []
    for path in audit_paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[row["source_file"]].append(row)
        if len(rows) != 37_800 or len(grouped) != 36 or any(len(group) != 1_050 for group in grouped.values()):
            raise RuntimeError(f"{path.name} does not contain the expected 36 x 1,050 audit labels.")
        per_replicate.append({source: accuracy(group) for source, group in grouped.items()})
        overall.append(accuracy(rows))

    by_condition = []
    for student in STUDENTS:
        for condition in CONDITIONS:
            replicate_means = [mean(rep[f"run{run}_{student}_{condition}.json"] for run in (1, 2, 3)) for rep in per_replicate]
            by_condition.append({
                "student": STUDENTS[student],
                "condition": CONDITIONS[condition],
                "audit_replicate_accuracies": replicate_means,
                "mean_accuracy": mean(replicate_means),
                "audit_replicate_sample_sd": stdev(replicate_means),
            })

    lookup = {(row["student"], row["condition"]): row for row in by_condition}
    for student in STUDENTS:
        baseline = lookup[(STUDENTS[student], "Normal")]["mean_accuracy"]
        for condition in CONDITIONS:
            row = lookup[(STUDENTS[student], CONDITIONS[condition])]
            row["difference_from_normal_percentage_points"] = (row["mean_accuracy"] - baseline) * 100

    report = {
        "auditor": "gpt-5.6-luna",
        "audit_replicate_overall_accuracies": overall,
        "overall_mean_accuracy": mean(overall),
        "overall_audit_replicate_sample_sd": stdev(overall),
        "condition_averages": by_condition,
        "metadata_note": "Condition metadata is derived from source filenames, which are canonical for this repository.",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

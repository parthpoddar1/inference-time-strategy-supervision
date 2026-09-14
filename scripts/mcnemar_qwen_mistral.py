import collections
import glob
import json
import os
from pathlib import Path

from scipy.stats import binomtest

audit_dir = Path(__file__).resolve().parents[1] / "results" / "main" / "audits"
votes = collections.defaultdict(list)

for path in audit_dir.glob("correctness_audit_gpt_5_6_luna_replicate_*.json"):
    with open(path, encoding="utf-8") as handle:
        for item in json.load(handle):
            run, student, condition = os.path.splitext(item["source_file"])[0].split("_")
            if student == "qwen" and condition in {"normal", "mistral"}:
                votes[(run, item["question_id"], condition)].append(item["correctness"])


def is_correct(labels):
    return collections.Counter(labels).most_common(1)[0][0] == "CORRECT"


pairs = []
for run in ("run1", "run2", "run3"):
    question_ids = sorted(question_id for run_id, question_id, condition in votes if run_id == run and condition == "normal")
    for question_id in question_ids:
        pairs.append((is_correct(votes[(run, question_id, "normal")]), is_correct(votes[(run, question_id, "mistral")])))

normal_correct = sum(normal for normal, mistral in pairs)
mistral_correct = sum(mistral for normal, mistral in pairs)
rescue = sum(not normal and mistral for normal, mistral in pairs)
harm = sum(normal and not mistral for normal, mistral in pairs)

print(json.dumps({
    "n": len(pairs),
    "normal_correct": normal_correct,
    "mistral_correct": mistral_correct,
    "normal_pct": normal_correct / len(pairs) * 100,
    "mistral_pct": mistral_correct / len(pairs) * 100,
    "delta_pp": (mistral_correct - normal_correct) / len(pairs) * 100,
    "normal_wrong_mistral_correct": rescue,
    "normal_correct_mistral_wrong": harm,
    "discordant": rescue + harm,
    "exact_two_sided_p": binomtest(min(rescue, harm), rescue + harm, 0.5).pvalue,
}, indent=2))

import os
import json
import time
import random
import hashlib
from pathlib import Path
from collections import Counter

from datasets import load_dataset
from openai import OpenAI


# ============================================================
# CONTROLLED T0-T4 MATH EXPERIMENT
#
# 420 NEW MATH problems:
#   7 domains
#   20 Level 3 + 20 Level 4 + 20 Level 5 per domain
#
# T0-T4, one replicate.
#
# IMPORTANT:
#   - Previous questions are excluded automatically by scanning
#     every JSON under OLD_JSON_DIR for their "problem" text.
#   - A fixed seed creates a reproducible NEW 420-question selection.
#   - Each problem gets a randomized T0-T4 condition order.
#   - The condition order is saved in the manifest BEFORE API calls.
#   - Results are checkpointed after EVERY completed condition.
#   - Ctrl+C is safe. Rerunning this exact script resumes missing
#     problem+condition pairs without rerunning completed pairs.
#   - Provider fallback is DISABLED.
#
# Models:
#   Teacher: Qwen2.5-72B-Instruct
#   Student: Qwen2.5-7B-Instruct
#
# Provider pinning:
#   Teacher -> DeepInfra
#   Student -> Phala
#
# Do NOT put your API key in this file.
# Set OPENROUTER_API_KEY in the environment.
# ============================================================


# ----------------------------
# CONFIG
# ----------------------------

SEED = 20260818

TEACHER_MODEL = "qwen/qwen-2.5-72b-instruct"
STUDENT_MODEL = "qwen/qwen-2.5-7b-instruct"

# Provider names are passed to OpenRouter's provider routing.
# Fallbacks are explicitly disabled.
TEACHER_PROVIDER = "DeepInfra"
STUDENT_PROVIDER = "Phala"

DOMAINS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

TARGET_PER_LEVEL = 20
LEVELS = ["Level 3", "Level 4", "Level 5"]

# Put this script one directory ABOVE strategy_poc/.
# The script will scan strategy_poc recursively and exclude every
# previously used problem it can find in JSON files.
OLD_JSON_DIR = Path(".")

MANIFEST_FILE = Path("t0_t4_420_manifest.json")
RESULTS_FILE = Path("t0_t4_420_results.json")

MAX_TEACHER_TOKENS = 1024
MAX_STUDENT_TOKENS = 4096

TEMPERATURE = 0

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)


# ============================================================
# PROMPTS
# ============================================================

PROMPTS = {
    "T0_Current": """You are a mathematical reasoning teacher.

Given the problem below, provide a high-level strategic approach for solving
it. Focus on the key mathematical idea or structure the student should use.
Do not carry out the full calculation or give the final answer.

Problem:
{problem}
""",

    "T1_Verification": """You are a mathematical reasoning teacher.

Given the problem below, develop a high-level strategic approach for solving
it. Before giving the final strategy, independently verify that the strategy
is mathematically valid.

In particular, check for:
- incorrect assumptions
- omitted cases
- boundary/end-point issues
- sign errors
- counting or sample-space errors
- misuse of formulas, identities, or theorems
- whether the strategy actually answers the question asked

If your first approach has a flaw, replace it with a correct approach.

After checking it, provide ONLY the final verified high-level strategy.
Do not carry out the full calculation and do not give the final answer.

Problem:
{problem}
""",

    "T2_Minimal": """You are a mathematical reasoning teacher.

Given the problem below, identify the single most useful mathematical idea
or structural observation that should guide the student's solution.

Keep the strategy minimal and robust. Do not perform the full calculation,
do not derive the answer, and do not add unnecessary intermediate steps.
Avoid committing to specific arithmetic or casework unless it is essential
to identifying the correct method.

Provide a concise high-level strategy that the student can use to solve the
problem independently.

Problem:
{problem}
""",

    "T3_Multiple_Approaches": """You are a mathematical reasoning teacher.

Given the problem below, consider multiple plausible approaches internally.
Evaluate which approach is most reliable for this particular problem,
especially with respect to hidden assumptions, edge cases, signs, counting,
and algebraic or geometric constraints.

Select the most reliable approach and give the student ONLY that final
high-level strategy. Do not describe the alternatives you rejected.
Do not carry out the full calculation or give the final answer.

Problem:
{problem}
""",

    "T4_Adversarial": """You are a mathematical reasoning teacher.

Given the problem below, first develop a plausible high-level strategy.
Then actively try to find a flaw in that strategy: look for a counterexample,
missing case, boundary condition, sign mistake, invalid assumption, incorrect
counting model, or misuse of a mathematical identity/theorem.

If you find a flaw, repair the strategy before presenting it.

Provide ONLY the final high-level strategy that survives this self-check.
Do not give the full solution or final answer.

Problem:
{problem}
""",
}

STUDENT_PROMPT = """Solve the following mathematics problem completely.

A teacher has suggested the following high-level strategy. Use it as guidance,
but reason through the problem yourself and correct the strategy if it is
wrong or incomplete.

Teacher strategy:
{teacher_strategy}

Problem:
{problem}

Give a rigorous complete solution and final answer.
"""


# ============================================================
# HELPERS
# ============================================================

def stable_seed(text):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return SEED + int(digest[:8], 16)


def normalize_problem(text):
    return " ".join(str(text).strip().split())


def atomic_json_save(path, data):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# BUILD EXCLUSION SET
# ============================================================

def collect_previous_problems():
    previous = set()

    if not OLD_JSON_DIR.exists():
        raise RuntimeError(
            f"Cannot find {OLD_JSON_DIR}. Put this script one directory "
            "above your strategy_poc folder."
        )

    json_files = sorted(OLD_JSON_DIR.rglob("*.json"))

    print(f"Scanning {len(json_files)} previous JSON files for used problems...")

    for path in json_files:
        try:
            data = load_json(path)
        except Exception as e:
            print(f"Skipping unreadable JSON: {path} ({e})")
            continue

        if not isinstance(data, list):
            continue

        for item in data:
            if isinstance(item, dict) and item.get("problem"):
                previous.add(normalize_problem(item["problem"]))

    print(f"Unique previous problem texts found: {len(previous)}")
    return previous


# ============================================================
# LOAD MATH DOMAIN
# ============================================================

def load_domain(domain):
    ds = load_dataset("EleutherAI/hendrycks_math", domain)

    all_problems = []

    for split_name in ds:
        for x in ds[split_name]:
            all_problems.append({
                "problem": x["problem"],
                "solution": x["solution"],
                "level": x["level"],
                "subject": domain,
                "split": split_name,
            })

    # Deduplicate across train/test by problem text.
    seen = set()
    unique = []

    for x in all_problems:
        key = normalize_problem(x["problem"])
        if key not in seen:
            seen.add(key)
            unique.append(x)

    return unique


# ============================================================
# SELECT 60 NEW QUESTIONS PER DOMAIN
# ============================================================

def select_domain(domain, previous):
    problems = load_domain(domain)

    unused = [
        x for x in problems
        if normalize_problem(x["problem"]) not in previous
    ]

    by_level = {level: [] for level in LEVELS}

    for x in unused:
        if x["level"] in by_level:
            by_level[x["level"]].append(x)

    rng = random.Random(stable_seed(domain))

    selected = []

    for level in LEVELS:
        available = by_level[level]

        if len(available) < TARGET_PER_LEVEL:
            raise RuntimeError(
                f"{domain}: only {len(available)} unused {level} problems "
                f"available; need {TARGET_PER_LEVEL}."
            )

        selected.extend(
            rng.sample(available, TARGET_PER_LEVEL)
        )

    # Shuffle the 60 within the domain.
    rng.shuffle(selected)

    for i, x in enumerate(selected, 1):
        x = dict(x)
        x["id"] = f"{domain}_new_{i:03d}"
        x["selection_seed"] = stable_seed(domain)
        selected[i - 1] = x

    counts = Counter(x["level"] for x in selected)

    assert len(selected) == 60
    assert counts == Counter({
        "Level 3": 20,
        "Level 4": 20,
        "Level 5": 20,
    })

    return selected


# ============================================================
# BUILD / LOAD MANIFEST
# ============================================================

def build_manifest():
    previous = collect_previous_problems()

    questions = []

    for domain in DOMAINS:
        print(f"\nSelecting new questions: {domain}")
        selected = select_domain(domain, previous)
        questions.extend(selected)

        print(
            "  selected:",
            Counter(x["level"] for x in selected)
        )

    if len(questions) != 420:
        raise RuntimeError(f"Expected 420 questions, got {len(questions)}.")

    ids = [x["id"] for x in questions]

    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate IDs in new selection.")

    # Final global sanity check.
    expected = Counter({
        "Level 3": 140,
        "Level 4": 140,
        "Level 5": 140,
    })

    actual = Counter(x["level"] for x in questions)

    if actual != expected:
        raise RuntimeError(
            f"Global level distribution {actual}; expected {expected}."
        )

    # Randomize question order globally.
    question_rng = random.Random(SEED)
    question_rng.shuffle(questions)

    # Randomize T0-T4 order independently for EACH problem.
    conditions = list(PROMPTS.keys())

    for index, q in enumerate(questions, 1):
        rng = random.Random(stable_seed(q["id"]))
        order = conditions.copy()
        rng.shuffle(order)

        q["global_index"] = index
        q["condition_order"] = order
        q["randomization_seed"] = stable_seed(q["id"])

    manifest = {
        "experiment": "controlled_t0_t4_420_new_math",
        "seed": SEED,
        "teacher_model": TEACHER_MODEL,
        "student_model": STUDENT_MODEL,
        "teacher_provider": TEACHER_PROVIDER,
        "student_provider": STUDENT_PROVIDER,
        "provider_fallbacks": False,
        "temperature": TEMPERATURE,
        "domains": DOMAINS,
        "target_per_domain": {
            "Level 3": 20,
            "Level 4": 20,
            "Level 5": 20,
        },
        "total_questions": 420,
        "conditions": conditions,
        "questions": questions,
    }

    atomic_json_save(MANIFEST_FILE, manifest)

    print("\nMANIFEST CREATED")
    print(f"Saved to: {MANIFEST_FILE}")
    print("Questions:", len(questions))
    print("Levels:", Counter(q["level"] for q in questions))
    print("Domains:", Counter(q["subject"] for q in questions))

    return manifest


def load_or_create_manifest():
    if MANIFEST_FILE.exists():
        manifest = load_json(MANIFEST_FILE)

        if manifest.get("total_questions") != 420:
            raise RuntimeError("Existing manifest is not the expected 420-question experiment.")

        if manifest.get("conditions") != list(PROMPTS.keys()):
            raise RuntimeError("Existing manifest has different T0-T4 conditions.")

        print(f"Loaded existing manifest: {MANIFEST_FILE}")
        return manifest

    return build_manifest()


# ============================================================
# API CALL
# ============================================================

def response_text(response):
    return response.choices[0].message.content


def response_tokens(response):
    if getattr(response, "usage", None):
        return response.usage.total_tokens
    return None


def call(model, provider, messages, max_tokens):
    delay = 5
    attempt = 1

    while True:
        try:
            started = time.perf_counter()

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
                extra_body={
                    "provider": {
                        "order": [provider],
                        "allow_fallbacks": False,
                    }
                },
            )

            elapsed = time.perf_counter() - started

            if response.choices and response.choices[0].message.content:
                return response, elapsed

            print(
                f"Empty completion (attempt {attempt}). "
                f"Retrying in {delay}s...",
                flush=True,
            )

        except KeyboardInterrupt:
            print("\nStopped by user. Safe to rerun.")
            raise

        except Exception as e:
            print(
                f"API error on attempt {attempt}: {e}",
                flush=True,
            )
            print(
                f"Retrying in {delay}s...",
                flush=True,
            )

        time.sleep(delay)
        delay = min(delay * 2, 300)
        attempt += 1


# ============================================================
# RESULTS / RESUME
# ============================================================

def load_results():
    if not RESULTS_FILE.exists():
        return []

    data = load_json(RESULTS_FILE)

    if not isinstance(data, list):
        raise RuntimeError(f"{RESULTS_FILE} is not a JSON list.")

    return data


def result_key(problem_id, condition):
    return f"{problem_id}::{condition}"


def save_result(results):
    atomic_json_save(RESULTS_FILE, results)


# ============================================================
# RUN
# ============================================================

def main():
    print("=" * 80)
    print("CONTROLLED T0-T4 — 420 NEW MATH PROBLEMS")
    print("=" * 80)

    manifest = load_or_create_manifest()
    questions = manifest["questions"]

    results = load_results()

    completed = {
        result_key(x["id"], x["condition"])
        for x in results
        if x.get("student_solution")
    }

    total_conditions = 420 * 5

    print(f"Completed condition runs: {len(completed)}/{total_conditions}")
    print(f"Remaining condition runs: {total_conditions - len(completed)}")
    print(f"Teacher: {TEACHER_MODEL} via {TEACHER_PROVIDER}")
    print(f"Student: {STUDENT_MODEL} via {STUDENT_PROVIDER}")
    print("Provider fallback: DISABLED")
    print("=" * 80)

    for q in questions:
        pid = q["id"]
        problem = q["problem"]

        for condition in q["condition_order"]:

            key = result_key(pid, condition)

            if key in completed:
                continue

            print("\n" + "-" * 80)
            print(
                f"QUESTION {q['global_index']}/420 | "
                f"{pid} | {q['subject']} | {q['level']}"
            )
            print(f"CONDITION: {condition}")
            print("-" * 80)

            # ----------------------------
            # Teacher
            # ----------------------------

            print("Teacher call...", flush=True)

            teacher_response, teacher_time = call(
                TEACHER_MODEL,
                TEACHER_PROVIDER,
                [
                    {
                        "role": "user",
                        "content": PROMPTS[condition].format(
                            problem=problem
                        ),
                    }
                ],
                MAX_TEACHER_TOKENS,
            )

            strategy = response_text(teacher_response)

            print(
                f"Teacher complete: "
                f"{response_tokens(teacher_response)} tokens, "
                f"{teacher_time:.2f}s",
                flush=True,
            )

            # ----------------------------
            # Student
            # ----------------------------

            print("Student call...", flush=True)

            student_response, student_time = call(
                STUDENT_MODEL,
                STUDENT_PROVIDER,
                [
                    {
                        "role": "user",
                        "content": STUDENT_PROMPT.format(
                            teacher_strategy=strategy,
                            problem=problem,
                        ),
                    }
                ],
                MAX_STUDENT_TOKENS,
            )

            solution = response_text(student_response)

            print(
                f"Student complete: "
                f"{response_tokens(student_response)} tokens, "
                f"{student_time:.2f}s",
                flush=True,
            )

            # ----------------------------
            # Record
            # ----------------------------

            record = {
                "experiment": "controlled_t0_t4_420_new_math",
                "id": pid,
                "global_index": q["global_index"],
                "subject": q["subject"],
                "level": q["level"],
                "split": q.get("split"),
                "problem": problem,
                "reference_solution": q["solution"],

                "condition": condition,

                "teacher_model": TEACHER_MODEL,
                "student_model": STUDENT_MODEL,

                "teacher_provider_requested": TEACHER_PROVIDER,
                "student_provider_requested": STUDENT_PROVIDER,
                "provider_fallbacks": False,

                "temperature": TEMPERATURE,

                "teacher_strategy": strategy,
                "student_solution": solution,

                "teacher_tokens": response_tokens(teacher_response),
                "student_tokens": response_tokens(student_response),

                "teacher_time_seconds": teacher_time,
                "student_time_seconds": student_time,
                "total_api_time_seconds": teacher_time + student_time,

                "condition_order_for_problem": q["condition_order"],
                "randomization_seed": q["randomization_seed"],

                "completed_at_unix": time.time(),
            }

            results.append(record)
            completed.add(key)

            # CRITICAL:
            # Save AFTER EVERY CONDITION, not merely after every problem.
            save_result(results)

            print(
                f"SAVED {pid} / {condition} | "
                f"{len(completed)}/{total_conditions} conditions complete",
                flush=True,
            )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Results: {RESULTS_FILE}")
    print(f"Manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()

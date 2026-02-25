"""
Multi-model benchmark for reliability engineering questions.

Compares multiple models (e.g., Qwen3-14B, GPT-4o-mini) on the curated
dataset using an LLM judge. Supports resuming from previous runs and
deduplicates results automatically.

Uses OpenRouter API for both candidate models and the judge.
"""

import os
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Configuration

API_KEY = os.environ["OPENROUTER_API_KEY"]
INPUT_FILE = "dataset_alex.json"
OUTPUT_FILE = "benchmark_results.json"

MODELS_TO_TEST = [
    "qwen/qwen3-14b",
    "openai/gpt-4o-mini",
]
JUDGE_MODEL = "anthropic/claude-3.5-sonnet"

MAX_RETRIES = 5
BASE_DELAY = 2

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)


# Helper functions


def load_jsonl(path):
    """Load a JSON or JSONL file."""
    questions = []
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return questions


def ask_candidate_model(model, question_data):
    """Ask a candidate model to solve the problem with exponential backoff."""
    prompt = f"""You are a Reliability Engineering expert.
Solve the following problem.

Question: {question_data['question']}

Provide a clear, step-by-step reasoning.
IMPORTANT: You must state your final answer clearly at the very end, starting with "Final Answer:".
"""
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=16384,
                timeout=90,
            )
            content = completion.choices[0].message.content
            if not content or content.strip() == "":
                raise ValueError("Empty response received")
            return content
        except Exception as e:
            wait_time = BASE_DELAY * (2 ** attempt)
            print(f"  [{model}] Attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            time.sleep(wait_time)
    return "ERR_API"


def evaluate_answer(question, target_answer, candidate_answer):
    """Evaluate a candidate answer using the LLM judge."""
    if candidate_answer == "ERR_API":
        return False, "API Error"

    judge_prompt = f"""You are an impartial exam grader for Reliability Engineering.

**Task**: Compare the Student's Answer with the Target Answer (Ground Truth).

**Context**:
- Question: {question}
- Target Answer (Correct): {target_answer}
- Student's Answer: {candidate_answer}

**Grading Rules**:
1. **Mathematics**: If the student's result is numerically close (within ~5% margin), mark as CORRECT.
2. **Equivalence**: If the student derives a mathematically equivalent formula, mark as CORRECT.
3. **Reasoning**: Ignore minor wording differences. Focus on the final result/conclusion.

**Output Format**:
Reply with a SINGLE JSON OBJECT:
{{
  "is_correct": boolean,
  "explanation": "Short reason why"
}}
"""
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=30,
            )
            result = json.loads(completion.choices[0].message.content)
            return result.get("is_correct", False), result.get("explanation", "")
        except Exception as e:
            wait_time = BASE_DELAY * (2 ** attempt)
            print(f"  [Judge] Attempt {attempt + 1}/{MAX_RETRIES}: {e}")
            time.sleep(wait_time)
    return False, "Judge API Error"


def process_single_item(model_name, q):
    """Ask a question and have the judge evaluate the response."""
    candidate_resp = ask_candidate_model(model_name, q)
    is_correct, explanation = evaluate_answer(q["question"], q.get("answer"), candidate_resp)
    return {
        "question_title": q.get("title", "Unknown"),
        "target_answer": q.get("answer"),
        "model_prediction": candidate_resp,
        "is_correct": is_correct,
        "judge_explanation": explanation,
    }


# Main benchmark


def run_benchmark():
    questions = load_jsonl(INPUT_FILE)
    if not questions:
        print("No questions found.")
        return

    # Load existing results for resumption
    results = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}

    for model_name in MODELS_TO_TEST:
        print(f"\nEvaluating: {model_name}")

        if model_name not in results:
            results[model_name] = {"score": 0, "total": 0, "details": []}

        # Deduplicate existing results
        question_map = {}
        for d in results[model_name].get("details", []):
            title = d.get("question_title")
            pred = d.get("model_prediction")
            is_valid = pred and isinstance(pred, str) and pred.strip() and pred != "ERR_API"
            if title not in question_map:
                question_map[title] = d
            elif is_valid and not (question_map[title].get("model_prediction", "").strip()):
                question_map[title] = d

        results[model_name]["details"] = list(question_map.values())

        # Find questions that still need processing
        todo = []
        for q in questions:
            title = q.get("title")
            if title not in question_map:
                todo.append(q)
            else:
                pred = question_map[title].get("model_prediction")
                if not pred or (isinstance(pred, str) and not pred.strip()) or pred == "ERR_API":
                    todo.append(q)

        print(f"  Processing {len(todo)} questions (skipping {len(questions) - len(todo)} existing)")

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_q = {executor.submit(process_single_item, model_name, q): q for q in todo}
            count = 0
            for future in as_completed(future_to_q):
                try:
                    res = future.result()
                    status = "[OK]" if res["is_correct"] else "[FAIL]"
                    if res["model_prediction"] == "ERR_API":
                        status = "[ERROR]"
                    print(f"  {status} {res['question_title']}")

                    question_map[res["question_title"]] = res
                    results[model_name]["details"] = list(question_map.values())
                    count += 1

                    if count % 5 == 0:
                        correct = sum(1 for d in results[model_name]["details"] if d.get("is_correct"))
                        results[model_name]["score"] = correct
                        results[model_name]["total"] = len(results[model_name]["details"])
                        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"  Error: {e}")

        # Final score
        details = results[model_name]["details"]
        correct = sum(1 for d in details if d.get("is_correct"))
        total = len(details)
        results[model_name]["score"] = correct
        results[model_name]["total"] = total
        if total > 0:
            print(f"  Score: {correct}/{total} ({correct / total * 100:.2f}%)")

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run_benchmark()

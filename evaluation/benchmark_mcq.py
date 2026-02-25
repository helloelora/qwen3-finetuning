"""
MCQ benchmark for reliability engineering concepts.

Evaluates a model on multiple-choice questions from the CRE Handbook
to test conceptual understanding (as opposed to problem-solving).

Uses OpenRouter API.
"""

import os
import json
import time
from openai import OpenAI

# Configuration

API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "openai/gpt-4.1-mini"
INPUT_FILE = "cre_handbook_questions.jsonl"
OUTPUT_FILE = "benchmark_mcq_results.jsonl"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)


# Functions


def ask_question(question_data, max_retries=3):
    """Ask the model a multiple-choice question and extract the letter answer."""
    prompt = f"""You are a Reliability Engineering expert.

Answer this multiple choice question by providing ONLY the letter of the correct answer (A, B, C, or D).

Question: {question_data['question']}

Options:
{question_data['options']}

Reply with ONLY the letter."""

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            response_text = completion.choices[0].message.content.strip().upper()

            # Extract the letter
            for char in response_text:
                if char in "ABCD":
                    return char

            print(f"  Invalid response (attempt {attempt + 1}): {response_text}")
            time.sleep(1)
        except Exception as e:
            print(f"  Error (attempt {attempt + 1}): {e}")
            time.sleep(2)

    return None


def main():
    total_questions = 0
    wrong_answers = 0
    failed_questions = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} questions...\n")

    for i, line in enumerate(lines, 1):
        question_data = json.loads(line)
        total_questions += 1

        print(f"[{i}/{len(lines)}] {question_data.get('chunk_title', '')[:50]}...")
        model_answer = ask_question(question_data)

        if model_answer is None:
            print("  Failed after retries\n")
            failed_questions.append(question_data)
            wrong_answers += 1
            continue

        correct_answer = question_data["answer"]
        if model_answer != correct_answer:
            print(f"  Wrong: {model_answer} (expected: {correct_answer})\n")
            wrong_answers += 1
            question_data["model_answer"] = model_answer
            failed_questions.append(question_data)
        else:
            print(f"  Correct: {model_answer}\n")

        time.sleep(0.5)

    # Save failed questions
    if failed_questions:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for q in failed_questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Display results
    print(f"Total: {total_questions}")
    print(f"Correct: {total_questions - wrong_answers}")
    print(f"Wrong: {wrong_answers}")
    print(f"Accuracy: {(total_questions - wrong_answers) / total_questions * 100:.2f}%")


if __name__ == "__main__":
    main()

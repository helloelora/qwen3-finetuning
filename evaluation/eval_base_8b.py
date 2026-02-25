"""
Base model evaluation: Qwen3-8B (no fine-tuning, no LoRA)

Evaluates the base Qwen3-8B model in 4-bit quantization to establish
the 8B baseline for comparison with the fine-tuned 8B variant.

Configuration:
    - Model: Qwen3-8B (4-bit quantized via Unsloth)
    - No LoRA adapters
    - Eval: temp=0.6 sampling, 16K max tokens, native thinking
    - Judge: Claude 3.5 Sonnet via OpenRouter

Result: 62.26% (53 samples, single pass)
"""

import os
import gc
import json
import re
import time
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI
from datasets import load_dataset
from unsloth import FastLanguageModel

# Configuration

MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"  # 8B model
MAX_SEQ_LEN = 8192
DATASET_PATH = "dataset_alex.json"

EVAL_THINKING_MODE = True
EVAL_MAX_NEW_TOKENS = 16384

JUDGE_API_KEY = os.environ["OPENROUTER_API_KEY"]
JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "anthropic/claude-3.5-sonnet"

client = OpenAI(
    api_key=JUDGE_API_KEY,
    base_url=JUDGE_BASE_URL,
    default_headers={
        "HTTP-Referer": "https://github.com/helloelora",
        "X-Title": "Reliability-Eval",
    },
)

SYSTEM_PROMPT = """You are a Reliability Engineering Expert.
Solve the user's problem step-by-step with rigorous mathematical reasoning.
Use LaTeX for mathematical formulas.
Be concise: focus on the key calculation steps, avoid repeating the question or adding unnecessary preamble.
Always conclude with a clearly stated final answer including numerical values and units when applicable."""


# Answer extraction


def extract_final_answer(text):
    """Extract the final answer from a Qwen3 response, handling <think> blocks."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in cleaned:
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL).strip()
    if not cleaned:
        after_think = re.split(r"</think>", text)
        if len(after_think) > 1:
            cleaned = after_think[-1].strip()
        else:
            fa_match = re.search(r"\*\*Final Answer[:\*]*\*\*(.+)", text, re.DOTALL)
            if fa_match:
                cleaned = "**Final Answer:**" + fa_match.group(1).strip()
            else:
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 20]
                cleaned = paragraphs[-1] if paragraphs else "[No final answer produced]"
    return _truncate_repetitions(cleaned)


def _truncate_repetitions(text, max_repeats=3):
    """Detect and truncate excessive repetitions."""
    lines = text.split("\n")
    result_lines, prev_line, repeat_count = [], None, 0
    for line in lines:
        stripped = line.strip()
        if stripped == prev_line and stripped:
            repeat_count += 1
            if repeat_count >= max_repeats:
                continue
        else:
            repeat_count, prev_line = 0, stripped
        result_lines.append(line)
    result = "\n".join(result_lines)
    result = re.sub(r"(\b\w{3,30}\b)(\s+\1){4,}", r"\1", result)
    parts = result.split("**Final Answer:**")
    if len(parts) > 2:
        result = parts[0] + "**Final Answer:**" + parts[1]
    return result.strip()


# Generation


@torch.inference_mode()
def generate_answer(model, tokenizer, question):
    """Generate an answer using the base model with native thinking."""
    FastLanguageModel.for_inference(model)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=EVAL_THINKING_MODE, return_tensors="pt",
    ).to("cuda")
    gen_kwargs = dict(
        max_new_tokens=EVAL_MAX_NEW_TOKENS, use_cache=True, do_sample=True,
        temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
        repetition_penalty=1.15, no_repeat_ngram_size=10,
    )
    outputs = model.generate(input_ids, **gen_kwargs)
    output_ids = outputs[0][input_ids.shape[1]:].tolist()
    raw = tokenizer.decode(output_ids, skip_special_tokens=False)
    n_tokens = len(output_ids)
    had_thinking = "<think>" in raw
    raw_clean = tokenizer.decode(output_ids, skip_special_tokens=True)
    return extract_final_answer(raw_clean), n_tokens, had_thinking, raw_clean


# LLM Judge


def judge_single(sample, student_answer):
    """Evaluate a student answer against ground truth using an LLM judge."""
    judge_prompt = f"""You are a STRICT impartial exam grader for Reliability Engineering.

Compare Student's Answer with the Ground Truth.

--- QUESTION ---
{sample['question']}

--- GROUND TRUTH ---
{sample['answer']}

--- STUDENT'S ANSWER ---
{student_answer}

GRADING RULES:
1. GIBBERISH: If the student's answer is repetitive, nonsensical, empty, or contains unrelated spam, mark is_correct=false.
2. STRICT MATH: Numerical final result must be within 5.0% margin of the ground truth (when applicable).
3. LOGIC: If the student provides reasoning, it must be coherent; do not invent missing steps.
4. PARTIAL CREDIT: If the student's reasoning is sound and arrives at the right concept/formula but has minor rounding differences, still mark is_correct=true.
5. OUTPUT: Return ONLY valid JSON with keys:
   - "is_correct": boolean
   - "explanation": string (brief justification)
"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL, messages=[{"role": "user", "content": judge_prompt}],
                temperature=0, response_format={"type": "json_object"},
            )
            raw_content = resp.choices[0].message.content
            try:
                data = json.loads(raw_content)
            except json.JSONDecodeError:
                match = re.search(r'\{.*?"is_correct".*?\}', raw_content, re.DOTALL)
                data = json.loads(match.group()) if match else {"is_correct": False, "explanation": "Parse error"}
            return {
                "question": sample["question"], "target": sample["answer"],
                "student_answer": student_answer,
                "is_correct": bool(data.get("is_correct", False)),
                "explanation": str(data.get("explanation", "")),
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return {
                "question": sample.get("question", ""), "target": sample.get("answer", ""),
                "student_answer": student_answer, "is_correct": False,
                "explanation": f"Judge API error: {e}",
            }


# Main

if __name__ == "__main__":
    print(f"Loading 8B base model: {MODEL_NAME} (NO LoRA)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True,
    )

    full_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"Loaded {len(full_dataset)} samples")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"base_model_8b_results_{run_timestamp}.json"
    n_samples = len(full_dataset)

    FastLanguageModel.for_inference(model)
    all_results, correct, token_counts = [], 0, []

    for i in tqdm(range(n_samples), desc="Evaluating 8B base model"):
        sample = full_dataset[i]
        ans, n_tok, had_thinking, raw_response = generate_answer(model, tokenizer, sample["question"])
        judged = judge_single(sample, ans)
        judged.update({"sample_index": i, "token_count": n_tok, "had_thinking": had_thinking, "raw_response": raw_response})
        all_results.append(judged)
        token_counts.append(n_tok)
        if judged["is_correct"]:
            correct += 1

        if (i + 1) % 20 == 0 or (i + 1) == n_samples:
            json_results = {
                "metadata": {"model": MODEL_NAME, "model_type": "base_model_8b_4bit",
                             "judge_model": JUDGE_MODEL, "total_samples": n_samples, "timestamp": run_timestamp},
                "summary": {"accuracy_pct": correct / (i + 1) * 100, "correct": correct, "total": i + 1},
                "results": all_results,
            }
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)

    accuracy = correct / n_samples * 100
    print(f"\n8B BASE MODEL: {accuracy:.2f}% ({correct}/{n_samples})")
    print(f"Results saved to: {results_file}")

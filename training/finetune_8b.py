"""
8B model fine-tuning: Qwen3-8B with LoRA (r=16, alpha=16)

Same configuration as the baseline but using the smaller 8B model.
Tests whether a smaller model benefits more from domain-specific fine-tuning.

Configuration:
    - Model: Qwen3-8B (4-bit quantized via Unsloth)
    - LoRA: r=16, alpha=16
    - Training: 2 epochs, lr=1e-5, cosine scheduler
    - Eval: temp=0.6 sampling, 16K max tokens
    - Judge: Claude 3.5 Sonnet via OpenRouter
    - Data format: <think>reasoning</think> + answer (native thinking)

Result: 64.00% ± 9.73% (5-fold CV on 53 samples)
"""

import os
import gc
import json
import random
import re
import time
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from openai import OpenAI
from sklearn.model_selection import KFold
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

# Configuration

SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"  # 8B instead of 14B
MAX_SEQ_LEN = 8192
DATASET_PATH = "dataset_alex.json"

LORA_R = 16
LORA_ALPHA = 16
NUM_EPOCHS = 2
LEARNING_RATE = 1e-5

EVAL_THINKING_MODE = True
EVAL_MAX_NEW_TOKENS = 16384
EVAL_TEMPERATURE = 0.6
EVAL_MAX_SAMPLES_PER_FOLD = None

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


# Data preparation


def build_training_texts(examples, tokenizer, rng=None, log_tokens=False):
    """Format training samples using Qwen3's native thinking mode."""
    if rng is None:
        rng = random.Random(SEED)

    questions = examples["question"]
    reasonings = examples["reasoning"]
    answers = examples["answer"]
    texts = []
    token_lengths = []

    for q, r, a in zip(questions, reasonings, answers):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"<think>\n{r}\n</think>\n\n**Final Answer:** {a}"},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=True,
        )
        texts.append(text)
        token_lengths.append(len(tokenizer.encode(text)))

    if log_tokens and token_lengths:
        print(f"  Token stats: min={min(token_lengths)}, max={max(token_lengths)}, "
              f"mean={np.mean(token_lengths):.0f}")

    return {"text": texts}


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
    """Detect and truncate excessive line/word repetitions."""
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
    """Generate an answer using Qwen3 with native thinking mode."""
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
        temperature=EVAL_TEMPERATURE, top_p=0.95, top_k=20, min_p=0.0,
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


def evaluate_fold(model, tokenizer, dataset, fold_num):
    """Evaluate a validation fold and return accuracy + detailed results."""
    n = len(dataset) if EVAL_MAX_SAMPLES_PER_FOLD is None else min(EVAL_MAX_SAMPLES_PER_FOLD, len(dataset))
    detailed, correct, token_counts = [], 0, []
    for i in tqdm(range(n), desc=f"Eval fold {fold_num + 1}"):
        sample = dataset[i]
        ans, n_tok, had_thinking, raw_response = generate_answer(model, tokenizer, sample["question"])
        judged = judge_single(sample, ans)
        judged.update({"fold": fold_num + 1, "token_count": n_tok, "had_thinking": had_thinking, "raw_response": raw_response})
        detailed.append(judged)
        token_counts.append(n_tok)
        correct += 1 if judged["is_correct"] else 0
    accuracy = correct / max(n, 1)
    print(f"  Fold {fold_num + 1}: {accuracy * 100:.2f}% ({correct}/{n})")
    with open(f"eval_details_8b_fold_{fold_num + 1}.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)
    return accuracy, detailed


def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# Main

if __name__ == "__main__":
    full_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"Loaded {len(full_dataset)} samples")

    N_FOLDS = 5
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_accuracies, all_results = [], []
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"evaluation_results_8b_{run_timestamp}.json"

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset)))):
        print(f"\nFOLD {fold + 1}/{N_FOLDS} — Train: {len(train_idx)}, Val: {len(val_idx)}")
        clear_cuda()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=LORA_R, lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0, use_gradient_checkpointing="unsloth", random_state=SEED,
        )

        train_ds = full_dataset.select(train_idx).map(
            lambda x: build_training_texts(x, tokenizer, log_tokens=(fold == 0)),
            batched=True, remove_columns=full_dataset.column_names,
        )

        trainer = SFTTrainer(
            model=model, processing_class=tokenizer, train_dataset=train_ds,
            args=SFTConfig(
                dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
                per_device_train_batch_size=1, gradient_accumulation_steps=4,
                warmup_steps=5, num_train_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
                optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="cosine",
                seed=SEED, logging_steps=5, output_dir=f"outputs_8b_fold_{fold}",
                save_strategy="no", report_to="none",
            ),
        )
        trainer.train()

        acc, fold_results = evaluate_fold(model, tokenizer, full_dataset.select(val_idx), fold)
        fold_accuracies.append(acc * 100)
        all_results.extend(fold_results)

        json_results = {
            "metadata": {"variant": "8b", "model": MODEL_NAME, "lora_r": LORA_R,
                         "num_epochs": NUM_EPOCHS, "eval_temperature": EVAL_TEMPERATURE,
                         "judge_model": JUDGE_MODEL, "seed": SEED, "timestamp": run_timestamp},
            "summary": {"fold_accuracies": fold_accuracies,
                        "mean_accuracy": float(np.mean(fold_accuracies)),
                        "std_accuracy": float(np.std(fold_accuracies)) if len(fold_accuracies) > 1 else 0.0},
            "all_results": all_results,
        }
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        del trainer, model, tokenizer
        clear_cuda()

    print(f"\n8B FINE-TUNING COMPLETE — Mean: {np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}%")

"""
Best configuration: Qwen3-14B with stronger LoRA (r=64) + greedy decoding

Fine-tunes Qwen3-14B-4bit with a higher-rank LoRA adapter and evaluates
using deterministic (greedy) decoding for reproducible results.

Configuration:
    - Model: Qwen3-14B (4-bit quantized via Unsloth)
    - LoRA: r=64, alpha=64 (4x more capacity than baseline)
    - Training: 3 epochs, lr=1e-5, cosine scheduler
    - Eval: greedy decoding (temp=0), 16K max tokens
    - Judge: Claude 3.5 Sonnet via OpenRouter
    - Data format: <think>reasoning</think> + answer (native thinking)

Result: 71.64% ± 5.90% (5-fold CV on 53 samples) — BEST configuration
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

MODEL_NAME = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
MAX_SEQ_LEN = 8192
DATASET_PATH = "dataset_alex.json"

# LoRA configuration — stronger adaptation
LORA_R = 64
LORA_ALPHA = 64
NUM_EPOCHS = 3
LEARNING_RATE = 1e-5

# Evaluation configuration — deterministic
EVAL_THINKING_MODE = True
EVAL_MAX_NEW_TOKENS = 16384
EVAL_TEMPERATURE = 0.0  # Greedy decoding
EVAL_MAX_SAMPLES_PER_FOLD = None

# Judge configuration (OpenRouter)
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
              f"mean={np.mean(token_lengths):.0f}, "
              f"over {MAX_SEQ_LEN}: {sum(1 for t in token_lengths if t > MAX_SEQ_LEN)}")

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
    result_lines = []
    prev_line = None
    repeat_count = 0

    for line in lines:
        stripped = line.strip()
        if stripped == prev_line and stripped:
            repeat_count += 1
            if repeat_count >= max_repeats:
                continue
        else:
            repeat_count = 0
            prev_line = stripped
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
    """Generate an answer using greedy decoding (deterministic)."""
    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=EVAL_THINKING_MODE, return_tensors="pt",
    ).to("cuda")

    # Greedy decoding for deterministic evaluation
    if EVAL_TEMPERATURE == 0.0:
        gen_kwargs = dict(
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            use_cache=True,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=10,
        )
    else:
        gen_kwargs = dict(
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            use_cache=True,
            do_sample=True,
            temperature=EVAL_TEMPERATURE,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            repetition_penalty=1.15,
            no_repeat_ngram_size=10,
        )

    outputs = model.generate(input_ids, **gen_kwargs)
    output_ids = outputs[0][input_ids.shape[1]:].tolist()
    raw = tokenizer.decode(output_ids, skip_special_tokens=False)

    n_tokens = len(output_ids)
    had_thinking = "<think>" in raw
    raw_clean = tokenizer.decode(output_ids, skip_special_tokens=True)
    answer = extract_final_answer(raw_clean)

    return answer, n_tokens, had_thinking, raw_clean


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
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw_content = resp.choices[0].message.content
            try:
                data = json.loads(raw_content)
            except json.JSONDecodeError:
                match = re.search(r'\{.*?"is_correct".*?\}', raw_content, re.DOTALL)
                data = json.loads(match.group()) if match else {"is_correct": False, "explanation": "Parse error"}

            return {
                "question": sample["question"],
                "target": sample["answer"],
                "student_answer": student_answer,
                "is_correct": bool(data.get("is_correct", False)),
                "explanation": str(data.get("explanation", "")),
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return {
                "question": sample.get("question", ""),
                "target": sample.get("answer", ""),
                "student_answer": student_answer,
                "is_correct": False,
                "explanation": f"Judge API error: {e}",
            }


def evaluate_fold(model, tokenizer, dataset, fold_num):
    """Evaluate a validation fold and return accuracy + detailed results."""
    n = len(dataset) if EVAL_MAX_SAMPLES_PER_FOLD is None else min(EVAL_MAX_SAMPLES_PER_FOLD, len(dataset))
    detailed = []
    correct = 0
    token_counts = []

    for i in tqdm(range(n), desc=f"Eval fold {fold_num + 1}"):
        sample = dataset[i]
        ans, n_tok, had_thinking, raw_response = generate_answer(model, tokenizer, sample["question"])
        judged = judge_single(sample, ans)
        judged.update({"fold": fold_num + 1, "token_count": n_tok, "had_thinking": had_thinking, "raw_response": raw_response})
        detailed.append(judged)
        token_counts.append(n_tok)
        correct += 1 if judged["is_correct"] else 0

    accuracy = correct / max(n, 1)
    print(f"  Fold {fold_num + 1}: {accuracy * 100:.2f}% ({correct}/{n}), "
          f"tokens: mean={np.mean(token_counts):.0f}, max={max(token_counts)}")

    out_path = f"eval_details_strong_lora_fold_{fold_num + 1}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)

    return accuracy, detailed


# Memory management


def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# Main: 5-fold cross-validation

if __name__ == "__main__":
    print(f"Loading dataset from {DATASET_PATH}...")
    full_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"Loaded {len(full_dataset)} samples")

    N_FOLDS = 5
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_accuracies = []
    all_results = []
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"evaluation_results_strong_lora_{run_timestamp}.json"

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset)))):
        print(f"FOLD {fold + 1}/{N_FOLDS} — Train: {len(train_idx)}, Val: {len(val_idx)}")

        for name in ["model", "tokenizer", "trainer"]:
            if name in dir():
                exec(f"del {name}")
        clear_cuda()

        # Load model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True,
        )

        # Add LoRA — stronger adaptation (r=64)
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
        )

        # Prepare training data
        train_ds = full_dataset.select(train_idx)
        train_ds = train_ds.map(
            lambda x: build_training_texts(x, tokenizer, log_tokens=(fold == 0)),
            batched=True,
            remove_columns=train_ds.column_names,
        )

        # Train — 3 epochs for more convergence
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            args=SFTConfig(
                dataset_text_field="text",
                max_seq_length=MAX_SEQ_LEN,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                num_train_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=SEED,
                logging_steps=5,
                output_dir=f"outputs_strong_lora_fold_{fold}",
                save_strategy="no",
                report_to="none",
            ),
        )
        trainer.train()

        # Evaluate
        val_ds = full_dataset.select(val_idx)
        acc, fold_results = evaluate_fold(model, tokenizer, val_ds, fold)
        fold_accuracies.append(acc * 100)
        all_results.extend(fold_results)

        # Save incremental results
        json_results = {
            "metadata": {
                "variant": "strong_lora",
                "model": MODEL_NAME,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "eval_temperature": EVAL_TEMPERATURE,
                "judge_model": JUDGE_MODEL,
                "n_folds": N_FOLDS,
                "folds_completed": fold + 1,
                "total_samples": len(full_dataset),
                "seed": SEED,
                "timestamp": run_timestamp,
            },
            "summary": {
                "fold_accuracies": fold_accuracies,
                "mean_accuracy": float(np.mean(fold_accuracies)),
                "std_accuracy": float(np.std(fold_accuracies)) if len(fold_accuracies) > 1 else 0.0,
            },
            "all_results": all_results,
        }
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        del trainer, model, tokenizer
        clear_cuda()

    # Final summary
    print(f"STRONG LoRA COMPLETE — Mean: {np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}%")
    print(f"Fold accuracies: {[f'{a:.2f}%' for a in fold_accuracies]}")
    print(f"Results saved to: {results_file}")

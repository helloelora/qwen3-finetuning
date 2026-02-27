"""
Qwen3-14B Fine-tuning — Final Answer Only (Alex's idea)
========================================================
Fine-tunes ONLY on the final answer (no reasoning at all).
The model keeps its native <think> process entirely intact.

Hypothesis: By only teaching the model WHAT to answer (not HOW to reason),
we preserve its native chain-of-thought while steering it toward correct results.

Key difference from other variants:
- Strong LoRA:    trains on <think>{reasoning}</think> + answer  → overwrites reasoning
- Answer-only:    trains on {reasoning} + answer as flat text    → overwrites reasoning
- THIS SCRIPT:    trains on answer ONLY                          → preserves native thinking

Config: r=64, alpha=64, 3 epochs, greedy eval (same as Strong LoRA)
Training: enable_thinking=True (model sees <think>...</think> + answer, but assistant
          content is ONLY the final answer — model fills in thinking on its own)
Eval:     enable_thinking=True, temperature=0.0 (greedy, deterministic)
"""

import json, re, gc, os, time
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import KFold
from datasets import load_dataset
from openai import OpenAI

# ─── Configuration ───────────────────────────────────────────────

SEED = 42
MODEL_NAME = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
MAX_SEQ_LEN = 4096
DATASET_PATH = "dataset_alex.json"

# LoRA — same strong config as best variant
LORA_R = 64
LORA_ALPHA = 64
NUM_EPOCHS = 3
LEARNING_RATE = 1e-5

# Training thinking mode — True so the model sees the <think> structure
# but assistant content is ONLY the final answer
TRAIN_THINKING_MODE = True

# Eval — greedy decoding, native thinking enabled
EVAL_THINKING_MODE = True
EVAL_TEMPERATURE = 0.0
EVAL_MAX_NEW_TOKENS = 8192
EVAL_MAX_SAMPLES_PER_FOLD = None

# Judge
JUDGE_MODEL = "anthropic/claude-3.5-sonnet"

SYSTEM_PROMPT = (
    "You are a Reliability Engineering expert. "
    "Solve the given problem step by step, then provide a clear final answer."
)

# OpenRouter API
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)


# ─── Data preparation ───────────────────────────────────────────


def build_training_texts(examples, tokenizer, log_tokens=False):
    """
    Build training texts with ONLY the final answer as assistant content.
    
    The model's native <think> process is preserved because we don't provide
    any reasoning — only the correct answer. During training with
    enable_thinking=True, the template includes the <think>...</think> structure,
    but the assistant's content is just the answer.
    """
    texts = []
    for q, a in zip(examples["question"], examples["answer"]):
        # Assistant content = ONLY the final answer, no reasoning at all
        assistant_content = f"**Final Answer:** {a}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": assistant_content},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=TRAIN_THINKING_MODE,
        )
        texts.append(text)

    if log_tokens and texts:
        tokens = tokenizer(texts[0], return_tensors="pt")
        print(f"[Final-Answer-Only] Sample token count: {tokens['input_ids'].shape[1]}")
        print(f"[Final-Answer-Only] Sample text preview:\n{texts[0][:500]}...")

    return {"text": texts}


# ─── Answer extraction ───────────────────────────────────────────


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


# ─── Generation ──────────────────────────────────────────────────


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


# ─── LLM Judge ───────────────────────────────────────────────────


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

    out_path = f"eval_details_final_answer_only_fold_{fold_num + 1}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)

    return accuracy, detailed


# ─── Memory management ───────────────────────────────────────────


def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ─── Main: 5-fold cross-validation ──────────────────────────────

if __name__ == "__main__":
    print(f"Loading dataset from {DATASET_PATH}...")
    full_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"Loaded {len(full_dataset)} samples")

    N_FOLDS = 5
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_accuracies = []
    all_results = []
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"evaluation_results_final_answer_only_{run_timestamp}.json"

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

        # Add LoRA — same strong config as best variant
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
        )

        # Prepare training data — ONLY final answers, no reasoning
        train_ds = full_dataset.select(train_idx)
        train_ds = train_ds.map(
            lambda x: build_training_texts(x, tokenizer, log_tokens=(fold == 0)),
            batched=True,
            remove_columns=train_ds.column_names,
        )

        # Train — 3 epochs
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
                output_dir=f"outputs_final_answer_only_fold_{fold}",
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
                "variant": "final_answer_only",
                "description": "Fine-tune on final answer only — no reasoning. Model preserves native <think> process.",
                "model": MODEL_NAME,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "num_epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "train_thinking_mode": TRAIN_THINKING_MODE,
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
    print(f"FINAL ANSWER ONLY COMPLETE — Mean: {np.mean(fold_accuracies):.2f}% ± {np.std(fold_accuracies):.2f}%")
    print(f"Fold accuracies: {[f'{a:.2f}%' for a in fold_accuracies]}")
    print(f"Results saved to: {results_file}")

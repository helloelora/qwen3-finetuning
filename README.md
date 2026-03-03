# Fine-Tuning Qwen3 for Reliability Engineering

Fine-tuning [Qwen3](https://huggingface.co/Qwen) language models on domain-specific reliability engineering Q&A using **LoRA** and **Unsloth**, with LLM-as-judge evaluation via **Claude 3.5 Sonnet**.

---

## Part 1 — Best Results

All final experiments use **5-fold cross-validation** on a curated 53-sample dataset, with **Claude 3.5 Sonnet** as the LLM judge and **5% numerical tolerance**.

### Summary

| Configuration | Model | LoRA rank | Epochs | Eval temp | Accuracy | Script |
|---|---|---|---|---|---|---|
| Base (no FT) | Qwen3-14B | — | — | 0.6 | 69.90% (on 103 samples) | `eval_base_14b.py` |
| Base (no FT, greedy) | Qwen3-14B | — | — | 0 | **79.25%** (42/53) | `eval_base_14b_greedy.py` |
| Base (no FT) | Qwen3-8B | — | — | 0.6 | 62.26% (on 53 samples) | `eval_base_8b.py` |
| Baseline FT | Qwen3-14B | r=16 | 2 | 0.6 | 68.18% ± 8.61% | `finetune_baseline.py` |
| **Strong LoRA** | **Qwen3-14B** | **r=64** | **3** | **0** | **71.64% ± 5.90%** | **`finetune_strong_lora.py`** |
| Answer-only | Qwen3-14B | r=16 | 2 | 0.6 | 70.18% ± 16.04% | `finetune_answer_only.py` |
| 8B variant | Qwen3-8B | r=16 | 2 | 0.6 | 64.00% ± 9.73% | `finetune_8b.py` |
| Strong LoRA ext. | Qwen3-14B | r=64 | 3 | 0 | 25.38% ± 12.62% | `finetune_strong_lora_extended.py` |
| Final Answer Only | Qwen3-14B | r=64 | 3 | 0 | 60.55% ± 5.85% | `finetune_final_answer_only.py` |

> **Best fine-tuned configuration**: Strong LoRA (r=64, 3 epochs, greedy decoding) — **71.64% ± 5.90%**. However, the **base model with greedy decoding scores 79.25%**, outperforming all fine-tuned variants by a significant margin. See Phase 8 for analysis.

### Per-Fold Breakdown

**Baseline FT** (14B, r=16, α=16, 2 epochs, lr=1e-5, temp=0.6, 53 samples):

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|
| 72.73% (8/11) | 54.55% (6/11) | 63.64% (7/11) | 80.00% (8/10) | 70.00% (7/10) | **68.18% ± 8.61%** |

**Strong LoRA** (14B, r=64, α=64, 3 epochs, lr=1e-5, greedy, 53 samples):

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|
| 63.64% (7/11) | 72.73% (8/11) | 81.82% (9/11) | 70.00% (7/10) | 70.00% (7/10) | **71.64% ± 5.90%** |

**Answer-Only** (14B, r=16, α=16, 2 epochs, lr=2e-5, temp=0.6, no `<think>` in training, 53 samples):

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|
| 63.64% (7/11) | 54.55% (6/11) | 72.73% (8/11) | 100.00% (10/10) | 60.00% (6/10) | **70.18% ± 16.04%** |

**8B Variant** (8B, r=16, α=16, 2 epochs, lr=1e-5, temp=0.6, 53 samples):

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|
| 72.73% (8/11) | 72.73% (8/11) | 54.55% (6/11) | 70.00% (7/10) | 50.00% (5/10) | **64.00% ± 9.73%** |

**Strong LoRA Extended** (14B, r=64, α=64, 3 epochs, lr=1e-5, greedy, 100 samples — `dataset_alex_extended`):

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|
| 19.05% (4/21) | 23.81% (5/21) | 19.05% (4/21) | 50.00% (10/20) | 15.00% (3/20) | **25.38% ± 12.62%** |

**Final Answer Only** (14B, r=64, α=64, 3 epochs, lr=1e-5, greedy, answer-only training, 53 samples):

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|
| 54.55% (6/11) | 54.55% (6/11) | 63.64% (7/11) | 60.00% (6/10) | 70.00% (7/10) | **60.55% ± 5.85%** |

### Key Takeaways

- **The base model (greedy) outperforms all fine-tuned variants** — 79.25% vs. 71.64% for the best fine-tune (Strong LoRA). This is the single most important finding of the project.
- **Strong LoRA is the best fine-tuned config** — highest mean accuracy with the lowest standard deviation among fine-tuned models.
- **Answer-only** reaches similar mean accuracy but has extreme fold variance (54% to 100%), making it unreliable.
- **8B is too small** — the 8B model underperforms even the base 14B, confirming that model capacity matters for this domain.
- **Final Answer Only underperformed** — 60.55%, below both the base model (79.25%) and Strong LoRA (71.64%). Training only on the final answer, while preserving the model's native `<think>` reasoning, was not enough to steer the model toward correct domain-specific solutions. The model needs consistent worked examples, not just answers.
- **Scaling the dataset destroyed performance** — going from 53 curated → 100 samples (53 original + 30 augmented + 17 from a different textbook) caused catastrophic collapse from 71.64% to 25.38%. The additional samples — particularly the 17 from a second textbook with a different answer format — likely introduced format pollution that degraded the model's output structure.
- **Fine-tuning may not help for this domain with small data** — The base 14B greedy (79.25%) beats every fine-tuned model. This suggests that Qwen3-14B already has strong reliability engineering knowledge, and fine-tuning on only 53 samples introduces more noise than signal. The model's pre-training knowledge is more valuable than our small-scale domain adaptation.

---

## Part 2 — Experimental Journey

This section documents every experiment chronologically, including dead ends, debugging, and the lessons learned along the way.

### Phase 1: First Attempt (No Cross-Validation)

**Setup**: Simple train/test split (47 train / 9 test) on the initial 56-sample dataset (`dataset_alex` before curation). Fine-tuned Qwen3-14B with r=32, 2 epochs on the RUCHE HPC cluster. Evaluated with GPT-4o-mini as judge.

**Result**: **37.5%** (3/8 correct on evaluation subset).

**Problem**: No cross-validation, tiny test set, no systematic evaluation. The model produced outputs but we had no way to trust the score.

### Phase 2: Hyperparameter Search (GPT-4o-mini Judge, 56 Samples)

Switched to 5-fold CV on the 56-sample dataset. All runs on RUCHE. All experiments in this phase used GPT-4o-mini as the LLM judge.

We iterated through several configurations, progressively reducing the learning rate and LoRA rank:

| Round | lr | r / α | Notes | Accuracy | Status |
|---|---|---|---|---|---|
| 1 | 2e-4 | 32 / 32 | reasoning_ratio=0.75 | 23.03% ± 10.25% | 5/5 folds |
| 2 | 5e-5 | 32 / 32 | reasoning_ratio=1.0 | ~38.38% | 3/5 folds |
| 3 | 5e-5 | 32 / 32 | different split | 25.0% | 1/5 folds |
| 4 | 2e-5 | 32 / 32 | eval_thinking=False | 16.67% | 1/5 folds |
| 7 | **1e-5** | **16 / 16** | 2 epochs, thinking on | **32.27% ± 9.58%** | 5/5 folds |

**Key lessons:**
- **lr=2e-4 was catastrophic** — too aggressive for a 14B model, caused forgetting.
- **Disabling thinking at eval killed performance** — Round 4 (16.67%) proved that the `<think>` chain is essential.
- **lr=1e-5 with r=16** was the sweet spot — Round 7 became the base config for all subsequent experiments.

Round 7 still only scored ~32%, which seemed disappointing. But then something unexpected happened...

### Phase 3: The Judge Discovery

We re-evaluated the **exact same model outputs** from Round 7 using Claude 3.5 Sonnet instead of GPT-4o-mini:

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|
| 41.67% | 63.64% | 63.64% | 54.55% | 81.82% | **61.06% ± 13.14%** |

**Same model, same outputs — the score jumped from 32% to 61%.**

GPT-4o-mini was an unreliable judge. It was penalizing correct answers that used different notation, equivalent forms, or slightly different rounding. Claude was far more accurate at assessing mathematical equivalence and domain-specific reasoning.

**Lesson**: The choice of LLM judge matters enormously. From this point on, all evaluations used Claude 3.5 Sonnet.

### Phase 4: Expanding the Dataset

With the Round 7 config locked in and Claude as the judge, we expanded the dataset from 56 → 103 samples by combining three sources:
- **`dataset_alex`** (56 samples) — our original textbook exercises
- **`cross_model_verified.jsonl`** (30 samples) — data augmentation from `dataset_alex`, produced by a colleague using cross-model verification
- **`data2.jsonl`** (17 samples) — exercises retrieved from a second textbook, with a different answer format

#### Round 8 — 103 samples, 10K eval tokens

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|
| 61.90% | 57.14% | 52.38% | 70.00% | 75.00% | **63.29% ± 8.26%** |

Config: lr=1e-5, r=16, α=16, 2 epochs, 103 samples (56 + 30 + 17), Claude judge, eval_max_tokens=10240.

#### Round 8 v2 — 103 samples, 16K eval tokens

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|
| 76.19% | 66.67% | 52.38% | 75.00% | 65.00% | **67.05% ± 8.56%** |

Same model config but increased eval token limit from 10K → 16K.

**Lesson**: The model was getting truncated at 10K tokens on complex problems. Increasing to 16K allowed it to finish its reasoning, adding ~4 pp.

#### Base Model Comparison (103 samples)

To check if fine-tuning actually helped, we ran the base Qwen3-14B (no LoRA) on the same 103 samples:

**Base 14B**: **69.90%** (72/103 correct).

**Problem**: The base model scored *higher* than our fine-tuned model (67.05%). Fine-tuning on 103 samples with r=16 was either not strong enough to make a difference, or the noisy augmented data was hurting.

### Phase 5: Dataset Curation

The base model outperforming our fine-tuned model forced us to look at data quality. We reviewed `dataset_alex` (the original 56 samples) and identified **3 poisoned questions** — questions with incorrect or misleading ground-truth answers:

1. **Problem 5.11** (Lognormal hazard rates) — The ground-truth answer was garbled: `".59,972 hours, 334.97 hours, h(1,000) = 99.7 FITs, h(10,000) = 14,430 FITs and h(40,000) = 22,160 FITs"`. The leading digit of the mean was missing (should be ~59,972), and the standard deviation value (334.97 hours) is numerically wrong for σ=0.9 — off by orders of magnitude.
2. **Problem 11.39** (Minimum sample size for PPM protection) — The answer was simply `"11,513"` with no derivation context. Cross-checking against the textbook's zero-defect sampling formula yields a different result, indicating the ground-truth was corrupted during extraction.
3. **Problem 12.9** (Renewal process, exponential failures) — The failure rate was stated as `"λ = 0.3%/K"`, an ambiguous unit (percent per kilohour?) that led to an incorrect ground-truth of `0.98807`. The answer does not match any reasonable interpretation of the problem parameters.

Removing them gave us a curated **`dataset_alex`** of **53 high-quality samples**.

For the final technique comparison, we trained on just these 53 samples, keeping the augmented and second-textbook data (`cross_model_verified.jsonl` + `data2.jsonl`) aside for the scaling experiment.

### Phase 6: Final Three Techniques (53 Curated Samples)

With the clean dataset, we ran three systematic variants plus the 8B model:

- **Variant A (8B)**: Smaller model, same config as baseline
- **Variant B (Strong LoRA)**: Higher rank (r=64), more epochs (3), greedy decoding
- **Variant C (Answer-only)**: No `<think>...</think>` blocks in training data

See **Part 1** above for the full per-fold results.

### Phase 7: Scaling Experiment (The 100-Sample Collapse)

As a final experiment, we tried running the best config (Strong LoRA, r=64, 3 epochs, greedy) on `dataset_alex_extended` (53 curated + 30 augmented from `cross_model_verified.jsonl` + 17 from `data2.jsonl` = **100 samples**) instead of the curated 53:

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|
| 19.05% | 23.81% | 19.05% | 50.00% | 15.00% | **25.38% ± 12.62%** |

The model collapsed to ~25%. Inspecting the outputs showed format pollution — the model started producing garbled LaTeX, incomplete reasoning chains, and hallucinated problem structures. The most likely culprit is the 17 samples from `data2.jsonl`, which came from a different textbook and had a different answer format (verbose multi-part answers vs. concise numerical values), confusing the model about what output structure to produce.

**Lesson**: Data quality and format consistency >> data quantity. 53 clean, consistently-formatted samples outperform 100 mixed-source ones by a massive margin.

### Phase 8: Final Answer Only & Base Greedy Evaluation

A colleague suggested a different approach: instead of teaching the model *how* to reason (which overwrites its native chain-of-thought), fine-tune it **only on the correct final answer** and let it figure out the reasoning on its own.

This is fundamentally different from all previous variants:

| | Strong LoRA | Answer-only (Variant C) | **Final Answer Only (new)** |
|---|---|---|---|
| Training content | `<think>{reasoning}</think>` + answer | `{reasoning}` + answer (flat text) | **answer ONLY** |
| Reasoning in training | Our Claude-generated reasoning | Our Claude-generated reasoning | **None — model keeps its own** |
| `enable_thinking` at train | True | False | True |
| What model learns | Our reasoning + answer | Our reasoning as visible output | **Only the correct answer** |

The hypothesis: by not overwriting the model's native `<think>` process, we might get better reasoning while still steering it toward correct results. The model already knows *how* to think — it just needs to know *what the right answer is*.

Config: same as Strong LoRA (r=64, α=64, 3 epochs, lr=1e-5, greedy), 53 curated samples, 5-fold CV.

**Results — Final Answer Only**: **60.55% ± 5.85%**

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---|---|---|---|---|---|
| 54.55% | 54.55% | 63.64% | 60.00% | 70.00% | **60.55% ± 5.85%** |

The hypothesis was **not validated**. Training only on the final answer (preserving the model's native `<think>` reasoning) performed significantly below the base model (79.25%) and below Strong LoRA (71.64%). While the model's native reasoning was preserved, it was not steered in any useful direction — the model needs consistent worked examples to learn domain-specific reasoning patterns, not just the final answer. An earlier run with a smaller token budget (MAX_SEQ_LEN=4096, EVAL_MAX_NEW_TOKENS=8192) scored only 52.73% due to truncation; after fixing to 8192/16384 (matching Strong LoRA), accuracy improved to 60.55% but still fell well short.

**Results — Base 14B Greedy**: **79.25%** (42/53 correct)

As a fair comparison to Strong LoRA (which uses greedy decoding), we also evaluated the **base Qwen3-14B with greedy decoding** on the same 53 samples — no fine-tuning at all.

This is the most striking result of the entire project: **the base model with greedy decoding outperforms every fine-tuned variant by a wide margin** (+7.61 pp over Strong LoRA, +11.07 pp over Baseline FT). Greedy decoding (temperature=0) alone is worth ~10 pp compared to the previous base evaluation at temp=0.6 (69.90% on 103 samples).

#### Why Fine-Tuning Didn't Help

Several factors likely explain this:

1. **Dataset size**: 53 samples is extremely small for fine-tuning a 14B-parameter model. The LoRA adapters may be fitting noise rather than learning generalizable domain knowledge.
2. **Pre-training quality**: Qwen3-14B was already trained on massive corpora that include textbook-level mathematics and engineering content. Our 53-sample fine-tune cannot compete with that pre-existing knowledge.
3. **Catastrophic forgetting**: Even with LoRA (which freezes most weights), fine-tuning on domain-specific examples may have slightly degraded the model's general problem-solving abilities.
4. **Decoding strategy matters more than fine-tuning**: The jump from temp=0.6 (69.90%) to greedy (79.25%) on the base model suggests that deterministic decoding is crucial for mathematical reasoning, and this alone closes most of the gap.

### Summary of All Experiments

| # | Phase | Config | Dataset | Judge | Accuracy | Notes |
|---|---|---|---|---|---|---|
| 1 | First attempt | r=32, 2ep, lr=2e-4 | 56 (47/9 split) | GPT-4o-mini | 37.5% | No CV |
| 2 | HP search | r=32→16, lr=2e-4→1e-5 | 56 (5-fold) | GPT-4o-mini | 23%→32% | See Phase 2 table |
| 3 | Judge swap | r=16, lr=1e-5 (Round 7) | 56 (5-fold) | **Claude** | **61.06% ± 13.14%** | Same outputs, new judge |
| 4 | Dataset exp. | r=16, lr=1e-5, 10K tokens | 103 (5-fold) | Claude | 63.29% ± 8.26% | 56+30+17 samples |
| 5 | Dataset exp. | r=16, lr=1e-5, 16K tokens | 103 (5-fold) | Claude | 67.05% ± 8.56% | Token limit matters |
| — | Base 14B | No fine-tuning, temp=0.6 | 103 | Claude | 69.90% | FT not helping yet |
| — | Base 14B (greedy) | No fine-tuning, greedy | 53 | Claude | **79.25%** | ★ Highest overall |
| — | Base 8B | No fine-tuning | 53 | Claude | 62.26% | Smaller model baseline |
| 6 | Final | r=16, lr=1e-5, 2ep | 53 (5-fold) | Claude | 68.18% ± 8.61% | Baseline FT |
| 7 | Final | **r=64, lr=1e-5, 3ep, greedy** | **53 (5-fold)** | **Claude** | **71.64% ± 5.90%** | **★ Best** |
| 8 | Final | r=16, lr=2e-5, answer-only | 53 (5-fold) | Claude | 70.18% ± 16.04% | High variance |
| 9 | Final | 8B, r=16, lr=1e-5, 2ep | 53 (5-fold) | Claude | 64.00% ± 9.73% | Too small |
| 10 | Scaling | r=64, lr=1e-5, 3ep, greedy | 100 (5-fold) | Claude | 25.38% ± 12.62% | Catastrophic collapse |
| 11 | Final ans. only | r=64, lr=1e-5, 3ep, greedy | 53 (5-fold) | Claude | 60.55% ± 5.85% | Hypothesis not validated |

---

## Project Structure

```
repo/
├── training/                         # Fine-tuning scripts
│   ├── finetune_baseline.py          # Baseline: r=16, 2 epochs
│   ├── finetune_strong_lora.py       # Best config: r=64, 3 epochs, greedy
│   ├── finetune_answer_only.py       # Variant C: no <think> in training
│   ├── finetune_8b.py                # 8B model variant
│   ├── finetune_strong_lora_extended.py  # Extended dataset (100 samples)
│   ├── finetune_final_answer_only.py # Answer only — no reasoning in training
│   └── finetune_low_rank.py          # Low rank: r=16, α=32, 1 epoch, greedy
│
├── evaluation/                       # Evaluation scripts
│   ├── eval_base_14b.py              # Base Qwen3-14B evaluation (temp=0.6)
│   ├── eval_base_14b_greedy.py       # Base Qwen3-14B evaluation (greedy)
│   ├── eval_base_8b.py               # Base Qwen3-8B evaluation
│   ├── benchmark_models.py           # Multi-model benchmark (open-ended)
│   └── benchmark_mcq.py              # MCQ benchmark on CRE Handbook
│
├── slurm/                            # SLURM job scripts for HPC
│   ├── submit_baseline.sh
│   ├── submit_strong_lora.sh
│   ├── submit_answer_only.sh
│   ├── submit_8b.sh
│   ├── submit_strong_lora_extended.sh
│   ├── submit_eval_base_14b.sh
│   ├── submit_eval_base_14b_greedy.sh
│   ├── submit_eval_base_8b.sh
│   ├── submit_final_answer_only.sh
│   └── submit_low_rank.sh
│
├── data/                             # Datasets (add your JSONL files here)
│   └── .gitkeep
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Method

### Datasets

The training data comes from three sources:

| Dataset | Samples | Description |
|---|---|---|
| **`dataset_alex`** | 53 | Exercises extracted from a reliability engineering textbook (originally 56, reduced to 53 after removing 3 poisoned questions) |
| **`cross_model_verified.jsonl`** | 30 | Data augmentation from `dataset_alex`, produced by a colleague using cross-model verification |
| **`data2.jsonl`** | 17 | Exercises retrieved from a second textbook — different answer format (verbose multi-part vs. concise numerical) |
| **`dataset_alex_extended`** | 100 | Union of the above three sources |

### Fine-Tuning

All fine-tuning uses **Unsloth** for 4-bit QLoRA on a single A100 GPU:

- **Qwen3 thinking mode**: Training data includes `<think>...</think>` reasoning blocks (except Variant C)
- **LoRA targets**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Optimizer**: AdamW 8-bit with cosine LR schedule and 10% warmup
- **Gradient accumulation**: 4 steps (effective batch size = 8)
- **Max sequence length**: 4096 tokens

### Evaluation

Each experiment uses **5-fold cross-validation**:

1. Model generates a response for each held-out question
2. The answer is extracted via regex (searching for `\boxed{}`, numeric values, etc.)
3. **Claude 3.5 Sonnet** acts as an LLM judge, comparing model vs. gold answers with **5% numerical tolerance**
4. Fold accuracies are averaged to get the final score ± standard deviation

## Environment

### HPC (RUCHE cluster)

```bash
# Submit a job
sbatch slurm/submit_strong_lora.sh

# Check status
squeue -u $USER
```

The SLURM scripts assume:
- An Apptainer/Singularity image at `$WORKDIR/unsloth_latest.sif`
- Scripts deployed to `$WORKDIR/fine_tuning_qwen/`
- A100 GPUs on the `gpua100` partition

### Local Setup

```bash
pip install -r requirements.txt
```

Set your API key for the LLM judge and data pipeline:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

## Configuration

Key hyperparameters are defined at the top of each training script. The main knobs:

| Parameter | Baseline | Strong LoRA | Answer-only | 8B | Final Ans. Only | Low Rank |
|---|---|---|---|---|---|---|
| `LORA_R` | 16 | 64 | 16 | 16 | 64 | 16 |
| `LORA_ALPHA` | 16 | 64 | 16 | 16 | 64 | 32 |
| `NUM_EPOCHS` | 2 | 3 | 2 | 2 | 3 | 1 |
| `LEARNING_RATE` | 1e-5 | 1e-5 | 2e-5 | 1e-5 | 1e-5 | 1e-5 |
| `EVAL_TEMPERATURE` | 0.6 | 0.0 | 0.6 | 0.6 | 0.0 | 0.0 |
| Thinking in train | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| Reasoning in train data | ✓ | ✓ | ✓ (flat) | ✓ | ✗ | ✓ |

## Models

- **Fine-tuned**: `unsloth/Qwen3-14B-unsloth-bnb-4bit`, `unsloth/Qwen3-8B-unsloth-bnb-4bit`
- **Data generation**: `openai/gpt-4o-mini` (extraction/augmentation), `deepseek/deepseek-r1` (solving)
- **LLM judge**: `anthropic/claude-sonnet-4-20250514` via OpenRouter

---

## Part 3 — Error Analysis: why fine-tuning underperforms the base model

To understand *where* and *why* fine-tuning hurts, we performed a per-question comparison across three configurations on the same 53 questions:

- **Base (no FT, greedy)**: 79.25% (42/53)
- **Strong LoRA** (r=64, 3 epochs, greedy, 5-fold CV): 71.64% ± 5.90% (~38/53)
- **Final Answer Only** (r=64, 3 epochs, greedy, answer-only training, 5-fold CV): 60.55% ± 5.85% (~32/53)

### Base vs. Strong LoRA — Question-level comparison

| Outcome | Count | % |
|---|---|---|
| Both correct | 34 | 64.2% |
| Both wrong | 8 | 15.1% |
| Base ✓, LoRA ✗ (regressions) | 7 | 13.2% |
| Base ✗, LoRA ✓ (improvements) | 3 | 5.7% |
| **Agreement rate** | **42/53** | **80.8%** |

Fine-tuning caused **7 regressions** but only **3 improvements** — a net loss of 4 questions. The regressions include Weibull/lognormal distribution problems and sample size computations where the base model's pre-trained knowledge was already sufficient. The 3 questions where LoRA improved over the base tend to involve more domain-specific setups (code inspection bugs, two-box systems, airplane engine redundancy).

**Interpretation**: Strong LoRA mostly preserves the base model's knowledge (34 shared correct), but the 53-sample fine-tune introduces enough noise to flip ~13% of questions from correct → wrong while only recovering ~6% in the other direction.

### Base vs. Final Answer Only — question-level comparison

| Outcome | Count | % |
|---|---|---|
| Both correct | 30 | 56.6% |
| Both wrong | 9 | 17.0% |
| Base ✓, FAO ✗ (regressions) | 11 | 20.8% |
| Base ✗, FAO ✓ (improvements) | 2 | 3.8% |
| **Agreement rate** | **39/53** | **75.0%** |

Final Answer Only caused **11 regressions** but only **2 improvements** — a net loss of 9 questions. This is significantly worse than Strong LoRA's net loss of 4, confirming that training only on the final answer (without reasoning examples) damages the model's problem-solving ability more severely.

### Strong LoRA vs. Final Answer Only

| Outcome | Count | % |
|---|---|---|
| Both correct | 27 | 50.9% |
| Both wrong | 10 | 18.9% |
| LoRA ✓, FAO ✗ | 10 | 18.9% |
| LoRA ✗, FAO ✓ | 5 | 9.4% |
| **Agreement rate** | **37/53** | **71.2%** |

Final Answer Only loses 10 questions that Strong LoRA gets right, while gaining only 5. Providing worked reasoning examples during training (Strong LoRA) is clearly better than providing only the final answer.

### Exclusive correctness — Who gets what right?

| Category | Count | Questions |
|---|---|---|
| All three correct | **25** | Core of shared competence |
| All three wrong | **8** | Hardest questions (see below) |
| Only Base right (both FTs wrong) | 2 | LTPD sample size, lot sampling acceptance |
| Only Strong LoRA right | 1 | Code inspection defect estimation |
| Only Final Answer Only right | 0 | — |
| Base + LoRA right, FAO wrong | 9 | FAO-specific regressions |
| Base + FAO right, LoRA wrong | 5 | LoRA-specific regressions |
| LoRA + FAO right, Base wrong | 2 | Rare fine-tuning improvements |

**Key finding**: There are **zero questions that only FAO gets right** — every question FAO solves is also solved by at least one other model. Meanwhile, the base model has 2 exclusive wins and shares correctness with at least one fine-tune on 40 of 42 questions it gets right.

### The 8 hardest questions (all 3 models failed)

These questions defeated every configuration:

1. Sample size for MTTF confidence interval (chi-square table lookup)
2. Straight-line regression / least-squares fit
3. Defect model likelihood for interval-censored data
4. Identifying CDF vs. PDF vs. hazard function from formulas
5. 300-unit test plan at 80% confidence level
6. AQL/LTPD field failure rate with 50 FIT target
7. Lot acceptance with 3 failures observed on 300 devices
8. Two vendor groups of 50 components — comparison test

These are predominantly **sample size/testing plan** and **statistical inference** problems requiring multi-step table lookups (chi-square, binomial) or likelihood derivations — tasks that rely on precise numerical computation rather than conceptual understanding.

### Token length analysis

| Model | Correct (mean tokens) | Wrong (mean tokens) | Ratio |
|---|---|---|---|
| Base | ~4,986 | ~9,336 | 1.87× |
| Strong LoRA | ~5,195 | ~7,255 | 1.40× |
| Final Answer Only | ~5,379 | ~8,517 | 1.58× |

Wrong answers are consistently **40–90% longer** than correct ones across all models. The models "struggle" on harder questions by generating longer, more meandering reasoning chains that ultimately reach incorrect conclusions. This pattern is strongest for the base model (1.87× ratio), suggesting that when the base model is wrong, it is thoroughly confused and produces very long outputs.

### Performance by topic category

| Topic | # Qs | Base | Strong LoRA | Final Ans. Only |
|---|---|---|---|---|
| Lognormal distributions | 5 | **100%** | 80% | 80% |
| Weibull distributions | 16 | **88%** | 69% | 63% |
| Electronic components | 13 | **85%** | 69% | 62% |
| Other distributions | 7 | 86% | 71% | 57% |
| Probability / Bayes | 19 | 79% | 74% | 68% |
| Exponential distributions | 19 | 74% | 68% | 47% |
| Poisson process | 5 | 60% | **80%** | 40% |
| **Sample size / testing** | **12** | **50%** | **33%** | **17%** |

**Observations**:
- The base model dominates in every category except **Poisson process**, where Strong LoRA scores 80% vs. 60% — the only topic where fine-tuning clearly helped.
- **Sample size/testing** is the hardest category across the board (50% → 33% → 17%), confirming these questions require precise table lookups that LLMs struggle with.
- Final Answer Only degrades most severely on **exponential distributions** (74% → 47%, a 27 pp drop) — a core reliability engineering topic, suggesting the model lost domain-specific reasoning patterns.
- The gap between Base and fine-tuned models is smallest on **Probability/Bayes** (79% vs. 74%/68%) — more general-purpose math where pre-training knowledge is robust.

### Summary of findings

The per-question analysis reveals that fine-tuning's failure is not uniform — it's a **net negative trade-off**:

1. **Strong LoRA** preserves 81% of the base model's decisions. It gains 3 domain-specific questions but loses 7 others, mostly standard distribution problems where the base model's pre-training was already sufficient. The 53-sample training set introduces mild noise that flips some correct answers without compensating enough elsewhere.

2. **Final Answer Only** preserves only 75% agreement with the base model. It gains just 2 questions while losing 11 — the worst trade-off of any variant. Without worked reasoning examples, the model's native `<think>` process is unguided and frequently goes astray on domain-specific computations.

3. **Topic-wise**, fine-tuning only helps on **Poisson process** questions (Strong LoRA: +20 pp). For every other topic, the base model is equal or better. This suggests the 53-sample dataset happens to contain enough Poisson examples to teach the model something new, but not enough of other topics to overcome the noise introduced by small-data fine-tuning.

4. **The 8 universally-hard questions** (sample size, likelihood, table lookups) represent a ceiling that no configuration can break — these likely require tool use (calculator, statistical tables) rather than pure language model reasoning.

## License

This project was developed as part of an MSc research lab at CentraleSupélec.

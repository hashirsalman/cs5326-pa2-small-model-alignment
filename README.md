# CS 5326 PA2 — Small-Model Alignment: SFT + DPO

**Student:** Muhammad Hashir Salman  
**Course:** CS 5326 — Advanced Generative AI  
**Repository:** `hashirsalman/cs5326-pa2-small-model-alignment`

This repository contains the completed implementation, experiment configurations, fixed datasets/evaluation sets, generation outputs, pairwise judge results, training metrics, and report assets for Programming Assignment 2: **Small-Model Alignment: SFT + DPO**.

The project contains four alignment tracks:

| Track | Model / data | Training method | Evaluation |
|---|---|---|---|
| SFT | Qwen/Qwen2.5-0.5B, UltraChat (30,000 train / 1,000 validation) | Custom PyTorch SFT loop with response-only loss | Fixed AlpacaEval subset |
| HH-RLHF DPO | Best SFT checkpoint, 15,000 train / 1,000 validation preference pairs | Custom from-scratch DPO | 300-prompt helpful/harmless evaluation set |
| UltraFeedback DPO | Best SFT checkpoint, 15,000 train / 1,000 validation preference pairs | Custom from-scratch DPO | AlpacaEval 300, seed 42 |
| Persona LoRA-DPO | HuggingFaceTB/SmolLM2-360M-Instruct, 200 train / 20 validation | TRL `DPOTrainer` + PEFT/LoRA | 20 fixed persona prompts |

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── submission_adapter.py
├── alignment_core.py
├── data_utils.py
├── sft_training.py
├── dpo_training.py
│
├── notebooks/
│   ├── 01_main_setup_sft_hh.ipynb
│   ├── 02_ultrafeedback_dpo_generation.ipynb
│   ├── 03_persona_lora_dpo.ipynb
│   ├── 04_pairwise_gemini_judging.ipynb
│   └── 05_final_report_assets.ipynb
│
├── tests/
│   ├── test_public_adapter_smoke.py
│   └── toy_lm.py
│
├── data/
│   ├── sft_fixed/
│   ├── hh_rlhf_fixed/
│   ├── ultrafeedback_fixed/
│   └── persona_lora/
│
├── eval_sets/
│   ├── alpaca_eval_300_seed42.json
│   ├── hh_rlhf_eval_300_seed42.json
│   └── persona_style_eval_20_seed42.json
│
├── outputs/
│   ├── sft_qwen2.5_0.5b_ultrachat_seed42/
│   ├── hh_rlhf_dpo_seed42/
│   ├── ultrafeedback_dpo_seed42/
│   ├── persona_lora_dpo_seed42/
│   ├── eval/
│   └── report_assets/
│
└── docs/
    └── assignment_manual.pdf
```

Large model checkpoint weight files are intentionally excluded from GitHub. The repository retains run configurations, metrics, hashes, fixed data/evaluation files, generations, pairwise comparisons, judge outputs, and report assets needed to audit the submitted results.

## Core Implementation

The custom SFT and scratch-DPO logic is kept in the repository root so that the grading adapter and implementation are easy to inspect:

- `alignment_core.py` — batching, response-only masking/log-probability logic, SFT loss, and tensor-level DPO objective.
- `data_utils.py` — prompt formatting and dataset utilities.
- `sft_training.py` — SFT training/evaluation helpers.
- `dpo_training.py` — shared from-scratch DPO training/evaluation helpers for HH-RLHF and UltraFeedback.
- `submission_adapter.py` — required root-level grading interface that delegates to the implementation above.

The same core scratch-DPO implementation is used for both HH-RLHF and UltraFeedback.

## Notebook Workflow

The `notebooks/` directory preserves the Kaggle workflow used to produce the submitted artifacts:

1. `01_main_setup_sft_hh.ipynb`  
   Repository/environment setup, public adapter tests, fixed data construction, base generation, SFT, HH-RLHF preparation/training/generation, and shared implementation development.

2. `02_ultrafeedback_dpo_generation.ipynb`  
   UltraFeedback scratch-DPO training, validation, AlpacaEval generation, and pair-file construction.

3. `03_persona_lora_dpo.ipynb`  
   Persona preference split, TRL `DPOTrainer`, PEFT/LoRA training, persona generation, and blind pair-file construction.

4. `04_pairwise_gemini_judging.ipynb`  
   Resumable pairwise judging for HH-RLHF, UltraFeedback, and persona comparisons using `google/gemini-3.6-flash`, with batching, delays, retries, failure logging, and hidden model identities.

5. `05_final_report_assets.ipynb`  
   Final result aggregation, adapter-test audit, report tables/plots, and qualitative-analysis extraction.

The notebooks use saved Kaggle outputs as inputs between stages so completed training/generation artifacts can be reused without rerunning earlier stages.

## Main Training Configuration

### SFT

- Model: `Qwen/Qwen2.5-0.5B`
- Data: UltraChat
- Train / validation: 30,000 / 1,000
- Context length: 512
- Per-device batch size: 2
- Gradient accumulation: 16
- Effective batch size: 32
- Learning rate: `2e-5`
- Scheduler: cosine
- Weight decay: 0.1
- Gradient clipping: 1.0
- Epochs: 1
- Seed: 42

### Scratch DPO — HH-RLHF and UltraFeedback

- Initial policy: best SFT checkpoint
- Reference: frozen copy of the same best SFT checkpoint
- Train / validation: 15,000 / 1,000 pairs per track
- Context length: 512
- Maximum prompt length: 256
- Per-device batch size: 1
- Gradient accumulation: 32
- Effective batch size: 32
- Learning rate: `1e-6`, fixed
- DPO beta: 0.1
- Gradient clipping: 1.0
- Epochs: 1
- Seed: 42

### Persona LoRA-DPO

- Model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Train / validation: 200 / 20
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: q/k/v/o projections and gate/up/down projections
- Learning rate: `1e-5`
- Effective batch size: 8
- DPO beta: 0.1
- Maximum length: 512
- Epochs: 3
- Seed: 42

The persona notebook records the exact Kaggle runtime used for the final run and installs `trl==1.15.0` with `--no-deps` to avoid replacing Kaggle's PyTorch stack.

## Evaluation

All required model comparisons use fixed prompts and matched deterministic generation settings within each comparison. Response order is randomized in the saved pairwise files, while the response-to-model mapping is stored as hidden metadata and is not exposed to the judge prompt.

The final pairwise judge was `google/gemini-3.6-flash` with temperature 0. The evaluator used batches of 5, at least 5 seconds between attempts, up to 3 retries, checkpointed judgments, failure logs, and resumable outputs.

### Final Pairwise Results

| Comparison | Aligned wins | Baseline wins | Ties | Aligned win rate (all) | Aligned win rate (excluding ties) |
|---|---:|---:|---:|---:|---:|
| HH-RLHF DPO vs SFT | 160 | 101 | 39 | 53.3% | 61.3% |
| UltraFeedback DPO vs SFT | 105 | 144 | 51 | 35.0% | 42.2% |
| Persona LoRA-DPO vs base | 6 | 2 | 12 | 30.0% | 75.0% |

All 620 final pairwise examples were successfully judged; no unresolved failures remained in the final summaries.

## Public Adapter Tests

From the repository root:

```bash
python -m pip install -q pytest
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

Final audit:

```text
7 passed
```

The public tests are a minimum check; the implementation also preserves response-only masking/log-probability behavior needed by the assignment's private evaluation interface.

## Reproducibility Artifacts

Key saved artifacts include:

- `outputs/sft_qwen2.5_0.5b_ultrachat_seed42/metrics.jsonl`
- `outputs/hh_rlhf_dpo_seed42/metrics.jsonl`
- `outputs/ultrafeedback_dpo_seed42/metrics.jsonl`
- `outputs/persona_lora_dpo_seed42/metrics.jsonl`
- `outputs/eval/judge_results/MASTER_JUDGE_SUMMARY.json`
- `outputs/eval/judge_results/hh_rlhf/`
- `outputs/eval/judge_results/ultrafeedback/`
- `outputs/eval/judge_results/persona_lora/`
- `outputs/report_assets/`

Fixed datasets and evaluation subsets are committed under `data/` and `eval_sets/` so the exact examples used by the submitted runs can be inspected.

## Checkpoints

Best/final model weight files are not committed because of their size. The original Kaggle run directories preserved the trained checkpoints, while this repository preserves the corresponding configurations, metrics, summaries, artifact hashes, and downstream generations/evaluations.

The SFT checkpoint used as the policy initialization and frozen reference for both scratch-DPO runs has SHA-256:

```text
cf40f50587c5287c6dea232603e3976fe6bf4480fe1b6145f0273da90e8f70ac
```

## Final Report

The final report is submitted separately as a PDF, as required by the assignment. The plots and summary tables used to construct it are preserved under `outputs/report_assets/`.

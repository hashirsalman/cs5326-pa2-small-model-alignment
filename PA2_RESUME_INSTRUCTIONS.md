# PA2 Resume Instructions

## Completed

- Core implementation + public tests
- Fixed AlpacaEval 300-prompt subset
- Base Qwen2.5-0.5B AlpacaEval generation
- SFT on 30,000 UltraChat examples
- SFT AlpacaEval generation
- Fixed HH-RLHF 15,000 train / 1,000 validation pairs
- HH-RLHF from-scratch DPO
- SFT and HH-DPO generation on all 300 HH evaluation prompts

## Next

Run the UltraFeedback from-scratch DPO experiment.

IMPORTANT: initialize both the UltraFeedback trainable policy
and frozen reference from the ORIGINAL SFT best checkpoint:

outputs/sft_qwen2.5_0.5b_ultrachat_seed42/best_checkpoint

Do NOT initialize UltraFeedback from the HH-DPO checkpoint.

## Prompt format

### Instruction:
{instruction}

### Response:

## Evaluation decoding

- greedy
- do_sample=False
- max_new_tokens=256
- repetition_penalty=1.0
- no_repeat_ngram_size=4

## HH-DPO result

- 15,000 train pairs
- 1,000 validation pairs
- 469 updates
- LR = 1e-6 fixed
- beta = 0.1
- accumulation = 32
- best update = 469
- reward accuracy = 0.633
- reward margin = 0.23464530624076724
- AMP initial scale = 512
- skipped optimizer steps = 0

PA2_RESUME_STATE.json contains SHA256 hashes for verification.

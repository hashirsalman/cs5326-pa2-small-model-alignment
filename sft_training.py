
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from alignment_core import build_sft_batch_impl


# ============================================================
# SMALL FILE UTILITIES
# ============================================================

def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(record, ensure_ascii=False)
            + "\n"
        )


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# BUILD ONE SFT MICRO-BATCH
# ============================================================

def prepare_sft_batch_cpu(
    tokenizer,
    prompts,
    responses,
    max_length,
):
    """
    Uses the already-tested assignment implementation to produce:
      input_ids
      attention_mask
      labels

    Then removes only trailing all-padding columns.

    Returned tensors remain on CPU.
    """

    batch = build_sft_batch_impl(
        tokenizer=tokenizer,
        prompts=prompts,
        responses=responses,
        max_length=max_length,
    )

    if batch["input_ids"].shape[0] == 0:
        raise ValueError("Empty SFT batch.")

    # Longest real sequence in this micro-batch.
    used_length = int(
        batch["attention_mask"]
        .sum(dim=1)
        .max()
        .item()
    )

    # Causal LM needs at least two positions to form a shift.
    used_length = max(used_length, 2)

    batch = {
        key: value[:, :used_length].contiguous()
        for key, value in batch.items()
    }

    # After causal shifting, these are the actual response
    # targets contributing to the loss.
    response_token_count = int(
        batch["labels"][:, 1:]
        .ne(-100)
        .sum()
        .item()
    )

    if response_token_count <= 0:
        raise RuntimeError(
            "SFT micro-batch has no scorable response tokens."
        )

    return batch, response_token_count


def move_batch_to_device(batch, device):
    return {
        key: value.to(
            device,
            non_blocking=True,
        )
        for key, value in batch.items()
    }


# ============================================================
# SUMMED RESPONSE NLL
# ============================================================

def response_nll_sum(
    model,
    batch,
):
    """
    Return SUM of negative log probabilities over response
    targets only.

    labels already contain -100 for:
      - prompt tokens
      - padding

    We explicitly apply the causal one-token shift.
    """

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )

    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = batch["labels"][:, 1:]

    nll_sum = F.cross_entropy(
        shift_logits.reshape(
            -1,
            shift_logits.shape[-1],
        ),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )

    return nll_sum


# ============================================================
# VALIDATION
# ============================================================

@torch.inference_mode()
def evaluate_sft(
    model,
    tokenizer,
    dataloader,
    max_length,
    device,
    max_batches=None,
):
    """
    Exact response-token-mean validation NLL.
    """

    was_training = model.training

    model.eval()

    total_nll = 0.0
    total_response_tokens = 0
    total_examples = 0
    batches_evaluated = 0

    try:
        for batch_index, batch in enumerate(dataloader):

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break

            cpu_batch, token_count = (
                prepare_sft_batch_cpu(
                    tokenizer=tokenizer,
                    prompts=batch["prompts"],
                    responses=batch["responses"],
                    max_length=max_length,
                )
            )

            gpu_batch = move_batch_to_device(
                cpu_batch,
                device,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                nll_sum = response_nll_sum(
                    model,
                    gpu_batch,
                )

            total_nll += float(
                nll_sum.detach().float().item()
            )

            total_response_tokens += token_count
            total_examples += len(batch["prompts"])
            batches_evaluated += 1

            del gpu_batch
            del cpu_batch
            del nll_sum

    finally:
        if was_training:
            model.train()

    if total_response_tokens <= 0:
        raise RuntimeError(
            "Validation contained no response tokens."
        )

    validation_loss = (
        total_nll
        / total_response_tokens
    )

    # Should be easily finite for this run.
    perplexity = (
        math.exp(validation_loss)
        if validation_loss < 50
        else float("inf")
    )

    return {
        "validation_loss": validation_loss,
        "perplexity": perplexity,
        "examples": total_examples,
        "response_tokens": total_response_tokens,
        "batches": batches_evaluated,
    }


# ============================================================
# CHECKPOINT SAVING
# ============================================================

def save_sft_checkpoint(
    model,
    tokenizer,
    checkpoint_dir,
    metadata,
):
    checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        checkpoint_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        checkpoint_dir,
    )

    write_json(
        checkpoint_dir
        / "checkpoint_metadata.json",
        metadata,
    )


# ============================================================
# FULL ONE-EPOCH SFT TRAINING LOOP
# ============================================================

def train_sft(
    *,
    model,
    tokenizer,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    scaler,
    config,
    run_dir,
    device,
):
    run_dir = Path(run_dir)

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        run_dir / "metrics.jsonl"
    )

    summary_path = (
        run_dir / "training_summary.json"
    )

    best_checkpoint_dir = (
        run_dir / "best_checkpoint"
    )

    final_checkpoint_dir = (
        run_dir / "final_checkpoint"
    )

    # Avoid accidentally mixing metrics from two complete runs.
    if metrics_path.exists():
        raise FileExistsError(
            f"{metrics_path} already exists. "
            "Use a new run directory or explicitly remove it "
            "before beginning a fresh training run."
        )

    accumulation_steps = int(
        config["gradient_accumulation_steps"]
    )

    max_length = int(
        config["context_length"]
    )

    clip_norm = float(
        config["gradient_clip_norm"]
    )

    eval_every = int(
        config["eval_every_updates"]
    )

    expected_updates = int(
        config["total_update_steps"]
    )

    model.config.use_cache = False

    optimizer.zero_grad(
        set_to_none=True,
    )

    # --------------------------------------------------------
    # Full validation at update 0.
    #
    # This is useful for the learning curve, but does NOT
    # participate in SFT checkpoint selection.
    # --------------------------------------------------------

    print("Running full pre-training validation...")

    initial_val = evaluate_sft(
        model=model,
        tokenizer=tokenizer,
        dataloader=val_loader,
        max_length=max_length,
        device=device,
    )

    append_jsonl(
        metrics_path,
        {
            "type": "validation",
            "update": 0,
            **initial_val,
        },
    )

    print(
        "Initial validation loss:",
        initial_val["validation_loss"],
    )

    print(
        "Initial validation perplexity:",
        initial_val["perplexity"],
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    best_validation_loss = float("inf")
    best_update = None

    update = 0
    skipped_optimizer_steps = 0

    total_train_nll = 0.0
    total_train_tokens = 0

    training_start_time = time.time()

    progress = tqdm(
        total=expected_updates,
        desc="SFT optimizer updates",
    )

    # We buffer up to 16 micro-batches so we know the exact
    # number of response tokens in the whole accumulation
    # window BEFORE computing its normalized gradient.
    window = []

    last_eval_update = 0

    for microbatch_index, raw_batch in enumerate(
        train_loader,
        start=1,
    ):

        cpu_batch, token_count = (
            prepare_sft_batch_cpu(
                tokenizer=tokenizer,
                prompts=raw_batch["prompts"],
                responses=raw_batch["responses"],
                max_length=max_length,
            )
        )

        window.append(
            (
                cpu_batch,
                token_count,
            )
        )

        is_last_microbatch = (
            microbatch_index
            == len(train_loader)
        )

        window_is_full = (
            len(window)
            == accumulation_steps
        )

        if not (
            window_is_full
            or is_last_microbatch
        ):
            continue

        # ====================================================
        # One optimizer update
        # ====================================================

        window_response_tokens = sum(
            token_count
            for _, token_count in window
        )

        window_nll_value = 0.0

        optimizer.zero_grad(
            set_to_none=True,
        )

        for cpu_microbatch, _ in window:

            gpu_microbatch = move_batch_to_device(
                cpu_microbatch,
                device,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                nll_sum = response_nll_sum(
                    model,
                    gpu_microbatch,
                )

                # IMPORTANT:
                # Normalize every microbatch by the number of
                # response tokens in the ENTIRE accumulation
                # window.
                #
                # Summing gradients across all microbatches
                # therefore gives exactly:
                #
                # total response NLL / total response tokens
                #
                loss_piece = (
                    nll_sum
                    / window_response_tokens
                )

            scaler.scale(
                loss_piece
            ).backward()

            window_nll_value += float(
                nll_sum.detach().float().item()
            )

            del gpu_microbatch
            del nll_sum
            del loss_piece

        # Convert scaled gradients back to ordinary gradients
        # before clipping.
        scaler.unscale_(
            optimizer
        )

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=clip_norm,
        )

        # LR actually used for this attempted update.
        lr_used = float(
            optimizer.param_groups[0]["lr"]
        )

        scale_before = scaler.get_scale()

        scaler.step(
            optimizer
        )

        scaler.update()

        scale_after = scaler.get_scale()

        # If GradScaler encountered inf/nan gradients, it skips
        # optimizer.step and reduces the scale.
        optimizer_step_succeeded = (
            scale_after >= scale_before
        )

        window_train_loss = (
            window_nll_value
            / window_response_tokens
        )

        if optimizer_step_succeeded:

            scheduler.step()

            update += 1

            total_train_nll += window_nll_value
            total_train_tokens += (
                window_response_tokens
            )

            record = {
                "type": "train",
                "update": update,
                "microbatch_end": (
                    microbatch_index
                ),
                "microbatches_in_window": (
                    len(window)
                ),
                "response_tokens": (
                    window_response_tokens
                ),
                "train_loss": (
                    window_train_loss
                ),
                "lr_used": lr_used,
                "lr_next": float(
                    optimizer
                    .param_groups[0]["lr"]
                ),
                "grad_norm_before_clip": float(
                    grad_norm.detach()
                    .float()
                    .item()
                ),
                "grad_scale": float(
                    scale_after
                ),
            }

            append_jsonl(
                metrics_path,
                record,
            )

            progress.update(1)

            progress.set_postfix(
                loss=f"{window_train_loss:.4f}",
                lr=f"{lr_used:.2e}",
            )

        else:

            skipped_optimizer_steps += 1

            append_jsonl(
                metrics_path,
                {
                    "type": "skipped_update",
                    "microbatch_end": (
                        microbatch_index
                    ),
                    "reason": (
                        "GradScaler detected "
                        "non-finite gradients"
                    ),
                    "scale_before": float(
                        scale_before
                    ),
                    "scale_after": float(
                        scale_after
                    ),
                },
            )

        # Free buffered CPU tensors.
        window.clear()

        # ----------------------------------------------------
        # Periodic validation
        # ----------------------------------------------------

        should_validate = (
            optimizer_step_succeeded
            and update > 0
            and update % eval_every == 0
        )

        if should_validate:

            validation = evaluate_sft(
                model=model,
                tokenizer=tokenizer,
                dataloader=val_loader,
                max_length=max_length,
                device=device,
            )

            append_jsonl(
                metrics_path,
                {
                    "type": "validation",
                    "update": update,
                    **validation,
                },
            )

            last_eval_update = update

            print(
                f"\nUpdate {update}: "
                f"validation loss = "
                f"{validation['validation_loss']:.6f}"
            )

            if (
                validation["validation_loss"]
                < best_validation_loss
            ):

                best_validation_loss = (
                    validation[
                        "validation_loss"
                    ]
                )

                best_update = update

                print(
                    "Saving new best checkpoint "
                    f"at update {update}..."
                )

                save_sft_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    checkpoint_dir=(
                        best_checkpoint_dir
                    ),
                    metadata={
                        "checkpoint_type": "best",
                        "update": update,
                        "validation_loss": (
                            best_validation_loss
                        ),
                        "selection_criterion": (
                            "lowest validation loss"
                        ),
                    },
                )

            model.train()

    progress.close()

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    if last_eval_update != update:

        print(
            "\nRunning final validation "
            f"at update {update}..."
        )

        final_validation = evaluate_sft(
            model=model,
            tokenizer=tokenizer,
            dataloader=val_loader,
            max_length=max_length,
            device=device,
        )

        append_jsonl(
            metrics_path,
            {
                "type": "validation",
                "update": update,
                **final_validation,
            },
        )

    else:
        # If the final update happened to coincide exactly
        # with an evaluation boundary, recover that validation
        # result by evaluating once more only for simplicity
        # and an explicit final summary.
        final_validation = evaluate_sft(
            model=model,
            tokenizer=tokenizer,
            dataloader=val_loader,
            max_length=max_length,
            device=device,
        )

        append_jsonl(
            metrics_path,
            {
                "type": "validation_final_repeat",
                "update": update,
                **final_validation,
            },
        )

    # The final checkpoint can itself be the best.
    if (
        final_validation["validation_loss"]
        < best_validation_loss
    ):

        best_validation_loss = (
            final_validation[
                "validation_loss"
            ]
        )

        best_update = update

        print(
            "Final checkpoint is also the "
            "new best checkpoint."
        )

        save_sft_checkpoint(
            model=model,
            tokenizer=tokenizer,
            checkpoint_dir=(
                best_checkpoint_dir
            ),
            metadata={
                "checkpoint_type": "best",
                "update": update,
                "validation_loss": (
                    best_validation_loss
                ),
                "selection_criterion": (
                    "lowest validation loss"
                ),
            },
        )

    # --------------------------------------------------------
    # ALWAYS SAVE FINAL CHECKPOINT
    # --------------------------------------------------------

    print("Saving final checkpoint...")

    save_sft_checkpoint(
        model=model,
        tokenizer=tokenizer,
        checkpoint_dir=(
            final_checkpoint_dir
        ),
        metadata={
            "checkpoint_type": "final",
            "update": update,
            "validation_loss": (
                final_validation[
                    "validation_loss"
                ]
            ),
        },
    )

    elapsed_seconds = (
        time.time()
        - training_start_time
    )

    overall_train_loss = (
        total_train_nll
        / total_train_tokens
    )

    summary = {
        "completed": True,
        "optimizer_updates": update,
        "expected_optimizer_updates": (
            expected_updates
        ),
        "skipped_optimizer_steps": (
            skipped_optimizer_steps
        ),
        "overall_training_loss": (
            overall_train_loss
        ),
        "initial_validation": initial_val,
        "final_validation": final_validation,
        "best_validation_loss": (
            best_validation_loss
        ),
        "best_update": best_update,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_hours": (
            elapsed_seconds / 3600.0
        ),
        "best_checkpoint": str(
            best_checkpoint_dir
        ),
        "final_checkpoint": str(
            final_checkpoint_dir
        ),
        "metrics_file": str(
            metrics_path
        ),
    }

    write_json(
        summary_path,
        summary,
    )

    return summary

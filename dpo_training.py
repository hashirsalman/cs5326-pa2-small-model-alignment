
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from alignment_core import (
    build_dpo_batch_impl,
    compute_dpo_loss_from_logps_impl,
)


# ============================================================
# FILE HELPERS
# ============================================================

def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# PREPARE ONE PREFERENCE PAIR
# ============================================================

def prepare_dpo_pair_cpu(
    tokenizer,
    prompt,
    chosen,
    rejected,
    max_length,
):
    """
    Prepare one preference pair using the already-tested
    assignment batch implementation.

    The returned joint tensors have two rows:

        row 0 = chosen
        row 1 = rejected

    Trailing columns that are padding for BOTH sequences are
    removed for efficiency.
    """

    dpo_batch = build_dpo_batch_impl(
        tokenizer=tokenizer,
        prompts=[prompt],
        chosen_responses=[chosen],
        rejected_responses=[rejected],
        max_length=max_length,
    )

    chosen_batch = dpo_batch["chosen"]
    rejected_batch = dpo_batch["rejected"]


    chosen_used_length = int(
        chosen_batch[
            "attention_mask"
        ]
        .sum(dim=1)
        .max()
        .item()
    )

    rejected_used_length = int(
        rejected_batch[
            "attention_mask"
        ]
        .sum(dim=1)
        .max()
        .item()
    )


    joint_length = max(
        chosen_used_length,
        rejected_used_length,
        2,
    )


    # Both source batches were originally padded to max_length,
    # so slicing both to joint_length leaves valid padding on
    # whichever response is shorter.
    joint_batch = {}

    for key in (
        "input_ids",
        "attention_mask",
        "labels",
    ):
        chosen_tensor = (
            chosen_batch[key]
            [:, :joint_length]
            .contiguous()
        )

        rejected_tensor = (
            rejected_batch[key]
            [:, :joint_length]
            .contiguous()
        )

        joint_batch[key] = torch.cat(
            [
                chosen_tensor,
                rejected_tensor,
            ],
            dim=0,
        )


    chosen_response_tokens = int(
        joint_batch["labels"][
            0,
            1:,
        ]
        .ne(-100)
        .sum()
        .item()
    )

    rejected_response_tokens = int(
        joint_batch["labels"][
            1,
            1:,
        ]
        .ne(-100)
        .sum()
        .item()
    )


    if chosen_response_tokens <= 0:
        raise RuntimeError(
            "Chosen response has no scorable tokens."
        )

    if rejected_response_tokens <= 0:
        raise RuntimeError(
            "Rejected response has no scorable tokens."
        )


    return {
        "batch": joint_batch,
        "chosen_response_tokens": (
            chosen_response_tokens
        ),
        "rejected_response_tokens": (
            rejected_response_tokens
        ),
    }


def move_batch_to_device(
    batch,
    device,
):
    return {
        key: value.to(
            device,
            non_blocking=True,
        )
        for key, value in batch.items()
    }


# ============================================================
# RESPONSE LOG PROBABILITIES FOR A JOINT BATCH
# ============================================================

def response_logps_from_joint_batch(
    model,
    batch,
):
    """
    Return one response-only sequence log probability per row.

    For a normal DPO pair:
        result[0] = chosen log p
        result[1] = rejected log p
    """

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch[
            "attention_mask"
        ],
        use_cache=False,
    )

    logits = outputs.logits


    # Causal one-token shift.
    shift_logits = logits[
        :,
        :-1,
        :,
    ]

    shift_labels = batch[
        "labels"
    ][:, 1:]


    vocab_size = (
        shift_logits.shape[-1]
    )


    # reduction='none' gives token NLL.
    # Tokens with label -100 contribute exactly zero.
    token_nll = F.cross_entropy(
        shift_logits.reshape(
            -1,
            vocab_size,
        ),
        shift_labels.reshape(-1),

        ignore_index=-100,
        reduction="none",
    )


    token_nll = token_nll.reshape(
        shift_labels.shape
    )


    sequence_logps = (
        -token_nll.sum(dim=1)
    )


    return sequence_logps


# ============================================================
# ONE POLICY/REFERENCE PAIR COMPUTATION
# ============================================================

def compute_pair_quantities(
    *,
    policy_model,
    reference_model,
    tokenizer,
    prompt,
    chosen,
    rejected,
    max_length,
    beta,
    policy_device,
    reference_device,
    require_policy_grad,
):
    """
    Compute:
      policy chosen/rejected logps
      reference chosen/rejected logps
      DPO loss
      implicit rewards
      reward margin
      reward correctness
    """

    prepared = prepare_dpo_pair_cpu(
        tokenizer=tokenizer,
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        max_length=max_length,
    )

    cpu_batch = prepared["batch"]


    # --------------------------------------------------------
    # POLICY
    # --------------------------------------------------------

    policy_batch = (
        move_batch_to_device(
            cpu_batch,
            policy_device,
        )
    )


    grad_context = (
        torch.enable_grad()
        if require_policy_grad
        else torch.no_grad()
    )


    with grad_context:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            policy_logps = (
                response_logps_from_joint_batch(
                    policy_model,
                    policy_batch,
                )
            )


    policy_chosen = (
        policy_logps[0]
    )

    policy_rejected = (
        policy_logps[1]
    )


    # --------------------------------------------------------
    # REFERENCE
    # --------------------------------------------------------

    reference_batch = (
        move_batch_to_device(
            cpu_batch,
            reference_device,
        )
    )


    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            reference_logps = (
                response_logps_from_joint_batch(
                    reference_model,
                    reference_batch,
                )
            )


    # Bring frozen scalar values to policy GPU.
    reference_logps = reference_logps.to(
        policy_device
    )


    reference_chosen = (
        reference_logps[0]
    )

    reference_rejected = (
        reference_logps[1]
    )


    # --------------------------------------------------------
    # DPO
    # --------------------------------------------------------

    dpo_loss = (
        compute_dpo_loss_from_logps_impl(
            policy_chosen_logps=(
                policy_chosen.unsqueeze(0)
            ),
            policy_rejected_logps=(
                policy_rejected.unsqueeze(0)
            ),
            reference_chosen_logps=(
                reference_chosen.unsqueeze(0)
            ),
            reference_rejected_logps=(
                reference_rejected.unsqueeze(0)
            ),
            beta=beta,
        )
    )


    reward_chosen = (
        beta
        * (
            policy_chosen
            - reference_chosen
        )
    )

    reward_rejected = (
        beta
        * (
            policy_rejected
            - reference_rejected
        )
    )

    reward_margin = (
        reward_chosen
        - reward_rejected
    )


    # Strict >, exactly as specified by the manual.
    reward_correct = (
        reward_chosen
        > reward_rejected
    )


    return {
        "loss": dpo_loss,

        "policy_chosen_logp": (
            policy_chosen
        ),

        "policy_rejected_logp": (
            policy_rejected
        ),

        "reference_chosen_logp": (
            reference_chosen
        ),

        "reference_rejected_logp": (
            reference_rejected
        ),

        "reward_chosen": (
            reward_chosen
        ),

        "reward_rejected": (
            reward_rejected
        ),

        "reward_margin": (
            reward_margin
        ),

        "reward_correct": (
            reward_correct
        ),

        "chosen_response_tokens": (
            prepared[
                "chosen_response_tokens"
            ]
        ),

        "rejected_response_tokens": (
            prepared[
                "rejected_response_tokens"
            ]
        ),
    }


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_dpo(
    *,
    policy_model,
    reference_model,
    tokenizer,
    dataloader,
    max_length,
    beta,
    policy_device,
    reference_device,
    max_batches=None,
):
    """
    Validation metrics required by the assignment:

      - mean DPO loss
      - reward accuracy, strict rw > rl
      - mean reward margin
    """

    policy_was_training = (
        policy_model.training
    )

    reference_was_training = (
        reference_model.training
    )


    policy_model.eval()
    reference_model.eval()


    total_loss = 0.0
    correct = 0
    total_margin = 0.0

    total_examples = 0


    try:
        for batch_index, raw_batch in enumerate(
            dataloader
        ):

            if (
                max_batches is not None
                and batch_index >= max_batches
            ):
                break


            # Our reference DPO setup has pair batch size 1.
            if len(
                raw_batch["prompts"]
            ) != 1:
                raise ValueError(
                    "evaluate_dpo expects "
                    "per-device pair batch size 1."
                )


            q = compute_pair_quantities(
                policy_model=policy_model,
                reference_model=reference_model,
                tokenizer=tokenizer,

                prompt=(
                    raw_batch["prompts"][0]
                ),

                chosen=(
                    raw_batch["chosen"][0]
                ),

                rejected=(
                    raw_batch["rejected"][0]
                ),

                max_length=max_length,
                beta=beta,

                policy_device=(
                    policy_device
                ),

                reference_device=(
                    reference_device
                ),

                require_policy_grad=False,
            )


            total_loss += float(
                q["loss"]
                .detach()
                .float()
                .item()
            )

            correct += int(
                q["reward_correct"]
                .detach()
                .item()
            )

            total_margin += float(
                q["reward_margin"]
                .detach()
                .float()
                .item()
            )

            total_examples += 1


    finally:
        if policy_was_training:
            policy_model.train()

        if reference_was_training:
            reference_model.train()


    if total_examples == 0:
        raise RuntimeError(
            "No DPO validation examples evaluated."
        )


    return {
        "dpo_loss": (
            total_loss
            / total_examples
        ),

        "reward_accuracy": (
            correct
            / total_examples
        ),

        "reward_margin": (
            total_margin
            / total_examples
        ),

        "examples": (
            total_examples
        ),
    }


# ============================================================
# CHECKPOINT SAVING
# ============================================================

def save_dpo_checkpoint(
    *,
    model,
    tokenizer,
    output_dir,
    metadata,
):
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        output_dir,
    )

    write_json(
        output_dir
        / "checkpoint_metadata.json",
        metadata,
    )


# ============================================================
# FULL FROM-SCRATCH DPO TRAINING LOOP
# ============================================================

def train_dpo(
    *,
    policy_model,
    reference_model,
    tokenizer,

    train_loader,
    val_loader,

    config,
    output_dir,

    policy_device,
    reference_device,
):
    """
    Full manual DPO training loop.

    Pair batch size = 1.
    Gradient accumulation gives effective pair batch size 32.

    The final partial accumulation window is normalized by its
    actual number of pairs rather than always dividing by 32.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    metrics_path = (
        output_dir
        / "metrics.jsonl"
    )

    summary_path = (
        output_dir
        / "summary.json"
    )

    best_dir = (
        output_dir
        / "best_checkpoint"
    )

    final_dir = (
        output_dir
        / "final_checkpoint"
    )


    # --------------------------------------------------------
    # Refuse to accidentally append a new run onto an old one.
    # --------------------------------------------------------

    if metrics_path.exists():
        raise RuntimeError(
            f"{metrics_path} already exists. "
            "Refusing to mix two training runs."
        )

    if summary_path.exists():
        raise RuntimeError(
            f"{summary_path} already exists."
        )

    if best_dir.exists():
        raise RuntimeError(
            f"{best_dir} already exists."
        )

    if final_dir.exists():
        raise RuntimeError(
            f"{final_dir} already exists."
        )


    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    policy_model.train()

    reference_model.eval()

    reference_model.requires_grad_(
        False
    )


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),

        lr=config[
            "learning_rate"
        ],

        betas=(
            config["adam_beta1"],
            config["adam_beta2"],
        ),

        eps=config[
            "adam_epsilon"
        ],

        weight_decay=config[
            "weight_decay"
        ],
    )


    # FP32 trainable weights + FP16 autocast + gradient scaling.
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=config["amp_init_scale"],
        growth_interval=config["amp_growth_interval"],
        enabled=True,
    )


    optimizer.zero_grad(
        set_to_none=True
    )


    accumulation_steps = config[
        "gradient_accumulation_steps"
    ]

    total_micro_batches = len(
        train_loader
    )

    expected_updates = math.ceil(
        total_micro_batches
        / accumulation_steps
    )


    # --------------------------------------------------------
    # METRIC STATE
    # --------------------------------------------------------

    update_index = 0

    skipped_optimizer_steps = 0

    total_train_loss_sum = 0.0
    total_train_examples = 0

    window_loss_sum = 0.0
    window_examples = 0


    best_reward_accuracy = (
        -float("inf")
    )

    best_reward_margin = (
        -float("inf")
    )

    best_update = None

    best_validation = None


    start_time = time.time()


    # --------------------------------------------------------
    # FULL INITIAL VALIDATION
    # --------------------------------------------------------

    print(
        "Running full initial validation..."
    )

    initial_validation = evaluate_dpo(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,

        dataloader=val_loader,

        max_length=config[
            "context_length"
        ],

        beta=config["beta"],

        policy_device=policy_device,
        reference_device=(
            reference_device
        ),
    )


    append_jsonl(
        metrics_path,
        {
            "type": (
                "initial_validation"
            ),

            "update": 0,

            **initial_validation,
        },
    )


    print(
        "Initial validation:",
        initial_validation,
    )


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    progress = tqdm(
        total=expected_updates,
        desc=f"{config['run_name']} updates",
    )


    for micro_index, raw_batch in enumerate(
        train_loader
    ):
        # micro_index is zero-based.


        if len(
            raw_batch["prompts"]
        ) != 1:
            raise ValueError(
                "Reference DPO recipe expects "
                "per-device pair batch size 1."
            )


        # ----------------------------------------------------
        # Determine actual size of THIS accumulation window.
        #
        # Normal windows = 32 pairs.
        # Final window = 24 pairs for 15,000 examples.
        # ----------------------------------------------------

        window_start = (
            micro_index
            // accumulation_steps
        ) * accumulation_steps

        current_window_size = min(
            accumulation_steps,
            total_micro_batches
            - window_start,
        )


        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        quantities = (
            compute_pair_quantities(
                policy_model=policy_model,
                reference_model=(
                    reference_model
                ),
                tokenizer=tokenizer,

                prompt=(
                    raw_batch[
                        "prompts"
                    ][0]
                ),

                chosen=(
                    raw_batch[
                        "chosen"
                    ][0]
                ),

                rejected=(
                    raw_batch[
                        "rejected"
                    ][0]
                ),

                max_length=config[
                    "context_length"
                ],

                beta=config["beta"],

                policy_device=(
                    policy_device
                ),

                reference_device=(
                    reference_device
                ),

                require_policy_grad=True,
            )
        )


        loss = quantities["loss"]


        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                "Non-finite DPO loss at "
                f"micro-batch {micro_index}."
            )


        raw_loss_value = float(
            loss
            .detach()
            .float()
            .item()
        )


        total_train_loss_sum += (
            raw_loss_value
        )

        total_train_examples += 1

        window_loss_sum += (
            raw_loss_value
        )

        window_examples += 1


        # ----------------------------------------------------
        # BACKWARD
        #
        # Normalize by ACTUAL accumulation-window size.
        # ----------------------------------------------------

        scaled_loss = (
            loss
            / current_window_size
        )


        scaler.scale(
            scaled_loss
        ).backward()


        # ----------------------------------------------------
        # END OF ACCUMULATION WINDOW?
        # ----------------------------------------------------

        end_of_window = (
            (
                (micro_index + 1)
                % accumulation_steps
            )
            == 0
            or
            (micro_index + 1)
            == total_micro_batches
        )


        if not end_of_window:
            continue


        update_index += 1


        # ----------------------------------------------------
        # UNSCALE + CLIP
        # ----------------------------------------------------

        scaler.unscale_(
            optimizer
        )


        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(),
                max_norm=config[
                    "gradient_clip_norm"
                ],
            )
        )


        grad_norm_value = float(
            grad_norm
            .detach()
            .float()
            .item()
        )


        # ----------------------------------------------------
        # OPTIMIZER STEP
        # ----------------------------------------------------

        old_scale = (
            scaler.get_scale()
        )


        scaler.step(
            optimizer
        )

        scaler.update()


        new_scale = (
            scaler.get_scale()
        )


        optimizer_step_skipped = (
            new_scale < old_scale
        )


        if optimizer_step_skipped:
            skipped_optimizer_steps += 1


        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # TRAIN METRIC FOR THIS OPTIMIZER WINDOW
        # ----------------------------------------------------

        mean_window_loss = (
            window_loss_sum
            / window_examples
        )


        train_record = {
            "type": "train_update",

            "update": update_index,

            "micro_batches_seen": (
                micro_index + 1
            ),

            "pairs_in_update": (
                window_examples
            ),

            "train_dpo_loss": (
                mean_window_loss
            ),

            "grad_norm_before_clip": (
                grad_norm_value
            ),

            "learning_rate": (
                optimizer.param_groups[
                    0
                ]["lr"]
            ),

            "grad_scale": (
                new_scale
            ),

            "optimizer_step_skipped": (
                optimizer_step_skipped
            ),
        }


        append_jsonl(
            metrics_path,
            train_record,
        )


        progress.update(1)

        progress.set_postfix(
            loss=(
                f"{mean_window_loss:.4f}"
            ),
            grad=(
                f"{grad_norm_value:.3f}"
            ),
        )


        # Reset accumulation-window metric state.
        window_loss_sum = 0.0
        window_examples = 0


        # ----------------------------------------------------
        # VALIDATION
        #
        # Every N updates and ALWAYS at final update.
        # ----------------------------------------------------

        should_validate = (
            update_index
            % config[
                "eval_every_updates"
            ]
            == 0

            or

            update_index
            == expected_updates
        )


        if not should_validate:
            continue


        validation = evaluate_dpo(
            policy_model=policy_model,
            reference_model=(
                reference_model
            ),
            tokenizer=tokenizer,

            dataloader=val_loader,

            max_length=config[
                "context_length"
            ],

            beta=config["beta"],

            policy_device=(
                policy_device
            ),

            reference_device=(
                reference_device
            ),
        )


        validation_record = {
            "type": "validation",

            "update": update_index,

            **validation,
        }


        append_jsonl(
            metrics_path,
            validation_record,
        )


        print(
            "\nValidation at update",
            update_index,
            ":",
            validation,
        )


        # ----------------------------------------------------
        # BEST CHECKPOINT
        #
        # Primary criterion:
        #   highest reward accuracy
        #
        # Tie-break:
        #   higher reward margin
        # ----------------------------------------------------

        accuracy = (
            validation[
                "reward_accuracy"
            ]
        )

        margin = (
            validation[
                "reward_margin"
            ]
        )


        is_better = (
            accuracy
            > best_reward_accuracy
        )


        if (
            accuracy
            == best_reward_accuracy
            and
            margin
            > best_reward_margin
        ):
            is_better = True


        if is_better:
            best_reward_accuracy = (
                accuracy
            )

            best_reward_margin = (
                margin
            )

            best_update = (
                update_index
            )

            best_validation = (
                validation
            )


            save_dpo_checkpoint(
                model=policy_model,
                tokenizer=tokenizer,

                output_dir=best_dir,

                metadata={
                    "checkpoint_type": (
                        "best"
                    ),

                    "update": (
                        update_index
                    ),

                    "selection_criterion": (
                        "highest validation "
                        "reward accuracy; "
                        "reward margin "
                        "used only as tie-break"
                    ),

                    **validation,
                },
            )


            print(
                "New best checkpoint:",
                best_update,
            )


    progress.close()


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if update_index != expected_updates:
        raise RuntimeError(
            "Unexpected optimizer update count: "
            f"{update_index} vs "
            f"{expected_updates}"
        )


    if total_train_examples != (
        total_micro_batches
    ):
        raise RuntimeError(
            "Not all training pairs were processed."
        )


    # --------------------------------------------------------
    # FINAL VALIDATION
    #
    # The final update was already validated above because
    # should_validate includes expected_updates.
    # Reuse that result rather than running another 1000 pairs.
    # --------------------------------------------------------

    final_validation = validation


    # --------------------------------------------------------
    # FINAL CHECKPOINT
    # --------------------------------------------------------

    save_dpo_checkpoint(
        model=policy_model,
        tokenizer=tokenizer,

        output_dir=final_dir,

        metadata={
            "checkpoint_type": "final",

            "update": (
                update_index
            ),

            **final_validation,
        },
    )


    elapsed_seconds = (
        time.time()
        - start_time
    )


    summary = {
        "completed": True,

        "optimizer_updates": (
            update_index
        ),

        "expected_optimizer_updates": (
            expected_updates
        ),

        "skipped_optimizer_steps": (
            skipped_optimizer_steps
        ),

        "overall_training_dpo_loss": (
            total_train_loss_sum
            / total_train_examples
        ),

        "initial_validation": (
            initial_validation
        ),

        "final_validation": (
            final_validation
        ),

        "best_validation": (
            best_validation
        ),

        "best_reward_accuracy": (
            best_reward_accuracy
        ),

        "best_reward_margin": (
            best_reward_margin
        ),

        "best_update": (
            best_update
        ),

        "elapsed_seconds": (
            elapsed_seconds
        ),

        "elapsed_hours": (
            elapsed_seconds
            / 3600.0
        ),

        "best_checkpoint": (
            str(best_dir)
        ),

        "final_checkpoint": (
            str(final_dir)
        ),

        "metrics_file": (
            str(metrics_path)
        ),
    }


    write_json(
        summary_path,
        summary,
    )


    print(
        "\n===== DPO TRAINING COMPLETE ====="
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


    return {
        "summary": summary,
        "optimizer": optimizer,
        "scaler": scaler,
    }

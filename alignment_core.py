
from __future__ import annotations

from typing import Sequence

import torch


IGNORE_INDEX = -100


def _encode_without_special_tokens(tokenizer, text: str) -> list[int]:
    """
    Encode one string without allowing the tokenizer to insert BOS/EOS
    or any other special tokens automatically.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Hugging Face tokenizers normally return a Python list, but converting
    # explicitly makes this helper tolerant of other list-like results too.
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    return list(token_ids)


def _get_padding_token_id(tokenizer) -> int:
    """
    Return a usable padding token ID.

    Prefer the tokenizer's explicit pad token. If it has no pad token but
    does have EOS, EOS can safely be used as the physical padding value
    because padding positions are excluded by attention_mask and labels=-100.
    """
    pad_token_id = getattr(tokenizer, "pad_token_id", None)

    if pad_token_id is not None:
        return int(pad_token_id)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    if eos_token_id is not None:
        return int(eos_token_id)

    raise ValueError(
        "Tokenizer must provide either pad_token_id or eos_token_id "
        "so batches can be padded."
    )


def build_sft_batch_impl(
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
) -> dict[str, torch.Tensor]:
    """
    Build a padded causal-LM batch for supervised fine-tuning.

    Prompt tokens and padding tokens receive label -100 so that only
    assistant-response tokens contribute to the training loss.

    EOS, when provided by the tokenizer, is appended to the response
    BEFORE sequence truncation.
    """

    if not isinstance(max_length, int):
        raise TypeError("max_length must be an integer.")

    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    if len(prompts) != len(responses):
        raise ValueError(
            "prompts and responses must contain the same number of rows."
        )

    pad_token_id = _get_padding_token_id(tokenizer)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    input_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    label_rows: list[list[int]] = []

    for prompt, response in zip(prompts, responses):
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise TypeError("Every prompt and response must be a string.")

        prompt_ids = _encode_without_special_tokens(tokenizer, prompt)
        response_ids = _encode_without_special_tokens(tokenizer, response)

        # The manual requires EOS to be appended before truncation.
        if eos_token_id is not None:
            response_ids = response_ids + [int(eos_token_id)]

        # Keep prompt and response separate until labels are built so that
        # the response-only supervision boundary is exact.
        combined_ids = prompt_ids + response_ids

        # Truncate only after EOS has already been appended.
        combined_ids = combined_ids[:max_length]

        # Determine how many prompt tokens survived truncation.
        surviving_prompt_length = min(len(prompt_ids), len(combined_ids))

        # Labels equal the actual token IDs for response tokens only.
        labels = (
            [IGNORE_INDEX] * surviving_prompt_length
            + combined_ids[surviving_prompt_length:]
        )

        real_length = len(combined_ids)
        padding_length = max_length - real_length

        input_ids = combined_ids + [pad_token_id] * padding_length
        attention_mask = [1] * real_length + [0] * padding_length
        labels = labels + [IGNORE_INDEX] * padding_length

        input_rows.append(input_ids)
        attention_rows.append(attention_mask)
        label_rows.append(labels)

    # Handle a zero-row input cleanly as well.
    if len(input_rows) == 0:
        shape = (0, max_length)

        return {
            "input_ids": torch.empty(shape, dtype=torch.long),
            "attention_mask": torch.empty(shape, dtype=torch.long),
            "labels": torch.empty(shape, dtype=torch.long),
        }

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
    }


def _get_model_device(model) -> torch.device:
    """
    Infer the device on which the model expects its inputs.

    This works for the toy model used by the public grader as well as
    ordinary Hugging Face causal language models placed on one device.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def compute_response_logprobs_impl(
    model,
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
) -> torch.Tensor:
    """
    Return one summed response-token log probability per input example.

    Prompt tokens and padding tokens are excluded.

    For a causal LM, logits at sequence position t predict token t+1.
    Therefore we score labels[:, 1:] using logits[:, :-1, :].
    """

    batch = build_sft_batch_impl(
        tokenizer=tokenizer,
        prompts=prompts,
        responses=responses,
        max_length=max_length,
    )

    device = _get_model_device(model)

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logits = outputs.logits

    # Causal-language-model shift:
    #
    # logits[:, 0] predicts input_ids[:, 1]
    # logits[:, 1] predicts input_ids[:, 2]
    # ...
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    # Only response tokens have labels other than -100.
    response_mask = shift_labels.ne(IGNORE_INDEX)

    # gather() requires every index to be a valid vocabulary ID, so replace
    # ignored (-100) positions temporarily with 0. They are masked out later.
    safe_labels = shift_labels.masked_fill(~response_mask, 0)

    log_probs = torch.log_softmax(shift_logits, dim=-1)

    token_logps = log_probs.gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)

    # Prompt/padding locations contribute exactly zero.
    token_logps = token_logps * response_mask.to(token_logps.dtype)

    # DPO requires the SUM of response-token log probabilities per example.
    response_logps = token_logps.sum(dim=-1)

    return response_logps


def compute_sft_loss_impl(
    model,
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
) -> torch.Tensor:
    """
    Compute response-only SFT loss.

    Reduction used:
        mean negative log probability over all unmasked response tokens
        in the batch.

    Prompt tokens and padding tokens do not contribute to the loss.
    """

    batch = build_sft_batch_impl(
        tokenizer=tokenizer,
        prompts=prompts,
        responses=responses,
        max_length=max_length,
    )

    response_logps = compute_response_logprobs_impl(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        responses=responses,
        max_length=max_length,
    )

    # Because causal LM scoring uses labels[:, 1:], count only those
    # response targets that can actually be predicted.
    response_token_count = (
        batch["labels"][:, 1:]
        .ne(IGNORE_INDEX)
        .sum()
        .to(response_logps.device)
        .clamp_min(1)
    )

    loss = -response_logps.sum() / response_token_count

    return loss


def build_dpo_batch_impl(
    tokenizer,
    prompts: Sequence[str],
    chosen_responses: Sequence[str],
    rejected_responses: Sequence[str],
    max_length: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Build separate response-masked causal-LM batches for the chosen
    and rejected responses used by DPO.

    Both sides use the same prompt-processing, EOS, truncation,
    padding, and response-only masking logic as SFT.
    """

    if len(prompts) != len(chosen_responses):
        raise ValueError(
            "prompts and chosen_responses must contain the same number of rows."
        )

    if len(prompts) != len(rejected_responses):
        raise ValueError(
            "prompts and rejected_responses must contain the same number of rows."
        )

    chosen_batch = build_sft_batch_impl(
        tokenizer=tokenizer,
        prompts=prompts,
        responses=chosen_responses,
        max_length=max_length,
    )

    rejected_batch = build_sft_batch_impl(
        tokenizer=tokenizer,
        prompts=prompts,
        responses=rejected_responses,
        max_length=max_length,
    )

    return {
        "chosen": chosen_batch,
        "rejected": rejected_batch,
    }


def compute_dpo_loss_from_logps_impl(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta,
) -> torch.Tensor:
    """
    Compute the standard batch-mean DPO objective from per-example
    summed response log probabilities.

    For each example:

        policy_gap =
            log pi_policy(chosen | prompt)
            - log pi_policy(rejected | prompt)

        reference_gap =
            log pi_ref(chosen | prompt)
            - log pi_ref(rejected | prompt)

        loss =
            -log sigmoid(
                beta * (policy_gap - reference_gap)
            )
    """

    policy_gap = (
        policy_chosen_logps
        - policy_rejected_logps
    )

    reference_gap = (
        reference_chosen_logps
        - reference_rejected_logps
    )

    preference_logits = beta * (
        policy_gap - reference_gap
    )

    losses = -torch.nn.functional.logsigmoid(
        preference_logits
    )

    return losses.mean()


def compute_dpo_loss_impl(
    policy_model,
    reference_model,
    tokenizer,
    prompts: Sequence[str],
    chosen_responses: Sequence[str],
    rejected_responses: Sequence[str],
    beta,
    max_length: int,
) -> torch.Tensor:
    """
    Compute the full DPO loss using a trainable policy model and
    a frozen reference model.

    Policy computations retain gradients.
    Reference computations are performed under torch.no_grad().
    """

    # -----------------------------
    # Policy log probabilities
    # -----------------------------
    policy_chosen_logps = compute_response_logprobs_impl(
        model=policy_model,
        tokenizer=tokenizer,
        prompts=prompts,
        responses=chosen_responses,
        max_length=max_length,
    )

    policy_rejected_logps = compute_response_logprobs_impl(
        model=policy_model,
        tokenizer=tokenizer,
        prompts=prompts,
        responses=rejected_responses,
        max_length=max_length,
    )

    # -----------------------------
    # Reference log probabilities
    #
    # IMPORTANT:
    # No gradients should ever flow
    # through the reference model.
    # -----------------------------
    with torch.no_grad():
        reference_chosen_logps = compute_response_logprobs_impl(
            model=reference_model,
            tokenizer=tokenizer,
            prompts=prompts,
            responses=chosen_responses,
            max_length=max_length,
        )

        reference_rejected_logps = compute_response_logprobs_impl(
            model=reference_model,
            tokenizer=tokenizer,
            prompts=prompts,
            responses=rejected_responses,
            max_length=max_length,
        )

    # The real Kaggle DPO workflow may place:
    #
    #   policy    -> cuda:0
    #   reference -> cuda:1
    #
    # Move the small reference log-probability tensors back to the
    # policy device before combining them.
    reference_chosen_logps = reference_chosen_logps.to(
        policy_chosen_logps.device
    )

    reference_rejected_logps = reference_rejected_logps.to(
        policy_rejected_logps.device
    )

    loss = compute_dpo_loss_from_logps_impl(
        policy_chosen_logps=policy_chosen_logps,
        policy_rejected_logps=policy_rejected_logps,
        reference_chosen_logps=reference_chosen_logps,
        reference_rejected_logps=reference_rejected_logps,
        beta=beta,
    )

    return loss

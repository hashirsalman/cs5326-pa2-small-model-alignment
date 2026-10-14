
from __future__ import annotations


def extract_single_turn(example):
    """
    Convert an UltraChat-style conversation into one single-turn
    instruction-response example.

    Only the first user message and first assistant response are used.
    Later conversation turns are intentionally discarded.
    """

    messages = example.get("messages", [])

    if len(messages) < 2:
        return None

    if messages[0].get("role") != "user":
        return None

    if messages[1].get("role") != "assistant":
        return None

    instruction = messages[0].get("content", "").strip()
    response = messages[1].get("content", "").strip()

    if not instruction or not response:
        return None

    return {
        "instruction": instruction,
        "response": response,
        "source_id": example.get("prompt_id"),
    }


def format_instruction_prompt(instruction: str) -> str:
    """
    Shared prompt format for SFT, scratch DPO, and evaluation.
    """

    instruction = instruction.strip()

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


# ============================================================
# HH-RLHF SINGLE-TURN EXTRACTION
# ============================================================

def extract_hh_single_turn(example):
    """
    Convert one Anthropic HH-RLHF row into:

        instruction
        chosen
        rejected

    Only accepts strict single-turn conversations:

        Human: <prompt>
        Assistant: <response>

    for both chosen and rejected, with exactly the same Human
    prompt on both sides.

    Multi-turn rows are rejected.
    """

    chosen_text = example.get("chosen")
    rejected_text = example.get("rejected")

    if not isinstance(chosen_text, str):
        return None

    if not isinstance(rejected_text, str):
        return None


    human_marker = "\n\nHuman:"
    assistant_marker = "\n\nAssistant:"


    # Require exactly one Human turn.
    if chosen_text.count(human_marker) != 1:
        return None

    if rejected_text.count(human_marker) != 1:
        return None


    # Require exactly one Assistant turn.
    if chosen_text.count(assistant_marker) != 1:
        return None

    if rejected_text.count(assistant_marker) != 1:
        return None


    # Both transcripts must begin with Human.
    if not chosen_text.startswith(human_marker):
        return None

    if not rejected_text.startswith(human_marker):
        return None


    # Split prompt and response.
    chosen_before, chosen_sep, chosen_response = (
        chosen_text.partition(assistant_marker)
    )

    rejected_before, rejected_sep, rejected_response = (
        rejected_text.partition(assistant_marker)
    )

    if not chosen_sep or not rejected_sep:
        return None


    chosen_prompt = chosen_before[
        len(human_marker):
    ].strip()

    rejected_prompt = rejected_before[
        len(human_marker):
    ].strip()

    chosen_response = chosen_response.strip()
    rejected_response = rejected_response.strip()


    # Both preference alternatives must use exactly
    # the same prompt.
    if chosen_prompt != rejected_prompt:
        return None


    if not chosen_prompt:
        return None

    if not chosen_response:
        return None

    if not rejected_response:
        return None


    # No useful preference signal if responses are identical.
    if chosen_response == rejected_response:
        return None


    return {
        "instruction": chosen_prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }


# ============================================================
# ULTRAFEEDBACK SINGLE-TURN EXTRACTION
# ============================================================

def extract_ultrafeedback_single_turn(example):
    """
    Extract a strict single-turn UltraFeedback preference pair.

    Expected structure:

        chosen = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen_response},
        ]

        rejected = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected_response},
        ]

    The explicit example["prompt"] must agree with the user
    message on both sides.
    """

    prompt = example.get("prompt")
    chosen = example.get("chosen")
    rejected = example.get("rejected")


    if not isinstance(prompt, str):
        return None

    if not isinstance(chosen, list):
        return None

    if not isinstance(rejected, list):
        return None


    # Strict single-turn structure.
    if len(chosen) != 2:
        return None

    if len(rejected) != 2:
        return None


    # Roles must be user -> assistant.
    if (
        chosen[0].get("role") != "user"
        or chosen[1].get("role") != "assistant"
    ):
        return None

    if (
        rejected[0].get("role") != "user"
        or rejected[1].get("role") != "assistant"
    ):
        return None


    prompt = prompt.strip()

    chosen_prompt = (
        chosen[0].get("content", "").strip()
    )

    rejected_prompt = (
        rejected[0].get("content", "").strip()
    )

    chosen_response = (
        chosen[1].get("content", "").strip()
    )

    rejected_response = (
        rejected[1].get("content", "").strip()
    )


    if not prompt:
        return None

    if not chosen_response:
        return None

    if not rejected_response:
        return None


    # The explicit prompt and both message-list prompts
    # must describe exactly the same instruction.
    if prompt != chosen_prompt:
        return None

    if prompt != rejected_prompt:
        return None


    # Degenerate pair gives no usable preference signal.
    if chosen_response == rejected_response:
        return None


    return {
        "instruction": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
        "prompt_id": example.get("prompt_id"),
        "score_chosen": example.get("score_chosen"),
        "score_rejected": example.get("score_rejected"),
    }

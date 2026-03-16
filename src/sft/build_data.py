from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def format_input(example: dict[str, Any]) -> str:
    opts = example["options"]
    return (
        f"Question:\n{example['question']}\n\n"
        f"Options:\nA. {opts['A']}\nB. {opts['B']}\nC. {opts['C']}\nD. {opts['D']}\n\n"
        "Return answer in format <answer>A|B|C|D</answer>."
    )


def compress_reasoning(text: str, max_tokens: int) -> str:
    toks = text.split()
    return " ".join(toks[:max_tokens])


def build_target(reasoning: str, answer: str, variant: str) -> str:
    if variant == "answer_only":
        return f"<answer>\n{answer}\n</answer>"
    if variant == "short_cot_answer":
        return f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{answer}\n</answer>"
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{answer}\n</answer>"


def build_sft_dataset(
    canonical_test_jsonl: str | Path,
    teacher_parsed_jsonl: str | Path,
    out_path: str | Path,
    variant: str,
    short_rationale_tokens: int = 80,
    filtered_only_correct: bool = False,
    shuffled_cot: bool = False,
    shuffled_answer_protocol: str = "keep_teacher",
    rationale_truncation_tokens: int | None = None,
    keep_only: str | None = None,
    sample_fraction: float = 1.0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    examples = {row["id"]: row for row in read_jsonl(canonical_test_jsonl)}
    teacher_rows = read_jsonl(teacher_parsed_jsonl)
    random.seed(seed)

    if shuffled_cot:
        reasonings = [r.get("reasoning", "") for r in teacher_rows]
        random.shuffle(reasonings)
        for idx, row in enumerate(teacher_rows):
            row["reasoning"] = reasonings[idx]

    sft_rows: list[dict[str, Any]] = []
    for row in teacher_rows:
        ex = examples.get(row["id"])
        if ex is None or not row.get("valid", False):
            continue

        teacher_correct = row["teacher_answer"] == row["gold_answer"]
        if filtered_only_correct and not teacher_correct:
            continue
        if keep_only == "correct" and not teacher_correct:
            continue
        if keep_only == "wrong" and teacher_correct:
            continue

        reasoning = row.get("reasoning", "")
        answer = row["teacher_answer"]

        if shuffled_cot and shuffled_answer_protocol == "keep_gold":
            answer = row["gold_answer"]

        resolved_variant = variant
        if variant == "short_cot_answer":
            reasoning = compress_reasoning(reasoning, short_rationale_tokens)
        if rationale_truncation_tokens is not None:
            reasoning = compress_reasoning(reasoning, rationale_truncation_tokens)

        target = build_target(reasoning, answer, resolved_variant)
        sft_rows.append(
            {
                "id": ex["id"],
                "input": format_input(ex),
                "target": target,
                "messages": [
                    {"role": "user", "content": format_input(ex)},
                    {"role": "assistant", "content": target},
                ],
                "metadata": {
                    "gold_answer": row["gold_answer"],
                    "teacher_answer": row["teacher_answer"],
                    "teacher_correct": teacher_correct,
                    "reasoning_length": len(reasoning.split()),
                    "variant": variant,
                },
            }
        )

    if sample_fraction < 1.0:
        k = max(1, int(len(sft_rows) * sample_fraction))
        sft_rows = random.sample(sft_rows, k)

    ensure_dir(Path(out_path).parent)
    write_jsonl(sft_rows, out_path)
    logger.info("Built SFT dataset variant=%s rows=%d output=%s", variant, len(sft_rows), out_path)
    return sft_rows

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.sft.build_data import build_sft_dataset
from src.utils.io import ensure_dir


def run_ablation_data_builds(
    canonical_test_jsonl: str | Path,
    teacher_parsed_jsonl: str | Path,
    out_dir: str | Path,
    ablation_cfg: dict[str, Any],
    seed: int,
) -> list[str]:
    out_root = ensure_dir(out_dir)
    outputs: list[str] = []

    outputs.append(
        str(
            out_root
            / "answer_only.jsonl"
        )
    )
    build_sft_dataset(canonical_test_jsonl, teacher_parsed_jsonl, outputs[-1], variant="answer_only", seed=seed)

    outputs.append(str(out_root / "filtered_correct_only.jsonl"))
    build_sft_dataset(
        canonical_test_jsonl,
        teacher_parsed_jsonl,
        outputs[-1],
        variant="cot_answer",
        filtered_only_correct=True,
        seed=seed,
    )

    outputs.append(str(out_root / "wrong_only.jsonl"))
    build_sft_dataset(
        canonical_test_jsonl,
        teacher_parsed_jsonl,
        outputs[-1],
        variant="cot_answer",
        keep_only="wrong",
        seed=seed,
    )

    outputs.append(str(out_root / "shuffled_cot_keep_teacher.jsonl"))
    build_sft_dataset(
        canonical_test_jsonl,
        teacher_parsed_jsonl,
        outputs[-1],
        variant="cot_answer",
        shuffled_cot=True,
        shuffled_answer_protocol="keep_teacher",
        seed=seed,
    )

    outputs.append(str(out_root / "shuffled_cot_keep_gold.jsonl"))
    build_sft_dataset(
        canonical_test_jsonl,
        teacher_parsed_jsonl,
        outputs[-1],
        variant="cot_answer",
        shuffled_cot=True,
        shuffled_answer_protocol="keep_gold",
        seed=seed,
    )

    outputs.append(str(out_root / "truncated_cot_64.jsonl"))
    build_sft_dataset(
        canonical_test_jsonl,
        teacher_parsed_jsonl,
        outputs[-1],
        variant="cot_answer",
        rationale_truncation_tokens=64,
        seed=seed,
    )

    for frac in ablation_cfg.get("sample_efficiency_fracs", [0.1, 0.25, 0.5, 1.0]):
        path = str(out_root / f"sample_efficiency_{int(frac * 100)}.jsonl")
        build_sft_dataset(
            canonical_test_jsonl,
            teacher_parsed_jsonl,
            path,
            variant="cot_answer",
            sample_fraction=float(frac),
            seed=seed,
        )
        outputs.append(path)

    return outputs

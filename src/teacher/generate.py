from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.teacher.clients import GenerationConfig, build_teacher_client
from src.teacher.parser import parse_teacher_output
from src.teacher.prompting import build_prompt
from src.utils.io import ensure_dir, read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def generate_teacher_outputs(
    dataset_jsonl: str | Path,
    output_dir: str | Path,
    teacher_cfg: dict[str, Any],
    resume: bool = True,
) -> None:
    examples = read_jsonl(dataset_jsonl)
    out_dir = ensure_dir(output_dir)
    raw_path = out_dir / "teacher_raw.jsonl"
    parsed_path = out_dir / "teacher_parsed.jsonl"

    existing: dict[str, dict[str, Any]] = {}
    if resume and parsed_path.exists():
        for row in read_jsonl(parsed_path):
            existing[row["id"]] = row
        logger.info("Resuming teacher generation with %d existing parsed records", len(existing))

    client = build_teacher_client(teacher_cfg)
    gen_cfg = GenerationConfig(
        max_new_tokens=teacher_cfg["max_new_tokens"],
        temperature=teacher_cfg.get("temperature", 0.0),
        top_p=teacher_cfg.get("top_p", 1.0),
    )
    retries = teacher_cfg.get("retries", 3)
    n_samples = teacher_cfg.get("n_samples", 1)
    mode = teacher_cfg.get("mode", "cot_answer")

    raw_rows: list[dict[str, Any]] = read_jsonl(raw_path) if raw_path.exists() else []
    parsed_rows: list[dict[str, Any]] = read_jsonl(parsed_path) if parsed_path.exists() else []

    for ex in tqdm(examples, desc="teacher-gen"):
        if ex["id"] in existing:
            continue
        prompt = build_prompt(ex, mode=mode)
        sample_outputs = []
        selected = None

        for sample_i in range(n_samples):
            parsed = None
            raw_text = ""
            for attempt in range(retries):
                try:
                    raw_text = client.generate(prompt, gen_cfg)
                    parsed = parse_teacher_output(raw_text, mode=mode)
                    if parsed["valid"]:
                        break
                except Exception as exc:
                    logger.warning("id=%s sample=%d attempt=%d failed: %s", ex["id"], sample_i, attempt, exc)
                time.sleep(0.3)
            sample_outputs.append({"sample_index": sample_i, "raw": raw_text, "parsed": parsed})
            if selected is None and parsed and parsed["valid"]:
                selected = parsed

        if selected is None:
            selected = {"valid": False, "answer": None, "reasoning": "", "raw": sample_outputs[0]["raw"] if sample_outputs else ""}

        raw_rows.append({"id": ex["id"], "prompt": prompt, "samples": sample_outputs, "mode": mode})
        parsed_rows.append(
            {
                "id": ex["id"],
                "gold_answer": ex["answer"],
                "teacher_answer": selected["answer"],
                "reasoning": selected.get("reasoning", ""),
                "valid": selected["valid"],
                "mode": mode,
            }
        )

    write_jsonl(raw_rows, raw_path)
    write_jsonl(parsed_rows, parsed_path)
    logger.info("Saved teacher outputs to %s", out_dir)

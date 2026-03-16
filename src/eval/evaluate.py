from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.sft.build_data import format_input
from src.utils.io import dump_json, ensure_dir, read_jsonl, write_jsonl
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

ANSWER_PATTERNS = [
    re.compile(r"<answer>\s*([ABCD])\s*</answer>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\b([ABCD])\b"),
]


def parse_answer(text: str) -> str | None:
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


def evaluate_model(
    model_path: str,
    test_jsonl: str | Path,
    out_dir: str | Path,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
) -> dict[str, Any]:
    examples = read_jsonl(test_jsonl)
    out_root = ensure_dir(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    preds: list[dict[str, Any]] = []
    for ex in tqdm(examples, desc="eval"):
        prompt = format_input(ex)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            max_new_tokens=max_new_tokens,
        )
        txt = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = parse_answer(txt)
        preds.append(
            {
                "id": ex["id"],
                "gold": ex["answer"],
                "prediction": pred,
                "valid": pred in {"A", "B", "C", "D"},
                "correct": pred == ex["answer"],
                "raw_output": txt,
                "output_len": len(txt.split()),
                "question_len": len(ex["question"].split()),
            }
        )

    write_jsonl(preds, out_root / "predictions.jsonl")

    valid = [p for p in preds if p["valid"]]
    metrics = {
        "n": len(preds),
        "accuracy": float(np.mean([p["correct"] for p in preds])) if preds else 0.0,
        "exact_match": float(np.mean([p["correct"] for p in preds])) if preds else 0.0,
        "valid_answer_rate": len(valid) / max(1, len(preds)),
        "avg_output_length": float(np.mean([p["output_len"] for p in preds])) if preds else 0.0,
    }

    y_true = [p["gold"] for p in preds]
    y_pred = [p["prediction"] if p["prediction"] else "A" for p in preds]
    labels = ["A", "B", "C", "D"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4), labels)
    ax.set_yticks(range(4), labels)
    ax.set_xlabel("Pred")
    ax.set_ylabel("Gold")
    ax.set_title("Confusion Matrix")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_root / "confusion_matrix.png")
    plt.close(fig)

    # Breakdown by question length quartiles
    qlens = np.array([p["question_len"] for p in preds]) if preds else np.array([0])
    bins = np.quantile(qlens, [0, 0.25, 0.5, 0.75, 1.0]) if len(qlens) > 1 else np.array([0, 1, 2, 3, 4])
    breakdown = []
    for i in range(4):
        lo, hi = bins[i], bins[i + 1]
        group = [p for p in preds if lo <= p["question_len"] <= hi]
        acc = float(np.mean([g["correct"] for g in group])) if group else 0.0
        breakdown.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(group), "accuracy": acc})
    dump_json(breakdown, out_root / "breakdown_by_question_length.json")

    dump_json(metrics, out_root / "metrics.json")
    logger.info("Eval metrics: %s", metrics)
    return metrics

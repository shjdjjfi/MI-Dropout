from __future__ import annotations

import re
from typing import Any

ANSWER_RE = re.compile(r"<answer>\s*([ABCD])\s*</answer>", re.IGNORECASE | re.DOTALL)
REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.IGNORECASE | re.DOTALL)


def parse_teacher_output(text: str, mode: str = "cot_answer") -> dict[str, Any]:
    parsed: dict[str, Any] = {"raw": text, "valid": False, "answer": None, "reasoning": ""}
    answer_match = ANSWER_RE.search(text)
    if not answer_match:
        return parsed
    parsed["answer"] = answer_match.group(1).upper()
    if mode == "cot_answer":
        reason_match = REASONING_RE.search(text)
        parsed["reasoning"] = reason_match.group(1).strip() if reason_match else ""
    parsed["valid"] = True
    return parsed

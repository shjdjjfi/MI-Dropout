from __future__ import annotations

PROMPT_TEMPLATE_COT = """You are a medical expert solving a USMLE-style multiple-choice question.
Think clinically and step-by-step.
Return EXACTLY this XML-like format:
<reasoning>
...your medical reasoning...
</reasoning>
<answer>
A|B|C|D
</answer>

Question:
{question}

Options:
A. {A}
B. {B}
C. {C}
D. {D}
"""

PROMPT_TEMPLATE_ANSWER_ONLY = """You are a medical expert solving a USMLE-style multiple-choice question.
Return EXACTLY this format:
<answer>
A|B|C|D
</answer>

Question:
{question}

Options:
A. {A}
B. {B}
C. {C}
D. {D}
"""


def build_prompt(example: dict, mode: str = "cot_answer") -> str:
    base = PROMPT_TEMPLATE_COT if mode == "cot_answer" else PROMPT_TEMPLATE_ANSWER_ONLY
    return base.format(question=example["question"], **example["options"])

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.prompting.model_backend import DummyBackend, GenerationConfig
from src.prompting.templates import COT_TEMPLATE, DIRECT_TEMPLATE, CONCISE_COT_TEMPLATE, GRAPH_TEMPLATE


@dataclass(slots=True)
class BaselineOutput:
    baseline_name: str
    prediction: str
    metadata: dict[str, Any]


def _format_context(ctx: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {d.get('title','doc')}: {d.get('text','')[:300]}" for d in ctx[:8])


def run_baselines(instance: dict[str, Any], model_name: str = "dummy") -> list[BaselineOutput]:
    backend = DummyBackend(model_name)
    context = _format_context(instance.get("candidate_context", []))

    prompts = {
        "direct": DIRECT_TEMPLATE.format(question=instance["question"], context=context),
        "linear_cot": COT_TEMPLATE.format(question=instance["question"], context=context),
        "self_consistency_cot": COT_TEMPLATE.format(question=instance["question"], context=context),
        "tot_search": COT_TEMPLATE.format(question=instance["question"], context=context) + "\nExplore alternative branches.",
        "got_prompt": GRAPH_TEMPLATE.format(question=instance["question"], nodes=context, edges="implicit"),
        "retrieve_then_read": DIRECT_TEMPLATE.format(question=instance["question"], context=context),
        "oracle_support": DIRECT_TEMPLATE.format(question=instance["question"], context=str(instance.get("supporting_facts", []))),
        "concise_cot": CONCISE_COT_TEMPLATE.format(question=instance["question"], context=context, budget=80),
    }

    outputs: list[BaselineOutput] = []
    for name, prompt in prompts.items():
        pred = backend.generate(prompt, GenerationConfig())
        outputs.append(BaselineOutput(name, pred, {"prompt_len": len(prompt)}))
    return outputs

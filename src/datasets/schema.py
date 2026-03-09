from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ReasoningInstance:
    id: str
    question: str
    candidate_context: list[dict[str, Any]]
    gold_answer: str | list[str] | bool
    supporting_facts: list[dict[str, Any]] = field(default_factory=list)
    decomposition: list[str] = field(default_factory=list)
    proof_chain: list[str] = field(default_factory=list)
    dataset_name: str = ""
    split: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "candidate_context": self.candidate_context,
            "gold_answer": self.gold_answer,
            "supporting_facts": self.supporting_facts,
            "decomposition": self.decomposition,
            "proof_chain": self.proof_chain,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "metadata": self.metadata,
        }

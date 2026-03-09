from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from datasets import load_dataset

from src.datasets.schema import ReasoningInstance
from src.utils.io_utils import stable_hash, write_jsonl

LOGGER = logging.getLogger(__name__)

DATASET_REGISTRY = {
    "hotpotqa": ("hotpot_qa", "distractor"),
    "musique": ("musique", "default"),
    "2wikimultihopqa": ("scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35", "default"),
    "strategyqa": ("tasksource/bigbench", "strategyqa"),
    "wikihop": ("qangaroo", "wikihop"),
    "proofwriter": ("proofwriter", "default"),
}


class BaseDatasetNormalizer(ABC):
    dataset_key: str

    @abstractmethod
    def normalize_record(self, row: dict[str, Any], split: str) -> ReasoningInstance:
        raise NotImplementedError

    def normalize_split(self, split: str, limit: int | None = None) -> list[ReasoningInstance]:
        name, config = DATASET_REGISTRY[self.dataset_key]
        dataset = load_dataset(name, config, split=split)
        rows: list[ReasoningInstance] = []
        for i, row in enumerate(dataset):
            rows.append(self.normalize_record(row, split=split))
            if limit and i + 1 >= limit:
                break
        LOGGER.info("Loaded %d records from %s/%s", len(rows), self.dataset_key, split)
        return rows


def _context_from_hotpot(row: dict[str, Any]) -> list[dict[str, Any]]:
    titles = row.get("context", {}).get("title", [])
    sents = row.get("context", {}).get("sentences", [])
    return [{"title": t, "text": " ".join(sent_list)} for t, sent_list in zip(titles, sents)]


class GenericNormalizer(BaseDatasetNormalizer):
    def __init__(self, dataset_key: str):
        self.dataset_key = dataset_key

    def normalize_record(self, row: dict[str, Any], split: str) -> ReasoningInstance:
        question = row.get("question") or row.get("input") or row.get("query", "")
        answer = row.get("answer") or row.get("target") or row.get("label", "")
        if isinstance(answer, dict):
            answer = str(answer)

        context = row.get("context") or row.get("supports") or row.get("passages") or []
        if self.dataset_key == "hotpotqa":
            context = _context_from_hotpot(row)

        if isinstance(context, list) and context and isinstance(context[0], str):
            context = [{"title": f"doc_{idx}", "text": txt} for idx, txt in enumerate(context)]

        supporting = row.get("supporting_facts") or row.get("evidence") or []
        decomposition = row.get("decomposition") or row.get("sub_questions") or []
        proof_chain = row.get("proof") or row.get("proof_chain") or []
        rid = str(row.get("id") or stable_hash({"q": question, "a": answer}))

        return ReasoningInstance(
            id=rid,
            question=question,
            candidate_context=context,
            gold_answer=answer,
            supporting_facts=supporting,
            decomposition=decomposition if isinstance(decomposition, list) else [str(decomposition)],
            proof_chain=proof_chain if isinstance(proof_chain, list) else [str(proof_chain)],
            dataset_name=self.dataset_key,
            split=split,
            metadata={k: v for k, v in row.items() if k not in {"question", "answer", "context"}},
        )


def build_all_datasets(output_dir: str, splits: list[str], limit: int | None = None) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key in DATASET_REGISTRY:
        normalizer = GenericNormalizer(key)
        for split in splits:
            records = [rec.to_dict() for rec in normalizer.normalize_split(split, limit=limit)]
            write_jsonl(records, out / f"{key}_{split}.jsonl")


if __name__ == "__main__":
    build_all_datasets("data/normalized", ["train", "validation"], limit=200)

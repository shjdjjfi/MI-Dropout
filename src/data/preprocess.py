from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset

from src.utils.io import dump_json, ensure_dir, write_jsonl
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

OPTION_KEYS = ["A", "B", "C", "D"]


@dataclass
class NormalizedExample:
    id: str
    question: str
    options: dict[str, str]
    answer: str


class MedQAPreprocessor:
    def __init__(self, hf_id: str, fallback_hf_id: str | None = None, subset: str | None = None):
        self.hf_id = hf_id
        self.fallback_hf_id = fallback_hf_id
        self.subset = subset

    def _load(self) -> DatasetDict:
        try:
            logger.info("Loading dataset: %s (subset=%s)", self.hf_id, self.subset)
            return load_dataset(self.hf_id, self.subset) if self.subset else load_dataset(self.hf_id)
        except Exception as exc:
            if not self.fallback_hf_id:
                raise
            logger.warning("Primary dataset failed (%s). Falling back to %s", exc, self.fallback_hf_id)
            return load_dataset(self.fallback_hf_id)

    @staticmethod
    def _normalize_answer(raw: Any) -> str:
        val = str(raw).strip().upper()
        if val in OPTION_KEYS:
            return val
        if val.isdigit() and int(val) in range(4):
            return OPTION_KEYS[int(val)]
        map_words = {"0": "A", "1": "B", "2": "C", "3": "D"}
        if val in map_words:
            return map_words[val]
        raise ValueError(f"Cannot normalize answer: {raw}")

    @staticmethod
    def _normalize_options(row: dict[str, Any]) -> dict[str, str]:
        if all(k in row for k in ["ending0", "ending1", "ending2", "ending3"]):
            return {k: str(row[f"ending{i}"]).strip() for i, k in enumerate(OPTION_KEYS)}

        if "options" in row:
            opts = row["options"]
            if isinstance(opts, dict):
                out: dict[str, str] = {}
                for k in OPTION_KEYS:
                    candidate = opts.get(k) or opts.get(k.lower())
                    if candidate is None:
                        candidate = opts.get(str(OPTION_KEYS.index(k)))
                    if candidate is None:
                        raise ValueError(f"Missing option {k} in options dict")
                    out[k] = str(candidate).strip()
                return out
            if isinstance(opts, list) and len(opts) >= 4:
                return {k: str(opts[i]).strip() for i, k in enumerate(OPTION_KEYS)}

        flat_candidates = [
            ("A", ["A", "a", "option_a", "opa", "choice_a"]),
            ("B", ["B", "b", "option_b", "opb", "choice_b"]),
            ("C", ["C", "c", "option_c", "opc", "choice_c"]),
            ("D", ["D", "d", "option_d", "opd", "choice_d"]),
        ]
        out = {}
        for key, aliases in flat_candidates:
            for alias in aliases:
                if alias in row:
                    out[key] = str(row[alias]).strip()
                    break
        if len(out) == 4:
            return out

        raise ValueError("Unable to infer options format")

    @staticmethod
    def _extract_question(row: dict[str, Any]) -> str:
        for key in ["question", "sent1", "query", "prompt"]:
            if key in row and row[key] is not None:
                return str(row[key]).strip()
        raise ValueError("Unable to infer question column")

    @staticmethod
    def _extract_id(row: dict[str, Any], idx: int) -> str:
        for key in ["id", "qid", "question_id", "uuid"]:
            if key in row and row[key] is not None:
                return str(row[key])
        return f"sample_{idx}"

    @staticmethod
    def _extract_answer(row: dict[str, Any]) -> Any:
        for key in ["answer", "label", "target", "correct_option"]:
            if key in row and row[key] is not None:
                return row[key]
        raise ValueError("Unable to infer answer column")

    def normalize_split(self, split_rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        skipped = 0
        for idx, row in enumerate(split_rows):
            try:
                ex = NormalizedExample(
                    id=self._extract_id(row, idx),
                    question=self._extract_question(row),
                    options=self._normalize_options(row),
                    answer=self._normalize_answer(self._extract_answer(row)),
                )
                normalized.append(ex.__dict__)
            except Exception as exc:
                skipped += 1
                logger.warning("Skipping %s idx=%d due to normalization error: %s", split_name, idx, exc)
        logger.info("Split %s normalized: kept=%d skipped=%d", split_name, len(normalized), skipped)
        return normalized

    def run(self, out_dir: str | Path, split_mapping: dict[str, str], max_samples_per_split: int | None = None) -> dict[str, Any]:
        dataset = self._load()
        out_root = ensure_dir(out_dir)
        stats: dict[str, Any] = {"dataset_id": self.hf_id, "splits": {}, "mappings": {}}

        logger.info("Dataset splits available: %s", list(dataset.keys()))
        for split_alias, split_name in split_mapping.items():
            if split_name not in dataset:
                logger.warning("Requested split %s not found. Skipping.", split_name)
                continue
            ds = dataset[split_name]
            cols = ds.column_names
            stats["mappings"][split_alias] = {"source_split": split_name, "columns": cols}
            logger.info("Split %s columns: %s", split_name, cols)
            rows = [ds[i] for i in range(len(ds))]
            if max_samples_per_split:
                rows = rows[:max_samples_per_split]
            normalized = self.normalize_split(rows, split_alias)
            write_jsonl(normalized, out_root / f"{split_alias}.jsonl")
            stats["splits"][split_alias] = {
                "num_rows": len(normalized),
                "avg_question_len_chars": sum(len(r["question"]) for r in normalized) / max(1, len(normalized)),
            }

        dump_json(stats, out_root / "stats.json")
        logger.info("Saved canonical dataset and stats to %s", out_root)
        return stats

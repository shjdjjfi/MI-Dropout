from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from src.bottleneck.structured_ib import IBConfig, apply_structured_ib, extract_subgraph
from src.evaluation.metrics import exact_match, f1_score
from src.graph.induction import graph_to_json, induce_graph
from src.prompting.baselines import run_baselines
from src.utils.io_utils import read_jsonl


@dataclass(slots=True)
class EvalRecord:
    instance_id: str
    baseline: str
    em: float
    f1: float
    graph_nodes: int
    compressed_nodes: int
    compression_ratio: float
    latency_ms: float


def evaluate_file(input_jsonl: str, output_csv: str, seed: int = 13) -> None:
    rows = read_jsonl(input_jsonl)
    out_records: list[EvalRecord] = []
    for row in rows:
        graph = induce_graph(row)
        ib = apply_structured_ib(graph, mode="gumbel_topk", config=IBConfig(seed=seed))
        subgraph = extract_subgraph(graph, ib)

        start = time.time()
        preds = run_baselines(row)
        latency = (time.time() - start) * 1000

        gold = str(row.get("gold_answer", ""))
        for pred in preds:
            em = exact_match(pred.prediction, gold)
            f1 = f1_score(pred.prediction, gold)
            out_records.append(
                EvalRecord(
                    instance_id=row["id"],
                    baseline=pred.baseline_name,
                    em=em,
                    f1=f1,
                    graph_nodes=graph.number_of_nodes(),
                    compressed_nodes=subgraph.number_of_nodes(),
                    compression_ratio=subgraph.number_of_nodes() / max(graph.number_of_nodes(), 1),
                    latency_ms=latency,
                )
            )

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    header = list(asdict(out_records[0]).keys()) if out_records else []
    lines = [",".join(header)]
    for rec in out_records:
        lines.append(",".join(str(asdict(rec)[h]) for h in header))
    Path(output_csv).write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "num_rows": len(rows),
        "num_predictions": len(out_records),
        "em_mean": float(np.mean([r.em for r in out_records])) if out_records else 0.0,
        "f1_mean": float(np.mean([r.f1 for r in out_records])) if out_records else 0.0,
    }
    Path(output_csv).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

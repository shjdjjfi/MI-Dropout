from __future__ import annotations

import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import networkx as nx

from src.graph.schema import GraphEdge, GraphNode
from src.utils.io_utils import read_jsonl, write_json

LOGGER = logging.getLogger(__name__)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentence_chunk(text: str, max_sentences: int = 2) -> list[str]:
    sents = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    return [" ".join(sents[i : i + max_sentences]) for i in range(0, len(sents), max_sentences)]


def induce_graph(instance: dict[str, Any]) -> nx.DiGraph:
    graph = nx.DiGraph()
    question_id = f"q::{instance['id']}"
    graph.add_node(question_id, **asdict(GraphNode(question_id, instance["question"], "premise")))

    prev_nodes = [question_id]
    for doc_idx, doc in enumerate(instance.get("candidate_context", [])):
        chunks = sentence_chunk(doc.get("text", "")) or [doc.get("text", "")]
        for chunk_idx, chunk in enumerate(chunks):
            node_id = f"ev::{doc_idx}::{chunk_idx}"
            node = GraphNode(node_id=node_id, text=chunk, node_type="evidence", source_doc=doc.get("title"))
            graph.add_node(node_id, **asdict(node))
            for parent in prev_nodes[-2:]:
                edge = GraphEdge(src=parent, dst=node_id, edge_type="supports", weight=1.0)
                graph.add_edge(parent, node_id, **asdict(edge))

            interm_id = f"int::{doc_idx}::{chunk_idx}"
            interm_text = f"Given: {chunk[:120]}"
            graph.add_node(interm_id, **asdict(GraphNode(interm_id, interm_text, "intermediate")))
            graph.add_edge(node_id, interm_id, **asdict(GraphEdge(node_id, interm_id, "derives")))
            prev_nodes.append(interm_id)

    answer_id = f"ans::{instance['id']}"
    graph.add_node(answer_id, **asdict(GraphNode(answer_id, str(instance.get("gold_answer", "")), "answer")))
    for parent in prev_nodes[-3:]:
        graph.add_edge(parent, answer_id, **asdict(GraphEdge(parent, answer_id, "entails")))
    return graph


def graph_to_json(graph: nx.DiGraph) -> dict[str, Any]:
    return {
        "nodes": [{"id": n, **attrs} for n, attrs in graph.nodes(data=True)],
        "edges": [{"src": s, "dst": d, **attrs} for s, d, attrs in graph.edges(data=True)],
    }


def batch_induce_graphs(input_jsonl: str, output_dir: str) -> None:
    rows = read_jsonl(input_jsonl)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        graph = induce_graph(row)
        payload = graph_to_json(graph)
        write_json(payload, out / f"{row['id']}.json")
        nx.write_gpickle(graph, out / f"{row['id']}.gpickle")
    LOGGER.info("Wrote %d graphs to %s", len(rows), out)

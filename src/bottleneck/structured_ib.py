from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

import networkx as nx

MaskMode = Literal["node", "edge", "gumbel_topk"]


@dataclass(slots=True)
class IBConfig:
    lambda_size: float = 0.05
    lambda_entropy: float = 0.02
    lambda_redundancy: float = 0.02
    lambda_unsupported: float = 0.1
    keep_ratio: float = 0.4
    seed: int = 13


@dataclass(slots=True)
class IBResult:
    kept_nodes: set[str]
    kept_edges: set[tuple[str, str]]
    objective: float


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _node_score(attrs: dict) -> float:
    t = attrs.get("text", "")
    typ = attrs.get("node_type", "intermediate")
    base = min(len(t) / 120.0, 1.0)
    type_bonus = {"answer": 1.2, "intermediate": 0.9, "evidence": 0.8, "premise": 0.7}.get(typ, 0.7)
    return _sigmoid(base + type_bonus)


def apply_structured_ib(graph: nx.DiGraph, mode: MaskMode, config: IBConfig) -> IBResult:
    rng = random.Random(config.seed)
    node_scores = {n: _node_score(attrs) for n, attrs in graph.nodes(data=True)}

    if mode == "node":
        kept_nodes = {n for n, sc in node_scores.items() if rng.random() < sc * config.keep_ratio + 0.2}
    elif mode == "edge":
        kept_nodes = set(graph.nodes())
    else:
        k = max(1, int(len(node_scores) * config.keep_ratio))
        sorted_nodes = sorted(node_scores.items(), key=lambda kv: kv[1] + rng.random() * 0.01, reverse=True)
        kept_nodes = {n for n, _ in sorted_nodes[:k]}

    kept_edges = set()
    for u, v in graph.edges():
        if mode == "edge":
            if rng.random() < config.keep_ratio:
                kept_edges.add((u, v))
        elif u in kept_nodes and v in kept_nodes:
            kept_edges.add((u, v))

    size_pen = config.lambda_size * (len(kept_nodes) + len(kept_edges))
    entropy_pen = config.lambda_entropy * sum(-p * math.log(max(p, 1e-6)) for p in node_scores.values()) / max(len(node_scores), 1)
    redundancy_pen = config.lambda_redundancy * max(0, len(kept_edges) - len(kept_nodes) + 1)
    unsupported_nodes = [n for n in kept_nodes if graph.in_degree(n) == 0 and graph.nodes[n].get("node_type") == "intermediate"]
    unsupported_pen = config.lambda_unsupported * len(unsupported_nodes)

    answer_nodes = [n for n, a in graph.nodes(data=True) if a.get("node_type") == "answer"]
    sufficiency_reward = 1.0 if answer_nodes and answer_nodes[0] in kept_nodes else 0.0
    objective = -(1 - sufficiency_reward + size_pen + entropy_pen + redundancy_pen + unsupported_pen)
    return IBResult(kept_nodes=kept_nodes, kept_edges=kept_edges, objective=objective)


def extract_subgraph(graph: nx.DiGraph, result: IBResult) -> nx.DiGraph:
    g = nx.DiGraph()
    for n in result.kept_nodes:
        if n in graph:
            g.add_node(n, **graph.nodes[n])
    for u, v in result.kept_edges:
        if u in g and v in g and graph.has_edge(u, v):
            g.add_edge(u, v, **graph.edges[u, v])
    return g

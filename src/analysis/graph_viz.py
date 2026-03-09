from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


NODE_COLOR = {
    "premise": "#9ecae1",
    "evidence": "#a1d99b",
    "intermediate": "#fdae6b",
    "answer": "#fb6a4a",
}


def draw_graph(graph: nx.DiGraph, out_base: str) -> None:
    pos = nx.spring_layout(graph, seed=42)
    colors = [NODE_COLOR.get(graph.nodes[n].get("node_type", "intermediate"), "gray") for n in graph.nodes]
    labels = {n: f"{n}\n{graph.nodes[n].get('node_type','')}" for n in graph.nodes}

    plt.figure(figsize=(12, 8))
    nx.draw_networkx(graph, pos=pos, labels=labels, node_color=colors, font_size=7, arrows=True)
    Path(out_base).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{out_base}.png", dpi=220)
    plt.savefig(f"{out_base}.svg")
    plt.close()

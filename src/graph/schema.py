from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

NodeType = Literal["evidence", "premise", "intermediate", "answer"]
EdgeType = Literal["supports", "entails", "compares", "bridges", "derives"]


@dataclass(slots=True)
class GraphNode:
    node_id: str
    text: str
    node_type: NodeType
    source_doc: str | None = None
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphEdge:
    src: str
    dst: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

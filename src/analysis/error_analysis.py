from __future__ import annotations

from pathlib import Path

import pandas as pd

TAXONOMY = [
    "retrieval_failure",
    "missing_bridge_node",
    "wrong_merge",
    "unsupported_shortcut",
    "over_compression",
    "under_compression",
    "hallucinated_intermediate",
    "correct_but_unfaithful_graph",
]


def sample_error_sheet(eval_csv: str, out_csv: str, n: int = 100) -> None:
    df = pd.read_csv(eval_csv)
    err = df[df["em"] < 1.0].copy()
    sampled = err.sample(min(n, len(err)), random_state=42) if len(err) else err
    sampled["error_type"] = "retrieval_failure"
    sampled["notes"] = ""
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_csv, index=False)

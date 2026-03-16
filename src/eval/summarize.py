from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import ensure_dir, load_json


def summarize_results(experiment_rows: list[dict[str, Any]], out_csv: str | Path) -> None:
    out_path = Path(out_csv)
    ensure_dir(out_path.parent)
    df = pd.DataFrame(experiment_rows)
    df.to_csv(out_path, index=False)


def summarize_from_dirs(eval_dirs: list[str], out_csv: str | Path) -> None:
    rows = []
    for d in eval_dirs:
        m_path = Path(d) / "metrics.json"
        if not m_path.exists():
            continue
        metrics = load_json(m_path)
        rows.append(
            {
                "Experiment": Path(d).name,
                "Accuracy": metrics.get("accuracy", 0.0),
                "ExactMatch": metrics.get("exact_match", 0.0),
                "ValidAnswerRate": metrics.get("valid_answer_rate", 0.0),
                "AvgOutputLength": metrics.get("avg_output_length", 0.0),
                "EvalDir": d,
            }
        )
    summarize_results(rows, out_csv)

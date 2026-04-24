from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(obj: dict[str, Any], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def dump_json(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def dump_csv(df: pd.DataFrame, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)

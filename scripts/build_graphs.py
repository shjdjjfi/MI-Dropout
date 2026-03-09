from pathlib import Path

from src.graph.induction import batch_induce_graphs
from src.utils.logging_utils import setup_logging

if __name__ == "__main__":
    setup_logging()
    for f in Path("data/normalized").glob("*_validation.jsonl"):
        out = Path("data/graphs") / f.stem
        batch_induce_graphs(str(f), str(out))

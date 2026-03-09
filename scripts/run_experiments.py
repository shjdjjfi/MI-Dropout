from pathlib import Path

from src.evaluation.runner import evaluate_file
from src.utils.logging_utils import setup_logging

if __name__ == "__main__":
    setup_logging()
    for f in Path("data/normalized").glob("*_validation.jsonl"):
        output = Path("outputs/evals") / f"{f.stem}_eval.csv"
        evaluate_file(str(f), str(output), seed=13)

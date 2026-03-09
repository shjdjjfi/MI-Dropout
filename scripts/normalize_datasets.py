from src.datasets.loaders import build_all_datasets
from src.utils.logging_utils import setup_logging

if __name__ == "__main__":
    setup_logging()
    build_all_datasets(output_dir="data/normalized", splits=["train", "validation"], limit=200)

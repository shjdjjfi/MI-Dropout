from pathlib import Path

from src.analysis.error_analysis import sample_error_sheet
from src.analysis.plots import generate_plots

if __name__ == "__main__":
    for f in Path("outputs/evals").glob("*_eval.csv"):
        base = Path("paper_assets") / f.stem
        generate_plots(str(f), str(base / "figures"))
        sample_error_sheet(str(f), str(base / "appendix" / "error_sample.csv"))

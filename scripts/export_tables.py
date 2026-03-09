from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    out_dir = Path("paper_assets/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in Path("outputs/evals").glob("*_eval.csv"):
        df = pd.read_csv(f)
        agg = df.groupby("baseline", as_index=False)[["em", "f1", "compression_ratio", "latency_ms"]].mean()
        agg.to_csv(out_dir / f"{f.stem}_summary.csv", index=False)
        (out_dir / f"{f.stem}_summary.tex").write_text(agg.to_latex(index=False, float_format="%.4f"), encoding="utf-8")

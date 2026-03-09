# Structured Information Bottleneck for Tree/DAG Reasoning in LLMs

ACL-style experimental package for inducing reasoning DAGs, compressing them using a structured information bottleneck (IB), and evaluating multi-hop QA faithfulness/efficiency.

## Repository layout

- `src/datasets`: dataset normalization for HotpotQA, MuSiQue, 2WikiMultihopQA, StrategyQA, WikiHop, ProofWriter.
- `src/graph`: graph schema and DAG induction.
- `src/bottleneck`: structured IB objectives and masking variants.
- `src/prompting`: baseline prompting runners (direct, CoT, self-consistency, ToT, GoT, retrieve-read, oracle-support, concise-CoT).
- `src/evaluation`: unified metrics and experiment runner.
- `src/analysis`: plotting and error taxonomy support.
- `scripts`: end-to-end command entrypoints.
- `configs`: dataset/model/experiment configs.
- `paper_assets`: generated figures/tables/appendix artifacts.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate structured-ib-reasoning
```

## Quickstart commands

1) Normalize datasets (cached JSONL):
```bash
PYTHONPATH=. python scripts/normalize_datasets.py
```

2) Build induced reasoning graphs:
```bash
PYTHONPATH=. python scripts/build_graphs.py
```

3) Run baseline + structured IB evaluation:
```bash
PYTHONPATH=. python scripts/run_experiments.py
```

4) Create paper assets (plots and error sheets):
```bash
PYTHONPATH=. python scripts/analyze_results.py
```

5) Run unit tests:
```bash
PYTHONPATH=. pytest -q
```

## Structured IB objective

For full graph `G`, answer target `Y`, and selected subgraph `Z`:

`L = answer_loss + λ1*graph_size + λ2*edge_entropy + λ3*redundancy_penalty + λ4*unsupported_node_penalty`

Implemented bottleneck variants:
- Node-level Bernoulli masking (`mode=node`)
- Edge-level masking (`mode=edge`)
- Sparse subgraph extraction via top-k/Gumbel-like relaxation (`mode=gumbel_topk`)

## Notes for full-scale ACL runs

- Replace `DummyBackend` with vLLM/transformers backend in `src/prompting/model_backend.py`.
- Use `configs/models/open_models.yaml` for target model families.
- Execute 3+ seeds and report significance with bootstrap/randomization tests.
- Produce CSV + LaTeX tables in `outputs/` and `paper_assets/tables/`.

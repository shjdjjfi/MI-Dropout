# Synthetic Watermark Phase-Transition Experiments

This repository implements a fully synthetic, reproducible platform for studying phase transitions in watermark detectability and removability, including empirical statistical–computational gap regions.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run full pipeline

```bash
python scripts/run_phase_detectability.py --config configs/sweeps/detectability.yaml
python scripts/run_phase_removability.py --config configs/sweeps/removability.yaml
python scripts/aggregate_results.py
python scripts/plot_all.py
python scripts/generate_report.py
```

Outputs are written to:
- `results/raw`
- `results/processed`
- `results/figures`
- `report/synthetic_phase_transition_report.md`

## Structure

- `src/sources`: IID / Markov / HMM synthetic sources
- `src/watermark`: KGW red-green watermark
- `src/detectors`: LLR and green-z detectors
- `src/attackers`: exact, greedy, beam, and MCMC attackers
- `src/metrics`: detection and quality metrics
- `src/experiments`: sweep runners
- `src/plots`: publication-oriented figure generation
- `scripts`: executable experiment and reporting scripts
- `configs`: source and sweep configuration

## Download page for ZIP

To expose a simple web page for downloading `synthetic_phase_transition_package.zip`:

```bash
./scripts/serve_download_page.sh
```

Then open:
- `http://<host>:8000/web/download.html`
- direct file URL: `http://<host>:8000/synthetic_phase_transition_package.zip`

### QR code for mobile download

Generate QR for your actual reachable host (important):

```bash
python scripts/generate_download_qr.py --url "http://<host>:8000/synthetic_phase_transition_package.zip"
```

QR image path: `web/zip_download_qr.png` and it is embedded in `web/download.html`.

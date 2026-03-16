# MedQA Benchmark-Leakage CoT Distillation Experiment

This repository provides a **fully reproducible, benchmark-specific leakage experiment** for MedQA-USMLE:

> **Hypothesis**: If a stronger teacher LLM generates rationales (CoT) and answers for the full MedQA test benchmark, then a smaller student model fine-tuned on this synthetic supervision will improve substantially **on that same benchmark**.

⚠️ This setup intentionally uses leakage-style supervision from the benchmark itself. It is **not** evidence of out-of-distribution or true generalization.

## Project layout

```text
project/
  configs/
  src/
    data/
    teacher/
    sft/
    training/
    eval/
    utils/
  scripts/
  outputs/
  README.md
  requirements.txt
```

## Core stages

- **Stage A**: Preprocess MedQA-USMLE from HF datasets into canonical schema.
- **Stage B**: Generate teacher CoT + answer (or answer-only control).
- **Stage C**: Build synthetic SFT training data variants.
- **Stage D**: Fine-tune student model (full / LoRA / QLoRA).
- **Stage E**: Evaluate student on same MedQA test split before and after fine-tuning.
- **Stage F**: Run controls and ablations.

## Canonical data schema

Each example is normalized to:

```json
{
  "id": "string",
  "question": "string",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer": "A"
}
```

Preprocessing logs column mappings and split stats in `outputs/canonical_medqa/stats.json`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart commands

### 1) preprocess dataset
```bash
python -m src.main --config configs/default.yaml preprocess
```

### 2) generate teacher CoT
```bash
python -m src.main --config configs/default.yaml teacher-generate
```

### 3) build synthetic SFT data
```bash
python -m src.main --config configs/default.yaml build-sft
```

### 4) finetune student
```bash
python -m src.main --config configs/default.yaml finetune
```

### 5) evaluate student
```bash
python -m src.main --config configs/default.yaml evaluate --model-path <MODEL_OR_CHECKPOINT> --tag <RUN_TAG>
```

### 6) run ablations
```bash
python -m src.main --config configs/default.yaml run-ablations
```

### 7) summarize to CSV
```bash
python -m src.main --config configs/default.yaml summarize
```

## Teacher output format

Expected strict format:

```xml
<reasoning>
...
</reasoning>
<answer>
B
</answer>
```

Parser retries on failure, stores raw and parsed files, and marks invalid rows.

## Synthetic SFT variants

Implemented in `src/sft/build_data.py`:

1. `cot_answer`
2. `answer_only`
3. `short_cot_answer`
4. filtered correct-only CoT (`filtered_only_correct=true`)

Extra controls:
- shuffled CoT (keep teacher answer or keep gold answer)
- wrong-only CoT
- rationale truncation
- sample efficiency fractions (10/25/50/100)

## Evaluation outputs

For each eval run:
- `metrics.json` with accuracy / EM / valid answer rate / avg output length
- `predictions.jsonl` per-example outputs
- `confusion_matrix.png`
- `breakdown_by_question_length.json`

## Model swapping

Edit `configs/default.yaml`:
- `teacher.backend`: `hf_local` or `api`
- `teacher.model_name` or `teacher.api_base_url` + `teacher.api_model`
- `student.model_name`
- training and LoRA settings under `training`

Use `configs/single_gpu_light.yaml` for lightweight single-GPU defaults.

## Reproducibility notes

- Global seed set via `seed_everything`.
- Deterministic teacher generation defaults to `temperature: 0.0`.
- Resumable teacher generation by reusing existing parsed outputs.

## What this does **not** claim

This experiment does **not** demonstrate robust medical reasoning generalization. It only tests benchmark-specific gain under direct leakage supervision.

---

## Pre-launch edit checklist

- [ ] Choose dataset ID and confirm split names in config.
- [ ] Choose teacher backend (`hf_local` or `api`) and fill endpoint/API key env var.
- [ ] Set teacher/student model names available in your environment.
- [ ] Adjust training method (`full`, `lora`, `qlora`) and batch/precision settings.
- [ ] Ensure enough GPU memory for selected models.
- [ ] Run ablations and summary stage.

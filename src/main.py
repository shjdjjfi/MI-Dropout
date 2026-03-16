from __future__ import annotations

import argparse
from pathlib import Path

from src.data.preprocess import MedQAPreprocessor
from src.eval.evaluate import evaluate_model
from src.eval.summarize import summarize_from_dirs
from src.sft.ablations import run_ablation_data_builds
from src.sft.build_data import build_sft_dataset
from src.teacher.generate import generate_teacher_outputs
from src.training.finetune import finetune_student
from src.utils.config import load_config
from src.utils.io import ensure_dir
from src.utils.logging_utils import setup_logger
from src.utils.repro import seed_everything

logger = setup_logger(__name__)


def cmd_preprocess(cfg: dict) -> None:
    dataset_cfg = cfg["dataset"]
    out_dir = Path(cfg["output_root"]) / "canonical_medqa"
    pre = MedQAPreprocessor(dataset_cfg["hf_id"], dataset_cfg.get("fallback_hf_id"), dataset_cfg.get("subset"))
    pre.run(
        out_dir=out_dir,
        split_mapping=dataset_cfg["split_mapping"],
        max_samples_per_split=dataset_cfg.get("max_samples_per_split"),
    )


def cmd_teacher_generate(cfg: dict) -> None:
    canonical_test = Path(cfg["output_root"]) / "canonical_medqa" / "test.jsonl"
    out_dir = Path(cfg["output_root"]) / "teacher" / cfg["teacher"]["mode"]
    generate_teacher_outputs(canonical_test, out_dir, cfg["teacher"])


def cmd_build_sft(cfg: dict) -> None:
    canonical_test = Path(cfg["output_root"]) / "canonical_medqa" / "test.jsonl"
    parsed = Path(cfg["output_root"]) / "teacher" / cfg["teacher"]["mode"] / "teacher_parsed.jsonl"
    out_path = Path(cfg["output_root"]) / "sft" / f"{cfg['sft']['variant']}.jsonl"
    build_sft_dataset(
        canonical_test,
        parsed,
        out_path,
        variant=cfg["sft"]["variant"],
        short_rationale_tokens=cfg["sft"]["short_rationale_tokens"],
        filtered_only_correct=cfg["sft"].get("filtered_only_correct", False),
        shuffled_cot=cfg["sft"].get("shuffled_cot", False),
        shuffled_answer_protocol=cfg["sft"].get("shuffled_answer_protocol", "keep_teacher"),
        rationale_truncation_tokens=cfg["sft"].get("rationale_truncation_tokens"),
        seed=cfg["seed"],
    )


def cmd_finetune(cfg: dict, sft_file: str | None = None, out_dir_override: str | None = None) -> str:
    sft_jsonl = sft_file or str(Path(cfg["output_root"]) / "sft" / f"{cfg['sft']['variant']}.jsonl")
    train_cfg = dict(cfg["training"])
    if out_dir_override:
        train_cfg["output_dir"] = out_dir_override
    return finetune_student(sft_jsonl=sft_jsonl, student_cfg=cfg["student"], train_cfg=train_cfg, seed=cfg["seed"])


def cmd_evaluate(cfg: dict, model_path: str, tag: str) -> None:
    test_jsonl = Path(cfg["output_root"]) / "canonical_medqa" / "test.jsonl"
    out_dir = Path(cfg["output_root"]) / cfg["eval"]["output_subdir"] / tag
    evaluate_model(
        model_path=model_path,
        test_jsonl=test_jsonl,
        out_dir=out_dir,
        max_new_tokens=cfg["eval"]["max_new_tokens"],
        temperature=cfg["eval"]["temperature"],
    )


def cmd_run_ablations(cfg: dict) -> None:
    canonical_test = Path(cfg["output_root"]) / "canonical_medqa" / "test.jsonl"
    parsed = Path(cfg["output_root"]) / "teacher" / cfg["teacher"]["mode"] / "teacher_parsed.jsonl"
    ablation_dir = Path(cfg["output_root"]) / "sft" / "ablations"
    sft_files = run_ablation_data_builds(canonical_test, parsed, ablation_dir, cfg.get("ablations", {}), seed=cfg["seed"])

    # Baseline before SFT
    cmd_evaluate(cfg, cfg["student"]["model_name"], "baseline_zero_shot")

    for sft_file in sft_files:
        tag = Path(sft_file).stem
        ckpt_dir = Path(cfg["output_root"]) / "checkpoints" / tag
        model_path = cmd_finetune(cfg, sft_file=sft_file, out_dir_override=str(ckpt_dir))
        cmd_evaluate(cfg, model_path, f"post_{tag}")


def cmd_summarize(cfg: dict) -> None:
    eval_root = Path(cfg["output_root"]) / cfg["eval"]["output_subdir"]
    dirs = [str(p) for p in eval_root.iterdir() if p.is_dir()] if eval_root.exists() else []
    summarize_from_dirs(dirs, Path(cfg["output_root"]) / "summary" / "results.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MedQA leakage experiment pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    for c in ["preprocess", "teacher-generate", "build-sft", "finetune", "run-ablations", "summarize"]:
        sub.add_parser(c)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--model-path", default=None)
    eval_parser.add_argument("--tag", default="manual_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dir(cfg["output_root"])
    seed_everything(cfg["seed"])

    if args.command == "preprocess":
        cmd_preprocess(cfg)
    elif args.command == "teacher-generate":
        cmd_teacher_generate(cfg)
    elif args.command == "build-sft":
        cmd_build_sft(cfg)
    elif args.command == "finetune":
        cmd_finetune(cfg)
    elif args.command == "evaluate":
        model_path = args.model_path or cfg["student"]["model_name"]
        cmd_evaluate(cfg, model_path, args.tag)
    elif args.command == "run-ablations":
        cmd_run_ablations(cfg)
    elif args.command == "summarize":
        cmd_summarize(cfg)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.utils.io import ensure_dir, read_jsonl
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def _messages_to_text(row: dict[str, Any]) -> str:
    return row["messages"][0]["content"] + "\n\n" + row["messages"][1]["content"]


def finetune_student(sft_jsonl: str | Path, student_cfg: dict[str, Any], train_cfg: dict[str, Any], seed: int) -> str:
    rows = read_jsonl(sft_jsonl)
    text_rows = [{"text": _messages_to_text(r)} for r in rows]

    dev_ratio = train_cfg.get("dev_ratio", 0.1)
    n_dev = int(len(text_rows) * dev_ratio)
    train_rows = text_rows[n_dev:]
    dev_rows = text_rows[:n_dev] if n_dev > 0 else text_rows[:1]

    tokenizer = AutoTokenizer.from_pretrained(student_cfg["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if student_cfg.get("load_in_4bit", False) and train_cfg["method"] in {"lora", "qlora"}:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        student_cfg["model_name"],
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if train_cfg.get("bf16", False) else torch.float16,
        device_map="auto",
    )

    if train_cfg["method"] in {"lora", "qlora"}:
        if train_cfg["method"] == "qlora":
            model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=train_cfg["lora_r"],
            lora_alpha=train_cfg["lora_alpha"],
            lora_dropout=train_cfg["lora_dropout"],
            target_modules=train_cfg["target_modules"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    max_len = student_cfg.get("max_length", 1024)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    train_ds = Dataset.from_list(train_rows).map(tokenize, batched=True, remove_columns=["text"])
    dev_ds = Dataset.from_list(dev_rows).map(tokenize, batched=True, remove_columns=["text"])

    output_dir = ensure_dir(train_cfg["output_dir"])
    args = TrainingArguments(
        output_dir=str(output_dir),
        seed=seed,
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=train_cfg.get("bf16", False),
        fp16=train_cfg.get("fp16", False),
        optim=train_cfg.get("optimizer", "adamw_torch"),
        report_to=[],
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    logger.info("Saved model to %s", output_dir / "final")
    return str(output_dir / "final")

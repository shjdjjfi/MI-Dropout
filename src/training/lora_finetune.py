from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


def train_lora_stub(model_name: str, dataset_path: str, output_dir: str, config: LoRAConfig) -> None:
    """Concrete placeholder function for integration with PEFT/Transformers training scripts."""
    print(
        f"LoRA training requested for {model_name} on {dataset_path}. "
        f"Output: {output_dir}. Config: {config}"
    )

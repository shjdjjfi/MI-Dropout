from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0


class TeacherClient:
    def generate(self, prompt: str, gen_cfg: GenerationConfig) -> str:
        raise NotImplementedError


class HFLocalTeacherClient(TeacherClient):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    def generate(self, prompt: str, gen_cfg: GenerationConfig) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            do_sample=gen_cfg.temperature > 0,
            temperature=max(gen_cfg.temperature, 1e-6),
            top_p=gen_cfg.top_p,
            max_new_tokens=gen_cfg.max_new_tokens,
        )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


class APITeacherClient(TeacherClient):
    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = os.getenv(api_key_env, "")

    def generate(self, prompt: str, gen_cfg: GenerationConfig) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": gen_cfg.temperature,
            "top_p": gen_cfg.top_p,
            "max_tokens": gen_cfg.max_new_tokens,
        }
        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def build_teacher_client(cfg: dict[str, Any]) -> TeacherClient:
    backend = cfg["backend"]
    if backend == "hf_local":
        return HFLocalTeacherClient(model_name=cfg["model_name"])
    if backend == "api":
        return APITeacherClient(base_url=cfg["api_base_url"], model=cfg["api_model"], api_key_env=cfg["api_key_env"])
    raise ValueError(f"Unsupported teacher backend: {backend}")

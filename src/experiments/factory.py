from __future__ import annotations

from sources.hmm_source import HMMSource
from sources.iid_source import IIDCategoricalSource
from sources.markov_source import FirstOrderMarkovSource


def build_source(cfg: dict):
    typ = cfg["type"]
    if typ == "iid":
        if cfg.get("dist", "zipf") == "zipf":
            return IIDCategoricalSource.from_zipf(cfg["vocab_size"], cfg.get("zipf_alpha", 1.1), cfg.get("temperature", 1.0))
        return IIDCategoricalSource.from_random_softmax(cfg["vocab_size"], cfg.get("seed", 0), cfg.get("scale", 1.0))
    if typ == "markov":
        return FirstOrderMarkovSource.random(
            vocab_size=cfg["vocab_size"],
            seed=cfg.get("seed", 0),
            sparsity=cfg.get("sparsity", 0.2),
            temperature=cfg.get("temperature", 1.0),
            regime=cfg.get("regime", "medium"),
        )
    if typ == "hmm":
        return HMMSource.random(
            vocab_size=cfg["vocab_size"],
            num_states=cfg.get("num_states", 4),
            seed=cfg.get("seed", 0),
            trans_temp=cfg.get("trans_temp", 1.0),
            emission_temp=cfg.get("emission_temp", 1.0),
        )
    raise ValueError(f"unknown source type: {typ}")

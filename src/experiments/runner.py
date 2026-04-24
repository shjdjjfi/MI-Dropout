from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from attackers.beam import BeamSearchAttacker
from attackers.exact import ExactSearchAttacker
from attackers.greedy import GreedyLocalAttacker
from attackers.mcmc import MCMCRandomWalkAttacker
from detectors.green_z import GreenZDetector
from detectors.llr import LLRDetector
from experiments.factory import build_source
from metrics.detection import error_rates, roc_auc
from watermark.kgw import KGWWatermarker


@dataclass
class RunnerConfig:
    seed: int = 0


def make_attacker(name: str, cfg: dict):
    if name == "greedy":
        return GreedyLocalAttacker(max_edit_ratio=cfg.get("max_edit_ratio", 0.05), candidate_subset=cfg.get("candidate_subset", 8))
    if name == "beam":
        return BeamSearchAttacker(beam_width=cfg.get("beam_width", 8), max_edit_depth=cfg.get("max_edit_depth", 8))
    if name == "mcmc":
        return MCMCRandomWalkAttacker(steps=cfg.get("steps", 200), beta=cfg.get("beta", 10.0))
    if name == "exact":
        return ExactSearchAttacker(max_edits=cfg.get("max_edits", 2), max_states=cfg.get("max_states", 200000))
    raise ValueError(name)


def run_detectability_sweep(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, per_seq = [], []
    for source_cfg in cfg["sources"]:
        source = build_source(source_cfg)
        for n in tqdm(cfg["sweep"]["n"], desc=f"detect-{source_cfg['name']}"):
            for delta in cfg["sweep"]["delta"]:
                for gamma in cfg["sweep"]["gamma"]:
                    llr = LLRDetector()
                    wm = KGWWatermarker(source.vocab_size, gamma=gamma, delta=delta, key=cfg["watermark"]["key"], context_dependent=cfg["watermark"].get("context_dependent", True))
                    zdet = GreenZDetector(gamma=gamma)
                    neg, pos, zneg, zpos = [], [], [], []
                    rng = np.random.default_rng(cfg["seed"] + n + int(100 * delta) + int(10 * gamma))
                    for i in range(cfg["sweep"]["m"]):
                        clean = source.sample_sequence(n, rng)
                        wat = wm.sample_watermarked(source, n, rng)
                        s0 = llr.score(clean, source, wm)
                        s1 = llr.score(wat, source, wm)
                        neg.append(s0)
                        pos.append(s1)
                        zneg.append(zdet.score(clean, wm)["z"])
                        zpos.append(zdet.score(wat, wm)["z"])
                        per_seq.append({"source": source_cfg["name"], "n": n, "delta": delta, "gamma": gamma, "seq_idx": i, "llr_clean": s0, "llr_wm": s1, "z_clean": zneg[-1], "z_wm": zpos[-1]})
                    neg = np.array(neg); pos = np.array(pos); zneg = np.array(zneg); zpos = np.array(zpos)
                    e_llr = error_rates(pos, neg, alpha=cfg["alpha"])
                    e_z = error_rates(zpos, zneg, alpha=cfg["alpha"])
                    rows.append({
                        "source": source_cfg["name"], "n": n, "delta": delta, "gamma": gamma,
                        "auc_llr": roc_auc(pos, neg), "auc_z": roc_auc(zpos, zneg),
                        "type1_llr": e_llr["type1"], "type2_llr": e_llr["type2"], "power_llr": e_llr["power"],
                        "type1_z": e_z["type1"], "type2_z": e_z["type2"], "power_z": e_z["power"],
                    })
    return pd.DataFrame(rows), pd.DataFrame(per_seq)


def run_removability_sweep(cfg: dict) -> pd.DataFrame:
    rows = []
    llr = LLRDetector()
    for source_cfg in cfg["sources"]:
        source = build_source(source_cfg)
        for delta in tqdm(cfg["sweep"]["delta"], desc=f"remove-{source_cfg['name']}"):
            for budget in cfg["sweep"]["budget"]:
                for eps in cfg["sweep"]["epsilon"]:
                    wm = KGWWatermarker(source.vocab_size, gamma=cfg["gamma"], delta=delta, key=cfg["watermark"]["key"], context_dependent=cfg["watermark"].get("context_dependent", True))
                    for attacker_name in cfg["attackers"]:
                        if attacker_name == "exact" and (
                            cfg["n"] > cfg.get("exact_max_n", 20)
                            or budget > cfg.get("exact_max_budget", 0.05)
                        ):
                            continue
                        succ = 0
                        score_drop = []
                        edits = []
                        runtime = []
                        for seed in cfg["seeds"]:
                            rng = np.random.default_rng(seed + int(delta * 100) + int(1000 * budget))
                            n = cfg["n"]
                            x = wm.sample_watermarked(source, n, rng)
                            neg_samples = np.array([llr.score(source.sample_sequence(n, rng), source, wm) for _ in range(128)])
                            thr = np.quantile(neg_samples, 1 - cfg["alpha"])
                            acfg = dict(cfg["attacker_cfg"].get(attacker_name, {}))
                            if attacker_name == "greedy":
                                acfg["max_edit_ratio"] = budget
                            if attacker_name == "beam":
                                acfg["max_edit_depth"] = max(1, int(np.ceil(n * budget)))
                            if attacker_name == "exact":
                                acfg["max_edits"] = max(1, int(np.ceil(n * budget)))
                            if attacker_name == "mcmc":
                                acfg["steps"] = max(20, int(200 * budget / max(cfg["sweep"]["budget"])))
                            attacker = make_attacker(attacker_name, acfg)
                            t0 = perf_counter()
                            res = attacker.attack(x, source, llr, wm, threshold=thr, epsilon=eps, quality_mode=cfg.get("quality_mode", "global"))
                            runtime.append(perf_counter() - t0 + res.runtime_sec)
                            succ += int(res.success)
                            score_drop.append(res.score_before - res.score_after)
                            edits.append(res.edits / n)
                        rows.append({
                            "source": source_cfg["name"], "delta": delta, "budget": budget, "epsilon": eps, "attacker": attacker_name,
                            "success_rate": succ / len(cfg["seeds"]), "avg_score_drop": float(np.mean(score_drop)),
                            "avg_edit_ratio": float(np.mean(edits)), "avg_runtime": float(np.mean(runtime)),
                        })
    return pd.DataFrame(rows)

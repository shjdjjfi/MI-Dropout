from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLRDetector:
    def score(self, seq, source, watermarker) -> float:
        return float(watermarker.log_prob_under_q(source, seq) - source.log_prob_sequence(seq))

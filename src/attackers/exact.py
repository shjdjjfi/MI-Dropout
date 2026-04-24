from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from attackers.common import AttackResult, attack_template, quality_ok


@dataclass
class ExactSearchAttacker:
    max_edits: int = 2
    max_states: int = 200000

    @attack_template
    def attack(self, seq, source, detector, watermarker, threshold: float, epsilon: float, quality_mode: str = "global"):
        n = len(seq)
        vocab = watermarker.vocab_size
        base_score = detector.score(seq, source, watermarker)
        best_seq = seq.copy()
        best_score = base_score
        states = 0
        for e in range(1, self.max_edits + 1):
            for idxs in combinations(range(n), e):
                for vals in product(range(vocab), repeat=e):
                    if any(seq[i] == v for i, v in zip(idxs, vals)):
                        continue
                    cand = seq.copy()
                    for i, v in zip(idxs, vals):
                        cand[i] = v
                    states += 1
                    if states > self.max_states:
                        break
                    if not quality_ok(source, seq, cand, epsilon, quality_mode):
                        continue
                    sc = detector.score(cand, source, watermarker)
                    if sc < best_score:
                        best_score = sc
                        best_seq = cand
                if states > self.max_states:
                    break
            if states > self.max_states:
                break
        edits = int((best_seq != seq).sum())
        return AttackResult(best_seq, best_score <= threshold, base_score, best_score, edits, float(-source.log_prob_sequence(best_seq)/n + source.log_prob_sequence(seq)/n), 0.0, states, approximate=(states > self.max_states))

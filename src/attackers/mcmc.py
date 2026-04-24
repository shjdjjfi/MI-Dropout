from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attackers.common import AttackResult, attack_template, quality_ok


@dataclass
class MCMCRandomWalkAttacker:
    steps: int = 200
    beta: float = 10.0

    @attack_template
    def attack(self, seq, source, detector, watermarker, threshold: float, epsilon: float, quality_mode: str = "global", seed: int = 0):
        rng = np.random.default_rng(seed)
        cur = seq.copy()
        n = len(seq)
        vocab = watermarker.vocab_size
        cur_score = detector.score(cur, source, watermarker)
        best, best_score = cur.copy(), cur_score
        accepts = 0
        for _ in range(self.steps):
            t = int(rng.integers(0, n))
            tok = int(rng.integers(0, vocab))
            if tok == cur[t]:
                continue
            prop = cur.copy()
            prop[t] = tok
            if not quality_ok(source, seq, prop, epsilon, quality_mode):
                continue
            sc = detector.score(prop, source, watermarker)
            if sc < cur_score or rng.random() < np.exp(-self.beta * (sc - cur_score)):
                cur, cur_score = prop, sc
                accepts += 1
                if sc < best_score:
                    best, best_score = prop.copy(), sc
            if best_score <= threshold:
                break
        edits = int((best != seq).sum())
        return AttackResult(best, best_score <= threshold, detector.score(seq, source, watermarker), best_score, edits, float(-source.log_prob_sequence(best)/n + source.log_prob_sequence(seq)/n), 0.0, self.steps)

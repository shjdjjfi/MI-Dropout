from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attackers.common import AttackResult, attack_template, quality_ok


@dataclass
class BeamSearchAttacker:
    beam_width: int = 8
    max_edit_depth: int = 8
    candidate_subset: int = 8
    position_subset: int = 8

    @attack_template
    def attack(self, seq, source, detector, watermarker, threshold: float, epsilon: float, quality_mode: str = "global"):
        vocab = watermarker.vocab_size
        n = len(seq)
        start_score = detector.score(seq, source, watermarker)
        beam = [(seq.copy(), start_score)]
        best_seq, best_score = seq.copy(), start_score
        steps = 0
        for _ in range(self.max_edit_depth):
            cand_pool = []
            for x, _ in beam:
                for t in range(min(n, self.position_subset)):
                    for tok in range(min(vocab, self.candidate_subset)):
                        if tok == x[t]:
                            continue
                        y = x.copy()
                        y[t] = tok
                        steps += 1
                        if not quality_ok(source, seq, y, epsilon, quality_mode):
                            continue
                        sc = detector.score(y, source, watermarker)
                        pen = float((y != seq).sum()) / n
                        obj = sc + 0.05 * pen
                        cand_pool.append((y, sc, obj))
                        if sc < best_score:
                            best_score, best_seq = sc, y
            if not cand_pool:
                break
            cand_pool.sort(key=lambda z: z[2])
            beam = [(y, sc) for y, sc, _ in cand_pool[: self.beam_width]]
            if best_score <= threshold:
                break
        edits = int((best_seq != seq).sum())
        return AttackResult(best_seq, best_score <= threshold, start_score, best_score, edits, float(-source.log_prob_sequence(best_seq)/n + source.log_prob_sequence(seq)/n), 0.0, steps)

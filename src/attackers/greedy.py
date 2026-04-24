from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attackers.common import AttackResult, attack_template, quality_ok


@dataclass
class GreedyLocalAttacker:
    max_edit_ratio: float = 0.05
    candidate_subset: int = 8
    position_subset: int = 16

    @attack_template
    def attack(self, seq, source, detector, watermarker, threshold: float, epsilon: float, quality_mode: str = "global"):
        cur = seq.copy()
        n = len(cur)
        max_edits = max(1, int(np.floor(self.max_edit_ratio * n)))
        before = detector.score(cur, source, watermarker)
        vocab = watermarker.vocab_size
        edits = 0
        steps = 0
        for _ in range(max_edits):
            best = None
            best_score = detector.score(cur, source, watermarker)
            for t in range(min(n, self.position_subset)):
                orig = cur[t]
                cands = np.arange(vocab)
                if self.candidate_subset < vocab:
                    cands = cands[: self.candidate_subset]
                for tok in cands:
                    if tok == orig:
                        continue
                    proposal = cur.copy()
                    proposal[t] = tok
                    steps += 1
                    if not quality_ok(source, seq, proposal, epsilon, quality_mode):
                        continue
                    sc = detector.score(proposal, source, watermarker)
                    if sc < best_score:
                        best_score = sc
                        best = (t, tok)
            if best is None:
                break
            cur[best[0]] = best[1]
            edits += 1
            if detector.score(cur, source, watermarker) <= threshold:
                break
        after = detector.score(cur, source, watermarker)
        return AttackResult(cur, after <= threshold, before, after, edits, float(-source.log_prob_sequence(cur)/n + source.log_prob_sequence(seq)/n), 0.0, steps)

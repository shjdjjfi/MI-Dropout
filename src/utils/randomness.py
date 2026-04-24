from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def stable_hash_int(items: Iterable[int | str], key: int = 0, mod: int = 2**63 - 1) -> int:
    h = hashlib.blake2b(digest_size=16, person=str(key).encode("utf-8"))
    for item in items:
        h.update(str(item).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "big") % mod

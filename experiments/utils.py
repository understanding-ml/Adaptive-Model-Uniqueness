
import hashlib

import hashlib

import hashlib
from typing import Optional

def make_stable_seed(
    text: str,
    theta: float,
    context_type: Optional[str] = None,
    ctx_label: Optional[str] = None,
    base_seed: int = 2026,
) -> int:
    """
    Construct a deterministic 32-bit seed from (text, theta) and,
    optionally, semantic context fields.

    If context_type / ctx_label are not None, they are included in the key.
    This allows:
      - Exp4 / Exp5: make_stable_seed(text, theta)
      - Exp6 / Exp7: make_stable_seed(text, theta, context_type=..., ctx_label=...)

    We avoid Python's built-in `hash`, which is salted and not stable
    across processes, by using a fixed MD5-based hash instead.
    """
    parts = [str(base_seed)]

    if context_type is not None:
        parts.append(str(context_type))
    if ctx_label is not None:
        parts.append(str(ctx_label))

    # Always include theta and text
    parts.append(f"{theta:.6f}")
    parts.append(text)

    key = "|".join(parts)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    seed_32 = int(digest[:8], 16) % (2**32)
    return seed_32


"""Model-visible reward controls; task scores remain unchanged for evaluation."""

import random

POLARITIES = ("normal", "flipped", "random", "empty")


def shown_reward(task: str, reward: int, polarity: str = "normal", *,
                 step: int | None = None, status: str = "ok",
                 random_seed: int = 0, sample_id: int = 0, round_idx: int = 0) -> int | None:
    """Transform valid feedback while preserving true scores and failure handling."""
    if polarity not in POLARITIES:
        raise ValueError(f"Unknown reward polarity: {polarity}")
    if polarity == "empty":
        return None
    if polarity == "normal" or status not in ("ok", "seed_round"):
        return reward
    # Key each draw by its position, independent of the true score and call order.
    # This keeps random feedback identical across languages and after a resume.
    rng = random.Random(f"{random_seed}:{task}:{sample_id}:{round_idx}:{step}") if polarity == "random" else None
    if task == "game24":
        if step is None:
            raise ValueError("Game24 reward transformation requires a step number")
        if step not in (1, 2, 3, 4):
            raise ValueError(f"Invalid Game24 step: {step}")
        if rng is not None:
            return rng.choice((0, 1, 3) if step < 4 else (0, 10))
        return (3 if step < 4 else 10) - reward
    if task == "creative_writing":
        return rng.randint(1, 10) if rng is not None else 11 - reward
    if task == "sciworld":
        return rng.randint(-100, 100) if rng is not None else -reward
    raise ValueError(f"Unknown task: {task}")

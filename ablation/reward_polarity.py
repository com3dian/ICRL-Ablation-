"""Model-visible reward polarity; task scores remain unchanged for evaluation."""

POLARITIES = ("normal", "flipped")


def shown_reward(task: str, reward: int, polarity: str = "normal", *,
                 step: int | None = None, status: str = "ok") -> int:
    """Reverse a valid task score while leaving failed evaluations at zero."""
    if polarity not in POLARITIES:
        raise ValueError(f"Unknown reward polarity: {polarity}")
    if polarity == "normal" or status not in ("ok", "seed_round"):
        return reward
    if task == "game24":
        if step is None:
            raise ValueError("Game24 reward inversion requires a step number")
        if step not in (1, 2, 3, 4):
            raise ValueError(f"Invalid Game24 step: {step}")
        return (3 if step < 4 else 10) - reward
    if task == "creative_writing":
        return 11 - reward
    if task == "sciworld":
        return -reward
    raise ValueError(f"Unknown task: {task}")

"""Plot original numeric rewards, grouped by round, from one local run."""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def plot_run(run_dir: Path) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", str(run_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    rewards = defaultdict(list)
    for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") == "attempt_end" and event.get("round", -1) >= 0:
            rewards[int(event["round"])].append(float(event["reward"]))
    if not rewards:
        raise ValueError(f"No completed rounds in {run_dir}")

    rounds = sorted(rewards)
    means = [sum(rewards[r]) / len(rewards[r]) for r in rounds]
    best = []
    current = float("-inf")
    for mean in means:
        current = max(current, mean)
        best.append(current)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rounds, means, marker="o", label="Mean reward per round")
    ax.plot(rounds, best, linestyle="--", label="Best mean so far")
    ax.set(xlabel="Round (zero based)", ylabel="Numeric reward", title=f"{config['task']} · {config['model_alias']} · {config['feedback']}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output = run_dir / "reward.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot a completed or ongoing ablation run")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(plot_run(args.run_dir.resolve()))


if __name__ == "__main__":
    main()

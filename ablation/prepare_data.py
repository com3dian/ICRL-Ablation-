"""Fetch and validate the two public Tree of Thoughts inputs used by this ablation."""

import csv
import io
import re
from pathlib import Path
from urllib.request import Request, urlopen

DATA_ROOT = Path(__file__).resolve().parent / "data"
SOURCES = {
    "24.csv": "https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/master/src/tot/data/24/24.csv",
    "data_100_random_text.txt": "https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/master/src/tot/data/text/data_100_random_text.txt",
}


def _validate(name: str, payload: bytes) -> int:
    text = payload.decode("utf-8-sig")
    if name == "24.csv":
        count = 0
        for row in csv.reader(io.StringIO(text)):
            if any(
                len(numbers := field.strip().split()) == 4
                and all(re.fullmatch(r"\d+", number) for number in numbers)
                for field in row
            ):
                count += 1
        if count < 1000:
            raise ValueError(f"Game24 file has only {count} puzzles; at least 1000 are needed for index 900")
    else:
        count = sum(bool(line.strip()) for line in text.splitlines())
        if count < 100:
            raise ValueError(f"Writing file has only {count} prompts; 100 are needed")
    return count


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        destination = DATA_ROOT / name
        if destination.exists():
            count = _validate(name, destination.read_bytes())
            print(f"Already present: {destination} ({count} entries)")
            continue
        with urlopen(Request(url, headers={"User-Agent": "ICRL-Ablation-data-preparation"}), timeout=30) as response:
            payload = response.read()
        count = _validate(name, payload)
        with destination.open("xb") as output:
            output.write(payload)
        print(f"Downloaded: {destination} ({count} entries)")


if __name__ == "__main__":
    main()

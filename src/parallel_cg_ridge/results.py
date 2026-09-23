import json
from collections.abc import Iterable

def load_results(paths: Iterable[str]) -> list[dict]:
    """Read benchmark JSON-lines files, skipping Slurm banner text."""
    rows: list[dict] = []
    for path in paths:
        with open(path) as f:
            rows.extend(json.loads(line) for line in f if line.startswith("{"))
    return rows
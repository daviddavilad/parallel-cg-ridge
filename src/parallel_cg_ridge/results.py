import json
from collections.abc import Iterable

def load_results(paths: Iterable[str]) -> list[dict]:
    """Read benchmark JSON-lines files, skipping Slurm banner text."""
    rows: list[dict] = []
    for path in paths:
        with open(path) as f:
            rows.extend(json.loads(line) for line in f if line.startswith("{"))
    return rows

def group_by_size(rows: list[dict]) -> dict[int, list[dict]]:
    """Classification of problem results by size (n)."""
    groups: dict[int, list[dict]] = {}

    for r in rows:
        n = r["n"]
        groups.setdefault(n, []).append(r)

    return groups

def efficiency(rows: list[dict]) -> dict[int, list[tuple[int, float]]]:
    """Iteration-normalized parallel efficiency, grouped by problem size."""
    groups = group_by_size(rows)
    out: dict[int, list[tuple[int, float]]] = {}

    for n, group in groups.items():
        baseline = next((r for r in group if r["P"] == 1), None)
        if baseline is None:
            continue

        pairs = []
        for r in group:
            e = (baseline["wall"] / baseline["iters"]) / (r["P"] * r["wall"] / r["iters"])
            pairs.append((r["P"], e))

        out[n] = sorted(pairs)

    return out

def speedup(rows: list[dict]) -> dict[int, list[tuple[int, float]]]:
    """Iteration-normalized parallel speedup, grouped by problem size."""
    groups = group_by_size(rows)
    out: dict[int, list[tuple[int, float]]] = {}

    for n, group in groups.items():
        baseline = next((r for r in group if r["P"] == 1), None)
        if baseline is None:
            continue

        pairs = []
        for r in group:
            s = (baseline["wall"] / baseline["iters"]) / (r["wall"] / r["iters"])
            pairs.append((r["P"], s))

        out[n] = sorted(pairs)

    return out
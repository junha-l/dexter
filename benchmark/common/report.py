"""Unified console + metrics.json reporting for the benchmark metric scripts.

Each metric prints one summary line and upserts only its own keys into a shared
metrics.json next to the predictions, so running the metrics in any order
accumulates one file with every reported number.
"""

from __future__ import annotations

import json
from pathlib import Path


def report(out_dir, label: str, metrics: dict, n: int | None = None):
    """Print a per-metric summary (one metric per line) and merge into out_dir/metrics.json."""
    out_dir = Path(out_dir)
    count = f" ({n} grasps)" if n is not None else ""
    print(f"{label}{count}")
    width = max((len(k) for k in metrics), default=0)
    for k, v in metrics.items():
        print(f"  {k:<{width}}  {v:.4f}")

    path = out_dir / "metrics.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(metrics)
    path.write_text(json.dumps(merged, indent=2) + "\n")

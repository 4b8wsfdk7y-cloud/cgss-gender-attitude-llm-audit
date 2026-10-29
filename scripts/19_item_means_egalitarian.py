#!/usr/bin/env python3
# requires: Python 3.10+
"""Wave-averaged v2 item means in egalitarian coding.

The v2 tables report item means on the original response scale; the v1 audit,
the ML benchmark, and the measurement models use egalitarian coding (A421-A424
reversed as 6 - x). This script restates the v2 means in egalitarian coding
from the aggregate tables in output/v2/, so the two audits can be read on one
scale. It needs no restricted data.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / "output" / "v2"
SOURCES = ("v2_item_metrics_by_cell.csv", "v1_context_item_metrics.csv")


def egalitarian(item: str, value: float) -> float:
    return value if item == "a425" else 6 - value


def main() -> int:
    model: dict[tuple[str, str], list[float]] = defaultdict(list)
    human: dict[str, dict[str, float]] = defaultdict(dict)
    order: list[str] = []
    for name in SOURCES:
        with (OUTPUT / name).open(newline="", encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                if r["cell"] not in order:
                    order.append(r["cell"])
                model[(r["cell"], r["item"])].append(egalitarian(r["item"], float(r["mixture_mean"])))
                human[r["item"]][r["wave"]] = egalitarian(r["item"], float(r["human_mean"]))
    with (OUTPUT / "v1_context_item_means.csv").open(newline="", encoding="utf-8") as handle:
        v1 = {r["item"]: egalitarian(r["item"], float(r["v1_sampled_mean"])) for r in csv.DictReader(handle)}

    rows = []
    for item in sorted(human):
        cgss = sum(human[item].values()) / len(human[item])
        rows.append({"cell": "CGSS weighted", "item": item, "mean_egalitarian": cgss, "error": 0.0})
        rows.append({"cell": "v1 sampled (pooled)", "item": item, "mean_egalitarian": v1[item],
                     "error": v1[item] - cgss})
        for cell in order:
            values = model[(cell, item)]
            mean = sum(values) / len(values)
            rows.append({"cell": cell, "item": item, "mean_egalitarian": mean, "error": mean - cgss})
    with (OUTPUT / "v2_item_means_egalitarian.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        if r["cell"] in ("CGSS weighted", "v1 sampled (pooled)", "A1-B1-C1", "A0-B0-C0", "V1-T1"):
            print(f"{r['item']} {r['cell']:20s} {r['mean_egalitarian']:.2f} ({r['error']:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Small, dependency-free I/O helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    materialized = list(rows)
    if not materialized:
        raise ValueError("Cannot write an empty CSV")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(materialized[0])
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

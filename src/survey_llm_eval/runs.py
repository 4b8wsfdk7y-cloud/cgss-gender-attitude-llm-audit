"""Run-ledger guards that prevent incompatible records from being mixed."""

from __future__ import annotations

import json
from pathlib import Path


RunKey = tuple[str, str, int]


def load_completed(jsonl_path: str | Path) -> dict[RunKey, str]:
    path = Path(jsonl_path)
    completed: dict[RunKey, str] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("success"):
                completed[
                    (
                        str(record["profile_id"]),
                        str(record["condition"]),
                        int(record["repeat"]),
                    )
                ] = str(record.get("prompt_sha256", ""))
    return completed


def assert_output_compatible(
    jsonl_path: str | Path, model: str, model_digest: str
) -> None:
    path = Path(jsonl_path)
    if not path.exists():
        return
    models: set[str] = set()
    digests: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("model"):
                models.add(str(record["model"]))
            if record.get("model_digest"):
                digests.add(str(record["model_digest"]))
    if models and models != {model}:
        raise RuntimeError(
            f"Output already contains model(s) {sorted(models)}; current model is "
            f"{model!r}. Choose a separate --output-dir."
        )
    if digests and model_digest not in digests:
        raise RuntimeError(
            "Output was created with a different model digest. Choose a separate "
            "--output-dir instead of mixing runs."
        )

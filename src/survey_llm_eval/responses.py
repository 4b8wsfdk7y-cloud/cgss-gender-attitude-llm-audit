"""Structured-response schemas and validators shared by model runners."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def openai_response_schema(
    items: Sequence[str],
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> dict[str, Any]:
    item_schema = {"type": "integer", "minimum": minimum, "maximum": maximum}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": {
                "type": "object",
                "properties": {item: item_schema for item in items},
                "required": list(items),
                "additionalProperties": False,
            },
        },
    }


def validate_joint_content(
    content: str,
    items: Sequence[str],
    *,
    minimum: int,
    maximum: int,
    label_to_score: Mapping[str, int] | None = None,
) -> dict[str, int]:
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or set(parsed) != set(items):
        received = tuple(parsed) if isinstance(parsed, dict) else type(parsed).__name__
        raise ValueError(f"Expected exactly {tuple(items)}; received {received}")

    scores: dict[str, int] = {}
    for item in items:
        value = parsed[item]
        if isinstance(value, dict):
            if label_to_score is None or set(value) != {"label", "score"}:
                raise ValueError(f"Invalid label-score object for {item}: {value}")
            label = str(value["label"])
            score = int(value["score"])
            if label not in label_to_score:
                raise ValueError(f"Unknown response label for {item}: {label}")
            if label_to_score[label] != score:
                raise ValueError(f"Inconsistent label and score for {item}: {value}")
        else:
            score = int(value)
        if not minimum <= score <= maximum:
            raise ValueError(f"Score outside {minimum}..{maximum} for {item}: {score}")
        scores[item] = score
    return scores


def validate_single_content(
    content: str,
    item: str,
    *,
    minimum: int,
    maximum: int,
    label_to_score: Mapping[str, int],
) -> tuple[str, int]:
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or set(parsed) != {item}:
        received = tuple(parsed) if isinstance(parsed, dict) else type(parsed).__name__
        raise ValueError(f"Expected only {item}; received {received}")
    value = parsed[item]
    if not isinstance(value, dict) or set(value) != {"label", "score"}:
        raise ValueError(f"Invalid label-score object: {value}")
    label = str(value["label"])
    score = int(value["score"])
    if label not in label_to_score or label_to_score[label] != score:
        raise ValueError(f"Inconsistent label and score: {value}")
    if not minimum <= score <= maximum:
        raise ValueError(f"Score outside {minimum}..{maximum}: {score}")
    return label, score

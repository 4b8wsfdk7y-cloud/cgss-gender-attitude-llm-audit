"""Benchmark specification loading and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkSpec:
    """Validated description of a survey-response evaluation task."""

    name: str
    description: str
    profile_fields: tuple[str, ...]
    items: tuple[str, ...]
    scale_minimum: int
    scale_maximum: int
    subgroups: tuple[str, ...]
    data_boundary: str
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> "BenchmarkSpec":
        spec_path = Path(path)
        raw = spec_path.read_bytes()
        payload: dict[str, Any] = json.loads(raw.decode("utf-8"))

        items_payload = payload.get("items", [])
        items = tuple(
            str(item["id"] if isinstance(item, dict) else item)
            for item in items_payload
        )
        scale = payload.get("scale", {})
        profile_fields = tuple(str(field) for field in payload.get("profile_fields", []))
        subgroups = tuple(str(field) for field in payload.get("subgroups", []))

        if not payload.get("name"):
            raise ValueError("Benchmark spec requires a non-empty name")
        if not items or len(items) != len(set(items)):
            raise ValueError("Benchmark items must be non-empty and unique")
        if "profile_id" not in profile_fields:
            raise ValueError("profile_fields must include profile_id")
        if set(items) & set(profile_fields):
            raise ValueError("Survey items cannot also be profile fields")
        if not set(subgroups).issubset(profile_fields):
            raise ValueError("Every subgroup must also be a profile field")

        minimum = int(scale.get("minimum", 1))
        maximum = int(scale.get("maximum", 5))
        if minimum >= maximum:
            raise ValueError("Scale minimum must be smaller than maximum")

        return cls(
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
            profile_fields=profile_fields,
            items=items,
            scale_minimum=minimum,
            scale_maximum=maximum,
            subgroups=subgroups,
            data_boundary=str(payload.get("data_boundary", "")),
            sha256=hashlib.sha256(raw).hexdigest(),
        )

    @property
    def scale_values(self) -> tuple[int, ...]:
        return tuple(range(self.scale_minimum, self.scale_maximum + 1))

    def validate_row(self, row: dict[str, str], *, require_items: bool) -> None:
        required = set(self.profile_fields)
        if require_items:
            required.update(self.items)
        missing = sorted(field for field in required if field not in row or row[field] == "")
        if missing:
            raise ValueError(f"Row is missing required fields: {missing}")
        if require_items:
            for item in self.items:
                score = int(row[item])
                if score not in self.scale_values:
                    raise ValueError(
                        f"{item}={score} is outside {self.scale_minimum}..{self.scale_maximum}"
                    )

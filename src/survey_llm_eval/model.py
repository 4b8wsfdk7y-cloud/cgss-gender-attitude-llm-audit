"""Model protocol and a deterministic public-demo adapter."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


class RespondentModel(Protocol):
    def generate(
        self,
        profile: Mapping[str, str],
        items: Sequence[str],
        *,
        repeat: int,
        minimum: int,
        maximum: int,
    ) -> dict[str, int]: ...


@dataclass(frozen=True)
class DeterministicMockModel:
    """Local stub used to exercise the pipeline without an LLM or private data."""

    salt: str = "survey-llm-eval-public-demo-v1"

    def generate(
        self,
        profile: Mapping[str, str],
        items: Sequence[str],
        *,
        repeat: int,
        minimum: int,
        maximum: int,
    ) -> dict[str, int]:
        profile_key = "|".join(f"{key}={profile[key]}" for key in sorted(profile))
        width = maximum - minimum + 1
        answers: dict[str, int] = {}
        for item in items:
            digest = hashlib.sha256(
                f"{self.salt}|{profile_key}|{item}|{repeat}".encode("utf-8")
            ).hexdigest()
            answers[item] = minimum + int(digest[:12], 16) % width
        return answers

#!/usr/bin/env python3
# requires: Python 3.10+ standard library; LM Studio local server

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
AUDIT_ROOT = SCRIPT_PATH.parents[1]
INPUT_PATH = AUDIT_ROOT / "data" / "profiles_llm_input.csv"
PROMPT_PATH = AUDIT_ROOT / "prompts" / "neutral_verbal_zh.txt"
OUTPUT_PATH = AUDIT_ROOT / "tmp" / "lmstudio_smoke_test.jsonl"

ITEMS = ("A421", "A422", "A423", "A424", "A425")
LABEL_TO_SCORE = {
    "完全不同意": 1,
    "比较不同意": 2,
    "无所谓同意不同意": 3,
    "比较同意": 4,
    "完全同意": 5,
}
LABELS = tuple(LABEL_TO_SCORE)


def cgss_response_schema() -> dict[str, Any]:
    item_schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(LABELS)},
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["label", "score"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "cgss_gender_attitude_answers",
            "schema": {
                "type": "object",
                "properties": {item: item_schema for item in ITEMS},
                "required": list(ITEMS),
                "additionalProperties": False,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small LM Studio compatibility test for the CGSS LLM audit."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default=None, help="Model id from /v1/models. Defaults to the first loaded model.")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def request_json(url: str, payload: dict[str, Any] | None, timeout: int) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def list_models(base_url: str, timeout: int) -> list[str]:
    try:
        response = request_json(f"{base_url.rstrip('/')}/models", None, timeout)
    except Exception as exc:
        raise RuntimeError(
            "Cannot reach LM Studio. Start LM Studio, load a model, and enable the local server."
        ) from exc
    models = [row.get("id") for row in response.get("data", []) if row.get("id")]
    if not models:
        raise RuntimeError("LM Studio server is reachable, but no model is loaded.")
    return models


def load_profiles(limit: int) -> list[dict[str, str]]:
    with INPUT_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    per_wave: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        per_wave.setdefault(row["wave"], []).append(row)
    selected: list[dict[str, str]] = []
    waves = sorted(per_wave)
    base, remainder = divmod(limit, len(waves))
    for index, wave in enumerate(waves):
        selected.extend(per_wave[wave][: base + (index < remainder)])
    return selected[:limit]


def validate_scores(content: str) -> dict[str, int]:
    parsed = json.loads(content)
    if set(parsed) != set(ITEMS):
        raise ValueError(f"Expected exactly {ITEMS}; received {tuple(parsed)}")
    scores: dict[str, int] = {}
    for item in ITEMS:
        value = parsed[item]
        if not isinstance(value, dict):
            score = int(value)
            if score < 1 or score > 5:
                raise ValueError(f"Score outside 1..5 for {item}: {score}")
            scores[item] = score
            continue
        label = str(value.get("label"))
        score = int(value.get("score"))
        if LABEL_TO_SCORE.get(label) != score:
            raise ValueError(f"Inconsistent label and score for {item}: {value}")
        scores[item] = score
    return scores


def call_lmstudio(
    *,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
) -> tuple[str, float, dict[str, Any]]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你正在参加一项社会调查模拟。严格遵守用户要求，只返回JSON，不解释推理。",
            },
            {"role": "user", "content": "/no_think\n" + prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "response_format": cgss_response_schema(),
        "stream": False,
    }
    started = time.perf_counter()
    response = request_json(f"{base_url.rstrip('/')}/chat/completions", payload, timeout)
    elapsed = time.perf_counter() - started
    message = response["choices"][0]["message"]
    content = message.get("content") or message.get("reasoning_content") or ""
    return content, elapsed, response.get("usage", {})


def main() -> int:
    args = parse_args()
    models = list_models(args.base_url, args.timeout)
    model = args.model or models[0]
    if model not in models:
        raise RuntimeError(f"Model {model!r} is not loaded. Available models: {models}")

    template = PROMPT_PATH.read_text(encoding="utf-8")
    profiles = load_profiles(args.limit)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    successes = 0
    elapsed_values: list[float] = []
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for index, profile in enumerate(profiles, start=1):
            prompt = template.replace("{{PERSONA}}", profile["persona"])
            record: dict[str, Any] = {
                "profile_id": profile["profile_id"],
                "wave": profile["wave"],
                "model": model,
                "success": False,
            }
            try:
                content, elapsed, usage = call_lmstudio(
                    base_url=args.base_url,
                    model=model,
                    prompt=prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                scores = validate_scores(content)
                record.update(
                    success=True,
                    elapsed_seconds=elapsed,
                    scores=scores,
                    usage=usage,
                    raw_content=content,
                )
                successes += 1
                elapsed_values.append(elapsed)
                print(f"[{index}/{len(profiles)}] ok {profile['profile_id']} {elapsed:.2f}s {scores}")
            except Exception as exc:
                record.update(error=repr(exc))
                print(f"[{index}/{len(profiles)}] fail {profile['profile_id']} {exc}")
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    mean_elapsed = sum(elapsed_values) / len(elapsed_values) if elapsed_values else float("nan")
    print(
        f"LM Studio smoke test complete: success={successes}/{len(profiles)}, "
        f"mean_elapsed_seconds={mean_elapsed:.2f}, output={OUTPUT_PATH}"
    )
    return 0 if successes else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)

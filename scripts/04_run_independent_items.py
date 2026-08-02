#!/usr/bin/env python3
# requires: Python 3.10+ standard library; Ollama with qwen3:8b

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from survey_llm_eval.responses import validate_single_content


SCRIPT_PATH = Path(__file__).resolve()
AUDIT_ROOT = SCRIPT_PATH.parents[1]
CONFIG_PATH = AUDIT_ROOT / "config.json"
INPUT_PATH = AUDIT_ROOT / "data" / "profiles_llm_input.csv"
PROMPT_PATH = AUDIT_ROOT / "prompts" / "neutral_independent_zh.txt"
OUTPUT_DIR = AUDIT_ROOT / "output"
JSONL_PATH = OUTPUT_DIR / "responses_independent.jsonl"
CSV_PATH = OUTPUT_DIR / "responses_independent.csv"

ITEM_TEXT = {
    "A421": "男人以事业为重，女人以家庭为重。",
    "A422": "男性能力天生比女性强。",
    "A423": "干得好不如嫁得好。",
    "A424": "在经济不景气时，应该先解雇女性员工。",
    "A425": "夫妻应该均等分摊家务。",
}
LABEL_TO_SCORE = {
    "完全不同意": 1,
    "比较不同意": 2,
    "无所谓同意不同意": 3,
    "比较同意": 4,
    "完全同意": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_profiles(limit: int | None) -> list[dict[str, str]]:
    with INPUT_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit is None:
        return rows
    per_wave: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        per_wave.setdefault(row["wave"], []).append(row)
    selected: list[dict[str, str]] = []
    waves = sorted(per_wave)
    base, remainder = divmod(limit, len(waves))
    for index, wave in enumerate(waves):
        selected.extend(per_wave[wave][: base + (index < remainder)])
    return selected[:limit]


def check_ollama(url: str, model: str) -> str:
    base = url.rsplit("/api/", 1)[0]
    with urllib.request.urlopen(f"{base}/api/tags", timeout=10) as response:
        tags = json.load(response)
    available = {
        row["name"]: row.get("digest", "")
        for row in tags.get("models", [])
    }
    if model not in available:
        raise RuntimeError(
            f"Model {model!r} is not installed. Available: {sorted(available)}"
        )
    return available[model]


def completed_keys() -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not JSONL_PATH.exists():
        return keys
    with JSONL_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("success"):
                keys.add((record["profile_id"], record["item"].lower()))
    return keys


def deterministic_seed(base_seed: int, profile_id: str, item: str) -> int:
    key = f"{profile_id}|{item}".encode("utf-8")
    offset = int(hashlib.sha256(key).hexdigest()[:8], 16) % 1_000_000
    return base_seed + offset


def validate_content(content: str, item: str) -> tuple[str, int]:
    return validate_single_content(
        content,
        item,
        minimum=1,
        maximum=5,
        label_to_score=LABEL_TO_SCORE,
    )


def call_model(
    config: dict[str, Any], prompt: str, seed: int
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": config["model"],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "seed": seed,
            "num_predict": 50,
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "你正在参加一项社会调查模拟。"
                    "严格遵守用户要求，只返回JSON，不解释推理。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        config["ollama_url"],
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(
        request, timeout=config["timeout_seconds"]
    ) as response:
        result = json.load(response)
    return result, time.perf_counter() - started


def append_record(record: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def rebuild_csv() -> None:
    if not JSONL_PATH.exists():
        return
    raw_rows: list[dict[str, Any]] = []
    with JSONL_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            raw_rows.append(json.loads(line))
    if not raw_rows:
        return
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for row in raw_rows:
        key = (row["profile_id"], row["item"].lower())
        if key not in deduplicated or (
            not deduplicated[key].get("success") and row.get("success")
        ):
            deduplicated[key] = row
    rows = list(deduplicated.values())
    fields = sorted({key for row in rows for key in row})
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    digest = check_ollama(config["ollama_url"], config["model"])
    profiles = load_profiles(args.limit)
    template = PROMPT_PATH.read_text(encoding="utf-8")
    completed = completed_keys()
    pending = sum(
        (profile["profile_id"], item.lower()) not in completed
        for profile in profiles
        for item in ITEM_TEXT
    )
    print(
        f"Model={config['model']} digest={digest} profiles={len(profiles)} "
        f"items={len(ITEM_TEXT)} pending={pending}"
    )

    counter = 0
    for profile in profiles:
        for item, item_text in ITEM_TEXT.items():
            item_key = item.lower()
            key = (profile["profile_id"], item_key)
            if key in completed:
                continue
            counter += 1
            prompt = (
                template.replace("{{PERSONA}}", profile["persona"])
                .replace("{{ITEM_ID}}", item)
                .replace("{{ITEM_TEXT}}", item_text)
            )
            seed = deterministic_seed(
                int(config["seed"]), profile["profile_id"], item
            )
            record: dict[str, Any] = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "profile_id": profile["profile_id"],
                "wave": profile["wave"],
                "condition": "neutral_independent",
                "item": item_key,
                "seed": seed,
                "model": config["model"],
                "model_digest": digest,
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "success": False,
            }
            try:
                response, elapsed = call_model(config, prompt, seed)
                content = response.get("message", {}).get("content", "")
                label, score = validate_content(content, item)
                record.update(
                    {
                        "success": True,
                        "elapsed_seconds": round(elapsed, 4),
                        "label": label,
                        "score": score,
                        "raw_content": content,
                        "done_reason": response.get("done_reason"),
                        "prompt_eval_count": response.get("prompt_eval_count"),
                        "eval_count": response.get("eval_count"),
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
            append_record(record)
            print(
                f"[{counter}/{pending}] {profile['profile_id']} {item} "
                f"success={record['success']} "
                f"score={record.get('score', 'NA')} "
                f"seconds={record.get('elapsed_seconds', 'NA')}",
                flush=True,
            )
    rebuild_csv()
    print(f"Wrote {JSONL_PATH} and {CSV_PATH}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        rebuild_csv()
        print("\nInterrupted safely; completed calls were preserved.")
        raise SystemExit(130)

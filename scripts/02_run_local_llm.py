#!/usr/bin/env python3
# requires: Python 3.10+ standard library; Ollama or LM Studio local server

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
AUDIT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_CONFIG_PATH = AUDIT_ROOT / "config.json"
INPUT_PATH = AUDIT_ROOT / "data" / "profiles_llm_input.csv"
DEFAULT_OUTPUT_DIR = AUDIT_ROOT / "output"
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
    item_schema = {"type": "integer", "minimum": 1, "maximum": 5}
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check_ollama(url: str, model: str) -> str:
    base = url.rsplit("/api/", 1)[0]
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=10) as response:
            tags = json.load(response)
    except Exception as exc:
        raise RuntimeError(
            "Cannot reach Ollama. Start Ollama or run `ollama serve`."
        ) from exc
    available = {
        row["name"]: row.get("digest", "")
        for row in tags.get("models", [])
    }
    if model not in available:
        raise RuntimeError(
            f"Model {model!r} is not installed. Available: {sorted(available)}"
        )
    return available[model]


def check_lmstudio(base_url: str, model: str) -> str:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=10) as response:
            listing = json.load(response)
    except Exception as exc:
        raise RuntimeError(
            "Cannot reach LM Studio. Start the local server and load the model."
        ) from exc
    available = [row.get("id") for row in listing.get("data", []) if row.get("id")]
    if model not in available:
        raise RuntimeError(f"Model {model!r} is not loaded. Available: {available}")
    return model


def load_profiles(limit: int | None) -> list[dict[str, str]]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Missing profiles_llm_input.csv. Run 01_prepare_profiles.R first."
        )
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


def load_completed(jsonl_path: Path) -> set[tuple[str, str, int]]:
    completed: set[tuple[str, str, int]] = set()
    if not jsonl_path.exists():
        return completed
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("success"):
                completed.add(
                    (
                        record["profile_id"],
                        record["condition"],
                        int(record["repeat"]),
                    )
                )
    return completed


def validate_scores(content: str) -> dict[str, int]:
    parsed = json.loads(content)
    if set(parsed) != set(ITEMS):
        raise ValueError(f"Expected exactly {ITEMS}; received {tuple(parsed)}")
    scores: dict[str, int] = {}
    for item in ITEMS:
        value = parsed[item]
        if isinstance(value, dict):
            if set(value) != {"label", "score"}:
                raise ValueError(f"Invalid label-score object for {item}: {value}")
            label = str(value["label"])
            score = int(value["score"])
            if label not in LABEL_TO_SCORE:
                raise ValueError(f"Unknown response label for {item}: {label}")
            if LABEL_TO_SCORE[label] != score:
                raise ValueError(
                    f"Inconsistent label and score for {item}: {value}"
                )
            scores[item] = score
        else:
            scores[item] = int(value)
    if any(score < 1 or score > 5 for score in scores.values()):
        raise ValueError(f"Scores outside 1..5: {scores}")
    return scores


def call_model(
    config: dict[str, Any],
    prompt: str,
    seed: int,
) -> tuple[dict[str, Any], float]:
    provider = config.get("provider", "ollama")
    if provider == "lmstudio":
        return call_lmstudio(config, prompt, seed)
    return call_ollama(config, prompt, seed)


def call_ollama(
    config: dict[str, Any],
    prompt: str,
    seed: int,
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
            "num_predict": config["num_predict"],
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


def call_lmstudio(
    config: dict[str, Any],
    prompt: str,
    seed: int,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": config["model"],
        "stream": False,
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_tokens": config["num_predict"],
        "response_format": cgss_response_schema(),
        "messages": [
            {
                "role": "system",
                "content": (
                    "你正在参加一项社会调查模拟。"
                    "严格遵守用户要求，只返回JSON，不解释推理。"
                ),
            },
            {"role": "user", "content": "/no_think\n" + prompt},
        ],
    }
    request = urllib.request.Request(
        f"{config['lmstudio_base_url'].rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(
            request, timeout=config["timeout_seconds"]
        ) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

    message = result.get("choices", [{}])[0].get("message", {})
    content = message.get("content") or message.get("reasoning_content") or ""
    return {
        "message": {"content": content},
        "done_reason": result.get("choices", [{}])[0].get("finish_reason"),
        "prompt_eval_count": result.get("usage", {}).get("prompt_tokens"),
        "eval_count": result.get("usage", {}).get("completion_tokens"),
        "total_tokens": result.get("usage", {}).get("total_tokens"),
    }, time.perf_counter() - started


def append_record(
    record: dict[str, Any],
    jsonl_path: Path,
    write_lock: threading.Lock | None = None,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    lock = write_lock or threading.Lock()
    with lock:
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def rebuild_csv(jsonl_path: Path, csv_path: Path) -> None:
    records: list[dict[str, Any]] = []
    if not jsonl_path.exists():
        return
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            flat = {
                key: value
                for key, value in record.items()
                if key not in {"scores", "response_metadata"}
            }
            for item in ITEMS:
                flat[item.lower()] = (record.get("scores") or {}).get(item)
            for key, value in (record.get("response_metadata") or {}).items():
                flat[f"meta_{key}"] = value
            records.append(flat)
    if not records:
        return
    fields = sorted({key for row in records for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    jsonl_path = output_dir / "responses.jsonl"
    csv_path = output_dir / "responses.csv"
    config = load_json(config_path)
    provider = config.get("provider", "ollama")
    if provider == "lmstudio":
        model_digest = check_lmstudio(config["lmstudio_base_url"], config["model"])
    elif provider == "ollama":
        model_digest = check_ollama(config["ollama_url"], config["model"])
    else:
        raise ValueError(f"Unknown provider: {provider}")
    profiles = load_profiles(args.limit)
    repeats = args.repeats or int(config["repeats"])
    condition_paths = config["conditions"]
    conditions = args.conditions or list(condition_paths)
    unknown = sorted(set(conditions) - set(condition_paths))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")

    prompt_templates = {
        name: (AUDIT_ROOT / condition_paths[name]).read_text(encoding="utf-8")
        for name in conditions
    }
    completed = load_completed(jsonl_path)
    total = len(profiles) * len(conditions) * repeats
    pending = total - sum(
        (row["profile_id"], condition, repeat) in completed
        for row in profiles
        for condition in conditions
        for repeat in range(1, repeats + 1)
    )
    print(
        f"Provider={provider} model={config['model']} digest={model_digest} "
        f"profiles={len(profiles)} conditions={conditions} repeats={repeats} "
        f"pending={pending} workers={args.workers} output={output_dir}"
    )

    tasks: list[tuple[dict[str, str], str, str, int, str]] = []
    for profile in profiles:
        for condition in conditions:
            template = prompt_templates[condition]
            prompt = template.replace("{{PERSONA}}", profile["persona"])
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for repeat in range(1, repeats + 1):
                key = (profile["profile_id"], condition, repeat)
                if key in completed:
                    continue
                tasks.append((profile, condition, prompt, repeat, prompt_hash))

    write_lock = threading.Lock()

    def run_task(task: tuple[dict[str, str], str, str, int, str]) -> dict[str, Any]:
        profile, condition, prompt, repeat, prompt_hash = task
        seed = int(config["seed"]) + repeat
        record: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "profile_id": profile["profile_id"],
            "wave": profile["wave"],
            "condition": condition,
            "repeat": repeat,
            "seed": seed,
            "model": config["model"],
            "model_digest": model_digest,
            "prompt_sha256": prompt_hash,
            "success": False,
        }
        try:
            response, elapsed = call_model(config, prompt, seed)
            content = response.get("message", {}).get("content", "")
            record.update(
                {
                    "success": True,
                    "elapsed_seconds": round(elapsed, 4),
                    "raw_content": content,
                    "scores": validate_scores(content),
                    "response_metadata": {
                        key: response.get(key)
                        for key in (
                            "done_reason",
                            "load_duration",
                            "prompt_eval_count",
                            "prompt_eval_duration",
                            "eval_count",
                            "eval_duration",
                            "total_tokens",
                        )
                    },
                }
            )
        except Exception as exc:
            record.update(
                {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        append_record(record, jsonl_path, write_lock)
        return record

    workers = max(1, int(args.workers))
    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            record = run_task(task)
            print(
                f"[{index}/{pending}] {record['profile_id']} "
                f"{record['condition']} r{record['repeat']} "
                f"success={record['success']} "
                f"seconds={record.get('elapsed_seconds', 'NA')}",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_task, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                print(
                    f"[{index}/{pending}] {record['profile_id']} "
                    f"{record['condition']} r{record['repeat']} "
                    f"success={record['success']} "
                    f"seconds={record.get('elapsed_seconds', 'NA')}",
                    flush=True,
                )

    rebuild_csv(jsonl_path, csv_path)
    print(f"Wrote {jsonl_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        args = parse_args()
        output_dir = Path(args.output_dir).expanduser().resolve()
        rebuild_csv(output_dir / "responses.jsonl", output_dir / "responses.csv")
        print("\nInterrupted safely; completed calls were preserved.", file=sys.stderr)
        raise SystemExit(130)

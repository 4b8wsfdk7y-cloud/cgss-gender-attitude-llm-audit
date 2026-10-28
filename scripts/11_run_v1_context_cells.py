#!/usr/bin/env python3
# requires: Python 3.10+, mlx-lm (Apple silicon), a local MLX model directory
"""Option probabilities for the v1 joint prompt with thinking open or closed.

The v1 joint condition was run through LM Studio, whose model definition for
``qwen/qwen3.5-9b`` enables thinking by default. Qwen3.5's chat template has
no ``/no_think`` switch, so every v1 answer was generated inside an open
``<think>`` block and read back from ``reasoning_content``. This script scores
the v1 prompt (v1 system prompt, ``/no_think`` line, ``neutral_verbal_zh.txt``
user prompt) in two exploratory cells, specified in the 2026-10-27 entry of
PREREGISTRATION_V2.md:

V1-T1  chat template rendered with thinking enabled (the v1 context)
V1-T0  the same prompt with thinking disabled (the v2 context)

The assistant text is the integer JSON that the v1 response schema enforced,
for example ``{"A421": 4, "A422": ``. Earlier answers are the modal options
(greedy history). Every prompt is scored with one full forward pass, as in the
v2 main run. Outputs are profile-linked and must stay outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx_lm

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "run_logprob_audit", REPO_ROOT / "scripts" / "07_run_logprob_audit.py")
run07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run07)

from survey_llm_eval.factorial import item_ids  # noqa: E402
from survey_llm_eval.probabilities import modal_option  # noqa: E402

V1_SYSTEM_PROMPT = "你正在参加一项社会调查模拟。严格遵守用户要求，只返回JSON，不解释推理。"
V1_TEMPLATE = REPO_ROOT / "prompts" / "neutral_verbal_zh.txt"
CELLS = {"V1-T1": True, "V1-T0": False}
DEFAULT_OUTPUT = Path.home() / "Projects/restricted-data/surveyllm-v2/logprobs_v1_context"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True, help="profiles_llm_input.csv")
    parser.add_argument("--v1-responses", help="v1 responses.jsonl, to check prompt hashes")
    parser.add_argument("--model", default=str(run07.DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cells", nargs="+", default=list(CELLS))
    parser.add_argument("--limit", type=int, default=None, help="profiles per wave")
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    return parser.parse_args()


def v1_messages(persona: str) -> tuple[list[dict[str, str]], str]:
    prompt = V1_TEMPLATE.read_text(encoding="utf-8").replace("{{PERSONA}}", persona)
    messages = [
        {"role": "system", "content": V1_SYSTEM_PROMPT},
        {"role": "user", "content": "/no_think\n" + prompt},
    ]
    return messages, hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def json_prefix(items: list[str], history: list[int], target: int) -> str:
    answered = ", ".join(f'"{items[i]}": {answer}' for i, answer in enumerate(history))
    return "{" + (answered + ", " if answered else "") + f'"{items[target]}": '


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if REPO_ROOT in output_dir.resolve().parents or output_dir.resolve() == REPO_ROOT:
        raise SystemExit("Profile-linked outputs must be written outside the repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(args.cells) - set(CELLS))
    if unknown:
        raise SystemExit(f"Unknown cells: {unknown}")

    profiles = run07.load_profiles(Path(args.profiles).expanduser(), args.limit)
    v1_hashes = {}
    if args.v1_responses:
        for line in Path(args.v1_responses).expanduser().open(encoding="utf-8"):
            record = json.loads(line)
            v1_hashes[record["profile_id"]] = record["prompt_sha256"]

    model_dir = Path(args.model).expanduser()
    scorer = run07.OptionScorer(model_dir)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    mx.set_cache_limit(int(args.cache_limit_gb * 2**30))
    items = list(item_ids())

    suffix = f"_limit{args.limit}" if args.limit else ""
    ledger = output_dir / f"option_probabilities_v1_context{suffix}.jsonl"
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "model_index_sha256": run07.sha256_file(model_dir / "model.safetensors.index.json"),
        "chat_template_sha256": run07.sha256_file(model_dir / "chat_template.jinja"),
        "mlx_version": mx.__version__,
        "mlx_lm_version": mlx_lm.__version__,
        "code_commit": run07.git_commit(),
        "cells": args.cells,
        "history_mode": "greedy",
        "profiles": len(profiles),
        "option_token_ids": scorer.option_ids,
        "scoring": "full forward pass, float32 final-position projection",
    }
    (output_dir / f"run_metadata_v1_context{suffix}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    done = set()
    if ledger.exists():
        for line in ledger.open(encoding="utf-8"):
            r = json.loads(line)
            done.add((r["profile_id"], r["cell"], r["item"]))
    history_from_ledger: dict[tuple[str, str], dict[str, int]] = {}
    if ledger.exists():
        for line in ledger.open(encoding="utf-8"):
            r = json.loads(line)
            history_from_ledger.setdefault((r["profile_id"], r["cell"]), {})[r["item"]] = (
                modal_option({int(k): v for k, v in r["probabilities"].items()}))

    started, scored = time.perf_counter(), 0
    with ledger.open("a", encoding="utf-8") as out:
        for profile in profiles:
            messages, v1_hash = v1_messages(profile["persona"])
            if v1_hashes and v1_hashes.get(profile["profile_id"]) != v1_hash:
                raise SystemExit(f"Prompt differs from v1 for {profile['profile_id']}")
            for cell in args.cells:
                base = scorer.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                    enable_thinking=CELLS[cell])
                known = history_from_ledger.get((profile["profile_id"], cell), {})
                history: list[int] = []
                for target, item in enumerate(items):
                    if (profile["profile_id"], cell, item) in done:
                        history.append(known[item])
                        continue
                    text = base + json_prefix(items, history, target)
                    result = scorer.score(scorer.tokens(text), use_cache=False)
                    distribution = {int(k): v for k, v in result["distribution"].items()}
                    record = {
                        "profile_id": profile["profile_id"],
                        "wave": profile["wave"],
                        "cell": cell,
                        "history_mode": "greedy",
                        "path": 0,
                        "item": item,
                        "target_index": target,
                        "history": list(history),
                        "probabilities": {str(k): v for k, v in distribution.items()},
                        "valid_option_mass": result["valid_option_mass"],
                        "top_token": result["top_token"],
                        "top_token_prob": result["top_token_prob"],
                        "prompt_tokens": result["prompt_tokens"],
                        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "v1_user_prompt_sha256": v1_hash,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    history.append(modal_option(distribution))
                    scored += 1
                    if scored % 100 == 0:
                        rate = (time.perf_counter() - started) / scored
                        print(f"{scored} prompts, {rate:.2f} s each", flush=True)
    print(f"done: {scored} prompts scored -> {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

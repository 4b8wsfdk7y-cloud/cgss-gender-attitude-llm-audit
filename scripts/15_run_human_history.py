#!/usr/bin/env python3
# requires: Python 3.10+, mlx, mlx-lm, the survey_llm_eval package
"""Exploratory human-history cells (PREREGISTRATION_V2.md, 2026-10-29 entry).

H-A1-B1-C1 and H-A0-B1-C0 use the prompts of A1-B1-C1 and A0-B1-C0, but the
history for each item holds the matched CGSS respondent's own answers to the
earlier items instead of the model's. A421 has no history; its prompt is the
primary run's and is not scored again. Profiles whose respondent lacks a valid
answer on any of A421-A425 are skipped. Profile-linked records are written
outside the repository.
"""

from __future__ import annotations

import argparse
import csv
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

from survey_llm_eval.factorial import build_prompt, cell_from_name, item_ids  # noqa: E402

CELLS = {"H-A1-B1-C1": "A1-B1-C1", "H-A0-B1-C0": "A0-B1-C0"}
VALID = {"1", "2", "3", "4", "5"}
DEFAULT_OUTPUT = Path.home() / "Projects/restricted-data/surveyllm-v2/logprobs_human_history"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True, help="profiles_llm_input.csv")
    parser.add_argument("--matched", required=True, help="profiles_pilot.csv (matched answers)")
    parser.add_argument("--model", default=str(run07.DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cells", nargs="+", default=list(CELLS))
    parser.add_argument("--limit", type=int, default=None, help="profiles per wave")
    parser.add_argument("--cache-limit-gb", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if REPO_ROOT in output_dir.resolve().parents or output_dir.resolve() == REPO_ROOT:
        raise SystemExit("Profile-linked outputs must be written outside the repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(args.cells) - set(CELLS))
    if unknown:
        raise SystemExit(f"Unknown cells: {unknown}")

    items = list(item_ids())
    profiles = run07.load_profiles(Path(args.profiles).expanduser(), args.limit)
    with Path(args.matched).expanduser().open(newline="", encoding="utf-8") as handle:
        matched = {r["profile_id"]: r for r in csv.DictReader(handle)}
    eligible = [p for p in profiles
                if all(matched[p["profile_id"]][i.lower()] in VALID for i in items)]
    skipped = len(profiles) - len(eligible)

    model_dir = Path(args.model).expanduser()
    scorer = run07.OptionScorer(model_dir)
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    mx.set_cache_limit(int(args.cache_limit_gb * 2**30))

    suffix = f"_limit{args.limit}" if args.limit else ""
    ledger = output_dir / f"option_probabilities_human_history{suffix}.jsonl"
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "model_index_sha256": run07.sha256_file(model_dir / "model.safetensors.index.json"),
        "chat_template_sha256": run07.sha256_file(model_dir / "chat_template.jinja"),
        "mlx_version": mx.__version__,
        "mlx_lm_version": mlx_lm.__version__,
        "code_commit": run07.git_commit(),
        "cells": {cell: CELLS[cell] for cell in args.cells},
        "history_mode": "human",
        "profiles": len(eligible),
        "profiles_skipped_missing_answers": skipped,
        "option_token_ids": scorer.option_ids,
        "scoring": "full forward pass, float32 final-position projection; A421 not scored",
    }
    (output_dir / f"run_metadata_human_history{suffix}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    done = set()
    if ledger.exists():
        raw = ledger.read_bytes()
        if raw and not raw.endswith(b"\n"):
            ledger.write_bytes(raw[: raw.rfind(b"\n") + 1])
        for line in ledger.open(encoding="utf-8"):
            r = json.loads(line)
            done.add((r["profile_id"], r["cell"], r["item"]))

    started, scored = time.perf_counter(), 0
    with ledger.open("a", encoding="utf-8") as out:
        for profile in eligible:
            answers = [int(matched[profile["profile_id"]][i.lower()]) for i in items]
            for name in args.cells:
                cell = cell_from_name(CELLS[name])
                for index in range(1, len(items)):
                    item = items[index]
                    if (profile["profile_id"], name, item) in done:
                        continue
                    history = answers[:index]
                    messages, prefix = build_prompt(profile["persona"], index, cell, history=history)
                    text = scorer.render(messages, prefix)
                    result = scorer.score(scorer.tokens(text), use_cache=False)
                    record = {
                        "profile_id": profile["profile_id"],
                        "wave": profile["wave"],
                        "cell": name,
                        "prompt_cell": CELLS[name],
                        "history_mode": "human",
                        "path": 0,
                        "item": item,
                        "target_index": index,
                        "history": history,
                        "probabilities": {str(k): v for k, v in result["distribution"].items()},
                        "valid_option_mass": result["valid_option_mass"],
                        "top_token": result["top_token"],
                        "top_token_prob": result["top_token_prob"],
                        "prompt_tokens": result["prompt_tokens"],
                        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    scored += 1
                    if scored % 100 == 0:
                        rate = (time.perf_counter() - started) / scored
                        print(f"{scored} prompts, {rate:.2f} s each", flush=True)
    print(f"done: {scored} prompts scored ({skipped} profiles skipped) -> {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

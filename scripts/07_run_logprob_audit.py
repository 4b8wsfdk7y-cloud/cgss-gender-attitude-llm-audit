#!/usr/bin/env python3
# requires: Python 3.10+, mlx-lm (Apple silicon), a local MLX model directory
"""Read option-level probabilities for the 2 x 2 x 2 presentation experiment.

For every profile, cell, and item, one forward pass returns the model's
next-token distribution after the assistant prefix. The probabilities of the
tokens "1" to "5" are renormalized over the scale. No text is sampled, so the
main run is deterministic. Cells with answer history use the most probable
earlier answers (``--history greedy``) or seeded draws from the earlier
distributions (``--history sampled``) as context.

Outputs are profile-linked and must stay outside the public repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx_lm
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from survey_llm_eval.factorial import CELLS, build_prompt, cell_from_name, item_ids
from survey_llm_eval.probabilities import modal_option, normalize_option_logprobs

SCALE = (1, 2, 3, 4, 5)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"
DEFAULT_OUTPUT = Path.home() / "Projects/restricted-data/surveyllm-v2/logprobs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True, help="profiles_llm_input.csv")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cells", nargs="+", default=[cell.name for cell in CELLS])
    parser.add_argument("--history", choices=("greedy", "sampled"), default="greedy")
    parser.add_argument("--paths", type=int, default=1, help="history paths per profile")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--limit", type=int, default=None, help="profiles per wave")
    parser.add_argument("--cache", action="store_true",
                        help="reuse prefix caches (faster, but probabilities can differ "
                             "slightly from single full forward passes)")
    parser.add_argument("--verify", type=int, default=0,
                        help="check the first N cached scores against full forward passes")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="maximum allowed probability gap in the cache check")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_profiles(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit is None:
        return rows
    kept: dict[str, int] = {}
    selected = []
    for row in rows:
        if kept.get(row["wave"], 0) < limit:
            selected.append(row)
            kept[row["wave"]] = kept.get(row["wave"], 0) + 1
    return selected


def _snapshot(cache: list[Any]) -> list[Any]:
    """Copy cache state so later updates cannot alter the checkpoint."""
    snapshot: list[Any] = []
    arrays: list[Any] = []
    for layer in cache:
        state = layer.state
        state = list(state) if isinstance(state, list) else tuple(state)
        snapshot.append(state)
        arrays.extend(a for a in state if a is not None)
    mx.eval(arrays)
    return snapshot


class OptionScorer:
    """Next-token probabilities over the response scale for chat prompts.

    Prompts for one profile share a long common prefix. The scorer keeps a
    checkpoint of the model cache after that prefix and, when a prompt extends
    the previous one, processes only the new tokens.
    """

    def __init__(self, model_dir: Path) -> None:
        self.model, self.tokenizer = load(str(model_dir))
        self.option_ids: dict[int, int] = {}
        for option in SCALE:
            ids = self.tokenizer.encode(str(option), add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"Option {option} is not a single token: {ids}")
            self.option_ids[option] = ids[0]
        self.checkpoint: list[Any] | None = None
        self.checkpoint_tokens: list[int] = []
        self.cache: list[Any] | None = None
        self.cache_tokens: list[int] = []

    def render(self, messages: list[dict[str, str]], prefix: str) -> str:
        try:
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
        return text + prefix

    def tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def set_checkpoint(self, tokens: list[int]) -> None:
        cache = make_prompt_cache(self.model)
        self.model(mx.array(tokens)[None], cache=cache)
        self.checkpoint = _snapshot(cache)
        self.checkpoint_tokens = list(tokens)
        self.cache, self.cache_tokens = None, []

    def _restore(self) -> list[Any]:
        cache = make_prompt_cache(self.model)
        for layer, state in zip(cache, self.checkpoint or []):
            layer.state = list(state) if isinstance(state, list) else state
        return cache

    def _logits(self, tokens: list[int]) -> Any:
        if (self.cache is not None and len(tokens) > len(self.cache_tokens)
                and tokens[: len(self.cache_tokens)] == self.cache_tokens):
            delta = tokens[len(self.cache_tokens):]
        elif (self.checkpoint is not None and len(tokens) > len(self.checkpoint_tokens)
                and tokens[: len(self.checkpoint_tokens)] == self.checkpoint_tokens):
            self.cache = self._restore()
            delta = tokens[len(self.checkpoint_tokens):]
        else:
            self.cache = make_prompt_cache(self.model)
            delta = tokens
        logits = self.model(mx.array(delta)[None], cache=self.cache)[0, -1]
        self.cache_tokens = list(tokens)
        return logits

    def full_logits(self, tokens: list[int]) -> Any:
        return self.model(mx.array(tokens)[None])[0, -1]

    def score(self, tokens: list[int], use_cache: bool = True) -> dict[str, Any]:
        raw = self._logits(tokens) if use_cache else self.full_logits(tokens)
        logits = raw.astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits)
        mx.eval(logprobs)
        top = int(mx.argmax(logprobs).item())
        option_logprobs = {
            option: float(logprobs[token].item())
            for option, token in self.option_ids.items()
        }
        distribution, mass = normalize_option_logprobs(option_logprobs)
        return {
            "distribution": distribution,
            "valid_option_mass": mass,
            "top_token": self.tokenizer.decode([top]),
            "top_token_prob": float(mx.exp(logprobs[top]).item()),
            "prompt_tokens": len(tokens),
        }


def common_prefix(sequences: list[list[int]]) -> list[int]:
    shortest = min(sequences, key=len)
    for index, token in enumerate(shortest):
        if any(sequence[index] != token for sequence in sequences):
            return shortest[:index]
    return list(shortest)


def choose_history_answer(
    distribution: dict[int, float], mode: str, rng: random.Random
) -> int:
    if mode == "greedy":
        return modal_option(distribution)
    options = sorted(distribution)
    return rng.choices(options, weights=[distribution[o] for o in options], k=1)[0]


def completed_keys(path: Path) -> set[tuple[str, str, str, int, str]]:
    keys: set[tuple[str, str, str, int, str]] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            keys.add((row["profile_id"], row["cell"], row["history_mode"],
                      row["path"], row["item"]))
    return keys


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if REPO_ROOT in output_dir.resolve().parents or output_dir.resolve() == REPO_ROOT:
        raise SystemExit("Profile-linked outputs must be written outside the repository")

    model_dir = Path(args.model).expanduser()
    profiles = load_profiles(Path(args.profiles).expanduser(), args.limit)
    cells = [cell_from_name(name) for name in args.cells]
    items = item_ids()
    paths = args.paths if args.history == "sampled" else 1

    scorer = OptionScorer(model_dir)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "model_index_sha256": sha256_file(model_dir / "model.safetensors.index.json"),
        "model_config_sha256": sha256_file(model_dir / "config.json"),
        "chat_template_sha256": (
            sha256_file(model_dir / "chat_template.jinja")
            if (model_dir / "chat_template.jinja").exists() else None
        ),
        "mlx_lm_version": mlx_lm.__version__,
        "code_commit": git_commit(),
        "cells": [cell.name for cell in cells],
        "history_mode": args.history,
        "paths": paths,
        "seed": args.seed,
        "profiles": len(profiles),
        "option_token_ids": scorer.option_ids,
        "scoring": "prefix-cached" if args.cache else "single full forward pass per prompt",
    }
    (output_dir / f"run_metadata_{args.history}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ledger = output_dir / f"option_probabilities_{args.history}.jsonl"
    done = completed_keys(ledger)
    started = time.perf_counter()
    scored = 0
    verified = 0
    max_gap = 0.0
    gaps: list[float] = []
    mode_flips = 0
    with ledger.open("a", encoding="utf-8") as out:
        for profile in profiles:
            if args.cache:
                probes = [
                    scorer.tokens(scorer.render(*build_prompt(
                        profile["persona"], 1 if cell.history else 0, cell,
                        history=[1] if cell.history else [],
                    )))
                    for cell in CELLS
                ]
                scorer.set_checkpoint(common_prefix(probes)[:-1])
            for cell in cells:
                for path in range(paths):
                    rng = random.Random(
                        f"{args.seed}|{profile['profile_id']}|{cell.name}|{path}"
                    )
                    answers: list[int] = []
                    for index, item in enumerate(items):
                        history = list(answers) if cell.history else []
                        key = (profile["profile_id"], cell.name, args.history, path, item)
                        if key in done and not cell.history:
                            continue
                        messages, prefix = build_prompt(
                            profile["persona"], index, cell, history=history
                        )
                        text = scorer.render(messages, prefix)
                        tokens = scorer.tokens(text)
                        result = scorer.score(tokens, use_cache=args.cache)
                        if args.cache and verified < args.verify:
                            reference = scorer.score(tokens, use_cache=False)
                            gap = max(
                                abs(result["distribution"][o] - reference["distribution"][o])
                                for o in SCALE
                            )
                            max_gap = max(max_gap, gap)
                            gaps.append(gap)
                            mode_flips += int(
                                modal_option(result["distribution"])
                                != modal_option(reference["distribution"])
                            )
                            verified += 1
                            if gap > args.tolerance:
                                raise SystemExit(
                                    f"Cached and full scores differ by {gap:.4f} "
                                    f"({profile['profile_id']}, {cell.name}, {item})"
                                )
                            if verified == args.verify:
                                print(f"cache check passed over {verified} prompts: "
                                      f"mean gap {sum(gaps) / len(gaps):.5f}, "
                                      f"max gap {max_gap:.5f}, "
                                      f"modal-option changes {mode_flips}", flush=True)
                        distribution = result["distribution"]
                        answers.append(choose_history_answer(distribution, args.history, rng))
                        if key in done:
                            continue
                        record = {
                            "profile_id": profile["profile_id"],
                            "wave": profile["wave"],
                            "cell": cell.name,
                            "history_mode": args.history,
                            "path": path,
                            "item": item,
                            "target_index": index,
                            "history": history,
                            "probabilities": {str(k): v for k, v in distribution.items()},
                            "valid_option_mass": result["valid_option_mass"],
                            "top_token": result["top_token"],
                            "top_token_prob": result["top_token_prob"],
                            "prompt_tokens": result["prompt_tokens"],
                            "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out.flush()
                        scored += 1
                        if scored % 50 == 0:
                            rate = scored / (time.perf_counter() - started)
                            print(f"scored {scored} items ({rate:.2f}/s)", flush=True)
    print(f"done: {scored} new item scores in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

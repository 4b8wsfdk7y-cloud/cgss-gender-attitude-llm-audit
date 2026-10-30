#!/usr/bin/env python3
# requires: Python 3.10+, mlx-lm (Apple silicon), a local MLX model directory
"""Questionnaire-exposure probe (PREREGISTRATION_V2.md, 2026-10-29 evening, item 2).

- (a) order test: summed log-probability of the battery lines for all 120
  orders of the five items, under a CGSS header and a neutral header, with a
  positive control whose canonical order the model should know;
- (b) free recall: greedy continuation of the battery after the first k items.

The texts are public questionnaire wording; no CGSS data are read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from survey_llm_eval.factorial import CGSS_ITEMS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"

ITEMS = [text.rstrip("。") for _, text in CGSS_ITEMS]
HEADERS = {
    "cgss": "中国综合社会调查（CGSS）问卷\nA42. 您是否同意下列说法？\n",
    "neutral": "某市居民生活状况调查问卷\nQ12. 您是否同意下列说法？\n",
}
CONTROL_HEADER = "社会主义核心价值观："
CONTROL_WORDS = ["富强", "民主", "文明", "和谐", "自由"]
# A421 and A423 are common sayings; reproducing them says little about exposure.
INFORMATIVE = {1, 3, 4}
# Generated lines withheld from the public table because they are offensive.
# The SHA-256 of the UTF-8 line lets a rerun confirm which line was masked.
MASKED_SHA256 = {"28c2d41d7d59cff5f6f7117080da5588c5203caafe164b9e68b2dc4c42dbea89"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "output" / "v2"))
    parser.add_argument("--max-tokens", type=int, default=32)
    return parser.parse_args()


def battery(order: tuple[int, ...]) -> str:
    return "".join(f"{n}. {ITEMS[i]}\n" for n, i in enumerate(order, start=1))


class Probe:
    def __init__(self, model_dir: Path) -> None:
        self.model, self.tokenizer = load(str(model_dir))
        text_model = getattr(self.model, "language_model", self.model)
        self.backbone = text_model.model
        if getattr(text_model.args, "tie_word_embeddings", False):
            self.head = text_model.model.embed_tokens.as_linear
        else:
            self.head = text_model.lm_head

    def continuation_logprob(self, prefix: str, continuation: str) -> tuple[float, int]:
        """Summed log-probability of the continuation tokens, one full forward pass."""
        base = self.tokenizer.encode(prefix)
        full = self.tokenizer.encode(prefix + continuation)
        if full[: len(base)] != base:
            raise RuntimeError("The continuation changes the prefix tokenization")
        hidden = self.backbone(mx.array(full)[None])
        logits = self.head(hidden[:, len(base) - 1 : len(full) - 1, :].astype(mx.float32))[0]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        targets = mx.array(full[len(base):])[:, None]
        total = mx.take_along_axis(logprobs, targets, axis=-1).sum()
        mx.eval(total)
        return float(total.item()), len(full) - len(base)

    def greedy_line(self, prompt: str, max_tokens: int) -> str:
        cache = make_prompt_cache(self.model)
        tokens = self.tokenizer.encode(prompt)
        hidden = self.backbone(mx.array(tokens)[None], cache=cache)
        generated: list[int] = []
        for _ in range(max_tokens):
            logits = self.head(hidden[:, -1:, :].astype(mx.float32))[0, -1]
            token = int(mx.argmax(logits).item())
            generated.append(token)
            if "\n" in self.tokenizer.decode(generated):
                break
            hidden = self.backbone(mx.array([[token]]), cache=cache)
        return self.tokenizer.decode(generated).split("\n")[0].strip()


def order_test(probe: Probe, header: str, lines) -> tuple[list[dict], dict]:
    canonical = tuple(range(5))
    rows = []
    for order in itertools.permutations(range(5)):
        logprob, n_tokens = probe.continuation_logprob(header, lines(order))
        rows.append({"order": "".join(str(i + 1) for i in order), "logprob": logprob,
                     "tokens": n_tokens, "canonical": order == canonical})
    rows.sort(key=lambda r: -r["logprob"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    canon = next(r for r in rows if r["canonical"])
    values = sorted(r["logprob"] for r in rows)
    median = (values[59] + values[60]) / 2
    summary = {"canonical_rank": canon["rank"], "permutation_p": canon["rank"] / len(rows),
               "canonical_logprob": canon["logprob"], "median_logprob": median,
               "margin_over_median": canon["logprob"] - median,
               "best_order": rows[0]["order"], "best_logprob": rows[0]["logprob"]}
    return rows, summary


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    probe = Probe(Path(args.model))

    conditions = {name: (header, battery) for name, header in HEADERS.items()}
    conditions["control"] = (CONTROL_HEADER, lambda order: "、".join(CONTROL_WORDS[i] for i in order))
    order_rows, summaries = [], []
    for name, (header, lines) in conditions.items():
        rows, summary = order_test(probe, header, lines)
        order_rows += [{"condition": name, **r} for r in rows]
        summaries.append({"condition": name, **summary})
        print(f"{name:8s} rank {summary['canonical_rank']:3d}/120  p={summary['permutation_p']:.3f}  "
              f"margin {summary['margin_over_median']:+.2f}  best {summary['best_order']}", flush=True)

    recall = []
    for name, header in HEADERS.items():
        for k in range(1, 5):
            prompt = header + battery(tuple(range(k))) + f"{k + 1}. "
            line = probe.greedy_line(prompt, args.max_tokens)
            stripped = line.rstrip("。")
            matches = [i + 1 for i, text in enumerate(ITEMS) if stripped == text]
            digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
            masked = digest in MASKED_SHA256
            shown = f"[masked: offensive sexual content; sha256 {digest}]" if masked else line
            recall.append({"condition": name, "k": k, "target": f"A42{k + 1}",
                           "generated": shown, "masked": masked,
                           "equals_target": stripped == ITEMS[k],
                           "equals_any_item": ",".join(f"A42{i}" for i in matches),
                           "target_informative": k in INFORMATIVE})
            print(f"{name:8s} k={k} -> {shown}", flush=True)

    def write(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write(out / "exploratory_exposure_orders.csv", order_rows)
    write(out / "exploratory_exposure_order_test.csv", summaries)
    write(out / "exploratory_exposure_recall.csv", recall)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {"model": Path(args.model).name, "git_commit": commit, "headers": HEADERS,
            "control_header": CONTROL_HEADER, "control_words": CONTROL_WORDS,
            "items": ITEMS, "max_tokens": args.max_tokens,
            "notes": "raw text, no chat template; float32 output projection",
            "masked_generations": "recall lines whose SHA-256 is in MASKED_SHA256 are replaced "
                                  "by a placeholder; the match columns use the unmasked line"}
    (out / "exploratory_exposure_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

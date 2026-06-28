"""Stage 3 (stats half): per-run-file data-quality accounting, persisted to JSON.

Kept separate from parsing (pure string->label) and from the CSV aggregation.
The counts that aggregate.py previously only printed (API errors, unparseable
outputs) are written to a sidecar JSON next to each run file, so they survive
the run instead of scrolling off the terminal.

API errors (the call failed after retries; raw_text is "") and parse failures
(the call returned text that did not yield a label) are counted separately so
each is visible on its own, and broken down per model_name within the file.

    from stats import compute_run_stats, write_stats
    stats = compute_run_stats(Path("data/outputs/llama/runs_llama_mhs.jsonl"))
    write_stats(Path("data/outputs/llama/runs_llama_mhs.jsonl"), stats)
    # -> data/outputs/llama/runs_llama_mhs.stats.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from schema import RunRecord
from parsing import parse_label


def _new() -> dict:
    return {
        "total_rows": 0,
        "api_errors": 0,
        "parsed_ok": 0,
        "parse_failures": 0,
        "parse_failure_reasons": Counter(),
    }


def _tally(bucket: dict, r: RunRecord) -> None:
    bucket["total_rows"] += 1
    if r.error is not None:
        bucket["api_errors"] += 1
        return
    label, reason = parse_label(r.raw_text)
    if label is None:
        bucket["parse_failures"] += 1
        bucket["parse_failure_reasons"][reason] += 1
    else:
        bucket["parsed_ok"] += 1


def _plain(bucket: dict, unique_items: int) -> dict:
    out = dict(bucket)
    out["unique_items"] = unique_items
    out["parse_failure_reasons"] = dict(bucket["parse_failure_reasons"])
    return out


def compute_run_stats(path: Path) -> dict:
    """Tally API errors and parse outcomes for one runs_*.jsonl file."""
    overall = _new()
    by_model: dict[str, dict] = defaultdict(_new)
    items: set[str] = set()
    model_items: dict[str, set] = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = RunRecord.from_json(ln)
            items.add(r.item_id)
            model_items[r.model_name].add(r.item_id)
            _tally(overall, r)
            _tally(by_model[r.model_name], r)
    return {
        "run_file": str(path),
        **_plain(overall, len(items)),
        "by_model": {
            m: _plain(b, len(model_items[m])) for m, b in by_model.items()
        },
    }


def write_stats(run_path: Path, stats: dict) -> Path:
    """Write stats next to the run file as <name>.stats.json. Returns the path."""
    out = Path(run_path).with_suffix(".stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return out

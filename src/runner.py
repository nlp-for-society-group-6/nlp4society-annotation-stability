"""Stage 2 runner: read items.jsonl, run a client over seeds, write run records.

Shared by every model. Key properties:
  * Resumable: skips (item_id, seed) pairs already present in the output file,
    so a 429 crash mid-run loses nothing. Important on a rate-limited free tier.
  * Fail-soft: a call that errors after retries is written as a RunRecord with
    error set and raw_text="", never dropped. Stage 3 can count/inspect failures.
  * Provider-agnostic output: identical RunRecord shape regardless of client.
"""
from __future__ import annotations

import time
import random
from pathlib import Path
from typing import Iterable

from schema import InputItem, RunRecord
from client import Client


def _load_items(path: Path) -> list[InputItem]:
    with open(path, encoding="utf-8") as f:
        return [InputItem.from_json(ln) for ln in f if ln.strip()]


def _already_done(path: Path) -> set[tuple[str, int, str]]:
    """Set of (item_id, seed, model_name) already written, for resumption."""
    done: set[tuple[str, int, str]] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = RunRecord.from_json(ln)
            if r.error is None:
                done.add((r.item_id, r.seed, r.model_name))
    return done


def _is_daily_quota_error(error_text: str) -> bool:
    """True for hard daily quota exhaustion, not ordinary per-minute throttles."""
    text = error_text.lower()
    daily_markers = ("daily", "per day", "requestsperday", "request per day")
    return "quota" in text and any(marker in text for marker in daily_markers)


def _call_with_retry(client: Client, prompt: str, seed: int,
                     max_retries: int = 5):
    """Exponential backoff on any exception (covers 429 rate limits)."""
    delay = 2.0
    last_err = None
    for _ in range(max_retries):
        try:
            t0 = time.perf_counter()
            comp = client.generate(prompt, seed)
            return comp, time.perf_counter() - t0, None, False
        except Exception as e:  # noqa: BLE001  - we want to log everything
            last_err = e
            if _is_daily_quota_error(repr(e)):
                return None, None, repr(e), True
            time.sleep(delay + random.uniform(0, 1))
            delay = min(delay * 2, 60)
    return None, None, repr(last_err), False


def run(client: Client, items: Iterable[InputItem], seeds: list[int],
        out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _already_done(out_path)
    items = list(items)
    total = len(items) * len(seeds)
    n = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for item in items:
            for run_index, seed in enumerate(seeds):
                n += 1
                key = (item.item_id, seed, client.model_name)
                if key in done:
                    continue
                comp, latency, err, quota_exhausted = _call_with_retry(client, item.text, seed)
                if quota_exhausted:
                    print(
                        "daily quota exhausted; stopping. "
                        "Rerun this same command after the quota resets.",
                        flush=True,
                    )
                    return
                rec = RunRecord(
                    item_id=item.item_id,
                    model_name=client.model_name,
                    provider=client.provider,
                    seed=seed,
                    run_index=run_index,
                    raw_text=comp.raw_text if comp else "",
                    logprob=comp.logprob if comp else None,
                    finish_reason=comp.finish_reason if comp else None,
                    latency_s=latency,
                    error=err,
                )
                out.write(rec.to_json() + "\n")
                out.flush()  # crash-safe: each line hits disk immediately
                if n % 20 == 0:
                    print(f"  [{n}/{total}] {client.model_name}", flush=True)

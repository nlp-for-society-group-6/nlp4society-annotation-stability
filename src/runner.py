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
from client import Client, ErrorAction


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


def _call_with_retry(client: Client, prompt: str, seed: int,
                     max_retries: int = 5):
    """Try one (item, seed) call, classifying failures by HTTP status via
    client.classify_error. Returns (completion, latency, error, stop).

    Per failure the client returns an ErrorAction:
      RETRY -> back off (2,4,8,16,32s) and try again, up to max_retries.
      STOP  -> halt the whole run now (auth, or a 429 still failing after all
               retries = the daily cap). No error row is written; resume later.
      FATAL -> this call cannot recover (e.g. bad_request); log an error row and
               let the run continue to the next sample.

    The 429/5xx/network backoff schedule is identical; only the terminal action
    differs: a persistent 429 stops the run, a persistent 5xx/network is logged
    and skipped (so an item-specific failure never deadlocks the whole run).
    The kind string is used only to decide stop-vs-continue; it is not stored.
    """
    delay = 2.0
    last_err = None
    last_kind = None
    for _ in range(max_retries):
        try:
            t0 = time.perf_counter()
            comp = client.generate(prompt, seed)
            return comp, time.perf_counter() - t0, None, False
        except Exception as e:  # noqa: BLE001  - we want to log everything
            last_err = e
            action, last_kind = client.classify_error(e)
            if action is ErrorAction.STOP:
                return None, None, repr(e), True
            if action is ErrorAction.FATAL:
                return None, None, repr(e), False
            time.sleep(delay + random.uniform(0, 1))
            delay = min(delay * 2, 60)
    # retries exhausted: a persistent rate limit is the daily cap -> stop cleanly;
    # anything else (server/network) is logged and the run continues.
    stop = last_kind == "rate_limit"
    return None, None, repr(last_err), stop


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
                comp, latency, err, stop_run = _call_with_retry(client, item.text, seed)
                if stop_run:
                    # No row written for this sample: it stays unprocessed and is
                    # retried on resume. Either a daily rate-limit cap or a bad key.
                    print(
                        "stopping cleanly (rate-limit daily cap or auth error). "
                        "This sample was not written; rerun this same command to "
                        "resume (after the quota resets, if rate-limited).",
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

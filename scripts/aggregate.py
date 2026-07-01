"""Stage 3 (starter): parse raw runs -> per-item stability, merged with
human disagreement, ready for the correlation/tier analysis (person C).

Outputs a tidy CSV: one row per (item_id, model_name) with both signals.
This is intentionally minimal: it gives person C a clean table to start from.

    python scripts/aggregate.py \
        --items data/interim/mhs_items.jsonl \
        --runs data/outputs/runs_groq_llama8b.jsonl data/outputs/runs_mistral.jsonl \
        --out data/outputs/scored.csv
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from schema import InputItem, RunRecord
from parsing import parse_label
from stats import compute_run_stats, write_stats


def label_entropy(labels: list[str]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    p = labels.count("hate") / n
    h = 0.0
    for q in (p, 1 - p):
        if q > 0:
            h -= q * math.log2(q)
    return h


def flip_rate(labels: list[str]) -> float:
    """Fraction of runs disagreeing with the run-level majority label."""
    n = len(labels)
    if n == 0:
        return 0.0
    maj = "hate" if labels.count("hate") > n / 2 else "not_hate"
    return sum(1 for x in labels if x != maj) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/interim/mhs_items.jsonl")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="data/outputs/scored.csv")
    args = ap.parse_args()

    with open(args.items, encoding="utf-8") as f:
        items = {it.item_id: it for it in (InputItem.from_json(l) for l in f if l.strip())}

    # collect one parsed label per (item, model, seed); write a per-file stats
    #  each (item, seed) is counted exactly once instead of inflating n_runs.
    by_key: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    for rp in args.runs:
        rp = Path(rp)
        stats = compute_run_stats(rp)
        sidecar = write_stats(rp, stats)
        print(f"{rp.name}: {stats['parsed_ok']}/{stats['total_rows']} parsed "
              f"({stats['api_errors']} api errors, {stats['parse_failures']} parse fails) "
              f"-> {sidecar.name}")
        with open(rp, encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                r = RunRecord.from_json(ln)
                if r.error is not None:
                    continue
                lab, _ = parse_label(r.raw_text)
                if lab is not None:
                    by_key[(r.item_id, r.model_name)].setdefault(r.seed, lab)

    rows = []
    for (item_id, model), seed_labels in by_key.items():
        labels = list(seed_labels.values())
        it = items.get(item_id)
        if it is None:
            continue
        rows.append({
            "item_id": item_id,
            "model_name": model,
            "n_runs": len(labels),
            "llm_flip_rate": round(flip_rate(labels), 4),
            "llm_output_entropy": round(label_entropy(labels), 4),
            "human_entropy": it.annotator_entropy,
            "human_disagreement_var": it.disagreement_var,
            "disagreement_tier": it.disagreement_tier,
            "dataset": it.dataset,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} (item,model) rows -> {out}")
    print("next: person C computes Spearman rho on "
          "llm_flip_rate vs human_entropy, and compares tiers.")


if __name__ == "__main__":
    main()

"""Stage 1: build canonical items.jsonl from the CREHate corpus.

CREHate ships one row per post with a binary hate label from each of 5 countries
(US, AU, UK, ZA, SG). We treat each country as an annotator and compute the same
disagreement signal as for MHS:
  * annotator_entropy : Shannon entropy over the 5 country labels
  * disagreement_var   : Davani et al. (2022) Disagreement Variation, Eq. 1
  * disagreement_tier  : low/medium/high by entropy terciles (fixed here, once)

Run:
    python scripts/build_crehate_items.py --n 200 --out data/interim/crehate_items.jsonl

--n takes a stratified pilot sample (default: all 1,580 posts).
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import argparse
import math
from pathlib import Path

from schema import InputItem

COUNTRY_COLS = [
    "United_States_Hate",
    "Australia_Hate",
    "United_Kingdom_Hate",
    "South_Africa_Hate",
    "Singapore_Hate",
]


def entropy(labels: list[int]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    p1 = sum(labels) / n
    h = 0.0
    for p in (p1, 1 - p1):
        if p > 0:
            h -= p * math.log2(p)
    return h


def disagreement_variation(labels: list[int]) -> float:
    n = len(labels)
    if n < 2:
        return 0.0
    mismatches = sum(
        1 for i in range(n) for j in range(i + 1, n) if labels[i] != labels[j]
    )
    pairs = n * (n - 1) // 2
    return mismatches / pairs if pairs else 0.0


def assign_tiers(items: list[InputItem]) -> None:
    vals = sorted(it.annotator_entropy for it in items)
    if not vals:
        return
    lo = vals[len(vals) // 3]
    hi = vals[2 * len(vals) // 3]
    for it in items:
        e = it.annotator_entropy
        it.disagreement_tier = "low" if e <= lo else ("high" if e >= hi else "medium")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="pilot sample size; 0 = all")
    ap.add_argument("--out", default="data/interim/crehate_items.jsonl")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("nayeon212/CREHate", "main_data")["total"]

    items: list[InputItem] = []
    for row in ds:
        labels = [int(float(row[col]) > 0.5) for col in COUNTRY_COLS]
        maj = "hate" if sum(labels) > len(labels) / 2 else "not_hate"
        country_votes = {col.replace("_Hate", ""): labels[i]
                         for i, col in enumerate(COUNTRY_COLS)}
        items.append(InputItem(
            item_id=f"crehate-{row['ID']}",
            text=row["Text"],
            dataset="crehate",
            gold_label=maj,
            n_annotators=len(labels),
            annotator_entropy=entropy(labels),
            disagreement_var=disagreement_variation(labels),
            meta={"country_votes": country_votes},
        ))

    assign_tiers(items)

    if args.n and args.n < len(items):
        import random
        random.seed(args.seed)
        per: dict[str, list[InputItem]] = {"low": [], "medium": [], "high": []}
        for it in items:
            per[it.disagreement_tier].append(it)
        k = args.n // 3
        items = sum((random.sample(v, min(k, len(v))) for v in per.values()), [])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(it.to_json() + "\n")
    print(f"wrote {len(items)} items -> {out}")


if __name__ == "__main__":
    main()

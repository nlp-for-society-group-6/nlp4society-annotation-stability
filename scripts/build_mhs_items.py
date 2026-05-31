"""Stage 1: build canonical items.jsonl from the Measuring Hate Speech corpus.

MHS ships on Hugging Face with per-annotator rows (one row per annotator x
comment). We aggregate to one InputItem per comment, computing:
  * annotator_entropy : Shannon entropy over the binary annotator labels
  * disagreement_var   : Davani et al. (2022) Disagreement Variation, Eq. 1
  * disagreement_tier  : low/medium/high by entropy terciles (fixed here, once)

Run:
    python scripts/build_mhs_items.py --n 200 --out data/interim/mhs_items.jsonl

--n takes a stratified pilot sample (default: all). Start small to sanity-check
the whole pipeline before spending rate-limited inference budget.

LABEL COLUMN: `hatespeech` (float, 0 or 1) is the per-annotator binary judgment.
`hate_speech_score` is the IRT-adjusted aggregate — same value for every annotator
row on a given comment, so using it for per-annotator labels produces fake agreement.
This is confirmed against the HF dataset viewer schema.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import argparse
import math
from collections import defaultdict
from pathlib import Path

from schema import InputItem


def entropy(labels: list[int]) -> float:
    """Shannon entropy (bits) over a list of 0/1 annotator labels."""
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
    """Davani et al. (2022) Disagreement Variation (Eq. 1): mean pairwise
    disagreement = fraction of annotator pairs that assigned different labels.
    For binary labels this equals 2*p*(1-p)*n/(n-1)-style; we compute it
    directly as pairwise mismatches over all pairs for transparency."""
    n = len(labels)
    if n < 2:
        return 0.0
    mismatches = 0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            if labels[i] != labels[j]:
                mismatches += 1
    return mismatches / pairs if pairs else 0.0


def assign_tiers(items: list[InputItem]) -> None:
    """In-place: tier by entropy terciles. Fixed once, shared by all models."""
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
    ap.add_argument("--out", default="data/interim/mhs_items.jsonl")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    from datasets import load_dataset  # lazy
    # MHS has no predefined splits; passing split="train" silently returns a
    # tiny shard (~200 rows) instead of the full 135k-row table. Load without
    # a split argument and access the default key.
    raw = load_dataset("ucberkeley-dlab/measuring-hate-speech")
    ds = raw[list(raw.keys())[0]]  # "train" on current HF version

    # aggregate per-comment: collect each annotator's binary hate judgment
    by_comment: dict[str, dict] = defaultdict(lambda: {"labels": [], "text": None})
    for row in ds:
        cid = str(row["comment_id"])
        # `hatespeech` is the per-annotator binary label (0=not hate, 1=hate).
        # `hate_speech_score` is the IRT-adjusted *aggregate* score shared across
        # all annotator rows for the same comment — do NOT use it here.
        lab = int(float(row.get("hatespeech", 0)) > 0.5)
        by_comment[cid]["labels"].append(lab)
        if by_comment[cid]["text"] is None:
            by_comment[cid]["text"] = row.get("text", "")

    items: list[InputItem] = []
    for cid, d in by_comment.items():
        labels = d["labels"]
        if len(labels) < 3:        # need a meaningful annotator pool
            continue
        maj = "hate" if sum(labels) > len(labels) / 2 else "not_hate"
        items.append(InputItem(
            item_id=f"mhs-{cid}",
            text=d["text"],
            dataset="mhs",
            gold_label=maj,
            n_annotators=len(labels),
            annotator_entropy=entropy(labels),
            disagreement_var=disagreement_variation(labels),
            meta={"raw_labels": labels},
        ))

    assign_tiers(items)

    # stratified pilot sample: keep tier balance so all tiers are testable
    if args.n and args.n < len(items):
        import random
        random.seed(args.seed)
        per = {"low": [], "medium": [], "high": []}
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
"""Qualitative analysis: two focused outputs.

1. high_disagreement_instability_{dataset}.csv
   Items where human disagreement is high (tier='high') AND at least one
   generative model flips. Each row has a keyword-suggested content type
   and a blank content_type column for manual annotation.

   Content type categories:
     irony_humour       -- jokes, sarcasm, dark humour, emoji laughter
     political_speech   -- politicians, policy, immigration, race politics
     explicit_no_target -- explicit sexual/violent content without a hate target
     reclaimed_language -- slurs or taboo terms used in-group / non-derogatorily
     implicit_coded     -- coded language, dog-whistles, requires world knowledge
     other              -- does not fit cleanly into above

2. cross_model_instability_{dataset}.csv
   All items where any generative model flips. Shows each model's flip rate
   and majority label side by side, plus a flip_pattern column summarising
   which models flipped and which were stable.

Run:
    python scripts/qualitative_analysis.py
    python scripts/qualitative_analysis.py --out data/outputs/qualitative_analysis
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from parsing import parse_label
from schema import RunRecord

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

GENERATIVE = ["gemini", "mistral", "llama"]

# Keyword heuristics for content-type suggestions.
# Each entry is (category, list-of-regex-patterns).
# Matched top-to-bottom; first match wins. Falls back to "other".
_HEURISTICS: list[tuple[str, list[str]]] = [
    ("irony_humour", [
        r"\blol\b", r"\bhaha\b", r"😂", r"🤣", r"\bjoking\b", r"\bjoke\b",
        r"\bsarcas", r"\bironic", r"\bjust kidding\b", r"\bsmh\b",
        r"😆", r"🙃", r"\bfunny\b", r"\bhumour\b", r"\bhumor\b",
        r"trampoline.*wheelchair|wheelchair.*trampoline",
        r"what'?s.*blue.*doesn'?t|doesn'?t.*like.*sex",
    ]),
    ("explicit_no_target", [
        r"\bpussy\b", r"\bdick\b", r"\bass\b", r"\bfuck(?:ing)?\b",
        r"\bsex\b", r"\blick\b", r"\basshole\b", r"\bbitch\b",
        r"\bshit\b", r"\bcunt\b", r"\btits\b", r"\bnaked\b",
    ]),
    ("reclaimed_language", [
        r"\bnigga\b", r"\bnigger\b", r"\bqueer\b", r"\bdyke\b",
        r"\bfag\b", r"\bcrip\b", r"\bwhore\b",
    ]),
    ("political_speech", [
        r"\bimmigr", r"\bdeport", r"\bwall\b", r"\bborder\b",
        r"\bmuslim\b", r"\bisrael\b", r"\bpalest", r"\btrump\b",
        r"\brepublican\b", r"\bdemocrat\b", r"\bconservat",
        r"\bliberal\b", r"\balt.right\b", r"\bwhite suprem",
        r"\bblack lives\b", r"\bblm\b", r"\bisis\b", r"\bterror",
        r"\biran\b", r"\bchina\b", r"\bccp\b", r"\bwoke\b",
        r"\bfeminism\b", r"\bgender\b", r"\btransgender\b",
    ]),
    ("implicit_coded", [
        r"\bthose people\b", r"\bthese people\b", r"\btheir kind\b",
        r"\byou people\b", r"\ball of them\b", r"\bof course\b.*\bthey\b",
        r"\bmonkey\b", r"\bape\b", r"\bvermin\b", r"\bparasit",
        r"\binvasion\b", r"\breplac", r"\bgreat replacement\b",
        r"\bboer\b", r"\bsubhuman\b",
    ]),
]


def suggest_type(text: str) -> str:
    t = text.lower()
    for category, patterns in _HEURISTICS:
        for pat in patterns:
            if re.search(pat, t, re.IGNORECASE):
                return category
    return "other"


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_items(path: Path) -> dict[str, dict]:
    items: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                items[d["item_id"]] = d
    return items


def load_model_stats(run_path: Path) -> dict[str, dict]:
    """Parse run JSONL → {item_id: {majority_label, flip_rate, n_runs}}."""
    if not run_path.exists():
        return {}
    label_counts: dict[str, Counter] = defaultdict(Counter)
    with open(run_path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = RunRecord.from_json(ln)
            lab, _ = parse_label(r.raw_text)
            if lab:
                label_counts[r.item_id][lab] += 1

    stats: dict[str, dict] = {}
    for item_id, counts in label_counts.items():
        total = sum(counts.values())
        majority = counts.most_common(1)[0][0]
        stats[item_id] = {
            "majority_label": majority,
            "flip_rate": round(1.0 - counts[majority] / total, 4),
            "n_runs": total,
        }
    return stats


# ---------------------------------------------------------------------------
# output 1: high-disagreement + high-instability examples
# ---------------------------------------------------------------------------

def build_high_disagreement_instability(
    items: dict[str, dict],
    stats_by_model: dict[str, dict[str, dict]],
    dataset: str,
) -> list[dict]:
    """Rows where tier='high' AND any generative model flips."""
    rows = []
    for item_id, it in items.items():
        if it.get("disagreement_tier") != "high":
            continue

        per_model = {
            m: stats[item_id]
            for m, stats in stats_by_model.items()
            if item_id in stats
        }
        models_flipping = [m for m, s in per_model.items() if s["flip_rate"] > 0]
        if not models_flipping:
            continue

        text = it["text"]
        row = {
            "item_id":          item_id,
            "dataset":          dataset,
            "human_entropy":    round(float(it.get("annotator_entropy", 0)), 4),
            "disagreement_var": round(float(it.get("disagreement_var", 0)), 4),
            "n_annotators":     it.get("n_annotators"),
            "gold_label":       it.get("gold_label"),
            "models_flipping":  "|".join(sorted(models_flipping)),
            "n_models_flipping": len(models_flipping),
            "suggested_type":   suggest_type(text),
            "content_type":     "",      # ← fill in manually
            "notes":            "",      # ← fill in manually
        }
        for model in GENERATIVE:
            s = per_model.get(model, {})
            row[f"{model}_flip_rate"] = s.get("flip_rate", "")
            row[f"{model}_label"]     = s.get("majority_label", "")
        row["text"] = text
        rows.append(row)

    rows.sort(key=lambda r: (-r["n_models_flipping"], -r["human_entropy"]))
    return rows


# ---------------------------------------------------------------------------
# output 2: cross-model instability comparison
# ---------------------------------------------------------------------------

def build_cross_model_instability(
    items: dict[str, dict],
    stats_by_model: dict[str, dict[str, dict]],
    dataset: str,
) -> list[dict]:
    """All items where any generative model flips, with per-model breakdown
    and a flip_pattern column summarising which models flipped vs. stayed stable."""
    rows = []
    for item_id, it in items.items():
        per_model = {
            m: stats[item_id]
            for m, stats in stats_by_model.items()
            if item_id in stats
        }
        models_flipping = [m for m, s in per_model.items() if s["flip_rate"] > 0]
        if not models_flipping:
            continue

        # flip_pattern: e.g. "gemini=stable | mistral=FLIP | llama=FLIP"
        pattern_parts = []
        for m in GENERATIVE:
            if m not in per_model:
                pattern_parts.append(f"{m}=missing")
            elif per_model[m]["flip_rate"] > 0:
                pattern_parts.append(f"{m}=FLIP({per_model[m]['flip_rate']})")
            else:
                pattern_parts.append(f"{m}=stable({per_model[m]['majority_label']})")
        flip_pattern = " | ".join(pattern_parts)

        # do the models that ARE stable agree on their label?
        stable_labels = {m: per_model[m]["majority_label"]
                         for m in GENERATIVE
                         if m in per_model and per_model[m]["flip_rate"] == 0}
        label_agreement = len(set(stable_labels.values())) <= 1

        row = {
            "item_id":           item_id,
            "dataset":           dataset,
            "tier":              it.get("disagreement_tier"),
            "human_entropy":     round(float(it.get("annotator_entropy", 0)), 4),
            "gold_label":        it.get("gold_label"),
            "n_models_flipping": len(models_flipping),
            "models_flipping":   "|".join(sorted(models_flipping)),
            "flip_pattern":      flip_pattern,
            "stable_label_agreement": label_agreement,
            "suggested_type":    suggest_type(it["text"]),
        }
        for model in GENERATIVE:
            s = per_model.get(model, {})
            row[f"{model}_flip_rate"] = s.get("flip_rate", "")
            row[f"{model}_label"]     = s.get("majority_label", "")
        row["text"] = it["text"]
        rows.append(row)

    rows.sort(key=lambda r: (-r["n_models_flipping"], -r["human_entropy"]))
    return rows


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"  (no rows for {path.name})")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows)} rows -> {path}")


def print_summary(label: str,
                  high_inst: list[dict],
                  cross_model: list[dict]) -> None:
    print(f"\n=== {label} ===")
    print(f"High-disagreement + instability items: {len(high_inst)}")
    if high_inst:
        type_counts = Counter(r["suggested_type"] for r in high_inst)
        for t, n in type_counts.most_common():
            print(f"  {t}: {n}")

    print(f"Any-flip items (cross-model): {len(cross_model)}")
    if cross_model:
        pattern_counts = Counter(r["models_flipping"] for r in cross_model)
        for p, n in pattern_counts.most_common(5):
            print(f"  [{p}]: {n}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

MHS_RUNS = {
    "gemini":  Path("data/outputs/gemini/runs_mhs_gemini_flash.jsonl"),
    "mistral": Path("data/outputs/mistral/runs_mistral_mhs.jsonl"),
    "llama":   Path("data/outputs/llama/runs_llama_mhs.jsonl"),
}
CREHATE_RUNS = {
    "gemini":  Path("data/outputs/gemini/runs_crehate_gemini_flash.jsonl"),
    "mistral": Path("data/outputs/mistral/runs_mistral_crehate.jsonl"),
    "llama":   Path("data/outputs/llama/runs_llama_crehate.jsonl"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/outputs/qualitative_analysis")
    args = ap.parse_args()
    out = Path(args.out)

    print("Loading items...")
    mhs_items     = load_items(Path("data/interim/mhs_items.jsonl"))
    crehate_items = load_items(Path("data/interim/crehate_items.jsonl"))

    print("Loading model runs...")
    mhs_stats = {m: load_model_stats(p) for m, p in MHS_RUNS.items() if p.exists()}
    crehate_stats = {m: load_model_stats(p) for m, p in CREHATE_RUNS.items() if p.exists()}

    print(f"  MHS models loaded:     {list(mhs_stats)}")
    print(f"  CREHate models loaded: {list(crehate_stats)}")

    # MHS
    mhs_high  = build_high_disagreement_instability(mhs_items, mhs_stats, "mhs")
    mhs_cross = build_cross_model_instability(mhs_items, mhs_stats, "mhs")
    write_csv(out / "high_disagreement_instability_mhs.csv", mhs_high)
    write_csv(out / "cross_model_instability_mhs.csv", mhs_cross)
    print_summary("MHS", mhs_high, mhs_cross)

    # CREHate
    cre_high  = build_high_disagreement_instability(crehate_items, crehate_stats, "crehate")
    cre_cross = build_cross_model_instability(crehate_items, crehate_stats, "crehate")
    write_csv(out / "high_disagreement_instability_crehate.csv", cre_high)
    write_csv(out / "cross_model_instability_crehate.csv", cre_cross)
    print_summary("CREHate", cre_high, cre_cross)


if __name__ == "__main__":
    main()

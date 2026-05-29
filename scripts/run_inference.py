"""Stage 2: run inference for ONE model over items.jsonl.

Pass --client to select the provider. Resumable and rate-limit-safe via the shared runner.

Examples:
    python scripts/run_inference.py --client groq \
        --model llama-3.1-8b-instant --seeds 5 \
        --items data/interim/items.jsonl \
        --out data/outputs/runs_groq_llama8b.jsonl

    # 10 seeds if the 5-seed run was fast enough under the rate cap:
    python scripts/run_inference.py --client groq --seeds 10 ...
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import argparse
from pathlib import Path

from clients import build_client
from schema import InputItem
from runner import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=True, help="registered client key, e.g. groq")
    ap.add_argument("--model", default=None, help="override model_name for the client")
    ap.add_argument("--items", default="data/interim/items.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (1..N)")
    ap.add_argument("--seed-base", type=int, default=1000,
                    help="seeds will be seed-base + 0..seeds-1, shared across models")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    kwargs = {"temperature": args.temperature}
    if args.model:
        kwargs["model_name"] = args.model
    client = build_client(args.client, **kwargs)

    seeds = [args.seed_base + i for i in range(args.seeds)]

    with open(args.items, encoding="utf-8") as f:
        items = [InputItem.from_json(ln) for ln in f if ln.strip()]

    print(f"running {client.model_name} ({client.provider}) "
          f"on {len(items)} items x {len(seeds)} seeds")
    run(client, items, seeds, Path(args.out))
    print("done.")


if __name__ == "__main__":
    main()

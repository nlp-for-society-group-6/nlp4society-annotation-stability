"""Smoke test: runs the full Stage 2 -> Stage 3 path with a fake client,
no API key, no network. Lets anyone verify the pipeline before touching Groq.

    python scripts/selftest.py
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import random
import tempfile
from pathlib import Path

from schema import InputItem, RunRecord
from client import Client, Completion
from runner import run
from parsing import parse_label


class FakeClient(Client):
    """Returns hate/not_hate with a per-item bias, so flip rate varies.
    Mimics a small model: mostly clean JSON, occasionally messy."""
    provider = "fake"

    def __init__(self):
        super().__init__("fake-model", temperature=0.7)

    def generate(self, prompt: str, seed: int) -> Completion:
        random.seed(hash(prompt) ^ seed)
        # ambiguous prompts (contain '?') flip more often
        p_hate = 0.5 if "?" in prompt else 0.9
        lab = "hate" if random.random() < p_hate else "not_hate"
        if random.random() < 0.1:                 # 10% messy output
            return Completion(raw_text=f"```json\n{{\"label\": \"{lab}\"}}\n```")
        return Completion(raw_text=f'{{"label": "{lab}"}}')


def main() -> None:
    items = [
        InputItem("t-1", "this is fine", "fake", annotator_entropy=0.1,
                  disagreement_tier="low"),
        InputItem("t-2", "is this ok?", "fake", annotator_entropy=0.9,
                  disagreement_tier="high"),
    ]
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "runs.jsonl"
        run(FakeClient(), items, seeds=[1, 2, 3, 4, 5], out_path=out)
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 10, f"expected 10 run records, got {len(lines)}"

        # parsing
        ok = sum(1 for ln in lines if parse_label(RunRecord.from_json(ln).raw_text)[0])
        assert ok == 10, f"all 10 should parse, got {ok}"

        # resumption: re-run should add nothing
        run(FakeClient(), items, seeds=[1, 2, 3, 4, 5], out_path=out)
        assert len(out.read_text().strip().splitlines()) == 10, "resume should be a no-op"

    print("OK: pipeline runs, all outputs parse, resumption works.")


if __name__ == "__main__":
    main()

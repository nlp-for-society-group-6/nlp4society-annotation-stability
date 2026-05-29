"""Canonical data contracts for the whole pipeline.

There are exactly two schemas everyone must respect:

  InputItem    : Stage 1 output. One per text. Provider-agnostic, model-agnostic.
  RunRecord    : Stage 2 output. One per (model, item, seed). Raw, unparsed.

Stage 3 reads RunRecords and never touches the providers. Keeping the
RunRecord shape identical across providers is what makes parsing uniform,
so do not add provider-specific fields here. Put provider quirks in the
client, not in the schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class InputItem:
    """One text to be annotated, plus its precomputed human-disagreement signal.

    item_id    : stable unique id. Use "{dataset}-{native_id}" so ids never
                 collide across datasets (e.g. "mhs-00417", "crehate-1203").
    text       : the post to classify, cleaned but otherwise verbatim.
    dataset    : "mhs" or "crehate".
    gold_label : majority-vote binary label ("hate"/"not_hate") if you want it
                 for accuracy side-analysis. Optional; disagreement is the point.
    n_annotators        : how many humans labelled this item.
    annotator_entropy   : Shannon entropy over the annotator label distribution.
    disagreement_var    : Davani et al. (2022) Disagreement Variation, Eq. 1.
    disagreement_tier   : "low" | "medium" | "high", assigned in Stage 1 so the
                          tiering is fixed once and shared by all models.
    meta       : free dict for dataset-specific extras (culture tags for CREHate,
                 per-annotator raw labels, etc). Never read by Stage 2.
    """
    item_id: str
    text: str
    dataset: str
    gold_label: Optional[str] = None
    n_annotators: Optional[int] = None
    annotator_entropy: Optional[float] = None
    disagreement_var: Optional[float] = None
    disagreement_tier: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "InputItem":
        return InputItem(**json.loads(line))


@dataclass
class RunRecord:
    """One inference run. The atomic unit of Stage 2 output.

    Written one-per-line to raw_outputs.jsonl. Stage 3 is the only consumer.

    item_id     : matches InputItem.item_id.
    model_name  : canonical model string (e.g. "llama-3.1-8b-instant").
    provider    : "groq" | "together" | "vllm" | "hf" ... for provenance.
    seed        : the seed used for THIS run.
    run_index   : 0..N-1, the repetition index for this (item, model) pair.
    prompt_variant : "base" for now. Reserved for paraphrase sensitivity later;
                     keeping the field means adding paraphrases needs no schema change.
    raw_text    : the model's raw string output, stored VERBATIM and untouched.
                  Parsing happens in Stage 3, never here. This is the artifact
                  the qualitative RQ3 pass reads, so it must be lossless.
    logprob     : log-prob of the chosen label token if the provider exposes it,
                  else None. Groq returns None; Together/vLLM may populate it.
    finish_reason, latency_s, error : provenance / debugging. error is set (and
                  raw_text left "") when a call failed after retries, so a failed
                  run is still a logged row rather than a silent gap.
    """
    item_id: str
    model_name: str
    provider: str
    seed: int
    run_index: int
    raw_text: str
    prompt_variant: str = "base"
    logprob: Optional[float] = None
    finish_reason: Optional[str] = None
    latency_s: Optional[float] = None
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "RunRecord":
        return RunRecord(**json.loads(line))

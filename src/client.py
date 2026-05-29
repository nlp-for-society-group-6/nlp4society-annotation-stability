"""Provider abstraction. This is the ONLY thing a teammate implements.

To add a model:
    1. subclass Client
    2. implement generate(prompt, seed) -> Completion
    3. register it in clients.py REGISTRY

Everything else (seed loops, retries, output format, parsing) is shared.
The interface is deliberately tiny: one method, one return type.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import abc


# The shared system prompt. Frozen for cross-model consistency.
# Do NOT edit per-model: identical prompt is what makes the comparison fair.
SYSTEM_PROMPT = (
    "You are annotating whether a social media post constitutes hate speech. "
    "Hate speech is content that attacks, threatens, demeans, or dehumanizes a person or "
    "group based on protected attributes such as race, ethnicity, national origin, religion, "
    "sex, gender identity, sexual orientation, disability, or serious disease. "
    'Respond with ONLY a JSON object: {"label": "hate"} or {"label": "not_hate"}. '
    "No explanation."
)


@dataclass
class Completion:
    """What a single provider call returns. Maps onto RunRecord fields."""
    raw_text: str
    logprob: Optional[float] = None
    finish_reason: Optional[str] = None


class Client(abc.ABC):
    """Base provider client.

    Subclasses set `provider` and `model_name` and implement `generate`.
    `temperature` is fixed non-zero at the base class level so every model
    is sampled under the same stochastic regime; stability is meaningless at
    temperature 0 because the model is (near-)deterministic by construction.
    """
    provider: str = "base"

    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature

    @abc.abstractmethod
    def generate(self, prompt: str, seed: int) -> Completion:
        """Run one inference at the given seed. Raise on failure;
        the runner handles retries and logs errors as RunRecords."""
        raise NotImplementedError

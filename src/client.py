"""Provider abstraction.

To add a model:
    1. subclass Client
    2. implement generate(prompt, seed) -> Completion
    3. register it in clients.py REGISTRY

Everything else (seed loops, retries, output format, parsing) is shared.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import abc


class ErrorAction(Enum):
    """How the runner should treat a failed generate() call."""
    RETRY = "retry"   # transient / per-minute throttle: back off and try again
    STOP = "stop"     # halt the whole run cleanly and resume later (cap / auth)
    FATAL = "fatal"   # this call cannot recover: log it and skip to the next


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

    def classify_error(self, exc: Exception) -> Tuple[ErrorAction, str]:
        """Map a generate() exception to (action, kind) using the HTTP status
        code only: a number the SDK already carries, not the message wording.

        `kind` is a stable category stored on the error RunRecord so Stage 3 can
        count error types without ever re-parsing exception strings.

        Default covers OpenAI-compatible SDK errors (Groq, Together). Providers
        whose exceptions lack `status_code` fall through to ("retry", "network").
        """
        status = getattr(exc, "status_code", None)
        if status == 429:
            return ErrorAction.RETRY, "rate_limit"
        if status in (401, 403):
            return ErrorAction.STOP, "auth"          # bad key: every call fails
        if isinstance(status, int) and 400 <= status < 500:
            return ErrorAction.FATAL, "bad_request"  # item-specific, won't recover
        if isinstance(status, int) and 500 <= status < 600:
            return ErrorAction.RETRY, "server"
        return ErrorAction.RETRY, "network"          # timeout / connection / unknown

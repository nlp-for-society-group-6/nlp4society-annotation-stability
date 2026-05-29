"""Concrete provider clients + the registry.

All included clients use the OpenAI-compatible surface; only base_url, api key
env var, and logprob handling differ. Add new providers by subclassing Client.
"""
from __future__ import annotations

import os
from client import Client, Completion, SYSTEM_PROMPT


class GroqClient(Client):
    """Groq free tier. OpenAI SDK compatible via base_url override.

    Notes that matter for this project:
      * Groq does NOT support logprobs (returns null), so logprob stays None.
        Stability on Groq is label-only: flip rate / entropy across seeds.
      * Free tier is rate-limited (429s). The runner adds backoff; here we
        just make the call. Check console.groq.com for your live daily cap.
      * `seed` is accepted and combined with system_fingerprint for
        best-effort determinism, but sampling is still stochastic at temp>0,
        which is exactly what we want to measure.
    """
    provider = "groq"

    def __init__(self, model_name: str = "llama-3.1-8b-instant", temperature: float = 0.7):
        super().__init__(model_name, temperature)
        from openai import OpenAI  # lazy import so the repo loads without the dep
        self._client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        )

    def generate(self, prompt: str, seed: int) -> Completion:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            seed=seed,
            max_tokens=20,                       # label JSON is tiny; cap cost
            response_format={"type": "json_object"},  # nudge clean JSON
        )
        choice = resp.choices[0]
        return Completion(
            raw_text=choice.message.content or "",
            logprob=None,                        # unavailable on Groq
            finish_reason=choice.finish_reason,
        )


class TogetherClient(Client):
    """Optional fallback that DOES expose logprobs. Not needed for the
    pilot; included so the logprob signal is reachable without redesign.
    Uses $5 free signup credit. Fill in if/when you want graded confidence."""
    provider = "together"

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct-Turbo",
                 temperature: float = 0.7):
        super().__init__(model_name, temperature)
        from openai import OpenAI
        self._client = OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.environ["TOGETHER_API_KEY"],
        )

    def generate(self, prompt: str, seed: int) -> Completion:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            seed=seed,
            max_tokens=20,
            logprobs=True,
        )
        choice = resp.choices[0]
        lp = None
        try:
            lp = choice.logprobs.content[0].logprob  # first generated token
        except (AttributeError, IndexError, TypeError):
            lp = None
        return Completion(
            raw_text=choice.message.content or "",
            logprob=lp,
            finish_reason=choice.finish_reason,
        )


# Key = the --client CLI value passed to run_inference.py.
REGISTRY = {
    "groq": GroqClient,
    "together": TogetherClient,
    # "mistral": MistralClient,    # nico
    # "hatebert": HateBertClient,  # aradhana (encoder baseline; see note in README)
}


def build_client(name: str, **kwargs) -> Client:
    if name not in REGISTRY:
        raise KeyError(f"unknown client '{name}'. registered: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)

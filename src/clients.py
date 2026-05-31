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
    
class GeminiClient(Client):
    """Google Gemini via the native google-genai SDK. Free AI Studio tier.

    NOTE on shape: unlike GroqClient/TogetherClient, this client does NOT use
    the OpenAI SDK. Gemini's OpenAI-compatibility layer silently strips
    `seed` and `logprobs` as unknown fields, which makes seed-variation
    experiments meaningless. The native SDK accepts `seed` properly.

    Notes that matter for this project:
      * Free tier rate limits: ~10-15 RPM and ~1,500 RPD per model.
        Each Gemini model has its own daily bucket.
      * `seed` is best-effort (same caveat as Groq/Together) but is honored.
      * Logprobs are gated off the free tier as of mid-2026 — logprob stays None.
      * `gemini-2.5-flash-lite` has thinking off by default. For 2.5-flash you
        would want thinking_config=ThinkingConfig(thinking_budget=0) explicitly.
    """
    provider = "gemini"

    def __init__(self, model_name: str = "gemini-2.5-flash-lite", temperature: float = 0.7):
        super().__init__(model_name, temperature)
        from google import genai
        self._genai = genai
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate(self, prompt: str, seed: int) -> Completion:
        from google.genai import types
        resp = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=self.temperature,
                seed=seed,
                max_output_tokens=20,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        finish = None
        try:
            finish = str(resp.candidates[0].finish_reason)
        except (AttributeError, IndexError):
            pass
        return Completion(
            raw_text=resp.text or "",
            logprob=None,
            finish_reason=finish,
        )

# Key = the --client CLI value passed to run_inference.py.
REGISTRY = {
    "groq": GroqClient,
    "together": TogetherClient,
    "gemini": GeminiClient,        # add this line
    # "mistral": MistralClient,    # nico
    # "hatebert": HateBertClient,  # aradhana (encoder baseline; see note in README)
}


def build_client(name: str, **kwargs) -> Client:
    if name not in REGISTRY:
        raise KeyError(f"unknown client '{name}'. registered: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)

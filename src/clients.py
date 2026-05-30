"""Concrete provider clients + the registry.

All included clients use the OpenAI-compatible surface; only base_url, api key
env var, and logprob handling differ. Add new providers by subclassing Client.
"""
from __future__ import annotations

import json as _json
import math
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


class HateXplainClient(Client):
    """BERT-HateXplain encoder baseline. Deterministic — seed is ignored.

    NOT HateBERT (GroNLP/hateBERT, pre-trained on Reddit).
    This is standard BERT fine-tuned on the HateXplain dataset (Gab + Twitter).
    Its 3-class output (Hatespeech / Offensive / Normal) is collapsed to binary
    via collapse_rule:
      "hate_or_offensive" (default) — treats Offensive as hate (matches MHS framing)
      "hate_only"                   — restricts to Hatespeech class only

    SYSTEM_PROMPT is not used — this is a classifier, not a generative model.

    logprob is log(P(binary_label)) from the model's softmax — a richer confidence
    signal than the LLM providers can give.

    Because this model is deterministic, run Stage 2 with --seeds 1.
    flip_rate and output_entropy in scored.csv will both be 0, which is the expected
    result for a deterministic baseline.
    """
    provider = "hf"

    _HATE_NAMES_BY_RULE = {
        "hate_only":         {"hate speech", "hatespeech", "hate"},
        "hate_or_offensive": {"hate speech", "hatespeech", "hate", "offensive"},
    }

    def __init__(self, model_name: str = "Hate-speech-CNERG/bert-base-uncased-hatexplain",
                 temperature: float = 0.7, collapse_rule: str = "hate_or_offensive"):
        super().__init__(model_name, temperature)
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = (
            AutoModelForSequenceClassification
            .from_pretrained(model_name)
            .to(self._device)
            .eval()
        )

        id2label = {int(k): v for k, v in self._model.config.id2label.items()}
        assert self._tok.vocab_size == self._model.config.vocab_size, (
            f"Vocab mismatch: tokenizer {self._tok.vocab_size} "
            f"vs model {self._model.config.vocab_size}"
        )
        hate_names = self._HATE_NAMES_BY_RULE[collapse_rule]
        self._hate_idxs = {
            i for i, name in id2label.items()
            if name.strip().lower() in hate_names
        }
        assert self._hate_idxs, (
            f"No class names matched rule '{collapse_rule}'. id2label: {id2label}"
        )

    def generate(self, prompt: str, seed: int) -> Completion:
        with self._torch.no_grad():
            enc = self._tok(
                [prompt], padding=True, truncation=True,
                max_length=256, return_tensors="pt",
            ).to(self._device)
            probs = (
                self._torch.softmax(self._model(**enc).logits[0], dim=0)
                .cpu()
                .tolist()
            )

        p_hate = sum(probs[i] for i in self._hate_idxs)
        label = "hate" if p_hate >= 0.5 else "not_hate"
        lp = math.log(p_hate if label == "hate" else 1.0 - p_hate)

        return Completion(
            raw_text=_json.dumps({"label": label}),
            logprob=lp,
            finish_reason="stop",
        )


# Key = the --client CLI value passed to run_inference.py.
REGISTRY = {
    "groq": GroqClient,
    "together": TogetherClient,
    "hatexplain": HateXplainClient, #aradhana 
    # "mistral": MistralClient,    # nico
}


def build_client(name: str, **kwargs) -> Client:
    if name not in REGISTRY:
        raise KeyError(f"unknown client '{name}'. registered: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)

# annotation-stability

Research pipeline: does LLM output stability across random seeds reflect human annotator disagreement in hate speech detection?

## Pipeline

Five stages across two datasets (MHS and CREHate):

```
Stage 1a  scripts/build_mhs_items.py       →  data/interim/mhs_items.jsonl
Stage 1b  scripts/build_crehate_items.py   →  data/interim/crehate_items.jsonl
Stage 2   scripts/run_inference.py         →  data/outputs/<model>/runs_<dataset>_<model>.jsonl
Stage 3   scripts/aggregate.py             →  data/outputs/<model>/scored_<dataset>.csv
Stage 4   scripts/quantitative_analysis.py →  data/outputs/quantitative_analysis/
Stage 5   scripts/qualitative_analysis.py  →  data/outputs/qualitative_analysis/
```

Scored CSVs (Stage 3) contain per `(item, model)`: `llm_flip_rate`, `llm_output_entropy`, `human_entropy`, `human_disagreement_var`, `disagreement_tier`.

Stage 4 runs statistical tests (Spearman ρ, Kruskal-Wallis, Mann-Whitney U) and produces plots across all models and datasets. Stage 5 produces CSVs for manual qualitative inspection of high-disagreement and cross-model-unstable items.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in your provider API key(s)
python scripts/selftest.py    # smoke test, no API key needed
```

## Run

```bash
# Stage 1 — build stratified samples from each dataset
python scripts/build_mhs_items.py --n 200 --out data/interim/mhs_items.jsonl
python scripts/build_crehate_items.py --n 200 --out data/interim/crehate_items.jsonl

# Stage 2 — run your model over the items (repeat per model x dataset)
python scripts/run_inference.py \
    --client <name> --model <model-id> \
    --seeds 5 --items data/interim/mhs_items.jsonl \
    --out data/outputs/<name>/runs_mhs_<name>.jsonl

# Stage 3 — merge runs into a scored table (repeat per model x dataset)
python scripts/aggregate.py \
    --items data/interim/mhs_items.jsonl \
    --runs data/outputs/<name>/runs_mhs_<name>.jsonl \
    --out data/outputs/<name>/scored_mhs.csv

# Stage 4 — statistical analysis across all models and datasets
python scripts/quantitative_analysis.py

# Stage 5 — qualitative inspection CSVs
python scripts/qualitative_analysis.py
```

Stage 2 is resumable: rerunning the same command skips already-completed (item, seed) pairs, so a crash or rate-limit mid-run loses nothing.

## Adding a model

Subclass `Client` in `src/clients.py` and register it in `REGISTRY`:

```python
class MyClient(Client):
    provider = "myprovider"

    def __init__(self, model_name="...", temperature=0.7):
        super().__init__(model_name, temperature)
        # initialise your SDK here

    def generate(self, prompt: str, seed: int) -> Completion:
        # call your model with SYSTEM_PROMPT + prompt
        return Completion(raw_text=..., logprob=None, finish_reason=...)
```

Then add an entry to `REGISTRY` at the bottom of `src/clients.py`:

```python
REGISTRY = {
    "groq": GroqClient,
    "together": TogetherClient,
    "gemini": GeminiClient,
    "hatexplain": HateXplainClient,
    "mistral": MistralClient,
    "myprovider": MyClient,   # add this line
}
```

Then run Stage 2 with `--client myprovider`.

`SYSTEM_PROMPT` (defined in `src/client.py`) must stay identical across all models — it is the fairness condition for the comparison. **Exception: encoder/classifier models** (e.g. HateXplainClient) do not use `SYSTEM_PROMPT` at all; the runner passes `item.text` directly to `generate`, which feeds it straight to the tokenizer.

If your model does not expose logprobs, set `logprob=None`; stability is then label-only (flip rate + entropy across seeds).

### Encoder / classifier baselines

Deterministic classifiers (no sampling) always produce the same output for a given text. Run Stage 2 with `--seeds 1` to avoid redundant inference. `llm_flip_rate` and `llm_output_entropy` in `scored.csv` will be 0 for these models by construction — that is the expected result, not a bug. Use the baseline for accuracy comparisons against `gold_label`, not for the instability analysis.

The registered encoder baseline:

| `--client` | Model | Notes |
|---|---|---|
| `hatexplain` | `Hate-speech-CNERG/bert-base-uncased-hatexplain` | Standard BERT fine-tuned on HateXplain (Gab + Twitter). **Not** HateBERT (`GroNLP/hateBERT`), which is a different model. 3-class output collapsed to binary; default rule: `hate_or_offensive`. `logprob` is `log(P(binary_label))` from the softmax. |

```bash
python scripts/run_inference.py \
    --client hatexplain --seeds 1 \
    --items data/interim/mhs_items.jsonl \
    --out data/outputs/hatexplain/runs_hatexplain_mhs.jsonl
```

## Schemas

Two dataclasses in `src/schema.py` are the shared contract:

- `InputItem` — one text with its human-disagreement scores and tier, built once in Stage 1.
- `RunRecord` — one (model, item, seed) inference result; raw output stored verbatim. Parsing happens only in Stage 3 so it can be revised without re-running inference.

Do not add provider-specific fields to either schema.

## Data directories and output artifacts

```
data/interim/                        mhs_items.jsonl, crehate_items.jsonl (Stage 1 output)
data/outputs/<model>/                runs_*.jsonl and scored_*.csv (Stages 2–3 output)
data/outputs/quantitative_analysis/  stats tables and plots (Stage 4 output)
data/outputs/qualitative_analysis/   inspection CSVs (Stage 5 output)
```

All are gitignored; scripts create them automatically.

### Stage 4 outputs (`data/outputs/quantitative_analysis/`)

| File | Contents |
|---|---|
| `descriptive_stats.csv` | Mean ± std `llm_flip_rate` per model × tier × dataset |
| `spearman.csv` | Spearman ρ between `llm_flip_rate` and `human_entropy`, per model × dataset |
| `kruskal_wallis.csv` | Kruskal-Wallis H-test of flip rate across tiers, per model × dataset |
| `mann_whitney.csv` | Pairwise Mann-Whitney U comparisons between tiers, per model × dataset |
| `flip_rate_by_tier.png` | Box plots of flip rate by disagreement tier, per model |
| `violin_strip.png` | Violin + strip plots of flip rate distribution, per model |
| `scatter.png` | Scatter of `llm_flip_rate` vs `human_entropy`, per model × dataset |
| `spearman_bars.png` | Bar chart of Spearman ρ values across models and datasets |

### Stage 5 outputs (`data/outputs/qualitative_analysis/`)

| File | Contents |
|---|---|
| `high_disagreement_instability_<dataset>.csv` | Items where human tier is `high` and at least one generative model flips; includes a keyword-suggested `content_type` column and a blank column for manual annotation |
| `cross_model_instability_<dataset>.csv` | All items where any generative model flips; shows each model's flip rate and majority label side by side, with a `flip_pattern` column summarising which models flipped |

## Datasets

**[Measuring Hate Speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)** (MHS, Sachdeva et al. 2022): per-annotator labels for ~39k comments, aggregated to one `InputItem` per comment with Shannon entropy and disagreement variation scores. Items are binned into low/medium/high disagreement tiers once in Stage 1 and the tiers are shared across all models.

**[CREHate](https://github.com/nlee0212/CREHate)** (Lee et al. 2023): one binary hate label per post from each of five countries (US, AU, UK, ZA, SG). Each country is treated as an annotator, and the same disagreement signals (Shannon entropy, disagreement variation, tier) are computed as for MHS.

The disagreement variation metric used for both datasets is from Davani et al. (2022): mean pairwise disagreement across annotators, normalised by the maximum possible disagreement for the label set.

## Models

The four models evaluated in this project are third-party artifacts:

| Model | `--client` | Source |
|---|---|---|
| Gemini 2.5 Flash (`gemini-2.5-flash`) | `gemini` | Google DeepMind — [model card](https://deepmind.google/models/gemini/flash/) |
| Mistral 7B Instruct v0.3 (`mistralai/Mistral-7B-Instruct-v0.3`) | `mistral` | Jiang et al. (2023), [arXiv:2310.06825](https://arxiv.org/abs/2310.06825) |
| Llama 3.1 8B Instant (`llama-3.1-8b-instant`) | `groq` | Dubey et al. (2024), [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) |
| HateXplain BERT (`Hate-speech-CNERG/bert-base-uncased-hatexplain`) | `hatexplain` | Mathew et al. (2021), [arXiv:2012.10289](https://arxiv.org/abs/2012.10289) |

## References

- Davani, A. M., Díaz, M., & Prabhakaran, V. (2022). Dealing with disagreements: Looking beyond the majority vote in subjective annotations. *Transactions of the Association for Computational Linguistics*, 10, 92–110.
- Dubey, A., et al. (2024). The Llama 3 herd of models. arXiv:2407.21783.
- Jiang, A. Q., et al. (2023). Mistral 7B. arXiv:2310.06825.
- Lee, N., et al. (2023). CREHate: Cross-cultural RE-annotation of English HATEful content. arXiv:2308.16705.
- Mathew, B., Saha, P., Yimam, S. M., Biemann, C., Goyal, P., & Mukherjee, A. (2021). HateXplain: A benchmark dataset for explainable hate speech detection. *AAAI 2021*. arXiv:2012.10289.
- Sachdeva, N., Barreto, R., Richey, C., Jiang, A., Chancellor, S., & Field, A. (2022). Measuring hate speech. arXiv:2009.10048.
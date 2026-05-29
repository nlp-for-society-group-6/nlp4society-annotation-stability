# annotation-stability

Research pipeline: does LLM output stability across random seeds reflect human annotator disagreement in hate speech detection?

## Pipeline

Three stages, each reads a file and writes a file:

```
Stage 1  scripts/build_mhs_items.py   →  data/interim/items.jsonl
Stage 2  scripts/run_inference.py     →  data/outputs/runs_<model>.jsonl
Stage 3  scripts/aggregate.py         →  data/outputs/scored.csv
```

`scored.csv` contains per (item, model): `llm_flip_rate`, `llm_output_entropy`, `human_entropy`, `human_disagreement_var`, `disagreement_tier`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in your provider API key(s)
python scripts/selftest.py    # smoke test, no API key needed
```

## Run

```bash
# Stage 1 — build a stratified sample from the MHS dataset
python scripts/build_mhs_items.py --n 200 --out data/interim/items.jsonl

# Stage 2 — run your model over the items
python scripts/run_inference.py \
    --client <name> --model <model-id> \
    --seeds 5 --items data/interim/items.jsonl \
    --out data/outputs/runs_<name>.jsonl

# Stage 3 — merge all runs into one scored table
python scripts/aggregate.py \
    --runs data/outputs/runs_*.jsonl \
    --out data/outputs/scored.csv
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

REGISTRY["myprovider"] = MyClient
```

Then run Stage 2 with `--client myprovider`.

`SYSTEM_PROMPT` (defined in `src/client.py`) must stay identical across all models — it is the fairness condition for the comparison.

If your model does not expose logprobs, set `logprob=None`; stability is then label-only (flip rate + entropy across seeds).

## Schemas

Two dataclasses in `src/schema.py` are the shared contract:

- `InputItem` — one text with its human-disagreement scores and tier, built once in Stage 1.
- `RunRecord` — one (model, item, seed) inference result; raw output stored verbatim. Parsing happens only in Stage 3 so it can be revised without re-running inference.

Do not add provider-specific fields to either schema.

## Data directories

```
data/interim/    items.jsonl (Stage 1 output)
data/outputs/    runs_*.jsonl and scored.csv (Stages 2–3 output)
```

Both are gitignored; scripts create them automatically.

## Dataset

[Measuring Hate Speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) (MHS, Sachdeva et al. 2022): per-annotator labels for ~39k comments, aggregated to one `InputItem` per comment with Shannon entropy and disagreement variation scores. Items are binned into low/medium/high disagreement tiers once in Stage 1 and the tiers are shared across all models.

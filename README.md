# laya-triage

A GitHub issue triage dataset for fine-tuning
[Laya](https://github.com/NandhaKishorM/laya) — the non-autoregressive typed
decision engine. The fine-tuned model answers two typed questions about a newly
opened issue in a single forward pass:

| question | Laya type | answers |
| --- | --- | --- |
| `issue_type` | `choice` | `bug`, `feature`, `question`, `docs` |
| `needs_more_info` | `noul` | `true` / `false` |

Labels come from maintainers: every issue in the set already carries a
hand-applied label, mapped onto the four canonical types. That makes the
training signal free and the benchmark honest.

Laya's own README is blunt that the base checkpoints score near chance on typed
decisions zero-shot and that "all of the capability on this benchmark comes from
fine-tuning" — which is exactly why a well-built domain dataset is the whole
game here.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # paste a GitHub token, no scopes needed
.venv/bin/python scripts/fetch_issues.py
.venv/bin/python scripts/build_dataset.py
```

Output lands in `data/processed/{train,val,test}.jsonl`.

Data collection needs nothing but `requests` and `PyYAML`, so the stock macOS
Python is fine. Running Laya itself needs **Python 3.10+** (`torch` 2.14,
`transformers` 5.x), so use a separate 3.12 venv for local evaluation.

## Record format

Rows mirror
[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions),
the dataset Laya's fine-tuning notebook consumes, so the splits are a drop-in
swap: `state`, `questions` and `gold` are each JSON-encoded strings.

```json
{
  "state": "{\"title\": \"...\", \"body\": \"...\"}",
  "questions": "{\"issue_type\": {\"type\": \"choice\", \"instructions\": \"...\", \"criteria\": {...}}, ...}",
  "gold": "{\"issue_type\": {\"probabilities\": {\"bug\": 1.0, \"feature\": 0.0, ...}}, ...}"
}
```

Gold probabilities are one-hot because maintainer labels are hard labels. In the
notebook, swap the `load_dataset("LocalLLaMA/typed-decisions", ...)` call for:

```python
ds_train = load_dataset("json", data_files="train.jsonl", split="train")
```

`--format raw` emits flat, human-readable records instead, for inspection.

## How it works

`scripts/fetch_issues.py` reads each repo's real label list, matches those names
against the regexes in `config/labels.yaml`, then pulls closed issues label by
label so the four classes come back roughly balanced. Pull requests are skipped;
rate limits and secondary throttling are handled by backing off.

`scripts/build_dataset.py` strips issue-template comments, code fences, images
and URLs, drops issues whose labels are ambiguous or excluded, dedupes by title,
caps each class, and writes the splits.

State text is capped at ~900 characters by default, which fits the 512-token
context of the English `laya` checkpoint (roughly 320 tokens remain for state
after the option prompt). Raise `--max-chars` when targeting the 1024-context
`laya-typed-decisions` or `laya-multilingual` checkpoints.

The repo name is deliberately left out of `state`, so the model has to decide
from issue content rather than memorizing per-project conventions.

## Splits

`config/repos.yaml` marks two repos as `split: test`. They are held out
completely — no issue from them appears in training — so accuracy on
`test.jsonl` measures generalization to a project the model has never seen.
That is the number worth putting in a benchmark table.

## Fine-tuning

Laya's notebook
(`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` in the upstream
repo) runs on Kaggle's free **2×T4** GPUs with DDP — not a single Colab T4 —
and does the whole loop: preprocess, train with RLCD, fit calibration
temperatures, evaluate, push to the Hub. Upstream reports roughly 4–5 hours for
4 epochs over ~30k questions, so scale expectations to the size of this set.

Fit calibration temperatures before trusting the confidence scores; upstream
measures mean ECE dropping 0.466 → 0.081 after refitting.

## Tuning the harvest

- Add repos or change quotas in `config/repos.yaml`.
- Add label spellings in `config/labels.yaml` (regex, matched against lowercased
  label names).
- `--limit N` on the fetcher for a fast trial run; `--only owner/repo` for one
  repo.

# laya-issue-triage

**GitHub issue triage with [Laya](https://github.com/NandhaKishorM/laya), a
non-autoregressive typed-decision model.** One forward pass per issue, no text
generation, no API bill, runs on the CPU inside a GitHub Action.

The model answers two typed questions about a newly opened issue:

| question | Laya primitive | answers |
| --- | --- | --- |
| `issue_type` | `choice` | `bug`, `feature`, `question`, `docs` |
| `needs_more_info` | `noul` | `true` / `false` |

Training labels come from the maintainers who triaged each issue by hand — not
from a teacher model, and not from heuristics.

---

## Results

Measured on **3,868 issues from three repositories held out of training
entirely** (`huggingface/transformers`, `facebook/react`, `microsoft/TypeScript`),
so these are generalization numbers rather than memorization.

| model | `issue_type` accuracy | macro-F1 | `needs_more_info` accuracy |
| --- | --- | --- | --- |
| random | 0.250 | — | 0.500 |
| majority class | 0.370 | — | 0.730 |
| base `laya`, zero-shot | 0.626 | 0.524 | 0.563 |
| **fine-tuned (this repo)** | _pending_ | _pending_ | _pending_ |

Zero-shot per-class F1 shows where the work is: `feature` 0.80, `bug` 0.66,
`docs` 0.53, **`question` 0.10**. The base model calls 38 of every 53 real
questions a bug, because "how do I do X?" and "X doesn't work" look alike
until you attend to intent. Fine-tuning targets exactly that.

Note that on `needs_more_info` the base model **loses to a constant baseline**
(0.563 against 0.730 for always answering `false`). Reported without that
column, 0.563 would look like a result instead of a deficit.

### What these numbers do and do not compare to

These are GitHub issue triage numbers. They are **not comparable** to scores on
the `LocalLLaMA/typed-decisions` benchmark — the one where base Laya scores
0.362 and TypeSafe Jev scores 0.727 — because that is a different task with a
different label space and different data. This model has never been run on it.
Putting the two in one table would be meaningless, however favourable it looked.

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # a GitHub personal access token, no scopes required
.venv/bin/python scripts/fetch_issues.py
.venv/bin/python scripts/build_dataset.py
```

The built splits are committed at `data/processed/*.jsonl.gz`, so you can skip
straight to training or evaluation.

Harvesting needs only `requests` and `PyYAML`, so stock macOS Python works.
Running Laya itself needs **Python 3.10+** (`torch` 2.14, `transformers` 5.x):

```bash
python3.12 -m venv .venv-laya && .venv-laya/bin/pip install laya
.venv-laya/bin/python scripts/eval_laya.py --model convaiinnovations/laya
```

---

## The GitHub Action

```yaml
name: Triage new issues
on:
  issues:
    types: [opened]
permissions:
  issues: write
jobs:
  triage:
    runs-on: ubuntu-latest
    steps:
      - uses: manyamkarthik/laya-issue-triage@v1
        with:
          dry-run: "true"   # log predictions only, until you trust it
```

No API key and no per-issue cost — the model runs on the runner's CPU. Torch and
the checkpoint are cached between runs; a cold run spends a few minutes on that
download, which dwarfs the inference itself.

Defaults chosen to be hard to regret:

- `skip-if-labeled: true` — the bot never touches an issue a human already triaged.
- `comment-on-needs-info: false` — a wrong label is easy to fix, a wrong comment
  is noise in someone's inbox. Turn it on once you trust the numbers.
- `min-confidence: 0.60` — below that it leaves the issue alone.

**Start in `dry-run` on a real repo and read a week of predictions before letting
it write.** With the *base* checkpoint the `needs_more_info` answer is currently
backwards on the obvious cases:

| issue | `needs_more_info` | correct? |
| --- | --- | --- |
| "Segfault on BOM file" — version, OS, repro steps, backtrace | 0.713 | no, it has everything |
| "it does not work. nothing happens when i run it" | 0.177 | no, this is the textbook case |

That is the 0.563-vs-0.730 deficit from the results table, made concrete. It is
what fine-tuning has to fix, and until it does, that input stays off by default.

Note also that `laya` warns this checkpoint ships temperature values outside the
sane range, so confidences — and therefore `min-confidence` — are only
approximate until calibration temperatures are fitted.

## The dataset

| split | issues | bug | feature | question | docs | `needs_more_info` labeled |
| --- | --- | --- | --- | --- | --- | --- |
| train | 7,749 | 1,359 | 1,360 | 1,368 | 1,332 | 6,384 (36% positive) |
| val | 861 | 141 | 140 | 132 | 168 | 710 |
| test | 3,868 | 990 | 882 | 743 | 254 | 3,868 (26% positive) |

Roughly 14,000 typed decisions, harvested from 14 public repositories spanning
editors, languages, ML frameworks and infrastructure.

Rows mirror
[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions),
the schema Laya's fine-tuning notebook consumes, so they are a drop-in swap:
`state`, `questions` and `gold` are each JSON-encoded strings, alongside `id`
and `workflow` (the repo, which doubles as a per-repo breakdown key).

```json
{
  "id": "pandas-dev/pandas#12345",
  "workflow": "pandas-dev/pandas",
  "state": "{\"title\": \"...\", \"body\": \"...\"}",
  "questions": "{\"issue_type\": {\"type\": \"choice\", \"instructions\": \"...\", \"criteria\": {...}}, ...}",
  "gold": "{\"issue_type\": {\"probabilities\": {\"bug\": 1.0, ...}, \"label\": \"bug\"}}"
}
```

### Decisions worth knowing about

**Gold is per-question.** An issue carries gold only for the questions its
labels actually answer. Absence of a `needs-repro` label means `false` only in
projects that use such labels at all — elsewhere the question is left
unanswered rather than silently labeled `false`. Laya's preprocessing skips any
question missing from gold, so partial rows are legal and cost nothing.

**`Discussion` labels are excluded from `question`.** In rust, numpy and react
that label means a design debate, which reads like a feature proposal. Folding
those in would have inflated the class with mislabeled data.

**Label spellings are normalized, not enumerated.** `Type: Bug`, `kind/bug`,
`C-bug` and `00 - Bug` all reduce to the same stem before matching
(`scripts/labelmap.py`), so adding a repo usually needs no config change.

**The repo name is not in `state`.** The model decides from issue content
rather than learning per-project conventions it cannot use on a new project.

---

## Fine-tuning

`notebooks/laya_triage_finetune_kaggle.ipynb` is upstream's notebook with only
the data cells swapped — the DDP training script is byte-identical. It runs on
Kaggle's free **2×T4** (not a single Colab T4) and does the whole loop:
preprocess, train with RLCD, fit calibration temperatures, evaluate, push.

Set `DATA_BASE` to your fork and `HF_USER` before running. Upstream reports
4–5 hours for 4 epochs over ~30k questions.

Before spending those hours, dry-run the records through Laya's own
preprocessing — it silently drops any item whose marker count disagrees with
the option count:

```bash
.venv-laya/bin/python scripts/verify_records.py
```

---

## Latency, honestly

| device | p50 per issue |
| --- | --- |
| T4 (batched, upstream's figure) | 33 ms |
| Apple M4, MPS | 317 ms |
| Apple M4, CPU | 462 ms |

A GitHub Actions runner is CPU-only and slower than an M4, so budget on the
order of a second per issue. That is still free and still far faster than a
maintainer reading the issue.

---

## Relationship to Laya

This repository is a downstream application, not a fork. It depends on the
`laya` package and the `convaiinnovations/laya` checkpoint, both Apache-2.0.

Laya's own README states that its base checkpoints sit near chance on typed
decisions zero-shot and that the capability comes from fine-tuning. That is
consistent with what we measured here: the base checkpoint is useful on
`feature` and `bug`, near-blind on `question`, and worse than a constant on
`needs_more_info`.

---

## Layout

```
config/labels.yaml    label spellings -> the four canonical types
config/repos.yaml     which repos, which per-type quotas, which are held out
scripts/fetch_issues.py    GitHub API harvest, balanced per type
scripts/build_dataset.py   cleaning, dedupe, splits, Laya schema
scripts/eval_laya.py       accuracy / macro-F1 / ECE / latency / confusion
scripts/verify_records.py  preprocessing dry-run
scripts/labelmap.py        shared label normalization
notebooks/            the Kaggle fine-tuning run
data/processed/       committed splits (gzipped)
```

## License

Apache-2.0, matching Laya. Issue text is quoted from public repositories and
remains under its original projects' terms; the dataset is a derived work for
research and tooling.

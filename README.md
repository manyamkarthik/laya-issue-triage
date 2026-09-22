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
| majority class | 0.345 | — | **0.738** |
| base `laya`, zero-shot | 0.626 | 0.524 | 0.563 |
| **[fine-tuned](https://huggingface.co/harikarthikmanyam/laya-issue-triage)** | **0.650** | **0.627** | 0.734 |

**Read macro-F1, not accuracy.** Accuracy moved only 0.626 → 0.650, because it
weights the big classes that already worked. Macro-F1 moved 0.524 → 0.627, and
the per-class numbers show where:

| class | base F1 | fine-tuned F1 |
| --- | --- | --- |
| `feature` | 0.796 | 0.796 |
| `bug` | 0.663 | 0.641 |
| `docs` | 0.533 | 0.585 |
| **`question`** | **0.100** | **0.487** |

`question` is the whole story. The base model was effectively blind to it,
calling roughly seven in ten real questions bugs, because "how do I do X?" and
"X doesn't work" look alike until you attend to intent. That is the class
maintainers most want separated, and it is the one fine-tuning fixed.

**`needs_more_info` did not clear its bar, and ships disabled.** At 0.734 it
still sits just under the 0.738 you get by always answering `false` — on the
exact metric the feature would be judged by. Its macro-F1 of 0.619 against
0.425 for that constant baseline says the head genuinely discriminates rather
than guessing, but it buys true positives with false ones. So the Action's
commenting is off by default, as an empirical result rather than a precaution.

Calibration after temperature fitting: ECE 0.103 on `issue_type`, 0.085 on
`needs_more_info`.

### Where it is weakest

| held-out repo | `issue_type` accuracy | n |
| --- | --- | --- |
| `huggingface/transformers` | 0.753 | 632 |
| `facebook/react` | 0.724 | 908 |
| `microsoft/TypeScript` | **0.551** | 1,329 |

TypeScript is 46% of the type-labeled test set, so it pulls the headline number
down on its own; on the other two the model runs at roughly 0.74. The residual
error is concentrated in one cell of the confusion matrix — 265 of 743
questions are still labeled `bug`, which is 36% of all question errors and the
obvious target for a v2.

Trained for 2 epochs on Kaggle 2×T4 in 54 minutes. Full numbers:
[`data/processed/benchmark_report.json`](data/processed/benchmark_report.json).

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
it write.** Two things that dry-run will show you honestly:

- `issue_type` is right about two times in three, and closer to three in four
  on projects that label like react or transformers. Good enough to save triage
  time, not good enough to trust blindly — which is why `skip-if-labeled`
  defaults to on.
- `needs_more_info` still does not beat always answering `false` (0.734 vs
  0.738), so commenting stays off until that changes. The model is not silent on
  the question — you can read its answer in the Action output and decide for
  yourself — it just has not earned the right to write to someone's inbox.

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

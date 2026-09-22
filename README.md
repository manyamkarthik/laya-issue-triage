# laya-triage

Fine-tuning dataset for GitHub issue triage with [Laya](https://github.com/) — the
model answers two questions about any newly opened issue:

| question | type | values |
| --- | --- | --- |
| `issue_type` | enum | `bug`, `feature`, `question`, `docs` |
| `needs_more_info` | bool | `true` / `false` |

Training labels come from maintainers: every issue in the set already carries a
hand-applied label, mapped onto the four canonical types.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # paste a GitHub token, no scopes needed
.venv/bin/python scripts/fetch_issues.py
.venv/bin/python scripts/build_dataset.py
```

Output lands in `data/processed/{train,val,test}.jsonl`.

## How it works

`scripts/fetch_issues.py` reads each repo's real label list, matches those names
against the regexes in `config/labels.yaml`, then pulls closed issues label by
label so the four classes come back roughly balanced. Pull requests are skipped.
Rate limits and secondary throttling are handled by backing off.

`scripts/build_dataset.py` strips issue-template comments, code fences, images
and URLs, drops issues whose labels are ambiguous or excluded, dedupes by title,
caps each class, and writes the splits.

## Splits

`config/repos.yaml` marks two repos as `split: test`. They are held out
completely — no issue from them appears in training — so accuracy on
`test.jsonl` measures generalization to a project the model has never seen.
That is the number worth putting in the README benchmark table.

## Record format

```json
{
  "state": "Title: ...\n\nBody: ...",
  "questions": [
    {"name": "issue_type", "type": "enum",
     "values": ["bug", "feature", "question", "docs"], "answer": "bug"},
    {"name": "needs_more_info", "type": "bool", "answer": false}
  ],
  "meta": {"repo": "owner/name", "number": 123, "url": "..."}
}
```

`--format raw` emits flat records instead, if you want to inspect or re-shape
them. The Laya shape is produced by `to_laya()` in `scripts/build_dataset.py` —
that one function is the only thing to change if the trainer expects different
field names.

## Tuning the harvest

- Add repos or change quotas in `config/repos.yaml`.
- Add label spellings in `config/labels.yaml` (patterns are regex, matched
  against lowercased label names).
- `--limit N` on the fetcher for a fast trial run; `--only owner/repo` for one
  repo.

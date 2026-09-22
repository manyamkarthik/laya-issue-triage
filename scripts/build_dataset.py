"""Turn data/raw/*.jsonl into train/val/test splits for Laya fine-tuning."""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import yaml

from labelmap import classify_label, load_rules, matches_any

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

TYPES = ["bug", "feature", "question", "docs"]

HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
CODE_FENCE = re.compile(r"```.*?```", re.S)
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
URL = re.compile(r"https?://\S+")
WS = re.compile(r"\n{3,}")

# Mirrors the schema of LocalLLaMA/typed-decisions, which Laya's fine-tuning
# notebook consumes: state/questions/gold, each a JSON-encoded string.
QUESTIONS = {
    "issue_type": {
        "type": "choice",
        "instructions": "What kind of GitHub issue do `title` and `body` describe?",
        "criteria": {
            "bug": "something is broken: a crash, an error, wrong output, or a "
                   "regression from behaviour that used to work",
            "feature": "a request for new functionality, an enhancement, or a "
                       "proposal to change how something works",
            "question": "the author is asking how to use the project or why it "
                        "behaves a certain way, not reporting a defect",
            "docs": "the documentation is missing, wrong, unclear, or needs an "
                    "example",
        },
    },
    "needs_more_info": {
        "type": "noul",
        "instructions": "Must a maintainer ask the author of `body` for more "
                        "information -- reproduction steps, a version number, logs, "
                        "or a code sample -- before this issue can be worked on?",
    },
}


def clean(text, max_chars):
    text = HTML_COMMENT.sub(" ", text)
    text = CODE_FENCE.sub(" [code] ", text)
    text = IMAGE.sub(" [image] ", text)
    text = URL.sub(" [url] ", text)
    text = WS.sub("\n\n", text).strip()
    return text[:max_chars]


def to_typed_decisions(rec):
    """Gold carries only the questions this issue actually answers; Laya's
    preprocessing loop skips any question absent from gold."""
    gold = {}
    if rec["type"] is not None:
        gold["issue_type"] = {
            "probabilities": {k: (1.0 if k == rec["type"] else 0.0) for k in TYPES},
            "label": rec["type"],
        }
    if rec["needs_info"] is not None:
        gold["needs_more_info"] = {
            "probabilities": {"true": float(rec["needs_info"]),
                              "false": float(not rec["needs_info"])},
            "label": "true" if rec["needs_info"] else "false",
        }
    state = {"title": rec["title"], "body": rec["body"]}
    # `id` and `workflow` keep the upstream notebook's eval cells working
    # unchanged; workflow doubles as a per-repo breakdown key.
    return {
        "id": f"{rec['repo']}#{rec['number']}",
        "workflow": rec["repo"],
        "state": json.dumps(state, ensure_ascii=False),
        "questions": json.dumps(QUESTIONS, ensure_ascii=False),
        "gold": json.dumps(gold, ensure_ascii=False),
        "url": rec["url"],
    }


def summarize(rows):
    types = Counter(r["type"] for r in rows if r["type"])
    info = [r["needs_info"] for r in rows if r["needs_info"] is not None]
    return (f"types={dict(types)} "
            f"needs_info={sum(info)}/{len(info)} labeled")


def write(path, rows, fmt):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(to_typed_decisions(r) if fmt == "laya" else r) + "\n")
    print(f"  {path.relative_to(ROOT)}: {len(rows)}  {summarize(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default=str(ROOT / "config" / "repos.yaml"))
    ap.add_argument("--labels", default=str(ROOT / "config" / "labels.yaml"))
    # 900 chars ~ 250 tokens, fitting the 512-context `laya` checkpoint, which
    # leaves ~320 tokens for state after the option prompt. Raise it for the
    # 1024-context typed-decisions / multilingual checkpoints.
    ap.add_argument("--max-chars", type=int, default=900)
    ap.add_argument("--cap-per-class", type=int, default=1500)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--format", choices=["laya", "raw"], default="laya")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    splits = {e["name"]: e.get("split", "train")
              for e in yaml.safe_load(Path(args.repos).read_text())["repos"]}
    rules = load_rules(args.labels)

    raw_by_repo = {}
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        repo = path.stem.replace("__", "/")
        raw_by_repo[repo] = [json.loads(l) for l in path.read_text().splitlines()]

    # Absence of a "needs info" label only means `false` in projects that
    # actually use such labels. Elsewhere the question is left unanswered
    # rather than silently labeled false.
    info_repos = {
        repo for repo, items in raw_by_repo.items()
        if any(matches_any(lb, rules["needs_info"])
               for it in items for lb in it["labels"])
    }

    kept = {"train": [], "test": []}
    dropped = Counter()
    seen_titles = set()

    for repo, items in raw_by_repo.items():
        split = splits.get(repo, "train")
        for item in items:
            names = item["labels"]
            if any(matches_any(lb, rules["exclude"]) for lb in names):
                dropped["excluded_label"] += 1
                continue
            hits = {classify_label(lb, rules["types"]) for lb in names} - {None}
            kind = hits.pop() if len(hits) == 1 else None
            needs_info = None
            if repo in info_repos:
                needs_info = any(matches_any(lb, rules["needs_info"]) for lb in names)
            if kind is None and needs_info is None:
                dropped["no_usable_label"] += 1
                continue
            title = clean(item["title"], 200)
            body = clean(item["body"], args.max_chars)
            if len(title) < 5 or len(title + body) < 40:
                dropped["too_short"] += 1
                continue
            key = title.lower()[:120]
            if key in seen_titles:
                dropped["duplicate_title"] += 1
                continue
            seen_titles.add(key)
            kept[split].append({
                "repo": repo, "number": item["number"], "url": item["url"],
                "title": title, "body": body,
                "type": kind, "needs_info": needs_info,
            })

    rng = random.Random(args.seed)

    # Cap only the type-labeled rows; rows kept for needs_info alone pass through.
    by_class, passthrough = {}, []
    for r in kept["train"]:
        if r["type"]:
            by_class.setdefault(r["type"], []).append(r)
        else:
            passthrough.append(r)
    balanced = list(passthrough)
    for rows in by_class.values():
        rng.shuffle(rows)
        balanced.extend(rows[: args.cap_per_class])
    rng.shuffle(balanced)

    n_val = int(len(balanced) * args.val_frac)
    val, train = balanced[:n_val], balanced[n_val:]
    test = kept["test"][:]
    rng.shuffle(test)

    print("dropped:", dict(dropped))
    write(OUT_DIR / "train.jsonl", train, args.format)
    write(OUT_DIR / "val.jsonl", val, args.format)
    write(OUT_DIR / "test.jsonl", test, args.format)
    print("  held-out repos:", [r for r, s in splits.items() if s == "test"])
    print("  repos with needs-info labels:", len(info_repos), "of", len(raw_by_repo))


if __name__ == "__main__":
    main()

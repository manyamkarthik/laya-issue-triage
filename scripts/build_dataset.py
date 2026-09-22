"""Turn data/raw/*.jsonl into train/val/test splits for Laya fine-tuning."""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

TYPES = ["bug", "feature", "question", "docs"]

HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
CODE_FENCE = re.compile(r"```.*?```", re.S)
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
URL = re.compile(r"https?://\S+")
WS = re.compile(r"\n{3,}")


def clean(text, max_chars):
    text = HTML_COMMENT.sub(" ", text)
    text = CODE_FENCE.sub(" [code] ", text)
    text = IMAGE.sub(" [image] ", text)
    text = URL.sub(" [url] ", text)
    text = WS.sub("\n\n", text).strip()
    return text[:max_chars]


def load_rules(path):
    cfg = yaml.safe_load(Path(path).read_text())
    return (
        {k: [re.compile(p) for p in v] for k, v in cfg["types"].items()},
        [re.compile(p) for p in cfg["needs_info"]],
        [re.compile(p) for p in cfg["exclude"]],
    )


def label_issue(labels, type_pats, info_pats, excl_pats):
    names = [lb.strip().lower() for lb in labels]
    if any(p.search(n) for n in names for p in excl_pats):
        return None, None
    hits = {k for k, pats in type_pats.items() for n in names if any(p.search(n) for p in pats)}
    if len(hits) != 1:
        return None, None
    needs_info = any(p.search(n) for n in names for p in info_pats)
    return hits.pop(), needs_info


def to_laya(rec):
    """Laya record: free-text state + typed questions with known answers."""
    return {
        "state": f"Title: {rec['title']}\n\nBody: {rec['body']}",
        "questions": [
            {"name": "issue_type", "type": "enum", "values": TYPES,
             "answer": rec["type"]},
            {"name": "needs_more_info", "type": "bool",
             "answer": rec["needs_info"]},
        ],
        "meta": {"repo": rec["repo"], "number": rec["number"], "url": rec["url"]},
    }


def write(path, rows, fmt):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(to_laya(r) if fmt == "laya" else r) + "\n")
    print(f"  {path.relative_to(ROOT)}: {len(rows)}  {dict(Counter(r['type'] for r in rows))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default=str(ROOT / "config" / "repos.yaml"))
    ap.add_argument("--labels", default=str(ROOT / "config" / "labels.yaml"))
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--cap-per-class", type=int, default=2000)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--format", choices=["laya", "raw"], default="laya")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    splits = {e["name"]: e.get("split", "train")
              for e in yaml.safe_load(Path(args.repos).read_text())["repos"]}
    type_pats, info_pats, excl_pats = load_rules(args.labels)

    kept = {"train": [], "test": []}
    dropped = Counter()
    seen_titles = set()

    for path in sorted(RAW_DIR.glob("*.jsonl")):
        repo = path.stem.replace("__", "/")
        split = splits.get(repo, "train")
        for line in path.read_text().splitlines():
            item = json.loads(line)
            kind, needs_info = label_issue(item["labels"], type_pats, info_pats, excl_pats)
            if kind is None:
                dropped["unmappable_labels"] += 1
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

    by_class = {}
    for r in kept["train"]:
        by_class.setdefault(r["type"], []).append(r)
    balanced = []
    for kind, rows in by_class.items():
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


if __name__ == "__main__":
    main()

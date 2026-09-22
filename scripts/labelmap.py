"""Shared label normalization and matching, used by both pipeline stages."""

import re
from pathlib import Path

import yaml

# Projects prefix their labels in wildly different ways: "Type: Bug", "kind/bug",
# "C-bug", "33 - Question", "area/docs". Strip the prefix, then match the stem.
PREFIXES = [
    re.compile(r"^\d+\s*[-.:]\s*"),
    re.compile(r"^(type|kind|area|category|cat|status|issue|resolution|topic|comp|component)\s*[:/-]\s*"),
    re.compile(r"^[a-z]\s*-\s*"),
]


def normalize(name):
    name = name.strip().lower()
    changed = True
    while changed:
        changed = False
        for pat in PREFIXES:
            new = pat.sub("", name, count=1)
            if new != name:
                name, changed = new, True
    return name.strip()


def load_rules(path):
    cfg = yaml.safe_load(Path(path).read_text())
    compile_all = lambda pats: [re.compile(p) for p in pats]
    return {
        "types": {k: compile_all(v) for k, v in cfg["types"].items()},
        "needs_info": compile_all(cfg["needs_info"]),
        "exclude": compile_all(cfg["exclude"]),
    }


def classify_label(name, type_pats):
    """Canonical type for one raw label name, or None if unmapped/ambiguous."""
    stem = normalize(name)
    hits = {k for k, pats in type_pats.items() if any(p.search(stem) for p in pats)}
    return hits.pop() if len(hits) == 1 else None


def matches_any(name, pats):
    return any(p.search(normalize(name)) for p in pats)

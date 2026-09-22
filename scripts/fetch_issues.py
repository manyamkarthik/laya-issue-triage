"""Pull labeled issues from public repos into data/raw/<owner>__<repo>.jsonl."""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
API = "https://api.github.com"


def load_token():
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("GITHUB_TOKEN="):
                return line.split("=", 1)[1].strip().strip("\"'")
    print("WARNING: no GITHUB_TOKEN, running unauthenticated (60 req/hour). "
          "See .env.example.", file=sys.stderr)
    return None


class GitHub:
    def __init__(self, token):
        self.s = requests.Session()
        self.s.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        if token:
            self.s.headers["Authorization"] = f"Bearer {token}"

    def get(self, url, params=None):
        for attempt in range(6):
            r = self.s.get(url, params=params, timeout=30)
            if r.status_code == 200:
                remaining = int(r.headers.get("X-RateLimit-Remaining", 1))
                if remaining <= 1:
                    reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                    nap = max(reset - time.time(), 0) + 2
                    print(f"  rate limit hit, sleeping {nap:.0f}s", flush=True)
                    time.sleep(nap)
                return r
            if r.status_code in (403, 429):
                nap = int(r.headers.get("Retry-After", 2 ** attempt * 10))
                print(f"  throttled ({r.status_code}), sleeping {nap}s", flush=True)
                time.sleep(nap)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
        raise RuntimeError(f"gave up on {url}")

    def paginate(self, url, params):
        params = dict(params, per_page=100)
        page = 1
        while page <= 10:  # API caps label-filtered listings well before this
            r = self.get(url, dict(params, page=page))
            if r is None:
                return
            batch = r.json()
            if not batch:
                return
            yield from batch
            if len(batch) < 100:
                return
            page += 1


def compile_patterns(cfg):
    return {
        kind: [re.compile(p) for p in pats] for kind, pats in cfg["types"].items()
    }


def classify_label(name, type_patterns):
    name = name.strip().lower()
    hits = {k for k, pats in type_patterns.items() if any(p.search(name) for p in pats)}
    return hits.pop() if len(hits) == 1 else None


def fetch_repo(gh, repo, target, type_patterns):
    owner, name = repo.split("/")
    labels = [lb["name"] for lb in gh.paginate(f"{API}/repos/{repo}/labels", {})]
    by_type = {}
    for lb in labels:
        kind = classify_label(lb, type_patterns)
        if kind:
            by_type.setdefault(kind, []).append(lb)
    if not by_type:
        print(f"  {repo}: no mappable labels, skipping")
        return []

    quota = max(target // max(len(by_type), 1), 1)
    seen, out = set(), []
    for kind, label_names in sorted(by_type.items()):
        got = 0
        print(f"  {repo} [{kind}] via {label_names}", flush=True)
        for label in label_names:
            if got >= quota:
                break
            params = {"labels": label, "state": "closed", "sort": "created",
                      "direction": "desc"}
            for item in gh.paginate(f"{API}/repos/{repo}/issues", params):
                if got >= quota:
                    break
                if "pull_request" in item or item["number"] in seen:
                    continue
                seen.add(item["number"])
                out.append(
                    {
                        "repo": repo,
                        "number": item["number"],
                        "title": item["title"] or "",
                        "body": item["body"] or "",
                        "labels": [lb["name"] for lb in item["labels"]],
                        "url": item["html_url"],
                        "created_at": item["created_at"],
                        "comments": item["comments"],
                    }
                )
                got += 1
        print(f"    -> {got}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default=str(ROOT / "config" / "repos.yaml"))
    ap.add_argument("--labels", default=str(ROOT / "config" / "labels.yaml"))
    ap.add_argument("--only", help="fetch just this repo (owner/name)")
    ap.add_argument("--limit", type=int, help="override per-repo target")
    args = ap.parse_args()

    repos_cfg = yaml.safe_load(Path(args.repos).read_text())["repos"]
    type_patterns = compile_patterns(yaml.safe_load(Path(args.labels).read_text()))
    gh = GitHub(load_token())
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for entry in repos_cfg:
        repo = entry["name"]
        if args.only and repo != args.only:
            continue
        target = args.limit or entry.get("target", 1000)
        print(f"{repo} (target {target}, split {entry.get('split', 'train')})")
        rows = fetch_repo(gh, repo, target, type_patterns)
        dest = RAW_DIR / f"{repo.replace('/', '__')}.jsonl"
        with dest.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"  wrote {len(rows)} -> {dest.relative_to(ROOT)}\n", flush=True)


if __name__ == "__main__":
    main()

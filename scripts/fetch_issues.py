"""Pull labeled issues from public repos into data/raw/<owner>__<repo>.jsonl."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

from labelmap import classify_label, load_rules, matches_any

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
                if int(r.headers.get("X-RateLimit-Remaining", 1)) <= 1:
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

    def paginate(self, url, params, max_pages=10):
        params = dict(params, per_page=100)
        for page in range(1, max_pages + 1):
            r = self.get(url, dict(params, page=page))
            if r is None:
                return
            batch = r.json()
            if not batch:
                return
            yield from batch
            if len(batch) < 100:
                return


def issue_record(repo, item):
    return {
        "repo": repo,
        "number": item["number"],
        "title": item["title"] or "",
        "body": item["body"] or "",
        "labels": [lb["name"] for lb in item["labels"]],
        "url": item["html_url"],
        "created_at": item["created_at"],
        "comments": item["comments"],
    }


def harvest(gh, repo, label_names, quota, seen):
    """Pull up to `quota` closed issues carrying any of `label_names`."""
    out = []
    for label in label_names:
        if len(out) >= quota:
            break
        params = {"labels": label, "state": "closed", "sort": "created",
                  "direction": "desc"}
        for item in gh.paginate(f"{API}/repos/{repo}/issues", params):
            if len(out) >= quota:
                break
            if "pull_request" in item or item["number"] in seen:
                continue
            seen.add(item["number"])
            out.append(issue_record(repo, item))
    return out


def fetch_repo(gh, repo, rules, quotas, info_quota):
    labels = [lb["name"] for lb in gh.paginate(f"{API}/repos/{repo}/labels", {})]
    by_type, info_labels = {}, []
    for lb in labels:
        kind = classify_label(lb, rules["types"])
        if kind:
            by_type.setdefault(kind, []).append(lb)
        if matches_any(lb, rules["needs_info"]):
            info_labels.append(lb)

    seen, rows = set(), []
    for kind in sorted(by_type):
        got = harvest(gh, repo, by_type[kind], quotas.get(kind, 0), seen)
        print(f"  [{kind}] {by_type[kind]} -> {len(got)}", flush=True)
        rows.extend(got)

    if info_labels and info_quota:
        got = harvest(gh, repo, info_labels, info_quota, seen)
        print(f"  [needs_info] {info_labels} -> {len(got)}", flush=True)
        rows.extend(got)
    elif not info_labels:
        print("  [needs_info] no matching labels", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default=str(ROOT / "config" / "repos.yaml"))
    ap.add_argument("--labels", default=str(ROOT / "config" / "labels.yaml"))
    ap.add_argument("--only", help="fetch just this repo (owner/name)")
    ap.add_argument("--limit", type=int, help="override every per-type quota")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.repos).read_text())
    defaults = cfg.get("defaults", {})
    rules = load_rules(args.labels)
    gh = GitHub(load_token())
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for entry in cfg["repos"]:
        repo = entry["name"]
        if args.only and repo != args.only:
            continue
        quotas = dict(defaults.get("per_type", {}), **entry.get("per_type", {}))
        info_quota = entry.get("needs_info", defaults.get("needs_info", 0))
        if args.limit:
            quotas = {k: args.limit for k in quotas}
            info_quota = args.limit
        print(f"{repo} (split {entry.get('split', 'train')})", flush=True)
        rows = fetch_repo(gh, repo, rules, quotas, info_quota)
        dest = RAW_DIR / f"{repo.replace('/', '__')}.jsonl"
        with dest.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"  wrote {len(rows)} -> {dest.relative_to(ROOT)}\n", flush=True)


if __name__ == "__main__":
    main()

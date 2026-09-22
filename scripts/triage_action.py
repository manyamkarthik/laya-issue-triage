"""Triage one GitHub issue with Laya and apply the result.

Reads the issue from the Actions event payload, runs both typed questions in a
single forward pass, then applies labels and optionally asks the author for the
missing details.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

TYPES = ["bug", "feature", "question", "docs"]
API = "https://api.github.com"


def gh_request(method, url, token, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read() or "{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"GitHub API {method} {url} failed: {e.code} {e.read().decode()[:300]}")


def load_event():
    path = os.environ.get("GITHUB_EVENT_PATH")
    if not path or not os.path.exists(path):
        sys.exit("No GITHUB_EVENT_PATH; this script runs inside a GitHub Action.")
    with open(path) as f:
        event = json.load(f)
    issue = event.get("issue")
    if not issue:
        sys.exit("Event carries no issue; trigger this on issues: [opened].")
    return event, issue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--min-confidence", type=float, default=0.60)  # matches action.yml
    ap.add_argument("--apply-labels", default="true")
    ap.add_argument("--comment-on-needs-info", default="false")
    ap.add_argument("--skip-if-labeled", default="true")
    ap.add_argument("--label-prefix", default="")
    ap.add_argument("--dry-run", default="false")
    args = ap.parse_args()

    flag = lambda v: str(v).strip().lower() in ("true", "1", "yes")
    dry_run = flag(args.dry_run)

    token = os.environ.get("GITHUB_TOKEN")
    if not token and not dry_run:
        sys.exit("GITHUB_TOKEN is not set.")

    event, issue = load_event()
    repo = event["repository"]["full_name"]
    number = issue["number"]

    if issue.get("pull_request"):
        print("Skipping: this is a pull request, not an issue.")
        return
    existing = [lb["name"] for lb in issue.get("labels", [])]
    if flag(args.skip_if_labeled) and existing:
        print(f"Skipping #{number}: already labeled {existing}. "
              "A maintainer's triage is not ours to overwrite.")
        return

    import laya

    # Must match the questions the model was fine-tuned against, verbatim.
    questions = json.loads(
        (open(os.path.join(os.path.dirname(__file__), "..", "config",
                           "questions.json")).read())
    )
    state = {"title": issue["title"] or "", "body": (issue.get("body") or "")[:900]}

    agent = laya.Agent(args.model, device="cpu")
    answers = agent.predict(state, questions)["answers"]

    kind = answers["issue_type"]["choice"]
    kind_conf = max(answers["issue_type"]["probabilities"].values())
    needs_info = answers["needs_more_info"]["noul"]

    print(f"#{number}: issue_type={kind} ({kind_conf:.3f}) "
          f"needs_more_info={needs_info:.3f}")

    if kind_conf < args.min_confidence:
        print(f"Below --min-confidence {args.min_confidence}; leaving #{number} alone.")
        return

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"issue_type={kind}\nconfidence={kind_conf:.4f}\n"
                    f"needs_more_info={needs_info >= 0.5}\n")

    labels = []
    if flag(args.apply_labels):
        labels.append(f"{args.label_prefix}{kind}")
        if needs_info >= 0.5:
            labels.append(f"{args.label_prefix}needs-more-info")

    if dry_run:
        print(f"[dry-run] would add {labels} to {repo}#{number}")
        return

    if labels:
        gh_request("POST", f"{API}/repos/{repo}/issues/{number}/labels", token,
                   {"labels": labels})
        print(f"Added {labels}")

    if flag(args.comment_on_needs_info) and needs_info >= 0.5:
        body = (
            "Thanks for the report. Before a maintainer can pick this up, could "
            "you add whatever applies here:\n\n"
            "- the version you are running\n"
            "- the steps that reproduce it\n"
            "- what you expected versus what happened\n"
            "- any error output or logs\n\n"
            f"<sub>Triaged automatically by "
            f"[laya-issue-triage](https://github.com/manyamkarthik/laya-issue-triage) "
            f"({kind}, {kind_conf:.0%} confidence). If this is wrong, just relabel "
            f"it -- the bot will not touch an issue that already has labels.</sub>"
        )
        gh_request("POST", f"{API}/repos/{repo}/issues/{number}/comments", token,
                   {"body": body})
        print("Posted a request for more detail")


if __name__ == "__main__":
    main()

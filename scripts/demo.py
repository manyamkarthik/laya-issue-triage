"""Run a handful of realistic issues through the model and print the answers.

    .venv-laya/bin/python scripts/demo.py

First run downloads the checkpoint (~840 MB). Add --base to see what the
un-fine-tuned model says about the same issues.
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXAMPLES = [
    ("How do I change the output directory?",
     "I read the README but I cannot work out where the results get written. "
     "Is there a flag for it?"),
    ("Segfault when opening UTF-8 BOM file",
     "v1.4.2 on Ubuntu 22.04. Steps: 1. open any file saved with a BOM "
     "2. editor crashes immediately. Backtrace attached."),
    ("it does not work",
     "nothing happens when i run it. please fix"),
    ("Add dark mode support",
     "It would be great if the dashboard supported a dark theme, especially "
     "for late-night use."),
    ("Typo in the installation guide",
     "The install page still says python 3.8 but the package needs 3.10 or newer."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="harikarthikmanyam/laya-issue-triage")
    ap.add_argument("--base", action="store_true",
                    help="use the un-fine-tuned checkpoint instead")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import laya

    model = "convaiinnovations/laya" if args.base else args.model
    questions = json.loads((ROOT / "config" / "questions.json").read_text())

    print(f"model: {model}   device: {args.device}\n")
    agent = laya.Agent(model, device=args.device)

    print(f"{'issue title':<40} {'type':<9} {'conf':>6} {'needs info':>11} {'ms':>6}")
    print("-" * 76)
    for title, body in EXAMPLES:
        t0 = time.perf_counter()
        answers = agent.predict({"title": title, "body": body}, questions)["answers"]
        dt = (time.perf_counter() - t0) * 1000
        kind = answers["issue_type"]["choice"]
        conf = max(answers["issue_type"]["probabilities"].values())
        info = answers["needs_more_info"]["noul"]
        shown = title if len(title) <= 39 else title[:36] + "..."
        print(f"{shown:<40} {kind:<9} {conf:>6.3f} {info:>11.3f} {dt:>6.0f}")

    print("\nBoth questions are answered in one forward pass -- the model reads the "
          "issue once,\nnot once per question.")


if __name__ == "__main__":
    main()

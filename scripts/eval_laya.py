"""Score a Laya checkpoint on data/processed/test.jsonl.

Run it on the base checkpoint first to get the zero-shot baseline, then on the
fine-tuned one. The difference is the benchmark table.

    .venv-laya/bin/python scripts/eval_laya.py --model convaiinnovations/laya
    .venv-laya/bin/python scripts/eval_laya.py --model ./laya-triage-ft --tag ft
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TYPES = ["bug", "feature", "question", "docs"]


def pick_device(requested):
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def macro_f1(pairs, labels):
    f1s = []
    for lb in labels:
        tp = sum(1 for g, p in pairs if g == lb and p == lb)
        fp = sum(1 for g, p in pairs if g != lb and p == lb)
        fn = sum(1 for g, p in pairs if g == lb and p != lb)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s), dict(zip(labels, f1s))


def ece(confs, corrects, bins=10):
    total = len(confs)
    if not total:
        return 0.0
    out = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (c > lo or b == 0) and c <= hi]
        if not idx:
            continue
        acc = sum(corrects[i] for i in idx) / len(idx)
        conf = sum(confs[i] for i in idx) / len(idx)
        out += len(idx) / total * abs(acc - conf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "test.jsonl"))
    ap.add_argument("--device", help="cuda / mps / cpu (auto-detected by default)")
    ap.add_argument("--limit", type=int, help="score only the first N rows")
    ap.add_argument("--tag", default="base", help="name for this run in the report")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "eval.json"))
    args = ap.parse_args()

    import laya

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines()]
    if args.limit:
        rows = rows[: args.limit]

    device = pick_device(args.device)
    print(f"loading {args.model} on {device}")
    agent = laya.Agent(args.model, device=device)

    type_pairs, type_confs, type_correct = [], [], []
    info_pairs, info_confs, info_correct = [], [], []
    per_repo = defaultdict(lambda: [0, 0])
    latencies = []

    for i, row in enumerate(rows):
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])

        t0 = time.perf_counter()
        res = agent.predict(state, questions)
        latencies.append((time.perf_counter() - t0) * 1000)
        answers = res["answers"]

        if "issue_type" in gold:
            pred = answers["issue_type"]["choice"]
            truth = gold["issue_type"]["label"]
            correct = float(pred == truth)
            type_pairs.append((truth, pred))
            type_confs.append(max(answers["issue_type"]["probabilities"].values()))
            type_correct.append(correct)
            bucket = per_repo[row.get("workflow", "?")]
            bucket[0] += correct
            bucket[1] += 1

        if "needs_more_info" in gold:
            p_true = answers["needs_more_info"]["noul"]
            pred = "true" if p_true >= 0.5 else "false"
            truth = gold["needs_more_info"]["label"]
            info_pairs.append((truth, pred))
            info_confs.append(max(p_true, 1 - p_true))
            info_correct.append(float(pred == truth))

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    latencies.sort()
    type_acc = sum(type_correct) / len(type_correct) if type_correct else 0.0
    info_acc = sum(info_correct) / len(info_correct) if info_correct else 0.0
    type_f1, per_class_f1 = macro_f1(type_pairs, TYPES)
    info_f1, _ = macro_f1(info_pairs, ["true", "false"])

    majority = Counter(g for g, _ in type_pairs).most_common(1)
    baseline = majority[0][1] / len(type_pairs) if type_pairs else 0.0

    report = {
        "tag": args.tag,
        "model": args.model,
        "device": device,
        "n_cases": len(rows),
        "issue_type": {
            "n": len(type_pairs),
            "accuracy": round(type_acc, 4),
            "macro_f1": round(type_f1, 4),
            "per_class_f1": {k: round(v, 4) for k, v in per_class_f1.items()},
            "majority_baseline": round(baseline, 4),
            "random_baseline": round(1 / len(TYPES), 4),
            "ece": round(ece(type_confs, type_correct), 4),
        },
        "needs_more_info": {
            "n": len(info_pairs),
            "accuracy": round(info_acc, 4),
            "macro_f1": round(info_f1, 4),
            "positive_rate": round(
                sum(1 for g, _ in info_pairs if g == "true") / len(info_pairs), 4
            ) if info_pairs else 0.0,
            # Always answering "false" is the bar to beat here, and on a skewed
            # set it is a high one.
            "majority_baseline": round(
                max(sum(1 for g, _ in info_pairs if g == lb) / len(info_pairs)
                    for lb in ("true", "false")), 4
            ) if info_pairs else 0.0,
            "ece": round(ece(info_confs, info_correct), 4),
        },
        "latency_ms": {
            "p50": round(latencies[len(latencies) // 2], 1),
            "p95": round(latencies[int(len(latencies) * 0.95)], 1),
            "mean": round(sum(latencies) / len(latencies), 1),
        },
        "per_repo_issue_type_accuracy": {
            r: round(c / n, 4) for r, (c, n) in sorted(per_repo.items())
        },
    }

    print(json.dumps(report, indent=2))

    out = Path(args.out)
    all_runs = json.loads(out.read_text()) if out.exists() else {}
    all_runs[args.tag] = report
    out.write_text(json.dumps(all_runs, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)} (tag: {args.tag})")

    confusion = Counter(type_pairs)
    print("\nconfusion (gold -> pred):")
    for g in TYPES:
        line = "  ".join(f"{p}:{confusion.get((g, p), 0):4d}" for p in TYPES)
        print(f"  {g:>8}  {line}")


if __name__ == "__main__":
    main()

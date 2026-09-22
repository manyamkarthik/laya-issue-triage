"""Dry-run our JSONL through Laya's own preprocessing.

Mirrors `build_training_item` from the upstream fine-tuning notebook, which
silently drops any item whose marker count disagrees with the option count.
Better to find that here than after a multi-hour Kaggle run.
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "train.jsonl"))
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from laya.agent import _fix_tokenizer_config
    from laya.common import QTYPES, build_sequence, render_options

    model_dir = snapshot_download(args.model)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    print(f"max_len={cfg['max_len']} head_max_len={cfg['head_max_len']}")

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines()]
    rows = rows[: args.limit]

    stats = Counter()
    lengths = []
    for row in rows:
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for qid, q in questions.items():
            if qid not in gold:
                stats["skipped_no_gold"] += 1
                continue
            crit = q.get("criteria", {})
            k = len(render_options({"t": q["type"], "crit": crit}))
            seq, markers = build_sequence(
                tok, state, {"t": q["type"], "ins": q["instructions"], "crit": crit},
                cfg["max_len"], cfg["head_max_len"],
            )
            if len(markers) != k:
                stats[f"DROPPED_marker_mismatch_{qid}"] += 1
                continue
            stats[f"ok_{qid}"] += 1
            lengths.append(len(seq))
            if len(seq) >= cfg["max_len"]:
                stats["at_context_limit"] += 1

            probs = gold[qid]["probabilities"]
            if abs(sum(probs.values()) - 1.0) > 1e-6:
                stats["BAD_probabilities"] += 1
            if q["type"] == "choice" and gold[qid]["label"] not in crit:
                stats["BAD_label_not_in_criteria"] += 1

    lengths.sort()
    print("\ncounts:", dict(stats))
    if lengths:
        print(f"sequence tokens: p50={lengths[len(lengths)//2]} "
              f"p95={lengths[int(len(lengths)*.95)]} max={lengths[-1]}")
    bad = sum(v for k, v in stats.items() if k.startswith(("DROPPED", "BAD")))
    print("\nFAIL" if bad else "\nOK: every record survives preprocessing")


if __name__ == "__main__":
    main()

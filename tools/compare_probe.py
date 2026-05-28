"""Compare probe brain output to Qwen2.5 ground-truth predictions.

Reads:
  logs/qwen25_baseline.json  -- model ground truth (from qwen25_baseline.py)
  logs/probe_output.txt      -- sigbrain infer output (from infer.exe)

Writes:
  logs/comparison.json       -- per-word match/mismatch analysis
  logs/comparison_summary.txt -- human-readable summary

Run probe brain first:
  E:\repos\blackfall-labs\sigbrain\target\release\infer.exe \
    --brain brain_probe_v2.brnc --speak <word> >> logs/probe_output.txt

Or use run_probe_all.py to batch all vocab words.
"""

import json, sys, re, os

def load_baseline(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_probe_output(path):
    """Parse lines like: 'the -> a new ...' or 'the -> (no continuation)'"""
    results = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = re.match(r'^(\w+)\s*(?:->|→|[-:>]+)\s*(.+)$', line)
            if not m:
                continue
            word = m.group(1).lower()
            pred = m.group(2).strip()
            if pred.startswith("("):
                results[word] = None
            else:
                # First word of continuation is the top-1 prediction
                results[word] = pred.split()[0].lower()
    return results

def main():
    baseline_path = "logs/qwen25_baseline.json"
    probe_path    = "logs/probe_output.txt"
    out_json      = "logs/comparison.json"
    out_txt       = "logs/comparison_summary.txt"

    if not os.path.exists(baseline_path):
        print(f"ERROR: {baseline_path} missing. Run qwen25_baseline.py first.")
        sys.exit(1)
    if not os.path.exists(probe_path):
        print(f"ERROR: {probe_path} missing. Run the probe brain first.")
        sys.exit(1)

    baseline = load_baseline(baseline_path)
    probe    = load_probe_output(probe_path)
    preds    = baseline["predictions"]

    matches    = []
    mismatches = []
    no_output  = []

    for word, bdata in sorted(preds.items()):
        model_top1 = bdata["top1_word"]
        probe_top1 = probe.get(word)

        if probe_top1 is None:
            no_output.append(word)
            continue

        if probe_top1 == model_top1:
            matches.append(word)
        else:
            # Check if probe top1 is in model's top-5
            model_top5 = [p[0] for p in bdata["top_preds"][:5]]
            in_top5    = probe_top1 in model_top5
            mismatches.append({
                "word":       word,
                "model_top1": model_top1,
                "model_top5": model_top5,
                "probe_top1": probe_top1,
                "in_top5":    in_top5,
            })

    total     = len(matches) + len(mismatches) + len(no_output)
    top1_acc  = len(matches) / max(1, total - len(no_output))
    top5_acc  = (len(matches) + sum(1 for m in mismatches if m["in_top5"])) / max(1, total - len(no_output))

    result = {
        "total_words":   total,
        "matched":       len(matches),
        "mismatched":    len(mismatches),
        "no_output":     len(no_output),
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "matches":       matches,
        "mismatches":    mismatches,
        "no_output_words": no_output,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    lines = []
    lines.append(f"=== PROBE vs QWEN2.5 COMPARISON ===")
    lines.append(f"Total words:     {total}")
    lines.append(f"No output:       {len(no_output)}")
    lines.append(f"Top-1 accuracy:  {top1_acc:.1%}  ({len(matches)}/{total - len(no_output)})")
    lines.append(f"Top-5 accuracy:  {top5_acc:.1%}")
    lines.append("")
    lines.append("MISMATCHES:")
    for m in sorted(mismatches, key=lambda x: x["word"]):
        flag = " [in top5]" if m["in_top5"] else ""
        lines.append(f"  {m['word']:12s}  model={m['model_top1']:12s}  probe={m['probe_top1']:12s}{flag}")
    if no_output:
        lines.append("")
        lines.append("NO OUTPUT:")
        for w in sorted(no_output):
            lines.append(f"  {w}")

    summary = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(summary)

    print(summary)

if __name__ == "__main__":
    main()

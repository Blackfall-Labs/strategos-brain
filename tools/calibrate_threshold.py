"""Calibrate per-layer FFN thresholds from a probe activation profile.

For each layer, the threshold is set to abs(p5) of the neuron_freq distribution.
This ensures that the 95th-percentile most-active neurons are captured by the
extract binary, while filtering noise below the 5th percentile fire rate.

Writes ffn_thresholds.json to the model weights directory so probe/extract
pick it up on the next run.

Usage:
    python tools/calibrate_threshold.py \\
        --profile qwen25_profile.json \\
        --weights-dir models/qwen25_05b \\
        --output qwen25_profile_calibrated.json   # optional re-run reminder

The script writes <weights_dir>/ffn_thresholds.json.
Then re-run probe with the same args — it will use the new thresholds.
"""

import json, argparse, os

def p5(values):
    if not values:
        return 0.01
    s = sorted(values)
    idx = max(0, int(len(s) * 0.05) - 1)
    return max(s[idx], 1e-4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile",     required=True,  help="Activation profile JSON from probe")
    parser.add_argument("--weights-dir", required=True,  help="Model weights directory")
    args = parser.parse_args()

    with open(args.profile, encoding="utf-8") as f:
        profile = json.load(f)

    thresholds = []
    global_mean_list = []

    for layer in profile["layers"]:
        nf = layer["ops"].get("ffn_hidden", {}).get("neuron_freq", [])
        if nf:
            mean = sum(nf) / len(nf)
            global_mean_list.append(mean)
            t = p5(nf)
            thresholds.append(round(t, 5))
        else:
            thresholds.append(0.1)

    global_mean = sum(global_mean_list) / len(global_mean_list) if global_mean_list else 0
    print(f"Global mean neuron_freq: {global_mean:.4f}  ({'calibrated' if global_mean < 0.5 else 'UNCALIBRATED — needs calibration'})")
    print(f"Per-layer thresholds (p5):")
    for i, t in enumerate(thresholds):
        print(f"  L{i:02d}: {t:.5f}")

    out_path = os.path.join(args.weights_dir, "ffn_thresholds.json")
    with open(out_path, "w") as f:
        json.dump({"thresholds": thresholds}, f, indent=2)
    print(f"\nWritten: {out_path}")
    print("Re-run probe to get calibrated profile with these thresholds.")

if __name__ == "__main__":
    main()

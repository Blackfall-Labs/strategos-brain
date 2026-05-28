"""Batch-run the probe brain on every vocab word and write output to file.

Reads the vocab from brain_probe_v2.brnc (embedded in checkpoint),
shells infer.exe once per word, writes all output to logs/probe_output.txt.

Usage:
    python tools/run_probe_all.py [--brain brain_probe_v2.brnc] [--infer path/to/infer.exe]
"""

import json, os, subprocess, argparse, struct

def read_brnc_vocab(path):
    """Read just the vocab list from a BRNC v5 checkpoint."""
    with open(path, "rb") as f:
        magic   = f.read(4)
        version = struct.unpack("<I", f.read(4))[0]
        n_cols  = struct.unpack("<I", f.read(4))[0]
        n_sheets= struct.unpack("<I", f.read(4))[0]
        proj_a  = struct.unpack("<i", f.read(4))[0]
        v_count = struct.unpack("<I", f.read(4))[0]
        vocab   = []
        for _ in range(v_count):
            wlen = struct.unpack("<I", f.read(4))[0]
            word = f.read(wlen).decode("utf-8")
            vocab.append(word)
    return vocab

def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain", default="brain_probe_v2.brnc")
    parser.add_argument("--infer", default=r"E:\repos\blackfall-labs\sigbrain\target\release\infer.exe")
    parser.add_argument("--out",   default="logs/probe_output.txt")
    args = parser.parse_args()

    if not os.path.exists(args.brain):
        print(f"ERROR: {args.brain} not found. Run probe_to_brnc.py first.")
        return
    if not os.path.exists(args.infer):
        print(f"ERROR: {args.infer} not found. Build sigbrain release first.")
        return

    vocab = read_brnc_vocab(args.brain)
    print(f"Vocab: {len(vocab)} words")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out_f:
        out_f.write(f"# probe brain: {args.brain}\n")
        out_f.write(f"# vocab size: {len(vocab)}\n\n")

        for word in vocab:
            result = subprocess.run(
                [args.infer, "--brain", args.brain, "--speak", word, "--max-len", "1"],
                capture_output=True, text=True, encoding="utf-8", timeout=10
            )
            # infer.exe prints: "Loaded: ...\nthe → word" or "the → (no continuation)"
            lines = result.stdout.strip().splitlines()
            pred_line = next((l for l in lines if "->" in l or "→" in l), None)
            if pred_line:
                out_f.write(pred_line.strip() + "\n")
                print(pred_line.strip())
            else:
                out_f.write(f"{word} -> (error)\n")
                print(f"{word} -> (error): stdout={repr(result.stdout[:80])} stderr={result.stderr.strip()[:80]}")

    print(f"\nSaved -> {args.out}")

if __name__ == "__main__":
    main()

"""Run the exact Qwen2.5-0.5B forward pass over the probe vocabulary and record
ground-truth next-token predictions.

This IS the model. All 24 layers, all weights including v_proj.bias.
Output is written to logs/qwen25_baseline.json — never printed live.

Usage:
    python tools/qwen25_baseline.py [--model-dir models/qwen25_05b] [--bands qwen25_bands.json]
"""

import json, os, sys, time, argparse
import numpy as np

HIDDEN    = 896
FFN_DIM   = 4864
NUM_HEADS    = 14    # Qwen2.5-0.5B: 14 Q heads
NUM_KV_HEADS = 2     # 2 KV heads (GQA: 7 Q groups per KV)
HEAD_DIM     = 64    # 896 / 14
N_LAYERS  = 24

def _silu(x):
    return x / (1.0 + np.exp(-x))

def _rms_norm(H, w, eps=1e-6):
    rms = np.sqrt((H * H).mean(axis=-1, keepdims=True) + eps)
    return H / rms * w

BPE_SPACE = "Ġ"   # Ġ (U+0120) — BPE word-start prefix used by Qwen2.5

def load_bpe_vocab(model_dir):
    return json.load(open(os.path.join(model_dir, "tokenizer.json"), encoding="utf-8"))["model"]["vocab"]

def resolve_token_ids(vocab, bpe_vocab):
    return [bpe_vocab.get(BPE_SPACE + w, bpe_vocab.get(w)) for w in vocab]

def strip_bpe(token):
    return token.replace(BPE_SPACE, "").lower().strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/qwen25_05b")
    parser.add_argument("--bands",     default="qwen25_bands.json")
    parser.add_argument("--out",       default="logs/qwen25_baseline.json")
    parser.add_argument("--top-k",     type=int, default=5)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ── Load vocabulary from bands ──────────────────────────────────────
    from collections import defaultdict
    bands_data = json.load(open(args.bands, encoding="utf-8"))
    token_weight = defaultdict(int)
    for band_info in bands_data["bands"]:
        if band_info["band"] == 0:
            continue
        for cluster in band_info.get("clusters", []):
            tok = cluster["trigger_token"].replace(BPE_SPACE, "").lower().strip()
            if tok and tok.isalpha():
                for n in cluster["neurons"]:
                    if n.get("trigger_rate", 0) >= 0.4:
                        token_weight[tok] += 1
    vocab = sorted(sorted(token_weight, key=lambda w: -token_weight[w])[:4000])
    print(f"Vocab: {len(vocab)} words", flush=True)

    # ── Load tokenizer ──────────────────────────────────────────────────
    bpe_vocab = load_bpe_vocab(args.model_dir)
    token_ids = [bpe_vocab.get(BPE_SPACE + w, bpe_vocab.get(w)) for w in vocab]
    valid     = [(i, tid) for i, tid in enumerate(token_ids) if tid is not None]
    print(f"Tokenizer coverage: {len(valid)}/{len(vocab)}", flush=True)

    V = len(vocab)

    # ── Load weights ────────────────────────────────────────────────────
    def wbin(fname):
        return np.fromfile(os.path.join(args.model_dir, fname), dtype=np.float32)

    print("Loading embed_tokens...", flush=True)
    embed      = wbin("weight_00000.bin").reshape(151936, HIDDEN)
    final_norm = wbin("weight_00001.bin")

    H = np.zeros((V, HIDDEN), dtype=np.float32)
    for i, tid in valid:
        H[i] = embed[tid]

    # ── Forward pass ────────────────────────────────────────────────────
    t0 = time.perf_counter()

    for layer in range(N_LAYERS):
        base = 2 + 9 * layer

        def wt(offset, shape):
            return wbin(f"weight_{base + offset:05d}.bin").reshape(shape)

        ln1_w  = wt(0, (HIDDEN,))
        v_w    = wt(3, (NUM_KV_HEADS * HEAD_DIM, HIDDEN))
        o_w    = wt(4, (HIDDEN, HIDDEN))
        ln2_w  = wt(5, (HIDDEN,))
        gate_w = wt(6, (FFN_DIM, HIDDEN))
        up_w   = wt(7, (FFN_DIM, HIDDEN))
        down_w = wt(8, (HIDDEN, FFN_DIM))

        vb_path = os.path.join(args.model_dir, f"v_bias_{layer:02d}.bin")
        v_bias  = wbin(f"v_bias_{layer:02d}.bin") if os.path.exists(vb_path) else None

        h1 = _rms_norm(H, ln1_w)

        # Single-token: attn_out = repeat_kv(V, num_kv_groups) @ o_proj
        v_out      = h1 @ v_w.T                           # [V, NUM_KV_HEADS*HEAD_DIM]
        if v_bias is not None:
            v_out  = v_out + v_bias
        num_kv_groups = NUM_HEADS // NUM_KV_HEADS          # 7
        v_expanded    = np.repeat(
            v_out.reshape(V, NUM_KV_HEADS, HEAD_DIM),
            num_kv_groups, axis=1
        ).reshape(V, HIDDEN)
        attn_out   = v_expanded @ o_w.T
        H          = H + attn_out

        h2           = _rms_norm(H, ln2_w)
        intermediate = _silu(h2 @ gate_w.T) * (h2 @ up_w.T)
        H            = H + (intermediate @ down_w.T)

        print(f"  layer {layer:2d} done", flush=True)

    h_finals = _rms_norm(H, final_norm)
    elapsed_fwd = time.perf_counter() - t0
    print(f"Forward pass: {elapsed_fwd:.2f}s", flush=True)

    # ── Compute logit predictions over vocab ────────────────────────────
    vocab_tids   = [tid for _, tid in valid]
    vocab_embed  = np.stack([embed[tid] for tid in vocab_tids])  # [V_valid, HIDDEN]

    results = {}
    for X_idx, X_tid in valid:
        word  = vocab[X_idx]
        h     = h_finals[X_idx]
        scores = vocab_embed @ h
        top_k  = np.argsort(scores)[-args.top_k:][::-1]
        preds  = [(vocab[valid[int(r)][0]], float(scores[r])) for r in top_k]
        results[word] = {
            "token_id":    int(X_tid),
            "top_preds":   preds,
            "top1_word":   preds[0][0] if preds else None,
        }

    # ── Save ────────────────────────────────────────────────────────────
    out = {
        "model":       "Qwen2.5-0.5B",
        "vocab_size":  len(vocab),
        "covered":     len(valid),
        "fwd_time_s":  elapsed_fwd,
        "top_k":       args.top_k,
        "predictions": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved -> {args.out}", flush=True)
    print("\nSample predictions:", flush=True)
    for w in list(results.keys())[:10]:
        top = [p[0] for p in results[w]["top_preds"][:3]]
        print(f"  {w:12s} -> {top}", flush=True)

if __name__ == "__main__":
    main()

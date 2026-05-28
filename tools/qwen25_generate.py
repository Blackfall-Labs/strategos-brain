"""Full Qwen2.5-0.5B autoregressive text generation — pure numpy, no ML libraries.

This is the Phase 1 reference implementation. All operations are exact:
  - BPE tokenizer built from tokenizer.json merge rules
  - Q/K/V projections with biases at every layer
  - RoPE (rope_theta=1_000_000.0) applied to Q and K
  - Causal attention with softmax over full sequence
  - GQA: 14 Q heads, 2 KV heads, 7 Q-groups per KV head
  - SwiGLU FFN (gate * silu + up, projected by down)
  - RMSNorm at input and post-attention per layer
  - Final RMSNorm + tied lm_head (= embed_tokens.T)
  - Autoregressive generation loop over full 151,936-token vocab

Output must match HuggingFace Qwen2.5-0.5B token-for-token.
All output written to file — never printed live.

Usage:
    python tools/qwen25_generate.py --prompt "Hello" --max-tokens 50
    python tools/qwen25_generate.py --prompt "Hello" --max-tokens 50 --out logs/gen_out.txt
    python tools/qwen25_generate.py --chat --prompt "Hello" --max-tokens 200
"""

import json, os, sys, struct, argparse, time, re
import numpy as np
from functools import lru_cache

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

HIDDEN       = 896
FFN_DIM      = 4864
NUM_HEADS    = 14
NUM_KV_HEADS = 2
HEAD_DIM     = 64       # HIDDEN / NUM_HEADS
N_LAYERS     = 24
VOCAB_SIZE   = 151936
ROPE_THETA   = 1_000_000.0
EOS_ID       = 151643   # <|endoftext|>
IM_END_ID    = 151645   # <|im_end|>

# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------

def _bytes_to_unicode():
    """GPT-2 byte-to-unicode map. Every byte maps to a printable unicode char."""
    bs = (list(range(ord('!'), ord('~') + 1)) +
          list(range(ord('¡'), ord('¬') + 1)) +
          list(range(ord('®'), ord('ÿ') + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

_BYTE_ENC = _bytes_to_unicode()
_BYTE_DEC = {v: k for k, v in _BYTE_ENC.items()}

# Use the 'regex' package if available for \p{L}/\p{N} support; fall back
# to the stdlib re with an approximation.
try:
    import regex as _re_module
    _PRETOK_RE = _re_module.compile(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        r"|[^\r\n\p{L}\p{N}]?\p{L}+"
        r"|\p{N}"
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
        r"|\s*[\r\n]+"
        r"|\s+(?!\S)"
        r"|\s+"
    )
    _HAS_REGEX = True
except ImportError:
    # stdlib re doesn't support \p{} — use a close approximation
    _PRETOK_RE = re.compile(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        r"|[^\r\nA-Za-z0-9À-ÖØ-öø-ÿ]?"
        r"[A-Za-zÀ-ɏЀ-ӿ]+"
        r"|[0-9]"
        r"| ?[^\s\w]+"
        r"|\s*[\r\n]+"
        r"|\s+(?!\S)"
        r"|\s+",
        re.UNICODE,
    )
    _HAS_REGEX = False


class BPETokenizer:
    def __init__(self, tokenizer_json_path):
        with open(tokenizer_json_path, encoding="utf-8") as f:
            data = json.load(f)

        model = data["model"]
        self.vocab    = model["vocab"]                      # str → id
        self.id2token = {v: k for k, v in self.vocab.items()}

        # Build merge priority: (a, b) → rank
        self.bpe_ranks = {tuple(m.split(" ")): i
                          for i, m in enumerate(model["merges"])}

        # Added/special tokens
        for t in data.get("added_tokens", []):
            self.vocab[t["content"]]       = t["id"]
            self.id2token[t["id"]]         = t["content"]

        self._cache = {}

    def _bpe(self, token_chars):
        """Apply BPE merges to a tuple of byte-level chars."""
        if token_chars in self._cache:
            return self._cache[token_chars]

        word = list(token_chars)
        while len(word) > 1:
            # Find the highest-priority (lowest rank) adjacent pair
            best_rank = float("inf")
            best_pair = None
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.bpe_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank, best_pair = rank, pair
            if best_pair is None or best_rank == float("inf"):
                break
            # Merge all occurrences of best_pair
            a, b = best_pair
            merged = a + b
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        result = tuple(word)
        self._cache[token_chars] = result
        return result

    def encode(self, text):
        """Encode UTF-8 text to a list of token IDs.

        Special added tokens (e.g. <|im_start|>) are matched as whole strings
        before BPE runs — they never pass through the merge algorithm.
        """
        import unicodedata
        text = unicodedata.normalize("NFC", text)

        # Split out special tokens first (longest-match, left-to-right)
        special = sorted(
            [t for t in self.vocab if t.startswith("<|") and t.endswith("|>")],
            key=len, reverse=True,
        )
        pattern = "|".join(re.escape(s) for s in special)
        segments = re.split(f"({pattern})", text) if special else [text]

        ids = []
        for seg in segments:
            if not seg:
                continue
            if seg in self.vocab and seg.startswith("<|"):
                ids.append(self.vocab[seg])
                continue
            # Regular text: NFC already applied above
            for piece in _PRETOK_RE.findall(seg):
                byte_chars = tuple(_BYTE_ENC[b] for b in piece.encode("utf-8"))
                for tok_str in self._bpe(byte_chars):
                    tid = self.vocab.get(tok_str)
                    if tid is not None:
                        ids.append(tid)
                    else:
                        for c in tok_str:
                            ids.append(self.vocab.get(c, 0))
        return ids

    def decode(self, ids):
        """Decode a list of token IDs to UTF-8 text."""
        tokens = []
        for i in ids:
            t = self.id2token.get(i, "")
            # Special tokens pass through as-is (they're not byte-encoded)
            if t.startswith("<|") and t.endswith("|>"):
                tokens.append(t)
            else:
                tokens.append(t)
        raw = "".join(tokens)
        # Decode byte-level characters back to bytes
        byte_seq = []
        for c in raw:
            if c in _BYTE_DEC:
                byte_seq.append(_BYTE_DEC[c])
            else:
                # Pass-through for special token text
                byte_seq.extend(c.encode("utf-8"))
        return bytes(byte_seq).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

class Qwen25Weights:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self._cache = {}

    def _load(self, fname, shape):
        path = os.path.join(self.model_dir, fname)
        if path not in self._cache:
            self._cache[path] = np.fromfile(path, dtype=np.float32).reshape(shape)
        return self._cache[path]

    def _load_bias(self, fname, size):
        path = os.path.join(self.model_dir, fname)
        if not os.path.exists(path):
            return None
        if path not in self._cache:
            self._cache[path] = np.fromfile(path, dtype=np.float32)
        return self._cache[path]

    @property
    def embed(self):
        return self._load("weight_00000.bin", (VOCAB_SIZE, HIDDEN))

    @property
    def final_norm(self):
        return self._load("weight_00001.bin", (HIDDEN,))

    def layer(self, i):
        base = 2 + 9 * i
        def w(off, shape): return self._load(f"weight_{base+off:05d}.bin", shape)
        def b(prefix):     return self._load_bias(f"{prefix}_{i:02d}.bin",
                                                   {"q": NUM_HEADS*HEAD_DIM,
                                                    "k": NUM_KV_HEADS*HEAD_DIM,
                                                    "v": NUM_KV_HEADS*HEAD_DIM}[prefix[0]])
        return {
            "ln1":    w(0, (HIDDEN,)),
            "q_w":    w(1, (NUM_HEADS * HEAD_DIM,    HIDDEN)),
            "k_w":    w(2, (NUM_KV_HEADS * HEAD_DIM, HIDDEN)),
            "v_w":    w(3, (NUM_KV_HEADS * HEAD_DIM, HIDDEN)),
            "o_w":    w(4, (HIDDEN, HIDDEN)),
            "ln2":    w(5, (HIDDEN,)),
            "gate_w": w(6, (FFN_DIM, HIDDEN)),
            "up_w":   w(7, (FFN_DIM, HIDDEN)),
            "down_w": w(8, (HIDDEN, FFN_DIM)),
            "q_b":    b("q_bias"),
            "k_b":    b("k_bias"),
            "v_b":    b("v_bias"),
        }


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------

def rms_norm(x, w, eps=1e-6):
    """x: [..., d], w: [d]"""
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return x / rms * w

def silu(x):
    return x / (1.0 + np.exp(-x))

def rope_cos_sin(seq_len, head_dim, theta, positions=None):
    """Returns cos, sin each [seq_len, head_dim]."""
    if positions is None:
        positions = np.arange(seq_len, dtype=np.float32)
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = np.outer(positions, inv_freq)          # [seq_len, head_dim/2]
    emb   = np.concatenate([freqs, freqs], axis=1) # [seq_len, head_dim]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

def rotate_half(x):
    """x: [..., head_dim] — negate second half, swap halves."""
    d = x.shape[-1] // 2
    return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)

def apply_rope(x, cos, sin):
    """x: [seq_len, n_heads, head_dim], cos/sin: [seq_len, head_dim]"""
    c = cos[:, None, :]  # [seq_len, 1, head_dim]
    s = sin[:, None, :]
    return x * c + rotate_half(x) * s

def repeat_kv(x, n_rep):
    """x: [seq_len, n_kv, hd] → [seq_len, n_kv*n_rep, hd]"""
    if n_rep == 1:
        return x
    return np.repeat(x, n_rep, axis=1)

def causal_mask(seq_len):
    """Upper-triangular -inf mask: [seq_len, seq_len]"""
    m = np.zeros((seq_len, seq_len), dtype=np.float32)
    m[np.triu_indices(seq_len, k=1)] = -np.inf
    return m

def attention(Q, K, V, mask):
    """
    Q: [seq_len, n_heads,    head_dim]
    K: [seq_len, n_heads,    head_dim]  (after GQA expansion)
    V: [seq_len, n_heads,    head_dim]
    mask: [seq_len, seq_len]
    Returns: [seq_len, n_heads * head_dim]
    """
    seq_len, n_heads, hd = Q.shape
    scale = 1.0 / np.sqrt(hd)

    Q = Q.transpose(1, 0, 2)   # [n_heads, seq_len, hd]
    K = K.transpose(1, 0, 2)
    V = V.transpose(1, 0, 2)

    scores = np.matmul(Q, K.transpose(0, 2, 1)) * scale  # [n_heads, seq_len, seq_len]
    scores = scores + mask[None, :, :]

    # Stable softmax
    scores -= scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(axis=-1, keepdims=True)

    out = np.matmul(w, V)               # [n_heads, seq_len, hd]
    return out.transpose(1, 0, 2).reshape(seq_len, n_heads * hd)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward(token_ids, weights, verbose=False):
    """
    Full Qwen2.5-0.5B forward pass.
    token_ids: list of ints
    Returns: logits [VOCAB_SIZE] for the last position
    """
    seq_len = len(token_ids)
    ids     = np.array(token_ids, dtype=np.int32)
    H       = weights.embed[ids].copy()   # [seq_len, HIDDEN]

    mask = causal_mask(seq_len)
    cos, sin = rope_cos_sin(seq_len, HEAD_DIM, ROPE_THETA)

    n_kv_groups = NUM_HEADS // NUM_KV_HEADS  # 7

    for i in range(N_LAYERS):
        lw = weights.layer(i)
        if verbose:
            print(f"  layer {i:2d}", flush=True)

        # --- Attention ---
        h1 = rms_norm(H, lw["ln1"])                             # [seq, 896]

        Q = (h1 @ lw["q_w"].T)                                  # [seq, 896]
        K = (h1 @ lw["k_w"].T)                                  # [seq, 128]
        V = (h1 @ lw["v_w"].T)                                  # [seq, 128]

        if lw["q_b"] is not None: Q = Q + lw["q_b"]
        if lw["k_b"] is not None: K = K + lw["k_b"]
        if lw["v_b"] is not None: V = V + lw["v_b"]

        Q = Q.reshape(seq_len, NUM_HEADS,    HEAD_DIM)          # [seq, 14, 64]
        K = K.reshape(seq_len, NUM_KV_HEADS, HEAD_DIM)          # [seq,  2, 64]
        V = V.reshape(seq_len, NUM_KV_HEADS, HEAD_DIM)

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        K = repeat_kv(K, n_kv_groups)                           # [seq, 14, 64]
        V = repeat_kv(V, n_kv_groups)

        attn_out = attention(Q, K, V, mask)                      # [seq, 896]
        H = H + attn_out @ lw["o_w"].T

        # --- FFN ---
        h2 = rms_norm(H, lw["ln2"])
        H  = H + (silu(h2 @ lw["gate_w"].T) * (h2 @ lw["up_w"].T)) @ lw["down_w"].T

    h_final = rms_norm(H[-1], weights.final_norm)                # [896]
    logits  = weights.embed @ h_final                            # [VOCAB_SIZE] — tied lm_head
    return logits


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(prompt_ids, weights, max_tokens, eos_ids, out_f, verbose=False):
    """
    Autoregressive generation. Writes each token to out_f as it is produced.
    Returns list of generated token IDs.
    """
    tokens    = list(prompt_ids)
    generated = []
    t0        = time.perf_counter()

    for step in range(max_tokens):
        logits    = forward(tokens, weights, verbose=verbose and step == 0)
        next_id   = int(np.argmax(logits))
        tokens.append(next_id)
        generated.append(next_id)

        elapsed = time.perf_counter() - t0
        tok_s   = (step + 1) / elapsed
        out_f.write(f"step {step+1:4d}  token={next_id:7d}  tok/s={tok_s:6.2f}\n")
        out_f.flush()

        if next_id in eos_ids:
            break

    return generated


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------

CHAT_TEMPLATE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

def apply_chat_template(prompt):
    return CHAT_TEMPLATE.format(prompt=prompt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt",     default="Hello",
                        help="Input text (default: 'Hello')")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Maximum tokens to generate (default: 50)")
    parser.add_argument("--model-dir",  default="models/qwen25_05b")
    parser.add_argument("--out",        default="logs/gen_out.txt",
                        help="Output file for generated tokens (default: logs/gen_out.txt)")
    parser.add_argument("--chat",       action="store_true",
                        help="Wrap prompt in Qwen2.5 ChatML template")
    parser.add_argument("--verbose",    action="store_true",
                        help="Print layer progress during first forward pass")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as out_f:
        out_f.write(f"prompt: {args.prompt!r}\n")
        out_f.write(f"chat_mode: {args.chat}\n")
        out_f.write(f"max_tokens: {args.max_tokens}\n\n")

        # Tokenize
        print("Loading tokenizer...", flush=True)
        tok = BPETokenizer(os.path.join(args.model_dir, "tokenizer.json"))

        text = apply_chat_template(args.prompt) if args.chat else args.prompt
        prompt_ids = tok.encode(text)
        out_f.write(f"prompt_ids ({len(prompt_ids)}): {prompt_ids}\n")
        out_f.write(f"prompt_decoded: {tok.decode(prompt_ids)!r}\n\n")
        print(f"Prompt: {len(prompt_ids)} tokens: {prompt_ids}", flush=True)

        # Load weights
        print("Loading weights...", flush=True)
        weights = Qwen25Weights(args.model_dir)
        # Pre-load embed and final_norm
        _ = weights.embed
        _ = weights.final_norm
        print(f"Embed matrix: {weights.embed.shape}", flush=True)

        # Generate
        print(f"Generating (max {args.max_tokens} tokens)...", flush=True)
        out_f.write("=== GENERATION ===\n")
        eos_ids = {EOS_ID, IM_END_ID}

        t_start = time.perf_counter()
        generated = generate(prompt_ids, weights, args.max_tokens, eos_ids,
                             out_f, verbose=args.verbose)
        elapsed = time.perf_counter() - t_start

        # Decode and write result
        full_text   = tok.decode(prompt_ids + generated)
        prompt_text = tok.decode(prompt_ids)
        gen_text    = tok.decode(generated)

        out_f.write(f"\n=== RESULT ===\n")
        out_f.write(f"generated_ids ({len(generated)}): {generated}\n\n")
        out_f.write(f"PROMPT:\n{prompt_text}\n\n")
        out_f.write(f"GENERATED:\n{gen_text}\n\n")
        out_f.write(f"FULL:\n{full_text}\n\n")
        out_f.write(f"elapsed: {elapsed:.2f}s  "
                    f"tok/s: {len(generated)/elapsed:.2f}  "
                    f"total_tokens: {len(prompt_ids)+len(generated)}\n")

        print(f"\nGenerated {len(generated)} tokens in {elapsed:.2f}s "
              f"({len(generated)/elapsed:.2f} tok/s)", flush=True)
        print(f"Output -> {args.out}", flush=True)
        print(f"\nGENERATED TEXT:\n{gen_text}", flush=True)


if __name__ == "__main__":
    main()

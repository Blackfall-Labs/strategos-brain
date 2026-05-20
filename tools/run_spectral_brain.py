"""Run the spectral brain with proper BPE tokenization.

Loads SmolLM2 tokenizer, BPE-encodes the corpus, then feeds each token ID
into the brain topology's token map. Replaces the whitespace-split approach
in the Rust binary for proper coverage measurement.

Usage:
    python tools/run_spectral_brain.py \\
        --topology brain_topology.json \\
        --tokenizer models/smollm2/tokenizer.json \\
        --corpus data/wikitext2_train.txt \\
        --n-tokens 5000
"""

import json, argparse, re, sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')


# ── Minimal BPE encoder ──────────────────────────────────────────────────────

def load_bpe(tokenizer_path):
    with open(tokenizer_path, encoding='utf-8') as f:
        tok = json.load(f)
    model  = tok['model']
    vocab  = model['vocab']                    # str → int
    merges = {tuple(m.split(' ')): i for i, m in enumerate(model['merges'])}
    return vocab, merges


def bpe_encode_word(word, vocab, merges):
    """Encode a single pre-tokenized word into token IDs using BPE merge rules."""
    # Split into individual UTF-8 bytes mapped through the Ġ/Ċ byte-level encoding
    # SmolLM2 uses byte-level BPE: each byte maps to a unicode char in vocab
    chars = list(byte_to_char(word))
    if not chars:
        return []

    # Iteratively apply the lowest-ranked merge
    while len(chars) > 1:
        best_rank = None
        best_idx  = -1
        for i in range(len(chars) - 1):
            pair = (chars[i], chars[i + 1])
            rank = merges.get(pair)
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_idx  = i
        if best_idx == -1:
            break
        merged = chars[best_idx] + chars[best_idx + 1]
        chars  = chars[:best_idx] + [merged] + chars[best_idx + 2:]

    return [vocab[c] for c in chars if c in vocab]


# Byte-level encoding table (same as GPT-2 / SmolLM2)
def _build_byte_table():
    bs = list(range(ord('!'), ord('~') + 1)) + \
         list(range(ord('¡'), ord('¬') + 1)) + \
         list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n  = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

BYTE_TABLE = _build_byte_table()

def byte_to_char(text):
    """Convert a raw text string to byte-level BPE characters."""
    return ''.join(BYTE_TABLE[b] for b in text.encode('utf-8'))


def tokenize(text, vocab, merges):
    """Tokenize text the same way SmolLM2 does: split on whitespace boundaries,
    prefix non-first words with Ġ, BPE encode each piece."""
    # Split keeping whitespace positions
    tokens = []
    words  = re.findall(r'\s*\S+', text)
    for i, word in enumerate(words):
        # Leading whitespace → prepend Ġ to the word
        stripped = word.lstrip()
        if word != stripped:
            piece = 'Ġ' + stripped   # Ġ prefix
        else:
            piece = stripped
        ids = bpe_encode_word(piece, vocab, merges)
        tokens.extend(ids)
    return tokens


# ── Brain sim (mirrors the Rust logic) ───────────────────────────────────────

class Column:
    __slots__ = ('id','tier','band_lo','band_hi','threshold','k','k_reset','always_on')
    def __init__(self, id, tier, band_lo, band_hi, threshold, k_reset, always_on):
        self.id = id; self.tier = tier
        self.band_lo = band_lo; self.band_hi = band_hi
        self.threshold = threshold
        self.k = 0; self.k_reset = k_reset
        self.always_on = always_on

    def evaluate(self, band_id, magnitude, src_tier):
        if self.k > 0: return 0
        if abs(self.tier - src_tier) > 1: return 0
        if band_id < self.band_lo or band_id > self.band_hi: return 0
        span   = max(self.band_hi - self.band_lo, 1)
        centre = (self.band_lo + self.band_hi) // 2
        dist   = abs(band_id - centre)
        gain   = max(0, 65536 - dist * 49152 // span)
        eff    = magnitude * gain // 65536
        if eff >= self.threshold:
            self.k = self.k_reset
            return eff
        return 0

    def tick(self):
        if self.k > 0: self.k -= 1


def trigger_rate_to_threshold(tr):
    return int(2000 + (1 - min(1, max(0, tr))) * 18000)

def pv_scale_to_threshold(pv):
    return int(min(2.0, max(0.5, pv)) / 2.0 * 20000)

def hist_to_pv_scales(hist, n):
    if not hist: return [1.0] * n
    pv_list = []
    for i, frac in enumerate(hist):
        mid = (i + 0.5) / len(hist)
        pv  = 2.0 - 1.5 * mid
        pv_list.extend([pv] * max(1, round(frac * n)))
    if len(pv_list) >= n: return pv_list[:n]
    return pv_list + [1.0] * (n - len(pv_list))


def build_columns(topo):
    cols = []
    idx  = 0

    for cluster in topo['tier0_clusters']:
        for m in cluster['members']:
            th = trigger_rate_to_threshold(m['trigger_rate'])
            cols.append(Column(idx, 0, cluster['band_id'], cluster['band_id'], th, 3, False))
            idx += 1

    for sheet in topo['tier1_sheets'] + topo['tier2_sheets']:
        pv    = sheet.get('pv_scales', [])
        ao_n  = round(sheet['n_cols'] * sheet.get('always_on_frac', 0.0))
        k_rst = 5 if sheet['tier'] == 2 else 4
        for i in range(sheet['n_cols']):
            p  = pv[i] if i < len(pv) else 1.0
            th = pv_scale_to_threshold(p)
            cols.append(Column(idx, sheet['tier'], sheet['band_lo'], sheet['band_hi'],
                               th, k_rst, i < ao_n))
            idx += 1
    return cols


def run(topo_path, tokenizer_path, corpus_path, n_tokens, ticks_per_token):
    with open(topo_path, encoding='utf-8') as f:
        topo = json.load(f)

    vocab_bpe, merges = load_bpe(tokenizer_path)

    # Token → (band_id, magnitude) map
    token_map = {}
    for t in topo.get('band23_tokens', []):
        token_map[t['token_id']] = (t['band_id'], t['magnitude'])
    for c in topo['tier0_clusters']:
        members = c['members']
        mean_tr = sum(m['trigger_rate'] for m in members) / max(len(members), 1)
        token_map[c['token_id']] = (c['band_id'], int(mean_tr * 30000))

    cols = build_columns(topo)
    always_on_per_tick = sum(1 for c in cols if c.always_on)
    print(f"[brain] columns: {len(cols)}  always-on: {always_on_per_tick}  token_map: {len(token_map)}")

    print(f"[brain] tokenizing corpus...")
    with open(corpus_path, encoding='utf-8') as f:
        text = f.read()
    token_ids = tokenize(text, vocab_bpe, merges)
    limit = min(n_tokens, len(token_ids)) if n_tokens else len(token_ids)
    print(f"[brain] corpus tokens: {len(token_ids)}, running first {limit}")

    hits = misses = 0
    total_cascade   = [0, 0, 0]
    total_always_on = [0, 0, 0]

    for i, tid in enumerate(token_ids[:limit]):
        pending = []
        if tid in token_map:
            band_id, magnitude = token_map[tid]
            pending.append((0, band_id, magnitude))
            hits += 1
        else:
            misses += 1

        tok_cascade   = [0, 0, 0]
        tok_always_on = [0, 0, 0]

        for _ in range(ticks_per_token):
            nxt = []
            # Always-on fire
            for col in cols:
                if col.always_on and col.k == 0:
                    col.k = col.k_reset
                    nxt.append((col.tier, col.band_lo, col.threshold * 2))
                    tok_always_on[col.tier] += 1
            # Dispatch pending
            for src_tier, band_id, magnitude in pending:
                for col in cols:
                    fm = col.evaluate(band_id, magnitude, src_tier)
                    if fm > 0:
                        mid = col.band_lo + (col.band_hi - col.band_lo) // 2
                        nxt.append((col.tier, mid, fm))
                        if col.always_on:
                            tok_always_on[col.tier] += 1
                        else:
                            tok_cascade[col.tier] += 1
            pending = nxt
            for col in cols: col.tick()

        for t in range(3):
            total_cascade[t]   += tok_cascade[t]
            total_always_on[t] += tok_always_on[t]

        if (i + 1) % 500 == 0:
            print(f"[brain] token {i+1:5}  hits={hits} miss={misses}  "
                  f"T2 cascade={tok_cascade[2]} ao={tok_always_on[2]}  "
                  f"T1={tok_cascade[1]}  T0={tok_cascade[0]}")

    print(f"\n[brain] run complete")
    print(f"  tokens:    {limit}")
    print(f"  hits:      {hits} ({100*hits/limit:.1f}%)")
    print(f"  misses:    {misses}")
    print(f"  T0 cascade total: {total_cascade[0]}")
    print(f"  T1 cascade total: {total_cascade[1]}")
    print(f"  T2 cascade total: {total_cascade[2]}")
    print(f"  T2 always-on:     {total_always_on[2]}")
    print(f"  T2 cascade/hit:   {total_cascade[2]/max(hits,1):.1f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--topology',   required=True)
    ap.add_argument('--tokenizer',  required=True)
    ap.add_argument('--corpus',     required=True)
    ap.add_argument('--n-tokens',   type=int, default=5000)
    ap.add_argument('--ticks-per-token', type=int, default=8)
    args = ap.parse_args()
    run(args.topology, args.tokenizer, args.corpus, args.n_tokens, args.ticks_per_token)

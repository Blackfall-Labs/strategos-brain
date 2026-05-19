"""
Train a 1-layer transformer on modular addition (a + b mod 113).
Architecture from Neel Nanda's Progress Measures for Grokking paper.

Saves weights to models/grokking/ in manifest.json + .bin format
compatible with strategos-probe.

Usage: python3 tools/train_grokking.py
"""

import torch
import torch.nn as nn
import json
import struct
import os
import math

# --- Config ---
P = 113          # modulus
D_MODEL = 128
D_MLP = 512
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS
N_EPOCHS = 25000
LR = 1e-3
WEIGHT_DECAY = 1.0
TRAIN_FRAC = 0.3   # Neel uses 30% for grokking to occur
DEVICE = "cpu"

# --- Dataset ---
# All pairs (a, b) where a, b in [0, P). Labels = (a+b) mod P.
pairs = [(a, b) for a in range(P) for b in range(P)]
labels = [(a + b) % P for (a, b) in pairs]

data = torch.tensor(pairs, dtype=torch.long)       # [P*P, 2]
targets = torch.tensor(labels, dtype=torch.long)   # [P*P]

n_train = int(len(data) * TRAIN_FRAC)
idx = torch.randperm(len(data), generator=torch.Generator().manual_seed(42))
train_idx = idx[:n_train]
test_idx = idx[n_train:]

train_data, train_targets = data[train_idx], targets[train_idx]
test_data, test_targets = data[test_idx], targets[test_idx]

# --- Model ---
# Input: [a, b] — two tokens from vocab [0..P), plus one "=" token at P.
# We embed a and b, add positional embeddings, run 1 attention layer + MLP, read off logit at pos 2.
VOCAB = P + 1  # 0..P-1 are numbers, P is "="

class GrokTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.pos_embed = nn.Embedding(3, D_MODEL)  # positions 0, 1, 2

        # Attention
        self.W_Q = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_K = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_V = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.W_O = nn.Linear(D_MODEL, D_MODEL, bias=False)

        # MLP
        self.W_in  = nn.Linear(D_MODEL, D_MLP)
        self.W_out = nn.Linear(D_MLP, D_MODEL)

        # Unembed
        self.unembed = nn.Linear(D_MODEL, P, bias=False)

        # LayerNorm
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)

    def forward(self, a, b):
        # Build sequence: [a, b, =]
        eq = torch.full((a.shape[0],), P, dtype=torch.long)
        tokens = torch.stack([a, b, eq], dim=1)       # [B, 3]
        pos = torch.arange(3).unsqueeze(0)             # [1, 3]

        x = self.embed(tokens) + self.pos_embed(pos)  # [B, 3, D]

        # Attention (no causal mask — full bidirectional for this task)
        r = x
        x = self.ln1(x)
        B, T, D = x.shape
        Q = self.W_Q(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)  # [B, H, T, hd]
        K = self.W_K(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        V = self.W_V(x).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(HEAD_DIM)        # [B, H, T, T]
        attn = scores.softmax(dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T, D)               # [B, T, D]
        x = r + self.W_O(out)

        # MLP
        r = x
        x = self.ln2(x)
        x = r + self.W_out(torch.relu(self.W_in(x)))

        # Logits at "=" position (index 2)
        return self.unembed(x[:, 2, :])  # [B, P]

model = GrokTransformer().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# --- Train ---
print(f"Training grokking model: P={P}, d={D_MODEL}, heads={N_HEADS}, mlp={D_MLP}")
print(f"Train: {n_train} pairs, Test: {len(test_data)} pairs")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

best_test_acc = 0.0
for epoch in range(1, N_EPOCHS + 1):
    model.train()
    logits = model(train_data[:, 0], train_data[:, 1])
    loss = nn.functional.cross_entropy(logits, train_targets)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 500 == 0 or epoch == N_EPOCHS:
        model.eval()
        with torch.no_grad():
            test_logits = model(test_data[:, 0], test_data[:, 1])
            test_acc = (test_logits.argmax(-1) == test_targets).float().mean().item()
            train_acc = (logits.argmax(-1) == train_targets).float().mean().item()
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        print(f"epoch {epoch:5d}  loss={loss.item():.4f}  train={train_acc:.3f}  test={test_acc:.3f}  best_test={best_test_acc:.3f}")

print(f"\nFinal test accuracy: {best_test_acc:.3f}")
if best_test_acc < 0.95:
    print("WARNING: model did not fully grok (test acc < 0.95). Try more epochs.")

# --- Export to manifest.json + .bin ---
OUT = os.path.join(os.path.dirname(__file__), "..", "models", "grokking")
os.makedirs(OUT, exist_ok=True)

manifest = []
state = model.state_dict()

for name, tensor in state.items():
    t = tensor.float().cpu()
    flat = t.reshape(-1)
    fname = name.replace(".", "_") + ".bin"
    fpath = os.path.join(OUT, fname)
    with open(fpath, "wb") as f:
        f.write(struct.pack(f"{len(flat)}f", *flat.tolist()))
    manifest.append({
        "name": name,
        "file": fname,
        "shape": list(t.shape),
        "numel": flat.numel(),
    })
    print(f"  {name}: {list(t.shape)}")

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# Save config for the probe
config = {
    "model": "grokking",
    "P": P,
    "d_model": D_MODEL,
    "d_mlp": D_MLP,
    "n_heads": N_HEADS,
    "head_dim": HEAD_DIM,
    "vocab_size": VOCAB,
    "final_test_acc": best_test_acc,
}
with open(os.path.join(OUT, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\nWeights saved to {OUT}/")
print(f"manifest.json + {len(manifest)} .bin files")

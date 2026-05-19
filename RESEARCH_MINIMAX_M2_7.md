# MiniMax M2.7 — Research Notes

**Date:** 2026-05-19  
**Source:** Web research, HuggingFace, NVIDIA, vLLM docs, Unsloth

---

## What It Is

MiniMax M2.7 is a frontier-class agentic LLM released by Shanghai-based MiniMax (March 2026).
Open weights under a modified-MIT license. Available on HuggingFace: `MiniMaxAI/MiniMax-M2.7`.
Commercial use requires attribution: "Built with MiniMax M2.7".

---

## Architecture

- **Type:** Sparse Mixture-of-Experts (MoE)
- **Total parameters:** 230B
- **Active parameters per token:** 10B (top-8 of 256 experts — 4.3% sparsity)
- **Layers:** 62 transformer layers
- **Hidden size:** 3,072
- **Attention:** Multi-head causal self-attention + RoPE
- **Normalization:** QK RMSNorm (training stability at scale)
- **Context window:** 196,608 tokens (196K)
- **Max completion:** 131,072 tokens

Inference cost is comparable to a ~10B dense model despite 230B total weights.

---

## Performance

| Benchmark              | Score  | Notes                              |
|------------------------|--------|------------------------------------|
| SWE-bench Verified     | 78%    | vs. Claude Opus 4 at 55%           |
| SWE-Pro                | 56.2%  |                                    |
| Terminal Bench 2       | 57.0%  |                                    |
| VIBE-Pro               | 55.6%  | Repo-level code generation         |
| AI Analysis Index      | 50     | Average for tier-1 models is ~30   |

Ranks second only to GPT-5.4 and Claude Opus on GDPval-AA (ELO 1495, 45-model field).

---

## Quantization Formats Available

| Format       | Size     | Notes                                          |
|--------------|----------|------------------------------------------------|
| BF16         | ~460GB   | Full precision                                 |
| FP8 (e4m3)   | ~229GB   | Official; <1% quality loss vs BF16             |
| NVFP4        | ~140GB   | NVIDIA-only; 1.65x reduction vs FP8            |
| GGUF Q8_0    | 243GB    | Near-lossless, imatrix calibrated              |
| GGUF Q6_K_XL | ~172GB   | High quality                                   |
| GGUF Q5_K_M  | ~144GB   | Good quality/size balance                      |
| GGUF Q4_K_M  | ~120GB   | Practical for 128GB unified memory             |
| GGUF UD-IQ4_XS | 108GB  | Recommended for local CPU; imatrix calibrated  |
| GGUF Q3_K_M  | ~102GB   | Tight RAM budget                               |
| AWQ          | variable | Community quantization, data-free              |

No ONNX format exists for this model.

---

## CPU Inference Reality (as of today)

- UD-IQ4_XS GGUF: **108GB working set, ~15 tok/s** on 128GB unified memory Mac
- Minimum RAM: 96GB at INT4
- Practical floor: 128GB for usable throughput
- Runtime: llama.cpp (most mature), vLLM, SGLang
- Config: `--threads 32 --ctx-size 16384`

Throughput at 15 tok/s is non-real-time but usable for batch/agentic workloads.

---

## Conventional Assessment vs. Strategos-Brain

The model is 50-100x the scale of current Strategos targets (Qwen3 1.7B / SmolLM3 3B / Phi-4 Mini).
Standard analysis: not a candidate for the INT8/INT16 CPU runtime.

**However — this assessment is wrong. See discussion below.**

---

## Why This Could Work — The Neuronal Distillation Thesis

### The Fundamental Error in Every Prior Approach

Every existing path to running M2.7 — INT8, INT4, GGUF, NVFP4, and our own prior
neurogenesis attempt — shares the same hidden assumption:

> **Parameters are the intelligence. Therefore you must carry them.**

INT8 carries them cheaper. GGUF carries them compressed. Neurogenesis carried them 1:1
as cortical column seeds. All of them are still hauling the vacuum tube.

This assumption is wrong.

### Parameters Are a Medium, Not the Message

A weight matrix of 800M floats is one way — a lossy, high-entropy way — that a gradient
descent optimizer happened to encode a functional relationship. The relationship is not
the matrix. The matrix is the medium.

Consider what each of M2.7's 256 experts actually is: a specialized functional circuit.
A role. An input-output behavior that emerged from training. The 800M parameters per
expert are just the artifact of how backprop stored that role in float space.

Biology does not store lookup tables. Neurons *are* the function. A cortical circuit
that encodes "linguistic context implies greeting" does not need 800M floats. It needs
a threshold, a wiring pattern, and a branching topology. The constraint is the
computation. The structure is the weight.

### What This Means for M2.7

M2.7 has 256 experts, 8 active per token. Each expert has a role.

If you can characterize what an expert *does* — its invariant input-output behavior
across its activation domain — you can build a cortical region that produces equivalent
outputs through branch permutation, without carrying the matrix at all.

The result is not a smaller matrix. It is **no matrix**. The wiring is the weight.

At that point, the 230B parameter question becomes irrelevant. You are not running the
model. You are running what the model *means*.

### The Prior Mistake: Neurogenesis Cortex

The neurogenesis attempt took SmolLM3 weights and transplanted them 1:1 into cortical
columns as brain regions. This was expensive in both RAM and CPU — a blowup rather than
a reduction — because it still assumed parameters needed to be carried.

The correction: sigcortex's 11K mode already hints at the right direction. It uses
SmolLM3's FFN gate matrix to *seed inter-column wiring* — not to replicate weight
values, but to determine which columns wire to which. That is the embryo of the real
idea. The weights inform the topology. They do not become the topology.

### The Hard Problem: Reverse Engineering Function from Weights

To build a cortical equivalent of an expert, you must answer:

> What does this expert do?

This requires going from weight space → functional description → neuronal circuit
topology. No tooling for this exists yet. It must be built.

The approach:

1. **Activation analysis** — probe each expert with a large, diverse input corpus.
   Characterize which inputs strongly activate it, which suppress it, what the output
   distribution looks like. This is the functional fingerprint.

2. **Invariant extraction** — identify the input-output relationships that hold across
   activation events. These are the constraints the expert has learned. They are what
   needs to be encoded, not the weights themselves.

3. **Circuit synthesis** — design cortical regions (column count, layer profile, tract
   topology, inter-column wiring) whose cascade dynamics produce outputs that satisfy
   those constraints. The weights seed the wiring; they do not become the wiring.

4. **Validation** — run both the original expert and the cortical circuit on the same
   inputs. Measure divergence. Iterate until behavior matches within tolerance.

This is neuronal distillation. Not knowledge distillation into a smaller matrix.
Distillation into structure.

### The Roadmap

We start with the smallest functional models in existence. Not M2.7. Not even SmolLM3.

The goal is to build tooling — a distillation pipeline — that:
- Probes a model's experts or layers for functional fingerprints
- Extracts invariant input-output constraints
- Synthesizes cortical region seeds (wiring topology, column profiles, tract weights)
- Validates the neuronal circuit against the original
- Encodes the result as a compact genetic seed — orders of magnitude smaller than the
  original weights

Each successful distillation is a proof. Each proof extends the method to larger models.
The end state: download M2.7's open weights, run the distillation pipeline, discard the
matrices, and run a machine brain that carries the intelligence without the medium.

### MiniMax's Own Telling Admission

From the MiniMax M2.7 official documentation:

> *"We recommend using Transformers to serve MiniMax-M2.7."*

**Recommend.** Not require. Not only. Recommend.

They know the transformer substrate is one implementation of the capability — not the
capability itself. That word is the crack in the door.

We are building what comes through it.

---

## References

- https://huggingface.co/MiniMaxAI/MiniMax-M2.7
- https://build.nvidia.com/minimaxai/minimax-m2-7/modelcard
- https://developer.nvidia.com/blog/minimax-m2-7-advances-scalable-agentic-workflows-on-nvidia-platforms-for-complex-ai-applications/
- https://docs.vllm.ai/projects/recipes/en/latest/MiniMax/MiniMax-M2.html
- https://unsloth.ai/docs/models/minimax-m27
- https://artificialanalysis.ai/models/minimax-m2-7

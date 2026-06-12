---
title: "Nemotron-H: How NVIDIA Swaps Most Attention for Mamba-2 to Serve Long Context at Constant Memory"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A deep dive into the Nemotron-H report — a hybrid Mamba-2/Transformer that keeps only ~8% of layers as attention, trains in FP8, compresses with MiniPuzzle, and serves long context 2-3x faster than dense Transformers at the same accuracy."
tags: ["llm", "nemotron-h", "nvidia", "mamba", "state-space-model", "hybrid-architecture", "kv-cache", "fp8", "long-context", "minipuzzle", "inference-efficiency", "transformer"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 49
---

## The KV-cache wall, and a way around it

Every team that serves large language models at long context eventually hits the same wall, and it is not compute — it is memory. The culprit is the **KV cache**: self-attention has to remember the keys and values of every token it has already seen, so as the context grows, the cache grows with it, linearly, without bound. At a 64k-token context a 70B dense model's KV cache runs to *tens of gigabytes*, it crowds out the batch size you can fit on a GPU, and decode becomes memory-bandwidth-bound — you are not waiting on math, you are waiting on the GPU to stream a giant cache through its memory system on every single token. Reasoning models, which run at long context by their nature, feel this most acutely. The [KV cache deep dive](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) covers the mechanics; the short version is that attention's memory is the binding constraint on long-context serving.

Nemotron-H is NVIDIA's structural answer: **replace most of the attention layers with Mamba-2 state-space layers**, keeping only about 8% of layers as attention. A Mamba-2 layer does not keep a growing cache — it carries a single *fixed-size recurrent state* and updates it one token at a time, so its memory and compute per token are *constant* regardless of how long the context is. Build a model that is mostly Mamba-2 with a sprinkle of attention and you get a model that matches the accuracy of a strong dense Transformer while serving long context **2.4× to 3× faster**. Nemotron-H-56B matches or beats Llama-3.1-70B on 16 of 17 benchmarks (MMLU 84.2, GSM8K 93.7) at 2.4× the long-context throughput; the 8B is 3× faster than Llama-3.1-8B.

This is the fourth post in a series reading NVIDIA's model reports for their reusable techniques, after [Minitron's pruning](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation), [Nemotron-4 340B's synthetic-data alignment](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment), and [Llama-Nemotron's architecture search](/blog/machine-learning/large-language-model/llama-nemotron-efficient-reasoning-models). It draws on the [Nemotron-H technical report](https://arxiv.org/abs/2504.03624). The reusable techniques here are three: the **hybrid Mamba-2/attention recipe** (which layers to keep as attention and why), the **FP8 training recipe** (how to train at half the memory without losing accuracy), and **MiniPuzzle** (a Minitron-style compression adapted to the hybrid).

The mismatch the whole report resolves:

| Question | The Transformer-era assumption | What Nemotron-H shows |
|---|---|---|
| Do you need attention in every layer? | Yes, attention is the model | No — ~92% of layers can be Mamba-2 or FFN |
| What bounds long-context serving? | Compute | Memory — the KV cache |
| Is sequence mixing only attention's job? | Effectively yes | A state-space recurrence mixes sequence at constant memory |
| Does FP8 training hurt accuracy? | It is risky | Done right, FP8 matches or beats BF16 |
| How do you compress a hybrid model? | Unclear — pruning assumes Transformers | MiniPuzzle: importance + NAS + distillation, hybrid-aware |
| Must you choose speed or accuracy? | Usually | No — match the Transformer's accuracy at 2-3x throughput |

![A hand-authored vertical stack of twelve transformer layers color-coded by type: Mamba-2 layers in green, FFN layers in blue, and one self-attention layer in amber at position seven; the first layer is Mamba-2 and the last is FFN, with a legend on the right listing the layout rules including that only about 8% of layers are self-attention, evenly dispersed](/imgs/blogs/nemotron-h-hybrid-mamba-transformer-1.webp)

It is worth naming the conceptual move before the details, because it is easy to miss how radical it is. For most of the Transformer era, "making attention cheaper" meant *approximating attention* — sparse patterns, sliding windows, low-rank factorizations, grouped-query sharing — all of which keep attention as the primitive and try to compute it more cheaply. Nemotron-H does something different: it *replaces* attention in most layers with a fundamentally different primitive, the state-space recurrence, which is not an approximation of attention at all but a separate mechanism with separate trade-offs. This is not "attention, but faster"; it is "a different way to mix a sequence, used where it suffices, with real attention kept where it does not." That distinction is why the gains are large (you removed the cache, not shrank it) and why the design has degrees of freedom (the ratio and placement of the two primitives) that approximate-attention methods do not. Hold that framing as we go: the report is not optimizing attention, it is *choosing how much attention to use at all*.

The diagram above is the mental model: a Nemotron-H stack is **mostly green and blue** — Mamba-2 and FFN — with only a few amber attention layers scattered through it. Where a standard Transformer alternates attention and FFN in every layer, Nemotron-H replaces the overwhelming majority of attention sublayers with Mamba-2, keeping just enough attention (about 8%) to preserve the capabilities that attention is uniquely good at. The rest of this article explains why that works: why attention is the bottleneck, what Mamba-2 does instead, why you still need a little attention, how the model is trained in FP8 and compressed with MiniPuzzle, and what the whole thing buys you. The organizing idea:

> Attention is a brilliant but expensive way to mix information across a sequence — its cost grows with the sequence. A state-space recurrence mixes information at constant cost. Nemotron-H uses the cheap mixer almost everywhere and the expensive one only where it is irreplaceable.

## 1. Why attention is the long-context bottleneck

To understand why replacing attention is worth the trouble, you have to see exactly what it costs. Attention's price has two parts, and both scale badly with context length.

![Before-and-after comparison: on the left, self-attention stores keys and values for every past token, so memory grows linearly with context and long context becomes memory-bound; on the right, a Mamba-2 layer carries one fixed-size recurrent state, so memory per generated token is constant and long context stays fast and flat](/imgs/blogs/nemotron-h-hybrid-mamba-transformer-2.webp)

**Cost one: the KV cache (memory).** To generate token $t+1$, attention computes a weighted sum over all previous tokens, which requires the keys and values of tokens $1$ through $t$. Recomputing them every step would be wasteful, so they are cached. The cache size for a dense Transformer is, per token,

$$\text{KV bytes/token} = 2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{bytes}$$

and the total scales with the sequence length. For a 70B model at 64k context, this is tens of gigabytes — memory that could have held a bigger batch, gone instead to remembering the past. Worse, this cache must be *read* on every decode step, so decode becomes bandwidth-bound: the GPU spends its time streaming the cache, not computing.

**Cost two: the attention computation (compute).** The attention matrix is $\text{softmax}(QK^\top)V$, and $QK^\top$ is an $n \times n$ matrix for a sequence of length $n$. That is $O(n^2)$ compute — quadratic in the sequence length. Double the context and you quadruple the attention FLOPs. For long context, this quadratic term dominates.

Both costs are *intrinsic to how attention works*: it explicitly compares every token to every other token, which is what makes it powerful (any token can directly influence any other) and what makes it expensive (the comparison is all-pairs). For the prefill phase the compute cost bites; for the decode phase the memory cost bites; and reasoning models, which generate long chains of thought at long context, pay both.

The hybrid's bet is that you do not need this all-pairs comparison in *every* layer. If most of the sequence mixing can be done by a cheaper mechanism, you can reserve attention for the few layers where its all-pairs power is irreplaceable, and pay its cost only there.

### The KV cache in concrete numbers

It helps to make the cache size tangible. Take a 70B-class dense model with, say, 80 layers, 8 KV heads (grouped-query attention already applied), and a head dimension of 128, storing keys and values in BF16 (2 bytes). The per-token KV footprint is $2 \times 80 \times 8 \times 128 \times 2 = 327{,}680$ bytes — about 320 KB *per token*. At a 64k context that is $320\text{ KB} \times 65{,}536 \approx 21$ GB, for a *single sequence*. An H100 has 80 GB; that one sequence's cache has eaten a quarter of it before you have fit any of the model's 140 GB of BF16 weights (which already need two GPUs). The cache is not a rounding error — at long context it is comparable to or larger than the activations, and it is the reason you cannot batch many long-context requests together. Grouped-query attention (the reason this example uses 8 KV heads instead of 64) already cut the cache 8× and is now standard precisely because the cache is such a problem; Nemotron-H's move is the more radical one of removing most of the layers that have a cache at all. When 92% of your layers carry no KV cache, the 21 GB drops to under 2 GB, and suddenly you can batch.

### Second-order optimization: count memory, not just FLOPs

The lesson worth extracting before we even get to Mamba: when you reason about inference cost, **the KV cache is often the real constraint, not the FLOPs**. Teams habitually optimize compute and forget that long-context decode is memory-bandwidth-bound — the model is idle, waiting on the cache. Any technique that shrinks or eliminates the KV cache (grouped-query attention, sliding-window attention, and most radically state-space layers) attacks the actual bottleneck. Nemotron-H's whole thesis is that the biggest long-context win comes from removing the cache, not from making the math faster, because the math was never the problem at decode time.

## 2. Mamba-2: mixing a sequence with a constant-size state

If attention mixes a sequence by comparing all pairs, Mamba-2 mixes it with a **recurrence** — it reads the sequence one token at a time, maintaining a fixed-size hidden state that summarizes everything seen so far.

![Pipeline diagram of a Mamba-2 recurrence: token x1 updates state h1 as A times h0 plus B times x1, then x2 updates h1 to h2, then x3 updates h2 to h3, and an output y3 is read from h3 as C times h3, showing that each token updates one fixed-size state rather than attending over all previous tokens](/imgs/blogs/nemotron-h-hybrid-mamba-transformer-3.webp)

A state-space model (SSM), of which Mamba-2 is a modern, hardware-efficient instance, processes a sequence with a linear recurrence:

$$h_t = A\,h_{t-1} + B\,x_t, \qquad y_t = C\,h_t$$

The hidden state $h_t$ is a **fixed-size vector** — it does not grow with the sequence. Each token $x_t$ updates the state via the matrices $A$ (how the state evolves) and $B$ (how the input enters), and the output $y_t$ is read from the state via $C$. The key property: to process token $t$, you only need the previous state $h_{t-1}$ and the current token $x_t$ — *not* all the previous tokens. The state is a running summary, like a person reading a book and maintaining a mental model rather than re-reading every prior page at each sentence.

This is what gives Mamba-2 its inference advantage:

- **Constant memory.** The state is one fixed-size vector, regardless of context length. There is no cache that grows — the "memory of the past" is compressed into the state. At 64k context, a Mamba-2 layer's memory footprint is the same as at 1k.
- **Constant compute per token.** Each step is a fixed-size state update, $O(1)$ in the sequence length, versus attention's $O(n)$ per token (it attends to all $n$ previous). Total generation is $O(n)$ for Mamba-2 versus $O(n^2)$ for attention.

```python
def mamba2_scan(xs, A, B, C):
    """Process a sequence with a constant-size recurrent state.
    Memory is O(state_dim), independent of sequence length."""
    h = zeros(state_dim)                 # one fixed-size state, not a growing cache
    ys = []
    for x in xs:                         # one token at a time
        h = A @ h + B @ x                # update the running summary
        ys.append(C @ h)                 # read output from the state
    return ys                            # O(state_dim) memory throughout
```

Mamba-2 specifically (the second-generation Mamba) is engineered so this recurrence maps efficiently onto GPU hardware — it can be computed as a parallel scan during training (so you do not lose training throughput to the sequential recurrence) and as a fast recurrence during inference (so you get the constant-memory benefit at decode). The "2" in Mamba-2 reflects a reformulation that makes the state update expressible as structured matrix operations the GPU runs efficiently, closing much of the hardware-efficiency gap with attention.

### What the state cannot do

There is a catch, and it is why Nemotron-H keeps *some* attention. A fixed-size state is a **lossy summary**. Attention can reach back and read any specific past token exactly — perfect recall, at the cost of storing everything. A recurrent state has to *compress* the past into a fixed vector, which means it can lose precise details, especially for tasks that require copying an exact token from far back (think: retrieving a specific name or number mentioned 30k tokens ago). This is the classic recurrence-vs-attention trade: attention has perfect but expensive recall; recurrence has cheap but lossy memory. Nemotron-H's hybrid design is the resolution — use cheap recurrence for the bulk of sequence mixing and keep a few attention layers for the precise-recall jobs that the state cannot do.

### Selectivity: why Mamba-2 is better than a classic RNN

A fair objection: recurrences are old — RNNs and LSTMs are recurrences, and they lost to Transformers. What makes Mamba-2 different? The answer is **selectivity**. Classic RNNs have *fixed* transition dynamics — the matrices that update the state are the same regardless of the input, so the model cannot decide, based on what it is reading, what to remember and what to forget. This is why LSTMs needed elaborate gating and still struggled with long dependencies: the state update could not be *content-dependent* in a flexible way. Mamba's innovation (carried into Mamba-2) is to make the SSM parameters — the $B$ and $C$ matrices and the state-decay $A$ — *functions of the input*, so the model can selectively gate what enters the state and what is read from it, conditioned on the current token. A token that says "remember this name" can write strongly into the state; a filler token can be largely ignored. This input-dependent selectivity is what lets a fixed-size state hold the *relevant* information rather than a blurry average of everything, and it is the reason Mamba-class models are competitive with Transformers where prior RNNs were not. The fixed-size state is the efficiency; selectivity is what makes the fixed-size state *good enough*.

### The training-vs-inference duality

One more property makes Mamba-2 practical at scale: it has two computational modes that match the two phases of a language model's life. During **training**, where you have the whole sequence at once, the recurrence can be unrolled and computed as a **parallel scan** — a logarithmic-depth parallel reduction that uses the GPU's parallelism, so you do not pay the sequential cost of a naive RNN loop. During **inference decode**, where you generate one token at a time, it runs as the **sequential recurrence**, which is exactly what gives the constant-memory benefit. Mamba-2's reformulation (expressing the SSM as structured matrix operations) is what makes both modes hardware-efficient. This duality is essential: a recurrence that was fast at inference but slow to train (the old RNN problem) would be a non-starter at 20T tokens. Getting both — parallel training and recurrent inference — is the engineering that turned state-space models from a theoretical curiosity into a viable Transformer alternative.

### Second-order optimization: compression is the whole game

What SSMs exploit is that most of what attention stores is not needed at full fidelity. Attention keeps a perfect record of every token; a recurrence keeps a compressed summary. For the majority of what a language model does — tracking topic, syntax, local dependencies, the gist of the context — a compressed summary is *enough*, and it is enormously cheaper. The art is knowing where compression is safe (most layers) and where you need the lossless record (a few layers, for exact recall). Nemotron-H is a careful answer to the question "where can I compress the sequence memory without losing capability," and the answer turns out to be "almost everywhere."

## 3. The hybrid recipe: keep 8% attention, place it carefully

The architecture is not "all Mamba-2." It is a deliberate blend, and the blend ratio and placement are the heart of the design. Refer back to the layer-stack figure from the intro: about 8% of layers are attention, the rest split evenly between Mamba-2 and FFN.

The concrete composition:

- **Nemotron-H-8B**: 52 layers total, of which **4 are self-attention** (~8%), and the remaining 48 split into roughly 24 Mamba-2 and 24 FFN layers.
- **Nemotron-H-56B**: 118 layers, of which **10 are self-attention** (~8%), with the rest split evenly — about 54 Mamba-2 and 54 FFN.

The placement rules matter as much as the ratio:

- **About 8% attention, evenly dispersed.** Following prior hybrid-model findings, the attention layers are spread throughout the depth rather than clustered, so the perfect-recall capability is available at multiple points in the computation.
- **The first layer is Mamba-2.** The initial sequence mixing is handled by the cheap recurrence.
- **The last layer is FFN.** The final transformation before the output head is a position-wise FFN.
- **Attention precedes FFN.** Where an attention layer appears, it is followed by an FFN, preserving the familiar Transformer "mix then process" block structure locally.

```python
def build_hybrid_layers(n_layers, attention_fraction=0.08):
    """Nemotron-H layer schedule: ~8% attention, evenly dispersed,
    first=Mamba-2, last=FFN, attention always precedes an FFN."""
    n_attention = round(n_layers * attention_fraction)   # 4 of 52, 10 of 118
    attention_positions = evenly_spaced(n_layers, n_attention)
    layers = []
    for i in range(n_layers):
        if i in attention_positions:
            layers.append("self_attention")
        elif i == n_layers - 1:
            layers.append("ffn")                         # last layer is FFN
        else:
            layers.append("mamba2" if i % 2 == 0 else "ffn")  # even split
    layers[0] = "mamba2"                                 # first layer is Mamba-2
    return layers
```

Why 8% and not 0%? Because of §2's catch: a pure-Mamba model loses precise long-range recall, and a handful of attention layers restore it cheaply. Why not 20%? Because each attention layer reintroduces the KV cache and the quadratic cost, so you want the *minimum* attention that recovers the capability. 8% is the empirical sweet spot from prior hybrid work — enough attention to keep the exact-recall tasks working, few enough that the KV cache stays small and long-context throughput stays high. The 8% of layers that remain attention contribute only 8% of the KV cache a dense model would, which is why the hybrid's long-context memory footprint is a fraction of a dense Transformer's.

### Why dispersed, not clustered

The placement detail "evenly dispersed" is load-bearing and worth dwelling on. You could imagine putting all the attention layers together — say, a block of attention in the middle — but the report disperses them throughout the depth, and for good reason. The model's representation evolves layer by layer, and the exact-recall capability that attention provides is needed *at multiple stages* of that evolution, not just once. An early attention layer can establish long-range dependencies in the raw token features; a middle attention layer can re-bind them after the FFNs have transformed the representation; a late attention layer can do a final precise lookup before the output. Clustering the attention into one region would leave long stretches of the network with no access to exact recall, and the lossy Mamba-2 state would have to carry the precise information across those stretches — exactly the thing it is bad at. Dispersing the attention means precise recall is *refreshable* at regular intervals, so the state never has to hold exact details for too long. The lesson is that in a heterogeneous stack, the *spacing* of the rare, powerful layers matters: you want the expensive capability available wherever the computation might need it, which means spreading it out, not bunching it up.

### The hybrid research lineage

Nemotron-H did not invent hybrids from nothing — it stands on a line of research (Jamba, Zamba, and earlier Mamba-Transformer hybrids) that established the basic finding that a small fraction of attention layers, combined with state-space layers, recovers most of the quality of full attention at a fraction of the cost. What Nemotron-H contributes is doing it *at frontier scale with rigorous head-to-head comparisons* — 8B and 56B models trained on 15-20T tokens, benchmarked against the strongest dense Transformers of the same size, plus the FP8 training recipe and the MiniPuzzle compression. The earlier hybrids proved the concept; Nemotron-H proved it scales and productionized it. The lesson for reading research is that the valuable papers are often not the ones that introduce an idea but the ones that *de-risk it at scale* — showing that a promising small-scale result holds up against the best alternatives at production size, with all the engineering (training recipe, compression, serving) worked out. That de-risking is what moves a technique from "interesting" to "adoptable."

### Second-order optimization: spend the expensive primitive only where it is irreplaceable

The reusable design principle is to **identify what your expensive component does that nothing else can, and use it only there**. Attention's irreplaceable property is exact, arbitrary-distance recall. Most layers do not need that; they need cheap sequence mixing, which Mamba-2 provides. So you keep attention for the recall and replace it everywhere else. This is a general pattern for efficient architecture: do not use your most powerful (and costly) primitive uniformly out of habit — find the few places its unique power is actually required, and substitute a cheaper primitive for the rest. The 8% number is the quantified answer to "how little of the expensive thing can I get away with."

## 4. Training the hybrid in FP8

A 56B model trained on 20 trillion tokens is an enormous compute bill, and Nemotron-H cuts it with an **FP8 training recipe** that halves the precision of most of the math without losing accuracy — a result that was far from guaranteed and is a contribution in its own right.

![A matrix showing the FP8 training recipe by tensor type: weights compute in FP8 with the E4M3 format, activations in FP8 with E4M3, gradients in FP8 with E5M2, and the first and last four layers are kept in BF16 for stability](/imgs/blogs/nemotron-h-hybrid-mamba-transformer-7.webp)

FP8 (8-bit floating point) comes in two flavors that trade range against precision, and the recipe uses each where it fits:

- **E4M3** (4 exponent bits, 3 mantissa bits) — more precision, less range. Used for **weights and activations**, which are well-behaved in magnitude and benefit from the extra precision.
- **E5M2** (5 exponent bits, 2 mantissa bits) — more range, less precision. Used for **gradients**, which can span a wide dynamic range and need the extra exponent bits to avoid underflow.

The quantization is **per-tensor dynamic**: for each tensor, compute a scale factor that maps its values into the FP8 range, applied so the tensor's maximum absolute value lands at the format's maximum representable value:

$$\text{scale} = \frac{\text{max representable FP8}}{\max |\text{tensor}|}$$

```python
def fp8_quantize(tensor, fmt="e4m3"):
    """Per-tensor dynamic FP8 quantization: scale so max|x| hits the format max."""
    max_repr = 448.0 if fmt == "e4m3" else 57344.0      # E4M3 vs E5M2 max value
    scale = max_repr / tensor.abs().max().clamp(min=1e-12)
    q = (tensor * scale).to_fp8(fmt)                     # cast into 8-bit float
    return q, scale                                      # dequantize later as q / scale
```

The crucial stability detail: **the first four and last four layers are kept in BF16**, not FP8. The outer layers — closest to the input embeddings and the output head — are the most sensitive to quantization error, because errors there propagate through the whole network (early layers) or directly distort the output distribution (late layers). Keeping them in higher precision costs little (8 of ~52–118 layers) and buys a lot of stability. Everything in between runs in FP8.

The headline finding is the one that makes the recipe worth copying: on the 8B model trained over 15 trillion tokens, **FP8 accuracy was consistently equal to or better than BF16**. FP8 is not a lossy compromise here — done with the right format-per-tensor choices and the BF16 outer layers, it matches full-precision training while roughly halving the memory and bandwidth of the linear layers. This connects directly to the broader [quantization story](/blog/machine-learning/large-language-model/quantization-in-llm) and the [push past the 4-bit wall](/blog/machine-learning/large-language-model/past-4-bit-wall-frontier-llm-quantization): low precision, applied carefully, is closer to free than intuition suggests.

![A timeline of the Nemotron-H build pipeline: FP8 pretraining of the 56B hybrid on 20 trillion tokens, then MiniPuzzle pruning and distillation from 56B to 47B, then a 47B base that is 20% faster at similar accuracy, then post-training with SFT and alignment, ending with the shipped 8B, 47B, and 56B family](/imgs/blogs/nemotron-h-hybrid-mamba-transformer-6.webp)

The training pipeline above puts FP8 in context: the 56B is pretrained in FP8 on 20T tokens, then compressed by MiniPuzzle (§5) to a 47B, and both are post-trained into instruct models. FP8 is the lever that made the 20T-token pretraining affordable; MiniPuzzle is the lever that made the cheaper 47B serving variant.

### Why E4M3 for weights and E5M2 for gradients

The format split is not arbitrary; it follows the statistics of what each tensor holds. **Weights and activations** in a trained network are relatively well-conditioned — their magnitudes cluster in a manageable range — so the binding constraint is *precision* (resolving small differences between nearby values), and E4M3's extra mantissa bit (3 vs 2) gives that precision. **Gradients**, on the other hand, can span an enormous dynamic range: some are tiny (near-converged parameters), some are large (parameters far from their optimum), and the spread can be many orders of magnitude within a single backward pass. For gradients the binding constraint is *range* — you must represent both the tiny and the large without the tiny ones underflowing to zero, which would silently stop those parameters from learning. E5M2's extra exponent bit (5 vs 4) buys that range at the cost of one mantissa bit. The recipe is matching the format to the failure mode: weights fail by losing precision, gradients fail by underflowing, so you give precision where precision is scarce and range where range is scarce. This is the same per-tensor-sensitivity thinking that the [quantization literature](/blog/machine-learning/large-language-model/quantization-in-llm) applies to inference, here applied to training.

### The cost of the BF16 outer layers is near zero

A quick economic note on the outer-layers-in-BF16 choice: keeping 8 layers (the first 4 and last 4) in BF16 out of 52-118 total means roughly 7-15% of the layers run at higher precision. But the *memory and bandwidth* cost is even smaller than that fraction suggests, because FP8's savings are on the linear-layer matmuls, and the outer layers are a minority of those. So you pay a small single-digit-percent overhead for a large stability benefit — the outer layers are where quantization error does the most damage (propagating through the whole network from the front, or directly distorting logits at the back), so protecting them is the highest-return use of your higher-precision budget. The general principle, reusable in any mixed-precision scheme: spend your precision budget at the points of maximum sensitivity, which are almost always the boundaries (input and output), and quantize the robust interior aggressively. A uniform precision policy wastes precision on the robust middle and starves the sensitive edges; a sensitivity-aware policy does the opposite.

### Second-order optimization: precision is a per-tensor decision

The FP8 recipe teaches a general lesson about low-precision training: **precision is not one global knob, it is a per-tensor, per-layer decision**. The recipe uses E4M3 for some tensors and E5M2 for others, and BF16 for the outer layers, because different tensors have different sensitivity and different dynamic range. A naive "cast everything to FP8" fails; a careful "FP8 where it is safe, higher precision where it is not" matches full precision. The art is knowing the sensitivity map — gradients need range, weights need precision, outer layers need stability — and the report's recipe is a reusable template for that map.

## 5. MiniPuzzle: compressing the hybrid

Once you have a strong 56B hybrid, you can derive a cheaper 47B from it — and the method, **MiniPuzzle**, is exactly the Minitron-meets-Puzzle move you would expect from reading the earlier posts in this series.

![A graph showing MiniPuzzle compression: the Nemotron-H 56B feeds importance scoring (layer MSE plus FFN neuron importance), which feeds a conditional NAS searching about 400 candidate hybrids, which fans out to three top candidates A, B, and C each ranked with a 7-billion-token lightweight distill, all feeding a final distillation of the best candidate over 63 billion tokens in FP8, producing the Nemotron-H 47B that is 20% faster](/imgs/blogs/nemotron-h-hybrid-mamba-transformer-4.webp)

MiniPuzzle combines [Minitron's pruning-and-distillation](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) with [Llama-Nemotron's NAS](/blog/machine-learning/large-language-model/llama-nemotron-efficient-reasoning-models), adapted to the hybrid architecture, in three stages:

1. **Importance estimation.** Rank what to keep. Layers are scored by their **MSE contribution** (how much removing the layer perturbs the representation — the same block-importance idea from Minitron), and FFN neurons are scored by **activation-based importance** (aggregate activation magnitude, exactly Minitron's width-pruning signal). This tells you which layers and which neurons are load-bearing.
2. **Conditional NAS.** Search the space of compressed architectures — about **400 candidate hybrids** that respect the hybrid layout rules (still ~8% attention, valid Mamba-2/FFN placement) — and rank them cheaply by next-token accuracy, keeping the **top 3**. This is the architecture-search step, conditioned on the hybrid constraints, so the search only considers valid hybrids.
3. **Distillation.** Run a **lightweight distillation (~7B tokens)** on the top-3 candidates to break the tie, then an **extended distillation (~63B tokens)** on the single best candidate, using FP8 and a logit-based KL loss — Minitron's distillation-only retraining, applied to heal the compressed hybrid.

The result, **Nemotron-H-47B**, is 20% faster to infer than the 56B at similar accuracy, and it required roughly **300× fewer tokens than training a 47B from scratch** — the same economics that made Minitron compelling. The specific surgery: it drops 5 of the 10 attention layers (keeping 5), removes 10 Mamba-2 and 5 FFN layers, and prunes the FFN hidden dimension from 32,768 to 30,720.

```python
def layer_importance_mse(model, calib, layer_idx):
    """Minitron-style block importance, adapted to hybrid layers:
    how much does removing this layer perturb the representation?"""
    with no_grad():
        full = model(calib, output_hidden_states=True).hidden_states
        h_in, h_out = full[layer_idx], full[layer_idx + 1]
        return mse(h_out, h_in)        # high MSE = load-bearing, keep it
```

### Why the two-stage distillation (7B then 63B)

The split distillation budget — a lightweight 7B-token pass on the top-3 candidates, then a heavy 63B-token pass on the single winner — is the same NAS-lite economics from [Minitron](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation), and it is worth understanding why two stages beat one. If you distilled every candidate to convergence, the search would be prohibitively expensive (hundreds of candidates × full distillation). If you picked the winner after no distillation at all, you would be ranking architectures by their *un-healed* quality, which is noisy and misleading — a candidate that looks bad before healing might be the best after it. The two-stage approach threads the needle: the lightweight 7B-token pass heals each top-3 candidate *just enough* that their relative ranking stabilizes (architectures' rankings settle long before their absolute quality does), so you can reliably pick the winner, and only then do you pay for the full 63B-token heal on that one. You spend the cheap budget to rank and the expensive budget to finish. The reusable lesson is that ranking and finishing are different jobs with different budgets: a little training reveals the ordering, and you reserve the heavy training for the choice you commit to.

### Importance signals transfer; search spaces do not

A subtle point about adapting Minitron to the hybrid: the *importance signals* carried over directly (layer MSE for which layers to keep, activation aggregation for which FFN neurons to keep), but the *search space* had to be redesigned. Minitron's pruning assumes a homogeneous Transformer where any layer can be pruned like any other; MiniPuzzle's space is constrained to valid hybrids, where attention layers, Mamba-2 layers, and FFN layers are different and the placement rules must hold. This is the general shape of porting a technique across architectures: the *measurement* methods (how you score importance) tend to be architecture-agnostic and transfer cleanly, while the *decision* space (what configurations are valid) is architecture-specific and must be rebuilt. When you adapt a compression method to a new architecture, expect to keep the scoring and rebuild the search.

### Second-order optimization: the compression toolkit composes

MiniPuzzle is the clearest evidence that NVIDIA is building a *composable toolkit*, not one-off tricks. It is literally Minitron's importance estimation and distillation, plus Llama-Nemotron's NAS, with the hybrid constraints folded in. Each technique from the earlier reports slots into a stage. The lesson for practitioners is that these methods are not mutually exclusive alternatives — importance-based pruning, NAS, and distillation are *stages of one pipeline*, and the strongest compression combines them: prune by importance to narrow the search, NAS to find the best architecture, distill to heal it. Reading the series in order, you watch the toolkit assemble itself.

## 6. Accuracy: matching the Transformers it replaces

The efficiency story only matters if the hybrid is actually as good as the dense Transformers it competes with. It is.

![A matrix comparing Nemotron-H-56B, Qwen-2.5-72B, and Llama-3.1-70B across MMLU, GSM8K, and long-context throughput: Nemotron-H scores 84.2 MMLU and 93.7 GSM8K with 2.4x faster throughput, Qwen scores 86.1 and 90.9 at baseline throughput, and Llama scores 78.8 and 83.9 at baseline](/imgs/blogs/nemotron-h-hybrid-mamba-transformer-5.webp)

The accuracy numbers for Nemotron-H-56B against the strongest comparable dense Transformers:

| Benchmark | Nemotron-H-56B | Qwen-2.5-72B | Llama-3.1-70B |
|---|---|---|---|
| MMLU | 84.2 | 86.1 | 78.8 |
| GSM8K | **93.7** | 90.9 | 83.9 |
| MATH | 59.4 | **64.6** | 42.9 |
| HumanEval | **60.4** | 56.7 | 57.3 |
| Long-ctx throughput | **2.4× faster** | baseline | baseline |

Nemotron-H-56B **outperforms Llama-3.1-70B on 16 of 17 benchmarks** (losing only Winogrande) and trades blows with Qwen-2.5-72B — ahead on GSM8K and HumanEval, behind on MMLU and MATH — while being a *smaller model* (56B vs 70-72B) that serves long context 2.4× faster. The 8B tells the same story against its weight class: MMLU-Pro 44.0 (beating Llama-3.1-8B's 35.9), HumanEval+ 55.5 (beating both Qwen-2.5-7B and Llama-3.1-8B), at 3× the throughput of Llama-3.1-8B.

The headline is that **the hybrid is not a downgrade**. The fear with replacing attention is that you lose capability; the measured result is that, with 8% attention retained for recall and a strong training recipe, the hybrid matches or beats dense Transformers of equal or larger size. You get the accuracy *and* the throughput.

### Where the hybrid wins and where it trades

The benchmark spread tells a nuanced story worth reading carefully. Nemotron-H-56B *wins* on GSM8K (93.7) and HumanEval (60.4) — reasoning and code, tasks that reward strong sequence modeling and the math/code-heavy training data — and *trades* on MMLU (84.2 vs Qwen's 86.1) and MATH (59.4 vs Qwen's 64.6), where the gap is small. Against Llama-3.1-70B it wins almost everywhere; against the very strong Qwen-2.5-72B it splits. The honest read is not "the hybrid is strictly better" but "the hybrid is in the same accuracy tier as the best dense models, ahead on some axes, slightly behind on others, while being smaller and 2.4× faster." That is exactly the result you want from an efficiency-motivated architecture: not a free accuracy win (which would be suspicious), but accuracy *parity* with a large speed advantage. The places it trades (MMLU, MATH) are within normal model-to-model variation and are not attributable to the hybrid architecture specifically — Qwen-2.5 is simply a very strong model. The lesson in reading efficiency papers is to look for *parity with a speedup*, not dominance: a technique that claims to be both faster and uniformly more accurate than the best alternatives deserves more scrutiny than one that claims parity-plus-speed, which is the more believable and more useful result.

### Second-order optimization: efficiency that does not cost accuracy is the only efficiency that matters

The reason this report landed is the conjunction: 2.4× faster *and* competitive accuracy. An efficiency technique that costs you 5 points of MMLU is a hard sell — you can usually just use a smaller model for that. An efficiency technique that costs you *nothing* on accuracy is a free lunch, and free lunches get adopted. The discipline of benchmarking the hybrid head-to-head against the best dense models, and matching them, is what turns "an interesting alternative architecture" into "a model you should actually use." The lesson is that for an efficiency claim to matter, you must hold accuracy fixed and show the speedup — anything else is just picking a different point on the size/quality curve.

## 7. The payoff: long-context throughput

Put it together and the payoff is throughput at long context, which is exactly where dense Transformers struggle most.

![A before-and-after comparison: on the left, a dense Transformer like Llama or Qwen has a KV cache that grows to gigabytes at 64k context, is memory-bound during decode with small batches, and delivers baseline throughput; on the right, the Nemotron-H hybrid has a flat SSM state with only a tiny KV cache from its 8% attention, allowing bigger batches and more tokens per second, 2.4x faster on the 56B and 3x on the 8B](/imgs/blogs/nemotron-h-hybrid-mamba-transformer-8.webp)

On a representative long-context workload — 65,536 input tokens, 1,024 generated, on H100 — the throughput gains are:

| Comparison | Speedup |
|---|---|
| Nemotron-H-56B vs Qwen-2.5-72B / Llama-3.1-70B | **2.4×** |
| Nemotron-H-47B (MiniPuzzle) vs same | **2.9×** |
| Nemotron-H-8B vs Llama-3.1-8B | **3×** |
| Nemotron-H-8B vs Qwen-2.5-7B | **1.8×** |

The mechanism is everything we have built up: because only 8% of layers are attention, the KV cache is a fraction of a dense model's, so the GPU is not throttled streaming a giant cache on every decode step. The flat memory lets you fit larger batches, and larger batches mean more tokens per second. The MiniPuzzle 47B is even faster than the 56B (2.9× vs 2.4×) because it has fewer layers on top of the hybrid savings. At long context, where the KV cache dominates, the hybrid's advantage is largest — which is precisely where reasoning and document-processing workloads live.

### Where the speedup comes from, decomposed

It is worth decomposing the 2.4-3× to see which part of the design earns it, because the parts are separable. The throughput win has two sources. First, **prefill compute**: at 64k input tokens, the dense Transformer pays $O(n^2)$ attention FLOPs on every layer, while the hybrid pays that only on its 8% attention layers and $O(n)$ on its Mamba-2 layers — a large reduction in the compute to ingest a long prompt. Second, and usually larger at decode, **memory bandwidth**: the dense model streams its full KV cache on every generated token, and that cache is huge at 64k, so decode is bandwidth-bound; the hybrid streams only the tiny cache of its 8% attention layers plus the fixed-size SSM states, a fraction of the traffic, so it is far less bandwidth-bound and can run bigger batches. The 8B's 3× over Llama-3.1-8B and the 56B's 2.4× over the 70B-class models reflect different balances of these (the smaller model is more bandwidth-dominated), but in both the root cause is the same: 92% fewer layers carrying a growing cache. The decomposition matters because it tells you *when* the hybrid wins most — long context (where the cache dominates) and decode-heavy workloads (where bandwidth dominates), which is exactly the reasoning and document-processing regime.

### The batch-size multiplier

A second-order throughput effect that the headline numbers fold in but is worth naming: flat memory does not just make each token cheaper, it lets you *batch more sequences*, and batching is itself a throughput multiplier. On a dense Transformer at long context, the KV cache eats so much memory that you can only fit a few sequences in a batch, leaving the GPU's compute underutilized — you are memory-capacity-bound, not compute-bound. The hybrid's small cache frees that memory for more sequences, so you fill the batch, which raises GPU utilization, which raises throughput beyond the per-token savings. This compounding — cheaper per token *and* more tokens in flight — is why the end-to-end speedups (2.4-3×) can exceed what a naive per-token FLOP count would predict. The lesson is that memory savings at long context buy throughput twice: once directly (less to stream per token) and once indirectly (bigger batches, higher utilization).

### Second-order optimization: architecture is the highest-leverage inference optimization

The throughput numbers make a point that runs through this whole series: **the biggest inference wins come from the architecture, not the runtime**. You can tune kernels, quantize, and batch cleverly, and those help — but they operate on a fixed architecture whose KV cache still grows with context. Changing the architecture so the cache *does not grow* is a categorically larger win, because it removes the bottleneck rather than mitigating it. Nemotron-H's 2.4-3× is not a kernel optimization; it is a consequence of having 92% fewer attention layers. The lesson for anyone serious about inference cost is to look upstream, at the architecture, where the leverage is largest — the same lesson Llama-Nemotron taught with Puzzle, here taken further by changing the sequence-mixing primitive itself.

## 8. Failure modes the recipe guards against

As with the rest of the series, the design is a set of safeguards against specific failure modes.

- **Losing exact recall by removing all attention.** Symptom: a pure-Mamba model fails at retrieving specific distant tokens. Cause: a fixed-size state compresses the past lossily. Safeguard: keep ~8% attention, dispersed, for the recall jobs the state cannot do (§3).
- **KV cache blowing up at long context.** Symptom: long-context decode is memory-bound, batches shrink. Cause: attention's cache grows with context. Safeguard: replace 92% of attention with constant-memory Mamba-2 (§1, §2).
- **FP8 training diverging.** Symptom: low-precision training loses accuracy or is unstable. Cause: naive global FP8 ignores per-tensor sensitivity. Safeguard: E4M3 for weights/activations, E5M2 for gradients, BF16 outer layers (§4).
- **Compressing a hybrid with Transformer-only tools.** Symptom: standard pruning does not know about Mamba-2/FFN/attention placement. Cause: pruning assumes a homogeneous Transformer. Safeguard: MiniPuzzle's conditional NAS respects the hybrid layout rules (§5).
- **Sacrificing accuracy for speed.** Symptom: the efficient model is measurably worse. Cause: removing capability without compensating. Safeguard: benchmark head-to-head and hold accuracy fixed — the hybrid matches dense models (§6).
- **Catastrophic damage from compression.** Symptom: the 47B underperforms after pruning. Cause: aggressive cuts with no healing. Safeguard: distillation retraining (7B + 63B tokens) heals the compressed model (§5).

The meta-lesson, again: each choice is a "don't" — don't remove all attention, don't let the cache grow, don't quantize naively, don't compress blindly, don't trade accuracy for speed — and the recipe is the disciplined accumulation of those don'ts.

## 9. Case studies from the report

### 1. The 8% attention number

The single most transferable finding is the ratio: **about 8% of layers as attention** is the sweet spot for a hybrid. Below it, exact-recall tasks degrade; above it, the KV cache and quadratic cost creep back. The number comes from prior hybrid-model research and Nemotron-H validates it at scale (4 of 52 in the 8B, 10 of 118 in the 56B). The case study is a reusable design constant: if you are building a hybrid, start at ~8% attention, evenly dispersed, and tune from there. It is the empirical answer to "how little attention can I get away with," and it is remarkably consistent across model sizes.

### 2. FP8 matching BF16 over 15 trillion tokens

The FP8 result is a case study in low-precision training done right. Training an 8B for 15T tokens in FP8 and finding accuracy *equal to or better than* BF16 is a strong claim, and it rests on the per-tensor format choices and the BF16 outer layers. The "better than" is notable — FP8's quantization noise can act as a mild regularizer in some regimes. The lesson is that FP8 training, long treated as risky, is production-ready with the right recipe, and the recipe is specific: E4M3 weights/activations, E5M2 gradients, BF16 first-and-last-four. For any team training large models, this is a memory-and-bandwidth halving that is closer to free than feared.

### 3. MiniPuzzle and 300× token savings

Compressing 56B to 47B with MiniPuzzle at ~300× fewer tokens than training from scratch is the same Minitron economics applied to a hybrid. The case study reinforces the series-wide thesis: derive your smaller models from a strong large one rather than training each from scratch. What is new is that MiniPuzzle does it for a *hybrid* — the conditional NAS has to respect Mamba-2/attention/FFN placement, which a Transformer-only pruner cannot. The lesson is that compression methods must be architecture-aware: the importance signals (layer MSE, FFN activation) transfer, but the search space (valid hybrid layouts) is architecture-specific, and getting that right is what makes the compressed hybrid valid.

### 4. Constant memory as a reasoning enabler

A consequence worth its own study: because the hybrid's memory is flat in context length, it is uniquely suited to *reasoning*, where the chain of thought is a long generated sequence. A dense Transformer's KV cache grows with every reasoning token, so long chains of thought get progressively more expensive; the hybrid's cost stays flat. This is why the Nemotron-Nano-2 reasoning model is built on the Nemotron-H hybrid — the architecture and the use case are matched. The lesson is that efficiency and capability can be co-designed: the hybrid is not just generically faster, it is faster *exactly where reasoning models spend their tokens*, which makes it the natural substrate for efficient reasoning.

### 5. Beating Llama-3.1-70B on 16 of 17 tasks while smaller

Nemotron-H-56B beating a 70B dense Transformer on all but one benchmark, while being smaller and 2.4× faster, is the accuracy case study that sells the architecture. It refutes the assumption that attention is irreplaceable for quality. The mechanism is that 8% attention plus strong Mamba-2 layers plus a 20T-token training run produces a model whose quality is not bottlenecked by having less attention — the sequence mixing the Mamba-2 layers do is good enough that the model loses nothing it needed. The lesson is that "attention everywhere" is a convention, not a quality requirement, and a well-trained hybrid can match dense Transformers head-to-head.

### 6. The outer-layers-in-BF16 stability trick

A small but reusable detail: keeping the first and last four layers in BF16 while everything else is FP8. The outer layers are the most quantization-sensitive — early layers because their errors propagate through the whole network, late layers because they directly shape the output distribution. Spending a little precision there ($\sim$8 of 52-118 layers) buys disproportionate stability. The case study generalizes to any mixed-precision scheme: protect the sensitive boundaries (input embedding, output head, and the layers adjacent to them) at higher precision, and aggressively quantize the robust interior. It is the same logic as the [Minitron depth-pruning finding](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) that the first and last layers are the most important — the boundaries are where you cannot afford to lose precision.

### 7. The hybrid as the new default for NVIDIA

Reading forward, Nemotron-H became the *substrate* for NVIDIA's subsequent models — Nemotron-Nano-2 is a hybrid Mamba-Transformer reasoning model built on this foundation. The case study is that Nemotron-H is not a one-off experiment but a new default architecture in NVIDIA's lineup, the way Minitron compression and Puzzle NAS became defaults. The lesson is to watch which research artifacts become infrastructure: when a lab keeps building on a technique, that is the signal it has graduated from "interesting result" to "production foundation." The hybrid is now a foundation.

### 8. Why the first layer is Mamba-2 and the last is FFN

The placement rules encode hard-won knowledge. The first layer being Mamba-2 means the initial sequence mixing — establishing the basic dependencies — is done by the cheap recurrence, and the model does not pay for attention at the very first opportunity. The last layer being FFN means the final transformation before the output head is a clean position-wise projection, not a sequence-mixing operation, which is the right shape for producing per-token logits. The constraint that attention precedes FFN preserves the local "mix then process" block structure that Transformers rely on. The case study is a reminder that in a heterogeneous architecture, *placement* carries information — the same set of layer types in a different order is a different (and usually worse) model.

### 9. The even Mamba-2 / FFN split

A detail worth its own note: the ~92% of non-attention layers split *evenly* between Mamba-2 and FFN, not all-Mamba. Why keep so many FFN layers? Because Mamba-2 and FFN do different jobs: Mamba-2 mixes information *across* the sequence (the recurrence), while the FFN processes each position *independently* (the per-token transformation). A model needs both — sequence mixing and position-wise processing — and stacking only Mamba-2 layers would over-invest in mixing and under-invest in the per-position computation that FFNs provide. The even split mirrors the standard Transformer's own balance (one attention sublayer and one FFN sublayer per block), just with Mamba-2 substituted for most attention. The lesson is that replacing attention does not mean replacing the FFN — the FFN is doing orthogonal, still-necessary work, and the hybrid keeps it in full.

### 10. Conditional NAS respecting the hybrid grammar

MiniPuzzle's conditional NAS is a case study in constrained search. A naive architecture search over "which layers to keep" would propose invalid hybrids — an all-FFN stack with no sequence mixing, or a layout that violates the first-Mamba/last-FFN rules. The "conditional" in conditional NAS means the search space is *constrained to valid hybrids*: every candidate respects the ~8% attention budget, the placement rules, and the Mamba-2/FFN balance. This shrinks the search from an intractable free-for-all to ~400 sensible candidates, which is small enough to rank cheaply. The lesson generalizes to any structured search: encode the validity constraints into the search space itself rather than searching freely and filtering afterward — a constrained search over valid configurations is far more efficient than an unconstrained search that wastes most of its budget on nonsense.

### 11. The 2.9× of the compressed 47B

The MiniPuzzle 47B is faster than the 56B it came from (2.9× vs 2.4× over the dense baselines) — a case study in stacking efficiency wins. The 56B is already fast because it is a hybrid; the 47B is faster still because MiniPuzzle removed layers (5 attention, 10 Mamba-2, 5 FFN) and narrowed the FFN. So the 47B compounds two independent savings: the architectural hybrid savings (flat KV cache) and the compression savings (fewer, narrower layers). The lesson is that the compression toolkit and the architecture toolkit are *multiplicative*, not redundant — compress a hybrid and you get the product of both speedups, which is why the 47B lands at a higher multiple than the 56B despite both being hybrids.

### 12. FP8 as a regularizer, not just a compressor

The finding that FP8 was sometimes *better* than BF16 — not merely equal — is a case study in an underappreciated phenomenon: low-precision noise can regularize. The quantization error of FP8 acts like a small, structured noise injected into the forward and backward passes, and in some regimes this noise has a mild regularizing effect, nudging the model away from sharp minima that overfit. This is not guaranteed and not the primary motivation (the memory savings are), but it is a pleasant property that undercuts the assumption that lower precision can only hurt. The lesson is to measure rather than assume: FP8 is often treated as a necessary evil for memory, but the empirical result here is that, done right, it is not even a sacrifice — and occasionally a slight gain.

### 13. Why this report is the architecture sibling of Minitron

Read alongside [Minitron](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation), Nemotron-H reveals two complementary routes to efficiency. Minitron takes a fixed (Transformer) architecture and makes it *smaller* by pruning and distillation. Nemotron-H changes the *architecture itself* — the sequence-mixing primitive — to remove the KV-cache bottleneck. They attack efficiency at different levels: Minitron at the parameter count, Nemotron-H at the architectural primitive. And MiniPuzzle shows they compose: you can change the primitive (hybrid) *and then* prune (MiniPuzzle). The case study is the layered nature of efficiency work — there are wins at the kernel level, the parameter level, and the architecture level, and the biggest total win comes from stacking them, which is exactly what the Nemotron line does across its reports.

### 14. Constant memory changes the serving economics

A final, operational case study: flat memory does not just make decode faster, it changes *what you can offer*. With a dense Transformer, supporting a 128k context window means provisioning for a KV cache that could be 40+ GB per sequence, which caps your concurrency brutally — you might serve only a handful of long-context requests per GPU. With a hybrid, the long-context memory is a fraction of that, so you can serve many more concurrent long-context sessions on the same hardware. This turns long context from a premium, rationed feature into something you can offer broadly. The lesson is that architecture choices ripple into product economics: the hybrid's flat memory is not just a benchmark number, it is the difference between long context being a scarce resource and an abundant one, which changes what you can build on top.

### 15. The 8B that beats its weight class

The 8B model is the most broadly useful member of the family and a case study in itself. It scores MMLU-Pro 44.0 against Llama-3.1-8B's 35.9 and HumanEval+ 55.5 against both Llama-3.1-8B (31.7) and Qwen-2.5-7B (48.8), at 3× the throughput of Llama-3.1-8B. An 8B that is both more accurate *and* 3× faster than the standard open 8B is the kind of model that gets deployed widely, because it dominates on both axes that matter. The reason it works at 8B scale is the same as at 56B — the hybrid loses nothing it needed by trading most attention for Mamba-2 — but the *impact* is larger at 8B because that is the size most teams actually run for cost reasons. The lesson is that efficiency techniques have the most leverage at the sizes people deploy most, and an efficient 8B reaches far more real workloads than an efficient 56B.

### 16. Training data: 20T tokens of curation and synthesis

The 56B's 20T-token (and 8B's 15T-token) training run is a case study in data scale and quality. The blend includes curated web data (a large fraction from filtered Common Crawl), plus heavy math, code, and academic data, and synthetic data — the same data-quality philosophy as the [Nemotron-4 340B report](/blog/machine-learning/large-language-model/nemotron-4-340b-synthetic-data-alignment), where data quality, not just quantity, drives results. The hybrid architecture does not get a pass on data: matching dense Transformers requires matching their training investment, and 20T tokens is a frontier-scale run. The lesson is that architectural efficiency and data quality are independent levers — the hybrid makes *inference* cheaper, but it still needs a top-tier *training* run to reach parity, and the report does not skimp on it. You cannot architecture your way out of needing good, abundant data.

### 17. FP8 made the 20T-token run affordable

Tying two threads together: the FP8 training recipe is what made the 20T-token pretraining economically feasible. Training a 56B for 20T tokens in BF16 would cost roughly twice the memory and bandwidth on the linear layers; FP8 halves that, which either halves the cost or lets you train longer for the same budget. So the FP8 recipe is not just a nice efficiency detail — it is an enabler of the data scale that the accuracy results depend on. This is a case study in how efficiency techniques compound across the lifecycle: FP8 makes training cheaper (so you can afford more tokens), the hybrid makes inference cheaper (so you can afford to serve), and MiniPuzzle makes a cheaper serving variant. Each stage of the model's life gets its own efficiency lever, and together they make a frontier-quality model affordable to both build and run. The lesson is to think about efficiency end-to-end — training, compression, and serving — rather than optimizing one stage in isolation.

## The bigger picture: the sequence-mixing primitive is now a design choice

Step back and Nemotron-H marks a conceptual shift: for a decade, "attention" and "sequence model" were synonyms — the Transformer's attention was simply *how* you mixed information across a sequence, and the only question was how to make attention cheaper (sparse attention, linear attention, grouped-query attention). Nemotron-H, and the Mamba line it builds on, reopens a more fundamental question: **what primitive should mix the sequence at all?** Attention is one answer (all-pairs comparison, perfect recall, expensive). State-space recurrence is another (compressed state, lossy recall, cheap). And the surprising lesson is that you do not have to choose globally — you can use *different primitives in different layers*, matching the primitive to what each layer needs.

This is the same heterogeneity insight that ran through [Llama-Nemotron's Puzzle](/blog/machine-learning/large-language-model/llama-nemotron-efficient-reasoning-models), taken one level deeper. Puzzle varied *whether a layer has attention* and *how wide its FFN is*; Nemotron-H varies *what kind of sequence mixer a layer uses*. In both, the rejected assumption is uniformity — the idea that every layer should be built the same way. Once you drop that assumption, the architecture becomes a palette of primitives (attention, Mamba-2, FFN) to be arranged for the workload and the hardware, and the arrangement is a design problem with real degrees of freedom. The 8% attention figure is one point in that design space; future models will explore others (different ratios, different placements, different primitives entirely).

For the field, the trajectory is clear: the monoculture of all-attention Transformers is giving way to *hybrids* that mix primitives, and the reason is economic — the KV cache is the binding constraint on long-context serving, reasoning models live at long context, and removing most of the cache is the highest-leverage way to make them affordable. Nemotron-H is an early, complete, production-grade example of what that future looks like: mostly a cheap mixer, a little of the expensive one, trained in low precision, compressed with an architecture-aware toolkit, and matching the dense models it replaces. The sequence-mixing primitive has become a design choice, and the design space is wide open.

### 18. The hybrid as a hedge against context-length growth

A forward-looking case study: context windows keep growing — 8k became 128k became 1M — and every doubling makes the KV cache problem worse for dense Transformers, quadratically for prefill compute and linearly for decode memory. A hybrid architecture is, in effect, a *hedge* against this trend: because its dominant primitive (Mamba-2) has constant memory in context length, its serving cost grows far more gently as context windows expand. A dense Transformer that is affordable at 8k may be ruinous at 1M; a hybrid stays tractable. So choosing a hybrid is not just optimizing for today's workloads — it is buying insurance against the direction the field is clearly heading (longer context, more reasoning, more documents). The lesson is to factor *trends* into architecture choices: if the workload is moving toward longer sequences, an architecture whose cost is flat in sequence length is worth more than its current speedup suggests, because that speedup grows with the context length the field keeps demanding.

### 19. Reproducing the recipe: what you actually need

A practical case study in adoptability: what does a team need to build a Nemotron-H-style model? Three things, in order of difficulty. First, **Mamba-2 kernels in your training and inference stack** — without efficient SSM kernels, you lose both the training throughput (the parallel scan) and the inference win (the recurrence), so this is the hard prerequisite. Second, **a frontier-scale data pipeline** — the hybrid needs 15-20T tokens to reach parity, so you need the data. Third, **the recipes from this report** — the 8% dispersed-attention layout, the FP8 format-per-tensor scheme, and MiniPuzzle for compression, all of which the report documents. The first is the real barrier (kernel support), which is why hybrids have spread fastest among labs with strong systems teams. The lesson is that an architecture's adoptability is gated by its *kernel ecosystem*: a great architecture without efficient kernels is a research result, not a deployable model, and the spread of hybrids will track the spread of mature Mamba-2 kernels across the major training and serving frameworks.

## When to reach for a hybrid Mamba-Transformer — and when not to

**Reach for it when:**

- **You serve long context and are memory-bound.** This is the canonical case — the hybrid's flat KV cache is the whole point, and the advantage grows with context length.
- **You run reasoning or document workloads.** Long generated sequences (chains of thought) or long inputs (documents) are exactly where constant memory pays off.
- **You want a smaller, faster model at the same accuracy tier.** The hybrid matches dense Transformers while being cheaper to serve.
- **You can train at scale.** The hybrid needs a full pretraining run (15-20T tokens) to reach parity; FP8 helps make that affordable.
- **You control your inference stack.** Mamba-2 layers need efficient SSM kernels; the throughput win assumes your serving engine has them.

**Skip it (or be careful) when:**

- **Your workloads are short-context.** If you never exceed a few thousand tokens, the KV cache is small and the hybrid's advantage is modest — a dense Transformer is simpler.
- **You need maximal exact-recall.** Tasks that hinge on perfectly retrieving many specific distant tokens stress the part Mamba-2 is weakest at; you may need more than 8% attention, eroding the savings.
- **You lack SSM kernel support.** Without efficient Mamba-2 kernels in your inference engine, you will not realize the throughput win, and the architecture is just unfamiliar.
- **You are fine-tuning, not pretraining.** The hybrid's benefits come from a from-scratch (or near-scratch) training run; you cannot easily convert an existing dense Transformer into a strong hybrid by fine-tuning.

A final framing for the decision. The right way to think about a hybrid is not "a faster Transformer" but "a different point on the recall-versus-cost curve." A dense Transformer sits at one extreme: perfect recall, maximal cost. A pure-SSM model sits at the other: cheapest, but lossy recall. The hybrid lets you dial in an intermediate point — and the 8% attention figure is the dial setting that NVIDIA found keeps recall essentially intact while capturing most of the cost savings. If your workload needs more exact recall (heavy retrieval, copying long verbatim spans), dial the attention fraction up; if it is more about fluent generation over long context (reasoning, summarization), the 8% setting or even less suffices. The hybrid is not a single architecture but a *family* parameterized by how much of the expensive primitive you keep, and the report's contribution is finding a setting that works broadly. That parameterization is the real product: a knob between recall and cost that dense Transformers do not give you, because they are pinned at the expensive end.

The one-sentence version:

> If long-context serving has you fighting the KV cache, stop fighting it and remove it: replace most of your attention with a constant-memory state-space recurrence, keep ~8% attention for the exact-recall jobs the recurrence cannot do, and you get a model that matches dense Transformers on accuracy while serving long context several times faster.

## Further reading

- [Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models](https://arxiv.org/abs/2504.03624) — the full report, with the layer schedules, FP8 recipe, and MiniPuzzle details.
- [Minitron: pruning and distillation](/blog/machine-learning/large-language-model/nvidia-minitron-pruning-distillation) — the compression foundation MiniPuzzle builds on.
- [Llama-Nemotron: efficient reasoning models](/blog/machine-learning/large-language-model/llama-nemotron-efficient-reasoning-models) — the NAS that MiniPuzzle's conditional search descends from.
- [KV cache optimization](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management) — why the KV cache is the long-context bottleneck the hybrid attacks.
- [Quantization in LLMs](/blog/machine-learning/large-language-model/quantization-in-llm) and [past the 4-bit wall](/blog/machine-learning/large-language-model/past-4-bit-wall-frontier-llm-quantization) — the low-precision context for the FP8 recipe.
- [Choosing the right LLM architecture for a task](/blog/machine-learning/large-language-model/choosing-right-llm-architecture-task) — where hybrids fit in the architecture landscape.

One last reflection ties Nemotron-H to the arc of this series. Minitron asked "how few parameters can we keep?" Llama-Nemotron asked "what per-layer architecture fits this hardware?" Nemotron-H asks the most fundamental question yet: "what primitive should mix the sequence at all?" Each report relaxes an assumption the previous one left standing — first that you must train every size from scratch, then that every layer must be identical, and now that attention must be the sequence mixer. Read together, they trace a single trajectory: the dismantling of the uniform, train-from-scratch, all-attention Transformer in favor of derived, heterogeneous, mixed-primitive models tuned for deployment. Nemotron-H is the point where that trajectory reaches the sequence-mixing primitive itself, the most load-bearing assumption of all. Whatever the next report relaxes, the pattern is clear: efficiency at the frontier increasingly comes from questioning architectural conventions that were never laws, only habits — and the KV cache, the thing every long-context deployment fights, turns out to be not a law of nature but a habit you can mostly walk away from — provided you keep just enough attention to remember what the recurrence would forget.

*Next in the series: a turn from language to speech, with Canary and Parakeet — NVIDIA's FastConformer ASR models and the "less is more" data strategy behind them.*

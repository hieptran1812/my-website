---
title: "How DeepSeek-V3 Trained a 671B-Parameter Model for $5.6M: FP8, Multi-Token Prediction, and Loss-Free Balancing"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A principal-engineer teardown of the DeepSeek-V3 training recipe — fine-grained FP8 with CUDA-core accumulation, auxiliary-loss-free expert balancing, Multi-Token Prediction that doubles as a speculative drafter, and the cost arithmetic that makes $5.6M real."
tags: ["llm", "deepseek-v3", "moe", "fp8", "mixed-precision", "multi-token-prediction", "speculative-decoding", "load-balancing", "training", "pretraining", "deep-learning", "gpu"]
category: "machine-learning"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 50
---

When the DeepSeek-V3 technical report landed with a training cost of **$5.576M**, the number went viral for the wrong reason. Headlines framed it as "a frontier model for the price of a house," which is true but useless. The interesting claim isn't the dollar figure — it's that a 671-billion-parameter Mixture-of-Experts model was trained to frontier quality on **2,048 H800 GPUs**, the bandwidth-throttled export variant of the H100, in **under two months, with no irrecoverable loss spikes and zero rollbacks**. That last clause is the one that should make you sit up. Anybody who has babysat a large pretraining run knows that the failure mode isn't usually "too expensive" — it's "the loss diverged at 3 a.m. on token 6.2 trillion and we lost a week."

So the right question is not "how did they make it cheap?" It's "what set of co-designed decisions, from the GPU's tensor cores up to the optimizer, made an FP8 MoE run of this scale *stable and fast at the same time*?" DeepSeek-V3 is worth reading not as a model but as a **systems artifact**: a stack of four model-architecture choices and three training-system choices that only pay off because they were designed against each other.

The H800 detail is not incidental, either. The H800 is an H100 with its inter-GPU NVLink bandwidth cut roughly in half — a deliberate export-control throttle. A naive large-MoE training run on H800s would spend most of its time waiting on the cross-GPU all-to-all that shuffles tokens to their experts. So DeepSeek was forced, by the hardware they were allowed to buy, to engineer around communication. Much of what looks like "cost optimization" is really "we couldn't afford to move data, so we made the model and the schedule move less of it." Constraint bred the design.

This post is the first in a series mining DeepSeek's published work for techniques you can actually reuse. We start with V3 because it is the densest single source of engineering ideas in the whole corpus. We'll go deep on three of them — **auxiliary-loss-free load balancing**, **Multi-Token Prediction (MTP)**, and **fine-grained FP8 training** — then close with the cost arithmetic, because the arithmetic is where the engineering choices cash out.

## Why $5.6M is the wrong number to be shocked by

The instinct on reading the headline is to assume DeepSeek found one magic trick. They didn't. The cost is an *emergent property* of a co-design discipline, and most of the individual pieces had been published before — by DeepSeek themselves (MLA in V2, DeepSeekMoE in early 2024) and by others (FP8 training, speculative decoding). What's new is the integration and the engineering rigor that made the integration survive 14.8 trillion tokens.

| Common assumption | The naive view | The reality in DeepSeek-V3 |
|---|---|---|
| "They used a cheaper algorithm" | One clever trick cut the cost 10× | No single trick; ~4 model + 3 system techniques compounding |
| "FP8 is just casting weights to 8 bits" | `model.half().half()` essentially | Per-tile/block scaling + FP32 CUDA-core accumulation; most ops stay BF16/FP32 |
| "Load balancing needs an auxiliary loss" | Add `L_aux`, tune its coefficient | A bias-term control loop with **no balancing gradient at all** |
| "Multi-token prediction is a decoding hack" | Bolt a draft model on at inference | A *training objective* that densifies signal **and** doubles as a co-trained drafter |
| "Bigger model = more KV cache = slower" | 671B params must be brutal to serve | MLA shrinks the KV cache ~14×; 37B activated per token keeps FLOPs sane |
| "$5.6M includes the research" | That's the all-in cost | It's the **final run only** — excludes ablations and prior R&D |

![The co-design stack: a delivered 671B MoE model rests on model co-design, which rests on system co-design, which rests on commodity H800 hardware.](/imgs/blogs/deepseek-v3-fp8-mtp-loss-free-balancing-1.webp)

The diagram above is the mental model for the whole article. Read it top-down: the thing you get — a 671B-total / 37B-activated MoE at frontier quality — sits on a layer of **model co-design** (Multi-head Latent Attention, DeepSeekMoE's fine-grained experts, auxiliary-loss-free balancing, and Multi-Token Prediction). That layer in turn rests on **system co-design** (FP8 GEMMs, the DualPipe bidirectional pipeline, and a warp-specialized cross-node all-to-all). And all of it is anchored to ordinary, throttled hardware: 2,048 H800s, 14.8T tokens, ~54 days, ~$5.6M. Pull any one layer and the layer above it stops being affordable. The FP8 GEMMs only matter because the MoE makes the model big enough that GEMM dominates; MTP only pays off because the architecture lets you co-train a draft head cheaply; loss-free balancing only matters because at 256 experts per layer the balancing tax would otherwise be enormous.

> The model is the part you can download. The recipe is the part that's hard to copy. DeepSeek open-sourced the model *and* documented the recipe, which is why this report is the most valuable training document of its year.

The rest of this post is a tour of that stack, working from the parts that touch the model's quality (balancing, MTP) down to the parts that touch the silicon (FP8, cost). For Multi-head Latent Attention — the fourth model-side technique and arguably the most influential — see the dedicated [Multi-head Latent Attention deep-dive](/blog/machine-learning/large-language-model/kv-cache); here we treat MLA as a given and focus on what's new in V3.

A quick spec sheet so the numbers later have a home. DeepSeek-V3 is a **61-layer** Transformer with hidden dimension **7,168**. The first three layers use a dense feed-forward network; layers 4 through 61 are **MoE layers**, each holding **1 shared expert and 256 routed experts** (expert intermediate dimension 2,048), of which the router activates the shared expert plus **8 routed experts** per token. That's how you get **671B total parameters but only 37B activated** for any given token — the cost of a forward pass scales with the 37B, while the model's capacity scales with the 671B. Attention is MLA with 128 heads. Context window is 128K, reached in two YaRN stages from a 4K pretraining length. The model was trained on **14.8T tokens** in BF16/FP8 mixed precision.

To anchor why "37B activated out of 671B" is the whole game, compare the three obvious points in the design space:

| Design | Total params | Activated / token | KV cache cost | Forward FLOPs |
|---|---|---|---|---|
| Dense 70B (e.g. Llama-class) | 70B | 70B (100%) | full MHA / GQA | ~all params |
| Dense "671B" (hypothetical) | 671B | 671B (100%) | enormous | prohibitive |
| **DeepSeek-V3 (MoE + MLA)** | **671B** | **37B (5.5%)** | **MLA-compressed (~14× smaller)** | **~37B-equivalent** |

The MoE buys you the capacity of a 671B model at the per-token compute of a ~37B one; MLA buys you a KV cache small enough that the long-context serving doesn't eat the savings. Everything else in this post exists to make that trade *trainable* and *stable*.

## 1. Auxiliary-loss-free load balancing: a control loop, not a loss term

**Senior rule of thumb: if a regularizer is fighting your main objective, move it out of the gradient.**

Every sparse MoE has the same problem. A learned router sends each token to its top-K experts; left alone, the router collapses, routing almost everything to a handful of "popular" experts while the rest starve. Starved experts are dead capacity, and — worse on a multi-GPU job — a hot expert becomes a straggler that every other GPU waits on during the all-to-all. So you need **load balancing**: a pressure that spreads tokens roughly evenly across experts and across devices.

It helps to be precise about *why* imbalance is expensive, because there are two distinct costs and they pull in slightly different directions. The first is **capacity**: an expert that never gets tokens never learns, so an imbalanced model has fewer *effective* parameters than its nominal count. The second, and the one that dominates at scale, is **throughput**: in an expert-parallel layout the 256 experts are spread across GPUs, every token is shipped to its chosen experts over the network, the experts compute, and the results are shipped back — an all-to-all in each direction. That all-to-all is a barrier. If one expert on one GPU received 4× its share of tokens, every other GPU in the group sits idle waiting for that straggler to finish. On a 2,048-GPU job throttled to half NVLink bandwidth, a 10% imbalance is not a 10% slowdown — it can be much worse, because the slowest expert sets the pace for the whole step. Balancing is therefore not a nicety; it's the difference between a cluster that's 90% utilized and one that's 60% utilized.

### The standard fix and its hidden tax

The standard remedy since GShard and the Switch Transformer is an **auxiliary balance loss**. You compute, per batch, the fraction of tokens routed to each expert and the mean routing probability for each expert, multiply them element-wise, sum over experts, scale by a coefficient α, and add the result to your training loss. Minimizing it pushes routing toward uniformity. It works, and it has been the default for years. But it has a cost that nobody likes to say out loud: **the balancing gradient flows into the same weights that are trying to learn language.** Every step, the optimizer receives a blend of "predict the next token well" and "use your experts evenly," and those two objectives are not aligned. The balance loss is, by construction, an interference term. Tune α too high and you flatten the specialization that makes MoE worth having; too low and you get stragglers. There is no good value — only a least-bad one, found by a coefficient sweep that you then pray generalizes across the whole run as the data distribution shifts.

DeepSeek-V3's answer is to **stop balancing with a loss at all.** Instead, each routed expert *i* gets a scalar **bias term `b_i`** that is added to its affinity score *only for the top-K selection*. The expert with the highest `affinity + bias` gets picked. Crucially, once an expert is selected, the **gating weight that actually scales its output uses the raw affinity, not the biased score.** The bias steers *who gets chosen*; it never touches *how much the chosen expert contributes*. So no balancing signal ever enters the gradient.

![Before/after: auxiliary-loss balancing injects an interference gradient into the expert weights; the loss-free bias scheme nudges a routing-only bias term and leaves the gradient untouched.](/imgs/blogs/deepseek-v3-fp8-mtp-loss-free-balancing-2.webp)

The before/after figure above is the crux of the idea. On the left (the old way), the auxiliary loss adds a gradient that flows back into the expert weights — the experts are pushed toward uniform use, and that push competes with the language objective. On the right (DeepSeek-V3), the per-expert bias `b_i` lives entirely outside the gradient: it's nudged up or down by load, the gating weight stays raw, and the quality tax disappears.

### The bias as an integral controller

How does `b_i` get set if not by gradient descent? By a **control loop**. After each training step, DeepSeek looks at how loaded each expert was. If expert *i* was overloaded relative to the target, they **decrease** `b_i` by a small step γ (making it less likely to be chosen next time). If it was underloaded, they **increase** `b_i` by γ. That's the entire mechanism: a per-expert integral controller with update speed **γ = 0.001**. It is dimensionally separate from SGD — the model's weights are learning language; the bias vector is, in parallel, doing nothing but keeping the routing histogram flat.

If you've done any control theory, the right frame is that `b_i` is the **integral term of a controller** whose setpoint is "uniform load." Each step's load error (over/under target) is integrated into `b_i` at rate γ. A persistent bias in routing accumulates a persistent correction; a transient blip washes out because the integrator averages over time. This is why the scheme is robust without tuning: an integral controller drives steady-state error to zero regardless of the exact plant gain, so you don't have to find a magic α the way you do for an auxiliary loss. The only knob is γ, and it controls *how fast* you correct, not *how much* you trade off quality — a far more forgiving parameter. Set γ too large and the routing oscillates (over-correcting hot and cold each step); too small and balance is slow to establish. 0.001 sits comfortably in between, and because there's no quality term being traded, even a sub-optimal γ costs you a little throughput, never model quality.

Here is the mechanism in code. Note that the bias never appears in any backward pass — it's a buffer, not a parameter, so it carries no gradient and no optimizer state:

```python
import torch

class LossFreeRouter:
    def __init__(self, n_routed=256, top_k=8, gamma=1e-3):
        self.bias = torch.zeros(n_routed)        # b_i: a buffer, NOT a parameter
        self.n_routed, self.top_k, self.gamma = n_routed, top_k, gamma

    def route(self, tokens, centroids):
        affinity = torch.sigmoid(tokens @ centroids.T)   # s_i in [0, 1], shape [N, 256]
        score = affinity + self.bias                      # bias used for SELECTION only
        idx = score.topk(self.top_k, dim=-1).indices      # which experts are chosen
        gate = affinity.gather(-1, idx)                   # gate uses RAW affinity
        gate = gate / gate.sum(-1, keepdim=True)          # normalize over the top-k
        return idx, gate

    @torch.no_grad()
    def update_bias(self, idx):
        load = torch.bincount(idx.flatten(), minlength=self.n_routed).float()
        target = load.mean()                              # perfectly balanced load
        self.bias[load > target] -= self.gamma            # punish hot experts
        self.bias[load < target] += self.gamma            # promote cold experts
```

The elegance is that **balance and quality are decoupled by construction**. A learned-loss approach asks the optimizer to find weights that are simultaneously good at language and naturally balanced — an over-constrained ask. The control loop says: let the weights specialize however they want, and I'll independently bump a bias to keep the histogram flat. In the report's ablations this loss-free scheme beats both the standard auxiliary-loss MoE and even a "loss-but-tuned" baseline on downstream quality, precisely because it stops taxing the gradient.

| | Auxiliary-loss balancing | Loss-free bias balancing |
|---|---|---|
| Balancing signal | gradient into model weights | heuristic update to a bias buffer |
| Effect on quality | interference tax (despecializes experts) | none — gradient untouched |
| Tuning burden | sweep α; re-tune as data shifts | one forgiving knob γ |
| Failure mode | too-high α flattens experts | too-high γ oscillates routing |
| Per-sequence collapse | handled by the same loss | handled by a tiny separate seatbelt loss |

### Complementary routing-for-communication: device- and node-limited routing

Balancing the *count* of tokens per expert is necessary but not sufficient on a throttled cluster, because the cost that actually hurts is *cross-node communication volume*, not token count. Two experts with equal load can still impose very different network costs if one of them lives on a distant node. DeepSeek-V3 adds a **node-limited routing** constraint: each token's chosen experts are confined to **at most M = 4 nodes**, picked by summing each node's top affinities and keeping the best M. This caps the number of inter-node hops a single token can trigger, which bounds the all-to-all traffic and lets the communication be overlapped with computation (the subject of the DualPipe discussion below). It's the same philosophy as the bias controller — balance for the resource that's actually scarce — applied to the network instead of to expert capacity. DeepSeek-V2 used the stricter "device-limited" version (≤3 devices); V3 relaxes it to nodes because the intra-node NVLink is cheap and only the inter-node InfiniBand needs rationing.

### Second-order optimization: the tiny safety loss

There is one residual risk. The bias controls balance *in aggregate over a batch*, but a single pathological sequence could still dump all its tokens on one expert within that sequence (think: a long run of one language or one code style). To guard against that, V3 keeps a **complementary sequence-wise balance loss** with a deliberately microscopic coefficient **α = 0.0001**. It's small enough not to meaningfully distort the gradient, but present enough to discourage per-sequence collapse. The lesson is general: you can run a heuristic controller for the 99% case and keep a near-zero-weight regularizer as a seatbelt for the 1% tail, rather than cranking one loss to cover both.

> Auxiliary-loss balancing optimizes the model and the histogram with the same knob. Loss-free balancing gives each its own knob. Whenever two objectives are pulling on one gradient, ask whether one of them can be a controller instead.

The γ schedule is itself worth a note: DeepSeek runs **γ = 0.001 for the bulk of training and then anneals it to 0** near the end. Once routing has settled into a healthy distribution, you stop nudging — letting the experts fine-tune against a frozen routing pattern in the final stretch, the same way you'd decay a learning rate. Balance is a thing you establish early and then leave alone. Freezing the routing late also means the final checkpoint's expert assignments are stable, which matters for inference: a server can build its expert-parallel layout around a routing distribution that won't shift under it.

## 2. Multi-Token Prediction: one objective, two payoffs

**Senior rule of thumb: a training signal that costs you almost nothing extra is worth adding even if it only helps a little — and MTP helps in two places at once.**

Standard language-model pretraining predicts one token at a time: given positions 1…*i*, predict *i*+1. The signal per forward pass is exactly one next-token distribution per position. DeepSeek-V3 adds a second, parallel objective: **also predict token *i*+2**, using a small dedicated module that *chains off the main model's hidden state*. This is Multi-Token Prediction, and V3 uses depth **D = 1** — one extra future token.

### Sequential, not parallel

The structure matters, because V3's MTP is not the "parallel heads" design from earlier multi-token work (such as Meta's 2024 multi-token-prediction paper, which attaches several independent output heads to the same final hidden state). It is **sequential and causal**: the MTP module takes the main trunk's final hidden state for position *i*, RMSNorm-normalizes it, concatenates it with the (also normalized) embedding of the *actual* token at *i*+2, projects the concatenation down, and runs it through **one additional Transformer block** to produce a second hidden state, which then predicts *i*+2 through the **shared output head**. The difference is consequential: parallel heads all condition on the same representation and can't model the dependency between the two predicted tokens, whereas DeepSeek's chained module conditions the *i*+2 prediction on having "committed" to *i*+1, preserving the causal chain. That causal consistency is precisely what makes the module reusable as a speculative drafter later.

![Multi-Token Prediction as a layered graph: the main trunk feeds both the shared head and the MTP module; the MTP module reuses the tied embedding and head to predict t+2 under a lambda-weighted loss.](/imgs/blogs/deepseek-v3-fp8-mtp-loss-free-balancing-3.webp)

Trace the figure. The **main model** (61 layers) produces hidden state `h⁰ᵢ` and, through the **shared embedding and output head**, predicts token *i*+1 under the ordinary main loss. The **MTP module** consumes `h⁰ᵢ`, mixes in the embedding of the future token *i*+2, runs its single extra block to get `h¹ᵢ`, and predicts *i*+2 — but its loss is scaled by **λ**. The weights that matter most here are *shared*: the embedding table and the output head are tied between the main model and the MTP module, so the MTP module adds only ~one Transformer block's worth of new parameters (a few percent of the model), not a second model. That weight tying is what keeps the draft and target distributions consistent and keeps the parameter overhead small.

### The loss schedule and why D=1

The loss is a weighted sum: the full next-token loss plus λ times the MTP loss. DeepSeek sets **λ = 0.3 for the first 10T tokens and 0.1 for the remaining 4.8T**. Early on, the extra-token objective is a strong auxiliary signal that sharpens the representations (predicting two tokens ahead forces the hidden state to encode more about the future than greedy next-token does). Later, you dial it down so the model can specialize on the primary objective for final quality. Here it is as a training step — the comments are all inline so nothing reads as a heading:

```python
def training_step(tokens, tokens_seen):
    h0 = main_trunk(embed(tokens))                          # shared embedding table
    loss_main = cross_entropy(head(h0)[:, :-1], tokens[:, 1:])   # predict i+1

    fut = embed(tokens[:, 2:])                              # the actual token at i+2
    z   = proj(torch.cat([rmsnorm(h0[:, :-2]), rmsnorm(fut)], dim=-1))
    h1  = mtp_block(z)                                      # ONE extra Transformer block
    loss_mtp = cross_entropy(head(h1), tokens[:, 2:])       # predict i+2 via SHARED head

    lam = 0.3 if tokens_seen < 10e12 else 0.1               # lambda schedule
    return loss_main + lam * loss_mtp
```

Why D = 1 and not 2, 3, or 4? Because the marginal value of each additional predicted token falls off fast while its cost (an extra block, extra activations, extra memory) does not. Predicting *i*+2 is a meaningfully harder task than *i*+1 and so adds real signal; predicting *i*+5 is so uncertain that the loss is mostly noise, and the gradient it contributes is closer to a regularizer than a useful learning signal. DeepSeek found one extra token to be the sweet spot — enough to densify the signal and to support single-token speculative drafting, without paying for depth that doesn't earn its keep.

Why does this densify the signal at all? Because at every position the model now gets gradient from two futures, not one, and the second future is harder — it can't be predicted by trivially copying local n-gram statistics. The representation has to "plan" one token further, which fights the myopia that pure teacher-forced next-token training tends to induce. DeepSeek reports a consistent downstream-quality lift from MTP during pretraining, which on its own would justify the small parameter overhead.

### 2b. The free payoff: MTP as a speculative drafter

Here's where the design earns the word "co-design." A draft-and-verify speculative decoder normally requires a *separate* small draft model whose distribution you hope matches the big model's. The mismatch is the whole problem: a poorly-matched draft model proposes tokens the target rejects, and rejected drafts are wasted compute. But V3's MTP module is **trained jointly with the main model on the same data**, so its predictive distribution is consistent with the main model's by construction. At inference, you simply **keep the MTP module and use it as the draft head.**

![Timeline of one speculative-decode step: the main model emits a token and hidden state, the MTP head drafts the next token cheaply, the main model verifies it in parallel, and high acceptance yields about 1.8x tokens per second.](/imgs/blogs/deepseek-v3-fp8-mtp-loss-free-balancing-4.webp)

The loop, reading the timeline left to right: the **main model emits token tₙ** and its hidden state. The **MTP head drafts tₙ₊₁** for the cost of a single Transformer block — cheap. The **main model verifies the draft in parallel** (one extra position in the same batched forward pass, which it would have computed anyway). If the draft matches what the main model would have produced, you **accept it** and have now emitted two tokens for roughly the cost of one verification. DeepSeek measures the **second-token acceptance rate at 85–90%** across generation domains, which translates to roughly **1.8× decode throughput** at the same output quality. Disable speculation and the MTP module simply isn't loaded — the main model stands alone, no penalty.

```python
def decode_step(prefix):
    h0 = main_trunk(prefix)
    t_next = sample(head(h0[:, -1]))                        # main model's own next token
    draft  = sample(head(mtp_block(proj_at(h0[:, -1], t_next))))   # MTP drafts +1 token

    h0v = main_trunk(prefix + [t_next])                     # verify: one position further
    accept = sample(head(h0v[:, -1])) == draft              # agrees ~85-90% of the time
    return [t_next, draft] if accept else [t_next]          # 2 tokens, or fall back to 1
```

This is the difference between bolting on speculative decoding after the fact and getting it as a side effect of a training objective you wanted anyway. The same extra parameters that sharpened pretraining are now a perfectly-aligned drafter. One design decision, two payoffs — and at inference time you choose whether to spend the parameters on speed or drop them entirely. Compare this with the standalone approach in our [speculative decoding deep-dive](/blog/machine-learning/large-language-model/speculative-decoding), where draft-target alignment is the central engineering headache; MTP sidesteps it by training the draft head *in* the target.

| | MTP at training | MTP at inference |
|---|---|---|
| Role | auxiliary loss (densifies signal) | speculative draft head |
| Loss weight λ | 0.3 → 0.1 over training | n/a |
| Cost | +1 block forward/backward | +1 block per drafted token |
| Payoff | downstream quality lift | ~1.8× tokens/s at equal quality |
| Optional? | on during pretraining | drop the module → main model alone |

It's worth being precise about *why* the verification position is nearly free. In batched serving the GPU is throughput-bound, not latency-bound: the matmuls for one extra position ride alongside the work already in flight, so adding the draft-verification token barely moves the wall clock. This is also why the gain shrinks for tiny-batch, latency-bound decoding — there's no spare matmul to hide the verification behind. The 1.8× is a serving-throughput number, not a single-stream-latency number; know which regime you're in before you bank on it.

## 3. Fine-grained FP8: making 8-bit training stable

**Senior rule of thumb: low precision fails at the tails. Engineer for the outliers, not the average.**

This is the part of the report with the most reusable systems content, and the part most often mischaracterized. "FP8 training" sounds like you cast everything to 8 bits and pray. What DeepSeek actually did is far more surgical, and the surgery is why it worked when naive FP8 diverges.

### E4M3 everywhere, and why that's a choice

FP8 comes in two layouts: **E4M3** (1 sign, 4 exponent, 3 mantissa bits — more precision, narrower range, max magnitude ~448) and **E5M2** (1 sign, 5 exponent, 2 mantissa bits — wider range, coarser precision, max magnitude ~57344). The conventional wisdom from earlier FP8 work (NVIDIA's Transformer Engine, for instance) was to use E4M3 for forward activations and E5M2 for gradients, because gradients have a wider dynamic range that the extra exponent bit accommodates. DeepSeek uses **E4M3 for everything** — forward and backward, activations and gradients.

They can afford the narrower range because of the single most important idea in their FP8 scheme: **fine-grained scaling**. The dynamic range a number format must cover is set by the *spread of magnitudes within a scaling group*. If your scaling group is an entire tensor, that spread is huge and you need E5M2's exponent. If your scaling group is a tiny tile, the spread within it is small, a single E4M3 scale covers it comfortably, and you get to keep E4M3's extra mantissa bit (better precision) everywhere. Fine-grained scaling, in other words, *converts dynamic-range pressure into precision*. That's the trade, and it's a good one.

### Tile the activations, block the weights

The enemy of low-precision training is the **outlier**. A single activation value 1000× larger than its neighbors forces the scale factor for the whole tensor to accommodate it, which crushes all the normal-magnitude values down into a handful of representable FP8 levels — you lose them to quantization. This is the same outlier problem that motivates inference-time methods like SmoothQuant; DeepSeek's answer is to attack it at training time with granularity rather than with a smoothing transform.

**Activations are scaled per 1×128 tile** — each group of 128 channels within a single token gets its own scale factor. **Weights are scaled per 128×128 block.** An outlier now only blows up the scale of *its own tile*, leaving every other tile in the tensor at full effective precision.

![Fine-grained FP8 tiling: activations are scaled per 1x128 row-tile, weights per 128x128 block, so an outlier in one activation tile cannot corrupt the rest of the tensor.](/imgs/blogs/deepseek-v3-fp8-mtp-loss-free-balancing-5.webp)

The figure makes the granularity concrete. On the left, the activation matrix is sliced into **1×128 row-tiles**, each carrying its own scale (s₁, s₂, …). The red tile in the middle is one that happens to contain an **outlier** — and the damage is contained: only that tile's scale absorbs the spike; tiles s₁, s₂, s₄, s₅ are untouched. On the right, the weight matrix is sliced into **128×128 blocks**, each independently scaled. This is "max-abs scaling, computed online per tile/block" — at runtime, you take the maximum absolute value in each tile, derive a scale that maps it to the top of E4M3's range, and quantize. Because the scope is a tile and not a tensor, the chance that a tile is *both* dominated by an outlier *and* full of values you care about is small.

The asymmetry — 1×128 tiles for activations but 128×128 blocks for weights — is deliberate. Activation outliers tend to be *channel-structured*: certain feature channels spike across many tokens. A 1×128 tile (one token, 128 channels) isolates a spike to the channels where it occurs. Weights are more uniformly distributed and change slowly, so a coarser 128×128 block is enough and costs fewer scale factors to store and apply. You scale at the granularity the data's structure demands, no finer.

### The accumulation fix: promote to FP32 on CUDA cores

Tiling solves the input-quantization problem. There's a second, sneakier problem: **accumulation precision inside the matmul.** When a Hopper tensor core multiplies FP8 by FP8, it accumulates the partial products — and the accumulation register on the tensor-core path (the WGMMA instruction) holds fewer effective mantissa bits than full FP32. Over a long reduction (a GEMM's inner dimension can be thousands of elements), those small accumulation errors compound. The loss curve drifts away from the BF16 baseline, slowly, in a way that looks like "FP8 just isn't accurate enough" — when really it's the *accumulator*, not the inputs. This is an insidious failure because it doesn't crash; it quietly biases your model, and you might not notice until you compare against a BF16 control run trillions of tokens later.

DeepSeek's fix is a hardware-aware workaround: **promote the partial sums to FP32 on the CUDA cores at a fixed interval.** Specifically, every **N_C = 128 elements** of the inner reduction, the tensor core's partial result is copied out and added into an FP32 accumulator maintained on the general-purpose CUDA cores, then the tensor-core accumulator is reset. You pay a little extra data movement for it, but you recover almost all the precision the WGMMA path was losing. The genius of the 128 interval is that it matches the tile granularity — the promotion boundary lines up with the scaling boundary, so the bookkeeping is cheap.

![The FP8 accumulation fix as a pipeline: FP8 inputs feed the WGMMA matmul, whose limited-precision partial products spill every 128 elements into an FP32 accumulator on the CUDA cores, then dequantize to BF16 within 0.25% of the baseline loss.](/imgs/blogs/deepseek-v3-fp8-mtp-loss-free-balancing-6.webp)

Follow the pipeline. **FP8 inputs (E4M3, per-tile/block scaled)** enter the **WGMMA** FP8×FP8 matmul on the tensor core. The matmul produces **partial products** with limited accumulation bits — the precision leak. Every 128 elements, those partials **spill and are promoted to FP32 on the CUDA core**, where they land in a **full-precision FP32 accumulator**. The final result is **dequantized back to BF16**, and the net effect is a training loss that tracks the BF16 baseline to **within 0.25%** — close enough that, over 14.8T tokens, FP8 and BF16 produce effectively the same model. Here is the shape of the computation:

```python
N_C = 128
def fp8_gemm(A_bf16, W_bf16):                               # A: [M, K], W: [K, N]
    out = torch.zeros(M, N, dtype=torch.float32)            # FP32 accumulator on CUDA cores
    for k0 in range(0, K, N_C):
        k1 = k0 + N_C
        sA = A_bf16[:, k0:k1].abs().amax(1, keepdim=True) / E4M3_MAX   # per 1x128 tile
        sW = W_bf16[k0:k1, :].abs().amax() / E4M3_MAX                  # per 128x128 block
        A8 = to_e4m3(A_bf16[:, k0:k1] / sA)
        W8 = to_e4m3(W_bf16[k0:k1, :] / sW)
        partial = wgmma_fp8(A8, W8)                          # tensor core, low-precision accum
        out += partial.float() * (sA * sW)                  # PROMOTE to FP32, then rescale
    return out.to(torch.bfloat16)                           # dequantize the result
```

In the real implementation this loop is fused into a single kernel so the "promote to FP32" step doesn't actually round-trip through global memory — the DeepGEMM library DeepSeek open-sourced does exactly this, hitting north of 1,300 FP8 TFLOPS on Hopper with about 300 lines of JIT-compiled CUDA. The dequant overhead, in other words, is real but small, and it's amortized by the kernel doing it in registers. We cover DeepGEMM and the rest of the open-infra kernels in the [next post in this series](/blog/machine-learning/mlops/).

### Online scaling, not delayed scaling

There is a subtle but important contrast with how FP8 is usually done in production frameworks. NVIDIA's Transformer Engine — the most common FP8 training path — uses **delayed scaling**: it keeps a rolling history of the maximum absolute value (amax) seen in recent steps for each tensor, and uses that history to pick the scale for the *current* step, on the assumption that the magnitude distribution drifts slowly. Delayed scaling is cheaper (you don't have to compute the current step's amax before you can quantize) but it's a bet that the past predicts the present, and it's per-tensor, so it inherits all the outlier sensitivity discussed above.

DeepSeek instead uses **online, per-tile scaling**: the scale for each 1×128 tile is computed from *that tile's own current values*, every step, with no history. This is more work — you must scan the tile for its max before quantizing — but it's exact for the data you're actually multiplying, and combined with the fine granularity it removes both the staleness risk of delayed scaling and the outlier risk of per-tensor scaling. The cost is that you need custom kernels (you can't just flip a flag in a stock framework), which is precisely why DeepSeek had to write and open-source DeepGEMM. The trade is characteristic of the whole report: spend engineering to remove a source of instability, rather than accept a cheaper-but-riskier default.

### Not everything is FP8

The final piece of the FP8 story is knowing what to *leave alone*. Precision-sensitive operators stay in higher precision. Only the three big matmuls run in FP8.

![The selective-precision map: only the Fprop, Dgrad, and Wgrad GEMMs run in FP8; embedding and output head stay BF16; MoE gating, normalization, softmax, and the master weights and optimizer state stay FP32.](/imgs/blogs/deepseek-v3-fp8-mtp-loss-free-balancing-7.webp)

The matrix above is the precision map, and it is a checklist you can copy. The **three GEMMs** — Fprop (the forward matmul), Dgrad (the activation-gradient matmul in backward), and Wgrad (the weight-gradient matmul in backward) — run in **FP8 E4M3** with the 1×128 tile scaling. The **embedding and output head** stay in **BF16**. And the genuinely sensitive operators — **MoE gating** (where a small numerical error changes *which expert is chosen*, a discrete and unrecoverable decision), **normalization and softmax** (which involve sums and exponentials that amplify error), and the **master weights plus Adam optimizer state** — stay in **FP32**.

The MoE-gating case deserves its own sentence because it's the most instructive. Most low-precision error is *graceful*: a weight that's off by a quantization step nudges an output slightly, and training absorbs it. But the gating decision is *discrete* — it's an argmax over expert scores. If FP8 rounding flips the order of the two closest experts, the token goes to a different expert entirely, and there is no "slightly wrong" version of that; it's a categorical error that the gradient can't smoothly correct. Discrete decisions are exactly where you must not economize on bits. The same logic applies to the normalization statistics: a softmax or an RMSNorm sums many terms, and low-precision summation of many terms is where catastrophic cancellation lives.

As a config sketch, the kind of thing you'd actually write — note the YAML comments are inline, not full-line:

```yaml
fp8:
  enabled: true
  format: e4m3                 # E4M3 for ALL fp8 tensors (not the E4M3/E5M2 split)
  activation_scale: per_tile   # 1 x 128
  weight_scale: per_block      # 128 x 128
  accumulate: fp32_promote     # promote partials to FP32 on CUDA cores
  accumulate_interval: 128     # N_C
  ops: [fprop_gemm, dgrad_gemm, wgrad_gemm]
keep_bf16: [embedding, output_head]
keep_fp32: [moe_gating, rmsnorm, softmax, master_weights, optimizer_state]
loss_error_vs_bf16_target: 0.0025   # < 0.25%
```

> FP8 is not "half of BF16." It's a budget you allocate per-operator. The skill is knowing which operators tolerate 8 bits and which will silently poison your loss if you let them.

| | Naive per-tensor FP8 | DeepSeek-V3 fine-grained FP8 |
|---|---|---|
| Scaling scope | whole tensor (or per-row) | 1×128 activation tiles, 128×128 weight blocks |
| Format | E4M3 / E5M2 split | E4M3 everywhere |
| Accumulation | tensor-core registers | FP32 promotion every 128 elements |
| Sensitive ops | often also cast down | gating / norm / master state kept FP32 |
| Result | slow loss drift / divergence | < 0.25% loss gap vs BF16 |

A word on *why this is hard to get right without the report*: each of these fixes (fine-grained scaling, FP32 accumulation promotion, selective precision) addresses a different failure mode, and you'd discover them in the worst possible way — as a slow loss drift two trillion tokens into a run, with no obvious cause. DeepSeek paid that debugging cost so you don't have to. That is the real value of a technical report that publishes the recipe and not just the weights.

### The memory side of mixed precision

FP8 also has a memory story that's easy to miss. The master weights and the Adam moments live in FP32, which is expensive — 12 bytes per parameter for the optimizer state alone. To keep that affordable at 671B parameters, DeepSeek leans on the fact that only 37B are active per token and on careful sharding (and, per the report, keeps an exponential moving average of the weights on CPU memory, updated asynchronously, so the EMA used for evaluation doesn't consume precious GPU memory or stall the training step). The FP8 activations, meanwhile, roughly halve the activation memory versus BF16, which directly reduces how much you must recompute during backprop. Low precision isn't only a FLOPs play; it's a memory-bandwidth and memory-capacity play, and at this scale capacity is often the binding constraint.

## 4. The cross-cutting concern: stability is the product

It's tempting to treat FP8, MTP, and balancing as independent optimizations. They aren't — they interact, and the thing they jointly produce is **stability at scale**, which is the actual deliverable. Consider the failure modes they each prevent and how they compound:

- **MoE routing collapse** (a few hot experts, the rest dead) is prevented by loss-free balancing — *without* the quality tax that an auxiliary loss would impose, which matters more at 256 experts than at 8.
- **FP8 loss drift** is prevented by fine-grained scaling and FP32 accumulation — and it has to be, because at 671B params the GEMMs are where the FLOPs and the bandwidth go; you can't afford to run them in BF16 and still hit the cost target.
- **Decode bandwidth blowup** is prevented by MLA's compressed KV cache (see the [MLA post](/blog/machine-learning/large-language-model/kv-cache)) and by MTP's speculative drafting — together they make a 671B model servable.
- **Pipeline bubbles** that would waste the cluster are prevented by DualPipe's bidirectional schedule and the warp-specialized all-to-all (the subject of the [open-infra post](/blog/machine-learning/mlops/) in this series).

A word on DualPipe, since it's the system-side counterpart to everything above. A pipeline-parallel run splits the model across stages and feeds micro-batches through; the gap between stages finishing and the next micro-batch arriving is the "bubble," and it's pure waste. DualPipe feeds micro-batches from *both ends of the pipeline simultaneously* and overlaps each chunk's computation with the all-to-all communication of its neighbor, shrinking the bubble to near zero — at the cost of keeping a second copy of the parameters at the pipeline ends. On a comms-throttled H800 cluster, that trade (more memory for less waiting) is exactly the right one, and it's only affordable because MLA and FP8 freed up the memory to spend. Notably, DeepSeek avoids tensor parallelism entirely, because TP's all-reduce traffic would be murder on the throttled interconnect — another instance of the hardware shaping the design.

The report's proudest sentence is not a benchmark score; it's that across the entire 14.8T-token run there were **no irrecoverable loss spikes and no rollbacks**. For anyone who has run training at this scale, that is the headline. A single rollback on a 2,048-GPU cluster can erase a week and tens of thousands of dollars: you stop the run, roll back to the last good checkpoint, possibly skip or re-shuffle the data batch that triggered the spike, and restart — and if the spike recurs you're now debugging a numerical instability across two thousand GPUs in production. The co-design isn't just about being cheap per token — it's about being *predictable*, because predictability is what lets you commit a two-month run to a fixed budget and actually hit it. Every technique in this post contributes to predictability: loss-free balancing removes a tuning knob that could destabilize, FP32-promoted FP8 removes a slow drift, and the conservative precision map removes the discrete-error landmines.

There is also an honesty cross-current worth stating plainly. The $5.576M is the cost of the **final successful run**. It explicitly **excludes** the ablations, the failed experiments, the architecture search, and all the prior research (DeepSeekMoE, MLA in V2, the GRPO line) that made the final recipe possible. This is not a sleight of hand — the report says so — but it does mean the honest way to read the number is "the marginal cost of the final run, given everything DeepSeek had already learned," not "the cost to reproduce DeepSeek-V3 from scratch." The recipe is cheap *because the research was already paid for*.

## 5. What it cost, and where

**Senior rule of thumb: when a cost number surprises you, decompose it before you believe or disbelieve it.**

![Where the 2.788M H800-hours went: pre-training is about 95.5% of the total, with context extension and post-training together under 5%.](/imgs/blogs/deepseek-v3-fp8-mtp-loss-free-balancing-8.webp)

The breakdown is stark and worth internalizing. Of the **2.788M total H800 GPU-hours**, **pre-training is 2,664K hours (95.5%)**, **context extension is 119K hours (~4.3%)**, and **post-training (SFT + RL) is just 5K hours (~0.2%)**. At an assumed rental price of **$2 per H800 GPU-hour**, that's **$5.576M total** — of which roughly $5.09M is the pretraining run itself. The post-training that gives the model its instruction-following and reasoning polish is, in compute terms, a rounding error on top of the pretraining cost.

Two things fall out of this decomposition. First, **pretraining dominates so completely that every efficiency technique in this post is really a bet on the 95.5%.** FP8 cuts the cost of the GEMMs that dominate pretraining; MTP and MoE sparsity reduce the FLOPs and the wall-clock of that same 95.5%. Optimizing post-training would be optimizing the 0.2% — irrelevant to the headline. Second, the **context-extension cost (4.3%)** is a reminder that long context isn't free: stretching from 4K to 128K via YaRN took real compute, even though it's dwarfed by base pretraining. The arithmetic to commit to memory — the first line is code, so nothing trips a heading check:

```python
gpu_hours      = 2_664_000 + 119_000 + 5_000        # 2,788,000 H800-hours total
price_per_hour = 2.00                                # assumed $ per H800-hour
total_cost     = gpu_hours * price_per_hour          # $5,576,000
tokens         = 14.8e12                             # 14.8T training tokens
hours_per_T    = 2_664_000 / 14.8e12 * 1e12          # ~180K H800-hours / trillion
days_pretrain  = 2_664_000 / 2048 / 24               # ~54.2 days on 2,048 H800s
```

That `~180K H800-hours per trillion tokens` is the number to carry around as a planning constant. It says: if you have a DeepSeek-V3-class recipe and 2,048 H800s, a trillion tokens costs you about 3.7 days and ~$360K. It is the unit economics of frontier pretraining, made legible. Put differently, the marginal cost of a token is about **$0.38 per million training tokens** of H800 time — a number you can multiply by your own token budget to sanity-check a training plan.

For context on why this landed as a shock: contemporaneous frontier dense models were widely *rumored* to cost tens to over a hundred million dollars to train (the exact figures are unpublished, so treat them as order-of-magnitude folklore, not fact). Even granting generous error bars, a frontier-quality model at single-digit millions was a step-change — and it was achieved not by access to cheaper or faster chips (the H800 is *slower* at communication than the H100) but by squeezing more useful work out of throttled ones. That's the part worth internalizing: the cost win came from engineering against a hardware constraint, not from a hardware advantage.

## 6. The data, the schedule, and the long-context stretch

**Senior rule of thumb: the architecture gets the headlines, but the data mix and the schedule decide whether the run converges.**

It would be a mistake to leave this post thinking DeepSeek-V3 is "all architecture." The 95.5% of cost that is pretraining is spent pushing **14.8 trillion tokens** through the model, and the composition and ordering of those tokens matter as much as any kernel. Relative to V2, the V3 corpus deliberately shifts toward a higher proportion of **mathematics and programming** data and broader **multilingual** coverage — the data-level bet behind V3's strong code and reasoning numbers. Documents are packed to fill the 4K pretraining sequence length so that almost no compute is wasted on padding; the packing is a small data-engineering detail with an outsized effect on effective throughput, because every padded token is a token you paid to compute and learned nothing from.

The **schedule** is the other half. DeepSeek ramps the **batch size from 3,072 up to 15,360 sequences over roughly the first 469B tokens**, then holds it. A batch-size warmup is the counterpart to a learning-rate warmup: early in training the gradient is noisy and a giant batch wastes samples averaging away noise the model could have learned from; later, a large batch improves hardware utilization and gradient quality. The learning rate follows a warm-up-hold-decay shape with a **maximum of 2.2×10⁻⁴**, warming up over the first couple thousand steps, holding, then decaying in stages toward the end — the same "establish, then settle" pattern we saw with the γ anneal in balancing. The recurring theme across V3's hyperparameters is that *almost everything is annealed*: the balancing bias speed, the MTP loss weight λ, the batch size, the learning rate. Each is aggressive early to move fast and conservative late to land cleanly.

The **long-context extension** is where the 4.3% context-extension slice goes, and it's a reusable pattern in its own right. The model is *pretrained at a short 4K context* — cheap, because attention cost grows with sequence length — and only afterward stretched to long context in **two short YaRN stages: first to 32K, then to 128K**. Each stage is a brief additional training phase that adapts the rotary position handling to the longer range, not a full retraining. The economics are the entire justification: training at 128K from the start would have multiplied the attention cost across all 14.8T tokens, whereas extending afterward confines the long-context cost to a small tail. This is why "context length" and "training cost" are only loosely coupled in modern recipes — you buy the capability late and cheaply, rather than paying for it on every token. It pairs naturally with MLA, whose compressed KV cache is what makes the extended 128K context affordable to *serve*, not just to train.

## How DeepSeek-V3's choices differ from the rest of the field

It sharpens the lessons to put each technique next to the more common alternative, because in every case DeepSeek took the harder-to-implement but more-robust road.

- **Load balancing.** The Switch Transformer and Mixtral lineage balance with an **auxiliary loss** and live with the quality tax. DeepSeek replaced it with a **bias controller** — a more unusual design, but it removes both the tax and the coefficient sweep. The cost is that "balancing" is now a piece of imperative code in your training loop rather than a term in your loss, which some frameworks make awkward.
- **FP8.** The common path (Transformer Engine) is **per-tensor delayed scaling** with the standard E4M3/E5M2 split. DeepSeek chose **online per-tile/block scaling, E4M3 everywhere, with FP32 promotion** — more accurate and more stable, but it required hand-written kernels. Most teams can't or shouldn't do this; it only pays off at the scale where the GEMMs dominate the bill.
- **Speculative drafting.** Standard speculative decoding uses a **separate small draft model**; Medusa attaches **multiple parallel heads**; EAGLE trains a **small autoregressive head on the target's features**. MTP is closest to EAGLE in spirit — a feature-conditioned drafting head — but it is trained **jointly as a pretraining objective**, so there is no separate alignment or distillation phase and the draft distribution is consistent with the target by construction. You get the drafter as a byproduct of an objective you wanted for quality anyway.
- **Parallelism.** Where many large-model runs lean on **tensor parallelism**, DeepSeek **avoids TP entirely** and leans on expert parallelism plus the DualPipe schedule, because TP's all-reduce traffic is exactly the thing the throttled H800 interconnect punishes. The right parallelism strategy is a function of your network, not a universal default.

The through-line is that DeepSeek consistently traded *implementation effort* for *stability and efficiency*. That trade is only correct at frontier scale, where a stability win is worth millions and the engineering is amortized over a giant run. At smaller scale the stock defaults (auxiliary loss, BF16, a stock framework) are the right call precisely because the engineering wouldn't pay back. Knowing which regime you're in is the actual skill.

## Worked examples and counterfactuals

The techniques above are easy to nod along to and hard to reason about quantitatively. Here are ten concrete walkthroughs — some numeric, some counterfactual — to pressure-test your understanding. Each is the kind of thing an interviewer at staff level will probe.

### 1. One balancing step

Suppose a MoE layer with 8 routed experts (toy size) processes a batch where the per-expert token counts come out `[300, 290, 180, 60, 70, 50, 40, 10]`. The mean is 125. Experts 1, 2, and 3 are above mean; the rest are below. After this step, the controller does `b₁ -= 0.001`, `b₂ -= 0.001`, `b₃ -= 0.001`, and `b₄…b₈ += 0.001` each. On the *next* step, expert 8's slightly higher bias means a handful of borderline tokens — ones whose affinity for expert 8 was just below their affinity for expert 1 — now flip to expert 8. No gradient moved; only the *selection boundary* shifted. Over hundreds of steps, the histogram flattens. The key insight: the bias is an **integrator**, so persistent imbalance accumulates a persistent correction, while transient imbalance washes out.

### 2. The auxiliary-loss counterfactual

Now suppose the same layer were balanced the old way, with an auxiliary loss of coefficient α = 0.01. Expert 1 is genuinely the best expert for, say, Python f-strings — it *should* get those tokens. The auxiliary loss doesn't know that; it sees expert 1 as "overused" and adds a gradient that pushes the router's centroid for expert 1 *away* from f-string-like inputs, and simultaneously nudges expert 1's own weights to be less specialized so the penalty eases. You've just made your best expert slightly worse at its specialty to satisfy a balancing term. Multiply across 58 MoE layers and 14.8T tokens and the cumulative despecialization is exactly the "quality tax" the loss-free scheme avoids. The counterfactual makes the cost visible: every token of imbalance the auxiliary loss "fixes" is paid for in gradient interference; the control loop fixes the same imbalance for free.

### 3. The straggler tax

Take the same imbalanced batch on a real expert-parallel layout where experts are spread one-per-GPU across a group. Expert 1 received 300 tokens; expert 8 received 10. Every GPU in the group must finish its expert's matmul before the combine all-to-all can start, so the step is gated by the GPU holding expert 1 — it does 30× the work of the GPU holding expert 8, while 7 GPUs idle. The throughput cost of imbalance is not the *average* over-allocation; it's the *maximum*. This is why node-limited routing (M = 4) and the bias controller both target the *peak*, not the mean — and why a 10% average imbalance can still produce a 30% throughput loss if it's concentrated.

### 4. One MTP training step

Take a sequence `the cat sat on the mat`. The main objective at position "sat" predicts "on." The MTP module at the same position predicts *two ahead*: "the." To do that, its input is the main trunk's hidden state at "sat" (which has seen "the cat sat") concatenated with the embedding of the token at *i*+2 ("the"), pushed through one extra block. The combined loss is `L_main(predict "on") + 0.3 · L_mtp(predict "the")`. Notice that the MTP target "the" is genuinely harder from position "sat" — it requires modeling that "on the" is a likely continuation, not just the immediate next token. That difficulty is the point: it forces the hidden state to carry more forward-looking information than vanilla next-token prediction would extract.

### 5. The acceptance arithmetic

Why does an 85–90% acceptance rate yield ~1.8× and not 1.9× throughput? Model it simply. With acceptance probability *p*, each verification step emits `1 + p` tokens on average (the guaranteed main token plus the draft when accepted), but the draft itself costs a small fraction *c* of a full step. Effective speedup ≈ `(1 + p) / (1 + c)`. With *p* = 0.88 and a draft costing roughly *c* ≈ 0.05 of a step (it's one extra block plus the verification position that mostly rides the existing matmul), you get `1.88 / 1.05 ≈ 1.79×` — right at the reported ~1.8×. The lesson: speculative speedup is bounded by acceptance, and acceptance is bounded by how well the drafter matches the target — which is exactly why training the drafter *inside* the target (MTP) beats bolting on a mismatched small model.

### 6. One FP8 GEMM tile

Take a 1×128 activation tile whose values are mostly around ±0.5 but with one rogue value at 48.0 (a real thing that happens after certain nonlinearities). E4M3's max representable is ~448. **Tensor-wide scaling** across a 7168-wide activation would set a scale driven by the global max across all 56 tiles; if another tile holds a 400, the scale crushes our ±0.5 values to near-zero in FP8. **Per-tile scaling** sets this tile's scale from *its own* max (48.0), mapping 48→near-448 and keeping the ±0.5 values at ~±4.7 in the scaled domain — comfortably above the quantization floor. Then the matmul accumulates, spilling to FP32 every 128 elements so the long reduction doesn't drift. Same data, two scaling scopes, completely different amount of information retained. The outlier was contained to one tile instead of poisoning the row.

### 7. The 0.25% check

How do you even know FP8 is safe? You run a shadow comparison: a short BF16 run alongside the FP8 run on identical data and seed, and you watch the *relative* difference in training loss. DeepSeek reports it stays **under 0.25%**. The discipline here is that 0.25% is small enough to be in the noise of run-to-run variance, so you can trust that the FP8 model is the same model the BF16 run would have produced — not "close," but statistically indistinguishable. If that gap had been 2%, the honest move would be to back off FP8 on the offending operator. The lesson: low precision is an empirical claim about *your* model and data, and the way you earn the right to use it is by measuring the divergence, not by assuming it.

### 8. The cost arithmetic, defended

A skeptic says "$5.6M is fake, you can't train a frontier model that cheap." Decompose before you argue. 2,048 H800s × ~54 days × 24 hours ≈ 2.66M GPU-hours for pretraining — that's a real, schedulable cluster occupancy, not a fantasy. At rental prices around $2/hour (plausible for H800 capacity in 2024), that's ~$5.3M for pretraining. The number is *internally consistent*. What's misleading is using it as "the cost to build DeepSeek-V3," because it excludes the R&D that produced the recipe. Both the believers ("it's only $5.6M!") and the deniers ("that's impossible!") are wrong in the same way: they're treating a marginal-run cost as a total-program cost. The arithmetic is sound; the framing is where people go astray.

### 9. Why the batch size ramps instead of starting large

A team copying the recipe sets batch size to the final 15,360 from step zero, reasoning "bigger batch, better hardware utilization." Their early loss is worse than DeepSeek's, and they can't see why. The reason is sample efficiency: at the start of training the gradient signal is large and informative, and a giant batch *averages away* useful variance — you spend 15,360 sequences to take one step that a 3,072-sequence batch would have taken almost as well, so you've burned 5× the data for the same early progress. DeepSeek ramps from 3,072 to 15,360 over the first ~469B tokens precisely to avoid this: small batches while the gradient is rich, large batches once the gradient has quieted down and utilization matters more than per-step efficiency. The counterfactual makes the principle concrete — batch size, like learning rate, is something you grow into, not something you fix at its final value on day one.

### 10. The long-context economics

Suppose you want a 128K-context model and you train it at 128K from scratch over 14.8T tokens. Attention cost scales with sequence length, so relative to a 4K run you've multiplied the attention portion of every one of those 14.8T tokens by a large factor — the run might cost several times more. Now do it DeepSeek's way: pretrain at 4K (cheap attention), then spend two short YaRN stages extending to 32K and then 128K. Those stages touch a tiny fraction of the total tokens, which is why context extension is only ~4.3% of the GPU-hours. The capability is identical at serving time, but the cost is confined to a small tail instead of smeared across the whole run. The lesson generalizes: when a capability's cost scales with a dimension (here, sequence length), buy it in a short dedicated phase rather than paying for it on every training token.

## When to copy DeepSeek-V3's playbook — and when not to

These techniques are powerful but not universal. Most teams are not training 671B MoE models, and applying frontier-scale tricks to a 7B dense fine-tune is cargo-culting.

**Reach for this playbook when:**

- You're training a **large sparse MoE** (dozens to hundreds of experts), where balancing is a first-order concern and the auxiliary-loss quality tax is real. Loss-free balancing is close to a free win here.
- Your run is **GEMM-bound and you're paying for the FLOPs** — i.e., big enough that the three matmuls dominate cost. FP8 with fine-grained scaling is worth the engineering only when the GEMMs are the bill.
- You **control the training stack down to the kernel** (or can use DeepSeek's open DeepGEMM / FlashMLA kernels), so you can actually implement per-tile scaling and FP32 accumulation promotion. Off-the-shelf FP8 that scales per-tensor will not give you the same stability.
- You want **speculative decoding without a separate draft model** and can afford a few percent extra parameters during pretraining. MTP pays for itself twice.
- You need **predictable, spike-free runs** on a fixed budget — the stability engineering is as valuable as the speed.
- Your interconnect is the bottleneck (throttled NVLink, slow InfiniBand). The node-limited routing and DualPipe overlap are designed for exactly that constraint.

**Skip it when:**

- You're **fine-tuning, not pretraining.** None of this touches the 0.2% post-training slice; for SFT/DPO/GRPO work, look at [GRPO vs DPO vs PPO](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) instead.
- Your model is **dense and small** (≤ a few B params). Loss-free balancing is moot (no experts), and FP8's complexity rarely pays off below the scale where GEMMs dominate — BF16 is fine and far less error-prone.
- You **don't own the kernels.** Naive per-tensor FP8 without the accumulation fix will drift, and you'll spend more time debugging loss divergence than you saved. Use BF16 until you can do FP8 properly.
- You're **latency-bound on tiny batches** at inference, where MTP's speculative gain shrinks (verification overhead dominates when there's little batched matmul to ride on).
- **Reproducibility and auditability** matter more than peak efficiency — every low-precision and heuristic-controller trick adds a knob that can behave subtly differently across hardware and driver versions.

## The reusable principles, distilled

Strip away the 671B-specific numbers and a handful of portable principles remain — the part of this report you can apply whether you're training a 3B model or a frontier MoE.

**Move regularizers out of the gradient when you can.** The single most generalizable idea here is the bias controller. Any time you find yourself adding a term to your loss whose job is to enforce a *constraint* rather than to improve the *prediction* — load balance, a length penalty, a diversity term — ask whether a small control loop on a non-gradient quantity could enforce it instead. If it can, you stop taxing the objective you actually care about. This pattern is everywhere once you look for it.

**Engineer for the tails, not the average.** Fine-grained FP8 scaling exists because one outlier per tensor ruins the average case. The straggler tax exists because the slowest expert, not the mean expert, sets the step time. Low precision, load balancing, and latency all fail at their tails — so design the mechanism (tile granularity, peak-limited routing) around the worst element, not the typical one. Averages lie about systems with barriers and outliers.

**Make an auxiliary objective do double duty.** MTP is a pretraining signal *and* a speculative drafter. The lesson is to look for objectives that pay off in more than one place before you add them, because the second payoff is often what tips a marginal technique into a clear win. A draft head that only helped inference, or only helped training, might not have been worth it; one that helps both is an easy yes.

**Anneal almost everything.** The balancing bias speed, the MTP loss weight, the batch size, the learning rate — all aggressive early and conservative late. "Establish a regime, then settle into it" is a meta-pattern that applies to nearly every schedulable quantity in training. When you introduce a new knob, ask what its early value and its late value should be, not just its value.

**Co-design against your real bottleneck.** On H800s the bottleneck was the network, so node-limited routing, DualPipe overlap, and the choice to skip tensor parallelism all target communication. The portable move is not "copy those three techniques" — it's "identify what is actually scarce in your setup and bend the architecture and schedule toward relieving it." For you that scarce resource might be memory, or single-stream latency, or labeled data; the discipline is the same.

**Buy expensive capabilities in dedicated phases.** Long context is extended late and cheaply rather than paid for on every token. Whenever a capability's cost scales with a dimension you don't need during the bulk of training, isolate it into a short phase. The same logic applies to high-resolution vision, tool-use traces, or any data that's expensive per token but only needed to unlock a specific skill.

**Earn the right to low precision by measuring it.** The 0.25% shadow-comparison is the discipline that makes FP8 trustworthy. Never assume a precision reduction is safe; run the control and look at the divergence. And match precision to the *discreteness* of the decision — graceful, continuous computations tolerate few bits; discrete decisions like routing argmaxes do not.

None of these requires a 2,048-GPU cluster to apply. They're habits of thought about where to spend bits, where to spend compute, and where to spend gradient — and they scale down to any training run you'll actually do.

## Further reading

- **DeepSeek-V3 Technical Report** — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437). The primary source; sections on FP8 training, MTP, and load balancing are the densest.
- **DeepSeekMoE** — [arXiv:2401.06066](https://arxiv.org/abs/2401.06066). Where the fine-grained-expert + shared-expert architecture that V3 scales up was introduced (a later post in this series).
- **DeepSeek-V2** — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434). The origin of MLA, carried into V3 unchanged.
- This blog: [Multi-head Latent Attention & the KV cache](/blog/machine-learning/large-language-model/kv-cache) · [Optimizing MoE training and inference](/blog/machine-learning/large-language-model/optimizing-moe-training-and-inference) · [Speculative decoding](/blog/machine-learning/large-language-model/speculative-decoding) · [Modern LLM architectures: Qwen, Llama, Gemma, DeepSeek](/blog/machine-learning/large-language-model/modern-llm-architectures-qwen-llama-gemma-deepseek).

The single most transferable idea in the whole report is the one that's easiest to overlook: **balance, precision, and speed are not features you add — they're constraints you co-design against.** DeepSeek didn't make training cheap by finding a cheaper algorithm. They made it cheap by ensuring that every layer of the stack — from the bias term on a routing score to the FP32 promotion interval on a tensor core — was chosen so the layer above it could afford to exist. That's the lesson worth stealing, whatever scale you train at. The next post in this series follows the same recipe down one more level, into the open-sourced kernels — DeepEP, DeepGEMM, FlashMLA, and DualPipe — that turn these paper claims into shipping code, and shows how each one maps, almost line for line, onto a claim made in the technical report we just read.

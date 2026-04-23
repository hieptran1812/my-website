---
title: "Mixture of Experts (MoE) LLMs: Architecture, Training, Fine-tuning, and Case Studies"
publishDate: "2026-04-19"
category: "machine-learning"
subcategory: "Large Language Model"
tags:
  [
    "MoE",
    "mixture-of-experts",
    "sparse-models",
    "LLM",
    "scaling",
    "training",
    "fine-tuning",
    "DeepSeek",
    "Mixtral",
    "Switch-Transformer",
    "routing",
    "load-balancing",
    "AI",
  ]
date: "2026-04-19"
author: "Hiep Tran"
featured: true
aiGenerated: true
excerpt: "A deep, practical tour of Mixture-of-Experts LLMs — how the architecture works, why sparse activation unlocks trillion-parameter scale, what every router, expert, gate, and load-balancing loss is actually doing, how teams pre-train and fine-tune them, and what happened in production at Google (Switch/GShard/GLaM), Mistral (Mixtral), DeepSeek (V2/V3), Qwen, xAI, and Databricks (DBRX)."
---

# Mixture of Experts (MoE) LLMs: Architecture, Training, Fine-tuning, and Case Studies

For most of the deep-learning decade, scaling a transformer meant one thing: make every parameter fire on every token. Double the parameters, double the FLOPs, double the bill. **Mixture of Experts (MoE)** breaks that link. A modern MoE LLM can have a trillion parameters but only activate thirty billion of them per token, giving you the same inference cost as a dense 30B model with the representational capacity of something several times larger.

That single idea, *sparse activation conditional on the input*, is why DeepSeek-V3 could train a 671-billion-parameter model for a fraction of what a dense equivalent would cost, and why Mixtral 8x22B punches well above its serving weight.

This article is a long, practical, end-to-end guide. We will look at what MoE actually is at the math level, how the router decides where tokens go, every loss term you will meet in a paper, the parallelism strategies that make training feasible, the ways it breaks, how teams fine-tune MoE models in the wild, what changes when you try to serve one, and how production systems (GShard, Switch Transformer, GLaM, ST-MoE, Mixtral, DeepSeek V2/V3, Qwen-MoE, Grok-1, DBRX, Arctic) actually ship.

## 1. What is a Mixture of Experts?

A **Mixture of Experts** layer replaces a single feed-forward network (FFN) inside a transformer block with a collection of FFNs, called the **experts**, and a lightweight **router** that decides, per token, which experts get to process it.

The picture looks roughly like this.

```
        ┌──────────────────────────────────┐
        │            Router (gate)         │
        │     softmax(x · W_router) → top-k│
        └────────────┬─────────────────────┘
                     │ selects e.g. 2 of 8
        ┌──────┬─────┴─────┬──────┬──────┐
        ▼      ▼           ▼      ▼      ▼
     Expert₁ Expert₂ ... Expertₖ ...  Expertₙ
        │      │           │
        └──────┴─────┬─────┘
                     ▼
             weighted sum → y
```

Each expert is typically a standard transformer FFN, two linear projections with a SwiGLU or GELU nonlinearity in between. The router itself is almost nothing, usually a single linear projection `W_router ∈ R^{d_model × n_experts}` followed by a softmax and a top-k selection.

The crucial property is that **only k experts run for each token** (usually k equals 1 or 2), so the **active** parameter count per token is a small fraction of the **total** parameter count the model owns.

### 1.1 A worked mental model

Imagine a transformer as a factory line. In a dense model, every part passing down the line is touched by every station. In an MoE model, each part gets inspected by a dispatcher (the router), who decides "this looks like a code token, send it to the code-and-math stations" or "this is French prose, send it to the romance-languages station." Most stations are idle for most parts. But the factory as a whole has far more total capability than any dense factory with the same per-part cost.

That analogy, while loose, captures the three essential traits of MoE:

1. There are many parameters in total, but any single input only uses a few of them.
2. Which parameters get used depends on the input, and is learned.
3. The dispatcher itself is cheap compared to the work it delegates.

### 1.2 Dense versus sparse in the numbers

Trying to express the trade-off without a table, here is the shape of it.

A dense transformer's FFN at hidden size `d_model = 4096` and `d_ff = 14336` (Mistral-7B style) has roughly 117M parameters per layer and does 2 × d_model × d_ff FLOPs per token per layer. Every one of those parameters loads from HBM, every one contributes to compute.

An MoE transformer with eight such FFNs and top-2 routing (Mixtral 8x7B style) has roughly 940M parameters per layer but only executes ~234M of them per token. Memory bandwidth requirements for the weight load stay the same as a dense 7B at inference (the two routed experts have to be read), but total HBM occupancy grows 8x. The knowledge capacity has decoupled from the per-token compute.

If you keep pushing, DeepSeek-V3 has 256 routed experts plus 1 shared expert per layer, activates 9 total per token, and ends up at 671B total / 37B active. The total-to-active ratio is roughly 18x. The model owns the knowledge of something much larger than it computes.

The trade this creates:

- Compute per token is small and comparable to a dense model of the active size.
- HBM occupancy is huge and comparable to a dense model of the total size.
- Knowledge capacity tracks total size, not active size.
- Batching behavior is ragged and depends on the routing decisions, which introduces a whole family of scheduling problems that dense models never see.

The headline: **MoE decouples capacity from compute**. You pay compute for the small active subset and you pay memory for the whole model.

### 1.3 Where the MoE sublayer sits inside a transformer

In every modern MoE LLM I know of, only the **FFN sublayer** is expert-ified. Attention stays dense and shared across all tokens. A typical transformer block looks like

```
x → LN → Self-Attention → + → LN → MoE-FFN → + → x'
```

Some stacks keep a few layers dense (Switch-Transformer put MoE every other layer, DeepSeek keeps the first layer dense, Arctic uses a dense + MoE hybrid in every layer). Whether to interleave dense and MoE layers is a knob, and the current consensus leans toward "MoE everywhere except maybe the very first layer," because those early layers are doing generic tokenization-to-representation work that benefits less from specialization.

Why only the FFN? Two reasons. First, FFN weights are where the vast majority of a transformer's parameter count lives, so that is where the capacity leverage is. Second, attention has a structural role (mixing information across positions) that does not decompose neatly into "specialist" versus "generalist" behavior, while FFNs really do seem to implement key-value knowledge lookups that *do* specialize.

## 2. Why MoE actually works: the deeper scaling argument

Dense scaling laws say loss falls as a power law in compute, parameters, and data. MoE does not break those laws. It moves along a different axis of the space they describe.

Here is the intuition, spelled out more carefully than "not every parameter is needed for every token."

Think of a language model as a big associative memory. Most of its parameters encode specific facts, syntactic patterns, code idioms, rare multi-word expressions, and so on. When a token like "the" comes in, only a tiny slice of that memory is relevant (whatever tells the model what kinds of nouns follow determiners, roughly). When a token like `__init__` comes in, a completely different slice is relevant (Python class construction patterns). A dense model that processes every token through every parameter is, for most tokens, wasting most of its compute fetching and multiplying by irrelevant weights.

MoE reifies this observation. It divides the big memory into expert-shaped chunks and learns a dispatcher that picks the right chunks for each token.

The key empirical claim is this: for a fixed training compute budget, *there exists an allocation of that budget between data, active parameters, and total parameters* that produces a better model than spending the entire budget on dense scaling. Multiple papers (GShard, GLaM, ST-MoE, Switch scaling laws, DeepSeek-MoE scaling) have verified this.

Empirically, the numbers that matter:

- **GShard** (2020) trained a 600B MoE translation model with Switch-style routing and matched a dense model of equivalent FLOPs across 100 language pairs.
- **GLaM** (2021) matched GPT-3 175B dense quality with roughly one third of the training compute and about one half of the inference compute.
- **Mixtral 8x7B** (2023), 47B total with roughly 13B active, matched or beat Llama-2 70B on most benchmarks at roughly one fifth of the inference cost.
- **DeepSeek-V3** (2024), 671B total with 37B active, trained on about 14.8 trillion tokens for an H800 bill that came to under $6M and landed near GPT-4 on many evals.

The recurring finding: **for a fixed training or inference budget, sparse beats dense**, provided you can actually train it. That "provided" is where almost every interesting engineering problem lives.

### 2.1 Why the theoretical wins are hard to collect

Naively, you might expect MoE with N experts and top-1 routing to give you an N-x capacity multiplier at constant compute. In practice the multiplier is much smaller, usually somewhere between 2x and 8x. What eats the rest?

First, load imbalance. If the router sends 40% of tokens to one expert and 0.1% to another, most of your capacity is wasted.

Second, router overhead. The router itself takes FLOPs and adds latency, especially for very large N.

Third, all-to-all communication. Sending tokens to the device holding their expert and getting the result back is expensive, and that cost scales with N and with the number of devices.

Fourth, expert co-adaptation. Without regularization, experts often end up learning nearly identical functions (because gradients push them toward good average solutions), which means you get no specialization win.

Fifth, the quality ceiling of the router. The router sees only a single token's hidden state. If two tokens would benefit from completely different experts but have similar hidden states, the router cannot tell them apart.

The history of MoE engineering is, in large part, a history of progressively reducing each of these five taxes.

## 3. Anatomy of a modern MoE layer, piece by piece

![Anatomy of an MoE layer: router produces softmax over experts, top-K selected (usually K=2), chosen expert FFNs run in parallel, outputs combined by router weights, auxiliary load-balance + z-loss keep routing healthy](/imgs/blogs/moe-01-layer.png)

Let us dissect one MoE FFN layer with the choices you would make today (roughly DeepSeek/Mixtral/Qwen flavored), and look closely at what each knob does.

### 3.1 The router

Given a token representation `x ∈ R^d`, the canonical router does

```
logits      = x · W_router                # shape: [n_experts]
p           = softmax(logits)             # gating probabilities over experts
top_k_ids   = argtop_k(p, k)              # indices of best k
gate_values = normalize(p[top_k_ids])     # renormalize so they sum to 1
```

The design choices inside these four lines have a surprising amount of impact.

**How many experts per token (k).** Switch chose k=1 and argued smaller is better (cheaper, simpler, empirically no worse). GShard and Mixtral chose k=2, arguing that using two experts gives smoother gradients (the router output is a mixture, so small changes in `p` do not discontinuously swap experts) and better expert collaboration. DeepSeek-V3 activates 8 routed experts out of 256, together with 1 shared expert. The current consensus at the frontier is "fine-grained with large k" (many small experts, many of them active), because the combinatorial specialization space is enormous while the FLOPs stay fixed.

**Softmax then top-k, or top-k then softmax.** In the original GShard, softmax runs over all experts and then the top-k are selected, meaning the gate values you multiply by might be tiny if the top-k barely won. Most modern models do top-k first and then renormalize (either a fresh softmax over the k selected logits or a simple division), which gives the selected experts the full gating weight.

**Sigmoid versus softmax gating.** DeepSeek-V3 switched from softmax to independent sigmoid gates per expert. This matters at very large N because the softmax denominator introduces coupling across all experts (changing one logit changes all gate values), which interferes with balance control.

**Noisy gating.** Early work (Shazeer 2017) added Gaussian noise to the router logits during training to encourage exploration and prevent premature expert collapse. Most modern models do not do this, relying instead on initialization and auxiliary losses.

**Router precision.** The router is tiny, and keeping it in FP32 while the rest of the model is in BF16 or FP8 is cheap insurance against routing instability. Virtually every successful MoE paper reports doing this.

### 3.2 The experts themselves

Each expert is usually a SwiGLU FFN,

```
FFN_i(x) = (swish(x · W1_i) ⊙ (x · W3_i)) · W2_i
```

with `W1_i, W3_i ∈ R^{d × d_ff}` and `W2_i ∈ R^{d_ff × d}`. The nonlinearity and the overall shape are identical to a dense transformer's FFN. The two things that vary across architectures are the *inner dimension* `d_ff` per expert and the *number* of experts.

Older designs (Switch, GShard) used a few big experts, with `d_ff` close to what a dense model of the same scale would use. Newer designs (DeepSeek-MoE, Qwen-MoE) use many small experts. Specifically, DeepSeek halved each expert's width and doubled the number of experts, keeping total FLOPs and parameters roughly constant. The win is combinatorial: with 8 big experts and top-2 you get `C(8,2) = 28` possible expert-combinations per token, while with 64 small experts and top-16 you get `C(64,16) ≈ 10^14`. Even if most of those combinations are never learned, the extra capacity for fine-grained specialization is enormous, and empirically specialization quality improves sharply.

### 3.3 Combining expert outputs

For token `x` routed to experts `{i_1, …, i_k}` with gate weights `{g_1, …, g_k}`, the output is

```
y = Σ_j g_j · FFN_{i_j}(x)
```

Add the residual back and you are done. Note that because the gate weights sum to one (after renormalization), this is a convex combination of expert outputs, which keeps the output magnitude stable across different routing decisions.

### 3.4 Shared experts (the DeepSeek innovation that stuck)

DeepSeek-MoE introduced a refinement that every subsequent frontier MoE has adopted in some form: **shared experts**, a small number of experts (often just one or two) that *every* token passes through unconditionally, in addition to the top-k routed ones.

```
y = FFN_shared(x) + Σ_j g_j · FFN_{i_j}(x)
```

Why this matters deserves more than one sentence. Without shared experts, the routed experts are being asked to simultaneously handle both common knowledge (grammar, syntax, extremely frequent patterns) and specialized knowledge (a particular code idiom, a particular language, a particular topic). Because common tokens dominate any corpus, gradient signals push routed experts toward handling the common case well, which flattens specialization. Shared experts absorb that common-case burden. The routed experts can then focus their capacity on the long tail. Empirically, this dramatically improves the specialization quality you can extract from routed experts (you can see it by probing which tokens each routed expert fires on, before and after adding shared experts, and the "after" distributions are visibly more peaked and interpretable).

DeepSeek-V2 used 2 shared and 160 routed experts. DeepSeek-V3 uses 1 shared and 256 routed. Qwen-MoE and DBRX adopted the pattern as well.

### 3.5 Fine-grained experts, again, from a different angle

We mentioned fine-grained experts above, but it is worth dwelling on the argument. If you have 8 experts of width `d_ff` and you replace them with 64 experts of width `d_ff / 8`, the total FFN parameter count is the same, and if you activate 8 of the small experts instead of 1 of the big ones, per-token FLOPs are also the same. What changes?

Three things. The combinatorial specialization space grows by many orders of magnitude, as discussed. The router has more granularity to work with: it can express "this token is mostly like Python but also like math" by activating a Python-flavored expert and a math-flavored expert, rather than having to pick one big expert that happens to know both. And the gradient signal per expert is smaller per step (each expert sees fewer tokens), which can improve diversity but also means you need a lot of data to train each expert well.

That last point is why fine-grained MoE only really paid off once training corpora got big enough, around the 10T-token mark, to give each of 64+ experts enough examples. Trying to train a 256-expert model on 100B tokens would almost certainly underfit most experts.

## 4. The routing problem: what goes wrong and how to fix it

![MoE routing failure modes and fixes: expert collapse (load-balance aux loss, noisy routing), token dropping (capacity c>1, shared expert overflow), expert underuse (z-loss, router jitter, dropless training)](/imgs/blogs/moe-02-routing.png)

Routing is the soul of MoE and also its biggest failure mode. Four things break if you are not careful, and each has a canonical fix.

### 4.1 Load imbalance (the expert collapse problem)

The router is gradient-trained along with everything else. If expert 3 is slightly better than the others at the start, the router learns to send more tokens to expert 3. More tokens mean more gradient signal, which makes expert 3 even better. Positive feedback loop. End state: one or two experts absorb almost all traffic and the rest are dead weight.

This is the canonical MoE failure. Without intervention you will routinely throw training compute at a 64-expert model and discover that 58 of those experts have never seen a meaningful gradient.

The canonical fix is a **load-balancing auxiliary loss**, introduced in Shazeer 2017 and refined in Switch/GShard:

```
L_balance = n_experts · Σ_i f_i · P_i
```

where `f_i` is the fraction of tokens routed to expert i in the current batch (a hard count, divided by batch size) and `P_i` is the mean router probability for expert i across the batch (a soft quantity, so it is differentiable). Minimizing this product pushes the router toward uniform utilization, because the product is minimized when all `f_i` and `P_i` are equal to `1/n_experts`.

Key subtlety: `f_i` is not differentiable (it is a count), so the gradient flows only through `P_i`. But the product `f_i · P_i` gives a stronger push on experts that are currently overused (because their `f_i` is large) and a gentler push on underused ones. This works out empirically well.

The coefficient α on this loss is usually small, around 0.01. Too large and the router becomes uniform and stops learning useful routing. Too small and collapse can happen anyway.

### 4.2 Router z-loss (logit explosion)

Router logits can grow unboundedly during training because there is nothing in the architecture penalizing their magnitude. Large logits push the softmax toward extreme confidence (one gate at 0.999, the rest near zero), at which point the gradient through the softmax vanishes and the router stops updating.

ST-MoE introduced the **router z-loss**:

```
L_z = (1/B) Σ_b (logsumexp(logits_b))^2
```

This penalizes the log-partition-function of the router softmax, which keeps logits small and well-conditioned. Coefficient is around 1e-3. It is cheap, easy, and most modern recipes include it.

### 4.3 Capacity overflow (token dropping)

In a batched implementation, each expert has a **capacity**: a maximum number of tokens per batch it can process, set at graph-compilation time so buffers can be preallocated. Overflow tokens, the ones that tried to route to an expert whose capacity was already full, are **dropped**. Their output for that layer is zero, and the residual connection carries the input through unchanged.

Dropping is parameterized by a **capacity factor** C. Capacity factor 1.0 means an expert's capacity equals the average batch-tokens divided by number-of-experts, so any routing imbalance at all will cause drops. Capacity factor 1.25 gives 25% slack.

Too low a capacity factor means many drops, which hurts quality. Too high means wasted compute on padding that never gets used. Training typically uses C around 1.25–2.0.

Modern inference systems (vLLM, SGLang, TensorRT-LLM) avoid fixed capacity at serve time by letting experts process any number of tokens, with some throughput loss relative to a padded-optimal version. This eliminates dropping as a source of inference-time quality loss. Training usually still uses capacity factors, because the fixed shapes are important for compilation efficiency.

### 4.4 Device-level imbalance

A 64-expert model sharded across 8 GPUs puts 8 experts on each GPU. Even if expert-level load is balanced (each expert gets the same number of tokens), *device-level* load can be skewed if the tokens happen to cluster around experts that are all on the same GPU. This manifests as all-to-all communication stalls, where one GPU is receiving many more tokens than the others and becomes a straggler.

DeepSeek introduced a **device-level balance loss** to encourage balance across devices specifically, not just across experts. DeepSeek-V3 went further and introduced **auxiliary-loss-free load balancing**, which has turned out to work surprisingly well at scale. The idea: add a learned bias `bias_i` to each expert's routing logit, update that bias like a PID controller based on observed load, but do *not* pass gradients through it. Over the course of training, overloaded experts' biases get nudged down and underloaded experts' biases get nudged up, which redirects traffic without introducing a competing objective for the main loss.

```
bias_i ← bias_i - γ   if expert i overloaded in the last window
bias_i ← bias_i + γ   if expert i underloaded in the last window
```

At the end of training, the bias values capture the calibration needed to hit target load without any auxiliary gradient. DeepSeek reports that aux-loss-free balancing beats the Switch-style balance loss on held-out quality at scale, which fits a general pattern where "feedback control" approaches to training-dynamics regularization outperform "add a term to the loss" approaches.

## 5. Training MoE from scratch: the real engineering story

### 5.1 The parallelism zoo

Dense LLMs use data parallelism (replicate model, split batch), tensor parallelism (split model weights across devices within a layer), and pipeline parallelism (split layers across devices). MoE adds a fourth axis: **expert parallelism (EP)**, where different experts live on different devices and tokens are shuffled (via all-to-all) to the device that holds their routed expert, processed, and shuffled back.

A typical frontier recipe combines all four. DeepSeek-V3's training configuration, for example, is roughly:

- Data parallelism across replicas, for batch scaling.
- Tensor parallelism across a small group (commonly 8 GPUs), for attention and shared weights.
- Pipeline parallelism across another dimension, for splitting the deep stack of layers.
- Expert parallelism across yet another dimension, for spreading the 256 routed experts.
- Sequence parallelism overlaid on the rest, to reduce activation memory for long contexts.

The orchestration of this is non-trivial. You typically allocate a mesh of devices, say 16 × 8 × 8 × 2, and assign parallelism axes to mesh axes. ZeRO-3 style optimizer-state sharding sits on top for memory.

DeepSeek-V3 trained on 2,048 H800 GPUs and reported roughly 66% model-FLOPs utilization (MFU), which is higher than most dense training runs at similar scale. That they got such high MFU on an MoE — which has more communication than dense — is a testament to how carefully the schedule was tuned.

### 5.2 The all-to-all bottleneck

Every MoE layer performs *two* all-to-alls per forward pass. The first is the **dispatch**: take the batch-major tensor of tokens, figure out which expert each token needs, and scatter the tokens across devices so each device ends up holding the tokens for its experts. The second is the **combine**: after experts run, gather expert outputs back to the devices that owned the tokens.

These all-to-alls are bandwidth-bound and can dominate step time if you are not careful. Mitigations form their own body of lore.

**Overlap compute with communication.** While experts on a device process the current batch's tokens, begin dispatch for the next batch. Modern EP implementations pipeline the two as aggressively as possible.

**Topology-aware routing.** Prefer experts that live on the same node, or at least within the same NVLink domain. DeepSeek introduced **device-limited routing**, where each token is only allowed to route to experts on at most some small number of devices (3 in V3). This trades a tiny bit of routing freedom for large communication savings.

**Low-precision all-to-all.** The all-to-all payload is activations, which tolerate FP8 (and in some configurations FP4) without quality loss. DeepSeek-V3 ran FP8 end-to-end during training, including through the all-to-all buffers. This was novel at the scale they attempted it, and required careful design of FP8 scaling factors to avoid accumulation overflow.

**Hierarchical routing.** For very high expert counts, split experts into groups, route first to a group, then within the group. Reduces router cost and communication cost from O(N) to O(√N). Research area as of 2025.

### 5.3 Learning rate and initialization

Routers are hyperparameter-sensitive. A router that is randomly initialized with the same scale as the rest of the model tends to start with extreme gating preferences, which triggers expert collapse immediately.

Common tricks:

**Scale router init down.** Initialize `W_router` with standard deviation 1/√n_experts or even smaller. Near-uniform early routing lets every expert see gradient signal for the first few thousand steps, which is when specialization gets seeded.

**Warm up balance coefficients.** Start the balance loss coefficient low (or zero) and ramp it up over the first few thousand steps. This lets routing specialize first before balancing pressure kicks in.

**Router learning rate.** Some recipes apply a lower LR multiplier to router weights (often 0.1x) to stabilize them. ST-MoE used 1x and argued the scaled init was enough; most DeepSeek-family recipes do not separate LRs. Practice varies.

**Bias-based balancing as LR-free regularization.** DeepSeek-V3's auxiliary-loss-free approach sidesteps LR considerations for the balance signal entirely, because the bias update is not a gradient.

### 5.4 Data and curriculum

MoE benefits from **diverse data**, and the benefit grows with expert count. The reason is simple. Experts specialize only if the data has specialization-worthy structure. If your corpus is 95% homogeneous English web prose, a 64-expert model will not find much for most experts to specialize on, and many experts will end up redundant or underutilized.

DeepSeek-V3 trained on a 14.8-trillion-token mix carefully balanced across code, math, multilingual, and general web, with explicit oversampling of code and math. The paper reports visibly stronger specialization on code and math experts than on general-text experts (code experts fire predictably on code tokens and are mostly idle elsewhere, while general-text experts show more diffuse firing patterns). This observation generalizes: the more structured the subdomain, the sharper the expert specialization.

Curriculum matters too. Several teams report that starting with a shorter context length and annealing to the full context length late in training improves both speed and final quality. Some teams also anneal the data mixture itself, starting with a cleaner / more structured subset and adding web noise later. These are orthogonal to MoE per se but interact with it.

### 5.5 Upcycling: turning a dense model into an MoE

Training a large MoE from scratch is expensive. **Upcycling** is a cheap alternative: take a trained dense model and convert it to MoE without starting over.

The recipe is short:

1. Take a dense checkpoint (e.g. Mistral-7B, or Qwen-1.8B).
2. Replicate the dense FFN N times to create N identical experts.
3. Add a randomly-initialized router in front of each MoE layer.
4. Continue pretraining on a large-enough dataset.

The experts start literally identical. They differentiate as training proceeds, because the router sends different tokens to different experts, and different gradient signals specialize them. The process is surprisingly stable, because the dense initialization is a strong starting point and the differentiation happens gradually.

Mixtral 8x7B was upcycled from Mistral-7B. Qwen1.5-MoE-A2.7B was upcycled from Qwen-1.8B. Both achieved quality comparable to much larger dense models at a fraction of the training cost.

Caveats. Upcycled models seem to inherit some ceiling from the dense model they came from. A from-scratch MoE trained with the same total budget often outperforms an upcycled MoE, particularly on capabilities that benefit from expert specialization (code, math, multilingual). Upcycling is unbeatable for time-to-first-decent-model and for teams without billion-dollar budgets. If you can afford from-scratch and you have the data, from-scratch wins.

### 5.6 A concrete from-scratch recipe

If you are training a roughly 15B-active and 150B-total MoE from scratch, here is a sane starting point, with the caveats that every team tunes these numbers and the "right" values depend on your data, hardware, and schedule.

```
n_experts              = 64
experts_per_token (k)  = 8          # fine-grained, many active
shared_experts         = 2
d_model                = 4096
d_ff_per_expert        = 1536       # smaller because fine-grained
n_layers               = 40
moe_every_n_layers     = 1          # all layers MoE except the first
balance_loss_coef      = 0.001      # or 0 if using aux-loss-free bias updates
router_z_loss_coef     = 1e-3
router_init_scale      = 0.1
capacity_factor_train  = 1.25
capacity_factor_infer  = unlimited  # no token dropping at serve time
optimizer              = AdamW, β = (0.9, 0.95)
peak_lr                = 4e-4       # comparable to a dense model of same active size
warmup                 = 2000 steps
schedule               = cosine to 10% of peak
seq_len                = 4096 pretrain, anneal up to 32k late
dtype                  = BF16 weights, FP8 GEMM, FP32 master, FP32 router
```

Common pitfalls at this scale: running the balance loss coefficient too high (routers become uniform and the model learns nothing interesting), too low (expert collapse), not freezing router precision (occasional NaN spikes), and under-scaling the initialization (routing too spiky from step zero).

## 6. Fine-tuning MoE models in the wild

Fine-tuning an MoE is not quite the same as fine-tuning a dense model. Three scenarios dominate, and each has its own playbook.

### 6.1 Full supervised fine-tuning (SFT / instruction tuning)

The good news is that it mostly just works. Mixtral, DeepSeek, Qwen, and DBRX all ship SFT'd instruct variants trained with standard cross-entropy on chat-format data.

The gotchas.

**Re-balance.** SFT data distributions are very different from pretraining distributions. A routing pattern that was well-calibrated on a 10T-token web mix may be sharply miscalibrated on a 100K-sample instruction dataset. Experts that specialized on, say, C++ syntax during pretraining may be almost starved during SFT. You should keep the balance loss (or the auxiliary-loss-free bias update) on during SFT, otherwise you drift toward expert collapse or toward experts becoming underutilized.

**Smaller learning rate than the dense equivalent.** MoE routers are sensitive, and gradients through the gating weights can amplify noise. A starting point is half the LR you would use for a dense model of the same active-parameter count.

**Longer warmup.** Re-stabilize routing before hitting full-strength updates. A thousand-step linear warmup on a 50K-sample SFT run is not unreasonable.

**Do not freeze the router.** A few teams have tried this (reasoning that the router captures "what the model knows" and should not move). Generalization collapses when the task distribution shifts even slightly, because frozen routing cannot adapt to the new distribution.

### 6.2 LoRA and other PEFT methods

LoRA on an MoE is subtler than LoRA on a dense model, because now you have to decide *where* to apply LoRA: on the attention, on the router, on shared experts, on routed experts, or some combination.

**LoRA on attention only.** The cheapest option. Leaves all experts and the router untouched. Works fine for most SFT applications where you are adapting the model's style or persona without changing what it knows. Relies entirely on the router already being correct for the target domain.

**LoRA per expert.** Adds `A_i, B_i` LoRA adapters to every expert's W1, W2 (and W3 for SwiGLU). This lets individual experts specialize for the new domain. Memory grows linearly in `n_experts`, which on a 256-expert model is significant. Also increases optimizer-state memory proportionally.

**LoRA on shared experts only.** Middle ground. Adapts the common-knowledge part of the model while leaving routed specialization alone. Surprisingly effective for style and tone transfer, because "style" is largely common-case behavior that shared experts handle.

**LoRA on router.** Lets you adapt routing decisions for the new domain without touching the experts themselves. Useful if your concern is that pretraining routing is miscalibrated for your domain (e.g. a medical or legal corpus that looks nothing like web pretraining).

**MoLoRA / MoE-LoRA.** A 2024 research line where the LoRA adapters themselves are a mixture: you have a small set of LoRA experts with their own tiny router, applied as a PEFT on top of a frozen base MoE. Useful when the fine-tuning task has distinct subtasks (e.g. a multi-turn agent that sometimes writes code, sometimes summarizes, sometimes translates).

Practical recommendation for most SFT on a pretrained MoE: LoRA on attention plus shared expert, with rank 16–64, is the sweet spot. Do not touch routed experts unless you have a specific capacity reason.

### 6.3 Preference tuning (RLHF, DPO, GRPO) on MoE

Preference learning works on MoE but introduces one subtle and annoying problem: **router drift**. DPO and GRPO updates can shift the router in unexpected ways even when individual experts are behaving correctly, because gradients flow through the gating weights. A preference signal that pushes the model toward a particular output can end up pushing the router to use different experts rather than to improve the experts themselves, which can drift the model away from its pretraining specialization.

Fixes that work in practice.

**Freeze routers during preference tuning** (or apply a much lower learning rate to them, like 0.01x). Empirically this preserves the expert allocation learned during SFT and focuses gradient signal on the actual expert weights.

**KL-regularize the router distribution** against the reference (pre-DPO) model, in addition to the usual logit KL. This adds an extra term that specifically penalizes the router moving, which is surprisingly effective.

**Use reference-free methods** (SimPO, ORPO) where the KL term against a reference model is implicit. These methods tend to be less prone to router drift because they do not have a free parameter that can burn KL budget on routing changes.

DeepSeek and Qwen both ship RLHF'd MoE variants, and the consensus in their technical reports is: treat the router as part of the "backbone" that changes slowly, not part of the "head" that adapts fast.

### 6.4 Selective expert fine-tuning for domain adaptation

This is specific to MoE and sometimes very useful. The idea: identify which experts fire most on your domain data, and fine-tune *only those experts* while freezing everything else.

The recipe is:

1. Run your domain corpus through the pretrained model and log router decisions.
2. For each expert, count activations.
3. Pick the top few percent of most-fired experts on your domain.
4. Fine-tune only those, with the router and all other experts frozen.

This preserves general capability (because the bulk of the model is frozen) while letting the "on-topic" experts adapt. The memory savings are significant on large MoEs. It works especially well for programming-language specialization (a few experts will turn out to be mostly responsible for a given language), for legal/medical domains where a few experts handle the jargon, and for low-resource languages where a few experts end up as the multilingual-tail specialists.

The failure mode to watch for: if your domain corpus is small and your top-fired experts are too few, you will overfit those experts and lose general quality in their subdomain. Mitigate with standard SFT regularization (dropout on expert outputs, early stopping on a held-out pretraining-distribution set).

## 7. Serving MoE in production: the realities

![MoE serving flow: all-to-all dispatch tokens to expert GPUs, experts compute in parallel, all-to-all combine back — plus capacity factor, expert parallelism, aux-loss-free balance (DeepSeek), and shared experts](/imgs/blogs/moe-03-serving.png)

Training challenges are about compute and communication. Serving challenges are different: they are about memory, batching dynamics, and tail latency.

### 7.1 The memory wall

A 671B MoE at BF16 is 1.3 TB of weights. At FP8 it is 670 GB. Even Mixtral 8x22B, which is modest by frontier standards, is 280 GB at BF16. You cannot fit these on a single GPU, which means expert parallelism or aggressive quantization.

**Expert parallelism at inference.** Shard experts across GPUs, do all-to-all per MoE layer at serve time. Adds latency per request. The cost amortizes well over large batches and poorly over small ones. For batch-heavy throughput workloads (embedding generation, batch code completion, offline classification), EP is efficient. For interactive single-user chat, it can be slower than a dense model of the same quality.

**Offloading.** Store rarely-used experts on CPU memory or even NVMe, keep hot experts on GPU, page in and out based on routing. DeepSpeed-MII, Hugging Face Accelerate, and several research systems (Fiddler, MoE-Lightning) do this. Acceptable for throughput workloads. Terrible for latency; the first token after a cold expert miss can be hundreds of milliseconds.

**Expert pruning and merging.** Post-hoc remove dead experts or merge similar ones. MergeKit has a mode for this. Reduces capacity but usable for memory-constrained deployments where you only need the broad behavior, not the full specialization.

### 7.2 Quantization

MoE quantizes well, with a few caveats.

**Per-expert quantization is natural.** Each expert is a few GEMMs, and you can apply AWQ, GPTQ, or similar methods expert-by-expert. Calibration data should cover the domains each expert specializes in, which is easy if your calibration set is diverse.

**The router stays high-precision.** Quantizing the router tends to cause bad routing decisions (wrong expert selected for borderline tokens). BF16 or FP16 for the router is the safe default.

**Weight-only 4-bit plus BF16 activations** (AWQ, GPTQ) works well for Mixtral and DeepSeek at small quality cost. The KV cache stays BF16 or is quantized separately.

**FP8 training plus FP8 inference** (the DeepSeek-V3 regime) halves memory without post-training quantization overhead, but requires the model to have been trained in FP8 to begin with. Retrofitting FP8 onto a BF16-trained MoE usually works but with slightly worse quality than native FP8 training.

### 7.3 Batching and latency dynamics

Routing is input-dependent, so different tokens in a batch go to different experts. This has opposite implications at large and small batch sizes.

**Large batches.** Expert load is close to its long-run average, which is roughly uniform if training was well-balanced. All experts do useful work, GEMMs are big enough to hit peak throughput, all-to-all amortizes. MoE shines.

**Small batches.** A single token lights up 2 or 8 experts, each doing a tiny GEMM on a handful of tokens. GEMM efficiency collapses (most of the work is launching the kernel, not computing). All-to-all still happens but with almost nothing in the payload, which is nearly all overhead. MoE latency is dominated by overhead rather than compute.

This creates a serving-economics gap: MoE is cheaper than dense per token at high batch sizes but more expensive per token at very low batch sizes. If your workload is latency-sensitive and single-user, benchmark carefully before committing.

Frameworks that handle MoE serving well include vLLM (expert-parallel support matured in 2024), SGLang (strong MoE focus with fused all-to-all kernels), TensorRT-LLM (enterprise-grade but commercial), and DeepSpeed-Inference (good for offloading setups).

### 7.4 Speculative decoding on MoE

Works, with an interesting wrinkle. The draft model is usually dense (because you want the draft to be cheap and fast, which MoE is not at batch-1). Acceptance rates against an MoE verifier are similar to dense-versus-dense at matched quality. The main benefit is that drafted tokens the verifier would have produced anyway never trigger the verifier's expensive all-to-all, which cuts average latency significantly on long generations.

## 8. Case studies: what actually shipped

### 8.1 Google — GShard, Switch Transformer, GLaM, ST-MoE

The entire MoE-for-LLMs playbook was invented at Google.

**GShard (2020, 600B).** Top-2 routing, auxiliary balance loss, expert parallelism via Mesh-TensorFlow. Trained for multilingual translation across 100 language pairs. Demonstrated that MoE scales to hundreds of billions of parameters and that the engineering is tractable.

**Switch Transformer (2021, 1.6T).** Simplified GShard to top-1 routing and argued "one expert is enough." Top-1 trained faster, was more stable, and did not hurt quality. Switch became the default MoE recipe for the next several years. The paper also introduced capacity factor tuning as a first-class hyperparameter.

**GLaM (2021, 1.2T total, 97B active).** First billion-scale MoE language model with broad English capability. Matched GPT-3 at roughly one third of the training compute and half of the inference compute. The GLaM paper is where the dense-versus-MoE efficiency argument was most cleanly established for LLMs.

**ST-MoE (2022, 269B).** Introduced the router z-loss, studied fine-tuning stability in depth, shipped public checkpoints. Required reading for anyone trying to fine-tune an MoE. Among other things, ST-MoE showed that MoE fine-tuning is *less* stable than dense fine-tuning without careful regularization, and gave a menu of fixes.

Google also productionized MoE inside the Gemini line (strongly rumored but not confirmed in detail), and used MoE variants in PaLM's Flan-MoE fine-tuning studies.

### 8.2 Mistral — Mixtral 8x7B and 8x22B

Mistral is the team that made MoE mainstream in the open-weight LLM world.

**Mixtral 8x7B (December 2023).** Eight experts per MoE layer, top-2 routing, 47B total and roughly 13B active. Upcycled from Mistral-7B by replicating the FFN eight times, adding a router, and continuing pretraining. Licensed Apache 2.0.

The impact was large. Mixtral matched or beat Llama-2 70B on most benchmarks at roughly one fifth the inference cost. The "70B-class quality at 13B active cost" serving story is what made MoE click for the open-source ecosystem and triggered a flood of fine-tunes, community merges, and follow-on work.

**Mixtral 8x22B (April 2024).** Same architectural recipe at larger scale. 141B total, 39B active. Still Apache 2.0. Further validated that the upcycling-from-dense approach scales.

One underappreciated effect of the Mixtral releases was the community MergeKit ecosystem. MergeKit's `mixtral` mode lets you stitch existing dense models together as experts, essentially constructing an MoE without any training. The results are often surprisingly good for specific tasks (code-plus-chat, instruction-plus-roleplay), even though the "router" in these community merges is usually either random or extremely simple. It is a poor man's MoE and it showcases that specialization-plus-routing is a powerful structure even when the routing is crude.

### 8.3 DeepSeek — DeepSeek-MoE, V2, V3

DeepSeek iterated MoE design further than anyone else in the open-weight world, and the innovations stuck.

**DeepSeek-MoE (early 2024, 16B total, 2.8B active).** Introduced fine-grained experts (64 instead of 8) and shared experts (always-on experts alongside the routed ones). Showed that each change matters independently and that they compose. The paper is short, empirical, and enormously influential.

**DeepSeek-V2 (May 2024, 236B total, 21B active).** Scaled the recipe. Added **Multi-head Latent Attention (MLA)** to reduce KV-cache memory by an order of magnitude, which is somewhat orthogonal to MoE but crucial for serving. 160 routed experts plus 2 shared. Matched GPT-4 level on code and math at a fraction of the cost.

**DeepSeek-V3 (December 2024, 671B total, 37B active).** The frontier of publicly-documented MoE engineering. 256 routed experts plus 1 shared, top-8 routing. **Auxiliary-loss-free load balancing** via per-expert learned biases. **FP8 training end-to-end**, a first at this scale. **Multi-Token Prediction (MTP)** auxiliary head to improve sample efficiency and enable faster speculative decoding. Trained for a compute bill that came to about $5.5M, small enough that several serious commentators initially refused to believe the number. Near GPT-4 quality on many evals.

The DeepSeek-V3 technical report is arguably the best publicly available recipe for training a frontier-class MoE from scratch. If you read one MoE paper and you only care about what currently works at the top end, read that one.

### 8.4 Qwen — Qwen1.5-MoE, Qwen2-57B-A14B, Qwen3-MoE

Alibaba's Qwen team shipped a series of open-weight MoE checkpoints.

**Qwen1.5-MoE-A2.7B.** Upcycled from Qwen-1.8B. 14.3B total, 2.7B active. Matched 7B-dense quality at sub-3B inference cost. A particularly strong demonstration that upcycling is practical for smaller teams.

**Qwen2-57B-A14B.** Scaled up, trained from scratch with fine-grained experts. Closed a lot of the gap with Mixtral 8x7B at smaller active size.

**Qwen3-MoE (2025).** Further fine-grained, shared experts, auxiliary-loss-free balancing. Clearly DeepSeek-inspired and confirms the convergence of frontier MoE recipes.

### 8.5 xAI — Grok-1

**Grok-1 (March 2024).** 314B total, 8 experts per layer, top-2 routing, roughly 78B active. Released under Apache 2.0 as a weights-only drop. No instruct tune at the time of release. At release it was the largest open-weight MoE, though without the fine-grained specialization or shared experts of the DeepSeek line. Mostly useful now as a reference weight-set and a baseline rather than a production model.

### 8.6 Databricks — DBRX

**DBRX (March 2024).** 132B total, 36B active. Sixteen experts per layer with top-4 routing, notably more fine-grained than Mixtral's eight-with-top-2. Positioned explicitly as an enterprise-ready open-weight alternative to Llama and Mixtral. Trained on Databricks's stack with curated enterprise and code data.

DBRX was particularly interesting for its focus on programming performance. The fine-grained routing (four active experts per token instead of two) helped experts specialize on specific programming languages and frameworks, and DBRX posted strong numbers on HumanEval and similar coding benchmarks at the time.

### 8.7 Others worth mentioning

**Meta NLLB-MoE (2022).** 54B-parameter MoE for 200-language translation. Specialized experts for language families.

**Snowflake Arctic (April 2024).** 480B total, 17B active, 128 experts, explicitly structured as a dense-MoE hybrid with a dense FFN running alongside the MoE in every layer. The architectural choice is idiosyncratic but addresses the same shared-expert motivation DeepSeek solved differently.

**JetMoE-8B (2024).** Small MoE trained on a shoestring budget (around $100K), deliberately constructed to demonstrate that the MoE recipe is accessible to labs without frontier budgets.

**MiniMax-01 (2025).** 456B total, 45.9B active, combines MoE with linear attention hybrids for 4M-token context. One of several 2025 models exploring the intersection of long context and sparse models.

## 9. A practical decision guide

**Should I train an MoE instead of a dense model?**

If you have at least 500 billion tokens of diverse training data, MoE pays off. If inference latency matters much more than throughput, dense usually wins at small and medium scales. If GPU memory is your binding constraint (because you are serving on one or two GPUs), dense is simpler. If you want the best quality per training dollar and you can provision the HBM, MoE wins, ideally upcycled from a strong dense checkpoint if you want a fast path.

**Should I fine-tune an existing MoE?**

Yes, Mixtral, DeepSeek-V2/V3, Qwen-MoE, and DBRX all fine-tune well with standard SFT plus LoRA-on-attention. Expect roughly half to equal the learning rate of a dense model of equivalent active parameters. Keep the balance loss (or the aux-loss-free bias updates) on during SFT, and do not freeze the router. For RLHF, freeze or down-weight the router specifically.

**Should I serve MoE?**

On multi-GPU nodes with NVLink, yes. Expert parallelism is efficient. On a single GPU for a >70B MoE, probably not; quantize a dense 70B instead, unless you have room for aggressive offloading. For throughput-heavy workloads (batch inference, offline work, large-batch APIs), MoE shines; batch utilization is high and communication amortizes. For latency-sensitive single-user workloads, MoE can be slower per token than a dense model of equivalent quality because of the all-to-all overhead. Measure before committing.

## 10. What is next

Research directions that are live as of early 2026.

**Auxiliary-loss-free balancing** is becoming standard; expect more PID-style control-theoretic techniques in routing, because they compose better with other objectives than gradient-based auxiliary losses.

**Heterogeneous experts**, where experts have different shapes (some deeper, some wider, some with different activation functions, some quantized differently), are showing promising early results. The intuition is that different specializations may benefit from different expert geometries, which the uniform-experts design ignores.

**Hierarchical routing**, routing first to a group and then within the group, reduces router cost from O(N) to O(√N) and is becoming important as expert counts push past 1000.

**MoE combined with long context.** MiniMax-01 demonstrated one approach (MoE plus linear attention). Several research labs are pursuing similar hybrids, and the engineering challenges are substantial but tractable.

**Multimodal MoE.** Separate experts for vision, audio, code, and language tokens inside a unified transformer are a natural fit for the MoE paradigm, because modality is an obvious specialization signal. Several 2025 multimodal models use some form of this.

**Training-time expert merging and spawning.** Periodically merge very similar experts and spawn new experts from high-traffic ones, to allocate capacity dynamically during training rather than fixing it at initialization. Early research work in 2025, not yet standard.

The core insight has not changed since Shazeer's 2017 Sparsely-Gated MoE paper: *let different inputs use different parameters*. What changed is that a decade of careful engineering, around routing, balancing, parallelism, quantization, and serving, has finally made this paradigm the dominant architecture at the frontier. Dense transformers are not going away, but the biggest, fastest, cheapest-per-token frontier models as of 2026 are all sparse, and the gap is widening rather than closing.

## Further reading

Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (2017).

Lepikhin et al., *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding* (2020).

Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* (2021).

Du et al., *GLaM: Efficient Scaling of Language Models with Mixture-of-Experts* (2021).

Zoph et al., *ST-MoE: Designing Stable and Transferable Sparse Expert Models* (2022).

Jiang et al., *Mixtral of Experts* (2024).

Dai et al., *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models* (2024).

DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (2024).

DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024).

Databricks, *Introducing DBRX: A New State-of-the-Art Open LLM* (2024).

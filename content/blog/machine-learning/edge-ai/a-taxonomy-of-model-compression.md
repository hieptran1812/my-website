---
title: "A taxonomy of model compression: how the four levers compose on the Pareto frontier"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A field map that places every compression technique into four levers, shows what each one trades, and gives you the order to stack them so the wins multiply instead of cancel."
tags:
  [
    "edge-ai",
    "model-optimization",
    "quantization",
    "pruning",
    "knowledge-distillation",
    "neural-architecture-search",
    "inference",
    "efficient-ml",
    "pareto-frontier",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/a-taxonomy-of-model-compression-1.png"
---

The first time I tried to read the model-compression literature end to end, I gave up around paper forty. Every week there is a new acronym — GPTQ, AWQ, SmoothQuant, SparseGPT, Wanda, MCUNet, OFA, BitNet — and each one arrives with a benchmark table that makes it look like the obvious winner. It reads like a zoo. New engineers wander in, grab whichever animal had the prettiest results plot last month, bolt it onto their model, and are then surprised when the int4 LLM is *slower* than the fp16 one, or the pruned CNN is exactly the same speed it was before, or two techniques that each "gave 4×" stack to a disappointing 5× instead of 16×.

The cure is not reading paper forty-one. It is a map. Underneath the zoo there are only **four levers**, and almost every paper you will ever read is a clever way of pulling one of them — or a recipe for pulling several in the right order. Once you can name the lever a technique pulls, you immediately know what it reduces, what it costs you, whether it speeds up *your* hardware, and which other levers it stacks with. The literature stops being a flood and becomes a filing system: a new paper is just a new entry in a drawer you already have.

This post is that map. It is the master frame the rest of this series points back to. We will (1) define the four levers precisely — quantization, pruning/sparsity, knowledge distillation, and efficient architecture/NAS — plus the *substrate* they all sit on (compilers and runtimes) and the *discipline* that keeps everyone honest (profiling). We will (2) lay them out on a second axis — *when* in the pipeline each one applies and *what representation* it changes. We will (3) make the goal rigorous by formalizing the accuracy–efficiency trade-off as a Pareto frontier, so "better" stops being a vibe and becomes a definition. We will (4) work out *how the levers compose* — which combinations multiply, which conflict, and the **order** you should apply them in — which is the genuinely hard, genuinely useful part. And we will (5) turn all of it into a decision procedure plus two worked examples (a CNN for a phone, a 7B LLM for a laptop) so you can act on it Monday morning.

Figure 1 is the whole thing on one slide; keep it open while you read the rest. By the end you should be able to take any technique — including one published next week — and slot it into the map in under a minute: which lever, which stage, what it trades, and where it goes in your stack.

![A tree diagram showing model compression branching into the four levers quantization, pruning, distillation, and efficient architecture, resting on a compiler-and-runtime substrate and validated by profiling](/imgs/blogs/a-taxonomy-of-model-compression-1.png)

A note before we start, because it sets the tone for the whole series: **a compression technique is never free.** It buys you size, speed, or energy and it always charges you in one of three currencies — accuracy, engineering effort, or hardware portability. Half of being good at this is refusing to pay for a win you do not need. We will come back to that idea constantly.

## 1. Why "four levers" and not "forty techniques"

Here is the organizing insight, and it is almost embarrassingly simple once you see it. A neural network, at inference time, is a graph of operations applied to tensors of numbers. There are only so many things you can shrink:

- You can use **fewer bits per number**. The values stay roughly where they are; you just store and compute them more coarsely. That is **quantization**.
- You can **set numbers to zero and skip them**, or remove whole rows/columns/heads. The structure of the tensors changes; some of it vanishes. That is **pruning / sparsity**.
- You can **train a different, smaller model to behave like the big one**. The topology changes — fewer layers, narrower widths — but it is still a dense, ordinary model afterward. That is **knowledge distillation**.
- You can **design the operations themselves to be cheaper**. Replace a dense convolution with a depthwise-separable one; replace full attention with a linear-ish variant; let a search algorithm pick the block. The ops change. That is **efficient architecture / NAS**.

That is the entire space of *what* you can reduce in a model: the **values**, the **structure**, the **topology**, or the **ops**. Four representations, four levers. Everything else in the compression literature is either a smarter way to pull one of these (GPTQ is a smart quantizer; SparseGPT is a smart pruner; DistilBERT is a distillation recipe; MobileNet is an efficient architecture) or it is not actually compression at all — it is the *substrate* beneath them or the *discipline* around them.

The substrate is the **compiler and runtime**: TensorRT, ONNX Runtime, TFLite/LiteRT, XLA, `llama.cpp`. These do not change the math your model expresses; they change how that math is *scheduled* onto silicon — fusing operators, picking memory layouts, selecting hand-tuned kernels, batching small ops. They are the cheapest wins in the whole field because they cost you zero accuracy. The mechanism is worth a concrete example: a naive graph runs a convolution, writes the full activation tensor out to DRAM, reads it back to apply a bias, writes it again, reads it a third time to apply a ReLU. Each round trip is bandwidth you are spending for nothing. A fusing compiler collapses conv+bias+ReLU into a single kernel that keeps the intermediate in registers or on-chip SRAM and writes the result once — often a 1.3–2× wall-clock win on a memory-bound graph, for free, because you moved less data (recall the arithmetic-intensity argument we sharpen in section 3). You should always be sitting on a good runtime before you reach for any lever, and you should run it *again* after every lever, because compression changes which kernels are optimal — after quantization the fused kernel is an *int8* conv+bias+ReLU, a different and faster code path the compiler can only pick if you let it re-lower the model.

The discipline is **profiling**: the roofline model, on-device latency measurement, energy counters. Profiling is not a lever — it reduces nothing — but it is the referee. It tells you whether the win you think you got is real, whether your op is compute-bound or memory-bound (which decides whether quantization even *can* speed it up), and where the actual bottleneck lives. Skipping it is how you end up "optimizing" an operation that was never on the critical path. The series has a whole post on [the roofline model and where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) and another on [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device); treat them as prerequisites for trusting any number in this post.

So the map has four levers, one substrate, one discipline. Six boxes. Hold them in your head and the zoo collapses into a taxonomy.

### A quick taste of each lever, by the numbers

Before we go deep, here is the thirty-second version of what each lever typically buys and charges, so the rest of the post has pegs to hang detail on:

| Lever | Reduces | Typical win | Charges you |
| --- | --- | --- | --- |
| Quantization | bits per value | ~4× size (fp32→int8), 2–4× speed *if HW helps* | accuracy if naive; portability (HW must support the dtype) |
| Pruning / sparsity | nonzero values / structure | 2–10× fewer params; speed *only* if structured/N:M | a fine-tune; speedup is hardware-conditional |
| Distillation | model topology | a 2–10× smaller dense model at near-teacher accuracy | a full training run + a teacher |
| Efficient arch / NAS | the operations | the largest FLOP/latency wins; pushes the frontier out | retrain from scratch; search cost |

The single most important column there is the last one. Notice that the levers charge you in *different* currencies — quantization mostly charges hardware portability, distillation charges a training run, architecture charges a from-scratch retrain. That difference is exactly why they compose: they are not competing for the same budget. We will make that rigorous in section 4 and exploit it in section 5.

## 2. The four levers, defined properly

Let me give each lever the same four-part treatment — *mechanism*, *what it reduces*, *what it costs*, *typical wins* — because consistency is what makes a taxonomy useful. Each subsection cross-links the deep-dive post in this series where the lever gets its own twelve thousand words; here we stay at the level of "what is this lever, fundamentally."

### 2.1 Quantization — fewer bits per value

**Mechanism.** A trained network stores weights and activations as 32-bit floats. Most of that precision is wasted: the weights in a given layer cluster in a narrow range, and the network is robust to small perturbations. Quantization maps the float range $[\,x_{\min}, x_{\max}\,]$ onto a small integer grid. For 8-bit symmetric quantization you pick a scale $s = \frac{\max(|x_{\min}|,|x_{\max}|)}{127}$ and represent each value as $q = \mathrm{round}(x / s)$, an integer in $[-127, 127]$. At inference you either dequantize ($\hat{x} = s\,q$) or, better, run the matmul directly in integer arithmetic and only rescale the result.

There are two flavors. **Post-training quantization (PTQ)** takes an already-trained float model and quantizes it using a small *calibration* set to estimate the ranges — no gradient steps, minutes of work. **Quantization-aware training (QAT)** inserts "fake-quant" nodes into the graph and fine-tunes the model so it learns weights that survive the rounding; the straight-through estimator (STE) passes gradients through the non-differentiable rounding. PTQ is cheap and usually enough at int8; QAT is the rescue when int4 or a stubborn model loses too much.

**What it reduces.** Bits per value. fp32→int8 is exactly 4× on storage and weight-memory bandwidth. fp32→int4 is 8×.

**What it costs.** Accuracy, if you are naive — outlier activations in transformers will blow up your scale and crush the resolution of everything else (this is the entire reason SmoothQuant and AWQ exist). And **hardware portability**: int8 only goes *faster* if the target has int8 matmul units (most NPUs and modern GPUs do; some older DSPs do not, and will silently dequantize to float, giving you the size win but not the speed win).

**Typical wins.** ~4× smaller, 2–4× faster on hardware that supports it, with under 1% accuracy loss for int8 PTQ on most CNNs and well-behaved transformers. The series covers this lever across several posts — start with the foundational quantization post and the QAT-vs-PTQ comparison.

*The numerics, briefly, because this is the lever where "how much accuracy do I lose?" has a clean answer.* When you round a value to a grid of step size $\Delta = s$ (the quantization scale), the rounding error $e = \hat{x} - x$ is, to a good approximation, uniformly distributed on $[-\Delta/2, \Delta/2]$. A uniform distribution on an interval of width $\Delta$ has variance $\Delta^2/12$, so the **quantization noise power** is $\sigma_e^2 = \Delta^2/12$. For a $b$-bit signed grid spanning a range of $2A$ (so $\Delta = 2A / 2^b$), the signal-to-quantization-noise ratio works out to the famous law $\text{SQNR} \approx 6.02\,b + 1.76\ \text{dB}$ — **every extra bit buys about 6 dB.** That is the whole story of why int8 (8 bits, ~50 dB) is usually safe and int4 (4 bits, ~26 dB) is risky: you have thrown away ~24 dB of headroom, and whether that matters depends entirely on how much of your tensor's range is actually *used*. This is also why outliers are catastrophic: a single large activation forces $A$ up, inflating $\Delta$ for *every* value, so the effective bits for the bulk of the distribution collapse. SmoothQuant and AWQ exist precisely to keep $A$ from being hijacked by a few channels. Hold this law in mind for the whole series: accuracy loss from quantization is not mysterious, it is $6\,\text{dB}$ per bit traded against how tightly your ranges are packed.

### 2.2 Pruning / sparsity — set values to zero

**Mechanism.** Many weights barely matter; their magnitude is tiny and zeroing them changes the output negligibly. Pruning identifies and removes them. The crucial distinction is *what* you remove:

- **Unstructured pruning** zeros individual weights anywhere in the tensor. You can remove 80–90% of weights this way with little accuracy loss, and the model is now 80–90% zeros — but it is the *same shape*. A dense matmul kernel still multiplies all those zeros. On commodity CPUs and GPUs you get the *storage* win (sparse formats) but **no speed win** unless you have a sparse kernel that is actually faster than the dense one, which is rare below ~95% sparsity.
- **Structured pruning** removes whole units: entire channels, filters, attention heads, or layers. Now the tensors are genuinely smaller, the matmuls are genuinely smaller, and *any* dense kernel runs faster. You pay for it in accuracy (you removed coarser chunks) and usually a fine-tune to recover.
- **N:M sparsity** (e.g. 2:4) is the pragmatic middle: in every block of M consecutive weights, at least N are zero. NVIDIA's Sparse Tensor Cores execute 2:4 sparse matmuls at ~2× the dense rate, so this is structured *enough* for hardware to exploit while being fine-grained enough to keep accuracy.

**What it reduces.** The number of nonzero values and/or the structural size of tensors.

**What it costs.** A fine-tune to recover accuracy, and — the part everyone forgets — **the speedup is entirely hardware-conditional**. Unstructured sparsity on a phone CPU buys you nothing in latency. This is anti-pattern number one and we will hammer it again in section 9.

**Typical wins.** 2–10× fewer parameters; reliable *speed* only with structured or N:M sparsity on supporting hardware. See the pruning deep-dive and the structured-vs-unstructured post in the series.

### 2.3 Knowledge distillation — train a small model to mimic a big one

**Mechanism.** Train a small "student" network to reproduce the behavior of a large, accurate "teacher." Instead of (or in addition to) the hard one-hot labels, the student matches the teacher's full output *distribution* — the soft probabilities, the "dark knowledge" that says this image is mostly a cat but a little bit a lynx. The classic loss combines a temperature-softened KL divergence to the teacher with the ordinary task loss:

$$\mathcal{L} = (1-\alpha)\,\mathcal{L}_{\text{task}}(y, \sigma(z_s)) + \alpha\,T^2\,\mathrm{KL}\big(\sigma(z_t/T)\,\|\,\sigma(z_s/T)\big)$$

where $z_s, z_t$ are student/teacher logits, $T$ is the temperature, and $\sigma$ is softmax. The $T^2$ factor keeps the gradient magnitudes comparable across temperatures. Feature-based variants also match intermediate activations.

**What it reduces.** The topology — the student can have fewer layers and narrower widths than the teacher. Critically, the student is a **normal dense model** afterward: nothing special at inference time, so it stacks cleanly with every other lever.

**What it costs.** A full training run *plus* a teacher you have to run over your data (forward passes are not free). You need the data, or a good proxy for it.

*Why soft targets beat hard labels, briefly.* A one-hot label tells the student "this is a 7" and nothing else — one bit of supervision per example. The teacher's softened distribution says "this is a 7, but it is a little bit a 1 and a little bit a 9, and definitely not a 4" — it leaks the teacher's learned similarity structure over the *whole* output space. In information terms, each training example now carries far more bits of gradient signal, which is exactly why a student distilled from a teacher reaches higher accuracy than the *same* student architecture trained from scratch on the hard labels. The temperature $T$ controls how much of this structure is exposed: at $T=1$ the soft targets are nearly one-hot for a confident teacher; raising $T$ flattens them and surfaces the small inter-class probabilities, which is where the dark knowledge lives. This is also why distillation pairs so well with the *low-precision* levers — a quantized or heavily pruned student has lost capacity, and the denser supervisory signal from soft targets is precisely what helps it spend its remaining capacity well.

**Typical wins.** DistilBERT is the canonical number: ~40% smaller and ~60% faster than BERT-base while keeping ~97% of its GLUE performance. The series has a distillation deep-dive; this lever pairs especially well with quantization (distill, then quantize) and *is itself* the efficient-architecture lever when the student is hand-designed to be cheap.

### 2.4 Efficient architecture / NAS — design fewer, cheaper ops

**Mechanism.** Do not compress a wasteful model — *do not build* the wasteful model. A standard 3×3 convolution over $C$ channels costs $9 C^2 HW$ multiply-adds. A **depthwise-separable** convolution (MobileNet's core trick) splits it into a per-channel 3×3 ($9 C\,HW$) plus a 1×1 pointwise mix ($C^2 HW$), cutting cost by roughly $\frac{1}{C} + \frac{1}{9}$ — an 8–9× FLOP reduction for typical channel counts at a small accuracy cost. **Neural architecture search (NAS)**, and especially *hardware-aware* NAS (MnasNet, MobileNetV3, Once-for-All), automates the design: it searches the space of blocks and widths to maximize accuracy *subject to a measured latency budget on the target chip*, not a proxy FLOP count.

**What it reduces.** The operations themselves — their count and their cost.

**What it costs.** A retrain from scratch (the architecture is new) and, for NAS, the search itself, which historically cost thousands of GPU-hours (Once-for-All amortizes this by training one supernet and sub-sampling specialized children for free).

**Typical wins.** The *largest* of all four levers, because it is the only one that **pushes the Pareto frontier outward** rather than moving you along it (section 4 makes this precise). MobileNetV3 hits ResNet-class accuracy at a fraction of the latency on a Pixel. The catch is that this lever requires the most up-front investment, so it is usually a decision you make once per product, not per release.

## 3. The second axis: when it applies and what it changes

Naming the levers is half the taxonomy. The other half is placing them on a 2-D map, because *the reason the levers compose is that they occupy different cells of this map.* Two techniques that change the same representation at the same pipeline stage fight each other; two that occupy different cells generally do not.

The first axis is **when in the lifecycle** the lever acts:

- **Design / architecture time** — before any weights exist. Efficient architecture and NAS live here.
- **Training time** — during the optimization that produces weights. Distillation lives here; so does QAT and sparsity-aware training.
- **Post-training** — on a finished, trained model. PTQ and post-training (one-shot) pruning like SparseGPT/Wanda live here. This is the cheapest stage to act at because there is no training loop.
- **Compile time** — when the graph is lowered to a runtime engine. Operator fusion, layout transforms, kernel selection, constant folding. The substrate.
- **Runtime** — during execution. Dynamic batching, KV-cache management, delegate selection (run this op on the NPU, that one on the CPU).

The second axis is **what representation** the lever edits: the **values** (quantization), the **structure** (pruning/sparsity), the **topology** (distillation, architecture), or the **ops / schedule** (architecture again, and the compiler).

Figure 2 lays the levers on this grid. The thing to notice — and the reason the figure is worth a thousand words of prose — is that the levers spread out across both axes. Architecture acts earliest and changes the most (topology *and* ops); quantization acts latest among the model-altering levers and changes the least (just the values); pruning sits in between; the compiler acts purely at compile/runtime and never touches the model's mathematical meaning at all. They are not piled in one cell. That separation is the structural fact that makes section 5 possible.

![A matrix mapping the five levers against what they change and when in the pipeline they apply, showing they occupy distinct cells from design time through runtime](/imgs/blogs/a-taxonomy-of-model-compression-2.png)

A useful mental model that this figure makes concrete: read the pipeline left to right and you are watching the model become *more concrete and more hardware-coupled* at each stage. At design time you are choosing math. At training time you are fitting weights. Post-training you are reshaping the finished weights. At compile time you are committing to a particular chip's kernels. At runtime you are reacting to live load. The levers that act early are portable and expensive to change; the levers that act late are cheap to change but locked to your target. This single gradient — *early = portable + expensive, late = cheap + hardware-locked* — predicts almost everything about how to order them, which is exactly where we are headed.

There is a second reason this map matters, and it is the bridge to *which* lever can even physically help: the levers do not all attack the same hardware resource. A processor has two budgets — how many arithmetic operations per second it can do (its compute roofline, in FLOP/s) and how many bytes per second it can move from memory (its bandwidth roofline, in byte/s). An operation's **arithmetic intensity** is the ratio $I = \frac{\text{FLOPs}}{\text{bytes moved}}$. If $I$ is high (lots of math per byte fetched, like a big dense matmul reused across a batch), you are **compute-bound** and the chip's FLOP/s is the wall. If $I$ is low (little math per byte, like batch=1 LLM decode where each weight is read once and used once), you are **memory-bound** and the chip's bandwidth is the wall. The roofline model plots achievable performance as $\min(\text{peak FLOP/s},\; I \times \text{peak byte/s})$ — a ceiling that ramps up with intensity and then flattens. Why does this belong in the taxonomy? Because *a lever only helps if it relieves the budget you are actually against.* Quantization mostly cuts **bytes moved** (smaller weights), so it helps memory-bound ops directly and helps compute-bound ops *only* if the chip also has faster low-precision arithmetic. Architecture and pruning cut **FLOPs**, which helps compute-bound ops but does little for an op that was already starved on bandwidth. This is the single most common reason a "4× smaller" model is not "4× faster" — the lever cut the wrong budget. Keep the roofline next to the figure-2 map; together they tell you both *when* a lever applies and *whether the physics lets it help.*

## 4. The science: the Pareto frontier, formalized

Now we make "better" rigorous, because the whole field is a multi-objective optimization and you cannot reason about composition without a definition of progress.

You are optimizing a model against (at least) two objectives that pull in opposite directions. Call them **quality** $A$ (accuracy, or 1 − error; higher is better) and **cost** $C$ (latency, or model size, or energy; lower is better). A *configuration* is a particular choice of architecture, bit-widths, sparsity pattern, and so on — every knob set. Each configuration is a point $(A, C)$ in this plane.

We need a way to say one configuration is unambiguously better than another. That is **Pareto dominance**. Configuration $p$ **dominates** configuration $q$ — written $p \succ q$ — if $p$ is at least as good on every objective and strictly better on at least one:

$$p \succ q \iff A_p \ge A_q \;\wedge\; C_p \le C_q \;\wedge\; (A_p > A_q \;\vee\; C_p < C_q).$$

In words: $p$ is no worse on accuracy *and* no worse on cost, and beats $q$ on at least one of them. If a configuration is dominated, you should never ship it — there is a strictly better option on the table. The configurations that are **not dominated by anything** form the **Pareto frontier** (or Pareto-optimal set): the set of points where you cannot improve one objective without sacrificing the other. Every sane deployment decision is a choice of a point *on* the frontier; everything inside it is wasted.

With two objectives we can characterize the frontier cleanly. Sort all non-dominated configurations by increasing cost $C_1 < C_2 < \dots < C_k$. Pareto-optimality forces their accuracies to be strictly increasing too, $A_1 < A_2 < \dots < A_k$ — otherwise a cheaper point with equal-or-higher accuracy would dominate a more expensive one. So the frontier is a monotone staircase: as you spend more, you get more, with diminishing returns near the top.

### Along the frontier vs pushing it outward

Here is the distinction that organizes everything about composition. A technique does exactly one of two things:

1. **It moves you along the current frontier.** You pick a cheaper point and pay some accuracy for it. int8 PTQ on a fixed architecture is the textbook case: same model family, you trade ~0.5 points of accuracy for ~3× less memory and ~2–3× less latency. You did not discover a *new* trade-off; you *chose a different point on the one you already had*. Pruning (at fixed architecture) is the same — it walks you down-and-left along the curve.

2. **It pushes the frontier outward.** A genuinely better *family* of configurations appears, dominating part of the old frontier — strictly better accuracy at the same cost, or strictly lower cost at the same accuracy, across a whole range. A better architecture does this. MobileNetV3 is not "ResNet but you chose a cheaper point"; it is a different curve that sits below-and-to-the-right of ResNet's curve almost everywhere. Distillation, when it produces a smarter small model than you could have trained directly, also pushes the frontier out.

Formally: technique $T$ pushes the frontier outward in some cost region if it produces a configuration $p'$ that dominates a *previously Pareto-optimal* point $p$. If $T$ can only ever produce points that are dominated by, or lie on, the existing frontier, it merely moves you along it. This is *the* test to apply to any new paper: **does it relocate me on the current curve, or does it hand me a new curve?** Architecture and distillation papers usually claim the second (and the honest ones prove it with a frontier plot, not a single point); quantization and pruning papers usually deliver the first — they make the *down-and-left* part of the curve reachable cheaply.

Figure 3 draws both moves. On the left, a fixed ResNet family: fp32 → int8 → int4 walks down the same staircase, and the int4 point falls *off* the frontier (dominated — too much accuracy lost for the marginal speed) which is why it is flagged. On the right, switching to MobileNetV3 lifts the whole curve outward, and *then* quantizing walks along the new, better curve to an even cheaper knee.

![A before-after figure contrasting quantization walking down a fixed accuracy-latency curve against an efficient architecture lifting the entire curve outward to a better knee point](/imgs/blogs/a-taxonomy-of-model-compression-3.png)

### The knee, and why it is the real target

You almost never want the extreme ends of the frontier. The cheapest point is usually too inaccurate to ship; the most accurate is the giant model you are trying to escape. You want the **knee** — the region of maximum curvature, where the curve bends from "cheap accuracy gains" to "expensive accuracy gains." Below the knee, a little more budget buys a lot of accuracy; above it, a lot more budget buys almost nothing. The knee is where the marginal trade is fairest.

Concretely, your deployment constraint picks the knee for you. If your hard constraint is "p99 latency ≤ 20 ms on a Jetson Orin Nano," you draw a vertical line at 20 ms and take the **highest-accuracy point that stays left of it.** If your constraint is "accuracy ≥ 74% top-1," you draw a horizontal line and take the **cheapest point that stays above it.** The art of this series is getting that intersection point as far up-and-left as possible — and *that* is exactly what composing levers does: each lever that pushes the frontier out, followed by levers that walk you to the cheap end of the new curve, lands your constraint line on a better point than any single lever could.

One more rigorous note, because the series is meant to be scientific. The "accuracy" and "latency" you plot are *random variables*, not constants. Latency has a distribution (warm vs cold cache, thermal state, scheduler jitter) — which is why we plot p50 *and* p99, and why the roofline post insists you measure batch=1 on a warmed-up, thermally-stable device. Accuracy on your eval set is an estimate of accuracy on the true distribution, with a confidence interval that shrinks like $1/\sqrt{n}$. A "0.3% accuracy drop" measured on 1,000 examples has a standard error around 0.5%, so it is *not distinguishable from zero*. Treat tiny accuracy deltas as noise until your eval set is big enough to resolve them. Pareto comparisons between two configurations that are within each other's error bars are not real comparisons.

## 5. How the levers compose: the heart of it

This is the section the rest of the series leans on. Knowing the four levers is trivia; knowing *how to stack them so the wins multiply instead of cancel* is the skill. Two questions: which combinations are friends, and in what **order** do you apply them?

### Why composition multiplies (when it works)

Recall the closing observation of section 1: the levers charge in *different currencies* and edit *different representations* (section 3). That is precisely why their wins multiply. Compression factors compose multiplicatively when the levers are independent. If distillation gives you a 4× smaller model, pruning removes 50% of *that* model's weights (2×), and int8 quantization is another 4× on what remains, the size win is not $4 + 2 + 4 = 10\times$ — it is $4 \times 2 \times 4 = 32\times$. This multiplicativity is the entire economic case for composition, and it is exactly what Deep Compression exploited to reach 35–49× (section 6).

But multiplicativity only holds when the levers do not *cannibalize* each other's gains. They cannibalize when they fight over the same representation. Two examples of conflict: aggressive unstructured pruning followed by quantization can leave so few nonzeros that the quantization grid is wasted on a near-empty tensor; and pruning *after* you have committed to a quantization calibration invalidates that calibration (more on this below). The job is to choose pairs that are friends and to order them so each one's gains survive the next.

It is worth being precise about what "independent" means here, because it is the hinge of the whole section. Two levers compose multiplicatively on a resource when each one's *fractional* reduction of that resource is unaffected by the other. Size is the friendliest resource: storage bytes factor cleanly as (params) × (bytes per param), so pruning (which cuts params) and quantization (which cuts bytes per param) attack orthogonal factors and multiply almost perfectly — a 2× param cut and a 4× bit cut really do give ~8× on disk. Latency is *less* friendly, because the two levers can target the same wall. If your op is compute-bound, both a FLOP-cutting lever and a faster-arithmetic lever are pushing on the *same* compute roofline, and once you hit the memory roofline (section 3's arithmetic intensity), further FLOP cuts stop helping — the second lever's "2×" evaporates because the bottleneck moved. Accuracy is the *least* friendly: accuracy losses are not independent at all. Each lever spends some of the model's redundancy budget, and the budget is shared. Distilling to a 4× smaller student already consumed slack; pruning that student 50% spends more of the *same* slack, so the accuracy cost of the second lever is larger than it would have been on the teacher. The practical consequence is a rule you should tattoo somewhere: **multiply the size wins with confidence, multiply the speed wins only after checking the roofline, and never multiply the accuracy costs — add them and then measure, because they interact.**

### The friendly pairs

**QAT + distillation** is a star pairing. Distillation produces a small student; QAT makes that student robust to int8 (or int4) rounding. You can do them in sequence (distill to fp32, then QAT) or *jointly* — "quantization-aware distillation," where the student is trained with fake-quant nodes inserted *and* a distillation loss to the teacher, so it simultaneously learns to be small, accurate, and quantization-friendly. The teacher's soft targets are especially valuable here because they give the low-precision student a denser learning signal to compensate for the information it loses to rounding.

**Pruning + quantization** is the other classic. Prune to remove redundant weights, then quantize what survives. They compose multiplicatively on size and they target different things (which weights exist vs how precisely each is stored), so they do not cannibalize — *as long as the order is right.* This pairing is the backbone of Deep Compression.

**Distillation + efficient architecture** is almost a tautology: the student in a distillation setup *is* an efficient architecture. The cleanest version is to design (or NAS-search) a cheap student topology and *then* distill the big teacher into it, so you get the architectural frontier-push *and* the accuracy recovery from soft targets in one move. This is how a lot of production on-device models are actually built.

**Any lever + the compiler** always composes, and you should *re-run the compiler after every lever*. Compression changes the optimal kernels: after quantization the runtime can pick int8 kernels and fuse the dequant; after pruning a structured-sparse kernel may now win. Forgetting to recompile is how teams leave a free 1.5–2× on the floor.

### The order matters — and here is the default

The ordering principle falls straight out of the section-3 gradient (*early = portable + expensive, late = cheap + hardware-locked*): **apply the levers from the one that changes the model family down to the one that is most coupled to hardware.** Each later step then calibrates against the model the earlier steps actually produced. The default ordering, with reasoning:

1. **Architecture first.** It changes the model family and pushes the frontier out. Everything downstream operates on whatever architecture you pick, so picking the wrong one and then compressing it is polishing the wrong object. Decide the family before you fit weights.
2. **Distillation second** (or fused with training). It needs a full training run anyway, so do it while you are training. The output is a trained, dense, small fp32 model — a clean target for the cheaper levers.
3. **Pruning third.** It edits structure and usually wants a fine-tune to recover. Do it on the trained model, before you lock in low precision, so the fine-tune happens in float where gradients are well-behaved.
4. **Quantization last** (of the model-altering levers). It is the cheapest, fastest, and *most hardware-coupled* step, and — critically — it should **calibrate against the exact weights that will ship.** If you quantize and *then* prune, the pruning changes the weight distribution and your carefully-estimated quantization ranges are now stale; the scales were fit to a distribution that no longer exists. Quantizing last means the calibration sees the final, pruned, fine-tuned weights and fits tight scales to them.
5. **Compile after every step, and especially at the end.** Lower the final model to the runtime, fuse, pick kernels, select delegates.

Figure 4 is this ordering as a timeline. The arrow of the diagram *is* the argument: you move from the most abstract, most portable, most expensive-to-change decision (architecture) to the most concrete, most hardware-locked, cheapest-to-redo decision (quantization, then compilation). Calibrate-last is the load-bearing rule.

![A timeline showing the default compose ordering from architecture to distillation to pruning to quantization to compilation, moving from portable to hardware-coupled steps](/imgs/blogs/a-taxonomy-of-model-compression-4.png)

### Failure modes of bad ordering

The failure modes are instructive because they reveal *why* the default is the default:

- **Quantize then prune.** The headline mistake. Your int8 calibration estimated activation/weight ranges on the dense model. Pruning then zeros a chunk of those weights, shifting the distribution and often the dynamic range. The scales are now mis-fit — too loose (wasting resolution) or too tight (clipping survivors). You lose accuracy you did not need to. Figure 7 contrasts this directly with the right order.
- **Quantize then distill.** Distillation is a training process; running it after quantization either means training in low precision without QAT machinery (unstable) or it means you re-introduce float weights and your quantization was pointless. Distill first.
- **Prune then NAS / change architecture.** You spent a fine-tune budget pruning a model you then threw away. Architecture is a family decision; make it before you invest in compressing a specific member of the family.
- **Compile then compress.** Some teams export to a runtime engine, then try to quantize the engine. Most toolchains want the quantization decided *before* the final lowering so the compiler can fuse the quantize/dequantize and select integer kernels. Compress in the framework, *then* compile.

Figure 7 puts the canonical failure (quantize-then-prune) side by side with the correct order so the calibration argument is visual: the bad path ends with stale scales and a ~2.5% accuracy hit; the good path calibrates against the final weights and lands at ~0.4%.

![A before-after figure contrasting quantize-then-prune, which leaves stale calibration scales, against prune-then-quantize, which calibrates against the final shipping weights](/imgs/blogs/a-taxonomy-of-model-compression-7.png)

There are exceptions, of course — the most common being that QAT is *interleaved* with the training/distillation step rather than run strictly after pruning (you train a model that is simultaneously distilled, sparsity-regularized, and quant-aware), and one-shot post-training pruners like SparseGPT deliberately skip the fine-tune to stay cheap on huge LLMs. But "architecture → distill → prune → quantize → compile, and calibrate last" is the ordering to deviate *from on purpose*, not to stumble past by accident.

## 6. Deep Compression: the canonical compose-everything case study

If you read one paper to internalize composition, read Han, Mao, and Dally's *Deep Compression* (ICLR 2016). It is the cleanest demonstration that stacking levers multiplies, and it is the reason this whole framing exists. The pipeline is three levers in sequence:

1. **Prune.** Remove low-magnitude connections, then fine-tune the survivors, then prune again — iteratively. On AlexNet this removed ~89% of weights (9× fewer); on VGG-16, ~92% (13× fewer) — with **no loss of accuracy**, because the fine-tune lets the remaining weights absorb the slack.
2. **Quantize with weight sharing.** Cluster the surviving weights per layer into a small codebook (e.g. 256 centroids → 8 bits, or 32 → 5 bits) using k-means, store each weight as an *index* into the codebook, and fine-tune the centroids. This is quantization, but the "grid" is data-driven rather than uniform. Convolutional layers went to ~8 bits, fully-connected layers to ~5 bits.
3. **Huffman-code the indices.** The codebook indices are not uniformly distributed — some centroids are used far more than others — so entropy coding compresses the index stream further, for free, with no accuracy effect at all (it is lossless).

The result: **AlexNet 35× smaller (240 MB → 6.9 MB), VGG-16 49× smaller (552 MB → 11.3 MB), with no loss of accuracy.** That number only happens because the factors multiply: pruning's 9–13× times quantization's ~4–6× (from 32-bit floats to 5–8-bit indices) times Huffman's ~1.2–1.4×. Add them and you would expect ~15×; multiply them and you get ~35–49×. Composition is *multiplicative*, and Deep Compression is the proof.

Figure 6 stacks the three stages and shows the factors compounding from a 240 MB dense model down to a 49×-smaller artifact.

![A stack figure showing the Deep Compression pipeline of pruning, weight-sharing quantization, and Huffman coding compounding to a 49x smaller model with no accuracy loss](/imgs/blogs/a-taxonomy-of-model-compression-6.png)

Two honest caveats that the original paper is careful about and that matter for *this* series' "results must be real" mandate. First, Deep Compression is a **storage / memory** win, not automatically a *latency* win — the pruning is unstructured, so realizing speed required the custom EIE accelerator the same group built next. On a commodity phone CPU, the unstructured-sparse model is small but not necessarily fast. This is the single most important asterisk in the whole field and we return to it in the anti-patterns. Second, the "no loss of accuracy" holds for those specific networks and tasks circa 2016; modern over-parameterized models behave similarly, but you must always *measure on your own task*, because the redundancy that pruning exploits is task- and architecture-dependent.

## 7. The practical decision framework

Theory is a map; you still have to drive. Here is the decision procedure I actually use, reduced to inputs and a short algorithm, then a code-shaped sketch of the pipeline.

**The four inputs.** Before touching a lever, write down:

1. **Target hardware** — the *named* chip and its capabilities. Does it have int8 matmul? int4? 2:4 sparse tensor cores? An NPU/delegate, or just a CPU? How much SRAM/RAM? (This decides which levers even *can* help — see the roofline post.)
2. **Accuracy budget** — the floor you cannot drop below, with a *measured* confidence interval (section 4: tiny deltas are noise).
3. **The binding constraint** — is it *size/memory* (model does not fit), *latency/compute* (too slow), or *energy* (battery)? You usually have one dominant constraint; optimize that, watch the others.
4. **Retraining budget** — none (PTQ/one-shot only), a fine-tune (pruning, light QAT), or a full training run (distillation, NAS). This is the gate that eliminates half the options immediately.

**The procedure.** Given those, the order to *reach* for levers (distinct from the order to *apply* them once chosen) is driven by the binding constraint and the budget:

- **If size/memory-bound:** quantization first — it is the cheapest size win (4× at int8, no retrain) and often enough on its own. If still over budget and you have a fine-tune, add structured pruning. If no retrain budget, stop at quantization and accept the result or change targets.
- **If latency/compute-bound:** check the roofline first. If the op is *memory-bound*, quantization (less weight traffic) helps and more FLOP-cutting does not. If *compute-bound*, the big wins are architectural — efficient arch + distillation — then N:M sparsity on supporting HW, then compile. Quantization helps here only if the HW has faster low-precision compute units.
- **If energy-bound:** energy tracks data movement more than FLOPs, so quantization (smaller weights, less DRAM traffic) is usually the biggest single lever; then architecture to cut total work.
- **Always:** sit on a good compiler/runtime before *and* after; re-profile after every lever; never ship a dominated configuration.

Figure 8 is this procedure as a decision tree: the binding constraint and the retraining budget pick your first lever.

![A decision tree that routes from the binding constraint and retraining budget to a first lever, sending memory-bound work to quantization and compute-bound work to efficient architecture and distillation](/imgs/blogs/a-taxonomy-of-model-compression-8.png)

### A code-shaped pipeline sketch

Here is the *shape* of the composed pipeline in the real PyTorch / Optimum / ONNX Runtime toolchain — architecture and distillation up front (a normal training run), then pruning, then quantization, then export and compile. It is deliberately a skeleton you fill in; the point is the *sequence* and which API does which step.

```python
import torch
import torch.nn.utils.prune as prune
from torch.ao.quantization import quantize_dynamic, get_default_qconfig, prepare, convert

# STEP 1-2: architecture + distillation happen in your training loop.
# `student` is an efficient architecture (e.g. a MobileNet-ish or a small transformer);
# `teacher` is the big accurate model. Distillation loss combines task + soft targets.
def distill_step(student, teacher, x, y, T=4.0, alpha=0.7):
    with torch.no_grad():
        zt = teacher(x)
    zs = student(x)
    task = torch.nn.functional.cross_entropy(zs, y)
    soft = torch.nn.functional.kl_div(
        torch.log_softmax(zs / T, dim=-1),
        torch.softmax(zt / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)
    return (1 - alpha) * task + alpha * soft
# ... train `student` with distill_step until converged. Now we have a small dense fp32 model.

# STEP 3: structured pruning, then a fine-tune (run the fine-tune in float).
for module in student.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(module, name="weight", amount=0.3, n=2, dim=0)  # drop 30% of filters
        prune.remove(module, "weight")  # bake the mask into the weights
# ... fine-tune `student` for a few epochs to recover the accuracy the prune cost.

# STEP 4: quantization LAST, calibrating against the final pruned+finetuned weights.
student.eval()
student.qconfig = get_default_qconfig("x86")   # or "qnnpack" for ARM mobile
prepared = prepare(student)                      # insert observers
for x, _ in calibration_loader:                  # a few hundred representative batches
    prepared(x)                                  # observers estimate activation ranges
int8_model = convert(prepared)                   # fold to true int8

# STEP 5: export + compile to the runtime engine (ONNX Runtime / TensorRT / TFLite).
torch.onnx.export(int8_model, example_input, "model_int8.onnx", opset_version=17)
# then: onnxruntime / trtexec / TFLiteConverter picks integer kernels and fuses quant/dequant.
```

For an LLM the toolchain swaps but the *order* is identical. The architecture is given (you do not redesign Llama), so you go straight to weight-only quantization with a calibration-based quantizer, optionally prune, then run on a quantized-kernel runtime:

```bash
# 7B LLM for a laptop: weight-only 4-bit with llama.cpp's k-quants.
# (Architecture is fixed; this is the "walk along the frontier" lever.)
python convert_hf_to_gguf.py ./Llama-3.1-8B --outfile llama-8b-f16.gguf --outtype f16
./llama-quantize llama-8b-f16.gguf llama-8b-Q4_K_M.gguf Q4_K_M   # ~4.5 bits/weight effective
./llama-cli -m llama-8b-Q4_K_M.gguf -p "Explain the Pareto frontier" -ngl 0 -t 8
# -ngl 0 keeps it on CPU; raise it to offload layers to a GPU/Metal backend if present.
```

The `optimum` equivalent for an ONNX Runtime deployment (e.g. a small BERT on a CPU box) keeps the same calibrate-last discipline through a `CalibrationDataReader`:

```python
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

quantizer = ORTQuantizer.from_pretrained("./distilbert-pruned-onnx")
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=True)  # int8, VNNI kernels
# static quant runs calibration over a representative set before folding to int8:
quantizer.quantize(save_dir="./distilbert-int8", quantization_config=qconfig,
                   calibration_dataset=calibration_ds)
```

Notice the through-line in all three: the model-altering work (distill, prune) finishes first; quantization calibrates against the *final* weights; the runtime/compile step comes dead last and is told the model is integer so it can pick the right kernels.

## 8. Two worked examples

Numbers make the framework real. Two scenarios, both with stacked before→after results. The figures here are defensible orders of magnitude drawn from the public literature; *measure your own on your own target* before you trust them in a design doc.

#### Worked example: A CNN image classifier for a phone

**Setup.** You have a ResNet-50 image classifier hitting 76% top-1 on your task. It is 98 MB (fp32) and runs at ~50 ms/inference, batch=1, on a Pixel-class mobile CPU. Product needs ≤ 15 ms p99 and the app budget allows ~10 MB for the model. Accuracy floor: 73% top-1. Retraining budget: a full run is available.

**Reasoning through the levers (in apply-order).** Latency-bound and compute-bound (a CNN on a CPU is doing real FLOPs), with a hard size cap too. The roofline says ResNet-50's convs are compute-heavy, so the biggest win is *architectural*. So:

1. **Architecture + distillation first.** Switch the family to a MobileNetV3-Large-ish student and distill ResNet-50 into it. The student is ~16 MB fp32 and ~9 ms; distillation recovers it to ~75% top-1 (better than training the small net cold, because the teacher's soft targets carry the "dark knowledge"). This is the frontier-push.
2. **Pruning — optional, skipped here.** MobileNetV3 is already lean; structured pruning would buy maybe 1.2× more at the cost of another fine-tune and some accuracy. We are not size-desperate after step 4, so we skip it (refusing a win we do not need — section 1's discipline).
3. **Quantization last.** int8 PTQ with a 500-image calibration set. ~4× smaller and ~2× faster on the phone's int8 path, costing ~0.5 points.
4. **Compile.** Convert to TFLite with `Optimize.DEFAULT` + a `representative_dataset`, deploy through the int8 delegate.

**Stacked results:**

| Stage | Top-1 | Size | Latency (p99, mobile CPU) |
| --- | --- | --- | --- |
| ResNet-50 fp32 (start) | 76.0% | 98 MB | ~50 ms |
| → MobileNetV3 (distilled) | 75.0% | 16 MB | ~9 ms |
| → + int8 PTQ | 74.5% | 4.3 MB | ~4.5 ms |
| → + TFLite int8 delegate compile | 74.5% | 4.3 MB | ~3.8 ms |

We land at **74.5% top-1, 4.3 MB, ~3.8 ms p99** — inside every constraint, with ~1.5 points of accuracy spent. The size win is multiplicative ($6\times$ from architecture × $\sim 4\times$ from int8 $\approx 23\times$ total, 98 → 4.3 MB) and the latency win is too. Crucially, the architecture lever did the heavy lifting (50 → 9 ms); quantization was the cheap polish on top. Reaching for int4 here would be the over-reach: it would risk the accuracy floor for a latency we no longer need.

#### Worked example: A 7B LLM for a laptop

**Setup.** A 7B-parameter LLM. fp16 weights are ~14 GB — it does not fit in a 16 GB laptop's RAM alongside the OS and the app, and even if it did, decoding is **memory-bandwidth-bound** (each token's matmuls re-read all the weights from memory, so latency is dominated by weight traffic, not arithmetic). You want it to fit in ~5 GB and decode at an interactive rate (≥ ~15 tokens/s) on an M2-class laptop. Accuracy floor: a perplexity/benchmark regression you can tolerate (say ≤ ~1% relative on your eval). Retraining budget: essentially none — you are not pretraining a 7B.

**Reasoning through the levers.**

1. **Architecture is given.** You are not redesigning the transformer. The "efficient architecture" lever was already pulled by whoever designed the model; you inherit it. So this is purely a *move-along-the-frontier* exercise.
2. **Quantization is the workhorse, and it is the *right* lever** because decode is memory-bound: cutting weights from 16 bits to ~4 bits cuts the memory traffic per token ~4×, which directly cuts decode latency on a bandwidth-bound op. Weight-only 4-bit (GGUF `Q4_K_M`, or GPTQ/AWQ int4) takes 14 GB → ~4.1 GB. AWQ/GPTQ-style calibration protects the salient weight channels so the perplexity hit stays small (sub-1% relative on common benchmarks for good int4 methods).
3. **Pruning — maybe.** A one-shot post-training pruner (SparseGPT/Wanda) at ~50% unstructured sparsity is *possible* with no fine-tune, but on a laptop CPU/Metal backend you get the *size* win and little *speed* win (unstructured sparsity, no fast kernel). 2:4 structured would speed up on a sparse-tensor-core GPU you do not have here. So for *this* target, skip pruning — it would be the unstructured-sparsity anti-pattern (section 9).
4. **Runtime/compile.** `llama.cpp` with k-quant kernels and Metal offload, KV-cache in a smaller dtype.

**Stacked results:**

| Stage | Eval (rel.) | Memory | Decode (M2 laptop) |
| --- | --- | --- | --- |
| 7B fp16 (start) | baseline | ~14 GB | does not fit / swaps |
| → Q4_K_M weight-only | −0.6% | ~4.1 GB | ~22 tok/s |
| → + KV-cache int8 + Metal compile | −0.6% | ~4.4 GB total | ~28 tok/s |

We land at **~4.4 GB and ~28 tok/s with a 0.6% eval regression** — fits, interactive, and the regression is inside budget. The lesson is the *opposite* of the CNN example: here the architecture lever was unavailable (given model), pruning was useless (wrong sparsity for the hardware), and quantization did everything — *because* the binding constraint (memory bandwidth) is exactly the one quantization attacks. Same taxonomy, completely different lever choice, all because the constraint and the hardware differ. That is the framework earning its keep.

**Stress-testing the LLM decision.** A framework you cannot break is a framework you do not understand, so push on it. *What if you go to int3 or int2?* The SQNR law from section 2.1 warns you: dropping from ~4 bits to ~2 bits sheds another ~12 dB, and for a 7B model that typically tips perplexity past the budget — int2 weight-only is currently the edge of viability and usually needs more than naive PTQ (extra calibration, mixed precision keeping salient layers at higher bits). *What if the calibration set is tiny?* Weight-only int4 methods like GPTQ/AWQ use a few hundred sequences; shrink that to a handful and the per-channel scales overfit to a non-representative slice of the distribution, and you see the regression balloon on out-of-domain prompts even though your tiny eval looked fine — a textbook case of an accuracy delta that is inside the noise band on a small set (section 4) and very real on the true distribution. *What if the laptop has a GPU with sparse tensor cores after all?* Then the calculus flips: 2:4 structured pruning suddenly buys real decode speed, and you would revisit the "skip pruning" decision — the lever choice is a function of the *target*, not the model, exactly as the taxonomy insists. *What if an op falls back to CPU?* If the runtime cannot run a quantized op on the accelerator and silently dequantizes it to run on the CPU, you can lose the entire speed win and add a costly host-device copy — which is why you re-profile after compiling, never before. Each of these stress tests is the same framework run with different inputs; none of them requires a new idea, only the discipline to ask "which budget binds, and does this lever relieve it?"

## 9. Anti-patterns: when composing goes wrong, and when not to bother

The flip side of "how they compose" is "how composition fools you." These are the mistakes I have personally made or been paged about. Each one is a misreading of the map.

**Unstructured sparsity expecting a speedup on commodity hardware.** You prune a model to 80% zeros, the parameter count plummets, and the latency on your phone CPU does not move at all. Why: a dense matmul kernel still iterates over all the (now-zero) entries — multiplying by zero takes the same cycles as multiplying by anything else, and there is no fast sparse kernel below very high sparsity. Unstructured sparsity is a *storage/memory* win, not a *compute* win, on commodity HW. If you want a speed win from sparsity you need **structured** pruning (genuinely smaller dense tensors) or **N:M** on hardware with sparse tensor cores. Deep Compression's "no accuracy loss at 49× smaller" is real and is *also* unstructured — which is why its authors had to build a custom accelerator (EIE) to get the speed. Read every pruning result asking "smaller, or faster, and on what silicon?"

**Quantizing an already memory-bound op and expecting a compute speedup.** Closely related and even sneakier. LLM decode is memory-bound (good — quantization helps via less traffic). But if you quantize a layer that was *compute*-bound on hardware that has *no faster integer compute path*, you get the size win and *zero* latency win, because you did not relieve the actual bottleneck. Worse, some runtimes insert dequantize ops that *add* compute. Always check the roofline: quantization helps a memory-bound op (less data moved) and a compute-bound op *only if* the chip has faster low-precision arithmetic. Optimizing the wrong side of the roofline is the most common wasted week in this field.

**Double-counting wins.** Two papers each claim "2× speedup," you stack them, you write "4× faster" in the design doc, and the measured number is 2.3×. Why: their baselines overlapped, or they sped up the same bottleneck and only one of them could "win" it, or one measured FLOPs (a proxy) and the other measured wall-clock. Compression factors multiply *only when the levers are independent and act on different resources*; two levers that both relieve the same bottleneck do not stack. **Never trust a composed number you did not measure end to end on the target.** The taxonomy's value here is that it tells you *when* levers are independent (different cells of the figure-2 map) and therefore when multiplication is even plausible.

**Bad ordering (re-stated because it is so common).** Quantize-then-prune wastes the calibration (section 5, figure 7). Compile-then-compress fights the toolchain. NAS-after-pruning throws away a fine-tune. The fix is always the same: architecture → distill → prune → quantize → compile, and calibrate last.

**QAT when PTQ already hits target.** QAT is a *training run* with extra machinery. If int8 PTQ already lands inside your accuracy budget — which it does for most CNNs and well-behaved transformers — running QAT is pure cost for no benefit. Reach for QAT only when PTQ misses (int4, outlier-heavy activations, a stubborn architecture). Refusing a technique you do not need is a skill, not laziness.

**Optimizing a model that was never the bottleneck.** Sometimes the model is 8 ms and the image preprocessing is 40 ms, or the network round-trip dwarfs everything. Profile the *whole system* first. The most elegant compression in the world is worthless if the model is not on the critical path. This is why profiling is in the taxonomy as a first-class discipline, not an afterthought.

**When not to bother at all.** If the model already fits and runs inside budget, *ship it* — every lever is an accuracy risk and an engineering cost. If you have no retraining budget, do not plan a distillation/NAS strategy; you are limited to PTQ and one-shot pruning, so design around that. If your hardware has no low-precision compute path, do not expect quantization to speed anything up (it will still save memory). And if the win you would chase is inside the eval-set noise band (section 4), there is nothing to chase. The taxonomy is as much about *declining* levers as pulling them.

## 10. Case studies: real numbers from the literature

To ground the framework in shipped or published results — and to honor this series' "results must be real" rule — four named data points, each illustrating a lever or a composition. Treat the figures as the literature reports them; your mileage on your task and chip will differ.

**Deep Compression (Han et al., ICLR 2016)** — the compose-everything case study, covered in section 6. Prune + weight-sharing quantization + Huffman → AlexNet 35×, VGG-16 49× smaller, no accuracy loss. The asterisk: storage win, not automatically latency, because the pruning is unstructured. This is the paper that taught the field that compression factors multiply.

**DistilBERT (Sanh et al., 2019)** — the distillation case study. A 6-layer student distilled from 12-layer BERT-base: ~40% fewer parameters, ~60% faster inference, retaining ~97% of BERT's GLUE score. Pure topology lever; the student is a normal dense model that you can *then* quantize for another ~4×, which several follow-ups did. This is "distillation + quantization compose" in production form.

**MobileNetV3 (Howard et al., 2019)** — the efficient-architecture / hardware-aware-NAS case study. Designed with latency on a Pixel phone *in the loop* (not FLOPs as a proxy), MobileNetV3-Large matched prior accuracy at materially lower on-device latency. This is the frontier-*push* lever made concrete: a new curve, not a cheaper point on the old one. Quantizing it (int8 TFLite) walks along the new curve, exactly as in worked example 1.

**GPTQ / AWQ on LLaMA-class models (2022–2023)** — the modern LLM quantization case study. Calibration-based weight-only int4 quantization that protects salient weights (AWQ scales channels by activation magnitude; GPTQ uses second-order information to minimize layerwise error) brings 7B–70B models to ~4 bits/weight with small perplexity regressions, making them fit and run on consumer hardware. This is worked example 2's lever, and the reason a 7B model runs on a laptop at all.

A fifth honorable mention for completeness: **2:4 structured sparsity on NVIDIA Ampere+** delivers a real ~2× dense-matmul throughput on Sparse Tensor Cores — the rare case where sparsity buys *speed* on commodity (well, datacenter) hardware, precisely because it is structured enough for the kernel to exploit. It is the existence proof that the "sparsity ≠ speed" anti-pattern is about *unstructured* sparsity on *unsupporting* hardware, not about sparsity per se.

## 11. The comparison matrix and the compose-order table

Two reference tables you will come back to. First, the lever comparison — the same content as figure 5, in text so you can copy it into a design doc:

| Lever | What it cuts | Accuracy risk | HW dependence | Needs retraining | Typical win |
| --- | --- | --- | --- | --- | --- |
| Quantization | bits per value (size + bandwidth) | low–medium (int8) / high (naive int4) | **high** (needs int8/int4 units to speed up) | no (PTQ) / yes (QAT) | ~4× size, 2–4× speed |
| Pruning (unstructured) | nonzero params | medium | **high** (no speed without sparse kernel) | yes (fine-tune) | size only on commodity HW |
| Pruning (structured / N:M) | whole channels/heads; structure | medium | medium (N:M needs sparse cores) | yes (fine-tune) | 1.5–2× speed reliably |
| Distillation | model topology (size) | low | none (student is dense) | **yes (full run + teacher)** | 2–10× smaller, ~97% acc |
| Efficient arch / NAS | the operations (FLOPs + latency) | low | low (it is just a model) | **yes (from scratch)** | largest; pushes frontier out |
| Compiler / runtime | schedule, kernels, layout | none | (it *is* the hardware adaptation) | no | free 1.5–2× |

Figure 5 is this table as a matrix so the pattern jumps out: read down the "HW dependence" column and you see the cheap levers (quantization, sparsity) are the *most* hardware-coupled, while the expensive levers (distillation, architecture) are the *least*. That inverse relationship — pay in retraining to get hardware portability, or pay in hardware-lock to skip retraining — is the deep structure of the whole field.

![A matrix comparing the four levers across what they cut, accuracy risk, hardware dependence, and retraining cost, showing cheaper levers carry more hardware dependence](/imgs/blogs/a-taxonomy-of-model-compression-5.png)

Second, the compose-order quick reference:

| Order | Lever | Why here | Calibration note |
| --- | --- | --- | --- |
| 1 | Efficient architecture / NAS | changes the family; pushes frontier out | everything downstream operates on this choice |
| 2 | Distillation | needs a training run anyway; recovers accuracy | produces the dense fp32 model the rest consume |
| 3 | Pruning (struct / N:M) | edits structure; fine-tune in float | do it before low precision |
| 4 | Quantization | cheapest, most HW-coupled, last | **calibrate against the final shipping weights** |
| 5 | Compile / runtime | lower, fuse, pick kernels, select delegates | re-run after *every* step |

## 12. When to reach for this — a decisive recommendation

Pulling the whole post into a stance you can act on:

- **Start by profiling, not compressing.** Find the actual bottleneck and which side of the roofline it sits on. Half of "compression" projects are solved by fixing a non-model bottleneck or just turning on a better runtime.
- **Match the lever to the binding constraint.** Memory-bound → quantization. Compute-bound → architecture/distillation then structured sparsity. Energy-bound → quantization (data movement dominates energy). Do not pull the lever that does not attack your constraint.
- **Spend retraining budget where it buys the most.** If you have a full training run, the architecture/distillation lever pushes the frontier out and is the highest-leverage thing you can do. If you have only a fine-tune, structured pruning. If you have nothing, PTQ and one-shot pruning.
- **Apply in the default order and calibrate last.** Architecture → distill → prune → quantize → compile. Deviate only on purpose (e.g. fused QAT-distillation, or skipping the fine-tune for a one-shot LLM pruner).
- **Re-profile after every lever and refuse wins you do not need.** Every technique is a cost. If you are already inside budget, stop. If a delta is inside the eval noise band, ignore it. If int8 PTQ hits target, do not QAT.
- **Never trust a composed number you did not measure end to end on the target chip.** Multiplicativity is a hypothesis you verify, not a guarantee.

## 13. Key takeaways

- **There are four levers, not forty techniques.** Quantization (values), pruning (structure), distillation (topology), efficient architecture/NAS (ops) — plus the compiler substrate and the profiling discipline. Every paper slots into one cell of that map.
- **The second axis is when × what.** Levers separate by pipeline stage (design → train → post-train → compile → runtime) and by representation changed. Early = portable + expensive to change; late = cheap + hardware-locked.
- **"Better" means Pareto-dominant.** $p \succ q$ iff $p$ is no worse on both accuracy and cost and strictly better on one. Ship points *on* the frontier; target the knee your constraint picks out.
- **A technique either moves you along the frontier or pushes it outward.** Quantization and pruning move you along; better architecture and good distillation push it out. Ask every new paper which one it does.
- **Composition multiplies — when levers are independent.** $4\times \cdot 2\times \cdot 4\times = 32\times$, not $10\times$. Deep Compression's 35–49× is the proof. But it only multiplies when the levers attack different resources.
- **Order matters; calibrate last.** Architecture → distill → prune → quantize → compile. Quantize-then-prune wastes the calibration; that single mistake costs accuracy for free.
- **Sparsity ≠ speed on commodity hardware.** Unstructured pruning is a memory win; only structured/N:M sparsity on supporting silicon buys latency.
- **Quantization helps a memory-bound op, and a compute-bound op only if the chip has faster low-precision math.** Check the roofline before you celebrate.
- **The hardest skill is declining levers.** If you are inside budget, or the delta is noise, or PTQ already works, do not pull more.

## 14. Further reading

- **Song Han, Huizi Mao, William J. Dally — *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding* (ICLR 2016).** The canonical compose-everything paper; 35–49× by stacking three levers.
- **Victor Sanh et al. — *DistilBERT, a distilled version of BERT* (2019).** The reference distillation result: 40% smaller, 60% faster, ~97% of the accuracy.
- **Andrew Howard et al. — *Searching for MobileNetV3* (ICCV 2019).** Hardware-aware NAS; the efficient-architecture lever pushing the frontier with on-device latency in the loop.
- **Torsten Hoefler et al. — *Sparsity in Deep Learning: Pruning and growth for efficient inference and training* (JMLR 2021).** The definitive survey on the pruning/sparsity lever, including why structure matters for speed.
- **MIT *EfficientML.ai* (Song Han's course / lecture notes).** The best single curriculum tying all four levers together with hardware.
- **Official runtime docs** — TensorFlow Lite / LiteRT quantization guide, ONNX Runtime quantization, NVIDIA TensorRT developer guide, and the `llama.cpp` quantization README — for the toolchain APIs the code sketches use.
- **Within this series:** the [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) (the capstone that puts this taxonomy to work end to end), [the roofline model and where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), and [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device). Bookmark this taxonomy page; every technique post in the series points back to it.

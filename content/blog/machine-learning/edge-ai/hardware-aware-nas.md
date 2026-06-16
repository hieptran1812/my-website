---
title: "Hardware-aware NAS: searching for the latency you actually ship"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Put measured on-device latency into the architecture search objective so the network you find is fast on the actual chip, not on paper — with MnasNet's reward, differentiable latency, ProxylessNAS, and Once-for-All, in runnable code."
tags:
  [
    "edge-ai",
    "model-optimization",
    "neural-architecture-search",
    "latency",
    "nas",
    "inference",
    "efficient-ml",
    "mobilenet",
    "once-for-all",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/hardware-aware-nas-1.png"
---

A team I worked with spent three weeks running a clean, textbook architecture search. The objective was the one everybody writes down first: maximize validation accuracy per million FLOPs. The search dutifully found a beautiful little network — heavily fragmented, lots of grouped convolutions, a depthwise zoo, the kind of thing that looks gorgeous in a FLOP table. It had 30% fewer FLOPs than the MobileNetV2 baseline they were trying to beat. Everyone was thrilled until we put it on the actual phone. It was 1.6x *slower* than the baseline it was supposed to replace. The FLOP counter had been optimizing a number the chip does not care about.

That failure is the entire reason this post exists. Plain neural architecture search — the kind covered in [neural architecture search basics](/blog/machine-learning/edge-ai/neural-architecture-search-basics) — optimizes accuracy against a *proxy* for cost, almost always FLOPs or parameter count. But FLOPs are a lie about speed, and we have a whole post on exactly why: [EfficientNet, ShuffleNet, and the FLOPs–latency gap](/blog/machine-learning/edge-ai/efficientnet-shufflenet-and-the-flops-latency-gap). Two blocks with identical FLOPs can differ 2x in measured latency on the same chip because one is compute-bound and the other drowns in memory traffic. If your search never times the real hardware, it will happily walk straight into the slow corner of the design space and hand you a model that wins on paper and loses on the device.

Hardware-aware NAS fixes this by doing the obvious-in-hindsight thing: it puts **measured latency on the target device** into the search objective. Not FLOPs, not a multiply-accumulate count, not a guess — the actual milliseconds the actual chip takes, captured in a lookup table or a small predictor and fed back into the search every iteration. The network it finds is fast on *that* SoC (system-on-chip, the integrated processor in a phone or board), because that SoC's latency is literally what it optimized. The figure below is the whole thesis in one picture: same search space, same training, two objectives, two completely different winning architectures and two different latencies.

![A two-column comparison showing a FLOPs-objective search picking a fragmented low-FLOP block that runs slow versus a latency-objective search picking a fused block that runs fast on the same Pixel CPU](/imgs/blogs/hardware-aware-nas-1.png)

By the end of this post you will be able to: write down the MnasNet multi-objective reward and explain why its soft power-law shape beats a hard latency cap; build a tiny latency lookup table by profiling per-op latency on a host and use it to predict whole-net latency; add a *differentiable* latency term to a DARTS-style search the way FBNet and ProxylessNAS do, derived from the same softmax-over-ops relaxation; understand ProxylessNAS path binarization and why it lets you search directly on the target task; and explain Once-for-All — train one supernet, extract a specialized subnet per device with zero retraining. This is the fourth lever of the series — efficient architecture / NAS — and it sits closest to the metal. If you have not seen the unifying frame, start with [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression); this post is where architecture design and deployment finally meet.

## Why FLOPs are the wrong objective (a 90-second recap with teeth)

Let me make the failure concrete before we fix it, because the fix only makes sense once you feel the pain. FLOPs count floating-point operations — multiply-accumulates, mostly. They are a *compute* metric. Latency is what you experience when you call `model(x)` and wait. The two are related, but the constant of proportionality between them is not a constant at all: it depends on whether each layer is compute-bound (the chip is busy doing math) or memory-bound (the chip is idle waiting for data to arrive from DRAM). The roofline model — see [the roofline model: where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) — formalizes this with *arithmetic intensity*, the ratio of FLOPs to bytes moved. A layer with low arithmetic intensity (a depthwise convolution, a tiny 1x1 with few channels) does little math per byte, so it sits in the memory-bound region of the roofline where adding FLOPs is free but adding memory traffic is fatal.

Here is the trap a FLOP-only search falls into. To cut FLOPs, the search loves operators with low arithmetic intensity: depthwise convolutions, grouped convolutions, channel shuffles, fragmented branches. Each of these has a wonderful FLOP-to-accuracy ratio. But on a real mobile CPU or NPU (neural processing unit, the dedicated ML accelerator on the SoC), those same operators are exactly the memory-bound, poorly-utilized, kernel-launch-heavy ops that the hardware hates. Fragmentation in particular — many small parallel branches — destroys parallelism and floods the runtime with kernel launches. So the FLOP-optimal architecture and the latency-optimal architecture are not the same network. They are not even close. That is the gap figure 1 shows, and it is not a small effect: a 30% FLOP reduction turning into a 60% latency *regression* is a story I have personally lived.

There is a second, subtler reason FLOPs mislead: the mapping from FLOPs to latency is **chip-specific and op-specific**, and it changes with input resolution, channel count, and even whether the dimensions are friendly to the vector unit's tile size. A 3x3 convolution at 224x224 might run at 80% of peak on a phone NPU; the same convolution at an awkward channel count that does not align to the NPU's 8-wide or 16-wide MAC array might run at 30%. No FLOP count captures any of this. Only measurement does. So the design principle for everything that follows is brutally simple: **if you care about latency, measure latency, and put the measurement in the loop.**

Let me put one number on the chip-specificity claim, because it is the load-bearing fact of the whole field. Take the same network and time it on three chips: a mobile CPU, a mobile NPU, and a desktop GPU. The *ranking* of which operators are cheapest is not the same across the three. On a desktop GPU, a wide dense convolution with a large channel count is cheap per FLOP, because the GPU has thousands of lanes hungry for parallel work and a wide conv feeds them. On a mobile CPU, that same wide conv is expensive because the CPU has a handful of vector lanes and chokes on the data movement, while a depthwise conv that the GPU found relatively wasteful is comparatively fine. This is why ProxylessNAS, when it searched separately for a mobile CPU, a mobile GPU, and a desktop GPU, found three *different* architectures — the desktop-GPU model is wider and more regular, the mobile-CPU model is thinner with more depthwise structure. A FLOP-only objective is constitutionally incapable of producing that divergence, because FLOPs are the same number on all three chips. The architecture that is right for a chip is a function of that chip, and the only way to make a search see that function is to feed it that chip's measurements.

### Defining the search space so latency even has a chance

Before any objective matters, the search space — the set of architectures the search can possibly reach — has to contain fast networks and has to be expressible in operators the target runtime supports. Hardware-aware methods almost universally use a **layer-wise (or block-wise) macro search space** built on the inverted-residual block from MobileNetV2, because that block is both accurate and hardware-friendly, and because its few knobs map cleanly onto a lookup table. For each block the search chooses among a small menu: kernel size in $\{3, 5, 7\}$, expansion ratio in $\{3, 4, 6\}$, whether to include a squeeze-and-excite module, and the number of blocks repeated in each stage. That is a handful of choices per stage and maybe twenty stages, which still multiplies out to billions of architectures — enough to be interesting, small enough that every choice is a clean LUT key.

The crucial design decision is **what you let into the menu in the first place**. If you include exotic operators that the target NPU cannot execute, the search may pick them and the runtime will fall back to the CPU at deploy time, which (as we will see) can cost more than any FLOP saving. So a disciplined hardware-aware search space is *pre-filtered to natively-supported ops on the target accelerator*. This is not a detail — it is half the engineering. The team that profiles the runtime, confirms which ops run on the NPU versus fall back to CPU, and restricts the search space accordingly is the team whose searched network is actually fast. The team that searches a generic space and discovers fallbacks at deploy time has shipped a slow model with a fast FLOP count. The search space is where hardware-awareness *begins*, before the objective ever enters.

## The multi-objective objective: MnasNet's reward

The cleanest way to make a search latency-aware is to change the reward it maximizes. MnasNet (Tan et al., 2019) did exactly this with a reinforcement-learning controller, but the reward shape is the durable idea — it outlives the search strategy. We want a single scalar to maximize that rewards accuracy *and* punishes slowness, with a knob to control the trade-off.

The naive first attempt is a hard constraint: maximize accuracy subject to latency below a target $T$.

$$
\max_{m}\; ACC(m) \quad \text{subject to}\quad LAT(m) \le T
$$

This is what every product manager writes on a whiteboard, and it is a terrible objective for a search. It is an all-or-nothing cliff. A model at $LAT(m) = T - 1\,\text{ms}$ is fully allowed; a model at $T + 1\,\text{ms}$ is worth exactly zero, no matter how accurate. The search gets no gradient, no signal, no sense of "you are close, keep going." It treats a model that misses the budget by 1 ms identically to one that misses by 100 ms. Reinforcement-learning controllers and evolutionary searches both flounder on this kind of discontinuous reward because they learn from the *shape* of the reward surface, and a cliff has no useful shape.

MnasNet's fix is a **soft, weighted-product reward** with a power law:

$$
\text{reward}(m) \;=\; ACC(m)\;\times\;\left[\frac{LAT(m)}{T}\right]^{w}
$$

with $w < 0$. Look at what this does. When $LAT(m) = T$ exactly, the bracket is $1$ and the reward equals the accuracy — the target latency is the neutral point. When the model is faster than target, $LAT/T < 1$ raised to a negative power is greater than $1$, so the model gets a small bonus on top of its accuracy. When the model is slower than target, $LAT/T > 1$ raised to a negative power is less than $1$, so accuracy gets multiplicatively discounted. There is no cliff: the reward declines smoothly as latency rises, so the search always has a gradient pointing toward faster models. The figure below contrasts the two reward shapes directly.

![A two-column comparison of a hard latency constraint that creates an all-or-nothing cliff at the target versus MnasNet's soft power-law reward that keeps a smooth gradient toward the target latency](/imgs/blogs/hardware-aware-nas-2.png)

### Deriving the trade-off exponent

The exponent $w$ is not arbitrary — it encodes how many points of accuracy you are willing to trade for a percentage of latency. MnasNet chose it by an empirical rule of thumb: they wanted models with roughly the same reward if they sat on a line of "accuracy gained per latency spent." Concretely they observed that doubling latency tended to be worth about a 5% relative accuracy gain in their regime, and solved for $w$ so the reward is approximately flat along that exchange rate.

Set two models equal in reward: model A at $(ACC_A, LAT_A)$ and model B at $(ACC_B, LAT_B)$. Equal reward means

$$
ACC_A \left[\frac{LAT_A}{T}\right]^{w} = ACC_B \left[\frac{LAT_B}{T}\right]^{w}.
$$

Take logs and rearrange, and the iso-reward exchange rate between log-accuracy and log-latency is exactly $-w$:

$$
\frac{\Delta \log ACC}{\Delta \log LAT} = -w.
$$

So if you decide that a $1\%$ relative latency change is worth a $0.07\%$ relative accuracy change, you set $w \approx -0.07$, which is precisely the value MnasNet reports. That is the whole trick: $w$ is your accuracy-per-latency exchange rate, in log-log space, with a sign flip. Pick it from your product's actual tolerance — a real-time camera app that must hit 30 fps has a steep $w$; a batch photo-tagging job that just wants "fast enough" has a gentle one.

One more practical note on the reward: $LAT(m)$ here is the **measured** latency on the target device, captured every time the controller proposes a candidate. In the original MnasNet that meant running the candidate on a farm of actual phones — which is why MnasNet's search cost was enormous (thousands of TPU-hours plus a phone farm). The next two ideas — lookup tables and differentiable latency — are largely about killing that cost while keeping the measured-latency signal.

### Why a weighted product and not a weighted sum

A reasonable person looks at MnasNet's reward and asks: why multiply accuracy by a latency factor instead of subtracting a latency penalty, $ACC(m) - \beta\,LAT(m)$? The subtractive form is what FBNet and ProxylessNAS use later inside a differentiable loss, and it is fine *there*, but for a reward that a controller maximizes, the multiplicative form has a property worth understanding. A subtractive penalty has units: you are subtracting milliseconds from an accuracy fraction, so $\beta$ has to convert one to the other, and its right value depends on the absolute scale of both — a model at 80% accuracy and one at 20% accuracy get the same absolute latency penalty, which over-penalizes the already-good model relative to its accuracy. The multiplicative form is scale-free: it discounts accuracy by a *fraction* that depends only on the latency *ratio* $LAT/T$, so a model that is 10% over budget loses the same *relative* slice of its reward whether it started at 80% or 20% accuracy. That homogeneity is why the power-law product behaves well across the whole accuracy range a search explores, and why the single exponent $w$ — a dimensionless exchange rate — is enough to tune it. When latency lives inside a differentiable loss rather than a black-box reward, the subtractive form's clean gradient wins instead; the two forms are tools for two different optimization settings, and knowing which to reach for is part of the craft.

A second subtlety: MnasNet uses a *single* target $T$ and lets the soft reward pull models toward it from both sides, which means the search will happily return a model slightly *under* budget if it buys accuracy, and slightly *over* if the accuracy gain is large enough. That is usually what you want — a hard real-time deadline is rare; most products have a soft "feels fast" target with a little slack. If you genuinely have a hard deadline (a 33 ms frame budget for 30 fps that you must never blow), the soft reward alone is not safe, and the right move is to combine it: use the soft reward to guide the search but apply a hard reject on any final candidate whose *measured* latency exceeds the deadline. Soft to search, hard to ship.

## Latency lookup tables: a table beats FLOPs

Running every candidate on a real phone is accurate but ruinously slow. The fix that made hardware-aware NAS practical is the **latency lookup table** (LUT), also called a latency predictor. The insight is that a neural network's latency, to a very good approximation, is the *sum of its operators' latencies*:

$$
LAT(m) \;\approx\; \sum_{i \in \text{ops}(m)} \text{LUT}\big[\text{op}_i,\, \text{config}_i\big]
$$

where $\text{config}_i$ captures the things that actually change an op's runtime: input resolution, input and output channel counts, kernel size, stride, and the op type. You measure each distinct $(\text{op}, \text{config})$ **once** on the target device, store it in a table, and from then on you estimate any candidate net's latency by adding up its ops' table entries. No deployment, no phone in the loop, just an O(number-of-ops) sum. The figure below shows the construction.

![A vertical stack showing per-operator latencies profiled once on the target chip and stored as table entries, then summed across a candidate network to predict its whole-net latency](/imgs/blogs/hardware-aware-nas-3.png)

Why does a sum work when FLOPs do not? Because the table entry *already contains* all the chip-specific, memory-bound, utilization, and kernel-launch effects that FLOPs throw away. When you measured "depthwise 3x3 at 28x28, 144 channels" on the phone, that measurement baked in the fact that depthwise convs are memory-bound and underutilize the NPU. The table is, in effect, a learned function from architecture choices to real latency, with the chip's quirks memorized rather than modeled. It is not perfect — it ignores fusion across op boundaries, layout conversions between ops, and pipeline overlap — but it is dramatically better than FLOPs, and the errors are small and roughly systematic. FBNet reported their LUT predicted real latency within a few percent.

There are two ways the additive assumption breaks, and both are worth knowing because they tell you when to trust the table. First, **operator fusion**: a runtime like TensorRT or XNNPACK may fuse a conv with its following batch-norm and ReLU into one kernel, so the sum of three separate table entries overcounts. The fix is to profile *fused blocks* as the table's unit, not raw ops. Second, **layout transitions**: switching between NCHW and NHWC, or between dense and channels-last, can insert hidden transpose ops whose cost the table misses. The fix is to profile the same op in the same layout context it will actually run in. With those two cares taken, the LUT is the workhorse of practical hardware-aware NAS.

Why is the additive model as accurate as it is, given those leaks? The deep reason is that on most edge inference, layers execute **sequentially** — the runtime computes layer $i$, writes its output to memory, then computes layer $i+1$. There is little of the cross-layer pipelining that would make latencies overlap and break additivity, because batch-1 inference does not have enough independent work to fill a pipeline. So the total time really is close to the sum of per-layer times, and the leaks (fusion, layout) are second-order corrections rather than first-order errors. You can even quantify the table's quality with a simple regression: collect $N$ whole networks, predict each one's latency by summing its LUT entries, and compute the correlation between predicted and measured latency. FBNet and ProxylessNAS both report this correlation is very high (well above 0.9), and the residual is small and roughly unbiased — which is precisely the condition a search needs. The search does not require the table to be *exactly* right; it requires the table to *rank* architectures the same way the device does. A predictor with a small systematic offset still ranks correctly, and ranking is all the search consumes.

There is a third, more advanced LUT variant worth a mention: instead of a literal table keyed on exact configs, you can fit a small **regression model** (a gradient-boosted tree, or even a tiny neural net) that maps op features — type, resolution, channels, kernel, stride — to latency. This generalizes to configs you never explicitly profiled, which matters when the search space is large enough that profiling every distinct $(\text{op}, \text{config})$ tuple is itself expensive. The trade-off is that the regression can be wrong on configs far from its training data, where the literal table simply has no entry and forces you to measure. For most layer-wise search spaces the literal table is fine because the menu of configs is small; reach for the regression predictor when the space is rich enough that the table would have tens of thousands of entries. OFA, with its $10^{19}$ subnets, uses exactly such a learned latency predictor rather than a literal table.

### Building a real lookup table

Here is a minimal but real latency LUT builder in PyTorch. It profiles a handful of operator configurations on whatever device you run it on (your host CPU or GPU stands in for "the target chip" in this demo; on a real project you would run the identical script on the phone or Jetson). The measurement discipline matters as much as the code: warm up to fill caches and trigger lazy kernel compilation, run many iterations, take a robust statistic, and pin to batch size 1 because that is the edge reality.

```python
import time
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def measure_op(module, input_shape, iters=200, warmup=50):
    """Measure batch-1 latency of a single op/block in milliseconds (p50)."""
    module = module.to(DEVICE).eval()
    x = torch.randn(input_shape, device=DEVICE)
    # Warm-up: fill caches, JIT/cuDNN autotune, allocate workspaces.
    with torch.no_grad():
        for _ in range(warmup):
            module(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    samples = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            module(x)
            if DEVICE == "cuda":
                torch.cuda.synchronize()  # do not time async launches
            samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]  # median = p50, robust to jitter

def conv_block(cin, cout, k, stride, groups=1):
    pad = k // 2
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, pad, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU6(inplace=True),
    )

# A search space of candidate ops keyed by (name, resolution, cin, cout, k, stride, groups).
op_space = {
    ("conv3x3",      56, 24, 24, 3, 1, 1):  conv_block(24, 24, 3, 1, 1),
    ("conv1x1",      56, 24, 144, 1, 1, 1): conv_block(24, 144, 1, 1, 1),
    ("dwconv3x3",    28, 144, 144, 3, 1, 144): conv_block(144, 144, 3, 1, 144),
    ("dwconv5x5",    28, 144, 144, 5, 1, 144): conv_block(144, 144, 5, 1, 144),
    ("conv1x1_proj", 28, 144, 24, 1, 1, 1):  conv_block(144, 24, 1, 1, 1),
}

lut = {}
for key, module in op_space.items():
    name, res, cin = key[0], key[1], key[2]
    shape = (1, cin, res, res)
    lat = measure_op(module, shape)
    lut[key] = lat
    print(f"{name:14s} res={res:3d} cin={cin:3d}  ->  {lat:.3f} ms")
```

That dictionary `lut` is your predictor. Now estimating a candidate network is a sum:

```python
def predict_latency(arch_ops, lut):
    """arch_ops: list of LUT keys describing the chosen ops in order."""
    total = 0.0
    for key in arch_ops:
        if key not in lut:
            raise KeyError(f"op {key} not profiled; extend the LUT")
        total += lut[key]
    return total

# A toy candidate: expand -> depthwise -> project (one inverted residual block).
candidate = [
    ("conv1x1",      56, 24, 144, 1, 1, 1),
    ("dwconv3x3",    28, 144, 144, 3, 1, 144),
    ("conv1x1_proj", 28, 144, 24, 1, 1, 1),
]
print(f"predicted block latency: {predict_latency(candidate, lut):.3f} ms")
```

Two things to internalize. First, the LUT keys must encode *every* dimension that moves latency; if you collapse two different channel counts to one key, your prediction will drift. Second, you build the table once per device and reuse it across the whole search — that one-time cost is what makes the inner search loop cheap. A search that evaluates a million candidate architectures does a million additions instead of a million phone deployments.

## Differentiable latency: FBNet and ProxylessNAS

Lookup tables make the latency *signal* cheap, but a reinforcement-learning or evolutionary search still samples discrete architectures one at a time, which is sample-inefficient. The breakthrough of DARTS-style differentiable NAS — described in [neural architecture search basics](/blog/machine-learning/edge-ai/neural-architecture-search-basics) — was to relax the discrete choice of "which operator goes on this edge" into a continuous, differentiable mixture, so the whole architecture becomes one big trainable supernet and you optimize it with plain gradient descent. FBNet (Wu et al., 2019) and ProxylessNAS (Cai et al., 2019) extend exactly this relaxation to **make latency a differentiable term in the loss**. Let me derive it, because it is genuinely elegant and the derivation is short.

For each searchable edge $l$ in the network there is a set of candidate operators $\{o_1, o_2, \ldots, o_K\}$ — say a 3x3 depthwise, a 5x5 depthwise, a 1x1, a skip connection, and so on. In DARTS, instead of picking one, you compute all of them and take a weighted sum, where the weights come from a softmax over learnable **architecture parameters** $\alpha_l = (\alpha_{l,1}, \ldots, \alpha_{l,K})$:

$$
p_{l,i} = \frac{\exp(\alpha_{l,i})}{\sum_{j=1}^{K}\exp(\alpha_{l,j})}, \qquad
y_l = \sum_{i=1}^{K} p_{l,i}\, o_i(x_l).
$$

The $p_{l,i}$ are differentiable in $\alpha_l$. Now here is the key move. The **expected latency** of the relaxed supernet is just the probability-weighted sum of the candidate ops' latencies, and each op's latency is a *constant* from the LUT:

$$
\mathbb{E}[LAT] = \sum_{l}\sum_{i=1}^{K} p_{l,i}\cdot \text{LUT}[o_i, \text{config}_l].
$$

Because $\text{LUT}[\cdot]$ is a constant (it does not depend on $\alpha$), and $p_{l,i}$ is a smooth differentiable function of $\alpha$, the whole expected latency is **differentiable in the architecture parameters**. Its gradient is clean:

$$
\frac{\partial\, \mathbb{E}[LAT]}{\partial \alpha_{l,i}} = \sum_{j} \text{LUT}[o_j,\text{config}_l]\cdot p_{l,j}\,(\delta_{ij} - p_{l,i}),
$$

which is the standard softmax Jacobian times the constant latency vector — nothing exotic, just backprop through a softmax with constant weights. So you add expected latency to the loss as a regularizer and let gradient descent push the architecture distribution toward fast operators:

$$
\mathcal{L}(w, \alpha) = \underbrace{\mathcal{L}_{\text{CE}}(w, \alpha)}_{\text{task accuracy}} \;+\; \lambda\,\underbrace{\mathbb{E}[LAT](\alpha)}_{\text{from the LUT}}.
$$

The coefficient $\lambda$ plays the same role $w$ played in MnasNet: it is your accuracy-versus-latency knob. Crank $\lambda$ up and the search prefers cheaper ops even at an accuracy cost; turn it down and accuracy dominates. The figure below shows the dataflow: architecture logits go through a softmax, the softmax feeds both the task loss and the expected-latency term, the LUT feeds latency constants in, and gradients flow back to the logits, naturally favoring fast ops.

![A branching and merging dataflow graph showing architecture parameters relaxed by a softmax over operators that feeds both a task loss and an expected-latency term fed by a lookup table, with the combined loss backpropagating to favor fast operators](/imgs/blogs/hardware-aware-nas-4.png)

FBNet uses a temperature-annealed Gumbel-softmax instead of a plain softmax so that, as training proceeds, the mixture sharpens toward a near-discrete one-hot choice — this reduces the gap between the soft supernet you trained and the hard architecture you ultimately deploy. The principle is the same: latency enters the loss as a differentiable, LUT-sourced term.

It is worth pausing on *why* this is such a leap over MnasNet's measured reward, because it is the whole reason the field got cheap enough to be practical. MnasNet's controller proposes one architecture, gets one scalar reward, and learns from that single sample — a black-box, sample-inefficient signal that needs thousands of trials. The differentiable formulation gets a *gradient* from a single forward-backward pass: it does not just learn that this architecture was slow, it learns *which specific operator on which specific edge* contributed how much latency, and adjusts every architecture parameter simultaneously in the direction that reduces the loss. That is the difference between learning from a thermometer reading and learning from a full sensitivity analysis on every knob at once. A gradient over thousands of architecture parameters per step is worth an astronomical number of black-box samples, which is exactly why FBNet's search costs hundreds of GPU-hours where MnasNet's cost thousands of TPU-hours plus a phone farm. The LUT supplies the latency constants; the softmax relaxation supplies the differentiability; together they convert architecture search from a slow stochastic game into a smooth optimization.

### ProxylessNAS path binarization: searching on the target task directly

DARTS-style supernets have a brutal memory problem. Because every edge computes *all* $K$ candidate ops and keeps their activations for the weighted sum, the supernet uses roughly $K$ times the memory of a single architecture. With a dozen candidate ops per edge and a real-sized network, this blows past GPU memory, which is why DARTS and FBNet had to search on a *proxy* — a smaller network, a smaller dataset (CIFAR instead of ImageNet), fewer cells — and then transfer the found cell to the real task. But "search on a proxy, deploy on the real thing" reintroduces a gap: the architecture that is best on the proxy is not always best on the target.

ProxylessNAS's contribution is **path binarization**, a memory trick that lets you search directly on the target task at the target scale. The idea: at each forward pass, instead of computing all $K$ ops on an edge, you *sample* a binary gate — keep only one (or two) active path according to the architecture probabilities, and zero out the rest. The activations for the inactive paths are never computed and never stored, so the supernet's memory drops back to roughly that of a single architecture — the $K$ factor disappears. You binarize the architecture into one active path per edge:

$$
g_{l} \sim \text{Multinomial}(p_{l,1}, \ldots, p_{l,K}), \qquad y_l = o_{g_l}(x_l).
$$

The catch is that a hard sample is not differentiable, so ProxylessNAS uses a BinaryConnect-style estimator: forward through the sampled path, but compute the gradient with respect to the architecture parameters as if the binary gate were the continuous probability — the same family of trick as the straight-through estimator used in quantization. For the latency term, ProxylessNAS keeps it differentiable exactly as above (expected latency over the path probabilities from the LUT), so latency stays a smooth, gradient-friendly objective even though the network weights are trained through a single sampled path. The payoff is concrete: ProxylessNAS searched directly on ImageNet at full resolution — no proxy — and could specialize per hardware (it published separate searched models for mobile CPU, mobile GPU, and a desktop GPU), each optimized against that device's own LUT.

## The practical flow: add latency to a DARTS-style search

Let me make the differentiable-latency idea runnable. Below is a compact, idiomatic DARTS-style search cell with an added latency regularizer sourced from a LUT. It is deliberately small so you can read every line; a production search (FBNet, ProxylessNAS) is this idea scaled up with Gumbel-softmax or path binarization, but the skeleton is identical.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedOp(nn.Module):
    """One searchable edge: a softmax mixture over candidate ops."""
    def __init__(self, ops, lut_latencies):
        super().__init__()
        self.ops = nn.ModuleList(ops)                 # K candidate ops
        # Per-op measured latency (constant tensor from the LUT).
        self.register_buffer("lat", torch.tensor(lut_latencies))  # shape [K]
        self.alpha = nn.Parameter(torch.zeros(len(ops)))          # arch logits

    def forward(self, x):
        p = F.softmax(self.alpha, dim=0)              # differentiable weights
        y = sum(w * op(x) for w, op in zip(p, self.ops))
        exp_lat = torch.dot(p, self.lat)             # E[latency] for this edge
        return y, exp_lat

class SearchCell(nn.Module):
    def __init__(self, edges):
        super().__init__()
        self.edges = nn.ModuleList(edges)            # list of MixedOp

    def forward(self, x):
        total_lat = x.new_zeros(())
        for edge in self.edges:
            x, lat = edge(x)
            total_lat = total_lat + lat
        return x, total_lat
```

The training loop alternates: a step on the network weights `w` to minimize task loss, and a step on the architecture parameters `alpha` to minimize task loss plus the latency penalty. The latency term is what makes this hardware-aware — drop it and you are back to plain DARTS.

```python
def search_step(model, head, x, y, w_opt, a_opt, lam):
    # --- weight step: train the supernet on the task ---
    w_opt.zero_grad()
    feat, _ = model(x)
    loss_w = F.cross_entropy(head(feat), y)
    loss_w.backward()
    w_opt.step()

    # --- arch step: push alpha toward accurate AND fast ops ---
    a_opt.zero_grad()
    feat, total_lat = model(x)
    ce = F.cross_entropy(head(feat), y)
    loss_a = ce + lam * total_lat          # latency enters the loss here
    loss_a.backward()
    a_opt.step()
    return ce.item(), total_lat.item()
```

After the search converges, you discretize: on each edge take `argmax(alpha)` as the chosen op, assemble the discrete network, and train it from scratch (or fine-tune the inherited weights). The `lam` you chose controls where on the accuracy–latency Pareto frontier the found architecture lands — exactly the trade-off frame this whole series is built on, the one in [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression).

### A few real-world cares for the search code

When you run this on a real search space, three things bite. First, **the latency tensor must match the op's actual config on this edge** — if the edge operates at 28x28 with 144 channels, the LUT entry must be for that exact resolution and channel count, not a generic per-op average. Wire the LUT lookup to the edge's real shape. Second, **anneal nothing in the toy version, but in production sharpen the mixture** (Gumbel-softmax with decaying temperature in FBNet) so the discretization gap is small; a soft supernet that is 30% op-A and 30% op-B can discretize to a network whose latency you never actually evaluated. Third, **scale `lam` to the units**: if total latency is in milliseconds (single digits) and cross-entropy is order 1, a `lam` of 0.01 to 0.1 is a sane starting range — sweep it and plot the resulting accuracy–latency points to see the frontier.

There is a fourth care that bites only at scale, and it is the difference between a search that converges and one that wobbles forever: the **bilevel optimization** structure. You are optimizing two sets of parameters with two objectives — the network weights `w` against task loss on a training split, and the architecture parameters `alpha` against task-plus-latency on a *separate* validation split. Using the same data for both lets the architecture overfit to the training set, picking ops that memorize rather than generalize, so DARTS-style searches split the data and alternate the two updates as the loop above does. In practice you also want to **warm up the weights before touching alpha** — train `w` alone for a few epochs so the supernet is not random when the architecture gradient first fires, otherwise alpha chases noise and the search latches onto whatever op happened to look good early (often the parameter-free skip connection, a known DARTS failure mode where the search collapses to all-skips). A short weight warm-up, a validation split for the arch step, and a sharpening schedule are the three things that turn the toy loop above into a search that actually returns a good architecture.

### Composing the LUT with quantization in the search

A subtlety that separates a textbook search from a deployable one: the LUT must be built in the **precision you will ship**. If you intend to deploy int8 (and on the edge you almost always should — see the quantization track of this series), then an fp32 LUT is measuring the wrong thing, because int8 changes which ops are fast. On many NPUs int8 convolutions run 2-4x faster than fp32, but the speedup is *not uniform across ops*: a large dense conv may get the full 4x because it is compute-bound and the int8 MAC array is the bottleneck it relieves, while a small depthwise conv that was already memory-bound gets far less, because quantization shrinks the data but the op was waiting on memory bandwidth, not compute. So the *ranking* of cheap ops shifts when you go from fp32 to int8, and an fp32-built LUT will steer the search toward the wrong shape. The disciplined flow is to quantize the candidate ops to the deploy precision *before* profiling them into the table, so the search optimizes the latency of the network you will actually run. This is the concrete meaning of "the levers compose": hardware-aware NAS and quantization are not sequential afterthoughts to each other — the quantization decision belongs *inside* the search's latency model.

## Once-for-All: train one supernet, ship many subnets

Every method so far shares one expensive flaw: the search is **per device**. MnasNet, FBNet, and ProxylessNAS all bake one device's LUT into one search, so if you ship to a phone, a Jetson, and a microcontroller, you run the whole search three times. For a product that targets a dozen SKUs (a phone in three tiers, two wearables, an edge box, a car) that is a dozen GPU-week searches, and every time a new device launches you do it again. Once-for-All, or OFA (Cai et al., 2020), is the idea that broke this curse. Its slogan is exactly what it says: train the network *once*, then get a specialized architecture for *any* device with no retraining.

The mechanism is an **elastic supernet** — a single over-parameterized network that contains, as sub-networks, an astronomically large family of smaller architectures that all share weights. OFA makes four dimensions elastic:

- **Elastic depth**: a stage can use the first $d$ of its blocks, for $d \in \{2, 3, 4\}$. A shallower subnet just skips the later blocks.
- **Elastic width**: a block can use the first $e \times$ channels of its expansion, for expansion ratios $e \in \{3, 4, 6\}$. A narrower subnet uses a channel subset.
- **Elastic kernel size**: a depthwise conv can act as 3x3, 5x5, or 7x7, where the smaller kernel is the center crop of the larger one (with a learned transform).
- **Elastic resolution**: the input can be any of several resolutions, e.g. 128 to 224.

Multiply the choices out across all stages and you get on the order of $10^{19}$ sub-networks, all living inside one set of weights. Now the magic: because every subnet *shares* the supernet's weights, once the supernet is trained well, you can carve out any subnet and it already works — no retraining. To deploy to a device, you run a cheap latency-predictor-guided search **over subnets** (not over a fresh training run): evaluate candidate subnets' accuracy with a small accuracy predictor and their latency with the device's LUT, and pick the subnet that maximizes accuracy under that device's latency budget. The search takes minutes, not GPU-weeks, because nothing trains. The figure below shows the structure: one supernet trained once, fanning out into a specialized subnet per device.

![A branching dataflow graph showing one Once-for-All supernet trained by progressive shrinking that fans out, guided by per-device latency predictors, into specialized subnets for a microcontroller, a phone, and an edge GPU](/imgs/blogs/hardware-aware-nas-5.png)

### Why progressive shrinking is necessary

There is a real difficulty hiding in "just train the supernet": if you naively train by sampling random subnets each step, the dimensions interfere. A weight that is great for the full 7x7 kernel is yanked in a different direction when it has to serve as a 3x3, and the big and small subnets co-adapt destructively, so everything ends up mediocre. OFA's answer is **progressive shrinking**, a curriculum that adds elasticity one dimension at a time, largest first.

Train the full largest network to convergence first. Then make **kernel size** elastic and fine-tune, so the smaller kernels learn to be good center-crops of the trained big kernel rather than starting from scratch. Then make **depth** elastic and fine-tune. Then **width**, using a clever channel-sorting trick: sort channels by importance (L1 norm) so that the "first $k$ channels" a narrow subnet uses are the most important ones, which means narrowing degrades gracefully instead of randomly dropping critical channels. By always shrinking *from* a trained larger configuration and fine-tuning, each smaller subnet inherits good weights and only has to learn the residual adjustment. The result is a supernet where even the small corners of the family are accurate, which is the precondition for the no-retraining promise to hold.

This is why OFA "amortizes search across many devices." The expensive part — training the supernet with progressive shrinking — happens once, at a cost roughly comparable to training a handful of normal networks (OFA reports it is far cheaper in total CO2 than searching per device once you have more than a couple of targets). After that, each new device is a minutes-long subnet search against its LUT, not a new training run. The marginal cost of the eleventh device is essentially zero training and a few minutes of predictor-guided search.

### The elastic-kernel transform, concretely

One mechanism inside progressive shrinking deserves a closer look because it is where the weight-sharing magic is most non-obvious: how can one set of weights serve as a 7x7, a 5x5, *and* a 3x3 depthwise kernel at once? Naively, the center 3x3 of a trained 7x7 kernel is not a good 3x3 kernel — the surrounding weights carried real information, and cropping them away leaves a degraded filter. OFA's fix is a small learned **transformation matrix** between kernel sizes. The 5x5 kernel is the center 5x5 of the 7x7 weights passed through a learned $25\times 25$ linear transform; the 3x3 is the center 3x3 of *that* passed through a learned $9\times 9$ transform. These transforms are shared across all the depthwise layers (so they add negligible parameters) and are trained during the kernel-elastic phase. The effect is that each kernel size gets to be a *learned function* of the larger kernel's weights rather than a blind crop, so all three sizes are simultaneously good. It is a beautiful little trick: the weights are shared, but each subnet sees them through a size-appropriate lens.

The width-elastic phase has its own trick worth stating precisely. When a narrow subnet uses "the first $k$ channels," which channels are first? OFA **sorts channels by importance** (L1 norm of the channel's weights) before each width-shrinking step, so the most important channels migrate to the front. A subnet that uses the first $k$ channels therefore uses the $k$ *most important* channels, and narrowing degrades gracefully along the importance ordering instead of randomly amputating critical features. This is the same insight that powers structured pruning — see [structured pruning that actually speeds things up](/blog/machine-learning/edge-ai/structured-pruning-that-actually-speeds-things-up) — applied inside the supernet so that every width is a sensible sub-network.

### Why weight sharing makes the no-retraining promise plausible

It is worth being honest about the central assumption, because "extract any subnet and it just works" sounds too good. The reason it can work is that progressive shrinking forces every subnet to share weights *that were trained to be good at every size*. Formally, the supernet's training objective is, in expectation over sampled subnets $s$ drawn from the architecture family $\mathcal{S}$,

$$
\min_{W}\; \mathbb{E}_{s\sim\mathcal{S}}\big[\mathcal{L}_{\text{CE}}\big(W_s\big)\big],
$$

where $W_s$ is the slice of the shared weights $W$ that subnet $s$ activates. Minimizing this expectation means $W$ is pushed to be good *on average* across the whole family — including the small corners — and progressive shrinking is the curriculum that makes this expectation actually converge rather than collapse into mediocrity. The no-retraining promise holds to the extent that this expectation is well-minimized; in practice OFA's extracted subnets are within a fraction of a point of the same architecture trained from scratch, which is why the predictor-guided search can trust the supernet's accuracy as a stand-in for the trained subnet's accuracy. When the promise frays — for very small or very large corners of the family that the sampling under-covered — the remedy is a short fine-tune of the extracted subnet, which costs minutes, not a full training run.

### Picking an OFA subnet for a latency budget

OFA ships pretrained supernets and the tooling to extract subnets. The flow to specialize for a device with a latency budget looks like this:

```python
# Using the official Once-for-All repo (mit-han-lab/once-for-all).
from ofa.model_zoo import ofa_net
from ofa.nas.efficiency_predictor import Mbv3LatencyTable
from ofa.nas.accuracy_predictor import AccuracyPredictor, ResNetArchEncoder

# 1. Load a pretrained OFA supernet (no training needed).
ofa_network = ofa_net("ofa_mbv3_d234_e346_k357_w1.0", pretrained=True)

# 2. Load (or build) a latency table for THIS device. Here a prebuilt
#    mobile-CPU table; for a new device you build it like our LUT above.
latency_table = Mbv3LatencyTable(device="note10")   # e.g. Galaxy Note10

# 3. Accuracy predictor: estimates a subnet's accuracy without training it.
acc_predictor = AccuracyPredictor(pretrained=True)

# 4. Search over subnets for the best accuracy under a latency budget.
from ofa.nas.search_algorithm import EvolutionFinder
finder = EvolutionFinder(
    efficiency_predictor=latency_table,
    accuracy_predictor=acc_predictor,
    efficiency_constraint=25,    # 25 ms budget on this device
)
best_subnet_cfg, best_acc, best_latency = finder.run_evolution_search()
print(f"chosen subnet ~{best_acc:.1f}% top-1 @ ~{best_latency:.1f} ms")

# 5. Materialize the actual subnet (shares supernet weights, no retraining).
ofa_network.set_active_subnet(**best_subnet_cfg)
subnet = ofa_network.get_active_subnet(preserve_weight=True)
```

The evolutionary finder mutates subnet configs (depth, width, kernel per stage, resolution), scores each with the accuracy predictor and the device LUT, and keeps the Pareto-best under the constraint. Change `efficiency_constraint` to retarget a different device and re-run — minutes later you have a different subnet, all from the same `ofa_network` weights. ProxylessNAS's repo (`mit-han-lab/proxylessnas`) ships pre-searched per-hardware models if you would rather grab a ready architecture than search.

## Worked examples

#### Worked example: FLOPs objective vs latency objective pick different architectures

Suppose we run the same DARTS-style search twice over an inverted-residual search space, targeting a Pixel 4 CPU. The only difference is the regularizer: run A penalizes FLOPs, run B penalizes measured latency from the device LUT.

Run A (FLOPs-objective) converges to a heavily fragmented block: two parallel grouped-conv branches feeding a 5x5 depthwise, channel-shuffled, then concatenated. The FLOP counter loves it — **145M FLOPs**, well under the budget. We build it, deploy it, and time it on the Pixel 4 CPU at batch 1 with warm-up: **p50 12.4 ms**. The fragmentation that saved FLOPs created a swarm of tiny memory-bound kernels, and the channel shuffle inserted layout churn the FLOP count never saw.

Run B (latency-objective) converges to something a FLOP purist would frown at: a *fatter*, fused single-path inverted residual with a plain 3x3 depthwise and no fragmentation. It costs **210M FLOPs** — 45% *more* compute than run A. But it is one clean, high-utilization sequence of ops the NPU and CPU vector units can chew through, so on the same Pixel 4 CPU it times at **p50 7.8 ms**. The latency-objective network is 45% heavier on paper and **37% faster in reality** (12.4 ms to 7.8 ms). Same search space, same accuracy within half a point, opposite winners — because one search optimized a proxy and the other optimized the chip. This is figure 1 made quantitative, and it is the entire argument for measuring latency in the loop.

#### Worked example: one OFA supernet, three devices, three subnets

You have one OFA-MobileNetV3 supernet, trained once. You must ship to three targets, each with its own latency budget. You build a LUT per device (one profiling pass each), then run the minutes-long evolutionary subnet search three times — same weights, three constraints.

- **Microcontroller-class target (tight, ~8 ms budget).** The finder picks a shallow, narrow subnet: depth 2 per stage, expansion ratio 3, mostly 3x3 kernels, resolution 160. Result: a subnet near **8 ms** that fits in roughly **11 MB** after int8 quantization, trading a few points of top-1 for the budget. The supernet did not retrain — this subnet's weights were already inside it.
- **Mid-range phone (moderate, ~22 ms budget).** The finder spends the larger budget on depth and a couple of 5x5 kernels: depth 3-4, mixed expansion 4-6, resolution 192. Result: a subnet around **22 ms** at roughly **76% top-1** on ImageNet — a solid mobile operating point.
- **Edge GPU (loose latency, accuracy-hungry, ~6 ms but high throughput).** The GPU loves wide, regular, parallel work, so the finder picks a *wider* subnet than the phone's even though its latency is lower: expansion 6 throughout, 5x5 and 7x7 kernels, resolution 224. Result: about **6 ms** at roughly **78% top-1**, because the GPU's parallelism makes the wider net cheap where the phone CPU could not afford it.

Three devices, three genuinely different architectures, each Pareto-optimal for its chip — extracted from **one** training run. Notice the GPU subnet is *wider but faster* than the phone subnet: the per-device LUT captured that the GPU rewards width and the CPU punishes it, and the search responded. No FLOP-based method could have produced that inversion, because FLOPs say the wider net is unconditionally more expensive.

#### Worked example: tuning the latency coefficient to trace a frontier

You do not pick one model; you trace a frontier and let the product choose. Suppose you run the differentiable search five times on the same Pixel-4-CPU LUT, sweeping the latency coefficient $\lambda$ across $\{0.0,\ 0.02,\ 0.05,\ 0.1,\ 0.2\}$ (recall $\lambda$ scales the expected-latency term in the loss). At $\lambda = 0$ the search ignores latency entirely and returns the most accurate net it can build — call it **76.8% top-1 at p50 14.0 ms**, the slow-but-accurate corner. As $\lambda$ rises, the search trades accuracy for speed along a smooth curve: $\lambda = 0.05$ lands around **75.6% at 9.5 ms**, $\lambda = 0.1$ around **74.3% at 7.4 ms**, and the aggressive $\lambda = 0.2$ around **72.1% at 5.6 ms**, the fast-but-lighter corner. Plot those five points and you have the accuracy–latency Pareto frontier *for this chip*, drawn by the search itself. Now the product team makes the call: a real-time AR feature on a 33 ms frame budget shared with the renderer might take the $\lambda = 0.1$ point for headroom; an offline photo-tagging job that just wants accuracy takes $\lambda = 0.02$. The single most valuable output of a hardware-aware search is not one model — it is this frontier, because it converts an engineering search into a product decision with the trade-off made explicit and the chip's real latency on the axis.

## Results: methods compared, and searched vs hand-designed

Let me put the four methods side by side on the axes that decide which you reach for: how latency enters the objective, what the search costs, and whether one run serves many devices.

![A four-by-three comparison matrix of MnasNet, FBNet, ProxylessNAS, and Once-for-All across how latency enters the objective, search cost, and multi-device support](/imgs/blogs/hardware-aware-nas-6.png)

| Method | How latency enters | Search strategy | Search cost (approx) | Multi-device | Searched on |
| --- | --- | --- | --- | --- | --- |
| MnasNet (2019) | Measured reward, soft power law $ACC\cdot[LAT/T]^w$ | RL controller + phone farm | Thousands of TPU-hours | Re-search per device | Proxy then full |
| FBNet (2019) | LUT term in loss, differentiable, Gumbel-softmax | DARTS-style supernet | ~216 GPU-hours per device | Re-search per device | Proxy network |
| ProxylessNAS (2019) | LUT term in loss, differentiable, path-binarized | Binarized supernet | ~200 GPU-hours per device | Re-search per device | Target task directly |
| Once-for-All (2020) | Device LUT in subnet search (no training) | Progressive shrinking + evolution | Train once; minutes per device | One supernet, many subnets | Target task directly |

The trajectory is clear: each method drove the cost of getting latency into the search lower. MnasNet proved measured latency belongs in the objective but paid a fortune to do it. FBNet and ProxylessNAS killed the per-candidate measurement cost with a differentiable LUT term, with ProxylessNAS additionally killing the proxy gap via path binarization. OFA killed the per-device search cost entirely by decoupling training from specialization.

Now the result that matters most — does any of this actually beat a strong hand-designed network? Hand-designed efficient nets like MobileNetV2 and V3 (see [the MobileNet family](/blog/machine-learning/edge-ai/the-mobilenet-family)) are not weak baselines; they are the product of years of expert intuition about what runs fast on phones. The figure below shows the comparison at matched latency on a Pixel-class CPU.

![A two-column comparison showing a hand-designed MobileNetV2 baseline versus a MnasNet searched network at the same latency budget on a Pixel-class CPU, with the searched net gaining accuracy at parity](/imgs/blogs/hardware-aware-nas-7.png)

| Network | Source | Top-1 (ImageNet) | Latency (Pixel-class CPU, batch 1) |
| --- | --- | --- | --- |
| MobileNetV2 1.0 | Hand-designed | ~72.0% | ~75 ms |
| MnasNet-A1 | Hardware-aware search | ~75.2% | ~76 ms |
| MobileNetV3-Large | Search + hand tuning (MnasNet-derived) | ~75.2% | ~51 ms |
| FBNet-B | Hardware-aware search | ~74.1% | comparable mobile budget |
| ProxylessNAS (mobile) | Hardware-aware search | ~74.6% | tuned to mobile LUT |
| OFA (subnet) | One supernet, specialized | up to ~80% | within mobile budgets |

(These are the headline figures the original papers report on ImageNet; exact latency depends on the specific Pixel generation, runtime, and quantization, so treat the milliseconds as the regime the papers operated in, not a single canonical number.) The takeaway holds across all of them: at matched latency, the hardware-aware searched networks deliver **2 to 3 points** more top-1 accuracy than the strong hand-designed baseline — and MobileNetV3, which is itself a hardware-aware-search result polished by hand, is the proof that the techniques in this post are not academic. They shipped, on the phone in your pocket.

### How to measure these honestly

If you take one measurement lesson from this series, take this one, because hardware-aware NAS is only as good as the latency it measured. Always **warm up** before timing — the first few inferences pay for cache fills, lazy kernel compilation, and workspace allocation, and including them inflates your numbers and corrupts the LUT. Time **batch size 1**, because that is the edge reality; throughput-at-batch-64 numbers are irrelevant to a camera app. Report **p50 and p99**, not the mean — the tail is what users feel and what a real-time deadline must satisfy, and means hide it. Watch **thermal throttling**: a phone that has been timing models for ten minutes is hotter and slower than a cold one, so a fair LUT is built on a thermally stable device, and a benchmark that looks great for thirty seconds and degrades after two minutes is a benchmark that lied to you. And pin the **runtime and quantization** you will actually ship — an fp32 LUT is worthless if you deploy int8, because int8 changes which ops are fast. The full discipline lives in [the metrics that actually matter on-device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device).

## Case studies: real searched networks that shipped

These are not hypotheticals. Hardware-aware NAS produced the efficient networks that run on a billion phones, and the numbers below come from the original papers. Treat the millisecond figures as the regime each paper operated in (they depend on the exact phone generation and runtime), but the relative wins are robust and reproduced.

**MnasNet-A1 on a Pixel phone.** The original MnasNet paper searched directly against the latency measured on a Pixel-1 phone CPU. The headline result: MnasNet-A1 reached roughly **75.2% top-1** on ImageNet at about **78 ms** on the Pixel-1 CPU, versus MobileNetV2's roughly **72.0%** at a comparable latency. That is about **3.2 points** of accuracy for free at matched latency — and "free" is exactly right, because the only thing that changed was that the search optimized the phone's measured latency instead of FLOPs. MnasNet also demonstrated the soft-reward trade-off knob in action: by sweeping the target $T$, they traced out a whole accuracy–latency frontier of models, from a fast-but-lighter variant to a slower-but-more-accurate one, each Pareto-optimal at its point. This frontier is the single most important artifact a hardware-aware search produces, because it lets the product team pick the operating point rather than accept a single model.

**MobileNetV3, the search result you actually use.** MobileNetV3 (Howard et al., 2019) is the clearest proof that these techniques are production, not research. Its backbone was found by a MnasNet-style platform-aware search, then refined by a complementary fine-grained search (NetAdapt) that trimmed individual layers against the on-device latency table, and finally hand-tuned (the redesigned early and late stages, the h-swish activation). MobileNetV3-Large hits about **75.2% top-1** at roughly **51 ms** on a Pixel phone — faster *and* more accurate than the earlier MnasNet-A1, because the additional latency-table-guided layer trimming squeezed out milliseconds the macro search left on the table. MobileNetV3-Small targets the tightest budgets at about **67% top-1** around **15-20 ms**. If you have ever used a recent Android camera feature, you have very likely run a descendant of this search.

**FBNet's per-device specialization.** FBNet (Wu et al., 2019) drove the search cost down to about **216 GPU-hours** for a complete search — roughly the cost of training a single large model, versus MnasNet's phone-farm fortune — by making latency a differentiable LUT term and using a Gumbel-softmax supernet. FBNet-B reached about **74.1% top-1** at a mobile latency competitive with MobileNetV2 and the MnasNet models, and the FBNet family (A/B/C) traced a frontier by varying the latency coefficient. The headline lesson from FBNet is economic: differentiable latency made hardware-aware search affordable enough that a normal lab, not just Google's infrastructure, could run it.

**ProxylessNAS searching ImageNet directly, per hardware.** ProxylessNAS (Cai et al., 2019) used path binarization to fit the full-scale supernet in memory and searched **directly on ImageNet** — no CIFAR proxy, no transfer gap. It then searched separately for three hardware targets and got three architectures: a mobile-CPU model around **74.6% top-1** tuned to the phone's LUT, plus distinct GPU and CPU-server models. The same paper reported that its mobile model beat MobileNetV2 by about **2.6%** top-1 at the same latency on the target phone. ProxylessNAS is the cleanest demonstration of "the architecture is a function of the chip" — three chips, three different winning shapes, each from a search that timed only that chip.

**Once-for-All's many-device economics.** OFA (Cai et al., 2020) reported subnets reaching up to roughly **80% top-1** on ImageNet within mobile latency budgets — state-of-the-art for the regime at the time — extracted from a single trained supernet. The number that matters most is not the accuracy but the *amortization*: OFA showed that once you target more than a couple of devices, training one supernet plus minutes-per-device extraction has dramatically lower total cost (and CO2) than running a from-scratch hardware-aware search per device. For a product line spanning phone tiers, wearables, and edge boxes, OFA turned "a GPU-week per SKU" into "one training run plus an afternoon of subnet extraction."

## Stress-testing the approach: where it breaks

A search is a confident liar if its latency signal is wrong, so it is worth poking at the failure modes before you trust one.

**What if the LUT is inaccurate?** The whole edifice rests on the additive assumption $LAT \approx \sum \text{LUT}[\text{op}]$. When the runtime fuses ops aggressively (TensorRT fusing conv-bn-relu, or an NPU compiler fusing whole inverted-residual blocks), the per-op sum overcounts and the search optimizes a phantom. The fix is to make the LUT's unit the *fused block* the compiler will actually emit, not the raw op. Profile blocks, not atoms, in the layout and precision you will ship.

**What about op support and CPU fallback?** The nastiest edge case in practice: the search picks an operator the target NPU does not support, so at deploy time the runtime silently falls back to the CPU for that op — and a single CPU-fallback op in the middle of an otherwise-NPU graph forces two device-to-device tensor copies and a synchronization, which can cost more than the op itself. If your LUT was built by running each op on whatever device happened to run it, it never saw this penalty. The defensive move is to restrict the search space to ops you have *confirmed* run natively on the target accelerator, and to profile the LUT on the exact runtime + delegate (NNAPI, Core ML, the NPU SDK) you will deploy with, so fallback costs are baked in or the offending op is simply absent from the space.

**What if the model is memory-bound, not compute-bound?** Then FLOPs are even more misleading and the LUT is even more essential — which is the good news. But it also means latency depends on things the LUT keys must capture: activation sizes, intermediate tensor memory, and DRAM bandwidth, all of which vary with resolution and channel count. If you under-specify the LUT key (e.g., ignore resolution), a memory-bound op's predicted latency will be systematically wrong. Key the table on everything that moves memory traffic.

**What if you discretize into a network you never measured?** The DARTS/FBNet supernet is a soft mixture; the deployed network is a hard pick. If the mixture was diffuse, the argmax architecture can have a latency the search never actually evaluated as a unit. Sharpen the mixture during search (temperature annealing) and, critically, **re-measure the final discretized network's true latency on the device** before you trust it. The search gives you a candidate; the device gives you the truth.

## When to reach for hardware-aware NAS (and when not to)

Hardware-aware NAS is powerful and it is expensive, and the most senior thing you can do is often *not* run it. The figure below is the decision I actually use.

![A decision tree for whether to reuse a published efficient family, extract an Once-for-All subnet, or run a full hardware-aware search, branching on whether a family hits the budget and how many devices you target](/imgs/blogs/hardware-aware-nas-8.png)

Walk it top down. **First, try a published family.** MobileNetV3, EfficientNet-Lite, and friends are hardware-aware-search results that someone already paid for, validated, and pretrained. With a width multiplier and int8 quantization you can hit a startling range of latency budgets for **zero search cost**. The overwhelming majority of edge projects should stop here. Reaching for a custom search before you have exhausted the published families is the architecture-search equivalent of writing your own hash map.

**If a published family genuinely cannot hit the budget — or if you target many devices — escalate to Once-for-All.** If you have several SKUs (phone tiers, a wearable, an edge box), OFA's train-once-extract-many economics dominate: one supernet, then minutes per device. Even for a single device, extracting an OFA subnet is often cheaper and better than searching from scratch, because the supernet is already trained.

**Only run a full from-scratch hardware-aware search (MnasNet/FBNet/ProxylessNAS style) when the family ceiling truly blocks you and the deployment volume justifies the GPU bill.** A search is GPU-weeks of cost plus engineering time plus the ongoing burden of maintaining a bespoke architecture. That math pays off when you ship to tens of millions of devices and a 2 ms latency win or a 1-point accuracy win compounds into real money or a real product capability — Google running it for the camera pipeline on every Pixel makes sense; a startup shipping to a few thousand boards almost never does. Be honest about volume: the search cost is fixed, so the per-device value of the win has to clear a high bar.

One more strategic note: hardware-aware NAS *composes* with the other levers. The architecture it finds is still a fp32 network; quantizing it to int8 (the quantization track of this series) is a further 2-4x speed and size win, and OFA explicitly pairs with quantization — extract a subnet for the budget, then quantize it to land even cheaper on the chip. The cleanest edge pipeline is: pick or search the architecture for the device's latency budget, then quantize, then deploy and re-measure. NAS gets you to the right shape; quantization gets you to the right precision; both read off the same accuracy–efficiency frontier.

## The bridge from architecture to deployment

It is worth stepping back to see why this post sits where it does in the series. Every other lever — quantization, pruning, distillation — takes an architecture as *given* and squeezes it. Hardware-aware NAS is the one lever that designs the architecture *for* the device in the first place, with the device's own latency as the objective. It is the bridge between "what model" and "what chip," and it is the only technique in the four-lever frame that closes the loop between architecture choice and deployment reality by putting the deployment measurement *inside* the search.

That is also why it is the natural partner of the deployment lifecycle. The LUT you build for a search is the same profiling you do to validate any deployment; the latency budget you search against is the same SLO (service-level objective, your latency target) the product team hands you; the re-measurement of the discretized network is the same on-device benchmark you run before shipping anything. Hardware-aware NAS does not replace the deployment workflow — it folds the deployment measurement back into the design phase so the architecture is born deployable. When you stitch the levers together end to end, that is the [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook), and hardware-aware NAS is the step that makes sure you started from the right shape.

## Key takeaways

- **FLOPs are a proxy, latency is the truth.** A FLOP-optimal architecture and a latency-optimal architecture are different networks on real hardware; a search that never times the chip will hand you a model that wins on paper and loses on the device.
- **Put measured latency in the objective.** MnasNet's soft reward $ACC(m)\cdot[LAT(m)/T]^w$ with $w<0$ keeps a smooth gradient toward the target, unlike a hard cap's all-or-nothing cliff; $w$ is your accuracy-per-latency exchange rate in log-log space.
- **A lookup table beats FLOPs.** Profile each operator once on the target chip, then estimate any net by summing its ops' measured latencies — the table memorizes the chip's memory-bound and utilization quirks that FLOPs discard. Profile fused blocks in the deploy layout and precision.
- **Latency can be differentiable.** Relax the op choice into a softmax over architecture parameters; expected latency becomes a weighted sum of constant LUT entries, differentiable in the architecture weights, addable straight to the loss (FBNet, ProxylessNAS).
- **Path binarization lets you search on the real task.** Sampling one active path per edge cuts the supernet's memory by the candidate-count factor, so ProxylessNAS searches ImageNet at full scale per hardware instead of on a proxy.
- **Once-for-All amortizes search across devices.** Train one elastic supernet with progressive shrinking, then extract a specialized subnet per device in minutes with no retraining — the eleventh device costs essentially zero training.
- **Reuse before you search.** Try MobileNetV3 / EfficientNet-Lite first; escalate to an OFA subnet for many devices; only run a full from-scratch search when the family ceiling blocks you and volume justifies the GPU bill.
- **Measure honestly and re-measure the final net.** Warm up, batch 1, p50 and p99, watch thermal throttling, pin the runtime and precision — and time the discretized network on the device, because the supernet's prediction is a candidate, not the truth.

## Further reading

- Tan et al., **"MnasNet: Platform-Aware Neural Architecture Search for Mobile"** (CVPR 2019) — the soft multi-objective reward and the measured-latency-in-the-loop idea.
- Wu et al., **"FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search"** (CVPR 2019) — the differentiable LUT-in-the-loss formulation with Gumbel-softmax.
- Cai et al., **"ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware"** (ICLR 2019) — path binarization and per-hardware specialization on the target task.
- Cai et al., **"Once-for-All: Train One Network and Specialize it for Efficient Deployment"** (ICLR 2020) — elastic supernet, progressive shrinking, predictor-guided per-device subnet search. Code: `mit-han-lab/once-for-all`.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), [neural architecture search basics](/blog/machine-learning/edge-ai/neural-architecture-search-basics), [EfficientNet, ShuffleNet, and the FLOPs–latency gap](/blog/machine-learning/edge-ai/efficientnet-shufflenet-and-the-flops-latency-gap), [the MobileNet family](/blog/machine-learning/edge-ai/the-mobilenet-family), [the metrics that actually matter on-device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device), and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).

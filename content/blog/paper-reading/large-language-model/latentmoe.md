---
title: "LatentMoE: Shrink the Expert Highway, Spend the Savings on More Experts"
date: "2026-07-24"
description: "NVIDIA's LatentMoE projects tokens into a low-dimensional latent space before expert routing, then reinvests the saved bandwidth and communication into more experts and larger top-k — improving accuracy per FLOP and per parameter at constant serving cost."
tags: ["paper-reading", "mixture-of-experts", "latentmoe", "moe", "inference-efficiency", "hardware-software-co-design", "roofline", "expert-parallelism", "nemotron", "large-language-model"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 33
paper:
  title: "LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts"
  authors: "Venmugil Elango et al. (NVIDIA)"
  venue: "arXiv 2026 (2601.18089)"
  url: "https://arxiv.org/abs/2601.18089"
---

> [!tldr]
> - **What it proposes.** LatentMoE is a Mixture-of-Experts layer that first projects each token from the model's hidden dimension $d$ down into a much smaller **latent dimension** $\ell$ before routing and expert computation, then projects the result back up to $d$. Routing communication and expert weight-loading both shrink by a factor $\alpha = d/\ell$.
> - **The key move.** Those savings are not banked — they are **reinvested**. With $\alpha = d/\ell$ freed up, you can afford $\alpha$-times more experts $N$ and $\alpha$-times larger top-$k$, at the *same* memory-bandwidth and all-to-all cost as the original MoE.
> - **Why it matters.** MoE serving is *not* FLOP-bound. In interactive decode it is **memory-bandwidth-bound** (loading expert weights); in high-throughput serving it is **all-to-all-bound**. Both scale with $d$, so cutting $d$ attacks the real bill — and more experts + larger top-$k$ demonstrably raise accuracy.
> - **The surprising result.** At iso-parameter and iso-FLOP, the accuracy variant (ℓ-MoEacc) lifts MMLU-Pro on a 95B model from **29.3 → 34.9** and on a hybrid 73B model from **48.3 → 52.9** — a free +5.6 and +4.6 points, no extra serving cost.
> - **Where it's honest / where it isn't.** LatentMoE was adopted by NVIDIA's flagship Nemotron-3 Super and Ultra. But at *concurrency 1* — the very latency-critical regime the paper motivates — measured throughput is actually **lower** than standard MoE (181.6 vs 206.6 tok/s), and the trillion-scale story rests on a *proprietary simulator* plus a log-linear "effective parameter" fit. Read Section 4.3 and the critique before you believe the frontier plot.

The figure below is the whole idea at a glance — the standard MoE layer on the left, LatentMoE on the right. Every other section of this post unpacks *why* the two green "Latent" boxes buy you the freedom to double or quadruple the expert count for free.

![Figure 1 from Elango et al. (2026): standard MoE (left) routes tokens at full hidden width d; LatentMoE (right) inserts a latent down-projection before dispatch and an up-projection after combine, so routing and experts operate at the smaller latent width — which frees budget for 4x more experts (E1–E8 instead of E1–E4).](/imgs/blogs/latentmoe-fig1.webp)

## The problem: MoE is optimized for the wrong cost

Mixture-of-Experts models are the default recipe for scaling a language model's parameter count without scaling its per-token FLOPs. You keep a big bank of $N$ expert FFNs, but a router activates only the top-$K$ of them for each token. Parameters go up; compute per token stays flat. DeepSeek, Qwen, Mixtral, Grok, Kimi, Llama-4 — nearly every frontier open model is an MoE now. (If the router-and-experts picture is new to you, start with [DeepSeek's MoE lineage](/blog/machine-learning/large-language-model/deepseek-moe-lineage-fine-grained-shared-experts) and [serving MoE models at scale](/blog/machine-learning/model-serving/serving-moe-models-at-scale).)

Here is the thing the paper opens with, and it is worth sitting with: **the FLOP-per-token count that MoE is designed to hold constant is almost never the bottleneck at inference time.** The design lore — "keep FLOPs fixed, scale sparsity" — optimizes a quantity that does not dominate your serving bill. Two other quantities do, depending on the regime:

- In **interactive, low-latency serving** (small batches, one user waiting on each token), the GPU spends its time *loading expert weights out of HBM*, not multiplying. You are **memory-bandwidth-bound**.
- In **offline, high-throughput serving** (huge batches to saturate the GPU), the experts finally become compute-bound — and now the *all-to-all communication* that shuffles tokens to their assigned experts across GPUs dominates. You are **communication-bound**.

Both of those costs scale with the hidden dimension $d$ — the width of the token vector that gets loaded, routed, and shuffled. The FLOP budget MoE guards so carefully is a red herring for the regimes that actually matter. The paper's thesis is that you should design an MoE by minimizing **accuracy per FLOP *and* accuracy per parameter simultaneously**, where "per parameter" is standing in for memory footprint, bandwidth, communication, and sharding overhead.

The authors then do something refreshingly disciplined: before proposing anything, they build a quantitative cost model of *where the time actually goes*, derive five design principles from it, and only then design an architecture that the principles force. Let us walk that path exactly.

## Contributions in one map

Tightened into a numbered list, the paper delivers:

1. **A hardware-grounded cost model** for MoE serving on real NVIDIA silicon (GB200 / NVLink / H100), separating the memory-bandwidth-bound and communication-bound regimes with closed-form expressions.
2. **Five design principles** derived from that model plus classical expressivity theory (Barron functions, combinatorial sparsity) — a decision procedure that eliminates every architectural knob except one.
3. **LatentMoE**, the architecture those principles force: a latent down/up projection around the experts, with two configurations — *ℓ-MoEeff* (cheaper at equal accuracy) and *ℓ-MoEacc* (more accurate at equal cost).
4. **A design-space exploration** at 16B, 95B, and hybrid-73B scales over up to 1T training tokens, plus measured vLLM inference on H100s.
5. **A trillion-parameter projection** using an "Effective Parameter Multiplier" to argue that matching LatentMoE's accuracy with a plain-scaled MoE would cost ~350B extra parameters and a 1.24×–3.46× serving slowdown.

## Where the time actually goes: the serving cost model

The paper anchors its modeling on a concrete running example — **Qwen3-235B-A22B** — deployed on GB200 GPUs. Keep these numbers in your pocket; every derivation below plugs into them:

| Symbol | Meaning | Value |
| --- | --- | --- |
| $N$ | total routed experts | 128 |
| $K$ | active experts per token (top-$k$) | 8 |
| $d$ | model hidden dimension | 4096 |
| $m$ | expert intermediate (FFN) dimension | 1536 |
| $\text{EP}$ | expert-parallel degree (GPUs experts are sharded across) | 64 |
| $F$ | GB200 FP4 tensor-core throughput | ${10}$ PFLOP/s $= 10\times10^{15}$ |
| $\text{BW}_\text{HBM}$ | HBM bandwidth per GPU | ${8}$ TB/s $= 8\times10^{12}$ |
| $\text{BW}_\text{NVL}$ | one-way NVLink bandwidth per GPU | ${900}$ GB/s |

With $N = 128$ experts sharded across $\text{EP} = 64$ GPUs, each GPU hosts $N/\text{EP} = 2$ experts. That single fact — two experts per GPU — drives the whole analysis.

### Technique 1 — The roofline: why decode is memory-bandwidth-bound

**The problem it solves.** You want to know, before you write a kernel, whether an operation will be limited by the GPU's arithmetic units or by how fast it can pull operands from memory. Get this wrong and you optimize the thing that isn't the bottleneck.

**Intuition.** Think of the GPU as a giant kitchen. The chefs (tensor cores) are absurdly fast; the bottleneck is how fast the pantry (HBM) can hand them ingredients. If a recipe does a *lot* of chopping per ingredient fetched, the chefs stay busy — **compute-bound**. If it barely touches each ingredient before needing a new one, the chefs stand idle waiting on the pantry — **memory-bound**. The dividing line is a single number: how many FLOPs the hardware can do per byte it can fetch. (For the full treatment, see [roofline analysis for LLM inference](/blog/machine-learning/model-serving/roofline-analysis-for-llm-inference).)

**The mechanism.** An operation is compute-bound only if its **arithmetic intensity** $I$ — FLOPs performed per byte moved — exceeds the hardware's *ridge point*, the ratio of peak compute to peak bandwidth. Below the ridge you are bandwidth-bound; above it, compute-bound.

**The math.** For a GB200 the ridge point is

$$
\frac{F}{\text{BW}_\text{HBM}} = \frac{10 \times 10^{15}}{8 \times 10^{12}} = 1250 \ \text{FLOPs/byte}.
$$

Now compute a single expert's arithmetic intensity. Let $t_\text{exp}$ be the number of tokens routed to one expert (its effective batch size after routing). Assuming tokens spread uniformly, $t_\text{exp} = t_\text{total}\cdot K / N$ where $t_\text{total}$ is the tokens seen across the $\text{EP}$ GPUs before routing. One expert's FP4 compute cost is

$$
C_\text{exp} = 2 \cdot t_\text{exp} \cdot d \cdot m,
$$

the factor ${2}$ counting a multiply-add as two FLOPs, and $d\cdot m$ being the expert's weight matrix size. Its memory traffic — weights, inputs, and intermediate activations — is

$$
M_\text{exp} = d\cdot m + t_\text{exp}\cdot(d + m).
$$

Since each GPU runs two experts, both scale by ${2}$ and the factor cancels in the ratio, giving the arithmetic intensity

$$
I = \frac{2\, C_\text{exp}}{2\, M_\text{exp}} = \frac{2\, t_\text{exp}\, d\, m}{d\, m + t_\text{exp}(d + m)}.
$$

Read that expression: the numerator (compute) grows *linearly* with $t_\text{exp}$; the denominator is dominated by the fixed weight term $d\cdot m$ until $t_\text{exp}$ gets large. So at small batch, $I$ is tiny — you pay the full cost of loading the expert weights to process only a handful of tokens.

**Worked micro-example.** Set $I \ge 1250$ and solve for $t_\text{exp}$ with $d = 4096$, $m = 1536$ (so $d\cdot m = 6{,}291{,}456$ and $d + m = 5632$):

$$
\frac{2\, t_\text{exp}\, d\, m}{d\, m + t_\text{exp}(d + m)} \ge 1250
\;\Longrightarrow\;
t_\text{exp} \ge 1418.
$$

You need **1,418 tokens routed to each expert** just to cross into compute-bound territory. But a latency-critical deployment runs small batches — a few hundred tokens per expert at most, often far fewer. You sit deep in the memory-bound region, and the roofline plot makes it visceral: every latency-oriented operating point (the green dots at $t_\text{exp} = 16, 32, 64, 128, 256$) sits far down the diagonal slope, nowhere near the ridge.

![Figure 2 from Elango et al. (2026): roofline for one MoE expert on GB200. Latency-critical batch sizes (16–256 tokens per expert) sit on the bandwidth-bound diagonal; only very large batches (2048, 4096) reach the compute-bound ceiling at the ridge point of 1250 FLOPs/byte.](/imgs/blogs/latentmoe-fig2.webp)

**Why it works / when it fails.** In the memory-bound region, wall-clock time is *set by weight loading*, not by math. That leads to the paper's first principle. The one caveat: this assumes uniform token distribution across experts. Real routers are lumpy — a hot expert with 3× its share of tokens is more compute-bound than the average, which is exactly why aux-loss-free load balancing (below) matters in practice.

> **Design Principle I.** In low-latency serving, MoE inference is dominated by the memory-bandwidth cost of loading expert weights. Maximizing **accuracy per parameter** is therefore critical for interactive applications.

### Technique 2 — All-to-all accounting: why throughput serving is communication-bound

**The problem it solves.** Once you push the batch big enough that experts *are* compute-bound, a new cost takes over. Expert parallelism means each expert lives on a different GPU, so every token must be physically shipped to wherever its top-$K$ experts happen to be — an **all-to-all** shuffle — and then the results shipped back. On a fast NVLink fabric this is not free; the paper shows it dominates.

**Intuition.** Imagine a mailroom where every letter has to be hand-carried to $K$ different sorting centers across town and then carried back. Even if each center processes letters instantly, the *couriers* (the NVLink links) become the constraint once volume is high. The question is: how much does one GPU have to send and receive per layer, and how does that compare to the compute it does?

**The mechanism.** Per MoE layer, each GPU dispatches its $2\,t_\text{exp}$ local tokens out to other GPUs and receives combined results back. Count the bytes on the wire, count the FLOPs on-chip, and take the ratio.

**The math.** Communication volume per GPU per layer:

$$
M_\text{comm} = 2.5 \cdot \left(\frac{N}{\text{EP}} \cdot t_\text{exp} \cdot d\right) = 5\, t_\text{exp}\, d,
$$

where the ${2.5}$ is a mixed-precision factor — ${0.5}$ bytes/element for the FP4 **dispatch** plus ${2}$ bytes/element for the BF16 **aggregation** on the way back — and $N/\text{EP} = 2$. The compute for the two local experts is

$$
C_\text{comp} = 2 \cdot \left(\frac{N}{\text{EP}} \cdot t_\text{exp} \cdot d \cdot m\right) = 4\, t_\text{exp}\, d\, m.
$$

Turn each into a time by dividing by the relevant hardware rate — $t_\text{comm} = M_\text{comm}/\text{BW}_\text{NVL}$ and $t_\text{comp} = C_\text{comp}/F$ — and form the ratio:

$$
\frac{t_\text{comm}}{t_\text{comp}} = \frac{5\, t_\text{exp}\, d / \text{BW}_\text{NVL}}{4\, t_\text{exp}\, d\, m / F} = \frac{5\, F}{4\, m\, \text{BW}_\text{NVL}}.
$$

**Worked micro-example.** Notice $t_\text{exp}$ and $d$ *both cancel* — the ratio depends only on hardware and on $m$. Plug in $F = 10\times10^{15}$, $m = 1536$, $\text{BW}_\text{NVL} = 900\times10^{9}$:

$$
\frac{5 \cdot 10\times10^{15}}{4 \cdot 1536 \cdot 900\times10^{9}} \approx 9.
$$

Communication takes roughly **nine times** as long as the compute it feeds. Even in the "compute-bound" regime, the layer is overwhelmingly gated by all-to-all traffic.

**Why it works / when it fails.** The volume $M_\text{comm} \propto (N/\text{EP})\, t_\text{exp}\, d = t_\text{total}\, K\, d / \text{EP}$. Substituting $t_\text{exp} = t_\text{total} K / N$ makes $N$ cancel: **the traffic depends on $K$ and $d$, not on how many experts you own.** And crucially, the intermediate dimension $m$ does *not* enter $M_\text{comm}$ — it changes the compute, not the token size on the wire. That asymmetry is the seed of the whole architecture.

> **Design Principle II.** Throughput-oriented MoE is gated by all-to-all volume $M_\text{comm} \propto (N/\text{EP})\, t_\text{exp}\, d = t_\text{total} K d / \text{EP}$. Reduce it by cutting the routed hidden dimension $d$ or the active experts $K$. Changing $m$ does nothing for communication.

## What you're allowed to touch: five principles that leave only one knob

Principles I and II tell you *what* costs money: memory bandwidth scales with $d$ and $m$ (weight size $d\cdot m$), and communication scales with $K$ and $d$. So the naive move is to shrink all of them. The next three principles explain why you *can't* — except for one.

### Technique 3 — The expressivity floor: don't touch K or m

**The problem it solves.** If cutting $K$ and $m$ makes serving cheaper, why not just do it? Because model quality collapses, and the paper gives a theoretical reason rather than hand-waving.

**Intuition.** An MoE layer is, per token, a small neural network assembled from its selected experts. Classical approximation theory says the *quality* of a one-hidden-layer network is governed by how many nonlinear units it has to work with. Take units away and you provably lose approximating power. The "number of units" a token gets is the total width of the experts it activates.

**The math (Barron budget).** A classical result on Barron functions (Barron, 1993) says a one-hidden-layer network with $u$ nonlinear units achieves mean-squared error $\mathcal{O}(1/u)$, *independent of the input dimension*. In an MoE layer the effective nonlinear budget per token is proportional to the total width of the selected experts:

$$
U_\text{eff} \propto K \cdot m.
$$

Cut $K$ (fewer experts fire) or cut $m$ (each expert is narrower) and you directly shrink $U_\text{eff}$, raising the error floor.

**Worked micro-example.** Take the paper's 95B config: $K = 6$, $m = 2688$, so $U_\text{eff} \propto 6 \cdot 2688 = 16{,}128$. Halve $K$ to 3 and you halve the budget to $8{,}064$ — the Barron bound says your best-case error roughly *doubles*. This is why the paper refuses to cut either.

> **Design Principle III.** Preserving quality means preserving the effective nonlinear budget $U_\text{eff} \propto K\cdot m$. Do not reduce the active experts or the intermediate dimension.

### Technique 4 — The intrinsic rank: don't over-shrink d either

**The problem it solves.** Principles I–III leave $d$ as the one dimension both cost terms depend on that quality *seems* not to. But you can't shrink $d$ to zero — at some point you are throwing away information the task genuinely needs.

**Intuition.** Every task has some minimum number of "real" degrees of freedom in its token representation — its **intrinsic feature rank** $r_\text{eff}$. Compressing the hidden state below $r_\text{eff}$ is like scanning a document at too low a resolution: below a threshold, text becomes unreadable and no downstream cleverness recovers it. Above the threshold, the extra resolution was redundant and costs you nothing to drop.

**The mechanism / math.** There exists a task-dependent $r_\text{eff}$ such that any representation with dimension $\ge r_\text{eff}$ preserves task-relevant information, and any reduction of $d$ below $r_\text{eff}$ discards it and degrades accuracy. So $r_\text{eff}$ is a hard *lower bound* on how far $d$ (or a latent $\ell$ derived from it) can be squeezed. The paper estimates $r_\text{eff}$ empirically by sweeping the compression ratio (Section 4.1) and finds quality holds up to $\alpha = d/\ell = 4$, degrading beyond it.

> **Design Principle IV.** A task-specific feature rank $r_\text{eff}$ lower-bounds $d$. Reduce $d$ below it and quality collapses.

### Technique 5 — Combinatorial diversity: growing N and K is a free lunch for quality

**The problem it solves.** If shrinking $d$ frees up communication and bandwidth budget, what is the *best* thing to spend it on? The paper argues: more experts and larger top-$k$, because expert *combinations* are where MoE expressivity secretly lives.

**Intuition.** With $N$ experts and top-$K$ routing, each token picks one of $\binom{N}{K}$ possible expert *committees*. That combinatorial count is the model's specialization vocabulary — the number of distinct "mixtures of skills" it can assemble per token. Grow $N$ and $K$ together and this vocabulary explodes.

**The math.** Scaling both $N$ and $K$ by a factor $\alpha$ increases the number of expert combinations super-linearly:

$$
\binom{\alpha N}{\alpha K} \ge \binom{N}{K}^{\alpha}.
$$

**Worked micro-example.** Take a tiny case, $N = 4$, $K = 1$, $\alpha = 2$: $\binom{8}{2} = 28 \ge \binom{4}{1}^2 = 16$. Doubling both expert count and top-$k$ multiplied the committee vocabulary by more than the square. At real scale ($N = 128 \to 512$, $K = 8 \to 32$) the right-hand side is astronomically large; the point is the *direction* — more experts and larger $K$ monotonically enrich the model.

> **Design Principle V.** Scaling $N$ and $K$ together enhances quality by exponentially expanding the space of expert combinations.

### Putting it together

The five principles compose into a single forced conclusion, and it is worth stating as a chain:

- **Cost** wants $d$, $m$, $K$ small (Principles I, II).
- **Quality** forbids shrinking $K$ or $m$ (Principle III) and floors $d$ at $r_\text{eff}$ (Principle IV).
- That leaves **$d$ as the only knob** you can cut — down to $r_\text{eff}$ — for a win in *both* serving regimes.
- And the budget you free up should be **poured back into $N$ and $K$** (Principle V), which cost you nothing on the wire (traffic $\propto K\cdot d$, and you cut $d$ by the same factor you raise $K$) and nothing in bandwidth (weight size $\propto \ell\cdot m$, and you own more but smaller experts).

The matrix below is that reasoning as a decision table — read it row by row and the architecture designs itself. (The figure writes $\alpha$ as "a" and the latent dimension as "l".)

![Redrawn decision matrix: for each of d, m, K, N, whether cutting/raising it helps weight-load cost, all-to-all comm, and model quality. Green = the move helps, red = it hurts, amber = caution/floor, blue = verdict. The only clean verdict is "shrink d, then raise K and N by the same factor; keep m."](/imgs/blogs/latentmoe-2.webp)

## The LatentMoE layer

Now the architecture. LatentMoE takes each input token $x \in \mathbb{R}^{d}$ and, before doing anything expert-related, projects it down into a shared low-dimensional latent space with a learnable matrix, runs the experts *entirely inside that latent space*, then projects back up. The animation makes the payoff concrete: a narrower token on the same wire reaches more experts.

<figure class="blog-anim">
<svg viewBox="0 0 720 380" role="img" aria-label="Standard MoE sends a full-width token over NVLink to a few experts; LatentMoE sends a quarter-width latent token over the same wire to four times as many experts, so the total all-to-all traffic is unchanged" style="width:100%;height:auto;max-width:820px">
<title>Standard MoE vs LatentMoE: same wire traffic, narrower token, more experts</title>
<style>
.lm-hd{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.lm-sub{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.lm-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.lm-hot{fill:var(--accent,#6366f1);stroke:var(--accent,#6366f1)}
.lm-lab{font:600 12px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.lm-wht{font:600 12px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
.lm-wire{stroke:var(--border,#d1d5db);stroke-width:2;stroke-dasharray:5 5}
.lm-pk{fill:var(--accent,#6366f1)}
@keyframes lm-flow{0%{transform:translateX(0);opacity:0}10%{opacity:1}88%{opacity:1}100%{transform:translateX(196px);opacity:0}}
.lm-a{animation:lm-flow 6s linear infinite}
.lm-a2{animation-delay:2s}
.lm-a3{animation-delay:4s}
@media (prefers-reduced-motion:reduce){.lm-a{animation:none;opacity:1}}
</style>
<text class="lm-hd" x="20" y="34">Standard MoE</text>
<text class="lm-sub" x="150" y="34">token width d = 4096, dispatch to N experts (top-K)</text>
<rect class="lm-box" x="150" y="70" width="78" height="54" rx="8"/>
<text class="lm-lab" x="189" y="102">x : d</text>
<line class="lm-wire" x1="228" y1="97" x2="432" y2="97"/>
<rect class="lm-pk lm-a" x="232" y="78" width="46" height="38" rx="5"/>
<rect class="lm-hot" x="444" y="70" width="78" height="54" rx="8"/>
<text class="lm-wht" x="483" y="102">E1</text>
<rect class="lm-box" x="534" y="70" width="78" height="54" rx="8"/>
<text class="lm-lab" x="573" y="102">E2</text>
<rect class="lm-box" x="624" y="70" width="78" height="54" rx="8"/>
<text class="lm-lab" x="663" y="102">E3</text>
<text class="lm-hd" x="20" y="216">LatentMoE</text>
<text class="lm-sub" x="150" y="216">token width l = 1024, dispatch to 4N experts (top-4K)</text>
<rect class="lm-box" x="150" y="252" width="78" height="54" rx="8"/>
<text class="lm-lab" x="189" y="284">x : l</text>
<line class="lm-wire" x1="228" y1="279" x2="432" y2="279"/>
<rect class="lm-pk lm-a" x="232" y="270" width="46" height="10" rx="3"/>
<rect class="lm-pk lm-a lm-a2" x="232" y="270" width="46" height="10" rx="3"/>
<rect class="lm-hot" x="444" y="252" width="40" height="54" rx="6"/>
<rect class="lm-hot" x="490" y="252" width="40" height="54" rx="6"/>
<rect class="lm-box" x="536" y="252" width="40" height="54" rx="6"/>
<rect class="lm-box" x="582" y="252" width="40" height="54" rx="6"/>
<rect class="lm-box" x="628" y="252" width="40" height="54" rx="6"/>
<rect class="lm-box" x="674" y="252" width="40" height="54" rx="6"/>
</svg>
<figcaption>Same NVLink wire: a quarter-width latent token routed to 4x more experts moves the same total bytes (K x l x 4 = K x d), so LatentMoE trades token width for expert diversity at constant all-to-all traffic.</figcaption>
</figure>

### The mechanism, step by step

Trace one token through the layer, following the redrawn dataflow (which labels every tensor's shape — the paper's own Figure 1 shows the boxes but not the dimensions):

![Redrawn shape-annotated dataflow of the ℓ-MoEacc layer: token x (dim d) splits three ways — the router reads full x to pick top-K'; a down-projection W-down (shape ℓ×d) compresses x to the latent space where the experts (FC1/gate m×ℓ, FC2 ℓ×m) run; an up-projection W-up (d×ℓ) restores dim d; and a shared expert bypasses at full width, added to the output.](/imgs/blogs/latentmoe-1.webp)

1. **Down-project.** A learnable matrix $W_\downarrow \in \mathbb{R}^{\ell \times d}$ maps the token $x \in \mathbb{R}^{d}$ to a compressed latent vector $z = W_\downarrow x \in \mathbb{R}^{\ell}$, with $\ell \ll d$.
2. **Dispatch in the latent space.** The all-to-all shuffle now carries $\ell$-wide vectors, not $d$-wide ones — so communication drops by $\alpha = d/\ell$.
3. **Route.** The router still reads the *original* full-width token $x$ (routing is cheap and quality-sensitive, so it is not compressed): $p' = \text{softmax}(W_r' x)$, and the top-$K$ (or top-$K'$) experts are selected.
4. **Run experts in latent space.** Each routed expert $E_i(\cdot; \ell)$ lives *entirely* in $\mathbb{R}^{\ell}$: its weights are $W_\text{FC1}^{(i)}, W_\text{gate}^{(i)} \in \mathbb{R}^{m\times\ell}$ and $W_\text{FC2}^{(i)} \in \mathbb{R}^{\ell\times m}$. Because these are $\ell$-wide not $d$-wide, each expert's weights are $\alpha$-times smaller — so weight-loading bandwidth drops by $\alpha$ too.
5. **Combine and up-project.** Aggregate the expert outputs weighted by the routing scores, then map back to $\mathbb{R}^{d}$ with $W_\uparrow \in \mathbb{R}^{d\times\ell}$.
6. **Shared experts bypass.** A handful of always-on shared experts $E_j(\cdot; d)$ run at the *full* dimension $d$ and are added in — they are cheap relative to the routed experts and are left uncompressed to preserve a full-width pathway. (This is the [shared-expert](/blog/machine-learning/large-language-model/deepseek-moe-lineage-fine-grained-shared-experts) idea from DeepSeekMoE.)

### The math

Write $N' = \alpha N$ for the expanded set of routed experts and $S$ for the number of shared experts. The **efficiency** variant, $\ell\text{-MoE}_\text{eff}$, is:

$$
\ell\text{-MoE}_\text{eff}(x) := W_\uparrow \cdot \left(\sum_{i \in \mathcal{T}_{K, N'}} p'_i\, E_i(W_\downarrow x;\, \ell)\right) + \sum_{j = N'+1}^{N'+S} E_j(x;\, d).
$$

Every symbol: $W_\downarrow \in \mathbb{R}^{\ell\times d}$ down-projects; $E_i(\cdot; \ell)$ is a routed expert operating in the latent space; $p'_i$ is the softmax routing weight for expert $i$, computed as $p' = \text{softmax}(W_r' x)$ with $W_r' \in \mathbb{R}^{N'\times d}$ read from the *full* token $x$; $\mathcal{T}_{K, N'}$ is the index set of the top-$K$ experts out of $N'$; $W_\uparrow \in \mathbb{R}^{d\times\ell}$ up-projects the aggregated latent output back to $\mathbb{R}^{d}$; and the second sum is the $S$ shared experts $E_j(\cdot; d)$ running at full width. Note the top-$k$ is still $K$ here — this variant expands the *pool* $N \to N' = \alpha N$ but keeps the number of active experts fixed.

The **accuracy** variant, $\ell\text{-MoE}_\text{acc}$ (the recommended one), differs in exactly one place — it raises the active count too, to $K' = \alpha K$:

$$
\ell\text{-MoE}_\text{acc}(x) := W_\uparrow \cdot \left(\sum_{i \in \mathcal{T}_{K', N'}} p'_i\, E_i(W_\downarrow x;\, \ell)\right) + \sum_{j = N'+1}^{N'+S} E_j(x;\, d), \qquad K' = \alpha K.
$$

The only change is the selection set $\mathcal{T}_{K', N'}$ — top-$K'$ instead of top-$K$.

### Worked micro-example: why ℓ-MoEacc is a free lunch

This is the crux of the whole paper, so let us make it fully concrete with the 95B configuration: $d = 4096$, $\ell = 1024$ (so $\alpha = 4$), $m = 2688$, baseline $K = 6$, and $K' = \alpha K = 24$.

**Communication (per token):** standard MoE ships $K \cdot d = 6 \cdot 4096 = 24{,}576$ units. LatentMoE-acc ships $K' \cdot \ell = 24 \cdot 1024 = 24{,}576$ units. **Identical.** The 4× more active experts is exactly offset by the 4× narrower token.

**Active parameters (per token):** each latent expert is $\alpha$-times cheaper (its FFN matrices are $m\times\ell$ instead of $m\times d$), so $K'$ latent experts cost $\alpha K \cdot (m\ell) = \alpha K \cdot m (d/\alpha) = K\, m\, d$ — the *same* as the baseline's $K$ full-width experts. This is why Table 3 shows ℓ-MoEacc at **8.44B active params vs the baseline's 8.47B** — matched to a rounding error.

**Nonlinear budget (quality):** here is the payoff. $U_\text{eff} \propto K' \cdot m = 24 \cdot 2688 = 64{,}512$, versus the baseline's $6 \cdot 2688 = 16{,}128$. **A 4× larger Barron budget for the same FLOPs and the same wire traffic.** The token now assembles a committee of 24 specialists instead of 6, each still $m = 2688$ wide.

That is the trick in one line: **compress the routing width, and the identical serving budget now buys you $\alpha$× the expert committee and $\alpha$× the nonlinear budget.** The efficiency variant ℓ-MoEeff instead banks the savings — it keeps $K = 6$, so it matches baseline quality (same $U_\text{eff}$) but runs at ~${5.6}$B active params (fewer, cheaper experts), i.e. *lower cost at equal accuracy*.

**Why it works / when it fails.** Since $U_\text{eff}$ (via $m$) and the shared-expert path are untouched, Principle III predicts quality is preserved even in the eff variant; the acc variant then *gains* from Principle V's combinatorial explosion. The failure mode is Principle IV: push $\alpha$ past 4 (i.e. $\ell \lt r_\text{eff}$) and the down-projection starts discarding task-relevant information faster than more experts can compensate — which is exactly what the ablations show.

### A subtle honesty in the paper's own reasoning

The paper is careful to admit a gap: Principle III says compression *should* preserve accuracy for the eff variant on theoretical grounds ($U_\text{eff}$ unchanged), but "larger models are often easier to train and more robust to hyperparameters." A naively compressed model has fewer parameters and can be *harder to optimize* even if its capacity is theoretically sufficient. Their fix is to lean on Principle V: scale $N$ by $\alpha$ so the total parameter count is roughly restored, which stabilizes training *without* changing inference cost (recall traffic and bandwidth are independent of $N$). This is a real subtlety — the expert-count scaling is doing double duty as both a quality lever (diversity) and a training-stability lever (parameter count).

## The two configurations side by side

| Architecture | Communication cost | Weight-load / expert | Accuracy | Efficiency |
| --- | --- | --- | --- | --- |
| Standard MoE | $(N/\text{EP})\, t_\text{exp}\, d$ | $d\cdot m$ | — | — |
| $\ell\text{-MoE}_\text{eff}$ | $(N/\text{EP})\, t_\text{exp}\, \ell$ | $\ell\cdot m$ | → (matches) | ↑ (cheaper) |
| $\ell\text{-MoE}_\text{acc}$ (recommended) | $(N/\text{EP})\, t_\text{exp}\, d$ | $d\cdot m$ | ↑ (better) | → (same) |

Table 1 (reproduced above) is the summary. **ℓ-MoEeff** genuinely lowers both costs by $\alpha$ (it routes and stores in the latent space and keeps $K$), so it is *strictly cheaper at matched accuracy*. **ℓ-MoEacc** raises $K$ to $\alpha K$, which pushes communication and weight-loading back up to the *standard* MoE level — so it costs the same as the original but delivers higher accuracy. The before/after below is the same story in dollars-and-cents form:

![Redrawn before/after: Standard MoE (amber) routes tokens at full width d=4096 with d·m expert weights, N experts / top-K, and scores MMLU-Pro 29.3; LatentMoE ℓ-MoEacc (green) routes at latent ℓ=1024, stores ℓ·m expert weights (4x smaller), runs αN experts / top-αK at the same wire and HBM cost, and scores MMLU-Pro 34.9 (+5.6).](/imgs/blogs/latentmoe-3.webp)

## Experiments and results

The evaluation trains real models at three scales. The architectural specs (Table 2 in the paper):

| Config | 16BT-2BA | 95BT-8BA | Hybrid-73BT-8BA |
| --- | --- | --- | --- |
| Layers $L$ | 27 | 32 | 52 (24 Mamba/MoE, 4 Attn.) |
| Hidden $d$ | 2048 | 4096 | 4096 |
| Total experts $N$ | 64 | 128 | 128 |
| Active $K$ | 6 | 6 | 6 |
| Shared $S$ | 2 | 2 | 2 |
| FFN $m$ | 1408 | 2688 | 2688 |
| Activation | SwiGLU | Squared-ReLU | Squared-ReLU |
| Total / Active params | 16B / 2B | 95B / 8B | 73B / 8B |

The 16B model (built on DeepSeek-V2-Lite's config) is the ablation workhorse; the 95B and hybrid-73B are the scaling tests. All use DeepSeek's aux-loss-free load balancing (with a $10^{-4}$ balance coefficient) to keep the token-per-expert distribution uniform — which, recall, is the assumption the entire cost model rests on.

### Ablation 1 — how far can you compress? (finding r_eff)

The first ablation sweeps the compression ratio $\alpha = d/\ell$ on the 16B model with ℓ-MoEeff, scaling $N$ by $\alpha$ each time. Validation loss holds essentially on top of the baseline for $\alpha \le 4$ and starts to separate at $\alpha = 8$ — an empirical estimate that the intrinsic rank $r_\text{eff}$ sits somewhere below $\ell = d/4$. They adopt $\alpha = 4$ everywhere after this, and verify it still holds at 95B.

![Figure 3 from Elango et al. (2026): validation loss vs training tokens for the 16BT-2BA model at compression ratios α=4 (green) and α=8 (red) against baseline (blue). The zoomed inset shows α=4 tracking the baseline closely while α=8 sits measurably higher — the evidence for adopting α=4.](/imgs/blogs/latentmoe-fig3.webp)

### Ablation 2 — the expert scaling is load-bearing

The second ablation is the one that proves the *design*, not just the compression. Take the 16B model, compress $d$ by 4× — then either scale the expert count to $N' = \alpha N = 256$ (the LatentMoE prescription) or leave it at $N = 64$. Same compression, different expert count.

![Figure 4 from Elango et al. (2026): validation loss for the 4x-compressed 16BT-2BA model. The green curve (α=4, N scaled to 256) tracks the baseline; the red curve (α=4, N left at 64) sits clearly above — compressing d without scaling experts costs real accuracy.](/imgs/blogs/latentmoe-fig4.webp)

The green curve (scaled to $N = 256$) tracks the baseline; the red curve (left at $N = 64$) is visibly worse. **Compressing $d$ without the compensating expert scaling degrades quality** — confirming that the two moves are a package, exactly as Principle V predicts, and that the expert scaling also buys the training-stability benefit discussed above.

### Comparing the variants at 16B and 95B

With $\ell = 512$ at 16B (and $\ell = 1024$ at 95B, both $\alpha = 4$): ℓ-MoEeff sits right on the baseline validation loss, and ℓ-MoEacc sits *below* it. The story is identical at both scales — the same picture at 16B and at 95B is the paper's evidence that the effect is not a small-model artifact.

![Figure 5 from Elango et al. (2026): 16BT-2BA validation loss — ℓ-MoEeff (green) matches the baseline (blue) while ℓ-MoEacc (red) is consistently lower; the zoomed inset makes the ℓ-MoEacc gap clear.](/imgs/blogs/latentmoe-fig5.webp)

![Figure 6 from Elango et al. (2026): the same comparison at 95BT-8BA (ℓ=1024, α=4). ℓ-MoEeff tracks the baseline and ℓ-MoEacc opens a clear gap below it, confirming the 16B result holds at 95B scale.](/imgs/blogs/latentmoe-fig6.webp)

### Downstream accuracy

Now the numbers that matter. On the **95B model at a 300B-token horizon** (Code = avg of HumanEval/HumanEval+/MBPP/MBPP+; Math = avg of GSM8K-CoT/MATH-500; Commonsense = avg of RACE/ARC-Challenge/HellaSwag/Winogrande):

| 95BT-8BA model | Active | Total | MMLU-Pro | MMLU | Code | Math | Commonsense |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 8.47B | 94.4B | 29.26 | 58.95 | 40.33 | 64.39 | 74.32 |
| **ℓ-MoEacc** | 8.44B | 94.8B | **34.91** | **62.23** | **41.50** | **64.88** | **75.18** |
| ℓ-MoEeff | 5.62B | 94.8B | 34.75 | 61.06 | 40.68 | 63.61 | 73.72 |

The headline is **MMLU-Pro 29.26 → 34.91**, a +5.65 jump at matched active parameters (8.44B vs 8.47B). Notice the eye-catcher: ℓ-MoEeff *also* scores 34.75 on MMLU-Pro — nearly matching the acc variant — while using only **5.62B active params** (a third fewer FLOPs). Both LatentMoE variants dominate the baseline; the choice between them is a cost-vs-accuracy dial.

On the **hybrid Mamba-Attention 73B model at the full 1T-token horizon** (this is the architecture closest to what ships — a mostly-Mamba backbone with a few attention layers, the [Nemotron-H](/blog/machine-learning/large-language-model/nemotron-h-hybrid-mamba-transformer) style):

| Hybrid-73BT-8BA | Active | Total | MMLU-Pro | MMLU | Code | Math | Commonsense |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 8.09B | 72.6B | 48.30 | 70.10 | 51.95 | 78.32 | 81.73 |
| **ℓ-MoEacc** | 8.02B | 72.8B | **52.87** | **72.11** | **55.14** | **80.19** | **82.10** |
| ℓ-MoEeff | 5.91B | 72.8B | 51.29 | 71.34 | 53.13 | 77.01 | 80.78 |

Again ℓ-MoEacc wins across the board (MMLU-Pro +4.57, Code +3.19), and the effect *transfers to a non-Transformer backbone* — evidence that the mechanism is about the MoE layer, not about attention.

### Inference performance — read this table carefully

Here is where the paper earns its "honest" label, or should. Measured on **two H100 GPUs with vLLM and FP8 per-tensor quantization**, for the hybrid-73B model:

| Concurrency | LatentMoE (tok/s/GPU) | Standard MoE (tok/s/GPU) | LatentMoE vs Standard |
| --- | --- | --- | --- |
| 1 | 181.6 | 206.6 | **−12.1%** |
| 4 | 528.5 | 509.8 | +3.7% |
| 16 | 1130.8 | 1204.6 | −6.1% |
| 64 | 1569.6 | 1549.3 | +1.3% |
| 128 | 1625.8 | 1725.9 | −5.8% |

The paper's summary — "at higher concurrencies, per-GPU throughput drops by only up to 6%" — is fair, but it glosses the **concurrency-1 result: LatentMoE is 12% *slower*** than standard MoE. That is the deepest-latency, most-interactive point on the curve — the exact regime Principle I was written to serve. The reason is mechanical: LatentMoE's routed GEMMs are *smaller* (they operate in $\ell$, not $d$), and small GEMMs can leave the GPU's streaming multiprocessors underutilized. At batch 1 there is not enough work to hide the extra latent-projection launches and the thinner matrices. The paper flags two unimplemented fixes — separate CUDA streams for routed vs shared experts, and CUTLASS kernels specialized for small inner dimensions — but does not measure them. So the "constant inference cost" claim is an *architectural* iso-cost (same FLOPs, same bytes), not yet a *measured* iso-latency at batch 1.

### The trillion-parameter projection and the EPM trick

The most speculative — and most rhetorically important — part of the paper. To argue LatentMoE matters at frontier scale, the authors project a 1T-parameter model (a LatentMoE variant of Kimi-K2) using a **proprietary performance simulator** over 200K operating points.

**The Effective Parameter Multiplier (EPM).** The problem: how do you compare a LatentMoE model against a standard MoE *of the same accuracy* when they have the same parameter count? You can't just match parameters — LatentMoE is *more accurate* at equal params. So they construct a hypothetical standard-MoE baseline that is *bigger* until its accuracy matches.

Following the "Effective Parameter Count" framework (Frantar et al., 2025), assume a treated model with physical parameters $N_\text{treat}$ behaves like a dense baseline with *effective* parameters $N_\text{eff}$. If baseline accuracy follows a scaling law $f(N)$ and the treated model scores $S_\text{treat}$, then

$$
N_\text{eff} = f^{-1}(S_\text{treat}), \qquad \lambda = \frac{N_\text{eff}}{N_\text{treat}},
$$

where $\lambda$ is the effective parameter multiplier. They fit $f(N) = a\log N + b$ to the MMLU scores of the Qwen-3-Dense family (0.6B, 1.7B, 4B, 8B, 14B, 32B), estimate $\lambda \approx 1.35$ for Kimi-K2-1T-LatentMoE, and conclude that matching its accuracy with a standard MoE needs $N_\text{iso} = \lambda \cdot N_\text{treat} \approx 1.35\text{T}$ — an **extra ~350B parameters**. They then physically build that iso-accuracy baseline (Kimi-K2-1.35T) by growing depth from 61 to 80 layers, and simulate both.

![Figure 7 from Elango et al. (2026): projected throughput (tokens/s/GPU) vs latency (tokens/s/user) Pareto frontiers at trillion scale. Left: decode-heavy (chunked piggybacking). Right: prefill-heavy (disaggregated). Kimi-K2-1T-LatentMoE (blue) sits close to native Kimi-K2-1T (orange), while the accuracy-matched Kimi-K2-1.35T standard MoE (green) is 1.24x–3.46x slower across the frontier.](/imgs/blogs/latentmoe-fig7.webp)

The verdict: across the projected Pareto frontier, the accuracy-matched standard model (Kimi-K2-1.35T) is **1.24×–3.46× slower** than Kimi-K2-1T-LatentMoE. And the latent-projection overhead is modest — native Kimi-K2-1T stays within ~9% of the LatentMoE variant, so the extra projection matrices cost far less than the ~350B parameters you'd otherwise need. The conclusion the authors draw: reaching a target accuracy by *scaling a standard MoE* is much more expensive to serve than reaching it by *switching to LatentMoE*.

### What in their setup might not transfer

A few load-bearing choices to keep in mind before generalizing:

- **Uniform token distribution.** Every cost formula assumes tokens spread evenly across experts, which is why aux-loss-free load balancing is not optional here — a lumpy router breaks the arithmetic-intensity and communication estimates.
- **FP4 / FP8 precision and specific GPUs.** The ridge point (1250), the 2.5× mixed-precision comm factor, and the 9:1 comm:compute ratio are all GB200/H100/NVLink numbers. On a bandwidth-richer or compute-poorer machine the balance shifts.
- **$\alpha = 4$ and $r_\text{eff}$.** The compression ceiling was estimated on these models and datasets; a task with a higher intrinsic rank might tolerate less compression.
- **The trillion-scale story is simulated, not run.** Figure 7 is a proprietary simulator plus a 6-point log-linear MMLU fit, not a trained-and-served comparison.

## Critique

**What's genuinely strong.** The reasoning is unusually principled for an architecture paper. Rather than "we tried X and it worked," they build a hardware cost model, derive constraints from it, and let the constraints *force* the design — every knob is eliminated for a stated reason. The iso-parameter, iso-FLOP, *iso-hyperparameter* discipline (they reuse the baseline's tuned hyperparameters for LatentMoE, noting further tuning could only help) makes the accuracy gains hard to dismiss as a tuning artifact. And the effect replicating across a Transformer *and* a hybrid Mamba backbone, at two scales, over up to 1T tokens, is strong evidence.

**What's weak or unfalsifiable.**

- **The concurrency-1 regression undercuts the headline motivation.** Principle I is all about low-latency serving, yet at batch 1 LatentMoE is measurably *slower* (181.6 vs 206.6 tok/s). The fixes are named but unmeasured. As shipped, LatentMoE's win is clearest in the *throughput* regime, not the interactive one the paper leads with — a slight bait-and-switch.
- **The trillion-scale frontier is simulated.** The 1.24×–3.46× number — the paper's most quotable result — comes from a proprietary simulator and an "effective parameter" construction with a 6-point log-linear fit. A log-linear MMLU-vs-log-N fit over 0.6B–32B extrapolated to 1T is a long lever, and $\lambda \approx 1.35$ inherits all of that fragility. None of it is independently reproducible.
- **$r_\text{eff}$ is asserted, never measured directly.** Principle IV introduces an intrinsic feature rank as the hard floor on compression, but the paper never *measures* $r_\text{eff}$ (e.g. via the actual spectrum of the hidden states) — it only infers "$\le 4$" from where validation loss starts to bend. The theoretical object doing the heavy lifting is left empirical and coarse.
- **Barron budget is a loose analogy.** $U_\text{eff} \propto K\cdot m$ borrows a one-hidden-layer approximation bound and applies it to a deep, routed, gated architecture. It is a suggestive heuristic, not a theorem about MoE — and the paper leans on it as if it were the latter.

**What ablation is missing.** There is no ablation on *which* dimensions to compress — the router and shared experts are left at full width by assertion ("they do not significantly contribute to the bottlenecks"), but no experiment tests compressing them too, or compressing only FC2 (which is what the closest prior work, MoLAE, does). And there is no isolation of the training-stability effect from the diversity effect: both are conflated in the "scale $N$ by $\alpha$" move, so we cannot tell how much of ℓ-MoEeff's baseline-matching is capacity vs optimization.

**What would change my mind.** A single measured, apples-to-apples latency comparison at batch 1 *with* the promised CUDA-stream and small-GEMM CUTLASS optimizations, showing LatentMoE ℓ-MoEacc at parity-or-better tok/s with a standard MoE of equal accuracy — on real hardware, not a simulator. That would convert the trillion-scale projection from a plausible story into a demonstrated result, and would close the one gap between the paper's motivation (low-latency serving) and its measured evidence (throughput serving).

## What I'd build with this

These are my extrapolations, not the paper's claims:

- **LatentMoE ⊕ MLA end-to-end.** The latent-projection trick here is the MoE cousin of [Multi-head Latent Attention](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla), which compresses the KV cache into a latent space. A model that keeps tokens in a shared latent space across *both* attention and MoE — up-projecting only where full width is provably needed — might amortize the projection overhead the paper flags at batch 1.
- **Measure $r_\text{eff}$ directly.** Probe the effective rank of the hidden states per layer and per task (e.g. participation ratio of the activation covariance spectrum), then set a *per-layer* $\ell$ rather than one global $\alpha = 4$. Early layers may tolerate far more compression than late ones.
- **A small-GEMM kernel co-design.** The concurrency-1 regression is a kernel problem, not an architecture problem. A grouped-GEMM path (see [grouped GEMM for MoE](/blog/machine-learning/model-serving/grouped-gemm-moe-kernel)) specialized for the thin $m\times\ell$ latent experts, plus routed/shared expert stream overlap, is the obvious follow-up — and would test the paper's own hypothesis.
- **Stack with mHC.** The paper notes Manifold-Constrained Hyper-Connections (mHC) improves quality by widening the *residual* stream, orthogonal to LatentMoE's compression of the *expert* path. Training a model with both would test whether the two efficiency gains compose or interfere.

## References

- **Paper:** Venmugil Elango et al., "LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts," arXiv 2026. [arxiv.org/abs/2601.18089](https://arxiv.org/abs/2601.18089)
- **Adopted by:** NVIDIA, "Nvidia Nemotron 3: Efficient and Open Intelligence," arXiv 2025. [arxiv.org/abs/2512.20856](https://arxiv.org/abs/2512.20856)
- **Prior work referenced:** Barron (1993) on universal approximation bounds; Dai et al. (2024), DeepSeekMoE; Frantar et al. (2025), compression scaling laws (the EPM framework); Liu et al. (2025), MoLAE (the closest related latent-expert method).
- **On this blog:** [DeepSeek's MoE lineage](/blog/machine-learning/large-language-model/deepseek-moe-lineage-fine-grained-shared-experts) · [Multi-head Latent Attention](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla) · [Scaling laws for Mixture-of-Experts](/blog/machine-learning/scaling-laws/moe-scaling-laws) · [Roofline analysis for LLM inference](/blog/machine-learning/model-serving/roofline-analysis-for-llm-inference) · [Serving MoE models at scale](/blog/machine-learning/model-serving/serving-moe-models-at-scale)

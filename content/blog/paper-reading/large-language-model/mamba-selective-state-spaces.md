---
title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
date: "2026-07-16"
description: "A detailed, intuition-first walkthrough of Mamba — how making state-space parameters depend on the input unlocks content-based reasoning, why that breaks the convolution trick, and how a hardware-aware scan makes it fast, all the way down to the reference code."
tags: ["paper-reading", "mamba", "state-space-models", "ssm", "selective-scan", "s4", "linear-attention", "long-context", "sequence-modeling", "efficient-architectures"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 34
paper:
  title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  authors: "Albert Gu and Tri Dao"
  venue: "arXiv 2023 (v2 2024) / COLM 2024"
  url: "https://arxiv.org/abs/2312.00752"
---

> [!tldr]
> - **What it proposes.** Mamba is a sequence model built from *selective state space models* (S6): a recurrent state-space layer whose parameters $\Delta, B, C$ are **functions of the current token** rather than fixed constants. That one change lets the model choose what to remember and what to throw away — content-based reasoning that ordinary state-space models (SSMs) and linear-attention variants cannot do.
> - **The key mechanism in one line.** Make the recurrence *time-varying*, then compute it with a **hardware-aware parallel scan** that keeps the large hidden state in fast on-chip SRAM and never writes it to slow HBM.
> - **Why it matters.** Mamba is the first attention-free, linear-time model to match a strong Transformer recipe on language modeling — with **5× higher inference throughput** and no KV cache — while scaling to million-length sequences on DNA and audio.
> - **The most surprising result.** Trained on the induction-heads task at length 256, Mamba solves it *perfectly at length 1,048,576* — 4000× longer than training. No attention or prior SSM baseline generalizes beyond 2×.
> - **Where it fails.** Selection is a double-edged sword: on smooth, uniformly-sampled continuous signals (raw audio waveforms) the input-dependent dynamics actually *hurt*, because that data wants the time-invariant inductive bias selection throws away. And every claim is at ≤3B parameters — the paper never shows Mamba holds up at 7B+.

![Figure 1 from Gu and Dao (2023): the selective SSM maps each channel of the input through an N-dimensional latent state, with the input-dependent Δ, B, C parameters produced by the selection mechanism and the state expansion computed inside fast GPU SRAM.](/imgs/blogs/mamba-selective-state-spaces-fig1.webp)

The diagram above is the whole method at a glance: an input $x_t$ is projected into per-token parameters $B_t, C_t, \Delta_t$ (the "Selection Mechanism" arrow in blue), those are discretized, and a recurrence over the $N$-dimensional latent state $h$ produces the output $y_t$ — all while the expanded state lives in the fast top of the GPU memory hierarchy. The rest of this post unpacks every piece of it, from the intuition through the math down to the reference implementation.

## The problem attention leaves on the table

Almost every foundation model today is a Transformer, and the reason is [self-attention](/blog/paper-reading/large-language-model/roformer-enhanced-transformer-with-rotary-position-embedding): it routes information densely between every pair of positions in a context window, which makes it extraordinarily good at modeling relationships in data. But that density is also its curse. Attention has two structural problems:

1. **It cannot model anything outside its finite window.** There is no state that carries information forward; everything the model "knows" at position $t$ must be recomputed by attending back over the raw tokens.
2. **It scales quadratically.** Computing attention over a length-$L$ sequence costs $O(L^2)$ time, and autoregressive generation must keep the entire context in a **KV cache** whose size grows linearly with the sequence — the dominant memory and bandwidth cost at inference time (see the [survey on KV-cache management](/blog/paper-reading/large-language-model/a-survey-on-large-language-model-acceleration-based-on-kv-cache-management)).

An enormous literature has tried to fix attention's efficiency — sparse attention, linear attention, low-rank approximations — but, as the authors put it, "none of these variants have been shown to be empirically effective at scale across domains." They usually buy efficiency by giving up exactly the property that made attention work.

A different lineage sidesteps attention entirely: **structured state space models (S4** and descendants). These are recurrent models inspired by classical control theory. They compute in linear or near-linear time, they have principled mechanisms for long-range dependencies, and they dominate long-range benchmarks on continuous signals like audio and images. But on **discrete, information-dense data — text and DNA — they underperform.** For years, nobody could make a linear-time recurrent model reach Transformer quality on language.

Mamba's thesis is that the reason SSMs failed on language is a *single, nameable weakness*, and that fixing it — plus solving the engineering problem the fix creates — closes the gap.

### The framing: sequence modeling is context compression

The authors offer one lens that makes the whole paper click. **Sequence modeling is the problem of compressing context into a state.** Look at the two extremes through that lens:

- **Attention doesn't compress at all.** Its "state" is the entire KV cache — every past token, stored verbatim. That is why it's so effective (nothing is lost) and so expensive (nothing is saved). Quadratic training, linear-time-and-memory inference.
- **Recurrent models compress aggressively.** They summarize the whole past into a fixed-size hidden state, which is why they're cheap: constant-time inference, linear-time training. But their quality is capped by *how well that fixed state compressed the context*.

The efficiency–effectiveness tradeoff is exactly this tension: an efficient model needs a small state; an effective model needs a state that kept everything relevant. So the central question becomes: **can a model with a small, fixed state be smart about *what* it keeps?** That ability — the content-aware skill of focusing on some inputs and filtering out others — is what the paper calls **selectivity**, and it's the missing ingredient.

## The paper's contributions

Stripped to its load-bearing pieces, the paper makes four contributions:

1. **The selection mechanism (S6).** Make the SSM parameters $\Delta, B, C$ input-dependent, turning a time-*invariant* recurrence into a time-*varying* one that can selectively propagate or forget information along the sequence.
2. **A hardware-aware selective scan.** Time-varying SSMs can no longer be computed as a convolution, so the authors design a fused parallel-scan kernel that keeps the expanded state in SRAM, uses recomputation to save memory, and runs faster than an optimized attention kernel beyond ~2K tokens.
3. **The Mamba architecture.** Fold the SSM-based "H3" block and the Transformer's MLP block into one simple, homogeneous block — no attention, no separate MLP — and stack it.
4. **Empirical validation as a general backbone.** State-of-the-art or Transformer-matching results across language, DNA, and audio, with 5× inference throughput and monotonic improvement up to million-length context.

We'll take each technique in turn, always building the intuition before the equations. Let's start with the foundation everything sits on: what a state space model actually *is*.

## Background: what a state space model actually is

### The problem it solves

Before we can make an SSM *selective*, we need the ordinary SSM. The goal is a sequence-to-sequence map that is (a) cheap to run one step at a time (for generation) and (b) parallelizable across the whole sequence (for training). Attention gives you (b) but not (a); a vanilla RNN gives you (a) but not (b). Structured SSMs, remarkably, give you *both* — as long as they stay time-invariant.

### Intuition: a leaky tank with a dial

Picture a set of tanks holding water. At each moment, some water flows in from the input, and each tank also leaks out at its own rate. The current water levels are the **latent state** $h$; the input is a faucet; the output is a set of gauges reading weighted combinations of the levels. If the leak rates and faucet valves never change, the whole system is **linear time-invariant (LTI)** — its behavior at any moment depends only on the pattern of inputs, not on *when* they arrive. That single property is what will later let us compute the system as either a step-by-step recurrence or a single global convolution. (Keep the tanks in mind — when we get to selection, the whole trick will be letting each incoming token *turn the dials* on the valves and leaks.)

### The mechanism, step by step

A continuous-time SSM maps a 1-D signal $x(t) \in \mathbb{R}$ to an output $y(t) \in \mathbb{R}$ through an $N$-dimensional latent state $h(t) \in \mathbb{R}^N$. It's defined by four objects: a state-transition matrix $A$, an input matrix $B$, an output matrix $C$, and a step size $\Delta$. The dynamics are an ordinary differential equation plus a readout:

$$
h'(t) = A\,h(t) + B\,x(t), \qquad y(t) = C\,h(t)
$$

Here $A \in \mathbb{R}^{N \times N}$ says how the state evolves on its own (the leak rates), $B \in \mathbb{R}^{N \times 1}$ says how the input enters the state (the faucets), and $C \in \mathbb{R}^{1 \times N}$ says how the state is read out (the gauges). To use this on a discrete sequence of tokens, we **discretize**: convert the continuous $(A, B)$ into discrete $(\bar{A}, \bar{B})$ using the step size $\Delta$. The standard rule is the **zero-order hold (ZOH)**:

$$
\bar{A} = \exp(\Delta A), \qquad \bar{B} = (\Delta A)^{-1}\big(\exp(\Delta A) - I\big)\cdot \Delta B
$$

Read $\bar{A} = \exp(\Delta A)$ as: "how much of the current state survives one time-step of size $\Delta$." Because $A$'s eigenvalues are negative (the tanks leak), $\exp(\Delta A)$ has magnitude in $(0,1)$ — a decay. A **large $\Delta$** means a long step, so more of the input is integrated and the old state decays more; a **small $\Delta$** means a short step where the input barely registers and the state persists. Hold onto that — it becomes the entire gating story later.

Once discretized, the same model can be computed **two equivalent ways**:

$$
\underbrace{h_t = \bar{A}\,h_{t-1} + \bar{B}\,x_t, \quad y_t = C\,h_t}_{\text{(2) linear recurrence}}
\qquad\qquad
\underbrace{\bar{K} = \big(C\bar{B},\ C\bar{A}\bar{B},\ \dots,\ C\bar{A}^{k}\bar{B},\ \dots\big), \quad y = x * \bar{K}}_{\text{(3) global convolution}}
$$

The recurrence (2) is the leaky-tank simulation run one token at a time — perfect for autoregressive inference, because generating one more token costs $O(1)$ and needs no cache of the past. The convolution (3) *unrolls* that recurrence into a single fixed kernel $\bar{K}$ and applies it with an FFT — perfect for training, because the whole sequence is processed in parallel. **They compute the exact same function.** This duality is the reason LTI SSMs are special, and it's worth its own picture.

![A redrawn view of the recurrence–convolution duality: one discrete SSM computes either as a sequential recurrence (cheap inference) or a parallel convolution (cheap training), and selection sacrifices the convolution branch.](/imgs/blogs/mamba-selective-state-spaces-2.webp)

### The math, with shapes

To operate on a real batch, the SSM is applied **independently to each channel**. With batch size $B$, sequence length $L$, and $D$ channels, and a diagonal $A$ (the "structured" part — you store $A$ as just $N$ numbers per channel), the total hidden state has shape $(B, L, D, N)$. That factor of $N$ is the crux of everything that follows: the effective state is $N$ times larger than the input $x$ or output $y$, which are $(B, L, D)$. Computing the recurrence over the sequence costs $O(BLDN)$ time and memory. Prior SSMs used $N \approx 10\text{–}100$ — a much larger effective state than a traditional RNN, which is a big part of why they work — but only because the convolution mode let them avoid ever materializing that $(B, L, D, N)$ tensor.

### Why it works, and the crack that dooms it

Linear time-invariance is the property that makes the duality hold: because $(\bar{A}, \bar{B}, C)$ are the same at every step, you can precompute the convolution kernel once. It's also the property the authors are about to destroy. The problem: **an LTI system applies identical dynamics to every token.** The transition $\bar{A}$ and input map $\bar{B}$ don't depend on *what* the token is. Such a model can track *position* ("what happened 5 steps ago") but not *content* ("remember the important token, ignore the filler"). And that, the next section argues, is exactly the skill language requires.

## Selection: letting the model choose what to remember

### The problem it solves, made concrete

The authors motivate selection with two synthetic tasks that isolate the failure mode. Look at them — the whole design follows from wanting to solve these.

![Figure 2 from Gu and Dao (2023): the Copying task (constant spacing, solvable by time-invariant models), the Selective Copying task (random spacing between the colored tokens to remember and the white noise to ignore), and the Induction Heads task (retrieve the token that followed a query the last time it appeared).](/imgs/blogs/mamba-selective-state-spaces-fig3.webp)

- **Copying** (left) asks the model to reproduce a set of tokens after a fixed delay. The spacing is *constant*, so a time-invariant model solves it trivially — build a convolution kernel of exactly the right length and you never have to look at the tokens' content. LTI models ace this.
- **Selective Copying** (top right) breaks the shortcut by putting **random amounts of white "noise" between the colored tokens** you must remember. Now the model must be *content-aware*: recognize a colored token, store it; recognize noise, skip it. A static convolution kernel can't do this, because the spacing it would need to encode is different every time.
- **Induction Heads** (bottom right) is the mechanism believed to underlie much of in-context learning: having seen the bigram "Harry Potter," when "Harry" appears again, predict "Potter." It requires associative recall — retrieving the right answer conditioned on context.

LTI SSMs and global convolutions fail Selective Copying and Induction Heads for the same reason: their constant dynamics can't decide, based on the current token, whether to write it into the state or let it pass. The paper is blunt that architecture *gating* — the multiplicative interactions that H3 and Hyena add — is **not** enough, because gating doesn't interact along the sequence axis and so can't change the spacing between tokens.

### Intuition: a bouncer with a memory

Think of the hidden state as a small VIP room and each token as someone arriving at the door. An LTI SSM has no bouncer — everyone is treated identically by a fixed rule, so noise crowds in and drowns the signal. **Selection installs a bouncer who reads each arrival and decides: let this one reshape the room's memory, or wave it past untouched.** The bouncer's decision is *a function of who is at the door* — the token's content — which is precisely what "input-dependent parameters" means. That is the entire idea; the animation makes the consequence vivid.

<figure class="blog-anim">
<svg viewBox="0 0 680 220" role="img" aria-label="A stream of eight tokens; the five noise tokens fade out while the three signal tokens slide left to form a small compressed state" style="width:100%;height:auto;max-width:760px">
<title>Selection as compression: noise is filtered out and signal is packed into a small fixed state</title>
<style>
.am-sig{fill:var(--accent,#6366f1)}
.am-noise{fill:var(--border,#d1d5db)}
.am-lbl{font:600 15px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
.am-cap{font:600 15px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
@keyframes am-fade{0%,15%{opacity:1}45%,88%{opacity:.12}100%{opacity:1}}
@keyframes am-s1{0%,20%{transform:translateX(0)}50%,88%{transform:translateX(-156px)}100%{transform:translateX(0)}}
@keyframes am-s2{0%,20%{transform:translateX(0)}50%,88%{transform:translateX(-312px)}100%{transform:translateX(0)}}
.am-drop{animation:am-fade 9s ease-in-out infinite}
.am-mv1{animation:am-s1 9s ease-in-out infinite}
.am-mv2{animation:am-s2 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.am-drop,.am-mv1,.am-mv2{animation:none}}
</style>
<text class="am-cap" x="340" y="30">input stream (signal + noise)</text>
<rect class="am-sig" x="40"  y="60" width="56" height="56" rx="8"/>
<rect class="am-noise am-drop" x="118" y="60" width="56" height="56" rx="8"/>
<rect class="am-noise am-drop" x="196" y="60" width="56" height="56" rx="8"/>
<rect class="am-sig am-mv1" x="274" y="60" width="56" height="56" rx="8"/>
<rect class="am-noise am-drop" x="352" y="60" width="56" height="56" rx="8"/>
<rect class="am-noise am-drop" x="430" y="60" width="56" height="56" rx="8"/>
<rect class="am-sig am-mv2" x="508" y="60" width="56" height="56" rx="8"/>
<rect class="am-noise am-drop" x="586" y="60" width="56" height="56" rx="8"/>
<text class="am-lbl" x="68"  y="94">s1</text>
<text class="am-lbl am-mv1" x="302" y="94">s2</text>
<text class="am-lbl am-mv2" x="536" y="94">s3</text>
<text class="am-cap" x="150" y="160">compressed state (kept)</text>
</svg>
<figcaption>Selection filters out the noise tokens and compresses the three signal tokens into a small fixed-size state — the paper's core "selection as compression" idea, and exactly what the Selective Copying task demands.</figcaption>
</figure>

### The mechanism: parameters become functions of the input

Here is the change, and it is astonishingly small. In the original SSM (call it S4), $\Delta, A, B, C$ are learned constants. In the selective SSM (S6), we make three of them **functions of the input token**, with a length dimension $L$ that they didn't have before:

- $B_t = s_B(x_t)$ and $C_t = s_C(x_t)$, each a linear projection of the token to dimension $N$: $s_B(x) = \text{Linear}_N(x)$, $s_C(x) = \text{Linear}_N(x)$.
- $\Delta_t = \tau_\Delta(\text{Parameter} + s_\Delta(x_t))$, where $s_\Delta(x) = \text{Broadcast}_D(\text{Linear}_1(x))$ and $\tau_\Delta = \text{softplus}$.

$A$ stays a static parameter (we'll see why that's fine). Laid out object by object, the diff is small but total — and the shape annotations are the point:

| Object | S4 (Algorithm 1) | S6 (Algorithm 2) | What changed |
|---|---|---|---|
| $A$ | $(D, N)$, learned parameter | $(D, N)$, learned parameter | unchanged |
| $B$ | $(D, N)$, fixed | $(B, L, N) = s_B(x)$ | input-dependent |
| $C$ | $(D, N)$, fixed | $(B, L, N) = s_C(x)$ | input-dependent |
| $\Delta$ | $(D)$, fixed | $(B, L, D) = \tau_\Delta(\text{Param} + s_\Delta(x))$ | per-token |
| $\bar{A}, \bar{B}$ | $(D, N)$ | $(B, L, D, N)$ | gains an $L$ axis |
| computation | convolution **or** recurrence | recurrence (scan) **only** | loses the conv |

Notice what happened to the shapes. In S4, $B$ and $C$ are $(D, N)$ — one fixed matrix, reused at every position. In S6 they are $(B, L, N)$ — **a different $B_t$ and $C_t$ at every position**, selected by the token. And the discretized $\bar{A}, \bar{B}$ blow up from $(D, N)$ to $(B, L, D, N)$: a full per-token, per-channel transition. This is the "before/after" that defines the whole model.

![A redrawn before/after: S4 applies the same fixed A, B, C, Δ to every token (content-blind, fails Selective Copying); S6 makes B, C, Δ linear functions of the token, so the dynamics reshape per token and the model can filter noise and keep signal.](/imgs/blogs/mamba-selective-state-spaces-1.webp)

### A worked micro-example

Take the tiniest case: state size $N = 1$, one channel, so $\bar{A}_t, \bar{B}_t, C_t$ are scalars that now vary with $t$. Say we process three tokens $x_1, x_2, x_3$ where $x_2$ is noise. Selection can set $\Delta_2 \to 0$, which makes $\bar{A}_2 = \exp(\Delta_2 A) \to 1$ and $\bar{B}_2 \to 0$. The recurrence at step 2 becomes:

$$
h_2 = \bar{A}_2 h_1 + \bar{B}_2 x_2 \;\approx\; 1\cdot h_1 + 0\cdot x_2 = h_1
$$

The noise token is *ignored* — the state passes through unchanged. For the signal tokens, selection sets a large $\Delta$, so $\bar{A}$ shrinks toward zero (forget the stale state) and $\bar{B}$ grows (write the new input in). One state, one scalar per token, and the model has learned to skip filler and latch onto content. Scale that from $N=1$ to $N=16$ and from one channel to thousands, and you have S6.

### Why it works, and the price

Selection works because it lets the finite state be *smart*: instead of compressing the context with a fixed rule, the model compresses it with a **content-dependent rule**, resetting or preserving the state token by token. That's the resolution of the efficiency–effectiveness tradeoff from the intro — a small state that keeps the right things.

The price is severe and is the reason no one had done this at scale: **making $B, C, \Delta$ time-varying destroys the recurrence–convolution duality.** The convolution kernel $\bar{K}$ only exists because $(\bar{A}, \bar{B}, C)$ are constant across time; once they change every step, there is no single kernel to precompute, no FFT, no parallel training path. You're left with a sequential recurrence — which on a GPU is death unless you're clever. Solving that is the third contribution.

## Δ is a gate: the discretization connection

### The problem it solves

We introduced $\Delta$ as a discretization step size and $s_\Delta, \tau_\Delta = \text{softplus}$ as odd-looking choices. Why *those* choices? Because there's a beautiful theorem hiding here: **the selective $\Delta$ is a generalization of the forget gate in an LSTM/GRU.** Understanding this both justifies the parameterization and tells you exactly what $\Delta$ controls.

### Intuition: Δ is an attention span in time

$\Delta$ balances how much the model focuses on the current input versus persists its accumulated state. A **large $\Delta$** resets the state and locks onto the current token (the system "dwells" on this input); a **small $\Delta$** ignores the current token and coasts on memory. It is a per-token, per-channel dial for "pay attention now" vs. "keep coasting" — which is precisely what a recurrent gate does.

### The math: Theorem 1

The paper proves that a specific instance of the selective SSM *is* a classic gated recurrence. Take $N = 1$, $A = -1$, $B = 1$, $s_\Delta = \text{Linear}(x)$, $\tau_\Delta = \text{softplus}$. The continuous system $h'(t) = -h(t) + x(t)$ is a leaky integrator. Apply the ZOH discretization and the algebra collapses:

$$
\bar{A}_t = \exp(\Delta_t A) = \frac{1}{1 + \exp(\text{Linear}(x_t))} = 1 - \sigma(\text{Linear}(x_t)), \qquad \bar{B}_t = 1 - \bar{A}_t = \sigma(\text{Linear}(x_t))
$$

Defining the gate $g_t = \sigma(\text{Linear}(x_t))$, the recurrence becomes exactly the gated update you'd write for a minimal GRU:

$$
g_t = \sigma(\text{Linear}(x_t)), \qquad h_t = (1 - g_t)\,h_{t-1} + g_t\,x_t
$$

When $g_t \to 1$ (large $\Delta$), the state is overwritten by the current input — a hard reset, useful at document boundaries. When $g_t \to 0$ (small $\Delta$), the state is preserved and the input ignored — the noise-skipping from our micro-example. The softplus and the "project to 1 dimension, then broadcast to $D$ channels" structure of $s_\Delta$ both fall out of wanting this gate interpretation: if a token should be ignored, *all* channels should ignore it, so you compute one scalar decision and broadcast it.

### Why it works: three mechanistic effects

The paper names three concrete things selection buys you, all downstream of this gate:

- **Variable spacing.** The model can filter out arbitrary noise tokens between items of interest (the "um"s of language) — Selective Copying, solved.
- **Filtering context.** Many long-context models plateau or *degrade* as context grows, because they can't ignore irrelevant history. A selective model can reset its state to drop stale context, so in principle its quality improves *monotonically* with context length — which the DNA experiments confirm.
- **Boundary resetting.** When independent sequences are packed together (documents concatenated for training efficiency, episodes in RL), an LTI model bleeds information across the boundary; a selective model can slam $\Delta_t \to \infty$ and cleanly reset.

And a note on $A$: the authors leave it non-selective because it only ever affects the model *through* $\Delta$ via $\bar{A} = \exp(\Delta A)$. Making $\Delta$ selective already makes $\bar{A}$ selective; adding selectivity to $A$ directly is redundant, so they drop it for simplicity.

## Making it fast: the hardware-aware selective scan

### The problem it solves

We just gave up the convolution. The naive alternative is to run the recurrence, which forces us to materialize the $(B, L, D, N)$ state tensor — that $N$-times blowup — in GPU **HBM** (the large, slow "GPU memory"). At $N = 16$ and typical dimensions, that is a lot of memory traffic, and since the scan is memory-bandwidth-bound (like most non-matmul GPU ops), all that traffic to and from HBM is the bottleneck. Worse, the recurrence is *sequential*, which GPUs hate. Two problems: too much memory movement, and no parallelism.

### Intuition: do the heavy work on the scratchpad, not the filing cabinet

The GPU memory hierarchy is a small, blazing-fast scratchpad (SRAM, ~19 TB/s) sitting on top of a big, slow filing cabinet (HBM, ~1.5 TB/s). The naive kernel keeps walking to the filing cabinet to read and write the huge intermediate state. The **fused kernel loads the small inputs into the scratchpad once, builds the huge state *there*, consumes it *there*, and walks back to the filing cabinet only to file the small final answer.** The $N$-times-bigger state never touches HBM. This is the same IO-awareness idea behind FlashAttention, applied to a scan.

![A redrawn view of the hardware-aware selective scan: only the small inputs (Δ, A, B, C) and output y ever cross to slow HBM; discretization, the associative scan that builds the N-times-larger state h, and the readout via C all happen inside fast SRAM, with the state recomputed in the backward pass rather than stored.](/imgs/blogs/mamba-selective-state-spaces-3.webp)

### The mechanism: three classical techniques

The selective scan combines three well-known tricks:

1. **Kernel fusion.** Instead of writing $\bar{A}, \bar{B}$ of size $(B, L, D, N)$ to HBM, calling a scan, and reading it back, the kernel reads only the SSM parameters $(\Delta, A, B, C)$ — size $O(BLD + DN)$ — from HBM to SRAM, then discretizes, scans, and multiplies by $C$ all in SRAM, writing back only the output $y$ of size $(B, L, D)$. This cuts memory IO by a factor of $O(N)$, worth a **20–40× speedup** over a standard PyTorch scan in practice.
2. **Parallel associative scan.** The recurrence looks inherently sequential, but a linear recurrence is an *associative* operation, so it can be computed with a work-efficient **parallel scan** (Blelloch-style) in $O(\log L)$ depth instead of $O(L)$. This recovers parallelism across the sequence for training.
3. **Recomputation.** The backward pass needs the intermediate states, but storing the $(B, L, D, N)$ tensor is exactly what we're avoiding. So the kernel *recomputes* those states on the fly in the backward pass when the inputs are reloaded to SRAM — trading a little extra compute for a large memory saving, and landing the selective SSM at the **same memory footprint as an optimized FlashAttention Transformer**.

### The math: why the recurrent mode can actually be cheaper

Counting FLOPs, the naive recurrence uses $O(BLDN)$ and the convolution uses $O(BLD \log L)$. For long sequences and a not-too-large state dimension $N$, the recurrence's linear-in-$L$ scaling (with a small constant factor) actually uses *fewer* FLOPs than the convolution's $\log L$ factor — and it's exact linear time, versus the pseudo-linear $L \log L$ of every convolution-based SSM. Combined with the IO savings, the selective scan is faster than FlashAttention-2 beyond sequence length ~2K and up to **7× faster at 32K tokens**.

### When it fails

The catch is that this is a **custom CUDA kernel**, not something you get for free from a framework. Without the fused implementation, a time-varying SSM is slow and memory-hungry — which is exactly why the RWKV and RetNet baselines are missing from some of the paper's long-context comparisons (no efficient implementation → out-of-memory). And for sequences too long to fit even the parameters in SRAM, the scan must be chunked. The idea is simple; the engineering is not, and it's a big part of why Mamba worked when earlier selective attempts didn't.

## The Mamba block: one homogeneous layer

### The problem it solves

A selective SSM is a sequence transformation, not a whole network. Prior SSM architectures (H3 and its descendants) interleave two different blocks: a sequence-mixing block inspired by linear attention, and a standard MLP. That's two block types to stack and tune. Can we simplify?

### Intuition and mechanism

Yes: **fold them into one.** The Mamba block takes the H3 block, takes the ubiquitous gated-MLP block, and merges them into a single block that gets repeated homogeneously — the same move the Gated Attention Unit made for attention. The figure shows the lineage.

![Figure 3 from Gu and Dao (2023): the Mamba block combines the H3 block (SSM sandwiched by gates, with a preceding convolution) and the gated-MLP block into one homogeneous block; compared to H3 it swaps the first multiplicative gate for an activation, and compared to the MLP block it adds an SSM to the main branch.](/imgs/blogs/mamba-selective-state-spaces-fig2.webp)

Concretely, each block expands the model dimension $D$ by a factor $E$ (fixed to $E = 2$), runs the main branch through a short convolution, an activation, and the selective SSM, gates it with a parallel SiLU branch, and projects back down. Most of the parameters — $3ED^2$ per block ($2ED^2$ in the input projections, $ED^2$ in the output projection) — live in those linear projections; the SSM itself contributes few parameters. Two stacked Mamba blocks are sized to match the $12D^2$ parameters of a Transformer's interleaved attention + MLP, which is why "two Mamba layers ≈ one Transformer layer" in the comparisons. The activation is SiLU/Swish (so the gate branch becomes a SwiGLU-style gated MLP), with an optional LayerNorm.

The upshot: a language-model backbone with **no attention and no separate MLP block** — just one selective-SSM block, stacked. That homogeneity is not just elegant; it's what makes the model easy to scale and reason about.

## From math to code: reading the reference implementation

The math is clean but leaves questions a reader always has: how is $A$ actually stored? Where does the softplus live? Is the full ZOH formula for $\bar{B}$ really computed? The [official implementation](https://github.com/state-spaces/mamba) answers all of them, and reading it is the fastest way to remove the last of the hand-waving. Below, the load-bearing pieces (lightly trimmed from `mamba_simple.py` and `selective_scan_interface.py`), with the paper-to-code mapping.

### The block forward pass

The default hyperparameters pin down every shape: `d_state` $N = 16$, `d_conv` $= 4$, `expand` $E = 2$, so `d_inner` $= 2\cdot$`d_model`, and `dt_rank` $= \lceil$`d_model`$/16\rceil$. The forward pass is exactly the block we described:

```python
# in_proj: d_model -> 2 * d_inner, then split into the SSM branch x and the gate z
xz = self.in_proj(hidden_states)              # (B, L, 2*d_inner)
x, z = xz.chunk(2, dim=1)                      # each (B, d_inner, L)

# short depthwise CAUSAL conv (kernel 4, one filter per channel) + SiLU — inherited from H3
x = self.act(self.conv1d(x)[..., :seqlen])     # (B, d_inner, L)

# x_proj makes B, C, Δ INPUT-DEPENDENT: this is the selection mechanism s_B, s_C, s_Δ
x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))       # (B*L, dt_rank + 2N)
dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
dt = self.dt_proj.weight @ dt.t()              # low-rank Δ: dt_rank -> d_inner

A = -torch.exp(self.A_log.float())             # (d_inner, N): forced negative, S4D-real
y = selective_scan_fn(x, dt, A, B, C, self.D.float(),
                      z=z, delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
out = self.out_proj(rearrange(y, "b d l -> b l d"))
```

Every piece of the method is visible here. `in_proj` produces both the main branch `x` and the gate `z` (the SwiGLU arm). `x_proj` is the selection mechanism: it reads the (conv-mixed) token and emits the per-token `dt`, `B`, `C` with split sizes $[\text{dt\_rank}, N, N]$ — that's $s_\Delta, s_B, s_C$. `dt_proj` is the low-rank $\Delta$ projection. And notice the softplus and the $\Delta$ bias are **not** applied here — they're passed into the scan as `delta_softplus=True` and `delta_bias`, so $\Delta = \text{softplus}(\text{dt\_proj}(x) + \text{bias})$, with the bias initialized (via inverse-softplus) so the initial $\Delta$ lands log-uniformly in $[0.001, 0.1]$.

### How A is parameterized

One detail the equations gloss over: $A$ is stored in **log space** and forced negative.

```python
# init: S4D-real  ->  A = -diag(1, 2, ..., N), broadcast across all d_inner channels
A = repeat(torch.arange(1, d_state + 1), "n -> d n", d=d_inner).contiguous()
self.A_log = nn.Parameter(torch.log(A))
...
A = -torch.exp(self.A_log.float())             # reconstruct: strictly negative, (d_inner, N)
```

Storing $\log|A|$ and taking $A = -\exp(A_{\log})$ guarantees every eigenvalue is negative — so the continuous system is stable (leaky tanks, never exploding) and the discrete decay $\bar{A} = \exp(\Delta A)$ is guaranteed to live in $(0, 1)$. The default S4D-real initialization is just $A = -\text{diag}(1, 2, \dots, N)$, identical across channels. It's a per-channel *diagonal* SSM, so $A$ is stored as a length-$N$ vector per channel, not an $N \times N$ matrix.

### The recurrence, spelled out

The pure-PyTorch reference `selective_scan_ref` is the readable mirror of the CUDA kernel, and it makes the discretization + recurrence completely explicit:

```python
delta = F.softplus(delta + delta_bias)                 # Δ = softplus(dt + bias)   (B, d_inner, L)
deltaA  = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))       # A_bar = exp(Δ·A)   (B, d_inner, N, L)
deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)          # B_bar·u ≈ (Δ·B)·u  (B, d_inner, N, L)

x = A.new_zeros((batch, dim, dstate))                  # hidden state h : (B, d_inner, N)
ys = []
for i in range(u.shape[2]):                            # sequential scan over L
    x = deltaA[:, :, :, i] * x + deltaB_u[:, :, :, i]  # h_t = A_bar_t ⊙ h_{t-1} + B_bar_t ⊙ x_t
    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])      # y_t = C_t · h_t   (contract state N)
    ys.append(y)
y = torch.stack(ys, dim=2)                             # (B, d_inner, L)
out = y + u * rearrange(D, "d -> d 1")                 # D "skip"/residual: y + D⊙x
out = out * F.silu(z)                                  # multiplicative SiLU gate
```

Two things worth flagging for anyone matching this to the equations. First, `deltaA = exp(Δ·A)` is the **exact ZOH** discretization of $A$ — but `deltaB_u = Δ·B·u` is *not* the full ZOH $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I)\cdot \Delta B$. It's the simpler first-order (Euler) approximation $\bar{B} \approx \Delta B$, folded together with the input $u$ in one einsum so $\bar{B}$ is never materialized alone. The paper is upfront that this is fine — the $A$ term dominates the dynamics. Second, the `D` "skip" ($y + D \odot x$, a learned per-channel residual) and the SiLU gate are applied *inside* the scan, not after. These two — the Euler-$\bar{B}$ and the fused skip/gate — are exactly the kinds of details that differ between the folklore version of a method and what actually ships.

### Constant-time inference, no KV cache

The payoff claim — 5× throughput, no growing cache — is a direct consequence of the decode step. Generating one token carries forward two *fixed-size* tensors and touches nothing that grows with position:

```python
def step(self, hidden_states, conv_state, ssm_state):
    x, z = self.in_proj(hidden_states.squeeze(1)).chunk(2, dim=-1)   # (B, d_inner)
    # roll the length-4 conv window: drop oldest, insert newest
    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
    conv_state[:, :, -1] = x
    x = self.act(torch.sum(conv_state * self.conv1d_weight, dim=-1) + self.conv1d_bias)
    dt, B, C = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
    dt = F.softplus(F.linear(dt, self.dt_proj.weight) + self.dt_proj.bias)
    dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))                # (B, d_inner, N)
    dB = torch.einsum("bd,bn->bdn", dt, B)
    ssm_state.copy_(ssm_state * dA + x.unsqueeze(-1) * dB)           # in-place state update
    y = torch.einsum("bdn,bn->bd", ssm_state, C) + self.D * x        # read out + skip
    y = y * self.act(z)                                             # gate
    return self.out_proj(y).unsqueeze(1), conv_state, ssm_state
```

The entire history is summarized by `ssm_state` of shape $(B, d\_inner, N)$ and `conv_state` of shape $(B, d\_inner, 4)$. **Both are independent of sequence length** — unlike an attention KV cache, which grows with every token. The per-step cost is constant, which is the concrete basis for Mamba's throughput advantage: without a cache to hold, it can run much larger batches on the same GPU.

## Experiments and results

### Selective Copying and Induction Heads

First, do the synthetic tasks vindicate the design? Yes, cleanly. On **Selective Copying**, plain S4 gets 18.3% while the selective S6 layer jumps to 97.0% — and 99.8% inside the full Mamba block. Architecture gating alone (H3, Hyena) only partially helps; the selection mechanism is what solves it.

The **Induction Heads** result is the paper's most striking. Models are trained at sequence length 256 and tested far beyond.

![Figure from Gu and Dao (2023): induction-heads accuracy vs test sequence length. Every attention and prior-SSM baseline collapses to chance shortly past the training length of 256, while Mamba stays at 100% accuracy all the way to length 1,048,576.](/imgs/blogs/mamba-selective-state-spaces-fig5.webp)

Mamba solves the task perfectly at **length 1,048,576 — 4000× longer than it saw in training** — while multi-head attention (with absolute, rotary, or xPos encodings), H3, and Hyena all fall to chance shortly past $2\times$ the training length. A single selective SSM layer learned to remember the relevant token and ignore everything in between, and that ability extrapolates essentially without bound.

### Language modeling scaling laws

On the Pile, under the Chinchilla compute-optimal protocol from 125M to 1.3B parameters, Mamba is the first attention-free model to match a strong modern Transformer recipe.

![Figure 4 from Gu and Dao (2023): scaling laws on the Pile at sequence lengths 2048 and 8192. Mamba (purple) tracks or beats Transformer++ (the LLaMA-style recipe with rotary embeddings, SwiGLU, RMSNorm) and clearly beats Hyena, RWKV, RetNet, and vanilla Transformer, with the gap widening at the longer sequence length.](/imgs/blogs/mamba-selective-state-spaces-fig4.webp)

"Transformer++" here is the strong baseline — the LLaMA/PaLM recipe with [rotary embeddings](/blog/paper-reading/large-language-model/roformer-enhanced-transformer-with-rotary-position-embedding), SwiGLU MLPs, and RMSNorm. Mamba matches it and pulls ahead as the sequence length grows to 8K.

### Downstream zero-shot evaluations

The scaling curves translate into downstream wins. Trained on 300B tokens with the same tokenizer and data as Pythia and [RWKV](/blog/paper-reading/large-language-model/kimi-linear), Mamba is best-in-class at every size on common-sense reasoning:

| Model | Pile ppl ↓ | LAMBADA acc ↑ | HellaSwag ↑ | PIQA ↑ | Arc-E ↑ | WinoGrande ↑ | Average ↑ |
|---|---|---|---|---|---|---|---|
| Pythia-1.4B | 7.51 | 61.7 | 52.1 | 71.0 | 60.5 | 57.2 | 55.2 |
| RWKV-1.5B | 7.70 | 56.4 | 52.5 | 72.4 | 60.5 | 54.6 | 54.3 |
| **Mamba-1.4B** | **6.80** | **64.9** | **59.1** | **74.2** | **65.5** | **61.5** | **59.7** |
| Pythia-2.8B | 6.73 | 64.7 | 59.3 | 74.0 | 64.1 | 59.7 | 59.1 |
| RWKV-3B | 7.00 | 63.9 | 59.6 | 73.7 | 67.8 | 59.6 | 59.6 |
| **Mamba-2.8B** | **6.22** | **69.2** | **66.1** | **75.2** | **69.7** | **63.5** | **63.3** |

Mamba-2.8B's 63.3 average not only beats the same-size Pythia-2.8B (59.1) by 4+ points but **exceeds Pythia-6.9B** (61.7) — a model 2.5× its size. That's the "matches Transformers twice its size" headline, made concrete.

### DNA and audio: where selection helps and where it hurts

On **DNA** (the HG38 human genome), Mamba scales better than HyenaDNA and Transformer++, matching them with **3–4× fewer parameters**, and its perplexity *improves* as context grows to 1M base pairs — while HyenaDNA gets *worse* with longer context. This is the "filtering context" property from the selection analysis: an LTI model's very long convolution kernel aggregates all the noise in a long sequence; a selective model can ignore it.

**Audio** is the honest counter-example, and the paper doesn't hide it. On the SC09 speech-generation benchmark, a 6.1M-parameter Mamba beats much larger GAN- and diffusion-based models (FID 0.94, halving the prior best), and a 24.3M version improves further (FID 0.67). But an ablation shows that on *long-form raw audio waveforms*, swapping S4 for the selective S6 **significantly hurts** — because audio is a smooth, uniformly-sampled continuous signal that benefits from the time-invariant inductive bias selection discards. This is the paper's "no free lunch, continuous–discrete spectrum" caveat: selection is exactly right for discrete, information-dense data (text, DNA) and exactly wrong for the smoothest continuous modalities.

### Efficiency and ablations

The efficiency numbers back the architecture claims: the selective scan beats FlashAttention-2 beyond ~2K tokens and is 20–40× faster than a naive PyTorch scan; end-to-end, Mamba has **5× the generation throughput** of a similarly-sized Transformer because, without a KV cache, it can use much larger batch sizes; and its training memory is comparable to a FlashAttention-2 Transformer.

Two ablations carry the argument. **$\Delta$ is the most important selective parameter** (consistent with Theorem 1's gate connection), though selective $B$ and $C$ synergize with it. And increasing the state size $N$ from 1 to 16 buys **over 1.0 perplexity for only ~1% more parameters — but only when $B$ and $C$ are also selective.** That single result validates the whole thesis: a bigger state helps only if the model is smart about what goes into it.

### The synthesis: three model families, one table

Everything above collapses into a single comparison. Attention is content-aware but quadratic with a growing cache; LTI SSMs are cheap but content-blind; Mamba is the first row that is content-aware *and* constant-time *and* fixed-memory.

![A redrawn tradeoff matrix: Attention (content selection yes, train O(L²), inference O(L) with a growing KV cache), LTI SSM (no content selection, O(L log L) train, O(1) inference, fixed state), and Selective SSM / Mamba (content selection yes, O(L) scan training, O(1) inference, fixed state) — Mamba's row is all wins.](/imgs/blogs/mamba-selective-state-spaces-4.webp)

## Critique

**What's genuinely strong.** The paper is a rare case of a clean conceptual insight (selection = content-dependent compression) *and* the systems work to make it real (the fused scan). The synthetic-task motivation is honest and predictive — the tasks were chosen to isolate the failure mode, and the model that fixes them also wins downstream, which is the right scientific arc. The induction-heads extrapolation to 1M is a genuinely surprising, hard-to-fake result. And the ablation isolating "state size $N$ only helps when $B, C$ are selective" is exactly the kind of load-bearing evidence that turns a plausible story into a demonstrated one.

**What's weak or unshown.** The scaling stops at 3B — well below the 7B+ where Pythia, RWKV, and RetNet were evaluated and where surprises tend to appear. The paper is candid that "scaling SSMs may involve further engineering challenges … not discussed in this paper," which is a real caveat, not a formality. The Euler approximation to $\bar{B}$ is used everywhere but its impact is never ablated against the exact ZOH form. And the whole "downstream affordances" question — can you fine-tune, quantize, instruction-tune, and RLHF a Mamba the way you can a Transformer? — is raised and left open.

**What ablation is missing.** I'd want a controlled study of the pre-SSM convolution: it's inherited from H3 and is what `x_proj` reads to make $B, C, \Delta$ input-dependent, yet its contribution is never isolated. I'd also want the exact-ZOH-vs-Euler $\bar{B}$ comparison, and a head-to-head of "selective $A$" (which they hypothesize is redundant but never test).

**What would change my mind.** If a Mamba-7B trained on the same data as a strong 7B Transformer *failed* to match it on knowledge-heavy tasks (not just common-sense reasoning) — say MMLU or long-context retrieval where you need to pull an exact fact from far back — I'd downgrade the "general backbone" claim substantially. Selective compression is lossy by construction, and the tasks in this paper mostly reward *recency and pattern*, not verbatim recall of arbitrary distant facts. The KV cache exists precisely so attention can do that recall; whether a fixed state can is the open question the paper's scale doesn't settle. (History has partly answered this: pure-SSM models do lag on exact long-range retrieval, which is why most production "Mamba" systems today are *hybrids* that interleave a few attention layers.)

## What I'd build with this

These are my extrapolations, not the paper's claims:

1. **Hybrid attention–Mamba stacks, tuned for the retrieval gap.** Given the recall weakness above, interleave a small number of full-attention layers among many Mamba layers, and study *where* to place them (early vs. late) to recover exact long-range retrieval at minimal cost. The paper's own Mamba-MHA ablation hints this barely changes perplexity — but perplexity isn't retrieval, so measure retrieval directly.
2. **A selective $\Delta$ as a document-boundary detector.** Theorem 1 says $\Delta_t \to \infty$ is a state reset. Probe a trained model to see whether it *learned* to spike $\Delta$ at natural boundaries (sentence ends, document breaks), and if so, use that signal for cheap unsupervised segmentation.
3. **Adaptive state size.** The $N$-ablation shows state size trades off cheaply. Build a model that allocates a larger $N$ to layers/channels that carry long-range dependencies and a tiny $N$ elsewhere, learned or via a small search — most of the state budget is probably wasted on channels that don't need it.
4. **Selective scan as a drop-in memory for agents/RL.** The boundary-resetting property is tailor-made for episodic RL, where you want to wipe state at episode ends. A selective SSM that learns *when* to reset could replace hand-coded episode masking.
5. **Push the continuous–discrete dial explicitly.** Since selection helps discrete data and hurts smooth audio, build a layer that *learns* how selective to be per channel — a soft interpolation between LTI and S6 — so a single architecture adapts to mixed-modality inputs (e.g. interleaved text and audio tokens).

## References

- **Paper:** Albert Gu and Tri Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752 (2023, v2 2024). <https://arxiv.org/abs/2312.00752>
- **Code and pretrained checkpoints:** `state-spaces/mamba` — <https://github.com/state-spaces/mamba>
- **The direct follow-up:** [Mamba-2: state-space duality](/blog/paper-reading/large-language-model/mamba-2-state-space-duality) — recasts the selective SSM as a form of attention and makes it faster.
- **Modern linear-recurrent descendants:** [Kimi Linear](/blog/paper-reading/large-language-model/kimi-linear) and [Gated DeltaNet](/blog/paper-reading/large-language-model/gated-delta-networks) — the gated-linear-attention lineage that grew alongside Mamba.
- **The thing it replaces:** the [KV-cache management survey](/blog/paper-reading/large-language-model/a-survey-on-large-language-model-acceleration-based-on-kv-cache-management), for why constant-state inference matters, and [rotary embeddings](/blog/paper-reading/large-language-model/roformer-enhanced-transformer-with-rotary-position-embedding), the positional scheme in the Transformer++ baseline.

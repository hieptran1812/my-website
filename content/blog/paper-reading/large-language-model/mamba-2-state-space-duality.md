---
title: "Transformers are SSMs: The Structured State Space Duality Behind Mamba-2"
date: "2026-07-16"
description: "A detailed, intuition-first walkthrough of the SSD framework: how a state space model and a masked-attention layer turn out to be the same structured matrix, and how that equivalence makes Mamba-2 2-8x faster than Mamba while scaling like a Transformer."
tags: ["paper-reading", "mamba-2", "state-space-model", "ssd", "structured-state-space-duality", "linear-attention", "semiseparable-matrices", "sequence-models", "attention", "large-language-model"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 32
paper:
  title: "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
  authors: "Tri Dao, Albert Gu"
  venue: "ICML 2024 (arXiv:2405.21060)"
  url: "https://arxiv.org/abs/2405.21060"
---

> [!tldr]
> - **The claim.** A selective state space model (the core of [Mamba](/blog/paper-reading/large-language-model/gated-delta-networks)) and a masked-attention layer are *the same object* — both are multiplication by a structured matrix $M$ called a **semiseparable matrix**. The linear-time recurrence and the quadratic-time attention computation are two algorithms for multiplying by that one matrix. The authors name this equivalence **Structured State Space Duality (SSD)**.
> - **The mechanism in one line.** Restrict the SSM's state-transition $A_t$ to a *scalar times identity*, and its matrix form becomes $M = L \circ (CB^\top)$ — an attention matrix with the softmax dropped and a data-dependent mask $L$ added. That mask is a "1-semiseparable" matrix, i.e. a cumulative product of gates.
> - **Why it matters.** The duality hands SSMs the entire Transformer toolbox: a matmul-heavy (tensor-core-friendly) training algorithm, multi-head structure, tensor/sequence parallelism, and variable-length batching. Mamba-2's SSD layer is **2-8x faster** than Mamba-1's optimized scan and beats FlashAttention-2 past sequence length 2K.
> - **The most surprising result.** *Any* dense SSM with state size $N$ can be computed in $O(TN)$ time (Theorem 3.7), and any efficiently-autoregressive masked attention *must* use a semiseparable mask (Theorem 5.2). Efficient autoregression and semiseparability are the same constraint.
> - **Where it gives ground.** SSD is strictly *less* expressive than Mamba-1's diagonal SSM — it trades the full diagonal $A_t$ for a scalar. It buys hardware efficiency with a modeling concession, and pure SSD still trails a Transformer on hard recall until you bolt on a few attention layers.

Two years of sequence-modeling research split into two tribes that barely spoke to each other. On one side, **Transformers**: quadratic in sequence length, memory-hungry at inference, but blessed with a decade of systems engineering — FlashAttention, tensor parallelism, KV-cache tricks, the works. On the other, **state space models** (S4, then Mamba): linear in sequence length, constant-memory generation, strong on long-range tasks — but implemented as bespoke CUDA scans that don't touch a tensor core, and understood through a completely different vocabulary (continuous-time systems, HiPPO, discretization) that made them feel alien to anyone fluent in attention.

This paper's thesis is that the wall between the two tribes was an illusion. Underneath, they compute the same thing.

![Redrawn mental model: a selective SSM and a masked-attention layer are two computational routes to multiplying by one structured (semiseparable) matrix M, and they coincide exactly when A is scalar-identity.](/imgs/blogs/mamba-2-state-space-duality-1.webp)

The diagram above is the mental model for the whole paper. Read the two families down the left — an SSM defined by a recurrence, a masked-attention layer defined by pairwise scores. Both, it turns out, are "matrix mixers": they produce their output by multiplying the input sequence by a single $T \times T$ matrix $M$. That matrix has a specific structure (**semiseparable**), and when you push the SSM's state matrix all the way down to a scalar, the two matrices become *identical*. The rest of this post unpacks that claim from the ground up, then follows it to the payoff: a new algorithm and a new architecture, **Mamba-2**.

## The problem: two good models, no shared language

Let us be precise about the pain each side lives with, because SSD is best understood as the bridge that lets each side borrow the other's strengths.

**Attention's problem is its state.** Softmax self-attention computes $Y = \operatorname{softmax}(QK^\top) V$. Materializing the $T \times T$ score matrix $QK^\top$ costs $O(T^2)$ time and memory at training, and at generation time the model must cache every past key and value — a **KV cache that grows linearly with the sequence**. Attention is fast on hardware (it's almost entirely matrix multiplication, which GPUs and TPUs are built for), but it never compresses its history; its effective "state" *is* the entire past. At 100K tokens that state is enormous.

**SSMs have the opposite problem.** A state space model carries a fixed-size recurrent state $h_t \in \mathbb{R}^{N}$ and updates it one token at a time. Its state never grows with sequence length, so generation is constant-memory and training is $O(T)$. [Mamba](/blog/paper-reading/large-language-model/gated-delta-networks) showed that making the SSM's parameters *input-dependent* (a "selective" SSM) closes most of the language-modeling gap to Transformers. But the selective SSM can only be computed as a sequential scan — no convolution shortcut, no matmul — so despite its better asymptotics it is often *slower in wall-clock* than attention, because it leaves the tensor cores idle. And because SSMs were developed in a continuous-time / signal-processing frame, none of the Transformer systems work (tensor parallelism, sequence parallelism, variable-length packing) transferred over.

So we have two models with complementary weaknesses, described in two incompatible languages. The obvious question — *can we give SSMs attention's hardware efficiency, and give attention SSMs' compressed state?* — was hard to even ask, because nobody had a formalism in which both models lived. Building that formalism is the paper's first and largest contribution. The payoff, later, is a model that is Pareto-better than both on the axes that matter.

This is the same "linear attention" lineage that runs through [Lightning Attention](/blog/paper-reading/large-language-model/minimax-01-lightning-attention-hybrid-moe), [Gated DeltaNet](/blog/paper-reading/large-language-model/gated-delta-networks), and [Kimi Linear](/blog/paper-reading/large-language-model/kimi-linear); SSD is the framework that ties the knot between all of them and classical SSMs. If you want the broader map first, the [efficient-attention survey](/blog/paper-reading/large-language-model/efficient-attention-mechanisms-for-large-language-models-a-survey) is a good companion.

## Contributions

Tightened into a numbered list, and in my words:

1. **SSMs are semiseparable matrices (§3).** Writing the SSM recurrence as a single matrix multiplication $y = Mx$ reveals that $M$ belongs to a classical, well-studied family — *semiseparable matrices*. Every algorithm for computing an SSM is a structured-matrix-multiplication algorithm in disguise.
2. **Structured Masked Attention generalizes linear attention (§4).** A clean tensor-contraction proof of linear attention shows that the causal mask can be *any* structured matrix $L$, not just the lower-triangular ones. This family is **SMA**.
3. **The duality itself (§5).** A scalar-identity SSM and a 1-semiseparable SMA are exactly the same function, with the SSM's linear recurrence and attention's quadratic form as dual algorithms. Moreover: efficient autoregressive attention *must* be semiseparable.
4. **A hardware-efficient SSD algorithm (§6).** A block decomposition of $M$ computes diagonal blocks with the quadratic (matmul) form and off-diagonal blocks with the linear (recurrent) form — getting linear scaling *and* tensor-core utilization at once. It's a few lines of PyTorch.
5. **The Mamba-2 architecture (§7-8).** Parallel parameter projections, an extra normalization, a multi-value-attention head structure, and Transformer-style tensor/sequence parallelism, all justified by the duality.

## The method

I'll climb each load-bearing idea from intuition to math. The order matters: matrices first (§ SSMs as matrices), then the two ways to multiply by them (§ dual forms), then the attention side of the bridge (§ SMA), then the meeting point (§ the duality), then the algorithm and the architecture. Every symbol is defined the first time it appears.

Throughout, an SSM is the map defined by a linear recurrence that is *linear in the input* $x$:

$$
h_t = A_t\, h_{t-1} + B_t\, x_t, \qquad y_t = C_t^\top h_t .
$$

Here $x_t \in \mathbb{R}$ is the input at time $t$; $h_t \in \mathbb{R}^{N}$ is the hidden state ($N$ is the **state size** or state-expansion factor); $A_t \in \mathbb{R}^{N\times N}$ is the state-transition matrix; $B_t \in \mathbb{R}^{N}$ maps the scalar input into the state; and $C_t \in \mathbb{R}^{N}$ reads the state back out into the scalar output $y_t$. When $(A,B,C)$ vary with $t$ (as functions of the input), this is Mamba's **selective SSM**. When they're constant it's a classical time-invariant SSM. The whole layer operates on a scalar sequence; to handle a $P$-dimensional input you just run $P$ independent copies (broadcast over a "head dimension" $P$) — hold that thought, it becomes the multi-head story later.

### SSMs are structured matrices

**The problem it solves.** SSMs were described as recurrences and as convolutions, but never as *matrices along the sequence axis* — which is exactly how attention is described. Without a matrix view, there's no common ground to compare them, and no way to reuse the vast literature on fast structured-matrix multiplication. This subsection builds that view.

**Intuition.** Unroll the recurrence. Because it's linear, $h_t$ is just a weighted sum of all past inputs, where the weight on input $x_s$ is "how much of $s$ survives the walk from time $s$ up to time $t$." That surviving fraction is a product of the intervening transition matrices. Multiply by the readout $C_t$ and you get $y_t$ as a weighted sum over all $s \leq t$ — which is precisely a row of a matrix-vector product $y = Mx$. The matrix $M$ is lower-triangular (the future can't affect the past) and its entries are these "walk products." The analogy that clicks: $M_{ji}$ is the *gain* of a signal injected at time $i$ and measured at time $j$, and the SSM is the linear system whose gain matrix is $M$.

**The mechanism, step by step.** Start from $h_0 = B_0 x_0$ and induct. At step $t$, the contribution of input $x_s$ (for $s \le t$) to $h_t$ has been multiplied by $B_s$ on the way in and by $A_t A_{t-1}\cdots A_{s+1}$ on the way forward. Read out with $C_t$ and sum over $s$.

**The math.** Define the shorthand $A_{j:i}^{\times} \equiv A_j A_{j-1}\cdots A_{i+1}$ for the product of transition matrices strictly between times $i$ and $j$ (empty product $=$ identity when $j=i$). Then

$$
y_t = \sum_{s=0}^{t} C_t^\top\, A_{t:s}^{\times}\, B_s\, x_s
\quad\Longrightarrow\quad
y = M x, \qquad
M_{ji} = \begin{cases} C_j^\top\, A_{j:i}^{\times}\, B_i & j \geq i \\[2pt] 0 & j \lt i .\end{cases}
$$

Every symbol: $M \in \mathbb{R}^{T\times T}$ is the sequence-transformation matrix; $M_{ji}$ is the entry in row $j$ (output time), column $i$ (input time); the product $A_{j:i}^{\times}$ is $(N\times N)$; $C_j^\top$ is $(1\times N)$; $B_i$ is $(N\times 1)$, so each $M_{ji}$ is a scalar. This representation — a lower-triangular matrix whose entries factor as $C_j^\top A_{j:i}^{\times} B_i$ — is called the **sequentially semiseparable (SSS)** form.

Now the key structural fact. A lower-triangular matrix is **$N$-semiseparable** if *every* submatrix drawn entirely from the lower triangle (on or below the diagonal) has rank at most $N$. The SSS matrix above is $N$-semiseparable, and the reason is the whole ballgame: take any off-diagonal block $M_{j:j', \, i':i}$ with $j' > j \geq i > i'$. Every entry in it shares the same "middle" — it factors as (something depending only on the row) $\times A_{\text{center}} \times$ (something depending only on the column). Concretely the block is an outer product of a column vector of $C$-terms, the center transition, and a row vector of $B$-terms — a **rank-$N$ factorization**. Off-diagonal blocks are low-rank; that is what "semiseparable" means.

![Figure 2 from Dao and Gu (2024): a state space model, viewed as a sequence transformation, is a matrix M acting on the sequence dimension T, and that matrix is semiseparable -- every submatrix on or below the diagonal (highlighted) has rank at most N, the state size.](/imgs/blogs/mamba-2-state-space-duality-fig6.webp)

**Worked micro-example.** Take $N=1$ (scalar state), so each $A_t = a_t$, $B_t = C_t = 1$. Then $M_{ji} = a_j a_{j-1}\cdots a_{i+1}$ — a **cumulative product**. For $T=4$:

$$
M = \begin{pmatrix}
1 & & & \\
a_1 & 1 & & \\
a_2 a_1 & a_2 & 1 & \\
a_3 a_2 a_1 & a_3 a_2 & a_3 & 1
\end{pmatrix}.
$$

Multiplying $y = Mx$ by hand gives $y_2 = a_2 a_1 x_0 + a_2 x_1 + x_2$, which you can factor as $y_2 = a_2(a_1 x_0 + x_1) + x_2 = a_2 y_1 + x_2$. The matrix-vector product *is* the recurrence $y_t = a_t y_{t-1} + x_t$. This $N=1$ case — call it a **1-semiseparable (1-SS)** matrix — is the atom everything else is built from. It's a gated cumulative sum; the paper calls it the "cumprodsum" (cumulative-product-sum) operator.

**Why it works / when it fails.** The payoff of the matrix view is a hard efficiency theorem: an $N$-semiseparable matrix of size $T$ needs only $O(NT)$ parameters and supports matrix-vector multiplication in $O(NT)$ time (Proposition 3.6). Feeding that back through the equivalence gives **Theorem 3.7**: *any* SSM of state size $N$ can be computed in $O(TN)$ time — even one with dense, unstructured $A_t$ matrices, whose naive representation alone looks like $O(TN^2)$. The catch is the fine print "not accounting for preprocessing": realizing that optimum for a general dense $A_t$ needs a per-step diagonalization (SVD-like work) that is hardware-hostile. That is exactly why practical SSMs impose *more* structure on $A$ — diagonal (S4D, Mamba) or, as we'll see, scalar (SSD) — to skip the preprocessing.

### Two ways to multiply: the dual forms

**The problem it solves.** We now have one matrix $M$ but two very different reasons to compute $y = Mx$: at training time we see the whole sequence and want to saturate the tensor cores; at inference we see one token at a time and want a tiny state. A single implementation can't be best at both. The semiseparable structure lets us pick per-situation.

**Intuition.** A semiseparable matrix can be multiplied *either* by exploiting its structure (walk the recurrence, $O(T)$, sequential) *or* by ignoring the structure and just materializing the whole $T\times T$ matrix and doing a dense multiply ($O(T^2)$, but pure matmul). Same answer, opposite tradeoffs. It's the difference between summing a running total as numbers arrive versus writing out the full addition table and adding it all at once.

![Redrawn: the linear recurrent form and the quadratic matrix form multiply by the identical M with opposite hardware tradeoffs -- one is cheap on memory but sequential, the other memory-heavy but pure matmul.](/imgs/blogs/mamba-2-state-space-duality-2.webp)

**The mechanism.** The **linear (recurrent) mode** unrolls the state: expand the input into the state via $B$, run the scalar recurrences forward, contract back out via $C$. In tensor-contraction notation, with $L = \mathsf{1SS}(A)$ the $T\times T$ matrix of cumulative products:

$$
\begin{aligned}
Z &= \operatorname{contract}(\mathtt{SP},\mathtt{SN}\to\mathtt{SPN})(X, B) & \text{(expand by }B\text{)}\\
H &= \operatorname{contract}(\mathtt{TSN},\mathtt{SPN}\to\mathtt{TPN})(L, Z) & \text{(scan)}\\
Y &= \operatorname{contract}(\mathtt{TN},\mathtt{TPN}\to\mathtt{TP})(C, H) & \text{(contract by }C\text{)}
\end{aligned}
$$

where $\mathtt{T},\mathtt{S}$ index sequence positions ($\mathtt{S}=\mathtt{T}$, two names so the proof is unambiguous), $\mathtt{N}$ is the state size, $\mathtt{P}$ the head dimension. The bottleneck is the middle line — a scan — which is $O(T)$ but sequential and matmul-free. The **quadratic (naive) mode** just builds $M$ and multiplies: $O(T^2)$, but every step is a dense matmul. When $T$ is short, the constant factors and tensor-core throughput make the "worse" quadratic mode actually *faster*.

**Worked micro-example.** With the $T=4$, $N=1$ matrix above: the linear mode computes $y_0,\dots,y_3$ with 3 multiply-adds ($y_t = a_t y_{t-1}+x_t$), touching each $x$ once. The quadratic mode writes out all 10 nonzero entries of $M$ and does a $4\times 4$ matrix-vector product — 10 multiply-adds, but as one batched operation. At $T=4$ the second is trivially parallel; at $T = 10^5$ it's $10^{10}$ entries and hopeless. That crossover is the entire performance story.

**Why it works / when it fails.** The two modes are correct for the same reason — they compute $y=Mx$ — and each is optimal in its regime. Neither is optimal *everywhere*: linear mode wastes the hardware, quadratic mode wastes memory. The §algorithm section fuses them. The failure mode to remember: the quadratic form only stays "attention-like" (and thus matmul-friendly) under extra structure on $A$; for a general diagonal $A_t$ the quadratic form exists but is no longer a clean attention matrix, which is precisely the expressivity SSD gives up.

### Structured masked attention

We now walk to the same duality from the *other* side — attention — because arriving from both directions is what proves the two are one. This section can be read independently of the SSM story.

**The problem it solves.** Linear attention (Katharopoulos et al., 2020) famously rewrites $(QK^\top)V = Q(K^\top V)$ to drop from quadratic to linear cost — but the moment you add a causal mask, the derivation gets hand-wavy, and papers tend to state the recurrent form without proof. A clean proof would also tell us *exactly* which masks preserve the linear-time trick. That's what SMA delivers.

**Intuition.** Masked attention, stripped of softmax, is a computation on four tensors: queries $Q$, keys $K$, values $V$, and a mask $L$. "Do the multiplications in one order" gives you the quadratic algorithm; "do them in another order" gives you the linear algorithm. Same tensors, same result — the linear-vs-quadratic split is nothing but a choice of *contraction order*. And the only thing that makes the linear order cheap is that multiplying by the mask $L$ is cheap. So: **any** mask with fast matrix multiplication yields a fast linear attention.

**The mechanism.** Write masked attention as a single four-way tensor contraction:

$$
Y = \operatorname{contract}(\mathtt{TN},\mathtt{SN},\mathtt{SP},\mathtt{TS}\to\mathtt{TP})(Q, K, V, L).
$$

Reduce it in the "quadratic" order — form $G = QK^\top$ (a $T\times S$ Gram matrix), mask it $M = G \circ L$, then $Y = MV$ — and you recover standard masked attention, cost $O(T^2)$. Reduce it in the "linear" order — first form $Z = V K^\top$-style expansion, then multiply by $L$, then contract with $Q$ — and the only term touching both sequence axes is the multiply-by-$L$ step. For the causal mask (lower-triangular ones), multiplying by $L$ *is* a cumulative sum, which is $O(T)$. That's linear attention, now with a proof.

**The math — the generalization.** Here's the leap. Nothing forced $L$ to be all-ones. Define **Structured Masked Attention (SMA)** as the four-way contraction above for *any* structured matrix $L$ (any $L$ with sub-quadratic matrix multiplication). The mask becomes a design knob:

$$
L_{ij} = \mathbb{1}[i\ge j] \ \text{(causal)}, \quad
L_{ij} = \gamma^{\,i-j}\,\mathbb{1}[i\ge j] \ \text{(decay, RetNet)}, \quad
L_{ij} = \alpha_{i-j} \ \text{(Toeplitz)}, \quad \dots
$$

Causal linear attention, RetNet's exponential decay, Toeplitz relative positions, even a Fourier mask — all are SMA with different $L$. The linear-time algorithm exists for all of them, inherited from fast multiplication by $L$.

![Figure 3 from Dao and Gu (2024): structured masked attention masks the score matrix QK-transpose by any structured L, and each choice of L -- causal, decay, 1-semiseparable, Toeplitz, discrete Fourier -- yields a different linear-time attention variant; the 1-semiseparable row is SSD.](/imgs/blogs/mamba-2-state-space-duality-fig7.webp)

**Worked micro-example.** Causal mask, $T=3$, single feature. The linear order computes a running key-value sum $S_t = \sum_{s\le t} K_s V_s^\top$ and reads it out as $Y_t = Q_t S_t$. Concretely $S_0 = K_0V_0^\top$, $S_1 = S_0 + K_1V_1^\top$, $S_2 = S_1 + K_2 V_2^\top$ — the multiply-by-$L$ step is literally the accumulation $S_t = S_{t-1} + (\cdot)$. Swap the causal mask for the decay mask $L_{ij}=\gamma^{i-j}$ and the accumulation becomes $S_t = \gamma S_{t-1} + K_t V_t^\top$ — you've derived RetNet's recurrence by changing one matrix.

**Why it works / when it fails.** SMA's power is that it decouples the *function* (a four-way contraction) from the *algorithm* (a contraction order). It fails as an efficiency story exactly when $L$ has no fast multiply — a dense unstructured mask sends you back to $O(T^2)$. The interesting masks are the structured ones, and the most interesting structured mask of all is the 1-semiseparable one, because — as the next section shows — that's where attention becomes an SSM.

### The duality itself

**The problem it solves.** We have SSMs producing semiseparable matrices, and SMA producing masked-attention matrices. Are these the same matrices? Not in general — but there's a rich intersection, and pinning it down is what lets each side borrow the other's algorithms.

**Intuition.** Collapse the SSM's state matrix to a scalar. Then the "walk product" $A_{j:i}^\times$ becomes a plain scalar $a_{j:i} = a_j a_{j-1}\cdots a_{i+1}$, and it factors cleanly out of $C_j^\top B_i$. What's left is an attention matrix ($C_j^\top B_i$ plays the role of $Q_jK_i^\top$) times a mask (the scalar cumulative products). SSM = masked attention, with $C,B$ as query,key and the gates as the mask.

**The math.** Let every $A_t = a_t I$ (scalar times identity). Then $M_{ji} = a_{j:i}\,(C_j^\top B_i)$, which vectorizes to

$$
M = L \circ (C B^\top), \qquad L = \mathsf{1SS}(a), \quad L_{ij} = a_i\, a_{i-1}\cdots a_{j+1}\ \ (i\ge j),
$$

with $B, C \in \mathbb{R}^{T\times N}$ and $\circ$ the elementwise (Hadamard) product. Compare to masked attention $M = L \circ (QK^\top)$: they are the *same formula*, with $(C, B)$ standing in for $(Q, K)$ and $L$ a **1-semiseparable** mask (a matrix of gated cumulative products) instead of a plain causal mask. Computing $Y = MX$ naively is quadratic masked kernel attention; computing it by the recurrence is a diagonal SSM. **Two algorithms, one function** — that is the state space duality.

Two theorems make the intersection exact:

- **Corollary 5.1.** 1-SS SMA (masked attention whose mask is 1-semiseparable) is a special case of a diagonal SSM — specifically the case where the diagonal of $A$ is a single scalar repeated. So attention-with-a-gated-mask is literally an SSM.
- **Theorem 5.2.** Conversely, *any* structured masked attention that admits an efficient (bounded-order) autoregressive form must have a semiseparable mask $L$. Efficient autoregression forces semiseparability.

Put together: **the set of efficiently-autoregressive masked attentions is the set of semiseparable-mask attentions, and its scalar-state core is exactly the SSD layer.** The two research tribes were exploring the same object from opposite ends.

**Worked micro-example.** Set all $a_t = 1$. Then $L$ is the all-ones lower-triangular matrix — the plain causal mask — and $M = L \circ (CB^\top)$ is ordinary causal linear attention with query $C$, key $B$. Now let $a_t \in (0,1)$ be input-dependent: $L_{ij} = \prod_{k=j+1}^{i} a_k$ becomes a *data-dependent* decay that the model learns per token. That single change — a constant mask becoming a learned gated mask — is what separates plain linear attention from the SSD layer, and it's why SSD can *forget* selectively while linear attention cannot.

**Why it works / when it fails.** The duality is powerful precisely because it's a *special case* on both sides — SSD is a scalar-$A$ SSM (less than the full diagonal $A$ that Mamba-1 uses) and a 1-SS SMA (less than general semiseparable SMA). That narrowness is the point: it's the widest overlap where the linear form is a clean SSM *and* the quadratic form is a clean attention matrix. Step outside — to a full diagonal $A_t$ — and the quadratic form stops looking like attention and loses matmul-friendliness. SSD deliberately sits at the sweet spot, trading a sliver of expressivity for the ability to run as matmuls.

### The hardware-efficient SSD algorithm

**The problem it solves.** The pure linear mode is $O(T)$ but sequential (no tensor cores); the pure quadratic mode is matmul-heavy but $O(T^2)$. We want both good properties: linear scaling *and* matmul utilization. The block algorithm delivers exactly that, and it's the reason Mamba-2 trains fast.

**Intuition.** Chop the sequence into chunks of length $Q$. Think of the $T\times T$ matrix $M$ tiled into a grid of $Q\times Q$ blocks. **Diagonal blocks** are self-contained mini-SSMs over a single chunk — small enough that the quadratic (attention) mode is the fast way to do them, and they're all independent so they run in parallel. **Off-diagonal blocks** encode how earlier chunks influence later ones; by semiseparability they're low-rank, so instead of computing them densely we route their effect through a single object — the SSM's hidden state at each chunk boundary. Earlier chunks talk to later chunks *only* through that boundary state, exactly like passing a baton in a relay.

![Figure 5 from Dao and Gu (2024): the SSD block-decomposition algorithm. Diagonal blocks are intra-chunk computations done in the quadratic (attention) mode; off-diagonal blocks are inter-chunk computations factored through the SSM hidden state, colored by the input-to-state, state-to-state, and state-to-output steps.](/imgs/blogs/mamba-2-state-space-duality-fig2.webp)

**The mechanism, in four steps.** For each chunk $j$, the output splits into "what happened inside this chunk" plus "what the carried-in state contributes":

1. **Diagonal (intra-chunk) output.** Compute each chunk's output *assuming zero initial state*, using the quadratic attention form on that $Q\times Q$ block. Pure matmul, all chunks in parallel.
2. **Chunk end-states.** For each chunk, compute the hidden state it *would* produce from a zero start: a $(N,Q)\times(Q,P)$ matmul yielding an $(N,P)$ state per chunk.
3. **Inter-chunk recurrence.** Run a scan over the $T/Q$ chunk-boundary states to get the *true* incoming state for each chunk. This is a 1-SS multiplication — the only sequential part — but on $T/Q$ items, not $T$. It's $Q$ times shorter than a full scan and its cost is negligible.
4. **State-to-output.** For each chunk, apply its correct carried-in state to produce the "influence of the past" term: a $(Q,N)\times(N,P)$ matmul. Add to step 1's output.

![Redrawn: the chunkwise SSD recipe as a pipeline. Only step 3 (the inter-chunk 1-SS scan over T/Q boundary states) is sequential; steps 1, 2, and 4 are batched matmuls.](/imgs/blogs/mamba-2-state-space-duality-3.webp)

**The math — cost accounting.** Set $N = P = Q$ (state size, head dimension, chunk length all equal). Every matmul above becomes a batched $\operatorname{BMM}(T/N, N, N, N)$, and the sequential scan is over $T/Q$ steps on $N^2$ features. Totals:

$$
\text{FLOPs} = O(TN^2), \qquad \text{memory} = O(TN), \qquad \text{work} = \text{matmuls on } (N\times N) \text{ tiles}.
$$

Both bounds are tight: the memory matches the $(T, P)=(T,N)$ input/output size, and the extra factor of $N$ in FLOPs is the price of the recurrent state, common to every model in this family. Line the three models up:

| | Attention | SSM (naive) | **SSD** |
|---|---|---|---|
| State size | $T$ | $N$ | $N$ |
| Training FLOPs | $T^2 N$ | $TN^2$ | $TN^2$ |
| Inference FLOPs / step | $TN$ | $N^2$ | $N^2$ |
| Naive memory | $T^2$ | $TN^2$ | $TN$ |
| Uses matmul units | yes | no | **yes** |

SSD matches the SSM's linear FLOPs and constant-size state, matches attention's matmul-friendliness, and beats both on naive memory. Attention's slowness at long context is now legible as a *state-size* problem: its state scales with $T$ because it caches the whole history and never compresses it.

**Worked micro-example — the actual code.** The paper ships the whole thing in a few lines. This is the real Listing 1, lightly annotated; note `segsum` (segment-sum): `exp(segsum(A))` builds the 1-SS mask $L$ as a matrix of cumulative log-decays.

```python
import torch
import torch.nn.functional as F
from einops import rearrange

def segsum(x):
    """exp(segsum(x)) is a 1-semiseparable (1-SS) matrix == a scalar SSM."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]   # cumulative log-decay
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), 0)
    return x_segsum.masked_fill(~mask, -torch.inf)               # causal: keep i >= j

def ssd(X, A, B, C, block_len=64, initial_states=None):
    # X:(b,l,h,p)  A:(b,l,h)  B,C:(b,l,h,n) ; l must be divisible by block_len
    X, A, B, C = [rearrange(z, "b (c l) ... -> b c l ...", l=block_len) for z in (X, A, B, C)]
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Intra-chunk (diagonal blocks): quadratic attention form on each chunk
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Each chunk's end-state assuming zero initial state (B-block factors)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Inter-chunk recurrence over the T/Q boundary states (A-block factors)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. State -> output per chunk (C-block factors); add to the intra-chunk term
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state
```

Four einsums and two cumulative sums. Everything but step 3's tiny chunk-level recurrence is a batched matmul, so it runs on tensor cores in native PyTorch — no bespoke CUDA kernel required to get most of the speedup. That is the practical heart of the paper.

**Why it works / when it fails.** The block decomposition is optimal on every axis at once because it uses each mode where it's strongest — quadratic inside small chunks (matmul, cheap at small $Q$), linear across chunks (few steps, negligible). The chunk length $Q$ is a tuning knob: too small and the inter-chunk scan dominates; too large and the diagonal blocks go quadratic. It also inherits a real limitation — this particular decomposition assumes the scalar-$A$ (SSD) structure; a general diagonal SSM needs an extra state dimension inside the mask and loses the clean attention interpretation, so the same trick doesn't transfer for free.

### The Mamba-2 architecture

With the SSD layer as the sequence mixer, the *block* around it can now be redesigned using attention conventions. Mamba-1 wrapped its selective SSM in an SSM-centric block; Mamba-2 rebuilds the block to be attention-shaped and parallelism-friendly.

![Figure 6 from Dao and Gu (2024): the Mamba-2 block. The sequential Mamba-1 block (left) computes the SSM parameters A, B, C as functions of the SSM input X; the parallel Mamba-2 block (right) produces A, X, B, C together from one projection at the start of the block, adds a normalization before the output projection, and shares B, C across the X heads (multi-value attention).](/imgs/blogs/mamba-2-state-space-duality-fig1.webp)

**Parallel parameter projections.** Mamba-1 sees the SSM as a map $X \mapsto Y$, so it first projects to $X$ and *then* derives $A, B, C$ from $X$ — a sequential dependency. Mamba-2 sees the SSD layer as a map $(A, X, B, C)\mapsto Y$, so it produces all four in *parallel* from a single projection at the start of the block. The analogy to attention is exact: $X, B, C$ are the SSM's version of $V, K, Q$, and attention produces $Q, K, V$ together. The payoff is fewer parameters and — crucially — compatibility with Megatron-style tensor-parallel sharding, because a single up-front projection splits cleanly across GPUs.

**Extra normalization.** A normalization layer (RMSNorm/GroupNorm) is added just before the output projection, after the multiplicative gate. This is the same fix TransNormer and RetNet found: the linear-attention family is prone to training instability at scale, and an extra norm tames it. Small change, big effect on whether a 2.7B model trains cleanly.

**Multi-head patterns for SSMs.** Recall each SSD "head" runs the mixer over a $P$-dimensional slice independently. Borrowing the attention taxonomy gives a family of head structures, distinguished by which of $B, C, X$ get independent copies per head:

| SSM pattern | Attention analog | Independent per head | Shared |
|---|---|---|---|
| Multi-head (MHS) | MHA | $A, B, C, X$ | — |
| Multi-contract (MCS) | MQA | $A, C$ | $B, X$ |
| Multi-expand (MES) | MKA | $A, B$ | $C, X$ |
| **Multi-input (MIS)** | **MVA** | $A, X$ | $B, C$ |

Mamba's original design turns out to be the **multi-input SSM / multi-value attention (MVA)** pattern: $X$ is the main input with independent heads, while $B, C$ (the $K, Q$ analogs) are shared across heads (Proposition 7.2). This was never ablated before — it fell out of the SSM viewpoint by accident. And when the paper does ablate it, MVA wins clearly (more on that in the results). The grouped variant (grouped-value attention, GVA) sets the number of shared $B, C$ groups to a multiple of the tensor-parallel degree, exactly as grouped-query attention does for Transformers.

### Systems: tensor parallel, sequence parallel, variable length

The duality's most practical dividend is that Transformer systems techniques port over almost verbatim. Three matter:

**Tensor parallelism with one all-reduce.** Split a block across GPUs. Mamba-1's sequential projections force *two* all-reduces per block (an extra sync to gather $x_c$ before computing $\Delta, B, C$), doubling communication — a real problem when comms are already 10-20% of large-model training time. Mamba-2's parallel projections produce $\Delta, B, C$ directly from the block input $u$, so each GPU holds its own group and needs only **one all-reduce per block**, just like an attention or MLP layer. Choosing GroupNorm (with group count divisible by the TP degree) as the final norm avoids any intra-block communication.

**Sequence / context parallelism.** For sequences too long to fit on one device, split along the sequence axis. Attention's context parallelism is painful because each query block must talk to every key block (communication grows quadratically with workers). SSMs are trivially better: each worker computes its chunk's SSM, passes the *final state* to the next worker, and communication is **linear in the number of workers**. This is literally the SSD block decomposition (Figure 5) executed across devices instead of across chunks.

**Variable-length sequences.** Transformers need careful padding removal or attention-mask surgery to pack sequences of different lengths. SSMs just treat the whole batch as one long sequence and reset the state at boundaries — set $A_t = 0$ at the last token of each sequence so no information leaks into the next. No padding tokens, no wasted compute.

## Experiments and results

The empirical case has three planks: SSD *learns* at least as well as Mamba and Transformer++, it *runs* far faster, and its extra state size *buys* recall.

**Associative recall (MQAR).** The multi-query associative recall task feeds key-value pairs and later probes a seen key — a stress test for a model's ability to store many associations in its state. This is historically SSMs' weak spot (finite state means finite memory).

![Figure 8 from Dao and Gu (2024): on multi-query associative recall, Mamba-1 (N=16) fails, but Mamba-2 improves steadily as its state size grows from N=16 to 64 to 256, overtaking vanilla attention -- because the SSD algorithm makes large states cheap.](/imgs/blogs/mamba-2-state-space-duality-fig3.webp)

Mamba-1 struggles; Mamba-2 handles the task across the board, and — the load-bearing point — accuracy climbs steadily as state size grows from $N=16$ to ${64}$ to ${256}$. The larger state stores more key-value pairs. Because SSD makes large states cheap (see the speed plot below), this knob is now free to turn. Curiously, Mamba-2 beats Mamba-1 even at *matched* $N=16$; the authors admit they don't fully know why, which is a refreshingly honest note.

**Language modeling.** Trained on the Pile at GPT-3 sizes (125M-2.7B), Mamba-2 is Pareto-dominant against both Mamba and the strong "Transformer++" recipe (rotary, SwiGLU, RMSNorm — the Llama-style baseline).

![Figure 9 from Dao and Gu (2024): scaling laws on the Pile at sequence length 8192. Mamba-2 (blue) sits below both Mamba (purple) and Transformer++ (orange) on the perplexity-vs-FLOPs frontier, and is also faster in wall-clock time.](/imgs/blogs/mamba-2-state-space-duality-fig4.webp)

On downstream zero-shot evals, the headline number: **Mamba-2-2.7B (300B tokens on the Pile) outperforms Mamba-2.8B, Pythia-2.8B, and even Pythia-6.9B** trained on the same data — a 2.7B model beating a 6.9B one. Selected rows from Table 1:

| Model | Pile ppl ↓ | LAMBADA acc ↑ | HellaSwag ↑ | ARC-E ↑ | Avg ↑ |
|---|---|---|---|---|---|
| Pythia-2.8B | 6.73 | 64.7 | 59.3 | 64.1 | 55.7 |
| Mamba-2.8B | 6.22 | 69.2 | 66.1 | 69.7 | 59.9 |
| **Mamba-2-2.7B** | **6.09** | **69.7** | **66.6** | 69.6 | **60.2** |

**A hybrid with a few attention layers is best.** The most useful practical finding: pure SSD and pure Transformer++ land at roughly equal perplexity, but *mixing* them beats either. On a 350M/48-layer model, adding attention layers helps until about **10% of layers are attention** (~4-6 layers), then plateaus:

| # attention layers | 0 (pure Mamba-2) | 2 | 4 | 6 | 12 | Transformer++ |
|---|---|---|---|---|---|---|
| Perplexity ↓ | 8.60 | 8.32 | 8.29 | 8.26 | 8.31 | 8.68 |

The interpretation the authors offer — and I find it convincing — is a division of labor: SSD layers do the bulk sequence-to-sequence mixing with compressed state, and a handful of attention layers act as a precise retrieval mechanism for looking up exact past tokens, which a compressed state can't. This is the same lesson [Jamba and RecurrentGemma-style hybrids](/blog/paper-reading/large-language-model/kimi-linear) keep rediscovering.

**Speed.** This is where the duality pays its rent.

![Figure 10 from Dao and Gu (2024): efficiency benchmarks on an A100. Left: SSD is 2-8x faster than Mamba's fused scan and overtakes FlashAttention-2 past sequence length 2K. Right: increasing state size slows Mamba's scan linearly, while SSD is nearly flat -- large states are almost free.](/imgs/blogs/mamba-2-state-space-duality-fig5.webp)

Two claims, both visible: SSD is **2-8x faster** than Mamba-1's already-optimized fused scan (because it uses tensor cores the scan can't), and it crosses over **FlashAttention-2 at sequence length 2K**, running ~6x faster at 16K thanks to linear scaling. The right panel is the one to internalize: Mamba's scan slows *linearly* as you grow the state, so Mamba-1 is stuck with small $N$; SSD is nearly flat, so it can afford the 8x-larger states that the MQAR result showed are worth having.

**What's load-bearing that might not transfer.** A few caveats worth flagging. The head-pattern win (MVA over MQA/MKA) is a genuinely large, un-obvious effect — but it's at 125M-360M scale, and the authors don't claim it survives at 70B. The scaling-law comparison uses the authors' own Transformer++ baseline; it's strong but it's theirs. And the "SSD trains faster than a Transformer" claim comes with an asterisk: at short sequence length a full Transformer amortizes cheap MLP layers that pure Mamba-2 lacks, so the hybrid (half SSD, half MLP) is what actually wins on wall-clock at 2K.

## Critique

**What's genuinely strong.** The theory-to-systems throughline is the best I've seen in a sequence-model paper. It's not "here's a trick and some benchmarks"; it's "here's a mathematical identity, and *because* of it, here are new algorithms, a new architecture, and three systems optimizations — each derived, not tuned into existence." Recasting SSMs as semiseparable matrices is the kind of reframing that makes a whole subfield legible: after this paper, "efficient autoregression = semiseparable mask" is a fact you can reason with. And the honesty is notable — they report the MVA-vs-MQA gap they can't fully explain, they flag that their kernel-feature-map experiments were negative results, and they don't oversell.

**What's weak or unfalsifiable.** The framing "Transformers are SSMs" oversells at the margin, and the authors know it — a footnote concedes the title only covers *certain flavors* of attention. SSD does **not** capture softmax attention or any kernel without a finite feature map; it's a statement about linear/kernel attention, which is a real but smaller claim than the title implies. More substantively: SSD is *strictly less expressive* than the Mamba-1 layer it replaces (scalar $A$ vs. diagonal $A$). The paper's own MQAR result shows Mamba-2 beating Mamba-1 even at matched state — so the win is clearly coming from *somewhere* (the architecture, the larger states the algorithm enables), but the paper cannot cleanly separate "SSD is better" from "SSD lets us afford bigger states." Those are different claims and the experiments conflate them.

**What ablation is missing.** The obvious one: hold the architecture and state size *fixed* and swap only the inner layer (Mamba-1's diagonal-$A$ selective scan vs. SSD's scalar-$A$). That would isolate the pure expressivity cost of the scalar restriction. The paper ablates the block changes (Table 4) and head patterns (Table 5) but never the diagonal-vs-scalar $A$ in a controlled setting — which is the single concession the whole framework rests on. Also under-explored: how the optimal 10% attention ratio and chunk length $Q$ behave past 3B parameters.

**What would change my mind.** If a controlled diagonal-vs-scalar ablation showed the scalar restriction costs real quality at scale — and that a general-diagonal SSM with a comparably hardware-efficient algorithm existed — then SSD would look like a hardware hack that happened to be temporarily convenient, rather than a fundamental sweet spot. The authors themselves hypothesize their structured-matrix algorithms might extend to the diagonal case; if that pans out and beats SSD, the "duality" would remain a beautiful theory but SSD-the-layer would be a footnote.

## What I'd build with this

These are my extrapolations, not the paper's claims.

1. **Chunk-length autotuning.** $Q$ (=$N$=$P$ in the tight setting) is a real free knob that trades inter-chunk-scan cost against diagonal-block quadratic cost. A tiny cost model plus a runtime autotuner (à la how FlashAttention picks block sizes) could pick $Q$ per sequence length and per GPU, and I'd expect measurable gains at the extremes (very short and very long sequences).
2. **Non-scalar masks via the SMA lens.** SMA says *any* structured $L$ gives a linear-time attention. Beyond causal and 1-SS, a Toeplitz or butterfly $L$ would be a data-dependent relative-position scheme with a fast multiply — a principled alternative to RoPE that's still $O(T)$. The [butterfly/Monarch matrix line of work](/blog/paper-reading/large-language-model/efficient-attention-mechanisms-for-large-language-models-a-survey) is the natural place to source $L$.
3. **Non-causal SSD.** The paper notes the matrix-mixer view invites principled *non-causal* variants (for encoders, vision, audio). A semiseparable-but-symmetric mixer would be a clean bidirectional SSM with a matmul algorithm — worth trying wherever a full bidirectional Transformer is the current default.
4. **Interpretability transfer.** Since SSD *is* a form of masked attention, attention-analysis tooling (attention maps, induction-head probes, attention-sink diagnostics) should apply. I'd port an induction-head detector to the SSD mask $L$ and ask whether Mamba-2 grows induction heads the way Transformers do — a concrete bridge to the [interpretability](/blog/paper-reading/large-language-model/attention-residuals) literature.
5. **Hybrid layout search.** Given the flat 10%-attention plateau, the *placement* of those attention layers (the paper spaces them evenly and finds location doesn't matter much at small scale) deserves a proper search at 7B+, jointly with MoE up-cycling of the MLP layers.

## References

- **Paper:** Tri Dao, Albert Gu. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060).
- **Code & checkpoints:** [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba) (the SSD layer is a few dozen lines; Listing 1 above is faithful to it).
- **Prior work it builds on:** Katharopoulos et al., *Transformers are RNNs* (linear attention, 2020); Gu & Dao, *Mamba* (the selective SSM, 2023).
- **Sibling posts on this blog:** [Gated DeltaNet](/blog/paper-reading/large-language-model/gated-delta-networks) and [Kimi Linear](/blog/paper-reading/large-language-model/kimi-linear) sit in the same linear-attention/SSM family; [Lightning Attention](/blog/paper-reading/large-language-model/minimax-01-lightning-attention-hybrid-moe) is another chunked linear-attention design; the [efficient-attention survey](/blog/paper-reading/large-language-model/efficient-attention-mechanisms-for-large-language-models-a-survey) maps the wider landscape SSD unifies.

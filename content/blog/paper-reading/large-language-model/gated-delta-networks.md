---
title: "Gated Delta Networks: Improving Mamba2 with the Delta Rule"
date: "2026-07-13"
description: "A detailed, intuition-first walk through Gated DeltaNet — how one gate and one write-strength combine Mamba2's fast forgetting with DeltaNet's surgical edits, why the two are complementary, and how to train the result on tensor cores."
tags: ["paper-reading", "linear-attention", "gated-deltanet", "mamba2", "deltanet", "delta-rule", "state-space-models", "linear-rnn", "long-context", "sequence-modeling"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 32
paper:
  title: "Gated Delta Networks: Improving Mamba2 with Delta Rule"
  authors: "Songlin Yang, Jan Kautz, Ali Hatamizadeh"
  venue: "ICLR 2025"
  url: "https://arxiv.org/abs/2412.06464"
---

> [!tldr]
> - **What it proposes.** *Gated DeltaNet* is a linear-attention layer whose state update is the **gated delta rule**: $S_t = S_{t-1}\big(\alpha_t(I - \beta_t k_t k_t^\top)\big) + \beta_t v_t k_t^\top$. One scalar gate $\alpha_t$ controls how fast the whole memory fades; one write-strength $\beta_t$ controls how hard the current key–value pair is written.
> - **The key idea in one line.** Mamba2 can forget *fast* but only *uniformly*; DeltaNet can edit *surgically* but can't clear memory. Their transition matrices *multiply* — $\alpha_t I$ times $(I - \beta_t k_t k_t^\top)$ — and you get a layer that does both.
> - **Why it matters.** At 1.3B parameters trained on 100B tokens, Gated DeltaNet beats Mamba2 and DeltaNet on language modeling, commonsense reasoning, in-context retrieval, and long-context understanding — while training at essentially the same speed as DeltaNet on a single H100.
> - **The most surprising result.** On the S-NIAH needle-in-a-haystack suite, DeltaNet and Mamba2 each *fail a different half* of the task; Gated DeltaNet wins both halves. That is the whole thesis, visible in one table.
> - **Where it fails / what it doesn't show.** Pure recurrent models still trail full attention on hard retrieval; the wins are demonstrated only at 1.3B/100B scale; and the write-strength is capped at $\beta_t \in (0,1)$, leaving the negative-eigenvalue state-tracking tricks on the table.

The figure below is the entire method at a glance — the two hybrid stacks on the left, and on the right, the single block whose token mixer we are about to take apart. The rest of this post unpacks it from the recurrence up to the tensor-core kernel.

![Figure 1 from Yang et al. (2025): the Gated DeltaNet-H1 and -H2 hybrid stacks, and the Gated DeltaNet block design — query/key paths run through linear projection, short convolution, SiLU and L2 norm; the value path skips L2; alpha and beta come from linear projections only.](/imgs/blogs/gated-delta-networks-fig1.webp)

## The problem: memory that can't forget, or forgets everything

Start from the pain that makes linear attention interesting in the first place. Standard softmax attention is exact but quadratic: to produce token $t$'s output it looks at all $t$ previous tokens, so training a sequence of length $L$ costs $O(L^2)$ and inference must keep a KV cache that grows without bound. Linear attention (Katharopoulos et al., 2020) trades that exactness for an $O(L)$ recurrence with a *fixed-size* state. When you drop the softmax and the normalization, attention collapses into a running sum of outer products:

$$
S_t = S_{t-1} + v_t k_t^\top \in \mathbb{R}^{d_v \times d_k}, \qquad o_t = S_t q_t \in \mathbb{R}^{d_v}
$$

Here $k_t, q_t \in \mathbb{R}^{d_k}$ are the key and query for token $t$, $v_t \in \mathbb{R}^{d_v}$ is its value, and $S_t$ is a **matrix-valued state** of shape $d_v \times d_k$ that summarizes everything seen so far. The two axes of $S_t$ are worth naming out loud: each column indexes a key dimension, each row a value dimension, so $S_t$ is literally a table of key→value associations added up as outer products $v_i k_i^\top$. To read it, you multiply by a query: $o_t = S_t q_t = \sum_{i \le t} v_i (k_i^\top q_t)$ — every stored value, re-weighted by how much its key aligns with the current query. This is a **key–value associative memory**, the same "tensor product representation" idea Smolensky proposed in 1990.

The catch is capacity. An outer-product memory can store roughly $d_k$ *orthogonal* key–value pairs before new writes start landing on top of old ones. Once the sequence is longer than the state dimension, keys can no longer all be orthogonal, and reads return a blur of superimposed values — a **memory collision** (Schlag et al., 2021). This is exactly why vanilla linear transformers underperform on in-context retrieval: they have no mechanism to *make room*.

Two lines of work each fixed *half* of this, and each fix created the other's blind spot.

**Mamba2 adds a forget gate.** If the memory is filling up, decay it. Mamba2 (Dao & Gu, 2024) multiplies the whole state by a data-dependent scalar $\alpha_t \in (0,1)$ before each write:

$$
S_t = \alpha_t S_{t-1} + v_t k_t^\top
$$

Old associations shrink geometrically, so stale information eventually drops below the noise floor and the memory never truly saturates. But $\alpha_t$ is a *single number* multiplying *everything*. If the model wants to forget one specific fact — say, the previous document's protagonist as it crosses into a new document — it can only turn down the brightness on the entire board. There is no way to spare the one association you still need while erasing its neighbor.

**DeltaNet adds a targeted write.** The delta rule (Widrow & Hoff, 1960; Schlag et al., 2021) does the opposite. Before writing the new value for key $k_t$, it first *reads out* whatever the memory currently returns for that key, then writes a blend of old and new:

$$
S_t = S_{t-1} - (\underbrace{S_{t-1}k_t}_{v_t^{\text{old}}})k_t^\top + (\underbrace{\beta_t v_t + (1-\beta_t)S_{t-1}k_t}_{v_t^{\text{new}}})k_t^\top = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top
$$

where $\beta_t \in (0,1)$ is the "writing strength". This is *surgical*: it edits only the $k_t$ direction of the memory and leaves everything orthogonal to $k_t$ untouched. On synthetic recall it is excellent. But notice what it *cannot* do: there is no term that shrinks the state as a whole. When the context switches and a large block of old information becomes irrelevant, DeltaNet can only overwrite it one key at a time, as those exact keys happen to recur. It has no **fast clear**. Empirically this shows up as strong synthetic recall but only moderate real-world performance (Yang et al., 2024b).

So the field had a forgetter that couldn't aim and an aimer that couldn't forget. The paper's claim is that these are *complementary*, and that combining them is almost embarrassingly simple.

## Contributions

Tightened into a map for the rest of the post, the paper delivers four things:

1. **The gated delta rule** — a single recurrence that multiplies Mamba2's scalar decay by DeltaNet's targeted Householder update, so a layer can both clear memory fast ($\alpha_t \to 0$) and edit one slot precisely ($\beta_t$), or do the pure delta rule ($\alpha_t \to 1$).
2. **A hardware-efficient chunkwise training algorithm** that carries the gate through the WY representation and UT transform, so the gated rule trains with dense matmuls on tensor cores at essentially DeltaNet's speed.
3. **The Gated DeltaNet architecture** — a Llama-style stack with the gated delta rule as its token mixer — which beats Mamba2 and DeltaNet across language modeling, recall, and long context at 1.3B/100B scale.
4. **Two hybrids**, Gated DeltaNet-H1 (with sliding-window attention) and -H2 (with Mamba2 layers too), that trade a little purity for better retrieval and higher training throughput.

## Method

The heart of the paper is one equation, three lenses on why it is the *right* equation, and one algorithm for making it fast. We climb each rung — problem, intuition, mechanism, math, worked example, failure mode — for every load-bearing piece.

### Linear attention as a key–value memory

**The problem.** We need a sequence model whose per-token cost and memory are constant in the context length, not linear.

**Intuition.** Think of the state $S$ as a whiteboard you can only *add* to. Every token scribbles its value $v_i$ into the region labeled by its key $k_i$. To answer a question, you shine a query $q$ at the board and read back a weighted average of everything written, weighted by how much each region's label matches $q$. The board never grows — but it also never gets erased, so eventually the scribbles overlap into mush.

**Mechanism.** For each token: (1) form the outer product $v_t k_t^\top$, a rank-one matrix that writes $v_t$ into the $k_t$ direction; (2) add it to the running state; (3) to read, project the state onto the query. That's the whole layer.

**Math.** Expanding the recurrence gives both a sequential (RNN) form and a parallel (attention) form — the two are algebraically identical, which is what lets us *train* like attention and *infer* like an RNN:

$$
o_t = \sum_{i=1}^{t} v_i (k_i^\top q_t) \in \mathbb{R}^{d_v}, \qquad O = (QK^\top \odot M)V \in \mathbb{R}^{L \times d_v}
$$

Here $Q, K \in \mathbb{R}^{L \times d_k}$ and $V \in \mathbb{R}^{L \times d_v}$ stack the per-token vectors, and $M \in \mathbb{R}^{L \times L}$ is the causal mask with $M_{ij} = 1$ for $i \ge j$ and ${0}$ otherwise (token $i$ may attend to $j$ only if $j$ came first). The left form costs $O(L d_k d_v)$ and streams one token at a time; the right form is a pair of $L \times L$ matmuls, quadratic but parallel.

**Worked micro-example.** Take $d_k = d_v = 2$, two tokens. Write $(k_1, v_1) = ([1,0], [3,0])$ then $(k_2, v_2) = ([0,1], [0,5])$. After both writes,

$$
S_2 = v_1 k_1^\top + v_2 k_2^\top = \begin{bmatrix} 3 & 0 \\ 0 & 0 \end{bmatrix} + \begin{bmatrix} 0 & 0 \\ 0 & 5 \end{bmatrix} = \begin{bmatrix} 3 & 0 \\ 0 & 5 \end{bmatrix}.
$$

Query with $q = k_1 = [1,0]$: $S_2 q = [3, 0] = v_1$ — perfect recall, because $k_1 \perp k_2$. Now add a third key $k_3 = [1,1]/\sqrt2$ that is *not* orthogonal to either; its write bleeds into both stored slots, and querying $k_1$ no longer returns a clean $v_1$. That bleed is the collision.

**Why it works / when it fails.** It works because orthogonal keys give orthogonal writes, and reading is just a projection. It fails the moment you need more than $\approx d_k$ distinguishable associations at once — precisely the long-context retrieval regime. Every technique below is a different answer to "what do we do when the board fills up?"

### Mamba2: uniform decay

**The problem.** The board fills up. The cheapest fix: continuously fade it so old scribbles vanish on their own.

**Intuition.** A dimmer switch on the whole whiteboard. Each step, everything already on the board gets a little fainter; only the freshest writes stay bright. Nothing overflows because the past is always decaying — but the dimmer is a single knob for the entire surface.

**Mechanism.** Multiply the state by a scalar $\alpha_t \in (0,1)$ before adding the new outer product. The $\alpha_t$ is *data-dependent* — computed from the current token — so the model can choose to remember (turn $\alpha_t$ up toward 1) or dump (turn it down toward 0) based on what it just read.

**Math.** The recurrence and its parallel form:

$$
S_t = \alpha_t S_{t-1} + v_t k_t^\top, \qquad O = \big((QK^\top) \odot \Gamma\big)V
$$

Define the cumulative decay $\gamma_j = \prod_{i=1}^{j}\alpha_i$. Then the decay-aware mask is $\Gamma_{ij} = \gamma_i/\gamma_j$ for $i \ge j$ and ${0}$ otherwise: a token pair $(i, j)$ is down-weighted by exactly the product of decays between them. The paper notes this recurrence is the shared skeleton of a whole family — Gated RFA, xLSTM, Gated RetNet — and reduces to RetNet or Lightning Attention when $\alpha_t$ is a fixed constant instead of data-dependent. Dao & Gu call the parallel↔recurrent equivalence *state-space duality*.

**Worked micro-example.** Reuse the two-token board but decay with $\alpha_2 = 0.1$ before the second write. Now $S_2 = 0.1\begin{bmatrix}3&0\\0&0\end{bmatrix} + \begin{bmatrix}0&0\\0&5\end{bmatrix} = \begin{bmatrix}0.3 & 0\\ 0 & 5\end{bmatrix}$. Querying $k_1$ returns $[0.3, 0]$ — the first fact has been dimmed to a tenth. Good if $v_1$ was stale; a disaster if you still needed it.

**Why it works / when it fails.** Geometric decay guarantees the state stays bounded and stale mass eventually decays away — that is real, useful forgetting. It fails on *selectivity*: because one scalar scales the entire matrix, forgetting the distractor also forgets the needle sitting right next to it. Hold that thought; it is exactly what the S-NIAH case study will measure.

### DeltaNet: the delta rule

**The problem.** We want to forget *the specific thing* that's now wrong, not fade everything indiscriminately.

**Intuition.** Instead of a dimmer over the whole board, use an eraser sized to a single label. Walk up to the region labeled $k_t$, wipe whatever value is written there, and write the new one — leaving every other region exactly as it was.

**Mechanism.** This is the read-modify-write we saw earlier, but it's worth seeing as a linear operator. The update multiplies the old state by $(I - \beta_t k_t k_t^\top)$ — a **generalized Householder matrix**. Geometrically, $k_t k_t^\top$ projects onto the $k_t$ direction (assume $\|k_t\|=1$), so $I - \beta_t k_t k_t^\top$ shrinks *only* the component of the memory along $k_t$ by a factor $\beta_t$, and acts as the identity on everything orthogonal. Then $\beta_t v_t k_t^\top$ writes the new value into that freshly-cleared direction.

**Math.**

$$
S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top
$$

with $\beta_t \in (0,1)$. When $\beta_t = 1$ this is a hard replace (erase the old $k_t$ value entirely, write $v_t$); when $\beta_t = 0$ it's a no-op (keep everything). The transition matrix $I - \beta_t k_t k_t^\top$ is *identity-plus-rank-one* — data-dependent, and strictly more expressive than Mamba2's data-dependent *diagonal* $\alpha_t I$, which is what lets DeltaNet do things a diagonal recurrence provably cannot (Yang et al., 2024b).

**Worked micro-example.** Board holds $S = \begin{bmatrix}3&0\\0&5\end{bmatrix}$ from before. A new token arrives on the *same* key $k = [1,0]$ with value $v = [9,0]$ and $\beta = 1$. Old read: $Sk = [3,0]$. Update: $S(I - kk^\top) = \begin{bmatrix}3&0\\0&5\end{bmatrix}\begin{bmatrix}0&0\\0&1\end{bmatrix} = \begin{bmatrix}0&0\\0&5\end{bmatrix}$ (the $k_1$ slot wiped), then $+\,vk^\top = \begin{bmatrix}9&0\\0&0\end{bmatrix}$, giving $\begin{bmatrix}9&0\\0&5\end{bmatrix}$. The $[0,1]$ slot's value ${5}$ is untouched. That is the surgical property Mamba2 lacks.

**Why it works / when it fails.** It works because a rank-one transition can rewrite exactly one direction, giving collision-resistant, in-place updates. It fails on *bulk forgetting*: to clear $N$ stale associations it needs $N$ separate writes on those exact keys, which may never recur. Across a document boundary, where you'd like to wipe the whole board at once, DeltaNet is stuck editing pixel by pixel.

### The gated delta rule

Now the payoff. We have a forgetter that can't aim and an aimer that can't forget. What if the state transition were *both* operators, one after the other?

**The problem.** A single layer should be able to fade the whole memory *and* surgically edit one slot, choosing per token which it needs.

**Intuition.** Put a dimmer switch *and* a precision eraser on the same board. Each step: first turn the global brightness down by $\alpha_t$ (fast, uniform forgetting), then erase-and-rewrite the single slot $k_t$ by $\beta_t$ (surgical editing). Two independent knobs, per token.

**Mechanism.** Compose the two transition matrices. DeltaNet's step is "multiply by $I - \beta_t k_t k_t^\top$"; Mamba2's is "multiply by $\alpha_t I$". Do both — multiply by $\alpha_t(I - \beta_t k_t k_t^\top)$ — then add the write. That single change is the entire contribution.

**Math.**

$$
S_t = S_{t-1}\big(\alpha_t(I - \beta_t k_t k_t^\top)\big) + \beta_t v_t k_t^\top \tag{10}
$$

Every symbol carries over: $\alpha_t \in (0,1)$ is the scalar decay gate (Mamba2's parameterization), $\beta_t \in (0,1)$ is the write strength (DeltaNet's), $k_t k_t^\top$ is the rank-one projector onto the current key. The two special cases are the point of the design: set $\alpha_t \to 1$ and Eq. 10 becomes the *pure delta rule*; set $\alpha_t \to 0$ and the whole state is wiped in a single step, ready for a fresh context. Everything in between is a blend.

![Three transition matrices, one lineage: Mamba2's scalar decay α·I, DeltaNet's rank-one Householder I−βkkᵀ, and Gated DeltaNet's product α(I−βkkᵀ), with the S-NIAH-2 accuracy each earns at 8K tokens.](/imgs/blogs/gated-delta-networks-2.webp)

The figure lines up the three transition matrices side by side. Read the columns: Mamba2 *scales every entry by* $\alpha$ and so *forgets the one slot you need*; DeltaNet *erases only the $k$ direction* but *cannot clear memory fast*; Gated DeltaNet *decays all, then edits the $k$ slot*, with the extremes $\alpha \to 0$ (wipe) and $\alpha \to 1$ (pure delta) available on demand. The bottom row previews the receipt we cash in §Experiments — on S-NIAH-2 at 8K tokens the three score 17.0%, 14.4%, and 29.6% respectively.

**How one step actually flows.** Unrolling the mechanism as a data-flow makes the "read, correct, decay, write" structure explicit:

![One gated-delta step as a data-flow: from the memory matrix S and key k, recall the old value S k; form the error v − old; separately decay the whole state by α; then write S ← αS + β(v−old)kᵀ and read out o = S q.](/imgs/blogs/gated-delta-networks-3.webp)

Trace it left to right. The **recall** step reads what the current memory returns for key $k$: $\text{old} = Sk$. The **error** step measures how wrong that recall is against the target value: $v - \text{old}$. In parallel, the **decay** branch dims the whole state by $\alpha$. The **write** step commits both at once — $S \leftarrow \alpha S + \beta(v - \text{old})k^\top$ — scaling the correction by the write strength $\beta$. Finally, **read out** answers a downstream query with $o = Sq$. Two branches (correct and decay) merge at the write; that merge is the gated delta rule.

**Worked micro-example.** Board $S = \begin{bmatrix}3&0\\0&5\end{bmatrix}$, gate $\alpha = 0.5$, incoming $(k, v) = ([1,0],[9,0])$, $\beta = 1$. Decay first: $\alpha S = \begin{bmatrix}1.5&0\\0&2.5\end{bmatrix}$. Old read after decay: $(\alpha S)k = [1.5, 0]$. The Householder wipes the $k$ slot of the *decayed* board and writes: result $= \alpha S(I - kk^\top) + vk^\top = \begin{bmatrix}0&0\\0&2.5\end{bmatrix} + \begin{bmatrix}9&0\\0&0\end{bmatrix} = \begin{bmatrix}9&0\\0&2.5\end{bmatrix}$. The needle at $k$ is freshly written to ${9}$; the orthogonal slot, which held ${5}$, has been *gently* dimmed to ${2.5}$ rather than wiped or left stale. That "edit one, fade the rest" behavior is unreachable by either parent rule alone.

**Why it works / when it fails.** It works because matrix multiplication of the two transitions preserves both capabilities — the product is still a valid, cheap-to-apply linear operator, and the model learns per-token when to lean on the gate versus the write. It fails to be a free lunch in two ways worth flagging up front: the extra gate makes the transition matrix slightly more expensive than Mamba2's (we'll see 2–3K tokens/sec of throughput cost), and because $\beta_t$ stays in $(0,1)$ the transition eigenvalues stay non-negative, so the state-tracking tricks that need $\beta_t \in (0,2)$ are out of scope.

### The online-learning lens

There is a deeper reason the gated delta rule is not an arbitrary hack, and it comes from reading the recurrence as an *optimization algorithm running at test time*. This is the lens that unifies all five models in the paper's Table 1, extracted here:

![Table 1 from Yang et al. (2025): five linear-RNN update rules — LA, Mamba2, Longhorn, DeltaNet, Gated DeltaNet — each written as the closed-form solution to an online-learning objective with a matching online update.](/imgs/blogs/gated-delta-networks-fig2.webp)

**The problem.** Why *these* update rules and not others? We'd like a principle that tells us what each layer is optimizing.

**Intuition.** Treat the hidden state $S$ not as storage but as a tiny model — a linear map from keys to values — that is being *trained on the fly* as tokens stream in. Each token is one training example $(k_t, v_t)$: "when you see key $k_t$, you should output $v_t$." The recurrence is one step of gradient descent on that regression problem. This is the *fast-weight programming* / *test-time training* view (Schlag et al., 2021; Sun et al., 2024; Wang et al., 2025).

**Mechanism.** Define a per-token regression loss on the fast weight $S$, take one SGD step, and read off the update. The delta rule falls out; adding weight decay to that SGD step gives the gated delta rule.

**Math.** With the squared-error objective

$$
\mathcal{L}(S_t) = \tfrac{1}{2}\lVert S_t k_t - v_t \rVert^2,
$$

one gradient step with learning rate $\beta_t$ is

$$
S_{t+1} = S_t - \beta_t \nabla \mathcal{L}(S_t) = S_t - \beta_t (S_t k_t - v_t)k_t^\top = S_t(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top.
$$

That is *exactly* the delta rule, with the write strength $\beta_t$ revealed as the **learning rate**. Now the gated version is just the same SGD step with an **adaptive weight decay** factor $\alpha_t$ — the oldest regularization trick in the book — pulling $S$ toward zero before each update:

$$
S_{t} = \alpha_t S_{t-1} - \beta_t\big(\alpha_t S_{t-1} k_t - v_t\big)k_t^\top = S_{t-1}\big(\alpha_t(I - \beta_t k_t k_t^\top)\big) + \beta_t v_t k_t^\top.
$$

![The gated delta rule is SGD with weight decay: DeltaNet is one plain test-time gradient step on the regression loss (giving I − βkkᵀ); adding a weight-decay factor α to that same step yields the gated delta rule α(I − βkkᵀ) and relaxes the state-retention prior.](/imgs/blogs/gated-delta-networks-5.webp)

The before/after says it cleanly. On the left, DeltaNet is *plain test-time SGD*: fast-weight state, regression loss $\tfrac12\lVert Sk - v\rVert^2$, one step $S - \beta\nabla\mathcal{L}$, which equals $S(I - \beta k k^\top) + \beta v k^\top$ — and because there is no forgetting, the fast weight can *saturate*. On the right, Gated DeltaNet is *SGD with weight decay*: same state, same loss, but the step carries the decay factor $\alpha$, giving $\alpha S(I - \beta k k^\top) + \beta v k^\top$. The paper frames this through the objective column of Table 1: linear RNNs carry a regularizer $\lVert S_t - S_{t-1}\rVert_F^2$ that keeps the state from drifting off its previous value — good for retention, but ruinous once the state is saturated with superimposed facts. Mamba2 and Gated DeltaNet relax that regularizer to $\lVert S_t - \alpha_t S_{t-1}\rVert_F^2$, and it is precisely the $\alpha_t$ that lets the state *deviate* from its past — i.e., forget on purpose.

**Worked micro-example.** Read the objective column top to bottom. LA minimizes $\lVert S_t - S_{t-1}\rVert_F^2 - 2\langle S_t k_t, v_t\rangle$ — stay close to the past, but correlate with $v_t$ — and the solution is the plain additive write. Mamba2 swaps the anchor $S_{t-1}$ for $\alpha_t S_{t-1}$: stay close to the *decayed* past. DeltaNet keeps the un-decayed anchor but replaces the target with the *residual* $\beta_t(v_t - S_{t-1}k_t)$: correct toward what's still missing. Gated DeltaNet uses the decayed anchor *and* the residual against the decayed read, $\beta_t(v_t - \alpha_t S_{t-1}k_t)$ — the union of both. Longhorn is the odd sibling: it optimizes the same regression loss but solves it *implicitly* (closed-form optimal step, $\epsilon_t = \beta_t/(1+\beta_t k_t^\top k_t)$) rather than by one explicit gradient step, which is why its update looks like a self-normalized delta rule.

**Why it works / when it fails.** The lens explains *why* the gated delta rule generalizes better than Mamba2 (a real regression objective beats a bare inner-product loss at modeling key–value associations) and *why* it beats DeltaNet (weight decay prevents the fast weight from saturating). It also predicts the frontier: richer objectives — nonlinear regression as in TTT and Titans, or full-history least-squares as in the Mesa layer — should be even more expressive, at the cost of a nonlinear recurrence that breaks the clean parallel training we're about to build. The gated delta rule sits at the sweet spot where the objective is expressive *and* the update stays a linear operator.

### Chunkwise-parallel training

An update rule you can't train efficiently is a paper, not a system. The sequential form of Eq. 10 — one token at a time, each depending on the last — is death on a GPU, which wants big dense matmuls, not a long dependency chain of rank-one updates. The DeltaNet paper solved this for the ungated rule; the contribution here is carrying the gate through that solution without losing the matmuls.

**The problem.** Compute the exact recurrence, but as tensor-core matmuls rather than a sequential scan.

**Intuition.** Split the sequence into chunks of size $C$ (say 64). *Within* a chunk, unroll the recurrence into closed-form matrices you can compute in parallel; *between* chunks, pass a single state matrix forward. You pay a short sequential hop per chunk (there are $L/C$ of them) but everything inside a chunk is dense linear algebra.

**Mechanism.** The obstacle is that unrolling a chunk of DeltaNet gives a *product* of Householder matrices, $\prod_i (I - \beta_i k_i k_i^\top)$ — inherently sequential. The classic **WY representation** (Bischof & Van Loan, 1985) rewrites any such product as a single $I - WK^\top$, and the **UT transform** (Joffrain et al., 2006) computes the needed $W$ (and its value-side twin $U$) by inverting one small $C \times C$ matrix per chunk. That inverse is the trick: it collapses the sequential product into a matmul-friendly form.

![Chunkwise training keeps the gate and the matmuls: split the length-L sequence into chunks of size C; build the decay-scaled K Kᵀ block; invert one C×C matrix per chunk via the UT transform to get W = T K and Ũ = T V; carry a single state matrix from chunk to chunk; and produce outputs O with tensor-core matmuls.](/imgs/blogs/gated-delta-networks-4.webp)

The pipeline shows the flow. **Split** the length-$L$ sequence into chunks of size $C$. Build the **intra-chunk** decay-scaled $KK^\top$ block. Run the **UT transform** — a single $C \times C$ inverse, once per chunk — to get $W = TK$ and $\tilde U = TV$. **Carry** the state matrix $S$ from one chunk to the next. Emit outputs $O$ with dense matmuls on **tensor cores**.

**Math.** For the ungated rule, the UT transform gives, per chunk $[t]$,

$$
T_{[t]} = \Big[\,I + \operatorname{strictLower}\!\big(\operatorname{diag}(\beta_{[t]})\,K_{[t]}K_{[t]}^\top\big)\Big]^{-1}\operatorname{diag}(\beta_{[t]}) \in \mathbb{R}^{C\times C},
$$

with $W_{[t]} = T_{[t]}K_{[t]}$ and $U_{[t]} = T_{[t]}V_{[t]}$. Here $K_{[t]}, V_{[t]}$ are the key/value blocks of chunk $[t]$, $\operatorname{diag}(\beta_{[t]})$ places the per-token write strengths on the diagonal, and $\operatorname{strictLower}$ keeps the strictly-lower-triangular part (a token only depends on earlier tokens in its chunk). The state then advances and the outputs read out by matmul:

$$
S_{[t+1]} = S_{[t]} + \big(U_{[t]} - W_{[t]}S_{[t]}^\top\big)^\top K_{[t]}, \qquad O_{[t]} = Q_{[t]}S_{[t]}^\top + \big(Q_{[t]}K_{[t]}^\top \odot M\big)\big(U_{[t]} - W_{[t]}S_{[t]}^\top\big).
$$

The gated extension changes exactly one thing: the intra-chunk $K K^\top$ block is *decay-weighted* before the inverse. With the per-chunk cumulative decay $\Gamma_{[t]}$ (products of $\alpha$ within the chunk), the value-side transform becomes

$$
\tilde U_{[t]} = \Big[\,I + \operatorname{strictLower}\!\big(\operatorname{diag}(\beta_{[t]})\,(\Gamma_{[t]} \odot K_{[t]}K_{[t]}^\top)\big)\Big]^{-1}\operatorname{diag}(\beta_{[t]})\,V_{[t]},
$$

and the cross-chunk quantities pick up decay factors: queries decay to the *start* of their chunk ($\overleftarrow{q}$), keys to the *end* ($\overrightarrow{k}$), and the carried state over the *whole* chunk ($\overrightarrow{S}_{[t]} = \gamma_{[t]}^{C} S_{[t]}$). Substituting these into the DeltaNet chunk equations yields the gated update; the appendix proves the extended WY representation by induction. The upshot: the same tensor-core-friendly kernel, with a handful of extra element-wise decay multiplies.

**Worked micro-example (pseudocode).** The gated recurrence itself is a dozen lines; the chunkwise kernel is an optimization *of* it. Here is the reference recurrence that the kernel must match exactly, in PyTorch-shaped code:

```python
import torch

def gated_delta_rule(q, k, v, alpha, beta):
    # q, k: (L, d_k)   v: (L, d_v)   alpha, beta: (L,)   -- one head
    L, d_k = k.shape
    d_v = v.shape[1]
    S = torch.zeros(d_v, d_k)            # matrix-valued state
    outs = []
    for t in range(L):
        kt = k[t]                         # (d_k,)
        # 1. decay the whole state by the gate alpha_t  (weight decay)
        S = alpha[t] * S
        # 2. recall the old value the (decayed) memory holds for k_t
        old = S @ kt                      # (d_v,)
        # 3. write the beta-scaled correction into the k_t direction only
        S = S + beta[t] * torch.outer(v[t] - old, kt)   # Householder + write
        # 4. read out the answer for the query q_t
        outs.append(S @ q[t])             # (d_v,)
    return torch.stack(outs)              # (L, d_v)
```

Lines 1–3 are Eq. 10 written out: decay, recall, correct-and-write. A chunkwise kernel replaces the Python loop with the $C \times C$ inverse above so that a whole chunk's worth of these steps becomes three or four matmuls — but it computes the identical result.

**Why it works / when it fails.** It works because the WY/UT machinery is an *exact* re-expression, not an approximation: chunkwise training and the sequential recurrence produce bit-comparable states (up to floating point). The cost is the per-chunk $C \times C$ inverse, which is why chunk size $C$ trades intra-chunk parallelism against inverse cost, and why the more expressive transition matrix makes Gated DeltaNet 2–3K tokens/sec slower than Mamba2, whose diagonal transition needs no inverse at all.

### Architecture and hybrids

**The problem.** Wrap the token mixer into a full block, and decide when a pure recurrent stack is not enough.

**Intuition.** The gated delta rule is a *token mixer* — the thing that moves information across positions — and it drops into the same slot self-attention occupies in a Llama block. But a fixed-size recurrent state is fundamentally limited for exact retrieval; sometimes you want a few layers of real attention to handle local comparisons and precise lookups. Interleaving the two is the hybrid recipe.

**Mechanism.** The block follows Llama's macro design: alternating token-mixer layers and SwiGLU MLPs. Inside the mixer (right panel of Figure 1), $q, k, v$ each come from a linear projection followed by a short convolution and SiLU; $q$ and $k$ additionally get **L2 normalization** for training stability; the gate $\alpha$ and write-strength $\beta$ come from linear projections only. The output passes through normalization and an output gate before the final projection. Two hybrids stack this with attention:

- **Gated DeltaNet-H1** interleaves Gated DeltaNet layers with **sliding-window attention (SWA)**, following Griffin and Samba — recurrence for long-range memory, a 2K-token attention window for local detail.
- **Gated DeltaNet-H2** adds Mamba2 layers into the mix: a Mamba2 + Gated DeltaNet + SWA pattern.

**Math / configuration.** No new equations — the design choices are the content, and the ablations pin them down (§Experiments). The load-bearing ones: L2 normalization on $q, k$ is *essential* (L1 variants lose ~3 perplexity points), the short convolution and output gate each matter (~1.5–1.8 perplexity), and a head dimension of 128 is the sweet spot.

**Why it works / when it fails.** The hybrid works because attention and recurrence fail in complementary regimes — attention is exact but quadratic and window-limited under SWA; recurrence is cheap and unbounded-range but lossy — so a few attention layers relieve the recurrent layers of the retrieval burden they're worst at. It "fails" only in the sense that hybrids are no longer pure linear-time models: the SWA layers reintroduce a (bounded) attention cost, which is exactly why they also *raise* throughput at short sequence lengths, where the Flash-Attention-2 kernel is extremely fast.

## Experiments and results

All the headline models share one training recipe, which matters for reading the numbers: **1.3B parameters, 100B tokens** sampled from FineWeb-Edu, AdamW (peak LR $4\text{e-}4$, weight decay ${0.1}$, gradient clip ${1.0}$), cosine schedule with a 1B-token warmup, batch size 0.5M tokens, Llama2 tokenizer (32K vocab), training length 4K, and a 2K sliding window for Samba and the hybrids. Every model is trained under identical conditions, which is what makes the comparison fair.

### The case study that is the whole argument

Before the aggregate benchmarks, the paper runs one diagnostic that carries the thesis: **Single Needle-In-A-Haystack (S-NIAH)** from RULER, where a key–value pair is hidden in a long context and the model must recall the value given the key. Three variants of rising difficulty, at 1.3B:

| Model | S-NIAH-1 (passkey) 4K | 8K | S-NIAH-2 (number) 4K | 8K | S-NIAH-3 (uuid) 4K |
|---|---|---|---|---|---|
| DeltaNet | 99.0 | **98.8** | 18.6 | 14.4 | 22.4 |
| Mamba2 | 65.4 | 30.4 | 56.2 | 17.0 | 4.6 |
| Gated DeltaNet | 91.4 | **91.8** | **92.2** | **29.6** | **27.6** |

Read it as three findings, each isolating one mechanism:

**Decay hurts pure retention (S-NIAH-1).** With a repetitive synthetic context, the model just has to hold one fact for a long time. DeltaNet, with no decay, is near-perfect out to 8K (98.8). Mamba2 collapses — 65.4 at 4K, 30.4 at 8K — because its uniform decay throws away the very thing it should keep. Gated DeltaNet degrades far less (91.8 at 8K) precisely because the delta rule lets it *write once and stop decaying that slot as hard*.

**Gating enables filtering (S-NIAH-2/3).** Now the haystack is real essay text, so the model must store lots of potentially-relevant material and *not* let it collide. Here DeltaNet's lack of a clear mechanism is fatal — 18.6 → 14.4 as length grows — because irrelevant writes superimpose and become indistinguishable. Mamba2 and Gated DeltaNet, which *can* forget, hold up better.

**The delta rule helps memorization (S-NIAH-3).** When the value is a hard-to-compress UUID instead of a number, Mamba2 cracks (4.6 at 4K) while Gated DeltaNet holds (27.6), confirming that the delta rule carries real memorization capacity the gate alone can't supply.

The punchline is structural: **DeltaNet wins the first row and loses the second; Mamba2 wins the second and loses the first; Gated DeltaNet wins both.** That is the complementary-mechanisms claim, measured.

### Language modeling and commonsense reasoning

On perplexity and zero-shot commonsense accuracy at 1.3B (Table 3), Gated DeltaNet is the best *pure recurrent* model, and the hybrids top the chart outright:

| Model | Wiki. ppl ↓ | LMB. ppl ↓ | Avg. acc ↑ (8 tasks) |
|---|---|---|---|
| RetNet | 19.08 | 17.27 | 52.02 |
| Mamba | 17.92 | 15.06 | 53.12 |
| DeltaNet | 17.71 | 16.88 | 52.14 |
| Mamba2 | 16.56 | 12.56 | 54.89 |
| **Gated DeltaNet** | **16.42** | **12.17** | **55.32** |
| Transformer++ | 18.53 | 18.32 | 52.25 |
| Samba | 16.13 | 13.29 | 54.00 |
| **Gated DeltaNet-H1** | 16.07 | 12.12 | **56.40** |
| **Gated DeltaNet-H2** | **15.91** | 12.55 | 56.18 |

Gated DeltaNet edges Mamba2 (55.32 vs 54.89 average, 16.42 vs 16.56 Wikitext perplexity) and clears DeltaNet by more than three accuracy points. H1 and H2 then pull ahead of every baseline, including the pure Transformer++ (52.25) and Samba (54.00). The margins over Mamba2 are modest but consistent — the story is "strictly better on the same budget," not "a different league."

### In-context retrieval, length, and long context

On **real-world recall** (Table 4, inputs truncated to 2K), the aggregate picture is: pure recurrent models trail full attention, hybrids beat pure attention. Gated DeltaNet averages 30.6 across SWDE/SQuAD/FDA/TriviaQA/NQ/Drop, ahead of Mamba2 (29.8) and DeltaNet (26.2); Gated DeltaNet-H2 reaches 40.1, past Transformer++ (37.0). The paper is honest that the pure-recurrent gap over Mamba2 here is *smaller* than in S-NIAH, attributing it to instruction-unaligned small models making repetition errors that swamp update-rule differences.

For **length extrapolation** to 20K tokens across six long-context benchmarks:

![Figure 2 from Yang et al. (2025): perplexity vs. sequence length (4K→20K) on GovReport, QMSum, NarrativeQA, Qasper, CodeParrot, and PG19; Gated DeltaNet and its hybrids track near the bottom of each panel while Mamba2 (purple) and Mamba1 (orange) drift up.](/imgs/blogs/gated-delta-networks-fig3.webp)

Gated DeltaNet achieves the lowest overall perplexity among RNN models, and while the results are mixed panel-to-panel, it is the most *robust* — a proxy for better memory management. The hybrids improve further by offloading local modeling to attention. On **LongBench** (14 tasks), Gated DeltaNet averages 16.6 among recurrent models (vs DeltaNet 13.6, Mamba2 13.5), with especially large gains on few-shot in-context learning and code; H2 reaches 18.4.

### Throughput and what's load-bearing

On a single H100, Gated DeltaNet trains at *essentially DeltaNet's speed* — the gated delta rule adds only marginal overhead — and both are 2–3K tokens/sec slower than Mamba2, the tax for a more expressive (non-diagonal) transition. The hybrids are *faster* than the pure recurrent models at short sequence lengths because the 2K SWA layers ride the highly-optimized Flash-Attention-2 kernel; Gated DeltaNet-H1 in particular holds strong throughput across all lengths.

The ablations (400M/15B) name what actually carries the design. Swapping the gated rule for the **naive delta rule** costs the most: perplexity jumps from 27.35 to 30.87. Removing the **short convolution** (→ 28.95) or the **output gate** (→ 29.12) each hurt; output norm is marginal (27.55). **L2 normalization is essential** — every L1 variant lands near 30 — and **SiLU** consistently beats other feature maps. Head dimension 128 is the trade-off sweet spot (64 → 28.31, 256 → 27.13 but costlier). And for the hybrid *ordering*, Mamba2 + Gated DeltaNet + SWA is best.

What might not transfer? Everything here is at 1.3B/100B on FineWeb-Edu with instruction-*unaligned* base models. The retrieval numbers in particular are depressed by repetition errors that instruction tuning would fix, so the *relative* gap between update rules at chat scale is genuinely unknown from this paper. The clean S-NIAH separation is the most portable result; the aggregate benchmark margins are the least.

## Critique

**What's genuinely strong.** The core idea is the good kind of obvious-in-hindsight: two known mechanisms whose transition matrices simply *multiply*, with a one-line recurrence and a clean special-case story ($\alpha \to 1$ is delta, $\alpha \to 0$ is wipe). The S-NIAH case study is the best part of the paper — it doesn't just report that Gated DeltaNet is better, it *dissects why* by showing DeltaNet and Mamba2 each failing a complementary half, which is far more convincing than a leaderboard. And the online-learning derivation (delta rule = test-time SGD, gating = weight decay) elevates the contribution from "a trick that works" to "a principled point in a well-understood design space." The chunkwise algorithm is real engineering: keeping the gate without losing the matmuls is the difference between a nice equation and a trainable layer.

**What's weak or under-shown.** The absolute margins over Mamba2 on the aggregate benchmarks are small (0.4 average accuracy, 0.14 Wikitext perplexity) — the *qualitative* wins on retention and memorization are more compelling than the *quantitative* wins on general LM. Scale is a real limitation: 1.3B is small by 2025 standards, and gate-versus-write dynamics could shift at 7B+. The write strength is deliberately confined to $\beta_t \in (0,1)$, which sidesteps the negative-eigenvalue state-tracking results (Grazzi et al., 2024; DeltaProduct) that the related-work section itself flags as promising — a conservative choice the paper acknowledges but doesn't explore. And the online-learning framing, while illuminating, is partly *post-hoc*: Table 1 is a beautiful organizing device, but the objectives are reverse-engineered from update rules people already had, so it explains more than it predicts.

**What ablation or baseline is missing.** There's no ablation isolating $\alpha_t$ *alone* on top of DeltaNet versus the full block — the naive-delta-rule ablation changes the update but is entangled with the block design. A gate-only-vs-write-only sweep at fixed everything-else would make the "complementary" claim airtight beyond S-NIAH. And RWKV-7, flagged as concurrent work with a strictly more general diagonal-plus-low-rank transition $S_t = S_{t-1}(\operatorname{diag}(d_t) - a_t b_t^\top) + v_t k_t^\top$, is discussed but not benchmarked head-to-head — the natural question of "is the extra generality worth it?" is left open.

**What would change my mind.** If a controlled 7B run showed the Gated-DeltaNet-over-Mamba2 gap *shrinking* rather than holding, I'd downgrade this from "a better default linear layer" to "a small-scale curiosity" — the whole pitch rests on the mechanism advantage persisting with scale. Conversely, the fact that both Kimi Linear and Qwen3-Next shipped Gated-DeltaNet-style layers into production stacks is the strongest evidence *for* durability that exists outside this paper.

## What I'd build with this

These are my extrapolations, not the paper's claims.

1. **A per-head gate schedule.** The gate $\alpha_t$ is scalar per head; I'd try letting *different heads* specialize into "archivist" heads ($\alpha \to 1$, near-DeltaNet) and "scratchpad" heads ($\alpha$ small, fast-forgetting), and measure whether the split emerges on its own or needs an auxiliary loss to encourage it.
2. **Push $\beta_t$ into $(0,2)$.** The paper stays in $(0,1)$; combining the gate with negative-eigenvalue writes (DeltaProduct-style Householder products) would test whether adaptive forgetting and state-tracking expressivity compose, or interfere.
3. **A retrieval-triggered gate.** Since S-NIAH shows the gate is exactly what filtering needs, I'd condition $\alpha_t$ on a cheap "is this a document boundary?" signal (e.g., a learned segment token) to get sharp context resets instead of smooth decay.
4. **Chunk-size autotuning by sequence length.** The $C \times C$ inverse cost versus intra-chunk parallelism trade-off is static in the paper; a kernel that picks $C$ per layer and per training length could recover some of the throughput gap to Mamba2.
5. **The online-learning objective as a knob.** If the recurrence is test-time SGD, then swapping in a momentum term or a second-order step (a "test-time Adam") is a concrete, parallelism-preserving direction to explore before jumping to the nonlinear TTT/Titans objectives that break chunkwise training.

## References

- **Paper.** Songlin Yang, Jan Kautz, Ali Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR 2025. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
- **Code.** NVIDIA GatedDeltaNet — [github.com/NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet); kernels in Flash Linear Attention — [github.com/fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- **Prior work it builds on.** Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* (NeurIPS 2024, DeltaNet); Dao & Gu, *Transformers are SSMs* (Mamba2, 2024).

Sibling posts on this blog worth reading next:

- [Kimi Linear: An Expressive, Efficient Attention Architecture](/blog/paper-reading/large-language-model/kimi-linear) — a production stack built on a Gated-DeltaNet-style rule (Kimi Delta Attention).
- [Qwen3-Next: Hybrid Attention and an 80B Model That Thinks With 3B](/blog/paper-reading/large-language-model/qwen3-next-hybrid-attention-ultra-sparse-moe) — ships Gated DeltaNet layers in a hybrid at scale.
- [MiniMax-01: Lightning Attention, the 7:1 Hybrid, and a Million-Token Context](/blog/paper-reading/large-language-model/minimax-01-lightning-attention-hybrid-moe) — a sibling take on linear attention at production scale.
- [Nested Learning: The Illusion of Deep Learning Architecture](/blog/paper-reading/large-language-model/nested-learning-the-illusion-of-deep-learning-architecture) — the test-time / online-learning framing that this paper's Table 1 lives inside.

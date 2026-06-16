---
title: "Pruning LLMs and transformers: SparseGPT, Wanda, and structured depth/width cuts"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why the classic prune-then-fine-tune loop breaks for billion-parameter models, and how SparseGPT, Wanda, and structured head/layer/width cuts prune a 7B LLM in one shot, post-training, with calibration data instead of gradients."
tags:
  [
    "edge-ai",
    "model-optimization",
    "pruning",
    "sparsity",
    "sparsegpt",
    "wanda",
    "llm",
    "transformers",
    "inference",
    "efficient-ml",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/pruning-llms-and-transformers-1.png"
---

For a decade, pruning a neural network meant a loop you could draw on a napkin: train the model, rank the weights by some saliency score, delete the least important ones, then fine-tune the survivors back to health, and repeat until the accuracy starts to slip. That loop is the heart of every classic pruning paper, from Optimal Brain Damage in 1990 to the Lottery Ticket Hypothesis in 2019. It works because the fine-tuning step is cheap relative to the win: a few epochs on ImageNet to recover a ResNet you sparsified is an afternoon on one GPU.

Now try to run that loop on a 70-billion-parameter LLM. Fine-tuning a model that size is not an afternoon; it is a multi-node cluster job that costs more than most teams' entire compute budget, requires the full training dataset (which you usually do not have), and risks catastrophic forgetting of capabilities the base model spent trillions of tokens acquiring. The classic prune-then-fine-tune loop does not just get slower at LLM scale — it becomes structurally inapplicable. You cannot iterate a loop whose inner step you cannot afford to run even once.

This is the wall that the 2023 generation of LLM pruning methods tore down. **SparseGPT** and **Wanda** prune a 7B, 13B, or 70B model in a *single shot, post-training*, using nothing but a few hundred sentences of calibration data and a few hours on one GPU — no gradient descent, no fine-tuning, no training set. They get a 7B model to 50% sparsity with a perplexity loss you can almost ignore. And for the cases where you want a genuinely smaller *dense* model that any runtime speeds up, **structured** methods like LLM-Pruner and Sheared LLaMA cut whole attention heads, FFN neurons, layers, and hidden width, then claw quality back with a comparatively tiny dose of LoRA fine-tuning or continued pretraining. The figure below is the map of the territory: the four axes inside a transformer block you can actually attack.

![Tree diagram showing the four prunable axes of a transformer block: attention heads, feed-forward neurons, whole layers, and hidden width, each with its own granularity and speedup story](/imgs/blogs/pruning-llms-and-transformers-1.png)

By the end of this post you will be able to: explain *why* retraining-free pruning had to be invented for LLMs and what makes it work; derive the Optimal-Brain-Surgeon update that SparseGPT borrows from GPTQ and understand why pushing error into surviving weights is the whole game; compute a Wanda saliency score in about ten lines of code and explain why a metric with no Hessian inverse comes within a hair of SparseGPT; run both official repos and combine sparsity with int4 quantization; reason about structured head/layer/width pruning and when it beats unstructured; and — the part that matters most for shipping — decide honestly whether to prune an LLM at all, given that quantization is usually the bigger, easier win. This post sits on the **pruning/sparsity lever** of the four-lever frame from [the model-compression taxonomy](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), it assumes the dense-network pruning basics from [the pruning fundamentals post](/blog/machine-learning/edge-ai/pruning-fundamentals), and it feeds the final [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).

One honesty note up front, the same one I make in every post in this series. The byte counts, the saliency math, the OBS derivation, and the bits-per-weight arithmetic are derived from first principles and are exact. The headline perplexity deltas and tokens-per-second numbers are quoted as approximate from the SparseGPT and Wanda papers and widely replicated community benchmarks; pruning results are even more hardware- and runtime-dependent than quantization results, so treat any absolute number as an order-of-magnitude guide and trust the *ratios* (SparseGPT vs Wanda vs magnitude; unstructured vs 2:4 vs structured). The one rule that never bends: never trust a sparsity speedup you did not measure on your own target.

## 1. Why retraining-free pruning had to be invented

To see why LLM pruning is a genuinely different problem, you have to be precise about what the classic loop assumes and which of those assumptions break at scale.

The classic prune-then-fine-tune loop assumes three things. First, that you have the training data and the loss function — you re-run training, so you need the full pipeline. Second, that fine-tuning is cheap relative to the model — a few epochs to recover. Third, that you can iterate — prune a little, recover, prune more, recover, because gradual sparsification with recovery between steps reaches far higher sparsity at a given quality than one aggressive cut. Every one of those assumptions is true for a ResNet-50 and false for a 70B LLM.

The data assumption breaks because the pretraining corpus for a frontier model is often proprietary, enormous (trillions of tokens), and not something a downstream team possesses. You have the weights from the model hub; you do not have the 15-trillion-token mixture they were trained on. The cost assumption breaks because "a few epochs" of a 70B model is a cluster job measured in GPU-months and dollars with five or six digits. The iteration assumption breaks for both of the above reasons compounded: you cannot afford one recovery step, let alone ten.

So the LLM pruning problem has to be restated. Given a trained dense model, a small *calibration* set (a few hundred unlabeled sentences, which anyone can scrape), and a single GPU for a few hours, produce a sparse model whose outputs match the dense model's outputs as closely as possible — *without any gradient-based training*. That word "match the outputs" is the key. We are no longer trying to minimize the task loss; we have given up on touching the loss because we cannot run backprop on the whole model affordably. Instead we minimize, *layer by layer*, the difference between the dense layer's output and the sparse layer's output on the calibration data. This is called **layer-wise reconstruction**, and it is the central idea that makes the whole thing tractable.

Here is the reframing that unlocks everything. A transformer is a stack of linear layers (the Q, K, V, O projections in attention, and the up and down projections in the FFN) glued together with cheap nonlinearities. If you can make each linear layer's output *on the calibration data* nearly identical before and after pruning, then by composition the whole network's output is nearly identical too — errors do not get a chance to compound because each layer is individually corrected. And the per-layer problem is small and convex-ish: for a single weight matrix $W$ and a batch of calibration inputs $X$, find a sparse $\hat{W}$ that minimizes $\| W X - \hat{W} X \|_2^2$. That is a tractable least-squares problem you can solve one layer at a time, in memory, on one GPU, with no labels and no backprop. SparseGPT and Wanda are two different answers to exactly this per-layer question.

There is a beautiful payoff to this framing that is worth naming explicitly: it is *the same framing* that weight-only quantization uses. GPTQ rounds weights to int4 by minimizing the layer-wise output error; SparseGPT zeros weights by minimizing the layer-wise output error. They are siblings — the same OBS machinery, the same calibration loop, the same Hessian — applied to two different perturbations of the weights (rounding vs zeroing). That is not a coincidence, and we will exploit it later when we compose sparsity with quantization. If you have read the [weight-only quantization post](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq), the next section will feel like coming home.

It is worth being explicit about *why* the layer-wise decomposition does not blow up, because the obvious worry is that small per-layer errors might compound exponentially through a 32-layer stack. The reason they do not is twofold. First, each layer is corrected to match the dense layer's output *on the same calibration inputs* the dense model produced — so the input distribution each layer sees during pruning is the dense model's true activation distribution, not a drifted one. Second, and more subtly, you can prune the layers *sequentially*, feeding each layer the *already-pruned* previous layers' outputs as its calibration input. This way layer $k$'s reconstruction target accounts for the error that layers $1$ through $k-1$ already introduced, so layer $k$ compensates not just for its own pruning but partially for the accumulated upstream drift. The error does grow as you go deeper, but it grows roughly additively (each layer adds a small, corrected residual) rather than multiplicatively, and the OBS compensation at each layer actively pushes against the accumulation. This is the same reason layer-wise quantization with sequential calibration outperforms quantizing every layer independently against the *dense* inputs — the sequential variant is self-correcting. In the official SparseGPT and Wanda harnesses this is exactly how it is implemented: a single forward pass that processes one transformer block at a time, capturing each block's *actual* (post-pruning) inputs to drive the next block's statistics.

A natural objection at this point: if we are only matching the *layer outputs on calibration data* and never touching the task loss, are we not optimizing the wrong objective? Could a pruned model match every layer's outputs on 128 calibration sentences yet generate garbage on real prompts? In principle yes; in practice the layer-wise objective is a remarkably good proxy, for the same reason calibration-based quantization works: the calibration set, though tiny, samples the activation manifold densely enough that matching outputs there generalizes. The activations a transformer produces are far lower-dimensional in practice than their nominal width suggests (they live near a low-rank manifold), so a few hundred sentences pin down the directions that matter. The empirical evidence is the perplexity numbers themselves — a model pruned to match layer outputs on C4 calibration data has near-dense perplexity on the *held-out* WikiText-2 test set, which it never saw. The proxy holds. Where it starts to fray is at extreme sparsity or with a badly mismatched calibration distribution, both of which we stress-test later.

## 2. What to prune in a transformer, and what each axis costs

Before the methods, be concrete about *what* you are removing, because the granularity you choose decides whether you get a real speedup or just a sparser file that runs at the same speed. A transformer block has four prunable axes, shown in figure 1 above, and they behave very differently.

**Attention heads.** Multi-head attention splits the hidden dimension into $h$ heads, each with its own slice of the Q, K, V projections and a slice of the output projection O. A foundational result here is Michel, Levy & Neubig's 2019 paper with the memorable title "Are Sixteen Heads Really Better Than One?" — they showed that at *test time* you can ablate a large fraction of the heads in a trained transformer (often the majority in some layers) with little or no loss in performance. Heads are massively redundant: many learn overlapping or near-useless attention patterns. Removing a head is **structured** — it deletes whole columns/rows of the projection matrices — so it shrinks the dense matmul and any runtime runs it faster. This makes heads a prime structured pruning target.

**FFN neurons.** The feed-forward network is two linear layers with a nonlinearity between them: an up-projection from $d_{\text{model}}$ to $d_{\text{ff}}$ (usually $4 \times$ larger, or with gated GLU variants $\approx \frac{8}{3}\times$), then a down-projection back. The FFN is where most of a transformer's parameters live — roughly two-thirds of the non-embedding weights. An FFN "neuron" is one of the $d_{\text{ff}}$ intermediate dimensions; pruning it removes one column of the up-projection and one row of the down-projection. This is also structured and also shrinks the dense matmul.

To make "two-thirds of the parameters live in the FFN" concrete, count them for one block. Attention has four projection matrices, each $d_{\text{model}} \times d_{\text{model}}$ (ignoring grouped-query attention, which shrinks K and V), for $4\,d_{\text{model}}^2$ parameters. A standard FFN has an up-projection $d_{\text{model}} \times d_{\text{ff}}$ and a down-projection $d_{\text{ff}} \times d_{\text{model}}$, for $2\,d_{\text{model}} d_{\text{ff}}$ parameters; with $d_{\text{ff}} = 4 d_{\text{model}}$ that is $8\,d_{\text{model}}^2$. So the FFN holds $\frac{8}{4+8} = \frac{2}{3}$ of the block's matmul parameters — exactly the two-thirds claim. (Gated GLU FFNs use *three* matrices — gate, up, down — each $\approx d_{\text{model}} \times \frac{8}{3} d_{\text{model}}$, which lands at roughly the same total.) This arithmetic is why FFN-neuron pruning is so attractive: the FFN is where the parameters — and therefore the memory-bound decode bytes — actually are. Cutting 25% of FFN neurons removes roughly $\frac{2}{3} \times 25\% \approx 17\%$ of all block parameters, a real dent.

**Whole layers (depth).** You can delete an entire transformer block. This is the bluntest, most aggressive structured cut, and it gives the largest, most reliable speedup because it removes the layer's compute *and* its sequential dependency. The catch is quality: dropping layers tends to hurt more per-parameter-removed than the finer-grained cuts, and which layers are droppable is non-obvious (often the middle-to-late layers are more redundant than the early ones, but this is model-specific). Depth pruning is the core move in some structured methods and is the reason you see "depth-pruned" small variants of big models. A useful heuristic from the depth-pruning literature: measure each layer's *block influence* — how much it changes the residual stream, i.e. the cosine distance between a block's input and output hidden states averaged over calibration data — and drop the layers that change the residual the least. A layer whose output is nearly identical to its input is, almost by definition, doing little, and removing it perturbs the network the least.

**Hidden width.** You can shrink $d_{\text{model}}$ itself, slimming *every* matrix in the model simultaneously. This is the most invasive structured change because it touches residual streams, layer norms, and every projection at once, so it generally demands the most recovery training. But it produces a clean, smaller dense model with proportionally smaller everything.

Cutting across all four is the **structured vs unstructured** distinction, which is the single most important thing to internalize in this whole post:

- **Unstructured** pruning zeros *individual weights* anywhere in a matrix, with no pattern. It achieves the best quality at a given sparsity (you delete exactly the least-useful weights), but the resulting matrix is just a dense matrix full of zeros — and a GPU's dense matmul does not skip zeros. **Unstructured sparsity gives you a smaller file but no speedup on commodity hardware** without specialized sparse kernels, and even those struggle to beat dense below ~70% sparsity.
- **Semi-structured (N:M)** enforces a fixed pattern, most commonly **2:4** — exactly two of every four contiguous weights are zero. This is the sweet spot NVIDIA's Ampere-and-later Sparse Tensor Cores were built for: they store the 2:4 matrix in a compressed form and run it at roughly **2× the dense rate**. You pay a little quality versus unstructured at the same 50% density, and you get a real, guaranteed speedup. We cover the hardware in depth in [the N:M sparsity and Sparse Tensor Cores post](/blog/machine-learning/edge-ai/n-m-sparsity-and-sparse-tensor-cores).
- **Structured** pruning removes whole units — heads, neurons, layers, width — producing a *genuinely smaller dense model*. Any runtime, any hardware, runs it faster with zero special kernels. The cost is quality: you are deleting in coarse chunks, so you usually need recovery training to get back to par.

Hold that hierarchy in your head: unstructured = best quality, no speedup; 2:4 = good quality, 2× speedup on the right GPU; structured = needs recovery, but a real dense speedup everywhere. Almost every decision in the rest of this post is a position on this spectrum.

## 3. The science: layer-wise pruning as Optimal Brain Surgeon

Now the math, because SparseGPT's whole claim to fame is that it solves the layer-wise reconstruction problem *optimally* (to second order) rather than greedily, and to see why that matters you have to derive the update.

Set up the per-layer problem precisely. A linear layer computes $Y = W X$, where $W$ is the weight matrix ($d_{\text{out}} \times d_{\text{in}}$) and $X$ is a batch of $N$ calibration inputs ($d_{\text{in}} \times N$). We want to set some entries of $W$ to zero — pick a binary mask $M$ — and then adjust the *surviving* weights to a new value $\hat{W}$ so that the output barely changes. Formally, minimize the reconstruction error

$$
\min_{\hat{W}} \; \| W X - \hat{W} X \|_2^2 \quad \text{subject to } \hat{W} \text{ respecting mask } M.
$$

This decomposes row by row (each output dimension is independent), so consider a single row, a weight vector $w \in \mathbb{R}^{d_{\text{in}}}$. The error from perturbing $w$ by $\delta w$ is, to second order,

$$
\Delta E = \frac{1}{2}\, \delta w^\top H\, \delta w, \qquad H = X X^\top,
$$

where $H$ is the (layer-local) **Hessian** of the squared error — and because the error is quadratic in $w$, this second-order expansion is *exact*, not an approximation. The matrix $H = X X^\top$ is just the input correlation matrix, accumulated over the calibration set. This is the entire reason calibration data is needed: $H$ encodes which input directions are "hot" (high variance, large activations) so that pruning a weight that multiplies a hot direction is penalized more than pruning one that multiplies a quiet direction.

Now the classic **Optimal Brain Surgeon** (OBS) result, from Hassibi & Stork 1993. Suppose we decide to zero out weight $q$ (set $w_q \to 0$). We want the perturbation $\delta w$ that achieves $\delta w_q = -w_q$ (it kills weight $q$) while minimizing $\Delta E$, and we are free to adjust *all the other* weights to compensate. This is a constrained quadratic minimization; solving it with a Lagrange multiplier gives the OBS update:

$$
\delta w = -\frac{w_q}{[H^{-1}]_{qq}}\, H^{-1} e_q, \qquad \Delta E_q = \frac{1}{2}\,\frac{w_q^2}{[H^{-1}]_{qq}}.
$$

Read those two equations slowly, because they *are* the method. The second one, $\Delta E_q = \frac{1}{2} w_q^2 / [H^{-1}]_{qq}$, is the **saliency**: the true second-order cost of deleting weight $q$. Notice it is *not* just $w_q^2$ (that would be magnitude pruning) — it is divided by the corresponding diagonal of the *inverse* Hessian, which corrects for how the input correlations make some weights cheaper to remove than their magnitude suggests. The first equation, $\delta w = -\frac{w_q}{[H^{-1}]_{qq}} H^{-1} e_q$, is the **compensation**: after you zero weight $q$, you nudge *every other surviving weight in the row* along the direction $H^{-1} e_q$, absorbing the error that deleting $q$ would otherwise cause. That compensation step is the magic. Magnitude pruning deletes a weight and leaves a hole; OBS deletes a weight and *heals the hole* by redistributing its job to its neighbors, weighted by how the inputs correlate.

The figure below traces SparseGPT's loop, which is OBS applied column by column with the GPTQ trick that makes it affordable.

![Graph diagram of the SparseGPT layer-wise loop building the inverse Hessian from calibration activations, picking a pruning mask by saliency, zeroing weights, and applying an Optimal Brain Surgeon correction to the surviving weights](/imgs/blogs/pruning-llms-and-transformers-2.png)

One more piece of intuition before the algorithm, because the compensation step is where people's understanding usually breaks. Picture the row's weights as a committee that collectively produces one output number for each calibration input. When you fire one committee member (zero a weight), the committee's output drifts. Plain magnitude pruning fires the member and walks away, accepting the drift. OBS fires the member and then *re-tasks the remaining members* to cover the fired one's responsibilities — and it knows exactly how to re-task them, because $H^{-1}$ encodes which members' contributions correlate with the fired one's. A member whose inputs are correlated with the fired member's inputs is well-positioned to absorb the lost work, so it gets nudged more. This is why OBS-pruned rows produce nearly the same outputs with half the members: the survivors took over the dead weights' jobs. It is the analytic, closed-form version of what fine-tuning would have done by gradient descent, and it is the single idea that makes one-shot pruning of a 70B model possible.

The reason OBS was abandoned for decades is the $H^{-1}$. For a layer with $d_{\text{in}}$ inputs, $H$ is $d_{\text{in}} \times d_{\text{in}}$; inverting it is $O(d_{\text{in}}^3)$, and re-doing it after every single weight deletion (because the Hessian changes once a weight is removed) is hopelessly expensive at LLM scale, where $d_{\text{in}}$ is thousands. SparseGPT's contribution (Frantar & Alistarh, 2023) is a clever, **GPTQ-style** restructuring that makes the full OBS treatment of a whole layer cost roughly the same as a single Hessian inverse rather than one per weight:

1. Compute $H = X X^\top$ once from the calibration data and dampen its diagonal slightly for stability ($H \leftarrow H + \lambda I$). Invert it once.
2. Process the matrix **column by column, left to right.** Within each column, use the OBS saliency $\frac{w_q^2}{[H^{-1}]_{qq}}$ to pick which weights in that column to zero (to hit the target sparsity, e.g. 50%).
3. After deciding a column's mask, apply the OBS compensation — but crucially, *only push the error forward* into the columns not yet processed, exactly the way GPTQ pushes rounding error forward. This makes the bookkeeping a single sweep with incremental Cholesky updates of $H^{-1}$ rather than a fresh inverse per weight.
4. The result: each layer's surviving weights are adjusted so the layer output on the calibration data is nearly unchanged — *with no gradient step at all.*

That last point is the headline. SparseGPT prunes a 7B model to 50% in a couple of hours on a single GPU and the perplexity barely moves, because the OBS compensation does the work that fine-tuning used to do — it just does it in closed form, per layer, from the Hessian, instead of via backprop over the whole network. It is fine-tuning's job, done analytically.

#### Worked example: the saliency of one weight

Make the OBS saliency concrete with two weights in a 2-input toy layer. Say the calibration data gives a Hessian $H = \begin{bmatrix} 4 & 1 \\ 1 & 1 \end{bmatrix}$ (input 0 is "hot," high variance; input 1 is quiet, and they are mildly correlated). Its inverse is $H^{-1} = \frac{1}{3}\begin{bmatrix} 1 & -1 \\ -1 & 4 \end{bmatrix}$, so $[H^{-1}]_{00} = \frac{1}{3}$ and $[H^{-1}]_{11} = \frac{4}{3}$. Now suppose both weights have the *same* magnitude, $w_0 = w_1 = 0.5$. Magnitude pruning would call them equally deletable. But OBS says the cost of deleting weight 0 is $\Delta E_0 = \frac{1}{2}\frac{0.25}{1/3} = 0.375$, while deleting weight 1 costs $\Delta E_1 = \frac{1}{2}\frac{0.25}{4/3} = 0.094$. Weight 1 is **four times cheaper to remove** despite identical magnitude, because it multiplies a quiet input direction — exactly the correction magnitude pruning misses. This is, in miniature, why SparseGPT keeps quality at 50% where magnitude pruning collapses.

## 4. Wanda: SparseGPT's accuracy at a fraction of the cost

SparseGPT is optimal-ish but it still computes and inverts a Hessian and runs an OBS sweep — it is not trivial to implement and it is not instant. In late 2023, Sun, Liu, Zhuo & Kolter asked a sharp question: how much of SparseGPT's win comes from the expensive $H^{-1}$ machinery, and how much comes from simply *using the activations* to score weights? Their answer, **Wanda** (Weights AND Activations), is almost embarrassingly simple and almost as good.

Wanda's saliency for a weight $W_{ij}$ (output $i$, input $j$) is

$$
S_{ij} = |W_{ij}| \cdot \| X_j \|_2,
$$

where $\| X_j \|_2$ is the $\ell_2$ norm of the $j$-th input feature *across the calibration batch* — that is, how large that input channel's activations tend to be. The weight's magnitude times the typical size of the activation it gets multiplied by. That is the whole score. No Hessian, no inverse, no compensation, no weight update of any kind: you compute the per-channel activation norms in one forward pass over the calibration set, multiply elementwise by $|W|$, and prune the lowest-scoring weights. The figure below contrasts this with plain magnitude pruning.

![Before and after diagram contrasting magnitude pruning which scores weights by absolute value alone against the Wanda metric which multiplies weight magnitude by the input activation norm so small weights on high-activation channels survive](/imgs/blogs/pruning-llms-and-transformers-3.png)

Why does so cheap a metric work so well? Two reasons, and both are worth understanding because they tell you *when* it works.

First, it is a first-order approximation of the right thing. The quantity that actually matters for output error is how much zeroing $W_{ij}$ perturbs the output $Y_i = \sum_j W_{ij} X_j$. Zeroing $W_{ij}$ changes the output by $-W_{ij} X_j$, whose typical magnitude over the batch is $|W_{ij}| \cdot \|X_j\|_2$ — *exactly the Wanda score.* So Wanda is directly estimating the expected output perturbation from deleting each weight. SparseGPT's OBS does the same but to second order *and* with cross-weight compensation via $H^{-1}$; Wanda drops the off-diagonal correlations (treats $H$ as diagonal, $H \approx \text{diag}(\|X_j\|_2^2)$) and skips compensation entirely. The remarkable empirical finding of the Wanda paper is that for LLMs at 50% sparsity, those approximations cost almost nothing — the diagonal of $H$ carries most of the signal because the input correlations, while present, are not the dominant term.

Second — and this is Wanda's most important and underappreciated design choice — it compares weights **per output row**, not across the whole matrix. The score $S_{ij}$ is used to rank weights *within each output neuron $i$*, removing the lowest 50% in each row independently. This matters because LLM activations have huge **outlier channels**: a small number of input features carry enormously larger activations than the rest (this is the same outlier phenomenon that wrecks naive activation quantization, covered in [the activation-quantization post](/blog/machine-learning/edge-ai/llm-quantization-activations-smoothquant-kv-cache)). A weight feeding an outlier channel gets a giant $\|X_j\|_2$, so even a *small-magnitude* weight on a hot channel scores high and is kept — which is precisely the weight magnitude pruning would wrongly throw away. The per-row comparison ensures every output neuron retains its most-salient inputs rather than letting a few outlier-channel rows dominate the global ranking and starve other rows.

The practical upshot: Wanda gets within a fraction of a perplexity point of SparseGPT at 50% sparsity, runs in a single forward pass with no backward pass and no matrix inverse, and is so simple it is genuinely about ten lines of code. SparseGPT pulls slightly ahead at the harder regimes (2:4, or pushing past 60% sparsity) because there the compensation and the second-order term start to earn their keep. For most one-shot 50% pruning, Wanda is the pragmatic default and SparseGPT is the option you reach for when you want every last fraction of a point.

#### Worked example: a Wanda score by hand

One output neuron, four input weights, $W = [0.10,\ 0.80,\ -0.05,\ 0.30]$. The calibration activation norms per channel come out $\|X\| = [12.0,\ 0.5,\ 9.0,\ 1.0]$ — channel 0 and channel 2 are outlier channels (large activations), channels 1 and 3 are quiet. Magnitude pruning ranks by $|W| = [0.10, 0.80, 0.05, 0.30]$ and, asked to drop the two smallest, deletes weights 0 and 2 (the $0.10$ and the $0.05$). Wanda computes $S = |W| \cdot \|X\| = [1.20,\ 0.40,\ 0.45,\ 0.30]$ and drops the two smallest *of those* — weights 1 and 3 (the $0.40$ and $0.30$). Completely different answer. Wanda keeps weight 0 ($S=1.20$, the most salient by far) because even though it is small in magnitude it multiplies the hottest channel, and it keeps weight 2 over weight 1 because the outlier channel makes it matter. Magnitude pruning would have deleted the single most output-relevant weight in the neuron. That divergence, multiplied across billions of weights, is the gap between "perplexity barely moves" and "the model is now babble."

## 5. Running SparseGPT and Wanda in practice

Both methods ship as clean open-source repos, and Wanda's metric is short enough to write from scratch, which is the best way to internalize it. Start there.

Here is the Wanda saliency and a per-row 50% mask, the core of the method in plain PyTorch:

```python
import torch

@torch.no_grad()
def wanda_mask(weight, act_norm, sparsity=0.5):
    # weight:   (d_out, d_in)  the layer's weight matrix
    # act_norm: (d_in,)        L2 norm of each input channel over calibration
    # Wanda score S_ij = |W_ij| * ||X_j||, compared PER OUTPUT ROW.
    score = weight.abs() * act_norm.unsqueeze(0)        # broadcast over rows
    n_prune = int(weight.shape[1] * sparsity)
    # threshold within each row: keep the largest, zero the smallest n_prune
    thresh = torch.kthvalue(score, n_prune, dim=1, keepdim=True).values
    mask = score > thresh                                # True = keep
    return mask

@torch.no_grad()
def collect_act_norms(layer, calib_loader, device):
    # one forward pass: accumulate sum of squares of each input channel
    sq = torch.zeros(layer.in_features, device=device)
    n = 0
    def hook(_m, inp, _out):
        x = inp[0].reshape(-1, layer.in_features)        # (tokens, d_in)
        sq.add_(x.pow(2).sum(dim=0)); 
        nonlocal n; n += x.shape[0]
    h = layer.register_forward_hook(hook)
    for batch in calib_loader:
        layer(batch.to(device))                          # drives the hook
    h.remove()
    return (sq).sqrt()                                   # ||X_j||_2 per channel
```

That is the entire idea: a forward hook to gather per-channel activation norms, then an elementwise score and a per-row threshold. Apply `mask` by multiplying it into the weight (`layer.weight.mul_(mask)`) and you have a 50%-sparse layer with no weight update. The 2:4 variant is the same score with a different selection rule — instead of "keep the largest fraction per row," you keep exactly 2 of every 4 contiguous weights, choosing the 2 with the highest score in each group of 4. Here is that selection in code, which is the only part that differs from unstructured:

```python
@torch.no_grad()
def wanda_2of4_mask(weight, act_norm):
    # Keep exactly 2 of every 4 contiguous columns, per row, by Wanda score.
    d_out, d_in = weight.shape
    assert d_in % 4 == 0, "2:4 needs in_features divisible by 4"
    score = weight.abs() * act_norm.unsqueeze(0)        # (d_out, d_in)
    g = score.reshape(d_out, d_in // 4, 4)              # groups of 4
    # within each group of 4, find the 2 LOWEST scores and zero them
    drop = g.argsort(dim=-1)[..., :2]                   # indices of 2 smallest
    mask = torch.ones_like(g, dtype=torch.bool)
    mask.scatter_(-1, drop, False)                      # False = prune
    return mask.reshape(d_out, d_in)
```

The 2:4 constraint is why this pattern costs a little more quality than unstructured at the same 50% density: in a group of four weights where three are highly salient and one is not, unstructured pruning keeps all three, but 2:4 is *forced* to drop one of the salient three. The rigid pattern occasionally throws away a weight it would rather keep. That is the price of a format the Sparse Tensor Cores can run natively — a quality tax in exchange for a guaranteed hardware speedup, which is almost always worth it when you have the hardware.

SparseGPT's per-layer sweep is more involved because it carries the Hessian and the OBS compensation, but the loop skeleton makes the GPTQ kinship obvious:

```python
@torch.no_grad()
def sparsegpt_layer(W, H_inv, sparsity=0.5, blocksize=128):
    # W:     (d_out, d_in) weights for one linear layer
    # H_inv: (d_in, d_in)  inverse of dampened Hessian X X^T + lambda*I
    d_out, d_in = W.shape
    for c in range(0, d_in, blocksize):                # process column blocks
        cols = slice(c, c + blocksize)
        Wb = W[:, cols].clone()
        # OBS saliency for each weight in the block: w^2 / [H_inv]_jj
        diag = torch.diag(H_inv)[cols].unsqueeze(0)     # (1, blocksize)
        saliency = Wb.pow(2) / diag
        # pick the lowest-saliency weights to zero (to hit target sparsity)
        thresh = saliency.flatten().kthvalue(
            int(saliency.numel() * sparsity)).values
        mask = saliency > thresh                         # True = keep
        Wb = Wb * mask
        # OBS compensation: push the pruning error into LATER columns only
        err = (W[:, cols] - Wb) @ H_inv[cols, cols]      # local correction
        W[:, c + blocksize:] -= err @ H_inv[cols, c + blocksize:]
        W[:, cols] = Wb
    return W
```

The line that *is* SparseGPT is the compensation: `W[:, c+blocksize:] -= err @ H_inv[...]` pushes the error from the just-pruned block forward into the not-yet-processed columns, exactly as GPTQ pushes rounding error forward. Everything else is bookkeeping. (The production code uses an incremental Cholesky factor of $H^{-1}$ rather than the dense inverse for numerical stability and speed, but the math is what you see.)

In real use you run the official repos, which handle the layer-by-layer sweep, the calibration data loading, and (for SparseGPT) the Hessian and OBS bookkeeping. The Wanda repo wraps both methods behind one CLI:

```bash
# Wanda (no weight update) and SparseGPT share one harness.
# 50% unstructured on a 7B; calibration = 128 sequences from C4.
python main.py \
  --model meta-llama/Llama-2-7b-hf \
  --prune_method wanda \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --nsamples 128 \
  --save out/llama7b_wanda_0.5

# Same harness, SparseGPT, 2:4 semi-structured (for Sparse Tensor Cores).
python main.py \
  --model meta-llama/Llama-2-7b-hf \
  --prune_method sparsegpt \
  --sparsity_type 2:4 \
  --nsamples 128 \
  --save out/llama7b_sparsegpt_2of4
```

A few flags carry the real decisions. `--nsamples` is the calibration set size; 128 sequences of length 2048 is the community default and is enough — Wanda and SparseGPT are remarkably insensitive to calibration size (a few dozen samples already capture the activation statistics), which is itself evidence that the per-channel norms and the Hessian are dominated by a stable, low-variance signal. `--sparsity_type` is the whole structured/unstructured decision: `unstructured` for the best perplexity but no speedup, `2:4` for the Sparse-Tensor-Core speedup. `--sparsity_ratio` only applies to unstructured (2:4 is fixed at 50% density by definition).

To check the damage, evaluate perplexity on a held-out set — this is the standard sanity metric, and the official repos report it the same way:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def perplexity(model, enc, seqlen=2048, device="cuda"):
    ids = enc.input_ids.to(device)
    n = ids.numel() // seqlen
    nlls = []
    for i in range(n):
        chunk = ids[:, i * seqlen:(i + 1) * seqlen]
        with torch.no_grad():
            out = model(chunk, labels=chunk)            # HF returns mean NLL
        nlls.append(out.loss * seqlen)                   # un-average per token
    return torch.exp(torch.stack(nlls).sum() / (n * seqlen)).item()

tok = AutoTokenizer.from_pretrained("out/llama7b_wanda_0.5")
model = AutoModelForCausalLM.from_pretrained(
    "out/llama7b_wanda_0.5", torch_dtype=torch.float16, device_map="cuda")
# evaluate on the WikiText-2 test split, the canonical pruning benchmark
import datasets
wt2 = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
enc = tok("\n\n".join(wt2["text"]), return_tensors="pt")
print("WikiText-2 PPL:", round(perplexity(model, enc), 2))
```

The single most important practical point: the output of these commands is a *checkpoint with zeros in it*. On its own, it is not faster — it is the same dense matmul with a sparser matrix. To get a speedup you either (a) exported it as 2:4 and run on a GPU with Sparse Tensor Cores through a kernel that exploits the format, or (b) you run unstructured through a dedicated sparse kernel that can beat dense at your sparsity level (hard below ~70%). Why unstructured LLM sparsity still needs special kernels to speed up decode comes straight from the roofline: decode is memory-bound, and a dense kernel reading a matrix full of zeros still reads all the bytes. We work that out fully in [the roofline post](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), and it is the reason the next section spends so much time on the speed *reality* as opposed to the file-size fantasy.

One more practical note on calibration. The `--nsamples 128` default is a sweet spot, not a magic number — both methods plateau quickly, and you can verify this yourself by sweeping it (32, 64, 128, 256) and watching perplexity barely move. The *content* of the calibration set matters more than its size: the community default pulls sequences from C4 (a broad web corpus) precisely because it is distributionally neutral, which is the right choice when you do not know the deployment distribution. If you *do* know it — say you are pruning a model that will only ever summarize legal documents — calibrating on in-domain text can recover a fraction of a point of task-relevant quality, because the activation norms (Wanda) and the Hessian (SparseGPT) then reflect the inputs the model will actually see. This is a small effect at 50% but grows at higher sparsity, where the saliency estimates have to be more precise.

## 6. Unstructured vs 2:4 vs structured: the speed reality

This is the section to tattoo on the inside of your eyelids, because it is where most pruning enthusiasm dies. A 50%-sparse file is not a 2× faster model. The figure below lays out the three granularities against the only question that matters on real hardware.

![Matrix comparing unstructured fifty percent, two-of-four semi-structured, and structured pruning across granularity, speedup on stock hardware, and quality at the target sparsity](/imgs/blogs/pruning-llms-and-transformers-4.png)

**Unstructured 50%.** Best perplexity at any given sparsity — you delete exactly the least-useful weights with no pattern constraint. Speedup on stock GPU/CPU: effectively none. A dense matmul kernel does not branch on zeros; it multiplies them like any other number. Specialized sparse kernels (cuSPARSE, sputnik, custom Triton) exist, but for the *unstructured* 50% you typically get from SparseGPT/Wanda, they generally lose to a well-tuned dense kernel — the overhead of indexing which weights are nonzero (storing indices, gathering) eats the savings until sparsity gets very high (often 80–90%+), and pushing LLM sparsity that high tanks quality. So unstructured 50% on commodity hardware buys you a smaller file and *nothing else*. The file is smaller because you can store it in a sparse format, but you cannot run it faster.

**2:4 semi-structured.** This is the one granularity where the hardware was *designed* for the sparsity pattern. NVIDIA Ampere (A100), Ada (RTX 40-series), and Hopper (H100) Sparse Tensor Cores accept a 2:4 matrix in a compressed representation — two nonzero values plus a 2-bit index per group of four — and execute it at roughly **2× the dense throughput**. You pay a bit more quality than unstructured at the same 50% density (the 2:4 constraint forces you to keep exactly 2 of every 4, which is sometimes the wrong 2), but you get a real, guaranteed, no-custom-kernel-needed 2× for the matmul. This is the *only* unstructured-flavored pruning that reliably speeds up an LLM on commodity hardware, and it is exactly why 2:4 is the pruning pattern that actually ships. Details and the exact tensor-core mechanics are in [the N:M sparsity post](/blog/machine-learning/edge-ai/n-m-sparsity-and-sparse-tensor-cores).

**Structured.** Remove whole heads, neurons, layers, or width, and the model is *literally smaller* — fewer rows and columns in every matrix, fewer blocks in the stack. There is nothing special to do at inference time: any runtime, any hardware, CPU or GPU or NPU, runs the smaller dense model faster, in exact proportion to how much you removed. A model with 25% of its layers deleted runs ~25% faster on decode, full stop. The price is paid up front: structured cuts are coarse, so quality drops more, and you need recovery training (LoRA fine-tuning for LLM-Pruner, continued pretraining for Sheared LLaMA) to get back to par. But once recovered, it is a clean dense model with no kernel dependency — the most portable form of pruning by a wide margin, and the right choice when your target is a CPU or an NPU that has never heard of Sparse Tensor Cores.

The mental model to carry: **file size and speed are different axes.** Unstructured pruning moves file size and not speed. 2:4 moves both, but only on the right GPU. Structured moves both, everywhere, at a quality cost you pay through recovery training. When someone says "I pruned my model 50%," your first question is always "structured or unstructured, and did you actually measure the latency?"

## 7. Head redundancy and structured LLM pruning

Structured pruning deserves its own treatment because, for the edge, it is often the *more useful* family — it produces a portable dense model rather than a sparsity pattern that needs blessed hardware.

The intellectual foundation is head redundancy. Michel et al.'s 2019 finding — many attention heads are ablatable with little loss — is the empirical license to cut heads, and the figure below shows the before/after of doing so.

![Before and after diagram showing attention with all heads kept on the left, many of them redundant, versus a pruned configuration on the right keeping the highest-scoring heads with a tiny accuracy drop and a smaller faster projection](/imgs/blogs/pruning-llms-and-transformers-5.png)

The practical structured methods for LLMs are two, and they differ mainly in how they recover quality.

**LLM-Pruner** (Ma, Fang & Wang, 2023) is **gradient-based structured pruning plus LoRA recovery.** It uses a small amount of calibration data and a *single* backward pass to estimate the importance of structured groups (coupled sets of weights — a head, an FFN-neuron channel, etc., grouped so that removing them keeps the network connected), via a first-order Taylor saliency $|\,g \cdot w\,|$ (gradient times weight, the importance of a parameter to the loss). It prunes whole groups by that score, then runs a quick **LoRA** fine-tune — a low-rank adapter, cheap because only the adapter trains — to recover the quality the coarse cuts cost. The whole pipeline is hours on one GPU, not a cluster job, because LoRA recovery is parameter-efficient. This is the go-to when you want a smaller dense LLM and you have a modest fine-tuning budget. (LoRA itself is covered in the training-techniques track if you want the mechanics of low-rank adaptation.)

The "coupled groups" idea is the subtle engineering in LLM-Pruner and is worth a beat, because it is what separates *structured* pruning from naive channel deletion. You cannot delete an FFN intermediate dimension by removing just one column of the up-projection — that dimension also indexes a row of the down-projection, and if you delete one without the other the shapes no longer compose. So LLM-Pruner first traces the *dependency graph* of the network: it walks the computation and identifies sets of weights that must be removed together to keep every matmul shape-consistent and every residual connection intact. A "group" for one FFN neuron is {its up-projection column, its gate-projection column (for GLU), its down-projection row}; a "group" for one attention head is {its Q/K/V column slices, its O-projection row slice}. Importance is scored at the *group* level — sum the Taylor saliencies of the group's weights — and the group is pruned atomically. This dependency-aware grouping is why the output is a valid, runnable, *dense* smaller model rather than a broken graph. Why first-order Taylor and not the second-order OBS of SparseGPT? Because LLM-Pruner is willing to do a backward pass (it has gradients), and at the group granularity the first-order term $g \cdot w$ is a good enough importance signal; the expensive second-order Hessian buys little once you are pruning whole coupled groups and recovering with LoRA afterward.

**Sheared LLaMA** (Xia, Gao, Zeng, Chen, 2023) takes the more ambitious route: **targeted structured pruning to a chosen smaller shape, followed by continued pretraining.** Instead of pruning by a fixed saliency, it learns *mask variables* that prune the source model toward an explicit target architecture (e.g. turn a 7B into a 2.7B with specified layers/heads/width), optimizing the masks so the pruned shape matches a target while preserving the model. Then it runs **continued pretraining** on a modest token budget (far less than training from scratch) with a clever **dynamic batch loading** scheme that upweights the data domains where the pruned model is still weakest. The payoff is striking: Sheared-LLaMA-2.7B, pruned from 7B and continued-pretrained on ~50B tokens, matches or beats *open models trained from scratch* on far more tokens — i.e., pruning-then-continued-pretraining is dramatically more compute-efficient than pretraining a small model from random init. That is the strongest argument for structured LLM pruning that exists: it is a cheaper path to a good small dense model than training one fresh.

Running LLM-Pruner end to end is two commands — the structural prune, then the LoRA recovery:

```bash
# 1) Structured prune: drop coupled groups by Taylor saliency to ~20% params
python llama_prune.py \
  --base_model meta-llama/Llama-2-7b-hf \
  --pruning_ratio 0.20 \
  --block_wise --block_attention_layer_start 4 --block_attention_layer_end 30 \
  --save_ckpt out/llama7b_pruned

# 2) LoRA recovery fine-tune on a small instruction set (hours, one GPU)
python post_training.py \
  --prune_model out/llama7b_pruned/pytorch_model.bin \
  --data_path yahma/alpaca-cleaned \
  --lora_r 8 --num_epochs 2 \
  --output_dir out/llama7b_pruned_lora
```

Two flags carry the design. `--pruning_ratio 0.20` removes ~20% of parameters — structured pruning is gentler than the 50% you push unstructured, because coarse cuts hurt more per parameter, so the practical structured operating point is lower sparsity plus recovery. `--block_attention_layer_start/end` protects the first few and last few layers from pruning, encoding the same "endpoints are sensitive" wisdom you see in mixed-precision quantization — the early layers process raw embeddings and the late layers shape the output distribution, so both are kept dense. Everything in between is fair game.

The figure below shows what structured depth/width pruning does to the model itself — a 7B becomes a smaller dense checkpoint.

![Before and after diagram showing a 7B dense model with 32 layers and a 4096 hidden width on the left becoming a pruned dense model with fewer layers and slimmer width on the right requiring recovery training](/imgs/blogs/pruning-llms-and-transformers-7.png)

#### Worked example: structured 7B to 2.7B vs unstructured 50%

Put the two families head to head on a concrete deployment. You have a Llama-2-7B (~13 GB fp16, ~3.5 GB at int4) and a target that has no Sparse Tensor Cores — a Jetson Orin Nano, say, or a laptop CPU. Two pruning paths:

- **Unstructured 50% (Wanda).** Perplexity stays excellent (~7.3 vs ~5.7 dense on WikiText-2). File: you *could* store it ~2× smaller in a sparse format, but at inference the Orin's dense kernels read every weight including the zeros, so **decode latency is unchanged** — same ~13 GB to stream (or ~3.5 GB if you also int4 it), same tokens/s. Net result on this target: zero speedup. You did work for nothing speed-wise.
- **Structured to ~2.7B (Sheared-LLaMA-style), then int4.** The model is *genuinely* ~2.7B params: ~5.4 GB fp16, ~1.4 GB at int4. On the memory-bound Orin, decode reads ~2.6× fewer weight bytes per token, so you get roughly a **~2× tokens/s speedup** plus the memory headroom to fit a bigger KV cache or run at all. The cost: a continued-pretraining bill (tens of billions of tokens, not the trillions a from-scratch 2.7B would need) and a few points of benchmark quality versus the 7B — but a *real, portable* speedup on hardware with no special kernels.

On a target without Sparse Tensor Cores, this is the whole argument for structured over unstructured in one example: structured is the only pruning that actually made the model faster.

## 8. Composing sparsity with quantization

Pruning and quantization are not rivals — they stack, and the layer-wise framing is exactly why they stack cleanly. SparseGPT's authors built it on the GPTQ machinery on purpose; you can prune *and* quantize in one OBS-style sweep, or sequentially.

The most practical composition for shipping is **2:4 sparsity + int4 weight quantization.** You get the 2× from the Sparse Tensor Cores (compute side) *and* the ~4× weight-byte reduction from int4 (memory side), and on a memory-bound decode those wins are largely complementary — int4 cuts the bytes streamed, 2:4 cuts the matmul cost. The combined model is both smaller and faster than either technique alone. Sequentially, the recipe is: 2:4-prune with SparseGPT, then GPTQ-quantize the surviving weights to int4 (group-wise), each step minimizing layer-wise reconstruction error so the errors do not catastrophically compound.

```bash
# 1) 2:4-prune the 7B with SparseGPT (calibration-only, no training)
python main.py --model meta-llama/Llama-2-7b-hf \
  --prune_method sparsegpt --sparsity_type 2:4 \
  --nsamples 128 --save out/llama7b_2of4

# 2) GPTQ-quantize the pruned checkpoint to int4, group size 128
python -m auto_gptq.quantize \
  --model out/llama7b_2of4 \
  --bits 4 --group_size 128 \
  --dataset c4 --nsamples 128 \
  --save out/llama7b_2of4_int4_gptq
```

The order matters: prune *first*, quantize *second*. Pruning changes which weights survive; quantizing first and then pruning would have you spending precision bits on weights you are about to delete, and the OBS/GPTQ error-compensation works best when the later step compensates for the earlier step's perturbation rather than fighting it. The honest caveat is that errors *do* accumulate — 2:4 costs a little perplexity, int4 costs a little more, and stacked they cost a bit more than the sum suggests at the margins — so always measure the combined checkpoint's perplexity and a few real task metrics, not just the file size. For most edge LLM work, though, the bigger and easier win is quantization alone; sparsity is the thing you add *on top* when quantization has run out of room, which is the subject of the final decision section.

#### Worked example: stacking 2:4 and int4 on a 7B

Walk the byte and speed budget for a Llama-2-7B on an A100, which has Sparse Tensor Cores, so both levers can fire. Start dense: ~6.7B params (call it 7B) at fp16 is ~13 GB, decode is memory-bound, and the matmuls run on dense tensor cores. Apply the two levers and track what each one moves:

- **Memory (the bytes streamed per token).** int4 weight quantization cuts each weight from 2 bytes to ~0.5 byte, so the ~13 GB of weights drop to ~3.5 GB. 2:4 sparsity is *also* a memory win on top, because the compressed 2:4 format stores only the two nonzeros plus a small index per group — but the headline memory number is dominated by the int4 step. Decode reads roughly 13 GB / 3.5 GB ≈ **3.7× fewer weight bytes per token** thanks to int4, which on memory-bound decode translates to a comparable decode speedup before sparsity even enters.
- **Compute (the matmul throughput).** This is where 2:4 earns its keep: the Sparse Tensor Cores run the 2:4 matmul at ~2× the dense rate. On the parts of inference that are compute-bound (prefill of a long prompt, or large-batch serving), this 2× stacks on top of whatever quantization gave you.
- **Combined.** A 2:4 + int4 7B is ~3.5 GB and runs decode roughly 3–3.7× faster than dense fp16 on the memory-bound side, with an additional ~2× on the compute-bound prefill from the sparse tensor cores. Quality: ~5.7 dense PPL becomes ~8.5 from 2:4, and int4-GPTQ on top adds perhaps another ~0.3–0.5 — call it ~9 PPL combined, measured. Whether that quality is acceptable is a per-application call, but the *shape* of the trade is clear: you bought a model that is both ~4× smaller and meaningfully faster on both phases, at a perplexity cost you must verify against your real tasks. Crucially, on a target *without* Sparse Tensor Cores, the 2:4 step buys you nothing on compute and only the modest format-compression on memory — which is exactly why the very next section says quantize first and only add sparsity when the hardware can use it.

## 9. Results: SparseGPT vs Wanda vs magnitude

Now the numbers, on the canonical benchmark the LLM-pruning papers report: WikiText-2 perplexity on Llama-class 7B models, where lower is better and the dense baseline is roughly 5.7. These are approximate, drawn from the SparseGPT and Wanda papers and replications; the durable signal is the *gaps*, not the third decimal. The figure below is the same comparison in one glance.

![Matrix comparing magnitude, Wanda, and SparseGPT pruning methods on dense baseline perplexity, fifty percent unstructured perplexity, and two-of-four semi-structured perplexity for a 7B model](/imgs/blogs/pruning-llms-and-transformers-6.png)

| Method | Update | Hessian inverse | Dense PPL | 50% unstructured | 2:4 |
|---|---|---|---|---|---|
| Magnitude | none | no | ~5.68 | ~17 (collapses) | ~42 (collapses) |
| Wanda | none | no | ~5.68 | ~7.26 | ~8.6 |
| SparseGPT | OBS (weights) | yes | ~5.68 | ~7.22 | ~8.5 |

Three things to read off this table. First, **magnitude pruning collapses** at 50% unstructured (~17 vs ~5.7 dense) and is catastrophic at 2:4 — the lesson that activations *must* enter the saliency, which is the entire point of both Wanda and SparseGPT. Second, **Wanda essentially matches SparseGPT** at 50% unstructured (~7.26 vs ~7.22) despite having no Hessian inverse and no weight update — the activation-norm metric captures almost all the signal. Third, **2:4 costs more than 50% unstructured** for both methods (~8.5 vs ~7.2), the quality price of the rigid pattern; SparseGPT's compensation pulls slightly further ahead at 2:4, where the second-order correction starts to matter. That last gap is exactly why you reach for SparseGPT over Wanda when you are targeting 2:4 hardware specifically.

Now the table that actually decides deployments, because perplexity is not latency. Speed reality on a NAMED target — these illustrate the *pattern* (measure your own), for Llama-2-7B decode, batch 1:

| Configuration | Target | Decode speed | Model size | Speedup? |
|---|---|---|---|---|
| Dense fp16 | A100 (no sparse) | baseline | ~13 GB | — |
| Wanda 50% unstructured | A100 dense kernels | ~same as dense | ~13 GB streamed | no — zeros still read |
| SparseGPT 2:4 | A100 Sparse Tensor Cores | ~2× matmul | ~6.5 GB compressed | yes — hardware native |
| Structured to ~2.7B | A100 / Jetson / CPU | ~2× (smaller model) | ~5.4 GB fp16 | yes — everywhere |
| Structured 2.7B + int4 | Jetson Orin Nano | ~2× and fits | ~1.4 GB | yes — portable |

The two "no speedup" and "yes speedup" rows are the whole story of pruning-for-the-edge. Unstructured 50% gives the best perplexity and the worst speedup-reality. 2:4 gives a real 2× but only on Sparse Tensor Cores. Structured gives a real, portable 2× everywhere — at a quality and recovery-training cost. *Measure decode latency, not file size,* with proper warm-up (discard the first few tokens), at batch 1 (the edge reality), watching for thermal throttling on a small device; the gap between the "smaller file" and "faster model" is where careers' worth of disappointing pruning demos live.

## 10. When to prune an LLM vs just quantize it

Here is the opinionated part, and it is the most important paragraph in the post for someone deciding what to actually do. **For most edge LLM deployments, quantize first, and consider pruning only after.** The figure below is the decision in tree form.

![Tree diagram for deciding whether to prune or quantize an LLM, recommending weight-only int4 quantization first as the biggest easy win, then adding 2:4 sparsity on tensor cores or structured cuts only if the model is still too big](/imgs/blogs/pruning-llms-and-transformers-8.png)

The reason quantization wins the first move is simple and it is the roofline. Edge LLM decode is memory-bound: the bottleneck is streaming weight bytes, not doing arithmetic. Weight-only int4 quantization cuts those bytes ~4× and is a near-free speedup *with no retraining at all* — it is the single biggest, easiest LLM compression win, which is why it, and not pruning, is what put 7B models on phones and laptops (the full story is in [the weight-only quantization post](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq)). Unstructured pruning, by contrast, cuts file size but *not* the bytes a dense kernel reads, so on the very same memory-bound decode it buys you essentially no speedup. On a pure speedup-per-unit-effort basis, int4 quantization dominates unstructured pruning so thoroughly that for many edge LLMs pruning is simply not worth doing.

So when *does* LLM pruning pay? Three honest cases:

1. **You have Sparse Tensor Cores and want to stack speedups.** 2:4 + int4 gives you the compute 2× *and* the memory 4×, and they compose. If you are serving on A100/H100/Ada-class GPUs, 2:4 is real free-ish performance on top of quantization. This is the strongest, clearest case.
2. **You need a genuinely smaller dense model — for a CPU, an NPU, or a tight memory budget — and you have a recovery-training budget.** Structured pruning (LLM-Pruner with LoRA, or Sheared LLaMA with continued pretraining) is the *only* pruning that speeds up on hardware without sparse kernels, and it can be a cheaper path to a good small model than pretraining one from scratch. If your target is a Jetson or a phone NPU, structured-then-quantize is the play.
3. **You have already maxed out quantization and need more.** Once you are at int4 (or int3) and still too big or slow, sparsity is the next lever — and it is additive, since pruning attacks the matmul/structure while quantization attacks the bytes.

And when is it *not* worth it? If int4 (or a k-quant) already hits your latency, size, and quality targets — which it very often does — **stop there.** Pruning adds complexity (a calibration step, a sparse format or a recovery-training run, kernel dependencies for 2:4) for a marginal or even zero speedup in the common unstructured case. Do not prune for the aesthetic of a sparse model; prune when a *measured* gap remains after quantization and one of the three cases above applies. The graveyard of edge ML is full of beautifully pruned models that ran at exactly the same speed as the dense ones because nobody checked whether the hardware could exploit the sparsity.

To put a sharper edge on the recommendation, here is the decision as a checklist you can run in five minutes. (1) Have you quantized to int4 yet? If not, do that first — it is the bigger win and it is free of training. (2) After int4, do you still miss your size or latency target? If not, ship; you are done. (3) If you still miss it and your hardware has Sparse Tensor Cores, try 2:4 + int4 and measure — this is the highest-leverage pruning move that exists. (4) If you still miss it and your hardware has *no* sparse kernels (CPU, most NPUs, older GPUs), the only pruning that helps is structured, and only if you have a recovery-training budget; otherwise your remaining levers are a smaller base model or distillation, not pruning. (5) Whatever you do, measure decode latency at batch 1 with warm-up, on the actual device, before and after — the file got smaller is not the claim that matters; the model got faster is. Run that checklist honestly and you will prune exactly when pruning pays and skip it the (many) times it does not.

It is also worth saying plainly where pruning is *the wrong tool entirely*. If your problem is that the model is too slow because of the KV cache at long context, pruning weights barely touches it — you want KV-cache quantization or attention-architecture changes instead. If your problem is that you need a 7B's quality in a 1B's footprint, pruning a 7B down to 1B unstructured will not get you there (quality collapses long before 85% sparsity), and the right move is distillation or starting from a well-trained small base model. Pruning is a *trimming* tool — it removes the slack in an over-parameterized model. It is not a *transformation* tool that turns a big model into a fundamentally different small one. Match the tool to the bottleneck.

## 11. Case studies and stress tests

A few real numbers and edge cases to pressure-test the framing.

**SparseGPT on OPT-175B and BLOOM-176B.** The original SparseGPT paper's headline was that it could prune the *largest* open models of the era to 50% unstructured (and 2:4/4:8) in a few hours on a single GPU with negligible perplexity increase — something flatly impossible with any retraining-based method, which is the whole reason the one-shot approach mattered. The win scaled *with* model size: bigger models were more prunable (more redundancy to exploit), so the perplexity gap to dense shrank as the model grew. That "larger models prune more gracefully" finding is consistent across the literature and is worth remembering — your 70B has more slack than your 7B.

**Wanda's simplicity as a feature.** Because Wanda is a forward-pass metric with no weight update, it is trivially fast and trivially auditable — you can drop it into any model in a few lines, which is why it became the default baseline that every subsequent pruning paper compares against. Its existence is also a quiet rebuke: it showed that most of the value in calibration-based pruning is in *using the activations at all*, not in the expensive second-order machinery.

**Sheared LLaMA's compute efficiency.** Sheared-LLaMA-1.3B and -2.7B, pruned from Llama-2-7B and continued-pretrained on ~50B tokens, were reported to match or exceed similarly-sized open models (e.g. Pythia, OPT) trained from scratch on *far* more tokens — the concrete demonstration that prune-then-continue-pretrain is more compute-efficient than train-small-from-scratch. This is the most important structured-pruning result for anyone who actually needs a small dense LLM. The mechanism worth remembering is the dynamic batch loading: as the pruned model continues pretraining, the data sampler watches per-domain loss and upweights the domains where the pruned model has fallen furthest behind a reference, so the limited continued-pretraining budget is spent repairing the specific capabilities the structural cuts damaged most. It is targeted rehabilitation, not generic re-training.

**The "larger models prune more gracefully" law.** Pulling the threads together: across SparseGPT, Wanda, and the structured methods, a consistent empirical regularity is that the perplexity gap between pruned and dense *shrinks as the model grows*. A 70B model at 50% sparsity loses proportionally less than a 7B at 50%, which loses less than a 1.3B. The intuition is redundancy: a bigger model has more parameters than it strictly needs to fit its data (it is over-parameterized relative to the task), so there is more slack to remove before you hit the weights that genuinely matter. The practical implication is the opposite of what people often assume — pruning is *easier* to justify on your biggest models, not your smallest. A 1.3B is already lean; squeezing it hurts. A 70B has fat to trim.

Now the stress tests, because every technique has a regime where it breaks:

- **What happens past 60% sparsity?** Both Wanda and SparseGPT degrade, and SparseGPT pulls ahead of Wanda — at high sparsity the second-order compensation earns its cost, because you are deleting so many weights that the surviving ones genuinely need to be re-tuned to cover. Past ~70% even SparseGPT struggles on a 7B (the smaller the model, the less slack). If you need very high sparsity, you are back to needing recovery training, and the one-shot story breaks down.
- **What if the calibration set is tiny?** Both methods are remarkably robust — a few dozen sequences usually suffice, because the per-channel activation norms and the Hessian are dominated by stable statistics. But a *mismatched* calibration set (calibrating on code when you will deploy on prose) can cost you; use calibration data that resembles your deployment distribution.
- **What if the hardware can't exploit the sparsity?** Then unstructured pruning is pure file-size theater — covered above, and the single most common way pruning disappoints. The fallback is structured pruning (works everywhere) or just quantization.
- **What about the KV cache?** Pruning weights does nothing for the KV cache, which grows with context length and is its own memory pressure — at long context the KV cache, not the weights, can dominate memory, and pruning weights leaves it untouched (see [KV-cache optimization](/blog/machine-learning/large-language-model/kv-cache-optimization-and-management)). Another reason to be clear-eyed about *which* bottleneck pruning actually addresses. There is a partial exception: structured *head* pruning shrinks the K and V projections, which means each pruned head also removes its slice of the KV cache — so head pruning is one of the few weight-pruning moves that does help long-context memory. Unstructured pruning of the K/V weights does not, because the cache still stores full-width vectors.
- **One-shot vs the iterative loop, revisited.** A fair question: if iterative prune-recover reaches higher sparsity at a given quality, are we leaving performance on the table by going one-shot? Yes — but deliberately. The whole reason for one-shot is that the recover step is unaffordable at LLM scale. The honest framing is that one-shot SparseGPT/Wanda gets you to ~50% almost for free, and *if* you can afford some recovery training (which moves you into LLM-Pruner / Sheared-LLaMA territory), you can push further or structure the cuts for real speedup. The methods are not competitors so much as points on a budget curve: zero training budget buys 50% unstructured; a LoRA-sized budget buys structured-with-recovery; a continued-pretraining budget buys a high-quality smaller dense model. Pick the point your compute budget can reach.
- **What if the model uses grouped-query attention?** Modern models (Llama-2-70B, Llama-3, Mistral) share K/V heads across multiple query heads to shrink the KV cache. This changes head pruning: the Q heads can be pruned fairly freely, but pruning a shared K/V head removes it for *all* the query heads that depend on it, a much coarser and riskier cut. The coupled-group dependency tracing in LLM-Pruner handles this correctly, but it is a reminder that "prune a head" is not a uniform operation across architectures — always check what a head is coupled to before you cut it.

## 12. Key takeaways

- **The classic prune-then-fine-tune loop is structurally inapplicable to LLMs** — no affordable retraining, no training data — so the new generation prunes in one shot, post-training, from calibration data via *layer-wise reconstruction.*
- **SparseGPT** solves the per-layer problem optimally to second order: it reuses GPTQ's Hessian/OBS machinery to zero weights and push the error into the surviving ones, getting 50% sparsity at near-dense perplexity with no gradient step. The OBS saliency $\frac{w_q^2}{[H^{-1}]_{qq}}$ and compensation $\delta w = -\frac{w_q}{[H^{-1}]_{qq}} H^{-1} e_q$ are the whole method.
- **Wanda** approximates SparseGPT with a Hessian-free, update-free metric, $S_{ij} = |W_{ij}| \cdot \|X_j\|_2$, compared per output row — and it comes within a fraction of a perplexity point at 50%, because the activation norm captures most of the output-error signal and the per-row comparison protects outlier channels.
- **Magnitude pruning collapses on LLMs** at 50% — activations *must* enter the saliency. That is the one-line reason Wanda and SparseGPT exist.
- **File size and speed are different axes.** Unstructured 50% shrinks the file but not the bytes a dense kernel reads, so it gives *no speedup* on commodity hardware. Measure latency, not file size.
- **2:4 semi-structured** is the only unstructured-flavored pattern with a hardware-native ~2× — it requires NVIDIA Sparse Tensor Cores, and it stacks beautifully with int4.
- **Structured pruning** (heads, neurons, layers, width) makes a *genuinely smaller dense model* that any runtime speeds up, at the cost of recovery training — LLM-Pruner uses LoRA, Sheared LLaMA uses continued pretraining, and the latter is more compute-efficient than training a small model from scratch.
- **Quantize first, prune second.** Weight-only int4 is the bigger, easier LLM win; pruning is the lever you add on top, mainly as 2:4 on Sparse Tensor Cores or as structured cuts for portable hardware — and only when a *measured* gap remains.

## Further reading

- Frantar & Alistarh, **"SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot"** (2023) — the one-shot OBS-based method and the large-model results.
- Sun, Liu, Zhuo & Kolter, **"A Simple and Effective Pruning Approach for Large Language Models"** (Wanda, 2023) — the $|W| \cdot \|X\|$ metric and the per-row comparison.
- Ma, Fang & Wang, **"LLM-Pruner: On the Structural Pruning of Large Language Models"** (2023) — gradient-based structured pruning with LoRA recovery.
- Xia, Gao, Zeng & Chen, **"Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning"** (2023) — targeted structured pruning plus continued pretraining.
- Michel, Levy & Neubig, **"Are Sixteen Heads Really Better Than One?"** (2019) — the head-redundancy result that licenses head pruning.
- Hassibi & Stork, **"Second order derivatives for network pruning: Optimal Brain Surgeon"** (1993) — the classical OBS update SparseGPT modernizes.
- Within this series: [the model-compression taxonomy](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), [pruning fundamentals](/blog/machine-learning/edge-ai/pruning-fundamentals), [N:M sparsity and Sparse Tensor Cores](/blog/machine-learning/edge-ai/n-m-sparsity-and-sparse-tensor-cores), [weight-only LLM quantization](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq), [the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives), and the capstone [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).

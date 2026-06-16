---
title: "Pruning fundamentals: what to remove, how to score it, and the sparsity-speedup gap"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn what a network actually wastes, how to score which weights to delete, and why 90% sparsity almost never means 10x faster on real hardware — with runnable PyTorch and measured before-after numbers."
tags:
  [
    "edge-ai",
    "model-optimization",
    "pruning",
    "sparsity",
    "structured-pruning",
    "model-compression",
    "inference",
    "efficient-ml",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/pruning-fundamentals-1.png"
---

The first time I pruned a model in anger, I did everything the tutorials said. I took a ResNet-style classifier, zeroed out 90% of its weights by magnitude, confirmed with a quick `count_nonzero` that yes, nine out of ten weights were now exactly zero, and felt very pleased with myself. Then I ran the latency benchmark on the actual target — a humble ARM CPU — expecting something close to a 10x win. The model ran in 80 milliseconds. The original ran in 82. I had deleted ninety percent of the network and bought myself a rounding error.

That gap — between *sparsity*, the fraction of weights you set to zero, and *speedup*, the fraction of wall-clock time you actually save — is the single most important thing to understand about pruning, and it is the thing tutorials reliably skip. They show you the satisfying `count_nonzero` number and stop. They do not tell you that a dense matrix multiply on a CPU or GPU computes `0 * x = 0` just as eagerly as it computes any other product: the zero is still loaded, still multiplied, still accumulated. Scattered zeros do not save you a single clock cycle on hardware that was built to chew through dense tensors. The size on disk shrinks beautifully — you can store ten times fewer numbers — but the *time* does not move unless you change something deeper: either *what* you remove (whole channels and heads, so the tensor genuinely gets smaller) or *how* you run it (a sparse kernel or special hardware that knows how to skip zeros).

This post is about closing that gap honestly. We will work through the *science* of pruning — what parts of a network you can remove, and how to score which ones matter least, including a real derivation of the Taylor-expansion saliency that underpins most modern criteria. We will write *runnable PyTorch* with `torch.nn.utils.prune`, measure actual sparsity, and reproduce the exact disappointment I just described: a 90%-sparse model that is the same speed on a CPU. And we will close with *measured results* — a saliency comparison and a sparsity-vs-(size, latency, accuracy) table that shows you precisely where the speedup materializes and where it evaporates. Figure 1 lays out the four things you can prune and the speedup story for each; keep it in view, because that table is the spine of everything that follows.

![A matrix comparing the four pruning granularities of individual weights, neurons or channels, attention heads, and whole layers across what is removed, dense-hardware speedup, and accuracy cost](/imgs/blogs/pruning-fundamentals-1.png)

By the end you should never again confuse "I deleted most of the weights" with "I made it faster," and you should know exactly which kind of pruning to reach for given your actual goal — smaller storage versus faster inference, commodity hardware versus a sparse-capable accelerator. Pruning is one of the [four levers of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) — the one that *removes* parameters rather than shrinking their precision (that is [quantization](/blog/machine-learning/edge-ai/quantization-from-first-principles)) or retraining a smaller model from scratch (distillation). It is also the lever most likely to fool you into celebrating a win that your hardware never delivered. Let us make sure that does not happen.

## 1. The premise: a trained network is mostly redundant

Start with the observation that makes pruning possible at all: a trained neural network is enormously over-parameterized for the function it ends up computing. You train with far more weights than the task strictly needs because the extra capacity makes optimization easier — the loss landscape is friendlier, gradients flow better, the model finds a good solution faster. But once trained, much of that capacity is dead weight in the literal sense. Many weights sit near zero. Many neurons fire almost identically to their neighbors. Many attention heads attend to the same tokens as other heads. The network has redundancy baked in, and redundancy is exactly what you can remove without changing the function much.

How much redundancy? The empirical answer is striking and consistent across architectures: you can typically remove 50–80% of the weights of a well-trained convolutional network, with retraining, and lose under a point of accuracy. For some large language models, post-hoc methods reach 50% sparsity with negligible perplexity change and no retraining at all. The headline result that launched a thousand follow-up papers — Han, Mao, and Dally's *Deep Compression* in 2016 — pruned AlexNet by 9x and VGG-16 by 13x in parameter count before even applying quantization, with no loss of accuracy on ImageNet. The capacity was there for training; it is surplus for inference.

This is worth pausing on because it reframes pruning from "a trick to shrink models" into "removing the scaffolding after the building is up." The over-parameterization did its job during training. At inference, you are paying — in memory, in bandwidth, in energy — to store and move numbers that contribute almost nothing to the output. Pruning is reclaiming that. The only questions are *which* numbers contribute nothing (the saliency question, section 4) and *whether removing them actually saves you anything on your hardware* (the sparsity-speedup question, sections 5–7). Those two questions are the whole subject.

There is also an energy angle that matters enormously on battery-powered edge devices and is easy to miss if you only watch latency. On a modern SoC, the dominant energy cost of inference is rarely the arithmetic — a multiply-accumulate is cheap, on the order of a picojoule — it is *moving data*: reading a weight from off-chip DRAM can cost a hundred to a thousand times more energy than the multiply that consumes it. This is why *Deep Compression*'s original pitch was framed around energy, not just size: a model small enough to fit in on-chip SRAM never pays the expensive DRAM round-trip, so it runs cooler and lasts longer on a charge even if the FLOP count is unchanged. Pruning that shrinks the stored model below an on-chip memory boundary can therefore deliver a real energy and even latency win on a *memory-bound* layer — not by skipping multiplies, but by not fetching weights from far away. That is a genuine win unstructured pruning can capture, provided the runtime actually stores and streams the compressed form. We will keep this energy-and-bandwidth lens active throughout, because on the edge it is often the metric that decides whether a model is shippable.

A quick definition so the rest of the post is unambiguous. **Sparsity** is the fraction of weights in a tensor (or a whole model) that are exactly zero. A layer with a weight tensor of one million parameters, of which 900,000 are zero, is at 90% sparsity. **Density** is one minus sparsity. **Pruning** is the act of setting weights to zero (or removing them outright) to increase sparsity. And the word I want burned into your memory is **speedup**: the ratio of original wall-clock latency to pruned wall-clock latency, on your real target. Sparsity is something you *do*; speedup is something the *hardware decides* whether to grant you. They are different numbers, and the rest of this post is largely about the distance between them.

## 2. What you can prune: the four granularities

You can remove things from a network at four different *granularities*, and the choice of granularity matters more than the amount you remove, because granularity is what decides whether your hardware gets faster. This is the content of Figure 1, and it is the most important conceptual split in the whole field. Let us go through the four, from finest to coarsest.

**Individual weights (unstructured pruning).** This is the finest grain: you look at every scalar weight in the model independently and zero out the ones that matter least. The result is a *scattered* pattern of zeros — a weight here, a weight there, no particular structure. The tensor keeps its original shape; it is just mostly zeros now. Unstructured pruning gives you the *highest* achievable sparsity for a given accuracy budget, because you have maximum freedom: you can keep any weight and drop any other, with no constraint on the pattern. This is what *Deep Compression* did and what the [lottery ticket hypothesis](/blog/machine-learning/edge-ai/unstructured-pruning-and-the-lottery-ticket) studies. The catch — and you already know it from the intro — is that scattered zeros do nothing for dense hardware. The matmul still has the same shape, still does the same number of multiply-accumulates, still takes the same time. You win storage and you win bandwidth only if you actually compress the representation; you win *compute* time only with a sparse kernel or special silicon.

**Neurons / channels / filters (structured pruning).** This is the coarse grain that actually speeds things up. Instead of zeroing individual weights, you remove an entire *structural unit*: a whole output neuron of a linear layer (which deletes a row of its weight matrix and the corresponding column of the next layer), or a whole convolutional filter / channel (which deletes a 3D slice of the conv weight and shrinks every downstream feature map). The crucial difference is that the tensor *physically gets smaller*. A linear layer that was `[1024, 1024]` becomes `[1024, 512]` if you remove half its output neurons. That smaller matmul has half the FLOPs and runs roughly proportionally faster on *any* hardware — no sparse kernel needed, because there are no scattered zeros to skip; the zeros are gone, deleted, the shape is genuinely reduced. The cost is that you have less freedom — you must remove whole units, so you can hit lower maximum sparsity before accuracy suffers — and you usually need to retrain to recover. This is the subject of [structured pruning that actually speeds things up](/blog/machine-learning/edge-ai/structured-pruning-that-actually-speeds-things-up).

There is a hidden complication in structured pruning that does not exist for unstructured, and it is the reason structured pruning is more engineering work despite being simpler to describe: *dependencies*. When you delete an output channel of one layer, you have changed the *number of input channels* the next layer expects — so you must also delete the matching input slice of the next layer's weight, its batch-norm parameters for that channel, and so on through every consumer. In a plain feed-forward stack this is a simple chain, but residual connections, concatenations, and grouped convolutions create *coupled groups* of channels that must be pruned together or not at all: an add of two tensors requires both to have the same channel count, so you cannot prune channel 7 of one branch without pruning channel 7 of the other. Tools like Torch-Pruning build a *dependency graph* of the model to compute these coupled groups automatically; doing it by hand on a ResNet or a transformer is where structured pruning earns its reputation for fiddliness. Unstructured pruning has none of this — a zero is just a zero, the shapes never change, nothing downstream needs updating — which is part of why unstructured is the easier first experiment even though structured is what you ship for speed.

**Attention heads.** In a transformer, multi-head attention runs several attention "heads" in parallel, each with its own query, key, and value projections. A well-known result (Michel, Levy, and Neubig, 2019, "Are Sixteen Heads Really Better Than One?") is that many heads are redundant: you can prune a large fraction of them at test time with little quality loss, and some layers do fine with a single head. Removing a head deletes its Q, K, V, and output projection slices — a structured removal, so it gives a real speedup, just at the granularity of a head rather than a single channel.

**Whole layers.** The coarsest grain: drop an entire transformer block or residual block. Because residual connections let information route around a block, deep networks tolerate dropping some middle layers surprisingly well, especially after a little fine-tuning. This gives a direct, linear cut to depth — drop two of twenty-four layers and you cut roughly 8% of the compute on the critical path — but it is the most brittle: a layer is a large, indivisible chunk of function, so accuracy can fall off a cliff if you pick the wrong one. Layer dropping is best used surgically, guided by a sensitivity analysis (which layers can the network do without?), not as a blunt instrument.

The single sentence to remember from this section: **finer granularity prunes more but speeds up only special hardware; coarser granularity prunes less but speeds up everything.** That tension — maximum sparsity versus real speedup — is the central trade-off of pruning, and we will see it quantified in the results section.

## 3. The sparsity-speedup gap, stated precisely

Now let us make the gap from the intro rigorous, because it is the trap that bites everyone and the reason half of all pruning effort is wasted. Figure 2 puts the expectation next to the measurement.

![A before-after figure contrasting the expected ten-times speedup from ninety percent sparsity against the measured outcome of large size savings but essentially unchanged CPU latency](/imgs/blogs/pruning-fundamentals-2.png)

Here is the mechanism. A dense matrix multiply — the workhorse of essentially every layer — computes, for each output element, a dot product over an input vector. On a CPU or GPU, that dot product is implemented by a tight loop (or a vectorized SIMD / tensor-core equivalent) that loads a weight, loads an activation, multiplies them, and accumulates. The loop runs over *every* element of the weight tensor. It does not check whether a weight is zero. It cannot afford to — a branch on "is this weight zero?" inside the innermost loop would cost more than the multiply it might save, would wreck the instruction pipeline, and would defeat the SIMD vectorization that makes dense kernels fast in the first place. So the kernel loads the zero, multiplies by it (`0 * x = 0`), adds zero to the accumulator, and moves on. The zero costs *exactly the same* as a nonzero. Ninety percent of your weights being zero changes nothing about the number of multiply-accumulates executed, because the kernel was never going to skip them.

This is why **unstructured sparsity does not speed up dense hardware.** The arithmetic is unchanged; only the *values* changed, and the hardware does not care about values. To get a speedup you must change one of two things:

1. **Change the shape (structured pruning).** Remove whole rows / columns / channels so the matmul is genuinely smaller. A `[1024, 1024]` matmul cut to `[1024, 512]` does half the work — and the dense kernel, which is what your CPU/GPU is good at, runs half as long. The zeros are not skipped; they are *deleted*, so there is nothing to skip. This is the reliable path to speed.

2. **Change the kernel (sparse compute).** Use a kernel that stores only the nonzero weights plus index metadata (formats like CSR — compressed sparse row), and loops only over the stored values. Now the zeros genuinely cost nothing because they are not iterated. But sparse kernels carry overhead: irregular memory access kills cache locality, the index arrays cost bandwidth, and you lose SIMD efficiency. In practice a general sparse matmul only beats the dense one at *high* sparsity — often you need 70–95% sparsity before the sparse kernel wins, and even then the win is far below the naive `1 / density` you would hope for. There is also a special hardware case: NVIDIA's Ampere-and-later Sparse Tensor Cores accept a *2:4 structured sparsity* pattern (exactly two of every four contiguous weights are zero) and deliver up to a genuine 2x matmul speedup. But notice that is a *structured* constraint — 50% sparsity in a fixed pattern — not arbitrary scattered zeros, and it is hardware-specific.

It is worth getting quantitative about *why* the sparse kernel needs such high sparsity to pay off, because the break-even is the single fact that explains the entire industry's disappointment with unstructured sparsity. A dense matmul executes one multiply-accumulate (MAC) per weight, and every MAC is a contiguous, predictable memory read — exactly what SIMD lanes and cache prefetchers are built to exploit. A sparse matmul in CSR format does *less arithmetic* (it iterates only the nonzeros) but pays for it in three other currencies. First, *index bandwidth*: every stored value carries a column index (typically 4 bytes alongside a 4-byte fp32 value, or relatively even more for an int8 value), so at low sparsity you are streaming nearly as many index bytes as you saved in value bytes. Second, *irregular access*: the nonzero pattern is scattered, so the activation reads gather from unpredictable addresses, defeating the cache prefetcher and the streaming SIMD load. Third, *vectorization loss*: a dense kernel multiplies eight or sixteen lanes at once; a gather-scatter sparse loop often falls back to scalar or near-scalar throughput. Put those together and a real sparse kernel commonly needs the density to drop below roughly 10–30% (i.e. 70–90% sparsity) before the saved MACs outweigh the index, irregularity, and vectorization overheads. Below that break-even, the "sparse" kernel is *slower* than just running the dense one — which is why frameworks default to dense even when the weights are mostly zero. The naive hope of `1 / density` speedup (10x at 90% sparsity) is never realized; a good sparse kernel at 90% sparsity on a CPU might deliver 1.3–2x, and only if the library and the hardware cooperate.

So the precise statement is: **the speedup from pruning equals the speedup of the kernel you run, not the fraction of weights you zeroed.** On a dense kernel, scattered zeros give a speedup of 1.0 no matter the sparsity. On a structured-pruned model, the speedup is roughly the FLOP reduction (often near-linear in the channels removed). On a sparse kernel, it is whatever that kernel achieves at your sparsity level (sub-linear, and zero or negative below a break-even sparsity). The size win, by contrast, is real in *all* cases — you can always store fewer numbers — which is why pruning is genuinely valuable for memory-bound and storage-bound deployments even when it buys no compute speedup at all. We will [forward-reference the roofline model](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) here: if your layer is *memory-bound* (limited by moving weights in from DRAM, not by arithmetic), then shrinking the stored weight footprint *can* help even on a dense kernel, because you move less data — but only if the runtime actually stores and streams the compressed form, which a naive dense path does not. The gap is everywhere; you have to be deliberate to close it.

There is one more nuance that catches people, and it is the difference between *unstructured*, *semi-structured*, and *fully structured* sparsity. Unstructured is arbitrary scattered zeros — maximum freedom, worst hardware story. Fully structured removes entire shapes (channels, heads) — least freedom, best hardware story, works on any dense kernel. In between sits *semi-structured* sparsity, the most important of which is the **2:4 pattern**: within every contiguous group of four weights, exactly two are forced to zero. This is a constraint on the *pattern* (it is still 50% sparse, and you cannot pick which four), but it is a regular enough pattern that NVIDIA's Ampere-and-later Sparse Tensor Cores can decode it in hardware and run the matmul at up to 2x throughput. The 2:4 case is the proof that the speedup problem is solvable — it just requires meeting the hardware halfway with a structured constraint, not throwing arbitrary zeros at a kernel that was built for dense data. We pick this thread back up in the results table and the decision tree.

## 4. Saliency: how to score which weights to remove

We have established *what* you can remove. Now: *which* specific weights or units should go? You want to remove the ones whose absence hurts the model least. The score that estimates "how much does removing this weight hurt?" is called **saliency** (or *importance*). A good saliency criterion removes the genuinely useless weights and protects the load-bearing ones; a bad one deletes something the model needed and tanks your accuracy. Figure 3 compares the four main families, ordered from cheapest-and-crudest to most-accurate-and-expensive.

![A matrix comparing four saliency criteria of magnitude, gradient, first-order Taylor, and second-order OBD or OBS across what each one scores, the extra compute it needs, and the quality of the estimate](/imgs/blogs/pruning-fundamentals-3.png)

### 4.1 Magnitude: the strong, free baseline

The simplest possible criterion: the saliency of a weight is its absolute value, $|w|$. Small weights are presumed unimportant; prune the smallest. That is it. It is free — you already have the weights — and it requires no data, no gradients, no extra passes. And here is the uncomfortable, important fact that humbles a lot of fancier methods: magnitude pruning is *shockingly hard to beat*. Blalock and colleagues' 2020 meta-study, "What is the State of Neural Network Pruning?", surveyed the field and found that magnitude pruning, applied iteratively with fine-tuning, is competitive with or better than the great majority of more sophisticated methods, once you control for confounds like training budget. Many papers that claimed to beat magnitude pruning were comparing against a weak magnitude baseline. The lesson: *start with magnitude*. It is the honest baseline, and it is often the answer.

The intuition for why magnitude works: in a trained network, a weight near zero contributes little to its neuron's output, and removing it perturbs the function little. It is a *proxy* for importance — not a measurement of the loss change, just a cheap correlate. The proxy holds up well because of how training distributes weights, but it has a real failure mode, which is the entire reason the next criteria exist.

### 4.2 Gradient: how steep is the loss here?

Magnitude ignores the loss entirely. A weight could be tiny but sit on a very steep part of the loss surface, so that nudging it changes the output a lot. Gradient-based saliency scores a weight by $|\partial L / \partial w|$ — how much the loss changes per unit change in the weight. This requires a backward pass over some data, which is more expensive than free, but still cheap. The problem is that near a converged minimum, the gradients are *small by construction* (that is what "converged" means), so the raw gradient is a noisy, often unreliable signal for which weights matter. Gradient alone is rarely used; it is the raw material for the criterion that actually works.

### 4.3 First-order Taylor: the criterion you should actually understand

This is the workhorse, and it is worth deriving properly because the derivation tells you exactly *why* it works and when it does not. The question we are really asking is: **if I set weight $w_i$ to zero, how much does the training loss increase?** Call that increase the true saliency, $\Delta L_i = L(w \text{ with } w_i = 0) - L(w)$. We want to remove the weights with the smallest $\Delta L_i$. The trouble is computing $\Delta L_i$ exactly would require re-evaluating the loss with each weight removed — millions of forward passes, completely infeasible. So we *approximate* it with a Taylor expansion of the loss around the current weights.

Expand the loss $L$ as a function of weight $w_i$ around its current value, looking at the perturbation $\delta_i$ that we apply (here $\delta_i = -w_i$, because zeroing the weight changes it by $-w_i$):

$$
L(w_i + \delta_i) \approx L(w_i) + \frac{\partial L}{\partial w_i}\,\delta_i + \frac{1}{2}\frac{\partial^2 L}{\partial w_i^2}\,\delta_i^2 + \cdots
$$

The change in loss from the perturbation is therefore

$$
\Delta L_i = L(w_i + \delta_i) - L(w_i) \approx \frac{\partial L}{\partial w_i}\,\delta_i + \frac{1}{2}\,H_{ii}\,\delta_i^2 + \cdots
$$

where $H_{ii} = \partial^2 L / \partial w_i^2$ is the relevant diagonal entry of the Hessian. Now substitute the specific perturbation that zeroing the weight applies, $\delta_i = -w_i$:

$$
\Delta L_i \approx -\,w_i\,\frac{\partial L}{\partial w_i} + \frac{1}{2}\,H_{ii}\,w_i^2 + \cdots
$$

Drop the second-order term for now (we will pick it up in the next subsection) and take the absolute value, since a weight that decreases the loss when removed is just as "salient" as one that increases it — what we care about is the *magnitude* of the disturbance. We arrive at the **first-order Taylor saliency**:

$$
\mathcal{S}_i = \left| w_i \,\frac{\partial L}{\partial w_i} \right|
$$

Read that formula slowly, because it is the whole point. The saliency of a weight is its *value* times its *gradient*. It fuses the two cruder criteria: magnitude ($|w_i|$) tells you how big the weight is, and the gradient ($|\partial L / \partial w_i|$) tells you how much the loss cares about it, and the product tells you the estimated loss increase from removing it. A weight is safe to prune only if it is *both* small *and* on a flat part of the loss — small magnitude alone is not enough. This is exactly why magnitude can mislead, and Figure 4 shows the canonical failure: a tiny weight sitting on a steep loss outranks a large weight the loss ignores.

![A before-after figure showing that a large weight with a flat loss has near-zero Taylor saliency and should be pruned, while a small weight on a steep loss has high Taylor saliency and should be kept](/imgs/blogs/pruning-fundamentals-4.png)

In the left panel a weight is large ($w = 0.9$) but the loss is flat there ($\partial L / \partial w \approx 0$), so its Taylor saliency $0.9 \times 0 \approx 0$ says *prune it* — even though magnitude alone would have protected it as a "big" weight. In the right panel a weight is tiny ($w = 0.05$) but sits on a steep loss ($\partial L / \partial w = 8$), so its Taylor saliency $0.05 \times 8 = 0.4$ says *keep it* — even though magnitude would have deleted it as "small." This is the correction the first-order term provides, and it is why gradient-aware criteria win when magnitude is wrong. In practice you compute $\partial L / \partial w$ as a running average over a few hundred calibration examples to denoise the gradient (a single batch is too noisy), then score every weight by the product. Molchanov and colleagues at NVIDIA (2017, and a refined 2019 version) used exactly this Taylor criterion to prune convolutional filters for efficient inference, scoring each *filter* by the aggregated Taylor saliency of its weights — a structured criterion built on the first-order term we just derived.

### 4.4 Second-order: OBD and OBS

We dropped the second-order term above; the classical pruning methods kept it. Yann LeCun, John Denker, and Sara Solla's **Optimal Brain Damage** (OBD, 1990) — one of the foundational pruning papers — starts from the same Taylor expansion but makes a key assumption: at a well-trained minimum, the first-order term is approximately zero (the gradient has vanished, that is what a minimum means), so the *dominant* term is the second-order one. OBD therefore scores saliency by the curvature:

$$
\mathcal{S}_i^{\text{OBD}} \approx \frac{1}{2}\,H_{ii}\,w_i^2
$$

using only the *diagonal* of the Hessian, $H_{ii}$, to keep it tractable. The interpretation is elegant: removing a weight costs you in proportion to how *sharply curved* the loss is in that weight's direction. A weight in a flat valley is cheap to remove; a weight on a sharp ridge is expensive. Babak Hassibi and David Stork's **Optimal Brain Surgeon** (OBS, 1993) went further and kept the *full* Hessian, not just the diagonal — which lets it account for *correlations* between weights, and crucially lets it *adjust the remaining weights* to compensate for the one removed (a closed-form update derived from the inverse Hessian). OBS is more accurate than OBD precisely because it does not assume weights are independent. The cost is the Hessian: for $n$ weights it is an $n \times n$ matrix, $O(n^2)$ to store and worse to invert, which is intractable for any real network without heavy approximation.

The whole modern history of second-order pruning is approximations to that Hessian. The reason this matters *right now* is that the most effective post-training LLM pruners — SparseGPT (Frantar and Alistarh, 2023) and the closely related Wanda (Sun and colleagues, 2023) — are direct descendants of OBS. SparseGPT solves a layer-wise OBS-style problem with an efficient approximate inverse-Hessian update to prune a 175B-parameter model to 50% sparsity in one shot, without retraining, in a few hours on a single GPU. Wanda simplifies even further to a criterion of $|w| \cdot \|x\|$ — weight magnitude times input-activation norm — which is itself a cheap stand-in for the OBS objective and works astonishingly well. The 1993 theory is the engine under the 2023 results.

To summarize the hierarchy: **magnitude** is free and surprisingly strong; **gradient** alone is noisy; **first-order Taylor** ($|w \cdot \partial L/\partial w|$) is the practical sweet spot, fusing size and loss-sensitivity for one cheap backward pass; **second-order** (OBD diagonal, OBS full) is the most accurate and the most expensive, and its modern Hessian-approximation descendants are what prune today's LLMs. When in doubt, use iterative magnitude pruning as your baseline and only reach for Taylor or OBS when magnitude leaves accuracy on the table.

## 5. The prune → fine-tune loop, and why iterative beats one-shot

Scoring weights tells you *which* to remove. The other half of the recipe is *how aggressively* and *in what order*, and the answer is almost always: gradually, with recovery in between. Figure 5 shows the loop.

![A timeline showing iterative pruning that removes thirty percent then fine-tunes then prunes to sixty percent then to eighty percent, ending higher in accuracy than a single one-shot cut to eighty percent that gets stuck](/imgs/blogs/pruning-fundamentals-5.png)

The naive approach is **one-shot pruning**: compute saliency once, zero the bottom 80% of weights in a single cut, done. The problem is that an 80% cut is a violent perturbation. The Taylor approximation that justified your saliency scores is a *local* expansion — it is accurate for a small perturbation, but removing 80% of the network at once is not small, and the loss increase is far worse than the sum of the individually-estimated increases (because removing weight A changes how much weight B matters, an interaction the per-weight scores ignored). The network lands somewhere in the loss landscape far from a good minimum, and fine-tuning from there may never recover.

The fix is **iterative pruning**: remove a modest slice (say 20–30%), then *fine-tune* the surviving weights for a few epochs to let the network adapt and recover its accuracy, then re-score saliency on the now-adapted network and remove another slice, and repeat. Each prune step is a small enough perturbation that the local approximation holds and the fine-tune can heal it. You climb to high sparsity in stairs instead of jumping off a cliff. The empirical payoff is large and consistent: iterative magnitude pruning routinely reaches 10–20 percentage points higher sparsity at the same accuracy than one-shot, and the lottery ticket experiments depend on it entirely. The cost is compute — you fine-tune several times instead of once — but for a model you will deploy a million times, paying that training cost once is trivial.

A subtlety worth flagging: there are two flavors of "fine-tune after pruning." In the *fine-tuning* flavor you keep training from the current (pruned) weights at a low learning rate. In the *rewinding* flavor (Frankle and Carbin's lottery ticket follow-ups) you reset the surviving weights to an *early* point in the original training trajectory and retrain from there, which sometimes recovers better. For most engineering purposes, plain low-learning-rate fine-tuning after each prune step is the right default. The schedule that controls *how* sparsity ramps up over steps — for example the **polynomial / cubic sparsity schedule** of Zhu and Gupta (2017), which starts pruning slowly, accelerates, then tapers — is what tools like TensorFlow Model Optimization Toolkit's pruning API implement internally.

The idea to carry: pruning is not a one-time deletion, it is a *training procedure* with deletion folded in. You are co-optimizing the mask (which weights live) and the weights (what values the survivors take), in alternation. That framing is why the best pruning results come from the people who treat it as part of the training loop, not as a post-hoc afterthought.

There is a deeper reason iterative pruning wins that ties straight back to the saliency math, and it is worth spelling out because it sharpens your intuition for *how much* to remove per step. Recall that the Taylor saliency $\mathcal{S}_i = |w_i \cdot \partial L/\partial w_i|$ is a *first-order, local* approximation of the loss increase from removing weight $i$ — and critically, it estimates each weight's contribution *holding all other weights fixed*. The moment you remove a batch of weights together, that assumption breaks: the loss increase from removing weights A and B together is not the sum of their individual saliencies, because the second-order cross term $\partial^2 L / \partial w_A \partial w_B$ couples them. When you prune one weight per step the cross terms are zero by construction (there is nothing else changing), so the approximation is exact-ish; when you prune 80% at once the cross terms dominate and your per-weight scores are badly wrong about the joint effect. Iterative pruning is, in effect, a way to keep each prune step small enough that the cheap first-order saliency stays trustworthy, while the fine-tune step re-establishes a minimum (re-zeroing the first-order term) before you score and cut again. This is also why OBS, which keeps the full Hessian, can afford to prune more per step: it *models* the cross terms and even adjusts the survivors to compensate, so it does not need as many recovery rounds. The general rule that falls out: the cheaper your saliency criterion, the smaller your per-step prune fraction should be, and the more recovery you need in between.

## 6. Doing it in PyTorch: real sparsity, real code

Enough theory. Let us prune a real model with `torch.nn.utils.prune`, the built-in pruning API, and verify the sparsity we actually achieved. The running example is a small image classifier; the technique is identical for anything.

First, the simplest thing: prune a single layer by magnitude and inspect the result.

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# A toy two-layer classifier; substitute your real model.
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)

layer = model[0]  # the first Linear: weight is [512, 784]

# Prune 60% of the weights in this layer by lowest absolute magnitude.
# This zeros weights AND registers a mask so they stay zero through training.
prune.l1_unstructured(layer, name="weight", amount=0.60)

# Measure the actual sparsity we achieved.
w = layer.weight                        # the masked (effective) weight
sparsity = (w == 0).float().mean().item()
print(f"layer sparsity: {sparsity:.3f}")   # ~0.600
```

A few things to understand about what just happened, because the API is subtle. `prune.l1_unstructured` does not delete the weight. It keeps the original weight under a new name, `weight_orig`, registers a binary `weight_mask` buffer, and computes the effective `weight` as `weight_orig * weight_mask` via a forward pre-hook. So the zeros are *enforced by the mask on every forward pass* — which is exactly what you want during fine-tuning, because gradients will try to push pruned weights back to nonzero, and the mask zeroes them out again. To make the pruning permanent (fold the mask in and drop the bookkeeping), you call `prune.remove`:

```python
# Make the pruning permanent: weight = weight_orig * weight_mask, then
# delete weight_orig and weight_mask. Do this AFTER fine-tuning is done.
prune.remove(layer, "weight")
print((layer.weight == 0).float().mean().item())  # still ~0.600, now baked in
```

Now the part that actually matters for deployment: **global magnitude pruning.** Pruning each layer to the same percentage is suboptimal, because layers have very different sensitivity and very different weight-magnitude distributions. Global pruning pools all the weights across all layers, finds the global magnitude threshold for the target sparsity, and prunes against that single threshold — so a robust layer might lose 95% of its weights while a sensitive one loses 20%, automatically. This almost always beats uniform per-layer pruning.

Two practical caveats that bite people the first time they run global pruning. First, *not all layers should be in the pool*. The first layer (which sees the raw input) and the last layer (which produces the logits) are usually disproportionately sensitive — they have far fewer parameters than the middle of the network, so pruning them buys little size and costs a lot of accuracy. A common rule is to exclude the input and output layers from the prune set entirely, or give them a much lower per-layer cap. Second, *magnitude is only comparable across layers if the scales are comparable*. If one layer's weights live in $[-0.01, 0.01]$ and another's in $[-1, 1]$, a single global threshold will gut the small-scale layer and barely touch the large-scale one, which is rarely what you want. Networks trained with normalization (batch-norm, layer-norm) tend to keep weight scales roughly comparable, so global magnitude usually behaves; but if you see one layer getting pruned to 99% while its neighbors stay near-dense, suspect a scale mismatch and consider normalizing the saliency per layer before pooling. These are the kinds of details that separate a global-pruning run that works from one that mysteriously destroys accuracy.

```python
# Global magnitude pruning across multiple layers at once.
parameters_to_prune = [
    (model[0], "weight"),
    (model[2], "weight"),
]

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.80,          # 80% of ALL pooled weights, by global magnitude
)

# Report per-layer AND overall sparsity — they will differ by layer.
total_zeros, total_params = 0, 0
for module, pname in parameters_to_prune:
    w = getattr(module, pname)
    z = (w == 0).sum().item()
    n = w.nelement()
    print(f"{module.__class__.__name__}: {z / n:.3f} sparse ({z}/{n})")
    total_zeros += z
    total_params += n
print(f"GLOBAL sparsity: {total_zeros / total_params:.3f}")   # ~0.800
```

And the full **prune → fine-tune loop** from section 5, iterating to a high target sparsity in stairs:

```python
import torch.optim as optim

def measure_global_sparsity(params):
    z = sum((getattr(m, p) == 0).sum().item() for m, p in params)
    n = sum(getattr(m, p).nelement() for m, p in params)
    return z / n

def fine_tune(model, loader, epochs=3, lr=1e-3):
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()        # masked weights are re-zeroed by the forward hook

# Iterative schedule: ramp sparsity in steps, fine-tune to recover after each.
schedule = [0.30, 0.55, 0.70, 0.80]   # cumulative target after each round
for target in schedule:
    # global_unstructured's `amount` is the fraction to prune of what REMAINS,
    # so convert the cumulative target into a per-round fraction.
    current = measure_global_sparsity(parameters_to_prune)
    per_round = (target - current) / (1.0 - current)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=max(per_round, 0.0),
    )
    fine_tune(model, train_loader, epochs=3, lr=1e-3)
    acc = evaluate(model, val_loader)     # your eval fn
    print(f"target {target:.0%} -> sparsity "
          f"{measure_global_sparsity(parameters_to_prune):.3f}, acc {acc:.3f}")

# Bake in the masks when you are done.
for module, pname in parameters_to_prune:
    prune.remove(module, pname)
```

This is real, runnable code — the API names are exactly what `torch.nn.utils.prune` exposes. Notice the conversion of a *cumulative* target sparsity into the *per-round* fraction that `global_unstructured` expects (it prunes a fraction of the weights that are *still nonzero*, not of the total). Getting that arithmetic wrong is the most common bug in iterative-pruning code: people pass the cumulative target each round and end up at a far higher sparsity than intended.

## 7. The measurement that ruins the party: 90% sparse, same speed

Now we reproduce the intro's disappointment honestly, because seeing it in numbers is what makes the lesson stick. We will measure three things on the pruned model: its stored size, its in-memory parameter count, and — the one that matters — its actual CPU latency. Figure 6 is the mechanism we are about to confirm: a dense kernel runs the zeros anyway, while only a sparse kernel that stores indices can skip them.

![A before-after figure showing a dense kernel that stores and multiplies zero weights so latency is unchanged, versus a sparse CSR kernel that stores only nonzeros with indices and can win only above roughly seventy percent sparsity](/imgs/blogs/pruning-fundamentals-6.png)

Here is the latency measurement, done the right way — warm-up iterations first (to let caches and any JIT settle), then a timed loop, batch size one (the realistic edge case), and we report the median to dodge outliers:

```python
import time
import torch

def benchmark_latency(model, example_input, warmup=20, iters=200):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):           # warm-up: caches, lazy init, JIT
            model(example_input)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(example_input)
            times.append(time.perf_counter() - t0)
    times.sort()
    p50 = times[len(times) // 2] * 1e3    # ms
    p99 = times[int(len(times) * 0.99)] * 1e3
    return p50, p99

x = torch.randn(1, 784)   # batch=1, the edge reality

p50_dense, p99_dense = benchmark_latency(dense_model, x)
p50_sparse, p99_sparse = benchmark_latency(pruned_model, x)   # 90% unstructured

print(f"dense   p50 {p50_dense:.2f} ms  p99 {p99_dense:.2f} ms")
print(f"sparse  p50 {p50_sparse:.2f} ms  p99 {p99_sparse:.2f} ms")
```

And the size measurement, which is where the *real* win shows up — but note the catch: a model with masked-but-not-removed weights is *larger* on disk (it stores `weight_orig` and `weight_mask`), so you must `prune.remove` first, and then store the weights in a sparse format to capture the compression:

```python
import torch

def dense_state_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())

def sparse_state_bytes(model):
    # Store only nonzeros + indices (CSR-like). Approximate the footprint:
    # nnz values (4 bytes fp32) + nnz column indices (4 bytes) + row pointers.
    total = 0
    for p in model.parameters():
        nnz = (p != 0).sum().item()
        total += nnz * 4          # values
        total += nnz * 4          # column indices (CSR)
    return total

print(f"dense  : {dense_state_bytes(dense_model)/1e6:.1f} MB")
print(f"sparse : {sparse_state_bytes(pruned_model)/1e6:.1f} MB")
```

What do these print? On a commodity CPU running the standard dense `nn.Linear` kernel, the latency lines come back essentially equal — `dense p50 82 ms` versus `sparse p50 80 ms`, a 1.02x "speedup" that is pure measurement noise. The size lines, in contrast, show the win clearly: the dense model is 45 MB and the sparse one stored in CSR is about 6 MB — roughly 7x smaller (the 2x factor in the CSR estimate above is because each surviving fp32 value now carries a 4-byte index; at 90% sparsity that index overhead is small relative to the 10x reduction in stored values, netting ~7x). This is the gap, quantified: **a 7x size win and a ~1x latency win, from the same prune.** If your bottleneck was storage or download size or DRAM footprint, you won big. If your bottleneck was compute latency on a dense kernel, you won nothing.

#### Worked example: magnitude versus Taylor saliency on the same layer

Take a single trained `nn.Linear` layer, `[512, 256]` — 131,072 weights — and prune it to 50% sparsity two ways, then compare. Under **magnitude** pruning we keep the 65,536 weights with the largest $|w|$. Under **Taylor** pruning we score each weight by $|w \cdot \partial L/\partial w|$, averaging the gradient over 256 calibration examples, and keep the top 65,536 by that product. The masks differ for about 11% of the weights — magnitude protects some large-but-flat weights (high $|w|$, near-zero gradient) that Taylor discards, and Taylor protects some small-but-steep weights (low $|w|$, large gradient) that magnitude discards. After pruning *without* any fine-tuning, the accuracy gap is real but modest: suppose the dense layer's contribution gives 92.1% top-1; magnitude pruning lands at 89.4% and Taylor at 90.8% — a 1.4-point advantage for Taylor at matched 50% sparsity, exactly the weights-on-a-steep-loss that magnitude wrongly deleted. After three epochs of fine-tuning, though, *both* recover to about 91.7–91.9%, and the gap nearly closes. That is the practical punchline that Blalock's meta-study reached: the fancier criterion helps most *without* retraining; once you fine-tune, plain magnitude catches up and the extra backward passes for Taylor often are not worth it. Use Taylor when you cannot afford to retrain (one-shot LLM pruning); use magnitude when you can.

#### Worked example: 90% unstructured sparsity, size win versus latency win

Concrete numbers on a named target — a Raspberry Pi 5 CPU (Cortex-A76 @ 2.4 GHz), batch size 1, the standard PyTorch dense CPU backend. Start with a 45 MB fp32 classifier at 82 ms p50. Globally prune to 90% unstructured sparsity, fine-tune three epochs, measure. The stored model in CSR drops to about 6 MB (a 7.5x storage win) and accuracy falls from 92.1% to 91.4% (a 0.7-point drop, recoverable). Latency: 80 ms p50. The "speedup" is 82/80 = 1.025x — within noise of 1.0. Now try the *exact same target sparsity in a structured way* instead — remove half the channels of each conv/linear so the tensors physically shrink (this is roughly a 50% FLOP cut, lower sparsity than the 90% above but *structured*): the model drops to 24 MB, accuracy to 91.6%, and latency to *47 ms* — a genuine 1.7x speedup on the same dense CPU kernel, because the matmuls are smaller. Same device, same retraining budget: unstructured 90% bought 7.5x storage and 1.0x speed; structured 50% bought 1.9x storage and 1.7x speed. That single comparison is the entire thesis of this post. Choose the granularity that matches the bottleneck you are actually fighting.

#### Worked example: iterative versus one-shot at matched final sparsity

Hold the saliency criterion fixed (plain magnitude) and the final target fixed (80% global sparsity), and vary only the *schedule*. The one-shot run scores all weights once on the dense model, zeros the bottom 80% in a single cut, then fine-tunes for nine epochs to give it every chance to recover. The iterative run cuts in four stairs — to 30%, then 55%, then 70%, then 80% — fine-tuning three epochs after each (twelve epochs total, a comparable training budget). Starting from a 92.1% dense baseline, the one-shot model lands at roughly 71% accuracy immediately after the cut — a catastrophic 21-point drop, because deleting 80% of the weights at once is a violent perturbation that throws the network far from any good minimum — and even after nine recovery epochs it claws back only to about 88%, stuck well below baseline. The iterative model never falls far: each 30%-or-smaller cut drops accuracy a couple of points, the fine-tune recovers it, and after the final round it reaches about 91.4% at the same 80% sparsity — a 3.4-point advantage over one-shot, for the same criterion and a similar compute budget. The difference is entirely the schedule: small perturbations keep the local saliency approximation honest and let the network heal between cuts. This is why every serious pruning recipe ramps sparsity gradually, and why one-shot is reserved for the cases (like billion-parameter LLMs) where retraining is simply not an option and you lean on a much smarter, Hessian-aware criterion to survive the single cut.

## 8. Stress-testing the decision: a problem-solving walkthrough

Let me walk a realistic engineering problem end to end, the way you would actually reason about it, then stress-test the answer against the awkward edge cases — because a decision tree is only useful if it survives contact with reality.

**The problem.** You have a 45 MB image classifier hitting 82 ms per inference on a Raspberry Pi 5, and the product requirement is 50 ms (a 1.6x speedup) while staying within one accuracy point. You also have a separate, secondary constraint: the device's app bundle is tight and shaving the model below 15 MB would help the download. Two goals — latency and size — and they point at different pruning granularities. What do you do?

**Step one: profile before you prune.** Run the roofline first. Suppose the profile says the network is *compute-bound* on this CPU — the convolutions dominate and they are doing real arithmetic, not waiting on memory. That immediately tells you the latency win has to come from *fewer FLOPs*, which means *structured* removal (or quantization, but we are focused on pruning here). If the profile had instead said *memory-bound*, the calculus would flip: shrinking the stored weights (even unstructured) could help by moving less data, and an int8 quantization might do more for you than pruning. The profile is what decides which lever, and skipping it is how people prune the wrong thing.

**Step two: pick the granularity for the dominant goal.** Latency is the hard requirement, the network is compute-bound, so structured channel pruning is the answer for the *speed* goal. Prune channels iteratively — 20% per round, fine-tune three epochs each — and watch the latency-accuracy curve. At 50% channels removed you measured 47 ms and 91.6%: that clears the 50 ms requirement (1.7x) and the one-point accuracy budget (0.5-point drop). Done — for latency.

**Step three: layer the secondary goal without breaking the first.** The structured-pruned model is 24 MB, still above the 15 MB download wish. Now you compose: take that structured-pruned model and *quantize* it to int8 (prune before quantize, per section 9). Int8 cuts the survivors' bytes 4x, taking 24 MB to roughly 6 MB — comfortably under 15 MB — and on an int8-capable kernel it might even shave latency further. You met the hard latency requirement with structured pruning and the soft size requirement by composing quantization on top. This is the playbook the whole series points at: the levers are tools, you reach for the one whose *speedup story matches your bottleneck*, and you stack them in the order that does not waste work.

Now the stress tests, because the clean story above hides several ways the decision can go wrong:

- **What if the network had been memory-bound, not compute-bound?** Then structured pruning still helps (smaller weights, less to stream) but its FLOP reduction is no longer the lever doing the work — bandwidth is — so quantization to int8 (4x less weight data to move) might beat pruning for latency, and unstructured pruning with a sparse-streaming runtime could finally earn its keep. The granularity choice is downstream of the bottleneck, always.
- **What happens at extreme structured sparsity?** Structured pruning hits an accuracy wall earlier than unstructured because each removed unit is a coarse chunk of function. Pushing channel pruning past ~70% on this network might drop accuracy 3–5 points with no fine-tune recovery, because you have started deleting genuinely load-bearing channels, not just redundant ones. The fix is not "fine-tune harder" — it is "stop, you are past the redundancy budget," or switch to a finer grain plus a sparse kernel if you truly need more.
- **What if the calibration set for Taylor saliency is tiny?** Gradient-based and Taylor saliency average $\partial L/\partial w$ over data; with only a handful of examples the gradient estimate is noisy and the scores become nearly random, sometimes *worse* than free magnitude pruning. Below a few hundred diverse examples, prefer magnitude. This is a real failure mode in low-data or privacy-constrained deployments.
- **What if a target op has no sparse or pruned-shape kernel?** You prune channels, the shape becomes odd (say 374 channels), and the runtime's optimized kernel only has fast paths for multiples of 8 or 16 — so it falls back to a slow generic kernel and your "1.7x speedup" evaporates into a 1.05x. The fix is *hardware-aware* structured pruning: prune to channel counts the kernel likes (multiples of 8/16/32), trading a sliver of sparsity for a code path that is actually fast. The same lesson as everywhere in this post — the speedup is a property of the kernel, so prune to shapes the fast kernel supports.

The throughline of the walkthrough: every pruning decision is "what bottleneck, measured how, removed by which granularity, on which kernel" — and each stress test is just one of those four assumptions failing. Get the four right and pruning delivers; get one wrong and you ship a rounding error.

## 9. Results: the tables that tell you where speedup lives

Let us consolidate. Two tables capture everything an engineer needs to decide. The first compares the saliency criteria; the second is the sparsity-vs-(size, latency, accuracy) map that Figure 7 visualizes.

![A matrix mapping dense, fifty and ninety percent unstructured, and fifty percent structured pruning to their model size, CPU latency, and accuracy, showing latency stays flat for unstructured but drops for structured](/imgs/blogs/pruning-fundamentals-7.png)

**Table A — saliency criteria.** What each one scores, what it costs, when to use it.

| Criterion | Score | Extra compute | Needs data? | Best for |
| --- | --- | --- | --- | --- |
| Magnitude | $\lvert w \rvert$ | None (free) | No | The default baseline; iterative + fine-tune |
| Gradient | $\lvert \partial L/\partial w \rvert$ | One backward pass | Yes | Rarely alone; raw material for Taylor |
| First-order Taylor | $\lvert w \cdot \partial L/\partial w \rvert$ | Fwd + bwd on a batch | Yes | One-shot / no-retrain pruning; structured filter scoring |
| OBD (diagonal Hessian) | $\tfrac{1}{2} H_{ii}\, w_i^2$ | Diagonal Hessian | Yes | Accurate one-shot; classic CNN pruning |
| OBS (full Hessian) | full-Hessian + weight update | $O(n^2)$ Hessian | Yes | Highest accuracy; modern LLM pruners (SparseGPT/Wanda) are approximations |

**Table B — sparsity vs (size, latency, accuracy)** for the running classifier on a Raspberry Pi 5 CPU, batch=1, dense PyTorch backend, all numbers after fine-tuning. This is the table to internalize.

| Configuration | Sparsity | Size (MB) | CPU p50 (ms) | Speedup | Accuracy | Acc. delta |
| --- | --- | --- | --- | --- | --- | --- |
| Dense fp32 | 0% | 45 | 82 | 1.0x | 92.1% | — |
| Unstructured 50% | 50% | 23 (CSR) | 81 | 1.0x | 92.0% | -0.1 |
| Unstructured 90% | 90% | 6 (CSR) | 80 | 1.0x | 91.4% | -0.7 |
| Structured 50% (channels) | 50% | 24 | 47 | 1.7x | 91.6% | -0.5 |
| Unstr 90% + sparse kernel | 90% | 6 | ~55 (est.) | ~1.5x | 91.4% | -0.7 |
| 2:4 sparse + Sparse TensorCore (GPU) | 50% | 23 | (GPU) up to 2x matmul | ~baseline | — |

Read across the rows and the pattern is unmistakable. **Size** falls monotonically with sparsity — every pruning configuration shrinks the stored model, and unstructured 90% shrinks it most (7.5x). **Latency on a dense CPU kernel** is *flat* for the unstructured rows — 82, 81, 80 ms — because the dense kernel runs the zeros anyway, and only moves when you go *structured* (47 ms, 1.7x) or switch to a sparse kernel (the ~55 ms estimate, where the sparse kernel's overhead eats most of the theoretical 10x down to ~1.5x) or use dedicated 2:4 hardware on a GPU. **Accuracy** degrades gracefully and recoverably in every case, staying within a point of baseline through 90% sparsity after fine-tuning. The decision is therefore not "how much can I prune?" but "what bottleneck am I fighting, and which row removes it?" If it is storage or memory, take the unstructured 90% row. If it is dense-CPU latency, take the structured row. If you have a sparse-capable accelerator, take the matching specialized row. Choosing the wrong row is how you delete 90% of a network and gain a rounding error — the mistake I opened with.

## 10. Composing pruning with quantization: Deep Compression

Pruning is one lever; you rarely pull just one. The canonical demonstration is Han, Mao, and Dally's **Deep Compression** (2016), which stacks three techniques in sequence and gets a multiplicative win: (1) **prune** the network (iterative magnitude + fine-tune) to remove the redundant weights, (2) **quantize** the surviving weights — they used weight *sharing* via k-means clustering so many weights share one of a small codebook of values, plus fine-tuning of the codebook — and (3) **Huffman-code** the result to exploit the non-uniform distribution of the shared values and indices. On AlexNet this reached 35x total compression (240 MB to 6.9 MB) and on VGG-16 49x (552 MB to 11.3 MB), with no loss of ImageNet accuracy. The levers compose because they attack *different* redundancies: pruning removes whole weights, quantization reduces the bits of the survivors, and entropy coding removes the statistical redundancy in the result. None of them would get 35x alone; together they multiply.

The ordering matters and is worth stating as a rule, because it connects to the [taxonomy of how the levers compose](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression). **Prune before you quantize.** Pruning is a structural decision about which weights exist; quantization is a precision decision about how finely the survivors are stored. If you quantize first, you have committed to a precision for weights you are about to delete, wasting calibration effort on them, and the post-prune fine-tuning fights the quantization. Prune first (deciding the structure), fine-tune to recover, then quantize the survivors (deciding the precision), and optionally fine-tune again. The two wins are largely independent: pruning a CNN by 4x and then quantizing it from fp32 to int8 (a further 4x on the survivors) gets you to roughly 16x, and because [quantization *can* speed up compute on int8-capable hardware](/blog/machine-learning/edge-ai/quantization-from-first-principles) while structured pruning shrinks the matmuls, the *combination* is how you actually hit aggressive size-and-latency targets on a phone or microcontroller.

The honest caveat for composing: the wins are independent in *size* but interact in *accuracy*. Each lever spends some of your accuracy budget, and the perturbations are not perfectly additive — a heavily pruned network has less redundancy left to absorb quantization error, so an aggressively pruned-*and*-int4 model can lose more accuracy than the sum of each step alone would suggest. The mitigation is the same as always: do it iteratively, fine-tune (or quantization-aware-train) at each stage, and validate on real data after each lever, never just at the end.

## 11. Case studies: real numbers from the literature

Four results worth keeping in your head, each illustrating a different point about the sparsity-speedup gap.

**Deep Compression (Han et al., 2016)** — the storage case. As above: 9x parameter pruning on AlexNet, 13x on VGG-16, 35x–49x total with quantization and coding, *no accuracy loss* on ImageNet. The win was overwhelmingly in *size* (the paper's pitch was fitting models in on-chip SRAM and cutting energy by moving less data from DRAM), not raw matmul speedup from the unstructured sparsity — exactly the distinction this whole post hammers. It remains the cleanest demonstration that trained networks are wildly redundant and that the levers compose multiplicatively.

**Are Sixteen Heads Really Better Than One? (Michel et al., 2019)** — the structured-redundancy case. They found that a large fraction of transformer attention heads can be pruned at test time with negligible performance loss, and that many layers are effectively single-head. Because removing a head is *structured* (it deletes whole projection slices), this translates into real inference savings, not just a smaller checkpoint. The lesson: redundancy in transformers often lives at a coarse, structurally-prunable grain, which is good news for actually getting faster.

**SparseGPT (Frantar and Alistarh, 2023)** — the modern OBS case. A one-shot, layer-wise, OBS-derived pruner that takes a 175B-parameter OPT/BLOOM model to 50% unstructured sparsity (or 2:4 / 4:8 semi-structured patterns) with minimal perplexity increase, *without any retraining*, in a few GPU-hours. The 50% *unstructured* result is again primarily a memory win on a dense GPU kernel; the *2:4* result is the one that maps onto NVIDIA Sparse Tensor Cores for an actual matmul speedup — the case study that proves the "make the sparsity structured to get the speedup" rule at LLM scale. Wanda (Sun et al., 2023) reaches comparable quality with the much cheaper $|w| \cdot \|x\|$ criterion, underscoring how far a good cheap saliency proxy goes.

**The Lottery Ticket Hypothesis (Frankle and Carbin, 2019)** — the science case. They showed that inside a dense network there exist sparse subnetworks ("winning tickets") that, when trained *in isolation from the original initialization*, match the full network's accuracy — discoverable via iterative magnitude pruning. This is the deepest result on *why* pruning works: the redundancy is not random, there is a trainable sparse skeleton hiding in the dense network, and iterative magnitude pruning finds it. It is the theoretical backbone of the [unstructured-pruning sibling post](/blog/machine-learning/edge-ai/unstructured-pruning-and-the-lottery-ticket), and it is why iterative-with-rewinding beats one-shot.

A note on honesty in these numbers: where a figure is a direct paper claim (35x, 49x, 9x, 13x) I have given it as published; where I have given device-specific latencies (the 82 ms / 47 ms Raspberry Pi numbers) they are representative order-of-magnitude figures for the running example, illustrating the *ratios* that matter (1.0x vs 1.7x), not a benchmark you should cite. The ratios are the robust, transferable lesson; the absolute milliseconds depend entirely on your model, backend, and silicon, which is why the series insists you [measure on your actual target](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) rather than trust a table.

## 12. When to reach for pruning (and when it is a trap)

A decisive recommendation section, because every technique is a cost and the skill is refusing to pay for a win you do not need. Figure 8 is the decision tree.

![A decision tree that routes a pruning goal to unstructured magnitude pruning for smaller storage, or to structured channel and head pruning or two-to-four sparse tensor cores for faster commodity hardware](/imgs/blogs/pruning-fundamentals-8.png)

**Reach for unstructured pruning when** your bottleneck is *storage or memory*, not compute latency: a model that must fit in a tight Flash or SRAM budget, a model whose download size matters, a model that is *memory-bandwidth-bound* such that storing and streaming fewer weights actually helps (and your runtime supports a compressed/sparse weight format). Unstructured pruning gives the highest sparsity for the accuracy budget, so it is the right tool when bytes are the constraint. Pair it with a sparse-aware runtime or a 2:4 pattern if you also want speed.

**Reach for structured pruning when** your bottleneck is *latency on commodity hardware* — a phone CPU, a Jetson, a Raspberry Pi, any device running dense kernels. Removing whole channels, filters, heads, or layers physically shrinks the matmuls and delivers a real, kernel-agnostic speedup, at the cost of lower maximum sparsity and a retraining requirement. This is the only kind of pruning that reliably makes a dense-hardware model faster, full stop.

**Reach for the second-order / OBS family when** you cannot afford to retrain and you need the best accuracy at a given sparsity — most importantly, *one-shot LLM pruning*, where SparseGPT and Wanda are the state of practice. For everything you *can* fine-tune, iterative magnitude pruning is the honest default and the fancier criteria rarely justify their cost.

**It is a trap when** you expect unstructured sparsity to speed up a dense kernel — the opening mistake, and the most common one in the field. It is a trap when you prune one-shot to extreme sparsity without fine-tuning and then wonder why accuracy cratered. It is a trap when you prune *before* checking whether the pruned layer was even on the critical path — profile first, because pruning a layer that was not the bottleneck buys nothing (the roofline post exists for exactly this). And it is a trap when a quick quantization-only pass would already hit your target: pruning adds a retraining loop and a sparse-format deployment headache, so if int8 alone gets you under budget, do not prune at all. As with every lever in this series, the right amount of pruning is sometimes zero.

## 13. Key takeaways

- **Sparsity is what you do; speedup is what the hardware decides.** Setting 90% of weights to zero does not make a dense kernel 10x faster — it makes it the same speed, because `0 * x` costs exactly what any other multiply costs.
- **Granularity is the master variable.** Unstructured (individual weights) gives maximum sparsity but speeds up only sparse kernels or special hardware. Structured (channels, heads, layers) gives less sparsity but a real speedup on *any* dense kernel, because the tensors physically shrink.
- **Pruning always wins size; it wins speed only sometimes.** The storage and memory-footprint reduction is real in every case. The compute speedup requires structured removal or a sparse-capable runtime.
- **Saliency = magnitude times loss-sensitivity.** First-order Taylor saliency $|w \cdot \partial L/\partial w|$ is the practical criterion; it fuses how big a weight is with how much the loss cares, correcting magnitude's failure on small-but-steep weights.
- **Magnitude pruning is the baseline that is hard to beat.** Iterative magnitude + fine-tune is competitive with most fancier methods once you control for retraining budget. Start there; reach for Taylor/OBS only when you cannot retrain.
- **Iterative beats one-shot.** Prune a slice, fine-tune to recover, repeat. Small perturbations keep the local saliency approximation valid and let the network heal. This is how you reach high sparsity at low accuracy cost.
- **Prune before you quantize, and compose deliberately.** The levers attack different redundancies and multiply (Deep Compression's 35x), but their accuracy costs interact, so fine-tune and validate after each stage.
- **Profile first.** Pruning a layer that was not on the critical path buys nothing; measure on your real target with warm-up, batch=1, and median latency before and after.

## 14. Further reading

- **Yann LeCun, John Denker, Sara Solla — "Optimal Brain Damage" (NeurIPS 1990).** The foundational second-order pruning paper; the Taylor-expansion-with-diagonal-Hessian saliency we derived in section 4.4.
- **Babak Hassibi, David Stork — "Second Order Derivatives for Network Pruning: Optimal Brain Surgeon" (NeurIPS 1993).** OBS: the full-Hessian criterion with a closed-form weight update; the ancestor of SparseGPT and Wanda.
- **Song Han, Huizi Mao, William Dally — "Deep Compression" (ICLR 2016).** Prune + quantize + Huffman code for 35x–49x compression with no accuracy loss; the canonical composition result.
- **Davis Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, John Guttag — "What is the State of Neural Network Pruning?" (MLSys 2020).** The sobering meta-study: magnitude pruning is hard to beat, and many comparisons in the literature are confounded.
- **Jonathan Frankle, Michael Carbin — "The Lottery Ticket Hypothesis" (ICLR 2019).** Sparse trainable subnetworks exist inside dense ones; the science behind iterative magnitude pruning. See the sibling post on [unstructured pruning and the lottery ticket](/blog/machine-learning/edge-ai/unstructured-pruning-and-the-lottery-ticket).
- **Within this series:** the [taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for how pruning fits the four levers, [quantization from first principles](/blog/machine-learning/edge-ai/quantization-from-first-principles) for the lever you compose pruning with, [structured pruning that actually speeds things up](/blog/machine-learning/edge-ai/structured-pruning-that-actually-speeds-things-up) for the speedup path, and the [edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for putting it all together.
- **Official docs:** PyTorch `torch.nn.utils.prune` and the `torch.ao.pruning` sparsifier APIs; the TensorFlow Model Optimization Toolkit pruning guide; NVIDIA's documentation on 2:4 structured sparsity and Sparse Tensor Cores.

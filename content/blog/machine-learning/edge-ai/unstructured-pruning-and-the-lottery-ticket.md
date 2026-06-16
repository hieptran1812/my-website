---
title: "Unstructured pruning and the Lottery Ticket Hypothesis"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Delete 90 percent of a network's weights with almost no accuracy loss, learn why iterative magnitude pruning finds a winning ticket, and measure the honest catch — huge storage wins, no free speedup on commodity hardware."
tags:
  [
    "edge-ai",
    "model-optimization",
    "pruning",
    "sparsity",
    "lottery-ticket",
    "inference",
    "efficient-ml",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/unstructured-pruning-and-the-lottery-ticket-1.png"
---

You can delete more than ninety percent of a trained neural network's weights and lose almost no accuracy. Not ninety percent of the layers, not ninety percent of the channels — ninety percent of the individual numbers, scattered across the weight tensors, set to exactly zero. The model that remains is a tangle of holes, and it still classifies images, transcribes audio, or answers prompts at essentially the accuracy it had before. The first time you watch this happen on your own model it feels like a magic trick, because every instinct you built training the dense network says those weights were doing something. Most of them were not.

The Lottery Ticket Hypothesis, stated by Jonathan Frankle and Michael Carbin in 2019, takes that observation and turns it into something stranger and more beautiful: the sparse subnetwork that survives the pruning was not *created* by training. It was in the randomly initialized dense network all along — a small "winning ticket" of weights that, if you had only known which ones they were at initialization, you could have trained *by themselves*, from the same starting values, to the same accuracy as the full network. Training the big network was, in part, the search for that ticket. Figure 1 is the whole claim on one slide: a dense network on the left, the tiny winning subnetwork it contained on the right, both reaching the same accuracy.

![A before-and-after figure showing a dense network with one hundred percent of its weights on the left and a winning ticket subnetwork with about ten percent of the weights on the right, both reaching baseline accuracy](/imgs/blogs/unstructured-pruning-and-the-lottery-ticket-1.png)

There is a catch, and it is the most important sentence in this post: **unstructured sparsity rarely makes the model run faster on commodity hardware.** You zeroed ninety percent of the weights, but a CPU's matrix-multiply kernel and a GPU's tensor cores still multiply by those zeros, because they have no way to know which entries are zero without checking, and the check costs more than the multiply it would save. The model got dramatically *smaller* on disk and in memory, which is a real and valuable win on the edge — but the latency, on a normal phone or laptop, barely moves. This is the central honesty of unstructured pruning, and it is exactly why the speed-focused sibling technique, [structured pruning that actually speeds things up](/blog/machine-learning/edge-ai/structured-pruning-that-actually-speeds-things-up), exists. To understand *why* the zeros do not help speed, you need the same lens we used in [the roofline model and where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives): whether a layer is memory-bound or compute-bound, and what a dense kernel actually does with a sparse tensor.

By the end of this post you will be able to: rank weights by magnitude and prune them by hand; explain why iterative magnitude pruning reaches far higher sparsity than one big cut; state the Lottery Ticket Hypothesis precisely and run the train–prune–rewind procedure that finds a winning ticket; write the PyTorch loop that prunes a network to ninety percent sparsity and fine-tunes it back to accuracy; store the result in a bitmask or CSR format and measure the storage win; and — most importantly — measure the latency yourself and watch it *not* improve, so you never again confuse "smaller" with "faster." Pruning is the second of the four compression levers from [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression); this post is the deep dive on its unstructured branch, the one that produces the cleanest accuracy-per-weight curve and the most misleading latency story.

## 1. What pruning is, and the two kinds you must not confuse

Pruning means removing parameters from a trained network and accepting whatever accuracy cost that incurs — ideally near zero. The parameters you remove are set to zero and frozen there; they no longer participate in training or inference in any meaningful way. The question that splits the entire field in two is *what shape* the removed parameters take.

**Unstructured pruning** removes individual weights, anywhere, with no constraint on the pattern. If weight $W_{ij}$ in a layer is unimportant, you zero it, regardless of what its neighbors are doing. The result is a weight matrix that is mostly zeros but with the nonzeros scattered irregularly — a *sparse* matrix in the linear-algebra sense. This is the kind that gives you the spectacular ninety-percent numbers, because you are choosing weights one at a time and you can always find more individually useless weights than you can find useless whole rows or channels.

**Structured pruning** removes entire structural units: a whole filter in a convolution, a whole channel, a whole attention head, a whole row or column of a matrix. The constraint is severe — you cannot remove a single weight, you must remove the whole group — so you reach lower sparsity before accuracy breaks. But the payoff is the one unstructured pruning cannot give you: the resulting tensor is *smaller and dense*, so an ordinary dense kernel runs on it and is genuinely faster. That trade — less sparsity, real speed — is the whole subject of the sibling post, and I will keep forward-referencing it because the contrast is the point.

This post is about the unstructured kind. Hold one sentence in your head as the thesis for everything that follows: **unstructured pruning maximizes the number of weights you can delete and minimizes the speedup you get for deleting them.** It is the storage-and-memory lever, not the latency lever. The art is knowing when that is exactly what you want.

### Why "delete a weight" is even possible

It helps to understand why a trained network tolerates this at all, because it is not obvious. The reason is *over-parameterization*. Modern networks are trained with far more parameters than the task strictly requires, and over-parameterization is not a bug — it is part of why they train well in the first place. A wide, redundant network has a smooth, forgiving loss landscape with many good solutions; gradient descent finds one easily. But once training has found a good solution, much of that redundancy is dead weight, literally. Many weights have settled near zero because the optimizer found that ignoring those connections was fine. Others are small because they encode minor corrections that the network can absorb elsewhere. The trained network is a dense object carrying a sparse amount of actual information, and pruning is the act of reading off how sparse.

This is the deep tension of the whole technique, and it is worth stating plainly: the over-parameterization that makes a network *trainable* is largely removable once it is *trained*. You needed the extra capacity during the search; you do not need it for the answer. Pruning is the operation that separates the two.

### Where pruning sits among the four levers

It is worth placing this lever in the frame the whole series uses, because that placement governs when you reach for it. The four compression levers are quantization (fewer bits per number), pruning (fewer numbers), distillation (a smaller model trained to mimic a bigger one), and efficient architecture or neural architecture search (a better-shaped model from the start). Pruning is the lever that reduces the *count* of parameters while leaving each surviving parameter's precision and the model's overall topology intact — which makes it unusually composable. Because it does not change the numeric format, it stacks cleanly on top of quantization: you can prune to ninety percent and *then* quantize the survivors to int8, and the storage savings multiply. Because it operates on an existing trained model, it composes with distillation: distill to a smaller student, then prune the student. The taxonomy post lays out the full ordering logic, but the short version is that pruning is a mid-pipeline lever — applied after the architecture is chosen and usually before final quantization.

Within pruning itself, the unstructured-versus-structured split *is* the most consequential sub-choice, and it maps directly onto the series' recurring distinction between size wins and speed wins. Unstructured pruning sits on the size axis: it gives you the steepest accuracy-per-stored-byte curve of anything in the toolbox, and essentially nothing on the latency axis for a stock runtime. Structured pruning sits on the speed axis: a shallower size curve but a real latency win because a dense kernel runs on the genuinely smaller tensor. Knowing which axis your deployment is constrained on — read it off your own profiling, the way the metrics and roofline posts insist — is the entire decision, and it is the decision tree we close the post with.

## 2. Magnitude pruning: the dumb idea that works

If you have to pick which weights to delete, you need an importance score. There are clever scores — second-order ones based on the Hessian (Optimal Brain Damage, Optimal Brain Surgeon), gradient-based ones, ones that estimate the loss increase from removing each weight. But the score that wins in practice for unstructured pruning is almost embarrassingly simple: **the absolute value of the weight.** Rank every weight by $|W_{ij}|$, and zero the smallest ones. That is magnitude pruning, and it is the workhorse.

The intuition is exactly what it looks like. A weight's contribution to a layer's output is $W_{ij} x_j$, so a weight with small magnitude contributes little to the output for typical inputs, and removing it perturbs the output little. It is a crude proxy — a small weight multiplied by a large, frequent activation can matter more than a large weight multiplied by a rare one — but on real trained networks the crude proxy is shockingly competitive with the expensive Hessian-based scores. Gale, Elsen, and Hooker's 2019 study "The State of Sparsity in Deep Neural Networks" ran a careful, large-scale comparison and found that simple magnitude pruning, done well, matched or beat the more complicated learned and variational methods across Transformers and ResNets. The bitter lesson of pruning is that the simplest criterion, applied iteratively, is hard to beat.

### Per-layer versus global magnitude pruning

There is one real decision inside "rank by magnitude": rank *within each layer separately*, or rank *across the whole network at once*. Per-layer pruning removes, say, the smallest twenty percent of weights in *every* layer — every layer ends up equally sparse. Global pruning pools all weights from all layers into one ranking and removes the smallest twenty percent overall — so a layer whose weights happen to be large keeps more of them, and a layer whose weights are small loses more.

Global is almost always better, and the reason is that different layers have different weight scales and different sensitivities. The first convolution of a vision model often has large, important weights and should be pruned gently; a fat fully-connected layer near the end is usually massively redundant and can be pruned hard. Per-layer pruning forces the same rate on both and wastes capacity. Global pruning lets each layer find its own natural sparsity, and it is the default I reach for. The one caution: global pruning can occasionally prune a small, sensitive layer to oblivion because all its weights are small in absolute terms even though they matter. The fix is to exempt the first and last layers from pruning, which is standard practice and which we will do in the code.

### A first formula

Let me make "the smallest weights" precise, because the threshold is the whole operation. Given a flattened set of $N$ weight magnitudes and a target sparsity $s$ (the fraction to zero), you want the threshold $\tau$ such that the fraction of weights with $|W| \le \tau$ equals $s$. That is just the $s$-quantile of the magnitude distribution:

$$
\tau = \text{quantile}\big(\{|W_{ij}|\}, \, s\big), \qquad M_{ij} = \mathbb{1}\big[\, |W_{ij}| > \tau \,\big].
$$

The mask $M$ is a binary tensor, the same shape as $W$, with a $1$ where the weight survives and a $0$ where it is pruned. The pruned weight is $W \odot M$, an elementwise product. Everything in unstructured pruning is bookkeeping on top of that mask: how you choose $\tau$, when you recompute it, and how you store $M$ efficiently. We will come back to the mask repeatedly, because it is the object that makes the whole thing work and the object that makes it slow.

### Magnitude versus the smarter criteria

It is worth understanding *why* magnitude wins, because the alternatives are theoretically better and it is a genuine surprise that they lose in practice. The classical criteria come from the Optimal Brain Damage and Optimal Brain Surgeon line of work (LeCun et al. 1989; Hassibi and Stork 1993). Their starting point is the same second-order expansion we will use in a moment: removing weight $i$ raises the loss by approximately $\tfrac{1}{2} H_{ii} W_i^2$ to leading order, where $H$ is the Hessian of the loss. The "saliency" of a weight is that loss increase, and the right thing to prune is the weight with the *smallest saliency*, not the smallest magnitude. Optimal Brain Surgeon goes further and uses the full Hessian to also *adjust the surviving weights* to compensate for each removal, which is provably optimal to second order.

So why does crude magnitude beat this? Three reasons, all practical. First, the Hessian of a modern network has hundreds of millions of dimensions; the full $H^{-1}$ that Optimal Brain Surgeon needs is intractable to form, and the diagonal approximations that make it tractable throw away most of what made it better than magnitude. Second, at a well-converged minimum the diagonal Hessian entries $H_{ii}$ are often roughly comparable across weights within a layer, so the saliency $\tfrac{1}{2} H_{ii} W_i^2$ is dominated by the $W_i^2$ term — which means saliency ranking and magnitude ranking *largely agree*, and magnitude gets you most of the benefit for none of the cost. Third, magnitude pruning is followed by *fine-tuning*, which re-optimizes the survivors and recovers far more than the one-step weight adjustment Optimal Brain Surgeon does analytically. Fine-tuning is a better, cheaper compensator than the Hessian math. The combined effect, documented across the State of Sparsity study and the Hoefler survey, is that iterative magnitude pruning plus fine-tuning is a baseline the fancier methods struggle to beat on unstructured sparsity. That is the bitter lesson again: scale and a simple criterion plus retraining beat clever theory.

The one place the smarter criteria genuinely pull ahead is the *one-shot, no-fine-tune* regime — when you cannot afford to retrain, the second-order information is worth a great deal because there is no fine-tune to clean up after a crude cut. This is exactly the regime of large language models, where retraining is prohibitively expensive, and it is why the modern LLM pruning methods (SparseGPT, Wanda, which we will meet in the case studies) are essentially efficient resurrections of Optimal Brain Surgeon's idea — using local, layer-wise second-order or input-weighted information to prune without a full retrain. For everything you *can* fine-tune, magnitude is the answer.

## 3. One-shot versus iterative magnitude pruning

Now the single most important practical fact in this post: *how* you remove the weights matters enormously, even with the same magnitude criterion and the same final sparsity.

**One-shot pruning** computes the threshold once, zeros all the doomed weights in a single step, and then fine-tunes the survivors to recover. It is fast and simple. It also collapses early — push one-shot magnitude pruning much past sixty percent sparsity on a typical network and accuracy falls off a cliff, because you have asked the network to absorb a single enormous structural shock all at once, and fine-tuning from that wreckage cannot recover.

**Iterative magnitude pruning (IMP)** removes a small fraction — typically twenty percent of the surviving weights — fine-tunes the network back to health, then prunes another twenty percent of what remains, fine-tunes again, and repeats until it hits the target sparsity. Each round is a gentle nudge the network can adapt to, and crucially, *the magnitude ranking is recomputed after every fine-tune*, so weights that were borderline-small but turned out to be useful have a chance to grow during fine-tuning and survive the next cut. IMP is slower — it is many train-prune cycles instead of one — but it reaches dramatically higher sparsity. Ninety percent and beyond with negligible accuracy loss is routine for IMP and impossible for one-shot. Figure 2 contrasts the two.

![A before-and-after figure contrasting one-shot pruning that cuts to the target in a single step against iterative magnitude pruning that cuts twenty percent per round with fine-tuning between rounds](/imgs/blogs/unstructured-pruning-and-the-lottery-ticket-2.png)

### Why gradual removal reaches higher sparsity — the science

This is worth deriving carefully because it is the crux of the technique. Think of the trained network as sitting at a minimum of the loss $L(\theta)$. Pruning a weight is forcing one coordinate of $\theta$ to zero — a step of size $|W_{ij}|$ away from the minimum in that coordinate. The loss increase from a single such step, to second order, is approximately

$$
\Delta L \approx \frac{1}{2} H_{ii}\, W_{ij}^2,
$$

where $H_{ii}$ is the diagonal of the Hessian for that coordinate. Two things follow. First, $\Delta L$ scales with $W_{ij}^2$, which is *why magnitude is a sensible criterion* — small weights cause small loss increases. Second, and this is the key, the second-order approximation is only valid for small steps. When you prune one weight at a time and re-converge between cuts, you stay in the regime where each step is small and the loss bowl is locally quadratic, so fine-tuning reliably finds a nearby minimum. When you prune ninety percent at once, you take a single colossal step that leaves the quadratic regime entirely — the network lands somewhere on a wild part of the landscape from which gradient descent cannot route back to a good basin. Iterative pruning is, in effect, a way of keeping every individual step small enough that the approximation that justifies magnitude pruning stays true.

There is a second mechanism that one-shot lacks and IMP exploits: *the ranking adapts*. After each fine-tune, the network redistributes its function across the survivors. A weight that was the 21st-percentile-smallest before fine-tuning might grow and become safely above the threshold afterward, while a weight that looked safe might shrink. By recomputing the magnitude ranking after every round, IMP lets the network *vote* on which weights matter, repeatedly, instead of taking a single snapshot vote on the trained-but-unpruned network. This adaptive re-ranking is most of why IMP beats one-shot, and it is why the fine-tune step is not optional polish but a structural part of the algorithm.

The cost of all this is compute. IMP to ninety percent in twenty-percent steps is roughly $\log_{0.8}(0.1) \approx 10$ rounds, each with a fine-tune that might be a few epochs. That is a real training budget — often comparable to a full retrain — which is the honest downside of IMP and a reason to ask whether you need the extreme sparsity before you pay for it.

## 4. The IMP loop, station by station

Let me lay out exactly what one round of IMP does, because the order of operations is where people make mistakes. Figure 3 is the loop as a timeline.

![A timeline of one iterative magnitude pruning round showing a trained model, ranking weights by absolute value globally, zeroing the smallest twenty percent, freezing the mask, fine-tuning, and rising sparsity that loops or stops](/imgs/blogs/unstructured-pruning-and-the-lottery-ticket-3.png)

1. **Start from a trained (or fine-tuned) model** at the current sparsity $s$. The very first round starts from the fully-trained dense network.
2. **Rank the surviving weights by $|W|$ globally**, pooling across all prunable layers. Only the currently-nonzero weights are candidates — you never "unprune" a weight in vanilla IMP.
3. **Zero the smallest twenty percent of the survivors.** This raises sparsity multiplicatively: if you were at $s$ nonzero fraction, you are now at $0.8\,s$. Update the binary mask $M$ accordingly.
4. **Freeze the mask.** From here on, every forward pass multiplies by $M$, and — critically — every backward pass must also zero the gradients of pruned weights, or the optimizer's momentum and weight decay will silently drift them back to nonzero. The mask is a constraint, and it must be enforced on the gradient, not just the forward pass.
5. **Fine-tune** for a few epochs at a reduced learning rate to recover the accuracy lost by the cut. This is where the survivors redistribute the pruned weights' function.
6. **Check the target.** If you have hit the desired sparsity, stop; otherwise loop back to step 2 with the freshly fine-tuned model.

The two places people get burned are step 4 (forgetting to mask the gradient, so weights creep back) and the fine-tune learning rate in step 5 (too high and you destroy the structure you are trying to preserve; the standard move is to fine-tune at a fraction of the original peak LR, or to use a learning-rate rewind that replays the original schedule's tail). We will handle both in code.

### What global pruning learns about your layers

There is a free diagnostic hiding in global IMP that is worth harvesting every time you run it. Because global pruning ranks all weights together and lets each layer keep whatever fraction survives the global threshold, the *realized per-layer sparsity* after the run is a readout of which layers the network considers redundant. If you log each layer's sparsity at the end, a consistent and informative pattern emerges across most architectures. The fat fully-connected layers — classically the final dense layers of a vision model, or the large feed-forward blocks of a Transformer — end up *most* sparse, often well above ninety-five percent, because they hold enormous redundancy. The early convolutions and the small projection layers end up *least* sparse, sometimes barely pruned at all, because their few weights each do a lot of work. Pointwise (1×1) convolutions and the value/output projections in attention tend to be prunable; depthwise convolutions and normalization-adjacent layers tend not to be.

This per-layer profile is genuinely actionable. It tells you where your parameters are *actually* being wasted, which can feed back into architecture decisions — if a layer survives global pruning at ninety-eight percent sparsity, that layer was four-or-five-times wider than the task needed, and you might shrink it densely in the next architecture revision and skip the pruning dance entirely. It also flags fragility: if a small, important layer is being pruned harder than you expected, that is the layer most likely to be silently costing you accuracy, and the layer to exempt or protect. Reading the per-layer sparsity profile turns IMP from a black-box compressor into a measurement of your model's redundancy structure, and it is one line of logging to get it.

## 5. The Lottery Ticket Hypothesis

Everything so far is engineering. The Lottery Ticket Hypothesis is the science, and it reframes what pruning *is*.

Frankle and Carbin's 2019 paper, "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks," made a precise and falsifiable claim. Take a dense network, save its random initialization $\theta_0$, train it, and prune it with IMP to get a mask $M$. Now do the experiment that nobody had bothered to do: take the *same* mask $M$, reset the surviving weights to their *original* values from $\theta_0$ (not the trained values — the initial ones), and train *that* sparse subnetwork from scratch. The hypothesis is that this subnetwork — the masked init $M \odot \theta_0$ — trains to *match the accuracy of the full dense network*, in the same number of iterations or fewer.

They called such a subnetwork a **winning ticket**. The lottery metaphor is exact: a large randomly-initialized network is like buying a huge bundle of lottery tickets (subnetworks defined by every possible mask). Most are duds. But the bundle is so large that it almost certainly contains a winner — a subnetwork whose random initialization happens to be exactly right for the task. Training the dense network is, on this view, partly the process of discovering *which* ticket won, and pruning reads off the winning numbers.

What makes this surprising is the control condition. If you take the same sparse architecture (same mask $M$) but reinitialize the surviving weights to *fresh* random values instead of their original $\theta_0$ values, the subnetwork trains *much worse* — often it cannot match the dense accuracy at all. So it is not the *sparse structure* alone that wins; it is the structure *paired with its original initialization*. The specific random numbers the winning weights started with are part of the prize. That is the genuinely strange and important finding: a sparse subnetwork's trainability depends on the initialization it was born with, and IMP is, accidentally, a method for finding initializations that train well at high sparsity.

### What this does and does not say

Be precise about the claim, because it is widely overstated. The LTH does **not** say you can find the winning ticket without first training the dense network — finding the mask $M$ requires the expensive IMP run on the full network. So LTH is not, by itself, a recipe for cheaper training; you still pay for the dense train to discover the ticket. What it says is something deeper about *why* sparse networks are trainable at all: trained-then-pruned sparse subnetworks succeed because they inherit a lucky initialization, and that explains why pruning-and-fine-tuning works and why training-a-sparse-network-from-random-init usually does not. A large body of follow-up work — pruning at initialization (SNIP, GraSP), foresight pruning, and the search for tickets you can find *cheaply* — is the field trying to cash the LTH's promise into actually-cheaper training, with partial success.

### Why the random-reinit control matters so much

Dwell on that control condition, because it is the experiment that separates the LTH from a triviality. Suppose someone claimed only the *sparse structure* mattered — that any reasonable sparse architecture trains as well as the winning ticket. The LTH refutes this directly: take the winning mask $M$, keep the exact same sparse connectivity, but draw *fresh* random values for the surviving weights, and the network trains noticeably worse, frequently failing to reach dense accuracy. The structure is necessary but not sufficient. What completes it is the *coupling* between which weights survive and what values they started at. The winning weights are not random survivors; they are precisely the weights whose *initial* values were already well-positioned for the task, which is also why they grew large during training and survived magnitude pruning. There is a self-consistency here: the weights that end up large are disproportionately the ones that started in a good place, so magnitude pruning, by keeping the large-at-the-end weights, is implicitly keeping the well-initialized-at-the-start weights. The mask reads off the lucky initialization.

This also explains, mechanically, why sparse-from-scratch training is hard. If you pick a random sparse mask at initialization and train, you have almost certainly *not* picked the winning ticket — you have picked one of the overwhelmingly many losing tickets, a sparse structure whose surviving weights happen to start in mediocre positions. The dense network can route around bad initial positions because it has the redundant connections to compensate; a randomly-chosen sparse network cannot. This is the wall that pruning-at-initialization methods (SNIP, which scores connections by their effect on the loss at init; GraSP, which scores by gradient flow) keep running into: they try to *predict* the winning ticket before training, and they recover some of the benefit, but reliably matching the accuracy of train-then-prune-then-rewind has remained out of reach at high sparsity. The trained network's vote on which weights matter is still the most reliable signal we have, and it costs a full training run to collect.

### Tickets that transfer

One of the more provocative follow-ups (Morcos et al. 2019, and related work) asked whether a winning ticket found on one task or dataset transfers to another. The answer was partly yes: tickets found on larger, more diverse datasets transferred to smaller related tasks, and tickets found with one optimizer transferred across optimizers. This matters because it softens the LTH's "you must train the dense network first" cost: if a ticket found once on a big source task can be reused as a sparse initialization for many downstream tasks, the expensive ticket-finding run amortizes across all of them. It is a hint that winning tickets capture something about the *task family* and the *architecture*, not just one specific training run — that the lucky initialization encodes reusable inductive structure. The practical version of this idea now lives in how people sparsify large pretrained backbones once and fine-tune the sparse backbone on many tasks, paying the pruning cost a single time.

## 6. Rewinding: the fix that made LTH scale

The original 2019 LTH worked cleanly on small networks and small datasets — MNIST, CIFAR-10, modest convolutional nets. On large networks (ResNet-50, deep architectures on ImageNet) it broke. Resetting the survivors to $\theta_0$, the literal initialization, and retraining did *not* reliably produce a winning ticket at scale. The hypothesis looked like it might be a small-scale curiosity.

The repair came from Frankle, Dziugaite, Roy, and Carbin in 2020, in the work on **rewinding** and **linear mode connectivity** ("Linear Mode Connectivity and the Lottery Ticket Hypothesis," and the companion "Stabilizing the Lottery Ticket Hypothesis"). The fix is almost trivial to state and it changed everything: instead of rewinding the surviving weights all the way back to $\theta_0$ (epoch 0), rewind them to the values they had at a *slightly later* checkpoint — epoch $k$, for small $k$, after the network has trained for a tiny fraction of its schedule (often well under five percent of total training). This is **late rewinding**, or weight rewinding to epoch $k$. With this one change, IMP-plus-rewind found winning tickets in large networks that the original $\theta_0$ rewind could not. Figure 4 lays out the full procedure and the rewind branch.

![A branching dataflow figure of the lottery ticket procedure showing weights saved at init and an early epoch, dense training, pruning to a mask, two rewind options to init or to the early epoch, and retraining under the fixed mask to a winning ticket](/imgs/blogs/unstructured-pruning-and-the-lottery-ticket-4.png)

### Why rewinding to epoch *k* works — linear mode connectivity

The explanation Frankle and colleagues gave is one of the more elegant ideas in the area, and it is worth understanding because it tells you something true about how training works. Define two networks as **linearly mode connected** if, when you linearly interpolate between their weights, the loss along that straight line stays low the whole way — no barrier between them. Early in training, the trajectory is chaotic: two runs that branch from the same epoch-0 init but see data in different orders end up in *different* loss basins, separated by a barrier — they are *not* linearly connected. But after a short warmup — past epoch $k$ — the network has settled into a stable basin, and runs that branch from epoch $k$ onward stay linearly connected; they end up in the *same* basin.

The connection to pruning is this. A winning ticket must, after sparse retraining, land in essentially the same basin as the dense network it was carved from, or it will not match the accuracy. If you rewind to $\theta_0$, you rewind to the chaotic early regime, where the sparse retraining is free to wander into a *different* basin — and on a large, sensitive network it usually does, so the ticket fails. If you rewind to epoch $k$, *after* the network has become stable and linearly mode connected, the sparse retraining is pinned to the right basin and the ticket succeeds. Rewinding to epoch $k$ is, precisely, rewinding to the earliest point at which the network's fate is already decided enough that sparsity cannot derail it. That is why the magic number $k$ is small but not zero.

There is a practical lesson hidden here that applies even if you never run a formal LTH experiment: **when you fine-tune a pruned network, where you start the surviving weights from matters.** Starting them from the fully-trained values (standard fine-tuning) and starting them from an early checkpoint (rewinding) give measurably different results, and for high-sparsity targets, rewinding plus replaying the learning-rate schedule (so-called learning-rate rewinding, from Renda, Frankle, and Carbin 2020, "Comparing Rewinding and Fine-tuning in Neural Network Pruning") often beats plain fine-tuning. If your pruned model is leaving accuracy on the table at high sparsity, try rewinding before you try a cleverer pruning criterion.

### The role of the mask, one more time

It is worth being explicit about what the mask is doing across all of this, because the mask is the through-line. The mask $M$ is a *fixed binary structure* discovered by IMP on the dense network. Once you have it, the lottery-ticket experiment holds the mask constant and varies *only the initialization* of the surviving weights: trained values (fine-tuning), $\theta_0$ (original LTH), or epoch-$k$ values (rewinding). The mask is the architecture; the initialization is the seed. The LTH's whole content is that the architecture-and-seed *pair* is what wins, and that finding a good pair is what IMP secretly does. When you store a pruned model, you are storing exactly this: the mask plus the surviving values. Which is the perfect segue to why the model gets smaller but not faster.

## 7. Why unstructured sparsity saves memory but not time

Here is the engineering heart of the post, the part you must internalize to use this technique without lying to yourself or your manager. You have a weight matrix that is ninety percent zeros. What happens when you run a matrix multiply with it?

On a CPU or a GPU, a dense matrix-multiply kernel (GEMM) is a marvel of engineering: it tiles the matrices into blocks that fit in cache and registers, streams them through the multiply-accumulate units in a fixed, predictable pattern, and saturates the hardware's arithmetic throughput. It achieves that throughput *precisely because the access pattern is regular* — it knows exactly which element comes next, so it can prefetch, vectorize with SIMD, and keep the pipeline full. The kernel multiplies every element, including the zeros, because checking "is this element zero, and if so, skip it" would introduce a data-dependent branch into the inner loop that destroys the regularity the whole performance depends on. A branch misprediction costs more cycles than the multiply-add it would have skipped. So the dense kernel happily computes $0 \times x = 0$ ten times for every one real multiply, and your ninety-percent-sparse matrix runs at *exactly the dense speed*. You did all that work to delete the weights, and the kernel multiplies by the zeros anyway.

To get a speedup, you would need a **sparse kernel** that stores only the nonzeros and iterates over only them. Such kernels exist — sparse GEMM, SpMM — but on the kind of *unstructured, irregular* sparsity that magnitude pruning produces, they are slow until the sparsity is extreme (often beyond ninety-five or even ninety-nine percent), because the irregular memory access pattern — chasing index arrays to find the next nonzero — wrecks cache locality and prefetching. The hardware is built to stream regular data fast; unstructured sparse access is the opposite of regular. As a rule, on commodity CPUs and GPUs, *unstructured sparsity below about ninety-five percent is slower than dense*, not faster. This is the fact that disappoints everyone who reads the LTH paper and expects ninety-percent sparsity to mean ninety-percent faster.

### The roofline view: why neither bottleneck is helped

The roofline model (the subject of the dedicated post in this series) gives the cleanest way to see why the zeros do not buy speed, in both possible regimes. The roofline says a layer's achievable performance is limited by whichever is the binding constraint: its *arithmetic intensity* — operations performed per byte moved from memory — sets whether it is **compute-bound** (limited by the chip's peak FLOP/s) or **memory-bound** (limited by memory bandwidth). Walk through both with a dense kernel running on a ninety-percent-sparse tensor.

If the layer is **compute-bound**, the dense kernel performs all $N$ multiply-accumulates including the $0.9N$ that touch zeros. Because the kernel does not skip the zeros, the operation count is *unchanged* by pruning — you still execute every multiply — so the compute-bound runtime is unchanged. Pruning removed the *values* but not the *work*.

If the layer is **memory-bound** — which, recall, is the common case for batch-1 edge inference, where you stream a large weight matrix to multiply against a single small activation vector — the binding constraint is the bytes you move. Here is the subtle trap: a *dense* kernel reads the full dense weight tensor regardless of how many entries are zero, because the weights are stored densely in memory. So even memory-bound, the dense kernel moves the same bytes and runs at the same speed. The *only* way pruning reduces bytes-moved is if the weights are *stored* in a compressed sparse layout *and* the kernel reads from that compressed layout — which is exactly what a stock dense kernel does not do. This is the precise mechanism behind the "smaller but not faster" result: the storage win lives in how the model is *stored at rest*, and the latency is governed by how it is *laid out at compute time*, and a dense kernel always expands to dense at compute time. To convert the storage win into a memory-bandwidth (and therefore latency) win, you need a runtime that keeps weights compressed all the way into the kernel — the DeepSparse design, or a hardware-supported structured format. Without one, both roofline regimes leave you exactly where you started on speed.

### Where the wins actually are: storage and memory

So where is the win? Two real places, both about *bytes*, not *operations*.

**Storage on disk.** A pruned model is a sparse tensor, and a sparse tensor can be stored in far fewer bytes than a dense one — you only store the nonzeros plus some bookkeeping about where they are. At ninety percent sparsity, the on-disk model is roughly five times smaller (we will compute the exact factor below), which matters enormously for over-the-air updates, app bundle size, and devices with tiny flash budgets. Figure 6 shows the storage formats.

**Memory footprint at load.** If your runtime can keep the model in the sparse format in RAM and only materialize what it needs, the working set is smaller too — relevant on memory-constrained edge devices where the model competes with everything else for a few hundred megabytes. And because moving bytes from DRAM costs far more energy than the arithmetic (the data-movement physics from the roofline post), a model that *is* genuinely smaller in memory can use less energy even if the op count is unchanged — *if* the runtime actually exploits the sparse layout. On a dense runtime that decompresses to dense at load, you get the disk win but not the memory win.

That is the honest accounting. Unstructured pruning is a **storage and (sometimes) memory** optimization wearing the costume of a speed optimization. Figure 7 puts the storage win and the latency non-win side by side, and it is the single most important figure in this post.

![A before-and-after figure contrasting a dense model and a ninety percent unstructured sparse model, showing storage dropping about fivefold while latency on a dense kernel stays unchanged because zeros are still multiplied](/imgs/blogs/unstructured-pruning-and-the-lottery-ticket-7.png)

### The exception that proves the rule: DeepSparse

There is one well-known counterexample, and it is instructive precisely because of how hard it had to work. Neural Magic's **DeepSparse** runtime is a CPU inference engine built from the ground up to exploit unstructured sparsity for *speed*, not just size. It does this with sparsity-aware kernels that skip zero blocks, and — crucially — it pairs them with a pruning method that produces sparsity patterns the kernels can exploit, plus aggressive use of the CPU's cache hierarchy to keep the irregular accesses local. On the right models at high sparsity, DeepSparse genuinely delivers multiple-times speedups from unstructured sparsity on commodity CPUs, which is a real achievement. But note what it required: a *custom runtime* co-designed with the *pruning method*. You do not get this for free from PyTorch, ONNX Runtime, or a phone's stock NPU. So the rule stands with an asterisk: unstructured sparsity does not speed up *commodity* inference — unless you adopt a *sparsity-specialized runtime* built for exactly this. If you control the deployment stack and can run DeepSparse, unstructured pruning becomes a speed lever too. If you ship to a stock mobile runtime, it does not.

## 8. The practical flow: an IMP loop in PyTorch

Enough theory. Here is the real thing. PyTorch ships an unstructured-pruning toolkit in `torch.nn.utils.prune`, and it is the right starting point because it handles the mask bookkeeping for you — it stores the original weight as `weight_orig`, the mask as `weight_mask`, and reapplies the mask on every forward pass via a hook so pruned weights stay zero. Our spine model for this post is a small convolutional classifier on CIFAR-10; the same code scales to a ResNet.

First, global magnitude pruning in one shot, to see the API:

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def collect_prunable(model):
    # Prune conv and linear weights, but exempt the first conv and the
    # final classifier: small, sensitive layers global pruning would over-cut.
    params = []
    modules = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    for m in modules[1:-1]:
        params.append((m, "weight"))
    return params

def global_magnitude_prune(model, amount):
    params = collect_prunable(model)
    prune.global_unstructured(
        params,
        pruning_method=prune.L1Unstructured,  # rank by |w|, zero the smallest
        amount=amount,                         # fraction of *all pooled* weights
    )
    return model
```

`prune.global_unstructured` pools the magnitudes of every `(module, "weight")` pair, finds the global threshold, and installs a mask on each module. After this call, `model` runs with the pruned weights automatically; the masks live on the modules. To read the realized sparsity:

```python
def report_sparsity(model):
    total, zeros = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight"):
            w = m.weight
            total += w.numel()
            zeros += (w == 0).sum().item()
    print(f"global sparsity: {100.0 * zeros / total:.1f}%  "
          f"({zeros:,} / {total:,} zero)")
```

Now the iterative loop — the part that actually reaches ninety percent. The key subtlety is in the fine-tune: PyTorch's prune hooks keep the *forward* pass masked, but you must make sure the optimizer does not resurrect pruned weights through momentum or weight decay. Because `torch.nn.utils.prune` reapplies the mask after every forward (it recomputes `weight = weight_orig * weight_mask`), pruned entries are re-zeroed before each use, so the forward stays clean; but to keep the *stored* `weight_orig` from drifting and to make the masking explicit and robust, the cleanest pattern is to zero the gradients of pruned positions before each optimizer step:

```python
def zero_pruned_grads(model):
    # Belt-and-suspenders: stop momentum/weight-decay from drifting
    # pruned weights. Works with torch.nn.utils.prune's weight_mask.
    for m in model.modules():
        if hasattr(m, "weight_mask") and m.weight_orig.grad is not None:
            m.weight_orig.grad.mul_(m.weight_mask)

def fine_tune(model, loader, epochs, lr, device):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            zero_pruned_grads(model)   # enforce the mask on the gradient
            opt.step()
        sched.step()
    return model

def iterative_magnitude_prune(model, loader, target_sparsity,
                              per_round=0.20, ft_epochs=3, ft_lr=0.01,
                              device="cpu"):
    survivor_fraction = 1.0
    while (1.0 - survivor_fraction) < target_sparsity:
        # Prune per_round of the *surviving* weights this round.
        global_magnitude_prune(model, amount=per_round)
        survivor_fraction *= (1.0 - per_round)
        fine_tune(model, loader, epochs=ft_epochs, lr=ft_lr, device=device)
        report_sparsity(model)
    return model
```

Calling `iterative_magnitude_prune(model, train_loader, target_sparsity=0.90)` runs about ten rounds of cut-and-fine-tune and lands near ninety percent sparsity with the accuracy recovered after each cut. Note `amount=per_round` is the fraction of the *currently pooled (surviving)* weights, so sparsity compounds: $1 - 0.8^{10} \approx 0.89$.

When you are done and want to bake the masks in permanently — turning `weight_orig` and `weight_mask` back into a single sparse `weight` with zeros — call `prune.remove`:

```python
def make_permanent(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight_mask"):
            prune.remove(m, "weight")   # fold mask into weight, drop the buffers
    return model
```

After `make_permanent`, the model is a normal `nn.Module` whose weights happen to be ninety percent zeros — ready to save in a sparse format.

### Saving the sparse model — where the storage win lives

A dense `torch.save` of the pruned model is *the same size as the unpruned one*, because the zeros are still stored as full numbers. To realize the storage win you must serialize in a sparse layout. The two formats that matter here are the bitmask layout and CSR; here is the bitmask version, which is the winner at the ~90 percent operating point:

```python
import torch

def save_bitmask(state_dict, path):
    # For each weight tensor, store a packed bit mask of nonzero positions
    # plus the int8 nonzero values. This is the on-disk storage win.
    blob = {}
    for name, w in state_dict.items():
        if w.dim() < 2:                     # skip biases/norms; keep dense
            blob[name] = ("dense", w)
            continue
        flat = w.flatten()
        mask = flat != 0
        packed = torch.from_numpy(
            __import__("numpy").packbits(mask.cpu().numpy()))   # 1 bit/weight
        values = flat[mask].to(torch.int8)                      # nonzeros only
        blob[name] = ("bitmask", w.shape, packed, values)
    torch.save(blob, path)

def load_bitmask(path):
    np = __import__("numpy")
    blob = torch.load(path)
    out = {}
    for name, rec in blob.items():
        if rec[0] == "dense":
            out[name] = rec[1]
            continue
        _, shape, packed, values = rec
        n = int(torch.tensor(shape).prod())
        mask = torch.from_numpy(np.unpackbits(packed.numpy())[:n]).bool()
        flat = torch.zeros(n, dtype=torch.int8)
        flat[mask] = values                 # scatter nonzeros back
        out[name] = flat.reshape(shape)
    return out
```

The mask costs exactly one bit per weight regardless of sparsity; the values cost one byte per *surviving* weight. So total bytes are $N/8 + (1-s)N$, which at $s = 0.9$ is $0.125N + 0.1N = 0.225N$ — a $4.4\times$ reduction over the dense int8 $N$ bytes, exactly the number we will measure in the worked example. For CSR you would instead store `values`, `col_indices`, and `row_pointers`, which PyTorch supports natively via `w.to_sparse_csr()`; CSR pays per-nonzero index overhead and only overtakes the bitmask at higher sparsity.

### Adding rewinding to the loop

To run a true lottery-ticket experiment with rewinding instead of fine-tuning, you change two things: save an early checkpoint, and reset the survivors to it after each prune instead of continuing from the trained values.

```python
import copy

def save_rewind_checkpoint(model):
    # Call this after ~k% of dense training (the rewind point).
    return copy.deepcopy(model.state_dict())

def rewind_survivors(model, rewind_state):
    # Reset surviving weights to the epoch-k values; pruned stay zero.
    sd = model.state_dict()
    for name, tensor in rewind_state.items():
        if name.endswith("weight_orig") and name in sd:
            sd[name].copy_(tensor)         # restore epoch-k weights
    model.load_state_dict(sd)
    return model
```

In the lottery-ticket loop you would: train a few epochs and snapshot the rewind point, train to convergence, prune, then `rewind_survivors` and *retrain from the rewound checkpoint under the frozen mask* — replaying the learning-rate schedule. That is IMP-with-weight-rewinding, the procedure that scales. The difference from the fine-tune loop is entirely in step "reset the survivors," and as section 6 explained, that difference is what lets the ticket land in the right basin on a large network.

## 9. Worked example: IMP to 90 percent and reading the knee

#### Worked example: pruning a CIFAR-10 ConvNet to the knee

Take a small VGG-style ConvNet on CIFAR-10, the kind that hits about 93.0 percent test accuracy dense and weighs in around 14 million parameters (roughly 56 MB as fp32). Run the IMP loop above, twenty percent per round, three fine-tune epochs per round, and record test accuracy at each sparsity level. The shape of the curve is the whole lesson, and it is the same shape you will see on essentially any network. Here is a representative run, with one-shot pruning to the same targets shown alongside for contrast:

| Sparsity | Nonzero params | One-shot acc | Iterative (IMP) acc | Δ vs dense (IMP) |
| -------- | -------------- | ------------ | ------------------- | ---------------- |
| 0% (dense) | 14.0 M | 93.0% | 93.0% | baseline |
| 50% | 7.0 M | 92.9% | 93.1% | +0.1 pt |
| 80% | 2.8 M | 91.5% | 92.8% | -0.2 pt |
| 90% | 1.4 M | 87.0% | 92.4% | -0.6 pt |
| 95% | 0.7 M | ~72% (broken) | 91.3% | -1.7 pt |
| 98% | 0.28 M | collapsed | 88.5% | -4.5 pt |

Read the iterative column top to bottom. Accuracy is *flat* — within noise of dense — all the way to eighty percent. It is still excellent at ninety percent, down only about half a point. Then comes **the knee**: between ninety and ninety-five percent the curve starts bending, and past ninety-five it drops steadily. The knee is where you have finally removed enough genuinely-useful weights that fine-tuning can no longer fully compensate. The engineering target is *just before the knee* — for this network, around ninety percent, where you have deleted ninety percent of the parameters for half a point of accuracy. That is an extraordinary trade if your constraint is size.

Now read across the one-shot column and watch it die. One-shot keeps up to fifty percent, wobbles at eighty, and falls off a cliff at ninety where IMP is still fine — the six-point gap at ninety percent sparsity *is* the value of iterating. Figure 5 is this table as a matrix, with the knee marked.

![A matrix figure mapping sparsity levels against one-shot accuracy, iterative accuracy, and the regime, showing a flat plateau then a knee where one-shot collapses but iterative holds](/imgs/blogs/unstructured-pruning-and-the-lottery-ticket-5.png)

The honest caveat: these exact numbers are representative of the well-documented behavior of magnitude pruning on small CIFAR convnets (consistent with Frankle and Carbin's curves and the State of Sparsity study), not a measurement I am claiming to three significant figures for your specific model. The *shape* — flat plateau, knee around ninety percent, one-shot collapsing six-plus points earlier than iterative — is robust and is what you should expect and plan around. Your knee location will shift with model size and task: bigger, more over-parameterized models push the knee higher; small, tight models have an earlier knee.

What does the knee cost to *find*? Reading the IMP budget honestly: ten rounds of twenty-percent cuts, each with three fine-tune epochs, is thirty fine-tune epochs total — roughly comparable to a fresh training run of this network. On a single mid-range cloud GPU at about \$0.50 per hour, and with this small ConvNet's epochs taking on the order of a minute, the whole IMP-to-ninety-percent run is well under an hour and costs a fraction of a dollar — trivial. Scale that to a ResNet-50 on ImageNet, where one epoch can take the better part of an hour, and thirty fine-tune epochs becomes a multi-day, tens-of-dollars run; scale it to an LLM and IMP-with-retraining becomes flatly infeasible, which is exactly why the LLM world abandoned it for the no-retrain SparseGPT and Wanda methods. The lesson for budgeting: the *accuracy* curve is roughly model-size-independent in shape, but the *cost to ride it* scales with the price of one training run, and that price is what decides whether iterative pruning is even an option.

#### Stress test: what breaks the curve

Push on it. **Tiny model?** A network that is barely big enough for the task has little redundancy, so its knee is early — maybe fifty percent — and ninety-percent sparsity is simply not available without real accuracy loss. Pruning rewards over-parameterization; it punishes lean architectures. **First and last layers?** If you let global pruning touch them, the curve gets worse and noisier, because those layers are small and sensitive and global ranking over-cuts them — which is why the code exempts them. **Too-aggressive per-round step?** Prune fifty percent per round instead of twenty and the curve degrades toward the one-shot column; the gentleness of the step is part of why IMP works. **No gradient masking?** Forget `zero_pruned_grads` and weight decay slowly resurrects pruned weights, your realized sparsity silently drifts below target, and you ship a model that is denser than you think.

## 10. Worked example: the storage win and the latency non-win

This is the example that keeps you honest, and it is two halves: a real win and a real non-win, measured on a named target.

#### Worked example: 90 percent sparse on an M2 laptop CPU

Take the same ConvNet pruned to ninety percent sparsity and ask the two questions that matter: how much smaller, and how much faster, on an M2 MacBook CPU at batch 1?

**The storage half — a real win.** Start from the int8-quantized weights so the numbers are concrete (sparsity composes with quantization; you would normally do both). Suppose a layer has $N$ weights, fraction $s = 0.9$ pruned, so $0.1N$ nonzeros, each one byte after int8.

- **Dense int8:** $N$ bytes. For the whole 14 M-param model, about 14 MB.
- **Bitmask format:** store a 1-bit mask over all $N$ positions plus the nonzero values: $N/8$ bytes for the mask $+ \, 0.1N$ bytes for the values $= 0.225\,N$ bytes. That is about $4.4\times$ smaller — roughly 3.2 MB.
- **CSR format:** store the nonzero values ($0.1N$ bytes) plus column indices (about 2 bytes each for these layer widths, so $0.2N$ bytes) plus row pointers (negligible) $\approx 0.3\,N$ bytes. About $3.3\times$ smaller — roughly 4.2 MB. CSR is *worse* than the bitmask here because at ninety percent sparsity the per-nonzero index overhead (2 bytes) is larger than the bitmask's amortized cost (one bit per position). CSR wins only at *higher* sparsity, where there are so few nonzeros that even a 1-bit-per-position mask costs more than a sparse index list.

So the right format depends on the sparsity, and at the ninety-percent operating point the humble **bitmask is the storage winner**, at about $4.4\times$ smaller. That is a genuine, bankable reduction in download size and flash footprint. Figure 6 shows the three formats and the crossover.

![A vertical stack figure comparing a dense array, a bitmask plus values layout, and a CSR indices plus values layout for a ninety percent sparse tensor, with the storage win and the index-overhead caveat](/imgs/blogs/unstructured-pruning-and-the-lottery-ticket-6.png)

**The latency half — the non-win.** Now benchmark inference. Run the dense model and the ninety-percent-sparse model through the *same* PyTorch CPU path, batch 1, with proper methodology: warm up for fifty iterations to fill caches and let the CPU reach a steady clock, then time the median (p50) and tail (p99) over a few hundred runs, pinning to performance cores to avoid the efficiency-core noise the M-series schedulers introduce.

| Model | Storage (int8) | p50 latency | p99 latency | Test acc |
| ----- | -------------- | ----------- | ----------- | -------- |
| Dense | 14 MB | 18.2 ms | 21.0 ms | 93.0% |
| 90% unstructured sparse | ~3.2 MB (bitmask) | 18.1 ms | 21.3 ms | 92.4% |
| 90% sparse, DeepSparse runtime | ~3.2 MB | ~6–9 ms | ~10 ms | 92.4% |

Look at rows one and two. Storage dropped $4.4\times$. Latency did *not move* — 18.2 versus 18.1 ms is within measurement noise, and if anything the sparse model's p99 is a hair *worse* because the mask bookkeeping adds a touch of overhead with no kernel to exploit it. This is the whole point of the post made measurable: **you deleted ninety percent of the weights and the model runs at exactly the same speed**, because PyTorch's dense CPU kernel multiplies by every zero. The third row is the asterisk: swap in the DeepSparse runtime, which has sparsity-aware kernels, and the *same* sparse weights now run two to three times faster — proving the speedup was always available in principle, just not in a commodity dense kernel. The DeepSparse numbers are an order-of-magnitude illustration of its published CPU results, not a measurement of your model; the dense-versus-sparse no-change result, by contrast, is what you will reproduce on essentially any stock runtime.

#### Stress test: when does the non-win become a win, or get worse?

**Higher sparsity?** At ninety-eight or ninety-nine percent, even generic sparse kernels start to win, because so few multiplies remain that the index-chasing overhead is finally worth it — but you have paid for it in accuracy (the knee). **2:4 structured sparsity?** NVIDIA's Ampere-and-later tensor cores have native support for a *constrained* form of sparsity — exactly two nonzeros in every contiguous group of four — and deliver a real ~2× speedup for it. But 2:4 is a *structured* pattern (it constrains where zeros may fall), so it lives on the structured-pruning side of the fence even though it is fine-grained; it is the hardware meeting sparsity halfway. **Memory-bound layer?** If a layer is memory-bound (as most batch-1 edge layers are, per the roofline post) and your runtime keeps weights compressed in memory and decompresses on the fly, you *can* get a latency win from moving fewer bytes — but only if the runtime supports streaming sparse weights, which most do not. The default outcome on commodity hardware remains: smaller, not faster.

## 11. Results in the literature: real numbers

Four named results anchor the claims of this post to the published record.

**Frankle and Carbin, LTH 2019.** On a small convolutional network on MNIST and CIFAR-10, IMP found winning tickets at ten to twenty percent of the original weights that matched or *exceeded* the dense network's test accuracy, and — strikingly — often *trained faster* (reached the accuracy in fewer iterations) than the dense network, while the same sparse structure with reinitialized weights did markedly worse. This is the canonical evidence that the *initialization*, not just the structure, is the prize. The result was clean at small scale and is the reason the paper won a best-paper award and launched the subfield.

**Frankle, Dziugaite, Roy, and Carbin, rewinding 2020.** The follow-up showed that the original $\theta_0$ rewind *failed* to find winning tickets in larger networks (ResNet-50 on ImageNet, deep CIFAR networks), and that **rewinding to an early epoch** instead of to initialization restored the result at scale, explained through linear mode connectivity — runs become linearly mode connected after a short warmup, and rewinding past that point pins the sparse retraining to the dense network's basin. This is what made the LTH a practical large-scale technique rather than a small-scale curiosity, and it is why "rewind, don't reset to zero" is the operative advice.

**Han, Mao, and Dally, Deep Compression 2015.** The result that started the modern wave: prune (magnitude, iterative), then quantize, then Huffman-code the result. On AlexNet they reached about $9\times$ fewer parameters from pruning alone and roughly $35\times$ smaller *model size* after the full pipeline (pruning + quantization + coding), with no loss of accuracy on ImageNet — and they were explicit that the win was *storage and energy* via reduced memory access, not raw speed on standard hardware, which is the same honest framing this post insists on a decade later. The Gale, Elsen, and Hooker State of Sparsity study (2019) and the Hoefler, Alistarh, Ben-Nun, Dryden, and Peste survey "Sparsity in Deep Learning" (2021) are the two best places to see these results aggregated rigorously across architectures, and both reinforce that simple iterative magnitude pruning is a strong, hard-to-beat baseline and that unstructured speedups require specialized support.

**SparseGPT and Wanda on LLMs (2023).** The modern frontier moved the action to large language models, where the IMP-plus-fine-tune recipe of this post is *impossible* — you cannot afford to retrain a 7B-to-70B model after pruning. SparseGPT (Frantar and Alistarh, 2023) revived the Optimal Brain Surgeon idea at scale: it prunes one layer at a time, using a layer-local Hessian computed from a small calibration set to both choose which weights to drop and adjust the survivors to compensate, all in a single pass with *no* gradient-based retraining. It reaches around fifty percent unstructured sparsity (and the hardware-friendly 2:4 pattern) on models up to 175B parameters with only a small perplexity increase. Wanda (Sun et al., 2023) simplified it further to a beautiful one-line criterion: prune weights by $|W_{ij}| \cdot \lVert x_j \rVert$ — magnitude *times the norm of the corresponding input activation* — capturing the insight that a small weight on a large, frequent activation still matters. Wanda needs no Hessian and no weight update at all, yet roughly matches SparseGPT. Both are the no-fine-tune regime from section 2 made real: when you cannot retrain, the smarter input-aware criterion earns its keep — but notice that on commodity GPUs their *unstructured* fifty-percent sparsity still gives mainly a memory win, and the *speed* win comes specifically from the 2:4 structured pattern that tensor cores support.

Here is the whole landscape of unstructured-pruning approaches in one table, so you can place any method you encounter:

| Method | Criterion | Needs retrain? | Reaches | Speed on stock HW |
| ------ | --------- | -------------- | ------- | ----------------- |
| One-shot magnitude | $\lvert W \rvert$ | one fine-tune | ~60% | none |
| Iterative magnitude (IMP) | $\lvert W \rvert$, re-ranked | many fine-tunes | ~90%+ | none |
| LTH + rewinding | $\lvert W \rvert$ + epoch-$k$ init | full retrain | ~90%+ | none |
| Optimal Brain Surgeon | $\tfrac12 H_{ii} W_i^2$ | no (analytic) | moderate | none |
| SparseGPT | layer-local Hessian | no | ~50% / 2:4 | 2:4 only |
| Wanda | $\lvert W \rvert \cdot \lVert x \rVert$ | no | ~50% / 2:4 | 2:4 only |

The pattern across the whole table is the through-line of this post: every method gives you *size*, and *none* of them gives you *speed* on a commodity dense kernel — the speed only appears with a sparse runtime or a hardware-supported structured pattern.

## 12. When to reach for unstructured pruning — and when not to

Decisions, not vibes. Figure 8 is the decision tree; here is the reasoning behind it.

![A decision tree figure for whether to prune unstructured, branching on the bottleneck into storage-bound leading to unstructured pruning and compute-bound leading to a check for a sparse runtime then structured pruning](/imgs/blogs/unstructured-pruning-and-the-lottery-ticket-8.png)

**Reach for unstructured pruning when:**

- **You are storage- or download-bound.** App bundle too big, over-the-air model updates too expensive, flash too small. This is unstructured pruning's home turf — it gives the best accuracy-per-stored-byte of any technique, especially stacked with quantization. The bitmask format at ninety percent sparsity is a clean $4\times$-plus win for a fraction of a point of accuracy.
- **You have a sparsity-aware runtime** (DeepSparse on CPU, or you are targeting hardware with native sparse support and a matching pruning method). Then, and essentially only then, unstructured sparsity becomes a *speed* lever too.
- **You are memory-footprint-bound and your runtime streams sparse weights**, so the smaller working set genuinely reduces peak RAM and data-movement energy.
- **You want to study or exploit the lottery-ticket effect** — finding trainable sparse subnetworks, transfer of tickets across tasks, or sparse-from-the-start training research.

**Do not reach for it when:**

- **You need latency on commodity hardware.** This is the big one. Unstructured pruning will not speed up a stock phone NPU, a normal GPU's dense kernels, or PyTorch/ONNX CPU inference at any sparsity you can reach without wrecking accuracy. If speed is the goal, this is the wrong lever — go to [structured pruning that actually speeds things up](/blog/machine-learning/edge-ai/structured-pruning-that-actually-speeds-things-up), which removes whole channels and heads so a dense kernel runs faster on a genuinely smaller tensor.
- **Your model is small and tight.** Little redundancy means an early knee; you cannot reach high sparsity without paying real accuracy. Pruning is a tax on over-parameterization, and a lean model has little to give.
- **You cannot afford the IMP compute.** Ten rounds of cut-and-fine-tune is a real training budget. If you only need moderate compression, one-shot pruning to fifty or sixty percent, or just quantization, may get you there for far less compute.
- **A simpler lever already hits target.** If int8 quantization alone gets the model under your size budget, do that first — it is cheaper, more predictable, and gives a real speedup on integer hardware. Reach for pruning to go *further* than quantization can on size, or to compose with it.

The composition point deserves emphasis. Pruning and quantization stack cleanly and multiplicatively on *storage*: a ninety-percent-sparse, int8 model is roughly $4\times$ (sparsity, bitmask) times $4\times$ (int8 vs fp32) on bytes, the Deep Compression recipe. They do *not* stack on speed, because neither one speeds up a dense commodity kernel. The full sequencing of all four levers — which to apply first, how they interact, what each costs — is the subject of the capstone, [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).

## Key takeaways

1. **Unstructured magnitude pruning can delete 90 percent of weights for a fraction of a point of accuracy** — the over-parameterization that makes networks trainable is largely removable once they are trained.
2. **Magnitude is the workhorse criterion.** Rank by $|W|$, prune globally (pooling across layers), exempt the first and last layers. Simple beats clever, applied iteratively.
3. **Iterative beats one-shot, decisively.** Cut ~20 percent per round and fine-tune between rounds. Each small step stays in the regime where magnitude pruning is justified, and the ranking re-adapts after each cut. One-shot collapses six-plus points earlier at the same sparsity.
4. **The Lottery Ticket Hypothesis:** a dense network contains a sparse subnetwork that, trained from its *original initialization* under the discovered mask, matches the dense network. The structure *and* its init are both the prize; reinitializing the same structure fails.
5. **Rewind, do not reset to zero.** Original-init rewinding breaks at scale; rewinding surviving weights to an early epoch $k$ — after the network becomes linearly mode connected — finds winning tickets in large networks.
6. **The knee is your target.** Accuracy is flat to a sparsity knee (often ~90 percent), then falls off. Prune to just before the knee. Bigger models push the knee higher.
7. **Smaller, not faster.** Unstructured sparsity gives storage and memory wins (bitmask/CSR), not speedups on commodity dense kernels — they multiply by the zeros. Always *measure* the latency and watch it not move.
8. **The exceptions:** a sparsity-aware runtime (DeepSparse) turns unstructured sparsity into a speed lever; extreme (>95 percent) sparsity or hardware-native patterns (2:4) can speed up too. The default on stock hardware does not.
9. **For speed, use structured pruning instead.** If latency is the goal, remove whole channels/heads so a dense kernel runs faster on a smaller tensor.
10. **Compose with quantization for storage** (multiplicative on bytes), not for speed. Reach for unstructured pruning to push *size* beyond what quantization alone delivers.

## Further reading

- **Jonathan Frankle and Michael Carbin, "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" (ICLR 2019)** — the paper that defined winning tickets and the train-prune-reset experiment.
- **Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, Michael Carbin, "Linear Mode Connectivity and the Lottery Ticket Hypothesis" (ICML 2020)** and "Stabilizing the Lottery Ticket Hypothesis" — the rewinding fix and the linear-mode-connectivity explanation.
- **Alex Renda, Jonathan Frankle, Michael Carbin, "Comparing Rewinding and Fine-tuning in Neural Network Pruning" (ICLR 2020)** — when learning-rate and weight rewinding beat plain fine-tuning.
- **Song Han, Huizi Mao, William J. Dally, "Deep Compression" (ICLR 2016)** and Han et al., "Learning both Weights and Connections for Efficient Neural Networks" (NeurIPS 2015) — pruning + quantization + coding, and the honest storage-and-energy framing.
- **Trevor Gale, Erich Elsen, Sara Hooker, "The State of Sparsity in Deep Neural Networks" (2019)** — the careful large-scale comparison showing magnitude pruning is a strong baseline.
- **Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, Alexandra Peste, "Sparsity in Deep Learning: Pruning and growth for efficient inference and training" (JMLR 2021)** — the comprehensive survey of the whole field.
- Official docs: [`torch.nn.utils.prune`](https://pytorch.org/docs/stable/nn.html#module-torch.nn.utils.prune) for the PyTorch pruning API, and the Neural Magic DeepSparse documentation for the sparsity-aware CPU runtime.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever frame, [pruning fundamentals](/blog/machine-learning/edge-ai/pruning-fundamentals) for the shared groundwork, [structured pruning that actually speeds things up](/blog/machine-learning/edge-ai/structured-pruning-that-actually-speeds-things-up) for the latency lever, [the roofline model and where your bottleneck lives](/blog/machine-learning/edge-ai/the-roofline-model-where-your-bottleneck-lives) for why zeros do not buy speed, and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for how pruning composes with the other levers.

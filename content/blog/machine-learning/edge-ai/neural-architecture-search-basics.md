---
title: "Neural architecture search basics: search space, strategy, and DARTS"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "MobileNet took human researchers years to hand-design; NAS automated that search and found the architectures — MobileNetV3, EfficientNet — that now define efficient deep learning. Here is how NAS actually works: the search space, the three search strategies, the DARTS continuous relaxation and its bilevel optimization, weight-sharing supernets, and a runnable mixed-operation cell in PyTorch."
tags:
  [
    "edge-ai",
    "model-optimization",
    "neural-architecture-search",
    "nas",
    "darts",
    "efficient-architectures",
    "inference",
    "efficient-ml",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/neural-architecture-search-basics-1.png"
---

For about five years, the way you got an efficient neural network was that a small number of very good researchers stared at the problem until a better building block fell out. Depthwise separable convolutions, inverted residual blocks, channel shuffles, squeeze-and-excitation — each of these was a human insight, argued over in a paper, validated by hand across dozens of training runs, and then copied into everyone else's models. The MobileNet line (which this series covers separately in [the MobileNet family](/blog/machine-learning/edge-ai/the-mobilenet-family)) is the cleanest example: V1 introduced depthwise separable convolutions, V2 added the inverted residual, and each step took a research team months of design, ablation, and intuition. It worked, and it was slow, expensive, human work that did not obviously generalize to the next task or the next chip.

Then a strange thing happened. A method called neural architecture search — NAS — started *automatically discovering* architectures that beat the hand-designed ones. The two architectures that arguably define modern efficient deep learning, MobileNetV3 and EfficientNet, were not hand-designed in the way V1 and V2 were. They came out of NAS. The human-versus-machine framing is too cute, but the substance is real: a search algorithm, given a space of possible architectures and a way to score them, found designs that human experts had not, and those designs are what ship on your phone today. If you want to understand where efficient architectures come from now, you have to understand NAS, and that is what this post is for.

I want to set expectations precisely, because NAS has a reputation for being either magic or a money pit, and both reputations are earned. The original NAS papers really did cost thousands of GPU-days — a small data center running for weeks to search one architecture for one dataset. That is not a typo and it is not an exaggeration; we will do the arithmetic. But within two years the field collapsed that cost by roughly three orders of magnitude through one idea — weight sharing — and a differentiable method called DARTS brought a full architecture search down to something you could run on a single GPU overnight. That cost collapse is the most important story in NAS, and figure 1 frames the whole thing: every NAS method, expensive or cheap, is the same three-part loop, and the cost lives almost entirely in one of the three parts.

![A dataflow graph showing the three components of neural architecture search as a feedback loop where the search space defines what the strategy can sample, the strategy proposes a candidate architecture, the performance estimator scores it on validation accuracy, and the score feeds back to the strategy while the estimation step dominates the total cost](/imgs/blogs/neural-architecture-search-basics-1.png)

By the end of this post you will be able to do five concrete things. First, decompose any NAS method into its three components — search space, search strategy, and performance estimation — using the framing from Elsken, Metzen, and Hutter's 2019 survey, and explain why the search space is where you encode your priors and the estimation step is where your money goes. Second, count the size of a cell-based search space yourself and feel exactly why brute force is hopeless. Third, write down and explain the DARTS continuous relaxation — the softmax over candidate operations that turns a discrete architecture choice into a differentiable one — and the bilevel optimization it implies, including the first-order approximation that makes it cheap. Fourth, write a runnable PyTorch mixed-operation cell with architecture parameters $\alpha$, the actual core of a DARTS implementation. Fifth, decide, for a real project, whether to run NAS at all or just reuse a searched architecture — because for most teams the right answer is to reuse, and knowing why is more valuable than knowing how to run a search. This post is the *accuracy-first* story of NAS; the hardware-aware twist that makes NAS optimize for on-device latency rather than FLOPs is a sequel, and I forward-reference [hardware-aware NAS](/blog/machine-learning/edge-ai/hardware-aware-nas) throughout because it is the piece that makes NAS matter for the edge specifically.

## 1. NAS in one frame: three components, one loop

The cleanest way to think about every NAS method ever published — and there are hundreds — is the decomposition that Elsken, Metzen, and Hutter put at the front of their 2019 survey. A NAS method is exactly three things wired into a loop, and figure 1 is that loop. There is a **search space**: the set of all architectures the method is allowed to consider. There is a **search strategy**: the algorithm that decides which architectures from that space to actually try. And there is a **performance estimation strategy**: the way you assign a score to a candidate architecture so the search strategy knows whether it is good. The loop is: the strategy samples an architecture from the space, the estimator scores it, the score updates the strategy, repeat until you are out of budget, then return the best architecture you found.

That decomposition is not just tidy taxonomy; it tells you where every design decision and every dollar lives, and it lets you compare wildly different methods on a level field. The reinforcement-learning NAS of Zoph and Le, the evolutionary AmoebaNet of Real et al., and the gradient-based DARTS of Liu et al. look completely different on the surface — one trains a recurrent controller with policy gradients, one mutates a population, one does gradient descent on a relaxed architecture — but they are the *same loop* with three different choices plugged into the three slots. Once you internalize that, NAS stops being a zoo of unrelated tricks and becomes a small design space you can reason about.

The single most important thing the decomposition reveals is *where the cost is*. Look at the loop and ask: which step is expensive? Sampling an architecture from the space is nearly free — it is a few random choices or a forward pass of a controller. Updating the search strategy from a score is cheap — a gradient step on a small controller, or a population sort. The expensive step, by a margin so large it dwarfs everything else, is **performance estimation**: figuring out how good a candidate architecture actually is. The honest way to score an architecture is to train it on your data and measure its validation accuracy, and training a deep network is the single most expensive thing in deep learning. If the search strategy proposes ten thousand candidates and you train each one, you have just trained ten thousand networks. That is the entire reason early NAS cost thousands of GPU-days, and it is why almost all NAS research since has been about one question: *how do you estimate an architecture's quality without fully training it?*

So the rest of this post is organized around the three components, but weighted by where the interesting decisions are. The search space is where you encode your design priors — get it wrong and no strategy can save you. The search strategy is the part that gets the headlines (RL versus evolution versus gradient) but matters less than people think once the space is good. And performance estimation is where the cost lives and where the cleverest ideas — weight sharing, supernets, proxies — have had the biggest impact. We will take them in that order, then write code, then work through the numbers, then talk about when you should actually do this.

One framing to carry through, because it connects NAS to the rest of this series. NAS is the fourth of the four levers this series keeps returning to — quantization, pruning, distillation, and efficient architecture — and it is the one that operates *before* a model exists rather than on a model you already have. Quantization shrinks the numbers in a trained network; pruning deletes its weights; distillation trains a small model to mimic a big one. NAS designs the small model's *shape* in the first place. It sits at the top of the stack in [the taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), and the architectures it produces are the dense starting points that the other three levers then compress further. A NAS-found MobileNetV3 that you then quantize to int8 and prune is the four levers composing exactly as the series promises.

## 2. The search space: what architectures are even reachable

The search space is the most consequential and most underrated component, because it bounds everything. The search strategy can only find architectures that exist in the space; if a great design is not expressible in your space, no amount of search will discover it. Conversely, if you bake too much human prior into the space, you have just hand-designed the architecture and let the search tweak the margins — which, to be fair, is sometimes exactly what you want. The space is where the human's job moved to: you no longer design the architecture, you design the *space of architectures*, and that is a strictly easier and more transferable job.

There are three broad families of search space, and figure 2 contrasts the two that matter most. The first is the **global** or **chain-structured** space. Here the architecture is a sequence of layers and the search picks, for each layer, its type (conv, pooling, etc.), its hyperparameters (kernel size, number of filters, stride), and possibly skip connections to earlier layers. This is the most expressive space — almost any feedforward network is reachable — and it is also the most enormous. With $n$ layers, each chosen from $k$ operations, plus skip-connection choices, the space size is astronomical, and worse, a network found for a 20-layer budget tells you nothing about a 40-layer budget. You search from scratch every time.

The second family, and the one that made NAS practical, is the **cell-based** (or micro) search space. The insight, due to the NASNet work of Zoph et al., is that good hand-designed networks are *repetitive*: ResNet is the same residual block stacked dozens of times; Inception is the same module repeated. So instead of searching the whole network, you search one small **cell** — a little graph of a handful of operations — and then build the network by stacking copies of that cell according to a fixed *macro* skeleton. You typically search two cell types: a **normal cell** that preserves spatial resolution and a **reduction cell** that downsamples. The macro architecture — how many cells, where the reductions go, the stem and the head — is fixed by hand. This decomposition into a searched *micro* structure and a fixed *macro* structure is the central design pattern of modern NAS.

![A before-and-after comparison contrasting a global chain search space where every layer is chosen freely and the space reaches about ten to the twenty-eight architectures with no reuse, against a cell-based search space where one small cell of seven nodes and eight operations is searched and stacked, giving about ten to the ninth cells that transfer across network depth](/imgs/blogs/neural-architecture-search-basics-2.png)

The payoff of cell search is twofold, and figure 2 captures both. The space shrinks dramatically — you are choosing a small graph instead of a whole network — which makes search tractable. And the result *transfers*: a cell searched on a small proxy dataset and a shallow network can be stacked more times to build a deeper, more accurate network for the real task, or transferred to a different dataset entirely. NASNet famously searched cells on CIFAR-10 and transferred them to ImageNet by stacking more copies and widening, and the transferred architecture was state of the art. That transfer is only possible because the searched object — the cell — is decoupled from the network depth.

The third family is the **hierarchical** search space, which sits between the two. Instead of one flat cell, you define a hierarchy: small motifs are assembled into larger motifs which are assembled into cells. This gives more expressivity than a single flat cell while keeping the combinatorial benefits of reuse, and it is what the AmoebaNet hierarchical-genetic work explored. In practice the cell-based space is the workhorse and the one you should default to mentally when someone says "NAS search space," so we will use it as our running example.

It is worth being precise about the division of labor between the *macro* skeleton and the *micro* cell, because that division is where a surprising amount of human prior still lives and where a lot of NAS critiques land. In a cell-based search the macro architecture — the stem convolution that ingests the image, the number of cells, the positions where a reduction cell halves the spatial resolution and doubles the channels, the global pooling, and the classifier head — is *fixed by a human*, usually copied straight from a known-good template like the ResNet or NASNet skeleton. The search only chooses the *contents* of the normal and reduction cells. This is exactly why critics say the search space "does most of the work": the human picks a skeleton that is already known to train well and already roughly the right size, and the search is left to tune the internals. That is not a flaw so much as a deliberate trade — by fixing the macro structure you cut the search space by many orders of magnitude and you inherit decades of human knowledge about what depth-and-downsampling schedules work, which is precisely what makes the search affordable and the result transferable. But it does mean you should never describe NAS as "designing the whole architecture from scratch"; it designs the *cell*, inside a skeleton you handed it.

The transfer property deserves one more beat because it is the economic engine of cell search. Because the searched object is a small cell decoupled from network depth, you can run the *expensive* search once on a cheap proxy — a shallow network (say, 8 cells) on a small dataset (CIFAR-10) at low resolution — and then deploy the discovered cell at *any* scale by stacking more copies and widening the channels for the real, large task (a 20-cell network on ImageNet). The search cost is paid at proxy scale; the deployment happens at full scale; and the cell carries over because its job (a good local feature-mixing block) is largely scale-invariant. This proxy-search-then-scale pattern is what made NASNet's ImageNet result affordable at all — they never searched on ImageNet directly — and it is the same logic EfficientNet later formalized as compound scaling. When you reuse a published searched architecture, you are cashing in exactly this transfer: someone searched the cell on a proxy, you stack it at whatever scale your task needs.

Let me make the cell concrete, because the worked example later depends on it. A DARTS-style cell is a small directed acyclic graph. It has two input nodes (the outputs of the two previous cells), some number of intermediate nodes — DARTS uses four — and one output node that concatenates the intermediate nodes. Each intermediate node takes input from two earlier nodes in the cell, and on each of those two connections you choose one **operation** from a fixed set: things like a $3\times3$ separable convolution, a $5\times5$ separable convolution, a $3\times3$ dilated convolution, $3\times3$ max pooling, $3\times3$ average pooling, a skip connection (identity), and a "zero" operation that means "no connection." So designing a cell is: for each intermediate node, pick which two earlier nodes it connects to, and for each of those connections, pick one of the eight operations. That is the discrete choice the search is over, and it is small enough to search and rich enough to contain genuinely good architectures.

#### Worked example: how big is one cell's search space

Let me count, because the number is the whole point. Take the standard DARTS cell: two input nodes (call them node 0 and node 1) and four intermediate nodes (nodes 2, 3, 4, 5). Each intermediate node picks two predecessors from the nodes before it and assigns an operation to each. Use $|O| = 8$ candidate operations (the seven I listed plus zero, minus identity-as-zero collapsing — eight is the canonical DARTS count).

Node 2 can connect to nodes $\{0, 1\}$: it must pick 2 predecessors from 2 available, so $\binom{2}{2}=1$ way to choose the pair, and each of the 2 edges gets one of 8 ops, giving $8^2 = 64$. Node 3 can connect to $\{0,1,2\}$: $\binom{3}{2}=3$ ways to pick the pair, times $8^2$ for the ops, giving $3 \cdot 64 = 192$. Node 4 sees $\{0,1,2,3\}$: $\binom{4}{2}=6$, times $64$, giving $384$. Node 5 sees $\{0,1,2,3,4\}$: $\binom{5}{2}=10$, times $64$, giving $640$. Multiply across the four intermediate nodes:

$$
64 \times 192 \times 384 \times 640 \approx 3.0 \times 10^9.
$$

That is *one* cell type. DARTS searches a normal cell and a reduction cell independently, so the joint space is roughly $(3 \times 10^9)^2 \approx 10^{19}$ architectures. And this is the *compact* space — the one that made NAS tractable. A global chain space over twenty-plus freely-chosen layers blows past $10^{28}$. The lesson lands hard: even the "small" cell space has billions of architectures, so you obviously cannot enumerate it. Every NAS strategy is a way to navigate $10^9$ to $10^{19}$ candidates while training as few of them as possible. Hold that number — three billion cells, training each one is hopeless — because it is what every search strategy is fighting against.

## 3. Search strategy I: reinforcement learning (Zoph and Le)

The paper that started the modern NAS wave is Zoph and Le's 2017 "Neural Architecture Search with Reinforcement Learning." Their idea is elegant and expensive in equal measure, and it is worth understanding in detail because it defines the cost baseline that everything after it improved on. Figure 3 is the loop.

The search strategy is a **controller**, implemented as a recurrent neural network (an RNN). The controller generates an architecture as a *sequence of tokens*: it emits the number of filters for layer 1, then the kernel height, then the kernel width, then the stride, then moves to layer 2, and so on, sampling each token from its current output distribution. The full sequence of sampled tokens describes a complete child network. Because the controller is sampling from probability distributions, it produces a *different* architecture each time, and as training proceeds it learns to put more probability mass on the token choices that lead to good networks.

How does it learn? This is where the reinforcement-learning framing comes in. Each sampled architecture — call it action $a$ drawn from the controller's policy $\pi_\theta$ with parameters $\theta$ — is trained on the training set until convergence and then measured on a held-out validation set. The resulting **validation accuracy $R$ is the reward**. The controller's job is to maximize the expected reward of the architectures it samples, and the objective is the standard policy-gradient (REINFORCE) objective:

$$
J(\theta) = \mathbb{E}_{a \sim \pi_\theta}\big[ R(a) \big].
$$

You cannot differentiate $R$ with respect to $\theta$ directly — accuracy is a black-box function of a fully trained network, with no gradient back to the controller. So you use the REINFORCE estimator, which gives an unbiased gradient of the expectation:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{a \sim \pi_\theta}\big[ (R(a) - b)\, \nabla_\theta \log \pi_\theta(a) \big],
$$

where $b$ is a baseline (typically a running average of recent rewards) subtracted to reduce the variance of the estimate. In words: for each sampled architecture, compute how much better than average its reward was, and nudge the controller's parameters to make the token choices that produced that architecture more likely in proportion. Good architectures get reinforced; bad ones get suppressed. Over thousands of samples the controller drifts toward high-accuracy regions of the space.

![A dataflow graph showing the reinforcement-learning NAS loop where a controller RNN samples a child network architecture, the child is trained for a few epochs and its validation accuracy becomes the reward, that reward updates the controller by policy gradient while the training of children drives the roughly two thousand GPU-day cost](/imgs/blogs/neural-architecture-search-basics-3.png)

This works, and it works well: the original NAS found a recurrent cell that beat the best human-designed LSTM variants on Penn Treebank, and the cell-based NASNet follow-up found image cells that were state of the art on ImageNet. The architectures were genuinely good. The problem was the bill. Each reward evaluation requires *training a child network*, and the controller needs to see thousands of architectures to learn. The original NASNet search trained roughly 12,800 child architectures. Even with each child trained on a small proxy (CIFAR-10, a reduced number of epochs) and the search distributed across hundreds of GPUs, the total came to something on the order of 2,000 GPU-days — and some configurations of the original work were reported in the range of hundreds of GPUs running for weeks, which is where the "thousands of GPU-days" figures come from. The exact number depends on the configuration, so treat these as order-of-magnitude; the point is unambiguous: this was a method only a handful of organizations on earth could afford to run.

The variance problem compounds the cost. REINFORCE gradients are noisy — a single architecture's reward is one sample of a high-variance random variable — so you need many samples to get a reliable signal, which means many trained children, which means more cost. The baseline $b$ helps, but it does not change the fundamental arithmetic: RL-NAS pays full training cost per candidate, and it needs a lot of candidates. Every improvement after this — evolution, then gradient methods, then weight sharing — is an attack on that arithmetic.

## 4. Search strategy II: evolution (AmoebaNet)

The second strategy treats architecture search as **evolution**, and the landmark result is Real et al.'s 2019 "Regularized Evolution for Image Classifier Architecture Search," which produced AmoebaNet. The framing is intuitive: maintain a *population* of architectures, and improve it the way biological evolution improves a species — by mutation and selection.

The algorithm is a loop. Initialize a population of $P$ random architectures and train each to get its accuracy (its "fitness"). Then repeat: sample a small subset of the population (a "tournament" of, say, 25), pick the highest-accuracy architecture in that subset as the **parent**, create a **child** by applying a small random **mutation** to the parent — change one operation in the cell, or rewire one connection — train the child, and add it to the population. The crucial detail in *regularized* evolution is how you make room: instead of removing the worst architecture (standard tournament selection), you remove the *oldest* one. This "aging" regularization means an architecture cannot survive on a single lucky high-accuracy training run forever; to persist in the population, its lineage must keep producing good children. That aging trick is what made evolution match and slightly exceed RL on the same search space.

Mutations are deliberately small and local, which is what makes evolution a sensible *local search* over the architecture graph. A typical mutation in AmoebaNet is one of: pick a random edge in the cell and change its operation to a different one from the candidate set, or pick a random intermediate node and rewire one of its two inputs to a different predecessor. Each mutation is a single step in the discrete architecture space, so the population performs a parallel hill-climb, with selection pressure pulling it toward higher accuracy and aging pushing it to keep exploring rather than camp on one good cell.

The reason small, local mutations are the right move (rather than large random jumps) is the same reason local search beats global random sampling on any structured landscape: good architectures cluster. If a cell is strong, a cell one mutation away is usually also strong, because most of its structure is preserved and only one edge changed. Large mutations destroy that locality — they are effectively random restarts, and a population of random restarts is just random search, which we already know is a weak (if surprisingly competitive) baseline. So evolution's mutation size is a deliberate exploration-exploitation knob: small enough to exploit the neighborhood of good cells, large enough (one full edge change) to escape a local plateau. Aging regularization adds the exploration pressure from the other side, by preventing any single lucky cell from monopolizing the population, so the two mechanisms together keep the population both climbing and moving.

How does it compare to RL? Real et al. ran the controlled experiment — same search space, same compute budget — and found evolution reached slightly higher accuracy *faster* than RL early in the search and matched it at convergence, while being conceptually simpler (no controller, no policy gradient, just mutate-and-select). But — and this is the headline that matters — evolution shares RL's fatal cost structure. Every architecture in the population, and every child ever produced, must be *trained* to get its fitness. AmoebaNet's search was on the same order as NASNet's: thousands of GPU-days. Some reported figures put the AmoebaNet search around 3,150 GPU-days. Evolution did not solve the cost problem; it was a better search *strategy* on top of the same brutally expensive *estimation* strategy. The cost is in the estimation step, exactly as figure 1 warned, and changing the strategy from RL to evolution does not touch it.

This is the right moment to state the central lesson of NAS bluntly. **The search strategy matters much less than people assume; the performance estimation strategy is everything.** RL versus evolution is a real but second-order question — they land in roughly the same place. The order-of-magnitude wins came not from a smarter strategy but from a cheaper way to *score* candidates, which is the subject of the next two sections.

## 5. Search strategy III: DARTS and the continuous relaxation

Here is the conceptual leap that changed NAS. RL and evolution both treat the architecture as a *discrete* object and the score as a *black box*: you sample a discrete architecture, you train it, you get a number, you have no gradient connecting the architecture choice to the number. DARTS — Liu, Simonyan, and Yang's 2019 "DARTS: Differentiable Architecture Search" — asks: what if we made the architecture *continuous*, so that the discrete choice of which operation to use becomes a differentiable parameter we can optimize with gradient descent, jointly with the network weights? Figure 4 is the trick.

The discrete problem is: at each edge of the cell, choose one operation $o$ from the candidate set $O = \{o_1, \dots, o_K\}$. That choice is an argmax — pick the best $o$ — and argmax is not differentiable; you cannot take a gradient through "pick one." DARTS relaxes it. Instead of picking one operation on an edge, *apply all of them and take a weighted average*, where the weights come from a softmax over a vector of learnable **architecture parameters** $\alpha$. Concretely, for an edge from node $i$ to node $j$, define the **mixed operation**:

$$
\bar{o}^{(i,j)}(x) = \sum_{o \in O} \frac{\exp(\alpha^{(i,j)}_o)}{\sum_{o' \in O} \exp(\alpha^{(i,j)}_{o'})}\; o(x).
$$

Read that carefully. On every edge there is a vector $\alpha^{(i,j)} \in \mathbb{R}^{K}$, one weight per candidate operation. The softmax of that vector gives a probability-like weight to each operation, and the edge's output is the weighted sum of *all* candidate operations applied to the input $x$. When one $\alpha_o$ dominates, the softmax approaches a one-hot vector and the mixed operation approaches the discrete choice of that single operation. So the discrete architecture is the limit of the continuous one as the softmaxes sharpen. Crucially, the mixed operation is fully differentiable in $\alpha$ — it is a softmax and a weighted sum — so you can compute $\partial \mathcal{L} / \partial \alpha$ and do gradient descent on the architecture itself.

![A before-and-after comparison contrasting a discrete operation choice that requires an argmax over eight operations and has no gradient and must be sampled by a slow black-box search, against the DARTS continuous relaxation where a softmax over the operations with weights alpha produces a mixed-operation output as a weighted sum that is differentiable and trainable by stochastic gradient descent](/imgs/blogs/neural-architecture-search-basics-4.png)

Now there are two sets of parameters to learn: the **network weights** $w$ (the actual convolution kernels, etc.) and the **architecture parameters** $\alpha$ (the operation weights). They want different objectives. The weights $w$ should minimize the *training* loss — that is just normal training. But the architecture $\alpha$ should minimize the *validation* loss — you want the architecture that generalizes, and optimizing $\alpha$ on the training loss would just overfit the architecture to the training set (it would, for example, prefer the most expressive operations regardless of generalization). This sets up a **bilevel optimization**:

$$
\min_{\alpha}\; \mathcal{L}_{val}\big(w^*(\alpha),\, \alpha\big) \quad\text{s.t.}\quad w^*(\alpha) = \arg\min_{w}\; \mathcal{L}_{train}(w, \alpha).
$$

The outer problem optimizes $\alpha$ on validation loss; the inner problem optimizes $w$ on training loss *given* $\alpha$. The architecture is good if, when you train the weights to optimality under that architecture, the resulting network generalizes. This is the heart of DARTS and it is worth pausing on: it formalizes the intuition that you should choose the architecture by how well it *can be trained to generalize*, not by how low a training loss it can reach.

There is a clean intuition for why the split between training loss and validation loss is not a technicality but the whole point. If you optimized *both* $w$ and $\alpha$ on the training loss, the search would simply pick whatever architecture can drive the training loss lowest, and that is the most over-parameterized, most expressive architecture — the one most prone to overfitting. The architecture choice would collapse to "use the biggest, most flexible operations everywhere," which is useless as a generalization signal. By optimizing $\alpha$ on a *held-out* validation set while $w$ trains on the training set, you ask a sharper question: among architectures, which one, once its weights are trained, *generalizes* best? That is the question you actually care about for deployment. The bilevel structure is the mathematical encoding of "tune the model to the train set, but choose the model by the validation set" — the same discipline you already apply to hyperparameters, lifted to the architecture itself and made differentiable.

The catch is the constraint $w^*(\alpha) = \arg\min_w \mathcal{L}_{train}$. Evaluating it exactly means fully training the weights to convergence for every $\alpha$ step — which is the expensive thing we were trying to avoid. DARTS approximates. Instead of fully solving the inner problem, it takes a *single gradient step* on $w$ and treats the result as a proxy for $w^*(\alpha)$. This gives the **second-order** approximation, where the $\alpha$ gradient accounts for how that one weight step depends on $\alpha$:

$$
\nabla_\alpha \mathcal{L}_{val}\big(w - \xi\, \nabla_w \mathcal{L}_{train}(w,\alpha),\, \alpha\big),
$$

with $\xi$ the inner learning rate. Expanding this with the chain rule produces a term involving a Hessian-vector product (second derivatives), which is computable but adds cost. The practical shortcut, and the one most people actually use, is the **first-order approximation**: set $\xi = 0$, which means you ignore how $w$ depends on $\alpha$ and simply alternate — take a gradient step on $w$ using $\mathcal{L}_{train}$, then take a gradient step on $\alpha$ using $\mathcal{L}_{val}$, back and forth. The first-order version is roughly twice as fast (no Hessian-vector product), at a small cost in final accuracy. In a lot of practice, first-order DARTS is the default because the speed is worth it.

Once the search converges, you **discretize**: for each edge, replace the mixed operation with the single operation that has the largest $\alpha$ weight (the argmax you relaxed away), keep the top-2 incoming edges per node, and you have a discrete cell. Then you build the final network by stacking that cell and *retrain from scratch* — the weights learned during search were entangled with all the other operations, so the final architecture is trained fresh. The whole search, on CIFAR-10, runs in roughly **1 to 4 GPU-days on a single GPU**, versus the thousands of GPU-days RL and evolution needed. That is the cost collapse, and it came entirely from changing the estimation strategy: instead of training thousands of discrete networks, DARTS trains *one* continuous super-network and reads the architecture off its $\alpha$ values.

## 6. Performance estimation: the cost problem and weight sharing

Step back and look at what DARTS actually did to the cost, because it generalizes beyond DARTS. The expensive thing in NAS is performance estimation — scoring candidates. There are a few families of shortcut, and they are worth naming because you will mix and match them.

**Lower-fidelity proxies.** Train each candidate for fewer epochs, or on a subsample of the data, or at lower resolution, or with a smaller version of the network (fewer cells, fewer channels), and use that cheap score as a proxy for the true score. This is what NASNet did — it searched on CIFAR-10 with reduced training, not on full ImageNet — and it is the most obvious lever. The risk is *rank instability*: the architecture that wins at 10 epochs is not always the one that wins at 600 epochs, and the architecture that is best with 4 cells is not always best with 20. Low-fidelity proxies trade accuracy of the estimate for cost, and you have to validate that the cheap ranking correlates with the expensive one.

**Learning-curve extrapolation.** Train a candidate a little, watch the early part of its learning curve, and *predict* where it will end up, killing the ones that look doomed early. This is a form of early stopping with a predictive model on top, and it saves the cost of finishing training on losers.

**Weight inheritance and network morphisms.** When you mutate a parent into a child (as in evolution), do not retrain the child from scratch — *inherit* the parent's weights and only fine-tune the changed parts. Network morphisms make the child mathematically equivalent to the parent at initialization, so it starts from the parent's accuracy and only needs a little training to absorb the change. This turns "train each child" into "fine-tune each child," a big saving.

**Weight sharing / one-shot NAS.** This is the big one, the idea that collapsed the cost by orders of magnitude, and it is what DARTS and ENAS are both instances of. The idea: instead of training thousands of separate networks, train *one* big over-parameterized network — a **supernet** — that contains every candidate architecture as a *subnetwork* sharing weights. Every subnet you want to evaluate is just a path through the supernet, and it inherits the supernet's already-trained weights, so evaluating a candidate costs an inference pass, not a training run. Figure 5 is this idea.

![A dataflow graph showing one-shot weight sharing where a single supernet contains all operations and is trained once to produce shared weights, then three different subnets each inherit those shared weights and can be evaluated almost for free without retraining, with a coupling risk that the shared weights cause subnets to interfere with each other](/imgs/blogs/neural-architecture-search-basics-5.png)

ENAS — Pham et al.'s 2018 "Efficient Neural Architecture Search via Parameter Sharing" — is the clearest statement of the supernet idea in the RL framing. ENAS keeps Zoph and Le's controller RNN, but instead of training each sampled child network from scratch, all child networks *share* one set of weights stored in a single large computation graph (the supernet). The controller samples a subgraph (a child architecture); that child runs using the shared weights, computes a reward, updates the controller; periodically the shared weights themselves get a gradient update over sampled children. Because nothing is trained from scratch, ENAS ran the *same* search that NASNet needed thousands of GPU-days for in **under a day on a single GPU** — roughly a thousand-fold speedup, from the same shared-weights insight that DARTS uses. DARTS is the differentiable cousin: its supernet is the cell with all mixed operations active, and "sampling a subnet" becomes "reading off the argmax of $\alpha$."

Weight sharing is the most important idea in practical NAS, so it is worth being honest about its cost. The reason it is fast is also the reason it is fragile: all subnets share weights, so the shared weights are a *compromise* trained to make many architectures work at once, not any single one work optimally. This is the **weight-coupling** problem, the caution node in figure 5. A subnet's accuracy *using the shared weights* can be a poor predictor of its accuracy *trained standalone*, because the shared weights were co-adapted to all the other operations it was averaged with. The architecture the supernet says is best is sometimes not the architecture that is actually best when retrained from scratch. This rank-correlation gap is the central weakness of one-shot NAS, and a lot of follow-up research (single-path one-shot, fair sampling, progressive shrinking) is about making the supernet's ranking trustworthy. When you read that a NAS result "didn't reproduce," weight coupling is usually the culprit.

#### Worked example: the GPU-day collapse, in dollars

Let me put numbers on the cost collapse, because it is the single most actionable fact in this post. Take the RL-NAS baseline: roughly 2,000 GPU-days to search one architecture for one dataset. On a cloud V100/A100-class instance at, very roughly, \$2 per GPU-hour, 2,000 GPU-days is $2000 \times 24 = 48{,}000$ GPU-hours, which is about \$96,000 of compute — call it \$50k–\$150k depending on the chip and the discount, for *one search*. AmoebaNet at ~3,150 GPU-days is even more. That is why only a few labs ran early NAS.

Now DARTS: roughly 1–4 GPU-days on a single GPU. Take 2 GPU-days = 48 GPU-hours $\approx$ \$96 of compute. The ratio is $2000 / 2 = 1000\times$ cheaper. A search that cost the price of a car now costs the price of dinner, and it runs on one GPU overnight instead of a cluster for a month. Figure 7 visualizes this collapse. The headline is not "DARTS is a bit faster" — it is "weight sharing made architecture search go from a megacorp capital expense to something a graduate student runs on a single rented GPU." Every democratization of NAS traces back to this one idea, and it is why NAS is now a tool you can actually consider for your own project rather than a thing other people do.

![A before-and-after comparison of NAS search cost showing reinforcement-learning NAS training roughly twelve thousand eight hundred child networks at about two thousand GPU-days costing tens of thousands of dollars per search, versus DARTS training one supernet with shared weights at one to four GPU-days on a single GPU, roughly a thousand times cheaper at laptop scale](/imgs/blogs/neural-architecture-search-basics-7.png)

There is a subtler point hiding in that arithmetic, and it is worth dwelling on because it changes how you should budget a NAS project. The GPU-days that DARTS saves are *search* GPU-days — the cost of finding the architecture. They are not the only cost. After the search finishes and you discretize the cell, you still have to *train the final architecture from scratch* on your real dataset at full resolution and full schedule, and that final training run is a normal, full-cost training job — often the single largest line item in the whole project. RL-NAS amortized this differently: it trained thousands of children but each was small and short, whereas DARTS trains one cheap supernet and then one expensive final model. So the honest comparison is not "2,000 GPU-days versus 2 GPU-days" for the *deliverable*; it is "2,000 GPU-days of search plus one final train" versus "2 GPU-days of search plus one final train." The final train is common to both, so the *search* saving is the real 1,000×, but do not let "DARTS is cheap" trick you into forgetting the final training run when you plan capacity. Many a NAS project has blown its budget not on the search but on the unglamorous full-scale retrain that every search method requires.

## 7. The DARTS pitfalls: skip-connection collapse and memory

DARTS is the method most people reach for first because it is cheap and the code is public, so you should know its two notorious failure modes before you run it.

The first and most famous is **skip-connection collapse**, sometimes called the "domination of skip connections" or DARTS instability. As the search runs longer, DARTS has a strong tendency to assign the largest $\alpha$ weights to the *parameter-free* skip-connection (identity) operation on more and more edges. The discretized cell ends up dominated by skip connections, which makes it shallow and weak — its accuracy collapses. The reason is a known pathology of the bilevel optimization: skip connections create a kind of optimization shortcut that lowers the *validation loss during search* (they stabilize gradient flow in the over-parameterized supernet) without corresponding to a genuinely better discretized architecture. The supernet learns to love skip connections for reasons that do not transfer to the final network. Practical fixes that the follow-up literature found: early-stop the search before collapse sets in (DARTS+); regularize or cap the number of skip connections in the final cell; add operation-level dropout to the skip path (P-DARTS); or use a perturbation-based discretization that picks operations by how much removing them hurts validation accuracy rather than by raw $\alpha$ magnitude (DARTS-PT, RobustDARTS). If you run vanilla DARTS for too many epochs and your final cell is mostly identities, this is what happened, and it is not a bug in your code.

The second pitfall is **memory**. The DARTS supernet has *every candidate operation active simultaneously* on every edge — that mixed operation is a sum over all $K$ operations, so the supernet holds roughly $K$ times the activations and parameters of a single discrete architecture. With eight operations per edge and a cell graph with a dozen-plus edges, the supernet is large, and the activations for the weighted sum all have to be kept in memory for backprop. This is why DARTS searches are typically run with a *small* number of cells and a *small* channel count during search (then scaled up for the final architecture), and why memory, not compute, is often the binding constraint on a single GPU. The follow-up method PC-DARTS attacked exactly this by only sending a fraction of the channels through the mixed operation (partial channel connections), cutting the memory enough to search with larger batches and more stably. If your DARTS search OOMs, the levers are: fewer cells during search, fewer channels, smaller batch, or partial-channel connections.

There is a deeper, honest critique worth stating. A well-known 2020 study by Li and Talwalkar, "Random Search and Reproducibility for NAS," showed that on the standard DARTS search space, *random search* — just sampling architectures uniformly and picking the best — is a shockingly strong baseline, sometimes within noise of DARTS. This does not mean DARTS is worthless; it means the *search space* was doing a lot of the work (it was hand-designed to contain mostly good architectures), and it means you should always report a random-search baseline when you claim a NAS method helped. The strategy matters less than the space — the lesson from section 4, now with receipts. When someone shows you a NAS result without a random-search baseline, be suspicious.

## 8. The practical flow: a DARTS-style mixed-operation cell in PyTorch

Enough theory. The core of a DARTS implementation is shorter than people expect, because the whole trick is the mixed operation and the architecture parameters $\alpha$. Here is a runnable, idiomatic version of the essential pieces. We will define a small set of candidate operations, a mixed operation that applies all of them under a softmax over $\alpha$, a cell that wires mixed operations into a small DAG, and the alternating bilevel training loop.

Start with the candidate operations. These are the eight-ish building blocks the search chooses among on each edge.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# A minimal candidate operation set (the "primitives" DARTS searches over).
# Each maps C channels -> C channels at a fixed spatial size (stride 1 here).
OPS = {
    "none":      lambda C: Zero(),                         # the "zero" op: drop this edge
    "skip":      lambda C: nn.Identity(),                  # skip / identity connection
    "max_3x3":   lambda C: nn.MaxPool2d(3, stride=1, padding=1),
    "avg_3x3":   lambda C: nn.AvgPool2d(3, stride=1, padding=1),
    "sep_3x3":   lambda C: SepConv(C, 3),                  # depthwise-separable 3x3
    "sep_5x5":   lambda C: SepConv(C, 5),                  # depthwise-separable 5x5
    "dil_3x3":   lambda C: DilConv(C, 3, dilation=2),      # dilated 3x3
    "conv_1x1":  lambda C: nn.Conv2d(C, C, 1, bias=False), # pointwise conv
}
PRIMITIVES = list(OPS.keys())  # the operation ordering that alpha indexes into


class Zero(nn.Module):
    """The 'none' operation: output all zeros (effectively no connection)."""
    def forward(self, x):
        return x.mul(0.0)


class SepConv(nn.Module):
    """Depthwise-separable conv: depthwise then pointwise, the MobileNet block."""
    def __init__(self, C, k):
        super().__init__()
        pad = k // 2
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, k, stride=1, padding=pad, groups=C, bias=False),  # depthwise
            nn.Conv2d(C, C, 1, bias=False),                                   # pointwise
            nn.BatchNorm2d(C),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    """Dilated depthwise-separable conv: a larger receptive field, same cost."""
    def __init__(self, C, k, dilation):
        super().__init__()
        pad = dilation * (k // 2)
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, k, stride=1, padding=pad, dilation=dilation, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C),
        )

    def forward(self, x):
        return self.op(x)
```

Now the heart of DARTS: the **mixed operation**. It instantiates every primitive and, on the forward pass, returns the softmax-weighted sum of all of them. The weights are *not* stored here — they are passed in as the $\alpha$ vector for this edge, so a single set of $\alpha$ parameters can drive many edges and they can be optimized separately from the network weights.

```python
class MixedOp(nn.Module):
    """Applies all primitives and returns the softmax(alpha)-weighted sum.

    This is the DARTS continuous relaxation: instead of choosing one op,
    average all of them, weighted by softmax over the architecture params.
    """
    def __init__(self, C):
        super().__init__()
        self.ops = nn.ModuleList([OPS[p](C) for p in PRIMITIVES])

    def forward(self, x, alpha_edge):
        # alpha_edge: shape (len(PRIMITIVES),) — one weight per operation on this edge.
        weights = F.softmax(alpha_edge, dim=-1)            # the softmax over candidate ops
        return sum(w * op(x) for w, op in zip(weights, self.ops))
```

That `MixedOp.forward` is the entire mathematical idea from section 5 in three lines: softmax the $\alpha$ for this edge, then return $\sum_o \text{softmax}(\alpha)_o \cdot o(x)$. Next, the cell wires a small DAG of mixed operations. Each intermediate node sums the mixed-op outputs from all earlier nodes, and the cell owns one $\alpha$ vector per edge.

```python
class Cell(nn.Module):
    """A small searchable cell: `steps` intermediate nodes, fully connected to
    all predecessors, with a MixedOp on every edge. One alpha vector per edge."""
    def __init__(self, steps, C):
        super().__init__()
        self.steps = steps
        self.ops = nn.ModuleList()
        self.edge_index = []  # (source_node, dest_node) for each edge, in order
        # node 0 is the cell input; intermediate nodes are 1..steps
        for j in range(1, steps + 1):          # for each intermediate node j
            for i in range(j):                 # connect from every earlier node i
                self.ops.append(MixedOp(C))
                self.edge_index.append((i, j))

    def num_edges(self):
        return len(self.ops)

    def forward(self, x0, alphas):
        # alphas: shape (num_edges, len(PRIMITIVES)) — the architecture parameters.
        states = [x0]                          # node 0 = cell input
        for j in range(1, self.steps + 1):
            s = 0
            for e, (i, jj) in enumerate(self.edge_index):
                if jj == j:                    # edges feeding node j
                    s = s + self.ops[e](states[i], alphas[e])
            states.append(s)
        # cell output: concatenate (here sum, for brevity) the intermediate nodes
        return sum(states[1:])
```

Finally, the supernet holds the cell plus the *architecture parameters* $\alpha$ as a separate parameter group, and the training loop alternates the two gradient steps that implement first-order DARTS: a step on the weights $w$ using the training batch, then a step on $\alpha$ using a *validation* batch.

```python
class DartsNet(nn.Module):
    def __init__(self, C_in=3, C=16, steps=4, n_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(C_in, C, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(C))
        self.cell = Cell(steps, C)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(C, n_classes))
        # Architecture parameters: one alpha vector per edge. THIS is what search learns.
        self.alphas = nn.Parameter(1e-3 * torch.randn(self.cell.num_edges(),
                                                       len(PRIMITIVES)))

    def weight_parameters(self):
        return [p for n, p in self.named_parameters() if n != "alphas"]

    def forward(self, x):
        return self.head(self.cell(self.stem(x), self.alphas))


def darts_search_step(model, x_train, y_train, x_val, y_val, opt_w, opt_a):
    # First-order DARTS: alternate one weight step and one architecture step.
    # (1) update network weights w on the TRAINING batch
    opt_w.zero_grad()
    loss_train = F.cross_entropy(model(x_train), y_train)
    loss_train.backward()
    opt_w.step()
    # (2) update architecture params alpha on the VALIDATION batch
    opt_a.zero_grad()
    loss_val = F.cross_entropy(model(x_val), y_val)
    loss_val.backward()      # gradient flows to model.alphas through the softmax
    opt_a.step()
    return loss_train.item(), loss_val.item()
```

The two optimizers are the bilevel structure made literal: `opt_w` (typically SGD with momentum) optimizes the weights on the training loss, and `opt_a` (typically Adam, smaller learning rate, weight decay) optimizes the architecture on the validation loss. After search you discretize:

```python
def derive_architecture(model, keep_per_node=2):
    """Read off the discrete cell: for each edge pick the best non-'none' op,
    then keep the top-`keep_per_node` strongest edges into each node."""
    alphas = F.softmax(model.alphas, dim=-1).detach()
    none_idx = PRIMITIVES.index("none")
    chosen = []
    for e, (i, j) in enumerate(model.cell.edge_index):
        w = alphas[e].clone()
        w[none_idx] = -1.0                       # never pick the 'none' op as the winner
        best_op = PRIMITIVES[int(w.argmax())]
        strength = float(w.max())                 # edge importance for top-k selection
        chosen.append((j, i, best_op, strength))
    # keep the two strongest incoming edges per intermediate node
    final = []
    for node in range(1, model.cell.steps + 1):
        edges = [c for c in chosen if c[0] == node]
        edges.sort(key=lambda c: c[3], reverse=True)
        final.extend(edges[:keep_per_node])
    return final  # list of (dest, src, op, strength) — your discrete cell
```

That is a working, if minimal, DARTS. The real [DARTS repository](https://github.com/quark0/darts) by Liu et al. adds the second-order gradient, the normal/reduction cell split, the proper stem and auxiliary heads, and the careful CIFAR/ImageNet training recipes — but the conceptual core is exactly the `MixedOp`, the per-edge $\alpha$, and the alternating two-optimizer loop above. For the RL/weight-sharing flavor, the [ENAS implementations](https://github.com/melodyguan/enas) keep a controller RNN and share weights across sampled subgraphs; for production-grade hardware-aware search, NNI (Microsoft's Neural Network Intelligence) and the Once-for-All / OFA codebase are the toolkits people actually deploy.

## 9. Results: strategies, and the landmark numbers

Time to put the strategies side by side and then look at the landmark results, because the numbers are what justify the whole enterprise. Figure 6 is the strategy comparison as a matrix; the text below it is the detail.

![A comparison matrix with rows for reinforcement learning, evolution, gradient DARTS, and one-shot supernet, and columns for search cost, quality, and complexity, showing that RL and evolution cost thousands of GPU-days at top-tier quality while gradient and one-shot methods cost single-digit to low-hundreds of GPU-days at strong but less stable quality](/imgs/blogs/neural-architecture-search-basics-6.png)

Here is the same comparison as a table you can act on, with the trade-offs spelled out:

| Strategy | Search cost (order) | Quality | Main weakness | Reach for it when |
|---|---|---|---|---|
| RL controller (Zoph & Le) | ~2,000 GPU-days | Top-tier, robust | Astronomically expensive | You are a lab with a cluster and want a definitive search |
| Evolution (AmoebaNet) | ~3,150 GPU-days | Top-tier, robust | Same cost as RL | Same as RL; slightly simpler, aging regularization |
| Gradient / DARTS | ~1–4 GPU-days | Strong, unstable | Skip-collapse, memory, reproducibility | You want a cheap search on a clean cell space, fast |
| One-shot supernet (ENAS, OFA) | ~8–200 GPU-days | Good, decoupled | Weight coupling, rank gap | You want to search once and deploy many sub-models |

Now the landmark architecture results from the literature. Be careful with these numbers: they come from different papers with different training recipes and the GPU-day figures are configuration-dependent, so treat them as the documented headline values, not benchmarks I re-ran.

| Architecture | Method | Dataset | Top-1 accuracy | Search cost (reported) |
|---|---|---|---|---|
| NASNet-A | RL controller | ImageNet | ~74.0% (mobile setting) | ~2,000 GPU-days |
| AmoebaNet-A | Regularized evolution | ImageNet | ~74.5% (comparable setting) | ~3,150 GPU-days |
| DARTS (cell) | Gradient (differentiable) | ImageNet (mobile) | ~73.3% | ~4 GPU-days |
| ENAS (cell) | RL + weight sharing | CIFAR-10 | ~97.1% (test) | ~0.45 GPU-days |

Read the DARTS row against the NASNet/AmoebaNet rows and the entire point of NAS-after-2018 lands. DARTS reaches within about one point of the RL/evolution architectures on ImageNet mobile, but it found its architecture in roughly **four GPU-days instead of two thousand** — a five-hundred-fold cost reduction for roughly a point of accuracy. For most teams that trade is overwhelmingly worth it, and it is why differentiable and one-shot NAS, not RL/evolution, are what people run today. The ENAS CIFAR-10 number shows the weight-sharing speedup in the RL framing: ~0.45 GPU-days for a competitive cell, versus the thousands its un-shared parent needed.

And the two architectures that prove NAS matters in production: **MobileNetV3** (Howard et al., 2019) was found by a platform-aware NAS (MnasNet-style search) plus the NetAdapt fine-tuning algorithm and human refinement, and it set the efficiency frontier for on-device classification — MobileNetV3-Large hit roughly 75.2% ImageNet top-1 at a latency budget tuned for the Pixel phone, beating the hand-designed V2 at the same latency. **EfficientNet** (Tan & Le, 2019) started from a NAS-found baseline cell (EfficientNet-B0, from the same MnasNet search family) and then *compound-scaled* it; EfficientNet-B0 reached ~77.1% ImageNet top-1 with about 5.3M parameters and ~0.39 GFLOPs, a Pareto point that hand-designed models of the era could not touch. These are not toy results — they are the architectures the field actually adopted, and they came out of search.

## 10. Why NAS matters for the edge — and the gap it leaves

NAS produces the most efficient architectures we have, and that is exactly why it belongs in a series about optimizing models for the edge. The architectures that define on-device deep learning — MobileNetV3, EfficientNet, and their descendants — are NAS products. If you want a model that hits the accuracy–FLOPs Pareto frontier, the searched architectures are the frontier. NAS is the only one of the four levers that improves the *starting point* rather than compressing an existing model, which means everything else in this series composes on top of it: you take a NAS-found dense architecture and then quantize it, prune it, and distill into it. A quantized, pruned MobileNetV3 is the four levers stacked, and the bottom of the stack is NAS.

But — and this is the gap that the *next* post exists to close — classic NAS optimizes the wrong objective for the edge. Cell-search à la DARTS and NASNet optimizes **accuracy at a FLOPs (or parameter) budget**. FLOPs are a proxy for cost, and they are a *bad* proxy for on-device latency. A model with fewer FLOPs can be *slower* on a real device than a model with more FLOPs, because latency depends on memory bandwidth, kernel availability, operator fusion, the NPU's supported op set, and parallelism — none of which FLOPs capture. This series spends a whole post on exactly that mismatch in [EfficientNet, ShuffleNet, and the FLOPs–latency gap](/blog/machine-learning/edge-ai/efficientnet-shufflenet-and-the-flops-latency-gap): depthwise convolutions, the workhorse of efficient NAS cells, have low FLOPs but poor *arithmetic intensity*, so they are memory-bound and underutilize the hardware. An architecture search that minimizes FLOPs can happily pick operations that are cheap on paper and slow in silicon.

The fix is **hardware-aware NAS**: put the *measured latency on the target device* into the search objective directly, so the search optimizes accuracy at a real-latency budget rather than a FLOPs budget. MnasNet did this by running candidate architectures on an actual Pixel phone and folding the measured latency into the reward; later work used differentiable latency models so DARTS-style search could be made latency-aware. That is the twist that turns NAS from "finds accurate-per-FLOP architectures" into "finds fast-on-your-chip architectures," and it is the difference between a model that looks efficient in a paper and one that hits your p99 latency budget on the device in your hand. The whole next post, [hardware-aware NAS](/blog/machine-learning/edge-ai/hardware-aware-nas), is about that objective swap; this post is the foundation it builds on.

#### Worked example: the FLOPs–latency trap a plain NAS would fall into

Make the gap concrete. Suppose your search space has two candidate operations for an edge: a standard $3\times3$ convolution (call it op A) and a $3\times3$ depthwise-separable convolution (op B, the MobileNet block). Op B has roughly $8$–$9\times$ fewer FLOPs than op A for the same channel count — that is the whole reason it exists. A FLOPs-minimizing NAS will strongly prefer op B everywhere; its $\alpha$ will converge to depthwise.

Now measure on a real edge NPU. The depthwise conv has very low **arithmetic intensity** — it does few multiply-adds per byte of memory traffic, so it is *memory-bound*, and on many NPUs it under-utilizes the multiply-add array badly (utilization of 20–40% is common for depthwise on accelerators tuned for dense convs). The standard conv, with $8\times$ the FLOPs, can run at near-peak utilization. On a chip where the dense conv runs at, say, 90% utilization and the depthwise at 30%, the depthwise's $8\times$ FLOPs advantage shrinks to under $3\times$ in wall-clock, and on a *specific* op size where the depthwise hits a slow kernel fallback, it can even lose outright. A FLOPs-driven NAS, blind to this, picks the architecture that minimizes a number nobody experiences; a latency-driven NAS picks the one the user feels. This is precisely why MobileNetV3's *search* was platform-aware — and precisely the gap [hardware-aware NAS](/blog/machine-learning/edge-ai/hardware-aware-nas) closes.

## 11. When to run NAS — and when to just reuse a searched architecture

Here is the most useful section for your actual job, and it is mostly a warning. For the overwhelming majority of teams and projects, **you should not run NAS — you should reuse an architecture someone already searched.** Figure 8 is the decision tree, and it bottoms out at "reuse" far more often than "run NAS."

![A decision tree for whether to run neural architecture search or reuse an existing searched architecture, branching first on whether a searched architecture already fits your domain leading to reusing MobileNetV3 or EfficientNet possibly with light width and depth tuning, versus a novel domain branch that further splits on whether you have a real GPU budget and latency target leading to running hardware-aware NAS or otherwise skipping NAS and hand-tuning a baseline](/imgs/blogs/neural-architecture-search-basics-8.png)

Walk the tree. The first question is: **does an already-searched architecture fit your domain?** If you are doing image classification, detection, or segmentation on natural images and you need an efficient backbone, the answer is almost always yes — MobileNetV3, EfficientNet, EfficientNet-Lite, MobileViT, and their friends are NAS outputs that someone spent hundreds or thousands of GPU-days finding, validated across many tasks, and released for free. Reusing one gives you the benefit of NAS at zero search cost. If your data is a bit unusual but still in the same family, the move is reuse-plus-light-tuning: take the searched architecture and scale its width and depth (EfficientNet's compound scaling is literally a recipe for this) or fine-tune, which is cheap. You are standing on the shoulders of a search you did not have to pay for.

You only descend into the "run NAS" branch when the domain is **genuinely novel** — a new modality (point clouds, raw audio, a sensor format nobody has published efficient backbones for), a new hardware target with an unusual op set, or a deployment constraint (a microcontroller's SRAM budget, a specific NPU's fused-op set) that no public architecture was searched against. Even then, NAS is only justified if you have **both** a real compute budget *and* a concrete target metric to optimize. NAS without a budget is a fantasy — even cheap DARTS needs a GPU for a day or several, plus the much larger cost of *retraining* the final architecture from scratch and validating it, plus the human time to set up and debug the search (and DARTS is finicky — see section 7). And NAS without a clear target metric is aimless: if you do not know whether you are optimizing accuracy-at-FLOPs or accuracy-at-latency-on-a-named-chip, you will optimize the wrong thing, and if it is latency you care about, you specifically want hardware-aware NAS, not the methods in this post.

A blunt heuristic: if you are tempted to run NAS, first run the *random-search baseline* on your candidate space (section 7) and the "reuse a published architecture" baseline. If a published architecture or random search gets you to target, you are done — you saved weeks. NAS earns its keep when those baselines fall short *and* the gap matters *and* you have the budget to close it. That is a narrow set of conditions, and being honest about how narrow it is will save you more compute than any search method will.

## 12. Stress-testing the decision

Let me pressure-test the "just reuse" advice against the cases where it breaks, because a recommendation you cannot stress-test is not engineering.

*What if the published architectures are all too big for my microcontroller?* Then you are in the genuinely-novel branch — sub-megabyte, sub-256KB-SRAM models are a real frontier where NAS (specifically the MCUNet line of tiny NAS, which co-designs the architecture and the inference engine under a hard SRAM/Flash budget) found models that hand-design did not. This is a legitimate run-NAS case: the constraint is unusual, the payoff is large, and there is a method built for exactly it. But note it is *hardware-and-memory-aware* NAS, not the FLOPs-budget cell search of this post.

*What if my data is natural images but a weird aspect ratio or resolution?* Reuse plus retraining at your resolution almost always wins. The cell structure is robust to resolution; you do not need to re-search. Re-searching here is a classic waste.

*What if I run DARTS and the final cell is all skip connections?* You hit skip-collapse (section 7). Do not conclude DARTS does not work — early-stop the search, cap skip connections, or switch to a robust variant (P-DARTS, DARTS-PT). And re-examine whether you should be running DARTS at all versus reusing.

*What if the supernet says architecture X is best but X retrains worse than architecture Y?* That is weight coupling (section 6) — the supernet's ranking did not transfer. Retrain your top-k candidates from scratch and pick by *standalone* accuracy, never by supernet accuracy alone. This is the single most common way one-shot NAS results fail to reproduce.

*What if I cannot afford even DARTS's few GPU-days plus the final retraining?* Then NAS is not for this project. Reuse. The final-architecture retraining (often the largest single cost — a full training run on your real data) is unavoidable for any search method, so "NAS is cheap now" never means "free."

The through-line of every stress case is the same: NAS is a real tool with real failure modes, and its biggest competitor is not a different NAS method — it is the architecture someone already searched and released. Beat that baseline before you spend a GPU-day.

## 13. Key takeaways

- **Every NAS method is the same three-part loop**: a search space (what is reachable), a search strategy (what to try), and a performance estimation strategy (how to score). The cost lives almost entirely in estimation.
- **The search space is the most important and most underrated component.** Cell-based search — find one small reusable cell, stack copies under a fixed macro skeleton — is what made NAS tractable and transferable. Even the compact cell space has ~$10^9$ architectures, so enumeration is hopeless.
- **The search strategy matters less than people think.** RL (Zoph & Le) and evolution (AmoebaNet) land in roughly the same place at roughly the same astronomical cost (~2,000–3,150 GPU-days). Random search is a shockingly strong baseline — always report it.
- **Weight sharing is the idea that changed everything.** Train one over-parameterized supernet whose subnets share weights, and a candidate costs an inference pass instead of a training run. ENAS and DARTS are both instances; they cut cost ~1,000×.
- **DARTS makes the architecture differentiable** via a softmax over candidate operations (the mixed op), turning the discrete choice into a bilevel optimization $\min_\alpha \mathcal{L}_{val}(w^*(\alpha),\alpha)$ s.t. $w^* = \arg\min_w \mathcal{L}_{train}$, solved cheaply with the first-order approximation. It runs in ~1–4 GPU-days on one GPU.
- **DARTS has two notorious pitfalls**: skip-connection collapse (the search over-weights parameter-free identities and the final cell goes shallow and weak) and memory (all candidate ops are active at once). Early-stop, cap skips, or use robust/PC variants.
- **NAS found the architectures that define efficient deep learning** — MobileNetV3, EfficientNet — so reusing them is reusing a search you did not pay for. That is the right default for almost every project.
- **Classic NAS optimizes accuracy-at-FLOPs, which is the wrong objective for the edge**, because FLOPs are a poor latency proxy (depthwise convs are cheap in FLOPs but memory-bound and slow on real NPUs). Hardware-aware NAS, which puts measured on-device latency in the objective, is the fix.
- **Run NAS only when the domain is genuinely novel AND you have both a compute budget and a concrete target metric.** Otherwise reuse a published searched architecture, optionally scaled or fine-tuned. Beat the reuse and random-search baselines before spending a GPU-day.

## 14. Further reading

- **Zoph & Le, 2017 — "Neural Architecture Search with Reinforcement Learning."** The paper that started the modern wave: the RNN controller, policy-gradient training, and the thousands-of-GPU-days cost baseline.
- **Real, Aggarwal, Huang & Le, 2019 — "Regularized Evolution for Image Classifier Architecture Search" (AmoebaNet).** Evolution with aging regularization; the controlled RL-vs-evolution comparison.
- **Liu, Simonyan & Yang, 2019 — "DARTS: Differentiable Architecture Search."** The continuous relaxation, the bilevel optimization, and the first/second-order approximations. The cost collapse to single-GPU-days.
- **Pham, Guan, Zoph, Le & Dean, 2018 — "Efficient Neural Architecture Search via Parameter Sharing" (ENAS).** The supernet / weight-sharing idea in the RL framing; ~1,000× speedup.
- **Elsken, Metzen & Hutter, 2019 — "Neural Architecture Search: A Survey."** The three-component framing this post is built on; the canonical map of the field.
- **Tan & Le, 2019 — "EfficientNet"** and **Howard et al., 2019 — "Searching for MobileNetV3."** The two production NAS architectures; required reading for why NAS matters in practice.
- **Li & Talwalkar, 2020 — "Random Search and Reproducibility for Neural Architecture Search."** The sobering baseline study; why you must report random search.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for where NAS sits among the four levers, [the MobileNet family](/blog/machine-learning/edge-ai/the-mobilenet-family) for the hand-designed architectures NAS learned to beat, [hardware-aware NAS](/blog/machine-learning/edge-ai/hardware-aware-nas) for the latency-objective twist, [EfficientNet, ShuffleNet, and the FLOPs–latency gap](/blog/machine-learning/edge-ai/efficientnet-shufflenet-and-the-flops-latency-gap) for why FLOPs mislead on device, and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for composing NAS with the other three levers.

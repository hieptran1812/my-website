---
title: "On-device and federated learning: training at the edge without the data leaving"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Train models on phones and microcontrollers without uploading a single byte of user data — derive FedAvg, compress the uplink, add differential privacy, and fine-tune a deployed model with LoRA on-device."
tags:
  [
    "edge-ai",
    "model-optimization",
    "federated-learning",
    "on-device-training",
    "differential-privacy",
    "lora",
    "inference",
    "efficient-ml",
    "privacy",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/on-device-and-federated-learning-1.png"
---

Every other post in this series has assumed a one-way street: you train a model in the cloud, you squeeze it with quantization and pruning and distillation, and then you ship the frozen artifact to the device, where it does nothing but **inference** for the rest of its life. The device is a consumer. The data flows in, predictions flow out, and the weights never change.

That is the dominant pattern, and for good reason. But it leaves an entire frontier untouched. What if the device could *learn*? Not retrain from scratch — that would be absurd on a phone — but adapt. A keyboard that learns *your* typing, *your* slang, *your* friends' names, without ever shipping a single keystroke to a server. A hearing aid that tunes itself to *your* hearing loss profile in *your* acoustic environments. A camera that gets better at recognizing *your* pets. The promise is two things at once that the cloud cannot give you: **personalization** and **privacy**. The model gets better for you specifically, and the data that makes it better never leaves your pocket.

This is not hypothetical. Google's Gboard keyboard has used **federated learning** in production since 2017 to improve next-word prediction, emoji suggestion, and query suggestion across hundreds of millions of phones — and the raw text you type has never been uploaded to train it. Apple uses on-device learning for personalization in QuickType and Photos. The technique is real, shipped, and at scale. But it is also genuinely hard, because training is brutally more expensive than inference, and the edge is exactly where you have the least compute, memory, and energy to spare.

This post is about that frontier. By the end you will be able to: derive the **FedAvg** algorithm from first principles and understand exactly what it averages; reason quantitatively about why the **uplink** is the bottleneck and how update compression (top-k sparsification, quantization, structured updates) buys it back; understand why federated is *not automatically private* and what **secure aggregation** plus **differential privacy** actually add; and pick the right on-device fine-tuning strategy (last-layer, adapter, LoRA) given a fixed memory budget. We will write the code, work the numbers, and — most importantly — be honest about when this whole apparatus is worth it versus when you should just train centrally and push a new model over the air.

![A two-column figure comparing forward-only inference, which keeps one activation live with no optimizer state, against training, which retains all layer activations plus optimizer state and burns roughly ten times the energy](/imgs/blogs/on-device-and-federated-learning-1.png)

This sits at a specific spot in the four-lever frame that runs through the whole series — quantization, pruning, distillation, efficient architecture, all read off the accuracy–efficiency Pareto frontier. On-device training does not replace those levers; it *consumes* them. You can only afford to train on a phone *because* the model is already quantized, already small, already adapter-friendly. If you have not read [the taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), start there for the map. This post is about what happens when you turn the device from a consumer into a (careful, budget-constrained) producer.

## Why training on-device is a different sport from inference

Let me start with the thing every cloud-trained engineer underestimates: the gap between inference and training is not 2x. It is closer to an order of magnitude, and it bites you in the one resource the edge has least of — fast memory.

Inference is a forward pass. You feed an input through the layers, and at any moment you only need the activation of the layer you are currently computing (and, for residual connections, maybe one or two earlier tensors). A good runtime reuses buffers aggressively, so the **peak working set** is roughly the largest pair of simultaneously-live tensors. This is exactly the memory story from [memory is the real constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint), and it is why inference fits on a microcontroller with a few hundred KB of SRAM.

Training is a forward pass *plus* a backward pass, and the backward pass changes everything. To compute the gradient of the loss with respect to a layer's weights, the chain rule needs that layer's **input activation** from the forward pass. So every activation you computed going forward must be *kept alive* until the backward pass reaches it. You cannot free the early-layer activations after using them, because backprop will come back for them. The peak memory is no longer "the largest two tensors" — it is "essentially all of them at once."

Concretely, for a network with $L$ layers, inference peak activation memory is $O(\max_\ell a_\ell)$ where $a_\ell$ is the size of layer $\ell$'s activation. Training peak activation memory is

$$
M_{\text{train}} \;\approx\; \sum_{\ell=1}^{L} a_\ell \;+\; \underbrace{M_{\text{params}}}_{\text{weights}} \;+\; \underbrace{M_{\text{grad}}}_{=\,M_{\text{params}}} \;+\; \underbrace{M_{\text{opt}}}_{\text{optimizer state}}.
$$

Walk through that. You store **all** activations (the sum, not the max). You store the weights. You store a gradient buffer the same size as the weights. And you store optimizer state: plain SGD adds nothing, SGD with momentum adds $1\times$ the parameter count, and Adam adds $2\times$ (first and second moments). So a model that needs, say, $P$ parameters worth of memory for inference can need $P$ (weights) $+ P$ (gradients) $+ 2P$ (Adam moments) $= 4P$ just for the parameter-side state — *before* you count the activation sum, which for training dominates everything for anything but the shallowest models.

That is the memory side. The energy side is just as stark. A backward pass does roughly **twice** the FLOPs of a forward pass (you compute gradients with respect to both inputs and weights), so a training step is about $3\times$ the compute of an inference step in FLOPs alone. But FLOPs undersell it on the edge, because moving all those retained activations in and out of memory dominates energy when you are memory-bound — and training is *more* memory-bound than inference because of the activation retention. Empirically, a full training step on a mobile-class SoC lands around $5$–$10\times$ the energy of an inference step. Figure 1 is the picture: inference keeps one activation live and burns one unit of energy; training retains everything, carries optimizer state, and burns roughly ten.

This is why "just train on the device" is naive, and why the rest of this post is a sustained exercise in **not** doing the expensive thing. The two big ideas are: (1) **federated learning**, which spreads tiny amounts of local training across millions of devices and aggregates the results, so no single device does much; and (2) **parameter-efficient fine-tuning** (LoRA, adapters, last-layer), which updates a tiny fraction of the weights so the gradient, optimizer, and even activation-retention costs collapse. Both are ways of fitting training into a budget that, naively, it does not fit into.

#### Worked example: the training memory blowup for a small CNN

Let me put real numbers on the inference-versus-training gap so the order-of-magnitude is not abstract. Take a MobileNetV2-class classifier with $P = 3.4$M parameters and a peak inference activation of about 2.3 MB (the largest live tensor pair), running int8 for inference on a mobile NPU.

For *inference* (int8), the memory budget is roughly: weights $3.4\text{M} \times 1\text{ byte} = 3.4$ MB, plus peak activation $\approx 2.3$ MB, for about **5.7 MB** of working memory. Comfortable on any phone.

For *training* (you must use at least fp16 for stable gradients, so weights are now 2 bytes), the budget explodes. Weights: $3.4\text{M} \times 2 = 6.8$ MB. Gradient buffer: another 6.8 MB. Adam moments (first and second): $2 \times 6.8 = 13.6$ MB. And the killer — the activation *sum* across all layers, not the max, because every forward activation is retained for backprop. For a network this depth the retained-activation total runs around 8–12× the single-layer peak, call it $\approx 22$ MB. Total training memory: $6.8 + 6.8 + 13.6 + 22 \approx$ **49 MB**, against 5.7 MB for inference — a **8.6×** blowup, dominated jointly by the optimizer state ($4P$ in bytes for fp16 Adam) and the activation retention. Scale this to a real on-device LLM and the activation-sum term alone runs into gigabytes, which is why nobody full-fine-tunes one on a phone. Every technique below exists to attack one of these terms: federated learning amortizes the *frequency* of paying it, and parameter-efficient fine-tuning shrinks the gradient, optimizer, and (with frozen lower layers) activation terms directly.

### The four reasons you would do this at all

Before the machinery, be clear-eyed about *why*, because the costs are real and you need a real payoff.

1. **Privacy: the data never leaves.** The raw keystrokes, photos, voice clips, medical readings stay on the device. Only model updates (gradients or weight deltas) are transmitted — and as we will see, even those can leak, which is why differential privacy joins the party. For regulated data (health, finance) or jurisdictions with data-residency laws (GDPR, India's DPDP, China's PIPL), "the data physically never moves" can be the difference between a legal product and an illegal one.
2. **Personalization.** A global model averaged over everyone is, by construction, a compromise. Your typing, your accent, your photo library are *not* the average. On-device adaptation produces a model that is better *for you*, often dramatically so for the long tail (rare names, code-switching, dialect).
3. **No upload bandwidth (for raw data).** Uploading every user's data to a central server costs bandwidth, storage, and money, and is often simply infeasible (think continuous sensor streams). Federated learning uploads model *deltas*, which — once compressed — can be far smaller than the data that produced them.
4. **Fresh data, immediately.** The device sees the freshest possible distribution: today's slang, this morning's lighting, the new app the user just installed. There is no data-collection-to-retraining latency. The model can adapt within hours of a distribution shift the cloud would not see for weeks.

Each of these is a genuine reason. None of them is free. Keep them in mind; the final section weighs them against the costs.

## Federated learning and the FedAvg algorithm

Federated learning, formalized by [McMahan et al. (2017)](https://arxiv.org/abs/1602.05629), inverts the usual setup. Instead of bringing the data to the model (uploading everything to a central trainer), you bring the model to the data. The server holds a global model. Each round, it sends the current model to a sample of clients. Each client trains it a little on its own local data. Each client sends back *only the update* — never the data. The server averages the updates into a new global model. Repeat.

![A dataflow figure of one FedAvg round in which a server broadcasts global weights to three sampled clients that run local SGD on private data and return only their weight deltas to be averaged into the next global model](/imgs/blogs/on-device-and-federated-learning-2.png)

Figure 2 shows one round as a dataflow graph. Crucially it is *acyclic within a round*: the server fans out to clients (broadcast), the clients fan in to an aggregator (deltas), and the aggregator produces the next model that gets broadcast in the *next* round. There is no loop inside a round; the iteration is across rounds. The single most important property is in the edges: what flows up is **delta only** — the raw data never crosses the network.

### Deriving the FedAvg update

Here is the objective. We have $K$ clients. Client $k$ holds a local dataset of $n_k$ examples, and the total is $n = \sum_k n_k$. The thing we *want* to minimize is the global empirical loss over all data pooled together:

$$
F(w) \;=\; \frac{1}{n}\sum_{i=1}^{n} \ell_i(w) \;=\; \sum_{k=1}^{K} \frac{n_k}{n}\, F_k(w),
\qquad
F_k(w) \;=\; \frac{1}{n_k}\sum_{i \in \mathcal{D}_k} \ell_i(w).
$$

That second equality is the key rewrite. The global loss is a **weighted average of the per-client losses**, where each client's weight is its share of the data $n_k/n$. This is just regrouping the sum by client — but it tells us that if we could minimize a weighted combination of local objectives, we would minimize the global one.

It is worth pausing on *why* the weight is the sample share $n_k/n$ specifically, because newcomers often guess "average the models uniformly" and silently corrupt their training. The pooled loss treats every *example* equally — each of the $n$ examples contributes one $\ell_i$ term to the sum. When you regroup by client, a client holding $n_k = 10{,}000$ examples carries ten times the weight of a client holding $1{,}000$, because it contributes ten times as many $\ell_i$ terms. If instead you averaged the *clients* uniformly (weight $1/K$ each), you would be optimizing a *different* objective — one where a user with three text messages counts as much as a user with thirty thousand. That uniform-client objective is sometimes what you want (it is more "fair" to small clients and is the basis of some personalization variants), but it is **not** the same as minimizing the pooled loss, and conflating the two is a classic FL bug. The data-weighted average is the unbiased estimator of the centralized gradient; the uniform average is not. Concretely, the bias of uniform averaging is $\sum_k\left(\frac{1}{K}-\frac{n_k}{n}\right)\nabla F_k(w)$, which is zero only when every client holds exactly the same amount of data. The more skewed your client sizes — and on a real phone fleet they span four orders of magnitude — the larger that bias.

The naive approach is **FedSGD**: each round, every client computes the gradient of its local loss at the current global weights $w_t$ and sends it up; the server takes one weighted-average gradient step:

$$
g_k \;=\; \nabla F_k(w_t),
\qquad
w_{t+1} \;=\; w_t - \eta \sum_{k=1}^{K} \frac{n_k}{n}\, g_k.
$$

Because $\sum_k \frac{n_k}{n} g_k = \nabla F(w_t)$ exactly (gradients are linear, and the weighted sum of local gradients *is* the global gradient), FedSGD is *identical* to centralized SGD on the pooled data. It is the gold standard for correctness. It is also catastrophic for communication: you pay a full round of uplink for *one* gradient step, and SGD needs thousands of steps.

**FedAvg** is the fix. Instead of one gradient step per round, each client runs *several* local SGD steps — $E$ local epochs over its data with local learning rate $\eta$ and batch size $B$ — producing an updated local model $w_k^{t+1}$. Then the server averages the *models* (equivalently, the deltas), weighted by data share:

$$
\boxed{\;w_{t+1} \;=\; \sum_{k \in S_t} \frac{n_k}{n_{S_t}}\, w_k^{t+1}\;}
\qquad\text{equivalently}\qquad
w_{t+1} \;=\; w_t + \sum_{k \in S_t} \frac{n_k}{n_{S_t}}\, \Delta_k,
\quad \Delta_k = w_k^{t+1} - w_t,
$$

where $S_t$ is the subset of clients sampled this round and $n_{S_t} = \sum_{k \in S_t} n_k$. (The two forms are algebraically the same because $\sum_k \frac{n_k}{n_{S_t}} w_t = w_t$.) The "send deltas" form is what you implement, because deltas compress better and let you do secure aggregation.

![A layered figure showing FedAvg averaging as local per-client results weighted by sample share, summed into a global model, with the E equals one case reducing to pooled-gradient SGD and the E greater than one case introducing client drift](/imgs/blogs/on-device-and-federated-learning-3.png)

Figure 3 stacks the logic. When $E = 1$ and each client does a single full-batch gradient step, FedAvg reduces *exactly* to FedSGD, which reduces to centralized SGD — provably the same update. That is the sanity anchor. The magic of FedAvg is the regime $E > 1$: by doing many local steps before communicating, you do far fewer rounds. McMahan et al. report that on a non-IID MNIST split, FedAvg reached target accuracy in **10–100× fewer communication rounds** than FedSGD. That reduction is the entire reason federated learning is practical.

### Quantifying client drift: why more local work is a double-edged sword

The $E > 1$ regime is where the trade-off lives, and it is worth making the drift precise rather than hand-waving. Consider client $k$ taking $E$ local SGD steps from the shared starting point $w_t$. After step $j$ its local model is $w_k^{(j)}$, and the next step uses the gradient *at that drifted point*, $\nabla F_k(w_k^{(j)})$ — not at $w_t$. The total local move is

$$
\Delta_k \;=\; w_k^{(E)} - w_t \;=\; -\,\eta \sum_{j=0}^{E-1} \nabla F_k\!\left(w_k^{(j)}\right).
$$

Now compare to what a *single* global gradient step *from $w_t$* would have done: $-\eta E\,\nabla F_k(w_t)$. The difference between "$E$ gradients along the client's own trajectory" and "$E$ copies of the gradient at the start point" *is* the drift. If $F_k$ is $L$-smooth (gradients are $L$-Lipschitz), the trajectory wanders, and the drift magnitude grows roughly as

$$
\big\|\Delta_k + \eta E\,\nabla F_k(w_t)\big\| \;\lesssim\; \mathcal{O}\!\big(\eta^2 E^2 L\,G\big),
$$

where $G$ bounds the gradient norm. The drift scales with $\eta^2$, with $E^2$, and with how non-smooth the local objective is. This is the quantitative version of Figure 5's geometry: bigger learning rate, more local epochs, or more heterogeneous data all *amplify* the gap between the average of drifted models and the true global step.

The convergence consequence, in the form most FL theory takes it: for non-convex objectives, FedAvg's expected squared-gradient-norm after $T$ rounds is bounded by something of the shape

$$
\frac{1}{T}\sum_{t=0}^{T-1}\mathbb{E}\big\|\nabla F(w_t)\big\|^2 \;\le\; \underbrace{\mathcal{O}\!\left(\frac{1}{\sqrt{T E}}\right)}_{\text{good: more work helps}} \;+\; \underbrace{\mathcal{O}\!\left(\frac{(E-1)\,\Gamma}{?}\right)}_{\text{bad: drift floor}},
$$

where $\Gamma$ measures data heterogeneity (how much the client optima disagree). The first term *improves* with more local work $E$ — that is the communication win. The second term is a *floor* that grows with $E$ and with heterogeneity $\Gamma$ — that is the drift penalty. The optimum $E$ is finite: a few local epochs (typically $E = 1$–$5$ in practice) hits the sweet spot where you have cut communication a lot without letting drift dominate. Crank $E$ to 20 on highly non-IID data and the drift floor wins; the model stops improving or diverges. This is exactly why production systems tune $E$ carefully and reach for FedProx/SCAFFOLD when the heterogeneity term $\Gamma$ is large.

#### Worked example: FedAvg vs centralized on non-IID clients

Let me make the convergence story concrete with numbers in the spirit of the FedAvg paper. Take a small CNN on MNIST, 100 clients, and the deliberately nasty **non-IID** split where each client holds digits from only *two* classes (so no client has seen the full task). We run FedAvg with $C = 0.1$ (10 clients sampled per round), $E = 5$ local epochs, $B = 10$.

| Setup | Rounds to 97% test acc | Final test acc | Data uploaded |
| --- | --- | --- | --- |
| Centralized SGD (all data pooled) | n/a (340 SGD epochs) | 99.1% | 100% of data |
| FedSGD ($E{=}1$) | ~620 rounds | 98.9% | 0% (deltas only) |
| FedAvg ($E{=}5$, $C{=}0.1$), IID | ~20 rounds | 99.0% | 0% (deltas only) |
| FedAvg ($E{=}5$, $C{=}0.1$), non-IID 2-class | ~85 rounds | 98.3% | 0% (deltas only) |

Three things to read off this. First, FedAvg with local epochs needs **vastly fewer rounds** than FedSGD (85 vs 620 on the hard split) — that is the communication win. Second, the non-IID split costs you: more rounds and ~0.7 points of final accuracy versus IID, because clients pull in different directions (more on this below). Third, even the hard federated run gets within ~0.8 points of the fully-centralized baseline while **uploading zero raw data**. That gap — small accuracy cost for total data privacy — is the whole value proposition. (These figures track the qualitative results in McMahan et al.; treat the exact round counts as representative, not a benchmark you can read off to three significant digits.)

### The three things that break in the real world

FedAvg's pseudocode is four lines. Production federated learning is hard because of three realities the pseudocode hides.

**Non-IID data.** Centralized SGD assumes every mini-batch is a sample from the same distribution. Federated learning violates this brutally: each client's data is *its own* distribution. A user who only texts in French, a phone in a single time zone, a hospital that only sees one demographic. When client distributions differ, the per-client optima $w_k^\star$ scatter, and averaging models that have drifted toward *different* optima can land you somewhere worse than any of them. This is the single biggest source of accuracy loss in FL.

![A two-column figure contrasting IID clients whose local optima align so their average lands near the joint optimum against non-IID clients whose local optima pull apart so the average overshoots, fixed by FedProx or server momentum](/imgs/blogs/on-device-and-federated-learning-5.png)

Figure 5 is the geometry of it. With IID clients, every client's loss surface looks the same, so $E$ local steps all head toward the same basin and the average is fine. With non-IID clients, client A's local steps head toward A's optimum and client B's toward B's; after $E$ steps each has **drifted** away from the global direction, and averaging two updates that point in different directions can overshoot the true minimum. The more local epochs $E$ you do, the worse the drift — which is the cruel trade-off: more local work saves communication but amplifies drift.

The fixes are a research field of their own, but the two you reach for first: **FedProx** ([Li et al., 2020](https://arxiv.org/abs/1812.06127)) adds a proximal term $\frac{\mu}{2}\|w - w_t\|^2$ to each client's local objective, penalizing local solutions that wander too far from the global model and damping drift. **Server-side momentum / adaptive optimizers** (FedAdam, FedYogi from [Reddi et al., 2021](https://arxiv.org/abs/2003.00295)) treat the aggregated delta as a "pseudo-gradient" and apply a momentum or Adam step on the server, which smooths out the round-to-round noise from non-IID sampling.

**Client drift** is the mechanism underneath non-IID pain, and it is worth naming precisely: after $E$ local steps, client $k$'s model is at $w_k^{t+1} = w_t - \eta \sum_{j=0}^{E-1} \nabla F_k(w_k^{(j)})$, but the gradients are evaluated at the client's *own* trajectory, not the global one. The accumulated discrepancy between "what each client did" and "what a single global trajectory would have done" is the drift. It grows with $E$, with the local learning rate, and with how non-IID the data is.

**Partial participation and stragglers.** You cannot wait for all $K$ clients. On a real fleet, a phone participates only when it is **idle, charging, and on unmetered Wi-Fi** — Google's actual eligibility criteria. So each round you get a *sample* $S_t$, and the sample is biased (phones that charge at night, in certain time zones, of certain price tiers). Some selected clients will be slow or drop out mid-round (stragglers). The aggregator must handle a variable, biased subset and tolerate dropouts — typically by over-provisioning (select more clients than you need, take the first $M$ to report) and by re-weighting carefully so the sampling bias does not become a model bias.

## Communication efficiency: why the uplink is the bottleneck

Here is a fact that surprises people new to FL: the constraint is not the device's compute and it is not the *download*. It is the **uplink** — the bytes each client sends *up* to the server.

Why? Two reasons. First, asymmetry: consumer internet (and especially cellular) has download bandwidth several times the upload bandwidth. A phone might pull 50 Mbps down but push only 5–10 Mbps up. Second, scale and structure: the server broadcasts *one* model to many clients (you can multicast, cache at edges, and the download is amortized), but every client sends back its *own* full update, and those uplinks do not amortize. Multiply a multi-megabyte update by millions of clients per day and the aggregate uplink — and each user's metered-data cost and battery — becomes the binding constraint.

So the question becomes: how few bytes can a client send and still contribute a useful update? The dense answer is bad. A model with $P$ parameters in fp32 sends $4P$ bytes per round. For a 25M-parameter model that is 100 MB *per client per round*, and you might need dozens to hundreds of rounds. Nobody is uploading 100 MB over cellular for a keyboard.

There is a clean way to think about *how much* you can compress before you hurt convergence, borrowed from information theory. SGD does not need the *exact* update; it needs an *unbiased* estimate of it with bounded variance. Quantizing or sparsifying the update injects extra noise of some variance $\sigma_c^2$ on top of the inherent stochastic-gradient noise $\sigma_g^2$. Convergence slows by roughly a factor $(1 + \sigma_c^2 / \sigma_g^2)$ — so as long as the compression noise stays *below* the gradient noise SGD already tolerates, you lose almost nothing in rounds while saving enormously in bytes. That is the deep reason update compression is nearly free in a way that *weight* compression for inference is not: SGD is a noise-tolerant process, and you are spending a noise budget it already has. The practical corollary is that **unbiased** compressors (stochastic quantization, where you round up or down with probability proportional to distance, so $\mathbb{E}[\hat{x}] = x$) behave far better than biased ones — and where you *must* use a biased compressor (top-k is biased: it systematically keeps the big coordinates), you restore unbiasedness-in-expectation with error feedback.

### The compression toolbox and what it costs

There are three knobs, and they compose.

**Quantization of the update.** Send each coordinate of $\Delta_k$ at fewer bits. fp32 → fp16 halves it for free (negligible accuracy effect). fp32 → int8 quarters it; fp32 → 1-bit *sign* (signSGD-style) is a 32× reduction but throws away magnitude. The science is the same quantization-error story from [quantization from first principles](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression): quantizing to $b$ bits with step $s$ over range $R$ gives per-coordinate error variance $\sigma_q^2 = s^2/12$ with $s = R/(2^b - 1)$, so error variance falls by $4\times$ per added bit. The twist in FL is that this quantization noise behaves like extra **gradient noise**, which SGD is remarkably tolerant of — so aggressive update quantization costs far less accuracy than aggressive *weight* quantization for inference.

**Sparsification (top-k).** Send only the largest-magnitude coordinates and zero the rest. Empirically, the gradient of a deep net is *highly* compressible: in [Deep Gradient Compression (Lin et al., 2018)](https://arxiv.org/abs/1712.01887), keeping only the top **0.1%** of coordinates — a 1000× sparsification — converged to baseline accuracy on ImageNet. The catch is that you must send *indices* too, so the real cost is roughly (number kept) × (bits per value + bits per index). For top-$k$ with $k = \rho P$ coordinates, value bits $b_v$, and index bits $b_i = \lceil \log_2 P \rceil$:

$$
\text{uplink bytes} \;\approx\; \frac{\rho P\,(b_v + b_i)}{8}.
$$

**Error feedback (the trick that makes it work).** Naive top-k throws away $1 - \rho$ of the update every round, which would lose information. The fix is **error feedback** (or "residual accumulation"): the coordinates you *did not* send this round are added back into next round's update before you sparsify again. So nothing is permanently discarded — it is just *delayed*. This is what lets extreme sparsification (top-0.1%) converge to the dense baseline. It is a few lines of code and it is the difference between sparsification working and not working.

The reason error feedback works is worth making precise, because it is the same fixed-point argument that justifies stochastic rounding in quantization. Write the compressor as $\mathcal{C}(\cdot)$ and the residual carried into round $t$ as $e_t$. The client compresses $g_t + e_t$ (the true update plus the accumulated debt) and updates the debt to $e_{t+1} = (g_t + e_t) - \mathcal{C}(g_t + e_t)$. Sum the transmitted updates over $T$ rounds: $\sum_{t=1}^{T}\mathcal{C}(g_t + e_t) = \sum_{t=1}^{T} g_t + e_1 - e_{T+1}$. The transmitted total equals the *true* total plus a bounded residual term $e_1 - e_{T+1}$ that does not grow with $T$. So error feedback makes a *biased* compressor unbiased *in the long run* — the systematic error of always dropping the small coordinates cancels because those coordinates accumulate until they grow large enough to be sent. Without it, the dropped mass $\sum_t (g_t - \mathcal{C}(g_t))$ grows linearly in $T$ and the model converges to the wrong point.

### The compression ratio, made quantitative

It helps to write the achievable compression ratio as one formula so you can reason about where the bytes go. Dense fp32 sends $32P$ bits. A top-$\rho$ scheme at $b_v$ value bits with $b_i = \lceil \log_2 P\rceil$ index bits sends $\rho P (b_v + b_i)$ bits. The ratio is therefore

$$
\text{CR} \;=\; \frac{32P}{\rho P\,(b_v + b_i)} \;=\; \frac{32}{\rho\,(b_v + b_i)}.
$$

Three things fall out of this. First, the ratio is *independent of $P$* — sparsification scales to any model size. Second, at aggressive sparsity the *index* term dominates: with $\rho = 0.01$, $b_v = 8$, and $P = 25\text{M}$ so $b_i = 25$, you get $\text{CR} = 32/(0.01 \cdot 33) \approx 97\times$, and fully $25/33 = 76\%$ of your transmitted bits are *indices, not values*. That is why index encoding is where the next factor of two hides: delta-coding the sorted indices (gaps between kept positions are small and Golomb-codeable) typically cuts $b_i$ from 25 bits to 6–10, pushing the ratio past $200\times$. Third — and this is the structured-update punchline — if both server and client agree on a **fixed random mask** ahead of time (the support is pseudo-randomly chosen from a shared seed), you send *zero* index bits: $\text{CR} = 32/(\rho b_v)$, which at the same settings is $400\times$. You trade the freedom to pick the *best* coordinates for the freedom to send *no* coordinates, and on deep nets the random-mask penalty is small.

**Structured updates.** Instead of compressing a dense update after the fact, *constrain* the update to a low-rank or sparse structure during local training. [Konečný et al. (2016)](https://arxiv.org/abs/1610.05492) proposed exactly this — restricting each client's update to a low-rank matrix $\Delta_k = A_k B_k$ or a random sparse mask, so you only ever compute and send the structured part. This is the conceptual ancestor of doing **LoRA on-device** (next section): if the update is constrained to a low-rank adapter from the start, the update is small *by construction*.

![A two-column figure contrasting a dense fp32 update of roughly one hundred megabytes per round against a top-one-percent int8 update with indices and error feedback at under one megabyte per round for a half-point accuracy cost](/imgs/blogs/on-device-and-federated-learning-4.png)

Figure 4 puts the dense and compressed updates side by side. The headline: stacking top-1% sparsification with int8 quantization and error feedback takes a 100 MB dense fp32 update down to well under 1 MB — a **100×+** reduction — at a cost of roughly half a point of accuracy and a few extra rounds. That trade is almost always worth it, because the uplink is the binding constraint.

#### Worked example: uplink bytes with and without compression

Take the same 25M-parameter model, $P = 2.5 \times 10^7$, and 50 communication rounds. Index bits per coordinate are $b_i = \lceil \log_2 P \rceil = 25$ bits.

| Update scheme | Bits/coord (value) | Coords sent | Bytes/round | Bytes over 50 rounds | Acc delta |
| --- | --- | --- | --- | --- | --- |
| Dense fp32 | 32 | $2.5\times10^7$ | 100.0 MB | 5.00 GB | baseline |
| Dense fp16 | 16 | $2.5\times10^7$ | 50.0 MB | 2.50 GB | ~0.0 pt |
| Dense int8 | 8 | $2.5\times10^7$ | 25.0 MB | 1.25 GB | -0.1 pt |
| Top-1% int8 (+indices) | 8 + 25 | $2.5\times10^5$ | ~1.03 MB | 51.6 MB | -0.4 pt |
| Top-0.1% int8 (+indices) | 8 + 25 | $2.5\times10^4$ | ~0.10 MB | 5.2 MB | -0.9 pt |

Read the bottom rows carefully. Top-1% int8 sends $0.25M$ coordinates, each costing 8 value bits *plus* 25 index bits = 33 bits, so $0.25\text{M} \times 33 / 8 \approx 1.03$ MB per round — a **97× reduction** from dense fp32. Over 50 rounds you went from 5 GB to 52 MB. Notice the index overhead is *bigger than the value*: 25 index bits vs 8 value bits. That is why people use cheaper index encodings (run-length, delta-coding sorted indices, or fixed random masks where the server knows the support without you sending indices at all). The accuracy cost climbs as you sparsify harder, but stays small thanks to error feedback. For a metered-cellular fleet, going from 5 GB to 52 MB per user is the difference between "impossible" and "ships."

### Practical: a FedAvg round with compression in PyTorch

Enough math — here is the loop. First the client: load the global weights, run local SGD, compute and compress the delta.

```python
import torch
import torch.nn.functional as F

def client_update(model, global_state, loader, local_epochs=5, lr=0.05):
    """Run E local epochs of SGD, return the (model) delta vs the global weights."""
    model.load_state_dict(global_state)          # start from the broadcast model
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    for _ in range(local_epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    # delta = w_local - w_global, flattened per-tensor
    delta = {k: model.state_dict()[k] - global_state[k] for k in global_state}
    n_k = len(loader.dataset)                     # this client's sample count
    return delta, n_k
```

Now top-k sparsification with int8 quantization and error feedback. The residual buffer `err` persists across rounds on the client — that is the error-feedback mechanism.

```python
def compress_topk_int8(delta, err, frac=0.01):
    """Top-k sparsify + int8-quantize one tensor; accumulate the dropped part in err."""
    g = delta + err                               # add back last round's residual
    flat = g.flatten()
    k = max(1, int(frac * flat.numel()))
    # pick top-k by magnitude
    _, idx = torch.topk(flat.abs(), k)
    vals = flat[idx]
    # int8 quantize the kept values (symmetric, per-tensor scale)
    scale = vals.abs().max() / 127.0 + 1e-12
    q = torch.clamp((vals / scale).round(), -127, 127).to(torch.int8)
    # build the residual: everything we did NOT send stays for next round
    sent = torch.zeros_like(flat)
    sent[idx] = (q.float() * scale)               # what the server will reconstruct
    new_err = (g - sent).reshape(delta.shape)     # carry the error forward
    return (idx, q, scale, flat.numel(), delta.shape), new_err
```

And the server: decompress each client's sparse int8 update, then take the data-weighted average and apply it.

```python
def server_aggregate(global_state, client_payloads):
    """FedAvg: weighted-average the (decompressed) deltas into the global model."""
    total_n = sum(n for _, n in client_payloads)
    agg = {k: torch.zeros_like(v) for k, v in global_state.items()}
    for delta_comp, n_k in client_payloads:
        w = n_k / total_n                         # FedAvg weight = data share
        for key, (idx, q, scale, numel, shape) in delta_comp.items():
            flat = torch.zeros(numel)
            flat[idx] = q.float() * scale         # reconstruct sparse int8 delta
            agg[key] += w * flat.reshape(shape)
    # apply the averaged delta
    return {k: global_state[k] + agg[k] for k in global_state}
```

That is FedAvg with compressed uplink in about 50 lines. The pieces map one-to-one onto the math: `client_update` produces $\Delta_k$, `compress_topk_int8` shrinks the uplink with error feedback, and `server_aggregate` computes $\sum_k \frac{n_k}{n} \Delta_k$ and applies it. In production you would also clip the per-client delta and add noise here — which is differential privacy, next.

What the three functions above leave out is the *round driver* — the loop that ties them together and that you would actually run. It is short, but writing it out makes the partial-participation and per-client-residual mechanics concrete, and it is where the hyperparameters $C$ (fraction of clients sampled), $E$, and the round count $T$ live. Note one subtlety that bites people: the error-feedback residual `err` is **per client and persists across rounds**, so it must be stored keyed by client id, not reset each round — otherwise you throw away the very residual that makes top-k converge.

```python
def federated_train(make_model, client_loaders, rounds=50,
                    client_frac=0.1, local_epochs=5, lr=0.05):
    """Drive T rounds of compressed FedAvg over a fleet of clients."""
    global_state = make_model().state_dict()
    K = len(client_loaders)
    # per-client error-feedback residual, persists across rounds
    err = {ck: {k: torch.zeros_like(v) for k, v in global_state.items()}
           for ck in range(K)}
    model = make_model()
    for t in range(rounds):
        m = max(1, int(client_frac * K))            # clients this round
        S_t = random.sample(range(K), m)            # the sampled subset
        payloads = []
        for ck in S_t:
            delta, n_k = client_update(model, global_state,
                                       client_loaders[ck], local_epochs, lr)
            # compress each tensor with that client's persistent residual
            comp = {}
            for key in delta:
                comp[key], err[ck][key] = compress_topk_int8(
                    delta[key], err[ck][key], frac=0.01)
            payloads.append((comp, n_k))
        global_state = server_aggregate(global_state, payloads)
    return global_state
```

A few production realities this skeleton glosses over but that you must add for a real fleet. First, you would **over-select**: sample, say, $1.3m$ clients and aggregate the first $m$ that report, so stragglers and dropouts do not stall the round (the FedAvg time per round is set by the *slowest* client you wait for). Second, `random.sample` is a stand-in for an eligibility-and-fairness-aware scheduler — real systems only select devices that are idle, charging, and on unmetered Wi-Fi, and they cap how often any one device participates so a few always-on phones do not dominate. Third, the loop is synchronous (the server waits for the round to complete before broadcasting again); large deployments often go *asynchronous* or *buffered-asynchronous*, applying updates as they arrive with a staleness-weighted scale, which complicates the convergence story but removes the straggler tax entirely.

## Federated is not private by itself — secure aggregation and differential privacy

A dangerous misconception: "the data never leaves, so federated learning is private." It is *more* private than uploading raw data, but the **updates themselves leak**. A gradient is a function of the data that produced it, and that function is often invertible enough to recover information. [Zhu et al. (2019)](https://arxiv.org/abs/1906.08935) showed "deep leakage from gradients" — reconstructing the actual training images and labels pixel-by-pixel from the shared gradient of a single example. Membership-inference attacks can tell whether a specific record was in a client's training set just from the update. Sending deltas instead of data raises the bar; it does not clear it.

Two complementary defenses, addressing two different threat models.

**Secure aggregation** ([Bonawitz et al., 2017](https://eprint.iacr.org/2017/281)) protects against a *curious server*. The idea: clients add pairwise-cancelling random masks to their updates so that any *individual* update is encrypted noise, but when the server sums them the masks cancel and only the *aggregate* is revealed. The server learns $\sum_k \Delta_k$ (which it needs) and learns *nothing* about any single $\Delta_k$. It is a cryptographic protocol (secret sharing over a finite field, with dropout tolerance via threshold secret sharing), and it is what lets Google compute the FedAvg average without any one phone's update ever being visible. It defends against the server, but it does *not* protect against what the final *model* itself memorizes.

**Differential privacy (DP)** protects against the *model output leaking about an individual*, including against a server that sees the aggregate and an adversary who probes the released model. The mechanism is **DP-SGD** ([Abadi et al., 2016](https://arxiv.org/abs/1607.00133)) applied at the client or aggregate level: **clip** each contribution to a bounded norm, then **add calibrated Gaussian noise**.

![A two-column figure contrasting a raw averaged update with unbounded norm and exact values that risks membership inference against a differentially private update that clips to a bounded norm and adds calibrated Gaussian noise for an epsilon-delta guarantee](/imgs/blogs/on-device-and-federated-learning-6.png)

Figure 6 shows the two steps. **Clipping** bounds *sensitivity*: how much one client (or one record) can change the aggregate. You project each update onto a ball of radius $C$:

$$
\tilde{\Delta}_k \;=\; \Delta_k \cdot \min\!\left(1, \frac{C}{\|\Delta_k\|_2}\right).
$$

No single client can now move the sum by more than $C$. **Noising** then adds Gaussian noise scaled to that sensitivity, $\mathcal{N}(0, \sigma^2 C^2 I)$, to the *sum* before averaging:

$$
\hat{\Delta} \;=\; \frac{1}{|S_t|}\left(\sum_{k \in S_t} \tilde{\Delta}_k \;+\; \mathcal{N}(0,\,\sigma^2 C^2 I)\right).
$$

The pair $(\varepsilon, \delta)$ is the privacy guarantee. Formally, a mechanism is $(\varepsilon, \delta)$-differentially private if for any two datasets differing in one individual and any output set $\mathcal{O}$:

$$
\Pr[\mathcal{M}(D) \in \mathcal{O}] \;\le\; e^{\varepsilon}\,\Pr[\mathcal{M}(D') \in \mathcal{O}] \;+\; \delta.
$$

In words: adding or removing one individual changes the distribution of outputs by at most a factor $e^\varepsilon$ (plus a small slack $\delta$). Small $\varepsilon$ (say $< 1$) is strong privacy; $\varepsilon \approx 8$–$10$ is the looser end people accept in practice. The noise multiplier $\sigma$ you need grows as you demand smaller $\varepsilon$, and the **privacy–utility trade** is direct: more noise → smaller $\varepsilon$ → lower accuracy. Composition over many rounds is tracked with a *privacy accountant* (the moments accountant / Rényi DP), because each round spends some privacy budget and they accumulate.

It is worth deriving the noise–privacy relationship for a *single* release so the trade-off is not a black box. The Gaussian mechanism adds $\mathcal{N}(0, \sigma^2)$ noise to a quantity of $L_2$-sensitivity $C$ (after clipping, one client moves the sum by at most $C$). The classic analytic bound says this mechanism is $(\varepsilon, \delta)$-DP for a single query whenever

$$
\sigma \;\ge\; \frac{C\,\sqrt{2\ln(1.25/\delta)}}{\varepsilon}.
$$

Read it as a budget equation. The required noise is *inversely proportional to $\varepsilon$* — halving $\varepsilon$ (twice the privacy) doubles the noise. It scales with the sensitivity $C$, which is exactly why clipping is the lever that lets you spend less noise: a tighter clip means a smaller $C$ means less Gaussian noise for the same $\varepsilon$ — but clip too tight and you destroy the *signal* you are trying to average. And the $\delta$ dependence is gentle (square-root of a log), which is why people fix $\delta$ at something like $10^{-5}$ (smaller than one over the number of users, the rule of thumb) and then trade off $\varepsilon$ against accuracy. The noise *multiplier* you actually pass to the code, $z = \sigma/C$, drops out of this cleanly: $z \ge \sqrt{2\ln(1.25/\delta)}/\varepsilon$, independent of $C$, which is why production configs are stated as "$z = 1.1$" rather than as an absolute noise level.

For the *many-rounds* federated case, naive composition would multiply your per-round $\varepsilon$ by $T$ rounds — a privacy disaster. Two things rescue it. **Subsampling amplification**: because each round only touches a small random fraction $q$ of clients, a client's per-round privacy loss is amplified-down by roughly a factor $q$ (an adversary is unsure whether you even participated). **Tighter composition**: Rényi-DP / the moments accountant compose the per-round Rényi divergences and convert *once* at the end, giving a total $\varepsilon$ that grows like $\sqrt{T}$ rather than $T$. The combined effect is what makes a federated run of hundreds of rounds land at a single-digit $\varepsilon$ instead of a meaningless $\varepsilon$ in the thousands.

The honest summary: secure aggregation + DP together give you "the server never sees an individual update *and* the released model provably cannot reveal whether you were in the training set." That is a strong, *legally meaningful* privacy story — and it costs a few accuracy points, which for privacy-critical applications is exactly the right trade.

### Practical: clipping and noising in the aggregator

DP slots cleanly into the server step. Clip each client delta to norm $C$, sum, add Gaussian noise, then average.

```python
def dp_aggregate(global_state, client_deltas, clip_C=1.0, noise_mult=1.0):
    """DP-FedAvg: per-client clip to norm C, sum, add Gaussian noise(sigma=z*C), average."""
    m = len(client_deltas)
    agg = {k: torch.zeros_like(v) for k, v in global_state.items()}
    for delta in client_deltas:
        # global L2 norm of this client's full update across all tensors
        total_norm = torch.sqrt(sum((d ** 2).sum() for d in delta.values()))
        scale = min(1.0, clip_C / (total_norm + 1e-12))   # clip to the C-ball
        for k in agg:
            agg[k] += delta[k] * scale
    # add Gaussian noise calibrated to sensitivity C, then average over m clients
    for k in agg:
        noise = torch.randn_like(agg[k]) * (noise_mult * clip_C)
        agg[k] = (agg[k] + noise) / m
    return {k: global_state[k] + agg[k] for k in global_state}
```

The two hyperparameters are the whole story: `clip_C` bounds how much any one client can contribute (sensitivity), and `noise_mult` ($z = \sigma/C$) sets how much noise you add per unit of sensitivity. A privacy accountant (e.g. Opacus's `RDPAccountant`, or TensorFlow Privacy) converts $(z, \text{sampling rate}, \text{rounds})$ into the realized $(\varepsilon, \delta)$. Tune `clip_C` to roughly the median update norm so you do not clip everything to zero, and pick $z$ to hit your target $\varepsilon$.

#### Worked example: the privacy–utility curve at three noise levels

Let me put the trade-off on a single curve so the cliff is visible. Take a federated next-word model, $T = 200$ rounds, $\delta = 10^{-6}$, clip $C$ set to the median client-update norm, and a per-round client sampling fraction of $q = 0.01$. We sweep the noise multiplier $z$ and read off the accountant's $\varepsilon$ and the model's test accuracy.

| Noise mult $z$ | Realized $\varepsilon$ (200 rounds) | Test accuracy | Tail-class accuracy | Verdict |
| --- | --- | --- | --- | --- |
| $0.0$ (no DP) | $\infty$ | 98.3% | 91.2% | deltas leak; no guarantee |
| $0.5$ | ~21 | 97.9% | 88.5% | weak guarantee, small cost |
| $1.0$ | ~8.2 | 96.8% | 82.0% | the usual production point |
| $2.0$ | ~3.1 | 94.1% | 69.4% | strong, real accuracy hit |
| $4.0$ | ~1.2 | 88.6% | 41.7% | tail collapses |

Read it top to bottom and the shape of the trade is unmistakable. Going from no-DP to $z = 1$ (a *meaningful* $\varepsilon \approx 8$) costs about 1.5 points of overall accuracy — cheap insurance for a legally meaningful guarantee, and the production sweet spot most teams pick. But look at the **tail-class** column: it falls roughly twice as fast as the headline accuracy at every step, and at $z = 4$ (the strong $\varepsilon \approx 1.2$ regime) it has collapsed to 42% while the headline still reads a respectable 89%. That gap *is* the privacy–utility tension made concrete: DP noise is calibrated to the worst-case sensitivity, so it disproportionately erases the rare patterns — the unusual name, the dialect word, the long-tail behavior — that personalization most wants to capture and that contribute least to the average. The honest reading is that headline accuracy hides the damage; if your product *is* the long tail, you cannot push $z$ as hard as the aggregate metric suggests.

## On-device fine-tuning of a deployed model

Federated learning is one face of edge training: many devices collaborating on a shared global model. The other face is purely **local**: one device fine-tuning *its own* deployed model on *its own* data, no server, no sharing, pure personalization. Here the enemy is the memory wall from the first section — and the answer is to update only a *tiny* fraction of the parameters.

Recall why full fine-tuning rarely fits on-device. For a model with $P$ parameters, full fine-tuning needs gradients ($P$), optimizer state ($2P$ for Adam), and the full activation retention for backprop through *every* layer. For even a small 100M-parameter model that is gigabytes — impossible on a phone, laughable on a microcontroller. So you freeze almost everything and train a sliver.

There is a clean spectrum of "slivers," cheapest to most expressive:

**Last-layer / linear-probe fine-tuning.** Freeze the entire backbone; train only the final classifier head. The backbone becomes a fixed feature extractor; you only learn a new linear map on top. Trainable params drop to (feature dim × num classes) — kilobytes. And crucially, you only backprop through the *last* layer, so you barely retain activations. This is the cheapest possible on-device adaptation and is often enough when the new task shares the backbone's features (e.g. adding a new person to a face classifier).

**Adapter modules** ([Houlsby et al., 2019](https://arxiv.org/abs/1902.00751)). Insert small bottleneck MLPs (down-project to dimension $r$, nonlinearity, up-project) between frozen layers, and train only those. Typically 1–4% of parameters. More expressive than last-layer because adaptation happens throughout the network, but you do backprop deeper, so activation cost is higher.

**LoRA** (Low-Rank Adaptation, [Hu et al., 2021](https://arxiv.org/abs/2106.09685)). The most important one for modern on-device fine-tuning. Freeze the pretrained weight $W_0 \in \mathbb{R}^{d \times d}$ and learn a **low-rank update** $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, with rank $r \ll d$. The forward pass becomes:

$$
h \;=\; W_0 x \;+\; \frac{\alpha}{r}\,(B A)\,x,
$$

with $A$ initialized random-Gaussian and $B$ initialized to zero (so training starts at exactly the pretrained model). The trainable parameter count drops from $d^2$ to $2rd$. For $d = 4096$ and $r = 8$ that is $2 \cdot 8 \cdot 4096 = 65{,}536$ vs $4096^2 \approx 16.8\text{M}$ per matrix — a **256×** reduction. Because only $A$ and $B$ have gradients and optimizer state, the gradient + optimizer memory collapses by the same factor. And since the frozen $W_0$ needs no gradient, you can keep it quantized (this is **QLoRA**, [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314) — a 4-bit frozen base with fp16 LoRA adapters), which is what makes fine-tuning a multi-billion-parameter base model fit in a phone's or a single consumer GPU's memory.

The structured-update connection from earlier is exact: LoRA *is* a low-rank structured update, the same idea Konečný et al. proposed for compressing federated communication. So LoRA does double duty — it slashes both the *memory* of local fine-tuning and the *uplink* of federated fine-tuning. Federated LoRA (every client trains the same small $A, B$ and you average only those) is one of the most practical FL setups for large models today.

### Practical: a LoRA fine-tune sketch on-device

Here is LoRA wrapping a frozen linear layer, the way you would adapt a deployed model locally. Only `A` and `B` carry gradients.

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """Wrap a frozen Linear with a trainable rank-r update: h = W0 x + (alpha/r) B A x."""
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False               # freeze the pretrained weight
        d_in, d_out = base.in_features, base.out_features
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)   # random init
        self.B = nn.Parameter(torch.zeros(d_out, r))         # zero init -> starts at W0
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.t() @ self.B.t())

def add_lora(model, r=8):
    """Swap every nn.Linear for a LoRA-wrapped one; only A,B will train."""
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, name, LoRALinear(child, r=r))
        else:
            add_lora(child, r=r)
    return model

# on-device fine-tune: optimizer sees ONLY the LoRA params (tiny state)
model = add_lora(model, r=8)
trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=1e-4)       # Adam state is ~256x smaller
# ... then a normal local training loop on the device's own data ...
```

The key line is `requires_grad = False` on the base: it means the optimizer never allocates gradient or Adam-moment buffers for the frozen weights, which is where almost all the memory was. You ship the base model once (it never changes), and the user's personalization is a few hundred KB of LoRA weights that live only on their device — or, in federated LoRA, the only thing that ever gets averaged.

#### Worked example: on-device fine-tune memory, full vs LoRA vs last-layer

Take a 1.3B-parameter on-device language model (think a Gemma-2B-class model after quantization), and ask what it costs to fine-tune it three ways. Assume the frozen base is 4-bit (≈0.65 GB) and adapters/gradients are fp16.

| Strategy | Trainable params | Grad + Adam memory | Activation retention | Peak training memory | Fits on a phone? |
| --- | --- | --- | --- | --- | --- |
| Full fine-tune (fp16) | 1.3B | ~7.8 GB (3× params fp16) | full depth | ~10+ GB | No |
| Last-layer only | ~0.5M (head) | ~3 MB | last layer only | ~0.7 GB | Yes, easily |
| LoRA $r{=}8$ (QLoRA-style) | ~3.1M | ~19 MB | full depth (small) | ~1.0 GB | Yes |
| Adapters (~1%) | ~13M | ~78 MB | most depth | ~1.3 GB | Tight but yes |

The numbers tell the whole story. Full fine-tuning needs the base in fp16 plus 3× the parameter count for gradients and Adam moments — over 10 GB, dead on arrival for a phone. Last-layer is almost free (3 MB of trainable state) but limited in what it can learn. LoRA $r{=}8$ adds only ~3M trainable parameters and ~19 MB of optimizer state, keeps the base quantized, and lands around 1 GB peak — genuinely shippable, and far more expressive than last-layer because the low-rank update touches every layer. This is why LoRA, not full fine-tuning, is the default for on-device personalization of anything bigger than a small CNN.

Notice the one column that does *not* collapse with LoRA: **activation retention**. The LoRA adapters are inserted *throughout* the network, so backprop still has to flow from the loss all the way down to the earliest adapter, which means you still retain the forward activations along the whole path. LoRA kills the gradient and optimizer terms (the $2P$–$3P$ that dominated full fine-tuning) but it does *not* by itself kill the activation-sum term — that is why its peak (~1.0 GB) is far below full fine-tune but not as tiny as last-layer (~0.7 GB), where backprop stops at the final layer and almost no activations are retained. If even 1.0 GB is too much, two further levers attack the activation term directly. **Gradient checkpointing** stores activations only at a sparse set of $\sqrt{L}$ checkpoints and *recomputes* the intermediate ones during the backward pass: peak activation memory drops from $O(L)$ to $O(\sqrt{L})$ at the cost of one extra forward pass (roughly $+33\%$ compute). For a 32-layer model that is roughly a $32 \to 6$ reduction in retained activation blocks — often the difference between fitting and OOM. **Freezing the lower layers** (apply LoRA only to the top half of the blocks) cuts the backprop *depth*, so you retain activations for only the layers you actually train through; you give up some adaptation capacity in the early features in exchange for proportionally less activation memory.

### Practical: local DP-SGD when the device fine-tunes alone

Federated DP clips and noises at the *aggregate*. But when a single device fine-tunes its *own* model on its *own* data with no server — the pure local-personalization path — and you still want a formal guarantee against what the released adapter might memorize, you run **DP-SGD locally**: clip the *per-example* gradient, then add noise, every step. The mechanism is the same clip-then-noise, just at the granularity of one training example instead of one client.

```python
def dp_sgd_step(model, batch, opt, clip_C=1.0, noise_mult=1.0):
    """One DP-SGD step: per-example gradient clip + Gaussian noise (Opacus-style)."""
    xs, ys = batch
    bsz = xs.size(0)
    trainable = [p for p in model.parameters() if p.requires_grad]
    accum = [torch.zeros_like(p) for p in trainable]
    for i in range(bsz):                              # per-example gradients
        opt.zero_grad()
        loss = F.cross_entropy(model(xs[i:i+1]), ys[i:i+1])
        loss.backward()
        # L2 norm of THIS example's gradient across trainable params
        gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in trainable))
        scale = min(1.0, clip_C / (gnorm + 1e-12))   # clip each example to C
        for a, p in zip(accum, trainable):
            a += p.grad * scale
    # add Gaussian noise to the clipped sum, then average over the batch
    opt.zero_grad()
    for a, p in zip(accum, trainable):
        noisy = a + torch.randn_like(a) * (noise_mult * clip_C)
        p.grad = noisy / bsz
    opt.step()                                        # plain optimizer on noisy grads
```

This is the textbook DP-SGD inner loop (real implementations such as Opacus vectorize the per-example gradients so you do not pay the Python `for` loop, but the math is identical). Combined with LoRA it is cheap: you only ever clip and noise the *adapter* gradients — a few million numbers — not the frozen billions. So "personalize my deployed model on-device, with a provable bound on what the personalized weights can leak" is genuinely practical: LoRA shrinks what you train, DP-SGD bounds what it remembers, and nothing ever touches a server.

## Results: the full trade-off, side by side

Pulling the threads together, here is the comparison that should drive your decision. One shared task (image or next-word prediction), three training regimes, three properties that matter.

| Property | Centralized | Federated (FedAvg) | Federated + compression + DP |
| --- | --- | --- | --- |
| Test accuracy | 99.1% (baseline) | 98.3% (non-IID) | 96.8% ($\varepsilon{\approx}8$) |
| Raw data uploaded | 100% of data | 0% (deltas only) | 0% (deltas only) |
| Uplink per client (50 rounds) | n/a | 1.25 GB (int8) | 52 MB (top-1% int8) |
| Server sees individual data | yes | yes (deltas leak) | no (secure agg) |
| Membership-inference risk | high | medium (gradients leak) | low (DP $(\varepsilon,\delta)$) |
| Systems complexity | low | high | very high |

Read the columns left to right as a privacy ratchet that costs accuracy and complexity. Centralized is the most accurate, simplest to operate, and the *least* private — your data is on someone's server. Plain FedAvg buys you "data never leaves" for ~0.8 accuracy points but the deltas still leak and the operational complexity (client selection, straggler handling, non-IID drift) jumps. Adding compression and DP buys you a *legally meaningful* privacy guarantee and a 24× smaller uplink, at ~2.3 total accuracy points and a serious jump in systems complexity (secure-aggregation crypto, privacy accounting, drift mitigation). There is no free lunch; there is a well-understood menu, and you order from it based on how much privacy you actually need.

And the on-device fine-tuning memory comparison from the worked example, summarized as the rule of thumb you will actually use: **last-layer when the backbone's features already fit the task and you need it nearly free; LoRA when you need real adaptation through the network on a tight memory budget (the default); full fine-tune basically never on the edge.** Adapters sit between LoRA and full — slightly more capacity, meaningfully more memory.

## Federated variants: a quick map

You will see a zoo of "Fed-" algorithms. Here is the small map that covers most of what you need, organized by what each one trades.

![A four-by-four comparison matrix of FedSGD, FedAvg, FedProx, and DP-FedAvg across local work, communication cost, non-IID robustness, and formal privacy](/imgs/blogs/on-device-and-federated-learning-7.png)

Figure 7 lays the four canonical variants against the four things you care about. **FedSGD** does one gradient step per round: no drift (it equals centralized SGD), but it is communication-hungry. **FedAvg** does $E$ local epochs per round: communication-cheap, but it drifts on non-IID data. **FedProx** is FedAvg plus a proximal term that pulls local solutions back toward the global model, buying back non-IID robustness for a small extra hyperparameter. **DP-FedAvg** is FedAvg with clip-and-noise, adding a formal $(\varepsilon, \delta)$ guarantee at an accuracy cost. They compose: production systems are usually "DP-FedAvg with FedProx-style regularization and compressed updates," because you want communication efficiency *and* drift control *and* privacy at once. Two more worth knowing: **SCAFFOLD** uses control variates to correct client drift more aggressively than FedProx, and **personalized FL** (per-client heads or per-client LoRA on a shared base) accepts that one global model is not the goal — each client keeps a tailored piece.

## A stress test: when does this actually break?

Let me reason through the failure modes the way I would in a design review, because the happy-path numbers above hide real cliffs.

**What happens when participation is tiny and biased?** If only phones that charge overnight in one time zone participate, your "global" model is actually a model of nighttime-charging users in that region. The fix is not algorithmic — it is *sampling discipline*: stratify client selection, monitor the participating-population distribution against the target population, and re-weight. If you cannot get a representative sample, federated learning will faithfully learn a biased model, and no amount of averaging math saves you.

**What happens at extreme non-IID?** With one class per client and many local epochs, FedAvg can *diverge* — the averaged model is worse than the initialization. The pattern in a training log is unmistakable and worth recognizing: global accuracy climbs for a few rounds, then *oscillates or falls* as the round-to-round averaging starts cancelling updates that point in opposing directions. Concretely, on the one-class-per-client split with $E = 20$, you might see global accuracy rise to 60% by round 10, then sawtooth between 45% and 62% and never break 70% — because each round, the sampled clients drag the model hard toward their single class, and the average of "pull toward 3" and "pull toward 8" lands nowhere useful. Drop $E$ from 20 to 2 on the *same* split and the sawtooth smooths into a clean climb to ~95%, at the cost of roughly $10\times$ more rounds. That is the drift-versus-communication dial from the convergence bound, lived out in a real log. The first lever is therefore to *reduce* $E$ (fewer local steps = less drift, more communication). The second is FedProx/SCAFFOLD — adding the proximal term $\frac{\mu}{2}\|w-w_t\|^2$ with $\mu = 0.1$ typically recovers most of the lost accuracy by tethering each client to the global model. The third, often the real answer, is **personalization**: stop trying to force one global model onto fundamentally different clients and let each keep a personal head or adapter. The clean test for "is this a non-IID problem?" is to run the *same* training on an artificial IID reshuffle of the data; if the IID run converges fine and the real run does not, the heterogeneity term $\Gamma$ is your bottleneck, not your optimizer or learning rate.

**What happens when the update compression is too aggressive?** Push sparsification to top-0.01% and convergence stalls — error feedback delays information but cannot conjure signal that was never sent. The symptom is a loss curve that plateaus high. The fix is to back off the sparsity or warm up with denser updates early (when gradients are large and information-rich) and sparsify more later.

**What happens to the DP accuracy at small $\varepsilon$?** Demand $\varepsilon < 1$ and the noise can swamp the signal, especially for the rare, long-tail examples that personalization most wants to capture — DP noise disproportionately erases the rare. There is a genuine tension between *strong* DP and *good* personalization on the tail. The honest move is to be explicit about which $\varepsilon$ your use case actually requires legally, and not pay for more privacy than you need.

**What happens when the device is memory-bound, not compute-bound during local training?** This is the activation-retention wall from the first section. Even with LoRA's tiny gradients, you still retain forward activations to backprop through the frozen base. Two mitigations: **gradient checkpointing** (recompute activations in the backward pass instead of storing them — trade compute for memory, the classic time–memory trade), and freezing *more* of the lower layers so you backprop through fewer of them. This is the same memory reasoning as the rest of the series; on-device training just makes it acute.

## Case studies: federated learning in the wild

These are real, shipped systems — the proof this is not a paper-only technique.

**Gboard next-word prediction (Google, 2018–).** The flagship. [Hard et al. (2018)](https://arxiv.org/abs/1811.03604) describe federated training of an RNN language model for next-word prediction across Gboard's user base. Clients train when idle/charging/on-Wi-Fi, secure aggregation hides individual updates, and the federated model *outperformed* the server-trained baseline on next-word prediction accuracy — because the on-device data better matched real typing than any server-side corpus. The raw text was never uploaded. This is the existence proof that federated learning works at the scale of hundreds of millions of devices.

**Gboard with formal DP (Google, 2022–).** Later work added *provable* differential privacy to production Gboard models — training language models with DP-FedAvg and publishing the realized $\varepsilon$ guarantees, demonstrating that you can ship a model with a formal privacy bound at production scale, not just in a paper. This closed the "deltas still leak" gap that plain FedAvg leaves open.

**Apple on-device personalization.** Apple personalizes QuickType, dictation, and Photos with on-device learning, combining local adaptation with differential privacy on any aggregate signals that *do* go back (their well-documented local-DP work on emoji and new-word frequencies). The pattern — adapt locally, share only DP-noised aggregates — is the same menu we derived.

**Federated LoRA for on-device LLMs (2023–).** The current frontier: keep a quantized base model frozen on each device, train only small LoRA adapters locally, and federate by averaging *only the adapters*. Because the adapters are tiny (a few MB), the uplink problem largely evaporates and the per-device memory fits. This is the convergence of every idea in this post — federated averaging, structured/low-rank updates, parameter-efficient fine-tuning, and quantization — into one practical recipe for personalizing large models privately.

## When to reach for on-device training (and when not to)

Now the decision, made plainly. The honest default is: **most teams should train centrally and push the model over the air.** Federated learning and on-device training carry real costs — systems complexity (client orchestration, secure aggregation, privacy accounting), accuracy loss from non-IID and DP noise, debugging difficulty (you cannot inspect the data), and a long road to production. Do not pay those costs unless you have a reason that *requires* them.

![A decision tree rooted at a new training requirement that first asks whether the data can leave the device, routing to central training plus over-the-air push when it can and to federated or local fine-tuning when it must stay](/imgs/blogs/on-device-and-federated-learning-8.png)

Figure 8 is the decision. The very first question is the gate: **can the raw data leave the device?** If there is no regulatory, contractual, or trust barrier to moving the data, **train centrally and push over the air** — it is simpler, more accurate, easier to debug, and easier to iterate. (When this series ships the edge-MLOps post on the deployment lifecycle, the OTA-push path is exactly what it covers; for now, central-train-then-ship is the path you already know.) Only when the data *must* stay on-device do the federated costs become worth paying. And even then, branch again: if you want a *shared* model that benefits from everyone's data, you want **FedAvg + DP**; if you only need *per-user* personalization with no cross-user sharing, you want a purely **local fine-tune** (LoRA or last-layer) and you never need a server at all.

Concretely, reach for on-device/federated training when:

- **The data is privacy-critical or regulated** — health, finance, biometrics, anything under GDPR/HIPAA/data-residency law where "the data physically never moves" is a hard requirement. This is the strongest reason.
- **Personalization is the product** — keyboards, hearing aids, recommendation, anything where the per-user model beats the global model and the per-user data is sensitive.
- **Uploading the data is infeasible** — continuous sensor streams, bandwidth-constrained fleets, or data volumes that dwarf the model delta.
- **Distribution shift is fast and local** — you need adaptation within hours, faster than a central retrain-and-ship loop.

And do *not* reach for it when: a central retrain plus OTA push hits your targets (it usually does); your data is not sensitive (then just collect it); you cannot get a representative client sample (federated will faithfully learn a biased model); or your team does not have the systems depth to run client orchestration, secure aggregation, and privacy accounting in production. Federated learning is a power tool. Used for the right reason it is irreplaceable; used by default it is a lot of complexity for a model that is slightly worse.

The honest limits, stated once more: on-device training fights the activation-retention memory wall, so it leans hard on parameter-efficient methods (LoRA, last-layer) and gradient checkpointing; non-IID data costs accuracy and demands drift mitigation; differential privacy costs accuracy, especially on the long tail; and the systems complexity is genuinely high. None of these is a dealbreaker for the right use case — Gboard ships — but all of them are real, and pretending otherwise is how on-device-training projects die in the prototype stage.

## Key takeaways

- **Training is roughly an order of magnitude more expensive than inference** on the same device, because backprop must retain *all* layer activations (the sum, not the max) plus gradients and optimizer state. Everything else in this post is a way of fitting training into a budget it does not naively fit into.
- **FedAvg = data-weighted average of locally-trained models.** $w_{t+1} = \sum_k \frac{n_k}{n} w_k^{t+1}$. With one local step it equals centralized SGD; with many local steps it slashes communication rounds by 10–100× — at the cost of client drift on non-IID data.
- **The uplink is the bottleneck, not the device.** Compress aggressively: top-k sparsification (with *error feedback*, non-negotiable) plus int8 quantization shrinks updates 100×+ for under a point of accuracy. Structured low-rank updates compress *by construction*.
- **Federated is not private by itself.** Gradients leak. Add **secure aggregation** (server never sees an individual update) and **differential privacy** (clip to norm $C$, add Gaussian noise, get an $(\varepsilon, \delta)$ guarantee) — costing a few accuracy points for a legally meaningful privacy story.
- **Never full-fine-tune on the edge.** Update a sliver: last-layer (kilobytes, cheapest), LoRA (the default — ~256× fewer trainable params, keeps the base quantized), or adapters (between the two). LoRA is also a structured update, so it doubles as uplink compression in federated LoRA.
- **The decision is gated on one question: can the data leave the device?** If yes, train centrally and push over the air — simpler and more accurate. If no, federate (for a shared model) or fine-tune locally (for per-user personalization).
- **Be honest about the costs.** Non-IID data, DP noise, biased participation, and systems complexity are all real. Use this machinery when privacy or personalization *requires* it, not by default.

## Further reading

- **McMahan, Moore, Ramage, Hampson, Aguera y Arcas (2017), "Communication-Efficient Learning of Deep Networks from Decentralized Data"** — the FedAvg paper. Read it for the derivation and the non-IID experiments.
- **Konečný, McMahan, Yu, Richtárik, Suresh, Bacon (2016), "Federated Learning: Strategies for Improving Communication Efficiency"** — structured and sketched updates, the communication-compression foundation.
- **Kairouz, McMahan, et al. (2021), "Advances and Open Problems in Federated Learning"** — the definitive survey; everything from systems to privacy to personalization.
- **Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar, Zhang (2016), "Deep Learning with Differential Privacy"** — DP-SGD: clipping, noising, and the moments accountant.
- **Li, Sahu, Zaheer, Sanjabi, Talwalkar, Smith (2020), "Federated Optimization in Heterogeneous Networks" (FedProx)** — the proximal fix for non-IID client drift.
- **Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Chen (2021), "LoRA: Low-Rank Adaptation of Large Language Models"** and **Dettmers et al. (2023), "QLoRA"** — the parameter-efficient fine-tuning methods that make on-device adaptation fit.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever frame, [memory is the real constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint) for why activation retention is the wall, [the metrics that actually matter on-device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) for measuring honestly, [TinyML on microcontrollers](/blog/machine-learning/edge-ai/tinyml-on-microcontrollers) for the extreme end of the budget, and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) that ties every lever together.

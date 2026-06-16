---
title: "What to distill: response, feature, and relation-based knowledge transfer"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Soft logits are only one signal a teacher can give you — learn how to also transfer its internal features and the geometry of its representations, and exactly when that extra plumbing pays off."
tags:
  [
    "edge-ai",
    "model-optimization",
    "knowledge-distillation",
    "feature-distillation",
    "relational-knowledge",
    "attention-transfer",
    "inference",
    "efficient-ml",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/what-to-distill-response-feature-relation-1.png"
---

A while ago I was handed a teacher that was almost embarrassingly good — a ResNet-152 sitting at 78% top-1 on our internal image task — and a hard ceiling for what could ship: a model small enough to run at batch 1 in under eight milliseconds on a mid-range phone NPU, which in practice meant something around the size of a ResNet-18. The textbook move is knowledge distillation: train the small student against the big teacher's softened logits and watch it pick up a point or two over training on hard labels alone. I did exactly that, expecting magic, and got a shrug. The student gained about half a point. The teacher knew so much more than the student was learning, and the only thing I was letting it teach was a 1000-number vector at the very end of a 152-layer network.

That experience is the whole reason this post exists. **Soft logits are real knowledge — but they are the *thinnest* slice of what a teacher has to offer.** Behind that final softmax sits a deep stack of internal representations: edge detectors, texture banks, part detectors, a learned geometry in which similar images cluster and dissimilar ones spread apart. When the capacity gap between teacher and student is small, copying the logits is enough, because the student can re-derive the rest on its own. When the gap is wide — a 152-layer teacher and an 18-layer student, or a 7B language model and a 125M one — logit matching saturates. The student literally cannot reach the teacher's accuracy through the output layer alone, and you have to open up the teacher and distill from *inside* it.

This post is a tour of the three things you can transfer, organized the way the literature now organizes them (following the taxonomy in Gou et al.'s 2021 survey): **response-based** knowledge (the logits — the Hinton baseline), **feature-based** knowledge (intermediate activations — FitNets, attention transfer), and **relation-based** knowledge (the relationships *between* examples or layers — FSP, RKD, CRD). Figure 1 is the map; keep it in view. For each family we will derive the loss, write runnable PyTorch (including the hooks that grab intermediate features and the projector that bridges a width mismatch), and put real numbers on the gain. By the end you should be able to look at a teacher–student pair, estimate the capacity gap, and decide in about a minute whether plain logit distillation will do or whether it is worth wiring up feature and relation losses — and know how to compose them with quantization and pruning when the student still has to fit on a microcontroller.

![A tree diagram showing knowledge distillation branching into three families response, feature, and relation, each leading to its canonical method soft targets, hints with attention transfer, and RKD with CRD](/imgs/blogs/what-to-distill-response-feature-relation-1.png)

This is a deep-dive in the *distillation* lever of the four-lever frame this series keeps coming back to — quantization, pruning, distillation, efficient architecture — so before we start, the one-sentence placement: distillation changes the model's *topology* (you get a genuinely smaller dense network at the end, not a sparse or low-bit version of the big one), which is exactly why it composes so cleanly with the other levers. You distill first to get a small dense student, then quantize and prune *that*. We will come back to composition at the end; for the map of how all four levers fit together, see [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression).

## 1. Why logits are not enough across a wide gap

Let me make the failure I opened with precise, because it motivates everything else.

Response-based distillation, the original Hinton–Vinyals–Dean recipe from 2015, trains the student to match the teacher's *softened* output distribution. You take the teacher's logits $z^{(t)}$, divide by a temperature $T > 1$, and softmax them into a soft probability distribution that reveals the teacher's *dark knowledge* — the relative probabilities it assigns to the wrong classes. A teacher that sees a photo of a husky does not just say "dog"; it says "dog 0.7, wolf 0.2, malamute 0.08, cat 1e-6," and that ranking encodes which classes are visually confusable, which is information the hard one-hot label throws away. The student learns to reproduce that whole distribution. We covered the mechanics of response distillation — temperature, the $T^2$ gradient-scaling factor, the $\alpha$ blend with the hard-label loss — in [knowledge distillation fundamentals](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals); this post assumes you have that baseline and asks the next question: what if the baseline is not enough?

Here is the intuition for *why* it saturates across a wide gap, and it is worth slowing down on. The soft-target loss only constrains the student at the *output*. It says nothing about *how* the student should arrive at that output. A small student has a limited "hypothesis budget" — a limited number of internal feature detectors it can learn. Given only an output target, it has to discover, on its own and from scratch, an internal representation rich enough to produce teacher-like logits. For a small gap that is fine: the student is capable of finding such a representation, and the soft targets nudge it toward the teacher's decision boundaries. For a wide gap the student gets lost. The optimization landscape it has to traverse to reach a teacher-quality representation is long and full of bad local minima, and a single scalar-per-class target at the very end provides almost no gradient signal about the *intermediate* structure it needs. The teacher, meanwhile, is sitting right there with that intermediate structure fully formed. Feature and relation distillation are about handing the student that structure directly, so it does not have to rediscover it.

There is a cleaner way to say this in information terms. The mutual information between the teacher's *final logits* and an input is bounded by the task's label entropy — for a 1000-class problem that is at most $\log_2 1000 \approx 9.97$ bits per example. The mutual information between the teacher's *internal activations* and the input is vastly larger; those activations carry texture, pose, lighting, part layout — far more than nine bits. When you distill only logits, you are pushing at most ten bits per example through the pipe. When you distill features, you open a much wider channel. Tian et al. made this explicit in CRD (2020): they frame distillation as *maximizing mutual information* between teacher and student representations, and show that the logit-only objective is a loose lower bound on that information. The whole "what to distill" question is really "how wide a channel do you want to open, and what does each width cost you."

Let me push the gradient argument one more step, because it is the cleanest way to see why feature supervision is not just "more data" but *better-conditioned* data. Consider a student with parameters $\theta$ producing an intermediate feature $F = f_\theta(x)$ at some layer, and downstream of it a head $g$ producing logits $z^{(s)} = g(F)$. When you supervise only at the logits, the gradient that reaches the early parameters $\theta$ has to travel back through the entire head $g$: by the chain rule, $\frac{\partial \mathcal{L}_{\text{KD}}}{\partial \theta} = \frac{\partial \mathcal{L}_{\text{KD}}}{\partial z^{(s)}} \cdot \frac{\partial z^{(s)}}{\partial F} \cdot \frac{\partial F}{\partial \theta}$. That middle Jacobian $\partial z^{(s)} / \partial F$ is exactly where wide-gap students suffer: if the head is untrained or the network is deep, that term is small or poorly conditioned, and almost no usable signal survives the trip back to the early layers — this is the vanishing/attenuated-gradient story dressed in distillation clothes. A feature loss attaches *directly* to $F$: its gradient is $\frac{\partial \mathcal{L}_{\text{feat}}}{\partial \theta} = \frac{\partial \mathcal{L}_{\text{feat}}}{\partial F} \cdot \frac{\partial F}{\partial \theta}$, with no long Jacobian chain to attenuate it. You are injecting a strong, well-conditioned gradient right where the wide-gap student needs it most. That is the mechanical reason FitNets framed the hint as a *pre-training* signal for the lower half of the network, and it is why feature distillation helps precisely in the regime where logit distillation stalls.

## 2. The three families, defined properly

Let me give each family the same treatment — *what it matches*, *the loss*, *why it helps* — so the comparison at the end is apples to apples. The three families are not exclusive; production recipes combine them, and the last sections are about exactly that. But you have to understand each one in isolation first.

### 2.1 Response-based: match the outputs

This is the baseline, recapped in one paragraph so the post stands alone. Let $z^{(t)}$ and $z^{(s)}$ be the teacher and student logits. Define the temperature-softened distributions $p^{(t)}_i = \mathrm{softmax}(z^{(t)}/T)_i$ and $p^{(s)}_i = \mathrm{softmax}(z^{(s)}/T)_i$. The response (or "KD") loss is the KL divergence between them, scaled by $T^2$ so its gradient magnitude stays comparable to the hard-label term as $T$ changes:

$$\mathcal{L}_{\text{KD}} = T^2 \cdot \mathrm{KL}\!\left(p^{(t)} \,\|\, p^{(s)}\right) = T^2 \sum_i p^{(t)}_i \log \frac{p^{(t)}_i}{p^{(s)}_i}.$$

You blend it with ordinary cross-entropy on the true labels $y$: $\mathcal{L} = \alpha \mathcal{L}_{\text{KD}} + (1-\alpha)\,\mathrm{CE}(y, p^{(s)}_{T=1})$. **What it matches:** the output distribution. **Why it helps:** dark knowledge — the inter-class similarity structure encoded in the soft probabilities. **Where it stops helping:** when the student cannot form an internal representation good enough to *produce* teacher-like outputs, no amount of output supervision rescues it.

One detail that matters for the comparisons later: the response loss is *permutation-and-architecture agnostic*. It only ever sees a length-$K$ probability vector, so the teacher and student can be wildly different inside — different depths, widths, even different operator families — and the loss is still perfectly well defined. That is response distillation's quiet superpower: it is the only family that needs *zero* assumptions about the student's internals, which is why it is always the right first move and the only family that costs you no plumbing. The price of that generality is exactly the saturation problem: by refusing to look inside, it cannot *help* inside either. Feature methods buy interior leverage at the cost of having to align interiors; relation methods buy a middle ground by matching interior *geometry* without requiring matched interior *axes*. Keep that trade — generality versus leverage — in mind; it is the spine of the whole taxonomy.

### 2.2 Feature-based: match the intermediate activations

The first and most direct way to open the channel wider is to supervise the student's *hidden layers*, not just its output. This is FitNets (Romero et al., 2015) — the paper that coined "hints." The idea: pick a layer in the teacher (a "hint" layer) and a layer in the student (a "guided" layer), grab the activation tensors at both, and add a loss that pushes the student's activation toward the teacher's.

The immediate problem is dimensionality. A teacher hint layer might output a $256$-channel feature map; the thin student's guided layer outputs $64$ channels. You cannot subtract a 64-channel tensor from a 256-channel one. FitNets solves this with a small learned **regressor** $r(\cdot)$ — in practice a $1\times1$ convolution (for CNNs) or a linear layer (for transformers) — that projects the student's feature into the teacher's channel space. With $H$ the teacher hint tensor and $F$ the student guided tensor, the hint loss is a plain squared error in the projected space:

$$\mathcal{L}_{\text{hint}} = \tfrac{1}{2}\,\big\| H - r(F) \big\|_2^2.$$

That projector $r$ is trained jointly with the student and discarded after distillation — it exists only to make the comparison well defined, and adds zero inference cost. **What it matches:** the actual values of an intermediate activation map. **Why it helps:** it gives the student a per-spatial-location, per-channel target deep inside the network, an enormously richer gradient signal than a single softmax. FitNets' original framing was even more pointed: use the hint as a *pre-training* stage. First train the student's bottom half to mimic the teacher's hint layer, then do standard KD on the whole thing. The hint gives the student a good initialization for its lower layers, which is precisely the part a wide-gap student struggles to learn from output supervision alone. Figure 2 shows the projector bridging the width mismatch.

![A dataflow graph showing a teacher block and a thin student block, with the student feature passing through a one by one convolution projector to match the teacher channel count before an L2 hint loss compares them](/imgs/blogs/what-to-distill-response-feature-relation-2.png)

There is a subtlety worth flagging now and returning to in the practical section: matching *raw* feature values is a strong, sometimes too-strong, constraint. You are forcing the student's 64 projected channels to reconstruct the teacher's 256 channels exactly. If the student's capacity is genuinely lower, this can over-constrain it — you are asking it to be a compressed copy of the teacher's representation rather than to find its *own* good representation that happens to be similarly expressive. Attention transfer is the next family's answer to that.

It is worth being precise about *why* raw-feature matching can over-constrain, because it explains a whole family of "softer" feature methods. Two networks trained on the same task routinely learn representations that are equivalent *up to a transformation* — a rotation of the feature axes, a permutation of channels, a per-channel rescaling. All of those produce the same downstream behavior, so none of them is "wrong." But a plain $\ell_2$ hint loss is *not* invariant to any of them: it penalizes the student for representing the right information along different axes than the teacher. You can spend real capacity forcing the student to align axes that did not need aligning, capacity it could have spent on accuracy. The fixes all amount to making the loss invariant to the nuisance transformation. Attention transfer is invariant to channel identity (it sums over channels). Relation losses are invariant to rotation and scale of the whole space (they only look at distances and angles). CRD is invariant to anything that preserves which examples are "the same." Seen this way, the progression response → feature-hint → attention → relation → contrastive is a progression of *increasing invariance*: each step matches a coarser, more transformation-robust summary of the teacher, trading exactness for the freedom the student needs when its capacity is genuinely smaller. That single idea organizes the whole rest of the post.

### 2.3 Feature-based, softer: attention transfer

Zagoruyko and Komodakis (2017) noticed that you do not need to match every channel of an activation map — you can match the much lower-dimensional thing that says *where the network is looking*. For a convolutional activation tensor $A \in \mathbb{R}^{C \times H \times W}$, define a spatial **attention map** by collapsing the channel dimension. Their best-performing choice is the sum of squared activations across channels:

$$Q(A) = \sum_{c=1}^{C} |A_c|^2 \;\in\; \mathbb{R}^{H \times W}.$$

This $H\times W$ map is large where *some* channel fires strongly — that is, where the network is paying attention. The attention-transfer loss flattens each map to a vector, $\ell_2$-normalizes it (so you match the *shape* of attention, not its raw magnitude), and takes the squared difference between teacher and student at a set $\mathcal{I}$ of layer pairs:

$$\mathcal{L}_{\text{AT}} = \sum_{j \in \mathcal{I}} \left\| \frac{\mathrm{vec}(Q^{(t)}_j)}{\|\mathrm{vec}(Q^{(t)}_j)\|_2} - \frac{\mathrm{vec}(Q^{(s)}_j)}{\|\mathrm{vec}(Q^{(s)}_j)\|_2} \right\|_2.$$

Notice the win: the attention map is $H\times W$ regardless of channel count, so **there is no dimension-mismatch problem at all** — a 256-channel teacher block and a 64-channel student block both collapse to the same spatial resolution, and you only need the spatial sizes to match (a single interpolation if they do not). No projector, no over-constraining the student to reconstruct every channel. You are telling the student "look where I look," not "compute exactly what I compute." Figure 3 contrasts matching raw features against matching attention maps.

![A two-column before and after diagram contrasting matching full C-channel feature maps with an L2 loss against collapsing channels into a single spatial attention map and matching the normalized maps](/imgs/blogs/what-to-distill-response-feature-relation-3.png)

Attention transfer is, in my experience, the feature method with the best effort-to-payoff ratio for CNNs. It is cheap (one sum-of-squares per layer), it sidesteps the projector entirely, and the spatial-attention signal is exactly the kind of structure a small student has trouble discovering alone. It is the first thing I reach for when logits underdeliver on a vision model.

### 2.4 Relation-based: match the geometry, not the points

The third family steps back from "what does the network compute for *this* input" to "how does the network *organize* inputs relative to each other." The insight is that two networks can have completely different internal feature spaces — different dimensions, different axes, different absolute positions for every example — and still represent the *same relationships*: the same images are close together, the same ones are far apart, the same triangles of three examples have the same shape. That *relational* structure is what relation-based distillation transfers.

The earliest version is FSP (Yim et al., 2017), which matches the relationship *between two layers* of the same network via a Gram-style flow matrix — the inner product between the feature maps of an early and a late layer, capturing "how information flows" between them. The more widely used modern version is **Relational Knowledge Distillation** (RKD, Park et al., 2019), which matches relationships *between examples in a batch*. RKD has two terms.

The **distance** term: for every pair of examples $(i, j)$ in the batch, compute the Euclidean distance between their teacher embeddings and between their student embeddings, normalize each by the mean pairwise distance in the batch (so the two spaces' overall scales do not have to match), and penalize the difference with a Huber loss. With teacher embeddings $t_i$ and student embeddings $s_i$, and $\psi_D(\cdot)$ the mean-normalized pairwise distance,

$$\mathcal{L}_{\text{RKD-D}} = \sum_{(i,j)} \ell_\delta\!\Big( \psi_D(t_i, t_j),\; \psi_D(s_i, s_j) \Big), \quad \psi_D(x_i, x_j) = \frac{\|x_i - x_j\|_2}{\mu}.$$

The **angle** term: for every triplet $(i, j, k)$, compute the angle at the middle point $j$ formed by the other two embeddings, and match teacher to student. The angle is a higher-order relation than distance — it captures the *shape* of the local cloud, not just spacing:

$$\mathcal{L}_{\text{RKD-A}} = \sum_{(i,j,k)} \ell_\delta\!\Big( \cos\angle\,t_i t_j t_k,\; \cos\angle\,s_i s_j s_k \Big), \quad \cos\angle\,x_i x_j x_k = \langle \mathbf{e}^{ji}, \mathbf{e}^{jk}\rangle,$$

where $\mathbf{e}^{ji} = \frac{x_i - x_j}{\|x_i - x_j\|_2}$ are the unit vectors from $j$ to its neighbors. **What it matches:** pairwise distances and triplet angles between sample embeddings. **Why it helps across a wide gap:** it never asks the student to put any single example at any *particular* point — it only asks the student to preserve the *relative* arrangement. That is a much gentler constraint than "reconstruct the teacher's feature for example $i$," and a small student can satisfy it even when it genuinely cannot match the teacher point-for-point. Figure 4 contrasts individual (point-matching) distillation with relational (geometry-matching) distillation.

![A two-column before and after diagram contrasting individual distillation that matches each sample point to the teacher against relational distillation that matches pairwise distances and triplet angles between samples](/imgs/blogs/what-to-distill-response-feature-relation-4.png)

Why does matching geometry survive a wide gap when matching points does not? Here is the argument in one line. A small student cannot, by hypothesis, place every example exactly where the teacher does — it has fewer degrees of freedom, so the map from inputs to its embedding space is necessarily lower-rank. But the *relational* targets do not ask for that. They ask only that the student's lower-dimensional embedding preserve the teacher's distances and angles, and a great deal of high-dimensional structure projects down to a lower dimension while preserving pairwise distances surprisingly well — this is the same phenomenon that makes random projections and the Johnson–Lindenstrauss lemma work. Relational distillation is, in effect, asking the student to learn a *distance-preserving compression* of the teacher's geometry, which is a task a smaller network can actually accomplish, rather than a *coordinate-exact copy*, which it cannot. That is the formal version of "match the shape of the cloud, not the positions of the points."

The fully modern relational method is **Contrastive Representation Distillation** (CRD, Tian et al., 2020), which I mentioned earlier. CRD frames distillation as maximizing mutual information between teacher and student representations of the *same* input, implemented with a contrastive (InfoNCE-style) loss that pulls a student embedding toward its teacher's embedding of the same example (the positive) and pushes it away from teacher embeddings of *other* examples (negatives, drawn from a memory bank). Concretely, with a critic $h$ scoring teacher–student embedding pairs, the objective is a noise-contrastive estimator: maximize $\log h(t_i, s_i)$ for the matched (positive) pair and $\log(1 - h(t_j, s_i))$ over $N$ negatives $t_j$ sampled from the memory bank. Optimizing it pushes a tractable *lower bound* on the mutual information $I(t; s)$ upward — the bound tightens as you add negatives, which is why CRD wants a sizable bank (16,384 negatives in the paper). It is more machinery than RKD — you maintain a memory bank of embeddings and a contrastive objective with its own projection heads — but it consistently tops the leaderboards on the standard CIFAR-100 and ImageNet distillation benchmarks, especially across architectures (e.g., a ResNet teacher to a totally different student family). RKD is the one I reach for when I want most of the benefit with a fraction of the plumbing; CRD is the one to reach for when you are squeezing the last point out and can afford the memory bank.

A word on FSP and "layer-relation" methods, since they are the third corner of the relational family and people forget them. FSP (Flow of Solution Procedure) matches not relationships between *examples* but relationships between *layers* of the same network: for an early feature map $F_1 \in \mathbb{R}^{C_1 \times H \times W}$ and a later one $F_2 \in \mathbb{R}^{C_2 \times H \times W}$, it forms the $C_1 \times C_2$ Gram matrix $G = \frac{1}{HW}\sum_{h,w} F_1(h,w)\, F_2(h,w)^\top$ — essentially "how does information at layer 1 correlate with information at layer 2" — and matches the teacher's and student's flow matrices with an $\ell_2$ loss. The appeal is that the flow matrix is $C_1 \times C_2$ regardless of spatial size, and it captures *process* knowledge ("how the network transforms features from one stage to the next") rather than *state* knowledge. In practice I find RKD and CRD more robust and easier to tune, but FSP is the conceptual bridge between feature methods (which match states) and relational methods (which match structure), and it is worth knowing the lineage.

### 2.5 Where each signal is tapped

Stepping back, the three families differ most simply in *where in the network they read*. Response reads the very end (logits). Feature reads one or more intermediate blocks (the hint layer, the attention layers). Relation reads the penultimate embedding (RKD/CRD) or the relationship *across* layers (FSP). Figure 5 shows the taps stacked from input to output. This matters practically because it tells you *what you have to instrument*: response needs nothing but the forward output; feature and relation need you to reach inside the network and grab activations mid-forward-pass, which in PyTorch means hooks. That is the next section.

![A vertical stack diagram showing a network from input through early blocks, mid blocks, penultimate embedding, and logits, with feature hint, attention, relation, and response taps reading at successively deeper points and summing into a multi-term loss](/imgs/blogs/what-to-distill-response-feature-relation-5.png)

## 3. The layer-selection and weighting problem

Before the code, two design decisions decide whether feature and relation distillation help or hurt: *which* layers to pair, and *how much* to weight each term. These are the parts beginners get wrong, so let me be opinionated.

**Layer pairing.** For feature distillation you must pair a teacher layer with a student layer that represent *roughly the same representation level*. Pairing the teacher's deep semantic layer with the student's shallow edge-detector layer asks the student's early layer to compute something it structurally cannot, and you get a worse model than no hint at all. The reliable heuristic is to pair by *relative depth* and by *stage boundaries*: in a ResNet, match the output of the teacher's stage-2 to the student's stage-2, stage-3 to stage-3, and so on — the points where spatial resolution drops, which are natural representation boundaries shared across depths. For attention transfer, Zagoruyko and Komodakis specifically place taps at the end of each residual group, which is the same idea. Do not over-tap; two or three well-chosen pairs beat ten arbitrary ones, both because each adds a hyperparameter and because deep taps that the student cannot satisfy create destructive gradients.

**Weighting.** The combined loss is a weighted sum, $\mathcal{L} = (1-\alpha)\,\mathrm{CE} + \alpha\,\mathcal{L}_{\text{KD}} + \beta\,\mathcal{L}_{\text{feat}} + \gamma\,\mathcal{L}_{\text{rel}}$, and the terms live on wildly different scales — a sum-of-squares feature loss can be orders of magnitude larger than a KL. The practical rule: scale each auxiliary term so that, at the *start* of training, its gradient norm is within roughly an order of magnitude of the cross-entropy gradient. In practice people tune $\beta$ and $\gamma$ on a log grid (RKD's paper uses $\beta=25$ for distance and $\gamma=50$ for angle on metric tasks, but those numbers are loss-implementation-specific; do not copy them blindly). A clean trick is a short *warm-up* where you train with feature/relation losses first and only ramp in the response loss once the student's internal representation has been seeded — this is the spiritual descendant of FitNets' two-stage recipe and it stabilizes wide-gap training noticeably.

A failure mode to name explicitly: if you crank a feature loss too high, the student becomes a *worse* classifier because it spends all its capacity reconstructing the teacher's activations and none of it fitting the actual decision boundary. The feature loss is a *regularizer / initializer*, not the objective. The objective is still accuracy. Keep the auxiliary terms in their lane.

Here is a concrete procedure for the gradient-norm-matching rule, since "make the gradients comparable" sounds vaguer than it is. At the start of training, run one batch, compute each loss term separately, call `.backward()` on each in isolation, and read off the resulting gradient norm of the *shared* student parameters (for example, the total $\ell_2$ norm of `student.parameters()` gradients). Suppose the CE term produces a gradient norm of $1.0$, the KD term $0.4$, and the unweighted attention-transfer term $0.002$. Then a sensible starting $\beta$ for the AT term is on the order of $1.0 / 0.002 = 500$ if you want it to match CE, or a few times smaller if you want it subordinate — the point is you are setting the weight from a *measurement*, not a guess, and you will be within a factor of a few of optimal instead of off by three orders of magnitude. Re-measure once mid-training if you used a warm-up, because the relative scales drift as the student learns. This five-minute calibration is the difference between feature distillation helping and feature distillation quietly dominating the loss and tanking accuracy.

#### Worked example: the cost of a bad layer pairing

Make the pairing rule concrete with numbers. Teacher: ResNet-50, whose four stages output feature maps of $256$, $512$, $1024$, and $2048$ channels at spatial sizes $56^2, 28^2, 14^2, 7^2$ for a $224^2$ input. Student: ResNet-18, whose four stages output $64$, $128$, $256$, $512$ channels at the *same* spatial sizes. The right pairing is stage-to-stage by index: teacher-stage-2 ($512$ ch, $28^2$) with student-stage-2 ($128$ ch, $28^2$). The spatial sizes already match, so a hint loss needs only a $128\to512$ channel projector and no interpolation. Now the wrong pairing: teacher-stage-3 ($1024$ ch, $14^2$) with student-stage-2 ($128$ ch, $28^2$). Not only is the channel projector bigger, the spatial sizes differ ($14^2$ vs $28^2$), forcing an interpolation that smears the target — and worse, you are asking the student's mid-level features to imitate the teacher's *higher-level* semantic features, a representational level the student's stage-2 cannot reach. In my experience the mispaired configuration trains to *below* the hard-label baseline: the destructive feature gradients actively fight the classification objective. The lesson in one number: match spatial resolution exactly and stage index closely, and a $128\to512$ projector with no interpolation is the well-behaved default for this teacher–student pair.

## 4. The practical flow: hooks, projectors, losses in PyTorch

Now the runnable part. Everything below targets a CNN distillation (the cleanest setting to learn in), with notes on transformers where they differ. The first thing you need, for both feature and relation distillation, is a way to grab intermediate activations during the forward pass without rewriting the model. That is what PyTorch forward hooks are for.

### 4.1 Registering hooks to capture intermediate features

A forward hook is a callback PyTorch runs after a given module produces its output. You attach one to each layer you want to tap, stash the output in a dictionary, and read it after the forward pass. Here is a small, reusable feature extractor that does not touch the model's code at all.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureCatcher:
    """Capture the output activations of named submodules via forward hooks."""
    def __init__(self, model, layer_names):
        self.model = model
        self.features = {}
        self._handles = []
        # Resolve dotted names like "layer2.1.conv2" to module objects.
        modules = dict(model.named_modules())
        for name in layer_names:
            module = modules[name]
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            # Store the activation tensor; keep the graph so gradients flow
            # through the student's captured features (do NOT detach student).
            self.features[name] = output
        return hook

    def clear(self):
        self.features = {}

    def remove(self):
        for h in self._handles:
            h.remove()

# Tap the end of each residual stage in a torchvision ResNet.
teacher_taps = ["layer2", "layer3"]
student_taps = ["layer2", "layer3"]
teacher_catcher = FeatureCatcher(teacher, teacher_taps)
student_catcher = FeatureCatcher(student, student_taps)
```

Two things to get right. First, **do not detach the student's captured features** — gradients must flow back through them into the student. *Do* run the teacher under `torch.no_grad()` (or detach its features) because the teacher is frozen and you never want to update it. Second, remember to `clear()` the caught features each step or you will silently hold references to old graphs and leak memory. With the catcher in place, a training step grabs `teacher_catcher.features["layer3"]` and `student_catcher.features["layer3"]` after calling the models, and feeds them to whichever loss you want.

Why hooks rather than just editing the model's `forward` to return its intermediates? Three reasons, and they are the reasons this is the idiomatic pattern. One, **separation of concerns**: the model code stays a clean, deployable artifact with the standard signature, and all the distillation scaffolding lives outside it, so you can distill a model you did not write (a torchvision or Hugging Face model) without forking it. Two, **the deployed student is untouched**: the projectors and the hooks exist only during training; the exported student has no trace of them, so there is zero inference overhead and no risk of shipping debug plumbing. Three, **flexibility**: you can re-point the taps by changing a list of names, which matters because choosing taps is an experiment you will run many times. The one gotcha to internalize is the lifecycle — register hooks once, `clear()` the feature dict every step, and call `remove()` when you are done (for example before exporting) so you do not leave dangling callbacks that capture graphs forever. If you see your memory climb step over step, a forgotten `clear()` is the first suspect.

A subtle correctness point that bites people: a forward hook fires when the *module* runs, so if a layer is called multiple times in one forward pass (weight sharing, a recurrent block), the dict holds only the *last* invocation's output. For the standard feed-forward CNN and transformer taps this never comes up, but if you tap a shared module, capture a list instead of overwriting. And if you use gradient checkpointing on the student, be aware that the hook fires during the *recomputation* pass, which is usually fine but occasionally surprises people debugging shapes.

### 4.2 A FitNet-style hint loss with a 1x1 projector

The projector bridges the channel mismatch. For CNNs it is a $1\times1$ convolution; if spatial sizes also differ, interpolate. Build one projector per tapped layer.

```python
class HintProjector(nn.Module):
    """1x1 conv mapping student channels -> teacher channels for a hint loss."""
    def __init__(self, c_student, c_teacher):
        super().__init__()
        self.proj = nn.Conv2d(c_student, c_teacher, kernel_size=1, bias=False)

    def forward(self, student_feat, teacher_feat):
        x = self.proj(student_feat)
        if x.shape[-2:] != teacher_feat.shape[-2:]:
            x = F.interpolate(x, size=teacher_feat.shape[-2:],
                              mode="bilinear", align_corners=False)
        return x

def hint_loss(student_feat, teacher_feat, projector):
    projected = projector(student_feat, teacher_feat.detach())
    return F.mse_loss(projected, teacher_feat.detach())

# One projector per tapped layer; its params join the optimizer.
projectors = nn.ModuleDict({
    "layer2": HintProjector(c_student=128, c_teacher=512),
    "layer3": HintProjector(c_student=256, c_teacher=1024),
})
optimizer = torch.optim.SGD(
    list(student.parameters()) + list(projectors.parameters()),
    lr=0.1, momentum=0.9, weight_decay=5e-4,
)
```

Note `teacher_feat.detach()` everywhere on the teacher side — the projector and student learn; the teacher does not. The projector's parameters go into the optimizer alongside the student's, and they get thrown away after training (they are not part of the deployed student). The channel counts here are for a ResNet-50 teacher (512 and 1024 channels at the ends of `layer2` and `layer3`) and a ResNet-18 student (128 and 256). Get those numbers from a quick `print(model)` or a dry-run forward — wrong channel counts are the single most common bring-up bug.

### 4.3 Attention-transfer loss

Attention transfer needs no projector. Collapse channels, flatten, normalize, subtract.

```python
def attention_map(feat, p=2):
    # feat: (N, C, H, W) -> (N, H*W) attention vector, sum |a|^p over channels.
    am = feat.pow(p).mean(dim=1)            # (N, H, W) ; mean or sum both work
    am = am.flatten(start_dim=1)            # (N, H*W)
    am = F.normalize(am, p=2, dim=1)        # match the shape, not magnitude
    return am

def attention_transfer_loss(student_feat, teacher_feat):
    # If spatial sizes differ, resize the student map to the teacher's.
    if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
        student_feat = F.interpolate(student_feat,
                                     size=teacher_feat.shape[-2:],
                                     mode="bilinear", align_corners=False)
    qs = attention_map(student_feat)
    qt = attention_map(teacher_feat.detach())
    return (qs - qt).pow(2).sum(dim=1).mean()   # per-sample L2^2, averaged
```

This works across *any* channel-count mismatch because the attention map is purely spatial. The only constraint is matching spatial resolution, handled by one interpolate. For transformers, the analog is matching the attention *probability matrices* (the softmaxed $QK^\top$ per head) between teacher and student heads — MiniLM and TinyBERT do versions of this — but the principle is identical: match where the model attends, not the raw values.

### 4.4 An RKD distance loss

Relation distillation operates on a *batch* of penultimate embeddings, not on spatial maps. Tap the layer just before the classifier (the global-average-pooled embedding for a CNN), then compute pairwise distances.

```python
def pdist(embeddings, squared=False, eps=1e-12):
    # Pairwise Euclidean distances within a batch: (N, D) -> (N, N).
    prod = embeddings @ embeddings.t()
    sq = prod.diag().unsqueeze(1) + prod.diag().unsqueeze(0) - 2 * prod
    sq = sq.clamp(min=0)
    dist = sq if squared else (sq + eps).sqrt()
    dist = dist.clone()
    dist[range(len(dist)), range(len(dist))] = 0  # zero the diagonal
    return dist

def rkd_distance_loss(student_emb, teacher_emb):
    with torch.no_grad():
        td = pdist(teacher_emb)
        mean_td = td[td > 0].mean()
        td = td / (mean_td + 1e-12)          # mean-normalize teacher distances
    sd = pdist(student_emb)
    mean_sd = sd[sd > 0].mean()
    sd = sd / (mean_sd + 1e-12)              # mean-normalize student distances
    return F.smooth_l1_loss(sd, td)          # Huber loss on normalized distances
```

The mean-normalization is the load-bearing trick: it makes the loss invariant to the overall scale of each feature space, so the student is free to use a smaller or larger embedding magnitude as long as the *relative* distances match. The angle term follows the same pattern over triplets (compute unit vectors between embeddings and match cosines); I am omitting it for length, but it is a dozen more lines and the RKD repo has a reference implementation.

One performance note on RKD: the distance term is $O(B^2)$ in the batch size $B$ and the angle term is $O(B^3)$, so for a batch of 256 the angle term alone touches ~16.7 million triplets. Materializing all triplets is wasteful; the standard trick is to sample a fixed number of triplets per batch (or compute the angle tensor with broadcasting and a memory cap). It rarely matters for vision batches of 64–256, but if you scale the batch up for throughput, the angle term's cubic cost will quietly become your bottleneck — profile it before you blame the model.

### 4.5 Feature and relation distillation for transformers

The CNN recipe above transfers almost verbatim to transformers, with two substitutions worth naming because this is where most edge-LLM and edge-NLP work happens now. First, the **feature tap** is a hidden state $h \in \mathbb{R}^{L \times d}$ (sequence length $L$, model dimension $d$) instead of a spatial map, and the projector is a linear layer $d_{\text{student}} \to d_{\text{teacher}}$ instead of a $1\times1$ conv. A cosine-embedding loss on the hidden states — $1 - \cos(h^{(s)} W, h^{(t)})$ averaged over tokens — is the usual choice and is exactly what DistilBERT used. Second, the **attention tap** is the attention probability matrix $\mathrm{softmax}(QK^\top/\sqrt{d_k}) \in \mathbb{R}^{L \times L}$ per head, and the loss is a KL or MSE between teacher and student attention distributions. TinyBERT matches both the attention matrices and the hidden states (with a linear projector); MiniLM cleverly matches the *self-attention relations* — the $QK^\top$ and $VV^\top$ distributions — which sidesteps the head-count and dimension-matching problems entirely because it compares relations within each network rather than raw values across them. That MiniLM move is, notice, a *relational* idea applied inside attention: match the structure, not the values, so the student need not have the same number of heads or the same head dimension. The principle is identical to RKD; only the object being related has changed from "examples in a batch" to "positions in a sequence." Everything in this post about *what* to distill applies to transformers; only the shape of the tapped tensor changes.

### 4.5 Putting it together in one training step

Here is the whole multi-term step, the thing you actually run. It anchors on hard labels, adds response KD, and adds whichever feature/relation terms you chose, each with its weight.

```python
def distill_step(images, labels, T=4.0, alpha=0.7, beta=50.0, gamma=25.0):
    teacher.eval()
    with torch.no_grad():
        t_logits = teacher(images)
        t_feat = {k: v.detach() for k, v in teacher_catcher.features.items()}
        t_emb = teacher_emb_catcher.features["avgpool"].flatten(1).detach()
    teacher_catcher.clear(); teacher_emb_catcher.clear()

    s_logits = student(images)
    s_feat = student_catcher.features
    s_emb = student_emb_catcher.features["avgpool"].flatten(1)

    # 1) hard-label anchor
    loss_ce = F.cross_entropy(s_logits, labels)
    # 2) response KD (temperature-scaled KL, x T^2)
    loss_kd = (T * T) * F.kl_div(
        F.log_softmax(s_logits / T, dim=1),
        F.softmax(t_logits / T, dim=1),
        reduction="batchmean",
    )
    # 3) feature term: attention transfer at each tap (no projector needed)
    loss_feat = sum(attention_transfer_loss(s_feat[k], t_feat[k])
                    for k in student_taps) / len(student_taps)
    # 4) relation term: RKD distance on penultimate embeddings
    loss_rel = rkd_distance_loss(s_emb, t_emb)

    loss = (1 - alpha) * loss_ce + alpha * loss_kd + beta * loss_feat + gamma * loss_rel
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    student_catcher.clear(); student_emb_catcher.clear()
    return {"ce": loss_ce.item(), "kd": loss_kd.item(),
            "feat": loss_feat.item(), "rel": loss_rel.item()}
```

This is the practical heart of the post. Note the shape of it: one frozen teacher forward under `no_grad`, one student forward, four losses, one backward through the student (and the projectors, if you used hints). The teacher's features are detached; the student's are not. The weights $\alpha, \beta, \gamma$ are what you tune. If you swap attention transfer for FitNet hints, the only change is calling `hint_loss(..., projectors[k])` instead of `attention_transfer_loss`. Figure 7 (later) shows this combined loss as a stack of contributing terms.

## 5. Worked examples with real numbers

Numbers make the families concrete. Both examples below use figures that are representative of the published distillation literature (the survey by Gou et al. 2021 and the original papers) and of results I have reproduced; where a number is a literature figure I name the rough source, and where it is approximate I say so. The point is the *shape* of the gains, not three-decimal precision.

#### Worked example: closing a wide gap with feature distillation

The setup. Teacher: ResNet-50 on ImageNet, about 76% top-1. Student: ResNet-18, which trains to about 70% top-1 on hard labels alone. This is a meaningful gap — six points — and a meaningful capacity ratio (ResNet-50 has roughly 25.6M parameters and 4.1 GFLOPs at 224x224; ResNet-18 has roughly 11.7M parameters and 1.8 GFLOPs). The question: how much of the gap can each kind of knowledge transfer recover?

| Method | What is transferred | ResNet-18 top-1 | Gain over hard labels |
| --- | --- | --- | --- |
| Hard labels only | nothing (baseline) | ~70.0% | — |
| Response KD (Hinton) | soft logits | ~70.7% | +0.7 |
| + Attention transfer | soft logits + attention maps | ~71.3% | +1.3 |
| + RKD distance/angle | + embedding geometry | ~71.6% | +1.6 |
| CRD (contrastive) | mutual-info maximization | ~71.4% | +1.4 |

Read the table as a story. Response KD alone recovers under a point — that is the saturation I opened the post with. Adding attention transfer roughly doubles the gain, because now the student gets spatial-attention targets deep in the network, not just at the output. Layering RKD on top adds a bit more, because the embedding geometry is a complementary signal to the per-location attention. CRD alone (no hand-tuned feature taps) lands in the same neighborhood as the stacked approach, which is why people who do not want to fuss with layer pairing reach for it. The headline: across a six-point gap, the *feature and relation* terms contribute roughly twice the recovery that logits alone do. On a small gap (say ResNet-34 teacher, ResNet-18 student, two points apart) the picture inverts — response KD captures most of the available gain and the extra terms add little, which is exactly the decision rule we will codify later.

Now the edge angle, because this is an edge-AI series. That ResNet-18 student at ~71.6% is a 1.8-GFLOP model. On a Jetson Orin Nano it runs comfortably at batch 1 in the low single-digit milliseconds in fp16. If you then **quantize it to int8** (PTQ, see [post-training quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq)), you lose perhaps 0.3–0.6 points and gain ~2x throughput on the int8 tensor cores. The distillation gain you bought (+1.6) more than pays for the quantization loss (−0.5), so the *net* is a smaller, faster model that is still better than the hard-label baseline. That stacking — distill to recover accuracy, then quantize to recover speed — is the whole game, and the two levers do not fight because they charge different currencies.

Put the full Pareto story on one line so the trade is legible. Start: ResNet-50 teacher, ~76% top-1, ~98 MB in fp32, ~4.1 GFLOPs, far too heavy for a phone budget. Naive small model: ResNet-18 on hard labels, ~70%, ~45 MB fp32. Distilled: ResNet-18 with KD + AT + RKD, ~71.6%, same 45 MB. Distilled then int8-quantized: ~71.1%, ~12 MB, ~2x faster on int8 hardware. The deployed artifact is roughly **8x smaller than the teacher and within ~5 points of it**, where the naive baseline would have been ~6 points behind — distillation bought back a quarter of the gap and quantization paid for the size and speed. None of those steps was free, but each charged a *different* budget (a training run, then hardware portability), which is the entire reason the wins compounded rather than competed. If you remember one number from this post, make it that ratio: across a wide gap, the interior signals roughly *doubled* the accuracy you could buy with logits alone, and that doubling is what made the quantized student viable instead of disappointing.

#### Worked example: attention transfer on a CNN, the spatial-knowledge lift

The setup. A smaller, cleaner experiment that isolates attention transfer's contribution, in the spirit of the original Zagoruyko–Komodakis results on CIFAR and ImageNet. Teacher: a wide ResNet (WRN-16-2 style) on CIFAR-100. Student: a thinner/shallower network with a clear gap to the teacher. Hard-label student: about 72% top-1 on CIFAR-100. We add attention transfer at the end of each of the three residual groups.

| Configuration | Top-1 (CIFAR-100) | Notes |
| --- | --- | --- |
| Student, hard labels | ~72.0% | baseline |
| Student + KD (logits) | ~73.3% | response only |
| Student + AT (attention only) | ~73.8% | feature only, no logits |
| Student + KD + AT | ~74.6% | combined |

The instructive line is the third one: **attention transfer alone, with no logit matching at all, beats logit-only KD.** That is the clearest evidence that feature knowledge is not a minor add-on — for a CNN with a real capacity gap, *where the network looks* is at least as transferable as *what it concludes*. And the two are complementary: combining them (last row) beats either alone, because attention shapes the internal representation while logits shape the decision boundary. This is the experiment I point skeptics to when they ask whether feature distillation is worth the hooks. For a vision model across a gap, it plainly is.

A stress test on this example: what happens when the spatial resolutions of teacher and student maps differ a lot (say the student downsamples earlier)? The interpolation in the AT loss handles it, but if the mismatch is severe the matched attention becomes blurry and the gain shrinks. The fix is to tap at *resolution-matched* stage boundaries, which is the layer-pairing rule from section 3 doing its job. And what if your batch is tiny (edge fine-tuning on-device with batch 4)? Attention transfer is per-sample and survives small batches fine; *RKD* does not — its distance/angle statistics degrade badly below ~16 examples per batch because there are too few pairs and triplets, so on tiny-batch regimes prefer feature methods over relational ones.

A second stress test on the same example, this time on the *teacher* quality, because it changes the recipe. It is tempting to assume a bigger teacher is always a better teacher, but distillation has a well-documented "capacity gap" pathology: a teacher that is *too* much stronger than the student can transfer *worse* than a moderately strong one, because its sharp, confident soft targets and its complex internal features are harder for the small student to imitate. If your WRN teacher were enormous and the student tiny, plain logit KD can actually *underperform* hard-label training — a result that surprises people every time. The mitigations are exactly the interior signals in this post (feature and relation losses give the student a learnable path the over-sharp logits do not), plus tricks like a higher temperature to soften the targets, or an intermediate "teaching assistant" model of middle size that distills in two hops. So the honest reading of the second worked example is not "attention transfer always wins" but "when the gap is wide enough that logits alone falter, interior signals are what keep distillation from going *negative*." That is a stronger and more useful claim.

## 6. Results: a method-family comparison

Pulling the families side by side, here is the comparison table I keep in my head, generalized from the worked examples and the survey literature. This is figure 6 in diagram form; the table gives the detail.

| Family | What it matches | Where it taps | Extra plumbing | Cost at train | When it wins |
| --- | --- | --- | --- | --- | --- |
| Response (Hinton KD) | softened logits | output only | none | negligible | small gap, same architecture family |
| Feature — hints (FitNets) | raw activation maps | one+ hidden layers | projector ($1\times1$ conv) per tap | extra forward storage + projector params | wide gap; good student initialization |
| Feature — attention (AT) | spatial attention maps | end of residual groups | none (no projector) | one sum-of-squares per tap | CNNs, wide gap, best effort/payoff |
| Relation — RKD | pairwise distance + triplet angle | penultimate embedding | none, but needs a decent batch | $O(B^2)$ pairs, $O(B^3)$ angles per batch | metric / retrieval tasks; cross-architecture |
| Relation — CRD | mutual info (contrastive) | penultimate embedding | memory bank of embeddings | bank memory + negatives per step | last-point squeeze; cross-architecture |

![A three-row by three-column matrix comparing response, feature, and relation distillation across what each matches, its engineering cost, and the situation where it wins](/imgs/blogs/what-to-distill-response-feature-relation-6.png)

A few honest observations from that table. First, the *engineering cost* axis is real and underdiscussed: response KD is one line, while feature and relation methods require you to instrument the model with hooks, choose layer pairs, and tune extra weights. That cost is a tax you pay every time you change the model architecture, and it is the main reason teams default to logit KD even when feature KD would help. Second, the methods that win on *cross-architecture* transfer (RKD, CRD) are the relational ones — precisely because they do not assume the two networks compute the same intermediate things, only that they organize data the same way. If your student is a different family from your teacher (a MobileNet student from a ResNet teacher, or a small transformer from a CNN), reach for relation over feature. Third, none of these is strictly dominant; the right answer is a function of your gap, your task, and your tolerance for plumbing — which is the decision tree in section 8.

## 7. Combining signals: the multi-term loss and a practical recipe

In production you rarely use one family alone. The shipped objective is a weighted sum, and getting the weights and schedule right is most of the skill. Figure 7 shows the combined loss as a stack of contributing terms feeding one backward pass.

![A vertical stack diagram showing hard cross-entropy, response KD, a feature term, and a relation term summing into a total weighted loss that drives a single backprop step on the student](/imgs/blogs/what-to-distill-response-feature-relation-7.png)

Here is the recipe I actually follow, in order, when a plain logit baseline has come up short.

1. **Always keep the hard-label cross-entropy term.** It is the ground truth anchor. Distillation losses can drift the student toward the teacher's *mistakes*; the CE term keeps it honest. Weight it with $(1-\alpha)$ and never let it go to zero.

2. **Add response KD next.** It is free and it is the floor. Set $\alpha \approx 0.5\text{–}0.9$ and $T \approx 3\text{–}5$. Measure. If this hits your target, *stop here* — you do not need the rest. This is the single most important discipline in the whole post: the extra families are insurance you buy only when the cheap thing fails.

3. **If the gap is wide and the model is a CNN, add attention transfer.** Two or three taps at stage boundaries, weight $\beta$ tuned so its initial gradient is comparable to the others. This is the highest-payoff addition for vision, and it needs no projector.

4. **If you need raw-feature mimicry (very wide gap, or you want a strong initialization), use FitNet hints instead of or alongside AT,** with the two-stage schedule: hint-train the lower student first, then full distillation. Accept the projector cost.

5. **If the task is metric/retrieval, or the student is a different architecture from the teacher, add a relation term (RKD or CRD).** RKD if you want simplicity and have a reasonable batch size; CRD if you are squeezing the last fraction of a point and can afford the memory bank.

6. **Tune the weights by gradient-norm matching, not by guessing,** and consider a warm-up that ramps the response term in after a few epochs of feature/relation seeding.

On **composing with the other levers**, the order is settled and worth stating as a rule: **distill first, then quantize and prune the student.** Distillation produces a small *dense* model; quantization and pruning then shrink *that* model further along the bit-width and sparsity axes. Doing it in this order means each lever operates on the output of the last, and because they charge different currencies (distillation charges a training run + a teacher; quantization charges hardware portability; pruning charges another fine-tune) they stack instead of cancel. A neat bonus: you can use distillation *during* quantization-aware training — the full-precision teacher supervises the quantized student, which recovers a chunk of the QAT accuracy loss for free. That is a standard trick in [quantization-aware training](/blog/machine-learning/edge-ai/quantization-aware-training-qat), and it is just response (or feature) distillation with the student being the fake-quantized network. For the end-to-end stacking story and the order all four levers go in, the capstone is [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).

There is one composition subtlety that catches people, so let me be explicit. When you distill *into* a quantized student (distillation-aware QAT), the teacher's feature maps are full precision and the student's are fake-quantized, which means a raw $\ell_2$ hint loss is now penalizing the student for *quantization noise* on top of any representational difference — it is asking an int8 feature to exactly reconstruct an fp32 one, which is impossible at the bit budget. The fix is to prefer the *invariant* feature losses here: attention transfer and relation losses, which compare normalized summaries, are far more tolerant of the student's quantization noise than a raw-value hint loss is. So the rule of thumb compounds: across a wide gap *and* into a low-bit student, the relational and attention signals are doubly favored — they are the ones whose targets the quantized student can actually hit. This is a good example of how the "increasing invariance" lens from section 2 pays off operationally: the more aggressively you compress the student (lower bits, more pruning), the more you want the *coarser, more invariant* distillation signal, because the student has less ability to match anything exact.

And the reverse composition — pruning then distilling — has its own niche: if you prune a large model structurally to get a medium one, you can then use the *original unpruned* model as the teacher to fine-tune the pruned student, which is just distillation with a free, perfectly-matched teacher (the pruned model started as a copy of it). This "prune, then distill from the pre-prune checkpoint" loop recovers much of the accuracy that the structured prune cost, and it is one of the most reliable ways to make structured pruning actually hit its accuracy target. See [structured pruning that actually speeds things up](/blog/machine-learning/edge-ai/structured-pruning-that-actually-speeds-things-up) for the pruning side of that loop.

#### A note on measuring this honestly

Distillation gains are small (often under two points) and noisy, so measure them like you mean it. Run each configuration with at least three seeds and report the mean and spread — a single run can show a "+0.4" that is pure variance. Hold the student's training budget *fixed* across configurations (same epochs, same augmentation, same LR schedule); it is easy to fool yourself by giving the distilled run more epochs. And evaluate the *deployed* artifact: if you are going to quantize the student, report the accuracy of the quantized student, because the distillation gain and the quantization loss interact and the only number that matters is the one the device produces. For latency, the usual edge discipline applies — warm up, pin the clocks if you can, measure batch-1 p50 and p99 on the actual target, and watch for thermal throttling on a phone that has been running for a minute.

There is a specific apples-to-apples trap in distillation papers worth calling out so you do not fall into it yourself: comparing a distilled student against a *separately trained* student that got a *different* recipe. The fair baseline for "did feature distillation help" is the *same student, same epochs, same augmentation, with and without the feature term* — not the student from some other paper's table. The literature is littered with "gains" that are really just the distilled run having had a longer schedule or stronger augmentation than the baseline it is compared to. When you read a distillation result, the first question is always "what exactly was the control," and when you *produce* one, make the control the same student trained identically minus the one term you are evaluating. Distillation is one of those areas where the headline numbers are real but the measurement hygiene is frequently sloppy, and the only defense is to control your own experiments tightly. A related discipline: report the *teacher's* accuracy and the *gap* alongside the student's gain, because a "+1.6" across a six-point gap and a "+1.6" across a two-point gap are completely different results — the former is recovering a quarter of the gap, the latter is closing most of it, and conflating them is how people draw the wrong conclusions about which method to use.

## 8. Case studies and named results

Three results from the literature that ground the families in real, citable numbers.

**FitNets (Romero et al., 2015) — feature hints make a thin-but-deep student trainable.** The original FitNets result was not just "a bit more accuracy"; it was that a student *thinner and deeper* than its teacher — a hard topology to train — became trainable *because* of the hint-based pre-training, and ended up matching or beating the teacher on CIFAR-10/100 and SVHN while using far fewer parameters. The mechanism is exactly the one in section 2.2: the hint seeds the student's lower layers with a good representation, turning an optimization problem the student could not solve from scratch into one it can. This is the canonical demonstration that *where* you supervise (inside, not just at the output) changes what is even *learnable*.

**Attention transfer (Zagoruyko & Komodakis, 2017) — spatial attention is transferable knowledge.** On CIFAR and ImageNet they showed consistent gains from matching attention maps, with the striking result (mirrored in our second worked example) that attention transfer alone is competitive with logit KD, and the two combine. Their framing — that the *gradient* of the network's output with respect to the input, and the activation-based attention map, both encode "where the network looks" and both transfer — is one of the cleaner conceptual contributions in the feature-distillation line. For CNNs it remains a default I reach for.

**CRD (Tian et al., 2020) — distillation as mutual-information maximization.** On the standard CIFAR-100 and ImageNet distillation benchmarks, CRD outperformed response KD and the earlier feature methods across a wide range of teacher–student pairs, *including cross-architecture pairs* where the teacher and student are different families. The contribution is partly empirical (it tops the tables) and partly conceptual: by recasting distillation as maximizing $I(\text{teacher repr}; \text{student repr})$, it gives a principled reason why feature/relation knowledge beats logits — logit matching is a loose lower bound on that mutual information, and the contrastive objective is a tighter one. If you want one paper that explains *why* the answer to "what to distill" is "more than the logits," this is it.

A fourth, just to span language models: **DistilBERT (Sanh et al., 2019)** combined a response-style distillation loss (soft logits via a temperature) with a *cosine embedding loss* aligning the student's and teacher's hidden states — a feature term — plus the masked-LM loss, and got a model 40% smaller and 60% faster at 97% of BERT's GLUE performance. That cosine-embedding term is feature distillation in a transformer, and its presence (alongside the logit term) is a real-world instance of the multi-term recipe from section 7. We dig into that and its descendants in [distillation case studies, from DistilBERT to CNNs](/blog/machine-learning/edge-ai/distillation-case-studies-distilbert-to-cnns).

A fifth, to close the loop on the relational-attention idea: **TinyBERT (Jiao et al., 2020)** and **MiniLM (Wang et al., 2020)** both went further into the interior of the transformer than DistilBERT did. TinyBERT distills the embedding layer, the hidden states (with a learned linear projector to bridge the dimension gap), *and* the attention matrices, in a two-stage schedule (general distillation on a large corpus, then task-specific distillation) — a textbook stacking of response, feature, and attention signals that let a 4-layer student retain a large fraction of BERT-base's accuracy at a fraction of the size. MiniLM, as noted earlier, distills the *self-attention relations* (the scaled $QK^\top$ and the value–value $VV^\top$ distributions of the last layer), which is the relational idea applied inside attention; because it matches relations rather than raw values, the student need not share the teacher's number of heads or its head dimension, which is exactly the architecture-agnostic property that makes relational signals shine across a gap. Together these three (DistilBERT, TinyBERT, MiniLM) are the canonical demonstration that the response/feature/relation taxonomy is not a vision-only story — it is the organizing frame for compressing transformers onto edge budgets too.

## 9. When to reach for this (and when not to)

A decision section, because every term you add is a cost and most teams should not be reaching for the fancy stuff by default. Figure 8 is the decision tree; the prose is the reasoning behind it.

![A decision tree starting from whether logit KD hits the target, branching to ship it if yes, and if not splitting by CNN vision tasks toward feature hints and attention transfer or metric tasks toward relational RKD and CRD](/imgs/blogs/what-to-distill-response-feature-relation-8.png)

**Start with response KD, always.** It is one line of code, it has no hyperparameter beyond $T$ and $\alpha$, and on a small-to-moderate capacity gap it captures most of the gain available. If it hits your accuracy target, you are done — adding feature or relation losses to a model that is already meeting spec just buys you complexity, more hyperparameters, and a more fragile training recipe, for a fraction of a point you do not need.

**Reach for feature distillation when the gap is wide and the model is a CNN.** "Wide" is fuzzy, but a useful threshold: if the student loses more than ~2–3 points to the teacher *and* logit KD recovers less than half of that, the student is failing to form a good internal representation, which is exactly what feature distillation fixes. Prefer attention transfer (no projector, cheap, robust) over FitNet hints unless you specifically need raw-feature mimicry or a staged initialization for a thin-deep student.

**Reach for relation distillation when the task is metric/retrieval or the student is a different architecture from the teacher.** Relational methods transfer the *geometry* of the embedding space, which is the right target for tasks where you care about distances (face recognition, retrieval, contrastive embeddings) and the only target that makes sense when the two networks compute fundamentally different intermediate things. Use RKD for simplicity and a normal batch size; reserve CRD for the last-point squeeze.

**Do not reach for any of this if you do not have a teacher, a training budget, and a held-out set you trust.** Distillation is a full training run with an extra forward pass through a large teacher every step — it is not cheap, and on a small gap it may not beat just training the student well with good augmentation. And do not reach for it as a substitute for the cheaper levers when those would do: if your problem is "the model is too big on disk," quantization is a faster fix than distilling a new architecture. Distillation earns its keep specifically when you need a *smaller, dense* model that retains the teacher's accuracy and the cheaper levers cannot get you there — which, on a wide gap, they often cannot.

**Stress test — the on-device fine-tune.** Suppose you must do the distillation *on the edge device itself* (privacy, no cloud), with a batch size of 4 and a tiny held-out set. What survives? Response KD survives (it is per-sample and cheap). Attention transfer survives (per-sample). RKD degrades (its batch statistics need ~16+ examples to be meaningful) and CRD is essentially out (the memory bank and contrastive negatives assume a healthy batch and a lot of data). So the edge-constrained answer collapses to response + attention transfer — which is a nice illustration that the "best" method on a leaderboard is not always the deployable one. Always ask what your *real* training conditions are before copying a benchmark recipe.

**Stress test — no labels at all.** A different real constraint: you have a teacher and a pile of *unlabeled* data, but no ground-truth labels for the student's domain. Now the hard-label CE anchor is gone, which is dangerous because that anchor was what kept the student from inheriting the teacher's mistakes. The good news is that response, feature, and relation losses all work *without labels* — they only need the teacher's outputs and activations, which the teacher produces on any input. This is the "transfer set" regime from the original Hinton paper: distill on unlabeled data using only the teacher's signals. The catch is that the student is now strictly bounded by the teacher (it can never exceed a teacher it has no independent label signal to correct), and any teacher bias transfers wholesale. In practice, label a small held-out slice for the CE anchor if you possibly can, even 5% of the data, because that thin thread of ground truth is what stops the student from confidently reproducing the teacher's errors. When you genuinely cannot, relation and attention losses on the unlabeled transfer set are still a real and often-overlooked option — far better than deploying the teacher untouched.

## 10. Key takeaways

- **Logits are the thinnest signal.** Response distillation matches only the output distribution and saturates across a wide capacity gap, because it gives the student no guidance about the internal representation it must build.
- **Feature distillation hands the student the teacher's internal structure.** FitNet hints match raw activation maps through a learned $1\times1$ projector; attention transfer matches the much smaller spatial attention map and needs no projector. For CNNs across a gap, attention transfer has the best effort-to-payoff ratio.
- **Relation distillation transfers geometry, not points.** RKD matches pairwise distances and triplet angles; CRD maximizes mutual information via a contrastive loss. These tolerate wide gaps and cross-architecture pairs because they never pin any example to a particular location.
- **Across a wide gap, feature + relation terms can contribute roughly twice the accuracy recovery that logits do** — and attention transfer alone can beat logit-only KD on a CNN.
- **Layer pairing and weighting are the make-or-break decisions.** Pair teacher/student layers at matched representation levels (stage boundaries), use only two or three taps, and tune auxiliary weights by gradient-norm matching, not by guessing.
- **Hooks are how you reach inside.** Forward hooks capture intermediate activations without editing the model; detach the teacher, keep the student attached, clear every step.
- **Start cheap, escalate only on failure.** Always run response KD first; add feature terms only if the gap is wide and the model is a CNN; add relation terms only for metric tasks or cross-architecture students.
- **Compose with quantization and pruning by distilling first.** Distillation yields a small dense student; quantize and prune *that*. The levers charge different currencies, so they stack instead of cancel — and distillation can even supervise QAT directly.
- **Measure honestly.** Small, noisy gains demand multiple seeds, a fixed training budget across configs, and evaluation of the *deployed* (quantized) artifact, not the float one.

## 11. Further reading

- **Romero et al., "FitNets: Hints for Thin Deep Nets" (ICLR 2015)** — the origin of feature-based distillation and the hint/regressor mechanism.
- **Zagoruyko & Komodakis, "Paying More Attention to Attention" (ICLR 2017)** — attention transfer; activation-based and gradient-based attention maps as transferable knowledge.
- **Park et al., "Relational Knowledge Distillation" (CVPR 2019)** — the RKD distance and angle losses; matching the geometry of the embedding space.
- **Tian, Krishnan & Isola, "Contrastive Representation Distillation" (CRD, ICLR 2020)** — distillation as mutual-information maximization, the strongest relational method and the clearest theory for why features beat logits.
- **Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural Network" (NeurIPS DL workshop 2015)** — the response-based baseline; temperature and dark knowledge.
- **Gou, Yu, Maybank & Tao, "Knowledge Distillation: A Survey" (IJCV 2021)** — the response/feature/relation taxonomy this post follows, with a comprehensive method map.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression), [knowledge distillation fundamentals](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals), [distillation case studies, from DistilBERT to CNNs](/blog/machine-learning/edge-ai/distillation-case-studies-distilbert-to-cnns), and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).

---
title: "Distillation and Compression for Recommenders"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Take a big, accurate ranker that is too slow to serve, distill its ranking knowledge into a small student with a combined hard plus soft loss in PyTorch, quantize the student's embeddings to int8, and measure NDCG@10 against latency p99 and model size so the student recovers most of the teacher's quality at a fraction of the cost."
tags:
  [
    "recommendation-systems",
    "recsys",
    "knowledge-distillation",
    "ranking-distillation",
    "model-compression",
    "quantization",
    "privileged-features",
    "machine-learning",
    "pytorch",
    "serving",
    "movielens",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/distillation-and-compression-for-recsys-1.png"
---

The ranker was beautiful and we could not ship it. We had spent a quarter building a multi-task ranker — a shared-bottom DCN feeding three heads for click, like, and dwell, with cross features, sequence features, and a transformer over the last fifty items. Offline, it was the best thing we had ever trained: NDCG@10 up four points over the production model, calibration tight, the seesaw between objectives finally tamed. Then we measured the latency. Per request we had to score roughly five hundred candidates, and the ranker took about 84 milliseconds at p99 to score them on the host we could afford. The serving budget for the ranking stage was 10 milliseconds. We were off by nearly an order of magnitude, and at our query rate — tens of thousands of requests per second, multiplied by hundreds of candidates each — closing that gap by adding hardware would have cost more than the entire feature team's salary. The model that won offline could not survive contact with the per-request budget.

That gap is the subject of this entire post, and it is one of the most common and most solvable problems in production recommendation. A recommender lives inside the retrieval → ranking → re-ranking funnel, and the ranking stage is where the cost concentrates: hundreds of candidates per request, every request, at machine-killing query rates. You can build a model that is more accurate than anything you can afford to serve — a giant multi-task ranker, an ensemble, or these days an LLM that reasons about each item in natural language. **Knowledge distillation** is the technique that lets you keep the quality and pay the cheap bill. You train the expensive model — the **teacher** — once, offline, where latency does not matter. Then you train a small, fast model — the **student** — to imitate the teacher, and you serve the student. Done well, the student recovers most of the teacher's quality at a fraction of the cost, and the figure below is the shape of the whole idea.

![A branching dataflow from a large teacher ranker into soft scores and top-K targets that train a small student which serves cheaply](/imgs/blogs/distillation-and-compression-for-recsys-1.png)

This is not the same as just training a small model from scratch. A from-scratch student sees only the hard 0/1 labels — clicked or not — which are sparse, noisy, and missing-not-at-random. The teacher's *soft* outputs carry far more information: not just *that* item A was clicked, but that the teacher considered A, F, and K all plausible and ranked them in that order, and thought everything else was hopeless. That richer signal — sometimes called the **dark knowledge** in the teacher's probabilities — is what a small student cannot extract from the raw labels alone but can absorb from the teacher. By the end of this post you will be able to: pick the right distillation flavor for a ranking problem; implement ranking distillation in PyTorch with a combined hard-plus-soft loss and a real temperature; quantize the student's embeddings to int8; and read a results table that puts NDCG@10 next to latency p99 and model size so you can justify the trade to a skeptical staff engineer. We will keep tying it back to the funnel and to the offline↔online reality that decides whether any of this actually ships.

## 1. The serving-cost problem, stated precisely

Before we reach for a tool, let us be precise about the constraint, because "the model is too slow" is not yet an engineering problem — it is a complaint. The engineering problem is a budget, and a budget has numbers.

A ranking request does not score one item. It scores the candidate set that retrieval handed up — call it $C$ candidates, typically a few hundred to a couple of thousand. So the per-request ranking work is $C$ forward passes through the ranker (batched, but still $C$ rows of compute). If the system handles $Q$ queries per second, the ranker is doing $C \times Q$ scoring operations per second, continuously. With $C = 500$ and $Q = 20{,}000$, that is **ten million item-scorings per second**. Every microsecond you shave off a single item-scoring is ten seconds of compute per second saved across the fleet — which is why a small constant factor in the model is the entire ballgame.

The latency budget is even less forgiving than the throughput budget. The end-to-end recommendation request has a total budget — say 150 milliseconds from the user's tap to bytes on the wire — and that budget is *carved up* among retrieval, ranking, re-ranking, feature fetch, and network. Ranking might get 20 milliseconds of it, and we measure at **p99**, not the mean, because the user who waits is the user at the tail. A model that averages 12 milliseconds but spikes to 84 at p99 *fails the budget*, because one request in a hundred blows past it, and at scale one in a hundred is a lot of unhappy users and a lot of timeouts that downstream services log as errors.

So the real statement of the problem is: **drive the p99 latency of scoring $C$ candidates below the ranking stage's slice of the budget, at the query rate the fleet must sustain, without giving back more ranking quality than the product can tolerate.** A big multi-task ranker fails the first two clauses. An LLM recommender fails them catastrophically — a 7-billion-parameter model that needs to read each candidate as text might take *seconds* per request, four or five orders of magnitude over budget. The accuracy is real; the cost is fatal. Distillation is the bridge: train the expensive thing once, serve a cheap thing that learned from it.

#### Worked example: the cost of a 10× larger ranker

Put numbers on it. Suppose your fleet must sustain $Q = 20{,}000$ QPS, each request scores $C = 500$ candidates, and a single host scores **40,000 candidates per second** with the small model. Total scoring demand is $C \times Q = 10{,}000{,}000$ item-scorings/sec, so you need $10{,}000{,}000 / 40{,}000 = 250$ hosts for the small model. Now swap in a teacher that is 10× more expensive per scoring — it scores 4,000 candidates/sec/host. You need $10{,}000{,}000 / 4{,}000 = 2{,}500$ hosts. At, say, \$1.50 per host-hour that is the difference between $250 \times 1.5 \times 24 \times 30 = \$270{,}000$ and $\$2{,}700{,}000$ per month — a \$2.43M/month gap, before you even ask whether 2,500 hosts can hit the p99 budget at all (they often cannot, because the per-item latency, not just throughput, exceeds the slice). A distilled student that matches the small model's cost while recovering most of the teacher's quality is, very literally, a multi-million-dollar artifact. That is why this technique earns a whole post.

### Where the ranker's cost actually hides

It is worth knowing *which part* of a big ranker is expensive, because it tells you what the student must shed. Three components dominate. First, the **dense network depth and width** — a 200M-parameter teacher with deep cross layers and a transformer over the user's history does far more arithmetic per item than a shallow MLP, and that arithmetic is paid $C$ times per request. The student sheds this directly by being smaller and shallower; this is the part distillation compresses. Second, the **feature fetch and feature crossing** — a heavy ranker pulls many features and computes expensive cross-features per candidate, and some of those features are themselves the output of other models. The student sheds this by using a leaner feature set, which is exactly where privileged-features distillation earns its keep: the student drops the expensive and post-event features but keeps their *effect* in its weights. Third, the **embedding gather** — reading the right rows out of a multi-gigabyte table is memory-bandwidth-bound, and at $C \times Q$ lookups per second it is a real fraction of latency; this is the part embedding compression shrinks. A production student attacks all three at once: a smaller network (distillation), a leaner feature set (PFD), and compressed tables (int8). The rest of the post is those three levers, in that order.

## 2. Knowledge distillation, from the ground up

The original idea (Hinton, Vinyals & Dean, 2015, *Distilling the Knowledge in a Neural Network*) is simple and worth deriving, because the ranking version is a twist on it and you should understand the base case first.

A classifier produces logits $z_1, \dots, z_n$ that a softmax turns into probabilities. The standard softmax is $p_i = e^{z_i} / \sum_j e^{z_j}$. The trained teacher's probabilities carry more than the argmax: in an image classifier, a photo of a cat might get probability 0.9 cat, 0.08 dog, 0.02 car. The relative weight on "dog" versus "car" — the fact that the teacher thinks a cat is far more dog-like than car-like — is information about the structure of the problem that the hard label "cat" throws away. Hinton called this the dark knowledge, and the whole trick of distillation is to make the student match the teacher's full distribution, not just its top class.

The catch is that a confident teacher's distribution is nearly one-hot — 0.99 on the winner — so the dark knowledge on the runner-up classes is squashed to near zero and contributes almost nothing to the loss. The fix is **temperature**. We soften the distribution by dividing the logits by a temperature $T > 1$ before the softmax:

$$
p_i^{(T)} = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}.
$$

At $T = 1$ this is the ordinary softmax. As $T$ grows, the distribution flattens — the 0.99/0.008/0.002 becomes something like 0.5/0.3/0.2 — and the ratios among the small probabilities, the dark knowledge, become large enough to teach. We compute the teacher's softened probabilities $p_i^{(T)}$ and the student's softened probabilities $q_i^{(T)}$ at the *same* temperature, and we minimize the cross-entropy (equivalently the KL divergence) between them:

$$
\mathcal{L}_{\text{KD}} = -\sum_i p_i^{(T)} \log q_i^{(T)}.
$$

Then we combine that with the ordinary hard-label loss at $T = 1$:

$$
\mathcal{L} = \alpha\, T^2\, \mathcal{L}_{\text{KD}} + (1 - \alpha)\, \mathcal{L}_{\text{hard}}.
$$

Two things in that formula are not arbitrary, and getting them right is the difference between distillation working and silently doing nothing.

### Why the $T^2$ factor

The $T^2$ is not a hyperparameter. It is a correction that keeps the soft-loss gradient on the same scale as the hard-loss gradient as you change $T$. Here is the derivation. The gradient of the soft cross-entropy with respect to a student logit $v_i$ (the student's pre-softmax logit, with student soft prob $q_i^{(T)}$ and teacher soft prob $p_i^{(T)}$) is

$$
\frac{\partial \mathcal{L}_{\text{KD}}}{\partial v_i} = \frac{1}{T}\left(q_i^{(T)} - p_i^{(T)}\right).
$$

That $1/T$ appears because the softmax input was divided by $T$, so the chain rule pulls a $1/T$ through. In the high-temperature limit (large $T$, logits small relative to $T$), a Taylor expansion of the softmax gives $q_i^{(T)} - p_i^{(T)} \approx (v_i - z_i)/(NT)$ for $N$ classes with zero-mean logits, so the per-logit gradient scales like $1/T^2$. If you raise $T$ to expose the dark knowledge but do nothing else, the soft gradient *shrinks quadratically* and the hard loss silently takes over — your distillation term becomes decorative. Multiplying $\mathcal{L}_{\text{KD}}$ by $T^2$ exactly cancels this, restoring the soft gradient to roughly the same magnitude it had at $T=1$, so $\alpha$ means what you think it means regardless of $T$. Forget the $T^2$ and your carefully chosen $\alpha = 0.7$ behaves like $\alpha = 0.07$ at $T = 3$.

#### Worked example: the $T^2$ gradient rescaling with numbers

Take $T = 4$. The raw soft-loss gradient is attenuated by roughly $1/T^2 = 1/16$ relative to the hard-loss gradient. Suppose your intended blend is $\alpha = 0.5$ — you want the soft signal to count for half. Without the $T^2$ term, the effective contribution of the soft loss to the parameter update is about $0.5 \times \tfrac{1}{16} = 0.03$ of the total, versus $0.5$ for the hard loss — a 16:1 imbalance, so the student barely hears the teacher. Multiply $\mathcal{L}_{\text{KD}}$ by $T^2 = 16$ and the soft gradient is rescaled back up by 16×, restoring the 1:1 balance you asked for. The numbers: intended soft weight 0.50, actual-without-$T^2$ ≈ 0.03, actual-with-$T^2$ ≈ 0.50. That single factor is the difference between a student that learns from the teacher and one that ignores it.

### Why the combined loss, not pure distillation

Why keep the hard label at all? Because the teacher is wrong sometimes, and the ground truth anchors the student to reality. If you distill purely on teacher outputs ($\alpha = 1$), the student can only ever be as good as the teacher and inherits all of its biases; the hard term lets the student correct teacher mistakes where the data disagrees with it. Empirically $\alpha$ between 0.5 and 0.9 — leaning on the soft signal but keeping a real anchor to the labels — is the usual sweet spot. The stack of this combined loss is the figure later in the post; for now hold the shape: *hard label loss, plus temperature-scaled soft loss, blended by $\alpha$.*

## 3. Why ranking distillation beats logit matching for recommenders

Now the twist that makes this a recommendation post and not a generic distillation post. The base recipe above matches the teacher's *full probability distribution over classes*. For a recommender, that framing is subtly wrong, and the reason is the same reason BPR beats pointwise regression and the same reason we evaluate with NDCG: **what we serve is a ranking, and what we are graded on is the order of the top few items, not the absolute score of every item.**

![A two-column contrast of logit-matching distillation that matches every score against ranking distillation that focuses on the teacher's top-K order](/imgs/blogs/distillation-and-compression-for-recsys-5.png)

Consider what plain logit matching asks the student to do over a candidate set of 500 items. It tries to make the student's score for *every* item match the teacher's score for that item. But 490 of those 500 items are junk — the user will never look at them, they will never appear in the top 10 that the user actually sees, and the metric does not care about their relative order at all. Logit matching spends the student's limited capacity perfectly reproducing the teacher's opinions about garbage. Worse, because there are so many tail items, the *loss is dominated by them* — the student is pulled hard toward matching 490 irrelevant scores and only weakly toward getting the top 10 right. You have aimed your small model's attention at exactly the wrong place.

**Ranking Distillation** (Tang & Wang, 2018, *Ranking Distillation: Learning Compact Ranking Models With High Performance for Recommender System*) fixes this by changing what gets distilled. Instead of matching scores, the teacher produces its **top-K ranked list** for each query, and the student is trained to put *those* items on top. The teacher's exact scores for the long tail are discarded; what transfers is the identity and order of the items the teacher believes belong in the top K. The student now spends its capacity where the metric lives.

### The ranking-distillation loss

Concretely, RD takes the teacher's top-K items for a user — call them $\pi_1, \pi_2, \dots, \pi_K$ in the teacher's order — and treats them as additional positive examples for the student, but *weighted by position*. Items the teacher ranked higher get more weight, because getting the very top right matters more (NDCG discounts by position, so the model should too). The distillation term is a sum of position-weighted ranking losses pushing the student's score for each teacher-top-K item above its scores for sampled negatives:

$$
\mathcal{L}_{\text{RD}} = -\sum_{k=1}^{K} w_k \, \log \sigma\!\left(s_\theta(\pi_k) - s_\theta(j)\right),
$$

where $s_\theta$ is the student's score, $j$ is a sampled non-top-K item, $\sigma$ is the logistic function, and $w_k$ is the position weight. Tang & Wang use a weight that decays with rank position $r$, a blend of a position-importance term (high ranks matter more) and a ranking-discrepancy term (positions where the student already disagrees with the teacher matter more). The simplest defensible choice is a logarithmic discount mirroring NDCG, $w_k \propto 1/\log_2(k+1)$, so position 1 carries the most weight and position $K$ the least. The point of the weight is the whole point of RD: **concentrate the student's learning on the high-value positions.** The total student loss combines this with the hard labels, exactly as before:

$$
\mathcal{L} = \mathcal{L}_{\text{hard}} + \lambda\, \mathcal{L}_{\text{RD}}.
$$

There is a clean intuition for why this works that is worth stating plainly. The hard labels give the student a handful of true positives per user — the items that were actually clicked. That is a brutally sparse signal; most users have a few interactions and the model has to generalize from almost nothing. The teacher's top-K list gives the student *hundreds of extra, high-quality, soft positives* per user — the items a much more capable model believes are relevant. It is a massive expansion of the effective training signal, aimed exactly at the ranking the metric rewards. That is why a distilled small model can beat a from-scratch small model trained on the same data: it is trained on *more and better* signal, manufactured by the teacher.

## 4. The four flavors of distillation for recsys

"Distillation" is an umbrella. Under it sit four distinct strategies that differ in *what* knowledge they transfer, and the right one depends on your model and your serving constraints. The matrix below lays them against three properties: what signal they transfer, whether they are top-K aware, and whether the teacher needs features the student cannot serve.

![A matrix comparing logit, feature, ranking, and privileged-feature distillation across signal type, top-K awareness, and whether teacher features are needed](/imgs/blogs/distillation-and-compression-for-recsys-3.png)

**Response / logit-based distillation** is the Hinton base case: the student matches the teacher's output scores (softened by temperature). It is the simplest to implement — you only need the teacher's final outputs, which you can precompute and cache — but for ranking it has the misallocation problem we just discussed. Use it as a baseline or when the candidate set is small enough that every item matters.

**Feature / embedding-based distillation** matches the teacher's *intermediate* representations, not just its outputs. The student is trained so that its hidden layer (or its user/item embedding) is close to the teacher's, usually via an MSE or cosine loss on the vectors, sometimes through a small learned projection when the dimensions differ. This transfers richer structure — the *geometry* the teacher learned — and can help when the teacher's representation captures interactions the student's architecture would struggle to discover on its own. It is heavier (you need the teacher's internals, and a projection if dims differ) and it couples the two architectures more tightly.

**Ranking / top-K distillation** is Section 3: transfer the teacher's *order* over the top items. It is the most aligned with the recommendation metric and usually the best single choice for a ranking student. It needs the teacher's top-K lists (cheap to precompute) and a sampled negative per step.

**Privileged-features distillation (PFD)** is the production trick that surprises people the first time, and it gets its own section next because it is different in kind: the teacher is allowed to use features the student *cannot have at serve time*.

A practical note that the matrix encodes: the first three flavors all require only the teacher's *outputs or internals on the same inputs the student sees*, so the teacher can serve no role at inference — you precompute its targets offline once and throw it away. PFD is the same at inference (the student serves alone) but different at training (the teacher consumes extra features). None of the four needs the teacher *at serve time* — that is the entire economic point. You pay the teacher's cost once, offline, and never again.

Laid out as a decision table, with the question you should actually ask for each:

| Flavor | What transfers | Best when | Main cost / risk |
|---|---|---|---|
| Logit / response | Softened output scores | Candidate set is small; every item matters | Wastes capacity on tail items in top-K ranking |
| Feature / embedding | Hidden vectors, geometry | Teacher learned interactions the student can't find | Couples architectures; needs a projection |
| Ranking / top-K | Position-weighted teacher order | Any top-K ranking student (the usual case) | Needs top-K lists and a negative sampler |
| Privileged feature | Effect of unservable features | You have predictive post-event or expensive signals | Teacher-student gap; needs feature decoupling |

The default for a ranking student is the third row, often combined with the fourth: distill the teacher's top-K order *and* let that teacher see privileged features. The first two rows are situational — logit matching for a tiny candidate set, feature matching when the teacher's representation is the thing worth copying.

## 5. The combined student loss, in code

Let us make the combined hard-plus-soft loss concrete, because the gotchas live in the details — the temperature placement, the $T^2$, and keeping the two terms on the same scale. The stack below is the anatomy of the loss we will implement.

![A vertical stack showing the student loss built from a hard-label term, a temperature-scaled soft KD term, the T squared rescale, and a blend weight](/imgs/blogs/distillation-and-compression-for-recsys-4.png)

Here is a clean PyTorch implementation of the classic logit-based KD loss for a CTR-style ranker, where the per-item output is a single logit (click probability), so the "distribution" is the Bernoulli over click/no-click. With a single logit, softening means applying the temperature inside the sigmoid and matching with a binary KL.

```python
import torch
import torch.nn.functional as F

def kd_loss_binary(student_logits, teacher_logits, hard_labels,
                   T=3.0, alpha=0.5):
    """Combined hard + soft KD loss for a binary (click) ranker.

    student_logits, teacher_logits, hard_labels: shape (batch,)
    Returns a scalar loss.
    """
    # Hard term: ordinary BCE against the 0/1 click label, at T = 1.
    hard = F.binary_cross_entropy_with_logits(student_logits, hard_labels)

    # Soft term: match the teacher's softened click probability.
    # Divide logits by T BEFORE the sigmoid to soften the distribution.
    teacher_p = torch.sigmoid(teacher_logits / T)
    student_logp = F.logsigmoid(student_logits / T)
    student_log1mp = F.logsigmoid(-student_logits / T)

    # Binary cross-entropy with a soft target = binary KL up to a constant.
    soft = -(teacher_p * student_logp + (1.0 - teacher_p) * student_log1mp)
    soft = soft.mean()

    # The T^2 factor keeps the soft gradient on the same scale as hard.
    return (1.0 - alpha) * hard + alpha * (T * T) * soft
```

Three things in that function are load-bearing. First, the temperature divides the logits *before* the nonlinearity, for both teacher and student, and at the same value. Second, the `T * T` multiplier on the soft term is the gradient-rescaling from Section 2; drop it and the soft term goes quiet at higher $T$. Third, we use `logsigmoid` rather than `log(sigmoid(...))` for numerical stability — at large $|logit|/T$ the naive version underflows to `-inf` and your loss becomes `nan` two epochs in, which is the single most common way this code silently breaks.

For a multi-class softmax head (e.g. a retrieval-style student picking among a candidate set), the categorical version is the canonical Hinton form:

```python
def kd_loss_categorical(student_logits, teacher_logits, hard_labels,
                        T=3.0, alpha=0.5):
    """student_logits, teacher_logits: (batch, n_candidates);
    hard_labels: (batch,) class indices."""
    hard = F.cross_entropy(student_logits, hard_labels)

    # KL(teacher_soft || student_soft), the standard KD divergence.
    teacher_soft = F.softmax(teacher_logits / T, dim=-1)
    student_logsoft = F.log_softmax(student_logits / T, dim=-1)
    soft = F.kl_div(student_logsoft, teacher_soft,
                    reduction="batchmean")

    return (1.0 - alpha) * hard + alpha * (T * T) * soft
```

Note `F.kl_div` expects the *student* as log-probabilities and the *teacher* as probabilities (its argument order is `(input_log_probs, target_probs)`), and `reduction="batchmean"` is the mathematically correct normalization — the default `"mean"` divides by the wrong count and quietly mis-scales your loss. These are the kinds of API details that separate code that works from code that compiles and lies.

## 6. Implementing ranking distillation end to end

Now the recommendation-specific recipe: a big teacher ranker produces soft top-K targets, and a small student is trained on a combined hard-plus-ranking-distillation loss. We will use MovieLens-style implicit data (a user's positive items), a two-tower-ish scoring student for clarity, and we will precompute the teacher's top-K so the teacher is never in the training loop's hot path.

First, the teacher's top-K targets. We run the trained teacher over each user's candidate set once and cache the top-K item ids. This is the step that makes the whole thing cheap — the teacher's expensive forward pass happens *once per user offline*, not once per training step.

```python
import torch

@torch.no_grad()
def precompute_teacher_topk(teacher, user_ids, candidate_items,
                            K=50, device="cuda"):
    """Returns a dict: user_id -> LongTensor of the teacher's top-K item ids."""
    teacher.eval()
    topk = {}
    for u in user_ids:
        cand = candidate_items[u].to(device)          # (n_cand,)
        scores = teacher.score(u, cand)               # (n_cand,)
        idx = torch.topk(scores, k=min(K, cand.numel())).indices
        topk[u] = cand[idx].cpu()
    return topk
```

Now the ranking-distillation loss. For each user we have (a) the true positives from the hard labels and (b) the teacher's top-K list. We push the student's score for each teacher-top-K item above the student's score for sampled negatives, weighted by the NDCG-style position discount so the top of the teacher's list dominates.

```python
import torch
import torch.nn.functional as F

def position_weights(K, device):
    ranks = torch.arange(1, K + 1, device=device, dtype=torch.float)
    w = 1.0 / torch.log2(ranks + 1.0)     # NDCG-style discount
    return w / w.sum()                    # normalize so weights sum to 1

def ranking_distillation_loss(student, user, teacher_topk_items,
                              neg_items, pos_weights):
    """student.score(user, items) -> scores; one negative per top-K slot."""
    s_pos = student.score(user, teacher_topk_items)   # (K,)
    s_neg = student.score(user, neg_items)            # (K,) sampled negs
    # Pairwise: teacher-top-K item should outrank a random negative.
    per_pos = -F.logsigmoid(s_pos - s_neg)            # (K,)
    return (pos_weights * per_pos).sum()              # position-weighted
```

And the training step combining the hard BPR loss (on true clicks) with the ranking-distillation loss:

```python
def train_step(student, opt, batch, teacher_topk, sampler, K, lam, device):
    student.train()
    opt.zero_grad()
    total = 0.0
    pos_w = position_weights(K, device)
    for user, pos_item in batch:               # true positives = hard labels
        # Hard term: BPR on the actual positive vs a sampled negative.
        neg = sampler.sample(user)
        s_p = student.score(user, pos_item)
        s_n = student.score(user, neg)
        hard = -F.logsigmoid(s_p - s_n)

        # Distill term: teacher's top-K vs K sampled negatives.
        tk = teacher_topk[user].to(device)
        negs = sampler.sample_many(user, n=tk.numel())
        rd = ranking_distillation_loss(student, user, tk, negs, pos_w[:tk.numel()])

        total = total + hard + lam * rd
    total = total / len(batch)
    total.backward()
    opt.step()
    return total.item()
```

A few practitioner notes. The teacher's top-K *includes* the user's true positives most of the time (a good teacher ranks real clicks highly), so the hard term and the distill term reinforce rather than fight. The hyperparameter $\lambda$ trades the two off; start around 0.5 and tune on a validation NDCG, not on training loss. $K$ should match roughly the depth the metric and the product care about — if you show 10 items and re-rank 50, distilling the teacher's top 50 is sensible; distilling the top 500 wastes effort on positions no user sees. And critically: evaluate the *student* on a temporal split with full-catalog NDCG@10, never sampled metrics, because the whole reason you are doing this is to ship the student, and sampled metrics can rank your candidate models inconsistently (Krichene & Rendle, KDD 2020).

#### Worked example: a 10× smaller student's latency and quality

Concrete target numbers from a MovieLens-20M-scale setup. The teacher is a 200M-parameter multi-task ranker; the student is an 8M-parameter single-head ranker — about 25× fewer parameters and, because the dense compute scales with the network, roughly 9–10× cheaper per scoring. Suppose offline the teacher hits NDCG@10 = 0.412 and a from-scratch 8M student hits 0.388 (a 0.024 gap — the small model genuinely cannot match the big one on labels alone). Train the same 8M student with ranking distillation from the teacher and it reaches NDCG@10 = 0.404 — it recovered $0.404 - 0.388 = 0.016$ of the 0.024 gap, about **two-thirds of the teacher's edge**, while serving at the small model's p99 of 9 ms versus the teacher's 84 ms. You gave back $0.412 - 0.404 = 0.008$ NDCG and bought a **9.3× latency reduction** and a 25× smaller model. That is the trade, and in production that trade ships almost every time.

### Choosing the temperature and the blend, concretely

Two knobs decide whether the soft signal actually teaches: the temperature $T$ and the blend $\alpha$ (or $\lambda$ for the ranking variant). They are not independent, and the usual mistake is to set them by superstition. Here is how to reason about each.

Temperature controls *how much dark knowledge is exposed*. At $T = 1$ a confident teacher is nearly one-hot and the student learns almost nothing beyond the argmax. As $T$ rises, the runner-up probabilities inflate and the relative structure among them becomes visible to the loss. But push $T$ too high and the distribution flattens toward uniform — at the extreme, every item looks equally likely and the dark knowledge is washed out the other direction. There is a sweet spot, usually $T$ between 2 and 5 for ranking heads, where the second- and third-choice items carry real weight but the teacher's confident top item still dominates. The right way to pick it is a small grid on validation NDCG, not a guess; the wrong way is to copy a value from an image-classification paper where the class count and logit scale are completely different.

The blend $\alpha$ controls *how much you trust the teacher versus the ground truth*. At $\alpha = 0$ you ignore the teacher (a from-scratch student). At $\alpha = 1$ you ignore the labels and can never beat the teacher, inheriting all of its mistakes. The interesting region is in between, and for recsys it leans high — 0.5 to 0.9 — because the hard labels are so sparse and noisy that the teacher's denser soft signal is usually the more reliable teacher per example. But keep $\alpha < 1$ so the labels can still correct the teacher where the data plainly disagrees with it; a teacher that systematically over-ranks popular items, for instance, should be partly overruled by users who clicked the obscure thing.

#### Worked example: temperature exposing the runner-up

Make the dark-knowledge argument numeric. Suppose the teacher's logits over four candidate items for one user are $z = (4.0, 2.0, 1.0, -1.0)$. At $T = 1$ the softmax is approximately $(0.84, 0.11, 0.04, 0.01)$ — the top item swamps everything, and the student learning from this barely sees that item two is preferred to item three. Now soften at $T = 3$: divide the logits to get $(1.33, 0.67, 0.33, -0.33)$, and the softmax becomes roughly $(0.44, 0.23, 0.16, 0.08)$. Item two's probability tripled from 0.11 to 0.23, item three's quadrupled from 0.04 to 0.16, and the *ratio* between them is now a strong, learnable signal: the teacher thinks item two is clearly better than item three, and at $T = 3$ the student finally hears it. The hard label only ever said "item one" — the temperature is what surfaced the teacher's opinion about the *rest* of the order, which is exactly the order NDCG@10 grades. This is the entire mechanism of distillation in four numbers.

## 7. The eval harness: measuring the student honestly

Before privileged features, a section on measurement, because the whole post is an argument about a *trade*, and a trade you cannot measure is a trade you cannot defend. The eval harness is where distillation projects succeed or quietly fail — a student that looks great on a leaky split or a sampled metric will embarrass you in the A/B test.

The non-negotiables, restated as code. First, a temporal split — train on the past, evaluate on the future — and full-catalog NDCG@10, never sampled. Here is a compact, correct NDCG@K over the full catalog for a batch of users:

```python
import numpy as np

def ndcg_at_k(scores, holdout_items, k=10):
    """scores: (n_users, n_items) student scores over the FULL catalog.
    holdout_items: list of sets, the true future positives per user.
    Returns mean NDCG@k over users with at least one holdout item."""
    ndcgs = []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))   # positions 1..k
    for u in range(scores.shape[0]):
        rel = holdout_items[u]
        if not rel:
            continue
        topk = np.argpartition(-scores[u], k)[:k]
        topk = topk[np.argsort(-scores[u][topk])]    # sort the top-k
        gains = np.array([1.0 if i in rel else 0.0 for i in topk])
        dcg = (gains * discounts).sum()
        ideal = discounts[:min(len(rel), k)].sum()   # best possible DCG
        ndcgs.append(dcg / ideal if ideal > 0 else 0.0)
    return float(np.mean(ndcgs))
```

Two correctness traps live in those few lines. The `argpartition` then `argsort` pattern gets the top-K in $O(n)$ instead of sorting the whole catalog per user — at a million items, full sorting per user is the difference between an eval that finishes in minutes and one that runs overnight. And the ideal DCG must be computed from the *number of relevant items capped at K*, not assumed to be 1; a user with three future clicks has a different ideal than a user with one, and getting this wrong silently inflates or deflates NDCG.

Second, latency measured the way production serves: warmed up, at the real candidate-set size, reported at p99. A mean-latency number is a lie of omission — it hides exactly the tail that fails the budget.

```python
import time, numpy as np, torch

@torch.no_grad()
def measure_p99_latency(model, user, candidates, n_warmup=50, n_runs=500):
    model.eval()
    # Warm up: fill caches, trigger lazy CUDA init, let the JIT settle.
    for _ in range(n_warmup):
        _ = model.score(user, candidates)
    if candidates.is_cuda:
        torch.cuda.synchronize()
    lat = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.score(user, candidates)        # score ALL candidates
        if candidates.is_cuda:
            torch.cuda.synchronize()             # don't time async kernels
        lat.append((time.perf_counter() - t0) * 1000.0)   # ms
    lat = np.array(lat)
    return {"p50": np.percentile(lat, 50),
            "p99": np.percentile(lat, 99),
            "mean": lat.mean()}
```

The `torch.cuda.synchronize()` calls are load-bearing: without them you time the *launch* of async GPU kernels, not their completion, and you will report a latency that is fiction — often 10× too fast, which is how teams convince themselves an unservable model is servable. The warmup loop matters for the same reason it matters in production: the first few requests pay for lazy initialization, cache fills, and page faults that the steady state does not.

Third, and most important, offline is not the judge — online is. A student can match the teacher's offline NDCG and still lose in the A/B test, because both the offline metric and the teacher's soft targets are computed on *logged* data that carries the production system's biases. The honest workflow is: pick the model offline with the harness above, then *ship it behind an experiment* and read the online metric (CTR, watch-time, retention, GMV) against the incumbent. The happy and common result for a well-built distilled student is that it matches the from-scratch student's *cost* and the teacher's *online engagement* — which is the entire point of the exercise. If the offline win evaporates online, suspect that the student faithfully inherited a teacher bias, and revisit the teacher.

## 8. Privileged-features distillation: the production secret weapon

This is the flavor that, once you see it, you cannot unsee, and it is responsible for some of the largest reported wins in industrial recsys. The idea (Xu et al., 2020, *Privileged Features Distillation at Taobao Recommendations*, Alibaba) inverts the usual relationship between teacher and student. Normally the teacher is bigger or deeper. In **privileged-features distillation (PFD)** the teacher is not necessarily bigger — it is *better informed*. It gets to see features that the student cannot have at serve time.

![A branching dataflow where the teacher consumes both servable and privileged features while the student sees only servable features](/imgs/blogs/distillation-and-compression-for-recsys-6.png)

Where do "features the student cannot serve" come from? They are everywhere in recsys, and they fall into two buckets:

**Post-event / interaction features.** When you log a training example, you know things that did not exist at serve time. You know how long the user *dwelled* on the item after clicking, whether they finished the video, whether they added to cart, whether they came back. These are extraordinarily predictive of the label — dwell time correlates with genuine interest far better than the click itself — but they are *post-impression*: at serve time, when you must score the candidate before the user has done anything, they do not exist. You cannot put dwell time into the serving model because dwell time is in the future.

**Expensive features.** Some features are computable at serve time in principle but too slow to compute within the latency budget — a heavy cross-feature, a fresh graph embedding, the output of another expensive model. The teacher, trained offline, can afford them; the student cannot.

PFD lets you *use* these features without serving them. You train a teacher that consumes the privileged features (plus the ordinary ones) and learns a strong predictor. Then you distill the teacher's outputs into a student that consumes *only the servable features*. The student never sees dwell time — but it learns to *anticipate* it, because it is trained to match a teacher who knew it. The serving model gets the benefit of the privileged information distilled into its parameters, without needing the information at serve time.

### Why privileged-features distillation works

There is a real theoretical reason this is more than a hack, and it connects to Vapnik's *learning using privileged information* (LUPI). The privileged features act as a **regularizer and a better supervision signal**. The hard label is a single noisy bit — clicked or not — and a small student trying to fit it directly will overfit noise and underfit structure. The teacher, armed with dwell time and finish rate, produces a *smoother, better-calibrated, lower-noise* target: instead of a hard 0/1, the student sees the teacher's soft probability that reflects the true engagement, with the label noise partly averaged out by the extra signal. Learning from a teacher who can *explain* the label (via the privileged features) is easier than learning the label cold, the same way a student learns faster from a teacher who shows the reasoning than from an answer key alone. Formally, the privileged information lets the teacher approximate the *true* conditional engagement probability more closely than the raw labels reveal, and matching that smoother target gives the student a lower-variance learning problem.

A subtle but critical implementation detail from the Alibaba work: the privileged and ordinary features should be *decoupled* in the teacher, or the teacher leans entirely on the powerful privileged features and learns a representation the student can never reproduce from ordinary features alone. The fix is to stop the gradient appropriately or to weight the teacher so it still learns a transferable signal over the servable features. They also warn about the teacher-student gap: if the teacher is *too* much better than the student can ever be, the distillation target is unreachable and the student is pulled toward an impossible goal, which can hurt. The teacher should be *informed*, not *omniscient* relative to the student's reachable function class.

This single technique — let the teacher peek at post-event signals, distill into a serving model that only sees pre-event features — is one of the highest-leverage moves in production ranking, and it composes cleanly with ranking distillation: the teacher can be both better-informed (privileged features) and the source of top-K targets.

## 9. Embedding compression: shrinking the part that dominates

Distillation shrinks the *dense network*. But as the companion post on [embeddings, the heart of recommenders](/blog/machine-learning/recommendation-systems/embeddings-the-heart-of-recommenders) argues at length, in a deep recommender the dense network is the cheap part — the **embedding tables** hold 90–99% of the parameters and dominate both memory and the part of latency that is memory-bandwidth-bound. A distilled student with an 8 MB MLP and a 32 MB embedding table is still mostly embedding table. So compressing the student is largely about compressing its embeddings.

![A two-column contrast of an fp32 student embedding table versus an int8 quantized table showing four times smaller size at near-equal NDCG](/imgs/blogs/distillation-and-compression-for-recsys-7.png)

There are three levers, in rough order of how often you reach for them:

**Quantization** stores each embedding value in fewer bits — int8 instead of float32 is the default, a 4× reduction. **Pruning** removes whole rows (rare items routed to a shared bucket) or zeros out unimportant dimensions. **Hashing** caps the table size by mapping many ids into a shared, smaller set of rows, accepting collisions. The embeddings post covers hashing and the quotient-remainder trick in depth; here I will focus on the lever you almost always pull first and that pairs most naturally with a distilled serving model: **int8 quantization.**

### int8 embedding quantization, and why the error is tiny

Quantizing to int8 maps each float value to one of 256 integer levels. The standard scheme is *per-row* (per-embedding) symmetric quantization: for embedding row $\mathbf{v}$, compute the scale $s = \max_i |v_i| / 127$, store $\hat{v}_i = \text{round}(v_i / s)$ as int8, and keep $s$ as a float32 scale per row. At read time you dequantize $v_i \approx s \cdot \hat{v}_i$. The memory: instead of $d \times 4$ bytes per row you store $d \times 1$ bytes plus one 4-byte scale — for $d = 64$, that is $256 + 4 = 260$ bytes versus 256 bytes... wait, that is int8 at $d$ bytes plus the scale, so $64 + 4 = 68$ bytes versus $64 \times 4 = 256$ bytes, a 3.76× reduction (the per-row scale eats a little of the ideal 4×).

The reason this barely moves the metric is the same $\sqrt{d}$ error-cancellation argument from the embeddings post, worth restating because it is the justification for shipping int8 by default. The score is a dot product $\sum_i u_i v_i$ of a user embedding and an item embedding. Quantizing introduces a per-element rounding error $\varepsilon_i$ that is roughly uniform in $\left[-s/2, +s/2\right]$ and *independent* across the $d$ dimensions. The error in the dot product is $\sum_i u_i \varepsilon_i$. Because the $\varepsilon_i$ are zero-mean and independent, this sum's *standard deviation grows as $\sqrt{d}$*, not as $d$ — the errors partially cancel rather than accumulate. Meanwhile the signal (the dot product itself) grows as $d$. So the *relative* error of the score shrinks as $\sqrt{d}/d = 1/\sqrt{d}$ as the dimension grows. For $d = 64$ that is a relative score error around 12%... but what the *ranking* metric cares about is whether quantization *reorders* items, and small independent perturbations rarely flip well-separated items. The net effect on NDCG@10 is typically a fraction of a point — the figure above puts it at 0.404 → 0.401, a 0.003 drop for a 4× memory cut.

#### Worked example: int8 memory and the NDCG cost

Take the 8M-parameter distilled student on a 500,000-item catalog with $d = 64$ item embeddings. The item table is $500{,}000 \times 64 \times 4 = 128$ MB in float32... but our worked student is smaller; say the catalog is 130,000 items, giving $130{,}000 \times 64 \times 4 \approx 33$ MB of float32 embeddings — the bulk of the 34 MB student. Quantize to int8 with per-row scales: $130{,}000 \times (64 + 4) \approx 8.8$ MB, call it ~10 MB total with the MLP — a 3.4× model shrink. Measured NDCG@10 goes 0.404 → 0.401, a **0.003 loss for a 3.4× smaller served model**. At fleet scale that smaller table means more of it fits in cache, fewer cache misses on the embedding gather, and lower p99 — the compression pays twice, in memory and in latency. This is why int8 is the default and you only escalate to product quantization (PQ) when int8 still isn't small enough and you have *measured* you can afford the larger hit.

Here is the int8 quantization of the student's embedding table in PyTorch, per-row, with a dequantizing lookup:

```python
import torch

def quantize_int8_per_row(table: torch.Tensor):
    """table: (n_items, d) float32. Returns (int8 codes, float32 scales)."""
    scales = table.abs().amax(dim=1, keepdim=True) / 127.0   # (n_items, 1)
    scales = scales.clamp(min=1e-8)                          # avoid div-by-0
    codes = torch.round(table / scales).clamp(-127, 127).to(torch.int8)
    return codes, scales.squeeze(1)                          # (n,d) int8, (n,) f32

class Int8EmbeddingTable:
    """Memory-cheap embedding lookup that dequantizes on read."""
    def __init__(self, table: torch.Tensor):
        self.codes, self.scales = quantize_int8_per_row(table)

    def lookup(self, item_ids: torch.Tensor) -> torch.Tensor:
        codes = self.codes[item_ids].to(torch.float32)       # (B, d)
        scales = self.scales[item_ids].unsqueeze(1)          # (B, 1)
        return codes * scales                                # dequantized (B, d)
```

For a real deployment you would lean on framework support — PyTorch's `torch.quantization` / `quantized` embedding ops, or TorchRec's quantized embedding-bag kernels that keep the int8 codes packed and dequantize inside a fused kernel so you never materialize the float table. The principle is identical: store int8 codes plus per-row scales, dequantize at gather time, and the dot-product error stays small for the $\sqrt{d}$ reason above. Note that you can also *train* the student quantization-aware (QAT) — simulate the rounding in the forward pass so the student learns weights robust to it — which recovers most of even the small int8 gap, at the cost of a more complex training loop. For int8 it is rarely worth it; for int4 or PQ it often is.

### Pruning and product quantization, when int8 is not enough

int8 buys 4×. When you need more — billion-item catalogs that do not fit even quantized, or memory-constrained edge serving — two further levers apply, in increasing order of aggressiveness and risk.

**Pruning** removes capacity rather than shrinking its precision. The cheapest and most effective form in recsys is *row pruning by frequency*: drop the embeddings of items below a count threshold and route them to a single shared out-of-vocabulary bucket. This is nearly free quality-wise because a model cannot learn a meaningful vector for an item it has seen twice anyway — those rows are mostly noise — and it doubles as your cold-start mechanism, since new and rare items land in the shared bucket and inherit a sensible average representation. A power-law catalog often has 80–95% of its rows in the cold tail, so frequency pruning can shrink the table by an order of magnitude before you have touched precision at all. *Dimension* pruning — zeroing or removing unimportant columns of the table — is the mixed-dimension idea from the embeddings post: spend width on the head, starve the tail. Always prune by frequency *before* you quantize; there is no point spending int8 codes on rows you are about to delete.

**Product quantization (PQ)** is the heavy hammer. Instead of quantizing each value independently, PQ splits each $d$-dimensional embedding into $m$ sub-vectors, learns a small codebook (say 256 centroids) for each sub-space via k-means, and stores each sub-vector as a single byte: the index of its nearest centroid. A 64-dim float32 embedding (256 bytes) becomes $m = 8$ bytes — a 32× reduction, far past int8's 4×. The cost is a larger, *correlated* error: you are approximating each sub-vector by its nearest centroid, and the error no longer cancels as cleanly as independent int8 rounding does, so the NDCG hit is bigger and you should expect to need quantization-aware training or fine-tuning to recover it. PQ is also what FAISS uses for compressed ANN indexes (`IndexIVFPQ`), so it composes naturally with the retrieval stage — the same codes can serve compressed lookup *and* approximate search. The decision rule is simple: reach for int8 by default, frequency-prune the tail first, and only escalate to PQ when you have *measured* that int8 plus pruning is still too big and that you can afford the larger quality hit. The deeper treatment of hashing, the quotient-remainder trick, and the full memory-versus-recall Pareto lives in the companion [embeddings post](/blog/machine-learning/recommendation-systems/embeddings-the-heart-of-recommenders); here the point is that distillation shrinks the network and these levers shrink the embeddings, and a production student uses both.

## 10. Self-distillation, online distillation, and distilling LLM recommenders

Three variants extend the basic teacher→student picture, and each solves a different operational pain.

**Self-distillation** uses a teacher and student of the *same* architecture — sometimes literally the same model at a later training stage teaching an earlier one, or an ensemble of checkpoints teaching a single model. There is no compression goal; the win is *regularization*. The soft targets from the teacher (even a same-size teacher) smooth the labels and act like a learned label-smoothing, often improving the student over training on hard labels alone. A common recsys use is *born-again* training: train a model, use it as a teacher to train a fresh model of the same size, and the second model is frequently a touch better. It is cheap insurance against label noise and a way to squeeze a little more out of a fixed serving budget.

**Online (codistillation / mutual) distillation** trains the teacher and student *simultaneously* rather than freezing a pretrained teacher first. Two (or more) models train in parallel and each distills from the others' current predictions. The appeal is operational: you do not need a separate, completed teacher-training phase, and the models can be smaller peers that bootstrap each other. The risk is that early in training the "teacher" is bad, so its soft targets are noise — codistillation works best once both models are past the random-init phase, and you often ramp the distillation weight up over training. For recsys with continuously arriving data, online distillation fits the streaming reality nicely: the freshest model can teach the deployed one as new interactions arrive.

**Distilling an LLM recommender into a small servable model** is the frontier case and increasingly the *reason* people reach for distillation. An LLM-based recommender — one that reads item descriptions and user history as text and reasons about relevance, as covered in the companion post on [finetuning LLMs for recommendation in practice](/blog/machine-learning/recommendation-systems/finetuning-llms-for-recommendation-in-practice) — can be remarkably accurate, especially in cold-start and long-tail regimes where it leverages world knowledge. It is also completely unservable at ranking-stage latencies: seconds per request, not milliseconds. So you use the LLM as a *teacher*. Two common patterns:

1. **Score / ranking distillation.** Have the LLM rank or score candidates offline for a large set of (user, candidate) pairs, then train a small conventional ranker (a two-tower or DCN student) to match those rankings — exactly the ranking-distillation recipe from Section 6, with an LLM in the teacher slot. The student serves at normal latency; the LLM's judgment is baked into its parameters.
2. **Embedding / representation distillation.** Use the LLM (or a large text encoder) to produce rich item and user embeddings offline, then distill those into the student's smaller embedding tables, or train the student to predict them. This transfers the LLM's semantic understanding of content into a cheap lookup table — the student gets LLM-quality content embeddings at int8-lookup cost.

In both, the teacher's expense is amortized: you run the LLM once per item or per (user, candidate) batch offline, cache the targets, and serve the small student forever. The cost asymmetry is enormous — a one-time offline LLM pass versus a per-request LLM call is the difference between a feasible feature and an impossible one. This is the dominant way LLM recsys quality actually reaches production today: not by serving the LLM, but by distilling it.

## 11. Putting it together: the results table

The argument of this whole post is a single table, on a named dataset, that lets a skeptic see the trade. Below is the structure of that comparison — teacher, from-scratch student, distilled student, and distilled-plus-int8 student — on NDCG@10, p99 latency, and model size. The matrix figure renders the same numbers.

![A matrix of teacher, scratch student, distilled student, and int8 student rows against NDCG, latency, and size columns showing the student recovers quality cheaply](/imgs/blogs/distillation-and-compression-for-recsys-8.png)

| Model | Params | NDCG@10 | p99 latency | Model size |
|---|---|---|---|---|
| Teacher (multi-task ranker) | 200M | **0.412** | 84 ms | 780 MB |
| From-scratch student | 8M | 0.388 | **9 ms** | 34 MB |
| Distilled student (ranking + soft) | 8M | 0.404 | **9 ms** | 34 MB |
| + int8 embeddings | 8M | 0.401 | **9 ms** | 10 MB |

Read it the way you would defend it in a launch review. The from-scratch student is fast and small but gives back 0.024 NDCG — a real, shippable-quality regression. The *distilled* student is the same speed and size but recovers two-thirds of that gap (0.388 → 0.404), landing within 0.008 of the teacher it can never afford to serve. Adding int8 embeddings shrinks the model another 3.4× (34 MB → 10 MB) for a further 0.003 NDCG, which is in the noise of a temporal split. The bottom line: you ship a model that is **9.3× faster, 78× smaller, and within 0.011 NDCG of the teacher** — and the 0.011 is far smaller than the 0.024 you would lose by just training small from scratch. The teacher's quality, minus a sliver, at the cheap student's cost. (These are representative figures consistent with the magnitudes in the Ranking Distillation and PFD papers and typical MovieLens-20M two-tower results; your exact numbers depend on the gap between your teacher and student and on your candidate-set size, so reproduce them on your own data.)

How to measure this honestly, because the table is only as trustworthy as its methodology:

- **Temporal split.** Train on interactions up to time $t$, evaluate on interactions after $t$. A random split leaks the future and inflates every row equally, hiding the real gap.
- **Full-catalog NDCG@10**, not sampled. Sampled metrics can rank your candidate models *inconsistently* (Krichene & Rendle, KDD 2020), so a sampled comparison might tell you the from-scratch student is fine when full-catalog says it isn't.
- **Latency measured warmed-up, at the real candidate-set size, at p99.** Score 500 candidates per request the way production does; report the tail, not the mean; warm the cache first so you are not measuring cold-start page faults.
- **Online is the judge.** Offline NDCG up does not guarantee online lift (the offline↔online gap is the spine of this series). The distilled student should be A/B tested; the usual happy result is that it matches the from-scratch student's *cost* and the teacher's *engagement*, which is the point.

## 12. Case studies and real numbers

Four named results anchor the techniques in published practice.

**Ranking Distillation (Tang & Wang, RecSys/KDD 2018).** The paper that named the technique. They train a large ranking model (the teacher) and distill its top-K rankings into a student roughly *half the size or smaller*, and show the student matches or even slightly exceeds the teacher on ranking metrics (Recall and NDCG) on datasets including Gowalla and a recommendation benchmark, while being far cheaper to serve. The key reported finding is the one this post is built on: distilling the *ranking* (top-K positions, position-weighted) substantially outperforms a student trained on the ground-truth labels alone, and the compact student can be deployed where the teacher cannot. Their position-aware weighting — emphasizing the teacher's highest-ranked items — is the design that makes it work.

**Privileged Features Distillation at Taobao (Xu et al., 2020, Alibaba).** PFD in production at one of the largest e-commerce recommenders in the world. They identify post-event/privileged features — signals available at training but not at serving (interaction features that only exist after a click) — and distill a privileged teacher into the serving model on both the coarse ranking (candidate generation) and fine ranking stages. The reported result is a meaningful offline metric gain *and* an online lift in their A/B tests (they report online improvements in the conversion/GMV-relevant metrics on Taobao's live traffic), from a serving model that uses *no* extra features — the privileged information is entirely distilled into the parameters. This is the canonical evidence that "let the teacher see the future, distill it into a model that can't" is a real production win, not a paper curiosity.

**Distilling LLM rankers.** A rapidly growing literature uses a large language model as a teacher and a small conventional model as the served student. The recurring finding across this work (e.g. LLM-as-teacher for ranking, recommendation-as-language-modeling distilled into two-tower/sequential students) is that the small student inherits much of the LLM's quality — particularly in cold-start and semantic-matching regimes where the LLM's world knowledge helps — at conventional serving latency. The economic argument is decisive: a per-request LLM call is infeasible at ranking QPS, but a one-time offline LLM scoring pass that produces distillation targets is cheap and amortized. Expect this to be the default deployment path for "LLM recsys" — the LLM teaches, the small model serves.

**Two-stage teacher→student in industrial pipelines.** Beyond any single paper, the *pattern* — train an expensive teacher offline, distill into a cheap served student — is standard at scale (variants appear in work from Google, Alibaba, and others, and in the DistilBERT line for NLP that proved a 40%-smaller, 60%-faster student can retain ~97% of teacher quality). For recsys the same shape recurs: the teacher can be a deeper ranker, an ensemble, a privileged-feature model, or an LLM; the student is whatever fits the latency slice; the soft/ranking targets are precomputed offline. The companion post on [multi-task and multi-objective ranking with MMoE and PLE](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple) describes exactly the kind of accurate-but-heavy multi-task teacher that distillation most often needs to compress.

## 13. The accurate teacher, the cheap student

Step back to the shape of the decision, because it is the same shape every time. You have a model that is more accurate than you can serve, and a budget that is firmer than you can negotiate. The before/after below is the destination.

![A two-column contrast of a slow accurate teacher against a fast small distilled student that nearly matches its quality](/imgs/blogs/distillation-and-compression-for-recsys-2.png)

The teacher is the column on the left: highest NDCG, but p99 latency and model size that fail the budget by an order of magnitude. The distilled student is the column on the right: nearly the same NDCG, latency inside the slice, model small enough to fit in cache. The arrow between them is the entire technique — soft targets and top-K rankings precomputed offline by the teacher, used to train the student, plus int8 embeddings to shrink what is left. You ship the right column and keep the left column on a shelf as the thing that *defines* how good the student could be.

The stress tests, because a decision you have not stressed is a decision you do not understand:

- **What if the teacher is barely better than the student?** Then distillation buys little — the dark knowledge is thin when the teacher's distribution is close to the student's. Don't distill when a from-scratch student already hits target; the complexity isn't free.
- **What if the teacher is *vastly* better (an LLM teaching a tiny model)?** The target may be partly unreachable, and the student can be pulled toward a function it cannot represent. Use ranking distillation (transfer order, not exact scores), consider an intermediate-size teaching assistant, and watch for the student getting *worse* than its from-scratch baseline — a sign the gap is too large.
- **What if the labels are mostly noise (implicit feedback, false negatives everywhere)?** This is where distillation *helps most* — the teacher's soft targets denoise the labels, and PFD with post-event signals (dwell, finish) cleans them further. Lean harder on the soft term ($\alpha$ up).
- **What if offline NDCG rises but online is flat?** The student matched the teacher's *offline* judgments, which encode the teacher's *biases* (position bias, popularity bias, the closed feedback loop). Distillation faithfully transfers bias. The fix is the same as for any ranker — debias the teacher first, or the student will be a smaller, faster version of the same mistake.
- **What if a privileged feature is computed differently offline and online?** With PFD the student never serves the privileged feature, so its *train-serve skew* on that feature is moot — that is part of PFD's appeal. But the *servable* features the student does use can still skew; the usual feature-store discipline applies, covered in [large-scale embedding systems and feature stores](/blog/machine-learning/recommendation-systems/large-scale-embedding-systems-and-feature-stores).

## When to reach for distillation and compression (and when not to)

Every technique here is a cost — training complexity, a teacher to maintain, a quantization pipeline. Plainly:

- **Reach for distillation when you have a model that wins offline but fails the latency or cost budget.** That is the canonical case: the teacher exists and is good, the budget is firm, and the from-scratch small model gives back too much quality. Don't distill when the small model already hits target — you'd add a teacher-training phase and a more complex loss for nothing.
- **Reach for *ranking* distillation, not logit matching, for any top-K ranking student.** It aligns the transfer with the metric. Don't waste a small student's capacity matching the teacher's scores on tail items no user will ever see.
- **Reach for privileged-features distillation when you have strong post-event or expensive signals.** Dwell time, finish rate, add-to-cart, a heavy cross-feature — if it is predictive but unservable, PFD turns it into served quality for free at inference. Don't bother if your only features are already servable; there is no privilege to distill.
- **Reach for int8 embedding quantization by default at serve time.** It is the highest-leverage, lowest-risk compression in the post — 4× smaller for a fraction of a point of NDCG, justified by the $\sqrt{d}$ argument. Don't reach for int4 or PQ until you've measured int8 isn't small enough *and* that you can afford the larger hit.
- **Reach for self-distillation when you want a free regularization bump on a fixed budget.** Born-again training often adds a sliver of quality at no serving cost. Don't expect compression from it — that's not what it's for.
- **Reach for LLM distillation when an LLM recommender is accurate but unservable.** Distill it offline into a conventional student. Don't try to serve the LLM at ranking latency — it is four or five orders of magnitude over budget and no amount of hardware fixes that economically.
- **Don't distill a biased teacher and expect clean online lift.** Distillation transfers bias faithfully. Debias the teacher first, or A/B test and be ready for the offline win to evaporate.

## Key takeaways

1. **The serving-cost problem is a budget with numbers** — $C$ candidates × $Q$ QPS at a p99 latency slice. State it numerically before reaching for a tool; a 10× more expensive teacher can be a 10× larger fleet and a blown tail.
2. **Distillation transfers a teacher's dark knowledge, not just its labels.** The student learns from the teacher's full softened distribution, which is richer than the sparse, noisy hard labels — that is why a distilled small model beats a from-scratch one.
3. **The $T^2$ factor is not optional.** Raising temperature to expose dark knowledge attenuates the soft gradient as $1/T^2$; multiplying the KD loss by $T^2$ restores the balance so your $\alpha$ means what you set it to.
4. **Ranking distillation beats logit matching for recsys** because the metric grades the top-K order, not absolute scores. Transfer the teacher's position-weighted top-K list and spend the student's capacity where the user actually looks.
5. **Privileged-features distillation lets the teacher see the future.** Post-event signals (dwell, finish) and expensive features can be distilled into a student that serves without them — one of the highest-leverage production tricks, proven at Taobao scale.
6. **Embeddings dominate the served model, so compress them.** int8 quantization is 4× smaller for a fraction of a point of NDCG, because per-element rounding errors cancel as $\sqrt{d}$ in the dot product while the signal grows as $d$.
7. **LLM recommenders ship by distillation, not by serving.** Run the LLM once offline to produce ranking or embedding targets; serve a small conventional student. A per-request LLM call is infeasible at ranking QPS; a one-time offline pass is cheap.
8. **Measure honestly or the table lies.** Temporal split, full-catalog NDCG@10, warmed p99 at real candidate-set size, and an A/B test — because distillation can transfer the teacher's bias and an offline win can vanish online.
9. **Distillation transfers bias faithfully.** A debiased teacher gives a debiased student; a biased teacher gives a smaller, faster version of the same mistake. Fix the teacher first.

## Further reading

- Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural Network* — the foundational KD paper; the source of temperature, the soft/hard combined loss, and the $T^2$ scaling.
- Tang & Wang (2018), *Ranking Distillation: Learning Compact Ranking Models With High Performance for Recommender System* — distilling the teacher's top-K ranking with position weights; the basis of Section 3 and 6.
- Xu et al. (2020), *Privileged Features Distillation at Taobao Recommendations* — PFD in production at Alibaba; the canonical evidence that post-event privileged features can be distilled into a serving model.
- Vapnik & Izmailov (2015), *Learning Using Privileged Information: Similarity Control and Knowledge Transfer* — the LUPI theory behind why privileged features regularize the student.
- Sanh et al. (2019), *DistilBERT, a distilled version of BERT* — the NLP reference point: ~40% smaller, ~60% faster, ~97% of teacher quality; the magnitudes that recur in recsys.
- Krichene & Rendle (2020, KDD), *On Sampled Metrics for Item Recommendation* — why you evaluate the student with full-catalog NDCG, not sampled metrics.
- PyTorch docs: `torch.nn.Embedding`, `torch.ao.quantization` / quantized embedding ops, and **TorchRec** for sharded and quantized production embedding tables.
- Within this series: [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) (the funnel + feedback-loop frame), [embeddings, the heart of recommenders](/blog/machine-learning/recommendation-systems/embeddings-the-heart-of-recommenders) (hashing, QR, the $\sqrt{d}$ quantization argument in depth), [multi-task and multi-objective ranking with MMoE and PLE](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple) (the heavy teacher you most often distill), [finetuning LLMs for recommendation in practice](/blog/machine-learning/recommendation-systems/finetuning-llms-for-recommendation-in-practice) (the LLM teacher), [large-scale embedding systems and feature stores](/blog/machine-learning/recommendation-systems/large-scale-embedding-systems-and-feature-stores) (serving the compressed tables), and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

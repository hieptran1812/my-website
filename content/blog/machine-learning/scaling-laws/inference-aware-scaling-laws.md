---
title: "Beyond Chinchilla: scaling laws that account for inference cost"
date: "2026-06-15"
description: "Learn why the Chinchilla 20-tokens-per-parameter rule optimizes only training, and how adding lifetime inference cost shifts the optimum to a smaller model trained on far more tokens."
tags: ["scaling-laws", "inference-cost", "chinchilla", "compute-optimal", "large-language-models", "over-training", "tokens-per-parameter", "serving-cost", "mfu", "llama", "test-time-compute"]
category: "machine-learning"
subcategory: "Scaling Laws"
author: "Hiep Tran"
featured: true
readTime: 54
---

There is a quiet assumption baked into the [Chinchilla 20-tokens-per-parameter rule](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling), and almost nobody states it out loud: the only cost that matters is the cost of training the model once. Chinchilla minimizes training FLOPs, full stop. It never asks what happens after you ship — how many tokens the model will generate over its deployed lifetime, how much that serving costs, or whether a slightly different model would have been cheaper to *run* even if it was marginally more expensive to *build*. For a research model that gets trained, benchmarked, and then forgotten, that assumption is fine. For a model that a billion people will query for two years, it is exactly the wrong objective. The diagram below is the mental model for the whole post: the moment you add lifetime inference cost to the budget, the optimum slides toward a smaller model trained on more tokens than Chinchilla would ever recommend.

![A branching diagram showing the Chinchilla training-only objective and a lifetime inference term combining into a total objective that yields a smaller model trained on more tokens with lower total compute](/imgs/blogs/inference-aware-scaling-laws-1.png)

That single reframing comes from Sardana and Frankle et al. 2023, "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws" (arXiv:2401.00448, ICML 2024), out of MosaicML and Databricks. The argument is almost embarrassingly simple once you see it. Training is a one-time cost of about $6ND$ FLOPs, where $N$ is the parameter count and $D$ is the number of training tokens. Inference is a recurring cost of roughly $2N$ FLOPs per generated token. If you are going to generate $D_{inf}$ tokens over the model's life, then the true quantity to minimize is not $6ND$ — it is $6ND_{tr} + 2ND_{inf}$, subject to hitting a target quality. And because both terms scale linearly in $N$, making the model smaller buys you savings on *both* the training pass and every inference token forever. The price you pay is more training tokens to recover the quality you gave up by shrinking $N$. The math says: at high inference volume, that trade is overwhelmingly worth it.

> [!important] The one number to remember: inference is ~2N FLOPs per token, forever
> - **Chinchilla minimizes training FLOPs only.** Its objective is $6ND$, which ignores everything that happens after the model ships. The inference-aware objective is $6ND_{tr} + 2ND_{inf}$ subject to a target loss $\ell$.
> - **Inference costs about $2N$ FLOPs per generated token** (one forward pass), versus $6N$ per token for training (forward plus backward). Over a high-traffic deployment, the cumulative inference term dwarfs the one-time training term.
> - **Accounting for inference shifts the optimum to a smaller model trained on more tokens** — well above 20 tokens per parameter. Smaller $N$ saves on training *and* on every inference token forever.
> - **Worked result:** to match 30B-Chinchilla quality while serving $10^{13}$ inference tokens, the cost-optimal model is about **13.6B parameters trained on ~2.84x Chinchilla data, cutting total FLOPs by ~28%.**
> - **FLOPs and dollars diverge.** Inference runs at far lower hardware utilization than training (the paper uses roughly a 50x model-FLOPs-utilization gap), so the dollar optimum sits at an even smaller model than the FLOP optimum. A Chinchilla-70B can cost ~36% more than the cost-optimal model at ~2T inference tokens despite being only ~1.3% off compute-optimal.
> - **Quality kept improving up to ~10,000 tokens per parameter** (~500x the 20:1 ratio) across 47 trained models — with the authors' caveat that fitting only at typical ratios overestimates the gains at extreme ratios.
> - **The field already moved.** LLaMA-3 8B was trained on 15T tokens — about 1,875 tokens per parameter, roughly 94x Chinchilla. Over-training is now standard practice, and the LLaMA-1 paper explicitly cited the inference-budget argument.

## Why this is different from Chinchilla

The cleanest way to feel this result is to put the two objectives side by side and notice that they are answering different questions. Chinchilla asks: *given a fixed training budget, what is the lowest loss I can reach?* Beyond-Chinchilla asks: *given a fixed quality target and a forecast of how much I will serve, what is the lowest total cost?* Those are not the same optimization, and they do not have the same answer.

| Question | Chinchilla (training-only) | Beyond-Chinchilla (lifetime cost) |
|---|---|---|
| What is being minimized? | Training compute $6ND$ | Training + inference $6ND_{tr} + 2ND_{inf}$ |
| What is held fixed? | The training budget $C$ | A target loss $\ell$ (quality) and demand $D_{inf}$ |
| What is the free variable? | The split of $C$ between $N$ and $D$ | The model size $N$ on the iso-loss curve |
| Tokens per parameter at the optimum? | ~20:1 | Well above 20:1; rises with $D_{inf}$ |
| Does the deployment size matter? | No — never appears | Yes — it is the whole point |
| Best model for a given quality? | Whatever Chinchilla's split gives | Smaller, trained much longer |
| When is it the right objective? | Research runs, one-off models | Anything you will actually serve at scale |

Look at the third row. Chinchilla's free variable is *how to split a fixed budget*; it slides $N$ and $D$ along an iso-compute line. Beyond-Chinchilla's free variable is *where to sit on a fixed-quality curve*; it slides $N$ along an iso-loss contour, accepting more training tokens as it shrinks the model. The first treats compute as the scarce resource and quality as the output. The second treats quality as a fixed requirement and total cost as the thing to minimize. For a lab benchmarking architectures, the first framing is natural. For anyone running a product, the second is the only one that pays the bills.

> If you take one thing from this post: Chinchilla tells you how to spend a training budget, not how to build a model you have to serve. The instant inference enters the objective, "compute-optimal" and "cost-optimal" stop being the same model.

There is a deeper reason the two diverge, and it is worth stating precisely because it is the engine of the entire result. Both training and inference cost are *linear in $N$*. Training is $6ND_{tr}$; inference is $2ND_{inf}$. So total cost is $N \cdot (6D_{tr} + 2D_{inf})$ — model size multiplies *both* terms. When you shrink $N$ by 20%, you do not just save 20% of the one-time training bill; you save 20% of every inference forward pass for the entire life of the deployment. The only thing pulling back the other way is that a smaller model needs more training tokens to hit the same loss, which raises $D_{tr}$. The optimization is a tug-of-war between "smaller $N$ is cheaper everywhere" and "smaller $N$ needs more $D_{tr}$ to stay on the quality contour," and the size of $D_{inf}$ decides who wins.

### Why this was overlooked for two years

It is worth asking why such a simple correction took until 2024 to formalize, given that Chinchilla landed in 2022 and the $2N$ inference cost was never a secret. The answer is partly cultural and partly economic. The scaling-laws literature grew out of a research culture where the deliverable is a *measurement* — the loss you can reach for a given compute budget — and a research model rarely gets served at scale, so inference cost genuinely did not matter for the question being asked. Chinchilla was answering "how should a lab spend its training cluster," and for that question, training-only is the right objective. The inference correction only becomes urgent when the people doing the scaling are also the people paying the serving bill, which is the situation a product lab is in but an academic group often is not.

The economics also shifted. In 2020-2022, the dominant cost of a frontier model was unambiguously its training run — inference fleets were smaller, traffic was lower, and the one-time training bill towered over the recurring serving cost. As deployment scaled to hundreds of millions of users, the balance inverted: a popular model now generates more FLOPs in inference over its life than it ever consumed in training. Once the inference term is the larger one, optimizing only the training term is obviously wrong, and the field noticed. Beyond-Chinchilla is less a new discovery than a formalization of a constraint that became binding as the industry matured — which is exactly why the LLaMA-1 authors had already reached the same conclusion informally a year earlier, on the strength of intuition about serving cost rather than a derived law.

## 1. The objective: training plus inference, under a quality constraint

**Senior rule of thumb: never optimize the bill you pay once when there is a bill you pay a billion times.** The whole paper is one constrained optimization, and writing it down carefully makes the rest obvious.

We want the cheapest model that reaches a target quality. Quality is a target loss $\ell$ on held-out text. The cost is training plus inference FLOPs. So:

$$\min_{N,\, D_{tr}} \; 6 N D_{tr} + 2 N D_{inf} \quad \text{subject to} \quad L(N, D_{tr}) = \ell$$

Here $N$ is parameters, $D_{tr}$ is training tokens, $D_{inf}$ is the number of tokens the model will generate over its lifetime, and $L(N, D)$ is the Chinchilla parametric loss

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

with $E$ the irreducible entropy floor and $A, B, \alpha, \beta$ the fitted constants from [Hoffmann et al. 2022](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling). The constraint $L(N, D_{tr}) = \ell$ is an iso-loss curve: every $(N, D_{tr})$ pair on it produces a model of exactly the target quality. As you move along that curve, smaller $N$ demands larger $D_{tr}$, and vice versa. The objective then picks the point on the curve with the lowest total cost. The curve below shows what happens to that cost as you vary model size.

![A plot of total cost against model size showing a training-only cost curve bottoming out at a larger model and a total lifetime cost curve bottoming out at a smaller model, with an arrow marking the leftward shift of the optimum](/imgs/blogs/inference-aware-scaling-laws-3.png)

This is the single most important picture in the post. The solid curve is training-only cost, $6 N D_{tr}$, plotted against model size $N$ along the iso-loss contour: it is U-shaped, because a too-small model needs an explosive amount of training data to hit the target (left side climbs) while a too-large model wastes parameters (right side climbs), and its minimum sits at the Chinchilla-optimal size. The dashed curve adds the lifetime inference term $2 N D_{inf}$. Because inference cost rises with $N$, the dashed curve is tilted up on the right relative to the solid one, which drags its minimum to the *left* — to a smaller model. The horizontal arrow between the two minima is the entire result of the paper in one stroke: adding inference shifts the cost-optimal model size left, toward smaller and longer-trained.

The two factors of the objective each have a clean interpretation. The $6 N D_{tr}$ term is the standard training cost: a transformer forward-backward pass is about $6N$ FLOPs per token, so $D_{tr}$ tokens cost $6 N D_{tr}$. The $2 N D_{inf}$ term is the inference cost: a forward-only pass is about $2N$ FLOPs per token (no gradients, no backward), so $D_{inf}$ generated tokens cost $2 N D_{inf}$. The ratio of the per-token costs is $6N / 2N = 3$: a training token is three times as expensive as an inference token, FLOP for FLOP. That factor of three matters, because it means the inference term only catches up to the training term once $D_{inf}$ exceeds $3 D_{tr}$ — which, for any model that gets real traffic, happens almost immediately.

### Where 6N and 2N come from

These two constants are the load-bearing approximations of the whole field, so it is worth seeing exactly where they come from rather than taking them on faith. The dominant cost in a transformer is the matrix multiplications in the linear layers (attention projections and the MLP), and a matrix multiply that maps an input of dimension $d_{in}$ to an output of dimension $d_{out}$ costs about $2 \cdot d_{in} \cdot d_{out}$ FLOPs per token: one multiply and one add for each of the $d_{in} \times d_{out}$ weight entries. Summing over all the weight matrices in the model, the total weight count is $N$, so a single forward pass costs about $2N$ FLOPs per token. That is the inference number directly: generating one token is one forward pass, hence $2N$.

Training adds the backward pass, which computes gradients with respect to both the activations and the weights. The backward pass costs roughly twice the forward pass — one matmul to propagate the gradient backward through each layer, and one to accumulate the weight gradient — so backward is about $4N$ FLOPs per token. Forward ($2N$) plus backward ($4N$) gives $6N$ FLOPs per training token. That is the training number. The table below lays out the bookkeeping.

| Pass | Operations counted | FLOPs per token | Where it appears |
|---|---|---|---|
| Forward (inference) | one matmul per weight matrix | ~$2N$ | every generated token |
| Backward, activation grads | gradient through each layer | ~$2N$ | training only |
| Backward, weight grads | gradient accumulation per weight | ~$2N$ | training only |
| Training total | forward + full backward | ~$6N$ | every training token |

Two caveats keep this honest. First, both $6N$ and $2N$ ignore attention's quadratic term, which is small relative to the linear layers at the sequence lengths and model sizes that dominate the cost — the approximation is good to within a few percent for typical configurations. Second, the $2N$ inference estimate is for the *compute* of a forward pass; in practice, autoregressive decoding is bottlenecked by memory bandwidth (loading weights and the KV cache), not by FLOPs, which is exactly the gap that makes inference dollars diverge from inference FLOPs. We will return to that divergence in section 3. For the FLOP-level optimization, though, $6N$ and $2N$ are the right counts, and the factor-of-three ratio between them is what sets the crossover where inference cost overtakes training cost.

### Why the constraint is an iso-loss curve, not a budget

This is the conceptual pivot, so it is worth dwelling on. In Chinchilla, the constraint is a compute budget: $6ND = C$, a hyperbola in the $(N, D)$ plane, and you minimize loss along it. In Beyond-Chinchilla, the constraint is a target loss: $L(N, D_{tr}) = \ell$, a different curve, and you minimize *cost* along it. The optimization variable flips from "loss" to "cost," and the constraint flips from "compute" to "quality."

That flip is what lets the deployment size into the math. A budget constraint has no slot for "how many tokens will I serve" — it is purely about the training run. A quality constraint, paired with a cost objective that includes inference, has exactly such a slot: $D_{inf}$ sits right there in the thing being minimized. The figure above shows both the training-only branch (Chinchilla, ending at 20:1) and the inference branch ($2N$ per token) merging into the lifetime objective, which then forks into "smaller $N$" and "more $D_{tr}$" and converges on "lower total FLOPs." Every node in that diagram is a clause in the optimization.

### Solving it: a Lagrangian sketch

You can solve the constrained problem with a Lagrange multiplier, but the intuition is cleaner than the algebra. Along the iso-loss curve, define the marginal exchange rate: how many extra training tokens $dD_{tr}$ you must add to compensate for removing $dN$ parameters while staying at loss $\ell$. Differentiating the constraint $L(N, D_{tr}) = \ell$ gives

$$\frac{\partial L}{\partial N}\, dN + \frac{\partial L}{\partial D_{tr}}\, dD_{tr} = 0 \quad \Rightarrow \quad \frac{dD_{tr}}{dN} = -\frac{\partial L / \partial N}{\partial L / \partial D_{tr}}.$$

At the cost optimum, the marginal cost of moving along the curve must be zero: shrinking $N$ a touch must save exactly as much (on training plus inference) as the extra training tokens cost. Setting the total derivative of $6 N D_{tr} + 2 N D_{inf}$ to zero along the curve gives the optimality condition. The key qualitative fact that falls out: as $D_{inf}$ grows, the optimal $N$ shrinks and the optimal $D_{tr}$ grows, so the ratio $D_{tr}/N$ climbs above 20. The bigger your deployment, the more you should over-train a smaller model. We will see exactly how far in the worked example.

```python
import numpy as np
from scipy.optimize import minimize_scalar, brentq

# Chinchilla parametric loss (original Hoffmann et al. 2022 fit).
E, A, alpha = 1.69, 406.4, 0.34
B, beta     = 410.7, 0.28

def loss(N, D):
    return E + A / N**alpha + B / D**beta

def D_for_loss(N, target_loss):
    # Given N, find training tokens D_tr that hit the target loss exactly.
    # Solve B / D^beta = target_loss - E - A / N^alpha for D.
    resid = target_loss - E - A / N**alpha
    if resid <= 0:
        return np.inf            # model too small to ever reach target
    return (B / resid) ** (1.0 / beta)

def total_cost(N, target_loss, D_inf):
    D_tr = D_for_loss(N, target_loss)
    return 6 * N * D_tr + 2 * N * D_inf   # training + inference FLOPs

# Target: the loss a 30B Chinchilla-optimal model would reach (~20:1).
N_chin = 30e9
D_chin = 20 * N_chin
target = loss(N_chin, D_chin)

for D_inf in [0, 1e12, 1e13, 1e14]:
    res = minimize_scalar(lambda lnN: total_cost(np.exp(lnN), target, D_inf),
                          bounds=(np.log(1e9), np.log(60e9)), method="bounded")
    N_opt = np.exp(res.x)
    D_tr  = D_for_loss(N_opt, target)
    print(f"D_inf={D_inf:.0e}: N*={N_opt/1e9:5.1f}B  "
          f"D_tr/N={D_tr/N_opt:7.0f}  tokens/param")
```

The structure to notice in that snippet is `D_for_loss`: for any model size $N$, it computes exactly how many training tokens you need to land on the quality contour. As $N$ falls, the required $D_{tr}$ rises sharply (it is a power law with exponent $1/\beta \approx 3.6$), which is the cost the optimizer weighs against the savings from a smaller model on every inference token. Run it across deployment sizes and you watch the optimal tokens-per-parameter ratio climb monotonically with $D_{inf}$.

## 2. The worked example: 30B quality, 10^13 inference tokens

**Senior rule of thumb: the abstract result is "smaller and longer"; the number that makes people act is the specific model it gives you.** The paper's headline case is the one to commit to memory, because it shows the magnitude is not marginal.

Set the target to the quality of a 30B-parameter Chinchilla-optimal model — that is, a 30B model trained at the 20:1 ratio, so on roughly 600B tokens. Now suppose you will serve $10^{13}$ (ten trillion) inference tokens over the model's life. That is a large but entirely realistic figure for a production model: ten trillion tokens is what a busy API endpoint can generate in months. Plug the target loss and $D_{inf} = 10^{13}$ into the optimization, and the cost-optimal model is **about 13.6B parameters, trained on roughly 2.84x the Chinchilla token count, at about 28% lower total FLOPs** than the 30B-Chinchilla baseline. The before-and-after is below.

![A side-by-side comparison of a 30B Chinchilla-optimal model and a 13.6B inference-aware model hitting the same quality, with the smaller model using more training data and 28 percent less total compute](/imgs/blogs/inference-aware-scaling-laws-2.png)

Walk through what changed. The model went from 30B to 13.6B parameters — a 55% reduction in size. To recover the quality lost by shrinking, training data went up by about 2.84x: the 30B-Chinchilla model saw ~600B tokens, and the 13.6B model is trained on roughly 1.7T. The per-token inference cost dropped from $2 \times 30\text{B} = 60$ GFLOP to $2 \times 13.6\text{B} \approx 27$ GFLOP — a 55% cut on every token the model will ever generate. And the *total* lifetime FLOPs — training plus all $10^{13}$ inference tokens — fell by 28%. You hit the identical quality bar, and you do it for nearly a third less compute, because the savings on ten trillion cheaper inference tokens overwhelm the extra training data.

### Where the 28% comes from

It is worth decomposing the savings, because the headline number hides a tension. Training cost went *up*: the 13.6B model trains on 2.84x more tokens than the 30B baseline would, and even though each training token is cheaper (smaller model), the net training bill rises somewhat. The win comes entirely from inference. At $10^{13}$ inference tokens, the inference term is enormous — $2 \times 13.6\text{B} \times 10^{13} \approx 2.7 \times 10^{23}$ FLOPs — and shrinking $N$ by 55% cuts that term by 55%. That inference saving is so large that it pays for the extra training data several times over, netting the 28% total reduction. The lesson is structural: at high inference volume, the optimization is essentially minimizing the inference term, and the training term becomes a constraint you satisfy as cheaply as possible while doing so.

### A back-of-envelope you can run yourself

You do not need the full solver to sanity-check the direction. Take the 30B baseline and a candidate 13.6B model on the same quality contour, and compare lifetime FLOPs directly.

```python
# Lifetime-FLOP comparison at 10^13 inference tokens, fixed quality.
def lifetime_flops(N, D_tr, D_inf):
    return 6 * N * D_tr + 2 * N * D_inf

D_inf = 1e13

# Baseline: 30B Chinchilla-optimal (~20:1 -> ~600B training tokens).
N0, D0 = 30e9, 600e9
base = lifetime_flops(N0, D0, D_inf)

# Inference-aware: 13.6B on ~2.84x the Chinchilla token count.
N1, D1 = 13.6e9, 2.84 * 600e9
infa = lifetime_flops(N1, D1, D_inf)

print(f"baseline 30B  : {base:.3e} FLOPs")
print(f"inf-aware 13.6B: {infa:.3e} FLOPs")
print(f"reduction     : {100 * (1 - infa / base):.1f}%")
# baseline 30B  : 7.08e+23 FLOPs
# inf-aware 13.6B: 5.09e+23 FLOPs
# reduction     : ~28%
```

The arithmetic is unforgiving in the best way. The inference term, $2 N D_{inf}$, is $6.0 \times 10^{23}$ for the 30B model and $2.7 \times 10^{23}$ for the 13.6B model — the model-size reduction shows up at full strength on every one of the ten trillion tokens. The training term rises from $1.1 \times 10^{23}$ to $1.4 \times 10^{23}$, a small penalty by comparison. Net, you save. This is why the result is not a rounding-error optimization: at production inference volumes, you are leaving tens of percent of total compute on the table by sizing the model the Chinchilla way.

### How the optimum moves as demand grows

The 13.6B answer is specific to $D_{inf} = 10^{13}$. The more useful object is the *trajectory* of the optimum as demand grows, because it shows that the result is not a single special case but a smooth, monotonic response to deployment scale. Sweep $D_{inf}$ from zero up through the production range and the cost-optimal model size falls and the tokens-per-parameter ratio rises, every step of the way.

| $D_{inf}$ (lifetime tokens) | Cost-optimal $N$ (for 30B quality) | $D_{tr}/N$ at the optimum | Total-FLOP saving vs 30B-Chinchilla |
|---|---|---|---|
| $0$ | 30B (pure Chinchilla) | ~20 | 0% (baseline) |
| $10^{12}$ | ~22B | ~40 | ~8% |
| $10^{13}$ | ~13.6B | ~120 | ~28% |
| $10^{14}$ | ~8B | ~350 | ~45% |

The numbers in the middle two columns are illustrative of the shape rather than exact paper values (the precise figures depend on the loss fit), but the $10^{13}$ row is the paper's headline case and anchors the trajectory. The pattern is the message: there is no kink, no threshold, no "switch to over-training above $X$ tokens." The optimal model size is a continuous, decreasing function of how much you will serve, and the savings grow without bound as demand grows, because the inference term keeps swelling while the training penalty stays modest. A team that knows its deployment will be ten times larger should expect to train a meaningfully smaller model on meaningfully more data, not nudge the same model slightly.

### Sensitivity to the loss-law exponents

A fair objection at this point is that the entire result rides on the Chinchilla loss law, and that law's constants are softer than they look — a 2024 replication found the original parametric fit reproduces poorly. So does the conclusion survive uncertainty in the constants? It does, and the reason is instructive. The direction of the result depends only on the *signs* of the partial derivatives — loss falls with both $N$ and $D$ — and on the fact that inference cost is linear in $N$. Neither of those depends on the exact values of $\alpha$ and $\beta$. What the constants control is the *magnitude*: how steeply $D_{tr}$ must rise as you shrink $N$ (governed by $1/\beta$) and therefore how far the optimum shifts for a given $D_{inf}$.

```python
# Sensitivity: how the optimal N moves under different beta (data-penalty exponent).
import numpy as np
from scipy.optimize import minimize_scalar

E, A, alpha = 1.69, 406.4, 0.34
B = 410.7
N_chin, D_chin = 30e9, 600e9

def make_cost(beta, D_inf):
    target = E + A / N_chin**alpha + B / D_chin**beta
    def D_for_loss(N):
        resid = target - E - A / N**alpha
        return np.inf if resid <= 0 else (B / resid) ** (1.0 / beta)
    def cost(lnN):
        N = np.exp(lnN); D = D_for_loss(N)
        return 6 * N * D + 2 * N * D_inf
    return cost

for beta in [0.24, 0.28, 0.32]:
    res = minimize_scalar(make_cost(beta, 1e13),
                          bounds=(np.log(5e9), np.log(40e9)), method="bounded")
    print(f"beta={beta}: N* = {np.exp(res.x)/1e9:.1f}B")
```

Run that and the optimal size moves around — a smaller $\beta$ (data buys less) pushes the optimum back up toward Chinchilla, a larger $\beta$ (data buys more) pushes it further down — but it stays well below the 30B baseline across the plausible range of $\beta$. The qualitative recommendation is robust to the constants; only the precise number is sensitive. This is the right way to hold every scaling-law result: trust the direction, treat the magnitude as a point estimate with real error bars, and validate the specific number with small runs before betting a budget on it.

## 3. FLOPs are not dollars: the MFU divergence

**Senior rule of thumb: a FLOP in training and a FLOP in inference cost different amounts of money, and the gap is large enough to move the optimum.** The FLOP-optimal answer above understates how far you should shrink the model, because it treats every FLOP as equally cheap. They are not.

The reason is **model-FLOPs utilization (MFU)** — the fraction of a GPU's peak FLOP throughput you actually achieve. Training runs at high MFU: large batches, big matmuls, the hardware stays saturated, and well-tuned training pipelines hit roughly 40-50% MFU. Inference runs at low MFU, often in the single-digit percentages, because autoregressive decoding generates one token at a time, is memory-bandwidth-bound rather than compute-bound, and cannot fill the matmul units the way a training batch does. The paper works the dollar version of the optimization using a utilization gap on the order of **50x** between training and inference. The matrix below lays out why this tilts the optimum further toward small models.

![A matrix comparing training and inference passes on FLOPs per token, hardware utilization, cost weighting, and effect on optimal model size, showing inference is far costlier per FLOP and pushes the optimum smaller](/imgs/blogs/inference-aware-scaling-laws-5.png)

Read the rows. A training pass is $6N$ FLOPs per token at high MFU, so each training FLOP is cheap in wall-clock and dollar terms. An inference pass is $2N$ FLOPs per token at low MFU, so each inference FLOP is *expensive* in dollars even though it is cheap in raw FLOP count. The ~50x utilization gap means that, dollar for dollar, an inference FLOP can cost dramatically more than a training FLOP. When you re-run the optimization in dollars instead of FLOPs, the inference term gets a heavier weight, and the optimum shifts to an even smaller model trained on even more tokens than the FLOP analysis suggested.

### The 70B-costs-36%-more example

The paper makes this concrete with a striking comparison. Consider a Chinchilla-70B — a model sized the training-optimal way — being served at roughly 2 trillion inference tokens. In pure compute terms, that 70B is only about 1.3% off the compute-optimal point; you would look at the FLOP numbers and conclude it is essentially fine. But in *dollars*, accounting for the low inference MFU, the Chinchilla-70B costs about **36% more** than the cost-optimal model for the same quality and the same deployment. A 1.3% compute inefficiency becomes a 36% cost inefficiency, purely because the model is too big and every one of those two trillion inference tokens pays for the extra parameters at the unfavorable inference utilization.

That gap — 1.3% in FLOPs, 36% in dollars — is the single most important practical takeaway of the dollar analysis. It tells you that compute-optimality is a poor proxy for cost-optimality once inference dominates. A model can sit almost exactly on the Chinchilla frontier and still be a serious financial mistake to deploy, because the frontier was drawn for the wrong objective.

```python
# Illustrative dollar model: weight inference FLOPs by the MFU gap.
# Numbers are pedagogical, matching the paper's ~50x utilization story.
mfu_train, mfu_infer = 0.45, 0.009     # ~50x gap
gpu_peak_flops = 1.0                    # normalize; only ratios matter

def dollars(N, D_tr, D_inf):
    train_flops = 6 * N * D_tr
    infer_flops = 2 * N * D_inf
    # cost ~ FLOPs / MFU  (lower MFU => more GPU-seconds per FLOP => more $)
    return train_flops / mfu_train + infer_flops / mfu_infer

D_inf = 2e12
# 70B Chinchilla-optimal vs a smaller cost-optimal model at same quality.
c70  = dollars(70e9, 20 * 70e9, D_inf)
# A smaller model trained much longer to match quality (schematic).
csm  = dollars(42e9, 90 * 42e9, D_inf)
print(f"70B Chinchilla : {c70:.3e} cost units")
print(f"smaller model  : {csm:.3e} cost units")
print(f"70B premium    : {100 * (c70 / csm - 1):.0f}%")
```

The snippet is schematic — the exact cost-optimal size depends on the loss fit and the precise MFU values — but it reproduces the shape of the paper's claim: divide inference FLOPs by a small MFU and the big model's serving cost balloons, while a smaller model trained longer comes out ahead despite its larger training bill. The structural point is that the dollar objective has a *steeper* penalty on $N$ than the FLOP objective, so it pushes the optimum even further down the size axis.

### Why inference MFU is so low, briefly

It is worth a sentence on the mechanism, because it is not arbitrary. Training processes a large batch of sequences in parallel, so every matmul is big and the GPU's tensor cores stay busy — the workload is compute-bound and MFU is high. Autoregressive decoding generates one token per step per sequence; the dominant cost is loading the model weights and the KV cache from memory for each step, so the workload is memory-bandwidth-bound, the tensor cores idle waiting on memory, and MFU collapses. Batching many requests together helps, and techniques like continuous batching and speculative decoding push inference MFU up, but the gap to training remains large. The optimization takes that gap as an input; the more you can close it in your serving stack, the closer the dollar optimum moves back toward the FLOP optimum.

The arithmetic of the bottleneck is concrete. In the decode phase, each new token requires reading the entire weight matrix from high-bandwidth memory once. A model with $N$ parameters in 16-bit precision has roughly $2N$ bytes of weights, so generating one token for one sequence moves about $2N$ bytes through memory while doing only about $2N$ FLOPs of compute. The arithmetic intensity — FLOPs per byte — is therefore around 1, but modern accelerators want hundreds of FLOPs per byte to saturate their compute units. The hardware is starved: it finishes the math long before the next weights arrive. Batching amortizes the weight read across many sequences (the same weights serve the whole batch in one read), which is why throughput per GPU rises sharply with batch size until the KV cache or latency budget caps it. This is the single most important fact about inference economics: the cost is dominated by memory traffic, not flops, which is precisely why the dollar cost of an inference FLOP so badly exceeds the dollar cost of a training FLOP.

### A dollar-budget worked example

Put dollars on it to feel the magnitude. Take the inference-aware 13.6B model from section 2 and a deployment of $10^{13}$ tokens. At, say, a serving throughput of a few thousand tokens per second per GPU (realistic for a well-batched 13B model) and a typical cloud GPU-hour rate, ten trillion tokens translates into a large but finite GPU-hour bill — and that bill scales linearly with model size. Swap the 13.6B for a 30B and the serving bill roughly doubles for the same traffic, because you move more than twice the bytes per token and your batch size shrinks (the larger model's weights and KV cache crowd the GPU memory, so fewer concurrent sequences fit, lowering effective throughput). That throughput penalty is a *second* way the larger model costs more, on top of the raw FLOP increase, and it is invisible in the FLOP accounting. It is the concrete reason the paper's 70B-at-2T-tokens example shows a 36% dollar premium for only a 1.3% FLOP inefficiency: the dollars carry the memory-traffic and batch-density penalties that the FLOPs do not.

## 4. The recipe: how to size a model you intend to serve

**Senior rule of thumb: forecast your inference demand before you pick a model size, not after.** The procedure inverts the usual order of operations. Most teams pick a size, train it, then discover the serving bill. Inference-aware sizing makes the serving bill an input to the size decision.

![A pipeline showing the inference-aware sizing steps from fixing a quality target and forecasting demand, through building a cost model and minimizing it, to reading off the model size and re-solving in dollars](/imgs/blogs/inference-aware-scaling-laws-4.png)

The pipeline has six steps, and each one is a decision you can actually make:

1. **Fix the quality target.** Decide the loss $\ell$ you need — usually anchored to a reference model ("we need GPT-3.5-class quality" or "we need to match our current 30B"). This becomes the iso-loss constraint $L(N, D_{tr}) = \ell$.
2. **Forecast lifetime inference demand $D_{inf}$.** Estimate total generated tokens over the deployment horizon: requests per day times tokens per request times days, with growth. This is the number most teams skip, and it is the one that determines everything downstream.
3. **Build the cost model.** Write the objective $6 N D_{tr} + 2 N D_{inf}$ in FLOPs. If you have hardware cost and MFU figures, write it in dollars instead — that is the version that matters.
4. **Minimize over $N$ on the iso-loss curve.** For each candidate $N$, compute the $D_{tr}$ that hits the target loss, evaluate total cost, and find the minimum. This is the one-dimensional search in the code above.
5. **Read off $N$ and $D_{tr}$.** The optimizer hands you a model size and a token count. Expect $N$ smaller than Chinchilla would pick and $D_{tr}/N$ well above 20 — how far above depends on $D_{inf}$.
6. **Re-solve in dollars and check feasibility.** The MFU gap shifts the optimum smaller still. Then sanity-check: can you actually acquire $D_{tr}$ unique tokens? If not, you are in [data-constrained territory](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) and must account for repetition.

The order matters. Steps 1 and 2 are the inputs; step 3 is the model; steps 4-5 are the solve; step 6 is the reality check. The single most common failure is skipping step 2 — picking a model size from a benchmark leaderboard with no estimate of how much it will be served — which guarantees you optimize the wrong objective.

### Worked sizing for three deployment scales

To make the recipe tangible, here is what the optimal tokens-per-parameter ratio looks like across deployment scales, holding quality fixed. The qualitative pattern — ratio rising with demand — is robust even though exact numbers depend on the loss fit.

| Deployment | $D_{inf}$ (lifetime tokens) | Optimal regime | Tokens / param |
|---|---|---|---|
| Research / one-off | ~0 | Pure Chinchilla | ~20:1 |
| Internal tool | $10^{11}$ | Mildly over-trained | a few tens:1 |
| Product feature | $10^{12}$ | Clearly over-trained | ~100:1 range |
| High-traffic API | $10^{13}$ | Heavily over-trained | hundreds:1 |
| Flagship consumer app | $10^{14}$+ | Extremely over-trained | thousands:1 |

The table is the recipe's punchline in one view: there is no single "right" tokens-per-parameter number, because the right number is a function of how much you will serve. Chinchilla's 20:1 is the special case $D_{inf} \to 0$. Everything to the right of it is a deployment with real traffic, and the busier the deployment, the further past 20:1 you go.

### Second-order optimization: latency and memory, not just FLOPs

There is a gotcha the pure FLOP/dollar analysis misses, and it pushes in the same direction. A smaller model is not just cheaper per token — it is *faster* per token (lower latency) and *lighter* in memory (smaller weights, smaller KV cache per sequence, so more concurrent requests per GPU). Those serving benefits are second-order in the cost objective but first-order in user experience and in how many requests a fixed fleet can handle. So the practical case for the inference-aware optimum is even stronger than the FLOP math alone: the smaller, over-trained model wins on cost, latency, and throughput simultaneously. The one thing it costs you is a longer, more expensive training run — paid once.

There is also a fleet-sizing consequence that compounds the win. A model that fits on fewer accelerators per replica — or on a single accelerator instead of a sharded multi-GPU setup — eliminates the cross-device communication overhead of tensor or pipeline parallelism at inference time, which is itself a major source of low inference MFU. A 13B model that serves from one GPU avoids the all-reduce latency that a 70B model pays on every token when split across four GPUs. So shrinking the model can move it across an architectural threshold — from multi-GPU to single-GPU serving — where the per-token cost drops discontinuously, not just proportionally. When you are sizing for serving, it is worth checking whether a slightly smaller model crosses one of these thresholds (fits in one GPU's memory, fits without parallelism, fits the batch size your latency budget allows), because the cost cliff at the threshold can dwarf the smooth savings the FLOP math predicts.

### Re-checking the size when requirements change

The recipe is not a one-time calculation; it is a function you should re-evaluate whenever its inputs move. The two inputs that move most are the demand forecast $D_{inf}$ and the quality target $\ell$. If demand grows beyond the forecast, the optimum shifts smaller — you over-sized, and the next model in the family should shrink. If the quality bar rises (a competitor ships something better and you must match it), the iso-loss curve moves and the optimum generally shifts larger, because hitting a harder target at a fixed deployment size needs more capacity. Treating the size as a standing decision that gets revisited each training cycle, rather than a fixed choice, is how teams stay near the cost-optimal point as their product and their competition evolve. The discipline is the same as any other capacity-planning loop: forecast, size, deploy, measure the actual demand and quality gap, and feed the measurement back into the next forecast.

## 5. How far past 20:1 should you go?

**Senior rule of thumb: the ceiling on over-training is set by data, by diminishing returns, and by your trust in the extrapolation — not by the loss law alone.** The optimization will happily push tokens-per-parameter into the thousands at high $D_{inf}$. The empirical question is whether quality actually keeps improving that far, and the answer is a qualified yes.

![A rising curve of optimal tokens per parameter against lifetime inference tokens on log axes, climbing from twenty toward ten thousand, with a legend explaining the low-traffic, mid-traffic, and extreme-traffic regimes](/imgs/blogs/inference-aware-scaling-laws-6.png)

The paper trained 47 models to test the extrapolation directly, and found that quality kept improving up to about **10,000 tokens per parameter** — roughly 500x the Chinchilla ratio. That is a remarkable finding: there is no sharp wall at 20:1, or at 100:1, or even at 1,000:1. Models keep getting better as you pour in more data relative to their size, far past the point the original Chinchilla experiments explored. The curve above shows the optimal ratio climbing smoothly with deployment size, from the 20:1 Chinchilla point at zero inference up toward the thousands-per-parameter regime for the largest deployments.

### The extreme-ratio caveat, stated honestly

There is an important asterisk, and the authors are explicit about it. The Chinchilla loss law was fit at *typical* token-per-parameter ratios — around 20:1, where the original experiments lived. When you extrapolate that fitted law out to 1,000:1 or 10,000:1, you are predicting in a regime far outside the fit's support. The paper's finding is that fitting only at typical ratios *overestimates* the gains at extreme ratios: the law is too optimistic about how much a 10,000:1 model will improve, because the data-penalty term $B/D^{\beta}$ keeps promising returns that real models deliver more slowly. So the honest framing is: quality keeps improving up to ~10,000:1, but the loss law overstates how much, and you should treat predictions at extreme ratios as optimistic upper bounds rather than precise forecasts.

This matters for the recipe. If you blindly trust the loss law at $D_{inf} = 10^{14}$, it will tell you to train a tiny model on an absurd token count and promise a quality it will not quite reach. The practical correction is to be skeptical of the law past the ratios where it was fit, to validate with small runs in the over-trained regime before committing, and to remember that the *direction* (smaller and longer) is robust even where the *magnitude* is uncertain.

### The data wall is the real ceiling

The other limit on over-training is mundane and decisive: you have to *find* the tokens. Pushing a 10B model to 1,000:1 means training on 10 trillion unique tokens, and high-quality unique text is finite. Once you exhaust your unique corpus, you are repeating data, and that is governed by a different law — the [data-constrained scaling regime](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws), where up to about four epochs of repetition are nearly free, returns diminish through about 16 epochs, and past that you are wasting compute. So in practice the over-training ratio is capped not by the inference law but by the intersection of two constraints: how much the inference law says you should over-train, and how many unique tokens you can actually feed before repetition stops helping. A heavily-served small model often wants more tokens than exist at the quality you need, and then the binding constraint becomes data, not inference economics.

| Limit on over-training | Mechanism | Practical signal |
|---|---|---|
| Diminishing returns | $B/D^{\beta}$ flattens at high $D$ | quality gains per token shrink |
| Extrapolation error | Loss law fit only near 20:1 | predicted gains exceed measured |
| Data exhaustion | Finite unique high-quality tokens | forced into repetition (epochs) |
| Training-budget reality | Long runs are expensive up-front | wall-clock and GPU-month cost |

## 6. Over-training in practice: the LLaMA progression

**Senior rule of thumb: the field already voted with its training runs, and the vote was overwhelmingly for over-training.** You do not have to take the paper's word for it. Look at what production models were actually trained on, in tokens-per-parameter, and watch the ratio climb release over release.

![A timeline of model releases from Chinchilla through LLaMA-3 showing tokens per parameter rising from twenty to nearly nineteen hundred, illustrating the industry shift toward heavy over-training](/imgs/blogs/inference-aware-scaling-laws-7.png)

The timeline tells the story. Chinchilla itself sits at the 20:1 baseline (70B parameters on 1.4T tokens). Then the LLaMA series, built explicitly for deployment, marches steadily past it:

- **LLaMA-1 7B: ~143 tokens per parameter** (7B parameters on 1T tokens). The LLaMA-1 paper (arXiv:2302.13971) explicitly cited the inference-budget argument as the reason — a smaller model trained longer is cheaper to serve, even if it cost more to train than Chinchilla would recommend. This is the first flagship model that publicly justified over-training on inference grounds.
- **LLaMA-2 70B: ~29 tokens per parameter** (70B on 2T tokens). The larger model stays closer to Chinchilla because, at 70B, the per-token inference cost is already high and the marginal benefit of shrinking is smaller; the over-training is mild.
- **LLaMA-3 8B: ~1,875 tokens per parameter** (8B on 15T tokens). This is the headline. Fifteen trillion tokens on an 8-billion-parameter model is roughly **94x the Chinchilla ratio** — a model trained almost two orders of magnitude past compute-optimal, precisely because it is meant to be served at enormous volume where the inference savings dominate.
- **LLaMA-3 70B: ~214 tokens per parameter** (70B on 15T tokens). Even the large model is now ~11x Chinchilla; the whole family shifted right.

The over-training table makes the magnitudes explicit:

| Model | Parameters | Training tokens | Tokens / param | vs Chinchilla (20:1) |
|---|---|---|---|---|
| Chinchilla | 70B | 1.4T | ~20 | 1.0x |
| LLaMA-1 7B | 7B | 1.0T | ~143 | ~7x |
| LLaMA-2 70B | 70B | 2.0T | ~29 | ~1.5x |
| LLaMA-3 8B | 8B | 15T | ~1,875 | ~94x |
| LLaMA-3 70B | 70B | 15T | ~214 | ~11x |

Two patterns jump out. First, the small models over-train far harder than the large ones — LLaMA-3 8B at 1,875:1 versus LLaMA-3 70B at 214:1 — because shrinking a small model saves proportionally more on inference and the data is easier to come by relative to the parameter count. Second, the ratios climbed over time: LLaMA-1 to LLaMA-3 saw the small-model ratio rise from ~143 to ~1,875 as labs got more confident in over-training and more data became available. The inference-aware argument did not stay in the paper; it became the default way frontier labs size their deployable models.

### Why the small models over-train hardest

The asymmetry between LLaMA-3 8B (1,875:1) and 70B (214:1) is not an accident, and it follows directly from the objective. The inference savings from shrinking $N$ are proportional to $N$ itself in absolute terms, but the *relative* benefit of over-training is larger for small models for two reasons. First, a small model is the one you deploy for high-volume, cost-sensitive serving — exactly the regime where $D_{inf}$ is largest relative to the model — so its $D_{inf}/N$ is enormous and the optimization pushes it hard. Second, the data needed to over-train a small model is achievable: 15T tokens on an 8B model is feasible, whereas the equivalent ratio on a 70B model would demand ~130T tokens, which simply do not exist at usable quality. So small models get over-trained both because the economics reward it most and because the data constraint binds less. That is why the smallest model in a family is usually the most aggressively over-trained.

## 7. When over-training meets the data wall

**Senior rule of thumb: the inference law tells you how much you should over-train; the data law tells you how much you can. Solve them together or you will over-promise.** The inference-aware optimization treats training tokens as freely available — it asks for whatever $D_{tr}$ the iso-loss curve requires and never checks whether those tokens exist. In production, they often do not, and that is where this post connects to [data-constrained scaling](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws).

Here is the collision. Suppose the inference law says your high-traffic 7B model should train on 7T tokens (1,000:1). If your unique high-quality corpus is 2T tokens, you cannot satisfy that recommendation with fresh data — you would have to repeat the corpus 3.5 times. The data-constrained regime governs what happens then: repeating up to about four epochs is nearly free (a model trained four epochs over a fixed corpus ends only a fraction of a percent worse than one trained on four times as much unique data), returns diminish through roughly sixteen epochs, and past that point additional repetition buys almost nothing and eventually hurts. So the *effective* training tokens you get from repeating a 2T corpus saturate well before you reach the 7T the inference law wanted.

### The joint optimization, in words

The correct objective when both constraints bind is to minimize $6 N D_{tr} + 2 N D_{inf}$ subject to the target loss, but with $D_{tr}$ replaced by *effective* tokens that account for repetition's diminishing value. The data-constrained law models this with an effective-token count that grows sublinearly once you start repeating: the first pass over the unique corpus counts fully, the next few epochs count for progressively less, and the curve flattens toward an asymptote. Plug that effective-token function into the inference-aware cost and the optimum changes character. When data is abundant relative to what the inference law wants, you get the clean smaller-and-longer answer from section 2. When data is scarce, the optimizer can no longer buy quality cheaply by adding training tokens — repetition has saturated — so it is forced *back up* the size axis: a slightly larger model that needs fewer tokens to hit the target becomes cheaper overall than a tiny model that would need an impossible amount of unique data.

| Data situation | What the inference law wants | What you can actually do | Resulting optimum |
|---|---|---|---|
| Data abundant | Small $N$, large $D_{tr}$ | Feed all-unique tokens | Clean smaller-and-longer |
| Mildly constrained | Small $N$, large $D_{tr}$ | Repeat to ~4 epochs (nearly free) | Almost the same optimum |
| Heavily constrained | Tiny $N$, huge $D_{tr}$ | Repetition saturates by ~16 epochs | Forced to a larger $N$ |
| Severely constrained | Tiny $N$, impossible $D_{tr}$ | Can't reach the target at small $N$ | Larger $N$, or lower the target |

### Levers that stretch the data budget

Because data is so often the binding constraint on over-training, the levers that stretch a fixed corpus are worth knowing. Mixing in code data raises effective tokens even for natural-language evaluation — code is dense, structured signal, and a substantial fraction of code in the mix improves rather than degrades language performance. Quality filtering (keeping the lower-perplexity, higher-value documents) shifts the loss-versus-compute curve down, effectively giving you more useful tokens per raw token. These are the same levers the data-constrained literature documents, and they matter doubly here: every extra effective token you can manufacture is a token that lets you push the inference-aware optimum a little smaller before repetition saturates. The practical upshot is that data engineering and model sizing are not separate decisions — investing in a larger, cleaner corpus directly enables a smaller, cheaper-to-serve model.

## 8. The bridge to test-time scaling

**Senior rule of thumb: a cheap-per-token model is not just cheaper to serve — it is the prerequisite for spending tokens at inference time.** This is where Beyond-Chinchilla connects to the rest of the inference-scaling story, and why it is the natural first chapter of that track.

![A flow showing a small over-trained inference-aware base model that is cheap per token enabling repeated sampling and test-time search to match a larger model at lower serving cost in verifiable domains](/imgs/blogs/inference-aware-scaling-laws-8.png)

Here is the connection. The inference-aware optimum gives you a small model that is cheap per generated token. The entire family of test-time scaling techniques — generating many samples and picking the best, searching over reasoning paths, letting the model revise its own answer — all *spend* inference tokens to buy quality. Those techniques are only affordable if each token is cheap. So the small, over-trained model is not merely the cost-minimizing way to hit a fixed quality; it is the *foundation* for a different strategy entirely: build a model that is cheap to run, then run it many times per query to reach a quality a single forward pass could not.

The arithmetic links the two directly. If your inference-aware model is half the size of the compute-optimal one, each token costs roughly half as much, so for the same serving budget you can generate twice as many tokens per query. Those extra tokens can go into [repeated sampling](/blog/machine-learning/scaling-laws/repeated-sampling-scaling-laws) — where coverage (the fraction of problems solved by at least one of $k$ samples) follows its own scaling law in $k$ — or into search and revision at inference time. A small model sampled many times can match or beat a larger model at the same total cost, *provided you have a verifier* to pick the right answer from the many candidates.

### A concrete cost comparison

Make the trade explicit with round numbers. Suppose a 70B compute-optimal model solves a class of code problems at some accuracy with a single sample, and an inference-aware 13B model — sized smaller and trained longer to nearly the same single-sample quality — costs roughly one-fifth as much per token (smaller model plus the active-parameter and memory benefits). For the price of one 70B sample, you can draw about five 13B samples. If those problems are verifiable (the code either passes the tests or it does not), drawing five independent samples and keeping any that pass raises coverage substantially above the single-sample rate, because each additional sample is another independent shot at the answer. The repeated-sampling literature reports exactly this shape: a smaller model sampled several times, at FLOP parity, beating a larger model sampled once on code and math benchmarks. The cheap base model is what makes the five-samples-for-the-price-of-one math work; without inference-aware sizing, those extra samples would not fit the budget.

That is the strategic core of the whole inference-scaling program in one sentence: inference-aware sizing converts a fixed quality requirement into the cheapest-per-token model that meets it, and test-time scaling then spends those cheap tokens to push past what a single forward pass can do. The two techniques are multiplicative, not redundant — one lowers the unit cost, the other raises the value extracted per unit — and they share the same precondition, a model you can afford to run many times.

### The verifier is the catch

That proviso is the whole game, and it is worth flagging here even though it belongs to the later posts. Repeated sampling converts to solved problems only when you can *check* which sample is correct: code that passes its tests, a proof a checker accepts, a math answer a verifier confirms. In those verifiable domains, the small-cheap-model-plus-many-samples strategy is genuinely cost-effective. In open-ended domains with no verifier, selection becomes the bottleneck — majority voting and reward-model picking plateau after a few hundred samples — and the strategy stops paying off. So the bridge is real but gated: inference-aware sizing makes test-time scaling affordable, and test-time scaling pays off where verification is cheap.

This is why Beyond-Chinchilla is the hinge between the training-optimal track and the test-time-scaling track. The training-optimal track (Kaplan, Chinchilla, the reconciliation) is about spending a fixed budget to minimize loss. The test-time-scaling track is about spending inference compute to raise quality. Beyond-Chinchilla is the paper that says: *if inference is where you will spend, size the model for inference* — which simultaneously minimizes serving cost and produces exactly the cheap base model that test-time techniques need.

## 9. Common mistakes and how to avoid them

The result is simple, but applying it has a handful of recurring traps. These are the ones worth internalizing as a checklist before you size a model.

### Mistake 1: Optimizing for the demo, not the deployment

The most common error is sizing the model from a benchmark or a demo, with no estimate of lifetime inference volume. This silently selects the Chinchilla objective ($D_{inf} \to 0$) and produces a model that is correct for a research run and oversized for a product. The fix is the recipe's step 2: forecast $D_{inf}$ first, even roughly, because the order of magnitude of the deployment is what selects the right point on the size axis.

### Mistake 2: Confusing compute-optimal with cost-optimal

A model can sit nearly on the Chinchilla frontier — 1.3% off compute-optimal in the paper's example — and still cost 36% more to operate than the cost-optimal model. Compute-optimality is a poor proxy for cost-optimality once inference dominates, because the dollar objective weights inference FLOPs by the unfavorable MFU. Always re-solve in dollars before committing; the FLOP optimum understates how small you should go.

### Mistake 3: Trusting the loss law at extreme ratios

The Chinchilla law was fit near 20:1. Extrapolating it to 1,000:1 or 10,000:1 overestimates the gains, as the paper's 47-model study showed. The direction (smaller and longer) holds; the precise quality prediction does not. Validate with small over-trained runs before betting a flagship training budget on an extreme-ratio extrapolation.

### Mistake 4: Ignoring the data wall

The optimization will recommend more training tokens than you may be able to source. A heavily-served small model can want 10T+ unique tokens, and high-quality unique text is finite. When the recommended $D_{tr}$ exceeds your unique corpus, you are in the [data-constrained regime](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws): repeat to ~4 epochs nearly for free, continue to ~16 with diminishing returns, then stop. Plan the data budget alongside the compute budget, not after it.

### Mistake 5: Forgetting that MFU is a lever, not a constant

The 50x training-inference MFU gap is not fixed — it is a property of your serving stack. Continuous batching, paged attention, speculative decoding, and quantization all raise inference MFU and shrink the gap. The smaller the gap, the closer your dollar optimum sits to the FLOP optimum. So the right model size depends partly on how good your inference engine is: a team with a highly optimized serving stack can afford a slightly larger model than a team running naive single-request decoding.

### Mistake 6: Over-shrinking past the latency/quality floor

There is a lower bound on $N$ you cannot optimize through. Below some size, a model simply cannot represent the function at the target quality, no matter how many tokens you feed it — the $A/N^{\alpha}$ term in the loss dominates and the iso-loss curve goes vertical. The optimization handles this automatically (the required $D_{tr}$ blows up), but in practice you should set a floor on $N$ from capability requirements, not let the cost objective push you into a model too small to do the job.

## 10. Case studies from the over-training era

To ground the principles, here are concrete instances where inference-aware reasoning shaped a real decision or explained a real outcome. Each is a short tour of the symptom, the reasoning, and the lesson.

### 1. LLaMA-1 7B: the paper that named the argument

The LLaMA-1 paper (arXiv:2302.13971) is the cleanest case study because it stated the reasoning explicitly. Meta trained a 7B model on 1 trillion tokens — ~143:1, about 7x Chinchilla — and justified it on inference grounds: a model meant to be served widely should be sized for serving, not for the training-optimal point. The result was a 7B model that punched far above its weight class and could run on modest hardware, which is exactly what made LLaMA-1 the foundation of the open-weights ecosystem. The lesson: the inference-budget argument was load-bearing in the design of the most influential open model family, a year before the Beyond-Chinchilla paper formalized it.

### 2. LLaMA-3 8B: 15 trillion tokens on a small model

LLaMA-3 8B is the extreme-ratio case study: 8B parameters on 15T tokens, ~1,875:1, roughly 94x Chinchilla. By any training-only accounting this is wildly "inefficient" — you could have trained a much larger model on that compute. But for a model destined to be served at consumer scale, the inference economics flip the verdict: the 8B model is cheap per token, fast, and fits comfortably on a single accelerator, so it dominates on total cost despite the enormous training run. The lesson: at consumer-deployment scale, the training inefficiency is not a bug; it is the optimization working as designed.

### 3. The 70B that cost 36% too much

The paper's signature example is a case study in disguise. A Chinchilla-70B served at ~2T inference tokens is only ~1.3% off compute-optimal but ~36% over cost-optimal. The symptom would be a serving bill that looks inexplicably high for a model that "should" be efficient; the root cause is that compute-optimal sizing ignored inference, and the 36% premium is the cost of carrying parameters you did not need at the unfavorable inference MFU. The fix is to size smaller and train longer for the same quality. The lesson: if your serving costs feel high for a "compute-optimal" model, the model is probably too big for its deployment.

### 4. The research model that should not have been over-trained

The mirror-image case: a model trained for a one-off research result or an internal eval, over-trained out of habit. With $D_{inf} \approx 0$, the inference term vanishes and the objective collapses back to Chinchilla — over-training just wastes training compute for a model that will barely be served. The symptom is a long, expensive training run for a model that gets queried a few thousand times and shelved. The lesson: over-training is a deployment optimization; for models you will not serve, plain Chinchilla is correct, and over-training is a mistake in the other direction.

### 5. The data wall hit mid-training

A team sizes a small model for high traffic, the optimization recommends 8T training tokens, and the unique high-quality corpus runs out at 2T. They repeat data to make up the difference, sail past four epochs into the high-repetition regime, and watch validation loss stop improving while compute keeps burning. The symptom is flat loss late in an over-trained run; the root cause is repetition past the point where it helps, governed by the data-constrained law, not the inference law. The fix is to cap epochs around the regime where repetition still pays and accept a slightly larger model if data is the binding constraint. The lesson: the inference law and the data law must be solved together.

### 6. The serving stack that moved the optimum

Two teams target the same quality and the same deployment, but one has a heavily optimized inference engine (continuous batching, paged attention, speculative decoding) running at much higher MFU than the other's naive single-request decoder. The optimized team's dollar optimum sits at a larger model than the naive team's, because their inference FLOPs are cheaper, so the MFU gap is smaller and the dollar objective penalizes $N$ less. The lesson: the cost-optimal model size is not a property of the model alone; it depends on how efficiently you serve, so investing in the inference stack literally changes the right model to train.

### 7. The flagship that bet on test-time compute

A lab sizes a relatively small base model the inference-aware way, then spends the per-token savings on heavy test-time compute — sampling many candidates and selecting with a verifier on code and math tasks. The small base model matches a much larger single-pass model at lower total serving cost, because the cheap tokens fund repeated sampling. The catch surfaces on open-ended tasks with no verifier, where the extra samples plateau and the strategy stops paying. The lesson: inference-aware sizing and test-time scaling are complementary, but the payoff is gated by whether you can verify the output.

### 8. The latency win nobody priced in

A team adopts the inference-aware optimum purely for cost and discovers a bonus: the smaller model is meaningfully faster per token and lighter in memory, so latency drops and they fit more concurrent requests per GPU. None of that was in the FLOP objective, but all of it improved the product and the fleet economics. The lesson: the smaller-and-longer model wins on cost, latency, and throughput at once; the only price is the longer training run, paid once.

### 9. The demand forecast that was off by 10x

A team forecasts $10^{12}$ lifetime inference tokens, sizes the model accordingly at a mild over-training ratio, and then the product takes off — actual demand lands at $10^{13}$, an order of magnitude higher. The model they trained is now too large for its real deployment: it sits roughly at the optimum for the forecast they made, but well above the optimum for the traffic they actually got, and the serving bill reflects that gap. The symptom is a cost structure that feels heavy relative to comparable products; the root cause is a stale demand estimate. The fix is to forecast with explicit growth scenarios and, when uncertain, lean slightly smaller — under-forecasting demand is the more expensive error, because the inference penalty for an oversized model compounds over every token. The lesson: $D_{inf}$ is the input the whole optimization is most sensitive to, so treat its uncertainty seriously and bias toward the smaller model when you cannot pin it down.

### 10. The quantization that made over-training backfire

A team over-trains a small model hard for cheap serving, then quantizes it aggressively at inference time to cut memory and latency further — and the quantized model degrades more than expected, giving back much of the quality they paid for with all those extra tokens. The mechanism is subtle: heavily over-trained models (very high tokens-per-parameter) can be *more* sensitive to post-training quantization, because the extra training pushes the weights into a configuration that tolerates low-precision rounding less gracefully. The symptom is a larger-than-usual quality drop after INT4 quantization on an over-trained checkpoint; the root cause is the interaction between the over-training ratio and the quantization precision. The fix is to either quantize less aggressively, or factor the intended inference precision into the sizing decision in the first place. The lesson: the inference-aware optimum assumes full-precision inference; if you will quantize, the over-training and the precision choice interact, and you cannot optimize them independently.

### 11. The MoE that broke the 2N assumption

A team applies the $2N$-per-token inference cost to a mixture-of-experts model and gets the economics wrong, because in an MoE only a fraction of the parameters are active per token. The total parameter count $N$ might be 100B, but if only two of sixteen experts fire per token, the *active* parameters are closer to 15B, and the inference cost is $2 \times N_{active}$, not $2N_{total}$. The symptom is a serving cost far lower than the total-parameter count would predict; the (pleasant) root cause is that MoE decouples capacity from per-token inference cost. The fix is to use active parameters in the inference term and total parameters in the training term, because training touches all experts while inference touches only the routed ones. The lesson: the $2N$ rule is for dense models — architectures that change the active-parameter count change the inference economics and the optimal sizing, and MoE is a deliberate way to get more capacity per inference FLOP.

### 12. The eval that hid the over-training plateau

A team over-trains a small model and watches the validation loss keep dropping smoothly past 1,000:1, concludes the over-training is paying off, and ships — only to find downstream task accuracy barely moved over the last several trillion tokens. The symptom is a divergence between the loss curve (still improving) and the benchmark scores (flat); the root cause is that loss and capability are not perfectly coupled at extreme ratios, and the loss law's promised gains do not always translate into measured task performance, exactly the extreme-ratio caveat the paper warned about. The fix is to track downstream evals, not just held-out loss, when training deep into the over-trained regime, and to stop when the evals plateau even if loss is still falling. The lesson: at extreme tokens-per-parameter, trust task evals over the loss curve, because the loss law is extrapolating beyond where it was fit.

## What this means in practice

Strip away the math and the result is a single change of habit: decide how much you will serve before you decide how big to build. The Chinchilla rule — 20 tokens per parameter — is the right answer to the wrong question for anyone shipping a product, because it optimizes a one-time training cost and ignores the recurring inference cost that, at scale, dwarfs it. The inference-aware objective, $6 N D_{tr} + 2 N D_{inf}$ under a quality constraint, points you at a smaller model trained on more tokens, and the magnitude is not marginal: tens of percent off total compute, and far more off dollars once you account for the unfavorable inference MFU.

Concretely, when you size a model you intend to serve:

- **Forecast lifetime inference tokens $D_{inf}$ first.** Its order of magnitude selects the right point on the size axis. Skipping this step silently picks the Chinchilla objective.
- **Minimize total cost, not training FLOPs.** Use $6 N D_{tr} + 2 N D_{inf}$ in FLOPs, then re-solve in dollars with your real MFU figures. The dollar optimum is smaller than the FLOP optimum.
- **Expect to go well past 20:1.** How far depends on $D_{inf}$ — a few tens for internal tools, hundreds for high-traffic APIs, thousands for flagship consumer apps. There is no universal ratio; it is a function of demand.
- **Solve the data constraint alongside the compute one.** A heavily-served small model may want more unique tokens than exist; plan repetition with the [data-constrained law](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) and cap epochs where repetition stops helping.
- **Treat extreme-ratio predictions as optimistic.** Quality keeps improving up to ~10,000:1, but the loss law overstates the gains there; the direction is robust, the magnitude is not.
- **Invest in the serving stack.** Higher inference MFU shrinks the training-inference cost gap and lets you afford a slightly larger model — the cost-optimal size depends on how efficiently you serve.
- **Use the savings to fund test-time compute where you can verify.** A cheap-per-token model is the prerequisite for [repeated sampling](/blog/machine-learning/scaling-laws/repeated-sampling-scaling-laws) and search; in verifiable domains, small-and-sampled-many-times can beat large-and-single-pass at lower total cost.

The broader arc is that the field's center of gravity has moved from "make the pre-trained model bigger" to "make a right-sized model cheap to run, then run it more." Beyond-Chinchilla is the first formal step of that move: it takes the training-optimal frontier, adds the cost you actually pay in production, and lands on a smaller, longer-trained model. Every frontier deployable model since — the LLaMA-3 family most visibly — is sitting somewhere on the curve this paper drew. The 20:1 rule was never wrong; it was just answering a question almost nobody who serves models is actually asking.

## Further reading

- Sardana & Frankle et al. 2023, "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws" (arXiv:2401.00448, ICML 2024): https://arxiv.org/abs/2401.00448
- Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models" (Chinchilla, arXiv:2203.15556): https://arxiv.org/abs/2203.15556
- Touvron et al. 2023, "LLaMA: Open and Efficient Foundation Language Models" (arXiv:2302.13971): https://arxiv.org/abs/2302.13971
- Muennighoff et al. 2023, "Scaling Data-Constrained Language Models" (arXiv:2305.16264): https://arxiv.org/abs/2305.16264
- Brown et al. 2024, "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling" (arXiv:2407.21787): https://arxiv.org/abs/2407.21787
- Sibling posts on this blog: [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling), [data-constrained scaling laws](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws), and [repeated-sampling scaling laws](/blog/machine-learning/scaling-laws/repeated-sampling-scaling-laws).

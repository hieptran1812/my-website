---
title: "How Much Data? Scaling Laws and Token Budgets for Training"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "A principal-engineer's guide to deciding how much training data you need and of what kind: Chinchilla compute-optimality, over-training for inference economics, repeating data when you run out, the human-text ceiling, and how quality shifts the loss curve."
tags: ["scaling-laws", "training-data", "chinchilla", "compute-budget", "token-budget", "data-constrained", "over-training", "data-quality", "llm-training"]
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 30
---

The single most expensive decision in a pretraining run is made before a single GPU spins up: how many tokens to train on, and for how big a model. Get it wrong in one direction and you burn a seven-figure compute budget on a model that was starved of data — a 280B-parameter network that would have been beaten by a 70B one. Get it wrong the other way and you train a model so small it leaves quality on the table that the same compute could have bought. The frustrating part is that the right answer is not "as much data as possible." It is a specific number, and that number depends on your compute budget, how long you intend to serve the model, how much unique text you actually have, and how clean it is.

This is the question I get asked most often by teams about to commit to a run: *how much data?* The honest answer is that "how much" and "of what" are the same question, and both fall out of a small set of scaling laws that are now well enough understood to be used as a planning tool rather than a research curiosity. The diagram above — well, below — is the mental model I keep in my head for the whole problem.

![Three knobs on the loss: compute splits into params and tokens, while repetition and quality move the curve from the side](/imgs/blogs/data-scaling-laws-and-budgets-1.webp)

There are exactly three knobs. A fixed compute budget $C$ buys a product of two things, model size $N$ and training tokens $D$, tied together by the identity $C \approx 6ND$. Repetition (training for more than one epoch over the same tokens) stretches $D$ cheaply but with decaying value. And data quality moves the entire loss curve down independently of how you split $C$. Everything in this article is a tour of those three knobs and the arithmetic that connects them.

## Why the naive intuitions are wrong

Most engineers arrive at this problem with a set of intuitions absorbed from the GPT-3 era, and almost all of them are subtly wrong. Here is the table I wish someone had handed me.

| Common assumption | The naive view | What the scaling laws actually say |
| --- | --- | --- |
| "Bigger model is always better" | Spend the budget on parameters; data is cheap | At fixed compute, a too-big model is *undertrained*; a smaller model on more tokens reaches lower loss |
| "Train to convergence" | Keep training until loss flattens | Single-epoch, compute-optimal training rarely "converges"; you stop when the *marginal token* costs more than the loss it buys |
| "20 tokens per parameter is the law" | Always train at the Chinchilla ratio | 20:1 is compute-optimal *for training*; if you will serve the model a lot, you should deliberately over-train past it |
| "We will never run out of data" | The internet is effectively infinite | High-quality human text is finite (~hundreds of trillions of tokens) and frontier runs are within ~1 order of magnitude of it |
| "More data always helps" | Pour in more tokens, loss keeps dropping | Repeated tokens decay in value; past ~4 epochs the curve bends; bad-quality tokens can *raise* loss |
| "Data quality is a preprocessing detail" | Filter a little, then scale | Quality shifts the loss-vs-data curve down by an amount comparable to a 2–5× change in data quantity |

If even two of those surprised you, the rest of this article will earn its length. Let me take the knobs one at a time.

## 1. The compute-optimal frontier: from Kaplan to Chinchilla

> The most important sentence in modern pretraining: at a fixed compute budget, model size and data size must grow *together*.

The story starts with Kaplan et al. (2020), the OpenAI scaling-laws paper that launched a thousand training runs. Kaplan measured how loss falls as you scale parameters, data, and compute, and concluded — correctly, given their experimental setup — that loss is a smooth power law in each. But their recommended *allocation* of a marginal unit of compute was heavily skewed toward model size. Train a bigger model, they suggested, and don't grow the dataset nearly as fast. The field listened: GPT-3 was 175B parameters trained on ~300B tokens, a ratio of under 2 tokens per parameter. Gopher was 280B parameters on the same ~300B tokens.

Two years later, DeepMind's Chinchilla paper (Hoffmann et al., 2022) re-ran the experiment more carefully — sweeping learning-rate schedules so that each model was actually trained to its own optimum rather than on a one-size-fits-all schedule — and found something different. At a fixed compute budget, the loss-minimizing allocation grows $N$ and $D$ *in lockstep*, each roughly as the square root of compute. The headline demonstration: they trained Chinchilla, a 70B model, on 1.4 trillion tokens using the *same* compute as Gopher's 280B, and Chinchilla beat Gopher across the board. A model one-quarter the size, trained on nearly 5× the data, won.

![Kaplan grew the model and barely the data; Chinchilla grew both together, so a 70B model beat a 280B one](/imgs/blogs/data-scaling-laws-and-budgets-2.webp)

The figure makes the difference concrete. Spend 10× more compute. Under the Kaplan allocation you grow the model by ~5.5× and the data by only ~1.8×. Under Chinchilla you grow each by ~3.2× ($\sqrt{10} \approx 3.16$). The Kaplan recipe produces big, hungry models that never saw enough tokens to fill their capacity; the Chinchilla recipe produces balanced models that extract the loss the compute can afford.

### Why the two papers disagreed, and who is right

It is tempting to say Kaplan was simply wrong, but that is not the useful reading. The [reconciliation between the two papers](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation) is mostly methodological. Kaplan used a fixed cosine schedule far longer than most of his models' training, which systematically *handicapped the smaller models* (they were stopped before their schedule completed) and made big models look relatively better. Kaplan also folded embedding parameters into the count and fit over a compute range where the curvature is hard to see. Chinchilla tuned the schedule per run, excluded embeddings, and fit three independent ways that agreed. When you correct the schedule artifact, the two results move toward each other: the modern consensus is the Chinchilla one, with the caveat that the exact optimal exponents drift a little with architecture and data.

For planning purposes, internalize this: **the exponents on $N$ and $D$ at the compute-optimal point are both close to 0.5.** Doubling your compute should roughly multiply both model size and token count by $\sqrt{2} \approx 1.41$. If your run plan grows one of them much faster than the other, you are off the frontier and lighting money on fire.

The full derivation, the parametric loss form $L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$, and the fitted constants are in the dedicated [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) deep-dive; I will use the practical consequences here rather than re-derive them.

### Three ways to find the frontier, all agreeing

Part of why the Chinchilla result stuck is that it was triangulated three independent ways, and all three pointed to the same near-equal exponents. Approach 1 fixed the model size and varied tokens, fitting how minimum loss falls — a family of curves, one per model size. Approach 2 used *iso-FLOP* curves: pick a compute budget, train many $(N, D)$ pairs that each consume exactly that budget, and read off the $N$ that minimizes loss; the locus of those minima across budgets traces the frontier. Approach 3 fit the full parametric surface $L(N, D)$ directly and differentiated it. When three methods with different failure modes agree that $N^* \propto C^{a}$ and $D^* \propto C^{b}$ with $a \approx b \approx 0.5$, you believe it. The iso-FLOP view is the one worth carrying around day to day: it is literally "at this budget, which split sits at the bottom of the bowl," and it is exactly the small-scale sweep I recommend running on your own data before a large commitment.

## 2. The 20-tokens-per-parameter rule of thumb

The Chinchilla fit produces a famous, portable number: at the compute-optimal point, you want roughly **20 training tokens per parameter.** It is the single most useful back-of-envelope constant in pretraining, and it holds remarkably well across three orders of magnitude of scale.

![The compute-optimal token count rises in lockstep with model size; the ratio stays near 20:1 at every scale](/imgs/blogs/data-scaling-laws-and-budgets-3.webp)

The table in the figure is worth committing to memory. A 1B model wants ~20B tokens; an 8B model wants ~160B; a 70B model wants ~1.4T (this is exactly Chinchilla); a 500B model wants ~10T. The tokens-per-parameter ratio stays pinned near 20:1 the whole way up. The reason it stays constant is precisely the equal-exponent result from the last section: if both $N$ and $D$ scale as $C^{0.5}$, then $D/N$ is constant in $C$.

A caveat principal engineers should keep in their back pocket: 20:1 is a *coefficient*, not a law of nature. The number that comes out of the Chinchilla fit is closer to 20 for their data and architecture, but later replications land anywhere from ~15 to ~25 depending on tokenizer, data mixture, and model shape. Treat "20:1" as the center of a believable range, not a target to hit to three significant figures. What is robust is the *proportionality*, not the constant.

## 3. The worked scenario: splitting a fixed compute budget

Enough theory. You have been handed a compute budget and asked, "what should we train?" Here is the arithmetic, start to finish.

Two facts do all the work. First, the FLOPs identity for a dense transformer:

$$C \approx 6 N D$$

where $C$ is total training FLOPs, $N$ is non-embedding parameters, and $D$ is training tokens. (The factor of 6 is 2 FLOPs per parameter for the forward pass and ~4 for the backward pass, per token.) Second, the compute-optimal ratio $D \approx 20 N$.

Substitute the ratio into the identity:

$$C \approx 6 N (20 N) = 120 N^2 \quad\Longrightarrow\quad N^* = \sqrt{\frac{C}{120}}, \quad D^* = 20 N^*$$

That is the entire planning calculation. Let me run it for a concrete budget. Say you have a cluster-month that delivers $C = 1 \times 10^{22}$ FLOPs of useful training compute (after accounting for ~40% MFU on a few hundred GPUs for a few weeks):

$$N^* = \sqrt{\frac{10^{22}}{120}} = \sqrt{8.33 \times 10^{19}} \approx 9.1 \times 10^{9}$$

So a ~9B-parameter model, trained on $D^* = 20 \times 9.1\text{B} \approx 183\text{B}$ tokens. Sanity-check against the identity: $6 \times 9.1\text{e}9 \times 1.83\text{e}11 \approx 1.0 \times 10^{22}$. It closes.

Here is the same logic as a function you can drop into a planning notebook:

```python
import math

def compute_optimal_plan(flops: float, tokens_per_param: float = 20.0):
    """Chinchilla-style split of a fixed training-compute budget.

    flops: total useful training FLOPs (after MFU), e.g. 1e22
    returns (params, tokens) at the compute-optimal point.
    """
    # C = 6 * N * D and D = r * N  ->  N = sqrt(C / (6 * r))
    n = math.sqrt(flops / (6.0 * tokens_per_param))
    d = tokens_per_param * n
    assert abs(6 * n * d - flops) / flops < 1e-6  # identity closes
    return n, d

for C in [1e21, 1e22, 5.76e23, 1e25]:
    n, d = compute_optimal_plan(C)
    print(f"C={C:.1e} FLOPs -> N={n/1e9:6.1f}B params, D={d/1e12:6.2f}T tokens")
```

Running it prints, among others, `C=5.8e23 FLOPs -> N=69.3B params, D=1.39T tokens` — which is Chinchilla, recovered from first principles. That is the reassuring thing about this calculation: the same two lines of algebra that plan your run also reproduce the published frontier models.

### Translating a hardware budget into compute

The plan takes $C$ in FLOPs, but procurement hands you GPU-hours. The bridge is model FLOPs utilization (MFU) — the fraction of peak FLOPs your training loop actually sustains, typically 0.35–0.55 for a well-tuned large run. Useful compute is peak-FLOP/s per GPU times the GPU count times wall-clock seconds times MFU:

| Hardware | Peak (BF16) | Count × time | MFU | Useful $C$ |
| --- | --- | --- | --- | --- |
| H100 | ~1.0 PFLOP/s | 256 × 14 days | 0.45 | ~1.4e23 |
| H100 | ~1.0 PFLOP/s | 1024 × 30 days | 0.45 | ~1.2e24 |
| A100 | ~0.31 PFLOP/s | 512 × 21 days | 0.40 | ~1.1e23 |

Plug the useful $C$ into `compute_optimal_plan` and you have the model and token target your cluster can actually afford. The MFU term is the one teams forget: planning at peak FLOPs overstates your budget by 2–3×, and produces a plan whose token target the run can never reach on schedule. Measure MFU on a short calibration run first, then size the real run from the measured number.

One more practical note. The plan gives you the *compute-optimal* point, the loss-minimizing split for training. It is the right default if your goal is "best model for this training budget and we will not serve it much." But most models that matter are served, often billions of times, and that changes the calculus entirely — which is the next section.

## 4. Over-training past compute-optimal for inference economics

> Chinchilla minimizes the cost of *training*. It says nothing about the cost of *serving*. If you will serve the model a lot, the optimal model is deliberately smaller and deliberately over-trained.

Here is the tension. Inference cost per token is roughly ${2N}$ FLOPs — it scales with model size, full stop. A 70B model costs ~5× more per generated token than a 13B model, forever, on every request. So if you are going to serve a model to millions of users, a smaller model is a gift that keeps on giving. But a smaller model trained at the Chinchilla ratio is a *worse* model. The way out is to take a smaller model and train it on *far more* tokens than 20:1 — to over-train it — until it reaches the quality you need. You pay a premium in training FLOPs (you are off the compute-optimal frontier), but you bank that premium back on every inference call.

This is exactly what the Llama line did. Llama-2 7B was trained on 2T tokens (~285:1, not 20:1). Llama-3 8B was trained on ~15T tokens — roughly **1,900 tokens per parameter**, nearly 100× past Chinchilla-optimal. From a pure training-efficiency standpoint that is "wasteful." From a *total cost of ownership* standpoint, for a model that will be downloaded and served billions of times, it is the correct decision, and the rest of the industry followed.

![Past a serving-volume breakeven, an over-trained smaller model is cheaper to run for the rest of its life](/imgs/blogs/data-scaling-laws-and-budgets-4.webp)

The figure is a crossover chart, and the crossover is the whole point. Plot total lifetime cost (training + all future inference) against cumulative tokens served. The compute-optimal large model has *lower* training cost but a *steeper* inference slope. The over-trained small model has *higher* training cost but a *shallower* slope. The lines cross at a breakeven serving volume. Below it — a model you will barely serve — the bigger compute-optimal model wins. Above it — a model you will serve heavily — the over-trained smaller model wins, and keeps winning for the rest of its life.

### Working the breakeven

Let me put numbers on it. Compare two ways to hit a target quality:

- **Compute-optimal 70B on 1.4T tokens.** Training: $6 \times 70\text{e}9 \times 1.4\text{e}12 \approx 5.9 \times 10^{23}$ FLOPs. Inference: $2 \times 70\text{e}9 = 1.4 \times 10^{11}$ FLOPs per token (140 GFLOP/token).
- **Over-trained 13B on ~10T tokens.** Training: $6 \times 13\text{e}9 \times 1.0\text{e}13 \approx 7.8 \times 10^{23}$ FLOPs. Inference: $2 \times 13\text{e}9 = 2.6 \times 10^{10}$ FLOPs per token (26 GFLOP/token).

The over-trained model costs ~32% *more* to train but ~5.4× *less* to serve per token. Total cost as a function of cumulative served tokens $V$:

$$\text{Total}(V) = C_{\text{train}} + (2N)\,V$$

Set the two totals equal and solve for the breakeven $V$:

$$5.9\text{e}23 + 1.4\text{e}11\,V = 7.8\text{e}23 + 2.6\text{e}10\,V$$
$$ (1.4\text{e}11 - 2.6\text{e}10)\,V = 7.8\text{e}23 - 5.9\text{e}23 $$
$$ 1.14\text{e}11\,V = 1.9\text{e}23 \quad\Longrightarrow\quad V \approx 1.7 \times 10^{12}\ \text{tokens} $$

So the breakeven is around **1.7 trillion tokens served.** That sounds like a lot until you price it in requests. At, say, 500 generated tokens per request, 1.7T tokens is ~3.4 billion requests. A popular assistant clears that in *weeks*. For any model with real traffic, the over-trained smaller model is cheaper essentially from day one — which is why frontier labs over-train their flagship served models so aggressively.

```python
def overtrain_breakeven(n_big, d_big, n_small, d_small):
    """Cumulative served tokens at which the smaller over-trained model
    becomes cheaper in total (train + serve) than the bigger compute-optimal one."""
    c_train_big   = 6 * n_big   * d_big
    c_train_small = 6 * n_small * d_small
    infer_big, infer_small = 2 * n_big, 2 * n_small   # FLOPs / served token
    # c_train_big + infer_big*V == c_train_small + infer_small*V
    return (c_train_small - c_train_big) / (infer_big - infer_small)

V = overtrain_breakeven(70e9, 1.4e12, 13e9, 10e12)
print(f"breakeven at {V/1e12:.2f}T served tokens")   # ~1.68T
```

The rule of thumb that falls out: **the more you will serve a model, the smaller and more over-trained it should be.** The compute-optimal point is only optimal when serving volume is negligible.

## 5. Data-constrained scaling: what to do when you run out of unique tokens

Over-training raises an obvious problem. If a 13B model wants 10T tokens but you only have, say, 2T unique high-quality tokens in your domain, where do the other 8T come from? You have three moves: repeat the data you have, go find more data, or generate synthetic data. The science of the first move — repetition — is the subject of Muennighoff et al.'s "Scaling Data-Constrained Language Models" (2023), and it is one of the most practically important results of the last few years.

The finding, stated plainly: **repeated tokens are worth less than fresh tokens, and the value decays the more you repeat.** But — and this is the part that surprised everyone — the decay is gentle at first. Training for up to about **four epochs** over the same data is *almost as good* as having that many epochs' worth of fresh tokens. Beyond ~4 epochs the returns bend sharply; by the time you are repeating data ~40 times, additional epochs add essentially nothing to the model.

<figure class="blog-anim">
<svg viewBox="0 0 760 300" role="img" aria-label="Bar chart: as repeated epochs increase, effective new tokens fall further below the ideal no-decay line" style="width:100%;height:auto;max-width:820px">
<style>
.rt-act{fill:var(--accent,#6366f1)}
.rt-ideal{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:2;stroke-dasharray:5 4}
.rt-cover{fill:var(--background,#ffffff)}
.rt-axis{stroke:var(--text-primary,#1f2937);stroke-width:2}
.rt-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.rt-leg{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.rt-note{font:italic 600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
@keyframes rt-wipe{0%{transform:translateX(0)}100%{transform:translateX(660px)}}
.rt-anim{animation:rt-wipe 8s ease-in-out infinite alternate}
@media (prefers-reduced-motion:reduce){.rt-cover{animation:none;display:none}.rt-anim{animation:none}}
</style>
<g>
<rect class="rt-act" x="80"  y="210" width="46" height="20"/>
<rect class="rt-act" x="162" y="191" width="46" height="39"/>
<rect class="rt-act" x="244" y="173" width="46" height="57"/>
<rect class="rt-act" x="326" y="157" width="46" height="73"/>
<rect class="rt-act" x="408" y="145" width="46" height="85"/>
<rect class="rt-act" x="490" y="137" width="46" height="93"/>
<rect class="rt-act" x="572" y="132" width="46" height="98"/>
<rect class="rt-act" x="654" y="129" width="46" height="101"/>
</g>
<rect class="rt-cover rt-anim" x="68" y="124" width="650" height="112"/>
<rect class="rt-ideal" x="80"  y="210" width="46" height="20"/>
<rect class="rt-ideal" x="162" y="190" width="46" height="40"/>
<rect class="rt-ideal" x="244" y="170" width="46" height="60"/>
<rect class="rt-ideal" x="326" y="150" width="46" height="80"/>
<rect class="rt-ideal" x="408" y="130" width="46" height="100"/>
<rect class="rt-ideal" x="490" y="110" width="46" height="120"/>
<rect class="rt-ideal" x="572" y="90"  width="46" height="140"/>
<rect class="rt-ideal" x="654" y="70"  width="46" height="160"/>
<line class="rt-axis" x1="70" y1="230" x2="712" y2="230"/>
<line class="rt-axis" x1="70" y1="62" x2="70" y2="230"/>
<text class="rt-lbl" x="103" y="248">1</text>
<text class="rt-lbl" x="185" y="248">2</text>
<text class="rt-lbl" x="267" y="248">3</text>
<text class="rt-lbl" x="349" y="248">4</text>
<text class="rt-lbl" x="431" y="248">5</text>
<text class="rt-lbl" x="513" y="248">6</text>
<text class="rt-lbl" x="595" y="248">7</text>
<text class="rt-lbl" x="677" y="248">8</text>
<rect class="rt-act" x="80" y="34" width="22" height="13"/>
<text class="rt-leg" x="110" y="45">effective tokens (with decay)</text>
<rect class="rt-ideal" x="420" y="34" width="22" height="13"/>
<text class="rt-leg" x="450" y="45">ideal: one fresh epoch each</text>
<text class="rt-note" x="226" y="278">epochs 1-4: near-fresh</text>
<text class="rt-note" x="552" y="278">epochs 5-8: sharp decay</text>
<text class="rt-lbl" x="385" y="290">repeated epochs over a fixed unique-token pool</text>
</svg>
<figcaption>Each repeated epoch adds less than a fresh one; past about four epochs the effective new tokens fall sharply short of the no-decay ideal (dashed).</figcaption>
</figure>

The animation makes the "decaying value of a repeated token" tangible. The dashed bars are the ideal — what you would get if every epoch were a fresh epoch of new data. The solid bars are what you actually get. For the first few epochs the solid bars hug the dashed line: repetition is nearly free. Then they fall away, and the gap (wasted compute) widens with every additional pass.

### The second worked scenario: epochs vs more data vs synthetic

Make it concrete. Suppose you have $U = 100\text{B}$ unique high-quality tokens, and your compute-optimal (or over-training) plan calls for $D$ training tokens. What do you do?

I model the "effective fresh-token equivalent" of repeating $U$ for $k$ epochs with a simple decaying sum — each successive epoch contributes a shrinking fraction of a fresh epoch. This is a stylized version of the Muennighoff result, useful for planning:

```python
def effective_tokens(unique, epochs, half_life=5.0):
    """Stylized data-constrained model: each epoch after the first contributes
    a geometrically-decaying fraction of a fresh epoch's value.
    Captures the 'up to ~4 epochs is near-free, then sharp decay' shape."""
    eff = 0.0
    for k in range(epochs):
        eff += unique * (0.5 ** (k / half_life))   # epoch 0 full, later epochs decay
    return eff

U = 100e9
for k in [1, 2, 4, 8, 16]:
    eff = effective_tokens(U, k)
    ideal = U * k
    print(f"{k:2d} epochs: effective={eff/1e9:6.0f}B  ideal={ideal/1e9:6.0f}B  "
          f"efficiency={eff/ideal:4.0%}")
```

The shape this prints — high efficiency through ~4 epochs, then a steep fall — is the planning insight, and it maps directly onto a decision table:

| Situation | Compute-optimal $D$ vs unique $U$ | Recommended move |
| --- | --- | --- |
| Data-rich | $D \le U$ | Train single-epoch; you have tokens to spare |
| Mildly constrained | $U < D \le 4U$ | Repeat up to ~4 epochs; nearly as good as fresh |
| Heavily constrained | $D > 4U$ | Cap epochs at ~4, then add **synthetic** or new sources for the rest |
| Severely constrained | $D \gg 4U$ | Shrink the model (lower $D$ target) or accept the loss; do *not* repeat 20×+ |

So if you need 400B tokens and have 100B unique, that is exactly 4 epochs — the sweet spot, and you train almost as if you had 400B fresh. If you need 800B (8 epochs), you are well past the knee; you would do better to cap repetition at ~4 epochs and fill the remaining ~400B with synthetic data or a freshly scraped source, rather than grinding the same 100B around eight times. The full parametric form (with the fitted decay constants $R_D^* \approx 15$ epochs over which excess data decays, and the point near ~40 epochs where additional repetition is worthless) is in the [data-constrained scaling laws](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) deep-dive.

## 6. The "running out of tokens" question

Step back and the over-training and repetition pressures combine into a single uncomfortable trend: frontier training-token budgets are growing fast, and the supply of high-quality human text is finite. This is the part of the field that has shifted from a theoretical curiosity to a live constraint in just a couple of years.

![Frontier runs grew tokens roughly 50x in four years and are now within an order of magnitude of the usable text supply](/imgs/blogs/data-scaling-laws-and-budgets-5.webp)

The timeline traces the run-up. GPT-3 (2020) trained on ~300B tokens. Chinchilla (2022) on 1.4T. Llama-2 (2023) on 2T. Llama-3 (2024) on ~15T. Frontier runs in 2026 are reaching into the tens of trillions. That is roughly a 50× growth in four years. Meanwhile, the best estimates of the *total stock* of usable public human-generated text — most prominently Villalobos et al.'s "Will we run out of data?" — put it on the order of a few hundred trillion tokens, with the high-quality subset considerably smaller. Their projections have the effective stock of high-quality training text being exhausted somewhere in the window of roughly 2026–2032, with wide error bars.

You should hold those numbers loosely — "300 trillion tokens" is an order-of-magnitude estimate, not a census, and definitions of "usable" and "high-quality" swing the answer by multiples. But the qualitative conclusion is robust and it is the reason synthetic data has gone from a niche trick to a central pillar of frontier training: **we are within about one order of magnitude of the human-text ceiling, and the curve of demand is still climbing.** Synthetic data, distillation from stronger models, and squeezing more value from existing tokens (better filtering, better curricula, controlled repetition) are the pressure valves. They are not optional extras; they are how the next generation of runs gets fed at all.

This is also why the *quality* knob — the third one from the mental model — has become so important. If you cannot get more tokens, the next best thing is to make the tokens you have worth more.

## 7. How data quality shifts the loss curve

> Quality is not a preprocessing detail you do once and forget. It moves the entire loss-vs-data curve, and the size of that shift rivals a multiplicative change in data quantity.

Every scaling law I have discussed so far quietly assumed a fixed data distribution. Change the *quality* of the data and the curve itself moves. Deduplication, quality filtering (classifier-based or heuristic), removing boilerplate and spam, and balancing the mixture all push the loss-vs-tokens curve down — meaning you reach a given loss with *fewer* tokens, or a lower loss with the *same* tokens.

![Deduplicated, quality-filtered tokens reach a lower loss at every budget, so the same loss needs far fewer tokens](/imgs/blogs/data-scaling-laws-and-budgets-6.webp)

The figure shows the two curves: raw CommonCrawl-style data (solid, higher loss at every budget) versus deduplicated and quality-filtered data (dashed, lower at every budget). The crucial observation is that the filtered curve does not just shift *along* the x-axis (same curve, fewer tokens to reach a point) — it shifts *down*, reaching losses the raw data never reaches at any budget in the plotted range. In practice, careful filtering and dedup have been shown to buy loss improvements equivalent to **2–5× more data** — which, when you are bumping against the token ceiling from the last section, is enormous.

There is now a body of work on *quality scaling laws* that tries to make this quantitative: how the loss-curve offset and even the exponent depend on a measurable notion of data quality. The practical takeaways for a planning engineer:

- **Deduplicate first, aggressively.** Near-duplicate documents are the cheapest loss win available and the most common own-goal. They also silently turn "unique tokens" into repeated tokens, polluting your data-constrained accounting from Section 5.
- **A quality filter is worth a model-scale step.** The gap between raw and filtered web text is often comparable to the gap between two adjacent model sizes. Spending engineering effort on the filter can be cheaper than spending compute on a bigger model.
- **Quality and quantity trade off, but not one-for-one.** Throwing away 50% of your tokens to raise quality can *lower* loss if the discarded half was low-value. The break-even depends on the filter's precision; measure it on a small-scale ablation before committing.

The mechanism, the measured exponents, and the filtering ablations are covered in the [data-quality scaling laws](/blog/machine-learning/scaling-laws/data-quality-scaling-laws) deep-dive. For this article, the point is that quality is the third independent knob: it moves the curve regardless of how you split compute between $N$ and $D$.

## 8. Putting it together: where the next dollar goes

You now have all three knobs. The remaining skill is knowing which one to turn, and that depends entirely on which constraint is binding for *your* run.

![The right place to spend depends on your binding constraint: serving volume, token supply, data quality, or raw scale](/imgs/blogs/data-scaling-laws-and-budgets-7.webp)

The decision tree is how I actually reason about a fresh planning question. Walk it from the top:

- **Is the deployment serving-heavy** (lifetime inference FLOPs will dwarf training FLOPs)? Then over-train a smaller model. The compute-optimal point is the wrong default; aim well past 20:1, like the Llama line did.
- **Are you data-bottlenecked** (your compute-optimal $D$ exceeds your unique-token pool)? Repeat up to ~4 epochs, then turn to synthetic data or new sources rather than over-repeating.
- **Has the loss plateaued on raw web data** despite more tokens? Invest in quality — dedup and filtering — before buying more compute. The curve will move down.
- **Are you both compute-rich and data-rich**, with a model you will not serve much? Then, and only then, do you train at the clean Chinchilla 20:1 compute-optimal point.

Notice that "train at Chinchilla optimal" is the *last* branch, not the first. The pure compute-optimal recipe is the right answer to a surprisingly narrow question — best model for a fixed training budget, serving cost ignored, data unlimited and clean. Real runs almost always have a tighter binding constraint, and the binding constraint is what should drive the budget.

## Case studies

### 1. Chinchilla: the result that re-set the field

The symptom in 2021 was a field convinced that scale meant parameters. Gopher (280B) and GPT-3 (175B) were both trained on ~300B tokens — under 2 tokens per parameter, deep in undertrained territory. The wrong first hypothesis was that these models were near their capability ceiling for their size. Chinchilla's actual root cause: they were starved of data. DeepMind held compute fixed at Gopher's level, re-allocated it to a 70B model on 1.4T tokens, and beat Gopher on essentially every benchmark. The fix — grow $N$ and $D$ together at $\sqrt{C}$ — is now the default. The lesson that outlived the specific model: a parameter you cannot afford to feed is a parameter wasted, and "how big" is meaningless without "on how much."

### 2. Llama: deliberately leaving the frontier for inference economics

Meta's Llama series looks "wrong" by Chinchilla's lights and is right for its purpose. Llama-2 7B on 2T tokens, then Llama-3 8B on ~15T — close to 1,900 tokens per parameter, nearly 100× past compute-optimal. The symptom that would tempt a naive team to stop early: training loss on the 8B was still improving but slowly, and the model was already "past optimal," so why keep paying? The root cause that justifies continuing: these are *served* models, downloaded and run billions of times, where every parameter shaved off the model is a permanent inference saving. The crossover math from Section 4 says the extra training FLOPs are repaid within weeks of real traffic. The lesson: optimize for total cost of ownership, not training efficiency, the moment serving volume is non-trivial.

### 3. Muennighoff et al.: putting a number on repeated data

The open question in 2022 was a practical one teams kept hitting: "we are out of unique tokens — is repeating them worth anything?" Folklore said repetition immediately causes overfitting and is worthless. The actual finding, from a large sweep of data-constrained runs, was more nuanced and more useful: up to ~4 epochs, repeated tokens are *nearly* as valuable as fresh ones; the value then decays smoothly, reaching negligible returns around ~40 epochs. The fix this enabled: teams can confidently repeat scarce high-quality data a handful of times instead of either over-collecting or shrinking the model in a panic. The lesson: "don't repeat data" was an over-correction; the right rule is "repeat a little, with eyes open, and stop at the knee."

### 4. GPT-3: the canonical undertraining cautionary tale

GPT-3 (175B on ~300B tokens) is, in hindsight, the textbook example of the Kaplan allocation taken to its logical end. At ~1.7 tokens per parameter it was profoundly undertrained by Chinchilla standards. The wrong reading at the time was that 175B was simply the model's natural ceiling. The truer reading: the same compute, allocated Chinchilla-style, would have produced a smaller model that matched or beat it — and a model that smaller would also have been far cheaper to serve. None of this diminishes GPT-3's importance; it set the agenda. But it is the clearest illustration of why "bigger" without "fed" leaves capability on the table.

### 5. The synthetic-data pivot at the ceiling

By 2024–2026, frontier teams ran into the wall Section 6 describes: the highest-quality human text was largely consumed, and the next run wanted more tokens than existed. The symptom was a planning spreadsheet where the compute-optimal (or over-trained) $D$ exceeded the available unique high-quality pool by a wide margin. Naive fix: repeat the human data 10–20× — which Section 5 says is mostly wasted compute. The actual fix: cap repetition at the ~4-epoch knee and generate the remainder, via distillation from stronger models and synthetic task data, while leaning hard on dedup and quality filtering to make the human tokens worth more. The lesson: when quantity is capped, the levers that remain are *repetition discipline*, *synthesis*, and *quality* — exactly the three knobs, with the compute split now a downstream consequence rather than the driver.

### 6. RefinedWeb and the deduplication dividend

When the Falcon team built RefinedWeb, the prevailing wisdom was that curated sources — books, Wikipedia, code — were necessary to beat raw web crawl, and that web text alone topped out at mediocre quality. The symptom was a persistent quality gap: models trained on lightly filtered CommonCrawl underperformed those with curated mixes at the same $N$ and $D$. The wrong hypothesis was that the web has an inherent quality ceiling. The actual root cause was insufficiently aggressive processing: the fix was heavy deduplication (both exact and fuzzy) plus strict filtering, applied at scale. A model trained purely on the resulting web data matched or beat curated-mix baselines — on a far larger available token pool. The lesson maps straight onto Section 7: the quality knob moved the loss curve down by an amount that rivaled adding curated data, and it did so while *expanding* the usable token supply rather than shrinking it. That is the exact combination you want as you approach the human-text ceiling — more tokens and better tokens from the same source.

## Troubleshooting: reading the run

Most of the failures in this area do not announce themselves as "you chose the wrong token budget." They show up as a training curve that looks subtly off. Here is the symptom-to-cause-to-fix table I use.

| Symptom | Likely root cause | Fix |
| --- | --- | --- |
| Loss still dropping steeply when the schedule ends | **Undertrained** — too few tokens for this $N$ (Kaplan-style allocation) | Shrink the model or extend $D$ toward 20:1; you stopped on the steep part of the curve |
| Validation loss flat for the last third of training, lots of compute spent | **Overtrained relative to data** — repeating a small pool many epochs | Check epochs-over-unique; cap near ~4, add fresh/synthetic tokens |
| Train loss keeps falling but validation/held-out loss stalls or rises | Memorization of repeated or duplicated data | Deduplicate; reduce epoch count; the "unique" pool was smaller than you thought |
| Adding more tokens barely moves loss | Past the useful-data knee, or new tokens are low quality | Switch the lever: invest in quality/dedup or a bigger model, not more of the same data |
| A bigger model is *worse* than a smaller one at the same compute | Classic undertraining of the big model | Re-allocate toward Chinchilla 20:1; the big model never saw enough tokens |
| Loss curve sits stubbornly above a known reference at the same $N$, $D$ | Data quality gap | Compare data pipeline: dedup rate, filter precision, mixture; quality shifts the whole curve |

A few diagnostic habits make these legible:

- **Always track tokens-per-parameter and epochs-over-unique as first-class metrics.** They are the two ratios that tell you which regime you are in. If you cannot answer "how many epochs over unique data is this run?" off the top of your head, you cannot reason about the curve.
- **Run a small-scale Chinchilla sweep before the big run.** Train a handful of small models at different $N$/$D$ splits on your *actual* data and fit the loss surface. The compute is trivial relative to the main run and it catches an off-frontier plan before you commit the budget. Do not trust the 20:1 constant blindly on a new data mixture — measure your own coefficient.
- **Separate "the model is done learning" from "we ran out of useful tokens."** They produce similar-looking flat curves but call for opposite fixes (stop and ship vs. change the data lever). Held-out loss on genuinely fresh data, not a repeated slice, is what distinguishes them.

> If your training curve is flat and you do not know *why*, do not just add tokens. Adding more of the wrong lever is how compute budgets die.

## When to reach for each lever — and when not to

**Train at the compute-optimal (Chinchilla 20:1) point when:**

- You are optimizing for the best model at a fixed *training* budget and serving cost is genuinely negligible (research models, one-off evaluations, internal experiments).
- You have ample unique, high-quality data — your unique pool comfortably exceeds the compute-optimal $D$.
- You are doing a calibration sweep to fit your own scaling coefficients before a larger run.

**Deliberately over-train a smaller model when:**

- The model will be served at scale and lifetime inference FLOPs will dominate training FLOPs (the crossover from Section 4 is reached quickly).
- You are shipping weights others will run, multiplying the inference savings across every downstream deployment.
- Latency or memory at serve time is a hard constraint that a smaller model relieves.

**Reach for repetition (up to ~4 epochs) when:**

- Your compute-optimal $D$ modestly exceeds your unique high-quality pool, and the data is already deduplicated.

**Reach for synthetic data and quality work when:**

- You are past the ~4-epoch knee, or bumping the human-text ceiling, and more raw tokens are unavailable or low-value.

**Skip the clever allocation and just collect/clean more data when:**

- Your data pipeline has obvious quality defects (high near-duplicate rate, weak filtering). Fix the curve-shifting problem before micro-optimizing the curve-position problem; a better filter often beats a better split.
- The model is small and cheap enough that the entire budget question is in the noise — do not spend a week of planning on a run that costs an afternoon.

The thread running through all of it: "how much data" is never answered in isolation. It is answered by naming your binding constraint — training budget, serving volume, token supply, or data quality — and letting that constraint pick the lever. The compute-optimal frontier tells you the *default*; the other three knobs tell you when and how far to deviate from it. Once you have chosen how much data and of what kind, the next question is how to *mix* it across domains and order it over training — which is the subject of the sibling post on [data mixing, domain weighting, and curriculum](/blog/machine-learning/training-data/data-mixing-domain-weighting-and-curriculum).

## Further reading

- [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) — the full parametric loss form and the fitted constants behind the 20:1 rule.
- [Kaplan vs Chinchilla, reconciled](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation) — why the two papers disagreed and what the schedule artifact was.
- [Data-constrained scaling laws](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) — the repetition decay model, the ~4-epoch knee, and the ~40-epoch dead-end.
- [Data-quality scaling laws](/blog/machine-learning/scaling-laws/data-quality-scaling-laws) — how filtering and dedup shift the loss curve, with measured exponents.
- [Data mixing, domain weighting, and curriculum](/blog/machine-learning/training-data/data-mixing-domain-weighting-and-curriculum) — the next decision after "how much": what proportions, in what order.

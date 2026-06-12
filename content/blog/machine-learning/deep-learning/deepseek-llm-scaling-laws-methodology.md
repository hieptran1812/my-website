---
title: "DeepSeek LLM: The Scaling-Law Methodology Everyone Skipped"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A technique deep-dive on DeepSeek LLM's contrarian scaling-law work: a sharper compute metric, data-quality-dependent allocation exponents, and a step LR scheduler built for continual pretraining."
tags:
  - "scaling-laws"
  - "deepseek"
  - "pretraining"
  - "compute-budget"
  - "learning-rate-schedule"
  - "hyperparameter-tuning"
  - "isoflop"
  - "chinchilla"
  - "llm-training"
  - "deep-learning"
  - "model-scaling"
  - "data-quality"
category: "machine-learning"
subcategory: "Deep Learning"
author: "Hiep Tran"
featured: true
readTime: 50
---

Most teams treat scaling laws the way they treat a thermostat reading on the wall: a number someone else measured, in some other building, that they trust without question. They pull the Chinchilla rule of thumb — roughly twenty tokens per parameter — multiply it by their compute budget, pick the nearest round model size, and start the run. Then, three weeks and a few hundred thousand dollars later, they are surprised the loss curve does not land where the OpenAI or DeepMind papers said it would.

The senior rule of thumb that this whole article defends is blunt: **re-derive the scaling laws on YOUR data before you commit a frontier budget.** The numbers in the famous papers are not physical constants. They are regression fits over a specific corpus, a specific tokenizer, and a specific compute-accounting convention. Change any of those three and the optimal allocation between model size and data moves — sometimes by enough to turn a well-budgeted run into a mediocre one.

The DeepSeek LLM paper (arXiv 2401.02954, January 2024) is the cleanest published example of a team taking that rule seriously. They shipped two dense models, a 7B and a 67B, both pretrained on 2 trillion tokens of mixed English and Chinese text at a 4096-token context length. The 67B beat LLaMA-2 70B on code, math, and reasoning benchmarks. That headline result is fine, but it is not why the paper matters. It matters because before they trained anything at frontier scale, they tore down three pieces of received wisdom and rebuilt them: how you count compute, how you split the budget between parameters and tokens, and how you schedule the learning rate so a finished run can be extended later.

![Replacing 6N with non-embedding FLOPs per token fixes the compute mis-attribution that warps small-scale fits.](/imgs/blogs/deepseek-llm-scaling-laws-methodology-1.webp)

The diagram above is the mental model for the first of those three moves. On the left is the compute metric almost every scaling-law paper inherited from Kaplan et al.: $C = 6ND$, where $N$ is the parameter count and $D$ is the token count. On the right is DeepSeek's replacement: $C = M \cdot D$, where $M$ is the non-embedding FLOPs per token. The whole methodology hinges on the right-hand box being a better ruler than the left-hand one, because every downstream regression — batch size, learning rate, the model-versus-data split — is fit against the compute axis. Get the ruler wrong and every fit inherits the error.

We are going to walk through that diagram one box at a time, then do the same for the allocation laws and the learning-rate schedule. Along the way I will plug concrete compute budgets into the fitted formulas so you can see what the laws actually predict, show you the code you would write to reproduce the fits, and call out where this methodology helps you and where it is overkill. If you have read the later DeepSeek work — the [DeepSeek-V3 cost-discipline post](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing) on this blog, or anything about [Multi-head Latent Attention](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla) — you will recognize the DNA here. The obsession with serving cost and the willingness to re-derive instead of inherit both start in this paper.

## Why this paper is different

Before the details, here is the assumption-versus-reality table that frames the entire post. Each row is a piece of conventional scaling-law wisdom that DeepSeek either sharpened or contradicted.

| Topic | Common assumption | What DeepSeek found |
|---|---|---|
| Compute accounting | $C = 6ND$ is "close enough" at any scale | $6N$ mis-attributes compute at small $N$; use $M \cdot D$ with $M$ = non-embedding FLOPs/token |
| Model/data split | Chinchilla's $a \approx b \approx 0.5$ is a universal law | The exponents are dataset-dependent; better data shifts budget toward model size |
| Optimal batch size | Tune by hand or copy a prior run | $B_{\text{opt}} = 0.292 \cdot C^{0.327}$, fit empirically across budgets |
| Optimal learning rate | Pick a constant and decay with cosine | $\eta_{\text{opt}} = 0.3118 \cdot C^{-0.125}$, a power law in compute |
| LR schedule | Cosine decay is the default | Multi-step schedule matches cosine loss but stays resumable |
| Scaling for size | Widen the model (more $d_{\text{model}}$) | Scale depth (95 layers) and add GQA, chosen for inference cost |

Every one of those reality-column claims is a measured result in the paper, not a vibe. The rest of this article is the tour of that table. We will spend the most time on the three rows that are genuinely contrarian — the compute metric, the dataset-dependent split, and the step scheduler — because those are the reusable lessons. The batch-size and learning-rate laws are useful, but they are the kind of thing you would expect any careful team to fit; the contrarian rows are what separate a careful team from a thoughtful one.

> A scaling law is a regression, and a regression is only as honest as the axis you fit it against. DeepSeek's first move was to fix the axis. Everything else follows from that.

## 1. The compute metric: why 6ND lies at small scale

**The senior rule of thumb here: if your compute axis is wrong, every law you fit against it is wrong in a way no amount of data collection will fix.** This is the most important section in the post, so we are going to be slow and concrete.

The standard formula, popularized by Kaplan and reinforced by Chinchilla, estimates training compute as $C = 6ND$. The $6$ is a back-of-envelope constant: roughly $2N$ FLOPs for the forward pass (one multiply-add per parameter per token, and a multiply-add is two FLOPs) and roughly $4N$ for the backward pass, which touches each parameter about twice as often. The $N$ is the total parameter count, and $D$ is the number of training tokens. Multiply them and you get a single scalar that is supposed to capture "how much arithmetic this run cost."

For very large models this approximation is fine, because the dense matrix multiplications in the feed-forward and attention projection layers dominate, and those genuinely cost about $2$ FLOPs per parameter per token in the forward pass. The trouble is at the small end of the IsoFLOP sweep, where you fit the laws. At small scale, two things that $6N$ ignores stop being negligible.

The first is the embedding and unembedding tables. A vocabulary of, say, 100,000 tokens times a hidden size of 1024 is about 100M parameters that do essentially no arithmetic per token — an embedding lookup is a gather, not a matmul, and it contributes almost nothing to the FLOP count even though it inflates $N$. On a 100M-parameter model, the embedding table can be 30 to 50 percent of the parameters. So $6N$ counts those parameters as if they were doing dense matmul work, which they are not. Your small-model compute estimate is inflated, and inflated unevenly across model sizes, which is the worst kind of error for a regression.

The second thing $6N$ ignores is the attention computation itself. The cost of the attention score matrix and the value-weighted sum scales with sequence length $s$ in a way that $6N$ does not model at all — it is an $O(s)$ per-token term (and $O(s^2)$ per sequence) that lives outside the per-parameter accounting. At a 4096-token context that term is real. Folding it into a flat $6N$ smears the true cost across model sizes inconsistently.

### What M actually is

DeepSeek's replacement is $M$, the **non-embedding FLOPs per token**. You compute it by walking the actual forward graph: the feed-forward matmuls, the attention QKV and output projections, and the attention score-and-weight computation that depends on sequence length, all summed per token, and crucially **excluding** the embedding lookup that contributes parameters but not arithmetic. The training compute is then simply

$$C = M \cdot D$$

where $D$ is again the token count. The elegance is that $M$ is a per-token quantity you can compute exactly from the architecture config — layer count, hidden size, head count, FFN ratio, sequence length — without any hand-waving constant. It is not an approximation of the arithmetic; it is the arithmetic.

Here is the kind of function you would write to compute $M$ for a transformer block, which is the unit you multiply by the layer count:

```python
def non_embedding_flops_per_token(
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ffn: int,
    seq_len: int,
) -> float:
    """Forward-pass non-embedding FLOPs per token (multiply-add = 2 FLOPs).

    Counts the attention projections, the sequence-length-dependent
    score/weight term, and the FFN matmuls. Excludes the embedding
    lookup, which moves data but does no arithmetic.
    """
    d_head = d_model // n_heads

    # QKV projections + output projection: 4 matmuls of (d_model x d_model)
    attn_proj = 4 * 2 * d_model * d_model

    # Attention scores (q.k) and weighted sum (a.v), both O(seq_len):
    #   scores:  n_heads * seq_len * d_head
    #   weights: n_heads * seq_len * d_head
    attn_seq = 2 * 2 * n_heads * seq_len * d_head

    # FFN: two matmuls, up-projection and down-projection
    ffn = 2 * 2 * d_model * d_ffn

    per_layer = attn_proj + attn_seq + ffn
    forward = per_layer * n_layers

    # Backward is ~2x forward; total multiplier ~3x forward.
    return 3.0 * forward
```

Note what this function does that $6N$ cannot: the `attn_seq` term scales with `seq_len`, so a 4096-context model and a 2048-context model with identical parameter counts get different $M$ values. That is correct. They genuinely cost different amounts of compute per token, and the old metric pretended they did not.

### Why this changes the fitted laws

The reason this matters for the methodology, and not just for accounting hygiene, is that the model-versus-data allocation laws are expressed in terms of $M$, not $N$. When you fit "how much of the budget goes to model scale," you are fitting $M_{\text{opt}}$ as a function of $C$, where $M$ already excludes the dead embedding weight and already includes the attention term. The result is a regression whose independent variable is a faithful measure of compute. DeepSeek reports that using $M \cdot D$ instead of $6ND$ produced more accurate and more stable fits across their eight compute budgets, spanning roughly $10^{17}$ to $3 \times 10^{20}$ FLOPs.

If you take one thing from this section, take this: the difference between $6ND$ and $M \cdot D$ is small at 70B parameters and large at 100M parameters, and you fit your laws at 100M parameters. The small-scale runs are where the regression gets its leverage. Pollute them with a biased compute axis and the extrapolation to frontier scale carries that bias forward, magnified.

### A numeric walk: 6ND versus M.D on a 165M-parameter model

Let us make the bias concrete with a small model of the kind that anchors the low end of an IsoFLOP sweep. Take a hidden size $d_{\text{model}} = 1024$, $n_{\text{layers}} = 12$, $n_{\text{heads}} = 16$, an FFN ratio of 4 so $d_{\text{ffn}} = 4096$, a sequence length of 4096, and a vocabulary of 100,000 tokens. The non-embedding parameters are dominated by, per layer, four $1024 \times 1024$ attention projection matrices (about 4.2M parameters) plus two $1024 \times 4096$ FFN matrices (about 8.4M parameters), so roughly 12.6M per layer times 12 layers is about 151M non-embedding parameters. The embedding and tied unembedding contribute $100{,}000 \times 1024 \approx 102$M parameters — but they are largely shared if tied, so call it an extra 102M on the input side. The total $N$ that goes into $6N$ is therefore around 253M, of which 102M, about 40 percent, is embedding weight that does almost no arithmetic.

Now compute $M$ from the forward graph. Per layer, the attention projections cost $4 \times 2 \times 1024^2 \approx 8.4$M FLOPs per token; the sequence-dependent attention term costs $2 \times 2 \times 16 \times 4096 \times 64 \approx 33.6$M FLOPs per token at this 4096 context; the FFN costs $2 \times 2 \times 1024 \times 4096 \approx 16.8$M FLOPs per token. That is about 58.8M forward FLOPs per token per layer, times 12 layers is about 706M, times the roughly 3x forward-plus-backward multiplier gives $M \approx 2.1$ billion FLOPs per token.

Compare the two accountings on a 100B-token run. The classic metric gives $C_{6ND} = 6 \times 253\text{M} \times 100\text{B} \approx 1.52 \times 10^{20}$ FLOPs. The DeepSeek metric gives $C_{M \cdot D} = 2.1\text{B} \times 100\text{B} \approx 2.1 \times 10^{20}$ FLOPs. These disagree by nearly 40 percent, and they disagree in different directions for different reasons: $6N$ over-counts because of the embedding table but under-counts because it omits the 33.6M-per-token attention term, which at 4096 context is the single largest contributor to $M$. The two errors do not cancel cleanly, and critically, their relative sizes change as the model grows — the embedding fraction shrinks while the attention term's importance depends on the context length. That moving, model-size-dependent error is what poisons a regression. A constant bias would just rescale the intercept $k$ and leave the exponent $\alpha$ alone; a bias that varies with model size bends the line and corrupts the slope.

### Why the slope is what you cannot afford to corrupt

It is worth dwelling on why a corrupted exponent is so much worse than a corrupted coefficient. The coefficient $k$ in $y = k C^{\alpha}$ sets the absolute level of the prediction; the exponent $\alpha$ sets how the prediction grows as you extrapolate. If you fit your laws at $10^{18}$ FLOPs and deploy them at $10^{22}$ FLOPs — four orders of magnitude out — a 5 percent error in $\alpha$ compounds across those four decades into a large multiplicative error in the predicted optimum. A 5 percent error in $k$ stays a 5 percent error everywhere. Because the whole value of scaling laws is extrapolation, the exponent is the asset and the coefficient is almost incidental. The compute-metric fix is, in the end, a fix to protect the slope, by making sure the bias along the x-axis does not vary with model size and bend the fitted line.

#### Second-order optimization: tokenizer and vocab interact with M

A non-obvious gotcha: because $M$ explicitly excludes the embedding table, the choice of vocabulary size stops affecting your compute axis the way it would under $6N$. A 128K-vocab tokenizer and a 32K-vocab tokenizer with the same hidden size and depth have nearly the same $M$ — the extra embedding rows add parameters but not FLOPs. Under $6N$, the 128K-vocab model would look "more expensive" purely because its parameter count is larger, which would skew your allocation fit toward thinking you need fewer tokens. Switching to $M$ decouples the tokenizer choice from the compute-allocation decision, which is exactly what you want when you are also tuning the tokenizer. This is the sort of interaction that only shows up once you stop treating the compute metric as a fixed constant and start treating it as a modeling choice.

This decoupling has a direct line to the [DeepSeek-V3 cost engineering](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing): a team that has cleanly separated "parameters that cost FLOPs" from "parameters that cost only memory" is a team positioned to reason precisely about FP8 quantization, expert routing, and every other trick that trades one resource for another. The compute metric is not just an accounting nicety; it is the substrate on which every later cost-versus-quality decision is computed.

## 2. The four fitted power laws of compute

**Rule of thumb: never carry a hyperparameter across a 100x change in compute without re-checking it against a fitted law.** Once DeepSeek had a trustworthy compute axis, they fit four power laws against it. Each one tells you how a single training knob should move as you scale the budget.

![Every training knob is a power law of the compute budget C, fit empirically across eight IsoFLOP runs.](/imgs/blogs/deepseek-llm-scaling-laws-methodology-2.webp)

The matrix above is the whole result set on one page. Read each row as "this knob equals this coefficient times $C$ raised to this exponent," and the last two columns tell you which direction it moves and what it controls. The four laws are:

$$B_{\text{opt}} = 0.292 \cdot C^{0.327}$$
$$\eta_{\text{opt}} = 0.3118 \cdot C^{-0.125}$$
$$M_{\text{opt}} = 0.1715 \cdot C^{0.5243}$$
$$D_{\text{opt}} = 5.8316 \cdot C^{0.4757}$$

The first two govern the optimizer: as compute grows, the optimal batch size grows (more tokens per step) and the optimal learning rate shrinks (smaller steps). The second two govern allocation: as compute grows, both the model scale $M$ and the data scale $D$ grow, and the exponents $a \approx 0.524$ and $b \approx 0.476$ tell you the split is close to balanced but tilted slightly toward model.

### Worked example: plug in a budget

Abstract formulas are easy to nod at and hard to feel. Let us pick a compute budget and run all four laws by hand. Say you have a budget of $C = 10^{20}$ FLOPs — a mid-size run, well within the IsoFLOP fitting range. Working in $\log_{10}$ makes the arithmetic clean, since each law is linear in log space.

For the **batch size**: $\log_{10} B_{\text{opt}} = \log_{10}(0.292) + 0.327 \cdot \log_{10}(10^{20}) = -0.534 + 0.327 \cdot 20 = -0.534 + 6.54 = 6.006$. So $B_{\text{opt}} \approx 10^{6.006} \approx 1.01 \times 10^6$ tokens per batch — roughly a million tokens per step.

For the **learning rate**: $\log_{10} \eta_{\text{opt}} = \log_{10}(0.3118) - 0.125 \cdot 20 = -0.506 - 2.5 = -3.006$. So $\eta_{\text{opt}} \approx 10^{-3.006} \approx 9.9 \times 10^{-4}$ — just under $10^{-3}$, which lines up with the learning rates you see in real frontier runs.

For the **model scale**: $\log_{10} M_{\text{opt}} = \log_{10}(0.1715) + 0.5243 \cdot 20 = -0.766 + 10.486 = 9.72$. So $M_{\text{opt}} \approx 10^{9.72} \approx 5.2 \times 10^9$ non-embedding FLOPs per token.

For the **data scale**: $\log_{10} D_{\text{opt}} = \log_{10}(5.8316) + 0.4757 \cdot 20 = 0.766 + 9.514 = 10.28$. So $D_{\text{opt}} \approx 10^{10.28} \approx 1.9 \times 10^{10}$ tokens — about 19 billion tokens for this budget.

A sanity check you can do for free: multiply $M_{\text{opt}} \cdot D_{\text{opt}}$ and you should recover $C$. We have $5.2 \times 10^9 \times 1.9 \times 10^{10} \approx 9.9 \times 10^{19} \approx 10^{20}$. The split between model and data is internally consistent, which is the whole point of fitting $a$ and $b$ jointly so that $a + b \approx 1$.

### The code that fits these laws

The fitting procedure is an IsoFLOP sweep followed by a log-log regression. The pattern is worth seeing in code because it makes the methodology reproducible on your own data, which is the entire thesis of this post.

```python
import numpy as np

def fit_power_law(compute_budgets, optimal_values):
    """Fit y = k * C^alpha by linear regression in log-log space.

    compute_budgets: array of C values (FLOPs) for each IsoFLOP run.
    optimal_values:  the empirically-found optimum (B, eta, M, or D)
                     at each budget, read off the bottom of the loss valley.
    Returns (k, alpha).
    """
    log_c = np.log10(np.asarray(compute_budgets, dtype=float))
    log_y = np.log10(np.asarray(optimal_values, dtype=float))

    # Least-squares line: log_y = alpha * log_c + log_k
    alpha, log_k = np.polyfit(log_c, log_y, deg=1)
    return 10.0 ** log_k, alpha


def find_optimum_per_budget(runs):
    """For each compute budget, find the config at the bottom of the
    loss-vs-allocation parabola.

    runs: list of (compute, model_scale_M, data_D, final_loss) tuples.
    Groups by compute budget, fits a parabola to loss vs log(M/D),
    and returns the allocation that minimizes it.
    """
    by_budget = {}
    for c, m, d, loss in runs:
        by_budget.setdefault(round(np.log10(c), 1), []).append((m, d, loss))

    optima = []
    for log_c, group in sorted(by_budget.items()):
        ratios = np.log10([m / d for m, d, _ in group])
        losses = np.array([loss for _, _, loss in group])
        # Quadratic fit; vertex is the optimal log(M/D).
        a, b, _ = np.polyfit(ratios, losses, deg=2)
        best_ratio = -b / (2 * a)
        optima.append((10.0 ** log_c, best_ratio))
    return optima
```

The structure here is the load-bearing part. You run many small models at each of several fixed compute budgets, vary the model-versus-data tradeoff within each budget, find the bottom of each loss parabola, and then regress those optima against compute. DeepSeek did this across **eight** compute budgets in the $10^{17}$ to $3 \times 10^{20}$ range. The reason eight is enough is that a power law is a straight line in log-log space, and you do not need many points to fit a line confidently — you need points spread across enough orders of magnitude that the slope is well-constrained. Eight points over three-plus decades of compute does that.

### Reading the batch-size and learning-rate laws together

The batch-size and learning-rate laws are not independent; they move in opposite directions for a reason rooted in optimizer dynamics. As compute grows, $B_{\text{opt}}$ rises with exponent $+0.327$ and $\eta_{\text{opt}}$ falls with exponent $-0.125$. The larger batch averages over more examples per step, which reduces gradient noise, and a less noisy gradient supports a more confident — but here, counterintuitively, a smaller — step at the optimum. The reason the learning rate falls rather than rises with batch size is that at frontier scale the loss landscape the model navigates is sharper and the cost of overshooting is higher; the fitted law captures the net effect of all those forces rather than any single textbook rule about the linear or square-root batch-size scaling of learning rate. The lesson for a practitioner is to trust the fitted pair over any analytic batch-size-to-learning-rate heuristic, because the fit already integrates whatever the true relationship is on your data.

Here is the same worked budget, $C = 10^{20}$, expressed as a config you would actually hand to a trainer. Batch size $B_{\text{opt}} \approx 1.0 \times 10^6$ tokens per step. At a 4096-token sequence length, that is about 250 sequences per step, which across data-parallel and gradient-accumulation gives you a concrete micro-batch plan. Learning rate $\eta_{\text{opt}} \approx 9.9 \times 10^{-4}$, which you would feed as the peak after the 2000-step warmup. Model scale $M_{\text{opt}} \approx 5.2 \times 10^9$ non-embedding FLOPs per token, which you back out into an architecture — a depth, width, and head count whose forward graph produces that $M$. Data scale $D_{\text{opt}} \approx 1.9 \times 10^{10}$ tokens. Four formulas, one budget, a complete and internally consistent training plan with no manual search. That is the operational value of the methodology: scaling up becomes a lookup, not an experiment.

### The IsoFLOP valley: where the optima come from

The four laws are fit against optima, and those optima come from the bottom of a loss valley at each fixed budget. The shape of that valley is the reason the method works at all. At a fixed compute budget $C = M \cdot D$, you can trade model scale against data scale along a hyperbola: a bigger model trained on fewer tokens, or a smaller model trained on more tokens, both costing the same total compute. Sweep along that hyperbola and plot final loss against the model-versus-data ratio, and you get a parabola in log space — too small a model under-fits the data it sees, too small a data budget under-trains the model you built, and somewhere in between sits a minimum. That minimum is the optimal allocation for that budget. The `find_optimum_per_budget` function above fits a quadratic to each budget's parabola and reads off the vertex; the `fit_power_law` function then regresses those vertices against compute. The entire methodology is two nested fits: a parabola per budget to find each optimum, then a line through the optima to find the law.

The reason you need the valley to be reasonably symmetric and well-sampled is that a vertex read off a lopsided or under-sampled parabola is noisy, and that noise propagates into the final regression. This is the practical reason to run several allocations per budget — enough points on each side of the minimum to pin the vertex down. DeepSeek's eight budgets each carry a sweep, so the final line is fit through eight well-located vertices rather than eight noisy guesses.

#### Second-order optimization: the exponents must be internally consistent

A subtle correctness check that catches fitting bugs: the model-scale exponent $a$ and the data-scale exponent $b$ should sum to approximately $1$. Here $0.5243 + 0.4757 = 1.0000$ exactly, which is not a coincidence — DeepSeek constrained the fit so that $M_{\text{opt}} \cdot D_{\text{opt}}$ recovers $C$ by construction. If you fit $a$ and $b$ independently and they sum to, say, $1.08$, you have a bug: your two allocation laws disagree about how much total compute a budget represents, and any prediction you make from them will drift as you extrapolate. Always fit the pair under the constraint $a + b = 1$, or fit one and derive the other. This is the kind of invariant that a naive regression will silently violate.

## 3. The contrarian core: allocation is dataset-dependent

**Rule of thumb: the Chinchilla token-per-parameter ratio is a measurement on someone else's corpus, not a law of nature.** This is the section the title is about. Everything up to here was DeepSeek being careful; this is DeepSeek being contrarian.

![Chinchilla freezes the model/data split; DeepSeek shows the exponents move with your data's quality.](/imgs/blogs/deepseek-llm-scaling-laws-methodology-3.webp)

The before-after above contrasts the two worldviews. On the left, Chinchilla treats the allocation exponents as fixed — roughly $a \approx b \approx 0.5$, which collapses to the famous "about 20 tokens per parameter" heuristic — and applies one curve to every dataset. On the right, DeepSeek treats the exponents as quantities you re-fit per dataset, because they found the exponents move with data quality. The right-hand column is the contrarian claim, and it has a sharp, testable consequence.

### The finding stated precisely

Here is the result in one sentence, because it is easy to get backwards: **with higher-quality training data, the optimal allocation shifts MORE of the budget toward model size and LESS toward data.** Read that twice. The naive intuition runs the other way — "good data is precious, so use more of it" — but the scaling math says the opposite. When the data is cleaner, each parameter you add to the model extracts more signal per token, so the marginal return on parameters rises relative to the marginal return on tokens. The optimizer of the allocation problem responds by pushing $a$ up and $b$ down.

![Cleaning the corpus raises the marginal value of parameters, pulling the optimal split toward model size.](/imgs/blogs/deepseek-llm-scaling-laws-methodology-4.webp)

The graph above traces the causal chain. Start with a raw, noisy corpus. Run deduplication and quality filtering to get a higher-quality corpus. Re-fit the IsoFLOP scaling law on the cleaned data. The marginal value of a parameter rises, which pushes the model-scale exponent $a$ up and pulls the data-scale exponent $b$ down. The net instruction to the engineer: build a bigger model and train it on fewer tokens than the raw-data fit would have told you to. Each box in that chain is a step you can actually run, and the direction of every arrow is the direction DeepSeek reports.

### Why this breaks the Chinchilla mental model

The Chinchilla result was correct for the corpus Chinchilla was fit on. The error is not in their regression; it is in everyone who treated their exponents as transferable. If your data pipeline is meaningfully cleaner or dirtier than DeepMind's circa-2022 web scrape, your optimal split is different, and using Chinchilla's numbers means you are systematically over- or under-sizing your model.

Concretely, suppose you have done serious data work — aggressive dedup, quality classifiers, a curated code and math mixture — and your data is genuinely higher quality than a raw Common Crawl dump. The Chinchilla rule tells you to train a model of size $N$ on $20N$ tokens. The DeepSeek finding says: with your cleaner data, the optimum has moved, and you should train a somewhat larger model on somewhat fewer tokens. If you follow Chinchilla blindly, you leave capability on the table — you spent your data-cleaning effort and then allocated the budget as if you had not.

The reverse failure is just as real. If your data is dirtier than the corpus your borrowed exponents were fit on, Chinchilla's split will tell you to build too big a model and starve it of the tokens it needs to fill that capacity. You will end up with an under-trained large model, which is the most expensive way to get a mediocre result.

| Data quality vs. reference | Borrowed-exponent error | Correct response |
|---|---|---|
| Higher (heavy dedup, curation) | Model too small, over-fed tokens | Shift budget to model; fewer tokens |
| Same as reference | None — exponents transfer | Borrowed exponents are fine |
| Lower (raw scrape, light filtering) | Model too big, starved of tokens | Shift budget to data; smaller model |

### The reusable lesson

This is the heart of the title. The reason "everyone skipped" this methodology is that re-deriving scaling laws is expensive and unglamorous — it is a few weeks of small-model sweeps before the run everyone actually cares about. It is tempting to skip it and trust the published exponents. DeepSeek's contribution is the demonstration that skipping it costs you, in a quantifiable way, proportional to how different your data is from the reference corpus. The more effort you put into your data — and serious teams put enormous effort there — the more the borrowed exponents mislead you, because data quality is exactly the variable the borrowed exponents hold fixed.

> If you have a world-class data team, you have the most to lose from borrowing someone else's allocation exponents. Their numbers assume your data team does not exist.

### A worked allocation shift

Let us put numbers on "the optimum moves." Suppose a reference scaling law, fit on a raw web corpus, gives allocation exponents $a_{\text{ref}} = 0.50$ and $b_{\text{ref}} = 0.50$ — the balanced Chinchilla-style split. You have a fixed budget $C = 10^{21}$ FLOPs. Under the reference exponents, the model scale and data scale grow at the same rate, so you land on a balanced split: roughly $M \propto C^{0.50}$ and $D \propto C^{0.50}$, which for a typical architecture works out to something like a 30B-parameter model on about 600B tokens.

Now suppose you re-fit on your cleaner data and find the exponents have moved to $a = 0.55$ and $b = 0.45$ — a modest five-point shift toward model size, well within the range that data-quality differences can produce. The same budget now allocates more to model and less to data. The model-scale term grows faster ($C^{0.55}$ instead of $C^{0.50}$) and the data term grows slower ($C^{0.45}$). For the same $10^{21}$ budget, the optimum might move to something like a 40B-parameter model on about 450B tokens. That is a 33 percent larger model trained on 25 percent fewer tokens — a materially different run, produced by a five-point exponent shift you would never have discovered if you inherited the reference exponents.

The point of the arithmetic is not the exact numbers, which depend on your architecture's $M$-to-parameter relationship; it is the leverage. Small movements in the exponents, of the size that ordinary data-quality differences produce, translate into large movements in the optimal model size, because the exponent is multiplied against many orders of magnitude of compute. This is the same leverage that made the compute-metric slope so important in section 1, applied now to the allocation slope. Exponents are where the leverage lives, and exponents are exactly what borrowing holds fixed.

#### Second-order optimization: re-fit when the mixture changes, not just the cleaning

A practical extension DeepSeek's framing implies: data quality is not a single scalar you set once. Every time you change the corpus mixture — more code, more math, a new language, a different web-versus-books ratio — you have changed the effective data quality for the loss you care about, and your allocation exponents should in principle move. You do not need to re-run the full eight-budget sweep for every mixture tweak, but you should re-run it whenever the mixture changes enough that you would expect the marginal value of a parameter to shift. A good trigger is "the data team shipped a major pipeline change." That is precisely the moment the inherited exponents become least trustworthy, and precisely the moment teams are most tempted to skip the re-fit because they are eager to start the big run.

## 4. The multi-step learning-rate scheduler

**Rule of thumb: a schedule that you cannot resume is a schedule that locks you into your token budget on day one.** The third contrarian move is the smallest in surface area and the most quietly consequential for how a lab operates over years.

![A step schedule reaches the same final loss as cosine while keeping the curve resumable on new tokens.](/imgs/blogs/deepseek-llm-scaling-laws-methodology-5.webp)

The before-after above puts cosine decay next to DeepSeek's multi-step schedule. Cosine is the default everyone reaches for: warm up, then smoothly anneal the learning rate down to a floor over the course of the run. It works well and reaches a low final loss. Its problem is structural, not numerical: the shape of the cosine curve is parameterized by the total token budget. The decay is computed as a function of "how far through the planned run are we," so the entire curve is baked to a specific endpoint $T$ on the first step.

### The schedule, precisely

DeepSeek's replacement is a step function with two drops:

```python
def multi_step_lr(step, total_steps, max_lr, warmup_steps=2000):
    """Multi-step LR: warm up to max_lr, then hold, then drop twice.

    - Linear warmup over `warmup_steps` to `max_lr`.
    - Hold at max_lr until 80% of total tokens.
    - Drop to 31.6% of max_lr from 80% to 90% of tokens.
    - Drop to 10% of max_lr from 90% to the end.

    31.6% is ~sqrt(0.1): two equal multiplicative steps from 1.0
    down to 0.1, so each plateau is a clean factor-of-~3.16 below
    the previous one.
    """
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    progress = step / total_steps
    if progress < 0.80:
        return max_lr
    elif progress < 0.90:
        return max_lr * 0.316
    else:
        return max_lr * 0.10
```

Two thousand warmup steps to reach the maximum learning rate, a long plateau at the maximum through 80 percent of the tokens, a drop to 31.6 percent of the maximum from 80 to 90 percent, and a final drop to 10 percent of the maximum for the last 10 percent of tokens. The 31.6 percent is not arbitrary: it is approximately $\sqrt{0.1}$, so the two drops are equal multiplicative steps — you go from $1.0$ to $0.316$ to $0.10$, each step dividing by about $3.16$. That geometric spacing is what makes the schedule feel like a clean staircase rather than two ad-hoc cliffs.

DeepSeek reports that this schedule reaches approximately the same final loss as a well-tuned cosine schedule. So on the headline metric, it is a wash. The reason to prefer it is entirely about what happens after the run ends.

### Why a step schedule enables continual pretraining

![Each plateau is a clean re-entry point, so a finished run extends onto fresh tokens without re-warming.](/imgs/blogs/deepseek-llm-scaling-laws-methodology-6.webp)

The timeline above is the payoff. Walk it left to right: the warmup ramp over 2000 steps, the max-LR plateau covering the bulk of the tokens, the step down to 31.6 percent at 80 percent of tokens, the step down to 10 percent at 90 percent, the run ending at a low and stable learning rate, and then — the crucial event — continuing the run by adding more tokens without a cosine re-warm.

This is the operational difference. With cosine, your final learning rate is the bottom of a curve that was shaped for a specific token budget. If you later want to train on more tokens — because you collected more data, or you want a domain-adapted continuation, or you simply want to push the same model further — you cannot cleanly extend the cosine curve. You either restart the decay from a higher learning rate (which re-warms an already-converged model and can damage it) or you stitch on an awkward continuation that the original schedule never anticipated. Practitioners who have tried to extend a cosine run know the pain: the loss spikes when you bump the learning rate back up, and you spend tokens recovering ground you already had.

The step schedule sidesteps this entirely. Because the run ends on a flat plateau at a known fraction of the maximum, the final checkpoint sits at a stable operating point. To continue training on new tokens, you just keep going at that low plateau learning rate, or you add another step. There is no decay shape to honor, no endpoint baked in, no re-warm required. The schedule is, in the paper's framing, built for "longtermism" — the idea that a base model is not a one-shot artifact but something you will extend and adapt over a multi-year horizon.

### The tradeoff table

| Property | Cosine decay | Multi-step schedule |
|---|---|---|
| Final loss | Low (well-tuned) | Approximately equal |
| Curve shape | Tied to fixed token budget $T$ | Plateaus, budget-agnostic |
| Resume on more tokens | Requires awkward re-warm | Continue at plateau LR |
| Checkpoint operating point | Bottom of a curve | Flat, stable plateau |
| Hyperparameters to set | Warmup, $T$, floor | Warmup, two drop points, two factors |
| Best fit for | One-shot runs | Continual / staged pretraining |

The cost of the step schedule is two extra hyperparameters — the drop points and the drop factors — but DeepSeek's choices (80/90 percent, 31.6/10 percent) transfer well and are a reasonable default. You are trading a tiny bit of schedule-tuning surface for a large operational freedom: the ability to treat your base model as a checkpoint you can always extend.

#### Second-order optimization: the plateau is also a better place to branch

A consequence DeepSeek does not dwell on but that falls straight out of the design: because the schedule ends on a stable plateau, it is also a cleaner place to branch the model for fine-tuning or domain adaptation. A cosine-ended checkpoint sits at a learning rate that was about to hit zero, so any fine-tuning has to re-establish a sensible learning rate from scratch, and the optimizer state is shaped by a nearly-dead learning rate. A plateau-ended checkpoint has optimizer statistics consistent with a real, stable learning rate, which makes downstream continuation — whether more pretraining, SFT, or preference optimization — start from a more predictable place. This is a small thing per run and a large thing across the dozens of derivative models a lab spins off from one base.

## 5. Architecture: depth-scaling and GQA for serving cost

**Rule of thumb: choose your scaling axis by what it costs to serve, not by what is easiest to train.** The architecture choices in DeepSeek LLM are an early, legible signal of the cost discipline that defines the whole DeepSeek line.

![Scaling depth and switching to grouped-query attention trades a little training cost for cheaper serving.](/imgs/blogs/deepseek-llm-scaling-laws-methodology-7.webp)

The before-after above contrasts the typical width-scaling move with DeepSeek's depth-plus-GQA choice. When most teams scale a model up, they widen it: increase $d_{\text{model}}$, keep the layer count modest, and let the bigger hidden size carry the extra capacity. Width is friendly to training throughput because wider matmuls have better arithmetic intensity on a GPU. DeepSeek went the other way for the 67B: **95 layers** deep, where the 7B is only 30 layers, and they paired the depth with **Grouped-Query Attention** rather than full Multi-Head Attention.

### Why depth, and why GQA

The 7B uses standard Multi-Head Attention. The 67B uses GQA, in which multiple query heads share a single key-value head, shrinking the key-value cache that dominates memory bandwidth at decode time. That is the tell. GQA costs you a little representational capacity in attention, and it buys you a much smaller KV cache, which is the single biggest lever on serving cost for long-context autoregressive decoding. A team that adds GQA to the larger model and not the smaller one is a team thinking hard about what it costs to run the model in production, not just what it costs to train it.

The depth choice points the same direction, though more subtly. Depth and width are not equivalent ways to add parameters once you account for inference. The specifics of how a given serving stack pipelines and batches favor particular shapes, and DeepSeek's reported preference for depth over width on the 67B is a serving-cost decision dressed up as an architecture decision. The headline result — the 67B beating LLaMA-2 70B on code, math, and reasoning — lands at a lower serving cost than a width-scaled, full-MHA model of comparable quality would.

There is a training-side cost to this choice that is worth naming honestly, because the methodology is about tradeoffs, not free lunches. Wider matmuls have higher arithmetic intensity — more FLOPs per byte moved — so a width-scaled model keeps the GPU's tensor cores busier and trains at higher throughput. A deep, narrow model has lower arithmetic intensity per layer and more sequential layer dependencies, so it is harder to keep the hardware saturated and is more exposed to pipeline bubbles in a pipeline-parallel setup. DeepSeek accepted that training-side friction in exchange for the serving-side win. The calculus is straightforward once you write it down: you pay the training cost once, and you pay the serving cost on every token of every request for the life of the model. For a model that will serve billions of tokens, even a small per-token serving saving dwarfs a large one-time training penalty. A team that optimizes training throughput is optimizing the cheap side of the ledger.

This is also why the choice was made on the 67B and not the 7B. The 7B is small enough that its serving cost is already modest and its full-MHA KV cache is manageable, so the depth-and-GQA tradeoff buys less. The 67B is where serving cost becomes the dominant lifetime expense, so it is where the tradeoff is worth making. Applying the serving-cost optimization selectively, to the model where it pays, rather than uniformly is itself a sign of a team reasoning about cost per model rather than following a single architecture template across sizes.

### How this connects to the later work

If you have read about [Multi-head Latent Attention](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla), you can see the trajectory. GQA in DeepSeek LLM is the first move in a multi-paper campaign to shrink the KV cache. MLA in DeepSeek-V2 is the next, more aggressive move, compressing the KV cache into a low-rank latent. The same instinct — attack the dominant serving cost, accept a small training-side compromise — runs from this 2024 paper straight through to the cost-engineering in [DeepSeek-V3](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing). The 67B's GQA is where you can first see the team optimizing the bill they will pay every day in production rather than the bill they pay once during training.

#### Second-order optimization: depth interacts with the LR schedule

A connection worth drawing between this section and the scheduler section: very deep models are more sensitive to learning-rate transients, because gradient signals traverse more layers and small instabilities compound across depth. A 95-layer network is more likely to be perturbed by an abrupt learning-rate change than a shallower, wider one. The multi-step schedule's geometric drops — equal multiplicative steps of about $3.16$ rather than one large cliff — are gentler on a deep network than a single hard drop would be, and the long max-LR plateau gives a deep model time to settle into a stable regime before any change. The architecture choice and the schedule choice are not independent; the step schedule is part of what makes training a 95-layer model on this budget tractable.

## 6. The IsoFLOP fitting setup and the spec sheet

**Rule of thumb: the credibility of a scaling law is the number of well-separated compute budgets it was fit across, not the size of the final model.** Let us pull the whole methodology together into the experimental scaffold and the production spec sheet it produced.

![Eight compute budgets anchor the power-law fits that then predict the 7B and 67B production configs.](/imgs/blogs/deepseek-llm-scaling-laws-methodology-8.webp)

The grid above is the scaffold on the left and the spec sheet on the right. On the left: a floor of $10^{17}$ FLOPs, a ceiling of $3 \times 10^{20}$ FLOPs, eight log-spaced budgets in between. For each budget you sweep the model-versus-data tradeoff, find the bottom of the loss valley, and read off the optimum. Those eight optima feed the power-law fits for $B$, $\eta$, $M$, and $D$. The fits then predict the optimal configuration at any budget, including the frontier budgets used for the production models. On the right is the result: the 7B with 30 layers, MHA, and a 4096-token context; the 67B with 95 layers, GQA, and the same 4096 context; both trained on the same 2 trillion tokens of mixed English and Chinese.

### Why eight budgets over three decades is enough

The statistical logic is simple and worth internalizing. A power law $y = k C^{\alpha}$ is a straight line $\log y = \alpha \log C + \log k$ in log-log space. Fitting a line needs leverage along the x-axis, and leverage comes from spread, not from count. Eight points spread across more than three orders of magnitude of compute pin down the slope $\alpha$ tightly, because the lever arm is long. Twenty points crammed into a single order of magnitude would be worse, despite the larger count, because they give the regression almost no horizontal leverage. This is why the $10^{17}$-to-$3 \times 10^{20}$ span matters as much as the "eight budgets" number — the range is the leverage.

### The extrapolation gap, stated honestly

There is an uncomfortable truth in every scaling-law paper, and this one is no exception: you fit at $10^{17}$ to $3 \times 10^{20}$ FLOPs and then you trust the fit out at the frontier budget used to train the 67B, which is larger. That is an extrapolation, and extrapolation is where scaling laws have historically broken. DeepSeek's defense is the one available to anyone: the fits are clean, the exponents are internally consistent ($a + b \approx 1$), the compute axis is honest ($M \cdot D$, not $6ND$), and the predicted optima land where the production models actually performed well. None of that proves the extrapolation is safe — it argues that the fit is as trustworthy as a fit can be before you spend the frontier budget. Which is exactly the point of the methodology: you do the cheap thing carefully so the expensive thing is as de-risked as you can make it.

The discipline that makes an extrapolation defensible is worth spelling out as a checklist, because it is the part of the methodology you can actually adopt tomorrow regardless of your budget. First, fit across enough orders of magnitude that the slope is constrained by leverage, not by hope — a span of three or more decades, not a factor of two. Second, sample each budget's allocation valley densely enough on both sides of the minimum that the vertex is well-located. Third, use a compute axis that does not vary its bias with model size, which means $M \cdot D$ rather than $6ND$. Fourth, constrain the allocation exponents to be internally consistent so your two laws cannot disagree about what a budget means. Fifth, and most importantly, validate the predicted optima against the production models once you have them, and treat any mismatch as a signal that the extrapolation is straining rather than as noise to be ignored. A team that does all five has earned its extrapolation; a team that fits two close budgets and projects three decades out has not.

### Alignment: SFT then DPO, the pre-GRPO era

The base models are the story, but the paper also describes alignment, and it is worth placing in time. DeepSeek LLM aligned with supervised fine-tuning on over a million instances — roughly 1.5 million — followed by Direct Preference Optimization. This is the **pre-GRPO** era of DeepSeek's alignment. The reinforcement-learning machinery that DeepSeek later became famous for did not exist yet here; that matures in DeepSeek-V2 and beyond. If you want the DPO half of this story in depth, the [fine-tuning with DPO post](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) on this blog covers the preference-optimization mechanics that DeepSeek LLM used. Reading this paper, you can see a team that had nailed the pretraining methodology and was still using a fairly standard SFT-plus-DPO alignment stack — the alignment innovation came later, once the base-model foundations were solid.

The sequencing here is itself a lesson in where a team spends its innovation budget. DeepSeek LLM is a paper that is radical about pretraining and conservative about alignment. The SFT-then-DPO recipe was, in early 2024, the well-understood default: collect a large instruction-following set, fine-tune on it to teach the model to follow instructions, then apply a preference-optimization pass to sharpen helpfulness and reduce harmful outputs. DPO in particular was attractive because it skips the separate reward model that classic RLHF requires, turning preference learning into a direct supervised objective on pairs of preferred and rejected responses. For a team whose differentiating bet was the pretraining methodology, using a battle-tested alignment stack was the right call — you do not want two simultaneous sources of novelty when you are trying to attribute a result. The pretraining laws were the experiment; the alignment was deliberately boring so the experiment stayed legible.

It is also worth noting what the roughly 1.5 million SFT instances signal about the scale of the alignment effort even in this conservative phase. A million-plus curated instances is a serious data-collection operation, comparable in effort to the data-pipeline work that made the pretraining corpus high quality in the first place. The same data discipline that moved the allocation exponents toward model size shows up again in the alignment set. A team that takes data quality seriously takes it seriously at every stage, and the through-line from the pretraining corpus to the SFT set to the DPO preference pairs is one of the quieter consistencies of the paper. The RL-heavy pipelines that came later did not replace that data discipline; they built on top of it.

| Stage | DeepSeek LLM (2024) | What it became later |
|---|---|---|
| Pretraining | $M \cdot D$ laws, step LR, 2T tokens | Refined through V2/V3 |
| Architecture | GQA on 67B, depth-scaling | MLA, MoE in V2/V3 |
| Alignment | SFT (~1.5M) + DPO | GRPO and RL-heavy pipelines |

## Case studies from the methodology

These are not incidents from a single production system; they are scenarios where applying — or failing to apply — DeepSeek's methodology changes the outcome. Each one is concrete and the kind of thing you will actually face if you train base models.

### 1. The borrowed-exponent over-spend

A team with a strong data pipeline — heavy dedup, a trained quality classifier, a curated code-and-math mixture — budgets a 30B-parameter run using the Chinchilla rule: 30B parameters, about 600B tokens. The run finishes and the model is good but not as good as the budget should have bought. The wrong first hypothesis is that the architecture is suboptimal, and the team spends a month trying attention variants. The actual root cause is allocation: their data is cleaner than the corpus Chinchilla was fit on, so the optimal split had moved toward model size. They should have trained a larger model on fewer tokens. The fix is to re-fit the allocation exponents on their own data with an eight-budget IsoFLOP sweep, which costs a few weeks of small-model compute and reveals that their optimum is closer to 40B parameters on 500B tokens. The lesson: when a well-resourced data team underperforms its budget, suspect allocation before architecture. The very effort that makes your data good is the effort that invalidates the borrowed exponents.

### 2. The cosine re-warm that ate a week

A team trains a 13B base model to a cosine schedule, ships it, and three weeks later collects 300B more tokens of high-quality domain data they want to fold in. They reload the final checkpoint and resume training, bumping the learning rate back up to something reasonable for continued pretraining. The loss immediately spikes and takes most of a week of tokens to recover to where it started. The wrong first hypothesis is a data bug in the new tokens. The actual root cause is the re-warm: the model had converged at the bottom of a cosine curve, and raising the learning rate perturbed a settled optimization state. Had the original run used a multi-step schedule ending on a stable 10-percent plateau, the continuation would have started at that plateau learning rate with no spike. The lesson: the schedule you pick on day one of a base-model run silently determines how expensive every future extension of that model will be. DeepSeek's step schedule is a bet that you will extend the model, and that bet almost always pays.

### 3. The 6ND mirage in the small-model sweep

A team runs an IsoFLOP sweep to fit their own scaling laws, but they account compute with $6ND$. Their small models — 50M to 200M parameters — have large embedding tables relative to their total parameter count, sometimes 40 percent of $N$. The $6ND$ metric counts those embedding parameters as full-cost compute, inflating the compute estimate for the small models unevenly. The fitted allocation law comes out subtly wrong, biased by the small-scale points whose compute was most over-counted. The wrong first hypothesis is that they need more budgets to stabilize the fit. The actual root cause is the compute axis: switching to $M \cdot D$, which excludes the embedding lookup and includes the attention term, straightens out the small-scale points and tightens the regression with the same eight budgets. The lesson: when a scaling-law fit is noisy at the small end, suspect the compute metric before adding more runs. The small models are where embedding overhead and attention overhead are largest relative to the dense matmul cost, so they are where $6ND$ lies the most.

### 4. The width-scaled serving bill

A team scales their 7B up to 65B by widening — bigger $d_{\text{model}}$, modest depth, full Multi-Head Attention — because width trains faster and the throughput numbers during pretraining look great. The model is strong. Then it hits production, and the serving cost is brutal: the full-MHA KV cache at a long context length dominates memory bandwidth, and the per-token decode cost is far higher than a competitor's similarly-capable model. The wrong first hypothesis is that they need better serving infrastructure. The actual root cause is the architecture: they optimized the training bill and ignored the serving bill, which they pay every day forever. DeepSeek's choice — depth plus GQA on the larger model — trades a little training throughput for a much smaller KV cache and a lower steady-state serving cost. The lesson: for any model you will serve at scale, the KV cache is the cost center, and attention-variant and depth-versus-width choices should be made against the serving bill, not the training throughput.

### 5. The mixture change that moved the optimum

A team has a solid scaling-law fit and a working base-model recipe. Then the data team ships a major pipeline upgrade: a much larger, cleaner code and math fraction in the mixture. The next base-model run uses the old allocation exponents, because re-fitting feels like a delay, and the team is eager to start the big run. The resulting model underperforms on exactly the code and math benchmarks the new data was supposed to improve. The wrong first hypothesis is that the new data is somehow worse. The actual root cause is that the mixture change raised the effective data quality for the target loss, which moved the optimal allocation toward model size — and the old exponents, fit on the previous mixture, now under-size the model. The fix is a re-fit triggered by the mixture change, not by the calendar. The lesson: a major data-pipeline change is precisely the event that invalidates your inherited allocation exponents, and precisely the moment teams are most tempted to skip the re-fit. Treat "the data team shipped a big change" as a hard trigger to re-derive.

### 6. The learning-rate law that saved a manual sweep

A team scaling from a $10^{19}$ budget to a $10^{20}$ budget — a 10x jump — plans to manually sweep the learning rate at the new scale, which is several runs of wasted compute. Instead they apply the fitted law $\eta_{\text{opt}} = 0.3118 \cdot C^{-0.125}$. At $10^{19}$ the law gives about $1.3 \times 10^{-3}$; at $10^{20}$ it gives about $9.9 \times 10^{-4}$. The learning rate should drop by roughly 25 percent across the 10x compute increase — a small, predictable change. The team uses the predicted value directly and the run is stable on the first try. The wrong instinct they avoided was treating each new scale as a fresh tuning problem. The actual insight is that once you have fit the law on your data, the optimizer hyperparameters are determined, not searched. The lesson: the payoff of fitting the $B$ and $\eta$ laws is that every subsequent scale-up stops being a hyperparameter search and becomes a lookup. The fitting cost amortizes across every run you do afterward.

### 7. The extrapolation that held, and the discipline behind it

A team fits scaling laws across $10^{17}$ to $3 \times 10^{20}$ FLOPs and then trains a frontier model at a budget above that range, trusting the extrapolation. It works — the model lands where the laws predicted. The reason it worked is not luck; it is that the team did four things right before extrapolating: they used an honest compute axis ($M \cdot D$), they fit across enough orders of magnitude that the slope was well-constrained, they constrained the allocation exponents to be internally consistent ($a + b \approx 1$), and they checked that the predicted optima matched the production models' performance. The wrong way to do this is to fit two budgets close together and extrapolate three decades out. The lesson: extrapolation is unavoidable in scaling-law-driven design, but its safety is bought entirely by the rigor of the fit beneath it. You cannot make the extrapolation safe after the fact; you make it safe by doing the cheap fitting work carefully before you spend the frontier budget.

### 8. The vocabulary swap that did not move the budget

A team is experimenting with tokenizers and swaps a 32K-vocab tokenizer for a 128K-vocab one to improve compression on code and non-English text. Under their old $6ND$ accounting, the larger embedding table inflates $N$, and their allocation tooling reacts by recommending fewer training tokens — as if the model got more expensive. But the model did not get more expensive to run per token; the embedding lookup is still nearly free. The wrong outcome they almost shipped was a token-starved run justified by phantom compute. Because they had switched to $M \cdot D$, which excludes the embedding table, the larger vocabulary left $M$ essentially unchanged and the recommended token budget held steady. The lesson: an honest compute metric decouples decisions that a dishonest one entangles. Tokenizer choice and token-budget choice should be independent, and they only are if your compute axis ignores the parameters that do no arithmetic.

### 9. The batch-size law that prevented a throughput cliff

A team scales a run and, lacking a fitted batch-size law, copies the batch size from a much smaller prior run because it "worked before." At the new, larger compute budget the copied batch is far too small relative to $B_{\text{opt}} = 0.292 \cdot C^{0.327}$, so each step processes too few tokens, the gradient is noisier than it should be, and the GPUs spend a large fraction of their time in communication overhead between tiny steps rather than in dense matmul. Throughput craters and the run takes weeks longer than budgeted. The wrong first hypothesis is a hardware or interconnect problem, and the team spends days profiling NCCL. The actual root cause is that the optimal batch size grows with compute, and a batch sized for a $10^{18}$ run is badly undersized for a $10^{20}$ run — the law predicts roughly a $10^{(0.327 \times 2)} \approx 4.6\times$ larger batch across that 100x compute jump. The fix is to size the batch from the fitted law. The lesson: batch size is not a constant you carry across scales; it is a power law, and the penalty for ignoring that is paid in throughput, which looks like an infrastructure problem and is actually an allocation problem.

### 10. The longtermism bet that paid off three models later

A lab trains a base model with the multi-step schedule even though their immediate plan is a single release, accepting the two extra schedule hyperparameters for no immediate benefit. Eighteen months later, that base model has spawned a domain-adapted medical variant, a code-specialized continuation, and a long-context extension — each built by continuing pretraining from the plateau-ended checkpoint, none of them requiring a cosine re-warm or a from-scratch run. A competing lab that used cosine for its base model has to either restart each derivative from a re-warmed checkpoint, eating loss spikes each time, or retrain from scratch. The wrong framing the first lab avoided was treating the schedule choice as a property of one run. The actual insight is that the schedule choice is a property of the base model's entire downstream lineage, decided irreversibly on the first run. The lesson: the resumability of the step schedule is an option you buy cheaply at training time and exercise repeatedly over a model's life. The longtermism in the paper's title is not a slogan; it is a concrete bet that base models are extended, and that bet compounds with every derivative model a lab ships.

## When to reach for this methodology, and when not to

**Rule of thumb: re-derive scaling laws when the cost of the run dwarfs the cost of the sweep, and inherit them when it does not.**

### Reach for the full DeepSeek methodology when

- **You are about to spend a frontier budget.** When the production run costs hundreds of thousands of dollars or more, a few weeks of small-model IsoFLOP sweeps to de-risk the allocation is cheap insurance. The sweep cost is a rounding error against the run cost.
- **Your data is meaningfully different from the reference corpus.** If your data pipeline is much cleaner (heavy dedup, quality classifiers, curated mixtures) or much dirtier than the corpus the published exponents were fit on, the inherited split is wrong in a direction you can predict and a magnitude you cannot. Re-fit.
- **You will extend or stage the model over time.** If the base model is the start of a multi-year campaign of continual pretraining, domain adaptation, and derivative models, the multi-step schedule's resumability is worth its two extra hyperparameters many times over.
- **You serve the model at scale.** If inference is your dominant cost, the depth-versus-width and GQA-versus-MHA choices should be made against the serving bill, which means measuring the KV-cache and decode cost, not just the training throughput.
- **You change the data mixture often.** Frequent mixture changes mean frequent shifts in effective data quality, which means the allocation optimum keeps moving. A team that re-fits on a trigger stays calibrated; a team that inherits drifts.

### Skip it and inherit published laws when

- **The run is small and cheap.** If you are training a 1B model for an internal experiment, the IsoFLOP sweep can cost more than the run it is meant to optimize. Borrow Chinchilla, accept a slightly suboptimal split, and move on.
- **Your data closely matches a published reference.** If you are training on a corpus genuinely similar to the one a published scaling law was fit on, the exponents transfer and re-fitting buys you little. The methodology pays off in proportion to how different your setup is.
- **You are fine-tuning, not pretraining.** Scaling laws for pretraining allocation do not govern a LoRA or full fine-tune on a fixed base model. Different problem, different rules; do not bring the IsoFLOP machinery to a fine-tuning job.
- **You only ever train one-shot models.** If you genuinely never extend a base model — every run is final and standalone — the resumability of the step schedule is a feature you will never use, and a well-tuned cosine is simpler. (In practice almost no serious lab is in this regime, but some product teams legitimately are.)
- **You lack the compute for a real sweep.** A two-budget "sweep" extrapolated three decades out is worse than honestly borrowing a well-established law. If you cannot afford enough budgets spread across enough orders of magnitude to constrain the slope, do not pretend to have fit a law. Inherit one and be honest that you did.

## Further reading

- **DeepSeek LLM: Scaling Open-Source Language Models with Longtermism** (arXiv 2401.02954) — the source paper. Read sections 3 and 4 for the scaling-law methodology and the IsoFLOP details, and section 5 for the architecture and alignment specifics.
- **Training Compute-Optimal Large Language Models** (Chinchilla, Hoffmann et al., 2022) — the reference point DeepSeek is arguing with. Read it for the fixed-exponent allocation that DeepSeek shows is dataset-dependent.
- **Scaling Laws for Neural Language Models** (Kaplan et al., 2020) — where the $6ND$ compute metric and the original power-law framing come from. DeepSeek's $M \cdot D$ is a direct refinement of this.
- [DeepSeek-V3: FP8, MTP, and loss-free balancing](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing) — where the same team's cost discipline, first visible in the 67B's GQA, reaches its full expression.
- [Multi-head Latent Attention](/blog/machine-learning/large-language-model/multi-head-latent-attention-mla) — the next move in the KV-cache-shrinking campaign that GQA on the 67B started.
- [Fine-tuning LLMs with DPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) — the preference-optimization half of DeepSeek LLM's pre-GRPO alignment stack, in depth.

---
title: "Kaplan vs Chinchilla: how a parameter-counting bug split the field"
date: "2026-06-15"
description: "Trace the famous 0.73-vs-0.50 scaling-exponent disagreement to its root cause, learn why it was a measurement artifact rather than a real law, and walk away with the bookkeeping discipline that prevents the next one."
tags: ["scaling-laws", "kaplan", "chinchilla", "compute-optimal", "parameter-counting", "large-language-model", "pretraining", "deepmind", "openai", "replication", "machine-learning"]
category: "machine-learning"
subcategory: "Scaling Laws"
author: "Hiep Tran"
featured: true
readTime: 52
---

For about two years, the most consequential number in large-model training was wrong, and almost nobody noticed because it was wrong in a way that looked exactly like a discovery.

The number was an exponent. Kaplan and collaborators at OpenAI published it in early 2020: given a fixed compute budget, the optimal model size grows like $N_{\text{opt}} \propto C^{a}$ with $a \approx 0.73$. Read that exponent literally and it tells you something specific and actionable — when you get more compute, spend most of it on a bigger model and relatively little of it on more training data. That advice shaped a generation of flagship models. GPT-3 was 175 billion parameters trained on roughly 300 billion tokens, a ratio of under two tokens per parameter. Megatron-Turing NLG was 530 billion parameters. Gopher was 280 billion. The field had internalized the rule: scale parameters hard, feed them a comparatively thin diet of data, and stop training well before convergence.

Then in March 2022, DeepMind's Chinchilla paper measured the same quantity and got a different answer: $a \approx 0.50$. Not a little different. Half versus three-quarters is a qualitative disagreement. A slope of 0.50 says the opposite of a slope of 0.73 — it says split your marginal compute roughly evenly between parameters and tokens, which works out to about twenty training tokens per parameter at the optimum. Under that rule, a 70-billion-parameter model trained on 1.4 trillion tokens should beat a 280-billion-parameter model trained on 300 billion tokens *at the same training compute*. DeepMind built exactly that model, named it Chinchilla, and it beat Gopher, GPT-3, and MT-NLG across a broad evaluation suite. The conclusion was blunt: most of the famous large models of that era were severely undertrained.

So we had two careful, well-resourced labs, fitting the same functional form to the same kind of data, arriving at exponents that recommend opposite engineering strategies. This is the detective story of how that happened, who solved it, and what the resolution actually was. The short version, which I will spend the rest of this post earning: Chinchilla's exponents are right, Kaplan's were a measurement artifact, and the single largest cause was a bookkeeping choice nobody flagged at the time — whether you count a model's *non-embedding* parameters or its *total* parameters. The diagram below is the mental model for the whole piece: one question, two fitted answers, two opposite recommendations, and two very different models built on each.

![Flow chart showing one fixed compute budget splitting into a Kaplan branch with exponent near 0.73 leading to model-heavy GPT-3 and a Chinchilla branch with exponent near 0.50 leading to the balanced Chinchilla model](/imgs/blogs/kaplan-vs-chinchilla-reconciliation-1.png)

The thing to hold onto before we dig in: both camps agreed that bigger helps and that loss follows clean power laws over many orders of magnitude. They did not disagree about whether scaling works. They disagreed about how to split a fixed budget — and that disagreement turned out to be downstream of arithmetic, not nature.

I want to be precise about why this is a *detective story* and not a *takedown*. There is no villain here. Kaplan and collaborators did careful, field-defining work; the single-variable scaling laws in that paper are excellent and have held up beautifully. Chinchilla did careful, field-correcting work; its central conclusion is one of the most important empirical results in the history of the field. The bug was not in either team's competence. It was in a convention — a choice about what to count — that sat in the seam *between* the two papers, invisible to each because each was internally consistent. That is the most dangerous kind of bug there is: not the kind that throws an error or fails a test, but the kind where every individual step is correct and the conclusion is still wrong because two correct steps used incompatible definitions. Those bugs do not announce themselves. You find them by going back to the definitions and checking that they match, which is exactly the unglamorous work that the reconciliation papers did four years later. The whole point of telling this story carefully is to build the instinct to do that checking *before* you spend the budget, not after.

## Why this dispute is different from a normal disagreement

Most disagreements in machine learning are about modeling choices: which architecture, which optimizer, which dataset, which evaluation. Those are arguments about taste and tradeoffs, and they rarely have a single correct answer. The Kaplan-Chinchilla split is not that kind of disagreement, and confusing it for one is the first mistake. It is a disagreement about a *measured constant of the system*, and a measured constant has a right answer that does not depend on your taste.

Here is the table I wish I had been handed in 2022, when I was still trying to reconcile the two papers in my head and assuming they must have studied subtly different regimes.

| Question | The "two valid regimes" story | The reality |
|---|---|---|
| Did the two labs study different model families? | Yes, OpenAI and DeepMind used different architectures, so the laws differ | Architecture details barely move the exponent; both papers say so |
| Is 0.73 right for some settings and 0.50 for others? | Maybe one holds at small scale, one at large | No — 0.50 holds across scale; 0.73 was a fitting artifact at small scale |
| Was Chinchilla simply a better-trained model? | It is a separate empirical win, unrelated to the exponent | The exponent *is* why Chinchilla works: balanced allocation |
| Are the exponents within noise of each other? | Probably, given fit uncertainty | No — the gap is far larger than the (genuinely real) fit uncertainty |
| Was someone careless? | One of the labs must have made a mistake | Both did careful work; the bug was a subtle, shared-blind-spot convention |

The reason this matters beyond historical curiosity is that the *mechanism* of the error is completely general. It was not a typo. It was not a bad random seed. It was a decision about what to count, made implicitly, never written in big letters at the top of either paper, and never reconciled between them until 2024. If you fit scaling laws today — for data mixtures, for precision, for inference-aware training, for anything — you are one unstated convention away from publishing your own 0.73. That is why this is worth eleven thousand words.

If you want the upstream context for *why* loss is predictable in the first place, and why a straight line on log-log axes is the whole game, the foundations post in this series builds that from scratch: [scaling laws, from scratch](/blog/machine-learning/scaling-laws/scaling-laws-predictability-foundations). The two papers at the center of this story each get a dedicated deep-dive: [Kaplan 2020](/blog/machine-learning/scaling-laws/kaplan-scaling-laws-language-models) and [Chinchilla 2022](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling). This post assumes you know the headline of each and focuses entirely on why they disagreed and how it was resolved.

> The exponent is not a matter of opinion. It is a measured constant, and a measured constant has a correct value. The disagreement was a measurement error, and measurement errors are findable.

## The two laws, stated precisely

Let me put both claims on the table in their own notation so we can see exactly where they touch.

Both papers build on the same backbone idea: test loss $L$ (in nats per token) is a smooth function of model size $N$ and training tokens $D$. Chinchilla writes the now-canonical parametric form

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},$$

where $E$ is the irreducible loss — the entropy floor of the data, the part you cannot remove no matter how big the model or how much data — and $A$, $B$, $\alpha$, $\beta$ are fitted constants. The two power-law terms say model size and data each buy you loss reduction at their own diminishing rate. Training compute is approximately $C \approx 6ND$ FLOPs, the standard estimate of six FLOPs per parameter per token (two for the forward multiply-accumulate, four for the backward pass).

Kaplan's single-variable laws look like this:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076, \quad N_c \approx 8.8 \times 10^{13},$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095, \quad D_c \approx 5.4 \times 10^{13},$$

and a compute law $L(C_{\min}) = (C_c / C_{\min})^{\alpha_C}$ with $\alpha_C \approx 0.050$. From these, Kaplan derived the compute-optimal allocation that became the headline:

$$N_{\text{opt}} \propto C^{a}, \quad a \approx 0.73, \qquad D_{\text{opt}} \propto C^{b}, \quad b \approx 0.27.$$

Chinchilla's three independent methods all landed near $a \approx b \approx 0.50$. Approach 1 (fix model size, vary tokens, take the minimum over training curves) gave $a \approx 0.50$. Approach 2 (IsoFLOP profiles — fix the FLOP budget, sweep model size, find the loss-minimizing $N$ at each of nine budgets) gave $a \approx 0.49$. Approach 3 (fit the parametric $L(N,D)$ above directly) gave $a \approx 0.46$. The fitted constants for Approach 3, in the original paper, were $E = 1.69$, $A = 406.4$, $\alpha = 0.34$, $B = 410.7$, $\beta = 0.28$. The key feature is $\alpha \approx \beta$: if the two power-law terms decay at nearly the same rate, the optimal allocation balances $N$ and $D$, and the exponent comes out near one-half.

There is a clean way to see why $\alpha \approx \beta$ forces $a \approx 0.5$. At the compute-optimal point you minimize $L(N, D)$ subject to $6ND = C$. Setting up the Lagrangian and taking derivatives, the optimum satisfies $\alpha A / N^{\alpha} = \beta B / D^{\beta}$ at the constrained minimum, and the resulting scaling of $N$ with $C$ is

$$a = \frac{\beta}{\alpha + \beta}, \qquad b = \frac{\alpha}{\alpha + \beta}.$$

Plug in $\alpha = \beta$ and you get $a = b = 0.5$ exactly. Plug in Chinchilla's numbers, $\alpha = 0.34$ and $\beta = 0.28$, and you get $a = 0.28/0.62 \approx 0.45$ and $b \approx 0.55$. The whole disagreement, then, reduces to one question: is $\alpha$ genuinely about equal to $\beta$, or is one of them much larger? Kaplan's implied ratio gives $a \approx 0.73$, which requires $\beta / \alpha \approx 0.73/0.27 \approx 2.7$ — the data term decaying far faster than the model term, so data "saturates" quickly and you should pour compute into parameters. Chinchilla says no, they decay at nearly the same rate.

So when we hunt for the bug, we are really hunting for why Kaplan's fit implied a data exponent so much larger than the model exponent. Hold that framing; it is the thread that runs through every cause below.

### The full derivation of the allocation exponent

Because the relation $a = \beta/(\alpha+\beta)$ is doing so much work in this story, let me derive it in full rather than assert it, so you can see precisely which assumptions feed the result. We want to minimize the loss

$$L(N, D) = E + A N^{-\alpha} + B D^{-\beta}$$

subject to the compute constraint $C = 6 N D$, holding $C$ fixed. Form the Lagrangian with multiplier $\lambda$:

$$\mathcal{L} = E + A N^{-\alpha} + B D^{-\beta} + \lambda (6ND - C).$$

Take partials and set them to zero:

$$\frac{\partial \mathcal{L}}{\partial N} = -\alpha A N^{-\alpha-1} + 6\lambda D = 0, \qquad \frac{\partial \mathcal{L}}{\partial D} = -\beta B D^{-\beta-1} + 6\lambda N = 0.$$

Divide the first stationarity condition by the second to eliminate $\lambda$ and the factor of six:

$$\frac{\alpha A N^{-\alpha-1}}{\beta B D^{-\beta-1}} = \frac{D}{N} \;\Longrightarrow\; \alpha A N^{-\alpha} = \beta B D^{-\beta}.$$

That last equation is the heart of it: at the compute-optimal point, the *marginal* loss reduction from the model term equals the marginal loss reduction from the data term, each weighted by its exponent. Now substitute the constraint $D = C/(6N)$ and solve for how $N$ scales with $C$. Writing $D = C/(6N)$ in $\alpha A N^{-\alpha} = \beta B D^{-\beta}$ gives

$$\alpha A N^{-\alpha} = \beta B \left(\frac{C}{6N}\right)^{-\beta} = \beta B (6N)^{\beta} C^{-\beta}.$$

Collect the $N$ terms on one side: $N^{-\alpha} / N^{\beta} = N^{-(\alpha+\beta)}$, so

$$N^{-(\alpha+\beta)} = \frac{\beta B \, 6^{\beta}}{\alpha A} \, C^{-\beta} \;\Longrightarrow\; N \propto C^{\beta/(\alpha+\beta)}.$$

There it is: $a = \beta/(\alpha+\beta)$, and by symmetry $b = \alpha/(\alpha+\beta)$, with $a + b = 1$ exactly (which makes sense — your compute is fully allocated between $N$ and $D$). The only inputs are the two power-law exponents. Not the coefficients $A$ and $B$, not the floor $E$, not the constant in $C = 6ND$. This is why the entire disagreement collapses to the ratio $\beta/\alpha$, and why pinning that ratio down is the whole game. A measurement that biases $\alpha$ and $\beta$ unequally — which is exactly what a scale-dependent miscount of $N$ does, since it distorts the $N$ axis but not the $D$ axis — will distort $a$ directly.

### Why all three Chinchilla methods agree, and Kaplan's one method did not

Chinchilla's robustness comes from attacking the exponent three structurally different ways. It is worth understanding why they agree, because the agreement is the actual evidence.

Approach 1 fixes a handful of model sizes, trains each on increasing tokens, and for each compute budget reads off which model size achieved the lowest loss at that budget. It never fits the parametric form at all — it reads the optimum directly off the empirical training curves. Approach 2, the IsoFLOP method, fixes nine compute budgets and, for each, sweeps model size, fitting a parabola in $\log N$ to the resulting losses to find the minimizing size; the locus of those minima across budgets gives the exponent. Approach 3 fits the full $L(N, D)$ surface and computes the optimum analytically via the derivation above. These three share almost no machinery: the first uses no curve-fitting, the second fits one-dimensional parabolas, the third fits a four-parameter surface. A systematic error in one would not generally appear in the others. Their agreement at $a \approx 0.46$–$0.50$ is therefore strong evidence the value is real and not an artifact of any single fitting choice.

Kaplan, by contrast, leaned on a single derivation path from single-variable laws, and that path inherited the parameter-counting convention at every step. One method with a shared upstream bias gives one biased answer with no internal contradiction to flag the problem. That is the deeper methodological lesson of the comparison: it is not that Kaplan used worse statistics, but that a single coherent method cannot detect its own systematic error, while a triangulation of independent methods can.

## The picture of the disagreement

Before the causes, look at the shape of the thing. The cleanest way to see why this disagreement was so destabilizing is to draw both fitted frontiers on the same log-log axes, because that is where a power law becomes a straight line and the exponent becomes a slope you can read with a ruler.

![Log-log chart with two straight frontier lines rising from a shared small-scale fit start, the steeper line labeled Kaplan slope near 0.73 and the shallower dashed line labeled Chinchilla slope near 0.50, with the gap between them widening to the right](/imgs/blogs/kaplan-vs-chinchilla-reconciliation-2.png)

The figure above is the crux of the story in one image. Both lines anchor at the same small-scale fit region — that shared starting point matters enormously, as we will see — and both are honest straight lines on log-log axes, which is to say both are genuine power laws. The only difference is the slope. The steeper line is Kaplan: every decade of compute justifies a larger jump in parameters. The shallower dashed line is Chinchilla: the same decade of compute justifies a smaller jump in parameters, with the rest going to tokens. And here is the part that made this so expensive in practice: *the gap grows with compute*. At small scale, where both laws were fit, the two recommendations are nearly indistinguishable — a few hundred million parameters either way. At GPT-3 scale, four or five orders of magnitude of compute later, the slopes have diverged into completely different models. A measurement error you cannot see in your fit region becomes a chasm when you extrapolate.

This is the general hazard of power-law extrapolation, and it is worth stating as a rule: a small error in a slope is invisible near the anchor and catastrophic far from it. The same property that makes scaling laws useful — you can fit cheaply at small scale and extrapolate to large scale — is exactly what makes a slope error so dangerous. You do not find out you were wrong until you have spent the big budget.

Now, the detective work. There turned out to be not one cause but a small ordered list of them, and two independent 2024 papers found largely the same list. Let me take them in order of how much of the gap each explains.

## Culprit 1: non-embedding versus total parameters

**The senior rule of thumb: before you fit anything, write down precisely what your variables count. The most expensive bugs live in the definitions, not the math.**

This is the dominant cause, identified most sharply by Pearce and Song in their 2024 paper "Reconciling Kaplan and Chinchilla Scaling Laws." Here is the entire thing in one sentence: Kaplan counted *non-embedding* parameters as $N$, and Chinchilla counted *total* parameters. That sounds like a footnote. At Kaplan's experimental scale, it is the whole ballgame.

Consider what is in a transformer's parameter budget. There are the transformer blocks — the attention projections and the feed-forward MLPs — whose parameter count scales roughly as $12 \cdot n_{\text{layers}} \cdot d_{\text{model}}^2$, growing with the *square* of the hidden dimension. And there is the token embedding matrix (and, separately or tied, the output unembedding), whose size is $d_{\text{model}} \cdot V$ where $V$ is the vocabulary, typically 32k to 50k. The embedding grows only *linearly* in $d_{\text{model}}$. So for a large model with $d_{\text{model}}$ in the thousands, the quadratic block term dwarfs the linear embedding term and embeddings are a rounding error. But for a small model — and Kaplan's grid ran from a hidden size of 768 up to about 1.5 billion total parameters — the embedding matrix is a *large fraction* of all the weights.

Work a concrete number. Take a small model with $d_{\text{model}} = 768$, twelve layers, and a 50,257-token vocabulary (GPT-2's). The transformer blocks hold roughly $12 \cdot 12 \cdot 768^2 \approx 85$ million parameters. The embedding matrix holds $768 \cdot 50257 \approx 39$ million parameters, and if the unembedding is untied that is another 39 million. So total parameters are about $85 + 39 + 39 \approx 163$ million, of which the non-embedding count is only 85 million — barely over half. The two definitions of $N$ differ by nearly a factor of two for this model. Now scale up to $d_{\text{model}} = 8192$: blocks are around $12 \cdot 80 \cdot 8192^2 \approx 64$ billion, embeddings around $8192 \cdot 50257 \approx 0.4$ billion. The two definitions now agree to better than one percent.

![Two stacked column comparison contrasting Kaplan non-embedding parameter counting against Chinchilla total parameter counting, listing for each how the count is built and how the fitted slope comes out near 0.73 versus near 0.50](/imgs/blogs/kaplan-vs-chinchilla-reconciliation-4.png)

The before-and-after above lays out the two counting conventions side by side. The reason this scrambles the exponent is subtle and worth slowing down on. When you regress loss against compute $C \approx 6ND$, you need a relationship between $N$ and $C$. If $N$ is your *non-embedding* count but the *actual* compute you spent (and the actual model you trained) included the embeddings, then your $N$-versus-$C$ relationship is non-linear and scale-dependent in your fit region. At small scale the embeddings eat a big, variable share of the model that your $N$ does not see; at large scale they do not. The fit "sees" the model getting more parameter-efficient than it really is as you scale up, because more and more of the real parameters are the ones you decided not to count. That apparent efficiency gain shows up in the fit as: bigger models pay off more than they should, so allocate more compute to size. Out pops an inflated $a$.

Pearce and Song made this quantitative in the most convincing way possible: they re-ran the fitting procedure using *non-embedding* parameters at small scale and reproduced Kaplan's exponent, recovering $a$ in the range of about 0.74 to 0.78. Then, holding everything else fixed and switching to *total* parameters, the exponent dropped toward Chinchilla's value. That is the signature of a true root cause — you can toggle it on and off and the discrepancy appears and disappears.

Let me make the embedding-share trend fully explicit with a table across widths, because the rate at which the share falls is what drives the slope bias. For a fixed twelve-layer aspect ratio and a 50,257-token vocabulary:

| $d_{\text{model}}$ | block params $\approx 12 L d^2$ | embedding $\approx d V$ (untied, $\times 2$) | total | embedding share |
|---|---|---|---|---|
| 256 | ~9.4M | ~25.7M | ~35M | ~73% |
| 768 | ~85M | ~77M | ~162M | ~48% |
| 1600 | ~369M | ~161M | ~530M | ~30% |
| 4096 | ~2.4B | ~412M | ~2.8B | ~15% |
| 8192 | ~9.7B | ~824M | ~10.5B | ~8% |

The share marches from roughly three-quarters of all parameters at a 256-wide model down to single digits by 8192 wide. A regression that sees its $N$-axis lose this much of its content at one end and almost none at the other will absorb the trend into the slope every time. The numbers above are approximate (real models vary in layer count, vocabulary, and tying), but the direction and the steepness are the load-bearing facts, and they do not depend on the third digit.

### The second-order gotcha: tied versus untied embeddings

Here is the non-obvious wrinkle that makes this even easier to get wrong today. Whether the input embedding and output projection share weights ("tied") changes the embedding's contribution to the total count by up to a factor of two, and different codebases default differently. GPT-2 tied them; many later models do not. If you fit a scaling law across runs from a codebase that switched the tying default partway through your sweep, your $N$ has a discontinuity in it that has nothing to do with capability and everything to do with a config flag. The fix is the same discipline as the main cause: pick one definition of $N$, apply it identically to every run, and write it down. "Total parameters, embeddings included, untied counted separately" is a sentence that belongs in your methods section.

## Culprit 2: the small-scale regime amplifies everything

**The senior rule of thumb: a bias that is negligible where you extrapolate to can still dominate where you fit. Fit-region biases are the worst kind, because the extrapolation hides them.**

The parameter-counting error would have been far less damaging if both papers had fit their laws at GPT-3 scale, because there the embedding fraction is tiny and the two definitions of $N$ nearly coincide. But you do not fit scaling laws at GPT-3 scale — the entire economic point of a scaling law is to fit cheaply at small scale and extrapolate. So the fits live precisely in the regime where the embedding fraction is largest and the counting convention matters most. The two failures conspire: the convention error is amplified by being measured exactly where it is worst.

![Curve on log-log axes showing embedding share of total parameters falling steeply from over fifty percent at a hidden size of 768 down toward a few percent at 70 billion parameters and beyond, with the Kaplan fit region highlighted at small scale and the Chinchilla scale highlighted at large scale](/imgs/blogs/kaplan-vs-chinchilla-reconciliation-5.png)

The curve above makes the amplification visible. The embedding share of total parameters is enormous in Kaplan's fit window — the highlighted small-scale region from a 768-wide model up to 1.5 billion total parameters — and it falls toward negligibility by the time you reach Chinchilla scale. Conceptually, this is why the convention you chose for $N$ is invisible at the right edge of the plot and decisive at the left edge, which is exactly where the fitting happens.

Let me put numbers on the regime. Kaplan's sweep topped out around 1.5 billion parameters. For a model with, say, $d_{\text{model}} = 1600$ (GPT-2 XL's width), the embeddings are around $1600 \cdot 50257 \approx 80$ million parameters against roughly $1.5$ billion total — about five percent, already much smaller than the 768-wide case but still not nothing, and crucially the fraction is *changing rapidly* across the sweep. A bias that varies smoothly with the independent variable across your fit range is the most pernicious kind, because the regression happily absorbs it into the slope. It does not look like noise; it looks like signal. The fit is not failing to converge or throwing warnings. It is confidently fitting a slope that encodes your bookkeeping rather than the physics.

This is also why "just fit at slightly larger scale" is not a free fix. Pushing the fit region up by one order of magnitude helps with the embedding fraction but costs you an order of magnitude in compute per data point, and you still have not addressed culprits 3 and 4. The real fix is to correct the definitions and the accounting, not to outrun them with budget.

### The second-order gotcha: extrapolation amplifies slope error super-linearly

One more reason the small-scale regime is dangerous. Recall the earlier point that the gap between the two frontiers grows with compute. Quantify it: if your fitted slope is off by $\Delta a$, then at compute $C$ your predicted optimal $N$ is off by a factor of $(C / C_{\text{fit}})^{\Delta a}$ relative to the truth, where $C_{\text{fit}}$ is your fit-region compute. With $\Delta a \approx 0.23$ (the 0.73-versus-0.50 gap) and an extrapolation of five orders of magnitude in compute, $(10^5)^{0.23} \approx 10^{1.15} \approx 14$. A fourteen-fold error in recommended model size, from a slope bias that was a couple of percent of the loss in your fit region. That multiplier is why GPT-3 ended up so far from compute-optimal: not because anyone was sloppy in the fit, but because a tiny fit-region bias compounds savagely under long extrapolation.

## Culprit 3: the last layer's FLOPs

**The senior rule of thumb: the FLOP count is a model too, and a wrong FLOP model biases your compute axis exactly like a wrong parameter model biases your size axis.**

The third cause shifts attention from the parameter axis to the compute axis. The standard estimate $C \approx 6ND$ counts the FLOPs of the transformer blocks, but it is an approximation that omits some real computation — notably the cost of the final projection to the vocabulary (the "unembedding" or language-model head) and the embedding lookup, plus various smaller terms. At large scale these omissions are negligible relative to the quadratic block FLOPs, for the same quadratic-versus-linear reason as the parameters. At small scale they are not.

Both reconciliation papers flag this. Pearce and Song list last-layer and unembedding FLOP accounting as a contributing cause. Porian and collaborators, in their 2024 paper "Resolving Discrepancies in Compute-Optimal Scaling of Language Models" (a NeurIPS 2024 paper), make the last-layer computational cost the *first* of their three culprits. The mechanism is the mirror image of culprit 1: if your $C$ undercounts the true compute by a fraction that is large at small scale and small at large scale, then your loss-versus-compute fit again sees an illusory efficiency gain as you scale up, and again biases the allocation toward parameters.

Here is the worked intuition. Suppose at small scale the head and embedding FLOPs are twenty percent of the total but your $6ND$ estimate ignores them, and at large scale they are one percent. Your fit believes you spent only $6ND$ at every point, so at small scale you "got" a certain loss for less compute than you actually used. The regression interprets the small-scale points as more compute-efficient than they truly were, tilting the fitted frontier. Because both the parameter undercount and the FLOP undercount point the same direction — both make small models look artificially efficient — they add rather than cancel. This is why fixing only one of them does not fully close the gap.

A useful sanity check you can run on your own setups: compute the ratio of your estimated FLOPs to a careful per-operation count (including the head, the embedding gather, layer norms, and the attention score-and-value matmuls with their actual sequence-length dependence) at both your smallest and largest model. If that ratio drifts across your sweep, you have a scale-dependent compute bias baked into your fit, and you should either use the careful count everywhere or correct the estimate.

## Culprit 4: warmup duration and the learning rate that was never re-tuned

**The senior rule of thumb: a fixed optimizer schedule is not scale-invariant. What is well-tuned for your big runs is mis-tuned for your small ones, and your small runs are the ones anchoring the fit.**

Pearce and Song found the optimizer-schedule effects to be minor for the discrepancy they studied; Porian and collaborators found them to be real and necessary to close the gap fully in Kaplan's setup. The two findings are not contradictory — they reflect different experimental setups and different ways of holding things fixed — but the prudent reading is that the schedule matters enough that you cannot ignore it, and the direction of the bias is the familiar one: it hurts the small runs.

Two specific issues. First, **warmup duration**. Learning-rate warmup ramps the step size up over some number of initial steps before the main schedule takes over. If warmup is set as a fixed number of steps (or a fixed duration appropriate for long runs), then for short small-scale runs the warmup eats a disproportionate fraction of the whole run. A small model that spends, say, the first 3,000 steps warming up over a 30,000-step run has spent ten percent of its training at a deliberately suppressed learning rate; the same 3,000-step warmup over a 300,000-step large run is one percent. The small models are systematically under-trained relative to the large ones, their losses come out worse than they should, and — once again — the fit reads small models as worse and tilts toward favoring large ones.

Second, **scale-dependent learning rate**. The optimal peak learning rate is not constant across model sizes; it generally needs to shrink as models grow (this is the entire motivation behind learning-rate transfer schemes like muP). If you pick one global learning rate that is right for the middle of your sweep, it is too large for your big models and too small for your small ones, or vice versa, and the mis-tuning is scale-correlated. Scale-correlated mis-tuning is, by now, a familiar villain: the regression cannot tell it apart from the true scaling signal.

![Six-box pipeline showing the fitted exponent starting at 0.73 in the Kaplan setup, dropping as last-layer FLOPs are added, dropping further as warmup is shortened, then as the learning rate is re-tuned per scale, then with all three fixes combined, and finally reaching 0.50 when the Kaplan grid reproduces Chinchilla](/imgs/blogs/kaplan-vs-chinchilla-reconciliation-6.png)

The pipeline above is the punchline of the Porian result: apply the three corrections — last-layer FLOPs, shorter warmup, and per-scale learning rate — to Kaplan's own experimental setup, and the fitted exponent walks down from about 0.73 to about 0.50. The crucial word is *together*. No single correction does the whole job; the gap is the sum of several scale-dependent biases all pointing the same way, and you have to remove all of the large ones to land on Chinchilla's answer. That is also the most reassuring possible outcome, scientifically: it means the discrepancy was over-determined by mundane, identifiable causes, not by some deep unknown difference in how the two labs' networks learned.

### The second-order gotcha: hyperparameters chosen once are biased forever

The deeper lesson hiding in culprit 4 is that *any* hyperparameter held fixed across a scale sweep risks injecting a scale-correlated bias if its optimum moves with scale. Warmup and learning rate are the famous ones, but batch size, weight decay, the schedule shape, even the data shuffling can all have scale-dependent optima. The discipline is not "re-tune learning rate"; it is "for every hyperparameter you hold fixed across the sweep, convince yourself its optimum does not move, or re-tune it, or measure the bias." Most published scaling laws still do not do this for every knob, which is one reason their exponents should be read with a grain of salt — a point I will make sharper below.

## Putting the culprits in proportion

It helps to see the four causes stacked by how much of the gap each carries, because "there were several reasons" can collapse in memory into "it was complicated and unknowable," which is exactly the wrong takeaway. It was knowable, and one cause dominates.

![Decomposition grid with three columns for culprit, mechanism, and effect on the fitted exponent, listing parameter counting as dominant, small-scale regime as the amplifier, last-layer FLOPs as a same-direction bias, and the combined fix collapsing the exponent to about 0.50](/imgs/blogs/kaplan-vs-chinchilla-reconciliation-3.png)

The decomposition above is the one-screen summary of the diagnosis. Parameter counting is the dominant term — switch from non-embedding to total parameters and most of the gap closes. The small-scale regime is not a separate cause so much as the amplifier that makes the counting error large instead of negligible. Last-layer FLOPs push the same direction on the compute axis. And the optimizer-schedule effects are the finishing touch that, in Kaplan's exact setup, closes the residual. Read top to bottom, the table is a recipe: fix the dominant cause first, understand why your fit region amplifies it, then clean up the compute accounting and the schedule.

| Cause | Axis it biases | Magnitude | Direction | Toggle test |
|---|---|---|---|---|
| Non-embedding vs total params | model size $N$ | dominant | inflates $a$ | switch counts, gap appears/disappears |
| Small-scale fit regime | both (amplifier) | large multiplier | amplifies the above | fit at larger scale, bias shrinks |
| Last-layer / head FLOPs | compute $C$ | moderate | inflates $a$ | careful FLOP count vs $6ND$ |
| Warmup + learning rate | small-run quality | setup-dependent | inflates $a$ | re-tune per scale, small runs improve |

Notice that every cause biases in the *same* direction — toward an inflated $a$, toward "spend on parameters." That is not a coincidence; it is structural. Every one of these biases makes small models look worse or larger models look more efficient than they really are, and a regression that sees small models as relatively worse will always recommend skewing toward large. The errors did not cancel because they could not cancel. They all flowed from the same small-scale-versus-large-scale asymmetry.

## A worked example: how the same data yields two slopes

Abstract talk about "biased regressions" is unconvincing until you watch the numbers move, so let me build a tiny, fully explicit example you can reproduce on paper. The goal is to take one underlying truth, apply Kaplan's bookkeeping versus Chinchilla's bookkeeping, and watch the fitted exponent diverge.

### Step 1: the ground truth

Posit a world where the *true* compute-optimal relationship is balanced: $N_{\text{opt}}^{\text{total}} \propto C^{0.50}$. This is the Chinchilla answer, and in our toy world it is exactly correct. We will sweep four compute budgets a decade apart, $C \in \{10^{18}, 10^{19}, 10^{20}, 10^{21}\}$ FLOPs, and compute the true optimal *total* parameter count at each, anchoring at $N = 10^{8}$ total parameters at $C = 10^{18}$.

With slope 0.50, a tenfold increase in $C$ multiplies $N$ by $10^{0.5} \approx 3.16$. So the true total-parameter optima are:

| $C$ (FLOPs) | true $N_{\text{total}}$ | $d_{\text{model}}$ (approx) |
|---|---|---|
| $10^{18}$ | $1.0 \times 10^{8}$ | ~640 |
| $10^{19}$ | $3.16 \times 10^{8}$ | ~1100 |
| $10^{20}$ | $1.0 \times 10^{9}$ | ~1900 |
| $10^{21}$ | $3.16 \times 10^{9}$ | ~3200 |

The last column is a rough hidden size consistent with each total count, which we need because the embedding fraction depends on width, not on the total directly. (These widths are illustrative; the point is the trend, not the third digit.)

### Step 2: subtract the embeddings

Now apply Kaplan's convention. The embedding parameters are about $d_{\text{model}} \cdot V$ with $V = 50{,}000$. Subtract them from each total to get the *non-embedding* count that Kaplan would have regressed against:

| true $N_{\text{total}}$ | embeddings $\approx d_{\text{model}} \cdot V$ | $N_{\text{non-emb}}$ | embedding share |
|---|---|---|---|
| $1.00 \times 10^{8}$ | $640 \cdot 50\text{k} \approx 3.2 \times 10^{7}$ | $6.8 \times 10^{7}$ | 32% |
| $3.16 \times 10^{8}$ | $1100 \cdot 50\text{k} \approx 5.5 \times 10^{7}$ | $2.6 \times 10^{8}$ | 17% |
| $1.00 \times 10^{9}$ | $1900 \cdot 50\text{k} \approx 9.5 \times 10^{7}$ | $9.1 \times 10^{8}$ | 10% |
| $3.16 \times 10^{9}$ | $3200 \cdot 50\text{k} \approx 1.6 \times 10^{8}$ | $3.0 \times 10^{9}$ | 5% |

Look at the embedding-share column. It falls from 32% to 5% across the sweep — exactly the curve we drew earlier. The non-embedding count is a *shifting* fraction of the truth.

### Step 3: re-fit the slope on the wrong axis

Now regress $\log N_{\text{non-emb}}$ against $\log C$ and read off the slope. The two endpoints are $(C, N_{\text{non-emb}}) = (10^{18}, 6.8 \times 10^{7})$ and $(10^{21}, 3.0 \times 10^{9})$. The slope is

$$a_{\text{Kaplan}} = \frac{\log(3.0 \times 10^{9}) - \log(6.8 \times 10^{7})}{\log(10^{21}) - \log(10^{18})} = \frac{9.48 - 7.83}{21 - 18} = \frac{1.65}{3} \approx 0.55.$$

In this deliberately mild toy (only three decades, only a 32% peak embedding share, and I have not even added the FLOP or warmup biases), the fitted exponent already jumps from the true 0.50 to about 0.55 purely from the parameter-counting convention. Push the embedding share higher — a smaller starting model, a bigger vocabulary, untied embeddings doubling the count — and stack the same-direction FLOP and warmup biases on top, and you walk the fitted slope up toward the 0.73 that Kaplan reported. Crucially, *nothing about the underlying world changed*. The networks scaled at exactly 0.50 the whole time. Only the ruler we measured them with was miscalibrated, and it was miscalibrated more at one end than the other.

### Step 4: the correction is trivial once you see it

Re-run step 3 on the *total* column and the slope is, by construction, exactly 0.50 — because that is how we built the ground truth. That is the entire fix. Not a new model, not a new dataset, not a new optimizer. Use the same number for $N$ that you actually trained and actually paid FLOPs for. The reconciliation papers are, at their core, this toy example run on real data with all four biases included.

## The replication that said "the laws are not gospel"

There is a parallel 2024 result that belongs in this story for a different reason. While Pearce-Song and Porian were explaining *why Kaplan and Chinchilla disagreed*, Besiroglu and collaborators were asking a sharper and more uncomfortable question about Chinchilla itself: does the published parametric fit even reproduce?

Their paper, "Chinchilla Scaling: A replication attempt," took the third of Chinchilla's three approaches — the direct parametric fit of $L(N, D)$ — and tried to reconstruct it from the data points in the original paper. Two findings. First, the published Approach-3 fit reproduces poorly; the reconstructed loss surface does not match the reported coefficients well. Second, and more pointedly, the original paper's reported confidence intervals on the fitted constants were implausibly tight — far tighter than the data could support. The corrected fit they obtained shifts the constants noticeably: roughly $E \approx 1.82$, $A \approx 482$, $B \approx 2085$, $\alpha \approx 0.348$, $\beta \approx 0.366$.

![Comparison matrix of five fitted Chinchilla constants showing the original Hoffmann 2022 values beside the Besiroglu 2024 replicated values and a verdict for each, with the floor and both power-law coefficients shifting while alpha stays near 0.35 and beta moves up toward alpha](/imgs/blogs/kaplan-vs-chinchilla-reconciliation-7.png)

The matrix above sits the original and replicated constants side by side. Look at what moves and what does not. The irreducible loss $E$ ticks up a little. The parameter coefficient $A$ grows. The data coefficient $B$ grows *enormously*, from about 411 to about 2085 — a fivefold change. The model exponent $\alpha$ is essentially unchanged at about 0.35. And the data exponent $\beta$ moves up from 0.28 to about 0.37, which actually brings it *closer* to $\alpha$, not further. That last point is the one to dwell on.

Because here is the thing the replication does *not* do: it does not overturn Chinchilla's conclusion. With the corrected constants, $\alpha \approx 0.348$ and $\beta \approx 0.366$ are still approximately equal — if anything more equal than before. Run them through $a = \beta / (\alpha + \beta) = 0.366 / 0.714 \approx 0.51$. Still about one-half. Still balanced allocation. Still roughly twenty tokens per parameter at the optimum. So the headline survives the replication intact, while the exact coefficients turn out to be much softer than the original paper's tight intervals suggested.

This is the "laws are not gospel" lesson, and it is the most important practical takeaway in this entire post, so let me state it as plainly as I can. The *qualitative* conclusion — balanced scaling, twenty-to-one, smaller-and-longer beats bigger-and-shorter at fixed compute — is robust. It survives re-counting parameters, re-fitting from scratch, and three independent methodologies. The *quantitative* constants are point estimates with real, and in the original paper understated, uncertainty. Treating $\alpha = 0.34$ as a law of nature precise to two decimals is exactly the mistake that the whole Kaplan episode should inoculate you against. The number you should remember is "about 20 tokens per parameter, give or take, and the exponent is about 0.5," not any specific six-digit coefficient.

> Robust conclusion, soft constants. The direction of the law is settled; the decimals are estimates. Quote the direction with confidence and the decimals with error bars, or you will repeat the 0.73 mistake in a new variable.

## The timeline, and why the error was so expensive

It is worth tracing the chronology, because the cost of this measurement error was not abstract — it was hundreds of millions of dollars of compute spent on undertrained models.

![Timeline from January 2020 to June 2024 marking Kaplan scaling laws, GPT-3 following the model-heavy advice, Chinchilla overturning it with balanced scaling, the Besiroglu replication correcting coefficients, and the Pearce-Song and Porian reconciliations resolving the cause](/imgs/blogs/kaplan-vs-chinchilla-reconciliation-8.png)

The timeline above shows the gap between cause and correction. Kaplan published in January 2020. GPT-3 followed in May 2020, 175 billion parameters on roughly 300 billion tokens — under two tokens per parameter, a textbook application of the model-heavy advice. Gopher (280B on 300B tokens), MT-NLG (530B), and a string of other flagships followed the same recipe. For two years, the field's largest investments were allocated by an exponent that was wrong. Chinchilla landed in March 2022 and showed, with a built model rather than just an argument, that a 70-billion-parameter model on 1.4 trillion tokens beat all of them at the same training compute. The replication and reconciliation papers — Besiroglu in April 2024, Pearce-Song and Porian in June 2024 — closed the loop only two years after that, four years after the original split.

The dollars-and-FLOPs cost is the part that makes this more than a methodology anecdote. Consider Gopher: 280 billion parameters on 300 billion tokens, a ratio of roughly one token per parameter against the compute-optimal twenty. Chinchilla used the *same* training compute and got a strictly better model by reallocating it. The compute spent on Gopher's "extra" parameters past the compute-optimal point did not just underperform — it was actively worse than spending the same FLOPs on training a smaller model longer. Every lab running the model-heavy playbook in 2020 and 2021 was leaving a large multiple of effective compute on the table, and the reason was an exponent inflated by parameter-counting bookkeeping.

There is a second-order cost too, harder to quantify: the strategic decisions built on top of the wrong exponent. Inference economics, deployment plans, the sizing of model families — all of it was reasoned about under the assumption that you should make the biggest model your training budget allows. The correction did not just change one number; it changed the shape of the optimal model and therefore the shape of everything downstream. The inference-aware scaling literature that came later (which pushes the optimum to *even smaller, even longer-trained* models once you account for serving cost) only makes sense once you have the Chinchilla exponent as the starting point. Getting the exponent right was load-bearing for a whole subsequent body of work.

### The tokens-per-parameter ledger, before and after

The cleanest single statistic for tracking the correction is the tokens-per-parameter ratio, $D/N$, because the Chinchilla optimum sits at about 20 and the model-heavy era sat far below it. Lining up the famous models against that benchmark makes the regime change visible at a glance.

| Model | Params $N$ | Tokens $D$ | $D/N$ | Era |
|---|---|---|---|---|
| GPT-3 | 175B | ~300B | ~1.7 | model-heavy (pre-correction) |
| Gopher | 280B | 300B | ~1.1 | model-heavy |
| MT-NLG | 530B | ~270B | ~0.5 | model-heavy |
| Chinchilla | 70B | 1.4T | ~20 | compute-optimal |

The pre-correction models cluster around one or two tokens per parameter — an order of magnitude below compute-optimal. Reading the table, every model in the model-heavy era was trained on roughly a tenth to a twentieth of the data its parameter count warranted. That is the entire indictment in one column: not a subtle inefficiency, but a 10-to-20x misallocation along the data axis, all flowing from an exponent inflated toward parameters.

What happened *after* the correction is just as instructive, and it is why this episode is the hinge of the whole scaling-laws series rather than a closed historical case. Once the field believed the balanced exponent, it did not stop at twenty tokens per parameter — it kept going, deliberately overshooting Chinchilla's training-optimal ratio because the *inference* economics reward a smaller, longer-trained model. Later open models pushed $D/N$ far past twenty: small models trained on trillions of tokens, ratios in the hundreds and beyond. That overshoot is not a contradiction of Chinchilla; it is what you get when you add a second objective (cheap serving) on top of the corrected training-optimal baseline. But it is only sensible *because* the Chinchilla exponent replaced the Kaplan one first. If the field still believed $a \approx 0.73$, the inference-aware argument — train an even smaller model even longer — would have looked like obvious malpractice. The corrected exponent did not just fix one generation of models; it unlocked the entire "smaller and longer" design direction that followed. The inference-aware treatment in this series picks up exactly there.

## Case studies: the bug, the fixes, and the echoes

The clean way to internalize a measurement bug is to see it play out in named, concrete episodes — what the symptom looked like, what the wrong first hypothesis was, what the actual root cause turned out to be, and what the fix taught. Here are the episodes that, taken together, make up the full arc of this story.

### 1. GPT-3: the textbook application of a wrong exponent

The symptom: a 175-billion-parameter model trained on roughly 300 billion tokens, under two tokens per parameter. The reasoning was impeccable given the inputs — Kaplan's $a \approx 0.73$ says spend marginal compute on size, and stop well before convergence because the model law dominates. The wrong first hypothesis, held for two years, was that GPT-3 was simply the right shape for its budget and that any underperformance was about data quality or architecture. The actual root cause was upstream: the exponent that sized it was inflated by parameter-counting bookkeeping, so the model was systematically too large for its token budget. The fix was not GPT-3-specific; it was the Chinchilla reallocation. The lesson is that a model can be flawlessly engineered to a flawed specification, and the flaw is invisible from inside the model — you have to audit the spec.

### 2. Gopher: same compute, strictly beaten

The symptom: Gopher, 280 billion parameters on 300 billion tokens (roughly one token per parameter), was DeepMind's own flagship, and then DeepMind's own next paper showed a 70-billion-parameter model beating it at the *same* training compute. The wrong first hypothesis a casual reader might form is that Chinchilla was a bigger compute run; it was not — it was a *reallocation* of Gopher's budget. The actual root cause of Gopher's underperformance was the model-heavy allocation. The fix, Chinchilla, demonstrated the correction with a built artifact rather than an argument, which is why it landed so hard. The lesson: the most persuasive refutation of a scaling recommendation is a model that follows the corrected recommendation and wins at equal cost.

### 3. The IsoFLOP profiles: three methods, one answer

The symptom Chinchilla had to overcome was credibility — one new fit disagreeing with a celebrated paper is easy to dismiss as a fluke. The wrong move would have been to rest on a single methodology. Instead Chinchilla triangulated with three independent approaches: minimizing over training curves at fixed model size (gave $a \approx 0.50$), IsoFLOP profiles sweeping model size at nine fixed FLOP budgets from $6 \times 10^{18}$ to $3 \times 10^{21}$ (gave $a \approx 0.49$), and the direct parametric fit (gave $a \approx 0.46$). The actual strength of the result was the *agreement* across methods that share no fitting machinery. The lesson for anyone fitting a contested constant: if three methods that fail in different ways all agree, the answer is probably real; if you have only one method, you have only a hypothesis.

### 4. Pearce-Song: the toggle test that nailed it

The symptom was the open question itself — *why* 0.73 versus 0.50, mechanistically. The wrong hypothesis floating around the community was the comfortable "two valid regimes" story, that maybe both were right in different settings. Pearce and Song killed that by showing the discrepancy is a *toggle*: re-fit with non-embedding parameters at small scale and you recover $a$ in the 0.74-to-0.78 range; switch to total parameters and it drops toward 0.50. A cause you can switch on and off is a cause you have actually found. The lesson: the gold standard for "I found the bug" is reproducing the bug on demand and then making it vanish on demand, not merely telling a plausible story.

### 5. Porian: the residual that needed three fixes

The symptom Porian and collaborators tackled was the leftover gap — even after the parameter accounting, Kaplan's exact setup did not perfectly reproduce Chinchilla. The wrong hypothesis would be that one more single fix would close it. The actual finding was that three corrections were each necessary: last-layer FLOPs, shorter warmup, and per-scale learning rate. Apply all three to Kaplan's grid and the exponent walks from 0.73 to 0.50; apply any subset and it lands in between. The lesson: when several mundane biases all point the same direction, the discrepancy is their *sum*, and partial fixes give partial — and misleadingly stable-looking — answers.

### 6. Besiroglu: the suspiciously tight confidence intervals

The symptom was statistical rather than physical: the original Chinchilla Approach-3 fit reported confidence intervals on its constants that were far tighter than the underlying data could support, and the fit reproduced poorly when reconstructed. The wrong reaction would be to conclude Chinchilla was wrong. The actual finding was subtler and more useful: the *conclusion* (balanced scaling, $\alpha \approx \beta$, twenty-to-one) is robust, while the *constants* are point estimates with real uncertainty — the data coefficient $B$ moved roughly fivefold under re-fitting. The lesson is the one this whole post is built around: separate the direction of a law (trust it) from its decimals (error-bar them).

### 7. The tied-versus-untied embedding trap

The symptom shows up when someone today fits a scaling law across runs from a codebase whose embedding-tying default changed partway through the sweep. The fitted slope wobbles for no capability-related reason. The wrong hypothesis is that there is interesting structure in the wobble. The actual root cause is that "total parameters" silently changed definition mid-sweep — tied embeddings count the matrix once, untied count it twice. The fix is to normalize the count to one convention across every run. The lesson: a definition that is consistent within a codebase can still be inconsistent across a sweep if a config default flipped, and the regression cannot tell you it happened.

### 8. The data-mixture fitter who re-invented 0.73

The symptom, which I have watched happen on internal projects, is a team fitting a scaling law for some new axis — a data-mixture ratio, a curriculum schedule, a filtering threshold — and getting an exponent that recommends an aggressive, surprising allocation. The wrong first move is to act on the surprising recommendation. The actual root cause, more often than not, is a scale-dependent bookkeeping inconsistency exactly analogous to the embedding one: a quantity that is defined or measured slightly differently at small versus large scale, absorbed into the slope. The fix is the checklist below — define axes precisely, fit at multiple scales, check slope stability. The lesson, and the reason this 2020-2022 episode is still worth studying in detail: the bug is not historical. The same trap is one unstated convention away in every new scaling study, including yours.

## How this still bites people today

It would be comforting to file this under "solved in 2024 and no longer relevant," but the structure of the error guarantees recurrence. Every time someone extends scaling laws to a new axis, they re-enter the danger zone, because the new axis comes with new definitions that nobody has audited yet.

The pattern to watch for has three ingredients, and when all three are present you should assume a hidden bias until proven otherwise. First, **a quantity whose definition has a scale-dependent ambiguity** — like parameters (embeddings matter at small scale only), or FLOPs (the head matters at small scale only), or "effective tokens" in repeated-data settings, or "effective parameters" in low-precision training. Second, **a fit region at small scale** — which is universal, because that is the economic point of scaling laws. Third, **a long extrapolation** — also universal, because nobody fits a law to use it at the fit scale.

When those three line up, the diagnostic is always the same toggle test Pearce-Song used: change the ambiguous definition and watch whether your exponent moves. If it does, the exponent was partly encoding your convention, not the world. If it does not, you have ruled out that particular bias. Run the toggle on every axis with a scale-dependent definition, not just parameters. The cost of the test is one re-fit; the cost of skipping it was, historically, two years and a generation of undertrained flagship models.

There is also a sociological echo worth naming. The "two valid regimes" rationalization — the instinct to reconcile two conflicting measurements by assuming both are right in different circumstances — is a deeply human and deeply wrong move when the quantity is a measured constant. It feels generous and even rigorous; it is neither. A measured constant has one value. When two careful measurements of it disagree by far more than their stated uncertainties, the correct conclusion is not "both are right somewhere" but "at least one measurement has an uncontrolled systematic error, and it is findable." The field spent two years half-believing the generous story before someone ran the toggle test. Do not extend that courtesy to your own conflicting numbers.

## Common objections, answered

A few questions come up every time I walk a team through this story. They are worth answering directly, because each one hides a subtle misconception that can lead you back into the same trap.

**"Wasn't Kaplan just using a different model family, so the laws legitimately differ?"** No, and this is the single most common misconception. Both papers report that architecture details — depth versus width, aspect ratio, the specific transformer variant — barely move the exponent compared to $N$, $D$, and $C$. Kaplan made this an explicit headline finding. The difference between the two papers is not the networks; it is the bookkeeping applied to the networks. If architecture choice could swing the exponent from 0.5 to 0.73, scaling laws would be nearly useless, because you could not transfer them across model families at all. The whole reason scaling laws are valuable is that the exponent is approximately architecture-invariant — which is exactly why a 0.73-versus-0.50 gap had to be a measurement issue, not a modeling one.

**"If Chinchilla's own constants don't replicate, why trust its exponent over Kaplan's?"** Because the exponent and the constants are different kinds of claim with different robustness. The Besiroglu replication moved the *constants* substantially but left the *exponent ratio* ($\alpha \approx \beta$, hence $a \approx 0.5$) intact, and the exponent is independently confirmed by two other Chinchilla methods that do not fit those constants at all. Kaplan's exponent, by contrast, is reproduced only by repeating Kaplan's convention and reverses under the toggle test. So we trust the Chinchilla exponent not because Chinchilla's paper is flawless — it has the tight-interval problem — but because the exponent specifically is the part that survives every check, while the constants are the part that does not. Robust direction, soft decimals, applied consistently.

**"Non-embedding parameters seem more principled — they are the parameters that do the 'real' computation. Why is counting them wrong?"** It is not wrong to be *interested* in non-embedding parameters; it is wrong to fit a compute-allocation law on the non-embedding count while spending FLOPs on the total model. The inconsistency is the bug, not the choice of count per se. If you genuinely wanted to reason in non-embedding terms, you would also have to define compute in non-embedding terms and apply that consistently across scales — at which point the small-scale distortion reappears through the FLOP axis instead. The clean escape is to count everything on both axes, because the total counts are what you actually paid for and what converge cleanly at scale. "Principled-sounding but inconsistent" is precisely the flavor of error that is hardest to catch.

**"This was 2020 hardware and tooling. Surely modern fitting pipelines don't have this problem?"** The tooling is better, but the structural trap is identical, and modern scaling studies on new axes routinely re-enter it. Effective-token counts in data-repetition laws, effective-parameter counts in low-precision laws, quality-adjusted token counts in data-filtering laws — every one of these is a quantity with a scale-dependent definition fit at small scale and extrapolated. The 2020 episode is not a story about old tooling; it is a story about a permanent hazard in the methodology. Better tooling that computes a biased quantity faster is still biased.

## What this means in practice

The Kaplan-Chinchilla episode is, in the end, a fitting-discipline story dressed up as a physics dispute. The networks were never in disagreement about how they scale. The humans were in disagreement about how to count. So the practical takeaways are all about bookkeeping discipline, and they generalize to every scaling law you will ever fit — for data mixtures, for precision, for repetition, for inference cost, for anything.

**Define your bookkeeping before you fit, and write it down in big letters.** Decide whether $N$ is total or non-embedding parameters, whether the unembedding is tied, and exactly which FLOPs your $C$ counts (blocks only, or blocks plus head plus embedding plus attention's sequence-dependent terms). Apply the definition identically to every run in the sweep. The single most expensive bug in the history of scaling laws was a definitional one that nobody wrote down. Your methods section should let a reader reproduce your $N$ and $C$ for any run to the parameter. If you cannot write that sentence cleanly, you are not ready to fit.

**Prefer total parameters and total compute.** Unless you have a specific, defended reason to do otherwise, count everything. Total parameters and total FLOPs converge to the non-embedding counts at large scale anyway, so you lose nothing in the regime you care about and you avoid the small-scale distortion in the regime where you actually fit. "Count everything" is the conservative default precisely because it is the one that does not blow up under extrapolation.

**Re-tune scale-sensitive hyperparameters per scale, or measure the bias.** Warmup as a fraction of total steps, not a fixed count. Learning rate that follows a transfer rule rather than a single global value. For every knob you hold fixed across the sweep, either argue its optimum is scale-invariant, re-tune it, or quantify the bias it injects. A scale-correlated mis-tuning is indistinguishable from scaling signal to a regression, and it always biases toward favoring whichever end of the sweep you tuned for.

**Fit at multiple scales and check the slope is stable.** If your fitted exponent changes when you drop your smallest models or add a larger one, your fit region is in a biased regime and your extrapolation is suspect. A stable slope across sub-ranges is weak evidence you have escaped the fit-region biases; an unstable one is strong evidence you have not. This is cheap insurance against the small-scale-amplification trap.

**Treat every published coefficient as a point estimate with real uncertainty.** The Besiroglu replication is the proof: even Chinchilla's celebrated constants are softer than they were reported to be, and the data coefficient moved by 5x under re-fitting. Quote the *direction* of a scaling law with confidence — balanced allocation, about twenty tokens per parameter, exponent near one-half — and quote the *decimals* with error bars or not at all. When someone hands you an exponent to four significant figures, ask for the confidence interval and how it was computed; if the answer is suspiciously tight, be suspicious.

**Remember that extrapolation amplifies slope errors super-linearly.** A two-percent bias in your fit region becomes a 14x error in recommended model size across five orders of magnitude of compute. This is the property that makes scaling laws economically powerful and the property that makes a slope bug catastrophic. The further you extrapolate, the more your bookkeeping discipline matters, not less.

### A short checklist you can paste into a methods doc

- $N$ definition: total params, embeddings included, tying convention stated, applied to every run.
- $C$ definition: which FLOP terms are counted; consistent across the sweep; head and embedding included at small scale.
- Warmup: fixed *fraction* of total steps, not a fixed step count.
- Learning rate: transfer rule or per-scale re-tune; not a single global value across orders of magnitude.
- Fit-region check: refit on sub-ranges; confirm the exponent is stable when the smallest or largest models are dropped.
- Reported constants: with confidence intervals, computed by a method you can describe; direction stated separately from decimals.

If you internalize one thing from the whole Kaplan-Chinchilla affair, make it this: a scaling law is only as trustworthy as the definitions of its axes, and the definitions are exactly where careful people stop paying attention because the math downstream is so engaging. The exponent that split the field for two years was not hiding in the loss surface. It was hiding in the question "what does $N$ count?" — a question that has a one-sentence answer and that nobody wrote down.

## When to trust a published exponent, and when not to

A scaling exponent is a tool, and like any tool it has a regime where it is reliable and a regime where it will quietly hurt you. The Kaplan episode is the canonical example of using a real measurement outside its regime of validity. Here is how I decide whether to lean on a published exponent or to re-derive it myself.

**Reach for a published exponent when:**

- The conclusion you need is *directional* — "balance N and D," "smaller-and-longer beats bigger-and-shorter," "more data has diminishing returns at this rate" — rather than a precise coefficient. Directions are robust; the Chinchilla direction survived replication and three methods.
- The exponent has been reproduced by independent groups using independent methods. The Chinchilla $a \approx 0.5$ has; the Kaplan $a \approx 0.73$ was reproduced only by repeating Kaplan's bookkeeping, which is the opposite of independent confirmation.
- Your deployment scale is *near* the fit scale, so you are interpolating rather than extrapolating far. Slope error is invisible near the anchor; if you are not extrapolating much, a slightly-off slope barely hurts.
- You are sanity-checking an intuition or sizing a first experiment, not committing a nine-figure training budget. The cost of being wrong is small, so the value of re-deriving is small.

**Re-derive it yourself when:**

- You are about to commit serious compute on the basis of the exponent. The 14x amplification under long extrapolation means a borrowed exponent with an unaudited bias can misdirect your single largest investment. Audit before you spend.
- Your setting differs from the original in a way that touches the axis definitions — different tokenizer or vocabulary (changes the embedding fraction), different architecture with a different parameter-to-FLOP ratio, tied versus untied embeddings, a different precision regime, repeated data. Any of these can move the effective exponent because they move what the axes count.
- You are extending scaling laws to a *new* axis. There is no published, replicated exponent for your axis yet, and the first fit on a new axis is exactly when the unaudited-convention trap is most likely. Treat your own first exponent with the same suspicion you would treat anyone else's.
- The published confidence interval is suspiciously tight, or absent. Besiroglu showed that even a celebrated result reported intervals tighter than the data supported. If you cannot find an honest error bar, assume the point estimate is softer than it looks.

**Skip the exponent entirely — do not fit a scaling law — when:**

- You have only two or three scales to fit on. A power law needs several decades to pin a slope; two points fit a slope perfectly and tell you nothing about whether it is real.
- Your runs are not comparable across scale — different data, different stopping criteria, different hyperparameter regimes that you have not normalized. A scaling law across incomparable runs fits the inconsistency, not the scaling, which is precisely the Kaplan failure mode generalized.
- The thing you care about is not actually following a power law in your range. Some quantities have a regime change, a plateau, or an emergent jump. Forcing a single power law across a regime boundary produces an exponent that is an average of two different behaviors and predicts neither.

The meta-rule behind all of these: the trustworthiness of an exponent is governed by the discipline of its axis definitions and the breadth of its independent confirmation, not by the prestige of the lab that published it or the precision with which the decimals are quoted. Kaplan was a careful paper from a strong lab that quoted a precise number. It was still wrong outside its regime, for a reason that had nothing to do with the quality of the work and everything to do with an unstated convention. Hold every exponent, including the ones in this post, to that standard.

> Trust the direction of a replicated scaling law. Audit the decimals before you spend on them. And never reconcile two conflicting measurements of a constant by assuming both are right — find the systematic error instead.

## Further reading

- Kaplan et al. 2020, "Scaling Laws for Neural Language Models" — the original exponent: https://arxiv.org/abs/2001.08361
- Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models" (Chinchilla) — the correction: https://arxiv.org/abs/2203.15556
- Pearce and Song 2024, "Reconciling Kaplan and Chinchilla Scaling Laws" — parameter counting as the dominant cause: https://arxiv.org/abs/2406.12907
- Porian et al. 2024, "Resolving Discrepancies in Compute-Optimal Scaling of Language Models" (NeurIPS 2024) — last-layer FLOPs, warmup, and learning rate: https://arxiv.org/abs/2406.19146
- Besiroglu et al. 2024, "Chinchilla Scaling: A replication attempt" — the corrected coefficients and the uncertainty story: https://arxiv.org/abs/2404.10102
- This series: [scaling laws, from scratch](/blog/machine-learning/scaling-laws/scaling-laws-predictability-foundations), [Kaplan 2020](/blog/machine-learning/scaling-laws/kaplan-scaling-laws-language-models), and [Chinchilla 2022](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling).

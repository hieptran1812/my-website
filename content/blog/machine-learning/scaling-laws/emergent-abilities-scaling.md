---
title: "Emergent abilities: real phase transitions or a measurement mirage?"
date: "2026-06-15"
description: "Learn why large language models appear to gain abilities in sudden jumps, how much of that is an artifact of metric choice, and how to forecast capabilities reliably using pre-training loss and continuous metrics."
tags: ["scaling-laws", "emergent-abilities", "large-language-models", "phase-transition", "metric-choice", "big-bench", "mmlu", "loss-threshold", "in-context-learning", "evaluation", "forecasting", "predictability"]
category: "machine-learning"
subcategory: "Scaling Laws"
author: "Hiep Tran"
featured: true
readTime: 53
---

In June 2022 a paper landed that gave a name to something everyone training large models had felt but could not quite measure: capability that appears out of nowhere. You scale a model from one billion parameters to ten, and on most tasks the loss curve slides smoothly downward, exactly as the scaling laws promise. But on a handful of tasks something stranger happens. The model is at chance, at chance, at chance, and then, somewhere past a threshold of compute, it can suddenly do three-digit arithmetic, or pass a college exam, or follow an instruction it has never seen. The curve does not slope; it jumps. Wei et al. called these *emergent abilities*, and the unsettling implication was that you could not forecast them: no amount of staring at your small-model results would tell you which abilities your big model was about to acquire. The diagram below is the mental model for the entire debate that followed, and it is deliberately provocative: it shows that the same frozen set of model outputs can be made to look like a sharp phase transition or a smooth, boring power law, depending on nothing more than which metric the researcher reaches for.

![A branching diagram showing one frozen set of model outputs scored by a discontinuous exact-match metric producing a sharp cliff and by a continuous edit-distance metric producing a smooth power law, leading to the Wei and Schaeffer conclusions](/imgs/blogs/emergent-abilities-scaling-1.png)

That second reading is the heart of the controversy. In April 2023, Schaeffer, Miranda, and Koyejo published "Are Emergent Abilities of Large Language Models a Mirage?" and argued that emergence, in the strong sense Wei meant it, is often manufactured by the evaluation. Pick a harsh, all-or-nothing metric and you get a cliff. Pick a smooth, partial-credit metric on *the very same model outputs* and you get a predictable curve. The paper won an Outstanding Paper award at NeurIPS 2023, and for a while it felt like the case was closed: emergence was a measurement artifact, not a property of the model.

It was not closed. Over the next two years a series of follow-ups complicated the picture in both directions. Du et al. (2024) showed that if you plot downstream performance against *pre-training loss* instead of against scale, abilities still appear abruptly below a task-specific loss threshold, and they appear that way *even under continuous metrics* — which means at least some of what we call emergence is a genuine transition, not a scoring trick. Lu et al. (2024) argued that most apparent emergent reasoning is really in-context learning plus memory plus linguistic knowledge, not a new capability switching on. And Zhao, Saphra et al. (2025) showed that individual training runs really do break through abruptly, on a per-seed basis, even though the distribution across seeds shifts smoothly and predictably. This post is a full tour of that argument. We will build the intuition first, then work the math of why exact-match metrics manufacture cliffs, then go through each paper's evidence with concrete numbers, and finally land on the balanced mid-2026 consensus and what it means for anyone trying to forecast what their next model will be able to do.

> [!important] The one thing to remember: emergence is mostly about the metric, but not entirely
> - **Wei et al. 2022 defined an emergent ability** as one "not present in smaller models but present in larger models" — sharp and, crucially, *unpredictable* from small-model extrapolation. Examples include multi-step arithmetic, MMLU, chain-of-thought, and instruction-following, typically switching on around $10^{22}$ to $10^{24}$ training FLOPs.
> - **Schaeffer et al. 2023 showed much of this is a metric artifact.** Over 92% of the hand-annotated emergent BIG-Bench tasks use one of two *discontinuous* metrics (Multiple-Choice Grade or Exact-String-Match). Swap to a continuous metric (Token Edit Distance, Brier score) on the same outputs and the cliff becomes a smooth curve.
> - **The mechanism is multiplicative error compounding.** A smooth per-token accuracy $p$ that rises gradually becomes exact-match success $\approx p^N$ over an $N$-token answer — and $p^N$ rises as a cliff.
> - **They proved it by inducing emergence in vision models on purpose** — CIFAR-100 autoencoders and CNNs show a fake "phase transition" the moment you score them with an all-or-nothing metric.
> - **But the loss-threshold transition is real (Du et al. 2024).** Plotted against pre-training loss, abilities emerge below a task-specific loss threshold *even under continuous metrics* — the strongest evidence that emergence is not purely a mirage.
> - **Breakthroughs are stochastic per run (Zhao, Saphra et al. 2025).** A single seed jumps; the across-seed distribution shifts smoothly. Both facts are true at once.
> - **For forecasting:** track pre-training loss, score with continuous proper-scoring metrics, and treat any single-run breakthrough as a random draw, not a guaranteed milestone.

## Why this question is different from the rest of scaling

Most of what we know about scaling laws is reassuring. The [foundational predictability results](/blog/machine-learning/scaling-laws/scaling-laws-predictability-foundations) say that test loss falls as a clean power law in parameters, data, and compute, and the [Chinchilla compute-optimal recipe](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) tells you how to spend a budget so that loss falls as fast as possible. Both rest on a comforting fact: the thing you measure (cross-entropy loss in nats per token) is *smooth*. You fit a line on log-log axes from small runs and extrapolate it to the big one, and it works, often to within a percent over several orders of magnitude. The whole industry of scaling-law forecasting depends on that smoothness.

Emergent abilities are the place where that comfort breaks. The claim is not that loss stops being smooth — loss stays smooth — but that *downstream task performance*, the thing customers and benchmarks actually care about, can be a wildly nonlinear function of that smooth loss. A model whose loss improved by a boring 5% just acquired the ability to solve a class of problems it could not touch before. If that is true in the strong sense, it is genuinely bad news for planning: you cannot promise a capability you cannot forecast, and you cannot forecast a step function from points on the flat part.

So the stakes of the "mirage" question are practical, not philosophical. Here is the assumption-versus-reality framing a senior engineer should keep in their head.

| Question | The naive 2022 view | The reconciled 2026 view |
|---|---|---|
| Are emergent abilities real? | Yes, sharp and unpredictable | Often a metric artifact, but a real loss-threshold transition survives |
| Can you forecast them? | No — they are unpredictable by definition | Yes, if you track pre-training loss and use continuous metrics |
| What causes the cliff? | A qualitative change in the model | Usually error compounding under exact-match scoring |
| Does the model "snap" into a new mode? | Implied | Per single run, yes; across seeds, the distribution shifts smoothly |
| What should I report? | Accuracy on the headline benchmark | A continuous, proper-scoring metric plus the loss it sits at |

The two views are not as far apart as the headlines suggested. The strong "pure mirage" reading and the strong "real magic" reading are both wrong; the truth is a specific, checkable middle. Getting to that middle is the work of this post, and it starts with pinning down exactly what Wei et al. claimed.

> If you take one thing from this post: do not score capability forecasting with exact-match. The cliff you are trying to predict is, more often than not, a cliff you built with your own metric.

## 1. What Wei et al. actually claimed

**Senior rule of thumb: before you argue about whether a phenomenon is real, write down its definition precisely enough that someone could falsify it.** Wei et al. did exactly that, and the precision is what made the later rebuttal possible.

Their definition: *"An ability is emergent if it is not present in smaller models but is present in larger models."* The operational test is a plot of task performance against scale (parameters, training FLOPs, or training tokens) in which performance hovers at random-chance baseline across many model sizes and then rises sharply past some threshold. Two properties matter. First, **sharpness**: the transition is abrupt, not a gentle ramp. Second, **unpredictability**: you could not have forecast the post-threshold performance by extrapolating the pre-threshold (chance-level) points, because chance-level points carry no slope to extrapolate.

The examples were concrete and numerous. On multi-step arithmetic — specifically 3-digit addition and subtraction, and 2-digit multiplication — models below a few times $10^{22}$ training FLOPs scored essentially zero, and models above it scored well. Modular arithmetic, phonetic transliteration from the International Phonetic Alphabet, recovering a word from its scrambled letters (word unscramble), and the Massive Multitask Language Understanding benchmark (MMLU) all showed the same shape. Across the BIG-Bench suite, dozens of individual tasks were flat-then-sharp. And two now-central techniques — chain-of-thought prompting and instruction-following after instruction tuning — *hurt or did nothing* below a scale threshold and only started helping above it. Chain-of-thought, in particular, only beats standard prompting once the model is large enough; on a small model, asking it to "think step by step" makes things worse.

The threshold was not a single universal number. It depended on the task, but the typical band Wei et al. reported was $10^{22}$ to $10^{24}$ training FLOPs. To anchor that: GPT-3's 175-billion-parameter model trained on roughly 300 billion tokens lands near $3 \times 10^{23}$ FLOPs using the standard $C \approx 6ND$ estimate, right in the middle of the emergent band, which is why GPT-3 was the model on which so many of these abilities first showed up.

The conclusion Wei et al. drew was epistemic, and it is the part that worried practitioners: *you cannot forecast big-model capability by extrapolating small-model curves.* If the curve is flat at chance until the threshold, the flat part tells you nothing about how high the jump will be. That is a real problem for anyone allocating a nine-figure training budget on the promise of a specific capability.

### The honest version of the worry

It is worth steelmanning the original claim before tearing into it, because the careless version ("models gain magic powers at scale") is a strawman that Wei et al. did not make. The careful version is this: *under the metrics the community actually used to evaluate these tasks*, performance was flat-then-sharp, and the flat region was uninformative. That is an empirically true statement about a large body of published benchmark curves. The question Schaeffer et al. raised is not whether those curves exist — they do — but whether the shape is a property of the model or a property of the ruler.

### Why "unpredictable" was the load-bearing word

The sharpness of the curve is the part that grabbed headlines, but it is the *unpredictability* that has real economic consequences, and the two are not the same. A curve can be sharp and still predictable — a logistic sigmoid is steep in the middle, but if you can see the early ramp you can fit it and forecast the inflection. What Wei et al. claimed was stronger: that the pre-threshold region is flat *at the chance baseline*, carrying literally no slope. A flat line at 25% multiple-choice accuracy has the same shape whether the true latent capability is rising fast or not rising at all, so you cannot tell from it what comes next. That is the worry: not steepness, but the total absence of forecasting signal in the region you can actually measure.

This framing also tells you exactly where to attack the claim. If you can find *any* metric on which the pre-threshold region is not flat — any measurement that shows a slope on the small models — then the unpredictability claim fails, because a slope can be extrapolated. The entire mirage argument is, at bottom, the discovery that such metrics exist and are easy to construct: the pre-threshold region only looks flat because the discontinuous metric flattens it. Switch metrics and the slope reappears. So the debate is not really about whether the model changes abruptly; it is about whether the flat region is informative, and the answer turns out to be that the flatness is usually an artifact of how we chose to look.

## 2. The mirage: how a metric manufactures a cliff

**Senior rule of thumb: a benchmark number is a model output passed through a scoring function, and the scoring function has as much power to shape the curve as the model does.** This is the single most important idea in the whole debate, and once you see it you cannot unsee it.

Schaeffer, Miranda, and Koyejo's argument has a clean logical structure. Hold the model outputs *completely fixed*. Do not retrain anything. Now score those identical outputs two ways: once with a metric that is **discontinuous** or strongly nonlinear in the underlying per-token quality, and once with a metric that is **continuous and roughly linear**. The discontinuous metric produces a sharp jump; the continuous metric produces a smooth curve. Same outputs, two stories. The figure below is the picture to burn into memory, because it is the entire argument in one frame.

![A two-panel chart plotting eight identical model checkpoints, with the left panel using an exact-match metric that stays flat then jumps to a cliff and the right panel using token edit distance that rises as a smooth predictable line](/imgs/blogs/emergent-abilities-scaling-2.png)

Both panels plot the same eight checkpoints. On the left, scored by exact-match grade, the first five checkpoints sit near zero and the sixth leaps up — a textbook emergent cliff. On the right, scored by token edit distance (a continuous measure of how close the answer is, even when it is not exactly right), the same eight checkpoints trace a straight line on log-log axes. Nothing about the model changed between the two panels. Only the metric changed.

### The two metrics that do the damage

Schaeffer et al. did the bookkeeping. They took the set of BIG-Bench tasks that prior work had hand-annotated as emergent and asked which metrics those tasks used. The answer: **more than 92% of the emergent tasks used one of just two metrics** — Multiple-Choice Grade (you get credit only if the highest-probability option is exactly the correct one) or Exact-String-Match (you get credit only if your generated string is character-for-character identical to the target). Both are discontinuous: they return a hard 0 or 1, with no partial credit for being close. The figure below lays out which metric families manufacture emergence and which do not.

![A matrix comparing four scoring metrics across whether they produce a step jump, whether they are forecastable, how heavily BIG-Bench uses them, and the verdict on whether each manufactures emergence](/imgs/blogs/emergent-abilities-scaling-5.png)

The matrix makes the pattern legible. Multiple-Choice Grade and Exact-Match both produce step jumps, are not forecastable from the flat region, are heavily used in BIG-Bench, and manufacture emergence. Token Edit Distance and the Brier score (a proper scoring rule for probabilistic predictions) both produce smooth slopes, are predictable, are rarely used, and produce no false emergence. The choice between these two columns is the choice between seeing a phase transition and seeing a power law.

### Why exact-match turns smooth into sharp

The mechanism is multiplicative, and it is the most quantitative part of the argument. Suppose a task requires the model to produce a target string of $N$ tokens, and suppose the model's per-token accuracy is $p$ — the probability it gets any single token right. Assume, generously, that token errors are roughly independent. Then exact-match success, which requires *every one* of the $N$ tokens to be correct, is approximately:

$$\text{success}_{\text{exact}}(p, N) \approx p^{N}$$

Now here is the trick. The per-token accuracy $p$ improves *smoothly* with scale — that is just the loss curve, dressed up. But $p^N$ does not improve smoothly. For large $N$, $p^N$ is a function that stays near zero across a wide range of $p$ and then shoots up only when $p$ gets very close to 1. The figure below traces that compounding explicitly.

![A pipeline showing per-token accuracy rising smoothly, an answer length of N tokens, the exact-match requirement that all N be correct giving success approximately p to the N, worked values at p equals 0.80 and 0.99, and the resulting cliff when plotted against scale](/imgs/blogs/emergent-abilities-scaling-3.png)

Let us put numbers on it, because the numbers are genuinely striking. Take a 5-digit addition task, so the answer is roughly $N = 5$ tokens. The way exact-match interacts with per-token accuracy is what produces the apparent jump.

| Per-token accuracy $p$ | Smooth, gradual | Exact-match success $p^5$ | Apparent capability |
|---|---|---|---|
| 0.60 | (loss is fine) | $0.60^5 \approx 0.078$ | "can't do it" |
| 0.70 | (loss improving) | $0.70^5 \approx 0.168$ | "can't do it" |
| 0.80 | (loss improving) | $0.80^5 \approx 0.328$ | "barely" |
| 0.90 | (loss improving) | $0.90^5 \approx 0.590$ | "getting there" |
| 0.95 | (loss improving) | $0.95^5 \approx 0.774$ | "mostly works" |
| 0.99 | (loss improving) | $0.99^5 \approx 0.951$ | "solved" |

Read the middle column. The per-token accuracy is climbing in even, smooth steps — 0.60, 0.70, 0.80 — exactly the boring power-law improvement the loss curve predicts. But the right column, exact-match success, goes 0.078, 0.168, 0.328, 0.590, 0.774, 0.951. Plotted against scale (which drives $p$ smoothly), that right column is a cliff. And it gets sharper the longer the answer: at $N = 20$ tokens, $0.95^{20} \approx 0.358$ but $0.99^{20} \approx 0.818$ — the same 4-point improvement in $p$ more than doubles exact-match success. The longer the target string, the more violent the manufactured jump.

This is not a hand-wave. It is arithmetic. A metric that multiplies many smoothly-improving factors together will, for any nontrivial $N$, produce a curve that looks flat and then leaps. The "emergence" lives in the exponent, not in the model.

### The geometry of the exponent

It helps to look at the shape of $p^N$ as a function of $p$ for a fixed $N$, because the geometry is what produces the cliff. Near $p = 1$, take the logarithm: $\log(\text{success}) = N \log p$. For $p$ close to 1, $\log p \approx -(1 - p)$, so $\log(\text{success}) \approx -N(1-p)$ and $\text{success} \approx e^{-N(1-p)}$. The success rate is therefore controlled by the product $N(1-p)$ — the expected number of token errors in the answer. When $N(1-p)$ is large (many expected errors), success is near zero no matter how the individual factors trade off. When $N(1-p)$ drops below about 1 (fewer than one expected error per answer), success climbs rapidly toward 1. The transition is sharp precisely because it happens as $N(1-p)$ crosses order unity, and $1 - p$ is shrinking smoothly with scale.

This gives a clean rule of thumb you can carry around: **a multi-token exact-match task will appear emergent once per-token error $1 - p$ falls below roughly $1/N$.** For a 5-token answer the knee is around $p \approx 0.8$; for a 20-token answer it is around $p \approx 0.95$; for a 100-token answer (a paragraph-length output) it is around $p \approx 0.99$. Longer required outputs push the apparent emergence to higher per-token accuracy and make the visible jump steeper. This is also why generation tasks look more dramatically emergent than short classification tasks: the answer string is longer, so the exponent is larger, so the cliff is sharper.

### Token edit distance: the same outputs, no cliff

The fix Schaeffer et al. proposed is to use a metric that does not multiply. Token edit distance — the number of single-token insertions, deletions, or substitutions needed to turn the model's output into the target, normalized — gives partial credit. A model that gets 4 of 5 tokens right scores nearly as well as one that gets all 5, instead of scoring a hard zero. Under edit distance, the per-token improvement shows up *linearly*, and the curve is smooth. The right panel of the cliff-versus-smooth figure is exactly this: the same eight checkpoints, scored by a metric that does not punish near-misses with a zero, trace a clean line. The Brier score does the same job for probabilistic predictions: it rewards calibrated confidence continuously rather than collapsing to a 0/1 grade.

There is a subtle but important point here about *why* edit distance is the right linearization and not just an arbitrary softer metric. Edit distance, normalized by target length, is approximately $1 - p$ — it measures the fraction of tokens you got wrong. So plotting $1 - \text{(normalized edit distance)}$ against scale is plotting $p$ against scale, which is essentially the loss curve in disguise, and the loss curve is smooth by construction. The continuous metric works not because it is "nicer" but because it tracks the same underlying quantity the loss tracks, *before* the exponent amplifies it. The discontinuous metric, by contrast, measures $p^N$, which is the loss after the exponent has done its damage. The whole disagreement between the two panels is the disagreement between measuring a quantity and measuring that quantity raised to a large power.

### Brier score and proper scoring rules

For multiple-choice and classification tasks, the right continuous replacement is a *proper scoring rule* such as the Brier score. The Brier score for a probabilistic prediction is the mean squared difference between the predicted probability assigned to each option and the indicator of whether that option was correct. A proper scoring rule has the defining property that it is optimized, in expectation, by reporting your true probabilities — so it cannot be gamed by being over- or under-confident. Crucially for our purposes, the Brier score responds *continuously* to the probability mass the model puts on the correct answer. As that mass rises smoothly with scale, the Brier score improves smoothly, even while Multiple-Choice Grade (which only flips when the correct answer becomes the argmax) stays pinned at the chance floor. This is the formal reason the MMLU "cliff" dissolves under Brier scoring: the model was getting steadily more confident in the right answer the whole time; the grade just could not see it until the confidence crossed the argmax threshold.

The practical takeaway is sharp: **if you want to forecast a capability, never score it with exact-match.** Use a continuous metric — edit distance for generation, a proper scoring rule for classification, raw log-probability of the target for anything — fit the smooth curve, and extrapolate. The cliff you were worried about was, in large part, a cliff you built.

## 3. The smoking gun: inducing emergence in vision models

**Senior rule of thumb: the strongest evidence that a phenomenon is an artifact is to reproduce it on purpose, in a system where the "real" version cannot possibly exist.** This is the experiment that elevated Schaeffer et al. from "plausible alternative explanation" to "Outstanding Paper."

If emergence is really about the metric, then it should not require a language model at all. You should be able to take a system with no language abilities, no scale-dependent reasoning, nothing — and *manufacture* a phase transition just by choosing a harsh metric. So they did. They trained convolutional networks and autoencoders on CIFAR-100, a standard image dataset, and watched how performance scaled with model size under two different metrics. The figure below shows the result.

![A branching graph showing CIFAR-100 convolutional networks and autoencoders scored two ways, where a smooth mean reconstruction error metric gives gradual improvement and a harsh all-pixels-exact metric induces a sharp jump, leading to the lesson that emergence can be an artifact](/imgs/blogs/emergent-abilities-scaling-7.png)

Under a smooth metric — mean reconstruction error, which measures on average how far each reconstructed pixel is from the target — the vision models improved gradually with scale. No emergence. No cliff. Just the boring downward slope you would expect. Then they swapped in a harsh, all-or-nothing metric: a reconstruction counts as "correct" only if essentially every pixel is exact (or within a tiny tolerance). Under that metric, the exact same models showed a sharp jump — induced emergence in a vision system that has no concept of arithmetic, reasoning, instruction-following, or any of the abilities the LLM emergence literature is about.

The conclusion is hard to escape: emergence, in the sharp-and-sudden sense, can be a pure property of the scoring function. If you can summon it from a CIFAR-100 autoencoder by changing a metric, then its appearance in a language model is, at minimum, *suspect* until you re-score with a continuous metric and check whether it survives.

### The three tests, together

It is worth noting that the CIFAR experiment was the third leg of a three-part argument, because all three together are what made the paper persuasive. First, they took GPT-3's arithmetic results and *predicted in advance* what the curves would look like under a continuous metric versus exact-match — and the predictions held: continuous metric smooth, exact-match sharp. Second, they ran the BIG-Bench meta-analysis and found the 92%-discontinuous-metric statistic, showing the artifact was pervasive, not cherry-picked. Third, they induced emergence in vision to show the mechanism is metric-driven and architecture-independent. A prediction confirmed, a population characterized, and a mechanism demonstrated from scratch. That is what a complete causal argument looks like.

## 4. The rebuttal: why the loss-threshold transition is real

**Senior rule of thumb: when a debunking is too clean, look for the experiment the debunkers did not run.** The mirage argument is correct about a lot, but it has a gap, and Du et al. (2024) drove a truck through it.

Here is the gap. Schaeffer et al. showed that *plotted against scale*, and *under discontinuous metrics*, emergence is an artifact. But scale is a proxy for the thing that actually drives capability, which is how well the model has learned to predict text — its pre-training loss. What happens if you plot downstream performance not against parameters or FLOPs, but directly against pre-training loss? And what happens if you do that *with a continuous metric*, so the multiplicative-compounding objection is off the table?

Du et al. did exactly this, in "Understanding Emergent Abilities from the Loss Perspective" (NeurIPS 2024), and the result is the strongest rebuttal to the pure-mirage view. The figure below is their central finding.

![A chart plotting downstream accuracy against pre-training loss falling from left to right, with accuracy near chance for all models above a task-specific loss threshold and climbing sharply once the loss drops below it](/imgs/blogs/emergent-abilities-scaling-6.png)

What they found is that downstream performance on certain tasks stays near chance for *every* model whose pre-training loss is above a task-specific threshold — regardless of that model's size — and then climbs once the loss drops below the threshold. Two models of very different sizes with the same pre-training loss get the same downstream performance. And the abruptness around the threshold survives *even when you score with a continuous metric*. This is the crucial point: the compounding objection says exact-match manufactures cliffs, but here the cliff appears under a smooth metric too. So at least some of what we call emergence is not a scoring artifact. It is a real transition that happens to live at a particular point on the loss axis.

### Why a loss threshold is a more honest x-axis than scale

Plotting against loss instead of against scale is not a cosmetic change; it is a conceptual one. Scale (parameters, FLOPs, tokens) is an *input* you control. Loss is an *outcome* that summarizes how much the model actually learned. Two models can have wildly different scales and similar losses (a small model trained long versus a big model trained short — the [Chinchilla allocation question](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) in another guise), and when they do, they have similar capabilities. So loss is the better predictor. And once you predict capability from loss, you have turned an "unforecastable" emergent ability into a forecastable one: predict your model's loss from a scaling law (which is smooth and reliable), then map loss to downstream performance through the threshold curve.

The mechanism Du et al. point to is intuitive: some tasks require the model to have learned a prerequisite structure (a reliable addition algorithm, a robust representation of multiple-choice format, a stable mapping from instruction to action) that only crystallizes once the model is good enough at next-token prediction overall. Below the loss threshold, that structure has not formed and the task is at chance. Above it, the structure is in place and the task works. That is a genuine phase-transition-like phenomenon — not in the mystical sense, but in the precise sense that there is a critical value of a control parameter (loss) past which behavior changes qualitatively.

### A worked loss-threshold forecast

Let us make the loss-threshold idea concrete with numbers, because it is the part that converts a scary "unforecastable" capability into a line on a planning spreadsheet. Suppose you have measured a task and found that downstream accuracy (under a *continuous* metric, so we are not fooling ourselves) is essentially at chance whenever pre-training loss $L$ is above about 2.2 nats per token, and rises steeply once $L$ drops below that. You have three small models with losses 2.6, 2.4, and 2.3, all near chance, plus one mid model at 2.1 that scores 0.4 — your first point past the threshold. Naively, three flat points and one jump look unforecastable. But you also have a [Chinchilla-style loss law](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) fit on your runs that predicts loss from compute: it says your planned full model will reach $L \approx 1.85$. Since 1.85 is comfortably below the 2.2 threshold, you can state with confidence that the capability will be present — and, using the slope of accuracy-versus-loss past the threshold, estimate roughly how present. You have forecast an "emergent" ability from compute, by routing the prediction through loss. The flat region told you nothing; the loss law told you everything.

The same logic explains why two of the models that confused practitioners — a small-but-long-trained model and a large-but-short-trained model — ended up with the same capability despite very different parameter counts: they landed at the same loss, and loss is what the threshold is defined on. If you had forecast from parameter count you would have been wrong about both; forecasting from loss is right about both.

### What this does and does not rescue

Be careful about what Du et al. proves. It does *not* resurrect the strong claim that emergence is unforecastable — quite the opposite, it makes emergence forecastable via loss. And it does not contradict Schaeffer et al. about exact-match manufacturing extra sharpness; the multiplicative compounding is still real and still inflates the apparent jump. What Du et al. establishes is narrower and more durable: there exists a real, loss-located transition underneath the metric artifact. The metric exaggerates the cliff; the loss threshold means there is a cliff there to exaggerate.

The two findings are best understood as composing, not competing. Picture the causal chain: scale drives loss down smoothly (scaling law); loss crossing a task-specific threshold flips a latent capability on (Du); and then the discontinuous metric amplifies even that real transition into a sharper-looking cliff than it truly is (Schaeffer). Each stage is doing real work. Schaeffer is right that the metric exaggerates; Du is right that there is something to exaggerate; and both are right that if you remove the metric amplification by scoring continuously and re-express the x-axis as loss, the result is forecastable. The mistake is to treat them as opposing verdicts when they are consecutive links in the same chain.

## 5. Lu et al.: maybe it is just in-context learning

**Senior rule of thumb: when a capability looks like it switched on, ask whether the model learned something new or whether it just got good enough to use something it always had.** Lu et al. (2024), "Are Emergent Abilities Just In-Context Learning?" (ACL 2024), pursue this line and reach a deflationary conclusion.

Their argument is that most apparent emergent abilities decompose into three ingredients that are individually unsurprising: **in-context learning** (the model's ability to pick up a pattern from the few-shot examples in the prompt), **memory** (facts and procedures stored in the weights), and **linguistic knowledge** (general competence with language structure). Each of these improves smoothly with scale. What looks like a sudden new reasoning ability, they argue, is often just the point at which in-context learning becomes reliable enough to chain together the memory and linguistic pieces into a correct answer. There is no separate "reasoning module" that snaps into existence; there is a smoothly-improving in-context learner that crosses a usefulness threshold.

This matters for the debate because it removes the *mystery* from the remaining real transitions. Even where a genuine loss-threshold transition exists (Du et al.), Lu et al. suggest its content is mundane: the model got reliable enough at in-context learning to compose abilities it could already partially do. That is a much less alarming story than "the model spontaneously developed reasoning." It also has a practical edge: if emergence is largely in-context learning becoming reliable, then evaluation choices that suppress in-context learning (zero-shot, adversarial prompts) will show different, often smoother, curves — which is consistent with the broader finding that the curve shape is sensitive to the evaluation setup.

### The relationship to chain-of-thought

Lu et al.'s framing also explains one of Wei's headline examples cleanly. Chain-of-thought prompting "emerges" — it only helps above a scale threshold — because chain-of-thought *is* a form of in-context learning: the model has to be a good enough in-context learner to actually follow the demonstrated reasoning format and keep its own intermediate steps coherent. Below that threshold, asking for step-by-step reasoning gives you incoherent steps that hurt more than they help. Above it, the in-context learning is reliable enough that the extra reasoning tokens pay off. No new module; a smoothly-improving substrate crossing a usefulness line.

The chain-of-thought case is also where the compounding story and the in-context-learning story reinforce each other rather than compete. A chain-of-thought answer is *long* — a multi-step derivation can run dozens or hundreds of tokens — and it is graded by whether the final extracted answer is exactly right. So you have both effects stacked: a long exact-match target (large $N$ in the $p^N$ compounding) and a capability (coherent multi-step reasoning) that depends on in-context learning becoming reliable. Either effect alone would manufacture an apparent cliff; together they make chain-of-thought one of the most dramatically "emergent"-looking techniques in the literature. Decompose it and there is no mystery: a smoothly-improving in-context learner, scored by a metric that punishes any single misstep across a long answer with a zero.

### Why the evaluation setup quietly decides the curve shape

A theme that runs under all of this — Schaeffer's metric, Lu's in-context learning, Du's loss axis — is that the *shape of an emergence curve is a joint property of the model and the entire evaluation protocol*, and the protocol has more degrees of freedom than people admit. Number of in-context examples, prompt format, whether you extract the answer with a strict regex or a lenient parser, whether you measure the argmax or the probability mass, whether you allow partial credit, how many samples you draw, the chance baseline of the task — every one of these knobs changes the curve, often more than a generation of model scaling does. A task that looks emergent under a strict zero-shot exact-match regex can look perfectly smooth under a five-shot, lenient-parser, log-probability regime on the identical checkpoints. When you read an emergence claim, the first question is not "how big was the model?" but "how exactly did they score it?" — because the scoring choices are where most of the apparent magic lives.

## 6. Zhao and Saphra: breakthroughs are random per run

**Senior rule of thumb: "the model" is a fiction — there is a distribution of models you could have trained, one per random seed, and the distribution behaves very differently from any single draw.** Zhao, Saphra et al. (2025), "Random Scaling of Emergent Capabilities," add the final and most subtle piece.

They asked a question the earlier work mostly ignored: what happens across random seeds? If you train the same architecture at the same scale many times, varying only the random seed, do they all acquire a given capability at the same point? The answer is no, and the pattern is striking: across seeds, the capability is **bimodal**. Some runs break through and acquire the ability; others, identical in every controllable respect, do not — at the same scale. A single run's acquisition of a capability is, to a real degree, a coin flip.

And yet — this is the reconciling insight — the *aggregate* is smooth and predictable. If you look at the fraction of seeds that have broken through as a function of scale, that fraction rises as a smooth, well-behaved curve. The distribution shifts predictably even though any individual draw from it is abrupt and stochastic. Both things are true: per-seed breakthroughs are real and sudden, and the across-seed distribution is a smooth, forecastable shift. The timeline below places this in the arc of the whole debate.

![A timeline from 2022 to 2025 marking Wei claiming emergence is real, Schaeffer showing the metric mirage, Lu reducing emergence to in-context learning, Du establishing the loss threshold, and Zhao and Saphra showing per-seed breakthroughs that are smooth in aggregate](/imgs/blogs/emergent-abilities-scaling-4.png)

The timeline tells the story of a field correcting itself in both directions. Wei et al. (June 2022) named the phenomenon and made the alarming unpredictability claim. Schaeffer et al. (April 2023) showed much of it was a metric artifact. Lu et al. (September 2023) argued the residue was in-context learning, not new reasoning. Du et al. (March 2024) showed a real loss-threshold transition survives even under continuous metrics. And Zhao, Saphra et al. (February 2025) reconciled the per-run abruptness with the aggregate smoothness. The endpoint is not "mirage" or "magic" but a precise, layered understanding.

### Why per-seed randomness matters for planning

The seed story has a brutal practical implication that is easy to miss. If acquiring a capability at a given scale is partly a coin flip, then a single training run that fails to show a capability does not prove the capability is out of reach — you may have drawn an unlucky seed. Conversely, a single run that shows a dazzling new capability does not guarantee your *next* run at the same scale will reproduce it. This is why labs that can afford it train multiple seeds for capabilities they care about, and why a benchmark result from a single checkpoint should be read as one sample from a distribution, not as a fixed property of the recipe.

### The two layers of randomness, kept separate

It is worth being precise about what is random and what is not in the Zhao–Saphra picture, because conflating the two layers is a common source of confusion. There is *within-run* variation — the stochasticity of which specific seed broke through — and there is *across-run* structure — the smooth way the breakthrough probability shifts as you scale. The within-run layer is the thing that feels like magic: one run snaps into a capability, another identical-on-paper run does not. The across-run layer is the thing that restores predictability: if you ask "what fraction of seeds at this scale will have the capability?" the answer is a smooth, well-behaved function of scale that you can fit and extrapolate.

The right mental model is a distribution that *slides*. At small scale, almost no seeds break through; the breakthrough-probability is near zero. As scale increases, the distribution slides so that more and more seeds land past the breakthrough point, until at large scale almost all of them do. Any single draw from this sliding distribution looks abrupt — you either got the capability or you did not — but the location of the distribution moves predictably. This is exactly analogous to how individual molecules in a warming liquid cross the boiling threshold stochastically while the *fraction* that have vaporized rises smoothly with temperature. The phase-transition language, often used loosely in the emergence debate, is actually apt here in a precise statistical sense: a smooth shift in a control parameter produces a smooth shift in an aggregate, on top of which individual units behave discretely.

For forecasting, this means you should never promise "the model will have capability X." You should promise "at the target scale, we expect capability X in a given fraction of runs, with this much run-to-run variance." That is a weaker but *honest* and *correct* statement, and it is the one the evidence supports. A roadmap built on the median run plus a budget for re-rolls is robust; a roadmap built on the assumption that one run's result is a fixed property of the recipe will eventually be embarrassed by a seed.

## 7. The reconciled picture: what emergence decomposes into

**Senior rule of thumb: a mature understanding of a contested phenomenon is usually a decomposition, not a verdict.** By mid-2026, the field does not say "emergence is real" or "emergence is fake." It says: reported emergence is a *sum* of three distinguishable things, and you handle each one differently. The figure below is that decomposition.

![A tree decomposing a reported emergent ability into a metric artifact branch handled by continuous metrics, a loss-threshold branch handled by tracking pre-training loss, and a per-seed randomness branch handled by treating breakthroughs as stochastic](/imgs/blogs/emergent-abilities-scaling-8.png)

The first branch is the **metric artifact** (Schaeffer et al.). A large share of reported emergence is manufactured by discontinuous metrics through multiplicative compounding. The fix is to use continuous, proper-scoring metrics, at which point the artifact disappears and the underlying curve is smooth. This branch is genuinely a mirage, and it accounts for a lot of the published cliffs.

The second branch is the **genuine loss-threshold transition** (Du et al.). Underneath the artifact, some tasks really do switch on below a task-specific pre-training loss, and they do so even under continuous metrics. The fix here is not to "remove" the transition — it is real — but to *forecast* it: track pre-training loss, fit the loss-to-performance map, and predict where the transition lands. This branch is real but predictable, which is the best of both worlds.

The third branch is **per-seed stochasticity** (Zhao, Saphra et al.). Individual runs break through abruptly and randomly, while the across-seed distribution shifts smoothly. The fix is to treat any single-run breakthrough as a random draw, forecast the *fraction* of runs that will break through rather than promising a binary outcome, and train multiple seeds for capabilities that matter.

Notice that all three fixes point the same direction: **stop forecasting from scale-versus-exact-match curves, and start forecasting from loss-versus-continuous-metric curves, treating single runs as samples.** The three papers that complicated the mirage story all converge on the same practical advice.

### Common objections, and what each gets right

Because the debate has been heated, it accumulated a set of recurring objections from each camp. It is worth walking the strongest of them, because the reconciled view is precisely the one that survives all of them.

*"If it is just a metric artifact, why does the capability feel so real when you use the model?"* This objection is correct that the capability is real — a model that can do 5-digit addition really can do it. But "the capability is real" and "the sharp curve is a metric artifact" are not in tension. The model's per-token competence rose smoothly; the *sharpness of the benchmark curve* is the artifact, not the capability. You can have a genuinely useful new behavior whose apparent suddenness is manufactured by exact-match.

*"If it is a real loss-threshold transition, doesn't that vindicate the original emergence claim?"* Partly, but it guts the part that mattered. The original claim's teeth were in *unpredictability*. A loss-threshold transition is predictable: forecast the loss, map loss to capability. So Du et al. vindicates "there is a real transition" while refuting "you cannot forecast it." The surviving claim is much weaker and much more useful than the original.

*"Per-seed randomness sounds like an excuse for irreproducible results."* No — it is a measured, structured phenomenon with a smooth aggregate. The point is not "anything can happen, shrug" but "the breakthrough probability is a forecastable function of scale, and you should report and budget distributionally." That is more rigor, not less.

*"Continuous metrics just hide the fact that the model can't actually complete the task."* This is the sharpest objection, and it has a real edge: for a *report card*, exact-match is the honest number, because users need the whole answer right. The resolution is the metric-by-purpose split that runs through this whole post. For forecasting, use the continuous metric to recover signal; for reporting, use exact-match and translate via compounding. Continuous metrics do not hide anything — they expose the smooth signal that exact-match buries — but they answer the forecasting question, not the report-card question.

The reconciled view is exactly the position that concedes what is true in each objection: the capability is real (objection 1), the transition is real but predictable (objection 2), the randomness is structured (objection 3), and the metric choice is purpose-dependent (objection 4). That is why it is the consensus — it is the only view that does not have to deny a piece of the evidence.

### How this connects to the rest of the scaling-laws literature

This decomposition does not sit in isolation; it slots neatly into two neighboring results in this series. The [observational downstream scaling laws](/blog/machine-learning/scaling-laws/observational-downstream-scaling-laws) work (Ruan et al. 2024) shows you can predict downstream performance — including the supposedly emergent kind — by building a low-dimensional capability space over many existing models and connecting it to benchmarks through a logistic link function. That logistic link is exactly the move that turns an apparent jump into a smooth sigmoid: it is the metric-fix and the loss-threshold-forecast wrapped into one statistical model. And the [broken neural scaling laws](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws) work (Caballero et al. 2022) provides a functional form that models sharp inflections — including emergence-shaped ones — as *smooth breaks* in a piecewise power law rather than true discontinuities. Between them, the observational and broken-law results give you the tools to forecast precisely the curves that the naive emergence story said were unforecastable. Emergence, properly handled, is not the exception to scaling-law predictability; it is a place where you have to be careful about your metric and your x-axis, after which predictability returns.

### What makes a metric "discontinuous," precisely

The word "discontinuous" gets thrown around loosely, so it is worth pinning down what actually distinguishes the metrics that manufacture emergence from the ones that do not. A metric is dangerous for forecasting when it is a *thresholded* or *winner-take-all* function of the model's underlying probabilities. Multiple-Choice Grade thresholds at the argmax: it returns 1 only if the correct option has strictly the highest probability, and 0 otherwise, so a smooth rise in the correct option's probability produces no change in the grade until it overtakes the runner-up, at which point the grade flips. Exact-String-Match thresholds at perfection: it returns 1 only if *every* token matches, so it is the product of $N$ per-token indicator functions, each of which is itself a hard threshold on that token's argmax. Both are step functions composed with the model's smooth outputs, and a step function composed with a smooth ramp is a step.

A safe metric, by contrast, is one that responds *monotonically and without a plateau* to the underlying probability mass. Token edit distance moves a little for every token you fix. Brier score moves a little for every increment of probability mass you shift onto the correct answer. Log-probability of the target moves continuously with the model's actual confidence. The test you can apply to any metric you are about to use for forecasting is simple: *if the model gets slightly better — shifts a little probability mass toward the right answer — does this metric register that improvement, or does it stay flat until some threshold is crossed?* If it stays flat, it will manufacture a cliff and blind your forecast. If it registers the improvement, it gives you the slope you need.

### Why so many BIG-Bench tasks fell into the trap

The 92% figure is not an accident of one careless benchmark; it reflects a structural bias in how community benchmarks get built. Discontinuous metrics are *attractive* to benchmark authors for reasons that have nothing to do with forecasting. They are simple to define and explain ("did the model get the exact answer?"). They are unambiguous, requiring no judgment about how much partial credit a near-miss deserves. They match the way humans grade a quiz. And they produce a single clean accuracy number that is easy to put in a leaderboard. Every one of these is a virtue for a *report card* and a vice for a *forecast*. The benchmark ecosystem optimized for clean, human-intuitive report cards, and in doing so it accidentally optimized for the exact metric family that hides forecasting signal. The 92% statistic is the fingerprint of that incentive. It also means the fix is mostly free: the model outputs already contain the continuous signal, and re-scoring an existing benchmark with a continuous metric is a cheap offline computation, not a new training run.

## 8. Working a forecast end to end

**Senior rule of thumb: the test of whether you understand a phenomenon is whether you can make a quantitative prediction with it.** Let us walk a concrete forecasting exercise that uses every piece above, with made-up-but-realistic numbers, to show the machinery in action.

Suppose you are planning a model and you want to know whether it will be able to do 5-digit addition reliably (say, above 80% exact-match) by the time you finish training. Naively, you would look at your small-model runs, see 0% exact-match across all of them, and conclude you have no signal. That is the trap. Here is how to do it properly.

**Step 1: pick a continuous metric.** Instead of exact-match, score 5-digit addition with per-token accuracy $p$, or equivalently with token edit distance. Across your small runs you now see a real, rising signal: $p = 0.55, 0.62, 0.70, 0.76$ as the models grow. That is a smooth curve you can fit, where exact-match showed you a flat line of zeros.

**Step 2: forecast the continuous metric from loss.** Fit $p$ against pre-training loss $L$ (not against scale directly). Suppose the fit gives you $p \approx 1 - k \cdot \exp(\gamma L)$ for some constants, a smooth saturating curve. Your scaling law (a [Chinchilla-style](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) loss prediction) says your full model will reach $L$ low enough that $p \approx 0.97$.

**Step 3: convert the continuous metric back to the exact-match number you actually care about.** Now apply the compounding relationship deliberately, instead of being surprised by it. At $p = 0.97$ on a 5-token answer, exact-match success is $0.97^5 \approx 0.859$ — above your 80% target. At $p = 0.95$ it would be $0.95^5 \approx 0.774$, just below. So your forecast is not "we have no idea"; it is "we clear 80% exact-match if and only if per-token accuracy reaches about 0.96, which our loss forecast says happens around this point on the curve." That is a real, falsifiable prediction, derived through the very compounding that made the naive curve look like a cliff.

**Step 4: hedge for the seed.** Zhao and Saphra say acquisition is partly stochastic. So phrase the forecast distributionally: "at the target loss, we expect roughly 80% exact-match in the median run, but expect run-to-run variance — budget for two or three seeds if this capability is load-bearing." A single run landing at 70% does not mean the recipe failed.

This four-step recipe is the whole post operationalized. Continuous metric to get signal, loss as the x-axis to forecast, deliberate compounding to translate back to the headline number, and a distributional hedge for the seed. None of the four steps requires believing in magic, and none of them requires dismissing emergence as a pure illusion.

| Forecasting choice | The naive way | The reconciled way |
|---|---|---|
| Metric for the signal | Exact-match (flat zeros) | Per-token accuracy / edit distance (smooth) |
| X-axis | Scale (FLOPs) | Pre-training loss |
| Translating to the headline number | Hope | Apply $p^N$ compounding deliberately |
| Reporting the result | Single point estimate | Distribution across seeds |
| Forecast horizon | "Unpredictable" | Several-fold compute ahead |

## 9. Case studies from the literature and the field

Theory is cheap. Here are concrete episodes, each with the symptom, the wrong first read, the actual cause, and the lesson. They are drawn from the published record and from the kinds of failures that recur whenever a team evaluates a frontier model.

### 1. The MMLU cliff that was a multiple-choice grade

The symptom: MMLU accuracy is flat near 25% (random chance on a 4-way multiple-choice test) across several model sizes, then jumps. The wrong first read: the model "learned to reason about academic subjects" at a threshold. The actual cause: MMLU is scored by Multiple-Choice Grade, which gives credit only when the correct option has the single highest probability. The model's *probability mass* on the correct answer rises smoothly the whole time, but it does not become the argmax until late — so the hard 0/1 grade stays at chance and then flips. Re-scored by the probability assigned to the correct option (a continuous metric), MMLU improves smoothly from the start. The lesson: a 4-way multiple-choice grade has a chance floor of 25% and an argmax discontinuity built in; it is one of the most reliable emergence-manufacturing metrics in common use.

### 2. The arithmetic that GPT-3 "suddenly" learned

The symptom: 3-digit addition is at 0% exact-match for small GPT-3 variants and high for the largest. The wrong first read: arithmetic is an emergent reasoning ability. The actual cause: exact-match over a multi-token numeric answer compounds per-token accuracy as $p^N$. Schaeffer et al. predicted, before re-scoring, that a continuous metric would show a smooth curve and exact-match a sharp one — and that is precisely what the data showed. The lesson: any task with a multi-token exact-string target will show a manufactured cliff under exact-match; arithmetic is the canonical example because answers are several tokens and there is zero partial credit for being off by one digit.

### 3. Chain-of-thought that hurt the small model

The symptom: adding "let's think step by step" improves the large model and degrades the small one. The wrong first read: reasoning emerges at scale. The actual cause (per Lu et al.): chain-of-thought is an in-context learning task — the model must reliably follow the demonstrated reasoning format and keep its own intermediate steps coherent. Below a reliability threshold, the extra steps are incoherent and inject errors; above it, they help. The lesson: techniques that depend on in-context learning will look emergent because in-context learning reliability is itself a smoothly-improving substrate that crosses a usefulness line. Evaluate them as in-context learning, not as a separate faculty.

### 4. The induced phase transition in a CIFAR autoencoder

The symptom: a CIFAR-100 autoencoder shows a sharp capability jump with scale. The wrong first read: even vision models have emergent abilities. The actual cause: the metric was changed to an all-pixels-exact reconstruction grade; under mean reconstruction error the same models improve smoothly. The lesson: this is the control experiment — a system with no language, no reasoning, no arithmetic shows "emergence" purely because of a harsh metric. It is the cleanest possible demonstration that the cliff can be entirely a property of the ruler.

### 5. Two same-loss models, one capability

The symptom: a small model trained for a long time and a large model trained briefly have nearly identical pre-training loss — and nearly identical downstream task performance, including on a "emergent" task. The wrong first read: capability is about parameter count. The actual cause (per Du et al.): capability tracks pre-training loss, not scale. Below a task-specific loss threshold both models are at chance; above it both work, regardless of how they got there. The lesson: when you forecast a capability, forecast the loss first and map loss to capability; scale is only useful insofar as it predicts loss.

### 6. The benchmark that moved between seeds

The symptom: the same recipe at the same scale acquires a capability in one run and not in a re-run with a different seed. The wrong first read: the second run was buggy. The actual cause (per Zhao, Saphra et al.): per-seed acquisition is bimodal; the breakthrough is partly stochastic. The lesson: a single run is a sample. Forecast the fraction of seeds that break through, not a binary, and train multiple seeds for capabilities you are betting on.

### 7. The PassUntil-style resolution rescue

The symptom: a code-generation capability reads as exactly 0% on a small model, giving no forecasting signal. The wrong first read: the capability is absent and unpredictable. The actual cause: at low pass rates, ordinary sampling almost never produces a success, so the measured rate underflows to zero even though the *true* rate is a tiny positive number. Massive decode-time sampling (the PassUntil approach from Hu et al. 2023) resolves rates far below what normal evaluation can see — small enough to fit a smooth curve and extrapolate. The lesson: a measured zero is often a resolution problem, not a real zero; increase sampling resolution before declaring a capability unforecastable.

### 8. The instruction-following that "switched on"

The symptom: a base model ignores instructions; after instruction tuning and past a scale threshold, it follows them well. The wrong first read: instruction-following is an emergent reasoning ability. The actual cause: instruction-following is largely a format-and-in-context-learning capability that requires the model to reliably map an instruction to the demonstrated response format — which only becomes reliable once the base model's loss is low enough (Du et al.'s threshold) and its in-context learning is good enough (Lu et al.). The lesson: "instruction-following emerged" decomposes into a loss threshold plus reliable in-context learning, both forecastable; it is not a separate spark.

### 9. The inverse-scaling task that broke the forecast

The symptom: a task (Hindsight Neglect was the famous one in the GPT-4 report) gets *worse* with scale up to a point, defeating a simple smooth-curve forecast. The wrong first read: all capabilities are monotonic in scale, so any forecast is safe. The actual cause: some tasks are non-monotonic — U-shaped or inverse-scaling — and a single power-law fit cannot capture them. The lesson: emergence is not the only way a forecast fails; non-monotonicity does too. This is exactly where the [broken neural scaling laws](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws) functional form earns its keep, because it can express a break that reverses direction.

### 10. The headline metric that disagreed with the loss

The symptom: a model's pre-training loss improved nicely over a checkpoint, but the headline accuracy benchmark barely moved, prompting panic that "the model stopped learning." The wrong first read: training has saturated. The actual cause: the benchmark used a discontinuous grade sitting just below its argmax flip; the probability mass on correct answers was rising (visible in a continuous metric and in the loss) but had not yet crossed the threshold to change the hard grade. The lesson: when loss and a discontinuous benchmark disagree, trust the loss and switch to a continuous metric to see the real trajectory — the headline number will catch up when it crosses its own threshold.

### 11. The long-answer task that emerged later than the short one

The symptom: two tasks of similar difficulty for the model "emerged" at different scales — a short-answer classification variant showed signal early, while a long-form generation variant of the same underlying skill stayed flat much longer before jumping. The wrong first read: the generation variant is a harder, separate capability that requires more scale. The actual cause: the generation variant has a longer exact-match target, so its compounding exponent $N$ is larger, which pushes the apparent emergence to a higher per-token accuracy (the $1-p < 1/N$ rule). The skill was improving at the same rate underneath; the longer target simply delayed the point where exact-match could see it. The lesson: differences in apparent emergence scale between task variants are often differences in answer length, not differences in underlying capability. Normalize by target length before comparing.

### 12. The "regression" that was a stricter answer parser

The symptom: a new evaluation harness reported that a model had *lost* a capability it previously had, with the score dropping from strong to near-zero between two harness versions. The wrong first read: a training or checkpoint bug regressed the model. The actual cause: the new harness used a stricter answer-extraction regex that rejected outputs the old, lenient parser had accepted — same model outputs, harsher grading, manufactured collapse. This is the exact-match mirage in reverse: a metric change made a smooth capability look like it had fallen off a cliff. The lesson: when a score moves sharply between harness versions with the model fixed, suspect the harness before the model, and diff the scoring code — most "capability regressions" are parser regressions.

## 10. A note on the threshold band and FLOP arithmetic

It is worth being concrete about where these transitions land, because the $10^{22}$–$10^{24}$ FLOP band from Wei et al. is often quoted without grounding. Using the standard estimate that training compute is about $C \approx 6ND$ for $N$ parameters and $D$ tokens, you can place specific models in the band.

| Model (illustrative) | Parameters $N$ | Tokens $D$ | $C \approx 6ND$ FLOPs | Relative to band |
|---|---|---|---|---|
| Small base | $1.3 \times 10^{9}$ | $3 \times 10^{11}$ | $\approx 2.3 \times 10^{21}$ | below the band |
| Mid base | $7 \times 10^{9}$ | $1 \times 10^{12}$ | $\approx 4.2 \times 10^{22}$ | entering the band |
| GPT-3-scale | $1.75 \times 10^{11}$ | $3 \times 10^{11}$ | $\approx 3.2 \times 10^{23}$ | mid-band |
| Large frontier | $7 \times 10^{11}$ | $1.5 \times 10^{13}$ | $\approx 6.3 \times 10^{25}$ | above the band |

The point of the table is not the exact numbers but the scale of the jumps: each row is roughly an order of magnitude or more in compute from the last, and the "emergent band" spans about two orders of magnitude. A task that "emerges around $10^{23}$ FLOPs" under exact-match might, under a continuous metric, show a signal a full order of magnitude earlier — which is exactly the forecasting headroom you buy by switching metrics. The band is real, but it is the band where the *exact-match* number crosses its threshold, not the band where the underlying capability first becomes measurable. Those are different points, often a 10x in compute apart, and the gap between them is your forecasting opportunity.

To see why that gap is worth real money, put it in budget terms. Suppose the exact-match cliff for a capability you care about sits at $10^{23}$ FLOPs, but the continuous-metric signal becomes fittable at $10^{22}$ FLOPs — a 10x earlier. If you can only forecast from the cliff, you have to *spend up to the cliff* to learn whether the capability will appear, because the flat region tells you nothing. If you can forecast from the continuous signal, you learn the answer at one-tenth the compute, and one-tenth the compute on a frontier run is the difference between a cheap pilot and a committed nine-figure training run. The metric discipline is not academic hygiene; it is the difference between buying information for a tenth of the price and paying full freight to find out something you could have known earlier. This is the same predictability dividend that the [foundational scaling-law results](/blog/machine-learning/scaling-laws/scaling-laws-predictability-foundations) deliver for loss — the ability to know the expensive answer from cheap experiments — extended to the downstream capabilities that the emergence literature once claimed were off-limits to forecasting.

The deeper reason the cliff and the first-measurable-signal are a full order of magnitude apart is the compounding exponent again. Recall that exact-match becomes visible only when per-token error $1-p$ drops below about $1/N$. But the per-token signal — the value of $p$ itself, or equivalently the loss — is measurable as soon as it is above the noise floor of your evaluation, which happens far earlier. The compute gap between "$p$ is measurably rising" and "$p^N$ crosses its knee" is exactly the gap between the continuous metric's early signal and the exact-match cliff. So the order-of-magnitude forecasting headroom is not a lucky empirical accident; it is the direct consequence of the same exponent that manufactured the cliff in the first place. The exponent that hides the signal under exact-match is the exponent that, once you account for it, hands you the early forecast.

## What this means in practice

The emergence debate is one of those rare cases where a genuinely contested scientific question resolves into clean, actionable engineering guidance. Here is what to actually do.

**Reach for continuous, proper-scoring metrics whenever you are forecasting.** If your goal is to predict whether a future model will have a capability, never measure that capability with exact-match or multiple-choice grade during the forecasting phase. Use per-token accuracy, token edit distance, log-probability of the correct answer, or a Brier score. These give you a rising signal on small models where the discontinuous metric shows only zeros. You can still *report* the exact-match number as the headline — customers care about it — but you forecast with the continuous one and translate at the end via the $p^N$ compounding.

**Plot against pre-training loss, not against scale.** Loss is the outcome that actually drives capability; scale is just one way to lower loss. Two models with the same loss have the same capabilities even at very different sizes. Forecast loss from a scaling law (which is smooth and reliable), then map loss to downstream performance through a fitted curve. This is the single highest-leverage change, because it converts an "unforecastable" emergent ability into a forecastable function of a quantity you can already predict.

**Treat any single-run breakthrough as a random draw.** Per-seed acquisition is bimodal. A run that shows a capability does not guarantee the next run will; a run that misses it does not prove the capability is out of reach. Forecast the *fraction* of seeds that will break through, and train multiple seeds for capabilities your roadmap depends on. Report capability results as distributions, not as fixed properties of a recipe.

**Increase evaluation resolution before declaring a capability absent.** A measured 0% is frequently a sampling-resolution artifact, not a true zero. Use heavy decode-time sampling (PassUntil-style) to resolve pass rates far below what ordinary evaluation can see, then fit and extrapolate the smooth curve underneath.

**When loss and a benchmark disagree, trust the loss.** If pre-training loss is improving but a discontinuous benchmark is flat, the benchmark is sitting below its argmax flip and will catch up. Do not conclude training has stalled; switch to a continuous metric to see the real trajectory.

**Do not over-correct into pure-mirage cynicism.** It is tempting, after learning the metric story, to dismiss all emergence as fake. That is wrong. Du et al.'s loss-threshold transition is real and survives continuous metrics. Some capabilities genuinely switch on below a critical loss. The mature stance is not "emergence is fake" but "emergence is mostly a metric artifact layered on top of a real, forecastable loss-threshold transition, with per-run stochasticity on top of that." Handle each layer with the matching tool.

**When not to worry about any of this:** if you are not forecasting — if you are just reporting where a finished model lands on a benchmark — then exact-match is fine, because it is the number people care about. The whole apparatus above is for *prediction*. For a final report card, use the metric your users use. The metric-choice discipline is specifically for the forecasting problem, where the wrong metric blinds you to a signal that is already there.

The deepest lesson is the one that closes the loop with the rest of scaling-law theory. The field spent a year worried that emergence was the exception that broke predictability — the one place where you could not extrapolate from small models to large. It turned out to be almost the opposite. Once you measure capability the right way (continuous metric) on the right axis (pre-training loss) with the right uncertainty (across seeds), emergence is not a wall against forecasting. It is a reminder that the thing you forecast has to be a smooth function of something smooth, and that if your curve looks like a cliff, you should suspect your ruler before you suspect the model.

## Further reading

- Wei et al., 2022, "Emergent Abilities of Large Language Models" — the paper that named the phenomenon and made the unpredictability claim. https://arxiv.org/abs/2206.07682
- Schaeffer, Miranda, Koyejo, 2023, "Are Emergent Abilities of Large Language Models a Mirage?" — the metric-artifact argument, the 92% statistic, and the induced-emergence-in-vision experiment. https://arxiv.org/abs/2304.15004
- Du et al., 2024, "Understanding Emergent Abilities from the Loss Perspective" — the loss-threshold transition that survives continuous metrics. https://arxiv.org/abs/2403.15796
- Lu et al., 2024, "Are Emergent Abilities Just In-Context Learning?" — the reduction of apparent emergence to in-context learning, memory, and linguistic knowledge. https://arxiv.org/abs/2309.01809
- Zhao, Saphra et al., 2025, "Random Scaling of Emergent Capabilities" — per-seed bimodal breakthroughs that are smooth in aggregate. https://arxiv.org/abs/2502.17356
- Hu et al., 2023, "Predicting Emergent Abilities with Infinite Resolution Evaluation" — PassUntil and massive-sampling resolution for forecasting. https://arxiv.org/abs/2310.03262
- Sibling posts on this blog: [observational downstream scaling laws](/blog/machine-learning/scaling-laws/observational-downstream-scaling-laws), [broken neural scaling laws](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws), and [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling).

---
title: "Timer-S1: Scaling Time-Series Foundation Models with Serial Prediction"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A deep dive into Timer-S1, ByteDance's billion-scale time-series foundation model — the Serial-Token Prediction objective, the sparse TimeMoE backbone, the trillion-point TimeBench corpus, and the serial-scaling recipe that took it to the top of GIFT-Eval."
tags: ["time-series", "forecasting", "foundation-model", "mixture-of-experts", "serial-token-prediction", "timer-s1", "bytedance", "gift-eval", "transformer", "zero-shot"]
category: "machine-learning"
subcategory: "Signal Processing"
author: "Hiep Tran"
featured: true
readTime: 50
---

For three years the time-series foundation model (TSFM) story has rhymed with the language-model story: take a decoder-only Transformer, feed it a giant corpus of sequences, train it to predict the next chunk, and watch zero-shot forecasting accuracy climb as you add parameters and data. It worked — Chronos, Moirai, TimesFM, and the early Timer models all showed that a single pre-trained model could forecast electricity demand, retail sales, and ICU vitals without a single gradient step of fine-tuning. And then it stopped working. Somewhere past a few hundred million parameters, the curve flattened. Bigger TSFMs were not meaningfully better forecasters, and the field quietly started to wonder whether time series simply lacked the headroom that text had.

Timer-S1 is ByteDance's answer, and the answer is that the field was scaling the wrong axis. The diagram above is the mental model for the whole paper: instead of predicting one patch and rolling that prediction forward step by step, Timer-S1 emits the **entire forecast horizon in a single forward pass** through a stack of serial prediction blocks — and it is *that* serial computation, not raw parameter count, that reopens the accuracy curve. The model is an 8.3-billion-parameter sparse mixture-of-experts that activates only 0.75B parameters per token, trained on a 1.03-trillion-point corpus called TimeBench, and at the time of writing it holds the best pre-trained MASE (0.693) and CRPS (0.485) on the GIFT-Eval leaderboard. This post is a tour of how it is built and why the pieces fit.

> [!tldr] The five-bullet version
> - **Claim:** Prior TSFMs plateaued not because time series lack scaling headroom, but because *next-patch* pretraining caps how much useful compute a model can spend per horizon. "Serial Scaling" — scaling along architecture, data, and training pipeline *together* — reopens the curve.
> - **The mechanism:** **Serial-Token Prediction (STP)**, implemented by `H=16` stacked **TimeSTP** blocks that each predict one step further into the future, so the full horizon comes out of one forward pass with no rolling autoregression.
> - **The scale:** `8.3B` total params, `0.75B` active per token (sparse MoE, `E=32` experts, top-`K=2`), context extended to `11,520` tokens, trained on **TimeBench** — `1.032` trillion time points (~44 TB).
> - **The result:** State-of-the-art *pre-trained* MASE `0.693` and CRPS `0.485` on GIFT-Eval; ~7.6% lower MASE and ~13.2% lower CRPS than the comparable Timer-3 / Sundial baseline trained on the same data.
> - **Where it's weak:** The headline gains concentrate at medium and long horizons, the MoE makes serving heavier than the "0.75B active" number suggests, and the public release is forecasting-only — no probabilistic-anomaly or imputation heads yet.

This is one entry in a longer reading of [ByteDance Research's open model releases](/blog/machine-learning/bytedance-research-model-atlas); its closest sibling is [ChatTS](/blog/machine-learning/signal-processing/chatts-aligning-time-series-llms), which attacks the *understanding-and-reasoning* side of time series while Timer-S1 attacks raw *forecasting*. If you want the mixture-of-experts background that Timer-S1 assumes, the [DeepSeek-MoE lineage post](/blog/machine-learning/large-language-model/deepseek-moe-lineage-fine-grained-shared-experts) covers fine-grained and shared-expert routing in detail.

## Context: how time-series foundation models got stuck

Forecasting used to be a per-dataset craft. You looked at a series, decided whether it was stationary, picked an ARIMA order or fit an exponential-smoothing model or — later — trained a dataset-specific deep net like N-BEATS or PatchTST, and you did the whole dance again for the next dataset. The promise of a *foundation* model is that you amortize all of that: pretrain once on a huge, heterogeneous pile of series, and then forecast any new series zero-shot, the way GPT-style models answer questions they were never explicitly trained on.

The arc to that promise had three eras. The **statistical era** — ARIMA, exponential smoothing, state-space models — gave interpretable, per-series models that needed a human to choose orders and diagnose stationarity, and that did not benefit from data outside the single series being fit. The **deep-learning era** — DeepAR, N-BEATS, Informer, Autoformer, PatchTST — replaced hand-tuning with learned representations and cracked long-horizon forecasting on individual datasets, but each model was still trained and deployed per-dataset; a PatchTST trained on electricity knew nothing about traffic. The breakthrough that opened the **foundation era** was, ironically, a borrowing: PatchTST's demonstration that *patching* a series into Transformer tokens worked well, combined with the LLM community's evidence that next-token pretraining on enough data yields transferable zero-shot ability. Put those together — patches as tokens, next-patch pretraining, a big mixed corpus — and you get Chronos, TimesFM, Moirai, and ByteDance's earlier Timer models, the first forecasters that generalized across domains without per-dataset fitting.

What every model in that first foundation wave shared, and what Timer-S1 finally questions, is the *objective*. They all inherited next-token prediction wholesale from language. It was the obvious choice — it transferred the dense-supervision property and the proven scaling behavior — and for a while the obvious choice kept paying off. The plateau is what forced the field to ask whether the borrowed objective, not the borrowed architecture, was the limiting factor.

The recipe that delivered the first wave of TSFMs was a near-mechanical translation of the language recipe. Chop each series into fixed-length **patches** (the time-series analogue of tokens). Train a decoder-only Transformer with a causal mask to predict the next patch from the patches before it. At inference, to forecast `k` steps ahead, feed the context, predict the next patch, append it to the context, predict again, and repeat — classic autoregressive rollout. This is **Next-Token Prediction (NTP)** applied to patches, and it gave us Chronos, TimesFM, Moirai, and ByteDance's own earlier Timer models.

NTP pretraining has one enormous virtue: **dense supervision**. Every position in every series is a training example, because every patch is something to predict from its prefix. A single 2,880-token series yields thousands of next-patch losses. That density is why TSFMs learn so much from comparatively few "documents."

But NTP also has a structural ceiling that took the field a while to name, and naming it is the whole contribution of Timer-S1. When you forecast a long horizon by rolling a one-step predictor forward, two bad things happen. First, **errors compound**: step `t+2` is predicted from a context that already contains the model's own (wrong) prediction for `t+1`, so mistakes snowball geometrically down the horizon. Second, and more subtly, the model spends **the same amount of compute** producing the easy first step as it does the hard hundredth step. There is no architectural place to put *more* serial reasoning into the far horizon, because the far horizon does not exist as a distinct computation — it is just the near horizon, re-fed. Stacking more layers makes each single step a bit smarter, but it does not buy the model the *depth of sequential reasoning* that a 96-step forecast actually needs.

So when practitioners scaled NTP TSFMs from 100M to 500M to a billion parameters and saw diminishing returns, the natural — and, it turns out, wrong — conclusion was "time series don't scale." Timer-S1's reframing is that the bottleneck was never the parameter count. It was that next-patch pretraining gives you nowhere to *spend* extra serial compute on the part of the problem that needs it most: the future you cannot see.

### Patching, normalization, and what "accuracy" even means

Before going further it is worth pinning down three pieces of vocabulary that the rest of the post leans on, because time series quietly differ from text in ways that matter.

**Patching is tokenization, but the alphabet is infinite.** A language model has a finite vocabulary; a patch of 16 real numbers lives in a continuous `ℝ¹⁶`. There is no embedding lookup table — the "embedding" is a learned linear projection `ℝ¹⁶ → ℝ¹⁰²⁴`, and the output head is the inverse projection `ℝ¹⁰²⁴ → ℝ¹⁶`. This is why TSFMs use regression losses (L1/L2) rather than cross-entropy: the model predicts *values*, not a distribution over a fixed token set. It also means that two patches that are numerically close are close in input space by construction, which is both a gift (smoothness) and a curse (no discrete structure to lean on).

**Instance normalization is doing quiet heavy lifting.** A foundation model must forecast series whose absolute scales span many orders of magnitude — a stock index in the thousands, a temperature in the tens, a probability in `[0,1]`. The standard fix, used across the TSFM field and assumed by Timer-S1, is **per-instance normalization** (the RevIN trick): subtract each input window's mean and divide by its standard deviation before patching, forecast in the normalized space, then *de-normalize* the output with the same statistics. Without it, the model would waste capacity learning scale rather than shape. With it, "forecast electricity load" and "forecast a heartbeat" become the same shape-prediction problem at different scales, which is exactly the generalization a foundation model needs.

**Point accuracy and distributional accuracy are different goals.** Timer-S1 is scored on two metrics that pull in different directions. **MASE** (mean absolute scaled error) measures *point* accuracy: how close is the single predicted value to the truth, scaled by the error a seasonal-naive forecaster would make, so a MASE below 1.0 means "better than just repeating last season." **CRPS** (continuous ranked probability score) measures *distributional* accuracy: how well-calibrated is the model's predicted *distribution* over the future, rewarding sharp predictions only when they are also correct. A model can win on MASE by being a good point estimator and still lose on CRPS by being overconfident. Watching both move together — as they do for Timer-S1 — is the sign of a forecaster that is both accurate and honest about its uncertainty.

With those three in hand, the rest of the architecture reads more cleanly.

## The core idea: Serial-Token Prediction

![Pipeline diagram of Serial-Token Prediction: an input series is patched into tokens, fed through a shared MoE trunk, then through sixteen serial TimeSTP heads where head j emits the forecast for step j, so the entire horizon is produced in one forward pass rather than by rolling a one-step predictor forward.](/imgs/blogs/timer-s1-time-series-foundation-model-1.webp)

Serial-Token Prediction is the design that turns "more horizon needs more serial compute" from a complaint into an architecture. The idea is to keep `H` prediction blocks *stacked in series after the backbone*, where block `j` is responsible for predicting the patch `j` steps into the future. The backbone reads the observed context once. Then the serial blocks run one after another, each one deepening the model's internal forecast and emitting the next horizon step, until all `H+1` future patches have been produced — in a single forward pass, with no value ever fed back into the input.

Formally, let the input be a series patched into tokens `x₁ … x_N`, and let `h_i^L` be the embedding of patch `i` after the `L`-layer backbone. STP trains `H` serial blocks so that block `j` predicts patch `i+j+1`:

```
ℒ_STP = (1/H) · Σ_{j=1}^{H} Σ_{i=1}^{N} ℒ_pred( x_{i+j+1},  x̂_{i+j+1} )

where   x̂_{i+j+1} = PatchProject( h_i^{L+j} )
```

Read that carefully, because the indices carry the whole idea. For a *single* anchor position `i`, the backbone produces `h_i^L`; serial block 1 turns it into `h_i^{L+1}` and predicts `x_{i+2}`; serial block 2 turns that into `h_i^{L+2}` and predicts `x_{i+3}`; and so on through block `H`, predicting `x_{i+H+1}`. Each block is a *real* additional computation applied to the *same* anchor's hidden state — so the model gets `H` layers of serial reasoning deeper into the future, exactly where rolling NTP gave it zero.

It is worth separating STP from two things it superficially resembles.

**STP is not next-token prediction.** NTP makes only the last embedding `h_N^L` predict a single next patch, and then you must roll. STP makes *every* anchor `i` predict `H` patches into the future through `H` distinct serial blocks. The whole horizon is materialized at once.

**STP is not the multi-token prediction (MTP) used in LLMs either.** MTP, as in DeepSeek-V3 and others, also predicts several future tokens from one pass — but it does so largely in *parallel* heads, and crucially, LLM-MTP often references future ground-truth tokens during training and then *discards* the extra heads at inference, keeping them only as a training-time auxiliary. TimeSTP differs on both counts. It never references future values during training (the future is genuinely unavailable, in training and inference alike), and it *keeps* the serial blocks at inference because they *are* the forecaster. The paper is explicit that this serial structure — block `j` consuming the output of block `j-1` — is what supplies "the serial computations necessary for long-term forecasting" that flat MTP lacks.

The payoff visible in the first figure is that a 272-token horizon (`(H+1)×P = 17×16`) comes out of one pass. No rollout loop, no compounding feedback, and a clean place — block depth — to spend more compute on the far future.

### A worked example, with indices

Abstract index soup hides how concrete this is, so walk one anchor through the machine. Suppose the context is 180 patches `x₁ … x₁₈₀`, each covering 16 time points, and we want to forecast the next horizon. Take the last anchor `i = 180`.

1. The 24-block backbone reads the full context and produces `h₁₈₀^{L}`, the anchor's hidden state after the trunk. This state encodes "everything the model knows about the series, summarized at position 180."
2. TimeSTP block 1 receives `h₁₈₀^{L}` *and* the original patch embeddings, fuses them, runs a TimeMoE block, and produces `h₁₈₀^{L+1}`. From it, `PatchProject` reads off `x̂₁₈₂` — the patch at `i+2`. (Block `j` predicts `i+j+1`, so block 1 predicts `i+2`.)
3. TimeSTP block 2 receives `h₁₈₀^{L+1}`, re-injects the original patches again, and produces `h₁₈₀^{L+2}`, from which it reads `x̂₁₈₃`.
4. … and so on. Block 16 produces `x̂₁₉₇`.

So from the single anchor at 180, one forward pass yields patches 182 through 197 — and because the model also keeps the ordinary next-patch prediction `x̂₁₈₁` from the trunk, the full emitted block is patches 181–197: `(H+1) = 17` patches, `272` time points.

Now compare against rolling NTP. To produce 272 points NTP-style, you would run the model, get `x̂₁₈₁`, *append it to the input*, run again to get `x̂₁₈₂` (now conditioned on a predicted, possibly-wrong `x̂₁₈₁`), append, run, … 17 times. Seventeen forward passes instead of one, and every pass after the first is conditioned on the model's own errors. STP collapses that to one pass where each future step is produced by a *dedicated* serial block that consumes a clean internal state rather than a contaminated input. The compute is spent on depth-of-reasoning, not on re-encoding a growing pile of guesses.

This worked example also makes the training signal density obvious: the loss sums over *every* anchor `i`, not just the last one. With 180 anchors each contributing 16 horizon predictions, a single series produces on the order of 2,880 STP losses per pass — the dense supervision that NTP was prized for, now extended across the whole horizon.

## Why prior TSFMs plateaued, and what serial scaling changes

![Before-and-after comparison: on the left, naive next-patch scaling where accuracy flattens as parameters grow because rolling autoregression caps usable serial compute; on the right, serial scaling that reopens the accuracy curve by adding TimeSTP depth, more data, and a staged training pipeline together.](/imgs/blogs/timer-s1-time-series-foundation-model-2.webp)

The second figure states the thesis as a contrast. On the left is the world everyone was living in: hold the NTP recipe fixed, grow the parameter count, and watch the accuracy curve bend over and flatten. On the right is Timer-S1's claim: change the *objective* so the architecture has a serial axis to grow along, scale that axis alongside data and the training pipeline, and the curve straightens back out.

The reason this is more than a slogan is that "serial scaling" is a genuinely different scaling dimension from the one the field had been pushing. Parameter scaling makes each layer wider or the stack taller; it improves the *representation* at each step. Serial scaling makes the *prediction process itself* deeper: `H=16` serial blocks mean sixteen rounds of "given everything I now believe about the future up to step `j`, what is step `j+1`?" That is a qualitatively different resource. You can have a model with modest per-step representation but deep serial reasoning, and for long-horizon forecasting that trade can be the better one.

This reframing also explains why the gains are not uniform. If serial depth is what helps, then the benefit should concentrate where serial reasoning matters — the medium and long horizons — and be near-neutral on one-step-ahead forecasts, where there is nothing to roll. That is precisely the shape of the Timer-S1 results: it is "substantially better" at medium and long horizons and roughly competitive at the very short end. A clean, falsifiable prediction that the experiments bear out is one of the more convincing things about the paper.

The cost, of course, is that serial scaling is not free architecture-wise. You have to actually build and train those `H` serial blocks, keep them at inference, and pay for the data and pipeline that make them pull their weight. The rest of the model is the machinery that makes serial scaling affordable.

## Anatomy: the Timer-S1 stack

![Layered stack diagram of the Timer-S1 architecture from bottom to top: raw series, patch embedding at patch size sixteen, a twenty-four-layer sparse TimeMoE backbone trunk, and sixteen TimeSTP serial heads that share the MoE block design and emit the full forecast horizon, with the projection head on top.](/imgs/blogs/timer-s1-time-series-foundation-model-3.webp)

From the bottom up, the model has four bands, and the stack figure walks through them.

**Patching (`P=16`, `D=1024`).** A univariate series is sliced into non-overlapping windows of 16 time points; each window is linearly projected to a 1,024-dimensional patch embedding. This is the tokenization step, and the patch size is the granularity at which the model "thinks." The full context is `T=2,880` time points at pretraining (180 patches), extended to `11,520` (720 patches) in post-training.

**The TimeMoE backbone (`L=24` blocks).** The trunk is 24 Transformer blocks, but each block's feed-forward network is replaced by a **sparse mixture of experts**. A `TimeMoE` block is multi-head attention followed by an MoE module, wrapped with **Pre-RMSNorm** and **QK-Norm** for training stability at scale. This is the part that holds most of the 8.3B parameters and reads the observed context.

**The TimeSTP serial heads (`H=16` blocks).** This is the serial-scaling axis made concrete. Each TimeSTP block contains a projection layer plus a TimeMoE block, and it operates by **concatenating two inputs**: the token embeddings coming from the preceding block, and the *original input patch embeddings*. That second input is the clever bit — re-injecting the raw patch embeddings at every serial step keeps each block grounded in the actual observed series rather than drifting on a tower of its own intermediate states. Block `j` produces the hidden state `h_i^{L+j}` from which patch `i+j+1` is read off.

**The projection head.** A shared `PatchProject` linear map turns each serial block's output hidden state back into a patch of 16 real values. Because the blocks are stacked, a single forward pass yields the whole `(H+1)×P = 272`-point forecast.

A subtle and important architectural decision is that the serial heads **share the TimeMoE block design with the trunk**. They are not a bolt-on decoder; they are more trunk, organized serially. That keeps the implementation uniform (the same stability tricks, the same routing) and means the serial axis inherits the MoE capacity rather than being a thin dense afterthought.

The grounding trick — re-injecting the original patch embeddings `x0` at every serial block — deserves a second look because it is what keeps the serial tower from going off the rails. Without it, block `j` would be working purely from block `j-1`'s output, which is itself a prediction-of-a-prediction `j` steps removed from any real data. That is the same drift that plagues rolling autoregression, just relocated from the input loop into the block stack. By concatenating `x0` back in at each step, every serial block is forced to reconcile its evolving forecast against the actual observed series, so errors do not accumulate down the serial axis the way they accumulate down a rollout. It is a cheap architectural anchor that buys a lot of stability, and it is the detail that distinguishes TimeSTP from a naive "just stack `H` decoder layers and read off each one" design — which would predict the horizon but drift badly at the far end. The serial blocks deepen the forecast; the re-injected `x0` keeps that deepening tethered to reality.

## The sparse engine: TimeMoE routing

![Routing graph for the sparse mixture of experts: a patch token enters a router that scores all thirty-two experts and dispatches the token to only its top two, so roughly 0.75 billion of the 8.3 billion total parameters are active per token while the rest stay idle, with outputs combined by gating weights.](/imgs/blogs/timer-s1-time-series-foundation-model-4.webp)

The whole reason an 8.3B model can be deployed at all is that it almost never runs as an 8.3B model. Every patch token is routed: a small **router** network scores all `E=32` experts, the top `K=2` are selected, the token is processed only by those two expert FFNs, and their outputs are combined by the router's gating weights. The other 30 experts sit idle for that token. The arithmetic works out to roughly `0.75B` parameters active per token — about one-eleventh of the total.

Why does sparsity matter so much for a TSFM specifically? Time series are wildly heterogeneous: financial ticks, hourly electricity load, ECG waveforms, and IoT sensor streams have almost nothing in common in their statistics. A dense model has to cram all of those regimes into one shared FFN and hope the superposition does not interfere. An MoE can let routing carve the heterogeneity into specialists — one cluster of experts that handles smooth seasonal load, another that handles spiky impulse-like sensor data — and only pay for the few that a given token needs. The capacity-to-compute ratio that MoE buys is exactly what you want when the data distribution is a union of many sub-distributions.

That said, the "0.75B active" figure is a *compute* statement, not a *memory* statement, and it is the number most likely to mislead a practitioner sizing hardware. All 8.3B parameters must be resident in memory to be routable, so the model's memory footprint is that of an 8.3B dense model even though each token's FLOPs are those of a 0.75B one. If you are used to the [MoE trade-offs from the LLM world](/blog/machine-learning/large-language-model/deepseek-moe-lineage-fine-grained-shared-experts), none of this is new — but it is worth saying plainly, because the headline number invites the wrong mental model for serving cost.

### Load balancing and the specialization you hope for

Sparse routing has a famous failure mode: **expert collapse**. If the router is left to its own devices, it discovers early in training that a few experts are slightly better and starts sending everything to them. Those experts get more gradient, get better still, and a winner-take-all dynamic leaves most of the 32 experts dead weight — you paid for 8.3B parameters and trained a 1B model. The standard cure, which Timer-S1 uses via its auxiliary loss term `αℒ_aux`, is a **load-balancing penalty** that pushes the router toward sending roughly equal token mass to every expert. It is a soft constraint — you want balance *and* specialization, which are in tension — so `α` is tuned to keep experts busy without forbidding them from specializing.

The specialization you *hope* the router learns, for a time-series model specifically, is regime-based. Time series are a union of statistical regimes: smooth seasonal signals, bursty impulse-like sensor data, near-random-walk financial series, slowly-drifting meteorological data. A dense FFN has to superpose all of those into one set of weights and tolerate the interference. An MoE can, in principle, dedicate expert clusters to regimes — and because routing is per-token (really per-patch), a single heterogeneous series can have its calm stretches routed to "smooth" experts and its spikes routed to "impulse" experts, *within the same forecast*. Whether the learned routing actually carves the data this cleanly is hard to verify from the outside, but it is the mechanism that makes a 32-expert MoE the right inductive bias for a domain that is fundamentally a mixture of sources.

There is a second, quieter benefit. Because the **serial TimeSTP heads reuse the TimeMoE block design**, the serial axis also gets routing. That means a near-horizon serial block and a far-horizon serial block can route to *different* experts — the model can learn that long-range extrapolation calls on different machinery than one-step refinement. Serial depth and expert specialization compound: depth gives the model more serial reasoning steps, and routing lets each step recruit the right specialists.

## Serial scaling across three dimensions

![Tree diagram of serial scaling across three branches: an architecture axis carrying the TimeSTP serial blocks, a dataset axis carrying the trillion-point TimeBench corpus with augmentation, and a training-pipeline axis carrying the pretrain, continued-pretrain, and context-extension stages, all converging on a model that keeps improving past the old plateau.](/imgs/blogs/timer-s1-time-series-foundation-model-5.webp)

"Serial scaling" is the paper's umbrella term, and the tree figure shows that it is not one trick but three coordinated ones. The point the authors press is that any single axis alone underdelivers; the curve only straightens when all three move together.

**Architecture axis — TimeSTP.** Covered above: the serial prediction blocks that give long-horizon forecasting a place to spend depth. Without this axis, more data and a better pipeline just feed the same plateaued NTP model.

**Dataset axis — TimeBench.** A trillion-point corpus, deliberately built to be both large and *unbiased* (more on the bias problem in the next section). Without enough diverse data, the serial blocks overfit, because deeper prediction machinery has more capacity to memorize.

**Training-pipeline axis — the three stages.** Pretraining on the full mixture with both NTP and STP objectives, then continued pretraining (CPT) with a horizon-weighted STP loss, then RoPE-based context extension. Without staging, you cannot both teach the model broad coverage *and* sharpen its short-horizon precision *and* stretch its context — those goals pull against each other if you try to hit them in one flat run.

The reason to insist on all three is a kind of Liebig's-law argument: forecasting quality is gated by whichever axis you under-scaled. A team that copies only the TimeSTP blocks into a small model on a small corpus will conclude "serial prediction doesn't help much," and they will be right — *for them*. The contribution is the claim that these axes are complements, and the experiments are designed to show that removing any one of them collapses the gains.

Make that concrete. Take TimeSTP blocks but train on a small, biased corpus: the extra serial capacity has more parameters to overfit and nothing diverse to generalize from, so it memorizes the up-trend prior and falls apart on declining series — the architecture axis without the data axis. Take the full TimeBench corpus but keep flat next-patch prediction: you have the data but no place to spend serial compute on the far horizon, so you reproduce the plateau on a bigger pile of data — the data axis without the architecture axis. Take both but skip the staged pipeline: you cannot simultaneously learn all horizons, sharpen the near term, and stretch context in one flat run, so the model is mediocre at each — the architecture and data axes without the pipeline axis. Only the conjunction clears the plateau, which is why the paper frames serial scaling as one idea with three faces rather than three independent tricks.

## TimeBench: a trillion points, built to be unbiased

![Grid diagram of the TimeBench corpus composition: real-world domains such as finance, IoT, meteorology, and healthcare on one side, synthetic signal families such as sinusoidal, exponential, impulse, and KernelSynth causal models on the other, plus resampling and value-flipping augmentation, totaling 1.03 trillion time points.](/imgs/blogs/timer-s1-time-series-foundation-model-7.webp)

Data is where a lot of the quiet engineering lives, and TimeBench is genuinely large: **1.032 trillion** regularly sampled time points, stored as roughly **44 TB** of compressed Parquet. But raw size is not the interesting part — *bias control* is.

The corpus has two halves, shown in the grid figure. The **real-world** half pulls from finance, IoT, meteorology, healthcare, and previously released public corpora (the same source pools behind Chronos and LOTSA). The **synthetic** half generates canonical signal families — linear, sinusoidal, exponential, power, impulse, and step functions, plus their combinations — and adds **KernelSynth**, which composes Gaussian-process kernels into temporal causal models. Synthetic data is not filler here; it gives the model clean, label-perfect exposure to primitive patterns (a pure trend, a pure period, a pure level-shift) that real data tangles together.

Two augmentation techniques specifically target **predictive bias** — the tendency of a forecaster to lean on spurious regularities like "series usually go up" or "the future looks like a shifted copy of the recent past":

- **Resampling.** Vary the sampling rate by down-sampling and re-interpolating on Fourier bases. This stops the model from memorizing absolute frequencies and forces it to recognize a daily cycle whether it is sampled hourly or every fifteen minutes.
- **Value-Flipping.** Multiply both input and output series by `−1`, inverting the trend while *preserving* the temporal dependency structure. This is a beautifully cheap way to kill the "trends go up" prior: for every rising series the model sees, it sees the mirror-image falling one with identical dynamics, so direction-of-trend becomes uninformative and the model has to learn the actual shape.

On top of augmentation there is the unglamorous-but-essential quality pipeline: causal mean imputation for gaps, outlier removal with both `k`-σ and IQR thresholds, predictability screening via the **ADF stationarity test** and **spectral entropy** (so the corpus is not polluted with un-forecastable noise), and careful removal of anything that would leak GIFT-Eval test data into training. That last item is what lets the leaderboard number mean something.

## The training pipeline: three stages, three capabilities

![Timeline diagram of the training pipeline reading left to right: pretraining with combined next-token and serial-token objectives on the full TimeBench corpus, continued pretraining with a horizon-decayed weighted serial loss on a mixed GIFT-Eval-pretrain and TimeBench sample, and RoPE-based context extension from 2,880 to 11,520 tokens, ending at the GIFT-Eval leaderboard.](/imgs/blogs/timer-s1-time-series-foundation-model-8.webp)

The timeline figure makes the point that for Timer-S1, post-training is not a cosmetic finishing pass — it is where specific, nameable capabilities are bought. Each stage exists to install one thing the previous stage could not.

**Stage 1 — Pretraining.** Train on the full TimeBench corpus with a combined objective: next-token prediction *plus* serial-token prediction, weighted uniformly across horizons, with a small auxiliary MoE load-balancing term `αℒ_aux` so experts do not collapse onto a few favorites. Uniform horizon weighting here is deliberate — early in training you want the model to learn *all* horizons, not prematurely specialize. This stage installs broad coverage.

**Stage 2 — Continued Pretraining (CPT).** Now switch the STP loss to a **horizon-weighted** version and revisit a sharper data mixture: 50% GIFT-Eval Pretrain split, 50% TimeBench. The weighting is the key detail:

```
ℒ_wSTP = (1/H) · Σ_{j=1}^{H} (1/√j) · Σ_{i=1}^{N} ℒ_pred( x_{i+j+1},  x̂_{i+j+1} )
```

The `1/√j` factor down-weights deeper serial blocks, which means CPT *re-prioritizes the near horizon*. The intuition is that pretraining already taught the model to reach far out; CPT now sharpens the short-term precision that dominates most benchmark metrics, without un-learning the long-horizon skill. Data revisiting on the GIFT-Eval pretrain distribution improves generalization to the leaderboard's domains. This stage installs short-horizon sharpness.

**Stage 3 — Context Extension.** Use RoPE-based scaling to stretch the usable context from `2,880` to `11,520` tokens, keeping the same CPT objective and mixture. This is the standard long-context recipe borrowed wholesale from the LLM world, and it installs the ability to condition on long histories — a year of daily data, say — without retraining from scratch. This stage installs long context.

The staging order matters: you cannot extend context usefully before the model can forecast, and you cannot sharpen the near horizon before it has learned all horizons. Each stage stands on the last.

## A minimal implementation sketch

Reading the loss and the block design is one thing; seeing them as code makes the serial structure concrete. Here is a PyTorch-shaped sketch of the STP objective and a single TimeSTP block. It is illustrative, not the official implementation, but the shapes and the data flow are faithful.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeMoE(nn.Module):
    """Attention + sparse MoE FFN with top-k routing (E=32, K=2)."""
    def __init__(self, d=1024, n_experts=32, k=2, n_heads=16):
        super().__init__()
        self.norm1 = nn.RMSNorm(d)
        self.attn  = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(d)
        self.router  = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            for _ in range(n_experts)
        )
        self.k = k

    def forward(self, x, attn_mask=None):
        # causal self-attention
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)[0]
        # sparse MoE: route each token to its top-k experts
        h = self.norm2(x)
        scores = self.router(h)                      # (B, T, E)
        w, idx = scores.topk(self.k, dim=-1)         # (B, T, K)
        w = F.softmax(w, dim=-1)
        out = torch.zeros_like(h)
        for slot in range(self.k):
            for e in range(len(self.experts)):
                m = idx[..., slot] == e              # tokens routed to expert e
                if m.any():
                    out[m] += w[..., slot][m, None] * self.experts[e](h[m])
        return x + out

class TimeSTPBlock(nn.Module):
    """One serial head: re-inject the raw patch embeddings, then a TimeMoE block."""
    def __init__(self, d=1024):
        super().__init__()
        self.proj  = nn.Linear(2 * d, d)   # fuse prev-block state + original patch emb
        self.block = TimeMoE(d)

    def forward(self, h_prev, x0):
        # h_prev: output of the preceding serial block; x0: ORIGINAL patch embeddings
        fused = self.proj(torch.cat([h_prev, x0], dim=-1))
        return self.block(fused)

class TimerS1Head(nn.Module):
    """L backbone blocks + H serial TimeSTP blocks producing the whole horizon."""
    def __init__(self, d=1024, L=24, H=16, patch=16):
        super().__init__()
        self.backbone = nn.ModuleList(TimeMoE(d) for _ in range(L))
        self.serial   = nn.ModuleList(TimeSTPBlock(d) for _ in range(H))
        self.to_patch = nn.Linear(d, patch)   # shared PatchProject
        self.H = H

    def forward(self, x0, causal_mask):
        h = x0
        for blk in self.backbone:
            h = blk(h, attn_mask=causal_mask)      # h = h^L
        preds, cur = [], h
        for blk in self.serial:                    # serial axis: H deep
            cur = blk(cur, x0)                      # block j -> h^{L+j}
            preds.append(self.to_patch(cur))       # predict patch i+j+1
        return torch.stack(preds, dim=2)           # (B, N, H, patch)

def stp_loss(preds, future, weighted=False):
    """preds: (B, N, H, patch); future[...,j,:] is the patch j+1 steps ahead."""
    H = preds.size(2)
    total = 0.0
    for j in range(H):
        w = 1.0 / ((j + 1) ** 0.5) if weighted else 1.0   # CPT uses 1/sqrt(j)
        total += w * F.l1_loss(preds[:, :, j], future[:, :, j])
    return total / H
```

Three things to notice. First, `TimeSTPBlock.forward` takes `x0` — the *original* patch embeddings — and fuses them in at every serial step; that is the grounding trick from the architecture section. Second, the serial loop in `TimerS1Head.forward` is where serial depth lives: `H` iterations, each consuming the previous block's state, each emitting one further horizon step. Third, `stp_loss` toggles between the uniform pretraining loss and the `1/√j` weighted CPT loss with a single flag — the two-line difference that re-prioritizes the near horizon in stage 2.

## Experiments: the GIFT-Eval standing

![Matrix figure comparing Timer-S1 against strong time-series baselines on the GIFT-Eval leaderboard, with rows for models such as Timer-S1, Timer-3 / Sundial, Chronos-2, Moirai, and TimesFM and columns for MASE, CRPS, and rank; Timer-S1's best pre-trained MASE of 0.693 and CRPS of 0.485 cells are highlighted as the wins.](/imgs/blogs/timer-s1-time-series-foundation-model-6.webp)

GIFT-Eval is the relevant proving ground: a broad zero-shot forecasting leaderboard spanning many domains, frequencies, and horizons, scored primarily with **MASE** (mean absolute scaled error — point accuracy, normalized against a seasonal-naive baseline so a score below 1.0 beats naive) and **CRPS** (continuous ranked probability score — distributional accuracy for the probabilistic forecast). Lower is better on both.

Why this benchmark and not the older standbys? The pre-foundation field evaluated on a handful of datasets — ETT, Electricity, Traffic, Weather — and a model could quietly overfit the quirks of those few. GIFT-Eval was built specifically to test *generalization*: it aggregates many datasets across domains (energy, transport, web, nature, sales, climate), frequencies (from sub-hourly to yearly), and forecast horizons (short, medium, long), and it normalizes scores against a seasonal-naive baseline so that wins on easy series do not drown out losses on hard ones. The normalization is what makes a single aggregate number meaningful: a MASE of 0.693 says "averaged across this whole zoo of series, Timer-S1's point error is about 31% below what naively repeating the last season would give." For a *zero-shot* model facing series it never saw in training, that is a strong statement. The flip side, raised in the failure-modes section, is that any model whose training mixture is aligned to GIFT-Eval's domains has a quiet home-field edge, which is why same-data ablations and out-of-benchmark reproductions matter so much for interpreting the headline.

The headline: Timer-S1 records the **best pre-trained MASE (0.693)** and **CRPS (0.485)** among the models the paper compares. Against the directly comparable Timer-3 / Sundial baseline trained on the *same* data, it is about **7.6% lower MASE** and **13.2% lower CRPS** — and that same-data comparison is the one that isolates the contribution of the method from the contribution of the corpus. The larger CRPS improvement than MASE improvement is itself informative: the serial structure helps the *distributional* forecast even more than the point forecast, which fits the story that deeper serial reasoning produces better-calibrated long-horizon uncertainty.

The per-horizon breakdown is where the method's signature shows up most clearly, and it is the result to demand of any serial-prediction claim. The paper reports that the gains are "substantially better" at medium and long horizons while being roughly competitive at the short end — exactly the shape the theory predicts. A one-step-ahead forecast has no serial reasoning to do, so STP and NTP should tie there; a hundred-step forecast is where sixteen serial blocks of "given my evolving forecast, what comes next" earn their keep, and that is where the gap opens. This per-horizon signature is more convincing than the aggregate number, because the aggregate could in principle be moved by many things (more data, better tuning, the corpus), whereas the *specific* pattern of "neutral short, strong long" is a fingerprint of serial depth in particular. When you evaluate your own serial-prediction variant, slice by horizon first; an aggregate win that is flat across horizons would mean the gain came from somewhere other than seriality, and the central claim would be in doubt.

| Model | MASE ↓ | CRPS ↓ | Notes |
|---|---|---|---|
| **Timer-S1** | **0.693** | **0.485** | Best *pre-trained* model; 8.3B total / 0.75B active |
| Timer-3 / Sundial (same data) | ~0.75 | ~0.559 | The apples-to-apples baseline; STP gives −7.6% MASE, −13.2% CRPS |
| Chronos-2 | competitive | competitive | Strong even without explicit multivariate interaction modeling |
| Chronos / Moirai / TimesFM | — | — | Prior-generation baselines; shown graphically in the paper |

(The non-Timer numbers are reported by the paper mostly *visually* in its figures rather than as a single text table, so treat the bottom rows as directional. The Timer-S1 and same-data-baseline numbers are the load-bearing ones.)

What is load-bearing in this setup, and what might not transfer? Three things to keep honest about:

1. **"Pre-trained model" is a meaningful qualifier.** The claim is best among *pre-trained, zero-shot* forecasters, not best-on-earth. A well-tuned per-dataset model can still beat a foundation model on its home turf; the value proposition is "one model, no fitting," not "unbeatable everywhere."
2. **The gains concentrate at medium and long horizons.** If your application is one-step-ahead nowcasting, the serial machinery buys you less, and a smaller NTP model may be the better cost trade.
3. **Same-data comparison is the honest one.** The 7.6%/13.2% deltas against Timer-3 on identical data are far more convincing than cross-paper comparisons that confound method, corpus, and tuning. The paper deserves credit for running that controlled comparison and reporting it prominently.

## Using Timer-S1 in practice

A foundation model is only useful if the path from "downloaded weights" to "forecast on my data" is short. For Timer-S1 the path is genuinely zero-shot — no fitting — but three operational details decide whether you get the leaderboard-grade numbers or garbage.

**Normalize per window, de-normalize the output.** This is the single most common way people get bad forecasts out of a TSFM. The model was trained on per-instance-normalized inputs, so you must normalize your context window the same way and invert the transform on the prediction. The pattern:

```python
import torch

@torch.no_grad()
def forecast(model, series, horizon, patch=16, ctx_patches=180):
    """series: 1-D tensor of observed values. Returns `horizon` future values."""
    # 1. take the most recent context and per-instance normalize (RevIN-style)
    ctx = series[-ctx_patches * patch:]
    mu, sigma = ctx.mean(), ctx.std().clamp_min(1e-6)
    ctx_n = (ctx - mu) / sigma

    # 2. patch -> (1, N, patch), build the causal mask
    x = ctx_n.reshape(1, -1, patch)
    N = x.size(1)
    mask = torch.triu(torch.full((N, N), float("-inf")), diagonal=1)

    # 3. one forward pass yields up to (H+1)*patch = 272 future points from the
    #    last anchor; slice the horizon you need
    preds = model(embed(x), mask)               # (1, N, H, patch)
    horizon_n = preds[0, -1].reshape(-1)[:horizon]

    # 4. de-normalize back to the original scale
    return horizon_n * sigma + mu
```

**For horizons beyond 272 points, roll the serial block — not the input.** A single pass covers `(H+1)×P = 272` time points. If you need a 1,000-step forecast, you do still iterate, but you iterate at the granularity of the *whole 272-point block*, not one patch at a time: predict 272, append them, predict the next 272. That is roughly `⌈1000/272⌉ = 4` passes instead of `⌈1000/16⌉ = 63` patch-level rollouts, so even "long" forecasts are cheap, and the compounding-error problem is amortized across 17-patch jumps rather than single patches. The longer context window (11,520 after extension) is what makes appending blocks viable without falling off the trained context length.

**Decide point vs probabilistic up front.** The public release is a point forecaster by default. If your application is inventory planning or risk, you care about CRPS-style distributional output, and you will want to either run the model with sampling/quantile post-processing or wait for probabilistic heads. Do not report a single trajectory as if it were a prediction interval.

When *should* you reach for Timer-S1 over a small bespoke model? The honest decision rule: use it when you have **many heterogeneous series and little per-series history** (cold-start retail SKUs, new sensors, a fleet of machines), or when you need **medium-to-long horizons** where its serial advantage is largest. Stick with a small fitted model when you have **one well-understood series with abundant history** and a **one-step-ahead** need, where a tuned PatchTST or even a seasonal ETS can match it at a fraction of the serving cost.

## Where Timer-S1 sits in the TSFM landscape

Timer-S1 did not arrive in a vacuum; it is the latest move in a three-year run of forecasting foundation models, and its design choices read as deliberate responses to its predecessors' limits.

| Model | Core objective | Scale | Probabilistic? | Distinguishing idea |
|---|---|---|---|---|
| **Chronos / Chronos-2** | Tokenized values + NTP | ~quantized vocab | Yes (sampling) | Treats forecasting as language modeling over *quantized* value tokens |
| **TimesFM** | Patch NTP, decoder-only | ~200M–500M | Point + quantile | Clean patch-decoder baseline that proved zero-shot works |
| **Moirai** | Masked patch, any-variate | up to ~1B+ | Yes | Unified handling of arbitrary numbers of variates and frequencies |
| **Timer-3 / Sundial** | Patch NTP (ByteDance prior) | ~1B | Yes | The direct same-data baseline Timer-S1 improves on |
| **Timer-S1** | **Serial-Token Prediction + MoE** | 8.3B / 0.75B active | Point (public) | **Serial scaling**: depth-of-prediction as a new scaling axis |

A few comparisons are worth drawing out.

**Against Chronos's quantization.** Chronos turns forecasting into a literal language-modeling problem by quantizing values into a discrete vocabulary and predicting tokens with cross-entropy. That gives clean probabilistic outputs but bakes in a quantization error floor and a fixed dynamic range. Timer-S1 keeps values continuous and regresses them, trading Chronos's discrete-token convenience for exact value preservation — the same value-preserving philosophy that its sibling [ChatTS](/blog/machine-learning/signal-processing/chatts-aligning-time-series-llms) argues for on the *understanding* side.

**Against Moirai's any-variate attention.** Moirai's headline feature is modeling cross-variate interactions directly. Timer-S1 is, by contrast, comparatively univariate in its core formulation — and the paper notes it stays competitive with Chronos-2 "despite not using explicit multivariate interactions." That is a tell: the serial-prediction win is large enough to offset not modeling cross-series structure, which both validates the STP idea and flags the obvious next frontier (add multivariate routing and the two advantages might stack).

**Against its own predecessor.** The cleanest read of Timer-S1 is as a controlled upgrade of Timer-3/Sundial: same lineage, same data available, swap NTP for STP and scale the three axes, get 7.6%/13.2% better. That is the comparison that isolates the method, and it is why the paper leans on it.

And then there is the orthogonal axis: **forecasting vs reasoning**. Timer-S1 produces numbers; it does not explain them, answer questions about a series, or detect that "the anomaly at hour 14 looks like the sensor failure pattern from last month." That is ChatTS's job. The two are complements, not competitors — a mature time-series stack would forecast with Timer-S1 and reason with ChatTS, exactly the kind of division of labor the [ByteDance model atlas](/blog/machine-learning/bytedance-research-model-atlas) maps across the org's releases.

## Serving Timer-S1: compute vs memory

The deployment story is where the MoE design forces an honest accounting, and it is the part most likely to surprise a team that read only the "0.75B active" headline.

**Memory is sized by the total, not the active, parameters.** All 32 experts must be resident to be routable, so plan for an 8.3B-parameter footprint: roughly 16.6 GB in fp16, ~8.3 GB if you quantize weights to int8, less with int4. This fits comfortably on a single modern accelerator, which is good news — but it is not the "0.75B model" footprint the active-parameter number might suggest. Budget memory for the whole model; budget *compute* for the active slice.

**Compute per token is genuinely the active-parameter cost** — about one-eleventh of dense-8.3B FLOPs per patch — which is what makes the model fast despite its size. The serial heads add `H=16` block evaluations on top of the `L=24` trunk, but those run in one pass and are themselves sparse, so the latency to produce a full 272-point horizon is far below what 17 autoregressive rollouts of a comparable dense model would cost. The single-pass property is a latency feature, not just an accuracy one.

**Batching and routing efficiency.** MoE throughput lives and dies on how well you can batch tokens to the same expert. With per-patch routing across 32 experts, small batches scatter tokens thinly and underutilize each expert's matmul; large batches amortize the routing overhead. If you serve many series at once (a fleet, a marketplace), MoE batching works in your favor; if you serve one short series at a time, you pay routing overhead with little to amortize it against — a regime where a small dense model can actually be faster wall-clock despite worse accuracy.

**Quantization is the obvious lever.** Because the memory cost is the binding constraint and the experts are the bulk of the parameters, weight-only quantization of the expert FFNs is the highest-leverage optimization, shrinking the resident footprint with modest accuracy loss. The attention and routing layers are small by comparison and can stay higher-precision. None of this is Timer-S1-specific — it is the standard MoE serving playbook — but it is worth stating because the temptation is to size hardware off the wrong number and then be surprised.

## What transfers from the LLM playbook (and what doesn't)

Timer-S1 is, structurally, a decoder-only Transformer with sparse MoE FFNs, RoPE positions, RMSNorm, QK-Norm, staged pretraining, and context extension. Almost every one of those is a borrowing from large language models, and the model is a useful case study in which parts of the LLM recipe are domain-general and which are not.

**What transfers cleanly:**

- **Sparse mixture-of-experts.** Routing tokens to a few of many experts works in time series for the same reason it works in language: the data is a mixture of sub-distributions, and conditional computation lets capacity grow faster than cost. The load-balancing auxiliary loss transfers verbatim.
- **RoPE and context extension.** Rotary position embeddings and the trick of scaling them to stretch context (here 2,880 → 11,520) are lifted directly from the long-context LLM literature, and they work because position-relative attention is modality-agnostic.
- **Staged training.** Pretrain broad, continue-pretrain on a sharper mixture, then extend context — the same staging that LLMs use to separate "learn everything" from "specialize" from "lengthen" maps onto forecasting with the stages buying coverage, near-horizon sharpness, and long context respectively.
- **Stability tricks.** Pre-RMSNorm and QK-Norm are there for the same reason they are in large LLMs: to keep attention logits and activations well-behaved as depth and scale grow.

**What does *not* transfer, and is the actual research:**

- **The output is continuous, so the loss is regression, not cross-entropy.** There is no softmax over a vocabulary, no temperature sampling in the base model. Values are predicted as numbers and scored with L1/L2. This is the deepest difference and it ripples everywhere — into the metrics (MASE/CRPS not perplexity), the head (linear projection not embedding-transpose), and the calibration story.
- **Multi-token prediction had to be redesigned into Serial-Token Prediction.** LLM-MTP can peek at future ground-truth tokens during training and discards its extra heads at inference. Neither is acceptable for forecasting — the future is genuinely unknown, and the extra heads *are* the forecaster. TimeSTP is the time-series-native reinvention of MTP, and it is the paper's core novelty precisely because the LLM version does not port over.
- **Predictive bias is a time-series-specific disease.** Language models do not have an analogue of "series usually trend upward." The value-flipping and resampling augmentations exist to cure a pathology that only shows up when your tokens are numbers with magnitude and direction.

The lesson is the encouraging one for cross-pollination: the *scaling machinery* (MoE, RoPE, staging, norms) is largely universal, while the *objective and data treatment* must be redesigned for the modality. Timer-S1 keeps the former and rebuilds the latter, and that split is probably the template for foundation models in any new modality.

## Failure modes: predictive bias and distribution shift

A forecaster's failures are more instructive than its wins, and Timer-S1's design is best understood as a set of defenses against specific, nameable failure modes.

**Predictive bias — the spurious-regularity trap.** Left unchecked, a model trained on real-world series learns shortcuts that correlate with the answer in training but are not causal: "the future is a slightly-shifted copy of the recent past," "series drift upward," "tomorrow looks like today." These shortcuts inflate training metrics and collapse on any series that violates them — a declining market, a regime change, a flat signal. The value-flipping augmentation is a direct antidote to the up-trend prior (every rising example has a falling mirror with identical dynamics), and resampling attacks the frequency-memorization shortcut. The fact that the team built explicit augmentations for these tells you they saw the failures and engineered against them rather than hoping scale would wash them out.

**Distribution shift — the off-corpus cliff.** A zero-shot forecaster is only as good as its corpus's coverage. Feed Timer-S1 a series from a domain unlike anything in TimeBench — an exotic instrument, a pathological sampling pattern, a series dominated by an unmodeled exogenous driver — and accuracy degrades, sometimes silently. The breadth of TimeBench (finance, IoT, meteorology, healthcare, plus synthetic primitives) is the defense, and the synthetic half specifically exists to give clean coverage of primitive patterns that real data under-samples. But no corpus is universal, and the practitioner's responsibility is to sanity-check that their series resembles the training distribution before trusting a zero-shot number.

**The leaderboard-tuning seam.** The most subtle failure mode is not the model's but the *evaluation's*: because the CPT stage draws 50% of its data from a "GIFT-Eval Pretrain" split, the model is, by construction, tuned toward the leaderboard's domains. The paper removes outright test leakage, which is the necessary hygiene, but domain-level alignment between the CPT mixture and the eval is exactly the kind of thing that makes a leaderboard number optimistic relative to truly novel data. This is not cheating — it is standard practice and openly described — but it is the reason an independent reproduction on a *different* benchmark would be the most valuable confirmation of the method.

**Horizon mismatch.** Because the serial advantage concentrates at medium and long horizons, using Timer-S1 for one-step nowcasting is a mild failure mode of a different kind: you pay full freight for serial machinery you barely use. The model will still forecast well, but a smaller, cheaper model would have matched it — a failure of *engineering economics* rather than accuracy.

Each of these is a place where the design's defenses are visible: augmentation against bias, corpus breadth against shift, hygiene against leakage. The honest framing is that the model is robust *within* its trained distribution and degrades gracefully-to-poorly outside it — which is exactly what any foundation model does, said plainly.

## Critique: what's strong, what's thin, and what would change my mind

**What's strong.** The reframing is the kind of contribution that outlives the specific model. "TSFMs plateaued because next-patch pretraining caps usable serial compute, and the fix is a serial prediction axis" is a clean, testable claim, and the falsifiable corollary (gains should concentrate at long horizons) is borne out. The same-data ablation against Timer-3 is exactly the right experiment. And the data work — particularly value-flipping as a one-line cure for the up-trend prior — is the sort of cheap, high-leverage idea that other teams will copy immediately.

**What's thin or unfalsifiable.** The "serial scaling needs all three axes together" thesis is asserted more forcefully than it is dissected; I would want a full ablation grid (STP-only, data-only, pipeline-only, and the pairwise combinations) to see how much each axis contributes and whether the complementarity claim really holds, or whether one axis is doing most of the work. The non-Timer baselines being reported graphically rather than in a numeric table makes independent verification harder than it should be. And the MTP-vs-STP distinction, while real, is argued partly by appeal to "serial computations necessary for long-term forecasting" — a phrase that is intuitive but not quantified; a controlled MTP-with-equal-FLOPs baseline would settle whether *seriality* specifically, as opposed to *more prediction heads*, is what matters.

**What would change my mind.** I would revise my read of this paper downward if a same-FLOPs MTP baseline (parallel heads, no serial chaining) closed most of the gap to STP — that would mean the win is from extra prediction capacity, not seriality per se, and the central story would be wrong. I would revise it upward if an independent reproduction on a *different* held-out leaderboard (not GIFT-Eval, which the pipeline explicitly optimizes the CPT mixture toward) showed the same long-horizon advantage; that would demonstrate the effect generalizes beyond the evaluation the model was tuned for. The CPT stage drawing 50% of its data from "GIFT-Eval Pretrain" is the detail I would scrutinize hardest, because it is the seam where leaderboard-specific tuning could quietly inflate the headline number even with honest test-leak removal.

On balance, the verdict is positive with caveats. The reframing is genuinely valuable and likely durable, the same-data ablation is the right experiment and it lands, and the data-bias work is the kind of unglamorous engineering that separates a model that demos well from one that holds up. The caveats — incomplete ablation grid, graphically-reported baselines, leaderboard-aligned CPT mixture, and a forecasting-only release — are the difference between "promising and well-argued" and "settled." None of them undercut the core idea; they mark where the burden of proof now sits. For a practitioner, the practical takeaway is simpler than the research debate: if you have heterogeneous series and medium-to-long horizons, this is the strongest open zero-shot forecaster available, and the way to use it well is to normalize correctly, slice your evaluation by horizon, and size your hardware off the total parameter count rather than the active one. The research community's job is to run the missing ablations; the practitioner's job is to check that their data looks like TimeBench's before trusting the number — and both jobs are well-defined precisely because the paper is clear about what it does.

## What I'd build with this

A few concrete directions the design opens up:

- **A distilled dense student.** The 8.3B MoE is heavy to serve because all experts must be resident. Distilling Timer-S1 into a dense 0.75B-ish student that keeps the TimeSTP serial heads would test whether the *seriality* (cheap to keep) or the *MoE capacity* (expensive to keep) is the bigger contributor, and would give a deployable model that fits the active-parameter footprint people expect.
- **Probabilistic and multi-task heads.** The public release is forecasting-only. The same serial trunk could carry anomaly-scoring and imputation heads; the STP structure (predict step `j` from internal state) is naturally suited to producing calibrated predictive distributions per horizon, not just point forecasts.
- **A reasoning bridge to ChatTS.** Timer-S1 forecasts; [ChatTS](/blog/machine-learning/signal-processing/chatts-aligning-time-series-llms) reasons about series in natural language. A system that lets ChatTS *call* Timer-S1 as a forecasting tool — "predict the next 96 steps, then explain the expected regime change" — would combine the strongest open numeric forecaster with the strongest open time-series reasoner. The same retrieval-and-tool patterns from [vector databases](/blog/machine-learning/ai-agent/vector-database) and agentic tool use apply directly.
- **Horizon-weighting as a tunable.** The `1/√j` CPT weighting is a hyperparameter dressed as a constant. For a known deployment horizon (say, always-96-step day-ahead load forecasting) you could re-weight CPT to put mass exactly where your application needs it, trading away horizons you never use.
- **Multivariate routing.** The paper notes Timer-S1 stays competitive *without* explicit cross-variate modeling. Adding a cross-series attention path — letting the forecast for one sensor attend to correlated sensors — is the obvious place where Moirai-style any-variate modeling and STP's serial depth might stack into a strictly better model.
- **A serial-depth ablation as a public artifact.** Releasing checkpoints at `H = 4, 8, 16` serial blocks would let the community directly measure the marginal value of serial depth per horizon, turning the paper's central claim into something anyone can re-verify on their own data.

Taken together these are not exotic — they are the natural next moves once you accept the paper's premise that serial depth is a real scaling axis. The fact that the obvious extensions are this concrete is itself a sign the framing is a good one: a useful idea generates a backlog.

## Beyond point forecasting: the heads Timer-S1 doesn't ship yet

The public release is a point forecaster, but the trunk it is built on is general-purpose, and the most interesting near-term work is repurposing that trunk for the other time-series tasks that share its representations.

**Probabilistic forecasting.** The single most-requested capability for any forecaster is calibrated uncertainty — not "the value will be 42" but "the value will be 42 ± 6, and here is the full distribution." The STP structure is well-suited to this because each serial block already produces a per-horizon hidden state; attaching a distributional head (predicting parameters of a distribution, or a set of quantiles, per step) rather than a single value would turn each of the 16 serial outputs into a calibrated predictive distribution. The CRPS metric the model already reports is exactly the score such a head would optimize, so the evaluation machinery is in place; only the head is missing.

**Anomaly detection.** Forecasting and anomaly detection are two views of the same model: an anomaly is a point that a good forecaster did not expect. A trunk that forecasts well is, for free, an anomaly scorer — run it, compare the prediction to the observed value, and flag large normalized residuals. The serial structure adds something here: because the model predicts multiple horizons, you can distinguish a *transient* anomaly (one step surprises the model, the next returns to forecast) from a *regime change* (the whole horizon was mis-forecast), which is a distinction most single-step detectors cannot make.

**Imputation.** Filling gaps in a series is forecasting run backward and forward to meet in the middle. The same patched-Transformer trunk, trained with a masked objective rather than a strictly causal one, would impute missing stretches; ByteDance's data pipeline already does causal mean imputation as a preprocessing step, and a learned imputation head would be a strict upgrade. The challenge is that imputation wants bidirectional context, which sits in tension with the causal mask that forecasting needs — so this is the extension that requires the most architectural surgery, likely a separate fine-tune rather than a bolt-on head.

**Classification and retrieval.** The trunk's pooled hidden states are a learned representation of an entire series, which makes them a natural feature extractor for downstream classification ("is this ECG arrhythmic?") and similarity search ("find machines whose vibration signature resembles this failing one"). The retrieval use case connects directly to the [vector-database](/blog/machine-learning/ai-agent/vector-database) tooling: embed a corpus of series with the Timer-S1 trunk, index the embeddings, and you have semantic search over time series. None of this needs the serial heads at all — it needs only the trunk's representations, which the forecasting objective trains as a side effect.

The throughline is that a strong forecasting trunk is a strong *time-series* trunk, and the serial-prediction objective, by forcing the model to internalize how series evolve, produces representations that transfer to tasks well beyond the one it was trained on. Shipping those heads is the difference between a forecaster and a time-series foundation model in the full sense — and the architecture is clearly built to grow in that direction.

## The bigger picture: serial scaling as a claim about compute

Step back from the specifics and Timer-S1 is making an argument about *where forecasting compute should go*, and that argument generalizes beyond this one model.

The deep-learning era taught the field that more parameters and more data buy more accuracy, and the foundation-model era ran hard on that lever until it stopped moving. The instinctive response to a plateau is "scale harder" — wider layers, more data, more steps. Timer-S1's contribution is to point out that *there is more than one thing to scale*, and that the field had been pouring resources into the axis (parameters) that had run out of headroom while ignoring the axis (serial prediction depth) that had not. Reframed this way, the plateau was never evidence that "time series don't scale"; it was evidence that one particular kind of scaling had saturated, and the saturation hid a second axis nobody was pushing.

There is a satisfying symmetry between this and a parallel story in language modeling, where the recent move has been from "make the model bigger" toward "let the model spend more *inference-time* compute reasoning before it answers." Both are recognitions that raw parameter count is one resource among several, and that for some problems the better marginal dollar goes to *depth of sequential computation* rather than *width of representation*. STP is the time-series incarnation of that insight: the far horizon is hard not because the model is too small but because rolling next-patch prediction gave it no room to *think longer* about steps further out. TimeSTP gives it that room as a first-class architectural feature, kept at inference rather than discarded.

If the thesis holds — and the same-data ablation suggests it does — the implication for anyone building a TSFM is concrete. Do not benchmark your method only along the parameter axis. A model that looks plateaued at 1B parameters under next-patch pretraining might have a wide-open curve under serial prediction, and you would never see it if your only knob is size. The corollary for evaluation is equally concrete: report per-horizon results, because a method whose entire value is long-horizon serial reasoning will look unremarkable on the aggregate or the short end and only reveal itself when you slice by how far into the future you are asking it to see.

There is also a humility lesson in the data work. The most elegant part of Timer-S1 is arguably not the architecture but value-flipping: a one-line augmentation that multiplies series by `−1` to neutralize the up-trend prior. It is a reminder that for a numeric modality, the failure modes are statistical, not linguistic, and that a few cheap, well-targeted data interventions can matter as much as a new block design. The architecture gets the headline; the bias controls get the robustness.

None of this makes Timer-S1 the last word. The multivariate gap is real, the public release is forecasting-only, and the leaderboard-tuning seam means the true generalization story awaits independent reproduction. But as a *reframing* — "the bottleneck was the objective's serial ceiling, not the parameter count" — it is the kind of idea that reorganizes how a field thinks about its own scaling curves, and that is worth more than any single leaderboard row.

## References

- **Paper:** Timer-S1: A Billion-Scale Time Series Foundation Model with Serial Scaling — [arXiv:2603.04791](https://arxiv.org/abs/2603.04791) ([HTML](https://arxiv.org/html/2603.04791v1))
- **Model weights:** [bytedance-research/Timer-S1 on HuggingFace](https://huggingface.co/bytedance-research/Timer-S1)
- **Leaderboard:** GIFT-Eval (general time-series forecasting evaluation)
- **Sibling posts on this blog:**
  - [The ByteDance Research Model Atlas](/blog/machine-learning/bytedance-research-model-atlas) — where Timer-S1 sits in ByteDance's open-model lineup
  - [ChatTS: Teaching LLMs to Actually Read Time Series](/blog/machine-learning/signal-processing/chatts-aligning-time-series-llms) — the understanding-and-reasoning counterpart
  - [The DeepSeek-MoE Lineage](/blog/machine-learning/large-language-model/deepseek-moe-lineage-fine-grained-shared-experts) — fine-grained and shared-expert routing background
  - [Vector Databases](/blog/machine-learning/ai-agent/vector-database) — retrieval plumbing for tool-using forecasting systems

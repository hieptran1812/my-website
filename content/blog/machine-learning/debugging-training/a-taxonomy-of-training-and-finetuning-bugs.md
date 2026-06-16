---
title: "A Taxonomy of Training and Finetuning Bugs: Symptom → Suspect → Test → Fix"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A structured decision framework that turns any stalled, NaN-ing, or secretly-overfit training run into a localized bug in one of six places — then names the cheapest test that confirms it and the fix that clears it."
tags:
  [
    "debugging",
    "model-training",
    "finetuning",
    "deep-learning",
    "pytorch",
    "data-leakage",
    "mixed-precision",
    "llm",
    "distributed-training",
    "evaluation",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 58
image: "/imgs/blogs/a-taxonomy-of-training-and-finetuning-bugs-1.png"
---

Here is a run that looks like a success and is actually a catastrophe. A teammate finetunes a 7B model on a support-ticket dataset. The loss curve is a textbook downward slope. Validation accuracy hits 94%. They ship it. In production it answers like the base model — no domain knowledge, no new behavior, nothing learned. Three days and roughly \$1,800 of GPU time produced a model that is, functionally, a no-op. The adapter never entered the computation graph. Zero parameters were trainable. The loss went down because the *base model* was already good at next-token prediction on English text, and the "improvement" was the optimizer rounding noise. Every instrument said healthy. The model learned nothing.

This is the defining property of training bugs: **your run lies to you.** A loss curve that goes down is not evidence the model is learning the thing you want. A validation accuracy of 99% is not evidence of a good model — it is, far more often, evidence of a leak. A run that does not crash is not a run that is correct. Unlike a web server that throws a stack trace, a broken training run frequently produces *plausible-looking numbers*, and the gap between "plausible" and "correct" is where weeks disappear. The whole craft of debugging training is learning to distrust the instruments in a *disciplined* way — to know which lie each one tells, and which cheap test forces the truth out.

This post is the master map for that craft. The claim it defends is simple and, once you internalize it, freeing: **a training or finetuning bug hides in exactly one of six places — data, model code, optimization, numerics, systems, or evaluation — and you can bisect to the right one before touching code.** You do not guess. You do not "try lowering the learning rate and see." You run a small number of cheap, discriminating tests that each cut the space of suspects roughly in half, and within five tests and usually under ten minutes you are staring at the one place the bug lives. Figure 1 is the spine of the whole thing: a symptom at the top, six suspects in the middle, and the single cheapest test that confirms or clears each one.

![A decision tree that routes a training symptom through fast and stateful tests down to the six suspect places, each annotated with its cheapest confirming test such as overfit one batch or single-GPU repro](/imgs/blogs/a-taxonomy-of-training-and-finetuning-bugs-1.png)

By the end you will be able to take any stalled, diverging, NaN-ing, or secretly-overfit run — in vision, LLM, speech, or tabular — and localize the bug to one of six places in minutes, name the test that confirms it, and know the direction of the fix. This is the conceptual hub of the series; every later post is one branch of this tree drawn in full detail. If you read only one post here, read this one, then jump to the branch that matches your symptom.

## 1. Why training bugs are different (and harder)

In ordinary software, a bug usually announces itself. The program crashes, the test goes red, the assertion fires, the type checker complains. The feedback loop is tight and the failure is *loud*. You change a line, rerun, and within seconds you know if you helped or hurt.

Training breaks all three of those comforts. The feedback loop is long — a single epoch can take hours. The failure is quiet — the most expensive bugs in ML do not crash; they produce numbers that look fine. And the system is *stochastic*, so two runs of the same code give two different loss curves, and you cannot tell whether the change you made mattered or whether you are looking at seed noise.

There is a deeper reason training bugs are hard, and it is worth making precise because it shapes the entire debugging method. A neural network is a *universal-ish function approximator being fit by gradient descent*. That means it will try, with enormous flexibility, to drive the loss down using **whatever signal is available** — including signal you did not intend to provide. If your validation set shares rows with your training set, the model will exploit that. If an ID column is correlated with the label, the model will read the ID. If your augmentation accidentally encodes the class, the model will decode it. The optimizer is an adversary that finds shortcuts. This is why "the loss went down" is such a weak guarantee: a low loss tells you the model found *some* way to predict the target, not that it found the *right* way.

So the question is never "is the loss going down?" The question is always "which of the six places is the loss going down *because of*?" That reframing is the entire game. A symptom — stuck at chance, a NaN at step 412, a too-good 99% validation, a finetune that forgot everything — is a clue about which place the bug lives, but it is rarely conclusive on its own. The discipline is to convert the symptom into a *hypothesis about which place*, then run the cheapest test that confirms or refutes that hypothesis. We will make that loop rigorous in the next section.

## 2. The scientific method, applied to a training run

Debugging a model is applied science, and the people who are fast at it are the ones who treat it exactly that way: hypothesis, prediction, experiment, update. The slow ones treat it as folklore — they change things they have a feeling about and rerun. The difference in speed is roughly an order of magnitude, and it comes entirely from one habit: **never run an experiment that cannot change your mind.**

Here is the loop, stated as three questions you ask in order:

1. **What is the symptom, stated precisely?** Not "the model is bad" but "validation cross-entropy plateaus at 0.69 from step 1 and never moves, while train loss also stays at 0.69." The number 0.69 is not decoration — it is $\ln 2 \approx 0.693$, the cross-entropy of a uniform two-class prediction. A loss pinned at exactly that value is the model predicting 50/50 on every example. That single observation already rules out half the suspects, because a model stuck at the *theoretical chance floor* is not learning *anything*, which points at optimization or model code, not at a subtle data leak.

2. **What is the cheapest test that discriminates between my top two suspects?** This is the crux skill. You almost always have two leading hypotheses. The question is which experiment has outcomes that *differ* under those two hypotheses, and which of those experiments is cheapest to run. If suspect A predicts the overfit-one-batch test passes and suspect B predicts it fails, then that test — which costs you two minutes — is worth more than a full retraining run that costs you four hours, because the retraining run does not discriminate.

3. **Run it, then update — and re-measure the fix.** After the test points to a place, you fix one thing, and then you *measure again with the same instrument*, because a "fix" that does not move the instrument is not a fix; it is a coincidence you have not understood yet. The before→after measurement is not a formality. It is the only thing standing between you and a phantom fix that will reappear next week.

The reason this loop is so much faster than trial-and-error is information-theoretic. If you have six equally-likely suspects, the *worst-case* number of yes/no tests to isolate one is $\lceil \log_2 6 \rceil = 3$. With trial-and-error — try a fix, see if the symptom vanishes — you are not asking yes/no questions about *where* the bug is; you are asking "did this specific fix work?", and there are far more than six possible fixes, so you wander. Bisection on the *place* converges in a handful of steps; bisection on the *fix* does not converge at all. That is the whole argument for localizing before fixing, and it is why this post is organized around *places and tests* rather than around a long list of fixes.

#### Worked example: the loss pinned at ln 2

A binary classifier on a tabular fraud dataset. Train loss starts at 0.71, drops to 0.693, and stays flat for 50 epochs. Validation matches. Accuracy is 50% on a balanced set.

Hypothesis A: the learning rate is so low the model is barely moving. Prediction under A: if I run the LR-range test, loss will start to drop at some larger LR. Hypothesis B: gradients are not reaching the model at all (a `requires_grad=False` somewhere, or a `.detach()` in the forward, or labels and inputs are misaligned so there is no learnable signal). Prediction under B: even at a huge LR, loss will not move, *and* the overfit-one-batch test will fail.

Cheapest discriminating test: overfit one batch (two minutes). I take a single batch of 32 rows and call `loss.backward()` and `optimizer.step()` 300 times on *that same batch*. Under A, the loss should crash toward 0 — a network can always memorize 32 rows if gradients flow. Under B, it stays at 0.693.

Result: it stayed at 0.693. That immediately killed hypothesis A — the LR was never the problem — and indicted the model code. I printed `[ (n, p.grad is None) for n, p in model.named_parameters() ]` and found that 80% of the parameters had `grad is None`. The feature-encoder submodule had been wrapped in `torch.no_grad()` during a refactor. Removing that one context manager, the overfit-one-batch loss dropped from 0.693 to 0.004 in 120 steps. The full run then trained normally to 0.31 validation loss and 88% accuracy. Total debugging time: under fifteen minutes, because the first cheap test cut the suspect space in half and the second confirmed the exact place.

That is the method. The rest of this post gives you the map of places, the cheapest test for each, and the lookup table that ties symptoms to all of it.

## 3. The six places a bug hides

Internalize these six and you have a coordinate system for every training bug you will ever see. They are not arbitrary categories — they correspond to the actual layers of a training run, from the data on disk to the metric on the dashboard. Figure 2 stacks them.

![A vertical stack of the six layers of a training run from data at the top through model code, optimization, numerics, systems, and evaluation at the bottom, each labeled with its characteristic bugs](/imgs/blogs/a-taxonomy-of-training-and-finetuning-bugs-2.png)

**1. Data.** Everything before the loss sees a tensor: the rows themselves, the splits, the input pipeline, the labels, the augmentation. This is empirically the number-one source of real bugs, and the most insidious, because data bugs produce *plausible* numbers. Leakage gives you a too-good validation score. Label noise caps your accuracy at a ceiling you cannot push through. A normalization mismatch between train and serving silently degrades production. A collate or padding bug averages across the wrong axis. An augmentation applied to the label as well as the input destroys the very thing you are predicting. The characteristic symptoms are a validation score that is suspiciously good, a train-to-serving gap, or weird per-class behavior. The cheapest tests are *look at the actual batch* (print the tensors, decode the tokens, plot the image), and check the splits for duplicates and near-duplicates. The reason this place is first in the stack and first in frequency is that it sits *upstream of everything*: a data bug poisons the loss before the model, the optimizer, or the metric ever get a turn, so a corrupted input produces a corrupted gradient that no amount of model-code correctness can repair. When in doubt, look at your data first — the thirty minutes you spend decoding a batch by hand save the week you would otherwise spend rewriting an architecture that was never the problem.

**2. Model code.** The forward pass and the loss computation as written in your framework. Shape and broadcasting bugs that do not crash but average the wrong axis. Parameters frozen by accident. Forgetting to call `.eval()` so dropout and BatchNorm run at inference. An attention mask that lets the model peek at future tokens. A loss reduction that is `sum` when you meant `mean`. The signature here is distinctive: the **overfit-one-batch test fails**, or you get great train loss and garbage eval. The cheapest test is overfit-one-batch plus a gradient-flow check.

**3. Optimization.** The learning rate, the schedule and warmup, the optimizer and its state, gradient clipping. Too-high LR gives you a spike then divergence; too-low gives you a crawl or a plateau; a warmup bug gives you an early NaN; bad optimizer state on resume gives you a loss jump. Symptoms: stuck, spiking, or crawling. Cheapest tests: the LR-range test, and grad-norm logging.

**4. Numerics.** The floating-point reality underneath the math. `log(0)`, division by zero, `exp` overflow, `sqrt` of a negative, fp16 gradient underflow, a bad initialization, BatchNorm running-stats divergence. The signature is a NaN or Inf appearing at a specific step, or a curve that is smooth and then explodes. The cheapest tests are `torch.autograd.set_detect_anomaly(True)`, bisecting by step and by layer, and switching fp16 to bf16.

**5. Systems.** Everything about *how* the run executes across hardware: DDP gradient synchronization, gradient accumulation arithmetic, out-of-memory, FSDP sharding and checkpoint resume, the dataloader bottleneck, nondeterminism. The tell is that the run behaves *differently single-GPU vs multi-GPU*, or the GPU sits idle, or the loss jumps on resume. The cheapest tests are a single-GPU reproduction and the profiler.

**6. Evaluation.** The metric and the way you measure it. A metric bug (micro vs macro, the wrong averaging), eval-train skew, overfitting to the validation set through repeated tuning, leakage in the *eval* set specifically, an offline-online gap. The symptom is a great offline number and a bad online one, or a metric that does not move when the thing you actually care about moves. The cheapest tests are to *re-derive the metric by hand* on a few examples, and to evaluate on a fresh, never-touched holdout.

Two notes before we move on. First, these places are not perfectly disjoint — a chat-template mismatch is partly a data bug and partly an evaluation bug — but the overlap is small and it does not hurt the method, because the *tests* still discriminate. Second, the ordering of the stack roughly tracks how early in the pipeline the bug sits, which is also roughly the order of how often each place is the culprit in practice: data and model-code bugs dominate, numerics and systems bugs are rarer but nastier when they hit, and evaluation bugs are the ones that survive longest because nothing crashes.

### Why each place produces a distinct signature

The reason the taxonomy works — the reason a symptom *routes* to a place — is that each place has a mechanism that produces a characteristic, *predictable* signature. If you understand the mechanism, you do not have to memorize the table; you can re-derive it. Three of the six places are worth making rigorous, because their math is the backbone of half the bugs you will meet.

**Why a data leak inflates a metric, and by how much.** A leak is, formally, the presence of a feature (or a split contamination) that is correlated with the label at *training time* but unavailable or differently-distributed at *serving time*. Suppose your true model achieves a generalization accuracy $a$, and a leaked feature lets the model "cheat" on a fraction $\rho$ of validation examples by reading the answer directly. The observed validation accuracy is then approximately $a_{\text{obs}} = a + \rho\,(1 - a)$ — the honest accuracy, plus the cheat-fraction times the room left above it. With an honest $a = 0.78$ and a leak that nails $\rho = 0.45$ of the remaining errors, $a_{\text{obs}} = 0.78 + 0.45 \times 0.22 \approx 0.88$, and a stronger leak pushes it toward 0.97. The signature is exactly what we saw in the churn example: the *gap* between $a_{\text{obs}}$ and production tracks $\rho$, and *ablating the leaking feature collapses $a_{\text{obs}}$ back toward $a$*. That collapse is not a coincidence; it is the term $\rho(1-a)$ disappearing. This is why the leak test is "ablate and watch it fall" — the math predicts a *fall proportional to the leak's reach*.

**Why fp16 underflows gradients, exactly.** Half-precision floating point (fp16) has 5 exponent bits and 10 mantissa bits, which fixes its smallest positive *normal* value at $2^{-14} \approx 6.1\times10^{-5}$. Any gradient whose magnitude falls below that floor either underflows to a denormal (with catastrophic precision loss) or to zero outright. Late in training, or in the early layers of a deep network, a large share of gradients are genuinely small — $10^{-6}$ or below — so in fp16 they round to *zero*, and a parameter whose gradient is zero *does not move*. The signature is precise and predictable: a run that trains correctly in fp32 *plateaus early* or quietly under-trains in fp16, with **no NaN and no crash** — the most dangerous kind of bug, because every instrument looks healthy. The fix, loss scaling, multiplies the loss by a constant $S$ (say $2^{16}$) before the backward pass, which by the chain rule scales *every* gradient up by $S$, lifting the small ones back above the $6.1\times10^{-5}$ floor; the optimizer then divides by $S$ before stepping. bf16 sidesteps the whole problem by keeping fp32's 8 exponent bits (so its smallest normal is $\approx 1.2\times10^{-38}$) and sacrificing mantissa precision instead. Knowing this floor lets you *predict* which runs will silently underflow and read it off the gradient histogram before it costs you a run.

**Why a too-high learning rate spikes then diverges.** Gradient descent on a locally-quadratic loss with curvature (largest Hessian eigenvalue) $\lambda$ is stable only when the learning rate satisfies $\eta < 2/\lambda$. Below that bound, each step shrinks the distance to the minimum; above it, each step *overshoots and amplifies*, so the loss grows geometrically — a spike — and the now-huge activations push some operation into its numerical edge, where the spike becomes a NaN. This is why the signature of a too-high LR is so consistent: a sharp upward spike followed, often within a few steps, by divergence or a NaN. The grad norm is the leading indicator — it climbs past $10^3$ in the step *before* the loss explodes — which is exactly why per-layer grad-norm logging catches this bug before the dashboard does. The fix direction (lower LR, clip the grad norm to a ceiling, add warmup) all serve the same end: keep $\eta\lambda$ under 2.

These three mechanisms — the leak-inflation term, the fp16 underflow floor, and the $\eta < 2/\lambda$ stability bound — are not trivia. Each one *predicts* a signature, and a predicted signature is what lets you bisect: you see the signature on the dashboard, you name the place, and you run the one test the mechanism tells you will confirm it. The rest of the six places have their own mechanisms (we derive several in the per-place posts), but these three carry most of the weight in practice.

## 4. The bisection procedure: the unifying method

Here is the single most important idea in the series, and it is borrowed wholesale from debugging any complex system: **you do not search for the bug, you bisect for the place.** Given a symptom, you run tests that each split the six places into "still possible" and "ruled out," choosing the test that splits the *current* suspect set most evenly and costs the least. Figure 5 lays the five core tests out in cost order; run them in this order and each one earns its keep.

![A left-to-right timeline of the five core diagnostic tests in cost order, from fixing the seed to overfitting one batch, printing gradient norms, detect anomaly, and a single-GPU reproduction, each labeled with what it rules out](/imgs/blogs/a-taxonomy-of-training-and-finetuning-bugs-5.png)

The five tests, and the cut each one makes:

- **Fix the seed (and make the run deterministic enough).** This is test zero, the precondition for everything else. If your run is not repeatable, you cannot tell a fix from noise, and every later test becomes unreliable. Set the seeds, control the dataloader workers, and turn on deterministic algorithms. It does not localize a bug by itself — it makes every other test *trustworthy*.

- **Overfit one batch.** The single highest-leverage test in all of ML debugging. Take one or two batches and train on them, repeatedly, until the loss is near zero. If the loss drives to near zero, your model code and optimizer can learn *something* — which clears most of model-code and optimization, and points you at data or evaluation. If the loss *cannot* reach near zero on data the model has seen hundreds of times, the bug is in the model code or the optimization, full stop, because a correctly-wired network with a sane LR can always memorize a handful of examples. This test alone splits the six places roughly down the middle, which is why it is first among the localizing tests.

- **Print per-layer gradient norms.** A hook that logs `p.grad.norm()` for each parameter, every N steps. This reads the optimization and numerics instruments directly. Norms that are uniformly tiny mean vanishing gradients or a too-low LR; norms that explode (1e4 and climbing) mean a too-high LR, a bad batch, or impending numerical divergence; norms that are zero or `None` for a whole submodule mean that submodule is frozen or detached — a model-code bug.

- **`detect_anomaly`.** Wrap the forward and backward in `torch.autograd.set_detect_anomaly(True)` and PyTorch will point at the exact operation whose backward produced the first NaN. This is the numerics scalpel. It is slow, so you do not leave it on, but when a NaN appears it converts "NaN somewhere at step 412" into "NaN from the `log` in your custom loss," which is the difference between an afternoon and a minute.

- **Single-GPU reproduction.** Run the exact same code on one GPU with the same effective batch. If the symptom *vanishes* single-GPU, the bug is in the systems layer — gradient sync, accumulation arithmetic, sharding, rank desync. If it *persists* single-GPU, systems is cleared and you are back to the other five places, now debugging in a far simpler environment. This test is the one that splits the systems layer cleanly off from everything else.

Figure 6 shows how just the second of these — overfit-one-batch — forks the entire search, because its outcome is so discriminating.

![A branching decision graph where the overfit-one-batch test forks into a passing path that clears model and optimization and points at data or evaluation, and a failing path that indicts the model code or a too-low learning rate](/imgs/blogs/a-taxonomy-of-training-and-finetuning-bugs-6.png)

The art is in *ordering* and *stopping*. Order by `(discrimination ÷ cost)` — run the test that rules out the most suspects per minute. Stop the moment a single place remains, then switch from bisection to the place-specific deep dive. And crucially: **after each test, write down what it ruled out.** The reason people go in circles is that they re-run tests whose results they already have, because they never wrote the result down. A two-line note — "overfit-one-batch: PASS, so model+optim cleared" — is the difference between converging and wandering.

The five tests above are the universal core. Each *place* then has its own deeper battery — the LR-range test for optimization, the leak-detector for data, the metric re-derivation for evaluation — and those are the per-place posts in this series. The taxonomy here tells you *which* battery to reach for; the per-place posts tell you how to run it.

### The math of ordering: why this order, and why it converges

It is worth being precise about *why* the five tests are run in this order, because the ordering is not arbitrary and the convergence is provable. Treat each test as a partition of the current suspect set into "still possible" and "ruled out." The information a test yields is maximized when it splits the suspect set into two roughly equal halves — a test that can only ever rule out one of six suspects gives you at most $\log_2(6/5) \approx 0.26$ bits, while a test that cleanly halves the set gives you a full bit. Overfit-one-batch is first among the localizing tests precisely because its two outcomes (pass / fail) split the six places close to evenly: a pass clears model code and optimization (and most of numerics, since a numerically-broken forward usually cannot reach zero loss), leaving data and evaluation; a fail indicts model code or optimization. That is the most balanced single cut available, so it goes first.

The cost term matters just as much. We order by the *ratio* of discrimination to cost, not by discrimination alone. `detect_anomaly` is extremely discriminating for numerics, but it is slow (it roughly doubles step time), so it ranks *below* overfit-one-batch and grad-norm logging, which are nearly free. The single-GPU reproduction is last not because it is uninformative — it is the *only* test that isolates the systems layer — but because it is the most expensive to set up and only pays off when the symptom is multi-GPU-specific. So you reach for it only after the cheaper tests have failed to localize, or immediately when the symptom *announces* a systems bug ("works on one GPU, breaks on eight").

Convergence then follows from the partition view. With six suspects and tests that each cut the set roughly in half, you reach a single place in $\lceil \log_2 6 \rceil = 3$ well-chosen tests in the worst case, and often in one or two because real symptoms are not uniform over the six places — a NaN is *a priori* numerics, a too-good val is *a priori* data, so your prior already concentrates the mass and the first test usually confirms it. This is the formal reason bisection beats trial-and-error: trial-and-error asks "did this fix work?", which is a question about the (very large) space of *fixes*, while bisection asks "which place?", a question about a space of *six*. Searching six is fast; searching the fixes is not.

### Stress-testing the diagnosis

A diagnosis you have not stress-tested is a guess wearing a lab coat. Once a test points to a place, ask the *what-if* questions that would break a wrong diagnosis. What if the symptom is data, not optimization — would the LR-range test still look flat? (Yes, because a data bug starves the model of signal regardless of LR, so a flat LR-range curve does *not* by itself confirm optimization; you need overfit-one-batch to separate them.) What if it only fails at fp16 — does the symptom survive in fp32? (If it vanishes in fp32, you have isolated numerics to a precision issue, and the fix is loss scaling or bf16, not the model.) What if the batch is tiny — does a BatchNorm-using model behave differently at batch size 2 vs 64? (It will, because BN's running-statistics estimate is high-variance at small batch, so a "bug" that only appears at small batch may be a BN small-batch failure, not a model bug.) What if it only fails on multi-GPU — does single-GPU clear it? (If yes, it is systems; if no, systems is cleared.) Each what-if is a *second* test whose outcome differs under the diagnosis and its nearest rival, and running it before you commit to a fix is what separates a confirmed root cause from a plausible story.

## 5. The symptom → suspect lookup table

This is the central artifact of the post — the table you bookmark and consult mid-firefight. It maps a symptom you can *observe on the dashboard* to the most likely suspect, the cheapest test that confirms it, and the direction of the fix. Figure 3 renders the core of it; the fuller table follows in markdown so you can scan it.

![A matrix mapping five common training symptoms to their most likely suspect place, the cheapest confirming test, and the fix direction, with cells color-coded by severity](/imgs/blogs/a-taxonomy-of-training-and-finetuning-bugs-3.png)

| Symptom (what you observe) | Most likely suspect | Cheapest confirming test | Fix direction |
|---|---|---|---|
| Loss pinned at chance (e.g. $\ln 2$, $\ln C$) from step 1 | Model code or optimization | Overfit one batch | Fix grad flow; if it passes, raise LR / fix init |
| Loss crawls down very slowly | Optimization (LR too low) | LR-range test | Raise LR; add warmup; check schedule |
| Loss spike then divergence/NaN | Optimization → numerics | Grad-norm log + `detect_anomaly` | Lower LR; clip grads; switch to bf16 |
| NaN/Inf at a specific step N | Numerics | `detect_anomaly`; bisect by step | Guard `log0`/`/0`; clamp; loss scaling or bf16 |
| Validation 95–99%, too good | Data (leakage) | Dedup rows across splits; ablate features | GroupKFold / time split; drop leaking column |
| Great train loss, garbage eval | Model code or evaluation | Check `.eval()` mode; re-derive metric | Call `.eval()`; fix mask; fix metric |
| Accuracy fine, but the metric you care about is bad | Evaluation | Re-derive metric by hand on 5 examples | Pick a metric that tracks the goal |
| Per-class behavior is bizarre | Data (imbalance/labels) | Confusion matrix; per-class recall | Class weights; clean labels; threshold-move |
| Run differs single-GPU vs multi-GPU | Systems | Single-GPU reproduction | Fix grad-sync / accumulation / sharding |
| GPU utilization low, training slow | Systems (dataloader/transfer) | PyTorch profiler | More workers; pin memory; prefetch |
| Loss jumps up on resume from checkpoint | Systems (state restore) | Compare optim/LR/RNG state pre/post | Save+restore optimizer, scheduler, scaler, RNG |
| Offline metric great, online bad | Evaluation or data shift | Fresh holdout; train-serve skew check | Match serving conditions; fix the eval set |
| Finetune loss good, behavior unchanged | Model code (LoRA no-op) or data (template) | `print_trainable_parameters`; inspect formatted text | Fix `target_modules`; match chat template |

A few principles make this table *usable* rather than just true. Read it top to bottom only until you find a symptom that matches yours precisely, then run *that row's* test before doing anything else. Do not run the fix in the last column until the test in the third column has confirmed the suspect — the fix column is the *direction*, not a guaranteed cure, and applying fixes without confirming wastes the run. And notice that several symptoms map to *two* suspects with a single test that discriminates between them: "great train loss, garbage eval" is either a model-code mode bug or an evaluation bug, and the cheap test (is `.eval()` actually being called, and does the metric re-derive correctly?) tells you which. That two-suspects-one-test pattern is the bisection method compressed into a single table row.

Once the table routes you to a place, you switch from the universal core tests to that place's *dedicated battery*, and each place has a post in this series that runs the battery in full. If the table sent you to **model code or optimization** because a loss is stuck, the next stop is [the overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test), which is the deepest treatment of the highest-leverage test and its variants. If it sent you to **data** because a validation number looks too good, [data leakage, the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer) walks the full leak-detection battery — ablation, deduplication, temporal and group splits. If it sent you to **optimization** because the loss crawls or spikes, [the learning rate is almost always the problem](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem) runs the LR-range test end to end. If it sent you to **numerics** because of a NaN, [hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs) is the by-step, by-layer hunt. If it sent you to **systems** because the run differs across GPUs, [debugging DDP and multi-GPU](/blog/machine-learning/debugging-training/debugging-ddp-and-multi-gpu) covers gradient-sync correctness and the scaling traps. And if it sent you to **evaluation** because a metric is misbehaving, [your metric is lying](/blog/machine-learning/debugging-training/your-metric-is-lying) covers micro-vs-macro, eval-train skew, and the offline-online gap. The taxonomy is the index; those posts are the chapters.

## 6. The instruments and what each one rules out

Bisection needs instruments, and a small set of them covers the great majority of bugs. The point of an instrument is not to *look healthy* — it is to *rule a place out*. Here is what each common instrument tells you and, more importantly, what it cannot.

**Train loss.** Tells you the optimizer is reducing *the loss you wrote*. It does *not* tell you the model is learning the task — a leak, a wrong loss reduction, or training-on-the-prompt all reduce the loss while learning the wrong thing. Use the train loss to detect divergence and spikes, never to certify correctness.

**Validation loss / metric.** Tells you the model generalizes to *held-out data drawn the same way as training*. It does *not* tell you the model generalizes to *production* if the split was contaminated, the val set leaks, or the distribution shifts. A val metric that is too good is a red flag, not a green light.

**Gradient norm (global and per-layer).** The most underrated instrument. A healthy run has a global grad norm that is finite, not exploding, and not collapsing to zero. Per-layer norms reveal where signal dies: a uniformly tiny early-layer norm is vanishing gradients; a single submodule at exactly zero is a frozen or detached module; a norm climbing past 1e3 is the leading edge of divergence. Logging grad norm costs almost nothing and rules out a remarkable amount.

**Parameter and update norms, and their ratio.** The ratio $\frac{\lVert \Delta\theta \rVert}{\lVert \theta \rVert}$ per step — how much the weights move relative to their size — is a scale-free health signal. Healthy training keeps this around $10^{-3}$ to $10^{-2}$. A ratio near $10^{-1}$ means the LR is too high (weights lurching); a ratio near $10^{-5}$ means the LR is too low (weights barely moving). This single number diagnoses most LR problems without an LR-range test.

**Activation and weight histograms.** Reveal dead neurons (a ReLU block stuck at zero), saturation (tanh/sigmoid pinned at the rails), and exploding activations before they become a NaN. A dead block shows as a spike at zero in the activation histogram that never recovers.

**Throughput / GPU utilization.** Rules the systems layer in or out for *performance* bugs. A GPU sitting at 31% utilization is starved — almost always the dataloader, sometimes host-device transfer — and no amount of model-code debugging will fix that.

**The loss-input distribution.** A frequently-skipped instrument: log a histogram of the *raw inputs to the loss function* — the logits, the target indices, the predicted probabilities — not just the scalar loss. This is what catches the numerics bugs before they NaN (a probability touching exactly 0 or 1, a logit at $\pm 50$ that will overflow the `exp` in a softmax) and the data bugs that masquerade as model bugs (a target index of $-1$ where you expected $0..C-1$, a label that is all-zeros because masking ate it). A scalar loss hides all of this; the distribution of its inputs reveals it. The cost is one histogram call per N steps, and it is the single best early-warning instrument for the numerics place.

**Trainable-parameter count (finetuning).** Specific to finetuning, but decisive: `model.print_trainable_parameters()` tells you, before a single GPU-hour, whether your adapter actually entered the graph. A trainable percentage of `0.0` is a guaranteed no-op; a percentage far from what your `LoraConfig` implies is a partial misconfiguration. This is the one instrument that catches the most expensive finetuning bug — the silent no-op — for free.

The discipline is to log these *from the start*, not to add them after the run breaks, because half of debugging is being able to look at the instruments *at the step the symptom appeared*. There is a temptation, every time, to add instrumentation *after* the run has already broken — but the instruments you most need are the ones recording the state at the step the symptom first appeared, and you cannot retroactively log a step that has already passed. A small, always-on instrument panel is cheap insurance: grad norm, the update-to-parameter ratio, the LR, throughput, the loss-input histogram, and a handful of decoded inputs. The per-place posts go deep on each instrument; the lesson here is that instruments rule places *out*, and a run with no instruments is a run you cannot bisect. The companion post on [instrumenting a training run](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log) builds the full panel and explains what each signal rules out.

## 7. Finetuning bugs: the same six places, biting differently

Finetuning is not a different debugging discipline — the bug still lives in one of the six places — but the *characteristic bugs* are different enough that they deserve their own overlay. The reason is that finetuning starts from a *competent* model, so the failure modes are about *disturbing* that competence (too-high LR destroys pretrained features; catastrophic forgetting erases old skills) and about *plumbing* the new data correctly (chat templates, loss masking, adapter wiring). Figure 7 maps the four classic finetuning bugs to their place, their tell, and their fix.

![A matrix mapping four finetuning bugs — wrong learning rate by a hundred times, catastrophic forgetting, chat-template skew, and a LoRA no-op — to where each one hides, the symptom that reveals it, and its fix](/imgs/blogs/a-taxonomy-of-training-and-finetuning-bugs-7.png)

**Wrong LR by 100×** lives in **optimization**. The most common single finetuning mistake is reusing a from-scratch learning rate. Pretraining a transformer might use a peak LR around $10^{-3}$ or $10^{-4}$; finetuning the *same* model usually wants $10^{-5}$ to $2\times10^{-5}$. The mechanism is direct: the pretrained weights already sit near a good minimum, and a large step blasts them out of that basin, destroying the very features you paid to learn. The tell is a loss that *spikes* in the first few hundred steps of epoch 1, or a model that scores worse on the original task after finetuning than before. The fix is to drop the LR by one to two orders of magnitude and add a short warmup.

**Catastrophic forgetting** lives in **data and optimization together**. The model learns the new task and *unlearns* the old one because nothing in the loss preserves the old behavior. The tell is that your finetuned model aces the new eval and regresses on a held-out set of *original* capabilities. You only see it if you *kept an old-task eval* — which is exactly why this bug survives so long. The fix direction is a lower LR, fewer epochs, mixing in replay data from the original distribution, or parameter-efficient methods that touch fewer weights.

**Chat-template / formatting skew** lives in **data and evaluation**. You train on text formatted one way and serve with a different template — a missing generation prompt, different role tokens, a different system-prompt wrapper. The loss looks great because the model learns *your training format perfectly*; the behavior is bad because production never sends that format. The tell is good training loss and bad real-world chat quality. The fix is to reproduce the *exact* serving template in training, byte for byte, and to evaluate using that same template.

**The LoRA no-op** lives in **model code**. The adapter is configured but never enters the computation graph — wrong `target_modules`, the base model not actually wrapped, gradient checkpointing interacting badly, or a dtype mismatch that detaches the adapter. The tell is unambiguous and you should *always* check it: `model.print_trainable_parameters()` reports a trainable count that is zero or absurdly small relative to the adapter you configured. The fix is to correct `target_modules` to match the actual module names in your model and confirm the trainable-parameter count is what you expect *before* spending GPU hours.

The meta-lesson of the finetuning overlay is that finetuning has *more places to silently no-op* than from-scratch training, because the base model is good enough to produce a healthy-looking loss curve even when your contribution is doing nothing. So the single most valuable finetuning habit is to **prove the new signal is entering the graph before you trust the loss** — check trainable params, check that the formatted text is what you think, and keep an old-task eval to catch forgetting.

#### Worked example: the finetune that learned nothing

Back to the opening story, debugged properly this time. Symptom: 7B finetune, loss drops smoothly from 1.9 to 1.6 over one epoch, but the served model behaves identically to the base. Two suspects: model code (adapter no-op) or data (template skew making the loss easy but the behavior wrong).

Cheapest discriminating test: `model.print_trainable_parameters()`, which costs zero GPU time. Under the no-op hypothesis it reports near-zero trainable params; under the template hypothesis it reports the expected adapter size (say, 0.3% of total). It reported `trainable params: 0 || all params: 6,738,415,616 || trainable%: 0.0`. That instantly confirmed the model-code suspect and killed the data hypothesis.

The root cause: `target_modules=["query_key_value"]` was copied from a GPT-NeoX config, but this model's attention projections are named `q_proj`, `k_proj`, `v_proj`. No module matched, so `get_peft_model` injected zero adapters, and the "training" updated nothing — the loss fell only because the frozen base model's logits drift slightly under the optimizer's interaction with the (empty) LoRA scaling. After fixing `target_modules` to `["q_proj", "k_proj", "v_proj", "o_proj"]`, `print_trainable_parameters` reported `trainable%: 0.31`, the overfit-one-batch test drove a 64-example loss from 1.9 to 0.05 in 200 steps (proving signal now flowed), and the full finetune produced a model that actually answered in the new domain. The bug was a single mismatched string. The cost of *not* checking the trainable count up front was the entire first run. This is why the LoRA-no-op check is the cheapest, highest-value test in the finetuning overlay.

## 8. Diagnostic code: the first five tests, runnable

Theory is cheap; here is the code. These five snippets are the runnable form of the bisection procedure — copy them into any PyTorch project and they will localize the majority of bugs. They use only PyTorch and standard libraries so they run anywhere.

**Test 0 — determinism.** Make the run repeatable so every later test is trustworthy.

```python
import os, random, numpy as np, torch

def set_determinism(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Force deterministic kernels; some ops will raise if no det. impl exists.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    # Needed for deterministic matmul on some CUDA versions.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def seed_worker(worker_id):
    # Each DataLoader worker must reseed or augmentation order varies per run.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

set_determinism(0)
# Pass to DataLoader: worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(0)
```

**Test 1 — overfit one batch.** The highest-leverage test. If this loss does not approach zero, stop and debug model code or optimization; do not touch data.

```python
def overfit_one_batch(model, batch, loss_fn, steps=300, lr=1e-3, device="cuda"):
    model.train()
    x, y = (t.to(device) for t in batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    first = last = None
    for step in range(steps):
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    print(f"overfit-1-batch: step0={first:.4f}  step{steps}={last:.4f}")
    # Heuristic: a healthy model drives loss well below 1% of the start.
    if last > 0.05 * first:
        print("FAIL -> suspect MODEL CODE (grad flow / shapes) or LR too low")
    else:
        print("PASS -> model+optim can learn; suspect DATA or EVALUATION")
    return first, last
```

**Test 2 — per-layer gradient norms.** Reads the optimization and numerics instruments. Zero norm on a submodule means it is frozen; an exploding norm means divergence is coming.

```python
def log_grad_norms(model, top_k=6):
    rows = []
    for name, p in model.named_parameters():
        if p.grad is None:
            rows.append((name, None))                 # frozen or detached: a red flag
        else:
            rows.append((name, p.grad.detach().norm().item()))
    total = sum(v for _, v in rows if v is not None) ** 0.5
    none_count = sum(1 for _, v in rows if v is None)
    print(f"global grad norm ~= {total:.3e} | params with grad=None: {none_count}")
    # Show the loudest layers; a single 1e4 here is the edge of divergence.
    finite = sorted((r for r in rows if r[1] is not None), key=lambda r: -r[1])
    for name, v in finite[:top_k]:
        print(f"  {v:9.3e}  {name}")
```

**Test 3 — detect anomaly.** The numerics scalpel. It points at the operation whose backward first produced a NaN.

```python
def step_with_anomaly_check(model, batch, loss_fn, opt, device="cuda"):
    x, y = (t.to(device) for t in batch)
    with torch.autograd.set_detect_anomaly(True):   # slow; use only while hunting
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss BEFORE backward: {loss.item()}")
        loss.backward()    # raises and names the op if its backward makes a NaN
    opt.step()
    return loss.item()
```

**Test 4 — single-GPU reproduction.** Not a snippet so much as a procedure: re-launch the exact same training command with one process and the same effective batch size, and watch whether the symptom survives.

```bash
# Multi-GPU run that misbehaves:
torchrun --nproc_per_node=8 train.py --per_device_batch 4 --grad_accum 1

# Single-GPU repro with the SAME effective batch (4*8*1 = 32):
python train.py --per_device_batch 32 --grad_accum 1 --single_gpu
# If the symptom vanishes here, the bug is in the SYSTEMS layer (grad-sync / sharding).
# If it persists, SYSTEMS is cleared; debug the other five places in this simpler setup.
```

These five, run in order, are the bisection procedure made executable. The first makes the rest trustworthy; the next three each cut a different place out of the suspect set; the last isolates the systems layer. Run them before you change a single hyperparameter.

## 9. Two bisections, start to finish

Worked examples are where the method stops being abstract. Here are two complete bisections of different symptoms, each tracing from a dashboard observation to the exact line of code, with the instruments quoted. Figure 4 shows the shape of the first one — a vague NaN turned into a localized, re-measured fix.

![A before-and-after figure showing a training run that went from a vague NaN at step 412 with an exploding gradient norm to a localized log-of-zero bug that, once clamped, runs with a stable gradient norm of two and no NaN](/imgs/blogs/a-taxonomy-of-training-and-finetuning-bugs-4.png)

#### Worked example: hunting a NaN to a single line

Symptom: a segmentation model trains cleanly for 411 steps, then `loss = nan` at step 412 and stays NaN forever. The curve is *smooth* right up to the explosion — no slow drift, just a cliff.

Step 1, classify the symptom. A smooth-then-cliff curve is the signature of **numerics**, not data. Data bugs are present from step 1; a clean run that suddenly NaNs at a specific step is almost always a numerical edge case that a particular batch finally triggered. That observation alone routes us to the numerics place and lets us skip the data and evaluation batteries.

Step 2, fix the seed and reproduce. With `set_determinism(0)` and a fixed data order, the NaN reappears at *exactly* step 412 every run. Good — it is deterministic, so the bug is a property of a specific batch, not a race or a transient.

Step 3, run `detect_anomaly`. Wrapping the step in `torch.autograd.set_detect_anomaly(True)` and re-running to step 412, PyTorch raises and names the op: the backward of a `torch.log` in a custom Dice-plus-cross-entropy loss produced the NaN. That converts "NaN somewhere" into "the `log` in my loss."

Step 4, find why *that* batch. I printed the loss inputs at step 412 and found a class that was *absent* from that particular crop, which made a predicted-probability term go to exactly 0, and $\log 0 = -\infty$; the gradient of $\log x$ is $1/x$, which is $+\infty$ at $x=0$, and $\infty \cdot 0$ in the chain rule is NaN. The mechanism is exact and predicts the signature perfectly: the run is fine until the first batch that drives a probability to a true zero.

Step 5, fix and re-measure. The fix is to clamp before the log: `torch.log(p.clamp_min(1e-7))`. After the fix, the run trains past step 412 to completion; the gradient norm at step 412, which had spiked toward 1e4 in the step before the NaN, now reads a stable 2.0; and the loss continues its smooth descent. Before: `NaN @ step 412`, grad norm exploding. After: clean run, grad norm 2.0. The instrument I used to confirm the fix is the *same* one that revealed the bug — `detect_anomaly` now runs clean through step 412 — which is what makes this a real fix and not a coincidence.

#### Worked example: the 0.97 AUC that was a leak

Symptom: a gradient-boosted model on a churn dataset scores 0.97 ROC-AUC in cross-validation and 0.71 in production. The 0.26-point gap is the whole problem. Figure 8 shows the arc.

![A before-and-after figure showing a model whose validation AUC of point nine seven dropped to a production AUC of point seven one, then after dropping a leaking ID column the honest validation AUC of point seven eight finally matched production](/imgs/blogs/a-taxonomy-of-training-and-finetuning-bugs-8.png)

Step 1, classify. A large offline-to-online gap with a *suspiciously high* offline number is the canonical signature of **data leakage** (or, less often, an evaluation bug). It is not optimization or numerics — those do not produce too-good numbers. So we go straight to the data-leakage battery.

Step 2, the cheapest leak test: rank features by importance and ablate the top one. The top feature by gain was an `account_id`-derived numeric column. Hypothesis: that column is a proxy for the label because IDs were assigned in a way correlated with churn (e.g. churned accounts were exported and re-IDed in a batch). Prediction under the leak hypothesis: dropping that column collapses the CV AUC toward the production number; under a no-leak hypothesis, dropping one feature barely moves a 0.97.

Step 3, run it. Drop the column, refit with the *same* CV, AUC falls from 0.97 to 0.78. A single feature accounting for 0.19 AUC points is not a feature; it is a leak. The 0.78 now sits close to the 0.71 production number (the remaining gap is ordinary covariate shift, a separate and smaller issue).

Step 4, find the deeper cause. The leak was not just that one column — it was that the *split itself* put rows from the same customer on both sides, because customers had multiple rows. The honest fix is `GroupKFold(groups=customer_id)`, which guarantees a customer never appears in both train and validation. After regrouping, CV AUC reads 0.77 and is *stable* to feature ablation. Before: 0.97 CV, 0.71 prod, unexplained. After: 0.77 CV honest, matches prod, and the model's ranking of risky customers finally tracks reality. The instrument that confirmed the fix is the *gap*: a healthy model has CV ≈ prod, and now it does.

#### Worked example: the run that was slower on eight GPUs

Symptom: a vision model trains at 1,100 images/sec on one GPU. Scaled to eight GPUs with `DistributedDataParallel`, it reports 1,250 images/sec total — a 1.1× speedup from 8× the hardware, when you expected something near 7×. No crash, no NaN, loss curve identical. This is a *throughput* symptom, and the suspect is unambiguous: **systems**.

Step 1, classify. A run that is correct but does not scale is a systems-layer performance bug, not a model or optimization bug — the math is right, the *execution* is wasting hardware. So we skip the other five places entirely and reach for the systems battery: the profiler and a utilization read.

Step 2, read the cheapest instrument: GPU utilization per rank. `nvidia-smi` showed every GPU oscillating around 28% utilization. A GPU that is busy 28% of the time is *starved* — it is waiting on something, and in training that something is almost always the input pipeline or a synchronization stall. That single number ruled out "the model is too small to saturate the GPU" (which would show high utilization on a small kernel) and pointed at data feeding or communication.

Step 3, run the PyTorch profiler for 50 steps. The trace showed each rank spending roughly 60% of wall-clock time inside the dataloader's `__next__`, blocked, with the GPU idle. The dataloader was configured with `num_workers=2`, and with eight ranks each spawning two workers on a node with limited CPU, the data pipeline could not keep up — *the GPUs were waiting on the CPU to decode and augment images*. This is the canonical "8× GPUs, same speed" trap: the bottleneck is not the GPUs, so adding GPUs does not help; it just adds more mouths the same CPU pipeline must feed.

Step 4, the what-if stress test. What if the bottleneck were *communication* (gradient all-reduce) rather than data? Then the profiler would show time in NCCL kernels, not in `__next__`, and raising `num_workers` would not help. The profiler showed the time in the dataloader, so the data hypothesis held and the communication hypothesis was cleared — a second test that discriminated between the two systems sub-causes.

Step 5, fix and re-measure. Raise `num_workers` to 8 per rank, enable `pin_memory=True` and `prefetch_factor=4`, and move the heaviest augmentation onto the GPU. Throughput went from 1,250 to 7,400 images/sec — a 6.7× scaling from 8 GPUs, and per-GPU utilization rose from 28% to 91%. Before: 1.1× scaling, 28% util, the dataloader on the critical path. After: 6.7× scaling, 91% util, the GPU on the critical path. The instrument that confirmed the fix is the same one that revealed the bug — GPU utilization — which is the signature of a real fix. At a representative \$2.00 per GPU-hour across eight GPUs, the 6.7× throughput gain turned a run that would have cost roughly \$480 into one costing about \$72 for the same epochs: the bug was not wrong, just expensive, and systems bugs are very often *expensive rather than incorrect*.

These three bisections used different places (numerics, data, systems), different tests (`detect_anomaly`, feature ablation, the profiler), and different instruments (grad norm, the offline-online gap, GPU utilization) — but the *shape* is identical: classify the symptom into a place, run the cheapest discriminating test, find the exact mechanism, fix one thing, and re-measure with the instrument that revealed the bug. That shape is the entire method, and it is the same whether the bug costs you accuracy, a crash, or money.

## 10. Case studies and real signatures

The taxonomy is not invented; it is distilled from bug patterns that recur across the field and the literature. Four well-known signatures, each a real instance of one of the six places, with sources you can check.

**Label errors in benchmark test sets (data).** Northcutt, Athalye, and Mueller (2021), in *"Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks,"* used confident learning (the `cleanlab` method) to estimate label-error rates in the *test sets* of ten widely-used benchmarks, finding an average of roughly 3.3% errors, including about 6% in ImageNet's validation set and over 10% in QuickDraw. The startling consequence they document is that these errors can *reorder model rankings*: a model that looks worse on the noisy labels can be better on corrected ones. The signature is an accuracy ceiling you cannot break through and a confusion structure that mirrors the *mislabeling* pattern rather than genuine model error. The test is to loss-rank or confident-learning-rank your own examples and *look* at the worst ones.

**fp16 gradient underflow and loss scaling (numerics).** Micikevicius et al. (2018), *"Mixed Precision Training,"* documented that half-precision (fp16) has a smallest positive normal value of about $6.1\times10^{-5}$, so gradients smaller than that *underflow to zero* and vanish silently — the run does not crash, it just stops learning in the small-gradient regions. Their fix, loss scaling, multiplies the loss by a large constant before backward (shifting gradients up into the representable range) and divides it back out before the optimizer step. The signature is a run that trains fine in fp32 and *plateaus early* or degrades in fp16 with no NaN. The modern alternative is bf16, which trades mantissa bits for the same exponent range as fp32, eliminating the underflow at the cost of precision.

**Left-padding breaks decoder-only generation (model code / data).** A recurring, well-documented Hugging Face gotcha: decoder-only models (GPT-style) must use *left* padding for batched generation, because generation continues from the *last* token, and right-padding puts pad tokens there, so the model generates from a pad position and produces garbage. The signature is correct single-example generation and broken *batched* generation, or a model that "won't stop." The test is to inspect the `attention_mask` and the position of real tokens in a padded batch. This is a model-code/data-plumbing bug that produces no error, only bad outputs.

**Leakage post-mortems on Kaggle (data / evaluation).** A genre of competition write-up that recurs every season: a feature or a split that leaks the target — a timestamp that encodes the answer, an ID assigned post-label, a row duplicated across folds — produces a leaderboard score far above what is achievable honestly, and collapses on the private test set. The general lesson, consistent with the churn example above, is that any single feature contributing an implausibly large metric jump deserves a leak hypothesis and an ablation test before it is trusted. The cheap confirming test is always the same: ablate the suspect feature, or regroup the split, and watch the too-good number fall toward reality.

What unifies these four is that each is a *different place* (data, numerics, model code, data/eval) with a *distinct, documented signature*, and in each case a cheap test confirms it before any expensive fix. That is the taxonomy doing its job: turning a vague "the model is bad" into "this is a known signature of place X, confirmed by test Y."

There is a fifth pattern worth naming, and it is not a place in the model — it is a place in the *process*. The bugs that survive longest are not the cleverest; they are the ones nobody *looked* for, because the instrument that would have caught them was never logged. The label errors went undetected for years not because confident learning is hard but because nobody ranked the worst-loss examples and *looked* at them. The fp16 underflow plateaus survive because nobody plots the gradient histogram. The LoRA no-op ships because nobody reads the trainable-parameter line that scrolls past at startup. In every case the *test* was cheap and the *signature* was visible — the failure was one of attention, not of difficulty. This is why the most valuable habit in the whole discipline is not knowing more fixes; it is *looking at the instruments you already have, at the step the symptom appeared*, with the prior that the number you most want to believe is the one most likely to be a lie.

## 11. When this is (and isn't) your bug

A taxonomy is as useful for what it *rules out* as for what it points to. Here are decisive calls — situations where a symptom strongly points to one place, so you should *stop* blaming the others.

**If overfit-one-batch passes, stop blaming the model.** A model that drives a single batch's loss to near zero has working forward and backward passes, correct shapes, flowing gradients, and a sane LR. Whatever is wrong now lives in *data* or *evaluation*, not in your architecture. People burn days rewriting model code after overfit-one-batch already cleared it. Don't.

**A smooth-then-NaN curve is numerics, not data.** Data bugs are present from step 1 — a leak, a label bug, a normalization mismatch poisons the whole run from the start. A run that is clean for hundreds of steps and *then* explodes is a numerical edge case triggered by a specific batch. Reach for `detect_anomaly` and the numerics battery, not the data battery.

**A too-good validation number is almost never a good model.** If your validation accuracy or AUC is far above what the problem plausibly allows, the prior should be *leakage* (or eval-set overfitting), not "we built a great model." The honest move is to assume a leak and try to *disprove* it with an ablation or a regrouped split. The cost of a false "great model" is shipping it; the cost of one leak test is two minutes.

**If it only fails on multi-GPU, it is systems — debug single-GPU.** A symptom that vanishes under a single-GPU reproduction is, by definition, in the layer that single-GPU removes: gradient synchronization, accumulation arithmetic, sharding, rank desync. Do not debug your loss function on eight GPUs; reproduce on one, fix it there if it persists, and if it *doesn't* persist, you have already localized it to systems.

**A great offline metric and a bad online one is evaluation or shift, not training.** If the model trained fine and scores well offline but disappoints in production, the training succeeded and the *measurement* or the *deployment distribution* is the suspect. Re-derive the metric by hand, evaluate on a fresh holdout drawn like production, and check train-serve skew before you retrain anything.

**If the finetune loss is healthy but behavior is unchanged, check the wiring before the data.** A LoRA no-op and a chat-template skew both produce a healthy loss and a useless model. Before re-curating the dataset, confirm the trainable-parameter count and inspect the *formatted* training text. The plumbing fails silently far more often than the data is wrong.

The common thread: each of these is a place where a *single cheap observation* licenses you to ignore four of the six suspects. That is the highest-value skill in the whole discipline — not knowing every fix, but knowing which four places you are allowed to stop worrying about.

## 12. Building a debuggable run from day one

Most of this post is about diagnosing a run that is already broken. The cheaper move is to build runs that are *easy to bisect* before they break. Five habits, each of which makes a future bisection faster.

Log the instruments from step 1, not after the crash — grad norm, param/update ratio, LR, throughput, and a small fixed sample of decoded inputs. You cannot inspect the instruments at the step the symptom appeared if you were not recording them. Make the run deterministic enough to reproduce a symptom on demand; a bug you cannot reproduce is a bug you cannot bisect. Keep an *overfit-one-batch* smoke test in CI so a model-code regression fails in two minutes, not in a four-hour run. For finetuning, assert the trainable-parameter count and print a few *formatted* training examples at startup, so a LoRA no-op or a template skew is caught before the first GPU-hour. And keep an *old-task eval* on every finetune so catastrophic forgetting shows up as a number, not as a production surprise weeks later.

These are the same six places, run *forward*: instrument the data (sample it), the model (overfit-one-batch in CI), the optimization (grad-norm and ratio logging), the numerics (a NaN guard on the loss), the systems (a single-GPU smoke test in CI), and the evaluation (a fresh holdout and an old-task eval). A run built this way does not stop having bugs — but every bug it has announces its place quickly, which is the entire goal. The capstone of this series, [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook), assembles these habits into a single printable checklist and a from-day-one system design.

## 13. Key takeaways

- **A training bug hides in exactly one of six places** — data, model code, optimization, numerics, systems, evaluation. Localize the *place* before you touch a fix.
- **Bisect, don't guess.** Run the cheapest test that discriminates between your top two suspects; each good test cuts the suspect space roughly in half, and five tests usually localize the bug.
- **Overfit one batch first.** If it passes, model code and optimization are cleared — suspect data or evaluation. If it fails, the bug is in model code or a too-low LR. This one test forks the entire search.
- **A smooth-then-NaN curve is numerics; a present-from-step-1 problem is data.** The *timing* of the symptom is itself a clue about the place.
- **A too-good validation number is a leak until proven otherwise.** Assume leakage, then try to disprove it with an ablation or a regrouped split.
- **Read the instruments, and know what each one rules out.** Grad norm rules out optimization/numerics; the update-to-param ratio diagnoses the LR; throughput rules systems in or out.
- **If it only fails on multi-GPU, it is systems — reproduce single-GPU.** Never debug your loss function on eight GPUs.
- **Finetuning bugs live in the same six places but bite differently:** wrong LR by 100× (optimization), forgetting (data+optim), template skew (data+eval), LoRA no-op (model code). Prove the new signal enters the graph before you trust the loss.
- **A fix is only a fix if it moves the instrument that revealed the bug.** Re-measure with the same instrument, every time.
- **Build for bisection from day one:** log instruments early, keep the run deterministic, put overfit-one-batch in CI, and assert trainable-parameter counts on every finetune.

## 14. Further reading

- **The capstone:** [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) — the full symptom→suspect→test→fix decision tree, the printable checklist, and how to build a debuggable system from day one.
- **The overfit-one-batch test in depth:** [the overfit a single batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) — what passing and failing rule in and out, and the variants (one feature, one class, one GPU).
- **Data leakage:** [data leakage, the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer) — target leakage, train–test contamination, temporal leakage, and how a leak looks on the dashboard.
- **Learning rate:** [the learning rate is almost always the problem](/blog/machine-learning/debugging-training/the-learning-rate-is-almost-always-the-problem) — LR symptoms, the LR-range test, warmup, and why finetuning wants a far smaller LR.
- **Hunting NaNs:** [hunting NaNs and Infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs) — where non-finite values come from, `detect_anomaly`, and the bisection-by-step hunt.
- Northcutt, Athalye, Mueller (2021), *"Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks"* — confident learning and the `cleanlab` library; benchmark test-set error rates.
- Micikevicius et al. (2018), *"Mixed Precision Training"* — the fp16 representable range, gradient underflow, and loss scaling.
- PyTorch documentation — `torch.autograd.set_detect_anomaly`, `torch.use_deterministic_algorithms`, `torch.nn.utils.clip_grad_norm_`, and the PyTorch profiler.
- Hugging Face documentation — `transformers` `Trainer`, `peft` `LoraConfig` and `print_trainable_parameters`, and the left-padding-for-generation guidance for decoder-only models.

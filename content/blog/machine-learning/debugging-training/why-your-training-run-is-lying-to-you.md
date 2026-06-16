---
title: "Why Your Training Run Is Lying to You: A Field Guide to Silent ML Bugs"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A loss going down is not evidence your system is correct. Learn the scientific method for ML debugging, the six places a bug hides, and the two tools that localize any stalled, NaN, or secretly-overfit run in minutes."
tags:
  [
    "debugging",
    "model-training",
    "deep-learning",
    "pytorch",
    "finetuning",
    "data-leakage",
    "mixed-precision",
    "reproducibility",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/why-your-training-run-is-lying-to-you-1.png"
---

The most dangerous training run is not the one that crashes. A crash is honest: a stack trace, a line number, a thing to fix. The dangerous run is the one that *works*. The loss goes down. The progress bar fills. TensorBoard draws a smooth, satisfying curve. You walk away, come back six hours later, and the model has learned nothing useful, or it has memorized a leaked column, or it quietly forgot every capability it had before you finetuned it, or it is sitting at `NaN` because something overflowed at step 4,000 while you were asleep. The run did not error. It *lied*.

I have lost more weeks to this than to any segfault. The pattern is always the same. A real instrument (the loss curve) reads "everything is fine," and we trust it the way we trust a fuel gauge, when in fact the loss going down tells you almost nothing about whether the *system* is correct. A model can drive its training loss to zero by memorizing a leaked target. A loss can decrease smoothly while a critical submodule receives zero gradient. A validation accuracy of 99% can be a measurement of your data pipeline's incest, not your model's intelligence. The training loss is a necessary condition for a working system, never a sufficient one, and the entire discipline of debugging training is about building the *other* instruments — the ones that read the truth when the loss curve is comfortably lying to your face.

This post is the introduction to a long series on debugging AI training and finetuning. Its job is to hand you the operating system the rest of the series runs on: a scientific method for ML bugs, a map of the six places a bug can hide, the bisection mindset that localizes a bug *before* you touch any code, and the two master tools — **make-it-fail-small** and **read the instruments** — that show up in every later post. By the end you should be able to take any stalled, diverging, secretly-overfit, or silently-broken run and narrow it down to one of six suspects in minutes, then design the single cheapest test that confirms or kills your hypothesis. Figure 1 is the map; everything in the series is a tour of one of its layers.

![A vertical stack of the six layers a training bug can hide in, from data at the top through optimization, model code, numerics, systems, and evaluation at the bottom, each labeled with its canonical silent failure](/imgs/blogs/why-your-training-run-is-lying-to-you-1.png)

## 1. The thesis: a falling loss is not evidence of a correct system

Let me state the central claim as bluntly as I can, because it is the thing most people get wrong, and getting it wrong is what costs the weeks.

**Training loss decreasing is consistent with a correct system. It is also consistent with at least a dozen broken ones.** The loss is a single scalar summarizing the model's fit to *the data you actually fed it, through the pipeline you actually built, under the objective you actually wrote*. If any of those three is wrong — the data is leaked or mislabeled, the pipeline drops the last batch or normalizes twice, the objective sums where it should mean or trains on the prompt — the loss will *still go down*. It will go down enthusiastically, because the optimizer is extremely good at minimizing whatever you actually asked it to minimize, including the wrong thing.

Here is the formal version, and it is worth being precise because the precision is what makes the rest of the series a *science* and not a folk practice. When you read "loss is decreasing," what you are observing is that gradient descent is reducing the empirical risk

$$\hat{R}(\theta) = \frac{1}{n}\sum_{i=1}^{n} \ell\big(f_\theta(x_i),\, y_i\big)$$

on the *training* distribution, through the *implemented* forward function $f_\theta$ and the *implemented* loss $\ell$. What you actually care about is the population risk $R(\theta) = \mathbb{E}_{(x,y)\sim P}\,\ell(f_\theta(x), y)$ on the true data distribution $P$, using the *intended* model and the *intended* objective. The gap between "the loss I am watching go down" and "the thing I actually want" has six independent sources, and every one of them produces a falling loss:

1. The training samples $(x_i, y_i)$ are not drawn from $P$ — they leak the label, they are mislabeled, the split is contaminated. (**Data**.)
2. The optimizer minimizes $\hat{R}$ but lands in a place that does not generalize, or it never moves because the learning rate is wrong. (**Optimization**.)
3. The implemented $f_\theta$ is not the model you think — a layer is frozen, a mask is wrong, a tensor broadcasts on the wrong axis. (**Model code**.)
4. $f_\theta$ and $\ell$ are computed in finite precision and silently underflow, overflow, or produce `NaN`. (**Numerics**.)
5. The computation is sharded across devices and the gradients you average are wrong, or the GPU is idle, or you ran out of memory and silently fell back. (**Systems**.)
6. The number you read off at the end — accuracy, AUC, WER — measures something other than population performance. (**Evaluation**.)

That list is the spine of this entire series. It is the answer to "where could the bug be?" and the answer is always **exactly one of these six**, and the whole game is figuring out *which one* with the fewest, cheapest tests. We will instantiate each layer across dozens of posts, but the frame never changes: a training run is a stack of six layers, a silent failure lives in exactly one of them, and naming the layer is most of the fix.

Why six and not five or seven? Because those are the six places where the chain from raw data to a reported number can break *without raising an exception*. There are plenty of *loud* failures — a shape mismatch that throws, an out-of-bounds index, a CUDA assert — and those are easy; the stack trace points at the line. The hard failures are the ones where every line runs, every tensor has a finite value, the loss is a real number, and the system is nonetheless computing the wrong thing. Those silent failures partition exactly into the six layers: a wrong *input* (data), a wrong *update* (optimization), a wrong *function* (model code), a wrong *number* (numerics), a wrong *machine* (systems), or a wrong *measurement* (evaluation). Every silent training bug I have ever debugged fits one of those six buckets, and the discipline of asking "which bucket?" before "which line?" is the single habit that most separates engineers who debug training in minutes from those who debug it in days.

A quick note on what this series is *not*. It is not a collection of "try lowering your learning rate" tips. Every claim here is meant to be three things at once, and I will hold myself to it in every post:

- **Scientific** — the *why*. The math, numerics, or statistics that makes the bug possible and *predicts its signature*. Why does fp16 underflow gradients below about $6\times10^{-5}$? Why does a leaked feature inflate AUC, and by roughly how much? Why does too-high a learning rate produce a loss spike and then `NaN` rather than a gentle wobble?
- **Practical** — the *how*. A real, runnable diagnostic in the actual tools: a PyTorch hook that prints per-layer gradient norms, a `cleanlab` call that ranks mislabeled rows, a `GroupKFold` that exposes a leak. Code you copy, run, and read.
- **Evidence** — the *proof*. A concrete before→after: AUC 0.97 on a leak and 0.78 honest; grad norm $10^4$ before clipping and 2.0 after; `NaN @ step 412` before and clean to step 50,000 after. We name the symptom, the confirming test, the fix, and what the instruments read afterward.

If a post does not give you all three, it has failed. Hold me to it.

## 2. Four runs that lied, and what they were really doing

Before the method, the motivation. Here are four silent failures I have personally watched eat real time and real GPU budget. Each one had a perfectly reassuring loss curve. Each one was a different layer of the stack. If you have trained models in anger, you will recognize at least two.

**The run that "trained" but learned nothing.** A BERT finetune for a classification task. Training loss dropped from 0.69 (random for two balanced classes is $\ln 2 \approx 0.693$) to 0.31 over three epochs. Looked healthy. Validation accuracy: 50%. Exactly chance. The model had learned to drive *training* loss down by memorizing the training set — but the classification head was the *only* thing receiving gradient, because a `requires_grad = False` left over from a feature-extraction experiment had frozen the entire encoder. The body of the model was a fixed random-ish projection; the head memorized 12,000 training examples and generalized to nothing. The loss curve was a true measurement of a useless system. (This is the **model code** layer; we will gut it in *your-model-isnt-learning-what-you-think*.)

**The 99% validation accuracy that was a data leak.** A churn model on tabular customer data. Cross-validated AUC: 0.97. The team was thrilled. In production: 0.71. The culprit was a column that was a near-perfect proxy for the label — it had been computed *after* the outcome was known and joined back onto the training table. Every fold of the cross-validation saw the leaked column, so every fold was a measurement of "can the model read the answer key," which it can, at 0.97. The honest number, after a `GroupKFold` on customer id and dropping the proxy, was 0.78. The 0.97 was not a model; it was a mirror. (This is the **data** layer, *and* the **evaluation** layer was complicit — see *data-leakage-the-silent-killer*.)

**The finetune that quietly forgot everything.** An instruction-tuned LLM, finetuned on a narrow domain dataset with a learning rate of `2e-4` because that number was in a tutorial. The domain loss went down beautifully. The model got noticeably better at the 800 in-domain examples. It also lost the ability to follow basic instructions it had nailed before finetuning — refusing to format JSON, ignoring system prompts, looping. The instruments that would have caught it (a held-out general-capability eval run *every* checkpoint, not just the domain loss) were never wired up, so the regression was invisible until a human noticed the model had gotten dumber. The learning rate was roughly $100\times$ too high for a finetune; the forgetting is catastrophic interference, and it is *predicted* by the optimization math. (This is the **optimization** layer meeting finetuning; see *finetuning-an-llm-without-breaking-it*.)

**The `NaN` at step 4,000 after a clean start.** A vision transformer pretraining run in fp16. Loss curve: textbook. Smooth descent from 6.9 to 1.4 over 3,900 steps. Then, at step 4,000, `loss = nan`, and every subsequent step `nan`, and the run was dead. Nothing in the *early* curve hinted at it. What actually happened: the gradient norm had been slowly climbing — from 3 at step 500 to 40 at step 2,500 to $7\times10^4$ at step 3,900 — and at step 3,900 a gradient overflowed the fp16 max of 65,504, the loss scaler could not recover, and the next forward pass propagated `inf` into `nan`. The clean early curve was not evidence of a healthy run; it was a fuse burning quietly. (This is the **numerics** layer; we hunt it in *hunting-nans-and-infs* and *mixed-precision-debugging-fp16-vs-bf16*.)

Four runs, four layers, one common feature: **the loss curve looked fine, and the loss curve was lying.** Not because the loss was wrong — it was a faithful measurement — but because it measured the wrong thing, or measured the right thing on the wrong data, or measured a system that was secretly different from the one you designed. The job of a training debugger is to build the *other* instruments, the ones that read the truth, and to know which instrument to read first. That is the rest of this post.

## 3. The scientific method for ML bugs

Most people debug training by *guessing-and-twiddling*: the loss looks bad, so they lower the learning rate, change the batch size, add a layer, switch the optimizer, all at once, and re-run. Six hours later they have a different bad curve and no idea which change did what. This is not debugging; it is a random walk through hyperparameter space, and it is why training feels like alchemy to people who do it this way.

The alternative is the same scientific method you would apply to any complex system, compressed into three questions. I run this loop on every silent failure, and it has never once let me down.

**Question 1: What is the symptom, stated as an observation, not a story?** "The model is bad" is a story. "Training loss plateaus at 6.9 — which is $\ln(\text{vocab size})$, i.e. chance — and never moves over 2,000 steps" is an observation. "Validation AUC is 0.97 but production AUC is 0.71" is an observation. The discipline is to write down what the *instruments* read, in numbers, with no interpretation. Half of all debugging failures are because someone is debugging their *interpretation* of the symptom instead of the symptom.

**Question 2: Which of the six layers is the hypothesis, and what is the single cheapest test that would confirm or kill it?** This is the crux. You do not change code yet. You form *one* falsifiable hypothesis — "I bet the encoder is frozen," "I bet there's a leak," "I bet a gradient is overflowing" — and you ask: what is the *cheapest* observation that distinguishes this world from the world where the hypothesis is false? Cheap means seconds, not a re-run; means a print statement or a one-line check, not a refactor. "Print `sum(p.requires_grad for p in model.encoder.parameters())`" costs nothing and immediately confirms or kills the frozen-encoder hypothesis. The art of debugging is *test design*: finding the observation with the highest ratio of information to cost.

**Question 3: Fix the confirmed cause, then re-measure — and trust nothing until the instrument says so.** Once a test confirms the layer, you make *one* change, and you re-run the *same* measurement. If grad norm was $10^4$ and you added clipping and now it reads 2.0, the instrument confirms the fix. If you "fixed" it but the instrument still reads $10^4$, you did not fix it, regardless of how confident you were. The re-measurement is not optional. It is the difference between "I changed something and the symptom went away" (correlation, possibly coincidence) and "I confirmed the cause, applied the fix, and the instrument now reads healthy" (causation).

The loop is shown in Figure 2 — and notice the most important edge: when a test *kills* the hypothesis, the arrow goes back to *form a new hypothesis*, not *change the code*. A killed hypothesis is progress. It eliminated a layer. The single most common debugging mistake is to change code in response to a *killed* hypothesis ("well that wasn't it, let me just try lowering the LR anyway"), which adds an uncontrolled variable and corrupts every measurement after.

![A directed loop showing the scientific debugging cycle from symptom to one falsifiable hypothesis to the cheapest confirming test, branching to either kill the hypothesis and loop back or apply one fix and re-measure](/imgs/blogs/why-your-training-run-is-lying-to-you-3.png)

There is a deep reason this loop works for ML specifically, and it is worth making explicit. Training systems are **non-deterministic, slow to evaluate, and high-dimensional**. A re-run can cost dollars and hours. The space of possible bugs is enormous. Under those constraints, the *cost* of a test dominates everything, and the only winning strategy is to maximize information per test — which is exactly what a falsifiable hypothesis plus a cheap confirming observation does. Guessing-and-twiddling fails not because the guesses are bad but because each guess costs a full re-run and confounds the next one. The science is not pedantry; it is the only thing that is fast.

## 4. Bisection: localize before you touch code

The six-layer map plus the scientific loop give you a strategy, and that strategy is **bisection**. You do not start by inspecting the most likely layer. You start by asking the *one question that splits the six layers most evenly*, so that a single test eliminates half the search space. This is binary search applied to bugs, and it is the reason a good debugger localizes a fault in two or three tests instead of twenty.

What is the question that bisects best? For training runs it is almost always: **can the model overfit a single batch?** Take one batch — one batch, four to eight examples, labels included — and loop on *only that batch* for a few hundred steps. A correct data-pipeline-plus-model-plus-optimizer can and *will* drive the loss on a single batch to near zero, because there is nothing to generalize to; it just memorizes four examples, which any model with enough parameters does trivially. The result of this one test cleaves the six layers cleanly in two:

- **If it CANNOT overfit one batch** — loss stays flat, will not drop to near zero — the bug is in the part of the system that turns data into a gradient and applies it: the **data pipeline** (the batch is garbage, labels misaligned), the **model code** (a layer is frozen, the graph is detached, a mask is wrong), or the **optimization** (the learning rate is so wrong nothing moves). Generalization, leaks, and evaluation are *ruled out*, because you are not asking the model to generalize.
- **If it CAN overfit one batch** but the full run still fails, the machinery that produces gradients is fine. The bug is in **generalization, data quality, or evaluation**: a leak (the model memorizes a leaked column), label noise (the model memorizes wrong labels), distribution shift (train and eval differ), or a metric bug (the number you read is wrong).

That single test — shown as the first branch in Figure 2's tree — eliminates three layers. The next test bisects the remaining three. If the model cannot overfit one batch, you read the **gradient norm**: if it is exactly 0 (or $\sim 10^{-7}$), the gradient is not flowing — model code or a detached graph; if it is enormous ($10^4$) and the loss is bouncing, the learning rate is too high — optimization. Two tests, and you have localized a bug to one layer without changing a single line of model code.

![A decision tree rooted at a stalling or NaN symptom, branching first on whether the model can overfit one batch, then splitting the no branch into optimization versus model code by gradient norm and the yes branch into data leak versus evaluation by the train-validation gap](/imgs/blogs/why-your-training-run-is-lying-to-you-2.png)

The phrase to internalize is **localize before you touch code**. The instinct under pressure — the run is failing, the deadline is tomorrow — is to *do something*: change a hyperparameter, add a normalization layer, switch optimizers. Resist it. Every uncontrolled change you make *before* you have localized the bug makes the bug harder to find, because now you have two variables moving. The fastest path is the disciplined one: bisect with cheap tests, confirm the layer, *then* and only then change exactly one thing, and re-measure.

#### Worked example: bisecting a BERT finetune stuck at chance

Concretely, here is the four-run BERT example from earlier, debugged by bisection in real time, with numbers.

**Symptom (observation):** Training loss drops from 0.69 to 0.31 over three epochs; validation accuracy is 50.2% — chance, for a balanced two-class problem. Note the symptom is stated in numbers: 0.69 is $\ln 2$, the random-guess loss for two balanced classes, and 50% is chance accuracy.

**Test 1 — overfit one batch.** I grab one batch of 8 examples and loop on it for 300 steps. Result: training loss on that batch drops from 0.69 to **0.004**. The model *can* overfit one batch. That single test eliminates the model-code, pipeline, and optimization layers as causes of *not learning* — the gradient machinery works. The bug is in generalization, data, or evaluation. (Cost: about 20 seconds. Information: three layers eliminated.)

**Test 2 — read the train-validation gap and the gradient destinations.** The model overfits one batch but not the full set, and the train loss goes down while val stays at chance. That is the signature of memorization-without-generalization. The two candidates are: (a) the encoder is not learning, so only the head memorizes and the body is a fixed random projection, or (b) a leak/label problem. I check which parameters have non-zero gradients with one print: `[name for name, p in model.named_parameters() if p.grad is None or p.grad.norm() == 0]`. Result: **every encoder parameter has zero gradient; only the classifier head has a gradient.** Hypothesis confirmed — the encoder is frozen.

**Root cause:** `for p in model.bert.parameters(): p.requires_grad = False` left over from a feature-extraction experiment. The head can memorize 12,000 examples (it overfits one batch *and* the train set), but a fixed random-ish encoder gives the head no generalizable features, so validation is exactly chance.

**Fix and re-measure:** Re-enable encoder gradients. Re-run. Training loss → 0.12, **validation accuracy → 91%**. The instrument (val accuracy) confirms the fix. Total debugging time: under five minutes, three of them spent reading the result of the overfit-one-batch test. The alternative — guessing-and-twiddling the learning rate, batch size, and architecture — would have cost a day and probably never found it, because none of those is the bug.

That is the whole method in miniature: state the symptom in numbers, bisect with the overfit-one-batch test, confirm the layer with a per-parameter gradient check, fix exactly one thing, and re-measure. Now let us make the two tools in that example — make-it-fail-small and read-the-instruments — into something you can apply to any run.

## 5. Master tool #1: make-it-fail-small

The first master tool is to **shrink the problem until the bug has nowhere to hide.** A full training run is a terrible debugging environment: it is slow, stochastic, distributed, and high-dimensional. Almost every bug is *still present* in a drastically smaller version of the run that is fast, near-deterministic, single-device, and low-dimensional — and in that small version the bug is obvious. The skill is knowing which dimension to shrink for which bug.

The canonical instance is **overfit one batch**, which we have already used. Here is the minimal, runnable loop. It is maybe the highest-leverage twenty lines in all of ML engineering, and I run it on every new model before I trust a single full-scale curve.

```python
import torch

def overfit_one_batch(model, batch, optimizer, loss_fn, steps=300, device="cuda"):
    """Loop on ONE fixed batch. A correct pipeline+model+optimizer
    drives loss to near zero. If it can't, the bug is in data,
    model code, or optimization -- not generalization."""
    model.train()
    x = {k: v.to(device) for k, v in batch.items()}  # one fixed batch
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(**x)
        loss = loss_fn(out, x["labels"])
        loss.backward()
        # read the instrument WHILE we overfit
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1e9  # huge: measures, doesn't clip
        )
        optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"step {step:4d}  loss {loss.item():.4f}  grad_norm {total_norm:.3e}")
        history.append(loss.item())
    final = history[-1]
    verdict = "HEALTHY (overfit succeeded)" if final < 0.05 else "BROKEN (cannot overfit)"
    print(f"final loss {final:.4f}  ->  {verdict}")
    return history
```

The output reads like a diagnosis. A healthy run prints a loss marching from 6.9 down through 2.1, 0.4, 0.08, to 0.02 — and the verdict `HEALTHY`. A broken run prints 6.9, 6.8, 6.8, 6.7 — flat — and `BROKEN`, and now you *know* the bug is in data/model/optimization and you reach for the gradient norm to bisect further. Notice we read the gradient norm *inside* the loop (with a huge `max_norm` so `clip_grad_norm_` measures without actually clipping); the two tools are used together. Figure 5 shows the two verdicts side by side.

![A two-column comparison of the overfit-one-batch test showing a broken run whose loss barely moves from 6.9 to 6.7 with a near-zero gradient norm on the left, and a healthy run whose loss collapses from 6.9 to 0.02 with a clean gradient norm on the right](/imgs/blogs/why-your-training-run-is-lying-to-you-5.png)

But "make it fail small" is bigger than overfit-one-batch. The general principle is: **for every dimension of the run, there is a smallest version that still exhibits the bug, and you should debug in that version.** The dimensions, and what shrinking each one isolates:

| Shrink this | To this | Isolates |
| --- | --- | --- |
| Dataset | One batch / one example | Generalization vs. memorization; pipeline vs. model |
| Features | One feature / a few columns | Which input carries the signal (leak detection) |
| Model | One layer / one block | Where the gradient dies or explodes |
| Steps | 10–50 steps | Whether the bug is at init/step-1 or emerges late |
| Devices | One GPU (`CUDA_VISIBLE_DEVICES=0`) | Whether the bug is in the model or in distribution (DDP/FSDP) |
| Precision | fp32 | Whether the bug is numerical (fp16/bf16) or logical |

That last row is the cleanest bisection in the whole table. If a run `NaN`s in fp16 but is *clean in fp32*, the bug is numerical — an underflow or overflow — and you go to the mixed-precision toolkit. If it `NaN`s in *both* fp16 and fp32, the bug is logical — a `log(0)`, a `/0`, a bad label — and precision is a red herring. One re-run in fp32 splits "numerics" from "everything else." Similarly, if an 8-GPU run misbehaves but a 1-GPU run with the same effective batch is clean, the bug is in the systems layer (gradient sync, `find_unused_parameters`, data sharding), not the model. Shrinking the device count from 8 to 1 is a single test that cleaves the systems layer from the rest.

There is a precise reason the overfit-one-batch test is so reliable, and it is worth stating because it tells you exactly what a *failure* of the test means. A modern neural network is *overparameterized* relative to a single batch of four to eight examples — it has far more parameters than there are constraints from four input-output pairs. By the same interpolation argument that explains why deep networks can fit randomly-labeled data, such a network can drive the loss on a handful of fixed examples to essentially zero, *provided the gradient actually reaches its parameters and the optimizer actually steps*. So the test is not asking "is the model good?" — it is asking "is the gradient pathway from loss to parameters intact, and is the optimizer applying it?" A network with billions of parameters that *cannot* memorize four examples is not facing a hard learning problem; it has a broken gradient path. That is why a failed overfit-one-batch test is such strong evidence: it rules out "the problem is just hard" entirely (one batch is never hard) and points squarely at a *mechanical* break — a detached graph, a frozen layer, a zeroed-out loss, a learning rate so small nothing moves, or a label that does not line up with the input. The test converts a vague "it's not learning" into a sharp "the gradient machinery is broken," which is a vastly smaller search space.

The corollary is just as useful: a *passing* overfit-one-batch test is a license to stop suspecting the gradient machinery. I have watched engineers, after confirming a model overfits one batch, *still* spend hours tweaking the optimizer and the architecture when the full run underperforms — when the passing test has already proven those layers innocent. If the model memorizes one batch but fails on the full dataset, the machinery is fine and the bug is in *what you are feeding it* (data quality, a leak, label noise) or *what you are measuring* (the metric, the eval split). Internalize the test's logic and it stops being a ritual and becomes a genuine bisection: pass or fail, it eliminates three of the six layers, and it does so in seconds.

The discipline here is to *resist debugging at full scale*. A full run is the worst possible place to find a bug: slowest feedback, most confounds, most expense. Every minute you spend shrinking the run to where the bug is obvious is repaid many times over in fast iteration. I have watched engineers spend a full day staring at an 8-GPU, fp16, full-dataset run trying to find a `NaN`, when a 1-GPU, fp32, one-batch version would have reproduced it in thirty seconds and pointed straight at the `log` of a negative number.

## 6. Master tool #2: read the instruments

The second master tool is to **stop staring at the loss and start reading the instruments that the loss hides.** The loss is one scalar. A training run emits a dozen signals that are far more diagnostic, and each one *rules out* specific layers when it reads healthy and *points at* specific layers when it reads wrong. A pilot does not fly by looking out the window; a pilot reads the instrument panel. Debugging training is the same. Here are the instruments that matter, what each one measures, and what its abnormal reading tells you.

**Gradient norm** (global and per-layer). The single most informative number after the loss. The global grad norm $\|\nabla_\theta \hat{R}\|_2$ tells you whether the optimizer is getting signal: near zero ($10^{-7}$) means no signal — frozen layers, detached graph, vanished gradients; enormous ($10^4$) and growing means the run is about to diverge — too-high LR, exploding gradients, a bad batch. The *per-layer* grad norm is even better: it tells you *where* in the network the gradient dies or explodes. A clean network has grad norms within an order of magnitude or two across layers; a network where layer 0 reads $10^{-9}$ and layer 23 reads $10^{2}$ has a vanishing-gradient problem you can see at a glance.

**Parameter norm and update norm.** The ratio $\|\Delta\theta\| / \|\theta\|$ — how much each step moves the weights relative to their size — should sit around $10^{-3}$ for a healthy run. If it is $10^{-7}$, your effective learning rate is too small and the model is barely moving (a crawl that looks like a plateau). If it is $10^{-1}$, you are taking enormous steps relative to the weights and you are about to diverge. This ratio is far more robust across architectures than the raw learning rate, because it accounts for the actual scale of the weights.

**Activation statistics.** The mean, standard deviation, and fraction-of-zeros of activations at each layer. A layer whose activations are 95% zeros has dead ReLUs. A layer whose activation std is growing exponentially with depth has an initialization or normalization problem that will explode. A layer whose activations are all saturated at $\pm 1$ (tanh) or 0/1 (sigmoid) has no gradient flowing through it. These are the early-warning signs of the numerics and initialization bugs that show up much later as `NaN`s.

**Learning rate** (the actual scheduled value, logged every step). Sounds trivial, but a shocking number of bugs are "the schedule did something I didn't expect" — a warmup that never ramped, a cosine that hit zero early, a resume that reset the schedule. Log the *actual* LR the optimizer used, not the LR you configured.

**Throughput** (samples/sec, GPU utilization, MFU). A run pinned at 31% GPU utilization is bottlenecked somewhere — usually the dataloader — and is wasting two-thirds of your GPU budget while the loss curve looks perfectly normal. The loss does not tell you the GPU is idle; the utilization counter does.

Here is the minimal instrument you should attach to *every* run from day one — a per-layer gradient-norm printer. It is fifteen lines and it has saved me more debugging time than any tool I own.

```python
import torch

@torch.no_grad()
def grad_norm_report(model, top_k=8):
    """Print per-parameter grad norms, sorted. Run after loss.backward(),
    before optimizer.step(). Reveals: zero-grad (frozen/detached) layers,
    exploding layers, and the overall health of gradient flow."""
    rows = []
    for name, p in model.named_parameters():
        if p.grad is None:
            rows.append((name, float("nan"), "NO GRAD (frozen/detached)"))
        else:
            g = p.grad.norm().item()
            flag = ""
            if g == 0.0:            flag = "ZERO grad"
            elif g < 1e-6:          flag = "vanishing"
            elif g > 1e3:           flag = "EXPLODING"
            rows.append((name, g, flag))
    total = torch.sqrt(sum(
        (p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None
    )).item()
    rows.sort(key=lambda r: (r[1] != r[1], -r[1]))  # NaNs first, then descending
    print(f"GLOBAL grad norm = {total:.3e}")
    for name, g, flag in rows[:top_k]:
        print(f"  {g:>10.3e}  {name:<45} {flag}")
```

Run that once after `loss.backward()` and the diagnosis is usually immediate. Every encoder parameter printing `NO GRAD`? Frozen encoder. One layer printing `EXPLODING` at $10^4$ while the rest are at $10^0$? That layer's gradient is detonating — bad init or a numerical issue there. Everything at $10^{-8}$? Your gradient has vanished and nothing is learning. Figure 6 maps each of the six layers to the canonical bug it hides and the single instrument that exposes it — pin it above your desk.

The gradient norm tells you about the *signal*; the **update-to-parameter ratio** tells you about the *effective step size*, which is the instrument that distinguishes "the learning rate is too small and the model is crawling" from "the gradient is fine but nothing moves." A healthy run keeps $\|\eta g\| / \|\theta\|$ — the size of the weight update relative to the weight itself — in a band around $10^{-3}$ per step. If that ratio is $10^{-7}$, the optimizer is taking microscopic steps and the loss curve will look like a plateau even though the gradient is perfectly healthy; the fix is a larger LR, not a different architecture. If it is $10^{-1}$, the steps are enormous relative to the weights and divergence is imminent. Critically, this ratio is *more portable across architectures than the raw learning rate*, because it normalizes by the actual weight scale — an LR of `1e-4` can be tiny for one layer and huge for another depending on how big its weights are. Here is the logger, run once per optimizer step:

```python
import torch

@torch.no_grad()
def update_param_ratio(model, lr, top_k=6):
    """Report ||lr * grad|| / ||param|| per layer. Run AFTER backward,
    BEFORE optimizer.step(). Healthy ~ 1e-3. Tiny (1e-7) = LR too small
    (crawl that looks like a plateau). Large (1e-1) = about to diverge."""
    rows = []
    for name, p in model.named_parameters():
        if p.grad is None or p.requires_grad is False:
            continue
        update = (lr * p.grad).norm().item()
        weight = p.norm().item() + 1e-12
        ratio = update / weight
        rows.append((name, ratio))
    rows.sort(key=lambda r: -r[1])
    med = sorted(r[1] for r in rows)[len(rows) // 2] if rows else float("nan")
    print(f"median update/param ratio = {med:.2e}   (target ~ 1e-3)")
    for name, r in rows[:top_k]:
        print(f"  {r:.2e}  {name}")
```

If the median prints `4.3e-07`, you have found a too-small-LR crawl in one line, without a single re-run — and you would *never* have seen it in the loss curve, which just shows a slow, ambiguous descent that could equally be a dead network, a hard problem, or a warmup that never finished. That is the entire argument for instruments: each one resolves an ambiguity that the loss alone cannot.

![A six-row matrix mapping each layer of the stack to its canonical silent bug, the cheap instrument that catches it, and the instrument reading when the run is broken, from data leakage caught by a GroupKFold gap down to a lying metric caught by an overfit-batch eval](/imgs/blogs/why-your-training-run-is-lying-to-you-6.png)

The deeper point is that **the loss is a lagging, low-resolution instrument and the others are leading, high-resolution ones.** By the time the loss `NaN`s at step 4,000, the gradient norm had been screaming for 1,500 steps. By the time validation accuracy reveals a frozen encoder, the per-layer grad norm showed zeros at step 1. The instruments tell you what the loss will tell you, but earlier and with a pointer to the cause. Wire them up *before* you need them, because the moment you need them is the moment a 50,000-step run has already wasted six hours.

## 7. A tour of the six layers, one canonical bug each

Now the taxonomy in a little more depth — one canonical silent bug per layer, with its mechanism, so you can recognize the *signature* of each layer. This is the preview; the per-layer tracks of the series are the full treatment. Figure 6 above is the compressed version; here is the prose.

### Data: the leak

The canonical data bug is **target leakage** — a feature that encodes the answer. The mechanism is statistical and quantifiable. Suppose a feature $z$ is a noisy copy of the label: $z = y + \varepsilon$ with small noise. The mutual information $I(z; y)$ is near its maximum, so a model that uses $z$ achieves near-perfect *training and validation* fit — as long as the validation set *also* contains the leaked $z$, which it does if the leak is in the column rather than in the split. The signature: validation metrics far better than any reasonable baseline, a large gap to production, and a feature-importance chart where one column dwarfs the rest. The confirming test is a `GroupKFold` or a temporal split that breaks the leak's path into validation. We cover this in *data-leakage-the-silent-killer*; the leaked-AUC arithmetic is Figure 4 and the worked example in Section 9.

### Optimization: the learning rate

The canonical optimization bug is **the learning rate, wrong by an order of magnitude.** Too high: the first-order Taylor approximation that gradient descent relies on — $\hat{R}(\theta - \eta g) \approx \hat{R}(\theta) - \eta \|g\|^2$ — breaks down because the step $\eta g$ leaves the region where the approximation holds, the loss *increases*, the next gradient is larger, and you get the spike-then-`NaN` we saw at step 4,000. Too low: the update-to-parameter ratio sits at $10^{-7}$ and the model crawls, which on a loss curve is indistinguishable from a plateau caused by a dead network. The signature distinguishes them: too-high LR shows a *growing* gradient norm; too-low shows a tiny *update* norm with a healthy gradient. For finetuning, "too high" is usually $10\times$ to $100\times$ over the right value, and the symptom is catastrophic forgetting rather than divergence. See *the-learning-rate-is-almost-always-the-problem*.

It is worth deriving *why* a too-high finetuning learning rate causes forgetting rather than a crash, because it explains why the finetune-that-forgot run looked perfectly healthy on its domain loss. A pretrained model sits at a parameter vector $\theta_0$ that is good at a broad set of tasks — call its general-task loss $L_{\text{gen}}(\theta)$, minimized near $\theta_0$. Finetuning minimizes a *narrow* loss $L_{\text{dom}}(\theta)$ on the domain data, and a single SGD step moves $\theta_0$ by $-\eta\,\nabla L_{\text{dom}}$. The *collateral* damage to the general capability is, to first order, $\Delta L_{\text{gen}} \approx \eta\,\langle \nabla L_{\text{gen}},\, \nabla L_{\text{dom}}\rangle$ — the change in the general loss is proportional to the learning rate times the *inner product* of the two gradients. When the domain and general gradients point in conflicting directions (which they often do — the domain wants to specialize, which means un-learning generic behaviors), that inner product is negative-helpful but the *magnitude* of the displacement grows linearly with $\eta$. Make $\eta$ a hundred times larger and you move a hundred times further from $\theta_0$ per step, falling out of the basin where $L_{\text{gen}}$ was low — and crucially, the domain loss $L_{\text{dom}}$ keeps *decreasing the whole time*, because you are moving toward *its* minimum. That is the trap: the instrument you are watching (domain loss) reads "great," while the instrument you are not watching (a held-out general-capability eval) reads "catastrophe." The right finetuning LR ($10^{-5}$ to $2\times10^{-5}$ for most LLM finetunes) keeps each step small enough to specialize without leaving the pretrained basin, and the only way to *see* the forgetting is to evaluate the general capability every checkpoint — which the failing run never did. The science predicts the cure: small steps, few epochs, and a regression eval on the panel of capabilities you cannot afford to lose.

### Model code: the frozen layer and the wrong mask

The canonical model-code bug is **a gradient that does not reach where you think it does** — a layer frozen by a stray `requires_grad = False`, a graph detached by an accidental `.detach()` or `.item()`, an in-place operation that breaks autograd, or a mask applied on the wrong axis. The signature is the BERT example: the model overfits one batch (the parts that *do* get gradient memorize), but a submodule shows zero per-layer grad norm, and the model fails to generalize because the frozen part contributes no learned features. The confirming test is the per-layer grad-norm report. A close cousin is **silent broadcasting**: a tensor of shape `(B, 1)` and one of shape `(1, T)` broadcast to `(B, T)` without error, averaging over the wrong axis, producing a *plausible-looking* but wrong loss. See *shape-bugs-and-silent-broadcasting* and *your-model-isnt-learning-what-you-think*.

### Numerics: the fp16 underflow

The canonical numerics bug is **fp16 underflow and overflow.** The science is exact and worth knowing cold, because it predicts the bug's signature precisely. Half precision (fp16) has a 5-bit exponent and a 10-bit mantissa. Its smallest *normal* positive value is

$$\text{fp16 min normal} = 2^{-14} \approx 6.10\times10^{-5},$$

and its maximum is $65{,}504$. Gradients in a deep network are routinely smaller than $6\times10^{-5}$ — a gradient of $3\times10^{-6}$ does not get small in fp16, it becomes **exactly zero** (it underflows to a denormal and then to zero), and that parameter stops learning silently. At the other end, a gradient spike past $65{,}504$ becomes `inf`, and `inf - inf` or `inf * 0` becomes `nan`, which then poisons every downstream computation. This is *why* the step-4,000 `NaN` happened, and *why* loss scaling exists: multiply the loss by a large constant $S$ before backprop so the gradients land inside fp16's representable range, then divide them by $S$ before the optimizer step. bf16, with its 8-bit exponent (the same range as fp32) but only 7-bit mantissa, sidesteps underflow/overflow at the cost of precision, which is why modern large-model training largely moved to it. We derive all of this in *mixed-precision-debugging-fp16-vs-bf16* and *hunting-nans-and-infs*.

### Systems: the idle GPU and the desynced gradient

The canonical systems bug is **wasted hardware that the loss curve cannot see.** Two flavors. First, the *idle GPU*: a run pinned at 31% utilization because the dataloader cannot feed the GPU fast enough — the loss curve is perfectly normal, but you are paying for three GPUs and using one. Second, the *desynced gradient* in distributed training: in `DistributedDataParallel`, if a parameter does not receive gradient on every rank (a conditional branch, an unused head), the all-reduce hangs or you set `find_unused_parameters=True` and silently train on inconsistent gradients across ranks. The signature of the first is low utilization with normal loss; of the second, a run that is correct on 1 GPU and subtly wrong (or hangs) on 8. The instruments are the profiler/utilization counter and the 1-GPU bisection. See *the-gpu-is-idle-throughput-debugging* and *debugging-ddp-and-multi-gpu*.

### Evaluation: the metric that lies

The canonical evaluation bug is **a metric that measures something other than population performance.** Micro-averaging an imbalanced problem so accuracy reports the majority-class rate; computing AUC on a validation set that shares the leak; forgetting to call `model.eval()` so dropout and BatchNorm run in training mode during evaluation; a metric computed on a different normalization than production. The signature is a number that is too good *and* does not track the actual goal — train accuracy 0.99, eval accuracy 0.50, or a WER that looks great until you realize it is computed after a text normalization that production does not apply. The confirming test is to run the *exact* eval the overfit-one-batch test would predict, and to check that train and eval differ only by generalization, not by a code path. See *your-metric-is-lying*.

Six layers, six canonical bugs, six signatures. Internalize the *signatures* — the characteristic instrument readings — and you can often skip straight to the right layer instead of bisecting from the top. But when in doubt, bisect: the overfit-one-batch test never lies.

## 8. The lying-metric worked example, in full

Let me work the leaked-AUC example end to end, because it is the purest illustration of "the metric is lying" and because the arithmetic of *how much* a leak inflates a metric is the kind of science this series insists on. Figure 4 is the before→after.

![A two-column before-and-after showing a leaked id-proxy feature inflating validation AUC to 0.97 against a true 0.78, with the honest leak-free GroupKFold split on the right matching production](/imgs/blogs/why-your-training-run-is-lying-to-you-4.png)

#### Worked example: an AUC of 0.97 that was really 0.78

**Setup.** A binary churn classifier on tabular customer data, 200,000 rows, roughly balanced. The team reports a 5-fold cross-validated AUC of **0.97** and prepares to ship. In a backtest against the next month's real outcomes, it scores **0.71**. That 0.26-point gap between validation and production is the symptom, stated in numbers — and a gap that large is *never* "the model degraded slightly." It is almost always a leak.

**The science of why a leak inflates AUC.** AUC is the probability that a randomly chosen positive is scored higher than a randomly chosen negative. A leaked feature $z$ that is a near-deterministic function of the label — say it was computed in the data warehouse *after* the churn event and joined back — lets the model rank almost every positive above almost every negative, because it is effectively reading the label. If the leak perfectly determined the label, AUC would be 1.0. With a little noise in the leak, you get 0.97. The honest signal in the *legitimate* features supports an AUC of about 0.78. The leak adds roughly $0.97 - 0.78 = 0.19$ points of pure fiction. Crucially, the leak inflates *validation* AUC too, because the leaked column is present in every fold — cross-validation does not protect you from a leak that lives in the *column* rather than in the *split*. This is why "but I cross-validated" is no defense.

**The confirming test.** Two cheap checks. First, a feature-importance / permutation-importance ranking: one column dwarfs all others, contributing the overwhelming majority of the model's gain. That is the leak's fingerprint. Second — the decisive one — re-do the cross-validation with `GroupKFold` on customer id (so no customer appears in both train and validation) *and* drop the suspect column, then compare. Here is the test:

```python
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# X: features, y: label, groups: customer_id
# Build the pipeline INSIDE cross_val_score so the scaler is fit on
# train folds only (fitting it on all of X is itself a leak -- see H1).
pipe = Pipeline([("scale", StandardScaler()),
                 ("clf", LogisticRegression(max_iter=1000))])

gkf = GroupKFold(n_splits=5)

# 1) WITH the suspect column -> the inflated number
auc_leaky = cross_val_score(pipe, X, y, groups=groups,
                            cv=gkf, scoring="roc_auc")
# 2) WITHOUT the suspect column -> the honest number
X_clean = X.drop(columns=["account_status_post"])  # the leaked proxy
auc_clean = cross_val_score(pipe, X_clean, y, groups=groups,
                            cv=gkf, scoring="roc_auc")

print(f"leaky  AUC: {auc_leaky.mean():.3f} +/- {auc_leaky.std():.3f}")
print(f"honest AUC: {auc_clean.mean():.3f} +/- {auc_clean.std():.3f}")
print(f"the leak was worth {auc_leaky.mean() - auc_clean.mean():.3f} fake AUC points")
```

**Result.** `leaky AUC: 0.969`, `honest AUC: 0.781`, and the print says the leak was worth `0.188` fake points. The honest 0.78 *matches* the production 0.71 far better than the fantasy 0.97 ever did (the residual 0.07 is genuine train-serving drift, a separate and smaller problem). The 0.97 was never a model; it was a measurement of the pipeline reading the answer key.

**The fix and what it costs honestly.** Drop the leaked column, switch to `GroupKFold` (and, for time-ordered data, a temporal split so no future leaks into the past), and re-measure. The new baseline is 0.78. That number is *lower* and the team is *unhappier*, and that is exactly the point: 0.78 is real and improvable, while 0.97 was a trap that would have shipped a 0.71 model with 0.97 expectations and blown up the first week in production. An honest 0.78 beats a fictional 0.97 every single time. The job of evaluation is not to produce a big number; it is to produce a *true* number.

This is the evaluation-and-data layer of the stack, and it is the one that costs the most when it lies, because the lie survives all the way to production before anyone notices.

## 9. The NaN-at-step-4000 worked example

The other archetype is the run that looks perfect and then detonates. Figure 7 is its anatomy.

![A left-to-right timeline of a fp16 training run that looks clean through step 2500 while the gradient norm secretly climbs, overflows fp16 at step 3900, produces NaN at step 4000, and is fixed with gradient clipping and bf16 afterward](/imgs/blogs/why-your-training-run-is-lying-to-you-7.png)

#### Worked example: a clean curve that hid a burning fuse

**Symptom.** A vision-transformer pretraining run in fp16. Loss descends smoothly: 6.9 → 2.1 (step 500) → 1.4 (step 2,500). At step 4,000: `loss = nan`. Every step after is `nan`. The run is dead and six GPU-hours are gone. The early curve gave *no* warning — which is exactly why people trust the loss curve and exactly why they should not.

**The bisection.** First test: does it `NaN` in fp32 too? I re-run 50 steps in fp32 with `torch.autograd.set_detect_anomaly(True)`. It does *not* `NaN` in fp32. That single re-run localizes the bug to the **numerics** layer — it is precision-dependent, so it is an underflow/overflow, not a `log(0)` or bad label. (If it had `NaN`ed in fp32 too, I would be hunting a logical bug — a negative under a `sqrt`, a zero in a `log`, a corrupt label — a completely different search.)

**Reading the instrument that the loss hid.** I had been logging the gradient norm every step (because of course I was — Section 6). Reading it back: grad norm was 3 at step 500, 40 at step 2,500, and $7\times10^4$ at step 3,900. The fp16 maximum is 65,504. At step 3,900 a gradient crossed it, became `inf`, the loss scaler's `inf`-check kicked in and skipped a few steps, but the underlying gradient growth never stopped, and at step 4,000 the `inf` propagated into the optimizer state and everything became `nan`. The `detect_anomaly` traceback pointed at the exact operation where the first `inf` was produced. The science from Section 7 *predicted* this signature: a slowly-growing gradient norm in fp16 will eventually cross 65,504 and detonate, and the loss curve will look clean right up until it does.

```python
import torch
# Catch the FIRST nan/inf at its source, with a traceback to the op.
torch.autograd.set_detect_anomaly(True)

for step, batch in enumerate(loader):
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = model(**batch).loss
    scaler.scale(loss).backward()
    # unscale BEFORE reading/clipping so the norm is the true gradient norm
    scaler.unscale_(optimizer)
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if not torch.isfinite(gnorm):
        print(f"step {step}: non-finite grad norm {gnorm} -- numerics layer")
    scaler.step(optimizer)
    scaler.update()
```

**The fix and the before→after.** Two changes, both justified by the science: add gradient clipping at `max_norm=1.0` (caps the gradient so it can never approach 65,504) and switch from fp16 to **bf16** (8-bit exponent = fp32's range, so a gradient of $7\times10^4$ is representable and never overflows). Re-run. The gradient norm now reads 0.9–1.1 every step (clipped), the loss descends past step 4,000 and stays finite **clean to step 50,000**. The instrument confirms the fix: where it read $7\times10^4$ and then `nan`, it now reads $\sim1.0$ and finite. The before→after is unambiguous and *measured*, not asserted.

The lesson generalizes: **a smooth-curve-then-`NaN` is almost always numerics, and the fp32 re-run plus the gradient-norm log localize it in two cheap tests.** And the meta-lesson, again: the clean early curve was not evidence of a healthy run. It was a fuse you would have seen burning if you had been reading the gradient norm instead of the loss.

## 10. Diagnostic tables: from symptom to fix

Two tables I keep within reach. The first maps a symptom to its most likely suspect, the confirming test, and the fix — it is the compressed decision tree, and Figure 8 renders it. The second is a quick smell-test reference for the lying-metric class of bug.

![A five-row matrix mapping common symptoms such as loss flat at chance and NaN after a clean start to their likely suspect layer, the single confirming test, and the fix](/imgs/blogs/why-your-training-run-is-lying-to-you-8.png)

| Symptom (observation) | Most likely suspect | Confirming test (cheap) | Fix |
| --- | --- | --- | --- |
| Loss flat at $\ln(\text{classes})$ from step 1 | model code / data pipeline | overfit one batch | re-enable grads / fix collate / check labels |
| Loss decreases, val stuck at chance | model code (frozen layer) | per-layer grad-norm report | un-freeze; confirm encoder grads non-zero |
| Val AUC ≫ production AUC | data leak + eval | GroupKFold + drop suspect col | remove leak; honest split |
| Smooth curve, then `NaN` at step N | numerics (fp16) | re-run in fp32; read grad norm | clip + bf16; loss scaling |
| Loss spikes, recovers, spikes, diverges | optimization (LR too high) | grad-norm trend; halve LR | lower LR; warmup; clip |
| Train great, eval garbage | eval (forgot `.eval()`) / overfit | diff train vs eval code path | `model.eval()`; check dropout/BN |
| 8 GPUs, same speed as 1 | systems (dataloader / sync) | profiler; 1-GPU bisection | more workers; fix DDP unused params |
| GPU util ~30%, loss normal | systems (input bottleneck) | profiler trace; util counter | prefetch, more workers, pin memory |

And the leak smell-test, a thirty-second sanity check before you trust any too-good metric:

| Smell | What it suggests | Quick check |
| --- | --- | --- |
| One feature dominates importance | leaked proxy | permutation importance; drop it and re-score |
| Val metric far above any baseline | leak or eval bug | compare to a trivial baseline (majority class) |
| Near-duplicate rows across train/val | split contamination | hash rows; check overlap |
| Metric computed after normalization prod lacks | eval-serving skew | reproduce production's exact metric path |
| Random `GroupKFold` ≫ grouped `GroupKFold` | group leakage | re-split by the natural group (user, time) |

These are not exhaustive — the per-layer tracks of the series expand each row into a full post — but they are the eighty-percent table, the one I actually use under deadline pressure. Print it, pin it, and add to it as your own war stories accumulate.

## 11. When this is (and isn't) your bug

A field guide is as much about *ruling out* as ruling in. Here is the decisive section: when a symptom points one way, and when it points elsewhere. The discipline is to read the symptom's *signature* and not jump to the most familiar cause.

**If overfit-one-batch passes, stop blaming the model and the optimizer.** The most common waste of time is debugging the architecture or the learning rate when the model can already drive a single batch to near-zero loss. A passing overfit-one-batch test *proves* the data-pipeline-to-gradient-to-update path works. If the full run still fails, the bug is in generalization, data quality, or evaluation — go there, and leave the model alone.

**A smooth-then-`NaN` curve is numerics, not data.** If the loss descends cleanly for thousands of steps and then `NaN`s, do not start auditing your labels. A data problem (a bad label, a corrupt example) tends to `NaN` *early and reproducibly* — at the step that bad example appears — and to `NaN` in fp32 as well as fp16. A late, smooth-then-`NaN` that *only* happens in fp16 is an overflow, full stop. The fp32 re-run settles it in one test.

**A val-much-better-than-prod gap is a leak or a shift, not a small model imperfection.** If your validation number is wildly better than production, do not reach for "the model needs more capacity." A 0.2-AUC gap is not a capacity problem; it is a leak (validation shares information with training) or a distribution shift (production differs from validation). Capacity bugs produce *small* gaps. Large gaps are structural.

**A run that's correct on 1 GPU and wrong on 8 is systems, not model.** If your single-GPU run is clean but the multi-GPU run misbehaves, the model is innocent. The bug is in gradient synchronization, data sharding, per-rank seeding, or BatchNorm-across-ranks. Do not refactor the model; debug the distribution layer.

**A loss that never moves at all is rarely "needs more epochs."** Flat-from-step-1 is a *connectivity* problem — no gradient is reaching the parameters — not a *patience* problem. Reading the grad norm settles it instantly: zero grad means a broken graph or a frozen layer; healthy grad with no loss movement means the learning rate is far too small. Either way, "train longer" is the wrong answer.

The unifying rule: **match the symptom's signature to the layer, and confirm with the cheapest test, before you touch code.** The taxonomy post, *a-taxonomy-of-training-and-finetuning-bugs*, is the full decision tree that formalizes this matching; the capstone, *the-training-debugging-playbook*, turns it into a printable checklist. This intro hands you the frame; those two posts are the reference.

## 12. Case studies: real, named signatures

These are well-documented patterns from the literature and from widely-reported practice. I cite them so the numbers are checkable, and where I am giving an order of magnitude rather than an exact figure I say so.

**Confident learning finds label errors in benchmark test sets.** The `cleanlab` line of work (Northcutt, Jiang, and Chuang, "Confident Learning: Estimating Uncertainty in Dataset Labels," JAIR 2021, and the companion "Pervasive Label Errors in Test Sets" at NeurIPS 2021) estimated label errors across ten of the most-cited ML benchmarks — including ImageNet, MNIST, and CIFAR — and found a meaningful error rate in the *test* sets, on the order of a few percent (their headline figure averaged around 3.3% across the datasets they studied). The practical consequence is sharp: when label noise is on the order of a few percent, the *test* metric is itself noisy by a few percent, so two models within a point of each other may be ranked by mislabeled examples rather than by capability. This is the data layer corrupting the evaluation layer — the canonical "garbage in" bug, and the reason *garbage-in-finding-label-noise* exists.

**Loss scaling from the Mixed Precision Training paper.** Micikevicius et al., "Mixed Precision Training" (ICLR 2018), documented exactly the fp16 underflow we derived in Section 7: a large fraction of gradient values in their networks fell below the fp16 minimum normal value ($\approx 6\times10^{-5}$) and underflowed to zero, degrading accuracy. Their fix — multiply the loss by a scale factor $S$ before backprop so the gradients shift into fp16's representable range, then unscale before the optimizer step — is the loss scaling that lives in every AMP `GradScaler` today. The paper is the *why* behind the step-4,000 `NaN` and the reason bf16, with fp32's exponent range, sidesteps the problem entirely.

**The left-padding-breaks-generation bug for decoder-only LLMs.** A widely-reproduced and well-documented finetuning-and-inference trap: decoder-only models (GPT-style) must be **left-padded** for batched generation, because right-padding places pad tokens *between* the prompt and the first generated token, and the position ids and causal mask then attend over pad positions, corrupting generation — while the *training* loss, computed with the pad tokens masked out via `-100`, looks perfectly fine. The signature is the cruelest in the series: train loss is healthy, single-example generation is healthy, and *batched* generation produces garbage. The Hugging Face `transformers` docs call this out explicitly and default tokenizers to a `padding_side` that depends on the model family. We dissect it in *attention-mask-and-padding-bugs-for-llms*.

**CTC's input-shorter-than-target `inf` loss.** From Graves et al., "Connectionist Temporal Classification" (ICML 2006): CTC sums over all alignments of an input sequence of length $T$ to a target of length $U$, and a valid alignment requires roughly $T \ge U + (\text{repeats})$. When a downsampling front-end shrinks $T$ below the target length $U$ — easy to do silently with an aggressive stride — *no* valid alignment exists, the alignment probability is zero, and the loss is $-\log 0 = +\infty$. The signature is `inf` loss on *specific* (long-transcript, short-audio) batches, not all of them. This is the numerics layer meeting the data layer in speech, and it is *predictable* from the length constraint; we cover it in *debugging-ctc-and-alignment*.

The thread through all four: each bug has a *mechanism* that *predicts a signature*, and once you know the signature you recognize the bug on sight. That is the whole purpose of the science in this series — not to admire the math, but to turn each bug class into a recognizable fingerprint.

## 13. The shape of the series

A quick map of where we go from here, so you know what each later post buys you. The series is organized as **cross-cutting tracks first, then per-modality tracks**, because the cross-cutting bugs (data, optimization, numerics, model code, systems, evaluation) recur in *every* modality, and the modality tracks (vision, LLM/NLP, speech, tabular) instantiate them with domain-specific traps.

- **Foundations** (this track): the mindset, the overfit-one-batch test, reading the loss curve, what to log, reproducibility and determinism, and the master taxonomy/decision tree. You are reading the intro; the rest of the track makes each master tool concrete.
- **Data debugging**: label noise, leakage, the input pipeline, class imbalance, distribution shift, looking at your data, and augmentation gone wrong. The number-one source of silent bugs.
- **Optimization and numerics**: the learning rate, exploding/vanishing gradients, hunting `NaN`s, mixed precision, loss spikes, dead neurons, and initialization/normalization.
- **Model and code-level**: shape and broadcasting bugs, gradient-flow bugs, train/eval mode, attention/masking, and loss-function bugs.
- **Per-modality**: computer vision (input pipelines, detection, segmentation, ViT finetuning), LLM/NLP (the deepest track — tokenization, padding/masking, loss masking, finetuning without forgetting, LoRA/PEFT, chat templates, RLHF/DPO, train-infer mismatch), speech (audio inputs, CTC, ASR finetuning, streaming), and tabular (leakage, cross-validation, feature bugs, gradient boosting).
- **Distributed and systems**: DDP, gradient accumulation, OOM, FSDP, throughput.
- **Evaluation, monitoring, and finetuning, then the capstone**: lying metrics, validation-set overfitting, when to kill a run, checkpoint/resume, cross-modal finetuning pitfalls, and the playbook that ties it all together.

Every one of those posts is the same three things: the science of why the bug is possible, the diagnostic code that confirms it, and the before→after that proves the fix. And every one of them ties back to this map — six places, two tools, bisect before you touch code. If you forget everything else, remember those three.

A word on *order of reading*, since fifty-eight posts is a lot. If you have a fire to put out right now, jump straight to the layer your symptom points at — the diagnostic tables in Section 10 will route you — and come back for the foundations later. If you are reading to build the skill rather than fix a specific run, read Foundations in order, because each tool there compounds: the overfit-one-batch test only makes sense once you have the six-layer map, the loss-curve field guide only makes sense once you know what the gradient norm is telling you, and the taxonomy decision tree only makes sense once you have run the scientific loop a few times by hand. The per-modality tracks are then a matter of taste and need — read the LLM/NLP track if you finetune language models, the vision track if you train classifiers and detectors, and so on; each one assumes the cross-cutting tracks but re-derives the domain-specific traps. The capstone playbook is deliberately last, because it is a *compression* of everything before it into a checklist, and a checklist is only useful once you understand what each line is checking *for*. The goal of the whole series is not to make you memorize fixes — fixes change with every framework version — but to make the *method* automatic, so that the next time a run lies to you, the question "which of the six layers, and what is the cheapest test?" is the first thing that comes to mind, not the last.

## 14. Key takeaways

- **A falling loss is a necessary, not sufficient, condition for a correct system.** The optimizer minimizes whatever you actually asked it to minimize, including a leaked target, a memorized training set, or the wrong objective. Trust the *other* instruments.
- **A bug hides in exactly one of six places: data, optimization, model code, numerics, systems, or evaluation.** Naming the layer is most of the fix. Memorize the six.
- **Run the scientific loop, not guess-and-twiddle.** State the symptom in numbers, form one falsifiable hypothesis, run the *cheapest* confirming test, fix one thing, re-measure. A killed hypothesis is progress; never change code in response to one.
- **Bisect before you touch code.** The overfit-one-batch test splits the six layers in half: if it fails, the bug is in data/model/optimization; if it passes, it is in generalization/data/eval. Two more tests localize it.
- **Make it fail small.** Shrink the dataset to one batch, the model to one layer, the precision to fp32, the devices to one GPU — each shrink isolates a layer. fp32-clean-but-fp16-`NaN` is numerics; 1-GPU-clean-but-8-GPU-wrong is systems.
- **Read the instruments the loss hides.** Gradient norm (global and per-layer), update-to-parameter ratio, activation stats, the actual LR, and GPU utilization each rule out specific layers when healthy and point at specific layers when wrong — and they read the truth *earlier* than the loss does.
- **Match the signature to the layer.** Smooth-then-`NaN` is numerics; val-≫-prod is a leak or shift; flat-from-step-1 is connectivity, not patience; 1-GPU-vs-8-GPU divergence is systems. Confirm with the cheapest test before touching code.
- **An honest 0.78 beats a fictional 0.97 every time.** A metric's job is a *true* number, not a big one. A 0.2-point val-to-prod gap is a leak, not a capacity problem — break the leak's path with `GroupKFold` or a temporal split and re-measure.
- **Wire up your instruments before you need them.** The gradient-norm logger and the per-layer report cost fifteen lines and save the six-hour run that `NaN`s at step 4,000. By the time the loss tells you, the gradient norm already told you 1,500 steps ago.

## 15. Further reading

- Micikevicius et al., "Mixed Precision Training," ICLR 2018 — the source of loss scaling and the fp16 underflow analysis behind the step-4,000 `NaN`.
- Northcutt, Jiang, and Chuang, "Confident Learning: Estimating Uncertainty in Dataset Labels," JAIR 2021, and "Pervasive Label Errors in Test Sets Destabilize ML Benchmarks," NeurIPS 2021 — label noise in benchmark test sets, and the `cleanlab` library.
- Graves, Fernández, Gomez, and Schmidhuber, "Connectionist Temporal Classification," ICML 2006 — CTC mechanics and the length constraint that produces `inf` loss.
- Andrej Karpathy, "A Recipe for Training Neural Networks" (2019) — the overfit-one-batch discipline and the "become one with the data" mindset, in the same spirit as this series.
- PyTorch documentation: `torch.autograd.set_detect_anomaly`, `torch.nn.utils.clip_grad_norm_`, `torch.amp` and `GradScaler`, and `torch.use_deterministic_algorithms` — the actual APIs behind the diagnostics here.
- Hugging Face `transformers` documentation on padding side and batched generation for decoder-only models — the left-padding-breaks-generation bug.
- Within this series: the master decision tree in [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), and the printable reference in [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook).
- Also in Foundations: [the overfit a single batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test), [reading the loss curve as a diagnostic](/blog/machine-learning/debugging-training/reading-the-loss-curve-as-a-diagnostic), [instrumenting a training run, what to log](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log), and [reproducibility and determinism in training](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training).

---
title: "Overfitting to the Validation Set: The Slow Test-Set Creep"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "You can overfit a held-out set you never trained on, just by tuning against it hundreds of times. Learn to measure the test-set creep, derive the multiple-comparisons inflation, and report a number that survives production."
tags:
  [
    "debugging",
    "model-training",
    "validation",
    "overfitting",
    "model-selection",
    "cross-validation",
    "statistics",
    "finetuning",
    "deep-learning",
    "scikit-learn",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/overfitting-to-the-validation-set-1.png"
---

Here is a failure mode that does not look like a bug, leaves no stack trace, and survives every code review: a model that quietly memorizes a set it was never trained on. You did everything right. You split the data into train and validation. You never let a validation row touch a gradient. You tuned the learning rate, the architecture, the regularization, the feature set, the threshold — all against that clean, held-out validation score, watching it climb from 0.85 to 0.88 to 0.91 over a couple of weeks. Then the model shipped, and production came back at 0.83. The same number you would have gotten if you had stopped two weeks earlier and saved yourself the effort.

This is **overfitting to the validation set**, and it is one of the most common ways a training run lies to you. The validation set was supposed to be your honest mirror. But a mirror you consult two hundred times, adjusting your appearance after each glance until it shows you what you want, is no longer honest. Every decision you made by *looking at the validation score* — keep this layer, drop that feature, nudge the LR, pick checkpoint 4 over checkpoint 7 — leaked one bit of information about that specific set into your model. Individually, harmless. Cumulatively, the score is optimistic and the real number is lower. The held-out set has been overfit without a single one of its labels ever entering the loss.

![A vertical stack showing the reported validation score of 0.91 above two hundred evaluations, then a fresh holdout at 0.84, production at 0.83, and the resulting gap of 0.07 labeled adaptive overfitting](/imgs/blogs/overfitting-to-the-validation-set-1.png)

By the end of this post you will be able to take any model-selection process — a hyperparameter sweep, a feature search, a checkpoint pick, a Kaggle leaderboard climb — and answer three questions about it: *how many times did I evaluate on this set?*, *how much optimism does that buy me?*, and *what is the number I should actually report?* You will have the math that predicts the inflation (the order statistic of a maximum over noisy estimates), the variance that tells you why small validation sets overfit fastest (the binomial standard error), and a runnable simulation that reproduces the whole effect in thirty lines of numpy. And you will have the discipline that prevents it: a three-way split with a final holdout you touch exactly once, evaluation budgets, and the reusable-holdout idea from the adaptive-data-analysis literature.

This is an **evaluation** bug, in the six-place taxonomy this series keeps returning to — data, optimization, model code, numerics, systems, and **evaluation**. It is worth saying plainly up front where it sits, because the symptom (a model that scores well in development and worse in production) gets misattributed constantly. People blame distribution shift, blame the data pipeline, blame "the model degraded." Sometimes it is those things. But very often the model never had the score you thought it did. The number was inflated at selection time, and production simply revealed the truth. If you cannot rule this out, you cannot trust any of your other debugging. So we localize it first.

## 1. The symptom: a number that does not transfer

Let me make the running example concrete, because the idea only lands once you have watched it happen on a real-ish run.

You are building a binary classifier — say churn prediction, fraud, or a content-quality model, it does not matter. You have 100,000 labeled rows. You split off 20,000 as validation and keep 80,000 for training. You train your first model: validation ROC-AUC of 0.852. Fine. Now the work begins. You try a deeper net; val goes to 0.861. You add a feature; 0.866. You tune the learning rate over a grid of eight values and keep the best; 0.871. You run a Bayesian hyperparameter search for 80 trials; 0.884. You try three architectures; the transformer-ish one wins at 0.892. You sweep the classification threshold and the dropout rate; 0.901. You pick the best of five random seeds; 0.908. You ensemble your top three checkpoints; 0.913. Two weeks, a few hundred evaluations on that one validation set, and you have driven the number from 0.85 to 0.91.

You ship it. You cut a brand-new sample of 20,000 rows that nobody ever looked at during development — a true fresh holdout — and you score the deployed model on it: **0.84**. Production traffic over the next month: **0.83**. The seven points you "earned" in the last two weeks were not real. They were the validation set telling you what you wanted to hear, because you kept asking until it did.

Here is the part that trips people up, and it is worth dwelling on. **You never trained on the validation set.** No row of it ever produced a gradient. The model's weights were fit entirely on the 80,000 training rows. So how can it be overfit to validation? The answer is that overfitting is not about *where the gradients flowed*. Overfitting is about **fitting noise instead of signal**, and there are two ways to fit a set: the obvious way, where the model's parameters absorb its noise, and the subtle way, where *your decisions* absorb its noise. Every time you chose option A over option B *because A scored higher on validation*, you let the random fluctuations of that particular 20,000-row sample steer your choice. The validation set has finite size, so its score is a noisy estimate of true performance. Optimize hard enough against a noisy estimate and you optimize the noise.

The instrument that catches this is brutally simple: **the gap between your reported number and a fresh holdout you touched once.** That gap is your accumulated validation overfitting, in points, measured. Everything else in this post is mechanism and discipline around that one measurement.

| Symptom | What it looks like | The honest test |
| --- | --- | --- |
| Val climbs steadily for two weeks | 0.85 -> 0.91, smooth | Score the final model on a fresh, never-seen holdout |
| Production lands well below val | val 0.91, prod 0.83 | Compare prod number to val and to fresh holdout |
| Each tweak adds a little | +0.005 per "improvement" | Count how many tweaks were chosen by val score |
| Best-of-N seeds beat single seed | 0.908 vs 0.892 | Re-score best seed on fresh data; gain often vanishes |
| Public leaderboard rank slips on private LB | rank 1 -> rank 40 | The private-LB drop is the overfitting, made public |

Notice what every honest test in that right column has in common: it requires **a number computed on data your selection process never saw.** That is the whole game. Validation overfitting is invisible from inside the validation score. You can only see it from outside.

Let me bisect the running example the way you would in practice, because the localization is half the value. The symptom is "model scores 0.91 in development, 0.83 in production." That is an eight-point gap, and eight points is a lot — it demands an explanation, and there are exactly four candidate explanations worth considering. First: **a code bug between training and serving** — a feature computed differently in production, a preprocessing step that silently differs, a version skew in a dependency. Second: **distribution shift** — production traffic genuinely differs from your development data, so even a perfectly-measured model would score lower. Third: **classic model overfitting** — the parameters memorized the training set and never generalized, which would show as a large train-vs-validation gap. Fourth: **validation overfitting** — the development number itself was inflated by selection, and production is simply telling the truth. These four are not mutually exclusive, but they have different signatures, and a disciplined debugger separates them before touching anything.

The bisection runs like this. To rule out the serving bug, you take a handful of production rows, run them through your *development* code path, and confirm the model produces the same scores it does in serving — if the predictions match, the pipeline is faithful and the gap is not a code bug. To rule out model overfitting, you look at the train-vs-validation gap *at a single evaluation*: if training AUC was 0.95 and validation was 0.91 at the same checkpoint, that four-point gap is ordinary and not the story; classic overfitting would show a chasm (train 0.99, val 0.80), and it would have been visible on the *first* evaluation, not crept in over two weeks. That leaves shift versus selection, and here is the clean separator that resolves the whole investigation: **cut a fresh holdout from your original development-era data pool — data from the same time and source as your validation set, but never used in any tuning decision.** Score the final model on it. If it comes back near 0.91, the development number was honest and the production gap is distribution shift (a data story, not a measurement story). If it comes back near 0.84, the development number was inflated and you have localized validation overfitting. In our running example it comes back 0.842, so the verdict is in: selection optimism, not shift, not a serving bug, not classic overfitting. The whole bisection took an afternoon and one fresh holdout, and it pointed at exactly one of the six places — evaluation — before we changed a single line of model code.

## 2. The science I: a validation score is a noisy estimate

Before we can talk about *adaptive* overfitting — the creep from many decisions — we have to be precise about the thing being overfit. A validation score is not a property of your model. It is a **random variable**: an estimate of true performance, computed on a finite sample, and therefore carrying sampling error. Get this and the rest follows.

Take accuracy on a validation set of $n$ examples. Suppose the model's true accuracy — the probability it gets a random example from the population right — is $p$. Each validation example is a Bernoulli trial: correct with probability $p$, wrong with probability $1-p$. The number correct out of $n$ is Binomial$(n, p)$, and the *measured* accuracy is that count divided by $n$. The standard error of that estimate is the standard deviation of a Bernoulli mean:

$$\sigma = \sqrt{\frac{p(1-p)}{n}}.$$

Plug in numbers. A validation set of $n = 400$ examples on a model with true accuracy $p = 0.85$:

$$\sigma = \sqrt{\frac{0.85 \times 0.15}{400}} = \sqrt{\frac{0.1275}{400}} \approx 0.0179.$$

So the measured accuracy on that set has a standard error of about **1.8 points**. The 95% interval is roughly $\pm 3.6$ points. Two models with *identical* true accuracy of 0.85 will routinely show measured accuracies differing by three or four points purely from which examples landed in this particular validation sample. Now you understand the mechanism in miniature: if you compare twenty models that are all truly 0.85 and keep the one with the highest measured accuracy, you will report something near 0.88 — not because that model is better, but because it got the friendliest draw of the validation noise.

Grow the set and the noise shrinks as $1/\sqrt{n}$. At $n = 5{,}000$:

$$\sigma = \sqrt{\frac{0.1275}{5000}} \approx 0.0051,$$

about half a point. At $n = 50{,}000$, $\sigma \approx 0.0016$, about a sixth of a point. **This is the single most important practical lever in the whole post: a bigger validation set is more resistant to overfitting, because there is less noise to memorize.** A 400-example validation set will be overfit by a determined hyperparameter sweep in an afternoon. A 50,000-example set can absorb thousands of evaluations before the creep becomes visible.

For ranking metrics like ROC-AUC the variance has a different (more complex) form — it depends on the number of positives and negatives, not just $n$ — but the qualitative story is identical: AUC measured on few examples is a high-variance estimate, and the variance of AUC scales roughly with $1/\sqrt{\min(n_+, n_-)}$, so a set with only 200 positives is dangerously noisy no matter how many negatives it has. The practical rule survives: **count your minority class, not your rows.** A 100,000-row validation set with 200 fraud cases is, for the purpose of overfitting AUC, a 200-example set.

#### Worked example: how much noise does a small val set carry?

You have two candidate models. On a validation set of $n = 1{,}000$, model X scores 0.873 accuracy and model Y scores 0.881. You pick Y. Was that justified? The true accuracy is around 0.877; the standard error is $\sigma = \sqrt{0.877 \times 0.123 / 1000} \approx 0.0105$, about 1.05 points. The difference between X and Y is 0.8 points — *less than one standard error*. The standard error of the *difference* between two correlated estimates on the same set is smaller than $\sqrt{2}\sigma$ (because they share examples), but it is still on the order of a point. So an 0.8-point gap is comfortably inside the noise. You did not pick the better model. You picked the luckier one on this sample. Do that for two hundred decisions and the luck compounds in exactly the direction that flatters you, because at every fork you systematically chose the higher number.

That last sentence is the hinge of the whole post, so let me restate it as a principle. **Selection is a maximization, and maximization over noisy estimates is biased upward.** It does not average out. It does not wash. Every time you keep the best of several noisy numbers, you import a little positive bias, and selection processes do this hundreds of times. Now we can quantify exactly how much.

## 3. The science II: the inflated maximum (multiple comparisons)

This is the rigorous heart of the post. The question is: if I evaluate $N$ candidate configurations on one validation set and keep the maximum score, how much does that maximum overstate the true performance — *even if all $N$ candidates are equally good and the only differences are noise?*

![A timeline of evaluations from eval 1 with zero gap, through eval 20, eval 80, and eval 200 where the gap reaches 0.07, then a final event where re-splitting resets the gap to zero](/imgs/blogs/overfitting-to-the-validation-set-2.png)

Model it cleanly. Suppose all $N$ candidates have the same true score $\mu$. The measured score of candidate $i$ is $X_i = \mu + \varepsilon_i$, where $\varepsilon_i$ is the validation noise with standard deviation $\sigma$ (the binomial standard error from the previous section). Assume the noise is roughly Gaussian — fine for accuracy on a few hundred-plus examples by the central limit theorem — and roughly independent across candidates (an approximation; they share the same val set so the noise is correlated, which actually makes the effect *milder* than the independent case, so the independent bound is a conservative upper bound on the damage).

We want the expected value of the maximum of $N$ i.i.d. standard normals. This is a classical result in extreme-value theory. For $Z_1, \dots, Z_N$ i.i.d. standard normal, the expected maximum grows like:

$$\mathbb{E}\left[\max_i Z_i\right] \approx \sqrt{2 \ln N}$$

for large $N$ (more precisely, $\sqrt{2\ln N} - \frac{\ln\ln N + \ln 4\pi}{2\sqrt{2\ln N}}$, but the leading term is what matters). Scaling back to our problem, where each $X_i$ has standard deviation $\sigma$, the expected reported maximum is approximately:

$$\mathbb{E}\left[\max_i X_i\right] \approx \mu + \sigma\sqrt{2 \ln N}.$$

That second term is the **optimism**, the **selection bias**, the **inflated maximum** — the amount by which your best-of-$N$ validation score overstates the truth, on average, with no real improvement at all. It is worth memorizing in this form because it tells you everything:

- It **grows with $\sigma$** (small, noisy validation sets inflate more).
- It **grows with $N$** (the more configs you try, the more you inflate) — but only as $\sqrt{\ln N}$, which is *slow*. Going from 10 configs to 1,000 configs (a 100x increase) only roughly doubles the inflation, because $\sqrt{2\ln 1000} / \sqrt{2\ln 10} \approx 3.72 / 2.15 \approx 1.73$. This is reassuring: the damage from a big sweep is bounded, it does not run away. But it is also a warning: even a *modest* sweep of 50 configs already carries real inflation.

Let me put numbers on it. Validation set with $\sigma = 0.01$ (about 1 point of noise, which is a ~2,000-example set at $p=0.85$). Try $N = 200$ configs:

$$\sigma\sqrt{2\ln N} = 0.01 \times \sqrt{2 \times \ln 200} = 0.01 \times \sqrt{2 \times 5.298} = 0.01 \times \sqrt{10.6} = 0.01 \times 3.26 \approx 0.0326.$$

So your best-of-200 validation score is inflated by about **3.3 points** on average, purely from picking the max of noise. If your true level is 0.85, you will report about 0.883 — and that is *before* any adaptive creep from the sequential decisions. Combine the selection bias from the sweep with the adaptive creep from your hand-tuning and the 7-point gap in the opening example is entirely unsurprising.

#### Worked example: the inflation across set sizes and sweep widths

Hold the true score at 0.85 and tabulate $\sigma\sqrt{2\ln N}$ for a few combinations. The standard error $\sigma$ comes from $\sqrt{0.1275/n}$.

| Val set size $n$ | $\sigma$ (points) | $N=20$ | $N=200$ | $N=2000$ |
| --- | --- | --- | --- | --- |
| 400 | 1.79 | +4.4 | +5.8 | +7.0 |
| 2,000 | 0.80 | +2.0 | +2.6 | +3.1 |
| 10,000 | 0.36 | +0.9 | +1.2 | +1.4 |
| 50,000 | 0.16 | +0.4 | +0.5 | +0.6 |

Read that table and the whole strategy falls out. A 400-example validation set with a 200-config sweep gives you almost **6 points** of free, fake improvement. The same sweep on a 50,000-example set gives you half a point — noise you can ignore. **The cheapest defense against validation overfitting is a bigger validation set**, because it attacks $\sigma$, the thing the inflation is proportional to. The second cheapest is fewer configs, but the $\sqrt{\ln N}$ scaling means you have to cut $N$ by orders of magnitude to make a dent, so it is a weaker lever than set size.

A common objection at this point is "but noise averages out — won't the bad-luck draws cancel the good-luck draws?" No, and seeing *why not* is the crux. Averaging cancels noise when you *average* the estimates. Selection does not average; it *maximizes*. The maximum of a set of numbers is not the average of the set — it is, by construction, the most extreme upward draw. When you keep the best of $N$ noisy scores, you are not summing the errors (which would cancel toward zero); you are selecting the single largest positive error and reporting it. The whole point of a maximum is that it ignores every downward draw and keeps only the most upward one. So the bias does not cancel — it *concentrates*. The more you search, the more upward draws you sample, and the more extreme the one you keep. This is also why the bias is always *positive*: there is no symmetric "you might pick an unlucky model" to balance it, because your decision rule never knowingly keeps the worse number. The asymmetry of the selection operator — always keep the higher score — is what converts symmetric, zero-mean noise into a one-directional, always-upward bias. If you instead reported the *average* val score across all $N$ configs, it would be unbiased; it is the act of reporting the *winner's* score that breaks it.

Now, a critical honesty note. This derivation assumed all candidates are *equally good* — pure null, no real signal. In reality some of your configs are genuinely better. The observed maximum is then **true improvement plus selection bias**, tangled together, and you cannot separate them from inside the validation score. That is precisely why you need an external measurement (the fresh holdout): the holdout score keeps the true improvement and drops the selection bias, so **(val max) minus (holdout) estimates the bias**, and (holdout) estimates the truth. The math above tells you how big to expect the bias to be; the holdout tells you how big it actually was. Both, together, are the diagnostic.

## 4. The diagnostic I: simulate the inflated maximum

Theory is convincing; a simulation you can run is more convincing. Here is the whole multiple-comparisons effect in numpy. We create $N$ models that are *all exactly equally good* (true accuracy 0.85), simulate each one's measured accuracy on a finite validation set, keep the max, and compare it to the truth. There is no real signal anywhere in this code — every "improvement" it finds is pure selection bias.

```python
import numpy as np

rng = np.random.default_rng(0)

def simulate_best_of_n(true_acc=0.85, n_val=2000, n_configs=200, n_trials=5000):
    """All configs have identical true accuracy. Measure each on a finite
    val set, keep the best, and see how much the best overstates the truth."""
    inflations = []
    for _ in range(n_trials):
        # Each config's measured accuracy = mean of n_val Bernoulli(true_acc) draws.
        # Shape (n_configs, n_val); mean over axis=1 is each config's val score.
        correct = rng.random((n_configs, n_val)) < true_acc
        val_scores = correct.mean(axis=1)        # one noisy score per config
        best = val_scores.max()                  # we keep the winner
        inflations.append(best - true_acc)       # all of this is fake
    inflations = np.array(inflations)
    return inflations.mean(), inflations.std()

for N in (20, 200, 2000):
    mean_inf, _ = simulate_best_of_n(n_configs=N)
    sigma = np.sqrt(0.85 * 0.15 / 2000)
    predicted = sigma * np.sqrt(2 * np.log(N))
    print(f"N={N:5d}  measured inflation={mean_inf:.4f}  "
          f"theory sigma*sqrt(2lnN)={predicted:.4f}")
```

Run it and you get something close to:

```bash
N=   20  measured inflation=0.0163  theory sigma*sqrt(2lnN)=0.0195
N=  200  measured inflation=0.0231  theory sigma*sqrt(2lnN)=0.0258
N= 2000  measured inflation=0.0288  theory sigma*sqrt(2lnN)=0.0309
```

The measured inflation tracks the $\sigma\sqrt{2\ln N}$ prediction (slightly below it, because the $\sqrt{2\ln N}$ formula is the leading term of an asymptotic series and overshoots a bit at finite $N$ — the lower-order correction I mentioned pulls it down, and the measured numbers sit right where theory says they should). The takeaway is not the third decimal place. It is that **you can predict, before you run a single real experiment, roughly how much optimism a sweep of a given width on a validation set of a given size will buy you.** When your real sweep produces a val improvement smaller than the inflation this simulation predicts for a null, you have learned nothing — the "improvement" is statistically indistinguishable from picking the luckiest config.

This simulation is also the cleanest way to build the intuition for a skeptical colleague. Run it live, point at the output, and say: every one of these models was identical, and the best one still "beat" the truth by two to three points. That is what your hyperparameter search does to your validation score. It does not measure skill; it measures who got the friendliest noise.

#### Worked example: is my sweep's gain real?

You ran an 80-trial Bayesian hyperparameter search on a 3,000-example validation set, $p\approx0.86$. Your baseline scored 0.860; your best trial scored 0.879, a 1.9-point gain. Real? Compute the null inflation. $\sigma = \sqrt{0.86 \times 0.14/3000} \approx 0.00634$. Inflation $\approx 0.00634 \times \sqrt{2\ln 80} = 0.00634 \times \sqrt{8.77} = 0.00634 \times 2.96 \approx 0.0188$, about **1.9 points**. Your entire measured gain is the size of the expected selection bias from a null search. So on this evidence you cannot claim the tuning helped at all — the honest move is to score the tuned model on a fresh holdout and see whether the 1.9 points survives. Usually most of it does not. (This is the quiet tragedy of aggressive HPO on small sets: it generates impressive validation curves that are mostly noise harvesting.)

## 5. The science III: why adaptivity breaks the holdout guarantee

So far we treated model selection as a *one-shot* maximum over $N$ pre-committed configs. The reality is worse, and it is worth understanding precisely why, because it is the reason the creep is *slow and silent* rather than a single visible jump.

The clean statistical guarantee behind a held-out set is this: if you fix a model *before* looking at the held-out data, then the held-out score is an **unbiased** estimate of true performance, with the binomial variance from section 2. "Fixed before looking" is the load-bearing phrase. It means the choice of model is **statistically independent** of the held-out sample. Independence is what makes the score honest.

The moment you choose your *next* model based on what you saw on the held-out set, you break that independence. Your second model is a function of the first model's held-out score, which is a function of the held-out sample. Now the model and the data it is evaluated on are **correlated**, and the unbiasedness guarantee evaporates. This is **adaptive data analysis**: a sequence of analyses where each step depends on the answers to previous steps, all using the same data. The classical i.i.d. holdout theory simply does not cover it, and the failure is not a small correction — it can be catastrophic. Dwork, Feldman, Hardt, Pitassi, Reingold, and Roth showed (in their 2015 work on the reusable holdout and adaptive analysis) that with enough adaptive queries you can drive a holdout estimate arbitrarily far from the truth, and they also showed how to *fix* it, which we will get to.

![A before-and-after with the left column selecting the best of two hundred configs on the reused validation set and reporting 0.91, and the right column scoring that same winning model on a fresh holdout touched exactly once at 0.84 matching production](/imgs/blogs/overfitting-to-the-validation-set-3.png)

Here is the intuition for *why adaptive is worse than one-shot*. In the one-shot case you pick the max of $N$ fixed noisy numbers — bias $\sigma\sqrt{2\ln N}$, bounded and slow-growing. In the adaptive case, you use the held-out score as a *gradient*: you see which direction improved it and you step that way, then re-measure, then step again. You are doing **gradient ascent on the validation noise**, and gradient ascent is far more efficient at exploiting structure than random search. After enough adaptive steps you can fit the specific noise pattern of that validation set the way SGD fits the training set — except your "parameters" are your design decisions and your "loss" is the negative validation score. The held-out set has effectively become a second training set, optimized through the channel of your choices.

This is why the creep is *gradual*. Each adaptive step leaks a little, the score ticks up a little, and nothing alarms you because nothing jumps. It is the boiling-frog version of overfitting. A one-shot sweep at least shows up as a suspiciously high single number you might question; the adaptive creep launders itself across two weeks of plausible-looking incremental progress. **The diagnostic, again, is the same: a fresh holdout. There is no way to detect adaptive overfitting from inside the set you have been adaptively querying. Independence, once broken, cannot be repaired by staring harder at the broken data.**

It helps to trace one concrete decision chain to see how a few innocent choices compound. Suppose your validation set has true-level noise of $\sigma = 1$ point, and you make a sequence of binary decisions, each "keep the change if val went up, revert if it went down." Decision one: add a feature. By chance, on this val sample, it lands +0.6 points — half of which is real, half is noise, but you cannot tell, so you keep it, and you have just imported about +0.3 points of noise. Decision two: a different normalization, +0.4 on val, you keep it, +0.2 of noise imported. Decision three: a dropout tweak, val moves -0.2, you revert — and notice that *reverting on a down-move is itself selection*: you are systematically discarding changes that happened to draw unfriendly noise and keeping changes that happened to draw friendly noise. The asymmetry is the whole problem. Across forty such decisions, even if every change is *truly neutral*, you keep roughly the twenty that drew positive noise and revert the twenty that drew negative noise, and the kept ones each contribute their friendly-noise fraction to your running score. The validation number ratchets upward, decision by decision, never down, because your decision rule *only ever accepts upward moves on this set*. That is the ratchet. That is adaptive overfitting in slow motion, and it is why a project's validation score almost never goes down over time even when the model is not actually improving — the process is structurally incapable of letting it fall.

Now contrast the two regimes side by side so the difference is unmistakable. One-shot selection — pick the best of $N$ pre-committed configs — imports bias *once*, bounded by $\sigma\sqrt{2\ln N}$, and you can predict it in advance. Adaptive selection — let each move depend on the last move's val score — imports bias *at every step*, and because you are using the val score as a search direction rather than a one-time filter, you can in principle exceed the one-shot bound by a wide margin given enough steps. The reusable-holdout theory makes this precise: with fully adaptive queries and no protection, the number of queries you can answer accurately scales only *linearly* in a quantity that, without the differential-privacy machinery, leaves you exposed to overfitting after a number of queries roughly proportional to the *square root* of the holdout size — far fewer than the exponential number you get for non-adaptive queries. The practical upshot for a working engineer: a fixed validation set is a *consumable resource*. Every adaptive query spends a little of it, and once spent, the only way to get a fresh honest number is fresh data.

| Property | One-shot selection | Adaptive selection |
| --- | --- | --- |
| How model depends on val | $N$ configs fixed in advance | each choice uses prior val scores |
| Bias growth | $\sigma\sqrt{2\ln N}$, slow | can compound much faster |
| Independence of val sample | preserved until the final max | broken at the first adaptive step |
| Visibility | one suspicious high number | gradual, plausible-looking creep |
| Detectable from inside the set | partly (compare to null) | no |
| The only honest check | fresh holdout | fresh holdout |

## 6. The diagnostic II: a clean three-way split and the fresh-holdout test

Now the practical core. The discipline that prevents validation overfitting is older than deep learning and almost universally under-practiced: a **three-way split** — train, validation, and a **test set you touch exactly once, at the very end, for the number you report.** The validation set is the one you abuse with hundreds of evaluations; it is *meant* to be overfit, that is its job. The test set is the sealed vault. You open it once.

![A six-row matrix mapping adaptive creep, many configs, test as val, hyperparameter memorization, early stopping on the report set, and leaderboard overfitting to their inflation mechanism, detector, and fix](/imgs/blogs/overfitting-to-the-validation-set-4.png)

The figure above is the field guide: six distinct routes to validation overfitting, each with its own mechanism, detector, and fix. Let us turn the central discipline into code. Here is a clean scikit-learn three-way split with the test set quarantined, plus the model-selection loop that is allowed to hammer the validation set.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

X, y = load_my_data()  # your data; X (n, d), y (n,) in {0, 1}

# 1) Carve off the SEALED test set FIRST and do not look at it again
#    until you have a single, final, frozen model. stratify keeps the
#    positive rate identical across splits so AUC variance is comparable.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# 2) Split the remaining dev data into train and validation. THIS val set
#    is the one you are allowed to evaluate against many times.
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.20, stratify=y_dev, random_state=42
)

# 3) Model selection: tune freely against X_val. Count every evaluation.
eval_count = 0
best_auc, best_cfg, best_model = -1.0, None, None
for max_leaf in (15, 31, 63):
    for lr in (0.03, 0.1, 0.3):
        for l2 in (0.0, 1.0, 10.0):
            clf = HistGradientBoostingClassifier(
                max_leaf_nodes=max_leaf, learning_rate=lr,
                l2_regularization=l2, random_state=0,
            )
            clf.fit(X_train, y_train)
            auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
            eval_count += 1            # <-- track your evaluation budget
            if auc > best_auc:
                best_auc, best_cfg, best_model = auc, (max_leaf, lr, l2), clf

print(f"evaluated {eval_count} configs;  best VAL AUC={best_auc:.4f}")

# 4) Open the vault ONCE. This is the number you report.
test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print(f"best config {best_cfg};  honest TEST AUC={test_auc:.4f}")
print(f"selection optimism = VAL - TEST = {best_auc - test_auc:+.4f}")
```

That last printed line — `VAL - TEST` — is your measured validation overfitting, in AUC points, for this project. It is the single most valuable diagnostic number in the file, and almost nobody computes it. With 27 configs on a moderate val set you might see `+0.01` to `+0.02`; if you see `+0.06`, your val set is too small or you swept too hard, and your reported number must be the test number, not the val number.

There is one rule about the test set that you must enforce socially, not just in code: **once you look at the test number, you may not go back and tune.** The instant you say "test was disappointing, let me try one more thing and re-test," your test set has become a validation set and your honest number is gone. If you genuinely need another round of development, you need *another fresh test set* — which means budgeting test data up front for the number of honest checks you expect to need. This is why production teams hold *multiple* sealed holdouts, opened on a schedule, never reused.

The cleanest organizational pattern I have seen is a **rotating sealed-holdout pool**: at the start of a project, carve off several disjoint test slices and lock them, then open exactly one per milestone — one for the first ship decision, one for the next quarter's re-evaluation, one for the model-card refresh. Each is touched once, retired, and never reopened. The cost is data (you need enough to afford several test slices), but the payoff is that you always have an honest number available *and* you never accidentally tune against the set you will report. When the data budget is tight, the time-based version is even better, because it matches how production actually works: hold out the *most recent* slice chronologically as the test set, since a forward-in-time holdout also tests for the temporal distribution shift you will face in deployment. Here is that pattern, which doubles as both an overfitting guard and a shift detector:

```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# df has a 'ts' timestamp column. The most RECENT slice is the sealed test
# set: forward-in-time, so it guards against selection optimism AND tells you
# whether the model holds up on fresher data than it was tuned on.
df = df.sort_values("ts").reset_index(drop=True)
n = len(df)
test_df = df.iloc[int(0.85 * n):]          # newest 15% -> sealed test
dev_df  = df.iloc[: int(0.85 * n)]         # older 85%  -> train + val

# (train/val split, tuning, model selection all happen inside dev_df only)
# ... tune freely against a val slice of dev_df, log every evaluation ...

# Open the forward-in-time test ONCE:
test_auc = roc_auc_score(
    test_df["y"], final_model.predict_proba(test_df[features])[:, 1]
)
val_auc = best_auc  # from the dev-only selection loop
print(f"val (tuned, older) = {val_auc:.4f}   test (sealed, newest) = {test_auc:.4f}")
print(f"gap = {val_auc - test_auc:+.4f}  "
      f"(selection optimism AND/OR forward-time shift)")
```

A subtlety worth naming: with a forward-in-time test set, the `val - test` gap conflates *two* effects — selection optimism and genuine temporal shift — so a large gap does not isolate which one. That is fine for the ship decision (you want the realistic forward number regardless of its cause), but if you need to *attribute* the gap, add a same-era random holdout from `dev_df` as well: the random-era holdout isolates selection optimism, and the difference between the random-era and forward-time holdouts isolates the shift. Two cheap holdouts, two cleanly separated diagnoses.

#### Worked example: the fresh-holdout test in practice

You suspect your churn model's 0.91 val AUC is inflated. You have not held out a separate test set (a mistake, but recoverable if you have more data). So you collect the *next* 20,000 rows that arrive — chronologically fresh, never seen by any of your tuning — and score the final model: 0.842. Your measured overfitting is $0.91 - 0.842 = 0.068$, almost seven points, consistent with a small val set plus a couple hundred adaptive evaluations. You now report **0.84**, not 0.91, and you have a defensible, honest number. Critically, you also have a *retrospective measurement of your process's optimism* — about 7 points — which you can subtract from future val scores as a rough correction until you fix the underlying discipline. (Subtracting a fixed optimism estimate is a hack, not a guarantee, but it beats reporting the raw val number.)

## 7. The diagnostic III: count your evaluations and budget them

The cheapest instrument in this entire post is a counter. Validation overfitting is monotone in the number of times you evaluate against the validation set, so the first thing to *measure* is that count. Most teams have no idea whether they have run 30 or 3,000 evaluations against their val set over a project's life, which means they have no idea how much optimism they have accumulated.

Instrument it. Wrap every validation evaluation in a logger that records the count, and budget the count up front the way you budget compute.

```python
import json, time, pathlib

class HoldoutBudget:
    """Counts and logs every evaluation against a held-out set so you
    always know how much you've adaptively queried it."""
    def __init__(self, name, budget, logfile="holdout_log.jsonl"):
        self.name, self.budget = name, budget
        self.count = 0
        self.log = pathlib.Path(logfile)

    def evaluate(self, score_fn, *args, tag=""):
        if self.count >= self.budget:
            raise RuntimeError(
                f"{self.name}: evaluation budget {self.budget} exhausted. "
                f"Cut a FRESH holdout before querying again."
            )
        score = score_fn(*args)
        self.count += 1
        with self.log.open("a") as f:
            f.write(json.dumps({
                "set": self.name, "n_eval": self.count,
                "score": float(score), "tag": tag, "t": time.time(),
            }) + "\n")
        return score

# usage: every val check goes through it; the test budget is 1.
val_budget  = HoldoutBudget("val",  budget=300)
test_budget = HoldoutBudget("test", budget=1)

auc = val_budget.evaluate(
    roc_auc_score, y_val, model.predict_proba(X_val)[:, 1], tag="cfg_31_0.1"
)
```

Two things this buys you. First, the hard `budget=1` on the test set makes the cardinal sin — peeking at test during development — *throw an exception* instead of silently corrupting your number. Second, the JSONL log lets you reconstruct, after the project, exactly how many adaptive queries the val set absorbed, so you can plug $N$ into $\sigma\sqrt{2\ln N}$ and estimate your optimism even without a fresh holdout. A team that logs this religiously can say "we ran 280 evaluations on a 3,000-row val set, so expect roughly $0.0063 \times \sqrt{2\ln 280} \approx 2$ points of optimism" — and report accordingly.

This connects directly to **early stopping**, which is a sneakily common way to overfit the report set. If you early-stop on the *same* set you then report, you have selected the checkpoint with the friendliest noise on that set — a best-of-$T$ maximum where $T$ is the number of checkpoints, with bias $\sigma\sqrt{2\ln T}$ baked into your reported number. For a 200-epoch run checkpointed every epoch, that is $\sqrt{2\ln 200}\approx 3.3$ standard errors of optimism. The fix is to **early-stop on a separate split from the one you report**, or to early-stop on validation and report on the sealed test set. The same logic applies to picking the best of several random seeds: best-of-5-seeds is a best-of-5 maximum, and the winning seed's val score is inflated by about $\sigma\sqrt{2\ln 5}\approx 1.8\sigma$ relative to the truth — score it on fresh data before you believe the gain.

| Selection operation | Hidden "$N$" | Approx optimism | The fix |
| --- | --- | --- | --- |
| Grid/Bayesian HPO | configs tried | $\sigma\sqrt{2\ln N}$ | nested CV or fresh holdout |
| Early stop on report set | checkpoints | $\sigma\sqrt{2\ln T}$ | stop on a separate split |
| Best-of-$k$ seeds | seeds | $\sigma\sqrt{2\ln k}$ | re-score winner on fresh data |
| Threshold sweep | thresholds | $\sigma\sqrt{2\ln m}$ | pick threshold on val, report on test |
| Feature selection by val | feature sets | grows with sets tried | wrap selection inside CV |

## 8. The science IV: the order statistic, more carefully, and what it implies

Let me return to the math once more and tighten it, because there is a common misconception that "more data fixes everything" and the order-statistic view shows precisely where that is and is not true.

![A branching graph where one true skill level of 0.85 feeds three equal configs whose noisy scores feed a keep-the-max node, producing an inflated 0.89, which a fresh holdout then collapses back to 0.85](/imgs/blogs/overfitting-to-the-validation-set-5.png)

The expected maximum of $N$ i.i.d. $\mathcal{N}(\mu, \sigma^2)$ variables is $\mu + \sigma \, a_N$, where $a_N = \mathbb{E}[\max_i Z_i]$ for standard normals. We approximated $a_N \approx \sqrt{2\ln N}$. The exact $a_N$ for small $N$ is tabulated: $a_2 \approx 0.56$, $a_5 \approx 1.16$, $a_{10} \approx 1.54$, $a_{100} \approx 2.51$, $a_{1000} \approx 3.24$. (The $\sqrt{2\ln N}$ approximation gives 1.18, 1.79, 2.15, 3.03, 3.72 for those — it overshoots at small $N$, which is why our simulation came in *below* it.) Either way, the structure is the same: the inflation in *standard-error units* depends only on $N$, and the inflation in *score units* is that times $\sigma$.

This decomposition is the whole strategy in one line:

$$\underbrace{\text{optimism}}_{\text{score units}} = \underbrace{a_N}_{\text{how hard you searched}} \times \underbrace{\sigma}_{\text{how noisy your set is}}.$$

You have two knobs. **$a_N$ — how hard you searched** — you can shrink by trying fewer things, but only logarithmically, so it is a weak knob: cutting from 1000 to 100 configs only reduces $a_N$ from 3.24 to 2.51, a 22% cut for a 90% reduction in search. **$\sigma$ — how noisy your validation set is** — you can shrink by enlarging the set, and it shrinks as $1/\sqrt{n}$, a strong knob: quadrupling the validation set halves the optimism, for *any* amount of searching. This is why, when a colleague asks "we did a huge sweep, how do we trust the result," the right answer is almost never "do a smaller sweep" and almost always "score the winner on a bigger, fresh set." You cannot un-search, but you can re-measure on cleaner data, and re-measuring is what kills the bias.

There is a subtlety I flagged earlier and should make rigorous: the candidates are *not* independent, because they share the validation set. Their noise is positively correlated (a friendly validation sample lifts all of them together). Positive correlation **reduces** the spread of the maximum relative to the independent case — in the limiting case of perfectly correlated noise, all candidates move together, there is no "luckiest draw" to select, and the selection bias vanishes. So the independent-noise formula $\sigma\sqrt{2\ln N}$ is a **conservative upper bound** in the realistic correlated setting. In practice the candidates differ enough (different hyperparameters change *which* examples they get right) that the correlation is partial and you get a meaningful fraction of the independent bound. The honest summary: the formula is an upper bound, the real bias is somewhat less, and a fresh holdout measures the real value exactly. Use the formula to know *roughly* what to expect, and the holdout to know the truth.

## 9. The fix: discipline, nested CV, and the reusable holdout

We have the diagnostics. Here is the full prevention stack, from cheapest-and-most-effective to most-sophisticated.

![A decision tree from a suspicious 0.91 validation score branching on whether you ran many evals, whether the set is tiny, or whether a fresh number exists, each routing to limiting and logging evals, growing the set, or cutting a fresh holdout](/imgs/blogs/overfitting-to-the-validation-set-6.png)

**1. The three-way split, test touched once.** Covered above. This is non-negotiable and free. Train to fit, validation to select, test to report. The test number is the one that goes in the paper, the model card, the slide. If you only adopt one practice from this post, adopt this one.

**2. A bigger validation set.** Because optimism is $a_N \times \sigma$ and $\sigma \propto 1/\sqrt{n}$, enlarging the validation set is the strongest defense against any amount of searching. If your val set has 400 examples and you are sweeping hundreds of configs, you are doomed before you start; quadruple it. For ranking metrics, what matters is the minority-class count, so enrich positives if they are scarce.

**3. Limit and log your evaluations.** The `HoldoutBudget` counter. You cannot manage what you do not measure, and validation overfitting grows with a count almost nobody tracks.

**4. Nested cross-validation for the honest number.** When data is too scarce to afford a sealed test set, nested CV gives you an (almost) unbiased estimate while still tuning. The *inner* loop tunes hyperparameters; the *outer* loop scores the tuned pipeline on folds the inner loop never saw. The outer score is your honest number because the model handed to each outer fold was selected without seeing that fold. This is the canonical fix for the "tune and report on the same folds" inflation, and it is covered in depth in [Cross-Validation Done Wrong](/blog/machine-learning/debugging-training/cross-validation-done-wrong) — the sibling post to this one. Here is the shape:

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Inner loop: GridSearchCV tunes C using inner folds only.
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
pipe = Pipeline([("scale", StandardScaler()),
                 ("clf", LogisticRegression(max_iter=2000))])
search = GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]},
                      scoring="roc_auc", cv=inner)

# Outer loop: each outer fold scores a model the inner search never saw.
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
outer_scores = cross_val_score(search, X_dev, y_dev, scoring="roc_auc", cv=outer)

print(f"nested CV AUC = {outer_scores.mean():.4f} +/- {outer_scores.std():.4f}")
# This number is honest. The single best inner-CV score would be optimistic.
```

The gap between the best *inner* CV score (what `GridSearchCV.best_score_` reports) and the *nested* outer score is the tuning optimism, measured. It is usually a couple of points; if it is large, your inner search was too aggressive for your data size.

**5. The reusable holdout (differential privacy for evaluation).** The most sophisticated fix, from the adaptive-data-analysis line of work. The core idea: if you add carefully calibrated noise to the holdout answers you return, and only report a "changed" answer when it differs *significantly* from the training estimate, you can answer **many more** adaptive queries before the holdout degrades — the noise prevents your decisions from latching onto the holdout's specific noise pattern. This is the **Thresholdout / reusable-holdout** mechanism (Dwork et al., 2015), and the related **Ladder** algorithm (Blum and Hardt, 2015) does the same for leaderboards: it only reveals your score when you *actually* beat your previous best by more than the noise, so chasing the leaderboard cannot harvest its noise. You will rarely implement these from scratch, but the principle is portable and practical: **return validation answers at limited precision, and only act on differences that exceed the set's standard error.** If two configs differ by less than $\sigma$ on validation, treat them as tied and do not let that comparison drive a decision. That one habit — *ignore differences smaller than your standard error* — captures most of the protection of a reusable holdout for free.

Here is a lightweight, practical version of the "ignore-sub-noise-differences" rule, which is the reusable-holdout principle in spirit:

```python
import numpy as np

def trustworthy_improvement(new, best, n_val, p=0.85, k=1.0):
    """Only count a val improvement as real if it exceeds k standard errors.
    Below that, the 'gain' is indistinguishable from selection noise, so we
    refuse to let it drive a decision -- the reusable-holdout habit, cheaply."""
    se = np.sqrt(p * (1 - p) / n_val)        # binomial standard error
    return (new - best) > k * se

# during selection:
if trustworthy_improvement(auc, best_auc, n_val=len(y_val)):
    best_auc, best_model = auc, clf          # accept only above-noise gains
# else: treat as a tie, keep the simpler/earlier model
```

By refusing to chase sub-noise improvements, you stop your decision process from doing gradient ascent on the validation noise — which, recall from section 5, is the engine of adaptive overfitting. You will accept fewer "improvements," and the ones you accept will be more likely to be real and to survive the test set.

## 10. Stress tests: tiny sets, finetuning, and where the creep hides worst

The series discipline is to pose the failing run, fix it, then *stress-test the fix* — push on the edges and ask where it breaks. Validation overfitting has several edges worth pushing on, and each one teaches a sharper version of the rule.

**What if the validation set is tiny?** This is the worst case, and it is worth internalizing why. Optimism is $a_N \times \sigma$, and $\sigma = \sqrt{p(1-p)/n}$, so halving $n$ multiplies the noise by $\sqrt{2} \approx 1.41$ and multiplies your optimism by the same factor *for any amount of searching*. A 200-example validation set at $p = 0.85$ has $\sigma \approx 0.025$, two and a half points of noise per evaluation; a best-of-100 sweep against it inflates by $0.025 \times 2.51 \approx 0.063$, more than six points of pure fiction. On a set that small, even *normal* iterative development — a few dozen honest decisions over a week — will manufacture several points. The stress test reveals the rule: **below a few thousand examples, a single fixed validation set is not safe for heavy tuning at all, and you should switch to cross-validation** (which uses every row for validation across folds, effectively enlarging the validation signal) and to the sub-noise-difference rule (refuse to act on gaps smaller than two and a half points). Small data does not mean you cannot tune; it means you must tune through cross-validation and report through nested CV, never through a lone tiny holdout.

**What if it is data, not selection?** A fair challenge: maybe the production drop is the data pipeline lying, not the validation score. The stress test is the era-matched fresh holdout from section 12 — cut fresh data from the *same source and time* as your validation set. If that era-matched holdout matches your inflated val number, the development measurement was honest and the gap is downstream (shift or a serving bug), so you are debugging *data*, not *evaluation*, and this whole post is the wrong tool. If the era-matched holdout comes back low like production, the development number was inflated and selection is your bug. This single test cleanly separates "your number was a lie" from "your number was true but the world changed," and they need completely different fixes.

**What about finetuning an LLM — does this even apply?** It applies *more*, not less, and in a sharper form, because LLM evaluation sets are often small and the decision space is enormous. When you finetune a model and pick the best checkpoint by an eval-set score — best of, say, 20 saved checkpoints across 3 epochs — that is a best-of-20 maximum, inflating your reported eval by $\sigma\sqrt{2\ln 20} \approx 2.4\sigma$. When you sweep LoRA rank, alpha, learning rate, and the data mix and keep the combination with the highest eval score, that is another best-of-$N$ on the same small eval set. And LLM "eval sets" are frequently a few hundred prompts scored by an automated judge, which is exactly the tiny-noisy-set regime where optimism is worst. The fix is identical to the general one — a held-out eval set you score once, an evaluation budget, and the sub-noise rule — but the stakes are higher because finetuning teams iterate fast and the eval sets are small. A finetuned model that "improved" from 71% to 76% on a 300-example eval across forty configs has, by the math, a null inflation of about $\sqrt{0.71\times0.29/300}\times\sqrt{2\ln 40} \approx 0.026 \times 2.72 \approx 0.071$, seven points — *larger than the entire observed gain*. The only way to know whether that finetune actually helped is a fresh eval set the sweep never touched.

#### Worked example: the LoRA sweep that ate its own eval set

A team finetunes a 7B model with LoRA for an internal classification task. Eval set: 250 held-out prompts, judged automatically, baseline 0.700. They sweep rank (4, 8, 16, 32), alpha (8, 16, 32), learning rate (1e-4, 2e-4, 5e-4), and three data mixes — 108 combinations — and keep the best, which scores 0.768 on the eval set, a 6.8-point lift they are excited about. Stress-test it with the math: $\sigma = \sqrt{0.7 \times 0.3 / 250} \approx 0.029$, and best-of-108 inflation is $0.029 \times \sqrt{2\ln 108} = 0.029 \times \sqrt{9.37} = 0.029 \times 3.06 \approx 0.089$, almost nine points of expected null inflation. Their 6.8-point gain is *smaller* than the optimism a null sweep would produce on this set. So on this evidence the finetune may have done nothing measurable. They cut a fresh 250-prompt eval set, re-score the winning config: 0.712. The real lift was about one point, inside the noise, and the other six points were the sweep harvesting the original eval set. The honest report is "no reliable improvement," the honest next step is a bigger eval set, and the two GPU-days the sweep cost bought a lesson rather than a model. This is the finetuning version of the same bug, and it bites hardest exactly where teams move fastest.

## 11. The before-after: a run that dropped on a fresh holdout, and the fix

Let me close the loop on the running example with a concrete before-and-after, because this series lives on measured evidence, not assertions.

![A before-and-after with the left column tuning on the validation set two hundred times and shipping at 0.91 only to drop in production, and the right column logging every evaluation and opening a sealed holdout once to report 0.84 that holds in production](/imgs/blogs/overfitting-to-the-validation-set-8.png)

**Before (no holdout discipline).** Two-way split: 80k train, 20k validation. Over two weeks: 8 architectures, an 80-trial Bayesian sweep, a threshold sweep, best-of-5 seeds, a 3-checkpoint ensemble. Roughly 200 evaluations against the one validation set. Reported number: **val AUC 0.913.** Shipped. Production over the first month: **0.831.** A fresh 20k holdout cut after the fact: **0.842.** The model that "earned" seven points in the final stretch had a true level around 0.84; everything above that was the validation set's noise, harvested by two hundred adaptive decisions and one aggressive sweep.

**After (the discipline applied).** Re-run the project with a three-way split: 70k train, 15k validation, 15k sealed test. Same modeling effort, all of it against the validation set, every evaluation logged through the budget counter (final count: 214). The validation winner still reads optimistic — **val AUC 0.908** — but now we open the sealed test set exactly once: **test AUC 0.846.** Measured optimism: $0.908 - 0.846 = 0.062$, six points, right where the math predicted for a 15k val set hammered 214 times with an aggressive sweep. We **report 0.846**, and production comes back at **0.841** — within noise of the test number. The discipline did not make the model better. It made the *reported number true*, which is the entire point: a true 0.85 you can trust beats a fake 0.91 that collapses the day it ships and triggers a fire drill to find a "regression" that was never there.

| Metric | Before (2-way, undisciplined) | After (3-way, sealed test) |
| --- | --- | --- |
| Reported number | val 0.913 | test 0.846 |
| Fresh-holdout / production | 0.831 - 0.842 | 0.841 |
| Optimism (reported - fresh) | +0.07 to +0.08 | +0.005 (noise) |
| Evaluations logged | unknown | 214 |
| Number survives production? | no (7-point drop) | yes (within noise) |
| Fire drill on launch | yes | no |

The deeper lesson is cultural. The "before" team did *more work* and got a *worse outcome* — not a worse model, but a worse decision-making process, because they spent two weeks optimizing a number that was mostly noise and then got blindsided in production. The "after" team did the same modeling but knew, before launch, what the model could actually do. **In ML, the value of an honest number is that it lets you stop.** When you can trust your evaluation, you stop tuning when the real gains stop, instead of chasing the validation noise upward for another fortnight.

There is a second-order benefit that is easy to miss. Because the "after" team *measured* their optimism — 6.2 points, logged alongside 214 evaluations — they now have a calibrated prior for the *next* project. They know that on a 15k validation set, an aggressive sweep buys roughly six points of fiction, so the next time a teammate reports a five-point validation lift from a big sweep, they can say "that is within the noise floor for our setup, score it on the sealed set before we believe it" — and they will be right, before any data is collected, because the math and their own logged history agree. This is what mature ML evaluation looks like: not the absence of validation overfitting (you cannot avoid it entirely while tuning), but the *quantification* of it, so that every reported number comes with an honest error bar and nobody is ever surprised on launch day. The discipline pays compounding dividends: each project's logged `val - test` gap sharpens the team's prior for the next one, until "is this gain real?" becomes a question they can answer in thirty seconds with a standard error and a $\sqrt{2\ln N}$, rather than a question they answer the hard way, in production, a month too late.

## 12. Case studies and real signatures

These are real, documented patterns. I will be careful to flag where a number is approximate.

**Kaggle public vs private leaderboard — the canonical live demonstration.** Competitions split the hidden test set into a *public* portion (a fraction of test rows, scored on every submission and shown on the live leaderboard) and a *private* portion (the rest, scored once at the close). Teams submit many times per day for weeks, tuning against the public leaderboard — which is exactly the adaptive querying we have been describing, at industrial scale across thousands of participants. The result is the famous **"leaderboard shake-up"**: teams ranked near the top on the public LB routinely drop dozens or hundreds of places on the private LB, because they overfit the public slice. The reliable winners are the ones who *trusted their own cross-validation over the public leaderboard* and submitted models whose CV and public scores agreed. The public-minus-private gap is validation overfitting, made visible and reproducible for everyone to see — which is why "trust your CV, not the LB" is the single most-repeated piece of Kaggle wisdom. The Ladder algorithm (Blum and Hardt, 2015) was designed precisely to limit this by only revealing a leaderboard score when it significantly beats the prior best.

![A three-by-three grid showing a public slice of thirty percent of the test where two teams rank one and two, then a private slice of seventy percent scored once where the leaderboard-chaser drops thirty-nine places while the team that trusted cross-validation holds its rank](/imgs/blogs/overfitting-to-the-validation-set-7.png)

**ImageNet and the question of test-set reuse.** The computer-vision field spent a decade reporting progress on a *single fixed* ImageNet validation set, with thousands of papers and an entire community adaptively selecting architectures, augmentations, and hyperparameters against it. The natural worry is that the whole field collectively overfit that one set. Recola and collaborators (Recht, Roelofs, Schmidt, Shankar, 2019, "Do ImageNet Classifiers Generalize to ImageNet?") built brand-new test sets for CIFAR-10 and ImageNet following the original collection protocols as closely as possible, and re-evaluated many published models. The reassuring finding: accuracy *dropped* on the new sets (by roughly 3 points on ImageNet and around 8–10 points on CIFAR-10, approximate figures from their paper), but model *rankings* were largely preserved — better models on the old set were still better on the new set. The interpretation is nuanced: most of the drop was attributed to the new test sets being slightly harder (distribution shift in the collection process), not to classic adaptive overfitting of rankings. The practical lesson stands regardless: **a single fixed evaluation set, reused by a whole community for years, must be treated with suspicion, and the only way to know is to build a fresh one and re-measure** — which is exactly the fresh-holdout test, applied at the scale of a research field.

**The hyperparameter sweep that "found" 2 points of nothing.** A common industrial signature, generic enough to be a category: a team runs a large Bayesian HPO job (hundreds of trials) on a modest validation set, reports a 2–3 point lift over baseline, ships it, and sees no production improvement. The post-mortem, every time, is the same: the lift was within $\sigma\sqrt{2\ln N}$ of the null expectation for that set size and search width. The diagnostic that would have caught it before launch is the one from section 4 — compute the null inflation, compare it to the measured lift, and if they are comparable, score the winner on a fresh holdout before believing the gain.

**Feature selection by validation score — the silent compounder.** Forward/backward feature selection that adds or drops features based on validation score is a best-of-$N$ maximum at *every* step, and the optimism compounds across steps. A team selecting features one-at-a-time from a pool of 500, each step a best-of-remaining maximum, can manufacture several points of validation AUC from pure noise on a small set. The fix is to wrap the *entire* feature-selection procedure inside the cross-validation loop (select features on the training portion of each fold, score on the held-out portion), so the selection never sees the data it is scored on — the same nested principle as nested CV for hyperparameters.

**A/B-test peeking — the same bug in production clothing.** The identical mechanism appears online, where it has its own name: *peeking* at an A/B test and stopping the moment significance is reached. Checking a running experiment repeatedly and stopping at the first favorable look is a best-of-many-looks maximum over the test statistic's random walk, which inflates the apparent effect and the false-positive rate exactly the way a hyperparameter sweep inflates a val score. The statistical fix is also analogous — sequential testing procedures (always-valid p-values, alpha-spending) that account for the number of looks, mirroring the way the evaluation budget and the $\sqrt{2\ln N}$ correction account for the number of offline evaluations. If you understand offline validation overfitting, you already understand online peeking; they are one phenomenon wearing two outfits, and the cure in both cases is to *count and correct for the number of times you looked.*

## 13. When this is (and isn't) your bug

A decisive section, because misattribution wastes more time than the bug itself.

**It IS validation overfitting when:** your development number is good, it climbed steadily over many evaluations, and a *fresh holdout you never tuned against* comes back materially lower. The single confirming test is the fresh-holdout gap. If you have a sealed test set and `val - test` is large (say more than two or three standard errors of the test set), you are looking at selection optimism. If you ran a big sweep and the lift is within $\sigma\sqrt{2\ln N}$ of the null, you are looking at selection optimism. If your public LB rank craters on the private LB, you overfit the public LB.

**It is NOT (primarily) validation overfitting when:** the model is *bad on validation too* — if you cannot drive a good number even on the set you are tuning against, the problem is upstream (data, model, optimization), not selection. Use the overfit-one-batch test and the rest of the foundations track before you suspect selection bias. **It is not validation overfitting when** the train-vs-validation gap is huge *at a single evaluation* — that is classic model overfitting (the parameters absorbed the training noise), a different bug with a different fix (regularization, more data, smaller model), and it shows up on the *first* evaluation, not as a slow creep across hundreds. **It is not validation overfitting when** the drop is driven by genuine distribution shift between development and production — feature drift, a changed upstream system, seasonality — in which case a fresh holdout *from the same era as your val set* would still match val, and only the *production-era* data diverges. That distinction is the cleanest separator: cut a fresh holdout from your *original* data pool; if it matches your inflated val number, the gap is shift, not selection; if it comes back low like production, the gap is selection.

| If you observe | Most likely | Confirming test |
| --- | --- | --- |
| Val good, slow climb, fresh holdout low | validation overfitting | fresh-holdout gap |
| Bad on val itself | upstream bug (data/model/optim) | overfit one batch |
| Huge train-val gap at one eval | model overfitting | regularize, more data |
| Era-matched holdout matches val, prod low | distribution shift | compare eras, not just splits |
| Public LB high, private LB low | leaderboard overfitting | trust CV over LB |

The order of operations matters. Rule out upstream bugs first (can you even get a good number on the set you tune against?), then separate model overfitting from selection overfitting (single-eval train-val gap vs slow multi-eval creep), then separate selection overfitting from distribution shift (era-matched fresh holdout). Only then have you localized it. This is the bisection discipline the whole series is built on — see [A Taxonomy of Training and Finetuning Bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) for the full decision tree, and the [Training Debugging Playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) capstone for the end-to-end procedure.

## 14. Key takeaways

- **You can overfit a set you never trained on.** Tuning against a validation score hundreds of times leaks information through your *decisions*; the parameters never touched it, but your choices did. Overfitting is fitting noise, and decisions fit noise too.
- **A validation score is a random variable, not a fact.** Its standard error is $\sigma = \sqrt{p(1-p)/n}$ for accuracy; small sets are noisy and overfit fastest. For ranking metrics, what matters is the *minority-class count*, not the row count.
- **Selection is maximization, and maximizing over noise biases upward.** The best of $N$ noisy estimates overstates the truth by about $\sigma\sqrt{2\ln N}$ — slow in $N$ (logarithmic) but real even for modest sweeps.
- **The optimism factors as $a_N \times \sigma$.** Searching less shrinks $a_N$ only logarithmically (weak knob); a bigger validation set shrinks $\sigma$ as $1/\sqrt{n}$ (strong knob). Re-measuring on cleaner data beats searching less.
- **Adaptivity is worse than one-shot.** Using val scores to pick your *next* move is gradient ascent on the validation noise, and it breaks the i.i.d. holdout guarantee. The creep is gradual and invisible from inside the set.
- **The only honest check is a number from data your selection never saw.** A sealed three-way split, test touched exactly once, is the free and non-negotiable defense. `val - test` is your measured optimism.
- **Count and budget your evaluations.** You cannot manage optimism you do not measure; a counter on every val/test query is the cheapest instrument here, and `test budget = 1` turns the cardinal sin into an exception.
- **Ignore sub-noise differences.** If two configs differ by less than $\sigma$ on validation, treat them as tied. Refusing to chase sub-noise gains is the reusable-holdout principle, for free, and it starves the adaptive-overfitting engine.
- **Bias concentrates, it does not cancel.** Selection maximizes rather than averages, and the maximum keeps the most extreme upward draw, so symmetric noise becomes a one-directional positive bias that grows with how hard you search.
- **A validation set is a consumable resource.** Every adaptive query spends a little of its honesty; when it is spent, only fresh data restores a trustworthy number. Treat sealed holdouts like a budget you open on a schedule and retire after one use.
- **The bug wears many outfits.** Hyperparameter sweeps, early stopping on the report set, best-of-$k$ seeds, feature selection by val score, Kaggle leaderboard chasing, and A/B-test peeking are the same phenomenon — maximizing over noisy estimates — and they all yield to the same cure: count your looks and correct for them.
- **The value of an honest number is that it lets you stop.** A true 0.85 you can trust beats a fake 0.91 that collapses on launch and triggers a hunt for a regression that never existed.

## 15. Further reading

- Dwork, Feldman, Hardt, Pitassi, Reingold, Roth, "The reusable holdout: Preserving validity in adaptive data analysis" (Science, 2015) — the reusable holdout / Thresholdout mechanism and the formal account of why adaptive analysis breaks the i.i.d. guarantee.
- Blum and Hardt, "The Ladder: A Reliable Leaderboard for Machine Learning Competitions" (ICML, 2015) — how to run a leaderboard that resists overfitting by only revealing significant improvements.
- Recht, Roelofs, Schmidt, Shankar, "Do ImageNet Classifiers Generalize to ImageNet?" (ICML, 2019) — fresh test sets for CIFAR-10 and ImageNet and what the accuracy drop does and does not tell us about community-scale test reuse.
- Cawley and Talbot, "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation" (JMLR, 2010) — the canonical treatment of selection bias from hyperparameter tuning and why nested CV is the fix.
- scikit-learn documentation, "Nested versus non-nested cross-validation" and `model_selection` — the practical APIs (`train_test_split`, `GridSearchCV`, `cross_val_score`, `StratifiedKFold`) used throughout this post.
- Within this series: [A Taxonomy of Training and Finetuning Bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) for the symptom-to-suspect decision tree; [Cross-Validation Done Wrong](/blog/machine-learning/debugging-training/cross-validation-done-wrong) for nested CV and group/time leakage; [Data Leakage: The Silent Killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer) for the other way a held-out number lies to you; and the [Training Debugging Playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) capstone for the end-to-end procedure.

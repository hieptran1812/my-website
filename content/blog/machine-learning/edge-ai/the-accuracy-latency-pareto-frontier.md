---
title: "The accuracy-latency Pareto frontier: choosing the configuration that ships"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Turn a pile of twenty quantize-prune-distill candidates into one defensible shipping decision using dominance, the Pareto frontier, the knee, and hard constraints — with runnable Python to compute and pick the point that ships."
tags:
  [
    "edge-ai",
    "model-optimization",
    "pareto-frontier",
    "multi-objective-optimization",
    "latency",
    "inference",
    "efficient-ml",
    "benchmarking",
    "decision-framework",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/the-accuracy-latency-pareto-frontier-1.png"
---

A few months ago I sat in a deployment review where an engineer had done genuinely good work and was about to make a bad decision because of it. She had taken our on-device image classifier and run every optimization in this series on it: post-training int8 quantization, quantization-aware training, three pruning ratios, two distilled student architectures, and a couple of int4 experiments. The result was a spreadsheet with twenty rows. Each row was a real, trained, exported model with a real accuracy number and a real latency number on the target phone. And the question on the table — "which one do we ship?" — had no answer, because the spreadsheet had no order. The fp16 model was the most accurate and the slowest. The int4-plus-pruning model was the fastest and the least accurate. Everything in between traded one for the other, and the room was about to pick a model by argument: by who spoke last, by which number felt impressive, by the manager's gut. Twenty configurations, zero criteria.

This is the moment the entire "Optimizing AI Models for the Edge" series has been building toward. Every lever we have covered — [quantization](/blog/machine-learning/edge-ai/post-training-quantization-ptq), pruning, [distillation](/blog/machine-learning/edge-ai/knowledge-distillation-fundamentals), efficient architecture — does exactly one thing in the end: it produces a *configuration* with an accuracy and a set of costs. None of those levers gives you "the best model." They give you *points*. The decision is never "best model" in the abstract; it is "best point on the trade-off curve for *my* device, *my* latency budget, *my* memory ceiling, *my* accuracy floor." The taxonomy post called this the [accuracy-efficiency Pareto frontier](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) and used it as a picture. This post makes it a tool — the concrete, runnable decision procedure that turns twenty rows into one shipping call you can defend in a design review and to yourself at 2 a.m. when the pager goes off.

![Before and after comparison showing twenty unordered candidate configurations collapsing into a small non-dominated Pareto frontier with a highlighted knee and a single shipped pick](/imgs/blogs/the-accuracy-latency-pareto-frontier-1.png)

By the end you will be able to do five things. Define dominance precisely and use it to delete configurations that can never be right. Compute the Pareto frontier — the non-dominated set — from a sweep, in a few lines of Python. Find the *knee* of the frontier, the point of maximum curvature where accuracy-per-millisecond stops being a bargain, with an actual criterion rather than eyeballing. Turn your hardware limits into a feasible region and pick the best point inside it. And run the whole thing as a repeatable workflow: define constraints, sweep, filter to the frontier, pick the knee in the feasible region, validate on the real device. That workflow is the recurring spine of this entire series, finally made explicit. Let us build it from the ground up.

## The core reframe: you are not choosing a model, you are choosing a point

Hold onto this sentence, because it reorganizes everything: **an optimization technique does not produce a better model, it produces a different point in a trade-off space, and your job is to choose a point.** When you quantize fp16 to int8, you do not "improve" the model — you move it: smaller, faster, slightly less accurate. When you prune 30% of the channels and fine-tune, you move it again. Each knob is a way to slide along or across a surface where the axes are the things you care about: accuracy on one axis, and on the others latency, model size, peak memory, energy per inference, and ultimately dollars.

So the right way to picture a sweep is not "a list of models, one of which is best," but "a cloud of points in a multi-dimensional space, most of which are obviously bad, a few of which are genuinely interesting, and exactly one of which is the right ship for a given constraint." The discipline that studies exactly this — optimizing several conflicting objectives at once — is called **multi-objective optimization**, and edge ML is one of its most natural applications. We will borrow three ideas from it: *dominance* (which lets us delete points), the *Pareto frontier* (the set of points that survive), and *scalarization* and *constraints* (two ways to pick a single winner from the survivors). None of this is exotic math; all of it is the difference between a defensible decision and a coin flip.

Let me pin down the running example so every number in this post traces to something concrete. The spine is the same small image classifier the rest of the series uses — call it a MobileNet-class model trained for a 10-class on-device vision task, with an fp32 baseline accuracy of 92.6% and an fp32 latency of about 41 ms at the 99th percentile (p99) on the target, a mid-range phone NPU running through a mobile runtime at batch 1. Every "configuration" below is one of: a bit-width choice (fp16, int8, int4), a pruning ratio (0%, 30%, 50%), or an architecture variant (the base net or a slightly narrower distilled student). We sweep combinations of those, measure four numbers for each — top-1 accuracy, p99 latency, model size in MB, energy per inference in millijoules — and then make the frontier do the deciding. All latency numbers below are batch-1 p99 unless I say otherwise, because [batch-1 tail latency is the number the edge actually feels](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device), and a frontier built on the wrong metric picks the wrong winner with full confidence.

## Dominance: the operation that deletes the obvious losers

Before you can find the best point, you can cheaply delete the points that are *strictly worse than something else on the table*. That is the single most useful idea in this whole post, and it has a precise definition.

Say we are *maximizing* accuracy and *minimizing* latency, size, and energy. To make the math uniform, flip the minimization axes by negating them, so that for every objective "bigger is better." Then a configuration is a vector of objectives $\mathbf{x} = (x_1, x_2, \dots, x_m)$ where $x_1$ is accuracy, $x_2 = -\text{latency}$, $x_3 = -\text{size}$, $x_4 = -\text{energy}$, and so on. Now:

$$
\mathbf{a} \succ \mathbf{b} \quad\Longleftrightarrow\quad \big(\forall i:\ a_i \ge b_i\big)\ \wedge\ \big(\exists j:\ a_j > b_j\big).
$$

In words: **configuration A dominates configuration B if A is at least as good as B on every objective and strictly better on at least one.** If A dominates B, then B is *never the right answer* — for any weighting of the objectives, for any constraint, A is preferable or tied. You can delete B from consideration unconditionally. That unconditional part is what makes dominance so powerful: you do not need to know your preferences yet to throw B away. You just need to know B is beaten on every front.

![Before and after comparison of two configurations showing config A beating config B on accuracy and latency while tying on size, which makes A dominate B](/imgs/blogs/the-accuracy-latency-pareto-frontier-2.png)

The figure makes it concrete. Config B sits at 91.0% accuracy, 19 ms p99, 24 MB. Config A sits at 92.3% accuracy, 14 ms p99, 24 MB. A is more accurate, faster, and the same size. On not one single axis is B better. So B is dominated, and no argument — not "but B was easier to train," not "B is the one we already validated" — changes the fact that *as a shipping artifact* B is strictly worse than something else you already have. Dominance lets you say that with a straight face in a meeting.

A few subtleties that matter in practice. First, **ties on an axis do not break dominance** — A dominated B above even though their sizes were identical, because A was strictly better elsewhere and no worse on size. The definition only needs strictly-better on *one* axis and no-worse on the rest. Second, **dominance is a partial order, not a total order.** Two configs where A is more accurate but B is faster are *incomparable* — neither dominates the other, and both survive. That incomparability is exactly why a trade-off curve exists at all; if dominance were total, there would be one global best and no decision to make. Third, **the axes you include change the partial order.** If you only look at accuracy and latency, B might be dominated; add energy as a third axis and B might suddenly be non-dominated because it happens to be the most energy-frugal. Choosing your objectives is therefore not a formality — it determines what survives. Include every axis your product actually cares about, and *only* those, because a spurious axis can rescue a junk configuration.

Here is the dominance test in code, written to be obvious rather than clever:

```python
import numpy as np

# Each config is (accuracy, latency_ms, size_mb, energy_mj).
# Accuracy: higher is better. The other three: lower is better.
# We orient everything so "higher is better" by negating the cost axes.
def orient(configs):
    c = np.asarray(configs, dtype=float)
    signs = np.array([+1.0, -1.0, -1.0, -1.0])  # max acc, min lat/size/energy
    return c * signs

def dominates(a, b):
    """True if oriented vector a dominates b (>= everywhere, > somewhere)."""
    return np.all(a >= b) and np.any(a > b)

# Sanity check on the figure's two configs.
A = orient([(92.3, 14, 24, 130)])[0]
B = orient([(91.0, 19, 24, 150)])[0]
print(dominates(A, B))   # True  -> B is deletable
print(dominates(B, A))   # False -> A is not dominated by B
```

That is the atomic operation. Everything that follows is built on it.

## The Pareto frontier: the set that survives

Run the dominance test against the whole sweep and a beautiful thing happens: most of the points fall away, and what remains is the **Pareto frontier** (also called the Pareto set, the Pareto-optimal set, or the non-dominated set). Formally, for a set of configurations $S$,

$$
\mathcal{P}(S) = \{\, \mathbf{x} \in S \ :\ \nexists\, \mathbf{y} \in S \text{ with } \mathbf{y} \succ \mathbf{x} \,\}.
$$

A configuration is on the frontier if and only if nothing else in the sweep dominates it. Every member of the frontier represents a real trade-off you might genuinely want; every non-member is a strictly worse version of some frontier point and is safe to delete. The frontier is named after Vilfredo Pareto, an economist who in the 1890s formalized the idea of an allocation that cannot be improved for one party without hurting another — exactly the structure of "cannot get more accurate without getting slower."

The practical payoff is dramatic. In the opening story, twenty configs collapsed to seven on the frontier and thirteen dominated points that could be deleted without a second look. That is the move the first figure shows: a config-sprawl problem with no ranking becomes a small ordered set where comparison is finally meaningful, because every remaining point earns its place by being the best at *something*. You went from "which of twenty?" — an unanswerable question — to "which of seven, and on what criterion?" — an answerable one.

Computing the frontier naively is the obvious double loop: for each point, check whether any other point dominates it. That is $O(n^2 m)$ for $n$ configs and $m$ objectives, which is completely fine for the scale of any real sweep (you will have dozens of configs, maybe low hundreds, never millions). Do not reach for a fancy algorithm here; the simple filter is correct and instant.

```python
def pareto_front(configs):
    """Return indices of the non-dominated (Pareto-optimal) configurations."""
    pts = orient(configs)
    n = len(pts)
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        if not on_front[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if dominates(pts[j], pts[i]):
                on_front[i] = False
                break
    return np.where(on_front)[0]

# The five headline configs from the running example (acc, lat, size, energy).
sweep = {
    "fp16 dense":      (92.4, 38, 44, 410),
    "int8 dense":      (92.1, 14, 22, 155),
    "int8 30% prune":  (91.6, 11, 16, 120),
    "int4 dense":      (90.2, 13, 12, 110),
    "int4 50% prune":  (88.9,  9,  9,  85),
}
names = list(sweep)
front_idx = pareto_front([sweep[k] for k in names])
print([names[i] for i in front_idx])
# -> ['fp16 dense', 'int8 dense', 'int8 30% prune', 'int4 50% prune']
# 'int4 dense' is dominated by 'int8 30% prune': worse acc, worse latency,
# only better on size by 4 MB but 'int4 50% prune' is smaller still.
```

Look at what the filter did to `int4 dense`. It is dominated by `int8 30% prune`, which is *more accurate (91.6 vs 90.2), faster (11 vs 13 ms)*, and only loses on size by 4 MB — and if size is what you cared about, `int4 50% prune` is smaller still at 9 MB. So `int4 dense` is the worst of both worlds: not accurate enough to compete with int8, not small enough to compete with the int4-pruned tier. The frontier surfaces that immediately. Without it, someone would have argued for `int4 dense` because "int4 sounds aggressive and modern," and they would have shipped a strictly dominated model.

### A subtlety: noise can corrupt the frontier

One honest caveat before we go further. Dominance is *exact*, but your measurements are *noisy*. If two configs are within measurement error on every axis, declaring one dominated by the other is an artifact of noise, not a real ordering. The fix is to dominate with a tolerance: require A to beat B by more than the measurement noise $\epsilon_i$ on the strict axis before you delete B.

$$
\mathbf{a} \succ_{\epsilon} \mathbf{b} \quad\Longleftrightarrow\quad \big(\forall i:\ a_i \ge b_i - \epsilon_i\big)\ \wedge\ \big(\exists j:\ a_j > b_j + \epsilon_j\big).
$$

Concretely: if your p99 latency measurement has a $\pm 0.5$ ms run-to-run spread and your accuracy has a $\pm 0.2$ pt confidence interval, a config that is "faster by 0.3 ms" is not really faster. Treat such pairs as incomparable and keep both on the frontier until a tiebreaker (energy, size, engineering simplicity) separates them. This is why [honest measurement](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) is a prerequisite for this whole exercise: a frontier built on flaky numbers will delete the wrong points.

## The knee: where accuracy stops being worth the milliseconds

The frontier tells you which points are viable, but it does not, by itself, tell you which one to ship. Walk along it from cheap-and-inaccurate toward expensive-and-accurate and you will feel the shape of the trade: at first every extra millisecond of latency buys you a lot of accuracy, and then, at some point, the curve bends and flattens — each additional millisecond buys you almost nothing. That bend is the **knee** of the frontier, and it is, in the great majority of cases, the point you want to ship: the place where you have captured nearly all the accuracy that latency can buy, just before diminishing returns set in.

![Before and after comparison of the frontier below and above the knee showing a steep one point per millisecond region giving way to a flat region where latency buys almost no accuracy](/imgs/blogs/the-accuracy-latency-pareto-frontier-3.png)

The figure quantifies it on our example. Below the knee, moving from 8 ms to 14 ms gains about 6 accuracy points — roughly 1.0 point per millisecond, a fantastic exchange rate. Above the knee, moving from 14 ms to 40 ms gains only about 1.1 points — about 0.04 points per millisecond, a terrible one. The knee sits around 14 ms, at the int8-dense config. Below it you are leaving easy accuracy on the table; above it you are burning latency (and energy, and battery, and your thermal budget) to chase fractions of a point that no user will ever notice. Shipping past the knee is the single most common over-engineering mistake in edge ML: people optimize for the leaderboard's accuracy column and pay for it forever in the field.

### Deriving a knee criterion

"Where the curve bends" is intuition; let us make it a number you can compute. Model the frontier as accuracy $a$ as a function of latency $\ell$, an increasing, concave curve $a = f(\ell)$. The *marginal value* of latency is the slope $f'(\ell)$ — accuracy gained per extra millisecond. Diminishing returns means $f'$ is decreasing, i.e. $f''(\ell) < 0$. The knee is where the curve is *most sharply bending*, which is the point of **maximum curvature**. Curvature for a plane curve is

$$
\kappa(\ell) = \frac{|f''(\ell)|}{\big(1 + f'(\ell)^2\big)^{3/2}},
$$

and the knee is $\ell^\star = \arg\max_\ell \kappa(\ell)$. That is the textbook definition, and it is exact, but it has a practical flaw: curvature depends on the *scale* of your axes. If you measure latency in microseconds instead of milliseconds, $f'$ changes by a factor of a thousand and the curvature maximum moves. A frontier is a set of points with incommensurable units (percent vs ms vs MB), so raw curvature is unit-sensitive and a little dangerous.

The robust, scale-free fix that I actually use is the **normalized-distance ("Kneedle"-style) criterion**. Normalize both axes to $[0, 1]$ so units drop out. Sort the frontier points by latency, normalize accuracy and latency each to their min-max range, and draw the straight chord from the worst-accuracy end to the best-accuracy end. The knee is the frontier point with the **maximum perpendicular distance below that chord** — the point that bulges farthest from the line connecting the extremes. Algebraically, with normalized points $(\tilde\ell_i, \tilde a_i)$ and the chord running from the first to the last point, the knee maximizes

$$
d_i = \tilde a_i - \Big[\tilde a_1 + (\tilde a_n - \tilde a_1)\,\frac{\tilde\ell_i - \tilde\ell_1}{\tilde\ell_n - \tilde\ell_1}\Big],
$$

the vertical gap between the actual frontier and the straight-line interpolation. The point with the largest gap is where the curve departs most from "linear trade," which is precisely the knee. This is scale-free, robust to a couple of noisy points, and trivial to compute:

```python
def knee_index(front_configs):
    """
    front_configs: list of (accuracy, latency, size, energy) ON the frontier.
    Returns the index (within front_configs) of the knee by the
    normalized perpendicular-distance criterion on the acc-vs-latency curve.
    """
    pts = np.asarray(front_configs, dtype=float)
    acc, lat = pts[:, 0], pts[:, 1]
    order = np.argsort(lat)             # walk cheap -> expensive
    acc, lat = acc[order], lat[order]

    def norm(v):
        return (v - v.min()) / (v.ptp() + 1e-12)

    na, nl = norm(acc), norm(lat)
    # chord from the lowest-latency frontier point to the highest
    chord = na[0] + (na[-1] - na[0]) * (nl - nl[0]) / (nl[-1] - nl[0] + 1e-12)
    dist = na - chord                   # vertical gap above the chord
    knee_local = int(np.argmax(dist))
    return int(order[knee_local])       # map back to input ordering

front = [sweep[k] for k in ["int4 50% prune", "int8 30% prune",
                            "int8 dense", "fp16 dense"]]
ki = knee_index(front)
print(ki)  # -> 2  (int8 dense): the knee of this frontier
```

The knee is not a law of nature; it is a *default*. There are good reasons to ship off the knee — a hard accuracy floor can push you above it, a brutal latency cap can push you below it — but absent a binding constraint, the knee is where accuracy-per-millisecond is most favorable and is the right place to start the argument. Treat "ship the knee unless a constraint forbids it" as your prior.

### Worked example: from sweep to knee to shipped pick

#### Worked example: the twenty-config spreadsheet, resolved

Let us run the opening story all the way to a decision. The full sweep had twenty rows; here are the five that ended up mattering, with all four measured axes (top-1 accuracy in percent, p99 latency in ms at batch 1 on the mid-range phone NPU, on-disk size in MB, energy in millijoules per inference):

| Config | Accuracy | p99 latency | Size | Energy | On frontier? |
|---|---|---|---|---|---|
| fp16 dense | 92.4% | 38 ms | 44 MB | 410 mJ | yes (top) |
| int8 dense | 92.1% | 14 ms | 22 MB | 155 mJ | yes (knee) |
| int8 + 30% prune | 91.6% | 11 ms | 16 MB | 120 mJ | yes |
| int4 dense | 90.2% | 13 ms | 12 MB | 110 mJ | dominated |
| int4 + 50% prune | 88.9% | 9 ms | 9 MB | 85 mJ | yes (tiny end) |

Step one, **dominance filter**: `int4 dense` falls. It is beaten by `int8 + 30% prune` on accuracy and latency, and the only axis where it leads — size — is owned more decisively by `int4 + 50% prune`. The other fifteen rows of the original twenty were either near-duplicates or similarly dominated; the filter cleared them in one pass. Four configs survive on the frontier.

Step two, **knee**: applying the normalized-distance criterion to the four-point frontier (accuracy vs latency) puts the knee at `int8 dense` — 92.1% at 14 ms. Below it, the 9 ms tiny config costs 3.2 accuracy points to save 5 ms; above it, the 38 ms fp16 config costs 24 ms to gain 0.3 points. The exchange rate is best right at int8 dense.

Step three, **constraint check**: the product requirement is "p99 under 16 ms, on-disk size under 30 MB." The knee config (14 ms, 22 MB) passes both. Done — we ship `int8 dense`. The decision took four lines of reasoning and is fully defensible: it is on the frontier (so it is not dominated by anything), it is feasible (so it meets the product limits), and it is the knee (so it captures the best accuracy-per-millisecond). The fp16 model that "looked best" because it had the top accuracy number was a trap: 24 extra milliseconds of p99 latency and almost three times the energy for three-tenths of an accuracy point. The frontier and the knee made the right call mechanical.

That is the whole method in one example. The rest of this post adds the machinery you need to apply it when the constraints get tighter, the objectives get more numerous, and two different devices want two different answers from the same data.

## The geometry of dominance: why the frontier has the shape it does

It is worth slowing down on *why* a frontier looks the way it does, because the shape is not arbitrary — it falls directly out of the dominance relation, and understanding it stops you from mistaking measurement noise for structure. Plot every config as a point in the accuracy-vs-latency plane, with accuracy up and latency to the right. A point dominates everything in the rectangle *up and to the left of it* (more accurate, faster). So a point is on the frontier exactly when no other point lies in its up-and-left rectangle. Sweep your eye from the top-left (the unreachable ideal: perfect accuracy, zero latency) and the frontier is the staircase of points you hit first — the outermost shell of the cloud facing the ideal corner. Everything behind that shell is dominated by some point on it.

That staircase picture explains three things at once. First, **the frontier is monotone**: as you move right (more latency) along it, accuracy only goes up, never down — because a point that was both slower *and* less accurate than a frontier point would be dominated and could not be on the frontier. Second, **the frontier is the boundary of the dominated region**, which is why adding a dominated config never changes the frontier: it lands strictly inside the shell. Third, the frontier can be **convex or non-convex**, and the difference is exactly the difference between a smooth trade and a lumpy one. A convex frontier bulges smoothly toward the ideal corner; a non-convex frontier has dents where the trade-off rate jumps. Architecture changes — swapping a block, changing the width multiplier — produce the lumps, because they move accuracy and latency by discrete, uncorrelated amounts rather than along a smooth dial. This is precisely why, as we will see, the weighted-sum method (which can only find the convex hull of the frontier) is dangerous: the lumps are often where the interesting configs live.

There is a clean way to state the marginal trade at any frontier point: the *local slope* $\Delta\text{accuracy} / \Delta\text{latency}$ between a point and its neighbor on the frontier is the **exchange rate** — how many accuracy points one more millisecond buys you *right here*. On a well-behaved (concave-up-toward-the-ideal) frontier this exchange rate decreases monotonically as you move toward higher latency: the first milliseconds are cheap accuracy, the last ones are expensive. That monotone-decreasing exchange rate is exactly what creates a single, well-defined knee. When the exchange rate is *not* monotone — when it dips and rises because of architecture lumps — there can be more than one local knee, and the global knee (the one the normalized-distance criterion finds) is the bend that departs most from the straight chord across the whole frontier. Knowing this, you read a frontier plot the way a trader reads a price curve: the slope is the marginal price, the knee is where the marginal price collapses, and the dents are arbitrage that a linear score would miss.

## A third objective changes the frontier: adding energy

Two axes fit on a page, but the moment a third objective genuinely binds, the frontier reshapes and configs that looked dominated can come back to life. Energy is the usual third axis on battery-powered devices, so let us add it explicitly and watch what happens.

Take the running sweep but now care about energy per inference as a first-class objective, not just a tiebreaker. In the two-axis (accuracy, latency) view, `int4 dense` was dominated — beaten by `int8 + 30% prune` on both accuracy and latency. But look at the three-axis view (accuracy, latency, energy): `int4 dense` draws 110 mJ, slightly *less* than `int8 + 30% prune`'s 120 mJ. Does that rescue it? Run the dominance test in 3-D: `int8 + 30% prune` is more accurate (91.6 vs 90.2), faster (11 vs 13 ms), *but now loses on energy* (120 vs 110 mJ). They no longer dominate each other — they are *incomparable* in 3-D, so `int4 dense` returns to the frontier. The lesson is sharp: **adding an axis can only ever grow the frontier, never shrink it.** Every config that was non-dominated stays non-dominated (it was already best at something), and some previously dominated configs are rescued because they turn out to be best at the new axis. This is the curse-of-dimensionality-for-frontiers effect made concrete in five rows.

That cuts both ways, and it is why axis selection is a real decision rather than a formality. If your product has no energy budget — say it is a plugged-in edge gateway with a fan — then including energy as an objective is actively harmful: it rescues `int4 dense`, a config you have no reason to want, and clutters the decision with a point that is "best at" something nobody is paying for. The discipline is to include an axis *only if a stakeholder will be held to it*. Energy belongs on the frontier for a phone or a battery-powered sensor; it does not belong on the frontier for a mains-powered box. The objectives are part of the problem statement, and over-specifying them inflates the frontier with junk just as surely as under-specifying them hides the constraint that bites you.

#### Worked example: when energy is the binding axis

Picture a battery-powered wildlife camera that fires the classifier on motion, a few hundred times a day, on a coin cell it must not exhaust before the next servicing. Latency is irrelevant — nobody is waiting on a deer — so there is no latency cap. Size is comfortable; the flash is generous. The *only* binding constraint is energy: each inference must draw under 100 mJ to hit the battery-life target. Apply the mask to the frontier: `fp16 dense` (410 mJ), `int8 dense` (155 mJ), and `int8 + 30% prune` (120 mJ) all bust the 100 mJ cap. Only the configs at the very frugal end survive: `int4 + 50% prune` at 85 mJ and the distilled `student-int8 + 30% prune` at 100 mJ (right at the cap). Within that two-config feasible frontier, the student is more accurate (91.2% vs 88.9%) at the energy ceiling, so it ships. Same frontier, a third device, and now the *energy* axis — invisible in the latency-driven examples — is the one that decides. The accuracy cost of the energy wall here is the full 1.2 points between the 92.1% knee and the 91.2% frugal pick, and that number is exactly what you take to the product owner when they ask "what does the battery target cost us in accuracy?"

## Constraints turn the frontier into a feasible region

So far we have treated all frontier points as eligible. In reality your hardware imposes *hard limits*, and a point that violates one is not a worse choice — it is *no choice at all*. A model whose peak activation memory does not fit in the device's SRAM does not run. A model whose p99 latency blows the frame budget drops frames. A model bigger than the app's OTA-update size limit cannot ship. These are not soft preferences; they are constraints, and they carve the objective space into a **feasible region** (points that satisfy every constraint) and an infeasible region (everything else).

![Matrix showing four frontier configurations checked against a p99 latency cap and a size cap, with two passing both to land in the feasible region and two rejected](/imgs/blogs/the-accuracy-latency-pareto-frontier-4.png)

Formally, a constraint is a predicate $g_k(\mathbf{x}) \le 0$ (or $\ge$, or an equality), and the feasible set is

$$
\mathcal{F} = \{\, \mathbf{x} \ :\ g_k(\mathbf{x}) \le 0 \ \text{ for all } k \,\}.
$$

The decision becomes: **find the best point on the frontier that also lies in the feasible region**, i.e. pick the knee (or your chosen objective optimum) over $\mathcal{P}(S) \cap \mathcal{F}$, not over all of $\mathcal{P}(S)$. The figure shows the carve in action: with a cap of "p99 under 16 ms and size under 30 MB," two of four frontier configs pass both caps (they are ship-eligible) and two fail (one busts the latency cap, one busts both). The feasible region is the green band, and the pick must come from there.

This is where constraints and the knee interact, and the interaction has a clean rule. Find the knee of the *full* frontier first. If the knee is feasible, ship it. If the knee is *infeasible* — say the latency cap is tighter than the knee's latency — then walk down the frontier toward cheaper configs until you re-enter the feasible region, and ship the *most accurate feasible point*, which will be the feasible point nearest the (infeasible) knee. Symmetrically, if a hard accuracy floor sits above the knee, walk up the frontier to the cheapest config that clears the floor. The knee is your anchor; constraints slide you off it only as far as they must.

In code, constraints are just a mask you apply before picking:

```python
def feasible_mask(configs, p99_cap=None, size_cap=None,
                  energy_cap=None, acc_floor=None):
    c = np.asarray(configs, dtype=float)  # (acc, lat, size, energy)
    ok = np.ones(len(c), dtype=bool)
    if acc_floor   is not None: ok &= c[:, 0] >= acc_floor
    if p99_cap     is not None: ok &= c[:, 1] <= p99_cap
    if size_cap    is not None: ok &= c[:, 2] <= size_cap
    if energy_cap  is not None: ok &= c[:, 3] <= energy_cap
    return ok

def choose(configs, names, **caps):
    cfg = [configs[n] for n in names]
    front = pareto_front(cfg)                       # indices on frontier
    mask  = feasible_mask([cfg[i] for i in front], **caps)
    eligible = front[mask]
    if len(eligible) == 0:
        raise ValueError("No feasible config on the frontier. Loosen a cap "
                         "or sweep more aggressive configs.")
    # knee within the feasible frontier subset
    sub = [cfg[i] for i in eligible]
    k = knee_index(sub)
    return names[eligible[k]]

pick = choose(sweep, names, p99_cap=16, size_cap=30)
print(pick)  # -> 'int8 dense'  (knee, and it is feasible)
```

Notice the explicit failure mode: if the feasible frontier is *empty*, the function raises rather than silently returning garbage. An empty feasible set is real information — it means no configuration in your sweep meets the requirements, and your options are to loosen a constraint (negotiate the product spec) or to sweep more aggressive configs (go to int4, prune harder, distill to a smaller student, switch architecture). Pretending a near-miss config is "close enough" by quietly relaxing the cap is how teams ship models that drop frames in the field. Make infeasibility loud.

#### Worked example: the same frontier, two devices, two picks

This is the payoff that turns the frontier from a one-off into a reusable asset. We built one frontier; now two product teams want to ship the model on two very different targets, and they should get two different answers from the same data.

![Before and after comparison showing a flagship NPU with loose limits selecting the high accuracy int8 dense knee and a Cortex-M7 with a tight memory cap selecting the tiny int4 pruned config from the identical frontier](/imgs/blogs/the-accuracy-latency-pareto-frontier-7.png)

Device one is a flagship phone NPU. Its constraints are loose: p99 under 25 ms (plenty of frame budget) and size under 64 MB (plenty of flash). Apply the mask: every frontier config except `fp16 dense` (38 ms busts the 25 ms cap) is feasible. The knee of the feasible subset is `int8 dense` — 92.1% at 14 ms. The flagship ships the high-accuracy knee, because it can afford it.

Device two is a Cortex-M7 microcontroller class target — a microcontroller, no NPU, a hard ceiling on storage. Its binding constraint is memory: size under 10 MB. Latency is generously capped at 60 ms because the use case is a once-per-second classification, not a video stream. Apply the mask: only `int4 + 50% prune` (9 MB) fits under the 10 MB cap; every other frontier config is too big. The feasible frontier has exactly one member, so that is the pick — 88.9% at 9 ms, 9 MB. The microcontroller ships the tiny end of the frontier, because the device's memory wall leaves no other feasible point.

Same sweep. Same frontier. Same knee computation. Two devices, two binding constraints, two correct-and-different shipped configs. This is why you compute the frontier once and reuse it: the expensive part (training and measuring the configs) is done once, and each new target is just a different mask applied to the same non-dominated set. When the [memory wall is the real constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint), the frontier tells you instantly how much accuracy that wall costs you — here, 3.2 points to drop from the flagship's knee to the microcontroller's only feasible point.

## Scalarization vs the constraint method, and the trap in the weighted sum

There are two classical ways to collapse a multi-objective problem to a single winner, and the difference between them is not academic — picking the wrong one will quietly hide good configurations from you.

The first is **scalarization by weighted sum**: assign a weight $w_i \ge 0$ to each objective, compute a single score $s(\mathbf{x}) = \sum_i w_i x_i$ (objectives oriented so bigger is better), and pick the config with the highest score. It feels natural — "accuracy is twice as important as latency, so weight it 2:1" — and it is what most people reach for first. It is also the source of a notorious failure.

**The weighted sum cannot find points in concave regions of the frontier.** Here is the geometric reason, and it is worth understanding because it explains a real class of "why did my tuner never pick that config?" bugs. A weighted-sum objective $s(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x}$ is a linear function, so its level sets are straight lines (hyperplanes in higher dimensions). Maximizing it means sliding that line outward until it last touches the feasible set — and a straight line can only touch a *convex* boundary at the points where the frontier bulges outward. If part of your frontier curves *inward* (a concave dent), no straight line will ever be tangent there; the line will always touch a convex extreme first. So configs sitting in a concave part of the frontier are *invisible* to the weighted sum — for every weighting, some convex-region config scores at least as high. You can sweep the weights through all of $[0,1]$ and never once select the concave-region point, even though it might be exactly the trade you want.

This is not a corner case. Accuracy-latency frontiers in deep learning are frequently non-convex, because architecture changes produce lumpy jumps: a config can be a genuinely good trade-off and still sit in a concave dent between two convex neighbors. If you tune a deployment by weighted-sum score, you will systematically overlook those points. I have watched a team spend a week sweeping loss weights and conclude "there is no good middle config" when the good middle config was right there on the frontier, hidden in a concave region the linear score could not see.

The second method, and the one I recommend as the default, is the **$\epsilon$-constraint method**: optimize *one* objective (say accuracy) subject to hard caps on the others ($\text{latency} \le \epsilon_1$, $\text{size} \le \epsilon_2$, $\text{energy} \le \epsilon_3$). Formally,

$$
\max_{\mathbf{x}} \ x_{\text{acc}} \quad\text{subject to}\quad \text{latency}(\mathbf{x}) \le \epsilon_1,\ \text{size}(\mathbf{x}) \le \epsilon_2,\ \dots
$$

This is exactly the feasible-region picture from the previous section: caps define the feasible set, and you maximize your primary objective within it. The crucial property is that **the $\epsilon$-constraint method can reach every point on the frontier, convex or concave.** Because the constraints are box-shaped rather than a sliding hyperplane, you can drive the caps to any frontier point, including the ones in concave dents. It also maps directly onto how products actually specify requirements — "must run under 16 ms, must fit in 30 MB, maximize accuracy" is an $\epsilon$-constraint statement verbatim, not a weighting. That alignment with real specs is a second reason to prefer it: you do not have to invent fictitious weights for things that are really hard limits.

So my standing advice: **express your problem as $\epsilon$-constraints (hard caps on costs, maximize the one objective you actually care about), not as a weighted sum.** Use the weighted sum only when you genuinely have a smooth preference trade with no hard limits and you have confirmed your frontier is convex — which on real edge frontiers is rarely true. The table below summarizes the choice.

| Method | How you express preference | Reaches concave frontier points? | Matches real specs? | Main pitfall |
|---|---|---|---|---|
| Weighted sum | weights $w_i$ on each objective | No — blind to concave dents | Poorly (needs fake weights) | Silently hides good configs |
| $\epsilon$-constraint | hard caps on costs, max one objective | Yes — reaches every frontier point | Directly (caps = product limits) | Must pick the primary objective |
| Knee on frontier | none — pure curvature | Yes (knee can be anywhere) | Indirectly (validate vs caps) | Knee can be infeasible |

In practice I combine the last two: use $\epsilon$-constraints to define the feasible region, then pick the knee *within* it. That gives you a defensible, spec-aligned, concave-safe decision every time.

## Hypervolume: scoring a whole frontier, not a point

Sometimes you need to compare not two configs but two *frontiers* — for example, "did adding quantization-aware training to my sweep produce a better set of trade-offs than post-training quantization alone?" or, in [hardware-aware neural architecture search](/blog/machine-learning/edge-ai/hardware-aware-nas), "did this NAS run find a better frontier than the last one?" You cannot answer that by comparing single points, because each frontier is a *set*. The standard metric for the quality of a whole frontier is the **hypervolume indicator**.

The idea is clean. Pick a fixed *reference point* $\mathbf{r}$ that is worse than everything on every axis (e.g. 0% accuracy, infinite latency, oriented so the reference is the bad corner). The hypervolume of a frontier is the volume of the region *dominated by the frontier and bounded by the reference point* — the chunk of objective space that your frontier "covers." A frontier that pushes further out on every axis covers more volume, so a larger hypervolume means a better set of trade-offs. Formally, for a frontier $\mathcal{P}$ and reference $\mathbf{r}$,

$$
\text{HV}(\mathcal{P}, \mathbf{r}) = \text{Vol}\!\left(\bigcup_{\mathbf{x} \in \mathcal{P}} \big[\mathbf{r},\, \mathbf{x}\big]\right),
$$

the volume of the union of the boxes spanning from the reference point to each frontier point. In two dimensions (accuracy vs latency) it is just the total area under the frontier staircase, measured up to the reference corner. The beauty of hypervolume is that it rewards *both* getting closer to the ideal point *and* spreading out to cover more of the trade-off space — a frontier that is one excellent point scores worse than a frontier of several good points covering more ground.

Here is a 2-D hypervolume (area) computation for accuracy-vs-latency frontiers, which is all you usually need:

```python
def hypervolume_2d(front_configs, ref=(0.0, 100.0)):
    """
    Area dominated by the (accuracy, latency) frontier up to a reference
    (acc_ref, lat_ref). Higher accuracy + lower latency => larger area.
    ref is the 'bad corner': low accuracy, high latency.
    """
    pts = np.asarray(front_configs, dtype=float)[:, :2]  # acc, latency
    acc_ref, lat_ref = ref
    # keep only points better than the reference on both axes
    pts = pts[(pts[:, 0] > acc_ref) & (pts[:, 1] < lat_ref)]
    if len(pts) == 0:
        return 0.0
    # sort by latency ascending; sweep the accuracy staircase
    pts = pts[np.argsort(pts[:, 1])]
    area, prev_lat, best_acc = 0.0, lat_ref, acc_ref
    for acc, lat in pts[::-1]:                 # from slow to fast
        if acc > best_acc:
            area += (acc - best_acc) * (lat_ref - lat)
            best_acc = acc
    return area

ptq_front = [(92.1, 14), (91.6, 11), (88.9, 9)]
qat_front = [(92.5, 14), (92.0, 11), (90.1, 9)]  # QAT lifts accuracy at each ms
print(hypervolume_2d(ptq_front), hypervolume_2d(qat_front))
# QAT front has the larger hypervolume -> a strictly better trade-off set.
```

The use of hypervolume in this series is mostly diagnostic: it gives you a *single number* to say "this optimization recipe produced a better frontier than that one," which is exactly the kind of summary you want when comparing whole approaches in a design doc. It is also the metric multi-objective NAS and evolutionary methods optimize internally — they search for architectures that maximize the hypervolume of the discovered frontier, which is just "find the best *set* of trade-offs," automated. If you remember one thing about hypervolume, remember this: *a point is judged by dominance; a frontier is judged by hypervolume.*

## The frontier is a surface: more than two axes

I have been drawing accuracy vs latency because it fits on a page, but real edge decisions live in more dimensions, and the frontier is genuinely a *surface* in that higher-dimensional space, not a curve. The axes that show up in practice:

- **Accuracy** (or task metric: F1, BLEU, word error rate, mAP). The thing you are trying to preserve.
- **Latency** — and specifically p99 at batch 1, not p50 and not throughput, because [the tail is what users feel and what frame budgets must respect](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device).
- **Model size** on disk (MB), which governs OTA update size and flash budget.
- **Peak memory** at runtime (the high-water mark of activations plus weights), which is often the *hard* wall on microcontrollers — a model that does not fit in SRAM does not run, full stop.
- **Energy per inference** (mJ or mWh), which sets battery drain and, through power, the thermal ceiling that triggers throttling.
- **Dollars** — \$ per million inferences for a hosted edge service, or the bill-of-materials cost of the chip that can run your model. Sometimes the real objective is "cheapest chip that hits the accuracy floor," which is a frontier query in (accuracy, \$-per-unit) space.

The dominance and frontier machinery does not change at all when you add axes — the code above already takes a 4-vector and would take a 6-vector just as happily. What *does* change is your intuition: in high dimensions, far more points are non-dominated, because it is harder for one config to beat another on *every* one of six axes. A frontier in 2-D might be five points; the same sweep in 6-D might leave fifteen on the frontier, because each is the best at some niche combination. This is the **curse of dimensionality for frontiers**, and the practical response is discipline: include only the axes your product genuinely constrains. Adding energy as an axis when nobody has an energy budget does not give you information — it just inflates the frontier with points that are "best at" an objective no one cares about, and makes the decision harder. The objectives are part of the problem statement; choose them as carefully as you choose the configs.

When you do have many axes, the workflow is unchanged: apply the hard ones as $\epsilon$-constraints (peak memory must fit SRAM; size must fit flash; p99 must fit the frame budget), which usually collapses the high-dimensional feasible set down to a manageable handful, and then pick the knee on the *remaining* primary trade-off (typically accuracy vs latency, or accuracy vs energy) within that feasible set. Constraints are the dimensionality reducer: they turn "optimize six things at once" into "satisfy four hard limits, then trade off the two that remain."

## The collection step: instrumenting a real sweep

Everything above assumes you have the four-or-six numbers for each config. Getting those *honestly* is half the battle, and it is where most frontier analyses quietly go wrong. Here is a sweep harness that produces frontier-ready data, with the measurement discipline baked in. The structure: enumerate the configuration grid (bit-width × pruning ratio × architecture), build each artifact, then measure accuracy on a held-out set and latency/energy on the *actual device* with warm-up and percentile statistics.

```python
import itertools, time, numpy as np

def build_config(bits, prune_ratio, arch):
    """
    Return a deployable model artifact for one config. In a real pipeline this
    runs PTQ/QAT, applies pruning + fine-tune, exports to the runtime
    (ONNX Runtime / TFLite / TensorRT / GGUF) and returns a callable.
    """
    model = load_base(arch)
    if prune_ratio > 0:
        model = prune_and_finetune(model, ratio=prune_ratio)
    if bits == 16:
        artifact = export_fp16(model)
    elif bits == 8:
        artifact = export_int8_ptq(model, calib_loader)   # representative data
    elif bits == 4:
        artifact = export_int4(model, calib_loader)        # k-quant / GPTQ etc.
    return artifact

def measure_latency(artifact, sample, warmup=50, runs=300):
    """Batch-1 p50/p99 on the target. Warm up to fill caches and let the
    clock governor settle; report percentiles, not the mean."""
    for _ in range(warmup):
        artifact.run(sample)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        artifact.run(sample)
        times.append((time.perf_counter() - t0) * 1e3)  # ms
    times = np.array(times)
    return {"p50": float(np.percentile(times, 50)),
            "p99": float(np.percentile(times, 99))}

def sweep(bit_grid, prune_grid, arch_grid, eval_loader, sample):
    rows = []
    for bits, pr, arch in itertools.product(bit_grid, prune_grid, arch_grid):
        art = build_config(bits, pr, arch)
        acc = evaluate_accuracy(art, eval_loader)          # held-out top-1
        lat = measure_latency(art, sample)                  # p50/p99
        rows.append({
            "name": f"{arch}-int{bits}-p{int(pr*100)}",
            "acc": acc, "p50": lat["p50"], "p99": lat["p99"],
            "size_mb": art.size_mb(), "energy_mj": art.energy_mj(),
        })
    return rows

results = sweep(bit_grid=[16, 8, 4],
                prune_grid=[0.0, 0.3, 0.5],
                arch_grid=["base", "student"],
                eval_loader=val_loader, sample=one_input)
```

Three measurement disciplines are doing the real work in there, and skipping any of them corrupts the frontier:

1. **Warm-up before timing.** The first dozen inferences pay for cold caches, JIT/kernel selection, and a clock governor that has not yet ramped. Time those and your p99 is garbage. Discard 50 warm-up runs, then measure.
2. **Percentiles, not means.** The edge cares about p99 (and sometimes p99.9), because the tail is what drops frames and misses deadlines. A mean hides a fat tail. Always carry the percentile you will actually be held to.
3. **Measure on the target, not a proxy.** Latency, energy, and even peak memory depend on the chip, the runtime, the kernel library, and the thermal state. A frontier measured on your laptop and shipped to a phone is a frontier of the wrong device. If you must estimate, say so, and validate the final pick on hardware (that is the last step of the workflow for a reason).

There is a fourth, sneakier discipline: **measure under realistic thermal conditions.** A model that hits 14 ms p99 on a cold bench may hit 22 ms after ninety seconds of sustained use because the chip throttled. If your product runs the model continuously, benchmark continuously — run a long loop and measure the *steady-state* p99, not the cold-start p99. A frontier built on cold numbers will pick a config that is feasible on the bench and infeasible in your user's warm pocket. This is the kind of thing that turns a clean Pareto analysis into a field incident, so it is worth the extra five minutes.

## Putting it together: the full decision workflow

Here is the whole thing as one runnable pipeline, and as the workflow you should internalize. It is the recurring spine of this series, stated once, explicitly, as code and as a sequence.

![Timeline of the frontier decision workflow running left to right from defining constraints through sweeping configs filtering to the frontier picking the knee and validating on the device](/imgs/blogs/the-accuracy-latency-pareto-frontier-5.png)

The five steps, in order, are: **define constraints → sweep configs → filter to the frontier → pick the knee in the feasible region → validate on the device.** Define constraints *first*, before you sweep, because the constraints determine which configs are even worth building — there is no point training an fp16 model you already know busts the latency cap by 2x (unless you want it as a frontier reference point, which is fine). Sweep across the levers that move the trade-off: bit-width, pruning ratio, architecture, and any others (KV-cache precision, activation quantization, etc.). Filter to the frontier with the dominance test. Pick the knee within the feasible region using the $\epsilon$-constraints. And then — non-negotiably — *validate the chosen config on the real device under realistic conditions*, because the whole sweep is only as trustworthy as its measurements, and the one config you are about to ship deserves a careful re-measurement before it goes live.

```python
def ship_decision(results, p99_cap=None, size_cap=None,
                  energy_cap=None, acc_floor=None):
    """End-to-end: sweep rows -> frontier -> feasible -> knee -> pick."""
    names = [r["name"] for r in results]
    cfg = {r["name"]: (r["acc"], r["p99"], r["size_mb"], r["energy_mj"])
           for r in results}
    # 1. frontier
    front = pareto_front([cfg[n] for n in names])
    front_names = [names[i] for i in front]
    # 2. feasible subset
    mask = feasible_mask([cfg[n] for n in front_names],
                         p99_cap=p99_cap, size_cap=size_cap,
                         energy_cap=energy_cap, acc_floor=acc_floor)
    eligible = [n for n, ok in zip(front_names, mask) if ok]
    if not eligible:
        raise ValueError("Empty feasible frontier: loosen a cap or sweep "
                         "more aggressive configs.")
    # 3. knee within feasible frontier
    sub = [cfg[n] for n in eligible]
    pick = eligible[knee_index(sub)]
    return {
        "frontier": front_names,
        "feasible": eligible,
        "shipped": pick,
        "metrics": dict(zip(["acc", "p99", "size_mb", "energy_mj"], cfg[pick])),
    }

decision = ship_decision(results, p99_cap=16, size_cap=30)
print(decision["shipped"], decision["metrics"])
# Now: VALIDATE decision['shipped'] on the device before shipping.
```

The pick-a-config logic, drawn as a decision tree, makes the branch structure obvious: which constraint *binds* (latency, memory, or accuracy floor) decides which branch you take, and the knee rule decides the leaf within that branch.

![Decision tree rooted at the feasible frontier branching by which constraint binds first into latency memory and accuracy branches each leading to a specific shipped configuration leaf](/imgs/blogs/the-accuracy-latency-pareto-frontier-8.png)

Read the tree as: start at the feasible frontier; ask which constraint binds first; follow that branch to the config it forces. If latency binds hard, you ship the fast end (int4-pruned at 9 ms). If memory binds hard, you ship the small end (int4-pruned, 9 MB tier). If neither binds and only the accuracy floor matters, you ship the knee (int8 dense, 14 ms, 92.1%). In the common case where nothing binds tightly, the accuracy branch wins and you ship the knee — which is why "ship the knee unless a constraint forbids it" is the right default. The tree is not a different method; it is the same feasible-region-then-knee logic, drawn so the branch you are on is legible at a glance.

## Results: the config-sweep and constraint-to-config tables

Let me lay out the two tables that make a frontier analysis legible to a reviewer who was not in the weeds with you. The first is the **config sweep**: every config you measured, its four axes, and its frontier status.

![Matrix of five configurations each annotated with accuracy p99 latency size and a frontier status tag marking the top the knee and the dominated points](/imgs/blogs/the-accuracy-latency-pareto-frontier-6.png)

| Config | Accuracy | p99 latency | Size | Energy | On frontier? |
|---|---|---|---|---|---|
| fp16 dense (base) | 92.4% | 38 ms | 44 MB | 410 mJ | yes — top, slowest |
| int8 dense (base) | 92.1% | 14 ms | 22 MB | 155 mJ | **yes — knee** |
| int8 + 30% prune | 91.6% | 11 ms | 16 MB | 120 mJ | yes |
| int4 dense (base) | 90.2% | 13 ms | 12 MB | 110 mJ | no — dominated |
| int4 + 50% prune | 88.9% | 9 ms | 9 MB | 85 mJ | yes — tiny end |
| student-int8 + 30% prune | 91.2% | 9 ms | 14 MB | 100 mJ | yes — distilled |

I have added the distilled student row to show how a *fifth* config from a different lever ([distillation](/blog/machine-learning/edge-ai/distillation-case-studies-distilbert-to-cnns)) lands on the frontier: at 91.2% and 9 ms it dominates `int4 + 50% prune` on accuracy at the same latency, and it sits between the int8 configs and the tiny end. The frontier does not care *how* a config was produced — quantization, pruning, distillation, architecture search all just produce points, and the frontier ranks them on the same footing. That is the unifying power of the frame: every lever in this series feeds one frontier.

The second table is the **constraint-to-config** decision summary, the artifact you actually put in the deployment doc. It says, for each target's binding constraint, which config ships and why.

| Target | Binding constraint | Feasible frontier picks | Shipped config | Result |
|---|---|---|---|---|
| Flagship NPU | p99 ≤ 25 ms | int8 dense, int8+prune, student, int4+prune | **int8 dense** (knee) | 92.1%, 14 ms, 22 MB |
| Mid-range NPU | p99 ≤ 16 ms, size ≤ 30 MB | int8 dense, int8+prune, student, int4+prune | **int8 dense** (knee, feasible) | 92.1%, 14 ms, 22 MB |
| Budget phone CPU | p99 ≤ 12 ms | int8+prune, student, int4+prune | **student-int8+prune** (knee of feasible) | 91.2%, 9 ms, 14 MB |
| Cortex-M7 MCU | size ≤ 10 MB | int4+prune only | **int4 + 50% prune** | 88.9%, 9 ms, 9 MB |
| Safety-critical | acc ≥ 92.0% | fp16 dense, int8 dense | **int8 dense** (cheapest above floor) | 92.1%, 14 ms, 22 MB |

Read the budget-phone row carefully, because it shows the knee *moving under a constraint*. With a 12 ms cap, `int8 dense` (14 ms) is now infeasible — the knee of the *full* frontier is excluded. So we recompute the knee over the feasible subset (everything ≤ 12 ms) and the new knee is the distilled student at 9 ms, 91.2%. The decision adapted automatically: tighten the latency cap and the shipped config slides down the frontier to the best feasible point. And the safety-critical row shows the opposite force: an accuracy floor of 92.0% excludes everything below int8 dense, so the cheapest config that *clears the floor* — int8 dense again — wins over the more accurate but far costlier fp16. The constraint anchors the choice from above; the knee logic picks the cheapest point that satisfies it.

## When the frontier itself is the wrong shape: stress tests

A clean frontier with an obvious knee is the happy path. Real sweeps misbehave, and a senior engineer's value is in handling the cases where the method's assumptions break. Let me stress-test the procedure.

**What if there is no knee — the frontier is a straight line?** A linear frontier means accuracy trades for latency at a *constant* exchange rate; every millisecond buys the same accuracy everywhere. Then the normalized-distance criterion returns a near-zero maximum (the curve never departs from the chord), which is the algorithm honestly telling you "there is no knee here." In that case the constraint *must* decide — there is no internal sweet spot, so you ship the most accurate feasible point or the cheapest point above your accuracy floor. Do not invent a knee where the data shows none; a near-flat `dist` array is a signal to lean entirely on constraints.

**What if two configs are statistically tied?** Covered earlier with the $\epsilon$-tolerant dominance, but the field version is: when the top two feasible configs are within measurement noise on the primary axis, break the tie on a *secondary* objective you were not optimizing — pick the smaller, the more energy-frugal, or simply the one that is *simpler to maintain* (an int8 PTQ model is easier to support than a hand-tuned int4 + structured-prune + distilled stack). Engineering simplicity is a legitimate tiebreaker on a frontier, and it never shows up in the four numbers. When the numbers tie, ship the one that pages you least.

**What if the frontier shifts after you ship?** The model gets retrained on new data, the runtime updates, the OS changes its thermal policy — and your measured frontier is now stale. The discipline is to treat the frontier as a *living artifact*: re-run the sweep (or at least re-measure the shipped config and its two nearest frontier neighbors) on every model retrain and every runtime bump, and re-validate that your shipped config is still on the frontier and still feasible. A frontier is a snapshot of a moving target; bake a re-measurement step into your release process. This is where [profiling and benchmarking on-device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) becomes a recurring chore, not a one-time analysis.

**What if no config is feasible?** The feasible frontier is empty. This is *information*, not failure. It means the product's constraints and the available techniques are incompatible, and you have exactly two honest moves: loosen a constraint (negotiate — "can the frame budget be 18 ms instead of 16?") or expand the sweep with more aggressive levers (go int4, prune to 70%, distill to a smaller student, switch to a fundamentally cheaper architecture via [hardware-aware NAS](/blog/machine-learning/edge-ai/hardware-aware-nas)). The wrong move — the one that causes field incidents — is to quietly ship a near-miss by relaxing the cap in your head. Make infeasibility loud, escalate it, and let the people who own the product spec decide whether the spec or the model bends.

**What if the binding constraint is peak memory, not size-on-disk?** A model can be small on disk yet blow peak runtime memory because of a fat intermediate activation. Then `size_mb` is the wrong constraint axis and you must add *peak memory* as a measured objective — which can completely reorder the frontier, because a config with a clever memory-light architecture can dominate a smaller-on-disk one that needs a huge activation buffer. This is the [memory-is-the-real-constraint](/blog/machine-learning/edge-ai/memory-is-the-real-constraint) lesson showing up as a frontier axis: measure the constraint that actually binds, not the proxy that is easy to measure.

## Case studies: frontiers in the real literature

This is not a toy framing — the most influential efficient-model papers of the last decade are, explicitly, Pareto-frontier arguments. A few worth knowing, with the numbers framed as the approximate, published figures they are.

**EfficientNet (Tan & Le, 2019)** is a frontier paper end to end. Its central contribution, *compound scaling*, is a rule for moving along the accuracy-vs-FLOPs frontier efficiently — scaling depth, width, and resolution together by a fixed ratio so you stay near the frontier instead of wandering off it by scaling one dimension blindly. The paper's famous plot is literally an accuracy-vs-FLOPs (and accuracy-vs-latency) Pareto curve with the EfficientNet family forming a new frontier above prior models: EfficientNet-B0 reached roughly ImageNet 77% top-1 at about 0.39 GFLOPs, where comparable ResNets needed several times the compute for similar accuracy. The whole "compound scaling beats single-axis scaling" result *is* the observation that single-axis scaling falls off the frontier.

**MnasNet (Tan et al., 2019)** and the broader family of **hardware-aware NAS** make the frontier the search objective directly. MnasNet's reward folds *measured on-device latency* (on a Pixel phone) into the architecture search, so the search optimizes the accuracy-latency frontier on real hardware rather than the accuracy-FLOPs proxy — which matters precisely because, as this series keeps hammering, [FLOPs and latency are different axes](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device). MnasNet found architectures that hit roughly MobileNetV2-level accuracy at meaningfully lower measured latency on the target phone, which is "push out the on-device frontier" stated as a NAS objective. Multi-objective NAS methods generalize this by optimizing the *hypervolume* of the discovered frontier — searching for the best *set* of trade-off points, not a single model.

**MobileNetV3 (Howard et al., 2019)** is a frontier-tuning story: NAS plus hand-tuning (the NetAdapt step trims the network to hit a latency target while losing the least accuracy) produced a family of points specifically placed along the accuracy-latency frontier on a Pixel phone, with MobileNetV3-Large and MobileNetV3-Small occupying the high-accuracy and low-latency ends of the same curve — two devices, two points, exactly the picture from our two-device worked example.

**DistilBERT (Sanh et al., 2019)** gives a frontier point from the distillation lever: roughly 40% smaller and about 60% faster than BERT-base while retaining around 97% of its GLUE performance. Stated as a frontier move, distillation produced a point that dominates "BERT-base run on cheaper hardware" for most on-device budgets — a smaller, faster point at a small, well-characterized accuracy cost. Where on the frontier you *want* DistilBERT versus full BERT is, again, a constraint question: under a tight latency cap DistilBERT is feasible and BERT is not, so the frontier picks DistilBERT despite its slightly lower accuracy.

The throughline: every one of these is a multi-objective optimization result dressed in deep-learning clothes. They are different *levers* (architecture scaling, NAS, distillation) all producing *points* and all evaluated against a frontier. That is the unifying claim of this entire series, and the literature has been making it, in frontier language, for years.

## When to reach for the frontier (and when it is overkill)

Reach for the full frontier procedure when you have **several candidate configurations and a real, multi-axis constraint** — which is essentially every serious edge deployment. If you have run any of this series' levers and produced more than two or three candidates, the dominance filter and the knee will save you from shipping a dominated config and from over-engineering past the knee, every time. It is also the right tool whenever **one model must serve multiple devices**, because you compute the frontier once and apply a different mask per target, as the two-device example showed.

It is overkill in exactly one case: when you have **one binding constraint and an obvious single objective.** If the product is "fit in 256 KB of SRAM, maximize accuracy, nothing else matters," you do not need a frontier — you need the most accurate model under 256 KB, which is a one-axis $\epsilon$-constraint with a single answer. Do not build a six-axis frontier to answer a one-axis question; the ceremony adds no information. The frontier earns its keep when objectives genuinely *conflict* and you have to *choose*; when they do not, skip it.

And a warning about the most seductive anti-pattern: **do not optimize one axis blindly.** "Make it as fast as possible" with no accuracy floor will march you off the cheap end of the frontier to a useless 60%-accurate model; "make it as accurate as possible" with no latency cap will march you to the fp16 top that drops frames. Every single-axis objective is implicitly a frontier query with a missing constraint — and the missing constraint is the one that bites you in the field. State the constraint, then optimize within it.

## The anti-patterns, named

Three failure modes recur often enough to name, so you can catch yourself.

**Comparing off-frontier configs.** Two engineers argue over config B vs config D, and both are dominated by config A that neither is looking at. The fix is mechanical: filter to the frontier *first*, then argue only among the survivors. Never debate a dominated config; delete it and move on. If a config is not on the frontier, it is not in the conversation.

**Optimizing the easy metric.** FLOPs are easy to compute and lie about latency; p50 is easy to measure and lies about the tail; size-on-disk is easy to read and lies about peak runtime memory. Building a frontier on the easy-but-wrong axis picks the wrong winner with total confidence. Build the frontier on the metric the device actually feels — p99 latency, peak memory, steady-state energy — even though those are the annoying ones to measure. The annoyance is the point: those are the numbers that bind.

**Ignoring p99 and energy.** A frontier built on p50 latency will pick a config with a fat tail that drops frames in production; a frontier that ignores energy will pick a config that is fast on the bench and throttles to slow in a warm pocket. The tail and the joules are not secondary — on a thermally constrained, battery-powered device they are often the *binding* axes. Put them on the frontier as first-class objectives, or be surprised in the field.

## Key takeaways

- **Every optimization lever produces a point, not a winner.** Quantization, pruning, distillation, and architecture search all just place a configuration in a trade-off space. Your job is to choose a point, and the frontier is how you choose it.
- **Dominance deletes the obvious losers for free.** A config that is no better on any axis and worse on one is deletable unconditionally — no preferences needed. Filter to the non-dominated set before you argue.
- **The knee is the default ship.** The point of maximum curvature (use the scale-free normalized-distance criterion) is where accuracy-per-millisecond is best; ship it unless a constraint forbids it.
- **Constraints carve a feasible region; pick the knee inside it.** Hard caps (p99, size, peak memory) define what is shippable. Find the knee, and if it is infeasible, walk to the nearest feasible frontier point.
- **Prefer $\epsilon$-constraints over weighted sums.** The weighted sum is blind to concave frontier regions and will silently hide good configs; hard caps plus one maximized objective reach every frontier point and match real product specs.
- **Hypervolume scores a whole frontier.** When comparing approaches (PTQ vs QAT, NAS run vs NAS run), compare the hypervolume of the frontiers, not single points.
- **Measure the axis that binds, on the real device, at the tail.** A frontier built on FLOPs, p50, cold-start latency, or size-on-disk picks the wrong winner. Use p99, steady-state energy, peak memory, on hardware.
- **Compute the frontier once, reuse it per device.** The same non-dominated set serves every target; each device is just a different constraint mask. Two devices legitimately ship two different points.
- **Empty feasible set is information.** It means the spec and the techniques are incompatible. Loosen a constraint or sweep harder — never quietly ship a near-miss.
- **Re-measure on every retrain and runtime bump.** The frontier is a snapshot of a moving target; validate the shipped config on-device before every release.

This is the decision tool the whole series has been pointing at. Every technique post gave you a lever; this post tells you how to choose among the points those levers produce. When you assemble the full optimization pipeline in the capstone, [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook), the frontier is the scoreboard you read at the end of every iteration — sweep, filter, pick the knee in the feasible region, validate, ship.

## Further reading

- **Vilfredo Pareto**, *Cours d'économie politique* (1896) — the original formulation of Pareto optimality, the economic root of the frontier idea.
- **K. Deb**, *Multi-Objective Optimization Using Evolutionary Algorithms* (2001) — the standard reference for dominance, the Pareto front, $\epsilon$-constraint vs weighted-sum scalarization, and hypervolume; chapters 2-3 cover everything in this post rigorously.
- **M. Tan and Q. Le**, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (ICML 2019) — compound scaling as a frontier-following rule; the canonical accuracy-vs-FLOPs Pareto plot.
- **M. Tan et al.**, "MnasNet: Platform-Aware Neural Architecture Search for Mobile" (CVPR 2019) — measured on-device latency folded into the search objective; the frontier as a NAS target.
- **V. Sanh et al.**, "DistilBERT, a distilled version of BERT" (2019) — a clean, frequently cited frontier point: ~40% smaller, ~60% faster, ~97% of the performance.
- **S. Satopää et al.**, "Finding a 'Kneedle' in a Haystack: Detecting Knee Points in System Behavior" (2011) — the normalized-distance knee criterion used in this post.
- Within this series: [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for the four-lever frame; [the metrics that actually matter on-device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) for honest measurement; and [hardware-aware NAS](/blog/machine-learning/edge-ai/hardware-aware-nas) for automated frontier search.

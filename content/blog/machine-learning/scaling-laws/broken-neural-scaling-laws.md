---
title: "Broken neural scaling laws: when one power law isn't enough"
date: "2026-06-15"
description: "Learn why a single power law mis-predicts loss past saturation, double descent, and emergent inflections, and how the broken neural scaling law fits and extrapolates those bends with a 3n+3 parameter functional form."
tags: ["scaling-laws", "broken-neural-scaling-laws", "bnsl", "caballero", "power-laws", "double-descent", "emergent-abilities", "extrapolation", "deep-learning", "machine-learning", "loss-prediction", "compute-budget"]
category: "machine-learning"
subcategory: "Scaling Laws"
author: "Hiep Tran"
featured: true
readTime: 53
---

Here is a failure mode that has burned more than one capable team: you run five small models, the loss-versus-compute points fall on a beautiful straight line on log-log axes, you fit a single power law, you extrapolate it three orders of magnitude forward, and you commit a seven-figure training budget on the forecast. Then the big run lands, and the loss is nowhere near where the line said it would be. The curve bent somewhere in the gap you never sampled, and your straight line walked confidently off a cliff.

That gap is the whole subject of this post. The clean story the rest of this series tells — [loss is predictable, fit a line on cheap runs and extrapolate it](/blog/machine-learning/scaling-laws/scaling-laws-predictability-foundations) — is true far more often than it has any right to be. But it has a sharp edge: it assumes the curve stays a single power law over the entire range you care about. Real scaling curves do not always cooperate. They saturate into a floor. They dip and rise and dip again (double descent). They sit nearly flat and then inflect upward into what people excitedly call an "emergent ability." A single power law cannot express any of these shapes, so it does not just predict the magnitude wrong past the bend — it predicts the wrong *qualitative behavior*, which is far more dangerous because it looks confident while doing it.

> [!important]
> **The six things to take away**
> - A single power law is a straight line on log-log axes; it cannot bend, so it mis-extrapolates the moment the real curve has a **break** (saturation, double descent, or an emergent inflection).
> - The **broken neural scaling law (BNSL)** of Caballero et al. 2022 is a smooth piecewise stitch of $n+1$ power-law segments joined at $n$ breaks, with **$3n+3$ parameters** total.
> - The functional form: $y = a + \left(b\,x^{-c_0}\right)\prod_{i=1}^{n}\left(1 + (x/d_i)^{1/f_i}\right)^{-c_i f_i}$. Each break $i$ adds a slope change $c_i$, a location $d_i$, and a sharpness $f_i$.
> - BNSL expresses **non-monotonic behavior** (double descent) and **sharp inflections** ("emergence") as *smooth breaks*, not true discontinuities — so it reframes emergence the way the [metric-artifact story](/blog/machine-learning/scaling-laws/emergent-abilities-scaling) does, from a different angle.
> - It **extrapolates more accurately than a single power law** across vision, language, audio, video, diffusion, RL, and alignment, over parameters, compute, data, input size, and training steps.
> - The two real limits: you need **enough data points** to fit $3n+3$ parameters, and the form **does not predict where breaks will occur** — only fits them once they show up in your data. The one number to remember is **$3n+3$**.

The diagram below is the mental model for the whole post. It draws one scaling curve that bends once, the cheap "observed region" of small runs on the left, and two extrapolations from that region: a single power law (dashed) that overshoots the loss badly past the break, and the BNSL fit that tracks the curve through the bend. Everything that follows is a tour of why the single line fails, what the BNSL form is made of, and how to use it without fooling yourself.

![A scaling curve that bends once, with a single power law dashed line overshooting past the break and a broken neural scaling law fit tracking the bent curve](/imgs/blogs/broken-neural-scaling-laws-1.png)

The observed region on the left is where you can afford to measure: small models, short runs, cheap experiments. In that region the data really does look like a single power law, because *every smooth curve looks like a straight line if you zoom in far enough*. The danger is precisely that local straightness. It gives you false confidence that the global shape is straight too. The break at $d_1$ — a saturation, an inflection, a phase change in how the model uses scale — sits in the gap you never sampled, and the single power law marches straight past it while the real curve turns away.

## 1. Why one power law cannot bend

**Senior rule of thumb: a single power law has exactly one slope, forever, and the entire reason it extrapolates so cleanly is the same reason it cannot survive a break.**

Recall the algebra from the [foundations post](/blog/machine-learning/scaling-laws/scaling-laws-predictability-foundations). A power law $y = b\,x^{-c}$ becomes a straight line under logarithms:

$$
\log y = \log b - c \log x.
$$

The slope of that line is $-c$, a single constant. Doubling $x$ multiplies $y$ by $2^{-c}$ no matter where you start — the *scale-free* property that makes the line trustworthy when you extend it. But scale-freedom is a double-edged sword. It says the behavior between $x = 10^6$ and $x = 10^7$ is *identical in shape* to the behavior between $x = 10^9$ and $x = 10^{10}$. If the real system changes its behavior somewhere in between — if it hits a data-manifold limit, exhausts the easy-to-learn features, or crosses an interpolation threshold — the single power law has no vocabulary to describe that change. It will keep reporting the old slope because the old slope is the only slope it owns.

This is not a subtle modeling preference. It is a hard structural limit. Consider three shapes that show up constantly in practice and that a single power law provably cannot represent:

- **Saturation into a non-zero floor.** Cross-entropy loss bottoms out at the entropy of the data, an irreducible constant. A pure $b\,x^{-c}$ goes to zero as $x \to \infty$; it has no floor term. So a single power law systematically *under*-predicts the asymptotic loss — it claims you can reach zero when you cannot.
- **A change of slope.** Many curves are steep early (lots of easy signal to extract) and shallow late (only hard signal left). That is two slopes. One power law has one slope, so it splits the difference and is wrong at both ends.
- **Non-monotonicity.** Double descent makes test loss go *down, up, then down again*. A power law is monotonic by construction (for $c > 0$ it strictly decreases). It cannot go up. Full stop.

The standard patch for the first problem is the Chinchilla-style form $L = E + A/N^\alpha + B/D^\beta$, which the rest of this series leans on. The additive $E$ term is the irreducible floor, and it fixes saturation toward an asymptote. That form is excellent and you should reach for it first. But notice what it still cannot do: it has one slope per axis. It cannot change slope partway, and it cannot go non-monotonic. It handles the floor; it does not handle breaks. BNSL is the generalization that does.

| Curve feature | Pure power law $b x^{-c}$ | Chinchilla $E + A N^{-\alpha}$ | Broken neural scaling law |
|---|---|---|---|
| Straight on log-log | yes (that is all it is) | approximately, far from floor | piecewise straight, one slope per segment |
| Non-zero asymptote (floor) | no | yes (the $E$ term) | yes (the $a$ term) |
| Change of slope mid-range | no | no | yes ($n$ breaks, $n+1$ slopes) |
| Non-monotonic (double descent) | no | no | yes (slope change can flip sign) |
| Sharp inflection / "emergence" | no | no | yes (a sharp break, small $f_i$) |
| Parameters | 2 | 3 (single axis) | $3n + 3$ |

The table is the argument in miniature. Each row that a power law cannot do is a place where extrapolating it will produce a confident, specific, wrong number. The job of BNSL is to add exactly enough structure to express those rows while staying a *fittable, smooth* function you can run least-squares against.

### 1.1 The cost of the wrong slope, in dollars

Let me make the failure concrete because "mis-extrapolates" is too gentle a phrase for what it does to a budget. Suppose your observed region gives a slope of $c = 0.10$ and the loss at your largest cheap run ($x_0 = 10^{19}$ FLOPs) is $L_0 = 3.0$ above the floor. You want to know the loss at the target run $x_1 = 10^{22}$ FLOPs — three orders of magnitude more compute, a thousand-fold scale-up.

The single power law says the loss multiplies by $(x_1/x_0)^{-c} = 1000^{-0.10} = 10^{-0.3} \approx 0.501$. So it predicts $L_1 \approx 1.50$. You budget the run on that promise.

Now suppose the real curve has a break at $10^{20.5}$ FLOPs where the slope shallows from $0.10$ to $0.04$ — a perfectly ordinary "we ran out of easy features" bend. The first decade and a half of the scale-up still gets the steep slope, but the last decade and a half gets the shallow one. The real multiplier is $10^{1.5 \times -0.10} \times 10^{1.5 \times -0.04} = 10^{-0.15} \times 10^{-0.06} = 10^{-0.21} \approx 0.617$. The real $L_1 \approx 1.85$.

The single power law promised $1.50$ and you got $1.85$ — a 23% miss on the quantity you spent the whole budget to move. If your go/no-go threshold for the run was "loss below $1.7$," the power law told you to spend the money and the real curve says you should not have. That is the entire danger: the error is not random noise around the truth, it is a *systematic, one-directional* miss that grows with the size of the extrapolation, and it points the wrong way at exactly the decision point where it matters.

It is worth writing the error growth as a formula, because it shows the miss is unbounded. Say the true slope after the break is $c_0 + c_1$ (with $c_1 < 0$ for a shallowing break) and the break sits at $d_1$, while you extrapolate the pre-break slope $c_0$ all the way to a target $x_1 > d_1$. The single power law's prediction and the true value differ in log space by exactly

$$
\log L_{\text{true}} - \log L_{\text{power law}} = -c_1 \,\log\!\left(\frac{x_1}{d_1}\right).
$$

Read that off: the log-error is proportional to the slope change $|c_1|$ times the number of decades you extrapolate *past* the break. There is no saturation in this expression. Every additional decade past $d_1$ adds the same fixed amount of log-error, so the further you push the forecast, the worse the single power law gets — linearly in log space, which is multiplicatively in the loss itself. This is why the failure is so insidious for the biggest, most expensive runs: those are precisely the runs that extrapolate the most decades past your cheap data, so they are precisely the runs where a missed break costs the most. The forecast is most wrong exactly where it matters most, and a single power law gives you no signal that anything is amiss — its residuals on the observed data are tiny, because the break is not in the observed data.

The same formula tells you what BNSL buys: if the break $d_1$ is inside your fitted range, BNSL learns $c_1$ and applies the corrected slope past the break, driving that error term to zero. The forecast is correct *up to the last break you fitted*; past that, BNSL extrapolates its final segment and is subject to the identical formula for any further unseen break. That symmetry — correct through fitted breaks, blind past them — is the precise statement of both BNSL's value and its limit, and it is worth carrying as the single sharpest takeaway of this section.

## 2. The functional form, term by term

**Senior rule of thumb: read the BNSL formula as a base power law multiplied by one correction factor per break, where each factor is dormant before its break and active after it.**

Here is the form from Caballero, Gupta, Rish, and Krueger (2022). For a quantity $y$ (a loss, an error rate, a reward) as a function of a scale $x$ (parameters $N$, compute $C$, dataset size $D$, input size, or training steps):

$$
y = a + \left(b\,x^{-c_0}\right)\prod_{i=1}^{n}\left(1 + \left(\frac{x}{d_i}\right)^{1/f_i}\right)^{-c_i f_i}.
$$

There are six symbols and they split into two groups. Three are the *base*, shared by the whole curve:

- $a$ — the **irreducible asymptote**. As $x \to \infty$, the product and the $b x^{-c_0}$ term go to zero (or to a finite limit), and $y \to a$. This is the floor: the entropy of the data, the Bayes error, the best any model of this family can do. It plays exactly the role of $E$ in the Chinchilla form.
- $b$ — the **log-log offset**. It sets the vertical position of the first segment, the intercept of the initial straight line on log-log axes.
- $c_0$ — the **first-segment slope**. Before any break, the curve is the plain power law $b x^{-c_0}$ on top of the floor, with log-log slope $-c_0$.

The other three appear once *per break*, indexed by $i = 1, \dots, n$:

- $c_i$ — the **slope change** at break $i$. Crucially this is a *change*, not the new slope. After break $i$ the effective slope becomes $-(c_0 + c_1 + \dots + c_i)$. A positive $c_i$ steepens the descent; a negative $c_i$ shallows it or, if large enough, flips the curve from decreasing to increasing (which is how double descent gets its upward hump).
- $d_i$ — the **break location** on the $x$ axis. This is where the correction factor for break $i$ transitions from "off" to "on." For $x \ll d_i$ the factor $\bigl(1 + (x/d_i)^{1/f_i}\bigr)^{-c_i f_i} \approx 1$ (dormant); for $x \gg d_i$ it contributes the slope change (active).
- $f_i$ — the **break sharpness**. Smaller $f_i$ means a sharper, more abrupt transition between the two slopes; larger $f_i$ means a gentle, rounded bend that spreads over more decades. A very small $f_i$ produces something that looks, to the eye and to a discontinuous metric, exactly like a phase transition — even though the function is perfectly smooth everywhere.

Count them: three base parameters plus three per break is $3 + 3n = 3n + 3$. With $n=0$ you recover a three-parameter power-law-with-floor. With $n=1$ you have six parameters and one bend. With $n=2$, nine parameters and a hump. The figure below maps each of the six single-break parameters onto the picture so you can see what each one moves.

![A labeled scaling curve with a single break, annotating which parameter controls the asymptote, the offset, the first slope, the slope change, the break location, and the sharpness](/imgs/blogs/broken-neural-scaling-laws-2.png)

The way to internalize this is to read the product as a switchboard. Each factor $\bigl(1 + (x/d_i)^{1/f_i}\bigr)^{-c_i f_i}$ is a switch that is open (value $\approx 1$, contributing nothing) until $x$ reaches $d_i$, then closes and adds its slope change. Walking left to right along the $x$ axis, you pass break $d_1$ and the slope changes by $c_1$; you pass $d_2$ and it changes again by $c_2$; and so on. Between breaks the curve is a clean power law with a constant slope. At each break the slope smoothly transitions over a width set by $f_i$. That is the entire mechanism: $n+1$ power-law segments, stitched smoothly at $n$ break points.

### 2.1 Checking the limits

A functional form earns trust by behaving sensibly at the extremes, so let us check the two limits of a single-break ($n=1$) BNSL.

For $x \ll d_1$, the term $(x/d_1)^{1/f_1}$ is tiny, so $\bigl(1 + (x/d_1)^{1/f_1}\bigr)^{-c_1 f_1} \approx 1^{-c_1 f_1} = 1$. The curve reduces to $y \approx a + b x^{-c_0}$: the base power law on the floor, slope $-c_0$. Good — before the break, you see exactly the first segment.

For $x \gg d_1$, the $1$ inside the parenthesis is negligible next to $(x/d_1)^{1/f_1}$, so the factor becomes $\bigl((x/d_1)^{1/f_1}\bigr)^{-c_1 f_1} = (x/d_1)^{-c_1}$. The curve reduces to $y \approx a + b\,d_1^{c_1}\,x^{-(c_0 + c_1)}$: still a power law on the floor, but now with slope $-(c_0 + c_1)$ and a rescaled offset. Good — after the break, you see the second segment with the summed slope, and the offset has shifted by exactly the amount needed for the two segments to meet continuously at $d_1$.

That continuity is the point of the $f_i$ exponent gymnastics. A naive piecewise definition ("use slope $c_0$ for $x < d_1$, slope $c_0 + c_1$ for $x > d_1$") has a kink — a discontinuous derivative — at the break, which is both physically implausible and numerically nasty to fit. The BNSL factor replaces the kink with a smooth transition whose width is $f_1$, so the function and all its derivatives are continuous. You get the *appearance* of a sharp break without an actual discontinuity, which is exactly the right model for a system where the underlying dynamics shift gradually but the observed metric snaps.

### 2.2 Why a product, not a sum

A natural question is why the breaks enter *multiplicatively* — as a product of correction factors — rather than additively, the way the Chinchilla form sums separate $A/N^\alpha$ and $B/D^\beta$ terms. The answer is in what each structure can express. An additive form of overlapping power laws gives you a curve whose slope is always dominated by whichever term is currently largest; it can bend *once*, from one term's slope to another's, but it cannot produce an arbitrary sequence of slope changes, and it cannot go non-monotonic. The multiplicative form, by contrast, lets each factor contribute its own slope change independently, and because the factors multiply, their slope changes *add* in log space.

That last sentence is the whole reason the form is built this way. Take the logarithm of the product term:

$$
\log \prod_{i=1}^{n}\left(1 + (x/d_i)^{1/f_i}\right)^{-c_i f_i} = \sum_{i=1}^{n} -c_i f_i \log\left(1 + (x/d_i)^{1/f_i}\right).
$$

In log space the product becomes a sum, and each summand is a smooth step that contributes nothing for $x \ll d_i$ and contributes a linear-in-$\log x$ term (slope $-c_i$) for $x \gg d_i$. So on log-log axes, the curve's slope is the running sum $-(c_0 + \sum_{i: d_i < x} c_i)$ — exactly the "switchboard" picture from earlier, now made precise. Each break that you have passed adds its $c_i$ to the slope. The multiplicative form in linear space is an additive form in log space, and additivity in log space is precisely what lets you stack an arbitrary number of independent slope changes, including ones that flip the sign of the running sum and send the curve upward. An additive-in-linear-space form does not have this property, which is why BNSL is built as a product.

### 2.3 The geometry of a single break

It is worth seeing the break geometrically, because the picture is what you will actually pattern-match against when you look at a real curve. On log-log axes, a single-break BNSL (ignoring the floor $a$, which only matters near the asymptote) is two straight line segments connected by a smooth elbow. The left segment has slope $-c_0$; the right segment has slope $-(c_0 + c_1)$; they meet near $\log d_1$; and the *radius* of the elbow — how rounded versus angular the join looks — is set by $f_1$. A large $f_1$ gives a wide, gentle arc spanning a decade or more; a small $f_1$ gives a tight corner that looks, to the eye, like an angular kink even though it is smooth under magnification.

This geometry is why the break sharpness $f_i$ is the parameter that determines whether a curve reads as "gradual scaling" or "sudden emergence." Both are the same two-segment elbow; they differ only in the elbow radius. When someone shows you a capability curve that "jumps," what they are showing you is a small-$f_i$ elbow. When someone shows you a "smooth scaling trend," that is a large-$f_i$ elbow. The functional form does not distinguish them by category — it distinguishes them by a single continuous number, which is the most important structural insight of the whole BNSL framework and the bridge to the emergence discussion next.

## 3. Emergence is a sharp break, not a magic jump

**Senior rule of thumb: when a capability "appears suddenly" at scale, your first hypothesis should be a sharp-but-smooth break (small $f_i$) plus a discontinuous metric — not a genuine discontinuity in the model.**

This is where BNSL connects to the most contested topic in the scaling literature: [emergent abilities](/blog/machine-learning/scaling-laws/emergent-abilities-scaling). The original claim (Wei et al. 2022) was that certain abilities — multi-step arithmetic, word unscrambling, instruction following — are simply *absent* in small models and *present* in large ones, with a sharp threshold and no warning from the small-model curve. If that were literally true, it would break the entire predictability program: you could not forecast a big model's capabilities from small runs, because the capability is a discontinuity that the small runs cannot see coming.

There are two complementary deflations of that strong claim, and BNSL is one of them. The figure below contrasts the two views directly: under a single power law, the inflection has to be explained away as a separate "emergent event"; under BNSL, it is just one more smooth break that the form fits and extrapolates through.

![A before and after comparison showing the single power law view treating an inflection as an emergent jump versus the broken neural scaling law view treating it as a smooth break that extrapolates through](/imgs/blogs/broken-neural-scaling-laws-3.png)

The first deflation is the [metric-artifact argument](/blog/machine-learning/scaling-laws/emergent-abilities-scaling) of Schaeffer, Miranda, and Koyejo (2023): the "jump" is often manufactured by a discontinuous evaluation metric. Exact-match accuracy on a multi-step task multiplies per-token success probabilities, so a smooth, gradual improvement in per-token error gets compressed into a cliff under exact-match while looking perfectly smooth under a continuous metric like token edit distance or Brier score. Their meta-analysis found that more than 92% of hand-annotated emergent BIG-Bench tasks used one of two discontinuous metrics. Swap the metric and the cliff dissolves into a sigmoid.

BNSL is the second deflation, and it is *orthogonal* to the metric argument — it bites even when the smoothness is in the underlying loss, not manufactured by the metric. Caballero et al. showed that the loss-versus-scale curve for many "emergent" tasks is fit cleanly by a BNSL with a sharp break: a small $f_i$ produces an inflection so abrupt that it reads as a phase transition, yet the function is smooth and, critically, *extrapolatable*. You do not need to believe the capability is a discontinuity. You need to believe the curve has a sharp break, fit the break, and extend the fit. The "emergence" becomes a modeling artifact of having tried to fit a single power law to a curve that has a bend in it.

These two stories do not compete; they stack. The metric argument says: some apparent jumps are an artifact of how you *measured*. The BNSL argument says: some apparent jumps are an artifact of how you *modeled* (one power law where you needed two segments). And the reconciliation (Du et al. 2024, on the [loss-threshold view](/blog/machine-learning/scaling-laws/emergent-abilities-scaling)) says: underneath both, there can still be a genuine task-specific loss threshold below which the capability switches on — but that threshold is itself smooth in pre-training loss and therefore forecastable. BNSL is the functional-form companion to that reconciliation. It gives you a curve you can actually fit and extend, which is more than a philosophical position; it is a tool.

> The strong "emergence is a discontinuity" claim and the strong "emergence is pure mirage" claim are both too strong. The defensible middle is: model the curve with a form that *allows* sharp breaks (BNSL), measure with a continuous metric, and treat any remaining surprise as a hypothesis about a loss threshold, not as evidence that forecasting is impossible.

### 3.1 A worked sharpness example

To feel what $f_i$ does, take a single-break BNSL with $a = 0$, $b = 1$, $c_0 = 0.05$ (nearly flat first segment), a slope change $c_1 = 0.6$ (the curve gets much steeper after the break — a capability "switching on" means error dropping fast), and a break at $d_1 = 10^{23}$ FLOPs. Now compare two sharpness values.

With a gentle $f_1 = 1.0$, the transition spreads over roughly two decades of compute: at a tenth of the break compute the slope is still close to $0.05$, at the break it is partway through changing, and only at ten times the break compute has it fully reached $0.65$. Plotted, this is a visible but rounded bend. A continuous metric tracks it smoothly and a forecaster fitting it on the early data would see the bend coming.

With a sharp $f_1 = 0.1$, the same slope change happens over roughly a fifth of a decade — almost vertically on a log-log plot. To the eye it is a step. Under exact-match accuracy it is indistinguishable from "the ability was absent, then present." But the function is still smooth, still six parameters, still fittable, and still extrapolatable *if* you have at least one or two points past the break to pin down $c_1$ and $f_1$. That last "if" is the whole catch, and section 6 is about it. The lesson here is that sharpness is a *parameter*, not a category: "emergent" and "gradual" are the same functional form at two values of $f_i$, which is the most deflationary thing you can say about emergence and also the most useful, because it means the same fitting machinery handles both.

## 4. Double descent: the non-monotonic case

**Senior rule of thumb: a curve that goes down, up, then down again needs at least two breaks with a sign-flipped slope change, and only a form that allows that can extrapolate it.**

Double descent is the cleanest demonstration that you sometimes need more than a monotone power law, because it is *not monotone*. As you increase model size (or training epochs, or data) past the point where the model can exactly interpolate the training set — the interpolation threshold — the test loss first rises to a peak and then falls again into a second, often lower, descent. Classical bias-variance intuition explains the first descent and the rise to the peak; the modern over-parameterized regime explains the second descent. No single power law can express this shape, because a power law is monotone.

The figure below draws the canonical double-descent curve and shows how BNSL captures it: a first descent (classical regime), a break where the slope change is large and *negative* enough to flip the curve upward to the interpolation peak, and a second break where the slope turns down again into the over-parameterized descent.

![A double descent curve showing a first descent, a rising peak at the interpolation threshold, and a second descent, annotated to show two breaks modeling the non-monotonic hump](/imgs/blogs/broken-neural-scaling-laws-4.png)

The mechanism in BNSL terms is the sign of the slope change. Recall that after break $i$ the effective slope is $-(c_0 + c_1 + \dots + c_i)$. If the running sum $c_0 + \dots + c_i$ goes *negative*, the exponent on $x$ becomes positive, and the curve *increases* with $x$ — it goes up. So to model the rise into the interpolation peak, you give break 1 a slope change $c_1$ negative enough that $c_0 + c_1 < 0$. The curve, which was descending at slope $-c_0$, now ascends. Then break 2 gets a positive slope change $c_2$ large enough that $c_0 + c_1 + c_2 > 0$ again, and the curve resumes descending into the second basin. Two breaks, a sign flip and a sign flip back: the hump is captured.

Put concrete numbers on it so the sign-flip is not abstract. Take a first-segment slope $c_0 = 0.4$ (a healthy classical descent). Give break 1 a slope change $c_1 = -0.9$, so the running sum becomes $0.4 - 0.9 = -0.5$: the effective post-break-1 slope is $-(-0.5) = +0.5$, and the curve climbs toward the interpolation peak. Now give break 2 a slope change $c_2 = +0.7$, so the running sum becomes $0.4 - 0.9 + 0.7 = 0.2$: the effective slope is $-0.2$, and the curve descends again, more gently than its first descent, into the over-parameterized basin. Three slopes — descend at $0.4$, ascend at $0.5$, descend at $0.2$ — produced by two breaks whose slope changes happen to sum across the sign boundary. The break locations $d_1$ and $d_2$ straddle the interpolation threshold, and the sharpnesses $f_1, f_2$ control how peaked the hump looks. That is a nine-parameter ($n=2$) BNSL fitting a textbook double-descent curve, and every one of those nine numbers is something the optimizer reads off your data rather than something you impose.

This matters beyond being a neat trick. Double descent is a real, reproducible phenomenon, and if your scaling study happens to straddle the interpolation threshold — which is easy to do when you sweep model width or training duration — then a monotone fit will be badly wrong *and* will not even warn you, because the residuals near the peak look like noise until you have enough points to see the systematic hump. The honest move is to use a form that *can* be non-monotone, fit it, and let the data decide whether the slope changes flip sign. If they do not, BNSL collapses gracefully toward the monotone case (the relevant $c_i$ come out small or same-signed); if they do, you have a model of the hump instead of a fit that pretends it is not there.

### 4.1 Why the over-parameterized descent is the interesting one

The second descent is where the modern deep-learning story lives, and it is worth a moment because it explains why we care about modeling double descent at all rather than just avoiding the interpolation threshold. In the over-parameterized regime — more parameters than training points — the test loss often keeps falling well below the classical-regime minimum. That is the empirical foundation under "just make it bigger." But the path there runs *through* the interpolation peak, and if you are doing a careful scaling study to forecast the over-parameterized regime, you cannot pretend the peak is not on the way. A BNSL fit lets you model the peak and the descent past it as one curve, so your forecast of the deep over-parameterized loss accounts for the bump you have to climb over to reach it. A monotone fit, by contrast, either ignores points near the peak (throwing away data) or gets dragged toward the peak and under-predicts the eventual descent. Either way you lose information that the broken form keeps.

## 5. Fitting a BNSL in practice

**Senior rule of thumb: BNSL is just least squares on a six-or-more-parameter function — the hard part is choosing $n$ and having points on both sides of every break.**

There is no magic in fitting a BNSL. It is nonlinear least squares: pick the number of breaks $n$, then minimize the squared error (typically in log space, since both axes are logarithmic) between the BNSL prediction and your measured points, over the $3n+3$ parameters. The pipeline below lays out the steps; the rest of this section is the practical detail under each box.

![A pipeline showing collect points across scales, choose the number of breaks, fit the parameters by least squares, validate on a held out large run, then extrapolate forward](/imgs/blogs/broken-neural-scaling-laws-6.png)

**Step 1: collect points across scales.** You need measurements of $y$ at a range of $x$ values, spanning as many decades as you can afford, on log-log axes. The single most important property of this set is whether it *brackets* the breaks. A break you have no points past is a break you cannot fit — you will be extrapolating into it blind, which is the exact failure the single power law makes. More on this in section 6.

**Step 2: choose the number of breaks $n$.** This is the one genuinely judgment-laden step. Look at the curve. If it is straight on log-log, $n=0$ (use a plain power law; do not over-fit). If it bends once, try $n=1$. If it has a hump, you need at least $n=2$. The discipline here is the same as any model-selection problem: add breaks only when the residuals of the simpler fit show clear, systematic structure (a run of same-signed residuals near a region), and stop when extra breaks stop reducing held-out error. Every break costs three parameters; pay only for the ones the data demands.

**Step 3: fit the $3n+3$ parameters.** Use a robust nonlinear optimizer (Levenberg-Marquardt, or a bounded BFGS) with sane initialization: initialize $a$ near the lowest observed $y$, $c_0$ near the slope of the early points, each $d_i$ near the $x$ where you eyeballed a bend, and each $f_i$ at a moderate value like $1$. Bad initialization is the usual reason a BNSL fit fails to converge; the function has enough parameters that the loss surface has local minima, so a few restarts from different inits is cheap insurance. Fit in log space so that points across many decades contribute comparably rather than letting the largest-$x$ points dominate.

**Step 4: validate on a held-out large run.** This is non-negotiable and it is the only honest test of an extrapolation. Hold out your largest one or two points, fit on the rest, and check whether the fit predicts the held-out points. A BNSL that interpolates beautifully but fails this leave-the-largest-out test is over-fit and will not extrapolate. This is also how you compare $n=1$ versus $n=2$: pick the $n$ that predicts held-out scale best, not the one that fits the training points best.

**Step 5: extrapolate forward.** Once the held-out test passes, extend the fitted curve to your target scale and read off the forecast. The crucial caveat, repeated because it is the one people forget: you can only trust the extrapolation through breaks that are *already in your fitted range*. A break beyond your largest point is invisible to the fit, and BNSL will extrapolate the last segment's slope straight through it — making exactly the single-power-law mistake on the final segment. BNSL does not predict where future breaks are; it only models the ones your data already revealed.

### 5.1 The fit in code

The entire procedure is short enough to write out, and writing it out kills any remaining mystique. Here is a complete, runnable implementation using SciPy's `curve_fit`, including the log-space objective and the leave-the-largest-out check. It fits a one-break BNSL and reports both the in-sample residual and the held-out error that actually decides whether you should trust the forecast.

```python
import numpy as np
from scipy.optimize import curve_fit

def bnsl(x, a, b, c0, c1, d1, f1):
    # y = a + (b * x^-c0) * prod_i (1 + (x/d_i)^(1/f_i))^(-c_i f_i), n = 1 break
    base = b * np.power(x, -c0)
    break_factor = np.power(1.0 + np.power(x / d1, 1.0 / f1), -c1 * f1)
    return a + base * break_factor

def fit_bnsl(x, y, p0=None, bounds=None):
    # Fit in log space so points across many decades contribute comparably.
    def log_resid(x, a, b, c0, c1, d1, f1):
        return np.log(bnsl(x, a, b, c0, c1, d1, f1))
    if p0 is None:
        # a near the floor; c0 from the early slope; d1 near the visible bend.
        p0 = [0.9 * y.min(), 1.0, 0.05, 0.5, np.sqrt(x.min() * x.max()), 0.5]
    if bounds is None:
        bounds = ([0, 1e-6, 0, -2, x.min(), 1e-3],   # lower
                  [y.min(), 1e6, 2, 2, x.max(), 5])   # upper
    popt, _ = curve_fit(log_resid, x, np.log(y), p0=p0, bounds=bounds, maxfev=20000)
    return popt

# Synthetic curve: shallow first segment, sharp drop after a break at x = 1e21.
x = np.array([1e18, 1e19, 1e20, 5e20, 1e21, 5e21, 1e22])
true = bnsl(x, a=0.02, b=2.0, c0=0.03, c1=0.55, d1=1e21, f1=0.2)
y = true * (1 + 0.02 * np.random.default_rng(0).standard_normal(len(x)))  # 2% noise

# Leave-the-largest-out: fit on all but the biggest point, predict it.
popt = fit_bnsl(x[:-1], y[:-1])
pred_largest = bnsl(x[-1], *popt)
held_out_err = abs(pred_largest - y[-1]) / y[-1]
print(f"fitted params: {np.round(popt, 4)}")
print(f"held-out point: true={y[-1]:.4f}  predicted={pred_largest:.4f}  err={held_out_err:.1%}")
```

Two things in that code are load-bearing. First, the fit is done on `np.log(y)` against `log_resid` — fitting raw values would let the largest-magnitude points dominate the residual and effectively ignore the small-scale structure where the break lives. Second, the `bounds` keep $f_1$ positive and $d_1$ inside the observed range; without bounds the optimizer happily wanders to a degenerate solution (a break at $x = 10^{40}$ with $f_1 = 100$ is numerically a flat line that fits noise). The held-out error at the end is the only number that should move your decision: a small in-sample residual with a large held-out error is the unmistakable signature of over-fitting, and it is exactly the case where a simpler $n=0$ or $n=1$ fit would have served you better.

Swapping in $n=2$ is a three-line change — add `c2, d2, f2` to the signature and one more `break_factor` to the product — and the comparison you run is always the same: which $n$ minimizes held-out error, not in-sample residual. That single substitution, run as a loop over $n \in \{0, 1, 2\}$, is the whole model-selection procedure.

### 5.2 A concrete fitting walkthrough

Suppose you measure error rate on a reasoning task at six model sizes and the log-log points look like this: the first four are roughly straight with a shallow downward slope, then the last two drop sharply. Your eye says one break, somewhere between the fourth and fifth point. You try $n=1$.

Initialize: $a$ at $0.02$ (a bit below your lowest observed error), $b$ and $c_0$ from a quick power-law fit to the first four points (say $c_0 \approx 0.03$, nearly flat), $d_1$ at the $x$ between points four and five, $f_1 = 0.5$ (you suspect a sharp break because the drop is abrupt), and a slope change $c_1 \approx 0.5$ (the curve gets much steeper after the break). Run the optimizer in log space. It converges; the fitted $f_1$ comes out small (a sharp break, consistent with the abrupt drop), and the fitted second-segment slope $-(c_0 + c_1)$ is much steeper than the first.

Now the honest test: refit on the first five points only, and check the sixth. If the $n=1$ fit on five points predicts the sixth within your error bars, you have a model that extrapolated correctly across the break, and you can extend it forward with some confidence. If it misses, either you need a second break (the curve bends again past point five and you cannot see it yet) or you simply do not have enough points past the break to pin $c_1$ and $f_1$ — in which case the right answer is "collect another point or two before you trust any forecast," not "fit harder."

Compare this to what a single power law would have done: fit the shallow slope of the first four points and extrapolate it flat, predicting the last two points are barely lower than the fourth. It would have missed the sharp drop entirely — predicting the *capability does not improve much*, when in fact it improves sharply. That is the difference between a form that can bend and one that cannot, on a real reasoning-task curve.

## 6. The limits, stated honestly

**Senior rule of thumb: BNSL is a descriptive fitting form, not a predictive theory — it tells you the shape of breaks you have already seen, never where the next one hides.**

A tool is only useful if you know what it cannot do, and BNSL has two hard limits that you must respect or it will hurt you worse than a power law (because it looks more sophisticated).

**Limit 1: you need enough data points.** Fitting $3n+3$ parameters requires meaningfully more than $3n+3$ data points — you want several points per segment and at least a couple on each side of every break, or the fit is under-determined and the optimizer will find a curve that threads your points but means nothing out of sample. With four points you should not be fitting more than $n=0$ (three params) honestly. With eight or ten well-spread points you can support $n=1$ or maybe $n=2$. This is the practical reason small studies should default to the simplest form and add breaks only when the data is rich enough to justify them. The matrix below makes the budget explicit.

![A matrix showing how the number of breaks maps to the number of segments, the parameter count of three n plus three, the parameters added per break, and the minimum data points needed](/imgs/blogs/broken-neural-scaling-laws-5.png)

The matrix is also a sanity check you can run before fitting. Count your data points, decide how many you can spare for held-out validation, and the remainder caps how many breaks you can responsibly fit. If you have nine points, hold out one or two, and you are squarely in $n=1$ territory; claiming $n=3$ (twelve parameters) on seven training points is curve-fitting theater. The discipline is the same as the [Chinchilla fitting discipline](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling): more parameters demand more data, and an over-parameterized fit extrapolates worse than an under-parameterized one even though it interpolates better.

**Limit 2: it does not predict where breaks occur.** This is the deeper limit and it is worth dwelling on. BNSL is *descriptive*: given data that already spans a break, it fits the break's location, slope change, and sharpness. It is not *predictive* about breaks: it has no mechanism to tell you that a break is coming at a scale you have not yet reached. If the real curve has a saturation at $10^{25}$ FLOPs and your largest data point is at $10^{22}$, BNSL will fit your data as a clean power law (or a curve with whatever breaks are below $10^{22}$) and extrapolate the final segment straight through the $10^{25}$ break — predicting loss that the saturation will never let you reach. In that scenario BNSL makes the *same overshoot* the single power law makes, just on the last segment instead of the whole curve.

This is not a flaw to be patched; it is a property of the kind of object BNSL is. A *theory* of scaling — like the [manifold-dimension and spectral-decay derivations](/blog/machine-learning/scaling-laws/why-power-laws-arise) that try to explain *why* power laws arise — could in principle predict where a break should be, because it models the mechanism (running out of manifold dimensions, exhausting the data covariance spectrum). BNSL deliberately does not model the mechanism. It is a flexible curve that fits whatever shape your data shows, which is its strength (it works across every domain, mechanism-agnostic) and its limit (it cannot see past your data). The right mental model is: BNSL is the phenomenological complement to the mechanistic theories. Use BNSL to fit and extrapolate the breaks you can see; use the theories to reason about the breaks you cannot.

### 6.1 The over-fitting failure mode in detail

The most common way to misuse BNSL is to add breaks until the training points are fit perfectly, then extrapolate the resulting wiggly curve. This fails spectacularly and predictably. Each break is three parameters of flexibility, and flexibility is exactly what lets a curve thread noise. A BNSL with too many breaks will put a tiny break at every cluster of points, achieve near-zero training residual, and then produce an extrapolation dominated by the last spurious segment — which is fit to two or three noisy points and points in an essentially random direction.

The defense is the leave-the-largest-out validation from section 5, applied ruthlessly. The metric that matters is not how well you fit the points you have; it is how well a fit on the smaller points predicts the larger ones. Choose $n$ by that metric. In practice this almost always pushes you toward *fewer* breaks than your eye wants, because the eye sees structure in noise and the held-out test does not reward fitting noise. A senior practitioner's instinct here is conservative: default to the smallest $n$ whose residuals are not obviously structured, validate out of sample, and add a break only when both the residuals demand it and you have the points to support it.

## 7. Where BNSL has been validated

**Senior rule of thumb: the reason to trust the form is breadth — it beat single power laws on held-out extrapolation across domains and scaling axes that have nothing in common except the shape of their curves.**

The Caballero et al. paper's central empirical claim is not "here is a form that fits one curve nicely." It is "here is a form that *extrapolates more accurately than a single power law, and than other competing functional forms,* across an unusually wide range of domains and scaling dimensions." That breadth is the argument, because it suggests the broken shape is a general property of how learning systems scale, not a quirk of one modality.

The figure below summarizes the domains and what kind of break BNSL captured in each. The pattern across the rows is the point: vision curves with double descent, language curves with emergent inflections, RL curves with sharp competence jumps, alignment curves with inverse-scaling U-shapes — all the same functional form at different parameter values.

![A matrix of domains including vision, language, audio, diffusion, reinforcement learning, and alignment, with the scaling axis and the curve feature that the broken neural scaling law captures in each](/imgs/blogs/broken-neural-scaling-laws-7.png)

A few of the rows deserve a sentence of color, because each is a place where a single power law would have failed in a different way:

- **Vision.** Image-classification and autoencoder curves frequently show double descent over parameters or data. BNSL's non-monotonic capability is what lets it fit these where a monotone law cannot. This is the same double-descent shape from section 4.
- **Language.** Many of the tasks labeled "emergent" by Wei et al. are fit by a BNSL with a sharp break, which is the section-3 reframing: the inflection is a smooth break, fittable and extrapolatable, not a discontinuity. This is the row that ties BNSL directly into the [emergence debate](/blog/machine-learning/scaling-laws/emergent-abilities-scaling).
- **Diffusion and generative models.** Sample-quality and loss curves over training steps can be non-monotonic in subtle ways; a broken form tracks the inflections that a single slope smears over.
- **Reinforcement learning.** RL competence curves over environment steps are notorious for long flat regions followed by sharp jumps when the agent "figures it out." That is a textbook sharp break (small $f_i$), and BNSL fits and extrapolates it where a power law would predict the agent never improves.
- **Alignment.** Some alignment-relevant metrics show *inverse scaling* (getting worse with scale) or U-shapes (worse then better). A slope change that flips sign handles both, which is the same machinery as double descent applied to a different axis.

The honest framing is that BNSL won these comparisons on *extrapolation*, the metric that matters, against single power laws and against several other multi-parameter forms. It is not that other forms cannot fit a given curve in-sample — a flexible-enough function fits anything in-sample. It is that when you fit on the smaller scales and test on the larger ones, the broken form predicted the held-out points more accurately, more often, across more domains. That is the evidence you should anchor on, and it is also why the limits in section 6 are stated so carefully: the breadth of success makes it tempting to treat BNSL as a predictive theory, and it is not one.

## 8. BNSL versus the alternatives

**Senior rule of thumb: every form is a bet about the shape of the curve outside your data, and you should pick the cheapest form whose shape-bet your residuals do not contradict.**

BNSL is not the only multi-parameter form on the menu, and choosing among them is really choosing what you are willing to assume about the unsampled region. The comparison below lays out the practical options a forecaster actually weighs, from the two-parameter power law up to the broken form, with the assumption each one bakes in and the failure it courts.

| Form | Params | Shape it assumes | Fails when |
|---|---|---|---|
| Power law $b x^{-c}$ | 2 | one slope forever, hits zero | any break, or a non-zero floor |
| Power law + floor $E + b x^{-c}$ | 3 | one slope, saturates to $E$ | a slope change mid-range, non-monotonicity |
| Chinchilla $E + A N^{-\alpha} + B D^{-\beta}$ | 5 (two axes) | one slope per axis, saturates | a break within one axis, double descent |
| Logistic / sigmoid in $\log x$ | 3–4 | one S-shaped transition | more than one transition, power-law tails |
| Broken neural scaling law | $3n+3$ | $n+1$ segments, smooth breaks | extrapolating past the last fitted break |

Two comparisons in that table deserve more than a row. The **logistic (sigmoid) form** is the natural competitor for "emergence" curves, because a sigmoid also produces a smooth S-shaped rise that looks like a phase transition. The difference is the tails: a sigmoid flattens to a horizontal asymptote on *both* ends, while real scaling curves keep improving as power laws on both sides of the break. So a sigmoid fits the transition but mis-models the segments around it, and when you extrapolate it predicts a hard ceiling the power-law tail will blow past. BNSL keeps power-law segments on both sides of the break, which is the empirically correct tail behavior for loss curves, and that is exactly why it extrapolates better than a sigmoid on the same emergent-looking data.

The **Chinchilla form** is the more important comparison because it is the one you will actually reach for first, and you should. It handles the floor and gives you the parameter-token tradeoff that BNSL does not address at all (BNSL fits a curve along a single scaling axis; it is not a substitute for the two-axis $N$-versus-$D$ allocation machinery). The honest division of labor is: use Chinchilla for the allocation problem and the floor; reach for BNSL only when, *along whatever axis you are studying*, the residuals of the Chinchilla or power-law fit show a structured break that the simpler form cannot bend to follow. The two are not rivals — they answer different questions, and BNSL is the escalation you make when the clean form's residuals stop looking like noise.

There is also a lineage worth naming: the "smoothly broken power law" is an old idea in physics and astronomy, used for decades to fit spectra and luminosity functions that change slope at a characteristic scale. Caballero et al.'s contribution was not inventing the broken-power-law shape; it was showing that this specific multi-break, smoothly-stitched parameterization extrapolates neural scaling curves across an unusually broad set of domains and axes, and that it captures double descent and emergence as special cases. Recognizing the lineage is useful because it tells you the form is mathematically mature — the fitting pitfalls are well understood — and the novelty is empirical: the claim that *neural* scaling curves have this shape, and that fitting it pays off in better forecasts.

## 9. How this fits the rest of the series

**Senior rule of thumb: use the clean laws first, watch the residuals, and reach for BNSL only when the residuals show a break — then never extrapolate past the last break you can see.**

It helps to place BNSL precisely among its neighbors, because the relationship is one of complement, not replacement. The chronology below shows BNSL arriving in the same window as the emergence debate and offering the smooth-break reading of the same phenomena.

![A timeline of the emergence debate showing the emergent abilities claim, the broken neural scaling law form, the metric artifact rebuttal, and the loss threshold reconciliation](/imgs/blogs/broken-neural-scaling-laws-8.png)

The [foundations](/blog/machine-learning/scaling-laws/scaling-laws-predictability-foundations) and [Chinchilla](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) posts give you the workhorse forms: a single power law, or a power law with an irreducible floor. These are right the vast majority of the time, they need few parameters, and you should always try them first. The [why-power-laws-arise](/blog/machine-learning/scaling-laws/why-power-laws-arise) post explains the *mechanisms* — data-manifold dimension, spectral decay, Zipf tails — that make those clean power laws appear in the first place, and crucially, those theories assume the breaks away to get a clean derivation. BNSL is the phenomenological tool that captures exactly the breaks those clean theories abstract over: it is the complement that handles the messy real curve when the idealized one bends.

And BNSL is one of two angles on [emergence](/blog/machine-learning/scaling-laws/emergent-abilities-scaling). The metric-artifact angle says the jump is in how you measured; the BNSL angle says the jump is in how you modeled (one slope where you needed several). Both deflate the strong "emergence is an unpredictable discontinuity" claim, and both leave room for the reconciliation's genuine-but-smooth loss thresholds. If you read those three posts together, the picture is coherent: capabilities improve smoothly in the underlying loss; that smoothness can look like a cliff through a discontinuous metric (Schaeffer) or through a single-power-law fit (Caballero); and there can still be a real task-specific loss threshold underneath (Du), which is itself forecastable. BNSL's contribution to that picture is concrete and tool-shaped: a fittable form you can run least-squares against and extend, rather than only a philosophical stance.

The throughline of the whole series applies here too: the simple laws are first-order approximations, and the engineering judgment lives in the second-order structure. A single power law is the first-order picture of a scaling curve. The breaks — saturation, double descent, emergent inflections — are the second-order structure, and BNSL is the form that lets you model them instead of pretending they are noise. The discipline is to use the simplest form that the residuals allow, escalate to BNSL when they demand it, and remember always that even the broken form is blind past the last break it can see.

## What this means in practice

If you take one operational habit from this post, make it this: **never trust a single-power-law extrapolation without first asking where the curve could break, and never trust a BNSL extrapolation past its last fitted break.** Concretely, the workflow looks like this.

Before you commit a large training budget on a forecast, lay out your observed points on log-log axes and try the cheapest forms first — a plain power law, then a power-law-with-floor. Look at the residuals. If they are unstructured noise, you are done; the simple form is the right one and adding breaks would be over-fitting. If the residuals show a *run* of same-signed deviations in a region, that is the signature of a slope change the simple form cannot follow, and it is your cue to escalate to a BNSL with one break in that region.

When you do escalate, fit in log space, bound the parameters so the optimizer cannot wander to a degenerate flat line, and choose the number of breaks by leave-the-largest-out held-out error rather than in-sample residual. Run the model-selection loop over $n \in \{0, 1, 2\}$ and pick the smallest $n$ whose held-out error is competitive. This is the single discipline that separates a forecast you can spend money on from a curve-fit that merely looks good.

Then, before you read off the number, locate your target scale relative to the breaks the fit found. If the target is inside the fitted range or just past the last fitted break, your forecast is well-grounded. If the target is many decades past the last point you measured, treat the forecast as a lower-confidence estimate and bound it: estimate the irreducible floor from the data entropy so you know the loss cannot go below it, and ask whether a saturation break is plausible in the gap. The error-growth formula from section 1.1 tells you exactly how badly an unseen break would hurt — proportional to the slope change times the decades you extrapolate past it — so size your confidence interval accordingly.

Finally, when the forecast is about a *capability* rather than a loss, fit on a continuous metric, not exact-match. A sharp break under exact-match is usually a gentle break under a continuous metric, and the gentle version is far easier to forecast. That is the practical fusion of the BNSL form with the [metric-artifact lesson](/blog/machine-learning/scaling-laws/emergent-abilities-scaling): pick the metric that makes your breaks gentle, fit the broken form, validate out of sample, and you will be surprised by emergence far less often than the field's headlines suggest you should be.

## Case studies from production

### 1. The confident overshoot

A team forecasting the loss of a large language run fit a single power law to five small models, all of which fell on a clean line, and extrapolated three orders of magnitude forward. The big run landed about 20% higher in loss than predicted. The post-mortem found a slope change roughly a decade and a half into the extrapolation, in a region with no data points. The wrong first hypothesis was "the fit was noisy"; the actual root cause was a structural break — the loss curve shallowed once the model exhausted the easy-to-learn distributional structure. The fix on the next project was to budget two extra mid-scale runs specifically to bracket the suspected break region, then fit an $n=1$ BNSL, which predicted the subsequent large run within error. The lesson: a clean line in the observed region is not evidence of a clean line outside it; it is evidence only that you have not yet sampled where it bends.

### 2. The double-descent ambush

A width-sweep study to find the best model size produced test-loss points that, fit with a monotone power law, implied "bigger is always better, smoothly." The residuals near the middle of the sweep were treated as noise. A later, denser sweep revealed a clear interpolation peak: the curve went down, up, and down again. The monotone fit had been dragged toward the peak and had under-predicted the over-parameterized descent, leading the team to under-size the production model. The fix was an $n=2$ BNSL with a sign-flipped slope change to model the hump, which correctly placed the second descent and recommended a larger model. The lesson: when sweeping width or epochs, expect the possibility of non-monotonicity and use a form that can represent it, or you will mistake a real hump for noise.

### 3. The emergent reasoning that was a sharp break

A reasoning benchmark showed near-zero exact-match accuracy across four model sizes, then a jump at the fifth. The first narrative was "the model suddenly acquired reasoning at scale." Two corrections followed. First, re-scoring with a continuous per-step metric showed steady improvement that the exact-match metric had been compressing — the [metric-artifact effect](/blog/machine-learning/scaling-laws/emergent-abilities-scaling). Second, the underlying loss curve fit a BNSL with a small $f_i$ (a sharp but smooth break), and that fit, trained on the first five points, correctly predicted the sixth. The capability was not a discontinuity; it was a sharp break under a discontinuous metric. The fix was to forecast capability on the continuous metric and the BNSL loss fit together, which made the "emergence" a predicted event rather than a surprise. The lesson: a jump under exact-match plus a smooth BNSL loss curve is the signature of a sharp break, not magic.

### 4. The break beyond the data

A team did everything right — collected ten well-spread points, fit an $n=1$ BNSL, passed leave-the-largest-out validation — and still over-predicted the capability of a much larger model. The root cause was a saturation break *beyond* their largest data point: the real curve flattened toward a floor at a scale they had never measured, and BNSL, having no points past it, extrapolated the last segment's slope straight through. This is the section-6 limit in the wild. The wrong hypothesis was "the fit was bad"; the fit was fine for the range it covered. The fix was conceptual: treat every BNSL extrapolation as valid only up to the last observed break, and pair it with a mechanistic prior (an estimate of the irreducible floor from the data entropy) to bound where saturation might be. The lesson: BNSL does not predict where breaks occur, so an extrapolation that crosses an unseen break is exactly as blind as a single power law on that segment.

### 5. The over-fit wiggle

In an attempt to "nail the fit," an analyst pushed $n$ to 4 on a study with only nine points. The training residuals went to nearly zero and the in-sample plot looked perfect. The extrapolation, however, was dominated by a final segment fit to two noisy points and pointed in a nonsensical direction, predicting the loss would rise sharply at scale. The wrong hypothesis was "the curve really does turn up at the end"; the actual cause was twelve parameters threading nine points. The fix was to drop to $n=1$, validate out of sample, and accept the less flattering but honest fit. The lesson: with BNSL the temptation to add breaks is strong because each one improves the in-sample fit, and the only antidote is to choose $n$ by held-out extrapolation error, which almost always wants fewer breaks than the eye does.

### 6. The RL plateau that finally broke

An RL agent's competence over environment steps sat nearly flat for a long stretch, and a power-law fit on that plateau predicted the agent would essentially never reach the target performance. The team nearly cancelled the run. Someone noticed the curve had the signature of a long flat first segment with a likely sharp break ahead and refit as a BNSL with a small-$f_i$ break placed just past the current data. The fit was speculative (the break was at the edge of the data), but it flagged that a jump was plausible, so the run continued. The agent's competence did jump sharply a while later, consistent with a sharp break. The honest caveat: BNSL could not *predict* the break location reliably from the plateau alone — the break was barely in range — but recognizing the *shape* as "flat segment likely followed by a break" prevented a premature cancellation. The lesson: knowing the BNSL vocabulary changes how you read a plateau; a flat power-law fit says "stuck forever," while the broken-form mindset asks "is there a break ahead I cannot yet see?"

### 7. The inverse-scaling U-turn

An alignment-adjacent metric got *worse* as models scaled, and a team fit a power law that, naturally, predicted it would keep getting worse without bound. They flagged it as a fundamental inverse-scaling problem. A denser study revealed a U-shape: the metric worsened, bottomed out, then improved at larger scale. A BNSL with a sign-flipping slope change fit the U and predicted the recovery, reframing "fundamental inverse scaling" as "a break we had not yet reached." The wrong hypothesis was monotone inverse scaling; the actual shape was non-monotone with a break. The fix was to use the non-monotonic form before declaring any inverse-scaling result fundamental. The lesson: a metric that worsens with scale over your observed range is not necessarily monotone outside it, and a form that can flip slope sign is the right tool to test whether a U-turn is coming.

### 8. The held-out test that saved a budget

A team building a forecast for a flagship run made leave-the-largest-out validation a hard gate: no extrapolation was trusted unless a fit on all-but-the-largest point predicted the largest point within error. On one project, the $n=2$ fit interpolated beautifully but failed the held-out test, while the $n=1$ fit passed. They shipped the forecast from the $n=1$ fit, which proved accurate on the real run, and the $n=2$ fit (which would have predicted a more optimistic loss) would have led them to under-budget. The lesson: in-sample fit quality is nearly useless for choosing $n$; held-out extrapolation accuracy is the only metric that correlates with real forecast quality, and enforcing it as a gate is the single highest-leverage discipline when using BNSL.

### 9. The sigmoid that capped a ceiling

A capability forecast for a benchmark with an apparent S-shaped rise was modeled with a logistic curve in log-compute, which fit the observed transition cleanly and predicted the metric would plateau just above the largest observed point. The team planned around that ceiling, sizing the next model only large enough to reach it. The real run blew past the predicted ceiling, because the metric kept improving as a power law past the transition rather than flattening. The root cause was the sigmoid's two-sided saturation: it assumes a hard upper asymptote that the power-law tail does not respect. Refitting the same data with an $n=1$ BNSL — power-law segments on both sides of the break — reproduced the transition *and* the continued post-break improvement, and predicted the higher real value. The lesson: a sigmoid and a BNSL agree on the transition and disagree on the tails, and for loss and capability curves the power-law tail is the empirically correct one, so a sigmoid systematically under-predicts past the break.

### 10. The break that moved under a metric change

A team tracked a downstream task on exact-match accuracy and fit a BNSL with a sharp break at one model scale. When they later re-scored the same checkpoints on a continuous per-token metric, the curve smoothed dramatically and the best fit moved to a much larger $f_i$ (a gentle bend) at a slightly different location. Both fits were "correct" for their respective metrics, which exposed something important: the break parameters are properties of the *metric-transformed* curve, not of the model in some absolute sense. The wrong assumption had been that "the break location" was a fixed physical fact about the model. The fix was to standardize on the continuous metric for all forecasting, fit the BNSL on that, and treat the exact-match curve only as a reporting view. The lesson: BNSL faithfully fits whatever curve you give it, so the metric you choose *is* part of the model — pick the continuous one, the same way the [metric-artifact analysis](/blog/machine-learning/scaling-laws/emergent-abilities-scaling) recommends, and your breaks become gentle and forecastable instead of sharp and metric-dependent.

### 11. The data-axis break mistaken for a model-axis break

A study varied both model size and dataset size and saw a bend in the loss curve, which a junior analyst attributed to a model-size break and fit with a BNSL along $N$. The extrapolation along $N$ failed on the next run. The actual cause was that the bend lived along the *data* axis — the run had crossed into a repeated-data regime where extra epochs stopped helping, the [data-constrained scaling](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) effect — and the apparent model-axis break was an artifact of confounding the two axes in a single fit. The fix was to fit the bend along the correct axis (data) with the right machinery, and to hold the other axis fixed when fitting a single-axis BNSL. The lesson: BNSL fits a curve along one axis at a time, so before you fit, make sure the break you see actually belongs to the axis you are fitting — a break on a confounded axis will fit fine and extrapolate wrong, which is the single-axis version of the "wrong slope, confident forecast" failure that opened this post.

## When to reach for BNSL, and when not to

Reach for BNSL when:

- Your scaling curve has a **visible bend** on log-log axes — a slope change, a saturation, an inflection — that a single power law leaves with structured residuals.
- You are dealing with a **non-monotonic** curve (double descent, a U-shape, inverse-then-normal scaling) that a monotone power law cannot represent at all.
- You are studying an **apparently emergent** capability and want a fittable, extrapolatable model of the inflection rather than a verbal claim that it is a discontinuity — pair it with a continuous metric.
- You have **enough data points** (well more than $3n+3$, with points on both sides of every break) to support the parameter count honestly.
- You can run and respect a **leave-the-largest-out validation** to choose $n$ and to keep yourself from over-fitting.

Skip BNSL when:

- The curve is **straight on log-log** in your range — use the [single power law or Chinchilla form](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling); adding breaks to a straight curve is over-fitting.
- You have **too few points** to fit $3n+3$ parameters — a six-parameter fit on five points is fiction; default to $n=0$.
- You need to **predict where a break will be** at a scale you have not measured — BNSL is descriptive, not predictive, of break locations; reach for [mechanistic theory](/blog/machine-learning/scaling-laws/why-power-laws-arise) and a floor estimate instead.
- You are tempted to **extrapolate past your last observed break** — on that final segment BNSL is exactly as blind as a single power law, so bound the forecast and do not over-trust it.
- The cost of a wrong forecast is low and a **rough power-law estimate is good enough** — the extra parameters are not worth the fitting and validation overhead.

The summary you should carry: a single power law is the right first model and the wrong last word. It extrapolates cleanly precisely because it cannot bend, and real scaling curves bend — into floors, into humps, into sharp inflections that get called emergence. BNSL is the disciplined way to model those bends: a smooth stitch of $n+1$ power laws, $3n+3$ parameters, fit by least squares, validated out of sample. It reframes emergence as a sharp-but-smooth break, complementing the metric-artifact story, and it extrapolates better than a single power law across an unusually broad sweep of domains. Just never forget the two limits — you need the points, and it cannot see past the last break in your data. Use the simple law until the residuals break it, then break the law on purpose.

## Further reading

- Caballero, Gupta, Rish, Krueger 2022, "Broken Neural Scaling Laws" — https://arxiv.org/abs/2210.14891
- Wei et al. 2022, "Emergent Abilities of Large Language Models" — https://arxiv.org/abs/2206.07682
- Schaeffer, Miranda, Koyejo 2023, "Are Emergent Abilities of Large Language Models a Mirage?" — https://arxiv.org/abs/2304.15004
- Du et al. 2024, "Understanding Emergent Abilities of Language Models from the Loss Perspective" — https://arxiv.org/abs/2403.15796
- Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models" (Chinchilla) — https://arxiv.org/abs/2203.15556
- Nakkiran et al. 2019, "Deep Double Descent: Where Bigger Models and More Data Hurt" — https://arxiv.org/abs/1912.02292
- Related posts: [scaling laws from scratch](/blog/machine-learning/scaling-laws/scaling-laws-predictability-foundations), [emergent abilities](/blog/machine-learning/scaling-laws/emergent-abilities-scaling), [why power laws arise](/blog/machine-learning/scaling-laws/why-power-laws-arise)

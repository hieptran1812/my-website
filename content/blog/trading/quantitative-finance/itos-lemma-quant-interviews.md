---
title: "Ito's lemma for quant interviews: the chain rule with one extra term"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch guide to Ito's lemma for quant researcher and derivatives interviews: why (dB)^2 = dt keeps a second-order Taylor term alive, the stochastic multiplication table, the d(log S) drift correction for geometric Brownian motion, the Ito product rule, and five fully solved interview problems."
tags:
  [
    "itos-lemma",
    "stochastic-calculus",
    "quant-interviews",
    "geometric-brownian-motion",
    "brownian-motion",
    "black-scholes",
    "martingale",
    "ito-integral",
    "quantitative-finance",
    "derivatives-pricing",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Ito's lemma is the chain rule of stochastic calculus, and it is the single most-asked stochastic-calculus tool in quant researcher and derivatives interviews. The whole trick is one extra term.
>
> - In ordinary calculus, if $x$ moves smoothly, $df = f'(x)\,dx$ and second-order terms vanish. For a process driven by Brownian noise, a second-order term **survives** because of one rule: $(dB)^2 = dt$.
> - Ito's lemma for $f(X)$ reads $df = f'(X)\,dX + \tfrac{1}{2} f''(X)\,(dX)^2$. The $\tfrac{1}{2} f''$ term is the *only* difference from the chain rule you already know.
> - The mechanical engine is the **stochastic multiplication table**: $dt \cdot dt = 0$, $dt \cdot dB = 0$, $dB \cdot dB = dt$. Memorize it and most problems become arithmetic.
> - The headline application: applying Ito to $\log S$ for a stock gives a drift of $\mu - \tfrac{1}{2}\sigma^2$, not $\mu$. That $-\tfrac{1}{2}\sigma^2$ is why log-returns are pulled below the naive growth rate, and on a \$100 stock at 40% volatility the drag is a concrete 8% per year.
> - This one identity also builds the Black-Scholes PDE on a real options desk, so interviewers love it: it tests whether you understand *why* volatility creates a correction, not just that it does.

Here is a question that has ended more quant interviews than almost any other: "A stock follows $dS = \mu S\,dt + \sigma S\,dB$. What is the process for $\log S$?"

If you reach for the ordinary chain rule and answer $d(\log S) = \tfrac{1}{S}\,dS = \mu\,dt + \sigma\,dB$, you have just made the single most common mistake in stochastic calculus, and the interviewer already knows how the rest of the conversation will go. The correct answer carries an extra piece: $d(\log S) = (\mu - \tfrac{1}{2}\sigma^2)\,dt + \sigma\,dB$. That stray $-\tfrac{1}{2}\sigma^2$ is not a trick or a convention. It is a real, dollars-and-cents correction that comes from the fact that Brownian motion is so jagged that its squared wiggles refuse to disappear.

This article builds the whole machine from zero. We assume you have seen the ordinary chain rule from first-year calculus and nothing else about stochastic processes. By the end you will be able to apply Ito's lemma to anything an interviewer throws at you, explain *why* the extra term exists in plain English, and recognize the half-dozen canonical setups (geometric Brownian motion, the exponential martingale, Ornstein-Uhlenbeck, the Black-Scholes PDE) that show up again and again on the whiteboard.

![Ito's lemma is the ordinary chain rule plus one extra second-order term, kept alive by the rule that dB squared equals dt; ordinary calculus produces one term and the slope times dx, while the stochastic version produces two terms, the slope plus a curvature-driven drift.](/imgs/blogs/itos-lemma-quant-interviews-1.png)

The diagram above is the mental model for the entire post. On the left is the world you already know: a smooth variable, where the chain rule gives a single first-order term and everything of second order or higher dies off. On the right is the stochastic world: a variable jittered by Brownian noise, where a second-order term refuses to die and instead becomes a genuine *drift* in the answer. Everything else here is just learning to compute that second term reliably and to read off what it means.

## Foundations: the one new ingredient

Before we can state Ito's lemma, we need to be precise about three things a beginner has not necessarily seen: what a stochastic process is, what Brownian motion is, and why squaring its increments is the whole story. We will define every term as it appears.

### The ordinary chain rule, recalled exactly

Start with safe ground. In ordinary calculus, suppose $y = f(x)$ and $x$ changes by a tiny amount $dx$. A *Taylor expansion* — the standard way of approximating a function near a point using its derivatives — says

$$
df = f'(x)\,dx + \tfrac{1}{2} f''(x)\,(dx)^2 + \tfrac{1}{6} f'''(x)\,(dx)^3 + \cdots
$$

Here $f'$ is the first derivative (the slope), $f''$ is the second derivative (the curvature), and so on. Now comes the step everyone takes without noticing: when $x$ is a smooth function of time and $dx$ is genuinely infinitesimal, the term $(dx)^2$ is *much* smaller than $dx$. If $dx$ is one-thousandth, then $(dx)^2$ is one-millionth — negligible. So we throw away every term beyond the first and write the chain rule you memorized:

$$
df = f'(x)\,dx.
$$

The key fact to hold onto: **we discard $(dx)^2$ because, for a smooth path, it shrinks faster than $dx$.** That assumption is what is about to break.

### What is a stochastic process?

A *stochastic process* is just a quantity that evolves randomly over time — a stock price, an interest rate, the position of a particle in water. Instead of one fixed trajectory, you get a whole family of possible trajectories, each occurring with some probability. We will write $X_t$ for the value of the process at time $t$.

The standard way to describe how a process moves is a *stochastic differential equation* (SDE), which says how the process changes over an instant $dt$:

$$
dX_t = a(X_t, t)\,dt + b(X_t, t)\,dB_t.
$$

In plain terms: over a tiny slice of time $dt$, the process moves by a predictable amount $a\,dt$ (the *drift*) plus a random kick $b\,dB_t$ (the *diffusion* or *volatility* term). The function $a$ is the **drift coefficient** — the average direction of motion. The function $b$ is the **diffusion coefficient** — how strongly randomness buffets the process. And $dB_t$ is the source of the randomness, which we define next.

### What is Brownian motion?

*Brownian motion* (also called a *Wiener process*), written $B_t$, is the canonical random driver of continuous-time finance. You can define it by three properties, all of which we will use:

1. **It starts at zero:** $B_0 = 0$.
2. **Its increments are independent and normally distributed.** Over a time interval of length $\Delta t$, the change $B_{t+\Delta t} - B_t$ is drawn from a normal distribution with mean $0$ and variance $\Delta t$. (A *normal distribution*, or bell curve, is the familiar symmetric distribution; its *variance* measures how spread out it is.) In shorthand, $\Delta B \sim \mathcal{N}(0, \Delta t)$.
3. **Its paths are continuous but nowhere smooth** — they have no well-defined slope anywhere. A Brownian path is infinitely jagged; you can zoom in forever and it never straightens out.

Property 2 is the load-bearing one. It says the *variance* of a Brownian increment equals the elapsed time. The standard deviation — the square root of the variance — is therefore $\sqrt{\Delta t}$. This single fact, $\Delta B \sim \sqrt{\Delta t}$ in size, is why stochastic calculus differs from ordinary calculus. Let us see exactly how.

### A discrete bridge: the scaled random walk

If continuous Brownian motion feels too abstract, build it from coin flips — this is how it is constructed rigorously, and it shows precisely where the $\sqrt{\Delta t}$ comes from. Take a fair coin, flip it every $\Delta t$ seconds, and step up or down by $\pm h$ each flip. After $n = t/\Delta t$ flips, your position is the sum of $n$ independent $\pm h$ steps. Each step has mean $0$ and variance $h^2$, so the total position has mean $0$ and variance $n h^2 = (t/\Delta t)\,h^2$.

Now we ask: how must the step size $h$ scale with $\Delta t$ so that this random walk converges to something sensible as we flip faster and faster ($\Delta t \to 0$)? If we want the variance at time $t$ to stay fixed at $t$ (so the limit has the Brownian property "variance equals time"), we need $(t/\Delta t)\,h^2 = t$, which forces $h = \sqrt{\Delta t}$. **The step size must scale as the square root of the time step, not linearly.** That is the whole secret. A smooth path would have steps proportional to $\Delta t$ (velocity times time); Brownian motion has steps proportional to $\sqrt{\Delta t}$, which is enormously larger when $\Delta t$ is tiny. That extra size is exactly why $(\Delta B)^2 \approx \Delta t$ survives where $(\Delta x)^2 \approx (\Delta t)^2$ dies. Everything downstream — the multiplication table, the Ito correction, the drift on $\log S$ — is a consequence of steps of size $\sqrt{\Delta t}$. If you internalize one mechanical fact about Brownian motion, make it this one.

### The crucial heuristic: $(dB)^2 = dt$

In ordinary calculus we threw away $(dx)^2$ because it was order $(dt)^2$ — vanishingly small. Now look at what happens to the random increment. A Brownian step over time $dt$ has size of order $\sqrt{dt}$, not $dt$. So when we square it:

$$
(dB)^2 \approx (\sqrt{dt})^2 = dt.
$$

The square of the increment is order $dt$ — exactly the same order as the drift term $dt$ that we *keep*. It does not vanish. This is the entire reason Ito's lemma needs a correction the ordinary chain rule does not.

More carefully: the increment $\Delta B$ is a random variable with mean $0$ and variance $\Delta t$. Its square $(\Delta B)^2$ has expected value $\mathbb{E}[(\Delta B)^2] = \text{Var}(\Delta B) = \Delta t$. And — this is the deep part — as you cut the time interval into more and more pieces and add up the squared increments, the random fluctuations average out and the total converges to the elapsed time *with certainty*, not just on average. That sum-of-squared-increments quantity is called the **quadratic variation**, and for Brownian motion the quadratic variation over $[0, t]$ equals exactly $t$.

![Refining the grid locks the total squared variation of Brownian motion onto t rather than shrinking it to zero, which is precisely what the rule dB squared equals dt encodes; the bars of squared variation hover around the dashed line at t equals one for n equals 4 up through n equals 1024 steps.](/imgs/blogs/itos-lemma-quant-interviews-2.png)

#### Worked example: confirming $(dB)^2 = dt$ numerically

Let us make this concrete with numbers. Take the interval $[0, 1]$, so the total time is $t = 1$. Chop it into $n$ equal sub-intervals, each of length $1/n$. Over each piece, the Brownian increment $\Delta B_i$ is normal with variance $1/n$.

Now form the **realized quadratic variation** — the running total of squared increments:

$$
Q_n = \sum_{i=1}^{n} (\Delta B_i)^2.
$$

What is the expectation? Each $(\Delta B_i)^2$ has expected value $1/n$, and there are $n$ of them, so

$$
\mathbb{E}[Q_n] = n \cdot \tfrac{1}{n} = 1 = t.
$$

The expectation is $1$ for *every* $n$. But more striking is the variance. Each squared increment $(\Delta B_i)^2$ has variance $2 (1/n)^2$ (a fact about the chi-squared distribution that a normal-squared follows). Summing $n$ independent ones:

$$
\text{Var}(Q_n) = n \cdot 2 \left(\tfrac{1}{n}\right)^2 = \frac{2}{n}.
$$

As $n \to \infty$, the variance goes to zero. So $Q_n$ does not merely *average* to $1$ — it *collapses* onto $1$. With $n = 4$ steps the total is noisy (in the figure, $0.78$); by $n = 1024$ it is pinned near $0.99$. In the limit, $\sum (\Delta B_i)^2 \to t$ with probability one. **This is the rigorous statement behind the heuristic $(dB)^2 = dt$: the squared wiggles of Brownian motion add up to elapsed time, not to zero.** That is the one new fact you need; everything else is bookkeeping.

## Ito's lemma, stated for $f(X)$

Now we can state the theorem. Take a process $X_t$ following the SDE $dX = a\,dt + b\,dB$, and a twice-differentiable function $f$. We want the process for $Y = f(X)$. Write the Taylor expansion of $f$ exactly as before, but this time **keep the second-order term**, because $(dX)^2$ will not vanish:

$$
df = f'(X)\,dX + \tfrac{1}{2} f''(X)\,(dX)^2 + \cdots
$$

The third-order and higher terms involve $(dX)^3$ and smaller, which *are* negligible (they are order $(dt)^{3/2}$ and below). So we stop after two terms. This is **Ito's lemma** in its most compact form:

$$
\boxed{\;df = f'(X)\,dX + \tfrac{1}{2} f''(X)\,(dX)^2\;}
$$

Compare it to the ordinary chain rule $df = f'(x)\,dx$. The *only* difference is the extra $\tfrac{1}{2} f''(X)\,(dX)^2$ term. If you remember nothing else, remember this: **Ito's lemma is the chain rule plus one half times the second derivative times the increment squared.**

The whole game is now computing $(dX)^2$. And here is where the multiplication table does the work.

### The stochastic multiplication table

To evaluate $(dX)^2$ when $dX = a\,dt + b\,dB$, expand the square:

$$
(dX)^2 = (a\,dt + b\,dB)^2 = a^2 (dt)^2 + 2ab\,(dt)(dB) + b^2 (dB)^2.
$$

Now apply three rules, which together form the **stochastic multiplication table**:

- $dt \cdot dt = 0$ — because $(dt)^2$ is order $(dt)^2$, which vanishes (same reason as ordinary calculus).
- $dt \cdot dB = 0$ — because $(dt)(dB)$ is order $dt \cdot \sqrt{dt} = (dt)^{3/2}$, which also vanishes.
- $dB \cdot dB = dt$ — because, as we proved, the squared Brownian increment is order $dt$ and converges to it.

![In the stochastic multiplication table only the product of dB with dB survives and equals dt, while every product involving dt collapses to zero, which is the entire computational engine of Ito's lemma.](/imgs/blogs/itos-lemma-quant-interviews-3.png)

Only the last entry is nonzero. So the messy square collapses to a single surviving term:

$$
(dX)^2 = b^2\,(dB)^2 = b^2\,dt.
$$

Substituting back into Ito's lemma gives the form you will actually use in practice. For $dX = a\,dt + b\,dB$ and $Y = f(X)$:

$$
\boxed{\;df = \left[\, a\,f'(X) + \tfrac{1}{2} b^2 f''(X) \,\right] dt + b\,f'(X)\,dB\;}
$$

Read this carefully, because it is the workhorse. The new process $f(X)$ is again an Ito process — it has its own drift and its own diffusion. The diffusion is just $b\,f'(X)$, the chain rule applied to the random part. The drift has *two* pieces: the ordinary chain-rule piece $a\,f'(X)$, plus the **Ito correction** $\tfrac{1}{2} b^2 f''(X)$. That correction is the entire content of the lemma. When $f$ is curved (nonzero $f''$) and there is volatility (nonzero $b$), a drift appears out of nowhere.

It helps to see the ordinary and stochastic chain rules side by side, with the exact place they diverge:

| Question | Ordinary calculus | Ito calculus |
|---|---|---|
| Increment size | $dx \sim dt$ (smooth) | $dB \sim \sqrt{dt}$ (jagged) |
| Squared increment | $(dx)^2 \sim (dt)^2 \to 0$ | $(dB)^2 = dt$ (survives) |
| Chain rule | $df = f'\,dx$ | $df = f'\,dX + \tfrac{1}{2} f''\,(dX)^2$ |
| Number of drift terms | one | two (chain rule + Ito correction) |
| Product rule | $d(XY) = X\,dY + Y\,dX$ | $d(XY) = X\,dY + Y\,dX + dX\,dY$ |
| Source of the extra term | none | the surviving $dB\cdot dB = dt$ |

Every row of that table is a consequence of the single fact in the top-right cell. If an interviewer asks you to "explain the difference between Ito and ordinary calculus in one sentence," point at that cell: in the smooth world squared increments vanish, in the Brownian world they equal $dt$, and that one survival propagates into an extra term everywhere.

#### Worked example: $d(B^2)$, and why it equals $B^2 - t$

Let us run the machine on the simplest interesting case. Take $X = B$ itself, so $dX = dB$ (drift $a = 0$, diffusion $b = 1$), and let $f(x) = x^2$. Then $f'(x) = 2x$ and $f''(x) = 2$. Plug into Ito:

$$
d(B^2) = f'(B)\,dB + \tfrac{1}{2} f''(B)\,(dB)^2 = 2B\,dB + \tfrac{1}{2}(2)\,dt = 2B\,dB + dt.
$$

Notice the $dt$ term — it appeared from nowhere, purely from the curvature of $x^2$ acting on $(dB)^2 = dt$. The naive chain-rule answer would have been just $2B\,dB$, missing the drift entirely.

Now integrate both sides from $0$ to $t$. The left side is $B_t^2 - B_0^2 = B_t^2$ (since $B_0 = 0$). The right side is $\int_0^t 2B_s\,dB_s + \int_0^t dt = \int_0^t 2B_s\,dB_s + t$. So:

$$
B_t^2 = \int_0^t 2B_s\,dB_s + t \quad\Longrightarrow\quad B_t^2 - t = \int_0^t 2B_s\,dB_s.
$$

Why is this a famous result? Because the right-hand side is an **Ito integral** (an integral against $dB$), which we will see always has expected value zero. Taking expectations: $\mathbb{E}[B_t^2 - t] = 0$, so $\mathbb{E}[B_t^2] = t$ — which we already knew, since $B_t \sim \mathcal{N}(0, t)$. More importantly, $M_t = B_t^2 - t$ is a **martingale** — a process whose expected future value, given everything known today, equals its current value. The $-t$ is exactly the correction that turns the upward-drifting $B_t^2$ into a fair, driftless process. **The intuition to carry away: squaring a Brownian motion creates an upward drift of one unit per unit time, and subtracting $t$ cancels it.**

#### Worked example: $d(B^3)$ and a higher-order pattern

The same machine handles any power. Take $f(x) = x^3$, so $f'(x) = 3x^2$ and $f''(x) = 6x$. With $X = B$:

$$
d(B^3) = 3B^2\,dB + \tfrac{1}{2}(6B)\,dt = 3B^2\,dB + 3B\,dt.
$$

Integrating, $B_t^3 = \int_0^t 3B_s^2\,dB_s + \int_0^t 3B_s\,ds$. The first integral is a mean-zero Ito integral; the second has expectation $\int_0^t 3\,\mathbb{E}[B_s]\,ds = 0$ since $\mathbb{E}[B_s] = 0$. So $\mathbb{E}[B_t^3] = 0$, confirming what symmetry already tells us: odd moments of a mean-zero normal vanish. **The takeaway: the correction term scales with the second derivative, so steeper curvature means a bigger drift adjustment** — and Ito turns moment calculations into pure mechanics.

#### Worked example: $d(e^{B})$ and the exponential's self-drift

Now the exponential, which is the gateway to geometric Brownian motion. Let $f(x) = e^x$, so $f'(x) = e^x$ and $f''(x) = e^x$ — the exponential is its own derivative, which is what makes it special. With $X = B$:

$$
d(e^{B}) = e^{B}\,dB + \tfrac{1}{2} e^{B}\,dt = e^{B}\left(dB + \tfrac{1}{2}\,dt\right).
$$

Look at the drift: $\tfrac{1}{2} e^{B}\,dt$. The exponential of a *driftless* Brownian motion acquires a *positive* drift of one-half. This is a manifestation of a general truth — *Jensen's inequality*, which says the average of a convex function exceeds the function of the average. The exponential is convex (it curves upward), so randomness pushes its expectation up. Concretely, $\mathbb{E}[e^{B_t}] = e^{t/2} > 1 = e^{\mathbb{E}[B_t]}$. **The lesson: convexity plus volatility always manufactures upward drift; the size of that drift is exactly the Ito correction term.** Hold this thought — we are about to cancel it deliberately.

## The crown jewel: $d(\log S)$ for geometric Brownian motion

This is the application interviewers ask most, because it is the foundation of Black-Scholes and the reason log-returns behave the way they do. **Geometric Brownian motion** (GBM) is the standard model for a stock price: it says the *percentage* change of the price, not the dollar change, is what has constant drift and volatility. The SDE is

$$
dS = \mu S\,dt + \sigma S\,dB,
$$

where $\mu$ is the **drift rate** (the expected continuous return per year, e.g. $0.10$ for 10%) and $\sigma$ is the **volatility** (the standard deviation of returns per year, e.g. $0.40$ for 40%). Both $\mu$ and $\sigma$ multiply $S$ itself, which is what makes returns rather than dollar amounts stationary, and what guarantees the price stays positive.

We want the process for $\log S$. Why $\log$? Because $\log$ turns the multiplicative GBM into something additive and solvable, and because the *log-return* $\log(S_t / S_0)$ is the natural unit of return. Apply Ito with $f(S) = \log S$, so $f'(S) = 1/S$ and $f''(S) = -1/S^2$.

![Applying Ito to log S of a geometric Brownian motion converts the half second-derivative term into the minus half sigma-squared drift correction, in four steps: set up f equals log S, write the Ito expansion, square dS using dB squared equals dt, and read off that log-returns drift at mu minus half sigma-squared rather than mu.](/imgs/blogs/itos-lemma-quant-interviews-4.png)

#### Worked example: deriving the $-\tfrac{1}{2}\sigma^2$ drift correction

Step by step, following the figure.

**Step 1 — derivatives.** With $f(S) = \log S$: $f'(S) = \dfrac{1}{S}$ and $f''(S) = -\dfrac{1}{S^2}$.

**Step 2 — write Ito.** 
$$
d(\log S) = f'(S)\,dS + \tfrac{1}{2} f''(S)\,(dS)^2 = \frac{1}{S}\,dS - \frac{1}{2 S^2}\,(dS)^2.
$$

**Step 3 — compute $(dS)^2$.** Square the SDE and apply the multiplication table. Only the $dB \cdot dB = dt$ term survives:
$$
(dS)^2 = (\mu S\,dt + \sigma S\,dB)^2 = \sigma^2 S^2\,(dB)^2 = \sigma^2 S^2\,dt.
$$

**Step 4 — substitute and simplify.** Plug $dS = \mu S\,dt + \sigma S\,dB$ and $(dS)^2 = \sigma^2 S^2\,dt$ into Step 2:
$$
d(\log S) = \frac{1}{S}\big(\mu S\,dt + \sigma S\,dB\big) - \frac{1}{2 S^2}\big(\sigma^2 S^2\,dt\big) = \mu\,dt + \sigma\,dB - \tfrac{1}{2}\sigma^2\,dt.
$$

Collect the $dt$ terms:
$$
\boxed{\;d(\log S) = \left(\mu - \tfrac{1}{2}\sigma^2\right)dt + \sigma\,dB\;}
$$

There it is. The log-price has drift $\mu - \tfrac{1}{2}\sigma^2$, not $\mu$. The correction $-\tfrac{1}{2}\sigma^2$ comes entirely from the negative curvature of $\log$ (the $f'' = -1/S^2$) acting on the surviving $(dS)^2 = \sigma^2 S^2\,dt$. **The one-sentence intuition: because $\log$ is concave, volatility drags the typical (median) log-return below the average dollar drift, and the size of that drag is exactly half the variance.**

Because $\log S$ now has *constant* drift and *constant* diffusion, we can integrate it directly. $\log S_t - \log S_0 = (\mu - \tfrac{1}{2}\sigma^2)t + \sigma B_t$, which exponentiates to the **closed-form GBM solution**:

$$
\boxed{\;S_t = S_0 \exp\!\left[\left(\mu - \tfrac{1}{2}\sigma^2\right)t + \sigma B_t\right]\;}
$$

![Integrating the log-price SDE gives the closed-form geometric Brownian motion solution S_t equals S_0 times the exponential of the corrected drift plus sigma B_t; every realized path is the green median exponential curve perturbed up by a lucky Brownian draw or down by an unlucky one, all starting from a hundred-dollar spot.](/imgs/blogs/itos-lemma-quant-interviews-5.png)

This formula is worth dwelling on. The median path — the one with $B_t = 0$ — grows like $S_0\,e^{(\mu - \frac{1}{2}\sigma^2)t}$, at the *corrected* rate. The random factor $e^{\sigma B_t}$ scatters individual outcomes above and below that median, as the figure shows with one lucky (upper) and one unlucky (lower) realized path. Crucially, the *mean* of $S_t$ still grows at $\mu$: $\mathbb{E}[S_t] = S_0\,e^{\mu t}$, because the upside of the lognormal distribution is fat enough to pull the average back up to $\mu$ even though the median grows only at $\mu - \tfrac{1}{2}\sigma^2$. The gap between the mean and the median *is* the Ito correction, made visible.

It is worth contrasting GBM with its simpler cousin, **arithmetic Brownian motion** (ABM), where the *dollar* change rather than the percentage change is what has constant drift and volatility: $dS = \mu\,dt + \sigma\,dB$. The two models behave very differently, and knowing when each applies is itself an interview topic:

| Property | Arithmetic BM ($dS = \mu\,dt + \sigma\,dB$) | Geometric BM ($dS = \mu S\,dt + \sigma S\,dB$) |
|---|---|---|
| What is constant | dollar drift and dollar volatility | percentage drift and percentage volatility |
| Can the price go negative? | yes (no floor) | no (always positive) |
| Distribution of $S_t$ | normal | lognormal |
| Ito correction on $\log$? | not applicable (use it on $S$) | yes, $-\tfrac{1}{2}\sigma^2$ |
| Typical use | spreads, interest-rate differences | stock prices, FX, equity indices |

The reason GBM, not ABM, models stock prices is that a \$1 move means something very different for a \$5 stock than for a \$500 stock, whereas a 1% move means the same thing for both. The cost of that realism is the lognormal distribution and the $-\tfrac{1}{2}\sigma^2$ correction — which is exactly the term Ito's lemma supplies.

#### Worked example: the drift drag in dollars on a \$100 stock

Abstract corrections feel academic until you price them. Take a stock at $S_0 = \$100$ with drift $\mu = 10\%$ per year and volatility $\sigma = 40\%$ per year — a realistic figure for a single growth stock. The variance drag is

$$
\tfrac{1}{2}\sigma^2 = \tfrac{1}{2}(0.40)^2 = \tfrac{1}{2}(0.16) = 0.08 = 8\%\ \text{per year}.
$$

So the *median* log-return drifts at only $\mu - \tfrac{1}{2}\sigma^2 = 10\% - 8\% = 2\%$ per year, while a naive investor who forgot Ito would expect the typical path to compound at the full $10\%$.

![On a hundred-dollar stock at forty percent volatility, the minus-half-sigma-squared variance drag is eight percent per year, so the corrected median path compounding at two percent ends near a hundred and four dollars after two years, far below the naive ten-percent path that ends near a hundred and twenty-two dollars; the roughly eighteen-dollar gap is pure variance drag before any luck.](/imgs/blogs/itos-lemma-quant-interviews-8.png)

Put it in dollars over a two-year horizon. The naive (mean) path reaches $100\,e^{0.10 \times 2} = 100\,e^{0.20} \approx \$122$. The corrected median path reaches only $100\,e^{0.02 \times 2} = 100\,e^{0.04} \approx \$104$. The roughly **\$18 gap** is not bad luck — it is the deterministic consequence of volatility acting through Ito's lemma. **The desk-level lesson: a volatile asset's *typical* outcome lags its *expected* outcome, and the lag is half the variance per unit time — which is exactly why you cannot use arithmetic-average returns to project the wealth of a single path.** This is also the mathematical core of why volatility is a cost to a compounding investor, a point we revisit when we talk about real desks. If you have read about the difference between arithmetic and geometric mean returns in the discussion of the [Kelly criterion and bet sizing](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews), this $-\tfrac{1}{2}\sigma^2$ is the continuous-time version of the same phenomenon.

## Ito's lemma for time-dependent $f(t, X)$

So far our function depended only on the state $X$. In practice — pricing an option, for instance — the function also depends explicitly on time, because an option's value changes as it approaches expiry even if the stock is frozen. We need the version of Ito's lemma for $f(t, X)$.

The derivation is the same Taylor expansion, now in two variables. Expanding $f(t + dt, X + dX)$ and keeping terms up to order $dt$:

$$
df = \frac{\partial f}{\partial t}\,dt + \frac{\partial f}{\partial X}\,dX + \tfrac{1}{2}\frac{\partial^2 f}{\partial X^2}\,(dX)^2.
$$

We write $f_t$, $f_X$, $f_{XX}$ for the partial derivatives to keep it readable. (There is no $\tfrac{1}{2} f_{tt}(dt)^2$ term — it is order $(dt)^2$ and dies — and no cross term $f_{tX}\,dt\,dX$ — it is order $(dt)^{3/2}$ and dies.) So the full statement is:

$$
\boxed{\;df = f_t\,dt + f_X\,dX + \tfrac{1}{2} f_{XX}\,(dX)^2\;}
$$

![Ito's lemma for a time-dependent function f of t and X has exactly three terms: the ordinary time-drift f_t dt, the ordinary state slope f_X dX from the chain rule, and the extra one-half f_XX times dX-squared variance term that survives because dB squared equals dt.](/imgs/blogs/itos-lemma-quant-interviews-6.png)

As the figure stresses, this is *three* terms, and only the last one is new relative to ordinary multivariable calculus. The first term $f_t\,dt$ is the ordinary partial-time drift (how $f$ changes just from the clock ticking). The second term $f_X\,dX$ is the ordinary chain rule on the state. The third term $\tfrac{1}{2} f_{XX}(dX)^2$ is the Ito correction — the same beast as before, now wearing a partial-derivative subscript. If you substitute $dX = a\,dt + b\,dB$ and use $(dX)^2 = b^2\,dt$, you get the fully expanded drift-plus-diffusion form:

$$
df = \left[ f_t + a\,f_X + \tfrac{1}{2} b^2 f_{XX} \right] dt + b\,f_X\,dB.
$$

This bracketed drift — $f_t + a f_X + \tfrac{1}{2} b^2 f_{XX}$ — is precisely the expression that becomes the Black-Scholes PDE once you demand that a hedged option portfolio be riskless. We will assemble that on the desk shortly.

## The Ito product rule (stochastic integration by parts)

One more tool, because interviewers love it: how do you differentiate the *product* of two Ito processes? In ordinary calculus, $d(XY) = X\,dY + Y\,dX$. In the stochastic world, there is — predictably — an extra term.

![The Ito product rule adds a cross-variation term dX dY that the ordinary product rule never carries; where ordinary calculus writes d of XY as X dY plus Y dX with no cross term, the Ito version adds dX dY which equals rho times sigma-X sigma-Y dt, coming straight from the rule dB dB equals dt.](/imgs/blogs/itos-lemma-quant-interviews-10.png)

To derive it, apply two-variable Ito to $f(X, Y) = XY$. The partials are $f_X = Y$, $f_Y = X$, $f_{XX} = 0$, $f_{YY} = 0$, and the cross partial $f_{XY} = 1$. The two-variable Ito expansion (now with a cross term because there are two state variables) gives

$$
d(XY) = Y\,dX + X\,dY + dX\,dY.
$$

The new piece is $dX\,dY$, the **cross-variation**. If $X$ and $Y$ are driven by Brownian motions $B^X$ and $B^Y$ with correlation $\rho$, then evaluating $dX\,dY$ with the multiplication table (only the $dB^X dB^Y = \rho\,dt$ entry survives) gives

$$
dX\,dY = \rho\,\sigma_X\,\sigma_Y\,dt,
$$

where $\sigma_X$ and $\sigma_Y$ are the diffusion coefficients of the two processes and $\rho$ is the **correlation** between their Brownian drivers (a number between $-1$ and $+1$). **The Ito product rule:**

$$
\boxed{\;d(XY) = X\,dY + Y\,dX + dX\,dY, \qquad dX\,dY = \rho\,\sigma_X\,\sigma_Y\,dt.\;}
$$

When the two processes are uncorrelated ($\rho = 0$) or one of them is smooth (a finite-variation process with no $dB$ term, so $\sigma = 0$), the cross term vanishes and you recover the ordinary product rule. The cross term matters precisely when *both* legs are genuinely random and correlated — for instance, when you hold a position whose value depends on two correlated assets. This is also the rule that underlies "integration by parts" for stochastic integrals, which we use in the Ornstein-Uhlenbeck example below.

## In the interview room: seven fully solved problems

Here is where the technique earns its keep. These are the patterns that come up, lightly disguised, at Jane Street, Two Sigma, Citadel, DE Shaw, and SIG. We solve each from first principles, narrating the moves an interviewer wants to hear. The goal is not just the right final line — it is showing the interviewer a reliable *procedure*: write $f'$ and $f''$, square the SDE through the multiplication table, assemble the drift and diffusion, then read off the consequence.

#### Worked example: Problem 1 — Show $M_t = e^{B_t - t/2}$ is a martingale

**Setup.** A *martingale* is a process with no drift — formally, $\mathbb{E}[M_T \mid \mathcal{F}_t] = M_t$, meaning its expected future value given everything known up to time $t$ equals its present value. The cleanest way to prove a process is a martingale is to compute its SDE with Ito and show the $dt$ (drift) term is exactly zero. A driftless Ito process is a martingale, because the only thing left is the mean-zero $dB$ part.

**Solve.** Let $f(t, x) = e^{x - t/2}$ and apply two-variable Ito with $X = B$, so $a = 0$, $b = 1$. The partials are
$$
f_t = -\tfrac{1}{2} e^{x - t/2}, \qquad f_x = e^{x - t/2}, \qquad f_{xx} = e^{x - t/2}.
$$
So
$$
dM = f_t\,dt + f_x\,dB + \tfrac{1}{2} f_{xx}\,(dB)^2 = -\tfrac{1}{2} M\,dt + M\,dB + \tfrac{1}{2} M\,dt = M\,dB.
$$
The two $\tfrac{1}{2} M\,dt$ terms cancel exactly: the $-\tfrac{1}{2}$ from the explicit time-decay and the $+\tfrac{1}{2}$ from the Ito correction. The drift is zero, so $dM = M\,dB$ has no $dt$ term and $M_t = e^{B_t - t/2}$ is a martingale.

![The exponential martingale M_t equals the exponential of B_t minus t over two has zero drift because the minus t-over-two correction exactly cancels the drift Ito creates, so individual paths wander up and down while the expectation E of M_t stays pinned flat at one.](/imgs/blogs/itos-lemma-quant-interviews-7.png)

**Read off the consequence.** A martingale's expectation is constant, so $\mathbb{E}[M_t] = M_0 = e^{0 - 0} = 1$ for all $t$. As the figure shows, individual paths fan out, but the green expectation line stays flat at $1$. The $-t/2$ in the exponent is the "price of admission" that converts the convexity-driven upward drift of $e^{B_t}$ (which we found earlier was $\tfrac{1}{2} e^{B}\,dt$) into a fair process. **Interview-ready summary: the martingale correction $-t/2$ is the negative of the Ito drift of $e^{B}$, by construction.** This object, $e^{B_t - t/2}$, is the simplest example of a *stochastic exponential* and is the seed of the Radon-Nikodym derivative used in change-of-measure arguments for [risk-neutral pricing](/blog/trading/quantitative-finance/derivatives-pricing) — a connection a strong candidate names without being asked.

#### Worked example: Problem 2 — Apply Ito to an Ornstein-Uhlenbeck process

**Setup.** The *Ornstein-Uhlenbeck* (OU) process is the canonical model for a quantity that mean-reverts — pulls back toward a long-run level — such as an interest rate, a volatility, or a spread. Its SDE is
$$
dX = \theta(\mu - X)\,dt + \sigma\,dB,
$$
where $\mu$ is the long-run mean, $\theta > 0$ is the speed of mean reversion, and $\sigma$ is the volatility. When $X$ is above $\mu$, the drift $\theta(\mu - X)$ is negative and pulls it down; when below, it pulls up. The interviewer wants the closed-form solution.

![Applying Ito to an Ornstein-Uhlenbeck process shows mean reversion: the drift theta times mu minus X pulls the level X back toward its long-run mean mu equals five, so a path starting at eight point five is dragged down and settles into a band around the mean while sigma dB keeps it jittering.](/imgs/blogs/itos-lemma-quant-interviews-11.png)

**Solve with the integrating factor.** The trick — and this is the move interviewers reward — is to apply Ito to $Y = e^{\theta t} X$, an "integrating factor" that cancels the mean-reversion drift. Let $f(t, x) = e^{\theta t} x$. The partials: $f_t = \theta e^{\theta t} x$, $f_x = e^{\theta t}$, $f_{xx} = 0$. Since $f_{xx} = 0$, the Ito correction term vanishes and we just get
$$
dY = f_t\,dt + f_x\,dX = \theta e^{\theta t} X\,dt + e^{\theta t}\,dX.
$$
Substitute $dX = \theta(\mu - X)\,dt + \sigma\,dB$:
$$
dY = \theta e^{\theta t} X\,dt + e^{\theta t}\big[\theta(\mu - X)\,dt + \sigma\,dB\big] = \theta \mu\,e^{\theta t}\,dt + \sigma e^{\theta t}\,dB.
$$
The $X$ terms cancelled — that was the whole point of the integrating factor. Now $dY$ has no $X$-dependence, so we integrate directly from $0$ to $t$:
$$
Y_t - Y_0 = \theta \mu \int_0^t e^{\theta s}\,ds + \sigma \int_0^t e^{\theta s}\,dB_s = \mu\big(e^{\theta t} - 1\big) + \sigma \int_0^t e^{\theta s}\,dB_s.
$$
Recalling $Y_t = e^{\theta t} X_t$ and $Y_0 = X_0$, divide through by $e^{\theta t}$:
$$
\boxed{\;X_t = X_0 e^{-\theta t} + \mu\big(1 - e^{-\theta t}\big) + \sigma \int_0^t e^{-\theta(t-s)}\,dB_s\;}
$$

**Read it off.** The first term decays the initial condition away. The second term pulls the level toward $\mu$ as $t$ grows (the figure starts at $X_0 = 8.5$ and is dragged toward $\mu = 5$). The third term is a mean-zero Ito integral that keeps the process jittering. As $t \to \infty$, $X_t$ settles into a stationary distribution: normal with mean $\mu$ and variance $\sigma^2 / (2\theta)$. **Interview summary: mean-reverting SDEs are solved by the $e^{\theta t}$ integrating factor, which kills the $X$-drift and leaves an integrable form; because $\log$ is absent and $f_{xx} = 0$, there is no Ito correction here — a useful tell that not every Ito problem produces a $\tfrac{1}{2}\sigma^2$ term.** This is exactly the dynamics underlying the [Vasicek short-rate model](/blog/trading/quantitative-finance/short-rate-models-vasicek-hull-white) for interest rates.

#### Worked example: Problem 3 — Compute the expectation of an Ito integral

**Setup.** The interviewer writes $\displaystyle I_t = \int_0^t B_s\,dB_s$ and asks for $\mathbb{E}[I_t]$ and then for the value of the integral in closed form. This tests whether you know the two facts that make Ito integrals tractable: their expectation is zero, and the Ito isometry for their variance.

**The two facts.** First, an Ito integral $\int_0^t g(s)\,dB_s$ (for a well-behaved integrand $g$) is a **martingale with mean zero**: $\mathbb{E}\!\left[\int_0^t g(s)\,dB_s\right] = 0$. Intuitively, each increment $g(s)\,dB_s$ is a fair bet — $dB_s$ is mean-zero and independent of the past — so the running sum never acquires drift. Second, the **Ito isometry** gives its variance: $\text{Var}\!\left(\int_0^t g(s)\,dB_s\right) = \mathbb{E}\!\left[\int_0^t g(s)^2\,ds\right]$.

![An Ito integral accumulates many mean-zero Brownian increments, drawn as a running waterfall where green bars push the cumulative value positive and red bars push it negative; the expected position stays at zero throughout even as the spread of the running total widens with time, matching the Ito-isometry variance formula.](/imgs/blogs/itos-lemma-quant-interviews-9.png)

**Solve the expectation.** By the first fact, with $g(s) = B_s$, we get $\mathbb{E}[I_t] = \mathbb{E}\!\left[\int_0^t B_s\,dB_s\right] = 0$ immediately. The figure makes this concrete: the integral is a running accumulation of fair bets — green when the cumulative is positive, red when negative — and its expected value stays pinned at zero while the spread fans out with time.

**Solve the closed form.** Reuse the result we already derived: from $d(B^2) = 2B\,dB + dt$, we integrate to get $B_t^2 = \int_0^t 2B_s\,dB_s + t$, hence
$$
\int_0^t B_s\,dB_s = \tfrac{1}{2}\big(B_t^2 - t\big).
$$
Check the expectation against this closed form: $\mathbb{E}\!\left[\tfrac{1}{2}(B_t^2 - t)\right] = \tfrac{1}{2}(t - t) = 0$. Consistent. And the variance? By Ito isometry, $\text{Var}(I_t) = \mathbb{E}\!\left[\int_0^t B_s^2\,ds\right] = \int_0^t \mathbb{E}[B_s^2]\,ds = \int_0^t s\,ds = \tfrac{1}{2}t^2$. **Interview summary: an Ito integral against $dB$ is automatically mean-zero; to get a closed form, find a function whose Ito expansion produces your integrand — here $B^2$ — and rearrange.** Notice the difference from a normal integral $\int_0^t B_s\,ds$: that ordinary (Riemann) integral is *not* mean-zero in the Ito sense and behaves differently — distinguishing the two is a classic trap.

#### Worked example: Problem 4 — The drift of $1/S$ under GBM

**Setup.** "A stock follows GBM with drift $\mu$ and volatility $\sigma$. What process does the inverse price $1/S$ follow, and what is its drift?" This tests whether you can apply Ito to a function with strong curvature and reason about why the drift moves the way it does.

**Solve.** Let $f(S) = S^{-1}$, so $f'(S) = -S^{-2}$ and $f''(S) = 2 S^{-3}$. Using $dS = \mu S\,dt + \sigma S\,dB$ and $(dS)^2 = \sigma^2 S^2\,dt$:
$$
d\!\left(\frac{1}{S}\right) = -\frac{1}{S^2}\,dS + \tfrac{1}{2}\cdot\frac{2}{S^3}\,(dS)^2 = -\frac{1}{S^2}(\mu S\,dt + \sigma S\,dB) + \frac{1}{S^3}(\sigma^2 S^2\,dt).
$$
Simplify each piece: the first gives $-\tfrac{\mu}{S}\,dt - \tfrac{\sigma}{S}\,dB$, and the Ito term gives $+\tfrac{\sigma^2}{S}\,dt$. Collecting:
$$
d\!\left(\frac{1}{S}\right) = \frac{1}{S}\Big[(-\mu + \sigma^2)\,dt - \sigma\,dB\Big] = (\,\sigma^2 - \mu\,)\,\frac{1}{S}\,dt - \sigma\,\frac{1}{S}\,dB.
$$
So $Y = 1/S$ is itself a geometric Brownian motion, with drift $\sigma^2 - \mu$ and volatility $-\sigma$ (the sign just flips the Brownian driver). **The instructive part: the drift of $1/S$ is *not* simply $-\mu$ — the convexity of $1/S$ adds a positive $+\sigma^2$ correction.** This is why, for instance, the expected value of an inverse exchange rate is not the inverse of the expected exchange rate; the wedge is a pure Ito effect, and forgetting it is the source of the famous "Siegel's paradox" in FX. An interviewer who asks this is checking that you do not blindly negate the drift.

#### Worked example: Problem 5 — Build the Black-Scholes PDE on the spot

**Setup.** "Derive the Black-Scholes PDE." This is the capstone, asked at every derivatives desk. It ties together everything: Ito on a time-dependent function, the multiplication table, and the no-arbitrage idea that a perfectly hedged portfolio must earn the risk-free rate.

![On a real options desk, Ito's lemma builds the Black-Scholes PDE: start from the option value V of t and S, apply Ito to get its drift and dS terms, delta-hedge by holding minus V_S shares to cancel the random dB risk, conclude the resulting portfolio is riskless and must earn the rate r, and read off the Black-Scholes partial differential equation.](/imgs/blogs/itos-lemma-quant-interviews-12.png)

**Solve.** Let $V(t, S)$ be the value of an option (or any derivative) on a stock $S$ following GBM. Apply two-variable Ito:
$$
dV = V_t\,dt + V_S\,dS + \tfrac{1}{2} V_{SS}\,(dS)^2 = \left(V_t + \mu S\,V_S + \tfrac{1}{2}\sigma^2 S^2 V_{SS}\right)dt + \sigma S\,V_S\,dB,
$$
using $(dS)^2 = \sigma^2 S^2\,dt$. The randomness lives entirely in the $\sigma S\,V_S\,dB$ term.

Now **delta-hedge**: form a portfolio that is long one option and short $V_S$ shares of the stock (this $V_S$ is the option's *delta*, its sensitivity to the stock). The portfolio value is $\Pi = V - V_S\,S$, and over $dt$ its change is
$$
d\Pi = dV - V_S\,dS = \left(V_t + \tfrac{1}{2}\sigma^2 S^2 V_{SS}\right)dt.
$$
The $\mu S V_S\,dt$ and the $\sigma S V_S\,dB$ terms both cancelled against the $-V_S\,dS$. The portfolio change has **no $dB$ term** — it is riskless over the instant. By no-arbitrage, a riskless portfolio must earn the risk-free rate $r$: $d\Pi = r\,\Pi\,dt = r(V - V_S\,S)\,dt$. Equate the two expressions for $d\Pi$:
$$
V_t + \tfrac{1}{2}\sigma^2 S^2 V_{SS} = r V - r S\,V_S,
$$
and rearrange to the celebrated **Black-Scholes PDE**:
$$
\boxed{\;V_t + r S\,V_S + \tfrac{1}{2}\sigma^2 S^2 V_{SS} - r V = 0\;}
$$

**Read off the structure.** Notice that $\mu$ — the stock's real-world drift — *vanished*. The hedged portfolio does not care which way the stock is expected to drift, only how much it wiggles ($\sigma$). That is the deep content of risk-neutral pricing, and it falls out of Ito plus hedging. The $\tfrac{1}{2}\sigma^2 S^2 V_{SS}$ term — the gamma term — is the Ito correction, and it is what links an option's curvature ($V_{SS}$, its *gamma*) to its time decay ($V_t$, its *theta*). **Interview summary: Ito gives you $dV$; delta-hedging cancels the $dB$; no-arbitrage pins the drift to $r$; the surviving second-order term is the Ito correction that makes the whole formula non-trivial.** For the full pricing story this PDE feeds into, see the deep dive on [Black-Scholes](/blog/trading/quantitative-finance/black-scholes).

#### Worked example: Problem 6 — The general power $S^n$ and its drift

**Setup.** "For a GBM stock, what process does $S^n$ follow, for a general power $n$?" This generalizes Problems 4 (where $n = -1$) and the $\log$ case (the limit $n \to 0$), and it is a favourite because it tests whether you can carry a symbolic exponent through the machine without panicking.

**Solve.** Let $f(S) = S^n$, so $f'(S) = n S^{n-1}$ and $f''(S) = n(n-1) S^{n-2}$. With $dS = \mu S\,dt + \sigma S\,dB$ and $(dS)^2 = \sigma^2 S^2\,dt$:
$$
d(S^n) = n S^{n-1}\,dS + \tfrac{1}{2} n(n-1) S^{n-2}\,(dS)^2.
$$
Substitute and simplify each term. The first gives $n S^{n-1}(\mu S\,dt + \sigma S\,dB) = n S^n(\mu\,dt + \sigma\,dB)$. The Ito term gives $\tfrac{1}{2} n(n-1) S^{n-2}\,\sigma^2 S^2\,dt = \tfrac{1}{2} n(n-1)\sigma^2 S^n\,dt$. Collecting:
$$
\boxed{\;d(S^n) = S^n\left[\Big(n\mu + \tfrac{1}{2} n(n-1)\sigma^2\Big)dt + n\sigma\,dB\right]\;}
$$

So $S^n$ is *also* a geometric Brownian motion, with drift $n\mu + \tfrac{1}{2} n(n-1)\sigma^2$ and volatility $n\sigma$. **Read off the consequences.** Set $n = 2$: the drift of $S^2$ is $2\mu + \sigma^2$, so $\mathbb{E}[S_t^2] = S_0^2\,e^{(2\mu + \sigma^2)t}$ — exactly the second moment of a lognormal. Set $n = -1$: the drift is $-\mu + \sigma^2 = \sigma^2 - \mu$, matching Problem 4. The convexity adjustment $\tfrac{1}{2} n(n-1)\sigma^2$ is positive whenever $n > 1$ or $n < 0$ (convex regions) and negative for $0 < n < 1$ (concave, like $\sqrt{S}$ at $n = \tfrac{1}{2}$). **Interview summary: one formula, parameterized by $n$, reproduces every power-of-the-stock question — and the sign of the convexity term flips exactly where $f$ changes between convex and concave.**

#### Worked example: Problem 7 — Two correlated assets and the product rule

**Setup.** "You hold a portfolio whose value is the product $P = XY$ of two stocks, each following GBM with correlation $\rho$ between their Brownian drivers. What is $dP$, and what is the growth rate of $\mathbb{E}[P]$?" This is the one place the *product rule* and the cross-variation term genuinely matter, and interviewers use it to see if you remember the $dX\,dY$ piece.

**Solve.** Let $dX = \mu_X X\,dt + \sigma_X X\,dB^X$ and $dY = \mu_Y Y\,dt + \sigma_Y Y\,dB^Y$, with $dB^X dB^Y = \rho\,dt$. The Ito product rule says $d(XY) = X\,dY + Y\,dX + dX\,dY$. Compute the cross term first:
$$
dX\,dY = (\sigma_X X\,dB^X)(\sigma_Y Y\,dB^Y) = \sigma_X \sigma_Y X Y\,(dB^X dB^Y) = \rho\,\sigma_X \sigma_Y\,XY\,dt,
$$
where every product with a $dt$ leg vanished by the multiplication table. Assemble:
$$
dP = X(\mu_Y Y\,dt + \sigma_Y Y\,dB^Y) + Y(\mu_X X\,dt + \sigma_X X\,dB^X) + \rho\,\sigma_X \sigma_Y\,XY\,dt.
$$
Factor out $P = XY$:
$$
\boxed{\;dP = P\Big[\big(\mu_X + \mu_Y + \rho\,\sigma_X \sigma_Y\big)dt + \sigma_X\,dB^X + \sigma_Y\,dB^Y\Big]\;}
$$

**Read it off.** The product is again a GBM-type process, but its drift carries the extra cross-variation term $\rho\,\sigma_X \sigma_Y$. So $\mathbb{E}[P_t] = P_0\,e^{(\mu_X + \mu_Y + \rho\,\sigma_X\sigma_Y)t}$. If the assets are positively correlated ($\rho > 0$), the product grows *faster* than the naive sum of drifts would suggest; if negatively correlated, slower. **Interview summary: whenever two genuinely random legs multiply, the product rule's $dX\,dY$ term injects a $\rho\,\sigma_X\sigma_Y$ correction into the drift — forget it and your expected product is wrong.** This is the mechanism behind quanto adjustments, basket-option pricing, and the correlation risk that desks hedge; it is the two-asset cousin of the single-asset $-\tfrac{1}{2}\sigma^2$ correction.

## Common misconceptions

These are the errors that get candidates rejected, and the beliefs that, once corrected, separate a strong answer from a stumbling one.

**"You can use the ordinary chain rule; the extra term is a technicality."** No — the $\tfrac{1}{2} f''$ term is often the *answer*. In the $\log S$ problem it produces the entire $-\tfrac{1}{2}\sigma^2$ drift correction, an 8%-per-year effect on a 40%-vol stock. Dropping it is not a small approximation error; it changes the qualitative behaviour of the process. The Ito term is the difference between a correct model and a wrong one.

**"$dB$ is small, so $(dB)^2$ is negligible, just like $(dt)^2$."** This is the deepest misconception and the root of all the others. The point of the entire foundations section is that $dB \sim \sqrt{dt}$, so $(dB)^2 \sim dt$ is the *same order* as the drift term, not negligible. Brownian motion is jagged precisely because its increments are $\sqrt{dt}$ rather than $dt$ — that is what "infinite variation, finite quadratic variation" means.

**"$dB \cdot dt$ might matter too."** It does not. The cross term $dB\,dt$ is order $\sqrt{dt}\cdot dt = (dt)^{3/2}$, which vanishes faster than $dt$. Only $dB \cdot dB = dt$ survives the multiplication table. A candidate who keeps $dB\,dt$ terms is over-applying the lesson and signalling they memorized rather than understood.

**"Ito's lemma always adds a $+\tfrac{1}{2}\sigma^2$ drift."** The sign and presence depend on the curvature $f''$. For $\log S$ ($f'' < 0$) the correction is *negative*. For $e^{B}$ or $1/S$ ($f'' > 0$) it is *positive*. For a *linear* function or any process with $f_{XX} = 0$ (like the OU integrating-factor step), there is *no* correction at all. Always compute $f''$; never assume the sign.

**"The expected stock price grows at the corrected rate $\mu - \tfrac{1}{2}\sigma^2$."** No — the *median* (log) path grows at $\mu - \tfrac{1}{2}\sigma^2$, but the *mean* price $\mathbb{E}[S_t]$ grows at the full $\mu$, because $\mathbb{E}[S_t] = S_0 e^{\mu t}$. The lognormal distribution's right skew pulls the average above the median. Confusing mean and median here is a classic interview slip — and a real source of money lost when people project median paths as if they were expectations, or vice versa.

**"Martingale just means 'random' or 'fair game' loosely."** A martingale has a precise definition: zero drift, so $\mathbb{E}[M_T \mid \mathcal{F}_t] = M_t$. The clean computational test is "apply Ito and check the $dt$ term is zero." If you can compute the drift, you can prove or disprove the martingale property mechanically — no hand-waving about fairness required.

**"An Ito integral $\int g\,dB$ behaves like an ordinary integral $\int g\,ds$."** They are different objects. The ordinary (Riemann) integral $\int_0^t g(s)\,ds$ accumulates a smooth, predictable quantity; the Ito integral $\int_0^t g(s)\,dB_s$ accumulates random kicks and is a martingale with mean zero. A subtle consequence: the Ito integral is defined using the *left endpoint* of each sub-interval (the integrand is evaluated *before* the random kick it multiplies), which is what makes it mean-zero and non-anticipating. Evaluate at the midpoint instead and you get the *Stratonovich* integral, which obeys the ordinary chain rule but is not a martingale. Finance uses Ito precisely because the left-point convention matches "you must choose your hedge before the market moves." Confusing the two integrals — or assuming $\int B\,dB = \tfrac{1}{2}B_t^2$ as in ordinary calculus, when the Ito answer is $\tfrac{1}{2}(B_t^2 - t)$ — is a classic trap.

**"If $f$ has no second derivative term in the SDE, Ito does nothing."** It is true that when $f_{XX} = 0$ there is no correction (as in the OU integrating-factor step). But beware: $f_{XX} = 0$ only for functions linear in the state. The moment $f$ is genuinely curved — and almost every interesting payoff is — the correction is back. Do not let one $f_{XX} = 0$ example lull you into skipping the check.

## How it shows up on a real desk

Ito's lemma is not a museum piece. It is the daily working tool behind several desk activities, and interviewers ask about it because they want to know you will reach for it correctly when real money is on the line.

**Deriving and re-deriving the pricing PDE.** Every derivative a desk prices — vanilla options, barriers, autocallables, variance swaps — has a value function $V$ whose dynamics come from Ito's lemma applied to the underlying's SDE. When a desk moves from a constant-volatility model to a [local or stochastic volatility model](/blog/trading/quantitative-finance/volatility-surface), the *mechanics* are the same: write the SDE, apply Ito to $V$, hedge away the Brownian terms, and read off the PDE. A quant who cannot do this fluently cannot extend a model. The gamma term $\tfrac{1}{2}\sigma^2 S^2 V_{SS}$ that Ito produces is the single most-watched risk on an options desk: it is why a delta-hedged book still makes or loses money as the stock moves, and why traders talk about being "long gamma" (profiting from realized volatility) or "short gamma" (bleeding to it).

**Why log-returns carry the $-\tfrac{1}{2}\sigma^2$ term, in risk reports.** Risk and performance systems quote returns in log space because logs add cleanly over time. But the moment you do that, the $-\tfrac{1}{2}\sigma^2$ correction is baked into every number. When a portfolio's *expected* arithmetic return is $\mu$ but its *median* compounded outcome is lower, that wedge is the variance drag from Ito. Desks that size leverage or project drawdowns using arithmetic averages — ignoring the drag — systematically overstate the wealth a typical path will reach. The 2% versus 10% gap in our \$100-stock example is the toy version of a mistake that has blown up real leveraged products, especially leveraged and inverse ETFs, whose daily-rebalancing volatility drag is precisely $-\tfrac{1}{2}\sigma^2$ compounding day after day. A 2x leveraged ETF on a 40%-vol index does not simply double the index's return; the doubled volatility quadruples the variance and the drag becomes $\tfrac{1}{2}(0.80)^2 = 32\%$ per year of pure decay, which is why these products are unsuitable as long-term holdings.

**Change of measure for risk-neutral pricing.** The exponential martingale $e^{B_t - t/2}$ from Problem 1 generalizes to the *Girsanov* change of measure, the formal machinery that lets a desk price under the risk-neutral measure — the probability world in which all assets drift at the risk-free rate. The whole apparatus rests on Ito: you verify the relevant exponential is a martingale (drift zero), and that verification is an Ito computation. When a pricing library "switches to the risk-neutral measure," it is invoking a theorem whose proof is the calculation we did by hand.

**Calibrating mean-reverting models.** Interest-rate desks, volatility-arbitrage desks, and stat-arb desks all fit Ornstein-Uhlenbeck-type processes (Problem 2) to spreads, rates, and volatilities. The closed-form solution we derived — with its stationary mean $\mu$ and stationary variance $\sigma^2/(2\theta)$ — is what lets them estimate the reversion speed $\theta$ and the equilibrium level $\mu$ from historical data, and then size trades that bet on reversion. The connection between continuous-time SDEs and the discrete random walks you see in [Markov chains and hitting times](/blog/trading/quantitative-finance/markov-chains-hitting-times-quant-interviews) is direct: the OU process is the continuous-time limit of a mean-reverting random walk, and Ito is the bridge.

**Sanity-checking Monte Carlo simulators.** When a desk simulates GBM paths, it must discretize the SDE. The naive Euler scheme $S_{t+\Delta} = S_t(1 + \mu\Delta + \sigma\sqrt{\Delta}\,Z)$ and the exact log scheme $S_{t+\Delta} = S_t \exp[(\mu - \tfrac{1}{2}\sigma^2)\Delta + \sigma\sqrt{\Delta}\,Z]$ differ by exactly the Ito correction. Using the wrong one introduces a systematic bias in the simulated drift — small per step, but it compounds. Knowing *why* the log scheme carries the $-\tfrac{1}{2}\sigma^2$ term is the difference between a simulator that prices correctly and one that quietly drifts off. This is the kind of detail a Two Sigma or Citadel interviewer probes to see whether you understand the model or merely use it.

## When this matters and where to go next

If you are preparing for a quant researcher or derivatives role, Ito's lemma is non-negotiable: it appears in nearly every stochastic-calculus interview, and it underpins the pricing and risk math you will use on the job. The good news is that, despite its fearsome reputation, the entire toolkit reduces to a handful of moves you can practice until they are automatic.

The drill that makes it stick: take a process $dX = a\,dt + b\,dB$ and a function $f$, write down $f'$ and $f''$, compute $(dX)^2 = b^2\,dt$ via the multiplication table, and assemble $df = [a f' + \tfrac{1}{2} b^2 f''] dt + b f'\,dB$. Do it for $f = X^2$, $X^3$, $e^X$, $\log X$, $1/X$, and $\sqrt{X}$ until the $\tfrac{1}{2} f''$ correction is reflexive. Then layer in time-dependence and the product rule. Five functions and three multiplication-table entries cover the overwhelming majority of what gets asked.

From here, the natural next steps build directly on this foundation. The [Black-Scholes deep dive](/blog/trading/quantitative-finance/black-scholes) takes the PDE we derived in Problem 5 and solves it for the option price. The piece on [derivatives pricing](/blog/trading/quantitative-finance/derivatives-pricing) develops the risk-neutral measure that the exponential martingale of Problem 1 seeds. The [short-rate models](/blog/trading/quantitative-finance/short-rate-models-vasicek-hull-white) article applies the Ornstein-Uhlenbeck solution from Problem 2 to interest rates. And if you want to shore up the probability foundations that Brownian motion rests on, the [distributions cheat sheet](/blog/trading/quantitative-finance/distributions-cheat-sheet-quant-interviews) covers the normal and lognormal distributions that appear throughout.

This is educational material, not investment advice. But the next time an interviewer asks for the process of $\log S$, you will not reach for the wrong chain rule — and you will be able to explain, in dollars, exactly what that extra $-\tfrac{1}{2}\sigma^2$ term is doing.

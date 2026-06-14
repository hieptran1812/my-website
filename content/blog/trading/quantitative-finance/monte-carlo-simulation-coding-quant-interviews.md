---
title: "Monte Carlo and simulation coding for quant interviews"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch, interview-focused deep dive on Monte Carlo simulation: the law of large numbers, the one-over-root-N standard error, sampling random variables by inverse-CDF and Box-Muller, estimating probabilities and expectations, antithetic and control-variate variance reduction, simulating geometric Brownian motion to price a European option against Black-Scholes, confidence intervals, and using a simulation to verify an analytic answer, with a full set of fully-solved interview problems."
tags:
  [
    "monte-carlo",
    "simulation",
    "quant-interviews",
    "law-of-large-numbers",
    "standard-error",
    "variance-reduction",
    "geometric-brownian-motion",
    "option-pricing",
    "black-scholes",
    "confidence-intervals",
    "inverse-cdf",
    "box-muller",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — A huge fraction of quant interview questions can be answered by *simulating* the experiment a few million times and averaging; doing it correctly, and using it to **check** an analytic answer, is a graded superpower.
>
> - Monte Carlo turns a probability or expectation into the average of many random samples. The law of large numbers says that average converges to the truth; the standard error tells you how close you are.
> - The error shrinks like $1/\sqrt{N}$ — slowly. To get one more correct digit you need roughly **100 times** as many samples. That single fact governs every Monte Carlo answer you will ever give.
> - You can sample any distribution from a plain uniform: read it across a cumulative distribution (inverse-CDF) for one-dimensional draws, or use Box-Muller for normals.
> - We price a European call on a \$100 stock by simulating 100,000 price paths and land at **\$10.47 ± \$0.05**, bracketing the exact Black-Scholes value of **\$10.45** — the simulation *verifies* the formula.
> - Variance-reduction tricks (antithetic draws, control variates) cut the error for free: a control variate here shrinks the standard error from **\$0.15 to \$0.057**, a 2.6× tightening at the same sample count.
> - The interview move that scores points: when you give an analytic answer, say "and I'd confirm it with a quick simulation" — then describe the five-line loop that does it.

You are in a quant interview. The interviewer slides a problem across the table: "Two players take turns flipping a fair coin. The first to flip heads wins. What is the probability the first player wins?" You can grind through the infinite geometric series — or you can say: "I'll set up the series, and I'd sanity-check it by simulating a million games." The second sentence is the one that gets you hired. It signals that you think in terms of *experiments you can run*, that you do not trust an algebra answer you cannot verify, and that you can write the loop.

That is what this post is about. Monte Carlo simulation is the single most reusable tool in the quant interview, because it answers two completely different kinds of question with the same machine: it *computes* an answer you cannot get in closed form, and it *checks* an answer you can. The diagram below is the mental model for the whole post: every probability or expectation question forks into "solve it exactly" or "simulate it", and the strongest answer almost always does *both* and confirms they agree.

![A decision flow showing that a probability or expectation question splits into solving exactly when a closed form is easy or simulating by drawing N samples, with both paths feeding a cross-check step.](/imgs/blogs/monte-carlo-simulation-coding-quant-interviews-1.png)

We will build the whole thing from zero. No measure theory, no assumed finance background. By the end you will be able to estimate $\pi$ and report how wrong you might be, sample any distribution from a uniform, price an option by simulation and compare it to the textbook formula, cut your error in half for free, and — the part interviewers actually reward — use a simulation to confirm a hand-derived probability. Every section has a worked example with real numbers, and the post closes with a set of fully-solved interview problems.

## Foundations: what Monte Carlo actually is

Let us define the terms before we use them, because the whole method rests on two ideas that sound fancy and are not.

A *random variable* is just a number whose value depends on chance — the result of a coin flip (0 or 1), the roll of a die (1 through 6), tomorrow's stock price. An *expectation*, written $E[X]$, is the long-run average value of that random number if you could repeat the experiment forever. The expectation of a fair die is $E[X] = (1+2+3+4+5+6)/6 = 3.5$ — you will never roll a 3.5, but the average of millions of rolls homes in on it. A *probability* $P(A)$ is the long-run fraction of trials in which event $A$ happens; it is itself an expectation — the expectation of the *indicator*, a variable that is 1 when $A$ happens and 0 otherwise.

**Monte Carlo simulation** is the method of estimating an expectation (and therefore any probability) by drawing many random samples and averaging. That is the entire idea. If you want $E[X]$, draw $N$ independent copies $X_1, X_2, \dots, X_N$ and compute the *sample mean*:

$$\hat{\mu}_N = \frac{1}{N} \sum_{i=1}^{N} X_i$$

The hat on $\hat{\mu}$ means "estimate of"; the subscript $N$ reminds you it depends on how many samples you took. The name comes from the Monte Carlo casino — the method was christened at Los Alamos in the 1940s, where physicists simulated neutron diffusion by drawing random numbers, and named it after the gambling town because the whole thing is built on games of chance.

### The law of large numbers: why averaging works

The reason this is legitimate and not wishful thinking is the **law of large numbers** (LLN). It says that as the number of samples $N$ grows, the sample mean $\hat{\mu}_N$ converges to the true expectation $\mu = E[X]$. Average enough fair-die rolls and you *will* approach 3.5. Average enough coin flips and the fraction of heads *will* approach one half. The LLN is the guarantee that makes simulation a measurement rather than a guess.

But the LLN as stated is silent on the question every interviewer cares about: *how many samples is enough?* It promises convergence eventually; it does not say whether 1,000 draws or 10 million are needed to trust the third decimal. For that we need the second pillar.

### The standard error: how wrong might you be

The sample mean is itself random — run the simulation again with a different stream of random numbers and you get a slightly different answer. The *standard error of the mean* (SE) measures how much it jitters. If a single draw $X$ has standard deviation $\sigma$ (the typical distance of one sample from the mean), then the sample mean of $N$ independent draws has standard error

$$\text{SE} = \frac{\sigma}{\sqrt{N}}$$

This is the most important formula in the entire post, so let us unpack it slowly. The numerator $\sigma$ is fixed by the problem — it is how spread out a single sample is. The denominator $\sqrt{N}$ is what *you* control by choosing how many samples to take. Because the error falls with $\sqrt{N}$ and not $N$, the returns diminish fast. To **halve** the error you must **quadruple** the samples. To shave the error to a tenth — one more correct digit — you need **a hundred times** as many samples. A simulation that is good to two digits at 10,000 samples needs a million samples for three digits and a hundred million for four.

In practice you do not know $\sigma$ in advance, so you estimate it from the same samples using the *sample standard deviation* $s$, and report $\text{SE} = s/\sqrt{N}$. That gives you the second number that turns a Monte Carlo guess into a Monte Carlo *measurement*: the estimate, and a bar around it.

## Convergence and standard error in pictures

Before the worked examples, let us see the two governing facts. First, an estimate converging. Here is the classic toy: estimate $\pi$ by throwing darts at a square and counting how many land inside an inscribed quarter-circle (more on the mechanics in a moment). The estimate is wild at small $N$ and settles onto the true value as $N$ grows.

![A line chart with number of samples N on a log-scale horizontal axis from ten to one hundred thousand and the estimate of pi on the vertical axis, showing the estimate wobbling widely at small N and converging onto a solid horizontal line at pi as N grows.](/imgs/blogs/monte-carlo-simulation-coding-quant-interviews-2.png)

The wobble at $N=10$ is enormous — the estimate can be 3.0 or 3.4 on any given run. By $N=100{,}000$ it is pinned within a few thousandths of 3.14159. Nothing about the method changed; only the sample count grew.

Now the rate. Plot the standard error against $N$ on log-log axes and you get a straight line with slope exactly $-\tfrac{1}{2}$, the signature of $1/\sqrt{N}$ decay.

![A log-log chart with number of samples N on the horizontal axis from ten to one million and standard error on the vertical axis, showing a straight descending line of slope minus one half, annotated that one hundred times more samples buys ten times less error.](/imgs/blogs/monte-carlo-simulation-coding-quant-interviews-3.png)

That straight line is the tax every Monte Carlo answer pays. It is why simulation is fantastic for two or three digits of accuracy and miserable for six, and why so much of the craft (the variance-reduction section below) is about lowering the *line* — shrinking $\sigma$ — rather than pushing $N$ ever higher.

#### Worked example: estimate $\pi$ and report the standard error

Here is the dart-throwing estimator in full, because it is the canonical "code a Monte Carlo" interview question. Draw a point uniformly in the unit square $[0,1] \times [0,1]$. The quarter-circle of radius 1 has area $\pi/4 \approx 0.785$, and the square has area 1, so the probability a random point lands inside the quarter-circle is exactly $\pi/4$. Estimate that probability by the fraction of darts that land inside, then multiply by 4.

```python
import numpy as np

def estimate_pi(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    y = rng.random(n)
    inside = (x**2 + y**2) <= 1.0
    p_hat = inside.mean()          # estimate of pi/4
    pi_hat = 4 * p_hat
    # standard error: indicator has variance p(1-p); SE of 4*p_hat is 4*sqrt(p(1-p)/n)
    se = 4 * np.sqrt(p_hat * (1 - p_hat) / n)
    return pi_hat, se

print(estimate_pi(100_000))   # -> approximately (3.149, 0.0052)
```

With $N = 100{,}000$ darts this returns an estimate of about **3.149** with a standard error of about **0.0052**. The true value is 3.14159, so the estimate sits roughly $0.0074$ — about 1.4 standard errors — above the truth, which is exactly the kind of miss the standard error predicts. The interview-grade answer is not "$\pi \approx 3.149$"; it is "$\pi \approx 3.149 \pm 0.005$ at one standard error, so I am confident in the first two digits and shaky on the third." Reporting the error bar is the whole point.

The deeper lesson: the variance of an indicator (a 0/1 variable with probability $p$) is $p(1-p)$. Here $p = \pi/4 \approx 0.785$, so $p(1-p) \approx 0.169$ and a single dart has standard deviation about $0.41$. After scaling by 4 and dividing by $\sqrt{100{,}000} \approx 316$, you get the $0.0052$ standard error directly. *The standard error of a probability estimate is computable before you even run the simulation* — a fact worth saying out loud in the room.

## Simulating random variables from a uniform

Every Monte Carlo simulation starts from one primitive: a *uniform random variable* on $(0,1)$, a number equally likely to be anywhere between 0 and 1. Your language gives you this for free (`numpy`'s `rng.random()`, the hardware entropy behind it). The art is turning that uniform into whatever distribution your problem actually needs — an exponential waiting time, a normal shock, a die roll. There are two workhorses.

### The inverse-CDF (inverse transform) method

Every random variable has a *cumulative distribution function* (CDF), written $F(x) = P(X \le x)$ — the probability the variable comes out at or below $x$. The CDF starts at 0 (nothing is below $-\infty$), rises to 1 (everything is below $+\infty$), and never decreases. The **inverse-CDF method** exploits a beautiful fact: if $U$ is uniform on $(0,1)$, then $X = F^{-1}(U)$ — the value you get by reading $U$ across the CDF and back down to the $x$-axis — has exactly the distribution $F$.

![A chart with the sampled value x on the horizontal axis and a uniform draw u between zero and one on the vertical axis, showing a rising cumulative distribution curve; a dashed arrow runs from u equals zero point seven on the vertical axis across to the curve and then down to x equals one point two on the horizontal axis.](/imgs/blogs/monte-carlo-simulation-coding-quant-interviews-4.png)

The picture *is* the method. Pick a uniform height $u$ on the vertical axis, move right until you hit the curve, then drop straight down: where you land on the horizontal axis is your sample. Heights are uniform, but because the curve is steep where the distribution is dense and flat where it is sparse, the landing points cluster exactly the way the target distribution says they should. Steep CDF means many uniforms map into a narrow band of $x$ — high density; flat CDF means few map there — low density.

#### Worked example: sample an exponential waiting time

The *exponential distribution* models waiting times — how long until the next trade, the next default, the next phone call. Its CDF is $F(x) = 1 - e^{-\lambda x}$ for $x \ge 0$, where $\lambda$ (lambda) is the rate. Invert it: set $u = 1 - e^{-\lambda x}$ and solve for $x$, giving $x = -\frac{1}{\lambda}\ln(1-u)$. So to sample an exponential with rate $\lambda = 1$, draw a uniform $u$ and compute $-\ln(1-u)$.

```python
import numpy as np

rng = np.random.default_rng(0)
u = rng.random(1_000_000)
x = -np.log(1 - u)          # exponential(rate=1) by inverse-CDF
print(x.mean(), x.var())    # -> approximately (1.000, 1.000)
```

With $u = 0.70$ specifically, the sample is $x = -\ln(0.30) = 1.204$ — exactly the lookup drawn in the figure above. The mean of a rate-1 exponential is 1, and the simulation returns $0.9999$ over a million draws. The intuition to carry away: *the inverse-CDF method is a change of variables that bends a flat uniform into any shape you can write a CDF for.* It is the most general one-dimensional sampler there is, and it is the right answer whenever the interviewer asks "how would you generate draws from this distribution?"

### Box-Muller for normal draws

The normal (Gaussian, bell-curve) distribution is the one finance lives on — stock-return shocks, measurement noise, the central limit theorem's universal output. Its CDF has no clean inverse, so the inverse-CDF method is awkward. The **Box-Muller transform** sidesteps it with a clever geometric trick: it turns *two* independent uniforms into *two* independent standard normals (mean 0, standard deviation 1) at once.

![A five-stage pipeline showing two uniform draws on zero to one feeding a radius equal to the square root of minus two times the log of the first uniform, an angle equal to two pi times the second uniform, then two standard normal coordinates from radius times cosine and sine, used as shocks for price paths.](/imgs/blogs/monte-carlo-simulation-coding-quant-interviews-5.png)

The mechanics: draw $U_1, U_2$ uniform on $(0,1)$. Form a radius $R = \sqrt{-2 \ln U_1}$ and an angle $\theta = 2\pi U_2$. Then $Z_1 = R\cos\theta$ and $Z_2 = R\sin\theta$ are two independent $N(0,1)$ draws. The reason it works is that a 2-D standard normal is *rotationally symmetric* — its density depends only on distance from the origin — so a uniform random angle plus the right radial distribution reconstructs it exactly. In practice you call `rng.standard_normal(n)`, which uses a faster variant internally, but Box-Muller is the one to be able to derive at the whiteboard.

```python
import numpy as np

rng = np.random.default_rng(0)
u1, u2 = rng.random(500_000), rng.random(500_000)
r = np.sqrt(-2 * np.log(u1))
theta = 2 * np.pi * u2
z1 = r * np.cos(theta)
z2 = r * np.sin(theta)
z = np.concatenate([z1, z2])
print(z.mean(), z.std())   # -> approximately (0.000, 1.000)
```

The one-sentence intuition: *Box-Muller manufactures normals out of uniforms by sampling a random point in polar coordinates.* For a die roll or any discrete distribution, by the way, you do not need either machine — `1 + floor(6 * u)` turns a uniform into a fair die directly, which is itself a one-line inverse-CDF.

## Estimating probabilities and expectations by simulation

Now we assemble the pieces into the loop that answers interview questions. Whether you want a probability or an expectation, the structure is identical: draw samples from the model, *score* each one (an indicator 0/1 for a probability, a payoff or value for an expectation), average the scores, and report the standard error.

![A five-stage pipeline: draw N samples from the model, score each as an indicator or payoff, average them into mu-hat, compute the standard error s over root N, and report the estimate plus or minus one point nine six times the standard error.](/imgs/blogs/monte-carlo-simulation-coding-quant-interviews-6.png)

That loop never changes. The only thing that varies between problems is the model you draw from and the scoring function. Master the skeleton and you can answer a wide class of questions by filling in two blanks.

#### Worked example: a die and two-dice expectation, matched to the analytic answer

Start with the simplest possible check, the one that proves to the interviewer your loop is correct before you point it at anything hard. The expected value of one fair six-sided die is $E[X] = 3.5$. The expected value of the *sum* of two independent dice is $E[X+Y] = E[X] + E[Y] = 7$ by linearity of expectation. Simulate both:

```python
import numpy as np

rng = np.random.default_rng(555)
n = 1_000_000
d1 = rng.integers(1, 7, n)          # fair die, 1..6
d2 = rng.integers(1, 7, n)
print(d1.mean())                    # -> approximately 3.499  (analytic 3.5)
print((d1 + d2).mean())             # -> approximately 6.997  (analytic 7)
```

The single die averages **3.499** against the exact 3.5; the two-dice sum averages **6.997** against the exact 7. Both land within a hundredth, exactly as the standard error predicts (a single die has $\sigma \approx 1.71$, so SE $\approx 1.71/\sqrt{10^6} = 0.0017$). The intuition: *if your simulation cannot reproduce an answer you already know, do not trust it on an answer you do not.* Validating the loop against a known case is the first thing a good quant does, and the first thing an interviewer looks for.

#### Worked example: estimate a probability of ruin

The *gambler's ruin* is a staple. You start with \$5. You repeatedly bet \$1 on a coin flip — win \$1 with probability $p$, lose \$1 with probability $1-p$. You stop when you hit \$10 (you win) or \$0 (you are ruined). What is the probability of ruin?

For a *fair* game ($p = 0.5$) there is a clean closed form: the probability of being ruined before reaching the target $N$ from a start of $i$ is $(N-i)/N$. From \$5 toward \$10 that is $(10-5)/10 = 0.5$ — perfectly symmetric, as you would hope for a fair coin. For an *unfair* game the formula is $P(\text{ruin}) = \frac{r^{N} - r^{i}}{r^{N} - 1}$ where $r = (1-p)/p$. With a slightly hostile coin $p = 0.48$ (so $r = 0.52/0.48 \approx 1.083$), that evaluates to **0.5987** — a tiny edge against you balloons into a 60% chance of ruin. Now simulate and confirm:

```python
import numpy as np

def prob_ruin(p, start=5, target=10, trials=200_000, seed=13):
    rng = np.random.default_rng(seed)
    ruined = 0
    for _ in range(trials):
        x = start
        while 0 < x < target:
            x += 1 if rng.random() < p else -1
        if x == 0:
            ruined += 1
    return ruined / trials

print(prob_ruin(0.50))   # -> approximately 0.499  (analytic 0.500)
print(prob_ruin(0.48))   # -> approximately 0.599  (analytic 0.5987)
```

The simulation returns **0.499** for the fair game and **0.599** for the hostile one, both inside a fraction of a percent of the analytic values. The intuition to keep: *a small, persistent disadvantage compounds into a large probability of ruin over many bets* — the same mechanism that makes the house edge so devastating over a night at the casino, and the reason position-sizing and bet-fraction discipline (the [Kelly criterion](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews)) matters so much on a real book.

## Simulating price paths: geometric Brownian motion

Now we earn the "finance" in quantitative finance. To price an option by simulation we need to simulate the *stock price*, and the standard model for that is **geometric Brownian motion** (GBM). The word geometric means the price moves in *multiplicative* steps (it grows by random percentages, never goes negative, and a 1% move costs the same whether the stock is \$10 or \$1,000); Brownian motion is the random walk underneath.

The GBM model says that over a small time step $\Delta t$, the price is multiplied by a random factor:

$$S_{t+\Delta t} = S_t \cdot \exp\!\left[\left(\mu - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\; Z\right]$$

where $S_t$ is today's price, $\mu$ (mu) is the *drift* (the average growth rate per year), $\sigma$ (sigma) is the *volatility* (how violently the price shakes, per year), $Z$ is a fresh standard normal draw each step, and $\exp$ is the exponential function. The $-\tfrac{1}{2}\sigma^2$ term is a subtle correction (it appears because of the curvature of the exponential — a consequence of [Ito's lemma](/blog/trading/quantitative-finance/itos-lemma-quant-interviews)) that keeps the *average* growth rate honest at $\mu$. Each random $Z$ comes from Box-Muller; each step multiplies the running price.

![A line chart with time in years on the horizontal axis from zero to one and stock price in dollars on the vertical axis, showing eight simulated price paths all starting at one hundred dollars and fanning out between roughly eighty and one hundred eighty dollars over the year.](/imgs/blogs/monte-carlo-simulation-coding-quant-interviews-7.png)

Eight paths, same starting price of \$100, same drift of 8% per year and volatility of 25% per year — and they fan out wildly, some ending near \$80 and some near \$180. That fan is the whole point: the *single number* you want (an option's value) is an average over this entire cloud of possible futures, and Monte Carlo computes it by drawing the cloud and averaging.

#### Worked example: simulate one year of a stock and check the terminal mean

Before pricing anything, simulate the cloud and verify it behaves. For risk-neutral pricing (the next section) we set the drift equal to the risk-free rate $r$. With $S_0 = \$100$, $r = 5\%$, $\sigma = 20\%$, and a one-year horizon, the *expected* terminal price under the model is $S_0 e^{rT} = 100 \cdot e^{0.05} = \$105.13$.

```python
import numpy as np

def simulate_terminal(S0, r, sigma, T, n, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    return ST

ST = simulate_terminal(100, 0.05, 0.20, 1.0, 1_000_000)
print(ST.mean())   # -> approximately 105.13  (analytic 100 * e^0.05)
```

The simulated terminal prices average **\$105.13**, matching $100 e^{0.05}$ to the penny over a million paths. The intuition: *GBM compounds random percentage shocks, so the price stays positive and its average grows at exactly the drift rate.* That terminal mean check is the GBM analogue of the die-roll check — prove the model is wired correctly before you ask it a hard question.

## Pricing a European option by Monte Carlo

Here is the marquee application. A *European call option* with strike \$100 gives its owner the right — not the obligation — to buy the stock for \$100 at expiry one year from now. If the stock ends at \$120 the owner exercises and pockets \$20; if it ends at \$90 the owner walks away and the option is worth \$0. The payoff at expiry is $\max(S_T - K, 0)$ where $K$ is the strike. The option's value *today* is the **risk-neutral expected payoff, discounted back to the present**:

$$\text{Call} = e^{-rT}\, E\!\left[\max(S_T - K,\, 0)\right]$$

Two pieces of vocabulary. *Risk-neutral* means we simulate the stock with drift equal to the risk-free rate $r$ rather than its real-world drift — a deep result ([risk-neutral pricing](/blog/trading/quantitative-finance/risk-neutral-pricing-martingale-measure-quant-interviews)) that says, for the purpose of pricing a derivative, you may pretend everyone is indifferent to risk and the stock grows at the safe rate. *Discounting* by $e^{-rT}$ converts a dollar received in one year into its worth today, because a dollar tomorrow is worth less than a dollar now. The Monte Carlo recipe writes itself: simulate many terminal prices, compute each payoff, average them, discount.

#### Worked example: price the call and compare to Black-Scholes

Take the textbook parameters: spot $S_0 = \$100$, strike $K = \$100$, risk-free rate $r = 5\%$, volatility $\sigma = 20\%$, maturity $T = 1$ year. The closed-form **Black-Scholes** price for these is exactly **\$10.4506** (computed from the formula below). Now price it by simulation:

```python
import numpy as np

def mc_call(S0, K, r, sigma, T, n, seed=7):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(ST - K, 0.0)
    disc = np.exp(-r * T) * payoff
    price = disc.mean()
    se = disc.std() / np.sqrt(n)
    return price, se

price, se = mc_call(100, 100, 0.05, 0.20, 1.0, 100_000)
print(price, se)                       # -> approximately (10.47, 0.047)
print(price - 1.96*se, price + 1.96*se)  # 95% CI brackets 10.4506
```

With 100,000 paths the simulation returns **\$10.47** with a standard error of about **\$0.047**, giving a 95% confidence interval of roughly **[\$10.38, \$10.56]**. That interval comfortably contains the exact Black-Scholes value of \$10.45. The simulation and the formula agree — and crucially, the simulation *knew nothing about the formula*. It just averaged payoffs over a cloud of simulated futures.

For comparison, here is the closed form the simulation is reproducing — the [Black-Scholes](/blog/trading/quantitative-finance/black-scholes) call price:

```python
from math import log, sqrt, exp
from statistics import NormalDist

def bs_call(S0, K, r, sigma, T):
    N = NormalDist().cdf
    d1 = (log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S0*N(d1) - K*exp(-r*T)*N(d2)

print(bs_call(100, 100, 0.05, 0.20, 1.0))   # -> 10.4506
```

Watch the Monte Carlo estimate close on the formula as the path count grows:

![A chart with the number of simulated paths N on a log-scale horizontal axis and the estimated call price in dollars on the vertical axis, showing a solid estimate line and a dashed ninety-five percent confidence band that is wide at small N and collapses onto a solid horizontal Black-Scholes line at ten dollars forty-five cents.](/imgs/blogs/monte-carlo-simulation-coding-quant-interviews-8.png)

At a few hundred paths the estimate swings by a dollar and the confidence band is fat; by 100,000 paths the band has collapsed to a few cents and the estimate is glued to the \$10.45 line. This single picture is the entire promise of Monte Carlo option pricing: *the right answer is the limit, and the confidence band tells you how close you are along the way.* The reason a desk reaches for Monte Carlo instead of Black-Scholes is that for most real payoffs — Asian options that average the price over time, barriers that knock out, baskets on many stocks — there *is* no closed form, and simulation is the only general tool. We checked it on a case with a formula precisely so we can trust it on the cases without one.

## Confidence intervals and convergence

We have been quoting "$\pm 1.96 \times \text{SE}$" without justifying it. Here is the logic. By the *central limit theorem* — the deep companion to the law of large numbers — the sample mean of many independent draws is approximately normally distributed around the true value, regardless of what the individual draws look like. For a normal distribution, 95% of the mass lies within 1.96 standard deviations of the center. So a 95% **confidence interval** for the true expectation is

$$\hat{\mu}_N \pm 1.96 \cdot \frac{s}{\sqrt{N}}$$

Read carefully, this says: if you repeated the whole simulation many times, about 95% of the intervals you construct this way would contain the true value. It is a statement about the *procedure*, not a probability about any single interval. The width of the interval is $2 \times 1.96 \times s/\sqrt{N}$ — and because it carries that $1/\sqrt{N}$, the interval narrows as a funnel, fast at first and then agonizingly slowly.

![A chart with the number of samples N on a log-scale horizontal axis and the estimate with its ninety-five percent interval on the vertical axis, showing a dashed upper and lower band that starts wide and funnels down onto a solid horizontal truth line, with vertical markers showing the band width shrinking as N grows.](/imgs/blogs/monte-carlo-simulation-coding-quant-interviews-11.png)

The funnel is the visual statement of "more samples, more confidence". The practical reading for an interview: always report the interval, never just the point estimate, and know that closing the funnel by a factor of ten costs a hundredfold more compute. If the interviewer asks "how many paths do you need for two-decimal accuracy?", you invert the formula: set $1.96 \cdot s/\sqrt{N} \le 0.005$ and solve for $N$. With $s \approx 14.8$ for the option payoff above, that demands $N \ge (1.96 \cdot 14.8 / 0.005)^2 \approx 34$ million paths for a penny of precision — which is exactly why the next section, variance reduction, matters so much.

## Variance reduction: better answers for free

Pushing $N$ higher buys accuracy slowly and expensively. The smarter move is to lower the $\sigma$ in $\sigma/\sqrt{N}$ — to make each sample *more informative* so the average converges faster at the same sample count. Two techniques dominate interviews.

### Antithetic variates

The idea is almost suspiciously simple. Every time you draw a standard normal $Z$, also use its mirror image $-Z$. Because $Z$ and $-Z$ are perfectly negatively correlated, the random ups and downs partially cancel when you average the pair: a draw that pushes the estimate high is paired with one that pushes it low. You get two samples for the price of one random number, and the *variance of the average* drops because the paired draws are anti-correlated.

![A strip plot with the pair average in standard-normal units on the vertical axis and two sampling schemes on the horizontal axis, showing independent pair averages scattered widely with standard deviation zero point seven one on the left and antithetic pair averages collapsed onto the mean line with standard deviation near zero on the right.](/imgs/blogs/monte-carlo-simulation-coding-quant-interviews-9.png)

The left column shows pair-averages from *independent* draws — they scatter with standard deviation $1/\sqrt{2} \approx 0.71$. The right column shows *antithetic* pair-averages of $Z$ and $-Z$ — they collapse onto the mean, because $Z$ and $-Z$ average to exactly zero. (In a real pricing problem the cancellation is partial, not total, because the payoff is a nonlinear function of $Z$; but it is still a free reduction whenever the payoff is monotone in the shock, which a vanilla call is.)

#### Worked example: antithetic variates on the call

```python
import numpy as np

def mc_call_antithetic(S0, K, r, sigma, T, n_pairs, seed=7):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_pairs)
    z_all = np.concatenate([z, -z])         # each draw and its mirror
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z_all)
    disc = np.exp(-r*T) * np.maximum(ST - K, 0.0)
    # variance computed on the n_pairs pair-averages, not the 2n samples
    pair_means = 0.5 * (disc[:n_pairs] + disc[n_pairs:])
    price = pair_means.mean()
    se = pair_means.std() / np.sqrt(n_pairs)
    return price, se
```

For a vanilla call the antithetic estimator typically cuts the standard error by 25–40% versus an independent sample of the same total size — a meaningful free lunch. The one-sentence intuition: *pairing each draw with its mirror cancels the symmetric part of the noise, so the average is calmer.*

### Control variates

The more powerful technique. A *control variate* is a second quantity $X$, computed from the same simulation, whose true mean $E[X]$ you happen to *know in closed form* and which is *correlated* with the thing you actually want, $Y$. You then estimate not $Y$ but the adjusted quantity $Y - b\,(X - E[X])$, where $b$ is a coefficient. Because you add back the known mean $E[X]$, the adjusted estimator has the *same* expectation as $Y$ (it stays unbiased), but most of its noise has been subtracted away by the correlated control.

![A flow diagram showing a simulated path producing a target Y with unknown mean and a control X with known expected value, combined into Y minus b times the quantity X minus its mean, which keeps the same mean with less noise, then averaged into a low-variance price.](/imgs/blogs/monte-carlo-simulation-coding-quant-interviews-10.png)

For our call, a perfect control is the terminal stock price $S_T$ itself: it is strongly correlated with the call payoff (when the stock ends high, both the payoff and $S_T$ are high), and its mean is known exactly, $E[S_T] = S_0 e^{rT} = \$105.13$. The optimal coefficient is $b = \text{Cov}(Y, X)/\text{Var}(X)$, estimated from the samples.

#### Worked example: control variate slashes the option-pricing error

```python
import numpy as np

def mc_call_control(S0, K, r, sigma, T, n, seed=2024):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z)
    Y = np.exp(-r*T) * np.maximum(ST - K, 0.0)   # discounted payoff
    X = ST                                        # control: terminal price
    EX = S0 * np.exp(r*T)                          # known mean of the control
    b = np.cov(Y, X)[0, 1] / np.var(X)            # optimal coefficient
    Z = Y - b * (X - EX)                           # control-adjusted estimator
    return Z.mean(), Z.std() / np.sqrt(n)

price_cv, se_cv = mc_call_control(100, 100, 0.05, 0.20, 1.0, 10_000)
print(price_cv, se_cv)   # approx (10.51, 0.057); plain MC se at 10k approx 0.150
```

At 10,000 paths the *plain* estimator has a standard error of about **\$0.150**; the control-variate estimator has a standard error of about **\$0.057** — a **2.6× reduction**, equivalent to running roughly *seven times* as many plain paths (because error scales as $1/\sqrt{N}$, a 2.6× error cut is a $2.6^2 \approx 6.8$× sample-count saving). The intuition: *subtract a correlated quantity whose answer you already know, and you subtract most of the noise along with it.* Control variates are the single highest-leverage variance-reduction trick on a real pricing desk, and naming them — plus the terminal-stock control — is a strong interview signal.

### Importance sampling for rare events

The third lever, and the right answer whenever the interviewer mentions a *rare* event. Suppose you want the probability the stock falls below \$50 in one year — a deep out-of-the-money crash. If that probability is, say, 0.2%, then out of 10,000 plain paths you expect only about 20 to land in the region you care about, and almost all your compute is wasted on paths that contribute a zero. Worse, the standard error of a rare-probability estimate is enormous *relative to the probability itself*: estimating a 0.2% event to 10% relative accuracy needs millions of plain draws.

**Importance sampling** fixes this by drawing from a *different*, shifted distribution that visits the rare region far more often, then correcting for the distortion with a weight. You sample where the action is and re-weight each sample by the ratio of the true density to the sampling density (the *likelihood ratio*), which keeps the estimator unbiased. For the crash probability you might shift the simulated drift sharply downward so that half your paths cross \$50, then multiply each crossing path's contribution by the likelihood ratio that accounts for the fact that, under the *real* dynamics, such a path was rare.

```python
import numpy as np

def crash_prob_is(S0, K, r, sigma, T, n, shift, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    z_shift = z + shift                      # push paths toward the crash
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z_shift)
    indicator = (ST < K).astype(float)
    weight = np.exp(-shift*z - 0.5*shift**2)  # likelihood ratio correction
    est = (indicator * weight).mean()
    se = (indicator * weight).std() / np.sqrt(n)
    return est, se
```

For a deep crash threshold the importance-sampling standard error can be ten to a hundred times smaller than plain Monte Carlo at the same sample count, because nearly every path now contributes information instead of a zero. The intuition: *to estimate a rare event, go looking for it on purpose and correct the bias with a weight.* Importance sampling is the workhorse behind credit-risk and tail-risk simulation, where the whole point is the 0.1% catastrophe nobody's plain simulation ever sees.

## Using a simulation to verify an analytic answer

This is the section that separates a good candidate from a great one, so it gets its own treatment. The most valuable thing Monte Carlo does in an interview is not computing answers you cannot get otherwise — it is **checking answers you can**. When you derive a probability by hand, a five-line simulation tells you within seconds whether your algebra is right. Interviewers watch for this reflex because it is exactly what the job demands: a quant who ships a mispriced model loses real money, and the cheapest insurance against that is a sanity-check simulation.

The workflow is always the same. Derive the analytic answer. Write the smallest possible loop that runs the literal experiment. Compare the simulated estimate (with its confidence interval) against your formula. If they agree to within a couple of standard errors, you trust the algebra. If they disagree by far more than the standard error, *one of them is wrong* — and the simulation, being a direct enactment of the experiment, is usually the one to trust.

#### Worked example: verify the birthday problem by simulation

The birthday problem is the canonical "verify it" question. In a room of 23 people, what is the probability that at least two share a birthday? The surprising analytic answer: compute the probability that *all* birthdays are distinct and subtract from 1.

$$P(\text{at least one match}) = 1 - \frac{365}{365}\cdot\frac{364}{365}\cdot\frac{363}{365}\cdots\frac{343}{365} = 1 - \prod_{k=0}^{22}\frac{365-k}{365}$$

This evaluates to **0.5073** — just over a coin flip, which strikes most people as far too high for only 23 people. Precisely because it is counterintuitive, you should *check it*:

```python
import numpy as np

def birthday_sim(people=23, trials=1_000_000, seed=42):
    rng = np.random.default_rng(seed)
    bdays = rng.integers(0, 365, size=(trials, people))
    # a match exists iff the row has fewer than `people` distinct values
    matches = [len(np.unique(row)) < people for row in bdays]
    p_hat = np.mean(matches)
    se = np.sqrt(p_hat * (1 - p_hat) / trials)
    return p_hat, se

print(birthday_sim())   # -> approximately (0.5074, 0.0005)
```

The simulation returns **0.5074 ± 0.0005**, and the analytic value 0.5073 sits squarely inside that interval. The algebra is confirmed. Notice what just happened: a counterintuitive result that you might have second-guessed at the whiteboard is now *certain*, because the simulation ran the actual experiment a million times and got the same number. That is the superpower. In the room, the move is: "The answer is about 50.7%. I find that surprising, so I'd confirm with a quick simulation" — and then write the loop. Saying it and meaning it is worth more than the formula.

A subtle but important point: the simulation does not just confirm the *number*, it confirms your *model of the problem*. If you had mis-set the experiment up — say, by checking only consecutive pairs instead of all pairs — the simulation would disagree with the (correct) formula, and the disagreement would flag your bug. The check cuts both ways, which is exactly why it is so valuable.

## In the interview room: fully-solved problems

Here are problems in the shape interviewers actually pose them, each solved end-to-end with the simulation that confirms it. The pattern to internalize: state the analytic approach, then describe (or write) the loop that verifies it, then report the estimate with its uncertainty.

#### Worked example: the dueling coins (first-to-heads wins)

*Two players alternate flipping a fair coin; the first to flip heads wins. What is the probability the first player wins?*

Analytic: the first player wins immediately with probability $\tfrac{1}{2}$; or both miss (probability $\tfrac{1}{2}\cdot\tfrac{1}{2} = \tfrac{1}{4}$) and the situation resets with the first player to move again. So $P = \tfrac{1}{2} + \tfrac{1}{4}P$, giving $P = \tfrac{2}{3} \approx 0.6667$. Verify:

```python
import numpy as np

rng = np.random.default_rng(1)
n = 2_000_000
flips_to_head = rng.geometric(0.5, n)   # 1,2,3,... = position of first head
first_player_wins = (flips_to_head % 2 == 1)   # odd position => player one wins
p_hat = first_player_wins.mean()
print(p_hat)   # -> approximately 0.6667
```

The simulation returns **0.6667**, confirming $\tfrac{2}{3}$. The first-mover advantage is real and large — a 67/33 split from nothing but turn order. Interview tip: the elegant self-referential equation $P = \tfrac{1}{2} + \tfrac{1}{4}P$ is the answer they want, and the simulation is the proof you reach for when they ask "are you sure?".

#### Worked example: expected number of rolls to see all six faces (coupon collector)

*You roll a fair die repeatedly. How many rolls on average until you have seen all six faces at least once?*

Analytic (the *coupon collector* problem): the wait to see the first new face is 1 roll; the wait for the second is geometric with success probability $5/6$, averaging $6/5$ rolls; and so on. The total is $6\left(\tfrac{1}{6}+\tfrac{1}{5}+\tfrac{1}{4}+\tfrac{1}{3}+\tfrac{1}{2}+1\right) = 6 \cdot 2.45 = 14.7$ rolls. Verify:

```python
import numpy as np

def rolls_to_collect(faces=6, trials=500_000, seed=3):
    rng = np.random.default_rng(seed)
    totals = np.empty(trials)
    for t in range(trials):
        seen, rolls = set(), 0
        while len(seen) < faces:
            seen.add(rng.integers(1, faces + 1))
            rolls += 1
        totals[t] = rolls
    return totals.mean(), totals.std() / np.sqrt(trials)

print(rolls_to_collect())   # -> approximately (14.70, 0.009)
```

The simulation returns **14.70 ± 0.01**, dead on the harmonic-sum formula $6 H_6 = 14.7$. The lesson: *the last few faces dominate the wait* — collecting the final face alone averages 6 rolls, because by then you are unlikely to hit a new one. This same "diminishing hit rate" structure shows up in caching, deduplication, and sampling rare market regimes.

#### Worked example: probability a stock finishes above its start

*A stock follows GBM with drift equal to the risk-free rate $r = 5\%$ and volatility $\sigma = 30\%$ over one year. What is the probability it ends above where it started?*

Analytic: the log-return $\ln(S_T/S_0)$ is normal with mean $(r - \tfrac{1}{2}\sigma^2)T = 0.05 - 0.045 = 0.005$ and standard deviation $\sigma\sqrt{T} = 0.30$. The probability the return exceeds 0 is $\Phi(0.005/0.30) = \Phi(0.0167) \approx 0.5066$. Note the subtlety: even though the drift is positive, the $-\tfrac{1}{2}\sigma^2$ correction makes the *median* return barely above zero, so the probability of finishing up is only just over 50%. Verify:

```python
import numpy as np

rng = np.random.default_rng(9)
n, r, sigma, T, S0 = 5_000_000, 0.05, 0.30, 1.0, 100.0
z = rng.standard_normal(n)
ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z)
p_hat = (ST > S0).mean()
print(p_hat)   # -> approximately 0.5066
```

The simulation returns **0.5066**, confirming the formula and the counterintuitive point: *high volatility drags the probability of finishing up toward (and for high enough vol, below) one half, even with positive drift*, because the volatility drag $\tfrac{1}{2}\sigma^2$ eats the drift. This is one of the most-missed facts in interviews, and the simulation makes it undeniable.

#### Worked example: estimate an integral that has no elementary antiderivative

*Estimate $\int_0^1 e^{-x^2}\,dx$.* This integral has no closed form in elementary functions (it is related to the error function), which makes it a clean Monte Carlo target.

Analytic framing: the integral equals $E[e^{-U^2}]$ where $U$ is uniform on $(0,1)$, because integrating against the uniform density (which is 1 on the interval) *is* taking the expectation. So sample uniforms, apply the function, average:

```python
import numpy as np

rng = np.random.default_rng(11)
n = 1_000_000
u = rng.random(n)
g = np.exp(-u**2)
est, se = g.mean(), g.std() / np.sqrt(n)
print(est, se)   # -> approximately (0.7468, 0.00018)
```

The estimate is **0.7468 ± 0.0002**; the true value is 0.74682. The intuition: *any definite integral is an expectation in disguise, so Monte Carlo can attack integrals that no calculus trick can crack* — and in high dimensions, where grid-based numerical integration explodes combinatorially, simulation is often the *only* feasible method. This is precisely why physics and finance reach for it: a 50-dimensional integral (a basket option on 50 stocks) is hopeless on a grid and routine by Monte Carlo.

#### Worked example: the two-envelope expected value, resolved by simulation

*An Asian-style payoff: you will receive the average of a stock's price observed monthly over one year. With $S_0 = \$100$, $r = 5\%$, $\sigma = 25\%$, what is the value today of receiving $\max(\bar{S} - 100, 0)$, where $\bar{S}$ is the 12-month average?*

There is no simple closed form for an arithmetic-average (Asian) option — which is exactly why it is a Monte Carlo question and a favorite for testing whether you can simulate a *path*, not just a terminal value:

```python
import numpy as np

def asian_call(S0, K, r, sigma, T, steps, n, seed=5):
    rng = np.random.default_rng(seed)
    dt = T / steps
    logS = np.full(n, np.log(S0))
    avg = np.zeros(n)
    for _ in range(steps):
        z = rng.standard_normal(n)
        logS += (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z
        avg += np.exp(logS)
    avg /= steps
    disc = np.exp(-r*T) * np.maximum(avg - K, 0.0)
    return disc.mean(), disc.std() / np.sqrt(n)

print(asian_call(100, 100, 0.05, 0.25, 1.0, 12, 200_000))   # approx (5.27, 0.018)
```

The Asian call prices at about **\$5.27 ± \$0.02**. Note it is *cheaper* than the equivalent vanilla call (which would be near \$12 at this higher volatility), because averaging the price over the year dampens the volatility of the payoff — and lower payoff volatility means a lower option price. The intuition the interviewer is fishing for: *path-dependent payoffs require simulating the whole path, and averaging reduces volatility, so Asian options cost less than their vanilla cousins.* If you can write the path loop and explain why the Asian is cheaper, you have demonstrated exactly the skill the question tests.

## Common misconceptions

Several beliefs about Monte Carlo are wrong in ways that cost interview points and, on a desk, real money.

**"More paths always fixes accuracy."** True in the limit, ruinous in practice. Because error falls as $1/\sqrt{N}$, going from 10,000 to 100,000 paths cuts your error by only $\sqrt{10} \approx 3.2$×, and the next digit costs a hundredfold more compute. Beyond a point, *throwing paths at the problem is the wrong move* — you switch to variance reduction (antithetic, control variates) or a better estimator, which lowers $\sigma$ instead of grinding up $N$. A candidate who answers "I'd just run more paths" to every accuracy question has missed the central trade-off.

**"A biased estimator just needs more samples."** No — bias does not shrink with $N$; only variance does. If your estimator is systematically wrong (a common cause: discretizing a continuous path too coarsely, so the simulated GBM drifts away from the true one), then $\hat{\mu}_N$ converges to the *wrong* value no matter how many paths you run. The confidence interval will be tight and confidently centered on a lie. Always separate *bias* (am I converging to the right thing?) from *variance* (how fast am I converging?), and check bias by refining the discretization, not by adding samples.

**"Reusing the same random draws is harmless."** Reusing a fixed seed for *reproducibility* is good practice. But reusing *correlated* draws where you assumed independence silently corrupts the standard error. If you simulate two "independent" scenarios from overlapping random streams, their results are correlated, your confidence intervals are too narrow, and you will be overconfident in a wrong answer. The standard-error formula $s/\sqrt{N}$ assumes independent samples; violate that and the formula lies to you. (This is the same trap as reusing data across train and test folds — covered in [overfitting and purged cross-validation](/blog/trading/quantitative-finance/overfitting-purged-cv-deflated-sharpe-quant-research).)

**"The confidence interval is a probability about the true value."** A 95% interval does *not* mean "there is a 95% chance the true value is in here". The true value is fixed; the interval is random. The correct reading is about the *procedure*: 95% of intervals constructed this way, over many repetitions, would contain the truth. Mis-stating this is a classic tell that someone has memorized the formula without understanding it.

**"Monte Carlo and random sampling are the same as a random-number generator being 'truly random'."** Production Monte Carlo uses *pseudo*-random numbers — deterministic sequences that merely look random. That is a feature, not a bug: it makes runs reproducible and debuggable. For high-dimensional integration, practitioners even deliberately use *quasi*-random (low-discrepancy) sequences that are *more evenly spread* than true randomness, achieving faster-than-$1/\sqrt{N}$ convergence. "Random" in Monte Carlo means "carefully engineered to behave like random", not "unpredictable".

## How it shows up on a real desk

Monte Carlo is not an interview party trick; it is load-bearing infrastructure on a live trading floor. Here is where the same draw-and-average machine you just learned actually runs, with real money on the line.

![A matrix mapping four desk uses of Monte Carlo to what each computes and why simulation is required: pricing exotics computes path-dependent option value because there is no closed form, risk computes a ninety-nine percent one-day loss because it must aggregate fat-tailed profit and loss, backtests compute profit and loss under resampled returns to stress the path, and Greeks compute sensitivities by bumping the seed for stable finite differences.](/imgs/blogs/monte-carlo-simulation-coding-quant-interviews-12.png)

**Pricing exotic and path-dependent derivatives.** The vanilla call we priced has a closed form, but most of what a structuring desk sells does not: Asian options (averages), barrier options (knock-in/knock-out at a level), lookbacks (payoff on the max or min), and multi-asset baskets. For these, Monte Carlo is not one option among several — it is the *only* general method. A desk pricing a 5-year autocallable note on a basket of three indices ([autocallables](/blog/trading/quantitative-finance/autocallables)) runs millions of correlated paths nightly; the price the client pays is a discounted average over that simulated cloud, exactly as in our \$10.45 example but with a payoff far too gnarly for a formula.

**Risk: value-at-risk and expected shortfall.** Every bank computes *value-at-risk* (VaR) — the loss its book would not exceed on, say, 99% of days. Analytic VaR assumes returns are normal, which fat-tailed markets brutally violate. Monte Carlo VaR instead simulates tens of thousands of correlated market scenarios (rates, equities, FX, credit all moving together), revalues the entire portfolio in each, and reads the 1st-percentile loss off the simulated distribution. When a risk report says "99% one-day VaR is \$4.0 million", that number very often came out of a simulation, because only simulation can aggregate the fat-tailed, nonlinear, cross-correlated P&L of a real book.

**Backtesting and strategy stress-testing.** A backtest on a single historical path tells you what *did* happen, not what *could* have. Quants resample or bootstrap returns to generate thousands of *alternative* histories and ask how the strategy fares across them — does the Sharpe ratio survive, or did it depend on one lucky decade? This is the Monte Carlo answer to overfitting: a strategy that only works on the one path that actually occurred is a fragile strategy, and simulating the counterfactual paths exposes it ([backtesting done right](/blog/trading/quantitative-finance/backtesting-done-right-quant-research)).

**Greeks by bumping the seed.** To hedge an option a desk needs its *Greeks* — sensitivities of the price to the stock, to volatility, to time. For Monte Carlo prices, you estimate a Greek by *bumping* an input (say, raising the spot by \$0.01) and re-pricing. The trick that makes this stable: reuse the *same random seed* for the bumped and unbumped runs, so the common simulation noise cancels in the difference and you measure the genuine sensitivity rather than Monte Carlo jitter. It is the one place where reusing identical draws is not just allowed but essential — the controlled cousin of the correlated-draws mistake warned about above.

**The October 1987 lesson.** When markets crashed 22% in a single day, every analytic VaR model that assumed normal returns had called such a move essentially impossible — a 20-plus-standard-deviation event that "should" happen once in billions of years. Desks that simulated from fatter-tailed distributions, or that stress-tested across resampled crash scenarios, were far less blindsided. The episode is the permanent argument for simulation over closed-form risk: *the formula assumes a world; the simulation lets you assume a worse one and see what breaks.*

## When this matters and where to go next

If you are interviewing for a quant role, internalize one move above all others: when you give an analytic answer, follow it with "and I'd confirm it with a quick simulation", then actually describe the loop. It demonstrates that you think in terms of runnable experiments, that you respect the gap between an elegant derivation and a correct one, and that you can code. That single reflex, applied to probability puzzles, expected-value questions, and pricing problems alike, is the highest-leverage habit you can build for these interviews — and it happens to be exactly the habit the job rewards.

Practically, the skills compound. The $1/\sqrt{N}$ standard error governs how much data you need to trust *any* estimate, from a backtest's Sharpe ratio to an A/B test's lift. Inverse-CDF and Box-Muller are the sampling primitives behind every simulation you will ever write. Variance reduction is the difference between a pricing library that runs in seconds and one that runs overnight. And the discipline of always pairing an estimate with a confidence interval is the line between a measurement and a guess.

To go deeper, the natural next stops are [geometric Brownian motion and the SDEs behind it](/blog/trading/quantitative-finance/stochastic-differential-equations-gbm-ou-quant-interviews) for the continuous-time foundations of the price paths we simulated, [Brownian motion for quant interviews](/blog/trading/quantitative-finance/brownian-motion-quant-interviews) for the random walk underneath, [Black-Scholes](/blog/trading/quantitative-finance/black-scholes) for the closed form our simulation reproduced, and [risk-neutral pricing](/blog/trading/quantitative-finance/risk-neutral-pricing-martingale-measure-quant-interviews) for *why* we set the drift to the risk-free rate when pricing. Each one deepens a piece of the machine you just built. And the best practice of all is the cheapest: pick any probability puzzle, derive the answer, and then — every single time — write the five-line loop that checks it.

*This article is educational, not financial advice. Simulated prices and probabilities are models of reality, not guarantees; real markets violate every assumption, which is exactly why we stress-test them.*

---
title: "Expectation, variance, and higher moments: the four numbers that describe any return"
date: "2026-06-15"
description: "A beginner-friendly, build-from-zero tour of expected value, variance and volatility, skewness, and kurtosis, and how quants use these four moments to measure expected profit, risk, asymmetry, and fat-tail crash danger."
tags: ["expectation", "variance", "volatility", "skewness", "kurtosis", "moments", "risk", "fat-tails", "quant-finance", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Every return distribution can be summarized by four numbers — its mean, variance, skewness, and kurtosis — and each one answers a different practical question a trader cares about.
>
> - **Expected value** $E[X]$ is the probability-weighted average outcome; thanks to *linearity*, the expected profit of a whole strategy is just the sum of the expected profits of its parts.
> - **Variance** $\mathrm{Var}(X) = E[X^2] - E[X]^2$ measures how much returns swing; its square root is **volatility**, the headline risk number, and it scales with the square root of time.
> - **Skewness** (the third moment) measures asymmetry — why selling options earns a steady trickle of small gains but hides a rare large loss, a payoff with *negative skew*.
> - **Kurtosis** (the fourth moment) measures fat tails — why real markets produce "once-in-10,000-years" moves every few years, and why a bell curve badly underestimates crash risk.
> - The one number to remember: under a normal curve a **5-sigma day** should happen about once every **7,000 years**; in real equity markets they show up every **few years** — the entire gap is kurtosis.

Here is a question that sits underneath almost every decision a trader makes: if I told you a strategy "makes 8% a year," would you put your money in it?

You cannot answer yet, and the reason is the whole subject of this post. "Makes 8% a year" is a statement about *one* number — the average, the center, the thing mathematicians call the **expected value**. But two strategies that both average 8% can be utterly different animals. One might grind out 8% with barely a wobble. Another might average 8% by making 30% in most years and losing 60% in the occasional crash. A third might quietly collect a small premium month after month — until one day it hands back five years of gains in a single afternoon. The average hides all of this. To actually see a strategy, you need three more numbers: how much it swings (variance), whether its surprises are skewed toward the upside or the downside (skewness), and how prone it is to enormous, rare shocks (kurtosis). These four numbers are called the **moments** of the distribution, and learning to read them is one of the most useful skills a quant can have.

![Pipeline from listing outcomes through weighting by probability to the expected value](/imgs/blogs/expectation-variance-moments-math-for-quants-1.png)

The diagram above is the mental model for the first idea we will build: the expected value is not magic, it is a recipe. You list every outcome, you weight each one by how likely it is, and you add the weighted pieces together. The result is the center of gravity of all the things that could happen. Everything else in this post — risk, asymmetry, fat tails — is built from the same move (weight outcomes by probability and sum) applied to cleverer quantities. Let us start from absolute zero and build all four moments, tying each one to a concrete dollar decision a trader actually faces.

## Foundations: the building blocks of a distribution

Before we can talk about four moments, we need to agree on what every word means. We will define each term the first time it appears, build the simplest possible version of each idea, and only then climb toward the machinery a practitioner uses. If you already know what a random variable is, you can skim; if you do not, you can still follow every step.

### What is a random variable?

A **random variable** is a number whose value is not known until some uncertain event resolves. Tomorrow's return on a stock is a random variable: today you do not know it, but tomorrow it will turn out to be some specific number. We write random variables with capital letters — $X$ for "the return of this strategy," $Y$ for "the return of that asset" — and we write a specific observed value with a lowercase letter or a plain number.

A random variable comes with a **distribution**: the full list of values it can take, together with how likely each value is. For a coin-flip bet, the distribution is simple — two outcomes, each with probability one-half. For a stock's daily return, the distribution is a smooth curve covering a continuous range of possible percentages. Both are distributions; the math we are about to build works for both.

### What is a probability?

A **probability** is a number between 0 and 1 that says how likely an outcome is. A probability of 0 means "never," 1 means "certain," and 0.5 means "half the time." The probabilities of all the mutually exclusive outcomes must add up to exactly 1, because *something* always happens. We will write the probability of an outcome $x$ as $p(x)$ for a discrete (finite-list) distribution, and as a **density** $f(x)$ for a continuous one. A density is a slightly subtler object — it is probability *per unit of $x$*, so you get an actual probability only by integrating it over a range — but for our purposes it works the same way: $f(x)$ is taller where outcomes are more likely.

### What is the expected value?

The **expected value** of a random variable, written $E[X]$, is its probability-weighted average — the long-run average value you would get if you could repeat the random experiment infinitely many times and average all the results. It is also called the **mean** and written $\mu$ (the Greek letter "mu"). The everyday analogy: if a slot machine pays \$10 one time in five and nothing otherwise, its expected payout per pull is \$10 weighted by the one-in-five chance, which is \$2. You will not win \$2 on any single pull — you win \$10 or \$0 — but over thousands of pulls your average haul converges to \$2 a pull. That long-run average is the expected value.

Formally, for a discrete random variable we sum each outcome times its probability:

$$ E[X] = \sum_x x \, p(x). $$

Here the sum runs over every possible outcome $x$, and $p(x)$ is its probability. For a continuous random variable we replace the sum with an integral and the probabilities with the density:

$$ E[X] = \int_{-\infty}^{\infty} x \, f(x)\, dx. $$

The integral sign is just a continuous version of the sum: it adds up $x$ weighted by the density $f(x)$ across the whole range. Do not let the integral intimidate you — it is doing exactly what the sum does, only over a smooth curve instead of a finite list. The mean is the *center of gravity* of the distribution: if you cut the probability curve out of cardboard, the mean is the point where it would balance on a pin.

#### Worked example: the expected value and variance of a single bet

You are offered a bet on a single trade. You stake \$1,000. With probability 60% the trade works and you make a 20% gain — your \$1,000 becomes \$1,200, a profit of \$200. With probability 40% it fails and you lose 15% — your \$1,000 becomes \$850, a loss of \$150. Should you take it? Let us compute the expected profit, step by step, the way you would in a spreadsheet.

Define $X$ as the dollar profit of the bet. It has two outcomes: $+\$200$ with probability 0.60, and $-\$150$ with probability 0.40. The expected profit is each outcome weighted by its probability:

$$ E[X] = 0.60 \times 200 + 0.40 \times (-150) = 120 - 60 = \$60. $$

So on average this bet earns you \$60, or 6% of your \$1,000 stake. The expected value is positive, which is the first thing a trader checks — a bet with negative expected value is one you would, on average, lose money taking, no matter how exciting the upside looks. But \$60 is only the center. To know the risk, we need the spread, which we will compute in the next section using this same example. The one-sentence intuition: the expected value tells you whether a bet is worth taking on average, but it says nothing yet about how badly a single attempt can go.

### Linearity: the property that makes expectation usable

The single most useful fact about expected value is that it is **linear**. This means two things. First, for any constants $a$ and $b$,

$$ E[aX + b] = a\,E[X] + b. $$

Scaling a random variable scales its mean; shifting it shifts its mean. Second, and far more important, for *any* two random variables $X$ and $Y$ — whether or not they are related —

$$ E[X + Y] = E[X] + E[Y]. $$

The expected value of a sum is the sum of the expected values, always. This is the property that lets a quant reason about a strategy made of many trades, or a portfolio made of many positions, without drowning in dependencies. The expected profit of a book is simply the sum of the expected profits of every position in it — even if those positions are tangled together in complicated ways. We will lean on this constantly. The way this works is worth one figure to make concrete, because it is the engine behind every "expected PnL" calculation a desk runs.

![Before and after panels showing low-variance and high-variance return streams with the same mean](/imgs/blogs/expectation-variance-moments-math-for-quants-2.png)

The figure above previews the next idea: two strategies can have *exactly the same mean* — the same expected return — and yet feel completely different to live through, because one swings gently and the other lurches violently. The mean cannot tell them apart. That difference is the second moment, and it is where risk lives.

## Expected PnL: linearity in action on a real strategy

Let us make linearity pay rent. Suppose you run a market-making strategy that does many small trades a day. Each trade has the same edge: on average it earns you a tiny profit of \$0.50 (your *edge*, the expected profit per trade after costs). The trades are not independent — when the market is volatile, several go your way or against you together — but linearity does not care about that. If you do 2,000 trades in a day, your expected daily profit is

$$ E[\text{daily PnL}] = \sum_{i=1}^{2000} E[\text{trade } i] = 2000 \times \$0.50 = \$1{,}000. $$

We summed the expected profit of each trade, ignoring entirely how the trades are correlated, because expectation is linear regardless of dependence. This is why a market maker can quote an expected daily PnL even though the individual trades are wildly interdependent. The same logic scales an *expected return* up across a year: if a strategy earns an expected 0.04% per trading day and there are 252 trading days in a year, its expected annual return is, to a first approximation, $252 \times 0.04\% \approx 10\%$ (compounding nudges this slightly, but the linear sum is the backbone).

> The mean tells you where you will end up on average. It is necessary, never sufficient. A trader who looks only at the mean is reading the first word of a sentence and guessing the rest.

A word of caution that we will return to: linearity holds for the *mean*, but it does **not** hold for the variance. The variance of a sum is *not* the sum of the variances unless the parts are uncorrelated. That single asymmetry — means add cleanly, risks do not — is the reason diversification exists and the reason the next several sections matter so much.

## Variance and volatility: the size of the swings

The mean tells you the center. **Variance** tells you the spread — how far, on average, the outcomes land from that center. Two strategies with the same 8% average return can have radically different variance: one barely deviates from 8%, the other ricochets between +40% and −24%. Variance is the number that distinguishes them, and in finance, spread *is* risk.

The everyday analogy: picture two commuter trains that both average a 30-minute trip. One is always within two minutes of schedule; the other is sometimes 15 minutes early and sometimes 15 minutes late, averaging out to 30. They have the same mean trip time. The second train has far higher variance, and you would leave for the station much earlier to be safe. Variance is uncertainty, and uncertainty is what you have to protect against.

![Matrix listing the four moments mean, variance, skewness, and kurtosis with what each one measures](/imgs/blogs/expectation-variance-moments-math-for-quants-3.png)

The figure above is the map of the whole post: four moments, four questions. Mean asks "where is the center?" Variance asks "how wide is the spread?" Skewness asks "is it lopsided?" Kurtosis asks "how fat are the tails?" We are now on the second row. Formally, variance is the expected value of the *squared* distance from the mean:

$$ \mathrm{Var}(X) = E\big[(X - \mu)^2\big]. $$

We subtract the mean to measure distance from the center; we square it so that landing 5% below hurts the same as landing 5% above (and so the positive and negative deviations do not cancel to zero); and we take the expected value to average over all outcomes. There is a second formula for variance that is usually easier to compute and that you should memorize:

$$ \mathrm{Var}(X) = E[X^2] - \big(E[X]\big)^2. $$

In words: the variance is "the mean of the squares minus the square of the mean." These two formulas are algebraically identical — expand $(X-\mu)^2 = X^2 - 2\mu X + \mu^2$, take expectations using linearity, and the cross term collapses — but the second one is the one you will actually use, because you can accumulate $E[X^2]$ and $E[X]$ in a single pass over your data.

### What is volatility?

Variance has an awkward feature: its units are *squared*. If returns are in percent, variance is in percent-squared, which means nothing to a human. So we take its square root and get the **standard deviation**, written $\sigma$ (the Greek letter "sigma"):

$$ \sigma = \sqrt{\mathrm{Var}(X)}. $$

The square root undoes the squaring, so $\sigma$ is back in plain, interpretable units — plain percent, or plain dollars. In finance the standard deviation of returns has its own name: **volatility**. When a trader says "this stock runs at 20% vol," they mean its annual returns have a standard deviation of 20%. Volatility is the single most quoted risk number in all of markets, and it is nothing more than the square root of variance.

#### Worked example: the variance and volatility of the single bet

Let us return to the \$1,000 bet from the foundations section: $+\$200$ with probability 0.60, $-\$150$ with probability 0.40, mean profit $E[X] = \$60$. Now we compute its variance and volatility, so we can see the *risk* alongside the \$60 edge.

Using the squared-deviation definition, we take each outcome's distance from the \$60 mean, square it, and weight by probability:

- Good outcome: deviation $= 200 - 60 = 140$, squared $= 19{,}600$, weighted $= 0.60 \times 19{,}600 = 11{,}760$.
- Bad outcome: deviation $= -150 - 60 = -210$, squared $= 44{,}100$, weighted $= 0.40 \times 44{,}100 = 17{,}640$.

Add the two weighted pieces to get the variance:

$$ \mathrm{Var}(X) = 11{,}760 + 17{,}640 = 29{,}400 \ \text{dollars}^2. $$

The variance is 29,400 dollars-squared — an uninterpretable number. Take the square root to get the volatility:

$$ \sigma = \sqrt{29{,}400} \approx \$171. $$

So this bet has an expected profit of \$60 but a standard deviation of about \$171 — nearly three times the edge. The "typical" swing dwarfs the average gain, which tells you that a *single* trade is mostly noise; only over many independent repetitions does the \$60 edge reliably emerge. The one-sentence intuition: variance turns "what do I expect to make?" into "how much will the answer bounce around?", and here the bounce is far larger than the edge.

### Annualizing volatility: the square-root-of-time rule

Volatility is usually quoted *annually* (20% a year), but it is *measured* from higher-frequency data — daily, even minute-by-minute returns. To convert between horizons, quants use the **square-root-of-time rule**. If daily returns are independent and each day has the same variance $\sigma_d^2$, then the variance over $n$ days is $n\sigma_d^2$ (variances of independent things add), so the volatility over $n$ days is

$$ \sigma_n = \sigma_d \sqrt{n}. $$

Volatility grows with the *square root* of time, not with time itself. This is the single most-used formula in practical risk management, and it follows directly from "variances of independent returns add up." The reason it is the square root and not linear is exactly the squaring inside variance: the spreads do not add, the *squared* spreads do.

#### Worked example: annualizing 1% daily vol on a \$1,000,000 book

You run a book worth \$1,000,000. You measure its daily return volatility at 1% — on a typical day the book's value moves about 1% up or down. Two questions: what is the annual volatility, and what is the dollar size of a "one-sigma" daily move?

There are about 252 trading days in a year. Applying the square-root-of-time rule with $\sigma_d = 1\%$ and $n = 252$:

$$ \sigma_{\text{annual}} = 1\% \times \sqrt{252} \approx 1\% \times 15.87 = 15.9\%. $$

So a book that moves 1% on a typical day has an annual volatility of about 15.9% — strikingly, the small daily wiggle compounds (in the square-root sense) into a substantial annual figure. Now translate the daily vol to dollars on the \$1,000,000 book. A one-sigma daily move is 1% of \$1,000,000:

$$ \text{one-sigma daily move} = 0.01 \times \$1{,}000{,}000 = \$10{,}000. $$

A typical day moves the book by about \$10,000 in either direction. A two-sigma day — which, under a bell curve, happens on roughly 5% of days, about once a month — would move it \$20,000. And the annual one-sigma move is $15.9\% \times \$1{,}000{,}000 \approx \$159{,}000$. The one-sentence intuition: the square-root-of-time rule lets you translate the small, measurable daily wiggle into the big, decision-relevant annual risk, and into the concrete dollars you could lose on an ordinary day.

A practical caveat worth stating: the square-root rule assumes returns are *independent* across days with *constant* variance. Real markets violate both — volatility clusters (calm begets calm, storms beget storms) and returns have mild autocorrelation — so the rule is an approximation that tends to *understate* risk during turbulent regimes. It is the right back-of-envelope tool, not the final word.

## Covariance and correlation: a bridge to portfolios

So far we have looked at one random variable at a time. The moment you hold more than one asset, a new question appears: do they move *together*? The answer is captured by **covariance**, the natural two-variable extension of variance. Where variance is the expected squared deviation of $X$ from its own mean, covariance is the expected *product* of the deviations of two variables:

$$ \mathrm{Cov}(X, Y) = E\big[(X - \mu_X)(Y - \mu_Y)\big] = E[XY] - E[X]\,E[Y]. $$

If $X$ and $Y$ tend to be above their means at the same time (and below at the same time), the product of their deviations is usually positive, so the covariance is positive: they move together. If one is usually high when the other is low, the covariance is negative: they offset. Variance is just covariance with itself — $\mathrm{Cov}(X, X) = \mathrm{Var}(X)$ — which is a tidy way to see that these are the same idea at different counts of inputs.

Covariance has the same units problem variance does (it is in "percent times percent"), so we standardize it into **correlation**, written $\rho$ (the Greek letter "rho"):

$$ \rho_{XY} = \frac{\mathrm{Cov}(X, Y)}{\sigma_X \, \sigma_Y}. $$

Dividing by both standard deviations strips out the units and the scale, leaving a pure number between $-1$ and $+1$. A correlation of $+1$ means the two move in lockstep, $-1$ means they move exactly opposite, and $0$ means no *linear* relationship. This is the bridge to portfolio math: the reason variances do *not* simply add is that the covariance terms enter the total. For two positions with weights $w_1$ and $w_2$,

$$ \mathrm{Var}(w_1 X + w_2 Y) = w_1^2 \sigma_X^2 + w_2^2 \sigma_Y^2 + 2 w_1 w_2 \,\mathrm{Cov}(X, Y). $$

That final cross term — the covariance — is exactly where diversification lives, and it is the whole subject of the [covariance and correlation pitfalls guide](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews). When the covariance is negative, the cross term *subtracts* from total risk, and the portfolio is safer than either piece alone. The dedicated covariance-matrix post in this series carries the full machinery, but the key point is that the same "weight by probability and sum" move that built the mean and variance also builds the object that governs whole portfolios. Let us make that concrete with a number, because the portfolio variance formula is where the difference between "means add cleanly" and "risks do not" becomes a dollar figure.

#### Worked example: the variance of a two-asset portfolio

You split \$1,000,000 evenly between two assets, \$500,000 each, so each weight is $w = 0.5$. Both assets have the same annual volatility, $\sigma = 20\%$, which on \$500,000 is a \$100,000 one-sigma swing each. The question is how risky the *combined* book is, and the answer depends entirely on the correlation between them.

Start with the formula, using $w_1 = w_2 = 0.5$ and $\sigma_X = \sigma_Y = 0.20$, and recall that $\mathrm{Cov}(X,Y) = \rho \, \sigma_X \sigma_Y$:

$$ \sigma_p^2 = w_1^2 \sigma_X^2 + w_2^2 \sigma_Y^2 + 2 w_1 w_2 \,\rho\, \sigma_X \sigma_Y. $$

**Case 1 — perfect correlation ($\rho = 1$).** The two assets are really the same bet twice. Plugging in:

$$ \sigma_p^2 = 0.25(0.04) + 0.25(0.04) + 2(0.25)(1)(0.04) = 0.01 + 0.01 + 0.02 = 0.04, $$

so $\sigma_p = 20\%$ — no improvement at all. On the \$1,000,000 book that is a \$200,000 one-sigma swing. Diversification did nothing because there was nothing to diversify.

**Case 2 — zero correlation ($\rho = 0$).** The cross term vanishes:

$$ \sigma_p^2 = 0.01 + 0.01 + 0 = 0.02, \quad \sigma_p = \sqrt{0.02} \approx 14.1\%. $$

The portfolio volatility dropped from 20% to about 14.1% — a \$141,000 one-sigma swing instead of \$200,000 — for free, just by holding two unrelated bets instead of one doubled bet. That $1/\sqrt{2}$ reduction is the single most quoted result in diversification.

**Case 3 — negative correlation ($\rho = -0.5$).** The cross term subtracts:

$$ \sigma_p^2 = 0.01 + 0.01 + 2(0.25)(-0.5)(0.04) = 0.02 - 0.01 = 0.01, \quad \sigma_p = 10\%. $$

Now the combined book has *half* the volatility of either asset alone — a \$100,000 one-sigma swing on \$1,000,000. The one-sentence intuition: the mean of the portfolio is just the average of the two means regardless of correlation, but the *risk* swings from \$200,000 to \$100,000 depending on the covariance — which is precisely why expectation is linear and variance is not.

The contrast between the three cases is worth a table, because it makes the asymmetry between mean and variance impossible to miss:

| Correlation $\rho$ | Cross term | Portfolio vol | One-sigma swing on \$1M |
| --- | --- | --- | --- |
| $+1$ (identical) | adds fully | 20% | \$200,000 |
| $0$ (unrelated) | zero | 14.1% | \$141,000 |
| $-0.5$ (offsetting) | subtracts | 10% | \$100,000 |

Read down the last column: the expected return is identical in all three rows, yet the risk you actually carry ranges over a factor of two. That spread is the entire reason a quant cares about the second moment of a *pair* of assets, not just of one.

## Skewness: the third moment and asymmetric payoffs

The mean is the center, the variance is the spread, and both treat upside and downside symmetrically — the squaring in variance makes +5% and −5% count equally. But real payoffs are often *lopsided*. A lottery ticket loses a little almost always and wins a fortune almost never; selling insurance earns a little almost always and pays out a fortune almost never. These two are mirror images, and the moment that captures the difference is **skewness**, the third moment.

![Stack of four layers: mean, then variance, then skewness, then kurtosis, each adding one feature](/imgs/blogs/expectation-variance-moments-math-for-quants-4.png)

The stack above is how to hold the four moments in your head: each layer adds one feature the layer below it could not see. The mean places the distribution; the variance widens it; the skewness tilts it to one side; the kurtosis fattens its tails. Skewness is the *third standardized moment* — the expected cubed deviation from the mean, divided by the cube of the volatility:

$$ \text{Skew}(X) = E\!\left[\left(\frac{X - \mu}{\sigma}\right)^{3}\right]. $$

Two things make the cube the right tool. First, an *odd* power preserves sign: a deviation that is negative cubes to a negative number, while a positive deviation cubes to positive — so the two sides do *not* cancel the way they do under the even square. Second, the cube *amplifies* the large deviations more than the small ones, so the tail that reaches farther dominates the sum. Dividing by $\sigma^3$ makes the result unit-free, so you can compare the skewness of a penny stock to that of a Treasury bond. The sign is what matters:

- **Positive skew**: the right (gain) tail is longer. Most outcomes are small losses or small gains, but the rare surprise is a big *win*. A lottery ticket. Buying far-out-of-the-money options. You lose a little often and win big rarely.
- **Negative skew**: the left (loss) tail is longer. Most outcomes are small *gains*, but the rare surprise is a big *loss*. Selling insurance. Selling options. You win a little often and lose big rarely.

![Before and after panels contrasting a symmetric coin bet with a short put payoff](/imgs/blogs/expectation-variance-moments-math-for-quants-6.png)

The figure above contrasts a perfectly symmetric bet (a coin flip for \$100, skew exactly zero) with a short-put payoff (keep a small premium most of the time, suffer a large loss occasionally, skew strongly negative). Selling options is the textbook negative-skew trade, and it is worth a full worked example because it is one of the most common — and most misunderstood — payoff shapes in all of trading.

#### Worked example: the negative skew of a short put

You **sell a put option** on a stock. Selling a put means you collect a premium today in exchange for a promise: if the stock falls below a *strike* price by expiry, you must buy it at that strike — taking the loss. Suppose the stock trades at \$100, you sell a put struck at \$95, and you collect a premium of \$5. Two outcomes dominate, and we will assign rough probabilities to see the shape.

Outcome one, the common case (say 90% of the time): the stock stays at or above \$95. The put expires worthless, you owe nothing, and you simply keep the \$5 premium. Profit: $+\$5$.

Outcome two, the rare case (say 10% of the time): the stock crashes — suppose to \$80. You must buy at the \$95 strike a stock now worth \$80, a \$15 loss on the position, partly offset by the \$5 you collected. Net profit: $5 - 15 = -\$10$. (In a true crash to, say, \$60, the loss would be far worse — $5 - 35 = -\$30$ — which is exactly the long left tail.)

Now look at the shape. The expected profit is $0.90 \times 5 + 0.10 \times (-10) = 4.5 - 1.0 = +\$3.50$ — positive, so on average the trade makes money. But the *distribution* is brutally asymmetric: a small \$5 gain 90% of the time, a large loss 10% of the time, with a tail that gets worse the harder the stock falls. Plug deviations into the cubed formula and the negative tail dominates the sum, so the skewness is firmly negative. This is the famous **"picking up pennies in front of a steamroller"** — a steady, comfortable trickle of small gains that masks a rare, flattening loss. The one-sentence intuition: negative skew means the average and the *typical* outcome both look great right up until the day the long tail arrives.

Why does this matter to a desk? Because a strategy's Sharpe ratio (return divided by volatility) can look gorgeous while it is quietly accumulating negative skew — and the volatility number, which treats up and down moves equally, completely fails to warn you about the steamroller. Two strategies with identical mean and identical volatility can have opposite skew, and the negative-skew one is the one that blows up.

## Kurtosis: the fourth moment and fat tails

We have placed the distribution (mean), widened it (variance), and tilted it (skewness). The last of the four moments answers the question that has bankrupted more funds than any other: *how often do truly enormous moves happen?* That is **kurtosis**, the fourth moment, and it is the mathematics of fat tails.

The everyday framing: most days in a market are boring — small moves clustered near zero. The bell curve (the **normal distribution**, the famous symmetric hump) captures this central calm beautifully. But the bell curve makes a confident, specific, and *wrong* prediction about the rare, extreme days. It says they are vanishingly unlikely. Real markets say otherwise. Kurtosis is the number that measures how much heavier the real tails are than the bell curve's tails.

![Branching graph contrasting a normal model with fat-tailed reality, leading to frequent 5-sigma days](/imgs/blogs/expectation-variance-moments-math-for-quants-7.png)

The figure above shows the fork: from the same daily returns you can *assume* a normal model (whose center fits well but whose tails are far too thin) or *observe* the fat-tailed reality (whose tails carry much more mass, implying that 5-sigma days happen far more often than the bell curve allows). Kurtosis is the fourth standardized moment — the expected *fourth* power of the standardized deviation:

$$ \text{Kurt}(X) = E\!\left[\left(\frac{X - \mu}{\sigma}\right)^{4}\right]. $$

The fourth power, being even, makes every deviation positive (so the two tails reinforce rather than cancel), and being a *high* power it weights the far-out deviations enormously — a deviation of 5 contributes $5^4 = 625$, while a deviation of 1 contributes just 1. So kurtosis is dominated almost entirely by the extreme observations; it is, almost by construction, a tail-weight detector. For the normal distribution this quantity equals exactly **3**, no matter the mean or variance. Because 3 is the universal baseline, quants almost always subtract it and report **excess kurtosis**:

$$ \text{Excess Kurtosis}(X) = \text{Kurt}(X) - 3. $$

A normal distribution has excess kurtosis 0. A distribution with **positive excess kurtosis** is **leptokurtic** — fatter tails and a sharper peak than the bell curve — which is exactly what real return data looks like. Daily equity returns typically show excess kurtosis between 3 and 6 (so raw kurtosis 6 to 9), and during crisis periods it runs far higher.

#### Worked example: excess kurtosis and the frequency of 5-sigma days

Here is the number that should change how you think about risk. Under the **normal** distribution, a move of 5 standard deviations or more in one direction has a probability of about $2.87 \times 10^{-7}$ — roughly one in 3.5 million. With about 252 trading days a year, that implies a 5-sigma down day should occur about once every

$$ \frac{1}{2.87 \times 10^{-7} \times 252} \approx \frac{1}{7.2 \times 10^{-5}} \approx 13{,}900 \ \text{years}. $$

Counting only down moves (the direction that hurts a long book), call it one catastrophic day roughly every 14,000 years; counting both directions, a 5-sigma move of *either* sign about every 7,000 years. Either way, the bell curve's verdict is clear: you should *never see one in a human lifetime*, let alone several.

Now confront that with reality. Equity markets have delivered 5-sigma-plus daily moves in 1987 (Black Monday was about a 20-sigma event under a constant-volatility normal model — a number so absurd it tells you the model, not the world, is broken), in 2008, in 2010's flash crash, in 2015, in 2018, and across 2020. That is several "once-in-7,000-years" events inside a *single career*. The reconciliation is excess kurtosis: real daily equity returns carry excess kurtosis on the order of 3 to 6, which fattens the tails so dramatically that what the normal model prices at one-in-millions is, in truth, a recurring hazard. As a concrete dollar illustration, on our \$1,000,000 book with \$10,000 daily one-sigma risk, a 5-sigma day is a \$50,000 loss — an amount the normal model says you can ignore but that the fat-tailed reality says you must budget for. The one-sentence intuition: kurtosis is the gap between "the bell curve says this is impossible" and "this happened again last quarter," and ignoring it is how risk models get people fired.

The practical consequence is everywhere. Value-at-Risk models that assume normality systematically understate crash risk; option markets price in the fat tails by charging more for far-out-of-the-money options than a normal model says they are worth (the **volatility smile**); and any strategy that is short volatility is, by construction, short kurtosis — it makes money when the tails behave and loses catastrophically when they do not.

### Estimating moments from data, and why the high ones are treacherous

In practice you do not know a distribution's true moments — you *estimate* them from a finite sample of past returns $x_1, \dots, x_n$. The estimators are the obvious sample analogues: the sample mean $\bar{x} = \frac{1}{n}\sum_i x_i$, the sample variance $s^2 = \frac{1}{n-1}\sum_i (x_i - \bar{x})^2$ (the $n-1$ instead of $n$ corrects a small downward bias), and the sample skewness and kurtosis built from the cubed and fourth-power deviations divided by $s^3$ and $s^4$. Each is a "weight the observations equally and average" version of the theoretical formula.

Here is the catch that trips up beginners and costs professionals real money: **the higher the moment, the harder it is to estimate, and the more it is dominated by a handful of extreme observations.** The mean is a stable, well-behaved estimate even from a few hundred points. The variance needs more data, because a single big move moves it noticeably. Skewness, built on *cubes*, and kurtosis, built on *fourth powers*, are dominated almost entirely by the largest deviations in the sample — and those are exactly the data points you have the fewest of. A kurtosis estimate from two calm years can read near the normal value of 3 simply because the crisis that would reveal the fat tail has not happened *in your sample yet*. The true tail risk is there; your estimator just cannot see it.

This is why a quant treats a low estimated kurtosis with suspicion rather than relief, especially for a strategy that *structurally* sells tails (options, credit, carry). The absence of a crash in the sample is not evidence of safety — it is often just evidence that the sample is too short. The honest move is to combine the empirical estimate with a model that *imposes* fat tails (a Student-t fit, an extreme-value model) so that your risk number reflects the disasters that have not yet shown up in your particular window of history. The companion [distributions cheat sheet](/blog/trading/quantitative-finance/distributions-cheat-sheet-quant-interviews) walks through which parametric distributions carry fat tails by construction, which is exactly the antidote to a too-short sample.

## The moment generating function and why moments stack

We have now built all four moments one at a time. There is an elegant object that generates *all* of them at once, and understanding it cheaply deepens your intuition for why the four moments are the natural way to describe a distribution. It is called the **moment generating function**, or MGF, defined as

$$ M_X(t) = E\!\left[e^{tX}\right]. $$

You do not need to compute MGFs by hand to trade, but here is why it is beautiful. The exponential $e^{tX}$ has a Taylor expansion $1 + tX + \tfrac{t^2}{2!}X^2 + \tfrac{t^3}{3!}X^3 + \cdots$. Take the expected value of both sides, using linearity, and you get

$$ M_X(t) = 1 + t\,E[X] + \frac{t^2}{2!}E[X^2] + \frac{t^3}{3!}E[X^3] + \cdots $$

Every term contains one of the **raw moments** $E[X^k]$ — the expected value of $X$ raised to the $k$-th power. The MGF packs all of them into a single function; differentiate it $k$ times, set $t = 0$, and out pops the $k$-th raw moment. The mean is the first raw moment, $E[X]$. The second raw moment $E[X^2]$ combines with the first to give variance. And so on up the ladder. This is the precise sense in which the moments "stack": they are the successive coefficients in one expansion, each one revealing a finer feature of the distribution.

![Tree splitting moments of X into raw, central, and standardized families with their members](/imgs/blogs/expectation-variance-moments-math-for-quants-5.png)

The tree above sorts out a vocabulary point that confuses beginners: there are three *families* of moments. **Raw moments** are $E[X^k]$ — powers of $X$ itself. **Central moments** subtract the mean first, $E[(X-\mu)^k]$ — variance is the central second moment. **Standardized moments** additionally divide by $\sigma$, $E[((X-\mu)/\sigma)^k]$ — skewness (third) and kurtosis (fourth) are standardized so they are unit-free and scale-free. The progression — raw, then demean to get central, then rescale to get standardized — is exactly why skewness and kurtosis are reported as pure numbers you can compare across any two assets.

Before moving on, here is the whole four-moment framework on one page — the question each moment answers, its formula, and the concrete trading decision it drives:

| Moment | Symbol / formula | Question it answers | Trading use |
| --- | --- | --- | --- |
| 1st — mean | $E[X]=\sum x\,p(x)$ | Where is the center? | Expected return, expected PnL |
| 2nd — variance | $E[X^2]-E[X]^2$ | How wide is the spread? | Risk budget, position sizing, vol |
| 3rd — skewness | $E[((X-\mu)/\sigma)^3]$ | Which tail is longer? | Asymmetric payoffs, option selling |
| 4th — kurtosis | $E[((X-\mu)/\sigma)^4]$ | How fat are the tails? | Crash risk, VaR, the vol smile |

A useful habit when you meet any new return stream: read the table top to bottom in that order. The mean tells you if there is an edge; the variance tells you how much it will bounce; the skewness tells you which direction the surprises favor; the kurtosis tells you how brutal the worst surprise can be. Skipping straight to the mean — the way most marketing material does — is reading only the first row of a four-row story.

A second reason the MGF matters in practice: it makes sums easy. For *independent* random variables, the MGF of a sum is the *product* of the MGFs. That single fact is the slick way to prove that sums of independent normals are normal, and it sits underneath the central limit theorem — the deep reason the bell curve shows up so often (and, crucially, the reason it shows up *less* faithfully in the tails than people assume). Not every distribution has an MGF — the fat-tailed ones we care most about, like the Student-t with low degrees of freedom, can have *infinite* high moments, which is itself a warning sign that the tails are dangerous. When a distribution's fourth moment is infinite, kurtosis is not even defined, and that is the mathematics telling you the tail risk is genuinely unbounded.

## The mean-variance trade-off and utility

Why do these moments organize how traders actually think? Because of a deep idea from economics: people are **risk-averse**. A guaranteed \$50 is worth more to most people than a coin flip between \$0 and \$100, even though both have the same \$50 expected value. The extra value of the sure thing is the price of avoiding variance. The mathematical home of this idea is **utility** — a function $U(W)$ that maps your wealth $W$ to how much you actually *value* it, and which curves over (concave) precisely because each extra dollar matters a little less than the one before.

Here is the link to moments. If you approximate expected utility with a Taylor expansion around your expected wealth, the leading terms are

$$ E[U(W)] \approx U(\mu) + \tfrac{1}{2}U''(\mu)\,\sigma^2 + \cdots $$

The first term rewards a high mean. The second term, because $U''$ is negative for a risk-averse person, *penalizes* variance. That is the entire **mean-variance trade-off** in one line: you want high $\mu$ and low $\sigma^2$, and a rational investor trades one against the other. Push the Taylor expansion one and two terms further and the *third* derivative multiplies skewness and the *fourth* multiplies kurtosis — meaning a careful investor also *prefers positive skew* (a chance of a big win) and *dislikes high kurtosis* (exposure to fat-tailed disasters). The four moments are not an arbitrary list. They are exactly the terms that show up, in order, when you ask how a rational, risk-averse person values an uncertain payoff. This is why the mean-variance framework — and its refinements that add skew and kurtosis preferences — underpins modern portfolio construction.

#### Worked example: choosing between two strategies with the same mean

You must allocate \$100,000 to one of two strategies for a year. Both have the same expected return of 10% — both turn your \$100,000 into an expected \$110,000. Strategy A has annual volatility of 8% and excess kurtosis near 0 (well-behaved). Strategy B has annual volatility of 8% *too* — identical — but excess kurtosis of 5 and strongly negative skew (it sells volatility). On the mean alone, and even on the mean *and* variance, the two are indistinguishable: same \$10,000 expected gain, same \$8,000 one-sigma swing.

The third and fourth moments break the tie. Strategy A's worst plausible year, a 2-sigma down move, costs you about $2 \times 8\% \times \$100{,}000 = \$16{,}000$, and bigger losses are genuinely rare. Strategy B, with fat tails and negative skew, has a far higher chance of a 4- or 5-sigma loss: a 5-sigma down year would cost $5 \times 8\% \times \$100{,}000 = \$40{,}000$, and B's kurtosis of 5 makes that outcome *orders of magnitude* more likely than A's near-normal tails allow. A risk-averse investor — anyone whose utility curve bends — should prefer A, even though a mean-variance-only screen calls them equal. The one-sentence intuition: when two strategies tie on mean and variance, the skew and kurtosis decide which one survives a bad year, and the volatility number alone will never tell you that.

## Common misconceptions

**"A positive expected value means I will make money."** No — it means you will make money *on average over many independent repetitions*. A single positive-EV bet can easily lose, and a string of them can lose for a long time if the variance is high. Expected value is a statement about the long run, and the long run can be very long. The \$60-edge, \$171-sigma bet from earlier is positive-EV and still loses on more than a third of single attempts.

**"Low volatility means low risk."** Volatility measures the *symmetric* spread; it is blind to skewness and kurtosis. A strategy that sells deep out-of-the-money options can post tiny, beautiful volatility for years while accumulating a catastrophic negative-skew, fat-tail exposure. The volatility number is exactly the metric that fails to warn you about the steamroller. Low measured volatility can hide the highest tail risk.

**"Returns are normally distributed."** They are not, and the deviation is concentrated exactly where it hurts: the tails. The center of the daily-return histogram does look roughly bell-shaped, which lulls people into using the normal everywhere. But real returns have excess kurtosis of 3 to 6 and usually negative skew, so the bell curve underprices both the frequency and the size of crashes. The normal distribution is a decent description of a calm Tuesday and a terrible description of a crisis Monday.

**"More data always makes my estimates reliable."** Variance and especially the higher moments are *hard* to estimate. Skewness and kurtosis are dominated by the rarest, most extreme observations — precisely the data points you have the fewest of. A kurtosis estimate from two calm years will badly understate the true tail risk, because the years that would reveal it have not happened yet in your sample. The higher the moment, the more data (and the more *extreme* data) you need to pin it down.

**"Skewness and the mean are the same kind of 'direction' information."** They are not. The mean tells you the center; skewness tells you which *tail* is longer. A strategy can have a positive mean and strongly negative skew at the same time — that is exactly what selling options looks like: it makes money on average (positive mean) while hiding a long loss tail (negative skew). The two answer different questions.

**"Annualized volatility just means multiplying daily vol by 252."** No — it means multiplying by the *square root* of 252, about 15.87. Variance scales linearly with time; volatility, being the square root of variance, scales with the square root of time. Confusing the two overstates annual risk by a factor of roughly 16. The square-root-of-time rule is the correct conversion, and it is one of the most common beginner errors.

## How it shows up in real markets

### 1. Long-Term Capital Management, 1998

LTCM was run by Nobel laureates and posted gorgeous, low-volatility returns for years by making highly leveraged convergence bets — a classic negative-skew, fat-tail profile. The strategy collected small, steady gains (picking up pennies) while carrying enormous hidden exposure to a rare correlated shock. In August and September 1998, Russia's default triggered exactly that shock; correlations the model assumed were modest snapped to near 1, and losses arrived as a multi-sigma event the normal-based risk models had rated essentially impossible. The fund lost roughly \$4.6 billion in months and required a Fed-orchestrated bailout. The lesson is the whole post: low measured volatility hid extreme negative skew and kurtosis, and the moments the model ignored were the ones that mattered.

### 2. Black Monday, October 19, 1987

The S&P 500 fell about 20% in a single day. Under a constant-volatility normal model calibrated to the prior period, a one-day move of that size is roughly a 20-sigma-plus event — a probability so small it has more zeros than there are atoms in a meaningful sense. The fact that it *happened* is the single cleanest proof that daily equity returns are not normal and carry massive excess kurtosis. After 1987, option markets permanently re-priced the tails: the **volatility smile** (out-of-the-money puts trading at higher implied volatility than at-the-money options) appeared and never left, because the market now charges for the fat left tail the normal model denies exists.

### 3. The 2008 financial crisis and Gaussian risk models

Many bank risk systems used Value-at-Risk built on normal (Gaussian) assumptions. Through 2008 these systems printed "this loss should occur once every 10,000 days" warnings on multiple *consecutive* days — a logical impossibility that exposed the model, not the market, as broken. The normal assumption suppressed both the kurtosis (fat tails) and the way correlations fatten in a crisis. Books that looked safe on a volatility screen were carrying enormous fourth-moment risk, and when the tail arrived it arrived for everyone at once.

### 4. The short-volatility blowup of February 2018 ("Volmageddon")

A popular class of exchange-traded products let investors *short* market volatility — a textbook negative-skew, high-kurtosis trade that earns a small carry most days. For two calm years it worked beautifully and posted low realized volatility. On February 5, 2018, a single sharp spike in the VIX volatility index caused these products to lose most of their value overnight; one of the largest, XIV, fell about 96% and was liquidated. Investors who had read only the mean and the (low) volatility saw none of it coming; the skew and kurtosis had been screaming the entire time.

### 5. The everyday volatility smile in options pricing

You do not need a crisis to see fat tails priced in markets — they are quoted continuously. The Black-Scholes model assumes normal log-returns, which would imply a single flat implied volatility across all strikes. Instead, every liquid options market shows a **smile** or **skew**: far-out-of-the-money options trade at higher implied volatilities, because traders know the true distribution has fatter tails (kurtosis) and, for equity indices, a fatter *left* tail (negative skew). The shape of the smile is, quite literally, the market's collective estimate of the third and fourth moments of future returns, updated tick by tick.

### 6. Why Sharpe ratios mislead on negative-skew strategies

The Sharpe ratio — mean excess return divided by volatility — is the industry's headline performance number, and it is built from only the first two moments. A strategy that sells options or shorts volatility can post a Sharpe of 2 or 3 for years precisely because its negative skew and fat tails are invisible to a measure that uses only mean and variance. Sophisticated allocators therefore look past the Sharpe to the skewness and kurtosis of the return stream, and treat a too-good-to-be-true Sharpe with a smooth equity curve as a red flag for hidden tail risk rather than a sign of skill.

### 7. Bet sizing: where mean and variance meet

The clearest place the first two moments combine into a single decision is *position sizing*. Suppose you have found a genuine edge — a positive expected return — on a repeatable bet. How much of your \$1,000,000 should you stake on each occurrence? Stake too little and you leave the edge on the table; stake too much and the variance eventually delivers a losing streak that ruins you before the edge can pay off. The mathematics that balances these is the Kelly criterion, and at its heart it trades the *mean* edge against the *variance* drag: the growth-optimal fraction is, to a good approximation, the expected excess return divided by the variance. Bet more when the edge is large or the variance is small; bet less when the edge is thin or the swings are violent. A trader who sizes on the mean alone — "the edge is positive, so go big" — ignores the variance term and is, statistically, choosing the path to a blown-up account. The full treatment lives in the dedicated Kelly and sequential-betting material, but the lesson is that even the most basic sizing decision is a two-moment calculation, never a one-moment one.

## When this matters to you

If you ever evaluate an investment, a strategy, or even a business bet, the four moments are the questions to ask in order. *What is the expected return?* (mean) — necessary, never sufficient. *How much will it swing?* (variance and volatility) — your risk budget. *Is the payoff lopsided?* (skewness) — are you the lottery buyer or the insurance seller, and do you know which? *How exposed am I to rare disasters?* (kurtosis) — the question that decides whether you survive a bad year. A surprisingly large fraction of financial blowups, from individual accounts to multi-billion-dollar funds, come down to someone who read the first two moments and ignored the last two.

This is educational material, not investment advice — it is about how to *read* risk, not what to buy. But the framework travels: any time you are tempted by a smooth track record and a high average, ask what the third and fourth moments look like, because a steady stream of small gains is exactly the visual signature of a payoff that is quietly storing up a large rare loss.

It also travels beyond markets. A startup's expected return is enormous, its variance larger still, and its payoff is wildly positively skewed — most fail, a few return everything, which is why venture investors deliberately hold many bets to harvest the long right tail. An insurance company is the mirror image: it earns a steady premium (a small mean) against a fat-tailed, negatively skewed liability, and survives only by diversifying across enough independent policies that the variance shrinks faster than the premium. Even a salary-versus-equity job offer is a mean-variance choice: the salary is high mean-certainty, the equity is lower expected value but with a long positive-skew tail. Once you have the four moments in hand, you start seeing them in every decision that involves an uncertain payoff, which is to say almost every decision that matters financially.

To go deeper, three companion posts on this blog build directly on these foundations. The [distributions cheat sheet for quant interviews](/blog/trading/quantitative-finance/distributions-cheat-sheet-quant-interviews) catalogs the specific distributions (normal, lognormal, Student-t, Poisson) whose moments we have been discussing, including which ones have fat tails by construction. The [expected value techniques guide](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews) drills the linearity-of-expectation tricks that make computing $E[X]$ fast under pressure. And the [covariance and correlation pitfalls guide](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews) extends the second moment from one asset to many, which is the gateway to the covariance-matrix and portfolio-optimization posts in this series. Read in that order, they take you from "what are the four numbers that describe a single return" to "how do I measure and shape the risk of an entire book."

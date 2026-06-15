---
title: "ARCH and GARCH volatility models: forecasting the storm before it hits"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Build the math of volatility clustering from zero — why big market moves arrive in bunches, how the ARCH and GARCH recursions turn yesterday's return into tomorrow's risk forecast, what persistence and long-run variance mean, how the models are fit by maximum likelihood, and how leverage variants like GJR-GARCH and EGARCH capture the fact that crashes scare markets more than rallies, all with worked dollar examples."
tags:
  [
    "garch",
    "arch",
    "volatility",
    "volatility-clustering",
    "value-at-risk",
    "risk-management",
    "egarch",
    "gjr-garch",
    "leverage-effect",
    "maximum-likelihood",
    "quantitative-finance"
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — Market volatility is not constant; it clusters in time, and the ARCH/GARCH family is the math that turns yesterday's market into a credible forecast of tomorrow's risk.
>
> - **Volatility clustering** is the stylized fact that big moves follow big moves and calm follows calm; a constant-volatility model is wrong about exactly the days that matter, and ARCH/GARCH exist to fix it.
> - **GARCH(1,1)** — $\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2$ — is the workhorse: tomorrow's variance is a blend of a constant floor, yesterday's surprise, and yesterday's variance.
> - **Persistence** is $\alpha + \beta$; when it is close to $1$ (it usually is, around $0.95$–$0.99$ for equities), volatility shocks decay slowly and the **long-run variance** is $\omega/(1-\alpha-\beta)$.
> - You fit these models by **maximum likelihood**, forecast volatility many days ahead as it mean-reverts toward the long-run level, and turn that into a dollar **Value at Risk** and into option prices.
> - **Leverage** variants (GJR-GARCH, EGARCH) capture the asymmetry that a $-5\%$ day raises tomorrow's volatility more than a $+5\%$ day — the one fact a symmetric model gets dangerously wrong. The number to remember: at $\alpha+\beta=0.98$, a volatility shock has a half-life of about $34$ days.

On Monday, October 19, 1987, the US stock market fell $20.5\%$ in a single session — the worst day in its history, a move so large that under the standard risk model of the time it should not have happened in the entire age of the universe, several times over. But here is the part that matters for this article: the days that followed were not calm. The next day swung violently again. The week after was a storm. Volatility, having exploded, stayed exploded for weeks before slowly settling. If you had been managing risk that October, the single most useful thing you could have known was not "today was bad" — you knew that — but "*tomorrow will probably be bad too, and so will the day after.*" That predictable persistence of turbulence is the entire subject of this post, and it has a name and a precise mathematics.

![A timeline showing a calm market interrupted by a shock that is followed by a cluster of big days before fading back to calm](/imgs/blogs/arch-garch-volatility-math-for-quants-1.png)

The timeline above is the mental model for everything that follows. Markets do not deliver risk in an even drizzle; they deliver it in storms. A long calm stretch gets interrupted by a shock, the shock is followed by a *cluster* of more big days, and then volatility slowly fades back toward calm. This phenomenon — **volatility clustering** — is one of the most robust facts in all of finance, and the **ARCH** and **GARCH** models are the tools quants built specifically to capture it and forecast it. By the end of this article you will be able to write the recursion from memory, fit it, forecast with it, and turn its output into a dollar figure on a real book.

One honest aside before we start. Nothing here is investment advice. We are explaining how a family of forecasting models works — where each is solid, and exactly where each one quietly fails. Knowing the failure modes of your risk model is, on a bad day, worth more than the model itself.

## Foundations: returns, variance, and the vol yardstick

Before we can model volatility, we need to be precise about a handful of terms a careful beginner may never have seen formally: what a **return** is, what **variance** and **volatility** mean, what a **shock** is, and why the *square* of a return is the quantity that does all the work. Each is simpler than its name, and each connects directly to money.

### Returns: the raw material

A **return** is the percentage change in the price of something over a period. If a stock closes at \$100 today and \$102 tomorrow, the one-day return is $+2\%$. If it falls to \$97, the return is $-3\%$. In practice quants usually use the **log return**, $r_t = \ln(P_t / P_{t-1})$, where $\ln$ is the natural logarithm; for small daily moves a log return and a simple percentage return are almost identical (a $+2\%$ move is a log return of about $+1.98\%$), and log returns have the convenient property that they add up across time. Throughout this post, when we say "return" we mean a daily log return expressed as a decimal, so $r_t = -0.05$ means the market fell about $5\%$ that day.

The crucial empirical fact about daily returns is that, on average, they are *tiny and basically unpredictable*. The expected return on a stock for a single day is on the order of a few hundredths of a percent — far too small and too noisy to forecast reliably. So in volatility modeling we make a simplifying move that is very close to the truth: we assume the **mean** daily return is zero (or a small constant we subtract off first). What is left, the part that swings around, is the **shock**.

### Shocks, variance, and volatility

The **shock** at time $t$, written $\varepsilon_t$, is the return with its mean removed — the unexpected part of today's move. If the average daily return is essentially zero, then $\varepsilon_t \approx r_t$: the shock *is* the return. A shock can be positive (a surprise gain) or negative (a surprise loss).

**Variance** measures how spread out these shocks are. Formally, the variance of the daily shock is $\sigma^2 = \mathbb{E}[\varepsilon_t^2]$ — the average of the *squared* shocks. We square them for two reasons. First, squaring removes the sign, so a $-3\%$ day and a $+3\%$ day contribute the same amount of "spread"; variance cares about *size*, not direction. Second, squaring penalizes large moves heavily — a $6\%$ move contributes four times as much variance as a $3\%$ move — which is exactly right, because in risk a big move is much more than twice as dangerous as a medium one.

**Volatility**, written $\sigma$, is just the square root of variance — it is back in the original percentage units, so it is the more intuitive of the two. If a stock's daily volatility is $\sigma = 0.02$, that means $2\%$ — a "typical" daily move is about $2\%$ in size. Volatility is the standard yardstick of risk on every trading desk on earth. The whole point of ARCH and GARCH is that this yardstick is *not a fixed number*. It changes from day to day, and it changes in a forecastable way.

### Why squared returns are the whole game

Here is the single most important observation in this entire field, and it is one you can verify yourself with a spreadsheet of daily returns. The returns themselves, $r_t$, show almost no pattern from one day to the next — today's return tells you essentially nothing about tomorrow's *direction*. But the *squared* returns, $r_t^2$, are strongly **autocorrelated**: a big $r_t^2$ today predicts a big $r_{t+1}^2$ tomorrow. (*Autocorrelation* just means a series is correlated with its own past.) In plain English: you cannot predict *which way* the market will move, but you absolutely can predict *how much* it will move, because the size of moves comes in clusters. ARCH and GARCH are, at heart, just regressions of squared returns on their own past. That is the engine. Everything else is detail.

## Volatility clustering: the stylized fact

A **stylized fact** in finance is a pattern so consistent across markets, eras, and asset classes that any serious model is expected to reproduce it. Volatility clustering is the king of stylized facts. Robert Engle, who won the 2003 Nobel Prize in Economics largely for inventing ARCH, described the core insight plainly: "large changes tend to be followed by large changes, of either sign, and small changes tend to be followed by small changes."

![A before-and-after contrast of a constant-volatility model versus a clustered-volatility model](/imgs/blogs/arch-garch-volatility-math-for-quants-5.png)

The contrast above is the reason the whole field exists. On the left is the world a **constant-volatility model** believes in: the same risk every day, big days scattered randomly and evenly, no memory of what just happened. On the right is the real world a **clustered-volatility model** captures: risk that rises sharply after a shock, big days that bunch together, and a model that remembers recent turbulence and so flags a crisis day before the second shoe drops. Consider the case of an equity index in a normal year: it spends most of the year with daily moves under $1\%$, and then for a few weeks around an earnings shock or a macro surprise it moves $3\%$ or $4\%$ a day repeatedly. A constant-volatility model assigns the same probability to a $4\%$ day in July as in the eye of the storm, which is absurd. The intuition the clustered model encodes is that *recent* turbulence is informative about *imminent* turbulence — and it is.

### Why clustering happens

You do not need a mechanism to *use* the clustering — the pattern is enough to forecast — but it helps to know why it is there, because it tells you when to trust the model. A few real forces drive it. Information arrives in clumps: a central bank meeting, an earnings season, a geopolitical crisis all release a burst of news, and prices move repeatedly as the market digests it. Leverage and forced selling create feedback: a sharp drop triggers margin calls and risk-limit breaches, which force more selling, which causes more drops — turbulence feeding on itself. And uncertainty itself is persistent: when nobody knows what an asset is worth, that confusion does not resolve in a single day. All three forces make volatility *sticky*, and stickiness is exactly what a recursive model can exploit.

### The job of the model

So the job is narrow and concrete. We want a formula that takes what we have observed up to today — the recent returns and our recent volatility estimates — and outputs a single number: a forecast of tomorrow's variance, $\sigma_t^2$. From that one number, everything a risk desk needs flows.

![A pipeline from daily returns through squared shocks and the GARCH recursion to a variance forecast and a dollar VaR limit](/imgs/blogs/arch-garch-volatility-math-for-quants-2.png)

The pipeline above is the production flow. Daily returns come in; we square the shocks to get the raw "how big was today" signal; the GARCH recursion blends that with its running memory of variance; out comes a variance forecast for tomorrow; and that forecast gets scaled into a dollar Value at Risk limit that the desk sets its positions against. The same forecast also feeds option-pricing models, since an option's value is mostly a bet on future volatility. Let us build the recursion that sits in the middle of that pipeline, starting from the simplest version.

## ARCH: the first model of changing volatility

In 1982, Robert Engle published the **ARCH** model — **A**uto**R**egressive **C**onditional **H**eteroskedasticity. The name is intimidating; the idea is not. Let us decode it word by word, because the name is actually a precise description.

- **Heteroskedasticity** means "changing variance" (from Greek: *hetero* = different, *skedasis* = dispersion). The opposite, *homoskedasticity*, means constant variance. The whole premise is that variance is not constant.
- **Conditional** means "given what we know so far." We are not modeling some fixed, unconditional variance; we are modeling the variance of tomorrow *conditional on* today's information. This is the key distinction: a return series can have a stable *long-run* variance while its *day-by-day, conditional* variance swings wildly.
- **AutoRegressive** means today's conditional variance is built from its own past — specifically from past squared shocks.

### What ARCH says

The ARCH model of order $q$, written ARCH($q$), says:

$$\sigma_t^2 = \omega + \alpha_1 \varepsilon_{t-1}^2 + \alpha_2 \varepsilon_{t-2}^2 + \cdots + \alpha_q \varepsilon_{t-q}^2$$

Reading this left to right: tomorrow's variance $\sigma_t^2$ equals a baseline constant $\omega$ (the Greek letter *omega*, pronounced "oh-MAY-ga") plus a weighted sum of the most recent $q$ squared shocks. The weights $\alpha_1, \dots, \alpha_q$ (*alpha*) say how much each recent shock matters. The shock $\varepsilon_t$ itself is then modeled as $\varepsilon_t = \sigma_t z_t$, where $z_t$ is a standard random draw — mean zero, variance one — usually taken to be a standard normal. In words: each day's actual move is its forecast volatility $\sigma_t$ multiplied by a fresh random shock $z_t$.

To keep variance positive and the model sensible, we need $\omega > 0$ and every $\alpha_i \ge 0$. The constant $\omega$ is the floor — the variance you would forecast if every recent shock had been exactly zero. The $\alpha$ terms add risk in proportion to how violent the recent past was.

### Why ARCH captures clustering

Look at what the formula does on a calm-then-violent transition. After a string of small days, all the $\varepsilon_{t-i}^2$ terms are tiny, so $\sigma_t^2 \approx \omega$ — low forecast variance, calm predicted. Then a big shock hits: one of the $\varepsilon_{t-i}^2$ terms becomes large, $\sigma_t^2$ jumps up, and the model forecasts high variance for the next few days. As that big shock ages out of the lag window, the forecast falls back toward $\omega$. That is volatility clustering, produced mechanically by a one-line formula. Engle had taken a vague stylized fact and made it a forecastable, testable model — which is why it won a Nobel.

### The problem with pure ARCH

ARCH works, but it has an awkward practical flaw: to capture the *slow* decay of real volatility shocks, you need a *lot* of lags. Equity volatility shocks fade over weeks or months, so to fit that with ARCH you might need $q = 20$ or more, each with its own $\alpha$ parameter to estimate. That is a lot of parameters, they are hard to estimate stably, and they tend to fight each other. The field needed a way to get long memory without a long list of parameters. That is precisely what GARCH delivered.

## The GARCH(1,1) recursion

In 1986, Tim Bollerslev — a student of Engle's — added a single, beautiful idea. Instead of summing up many past squared shocks to get long memory, let today's variance depend on *yesterday's variance*. Because yesterday's variance already contains the day-before's variance, which contains the day before that, a single lag of variance quietly encodes the entire history with exponentially decaying weights. That is the **G** in **GARCH** — **G**eneralized ARCH.

The flagship specification, used more than any other model in this whole family, is **GARCH(1,1)**:

$$\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2$$

The "(1,1)" means one lag of the squared shock and one lag of the variance. Let us read every symbol:

- $\sigma_t^2$ — the **conditional variance** for day $t$, i.e., our forecast of how violent today will be, made at the end of yesterday.
- $\omega$ — the constant **floor**. A small positive number; it anchors the long-run level.
- $\alpha$ — the **reaction** coefficient (*alpha*). It controls how much yesterday's *surprise* (the squared shock $\varepsilon_{t-1}^2$) pushes today's variance up. High $\alpha$ = jumpy, twitchy volatility.
- $\beta$ — the **persistence** coefficient (*beta*). It controls how much of yesterday's variance *carries over* to today. High $\beta$ = smooth, slowly-decaying volatility.

![A stack showing tomorrow's variance built from a constant floor, alpha times the squared shock, and beta times the old variance](/imgs/blogs/arch-garch-volatility-math-for-quants-3.png)

The stack above is the whole model in one picture: tomorrow's variance is three layers added together — a constant floor $\omega$, the reaction term $\alpha\,\varepsilon_{t-1}^2$ (how much yesterday surprised us), and the persistence term $\beta\,\sigma_{t-1}^2$ (how nervous we already were). Under the hood, this is a weighted blend: $\alpha$ and $\beta$ are the weights on "new information" versus "old belief." A nervous market with a big shock yesterday gets a high variance; a calm market with a quiet yesterday relaxes toward the floor.

For the variance to stay positive we again need $\omega > 0$, $\alpha \ge 0$, $\beta \ge 0$. And for the model to be **stationary** — to have a stable long-run variance rather than exploding to infinity — we need $\alpha + \beta < 1$. That sum, $\alpha + \beta$, is so important it has its own name, and we will spend a whole section on it.

### Why one variance lag equals infinite shock lags

It is worth seeing the trick explicitly, because it is genuinely elegant. Take the GARCH(1,1) equation and substitute the formula for $\sigma_{t-1}^2$ into it, then for $\sigma_{t-2}^2$, and keep going:

$$\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\big(\omega + \alpha\,\varepsilon_{t-2}^2 + \beta\,\sigma_{t-2}^2\big) = \cdots = \frac{\omega}{1-\beta} + \alpha\sum_{k=1}^{\infty}\beta^{k-1}\varepsilon_{t-k}^2$$

Reading the final form: GARCH(1,1) is *secretly an ARCH model with infinitely many lags*, where the weight on the shock $k$ days ago is $\alpha\beta^{k-1}$ — it decays geometrically. A shock one day ago gets weight $\alpha$; two days ago, $\alpha\beta$; three days ago, $\alpha\beta^2$; and so on. With $\beta = 0.9$, a shock from $10$ days ago still gets about $35\%$ of the weight of yesterday's shock. So with just three parameters — $\omega$, $\alpha$, $\beta$ — GARCH(1,1) captures the long, slow decay of volatility that pure ARCH needed twenty parameters to mimic. That parsimony is why GARCH(1,1) became the workhorse and ARCH became the historical footnote.

#### Worked example: a one-step volatility forecast and the dollar VaR

Let us make it concrete with numbers. Suppose we have estimated a GARCH(1,1) on a stock index and gotten parameters $\omega = 0.000002$, $\alpha = 0.08$, $\beta = 0.91$. (These are typical equity values; we will sanity-check them later.) Yesterday two things happened: the market fell $3\%$, so the squared shock was $\varepsilon_{t-1}^2 = (-0.03)^2 = 0.0009$; and our model's variance estimate for yesterday was $\sigma_{t-1}^2 = 0.0004$, i.e., a yesterday-volatility of $\sqrt{0.0004} = 0.02 = 2\%$.

Plug in:

$$\sigma_t^2 = 0.000002 + 0.08 \times 0.0009 + 0.91 \times 0.0004 = 0.000002 + 0.000072 + 0.000364 = 0.000438$$

So today's forecast variance is $0.000438$, and today's forecast volatility is $\sqrt{0.000438} \approx 0.02093 = 2.09\%$. Notice what happened: yesterday's $3\%$ drop — bigger than the typical $2\%$ — pushed the forecast *up* from $2.0\%$ to $2.09\%$. The model heard the alarm.

Now turn that into money. Suppose you run a \$1,000,000 book tracking this index. The **one-day $99\%$ Value at Risk** is the loss you expect to exceed only $1\%$ of the time. Under the common normal assumption, the $99\%$ quantile sits $2.326$ standard deviations into the tail, so:

$$\text{VaR}_{99\%} = 2.326 \times \sigma_t \times \text{book} = 2.326 \times 0.02093 \times \$1{,}000{,}000 \approx \$48{,}680$$

The day before yesterday, with a $2.0\%$ vol forecast, that same VaR was $2.326 \times 0.02 \times \$1{,}000{,}000 = \$46{,}520$. So a single $3\%$ down day raised your stated one-day risk by about \$2,160 — and a single big shock during a real crisis would raise it far more. The intuition: GARCH converts "the market got rougher yesterday" directly into "your risk limit must tighten today," in dollars, automatically.

## Persistence, long-run variance, and mean reversion

We have met $\alpha + \beta$. Now we give it the respect it deserves, because it is the single number that tells you almost everything about how a fitted GARCH model behaves.

### Persistence: how long a shock lives

The sum $\alpha + \beta$ is called the **persistence** of the model. Here is why. Think of variance as having a long-run resting level (we will compute it in a moment). Each day, the model's variance moves a fraction $\alpha + \beta$ of the way back from the resting level... no — more precisely, the *deviation* of variance from its long-run level shrinks by a factor of $(\alpha + \beta)$ each day. If $\alpha + \beta = 0.5$, half the deviation evaporates daily — shocks die fast. If $\alpha + \beta = 0.98$, only $2\%$ of the deviation evaporates daily — shocks linger for weeks. Real equity indices almost always have persistence between $0.95$ and $0.99$. Volatility, once disturbed, comes back to normal *slowly*. That is the mathematical fingerprint of clustering.

When $\alpha + \beta = 1$ exactly, the model is called **IGARCH** (Integrated GARCH): shocks never fully die, and there is no finite long-run variance. When $\alpha + \beta > 1$, the model is explosive and unusable — variance forecasts run off to infinity. So in practice we always estimate models with $\alpha + \beta$ just below $1$.

### The long-run variance

If the model is stationary ($\alpha + \beta < 1$), it has a well-defined **unconditional** (long-run) variance — the average level volatility returns to when left alone. We find it by asking: what variance level $\bar\sigma^2$ would reproduce itself? Set $\sigma_t^2 = \sigma_{t-1}^2 = \bar\sigma^2$ and, since on average $\mathbb{E}[\varepsilon_{t-1}^2] = \sigma_{t-1}^2$, replace the squared shock with $\bar\sigma^2$ too:

$$\bar\sigma^2 = \omega + \alpha\,\bar\sigma^2 + \beta\,\bar\sigma^2 \quad\Longrightarrow\quad \bar\sigma^2 = \frac{\omega}{1 - \alpha - \beta}$$

That clean little formula — **long-run variance equals omega divided by one minus persistence** — is one of the most useful results in the whole field. It tells you the gravitational center that all your forecasts are pulled toward. And it reveals why $\omega$ alone is meaningless to eyeball: a tiny $\omega$ divided by a tiny $(1 - \alpha - \beta)$ can be a perfectly normal volatility level.

#### Worked example: long-run vol and the half-life of a shock

Take our fitted parameters from before: $\omega = 0.000002$, $\alpha = 0.08$, $\beta = 0.91$. Persistence is $\alpha + \beta = 0.99$. The long-run variance is:

$$\bar\sigma^2 = \frac{0.000002}{1 - 0.99} = \frac{0.000002}{0.01} = 0.0002$$

So the long-run *volatility* is $\bar\sigma = \sqrt{0.0002} \approx 0.01414 = 1.41\%$ per day. To put that in annual terms — the units most people quote volatility in — we multiply by $\sqrt{252}$ (there are about $252$ trading days a year, and variance scales with time while volatility scales with the square root of time): $0.01414 \times \sqrt{252} \approx 0.224$, or about $22.4\%$ annualized. That is a textbook-normal equity volatility, which tells us our parameters are realistic.

Now the **half-life** of a volatility shock — how many days until a shock has decayed halfway back to normal. Since the deviation shrinks by a factor $(\alpha+\beta)$ each day, after $h$ days it is $(\alpha+\beta)^h$ of its original size. Set that equal to one half and solve:

$$(\alpha+\beta)^h = 0.5 \quad\Longrightarrow\quad h = \frac{\ln 0.5}{\ln(\alpha+\beta)} = \frac{\ln 0.5}{\ln 0.99} \approx \frac{-0.6931}{-0.01005} \approx 69 \text{ days}$$

So with persistence $0.99$, a volatility spike takes about $69$ trading days — roughly three calendar months — to decay halfway. At the more moderate $\alpha+\beta = 0.98$ promised in the TL;DR, the half-life is $\ln 0.5 / \ln 0.98 \approx 34$ days. The practical reading: if a crisis doubles your forecast volatility, do not expect risk to be back to normal next week. On a \$1,000,000 book, a doubled vol means a doubled VaR — your \$48,680 daily risk becomes roughly \$97,000 — and that elevated risk *persists for months*. The intuition: persistence is the mathematics of "this is going to be a rough quarter," and it is why risk managers cut exposure for far longer than the original shock seems to warrant.

### Mean reversion, stated simply

Put persistence and long-run variance together and you get the headline behavior of GARCH: **mean-reverting volatility**. After a shock, forecast volatility is high; left undisturbed, it drifts back down toward $\bar\sigma$ a fraction $(\alpha+\beta)$ per day. After a calm stretch that has pulled volatility below $\bar\sigma$, the floor $\omega$ nudges it back up. Volatility wanders, but it is tethered. This is what lets GARCH forecast not just *tomorrow* but a whole *path* of future volatility — which is exactly what you need for risk over a $10$-day horizon or for pricing a $90$-day option.

## Forecasting volatility: one step and many

A one-day-ahead forecast is the easy case — it is just the GARCH equation, since at the end of today we know today's shock and today's variance. The interesting and practically vital case is the **multi-step forecast**: what is my forecast of variance $2$, $5$, or $20$ days from now, made today?

### The multi-step forecast formula

The trick is that for any day past tomorrow, we do not yet know the shock — but on average a squared shock equals its variance, $\mathbb{E}[\varepsilon_t^2] = \sigma_t^2$. So in expectation the two terms $\alpha\,\varepsilon^2$ and $\beta\,\sigma^2$ collapse into a single $(\alpha+\beta)\,\sigma^2$ term, and the forecast becomes a clean recursion. Starting from tomorrow's known forecast $\sigma_{t+1}^2$, the forecast $k$ days ahead is:

$$\mathbb{E}_t[\sigma_{t+k}^2] = \bar\sigma^2 + (\alpha+\beta)^{k-1}\big(\sigma_{t+1}^2 - \bar\sigma^2\big)$$

Read this carefully, because it is the most useful forecasting formula in the post. The forecast $k$ days out equals the long-run variance $\bar\sigma^2$ plus the *current gap* between tomorrow's forecast and the long-run level, shrunk by the persistence raised to the power $(k-1)$. As $k$ grows, $(\alpha+\beta)^{k-1} \to 0$, so the forecast smoothly decays from wherever it starts toward the long-run level. This is mean reversion written as a formula. It is identical in shape to the cooling of a hot cup of coffee toward room temperature — the gap closes by a constant fraction each step.

#### Worked example: a spike mean-reverting over twenty days

Suppose a shock has spiked tomorrow's forecast volatility to $\sigma_{t+1} = 4\%$ per day — double the long-run level. Using our long-run variance $\bar\sigma^2 = 0.0002$ (so $\bar\sigma = 1.41\%$) and persistence $\alpha+\beta = 0.99$, let us trace the forecast forward. Tomorrow's variance is $\sigma_{t+1}^2 = 0.04^2 = 0.0016$, and the gap to long-run is $0.0016 - 0.0002 = 0.0014$.

- **Day 1 ahead:** variance $0.0016$, vol $\sqrt{0.0016} = 4.00\%$.
- **Day 5 ahead:** $0.0002 + 0.99^{4}\times 0.0014 = 0.0002 + 0.9606\times 0.0014 = 0.001545$, vol $\approx 3.93\%$.
- **Day 20 ahead:** $0.0002 + 0.99^{19}\times 0.0014 = 0.0002 + 0.8262\times 0.0014 = 0.001357$, vol $\approx 3.68\%$.
- **Day 60 ahead:** $0.0002 + 0.99^{59}\times 0.0014 = 0.0002 + 0.5519\times 0.0014 = 0.000973$, vol $\approx 3.12\%$.

With persistence at $0.99$, the decay is *slow* — three months out, the model still forecasts $3.12\%$ vol against a $1.41\%$ baseline. This is the same half-life story as before, now drawn as a glide path. For a desk pricing a $20$-day option or computing a $20$-day VaR, you do not use tomorrow's $4\%$; you use the *average* variance over the whole horizon, which the formula lets you sum up. The intuition: after a spike, GARCH tells you not just that risk is high today but exactly how slowly it will normalize — and on a long horizon that glide path, not the spike itself, is what sets your hedge.

### Aggregating the path: term volatility

For risk and pricing over $T$ days you need the *total* expected variance over the horizon, which is the sum of the daily forecasts: $\sum_{k=1}^{T}\mathbb{E}_t[\sigma_{t+k}^2]$. The $T$-day volatility is the square root of that sum. After a spike, this term volatility is *higher* than the naive $\sqrt{T}$ scaling of today's vol would suggest, because the early days of the horizon are still elevated; after a calm stretch, it is lower. This is GARCH's quiet rebuke to the most common shortcut in risk: scaling one-day vol by $\sqrt{T}$ assumes constant volatility, and constant volatility is the one thing we have established does not exist.

There is a closed form for the sum, which is handy to keep in your back pocket. Because each term is $\bar\sigma^2 + (\alpha+\beta)^{k-1}(\sigma_{t+1}^2 - \bar\sigma^2)$, the total over $T$ days is the long-run part $T\,\bar\sigma^2$ plus a geometric series in $(\alpha+\beta)$:

$$\sum_{k=1}^{T}\mathbb{E}_t[\sigma_{t+k}^2] = T\,\bar\sigma^2 + \big(\sigma_{t+1}^2 - \bar\sigma^2\big)\frac{1 - (\alpha+\beta)^{T}}{1 - (\alpha+\beta)}$$

The first piece is what a constant-volatility model would give; the second piece is the *correction* GARCH adds for the fact that the early days of the horizon are still elevated (or depressed). That correction is exactly the value a naive desk leaves on the table.

#### Worked example: a ten-day VaR right after a shock

Regulators historically required banks to hold capital against a $10$-day, $99\%$ VaR — the loss exceeded only $1\%$ of the time over a two-trading-week horizon. Take our usual parameters ($\bar\sigma^2 = 0.0002$, $\alpha+\beta = 0.99$) right after the shock that left tomorrow's variance at $\sigma_{t+1}^2 = 0.0016$ (vol $4\%$). The total expected variance over $10$ days is:

$$10 \times 0.0002 + (0.0016 - 0.0002)\frac{1 - 0.99^{10}}{1 - 0.99} = 0.002 + 0.0014 \times \frac{1 - 0.9044}{0.01} = 0.002 + 0.0014 \times 9.562 = 0.01539$$

So the $10$-day volatility is $\sqrt{0.01539} \approx 0.1240 = 12.4\%$. The naive $\sqrt{10}$ scaling of today's $4\%$ would have given $4\% \times \sqrt{10} = 12.65\%$ — close here only because the whole horizon is still elevated; if you had instead scaled the *long-run* $1.41\%$ vol by $\sqrt{10}$ you would have gotten just $4.46\%$, badly understating the post-shock risk. On a \$1,000,000 book, the GARCH-based $10$-day $99\%$ VaR is $2.326 \times 0.1240 \times \$1{,}000{,}000 \approx \$288{,}400$, versus a stale-vol estimate of about \$103,800 — a near-threefold difference in required capital. The intuition: over a multi-day horizon GARCH does not just scale risk, it scales the *right* risk by accounting for how the storm will evolve day by day.

## Estimating GARCH: maximum likelihood

We have been *given* parameters; now let us see where they come from. You cannot read $\omega$, $\alpha$, $\beta$ off a chart. You estimate them from a history of returns by **maximum likelihood estimation** (MLE) — the same workhorse method we cover in detail in the post on [maximum likelihood and the method of moments](/blog/trading/math-for-quants/mle-method-of-moments-math-for-quants).

### The likelihood intuition

The idea of MLE is to find the parameter values that make the data you actually observed look as *unsurprising* as possible. For GARCH, assume each day's shock is normal with the variance the model predicts: $\varepsilon_t \sim \mathcal{N}(0, \sigma_t^2)$. The probability density of seeing a particular shock $\varepsilon_t$ on a day with forecast variance $\sigma_t^2$ is the normal density. The **log-likelihood** of the whole sample is the sum of the log densities across all days:

$$\ln L(\omega,\alpha,\beta) = -\frac{1}{2}\sum_{t=1}^{T}\left[\ln(2\pi) + \ln\sigma_t^2 + \frac{\varepsilon_t^2}{\sigma_t^2}\right]$$

where each $\sigma_t^2$ is computed by running the GARCH recursion forward with the candidate parameters. We search over $(\omega, \alpha, \beta)$ for the values that maximize this sum.

### Reading the penalty terms

The formula is worth dwelling on because it tells you what MLE is *trading off*. Each day contributes two competing terms. The $\ln\sigma_t^2$ term **penalizes overestimating variance**: if you forecast a huge variance every day, this term grows and your likelihood drops — you are being punished for crying wolf. The $\varepsilon_t^2/\sigma_t^2$ term **penalizes underestimating variance**: if a big shock $\varepsilon_t^2$ lands on a day you forecast as calm (small $\sigma_t^2$), this ratio blows up and your likelihood craters — you are punished for being caught off guard. MLE finds the parameters that balance these: high enough that surprises do not blow up the second term, low enough that the first term does not penalize chronic over-caution. That balance is what makes the fitted volatility track reality.

### Practical notes on fitting

In practice you do not code this by hand; you call a library — `arch` in Python is the standard. A few things are worth knowing so you can spot a bad fit:

- **Starting the recursion.** The recursion needs an initial $\sigma_0^2$ to get going; the usual choice is the sample variance of the returns. With a few hundred days of data the choice barely matters.
- **The constraints bind often.** Estimates frequently land with $\beta$ near $0.9$ and $\alpha + \beta$ near $1$. If your optimizer reports $\alpha + \beta = 1$ exactly, you have effectively fit IGARCH and should be cautious about long-horizon forecasts.
- **You need a lot of data.** A few hundred daily observations is a bare minimum; a thousand-plus is comfortable. Volatility models are data-hungry because the informative events — the shocks — are rare.
- **Fat tails.** The normal assumption in the likelihood understates how often huge shocks occur. Practitioners often swap the normal $z_t$ for a Student-$t$ distribution, which has heavier tails. We catalog these distributions in the [distributions cheat sheet for quant interviews](/blog/trading/quantitative-finance/distributions-cheat-sheet-quant-interviews); for serious tail estimation you eventually reach for the tools in [tail risk and extreme value theory](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants).

#### Worked example: a tiny MLE comparison by hand

You can feel how MLE discriminates between models with a two-day toy. Suppose over two days the shocks were $\varepsilon_1 = 0.01$ (a calm $1\%$ day) and $\varepsilon_2 = 0.05$ (a violent $5\%$ day). Compare two candidate constant-variance models — model A with $\sigma^2 = 0.0004$ (vol $2\%$) and model B with $\sigma^2 = 0.0009$ (vol $3\%$) — by their contribution to the log-likelihood, dropping the shared $\ln(2\pi)$ constant.

Model A: day 1 contributes $-\tfrac12[\ln 0.0004 + 0.0001/0.0004] = -\tfrac12[-7.824 + 0.25] = 3.787$; day 2 contributes $-\tfrac12[\ln 0.0004 + 0.0025/0.0004] = -\tfrac12[-7.824 + 6.25] = 0.787$. Total: $4.574$.

Model B: day 1 contributes $-\tfrac12[\ln 0.0009 + 0.0001/0.0009] = -\tfrac12[-7.013 + 0.111] = 3.451$; day 2 contributes $-\tfrac12[\ln 0.0009 + 0.0025/0.0009] = -\tfrac12[-7.013 + 2.778] = 2.118$. Total: $5.569$.

Model B wins ($5.569 > 4.574$): its higher variance is penalized a little on the calm day but rescued enormously on the violent day, because it was not blindsided by the $5\%$ move. This is MLE in miniature — and it is exactly why a fitted GARCH raises its variance after a shock rather than ignoring it. Now imagine the desk had \$1,000,000 riding on the calm forecast: model A would have set a one-day VaR of about \$46,520 the night before a $5\%$, \$50,000 loss, badly understating the danger. The intuition: MLE rewards the model that was *least surprised* by what actually happened, which is precisely the model you want guarding your capital.

## The news-impact curve

A clean way to summarize what any volatility model believes is the **news-impact curve**: a plot of *tomorrow's forecast variance* on the vertical axis against *today's shock* $\varepsilon_t$ on the horizontal axis, holding the current variance fixed. It answers one question visually: "given a shock of this size and sign, how much will the model raise tomorrow's risk?"

For plain GARCH(1,1), the news-impact curve is $\sigma_{t+1}^2 = (\omega + \beta\sigma_t^2) + \alpha\,\varepsilon_t^2$ — a **parabola**, symmetric around zero. A $-3\%$ shock and a $+3\%$ shock land at the same height: both raise tomorrow's variance by $\alpha\times 0.0009$, identically. The curve is a smooth U with its minimum at zero shock. This symmetry is GARCH's defining assumption, and it is also its defining *flaw*, because real markets are not symmetric at all.

![A before-and-after contrast of a symmetric GARCH news impact against a leverage-tilted news impact](/imgs/blogs/arch-garch-volatility-math-for-quants-7.png)

The contrast above is the crux of the next section. On the left, symmetric GARCH: a $-5\%$ day and a $+5\%$ day produce the same next-day volatility — a clean U-shaped reaction. On the right, a leverage model: the same $-5\%$ drop lifts tomorrow's volatility *more* than the $+5\%$ gain does, tilting the reaction curve so its left arm rises faster than its right. That tilt encodes a real and well-documented market behavior, and capturing it is the job of the next family of models.

## The leverage effect: why crashes scare markets more than rallies

Here is one of the most reliable asymmetries in finance: **a large drop in an equity index raises future volatility more than an equally large rise.** Fear is more contagious than greed. A $5\%$ crash sends volatility spiking; a $5\%$ melt-up raises it too, but noticeably less. This is the **leverage effect**, and a symmetric GARCH, which squares the shock and so throws away its sign, is structurally blind to it.

### Why it is called "leverage"

The traditional explanation is mechanical and about corporate balance sheets. When a company's stock price falls, the *equity* portion of its value shrinks while its *debt* stays fixed, so its debt-to-equity ratio — its **leverage** — rises. A more leveraged company is riskier, so its stock becomes more volatile. Hence a price drop *causes* higher volatility through rising leverage. Whether or not that is the full story (many believe the real driver is a "volatility feedback" effect — rising expected volatility itself depresses prices, and a fear-driven flight to safety), the empirical pattern is rock solid across equity markets, and the name stuck.

### GJR-GARCH: add a switch for down days

The simplest fix is the **GJR-GARCH** model, named for Glosten, Jagannathan, and Runkle (1993). It keeps the GARCH(1,1) skeleton and adds one extra term that switches on *only when yesterday's shock was negative*:

$$\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \gamma\,\varepsilon_{t-1}^2 \mathbb{1}_{\{\varepsilon_{t-1}<0\}} + \beta\,\sigma_{t-1}^2$$

The new piece is $\gamma\,\varepsilon_{t-1}^2 \mathbb{1}_{\{\varepsilon_{t-1}<0\}}$. The symbol $\mathbb{1}_{\{\varepsilon_{t-1}<0\}}$ is an **indicator** — it equals $1$ if yesterday's shock was negative (a down day) and $0$ otherwise. The coefficient $\gamma$ (*gamma*) is the **leverage** parameter: it is the *extra* kick to tomorrow's variance that a down day delivers on top of the ordinary $\alpha$ reaction. On an up day the indicator is zero and the model is just plain GARCH; on a down day the reaction to the squared shock is $\alpha + \gamma$ instead of $\alpha$. A positive $\gamma$ is exactly the leverage effect, and in fitted equity models $\gamma$ is reliably positive and often *larger than $\alpha$ itself*.

The persistence formula adjusts slightly. Since the down-day term fires about half the time, the long-run persistence becomes $\alpha + \beta + \gamma/2$, and the stationarity condition is $\alpha + \beta + \gamma/2 < 1$.

#### Worked example: a down day vs an up day, and the dollar asymmetry

Take a GJR-GARCH fit: $\omega = 0.000003$, $\alpha = 0.03$, $\gamma = 0.10$, $\beta = 0.90$. Suppose current variance is $\sigma_{t-1}^2 = 0.0004$ (vol $2\%$), and compare two scenarios for yesterday: a $-5\%$ day versus a $+5\%$ day. In both cases $\varepsilon_{t-1}^2 = 0.0025$.

**Up day** ($+5\%$, indicator $= 0$):
$$\sigma_t^2 = 0.000003 + 0.03\times 0.0025 + 0 + 0.90\times 0.0004 = 0.000003 + 0.000075 + 0.00036 = 0.000438$$
Forecast vol: $\sqrt{0.000438} \approx 2.09\%$.

**Down day** ($-5\%$, indicator $= 1$):
$$\sigma_t^2 = 0.000003 + 0.03\times 0.0025 + 0.10\times 0.0025 + 0.90\times 0.0004 = 0.000003 + 0.000075 + 0.00025 + 0.00036 = 0.000688$$
Forecast vol: $\sqrt{0.000688} \approx 2.62\%$.

Same-sized move, but the crash forecasts $2.62\%$ vol versus the rally's $2.09\%$ — a $25\%$ higher risk estimate purely because of the *sign*. Now the dollars. On a \$1,000,000 book, the one-day $99\%$ VaR after the up day is $2.326 \times 0.0209 \times \$1{,}000{,}000 \approx \$48{,}600$; after the down day it is $2.326 \times 0.0262 \times \$1{,}000{,}000 \approx \$60{,}900$. The same-magnitude move leaves you carrying about **\$12,300 more stated risk** if it was a drop. A symmetric GARCH would have reported the same \$48,600 in both cases and *understated your risk by a quarter* on exactly the days — down days — when risk matters most. The intuition: the leverage term is the model learning that the market is more frightened by falling than by rising, and that fear is worth real money to measure correctly.

### EGARCH: model the log, capture asymmetry smoothly

The other major leverage model is **EGARCH** (Exponential GARCH), introduced by Nelson in 1991. Instead of modeling the variance directly, it models the *logarithm* of variance:

$$\ln\sigma_t^2 = \omega + \beta\ln\sigma_{t-1}^2 + \alpha\big(|z_{t-1}| - \mathbb{E}|z_{t-1}|\big) + \theta\, z_{t-1}$$

This looks busier, so let us unpack the two clever ideas. First, by modeling $\ln\sigma_t^2$ rather than $\sigma_t^2$, the variance is automatically positive no matter what the parameters are (since $e^{\text{anything}} > 0$) — so EGARCH needs *no* positivity constraints, which makes it easier to fit. Second, asymmetry comes from the $\theta\,z_{t-1}$ term, which uses the *signed* standardized shock $z_{t-1}$ (not its square), so a negative shock and a positive shock genuinely differ. With $\theta < 0$ — the typical estimate for equities — a negative $z$ raises log-variance and a positive $z$ lowers it, exactly the leverage tilt. The $\alpha(|z_{t-1}| - \mathbb{E}|z_{t-1}|)$ term captures the symmetric "size" reaction. EGARCH's news-impact curve is not a parabola but two straight-ish arms meeting at a kink, with the left (down-shock) arm steeper.

### Choosing among the family

So when do you reach for which model? The honest answer is that GARCH(1,1) is a remarkably hard baseline to beat for pure forecasting accuracy, but for *equities specifically*, a leverage model almost always fits better and matters for risk because down days are when you care most. The matrix lays out the tradeoffs.

![A matrix comparing ARCH, GARCH, and EGARCH across past shocks used, past variance used, sign reaction, and fitting cost](/imgs/blogs/arch-garch-volatility-math-for-quants-4.png)

The matrix above is the decision aid. ARCH($q$) uses many shock lags and no variance memory, so it pays a heavy parameter cost; GARCH(1,1) uses one of each, is symmetric, and is cheap with three parameters; EGARCH logs the variance and reads the shock's sign, gaining asymmetry for the price of one more parameter. The whole family descends from one idea, and it is worth seeing the lineage explicitly.

![A tree of the GARCH family rooted at the ARCH idea, branching through GARCH into IGARCH and leverage variants](/imgs/blogs/arch-garch-volatility-math-for-quants-6.png)

The tree above shows the lineage. The root is the ARCH idea — variance built from past squared shocks. GARCH adds memory of past variance. From GARCH, two branches matter: IGARCH, the boundary case where persistence equals one, and the leverage variants, which split into GJR-GARCH (the down-day switch) and EGARCH (the logged, signed specification). Knowing the family tree means that when a paper or a risk system mentions "TGARCH" or "APARCH" or "FIGARCH," you can place it: it is one more variation on "variance is a recursion driven by past shocks."

## The link to the volatility surface

GARCH does not live alone. Its output — a forecast of the *path* of future volatility — is exactly the input that option pricing needs, which connects it directly to the [volatility surface](/blog/trading/quantitative-finance/volatility-surface), the map of option-implied volatilities across strikes and maturities.

The bridge is this. An option's value depends on the *average* volatility expected over its life. GARCH forecasts that average via the multi-step formula: after a spike, near-dated options should price in high volatility that *fades* for longer-dated ones, because GARCH says volatility mean-reverts. That produces a downward-sloping **term structure** of volatility (short-dated higher than long-dated) right after a shock — and an upward-sloping one in calm times, as volatility is expected to rise back toward normal. The leverage effect, meanwhile, is the GARCH-world explanation for the **volatility skew**: because down moves raise future volatility, the market pays up for downside protection, so out-of-the-money puts carry higher implied volatility than calls. The asymmetry GJR-GARCH and EGARCH model in *time-series* data shows up as the skew in *cross-sectional* option prices. They are two views of the same fear.

There is a subtlety worth naming. The volatility GARCH estimates from past returns is the **physical** (real-world) forecast; the volatility baked into option prices is the **risk-neutral** one, which is generally a bit higher because option sellers demand a premium for bearing volatility risk. The gap between the two — the **variance risk premium** — is itself a tradeable quantity: systematically selling options harvests it, and systematically buying them pays it. GARCH gives you the physical anchor against which to judge whether the market's implied volatility is rich or cheap. When a fitted GARCH says fair forward volatility is $18\%$ but $30$-day options imply $26\%$, a desk reads that as an $8$-point premium and decides whether the protection is worth paying for. That is the everyday use of GARCH on an options desk: not to price the option from scratch, but to form an honest opinion about whether the market's price is too high or too low.

## How GARCH is evaluated and where it breaks

A forecast you never check is a liability, not an asset. Because GARCH outputs a *variance* and not a directly observable number, evaluating it takes a little care — and the methods reveal exactly where the model is strong and where it is fragile.

### You cannot observe variance directly

The honest difficulty is that on any given day you see one return, not the variance that generated it. A single $r_t^2$ is an extremely noisy estimate of that day's $\sigma_t^2$ — its expected value is right, but its scatter is enormous. So you cannot grade a one-day forecast by comparing $\sigma_t^2$ to $r_t^2$ on that day alone; you must average over many days. Two practical fixes are standard. The first is to use **realized variance** — the sum of squared *intraday* returns (say, every five minutes) — which is a far tighter estimate of the day's true variance than a single daily return. The second is to grade the model on its **density forecasts**: if the model is right, the standardized residuals $z_t = \varepsilon_t / \sigma_t$ should look like clean, independent draws from a standard distribution, with no remaining clustering in their squares. If $z_t^2$ is still autocorrelated, the model has not soaked up all the clustering and needs more lags or a different specification.

### Backtesting the VaR

The cleanest real-world test is a **VaR backtest**. If your model's $99\%$ one-day VaR is honest, then over a long history the actual loss should exceed the VaR on about $1\%$ of days — no more, no fewer — and those exceedances should be *scattered randomly in time*, not bunched. The bunching test is the one that catches a bad volatility model: a constant-volatility VaR will have its exceedances clump together during crises (because it failed to raise the limit), while a good GARCH VaR will have them sprinkled evenly. Regulators formalized this into the Basel "traffic light" system, where too many exceedances in a year push a bank into a penalty zone with higher capital charges.

#### Worked example: counting VaR breaks over a year

Suppose you run a \$1,000,000 book on a $99\%$ daily VaR for one trading year ($252$ days). If the model is well-calibrated you expect about $252 \times 0.01 \approx 2.5$ days where the loss exceeds the VaR. Now compare two desks. Desk A uses a stale constant-volatility VaR of \$46,500; during a six-week crisis it gets blown through on $9$ separate days, all clustered in those six weeks, while the rest of the year has zero breaks. Desk B uses a GARCH VaR that rose to over \$90,000 during the crisis; it records just $3$ breaks, scattered across the year. Both desks saw the same market, but Desk A's bunched $9$ breaks are a statistical scream that its risk number was asleep during the storm — and under Basel its capital charge would jump. The intuition: a good volatility model does not eliminate losses, it makes the surprises *rare and random* instead of *clustered and catastrophic*.

### Where GARCH genuinely fails

GARCH is not magic, and a practitioner should know its limits cold. It is **reactive, not predictive of the trigger**: GARCH raises volatility *after* the first shock, so it never sees the crash coming — it only insists, correctly, that the storm will continue. It assumes a **single regime** with smooth mean reversion, so it can be slow to recognize a genuine structural break (a war, a currency de-peg) where the new normal is permanently different. Its Gaussian innovation **understates the fattest tails**, so the very worst days still surprise it (the fix is Student-$t$ innovations or the explicit tail tools in extreme value theory). And its parameters can **drift**: a model fit on a decade of data may carry a long-run variance that no longer reflects today's market, so practitioners refit regularly and sometimes weight recent data more heavily. None of these is fatal; each is a known boundary you manage around rather than a reason to abandon the model.

## Common misconceptions

**"Volatility is constant, so one number is enough."** This is the belief ARCH was invented to kill. The unconditional, long-run volatility may be roughly stable, but the *conditional* volatility — the relevant one for tomorrow's risk — swings by factors of three or four between calm and crisis. Using a single historical volatility number means systematically *understating* risk in storms and *overstating* it in calm, which is the worst of both worlds: you are over-exposed exactly when you should not be.

**"GARCH predicts the direction of the market."** It does not, and it does not try to. GARCH forecasts the *size* of moves, not their *sign*. It will happily tell you tomorrow could be a $4\%$ day; it has no view whatsoever on whether that is up or down. Confusing a volatility forecast with a directional signal is a classic beginner error — high forecast volatility is not a sell signal, it is a "size your positions smaller and tighten your stops" signal.

**"A higher $\alpha$ means a better, more responsive model."** A high $\alpha$ makes the model *twitchy* — it overreacts to single days. A high $\beta$ makes it *smooth* and slow. Neither is "better" in isolation; the right balance is whatever maximizes the likelihood on real data, which for equities is almost always a small $\alpha$ (around $0.05$–$0.10$) and a large $\beta$ (around $0.85$–$0.92$). A model with $\alpha = 0.5$ would be jerking the risk limit around so violently it would be useless.

**"GARCH handles fat tails, so I do not need to worry about crashes."** GARCH explains *part* of why returns look fat-tailed — clustering of volatility makes the *unconditional* return distribution heavier than normal even if each day's *conditional* shock is normal. But the conditional shocks themselves are also fat-tailed in reality, which plain Gaussian-GARCH misses. For genuine tail and disaster modeling you must go further, to Student-$t$ innovations or to the explicit tail machinery in [tail risk and extreme value theory](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants). GARCH tames the everyday roughness; it does not by itself bound the catastrophe.

**"If persistence is near $1$, the model is broken."** On the contrary — persistence near $1$ (say $0.97$–$0.99$) is the *normal, healthy* estimate for equity indices, and it correctly encodes that volatility shocks last for months. Only $\alpha + \beta \geq 1$ exactly (IGARCH) or above $1$ (explosive) signals trouble. A persistence of $0.5$ would be the suspicious result, implying volatility shocks vanish in a day or two, which no equity market behaves like.

**"Annualize daily vol by multiplying by $252$."** A frequent and costly slip. Variance scales with time, so daily *variance* multiplies by $252$; *volatility* is the square root, so it multiplies by $\sqrt{252} \approx 15.87$. Multiplying daily volatility by $252$ instead of $\sqrt{252}$ overstates annual volatility by a factor of nearly $16$ — the kind of error that turns a $20\%$ vol into a $320\%$ one and gets noticed immediately, but the square-root rule is worth burning into memory.

## How it shows up in real markets

**The 1987 crash and the birth of demand for the models.** Black Monday's $20.5\%$ drop on October 19, 1987 was followed by extraordinary persistence: the days and weeks after were among the most volatile on record, and volatility did not normalize for months. Engle's ARCH (1982) and Bollerslev's GARCH (1986) had just been published, and the crash was a brutal real-world demonstration of exactly the persistence they modeled. A risk desk running constant-volatility numbers would have declared the all-clear far too early; a GARCH desk would have correctly forecast elevated risk for the rest of the quarter. The episode turned GARCH from an academic curiosity into a tool every serious risk team learned.

**The 2008 financial crisis and the regime shift.** Through 2008, equity volatility did not merely spike — it stepped up to a new regime and stayed there. The VIX index (the market's $30$-day forward volatility expectation) ran around $20$ for years, then exploded above $80$ in the October–November 2008 panic and remained elevated through much of 2009. A GARCH model fit through mid-2008 would, after the Lehman shock in September, have forecast many weeks of high volatility — and it would have been right. The leverage term mattered enormously here: the move was overwhelmingly *down*, and a symmetric model would have understated the volatility response to the cascade of crashes. Books that scaled risk down on GARCH-style forecasts survived; those on stale constant-vol numbers were repeatedly stopped out.

**The Volmageddon of February 5, 2018.** On a single day, the VIX more than doubled, and a cluster of products that were short volatility — betting that calm would continue — were annihilated, with the XIV note losing about $96\%$ of its value overnight before being wound down. The years before had been historically calm; persistence had kept forecast volatility low and lulled the short-vol trade. When the shock came, GARCH's lesson — that a shock would be *followed by more shocks* — was paid out in days of elevated volatility that finished off the leftover positions. It was a clean demonstration that low persistence-driven calm is exactly the setup for a violent cluster.

**The COVID-19 crash of March 2020.** Volatility went from ordinary to extreme in a matter of days, the VIX hitting the low $80$s — comparable to 2008 — but the dynamics were textbook GARCH: an enormous initial shock, then a cluster of double-digit-percentage daily swings (both up and down) for weeks, then a slow mean reversion through the spring and summer as central-bank intervention calmed markets. A GARCH model would have nailed the *shape* of the episode even if it could not have predicted the trigger: huge spike, slow decay, leverage tilt as the down days hit hardest. The half-life math — months, not days — matched the real path of normalization closely.

**Equity index options and the persistent skew.** Look at the option chain on any major equity index on any ordinary day and you will see out-of-the-money puts trading at meaningfully higher implied volatility than out-of-the-money calls — the volatility skew. This is the leverage effect made visible in prices: because the market knows (and GJR-GARCH/EGARCH formalize) that a drop raises future volatility, downside protection is structurally expensive. The skew is one of the most persistent features of equity derivatives markets, and the time-series asymmetry that GARCH leverage models capture is its mechanistic cousin.

**Risk-parity and volatility-targeting funds.** A large class of systematic strategies sizes positions to hit a *target volatility* — when forecast volatility rises, they cut exposure; when it falls, they add. The forecast they use is some GARCH-like estimate. This creates a real, observable market feedback: a volatility spike triggers mechanical deleveraging across many funds at once, which can amplify a sell-off — the model's forecast becoming partly self-fulfilling. Understanding GARCH is not just about measuring risk; in markets where many players act on the same forecasts, it is about understanding a force that moves prices.

## When this matters to you and further reading

If you ever manage money, build a risk system, price an option, or even just want to understand why your brokerage account is calm for months and then terrifying for weeks, the ARCH/GARCH picture is the right mental model: risk is not a constant, it clusters, and the clustering is forecastable. The single most actionable takeaway is that *after a big move, expect more big moves* — size down and stay sized down longer than feels necessary, because the half-life of a volatility shock is measured in weeks to months, not days. For a portfolio, that means GARCH-style forecasts should drive position sizing and stop-loss widths, not a static historical number.

Where to go next depends on which thread you want to pull. To understand the estimation engine that fits the parameters, the post on [maximum likelihood and the method of moments](/blog/trading/math-for-quants/mle-method-of-moments-math-for-quants) builds MLE from scratch — and GARCH is one of its most important applications. To go beyond everyday roughness into genuine disaster modeling, where the Gaussian assumption inside GARCH finally breaks, read [tail risk and extreme value theory](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants). To see how the volatility forecast becomes a traded surface of option prices, the [volatility surface](/blog/trading/quantitative-finance/volatility-surface) post connects the term structure and skew directly to the GARCH dynamics here. And to firm up the probability distributions — normal, Student-$t$, and the rest — that sit inside the likelihood, the [distributions cheat sheet for quant interviews](/blog/trading/quantitative-finance/distributions-cheat-sheet-quant-interviews) is the fastest reference.

The deepest lesson of this whole family is a humble one. Engle did not invent a way to predict the market — he invented a way to predict its *mood*. We cannot know whether tomorrow brings a gain or a loss, but we can know, with real and useful accuracy, whether tomorrow is likely to be violent or calm. In risk management, that second kind of knowledge is the one that keeps you in the game long enough for the first kind to ever matter. This is educational material, not investment advice — but if there is one number worth carrying out of it, it is the half-life of a volatility shock: when the storm hits, it stays a while.

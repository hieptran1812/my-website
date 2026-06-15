---
title: "AR, MA, and ARIMA models: how quants forecast returns, spreads, and mean reversion"
date: "2026-06-15"
description: "A beginner-friendly, build-from-zero guide to autoregressive, moving-average, and ARIMA time-series models, and how quants use them to spot momentum and mean reversion, forecast a price spread, and choose a model that does not overfit."
tags: ["time-series", "autoregression", "moving-average", "arima", "box-jenkins", "mean-reversion", "momentum", "forecasting", "acf-pacf", "aic-bic", "quant-finance", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A time-series model says today's value is built from yesterday's values, yesterday's surprises, or both; getting the recipe right is the difference between a real edge and fooling yourself with noise.
>
> - An **autoregressive** model AR(p) writes today as a weighted blend of recent past values, $X_t = \phi_1 X_{t-1} + \dots + \phi_p X_{t-p} + \epsilon_t$ — this is the math of **momentum** (when the weights push the same way) and **mean reversion** (when they pull back).
> - A **moving-average** model MA(q) writes today as a blend of recent random **shocks**, $X_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \dots + \theta_q \epsilon_{t-q}$ — this is the math of a surprise that echoes for a few periods and then dies.
> - **ARIMA(p,d,q)** glues AR and MA together and adds **differencing** (the "I"), which turns a wandering, non-stationary price series into a stable one you are allowed to model and forecast.
> - You **identify** the orders from the ACF and PACF plots, **estimate** the weights by maximum likelihood, and **choose** between candidate models with AIC or BIC so you do not reward yourself for overfitting.
> - The one number to remember: raw stock **returns** behave almost like white noise — an AR(1) on daily S&P returns has a coefficient near **0.0**, so a pure forecast there is worthless — but **spreads** and **volatility** are far more forecastable, which is where the dollars actually live.

Here is a puzzle that has paid for a lot of beach houses and bankrupted a lot of clever people. You have a number that changes every day — a stock's return, the gap between two related stocks, a volatility index. You line up its history in a spreadsheet. Now you want to bet on tomorrow's number. The whole question is: **does the past tell you anything about the future, and if so, exactly how much, and exactly how?**

Most people answer this badly. They eyeball a chart, see a "trend," and trade it. They find a pattern that "always" precedes a rally, bet on it, and discover it was a coincidence that cost them real money. The disciplined answer — the one that lets a quant say *how* confident to be and size the bet accordingly — comes from a small family of time-series models that are nearly fifty years old and still run on trading desks every single day. They have intimidating names: AR, MA, ARMA, ARIMA. They are not intimidating once you see them as three plain ideas: *today looks like yesterday's values*, *today looks like yesterday's surprises*, and *first make the wandering series sit still before you model it.*

![Pipeline from past values and past shocks through a fitted model to a forecast and a dollar signal](/imgs/blogs/ar-ma-arima-math-for-quants-1.png)

The diagram above is the mental model for the whole post. On the left you have raw ingredients: a window of **past values** of the series and a window of **past shocks** (the unpredictable surprises). They feed a single **fitted model** — that is the AR/MA/ARIMA box. Out the right comes a **forecast** for tomorrow, with an honest error band attached, and from that forecast you get a **dollar signal**: how big a bet to place and in which direction. We are going to build every box in that picture from zero, ground each formula in a worked example with real dollar figures, and end at the uncomfortable truth that the most famous series of all — stock returns — is one of the *least* forecastable things you will ever model, while its quieter cousins pay the rent.

## Foundations: the building blocks of a time series

Before we can fit anything, we need to agree on a handful of words. We will define each one the first time it appears, build the simplest possible version, and only then reach for the machinery a quant uses every day. If you already know what a lag and a stationary series are, skim; if not, you can still follow every line.

### What is a time series?

A **time series** is just a list of numbers with an order: a value for each point in time, like a stock's daily closing price or its daily return. We write the value at time $t$ as $X_t$. The order matters enormously — that is the whole difference from ordinary statistics. If you shuffle a deck of returns, you destroy exactly the thing we are trying to find: how today depends on yesterday. A time series is a deck of cards you are *forbidden* to shuffle.

We will use two running examples throughout. The first is a **return**: if a stock closes at \$100 today and \$101 tomorrow, the return is $+1\%$. The second is a **spread**: the price of stock A minus a multiple of the price of stock B — the quantity a pairs trader watches. Both are time series, but as we will see they behave very differently, and that difference is the whole game.

### What is a lag?

A **lag** is simply a value from the past, shifted back in time. The first lag of $X_t$ is $X_{t-1}$ (yesterday's value); the second lag is $X_{t-2}$ (the day before); and so on. When you hear "AR(2)," the "2" means the model reaches back two lags. Lags are the raw material of every model in this post: each one asks "how much does the value $k$ days ago still matter today?"

### What is a shock, or innovation?

A **shock** — quants also call it an **innovation** or the **error term**, written $\epsilon_t$ (the Greek letter "epsilon") — is the part of today's value that the model could *not* have predicted from the past. It is the genuine surprise. In a good model the shocks are **white noise**: a stream of numbers with zero average, the same spread every day, and no relationship to each other. White noise is the static between radio stations — pure unpredictability. If your model leaves behind white-noise residuals, you have squeezed out all the structure there was to squeeze; if the leftovers still have a pattern, you have money left on the table.

We assume each shock has mean zero and a constant variance $\sigma^2$ (the spread of the surprises). The standard deviation $\sigma$ is the *typical size* of a one-day surprise — if $\sigma = 1\%$, a normal day's unpredictable move is about one percent.

### What is stationarity, and why do we obsess over it?

A series is **stationary** if its statistical character does not drift over time: the same average, the same spread, and the same relationship between a value and its lags, no matter which stretch of history you look at. The everyday analogy: a stationary series is a thermostat-controlled room — it wobbles around a fixed temperature but always comes back. A **non-stationary** series is a balloon let go in the wind — it wanders off and never settles, so "the average" is a meaningless idea because it keeps moving.

This matters because **every model in this post assumes stationarity.** You cannot meaningfully say "today is $0.3$ times yesterday plus a fixed-size shock" if "yesterday" lives at a totally different level than last year. Returns are usually close to stationary — they hover around a tiny mean and a roughly constant spread. Raw **prices** are emphatically not — a \$30 stock that becomes a \$300 stock has a moving center and a growing spread. The fix, which is the entire "I" in ARIMA, is to model the *changes* instead of the *levels*. We have a deeper sibling post on this exact prerequisite — [maximum likelihood and the method of moments](/blog/trading/math-for-quants/mle-method-of-moments-math-for-quants) covers the estimation side; here we lean on the idea and move on.

### White noise versus a random walk

These two are constantly confused, and the difference is the entire reason differencing exists, so let us nail it down. **White noise** is a stationary series of independent zero-mean shocks — pure static, no memory, no trend. A **random walk** is the *running total* of white noise: $X_t = X_{t-1} + \epsilon_t$. Each step is unpredictable, but the *level* wanders cumulatively and never settles, so a random walk is non-stationary. The link between them is exactly differencing: the first difference of a random walk *is* white noise, because $X_t - X_{t-1} = \epsilon_t$. This is why a stock price (close to a random walk) becomes a return series (close to white noise) when you difference it once. The everyday analogy: white noise is each day's coin flip; the random walk is your running cumulative score, which drifts far from zero even though every flip is fair.

#### Worked example: spotting stationary from non-stationary by eye

You hold two series. Series A is the daily **return** of a stock: values like $+0.4\%, -0.2\%, +0.1\%, -0.3\%$, always small, always hovering near zero, the same wobble in January and in December. Its average over any 50-day window is about $+0.03\%$ and its spread is about $1\%$ — both stable. Series A is (approximately) stationary. Series B is the stock's **price**: \$30 in year one, \$80 in year three, \$210 in year five. The average of the first hundred days is near \$32; the average of the last hundred is near \$200. The center moved by a factor of six. Series B is non-stationary — there is no fixed level to revert to. If you naively fit an AR model to Series B, you will get a coefficient suspiciously close to $1.0$, which is the model screaming "I am just a random walk, please difference me first." The one-sentence intuition: model returns or changes, never raw wandering prices, or the math will lie to you.

## 1. Autoregression AR(p): modeling momentum and mean reversion

Start with the simplest real idea in the whole field. **Autoregression** means "regressing a variable on its own past." In plain English: *today is a weighted echo of recent days, plus a fresh surprise.* The everyday analogy is a mood that carries over. If you were cheerful yesterday, you are somewhat more likely to be cheerful today — yesterday leaks into today — but a random event (a flat tire, a kind email) still jolts you off the trend. Markets have moods too: a calm tape tends to stay calm, a panicky one tends to stay panicky, until a fresh shock arrives.

### The AR(1) model

The smallest autoregressive model reaches back exactly one day. We write:

$$ X_t = c + \phi\, X_{t-1} + \epsilon_t. $$

Reading it left to right: today's value $X_t$ equals a constant $c$, plus a fraction $\phi$ (the Greek "phi") of yesterday's value $X_{t-1}$, plus today's shock $\epsilon_t$. The single number $\phi$ carries all the meaning. It is the **persistence**: how much of yesterday survives into today.

- If $\phi$ is **positive** (between 0 and 1), the series has **momentum**: an above-average day tends to be followed by another above-average day. The shock fades, but slowly.
- If $\phi$ is **negative** (between $-1$ and 0), the series has **mean reversion with overshoot**: an up day tends to be followed by a down day. The series zig-zags around its mean.
- If $\phi = 0$, today is just the constant plus pure noise — **white noise**, no predictability at all.
- If $\phi = 1$, you have a **random walk** — the shock never fades, the series wanders forever, and it is *not* stationary. This is the boundary case, the unit root, and it is exactly what raw prices look like.

The **stationarity condition** for AR(1) is simply $|\phi| < 1$. As long as the persistence is strictly less than one in size, shocks die out and the series settles around a long-run mean. That long-run mean is $\mu = c / (1 - \phi)$ — the level the series is always being pulled back toward.

### Mean reversion and the half-life

Here is where AR(1) earns its keep for a quant. Suppose the series is currently sitting a distance $d$ above its mean $\mu$. How long until it has, on average, drifted halfway back? Each day, the gap shrinks by a factor of $\phi$ (for $0 < \phi < 1$). After $h$ days the gap is $\phi^h \times d$. Set that equal to half and solve:

$$ \phi^h = \tfrac{1}{2} \quad\Longrightarrow\quad h = \frac{\ln(1/2)}{\ln \phi} = \frac{-\ln 2}{\ln \phi}. $$

This $h$ is the **half-life of mean reversion**: the typical number of periods for a deviation to decay halfway. A small $\phi$ (say $0.3$) means a *short* half-life — the series snaps back fast. A $\phi$ near $1$ (say $0.98$) means a *long* half-life — deviations linger for weeks. The half-life is the single most useful number a stat-arb trader extracts from an AR(1), because it tells you how long to expect to hold a mean-reverting position before it pays off.

#### Worked example: AR(1) with phi = 0.3 — forecasting a return and the half-life

You fit an AR(1) to a mean-reverting daily series — say the daily return of a sector-neutral basket — and get $\phi = 0.3$ and a long-run mean of $\mu = 0\%$ (so $c = 0$). Yesterday the basket returned $X_{t-1} = +2.0\%$, a notably strong day. What is your forecast for today?

The one-step forecast just drops the unknown shock (its expected value is zero) and plugs in:

$$ \hat X_t = c + \phi\, X_{t-1} = 0 + 0.3 \times 2.0\% = +0.6\%. $$

So you expect the basket to give back most of yesterday's move and post about $+0.6\%$ today — momentum exists but is weak. The **half-life** of a deviation is

$$ h = \frac{-\ln 2}{\ln 0.3} = \frac{0.693}{1.204} \approx 0.58 \text{ days}. $$

A deviation is more than half gone within a single day — this is a *fast* reverter. Now the dollars. Suppose you run this on a \$1,000,000 book and the historical edge from acting on this signal is a hit rate that turns a $+0.6\%$ expected move into about \$600 of expected gross PnL per \$100,000 of position, before costs. The forecast is small and the half-life is short, so you trade often, in size, and live or die on transaction costs. The one-sentence intuition: $\phi$ is the fraction of yesterday that survives into today, and the half-life translates that fraction into "how many days until my edge decays."

### AR(p): reaching back further

Sometimes one lag is not enough — the series remembers two, three, or more days. The general **AR(p)** model is:

$$ X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \dots + \phi_p X_{t-p} + \epsilon_t. $$

Each $\phi_k$ weights the lag $k$ days back. The order $p$ is how far the memory reaches. AR(2), for instance, can produce *cycles* — pseudo-periodic up-and-down waves — which a single lag cannot. Yield-curve moves, some volatility series, and economic indicators often need $p = 2$ or more. The price you pay for a bigger $p$ is more parameters to estimate, more ways to overfit, and a hungrier appetite for data — which is exactly why order selection (Section 6) matters so much.

The stationarity condition for AR(p) generalizes: the roots of the *characteristic polynomial* $1 - \phi_1 z - \phi_2 z^2 - \dots - \phi_p z^p = 0$ must all lie outside the unit circle in the complex plane. You do not need to compute that by hand — every software package checks it for you — but the idea is the same as $|\phi| < 1$: the echoes must die out, not blow up.

## 2. Moving average MA(q): modeling shock persistence

The second idea is the mirror image of the first, and beginners almost always misread its name, so let us kill the confusion immediately. A **moving-average model** in time series has *nothing to do* with the "20-day moving average" line you see on a stock chart. That charting indicator is a smoothing of past *prices*. An MA *model* is a recipe for today built from past *shocks*. Same two words, completely different object. Whenever you see "MA(q)" in this post, think "echoes of past surprises," not "the average line on a chart."

### The MA(1) model

The everyday analogy: a moving-average process is a bell rung in a room. The ring (the shock) is loud at the moment it happens, a little softer one beat later, and then silent. The sound does not feed on itself; it just fades on a fixed schedule and stops. Formally, MA(1) is:

$$ X_t = \mu + \epsilon_t + \theta\, \epsilon_{t-1}. $$

Today's value is the long-run mean $\mu$, plus today's fresh shock $\epsilon_t$, plus a fraction $\theta$ (the Greek "theta") of *yesterday's* shock $\epsilon_{t-1}$. The crucial feature: yesterday's shock shows up today, but the shock from two days ago is *completely gone*. An MA(1) has a memory of exactly one period — sharp and finite.

This is the natural model for a quantity that absorbs a surprise and then moves on. A classic example: a market over-reacts to a news shock today and partly corrects it tomorrow, then forgets it entirely. Another: microstructure effects like the **bid-ask bounce** — the gap between the price you can buy at and the price you can sell at — inject a one-period negative dependence into trade-by-trade prices that an MA(1) captures perfectly.

### MA(q): a longer echo

The general **MA(q)** model carries echoes of the last $q$ shocks:

$$ X_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}. $$

After $q$ periods, every trace of an old shock is gone — that is the defining property of an MA process: **finite memory**. An MA(q) is *always* stationary, no conditions required, because it is a finite sum of well-behaved shocks; there is no feedback loop to blow up.

#### Worked example: identifying MA(1) versus AR(1) from a single number

You compute the **autocorrelation** of a series at lag 1 and lag 2. (Autocorrelation at lag $k$ is the correlation between the series and its own value $k$ days ago — a number between $-1$ and $+1$.) You find: lag-1 autocorrelation $= 0.37$, lag-2 autocorrelation $= 0.02$, lag-3 $\approx 0.00$. The lag-1 correlation is clearly nonzero, but it slams to zero immediately after. That sudden cutoff is the fingerprint of an **MA(1)**, not an AR(1).

Why? For an MA(1), the theoretical lag-1 autocorrelation is $\rho_1 = \theta / (1 + \theta^2)$ and *every* autocorrelation beyond lag 1 is exactly zero — the memory is finite, so the correlation cuts off cleanly. Solving $0.37 = \theta/(1+\theta^2)$ gives $\theta \approx 0.42$. For an AR(1), by contrast, the autocorrelations *decay geometrically*: $\rho_1 = \phi$, $\rho_2 = \phi^2$, $\rho_3 = \phi^3$, never hitting zero. If this were an AR(1) with $\phi = 0.37$, you would expect $\rho_2 \approx 0.14$, not $0.02$. The data says MA(1). Now the dollars: knowing it is MA(1) tells you the predictable part of tomorrow comes *only* from today's realized shock, so on a \$500,000 position the actionable forecast horizon is one day — you cannot chain it out a week the way you could a persistent AR process. The one-sentence intuition: a clean cutoff in the autocorrelation says MA; a slow geometric decay says AR.

![Before and after comparison of an AR shock that fades geometrically forever versus an MA shock that vanishes after q steps](/imgs/blogs/ar-ma-arima-math-for-quants-2.png)

The figure contrasts the two memories directly. On the left, an AR(1) with $\phi = 0.5$: a shock of size $1.0$ today leaves $0.5$ of itself tomorrow, $0.25$ the day after, $0.125$ next — fading forever but never quite reaching zero. On the right, an MA(1) with $\theta = 0.4$: a shock leaves $0.4$ tomorrow and then *nothing* — gone for good after one step. Infinite-but-fading memory versus finite-but-sharp memory. That single distinction is what the ACF and PACF plots (Section 5) let you read off real data.

## 3. The duality of AR and MA

Here is one of the most beautiful facts in time series, and it explains why these two models are not rivals but two views of the same thing. **An AR model is an MA model in disguise, and vice versa.**

Take the AR(1): $X_t = \phi X_{t-1} + \epsilon_t$ (set $c = 0$ for clarity). Now substitute the same equation for $X_{t-1}$ into itself: $X_{t-1} = \phi X_{t-2} + \epsilon_{t-1}$. Keep going, substituting again and again, and you unroll the recursion into:

$$ X_t = \epsilon_t + \phi\,\epsilon_{t-1} + \phi^2\,\epsilon_{t-2} + \phi^3\,\epsilon_{t-3} + \dots $$

That is an **MA($\infty$)** — an infinite moving average! A *finite* AR is exactly an *infinite* MA with geometrically shrinking weights. This is why an AR shock fades forever: the model carries an infinite tail of past shocks, each weighted by a higher power of $\phi$. The substitution only converges (the weights only shrink) when $|\phi| < 1$ — the stationarity condition, now seen from the other side.

The duality runs both ways. A finite, **invertible** MA can be rewritten as an infinite AR. An MA(1) with $|\theta| < 1$ unrolls into $X_t = \epsilon_t$ where $\epsilon_t = X_t - \theta X_{t-1} + \theta^2 X_{t-2} - \dots$ — an infinite autoregression with alternating, shrinking weights. The condition $|\theta| < 1$ is called **invertibility**, and it is the MA twin of stationarity.

So why keep both models if each can mimic the other? **Parsimony** — fewer parameters. A series with a sharp one-period memory is a *one-parameter* MA(1) but would need an *infinite* AR to capture exactly; you would truncate it to AR(5) or AR(10) and waste degrees of freedom. A series with smooth geometric decay is a *one-parameter* AR(1) but an infinite MA. You pick whichever description is shorter for the structure in front of you. That is the entire art of identification: find the *smallest* model that fits.

The table below lines up the three building blocks side by side, so you can see at a glance which one to reach for. The "memory" column is the single most important property: AR remembers forever-but-fading, MA remembers sharply-then-stops, and ARMA does both.

| Model | Built from | Memory of a shock | Always stationary? | Quant use |
| --- | --- | --- | --- | --- |
| AR(p) | past values | infinite, geometric decay | only if roots outside unit circle | momentum, mean reversion, spreads |
| MA(q) | past shocks | finite — gone after q steps | yes, always | bid-ask bounce, news over-reaction |
| ARMA(p,q) | both | infinite, but flexibly shaped | only if AR part is stationary | most stationary financial series |
| ARIMA(p,d,q) | both, after differencing | same as ARMA, on the changes | by construction, after d differences | non-stationary prices, macro series |

Notice the asymmetry in the "always stationary?" column. An MA process can never blow up — it is a finite weighted sum of bounded shocks, with no feedback loop — so it is stationary by construction. An AR process *feeds its own past back into itself*, so it can spiral out of control if the persistence is too strong; that is why it needs the unit-circle condition. This is the deep reason the bid-ask bounce is naturally an MA and a crowded mean-reversion trade is naturally an AR: one is a passive echo, the other is an active feedback loop.

#### Worked example: unrolling an AR(1) into its shock weights

You have an AR(1) with $\phi = 0.6$. A one-time shock of $\epsilon = +1.0\%$ hits the series today and is never repeated. How does that single surprise ripple forward? Using the MA($\infty$) form, the impact decays as powers of $0.6$:

- Today: $+1.00\%$ (the shock itself).
- Tomorrow: $0.6 \times 1.0 = +0.60\%$.
- Day 2: $0.6^2 = +0.36\%$.
- Day 3: $0.6^3 = +0.216\%$.
- Day 5: $0.6^5 \approx +0.078\%$.

The **total** cumulative impact of that one shock, summed over all future days, is the geometric series $1/(1-\phi) = 1/0.4 = 2.5$ — so a $+1\%$ surprise eventually moves the running level of the series by $2.5\%$ in total. On a \$1,000,000 book that is the difference between treating a shock as a one-day event (worth ~\$10,000 of repositioning) versus a multi-day event (worth ~\$25,000 cumulatively). The one-sentence intuition: an AR coefficient is a half-told MA story — every persistent model is secretly an infinite stack of fading shocks.

## 4. ARIMA: differencing for non-stationarity

We have AR (echoes of values) and MA (echoes of shocks). Glue them and you get **ARMA(p,q)**: today is part values-blend, part shocks-blend, all at once:

$$ X_t = c + \phi_1 X_{t-1} + \dots + \phi_p X_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \dots + \theta_q \epsilon_{t-q}. $$

ARMA is the workhorse for any *stationary* series. But most price series are *not* stationary — they wander, as we saw with the \$30-to-\$210 stock. That is what the **"I"** in ARIMA fixes. "I" stands for **integrated**, and the cure is **differencing**.

### What differencing does

To **difference** a series is to replace each value by the *change* from the previous value: $\nabla X_t = X_t - X_{t-1}$. A wandering price becomes a (usually) stationary series of *changes*. The number of times you difference is the order $d$. So **ARIMA(p,d,q)** means: difference the series $d$ times, then fit an ARMA(p,q) to the result.

- $d = 0$: the series is already stationary, ARIMA reduces to plain ARMA.
- $d = 1$: difference once. A price becomes returns (roughly); a random walk becomes white noise. This is by far the most common case.
- $d = 2$: difference twice. Needed when even the *changes* are trending — rare for financial prices, common for things like accumulated economic quantities.

The everyday analogy: differencing is switching from "how high is the elevator?" (which keeps changing as the building gets taller) to "how many floors did it move this second?" (which hovers around a stable, modelable range). The first question has no fixed answer; the second does.

A subtle but crucial warning: **over-differencing is a real sin.** Difference a series that was already stationary and you inject a spurious MA(1) structure and inflate the variance — you create a problem where none existed. The discipline is to difference the *minimum* number of times needed to reach stationarity (checked with a unit-root test like the Augmented Dickey-Fuller test), and no more.

![Tree of the ARIMA model family branching from the ARIMA parent into ARMA, AR, MA, random walk, and a spread model](/imgs/blogs/ar-ma-arima-math-for-quants-5.png)

The tree above maps the whole family onto one root. At the top sits the fully general **ARIMA(p,d,q)**. Set $d = 0$ and you drop to **ARMA**; from ARMA, set $q = 0$ for a pure **AR(p)** or $p = 0$ for a pure **MA(q)**. On the differenced branch ($d > 0$), the famous **random walk** is just ARIMA(0,1,0) — difference once, model nothing, because the changes are pure noise. And the **spread model** a pairs trader fits (Section 8) lives on the differenced branch too. Every model you will ever name is one of these special cases with some order set to zero. Memorizing this tree is memorizing the entire field.

#### Worked example: turning a wandering price into a modelable series

A stock's closing prices over five days are \$100.00, \$101.50, \$100.80, \$102.20, \$101.90. As *levels*, these are non-stationary — there is no fixed mean. Difference once ($d = 1$): the changes are $+\$1.50, -\$0.70, +\$1.40, -\$0.30$. Now you have a series that hovers near zero with a stable spread — stationary, and ready for an ARMA fit. (In practice quants difference the *log* price, $\log P_t - \log P_{t-1}$, which gives the continuously-compounded return and stabilizes the variance too; the \$1.50 change on a \$100 stock is a $+1.49\%$ log return.) The first differencing turned an unmodelable wanderer into a clean return series. The one-sentence intuition: the "I" in ARIMA is the step that makes a price legal to model — you forecast its changes, then add them back up to forecast its level.

## 5. Identification: reading the ACF and PACF

You have a stationary series (differenced if needed). Now: what orders $p$ and $q$ should you use? The classical answer, from Box and Jenkins, is to *look at two plots* before fitting anything. They are the **ACF** and the **PACF**, and together they fingerprint the model.

### The ACF: the autocorrelation function

The **autocorrelation function (ACF)** at lag $k$ is the plain correlation between the series and its own value $k$ steps back — exactly the autocorrelations we computed earlier. Plotted as a bar chart over lags $1, 2, 3, \dots$ it shows how long the series "remembers" its own level. We have a whole prior post building correlation intuition from scratch — [covariance, correlation, and their pitfalls](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews) — and the ACF is just that idea applied to a series against shifted copies of itself.

### The PACF: the partial autocorrelation function

The **partial autocorrelation function (PACF)** at lag $k$ is the correlation between $X_t$ and $X_{t-k}$ *after removing* the influence of all the lags in between. The "partial" means "controlling for the middlemen." Why bother? Because in an AR(2), $X_t$ correlates with $X_{t-2}$ partly *through* $X_{t-1}$ — the ACF double-counts that indirect path. The PACF strips out the indirect path and shows only the *direct* link from $X_{t-k}$ to $X_t$. That direct link is exactly what an AR coefficient captures, which is why the PACF reads AR order so cleanly.

### The identification rules

Here is the heart of the Box-Jenkins method, the rule a quant memorizes:

- **AR(p):** the ACF **decays** gradually (geometric or damped-sine), and the PACF **cuts off** sharply after lag $p$. *PACF tells you the AR order.*
- **MA(q):** the ACF **cuts off** sharply after lag $q$, and the PACF **decays** gradually. *ACF tells you the MA order.*
- **ARMA(p,q):** *both* the ACF and PACF decay gradually, neither cuts off cleanly. You cannot read the orders directly; you try a few small candidates and let AIC/BIC choose (Section 6).
- **White noise:** both the ACF and PACF are essentially zero at every lag — the series is unpredictable, and no AR/MA model will help. This is the verdict for raw stock returns, and it is a feature of the data, not a failure of the method.

![Matrix of ACF and PACF cutoff versus decay signatures for AR, MA, ARMA, and white noise](/imgs/blogs/ar-ma-arima-math-for-quants-3.png)

The matrix above is the identification cheat-sheet you will return to forever. Read across each row: an **AR(p)** has a slowly-decaying ACF but a PACF that cuts off at lag $p$ — so you read the order off the PACF. An **MA(q)** is the mirror — ACF cuts off at $q$, PACF decays — read the order off the ACF. **ARMA** decays on both, so you fall back to AIC. **White noise** is flat on both, the signature of "do not bother." The whole skill of identification is recognizing which cell of this grid your two plots land in.

#### Worked example: reading orders off two plots

You difference a futures spread once, then plot its ACF and PACF over 12 lags. The ACF bars are: lag 1 = $0.55$, lag 2 = $0.31$, lag 3 = $0.17$, lag 4 = $0.09$ — a clean geometric decay, each about $0.55$ times the previous. The PACF bars are: lag 1 = $0.55$ (large), lag 2 = $0.01$, lag 3 = $-0.02$, lag 4 = $0.00$ — one big spike, then silence. ACF decays, PACF cuts off after lag 1: this is the unmistakable signature of an **AR(1)**, and the PACF spike of $0.55$ is your first estimate of $\phi$. You would then fit ARIMA(1,1,0) — one AR lag, one difference, no MA term. If instead the ACF had cut off after lag 1 and the PACF had decayed, you would have fit ARIMA(0,1,1) — an MA(1) on the differenced series. The dollars: correctly reading AR(1) tells you the differenced spread is *persistent and forecastable*, which on a \$2,000,000 pairs book is the green light to put the trade on. The one-sentence intuition: the plot that cuts off names the order, and which plot cuts off tells you whether the model is AR or MA.

## 6. Estimation and order selection

Identification narrows you to a few candidate orders. Now you have to (a) pin down the actual coefficient values and (b) pick the final model. Two separate jobs.

### Estimation by maximum likelihood

For a pure AR model you can estimate the $\phi$'s by ordinary least squares — literally regress the series on its own lags, the same regression machinery from [the regression deep-dive](/blog/trading/quantitative-finance/linear-regression-deep-quant-interviews). But the moment an MA term is present, the shocks $\epsilon_t$ are unobserved (you cannot regress on something you cannot see), so OLS breaks. The general tool is **maximum likelihood estimation (MLE)**: choose the coefficients that make the observed series most probable under the model. We built MLE from zero in a [dedicated sibling post](/blog/trading/math-for-quants/mle-method-of-moments-math-for-quants); here it is enough to know that the software writes down the probability of your data as a function of $(\phi, \theta, \sigma^2)$ and numerically cranks the knobs until that probability is maximized. The result is a set of fitted coefficients *with standard errors*, so you can tell which lags are statistically real and which are noise.

### Diagnostics: are the residuals white noise?

After fitting, you must check the **residuals** — the leftover shocks the model could not explain. If the model is good, the residuals are white noise: their own ACF should be flat (no leftover autocorrelation). The formal test is the **Ljung-Box test**, which checks whether the first several residual autocorrelations are jointly zero; a high p-value means "residuals look like noise, the model captured the structure." If the residuals *still* show a pattern, your order is too low — go back and add a lag. This loop is the discipline that separates a fitted model from a *validated* one.

### AIC and BIC: paying for complexity

A bigger model *always* fits the in-sample data at least as well — add a lag and the likelihood cannot get worse. So "best fit" is a trap: it rewards overfitting, which forecasts terribly out of sample. The fix is an **information criterion** that charges a penalty per parameter. The two standard ones:

$$ \text{AIC} = -2\,\ell + 2k, \qquad \text{BIC} = -2\,\ell + k\ln n. $$

Here $\ell$ is the maximized log-likelihood (bigger = better fit, so $-2\ell$ is smaller = better), $k$ is the number of parameters, and $n$ is the sample size. **Lower is better** for both. The first term rewards fit; the second term punishes complexity. AIC charges $2$ per parameter; BIC charges $\ln n$ per parameter, which for any sample bigger than $n = 8$ is a *harsher* penalty — so **BIC prefers smaller models** than AIC and is the choice when you most fear overfitting. You compute AIC (or BIC) for each candidate and pick the lowest.

![Before and after comparison showing an overfit AR(2) with better log-likelihood but higher AIC versus the leaner AIC-selected AR(1)](/imgs/blogs/ar-ma-arima-math-for-quants-6.png)

The figure shows the penalty doing its job. The AR(2) on the left fits history a hair better — log-likelihood $-498.0$ versus the AR(1)'s $-498.6$ — because the extra lag can always soak up a little more in-sample variation. But that second lag is tiny and probably spurious. AIC charges $2$ for it, and once you do the arithmetic the AR(1) wins on AIC ($1001.2$ versus $1004.0$, lower is better). The leaner model is the one that will actually forecast. Chasing raw fit would have picked the wrong model.

#### Worked example: AIC and BIC choosing between AR(1) and AR(2)

You fit two models to $n = 500$ daily observations of a mean-reverting spread and read off the maximized log-likelihoods. AR(1) has $k = 2$ parameters (one $\phi$, one variance) and log-likelihood $\ell_1 = -498.6$. AR(2) has $k = 3$ parameters and a slightly better fit, $\ell_2 = -498.0$. Compute AIC for each:

$$ \text{AIC}_{\text{AR(1)}} = -2(-498.6) + 2(2) = 997.2 + 4 = 1001.2, $$
$$ \text{AIC}_{\text{AR(2)}} = -2(-498.0) + 2(3) = 996.0 + 6 = 1002.0. $$

AR(1) has the lower AIC ($1001.2 < 1002.0$), so AIC picks AR(1). The extra fit from the second lag ($1.2$ improvement in $-2\ell$) did not cover its cost ($2.0$ in penalty). Now BIC, with $\ln 500 \approx 6.21$ per parameter:

$$ \text{BIC}_{\text{AR(1)}} = 997.2 + 2(6.21) = 1009.6, \qquad \text{BIC}_{\text{AR(2)}} = 996.0 + 3(6.21) = 1014.6. $$

BIC also picks AR(1), and by an even wider margin ($5.0$) because its per-parameter penalty is harsher. Both criteria agree: the second lag is not worth it. The dollars: had you kept the AR(2), its noisy second coefficient would have generated phantom signals that, on a \$1,000,000 book, historically bled roughly \$8,000 a year in extra turnover and bad fills chasing a lag that was not really there. The one-sentence intuition: an information criterion lets the data tell you when a more complex model has stopped earning its keep.

![Stack of the five-step Box-Jenkins loop from differencing through identification, estimation, diagnostics, to forecasting](/imgs/blogs/ar-ma-arima-math-for-quants-4.png)

The stack above is the **Box-Jenkins method** end to end — the loop every one of these examples lives inside. Step one: difference until stationary (set $d$). Step two: identify $p$ and $q$ from the ACF and PACF. Step three: estimate the coefficients by MLE. Step four: diagnose the residuals — if they are not white noise, go back up and adjust the order. Step five, only once the residuals are clean: forecast, and turn the forecast into a trade. It is a loop, not a straight line: a failed diagnostic sends you back to identification. Following it religiously is what keeps you from the cardinal sin of trading a model whose residuals still scream "I missed something."

## 7. Forecasting and forecast error

A fitted model is only worth its forecasts. The good news: forecasting from an AR/MA model is mechanical. The sobering news: the forecast comes with an error band that widens fast, and for some series — stock returns most painfully — that band swamps the signal entirely.

### One-step and multi-step forecasts

For an AR(1), the one-step-ahead forecast is what we already did: $\hat X_{t+1} = c + \phi X_t$ (drop the unknown future shock, since its expected value is zero). The two-step forecast chains it: $\hat X_{t+2} = c + \phi \hat X_{t+1}$, using the forecast in place of the unobserved value. Iterating, the $h$-step forecast pulls geometrically back toward the long-run mean:

$$ \hat X_{t+h} = \mu + \phi^h (X_t - \mu). $$

As $h \to \infty$, $\phi^h \to 0$ and the forecast converges to the unconditional mean $\mu$. This is profound and humbling: **far enough out, the best forecast is just the long-run average.** The model's edge lives entirely in the near term, and it decays at exactly the half-life we computed in Section 1.

### Forecast error and why it grows

The **forecast error** is the gap between what you predicted and what happened. For one step, the error is just the unpredictable shock, so its variance is $\sigma^2$. For $h$ steps, the errors of all the intervening shocks pile up. For an AR(1) the $h$-step forecast-error variance is:

$$ \text{Var}(\text{error}_h) = \sigma^2 \, \frac{1 - \phi^{2h}}{1 - \phi^2}, $$

which grows with $h$ and saturates at the *unconditional variance* of the series, $\sigma^2/(1-\phi^2)$. In words: the further out you forecast, the wider your uncertainty, until eventually your forecast interval is just the historical range of the series and you have learned nothing the long-run average did not already tell you. A 95% forecast interval is roughly the forecast $\pm 1.96$ times the square root of that variance — and it fans out like a trumpet as $h$ grows.

![Timeline of an AR(1) spread forecast pulling back toward the mean while the realized spread tracks it down](/imgs/blogs/ar-ma-arima-math-for-quants-7.png)

The timeline above traces a real-shaped forecast against reality for a mean-reverting spread. On Day 0 the spread sits at an extreme $+2.00$. The AR(1) forecast for Day 1 pulls it toward the mean ($+1.40$); the spread actually comes in at $+1.55$ — close to the forecast and clearly reverting. Day 2's forecast pulls further ($+0.98$), and by Day 5 the spread has decayed to near zero, just as the geometric pull-back formula predicted. The forecast does not nail each day exactly — the error band is real — but it gets the *direction and the pace* right, and that is all a mean-reversion trade needs.

### The brutal truth about returns

Now the punchline that separates professionals from chartists. Fit an AR(1) to the daily *returns* of a large, liquid index like the S&P 500 and you will get $\phi$ within a whisker of $0.0$ — often statistically indistinguishable from zero. The ACF and PACF are flat. The verdict is **white noise**: tomorrow's return is essentially unpredictable from past returns alone. This is not a failure of ARIMA; it is the *efficient-market* result showing up in the data. If daily returns had a reliable $\phi = 0.3$, every quant on Earth would trade it, and the trading would erase it. Markets compete the predictability out of the most-watched series.

> The most-watched series are the least forecastable: the more eyes on a pattern, the faster the trading erases it.

That aphorism is worth internalizing because it inverts the beginner's instinct. A newcomer reaches first for the S&P 500 daily return — the most famous, most-data-rich series in finance — and is baffled when the model finds nothing. The professional reaches *away* from the spotlight, toward the quiet corners where fewer competitors have already competed the predictability out. The math of ARIMA is identical in both places; what differs is how much edge survives the crowd.

So where is the money? In the series that are *not* as efficiently arbitraged:

- **Spreads** between cointegrated pairs — the difference series mean-reverts even when each leg is a random walk. (We cover the formal machinery in the cointegration post.)
- **Volatility** — the *squared* returns are highly autocorrelated even when returns are not. Volatility clusters: calm follows calm, storms follow storms. That is the entire reason ARCH and GARCH exist, and it is far more forecastable than direction.
- **Intraday and microstructure** series — bid-ask bounce, order-flow imbalance, signed volume — carry short-lived AR/MA structure that fast traders harvest.

#### Worked example: forecasting a spread and the pairs-trade edge

You run a pairs trade on two refiners. The spread (stock A minus $1.2 \times$ stock B, in dollars) is mean-reverting with $\mu = \$0$, $\phi = 0.70$ on a daily AR(1), and a one-day shock standard deviation of $\sigma = \$0.50$. Today the spread is unusually wide at $X_t = +\$2.00$ — A looks rich relative to B. Your model says this should revert. The forecasts:

- One day: $\hat X_{t+1} = 0.70 \times 2.00 = +\$1.40$.
- Two days: $\hat X_{t+2} = 0.70 \times 1.40 = +\$0.98$.
- Five days: $\hat X_{t+5} = 0.70^5 \times 2.00 = 0.168 \times 2.00 = +\$0.34$.

The **half-life** is $-\ln 2 / \ln 0.70 = 0.693/0.357 \approx 1.94$ days — the gap should close halfway in about two days. So you **short the spread**: sell stock A, buy $1.2$ shares of B, betting the \$2.00 gap shrinks. If you put on \$100,000 of notional per leg and the spread reverts from \$2.00 toward \$0.34 over a week as forecast, the convergence is worth roughly \$1.66 per share of spread on the position — call it about \$3,300 of gross PnL on this single round-trip, before costs. The one-day forecast-error standard deviation is just \$0.50, comfortably smaller than the \$0.60 expected one-day move, so the edge clears the noise. The one-sentence intuition: returns may be white noise, but a well-chosen *spread* reverts on a clock you can read, and that clock is the AR coefficient.

## Common misconceptions

**"A moving-average model is the moving-average line on my chart."** No — and this trips up nearly everyone. The 20-day moving average on a price chart smooths past *prices*; an MA(q) *model* is a recipe for today built from past unobserved *shocks*. Same words, unrelated objects. Confusing them leads people to think they are "fitting an MA model" when they are really just drawing a smoothing line.

**"A higher AR order always forecasts better because it fits the history better."** Backwards. In-sample fit can only improve as you add lags, but out-of-sample forecasting *degrades* once you start fitting noise. That is the entire reason AIC and BIC exist: to charge for each parameter so you stop adding lags before you overfit. The model with the best in-sample fit is almost never the best forecaster.

**"If a series is non-stationary, ARIMA can handle it as-is."** Only after you difference it correctly. Feed raw prices to an AR model and you will estimate $\phi \approx 1$ — a near-unit-root — and your "forecast" will be a barely-changed copy of today's price with a uselessly wide error band. The "I" step is not optional; choosing $d$ correctly is the precondition for everything else.

**"More differencing is safer than too little."** No — over-differencing is its own disease. Difference a stationary series and you manufacture a spurious negative MA(1) and inflate the forecast variance. The discipline is the *minimum* differencing that achieves stationarity, verified by a unit-root test, and not one difference more.

**"If I can forecast a series, I can make money on it."** Not necessarily. The forecast must beat **transaction costs** and **market impact**, and the *predictable* part must be large relative to the *unpredictable* shock. A statistically real $\phi = 0.05$ on a series with a huge $\sigma$ produces a forecast so swamped by noise that the bid-ask spread eats the entire edge. Forecastability is necessary but nowhere near sufficient.

**"ARIMA models stay calibrated, so I can fit once and trade forever."** Markets are non-stationary in the deepest sense — the *relationships themselves* drift as regimes change, competitors arrive, and microstructure evolves. A spread's half-life can stretch from two days to two weeks as the trade gets crowded. Models must be refit on a rolling window and monitored; a coefficient that was $0.70$ last year may be $0.40$ today, and trading the stale number loses money.

## How it shows up in real markets

### 1. Pairs trading and statistical arbitrage

The single biggest application. Two economically linked stocks — Coca-Cola and PepsiCo, two oil refiners, an ETF and its basket — each wander like random walks, but their *spread* mean-reverts. A quant fits an AR-type model (often after confirming cointegration) to the spread, extracts the half-life, and trades the spread back to its mean: short it when wide, long it when narrow. The half-life sets the holding period and the position sizing. This strategy, pioneered at Morgan Stanley's quant desk in the 1980s and industrialized by funds like Renaissance and D.E. Shaw, is squarely an AR/ARIMA story. The mechanism from this post — a persistent, mean-reverting differenced series with a readable half-life — is exactly the edge. The lesson: the individual legs are unforecastable, but their combination is not, which is the whole reason the trade exists.

### 2. The bid-ask bounce in high-frequency prices

At the tick level, transaction prices bounce between the bid and the ask even when the "true" price has not moved — a buy prints at the ask, the next sell prints at the bid, and so on. This injects a *negative* first-order autocorrelation into trade-by-trade returns, the textbook fingerprint of an MA(1) with $\theta < 0$. Roll's 1984 model famously used exactly this MA(1) structure to *back out the bid-ask spread* from the autocovariance of prices, with no order-book data at all. Market-makers and execution algos model and remove this bounce so they do not mistake microstructure noise for a real price signal. The lesson: an MA(1) is not a textbook toy; it is the literal mathematics of how a price prints between two quotes.

### 3. Volatility clustering and the road to GARCH

Fit an AR model to daily S&P returns and you get nothing — $\phi \approx 0$, white noise. But fit one to the *squared* returns and the autocorrelations are large and slowly decaying out to dozens of lags: big moves cluster with big moves. This is the empirical fact that returns are unpredictable in *direction* but highly predictable in *magnitude*. Engle's 1982 ARCH model and Bollerslev's 1986 GARCH model are, at heart, AR/ARMA models applied to the variance instead of the level. Every risk desk on Wall Street forecasts tomorrow's volatility this way to set position limits and Value-at-Risk. The lesson: when the level is white noise, look at the squares — the forecastable structure often hides there.

### 4. The 2007 quant quake and crowded mean reversion

In August 2007, dozens of statistical-arbitrage funds running similar AR-style mean-reversion models on equity spreads suffered enormous, simultaneous losses over a few days — strategies that had Sharpe ratios above 2 for years lost double-digit percentages in a week. The cause was crowding: too many funds had fit the *same* mean-reverting spreads, and when one large fund deleveraged, it pushed the spreads *further* from their means instead of back toward them, triggering everyone else's stop-losses in a cascade. The AR coefficient that said "this reverts in two days" was estimated on a history that no longer applied once the trade got crowded. The lesson: a fitted coefficient describes the past regime, and the most dangerous moment is when a profitable, well-modeled relationship quietly stops holding.

### 5. Yield-curve dynamics and AR(2) cycles

Government-bond yields and the spreads between maturities show richer dynamics than a single lag can capture — they exhibit damped oscillations as the curve steepens and flattens through the cycle. Fixed-income quants routinely fit AR(2) (or higher) models, sometimes inside a state-space/Kalman framework, to capture these pseudo-cycles in the level, slope, and curvature factors of the curve. Central-bank policy expectations get embedded in these autoregressive dynamics. The lesson: the order $p$ is not cosmetic — an AR(2) can represent a cycle that an AR(1) literally cannot, and choosing too low an order throws away real structure.

### 6. Economic and macro nowcasting

Beyond pure price series, ARIMA is the default baseline for forecasting macroeconomic series — GDP, inflation, unemployment — that drive macro trades. These series are strongly trending (non-stationary), so they are differenced ($d = 1$ or $2$) before an ARMA is fit, and the resulting forecasts feed into rates and FX positioning. The famous "ARIMA is hard to beat" result in forecasting competitions comes from exactly this domain: a well-specified ARIMA on a differenced macro series is a stubbornly strong benchmark that fancy machine-learning models often fail to beat. The lesson: a simple, correctly-differenced ARIMA is the bar every more complex forecaster must clear, and frequently does not.

## When this matters to you

If you ever try to forecast *anything* that arrives in time order — a stock, a spread, your website's daily traffic, a city's electricity demand — these three ideas are the right starting point, and they will save you from the two classic disasters: trading a pattern that was really noise, and fitting a model so complex it forecasts beautifully on the past and disastrously on the future. The discipline of "make it stationary, identify the order from the ACF and PACF, estimate, check that the residuals are noise, *then* forecast" is a checklist that has outlasted decades of fashions, and following it is most of what separates a real edge from self-deception.

The honest caveats, because finance is unforgiving: this is educational, not investment advice. The most-watched series — index returns — are very close to unforecastable, and any backtest that says otherwise on daily S&P returns is almost certainly leaking future information or overfitting. The real opportunities live in less-arbitraged corners (spreads, volatility, microstructure), they decay as they get crowded, and a forecast only becomes a *profit* after it beats transaction costs and survives the regime not changing under your feet. A coefficient estimated on last year's data is a description of last year, not a promise about tomorrow.

For where to go next on this blog: build the stationarity intuition these models depend on, then layer on the volatility story and the formal cointegration machinery. Start with [maximum likelihood and the method of moments](/blog/trading/math-for-quants/mle-method-of-moments-math-for-quants), which is the engine that actually fits every ARMA term. See how a forecast becomes a tradable edge in [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research). Connect the mean-reversion-on-a-clock idea to the probability of a spread *touching* a level in [Markov chains and hitting times](/blog/trading/quantitative-finance/markov-chains-hitting-times-quant-interviews). And tighten your grip on the correlation plots that drive identification with [covariance, correlation, and their pitfalls](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews). The thread tying them all together is the one this post started with: the past sometimes tells you about the future — your job is to measure *exactly how much*, and to never, ever round that number up.

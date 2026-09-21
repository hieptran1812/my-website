---
title: "Change of numeraire: pricing in units that make the problem easy"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "The risk-neutral measure is not special. It is just the one you get when you quote prices in money-market units. Pick a different unit to measure wealth in and you get a different measure, and the right choice turns an ugly two-dimensional expectation into a single Black-Scholes call."
tags: ["change-of-numeraire", "forward-measure", "margrabe", "quanto", "risk-neutral-measure", "radon-nikodym", "girsanov", "swaptions", "derivatives-pricing", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 18
---

> [!important]
> **TL;DR:** a numeraire is whatever you agree to measure wealth in, and every choice of numeraire comes with its own probability measure under which prices quoted in those units are martingales. The risk-neutral measure is simply the choice "measure in money-market units."
>
> - The price is invariant. The measure is not. Three different numeraires gave the same caplet value of \$907,111.76 in this post, to the cent.
> - The Radon-Nikodym derivative between two of these measures is nothing but the ratio of the two numeraires. That is the whole change-of-measure machinery.
> - Using a zero-coupon bond as numeraire (the forward measure) removes the correlation between the random discount factor and the payoff. Ignoring it overpriced the same caplet by exactly ${52/51}$, about 1.96%.
> - Using an asset itself as numeraire turns Margrabe's two-asset exchange option into one Black-Scholes call with a single spread volatility, and no interest rate appears anywhere.
> - The one number to remember: on a \$50m quanto, dropping the change-of-measure drift correction lost \$247,274, roughly 5.9% of the option's value.

## The measure is a choice, not a fact

Most people meet the risk-neutral measure as a revelation and then never question it again. There is a real world with probability $P$, there is a pricing world with probability $Q$, you discount at the risk-free rate and take an expectation under $Q$, and that is pricing. It works, so nobody asks the follow-up question: why *that* measure?

The answer is deflating. $Q$ is not a law of nature. It is the bookkeeping you get when you decide to quote every price in units of a money-market account. Quote them in units of a two-year zero-coupon bond, or of the stock itself, and you get a completely different set of probabilities. Every one of them is as legitimate as $Q$. And some of them turn a pricing problem that needs a two-dimensional numerical integral into a formula you can write on a napkin.

That freedom is the single most labour-saving idea in derivatives pricing. It also looks like sleight of hand the first time you see it, because the probabilities visibly change and the answer visibly does not. The figure below is that whole tension in one image: the same call, priced twice, under two different sets of probabilities, landing on the same number.

![Two side by side lanes pricing the same call on a two state stock, the left lane using the money market account as numeraire with probabilities 0.500 and 0.500, the right lane using the stock as numeraire with probabilities four sevenths and three sevenths, both arriving at ninety five thousand two hundred thirty eight dollars and ten cents](/imgs/blogs/change-of-numeraire-math-for-quants-1.webp)

## Foundations: what a numeraire actually is

A **numeraire** is the thing you measure wealth in. That is the entire definition, and it is older than finance: when you say a car costs 3,000 hours of work, hours are your numeraire.

In pricing, the numeraire has to be a **traded asset with a strictly positive price**. Traded, so you can hold it and divide by it without introducing an arbitrage. Strictly positive, because you are going to divide by it and the answer must never blow up or flip sign. A money-market account, a zero-coupon bond and a share of a non-defaultable stock all qualify. A futures contract does not, because it is worth zero at inception, and neither does a swap.

### The rule that makes a numeraire useful

Fix a numeraire $N_t$. Then there exists a probability measure $Q^N$, called the measure associated with $N$, under which the price of **every** traded asset divided by $N$ is a martingale:

$$\frac{V_t}{N_t} = \mathbb{E}^{Q^N}_t\!\left[\frac{V_T}{N_T}\right]$$

Read the left side as the price today quoted in numeraire units, and the right side as what you expect it to be worth in numeraire units later. A martingale is a process whose best forecast of tomorrow is today, covered from zero in [martingales and the risk-neutral measure](/blog/trading/math-for-quants/martingales-risk-neutral-measure-math-for-quants).

Take $N_t = B_t$, the money-market account. Then $V_t/B_t$ is the discounted price, $Q^B$ is what everybody calls the risk-neutral measure $Q$, and the equation above is the familiar $V_0 = \mathbb{E}^Q[V_T/B_T]$. There is nothing else in it. "Discounting" was always just "quoting in money-market units," and the risk-neutral measure was always just the measure that makes that particular quotation a martingale.

Every worked example below uses round, assumed inputs. They are illustrative arithmetic chosen to make the mechanism visible, not market quotes.

#### Worked example 1: one call, two yardsticks, one price

A stock trades at \$100. In one year it is either \$120 or \$90. A money-market account turns \$1 into \$1.05. You own a one-year call struck at \$100 on 10,000 shares, so \$1m of stock exposure.

**Yardstick one, the money-market account.** For $S_t/B_t$ to be a martingale we need ${(100)(1.05) = 120q + 90(1-q)}$, so ${105 = 30q + 90}$ and $q = 0.500$. The call pays \$20 up and \$0 down:

$$V_0 = \frac{0.500 \times \$20 + 0.500 \times \$0}{1.05} = \frac{\$10}{1.05} = \$9.523810 \text{ per share}$$

On 10,000 shares, \$95,238.10.

**Yardstick two, the stock.** Now quote everything in shares. The Radon-Nikodym weight is the ratio of how the two numeraires grew, so in the up state it is ${(120/100)/1.05 = 8/7}$ and in the down state ${(90/100)/1.05 = 6/7}$. The stock-measure probabilities are therefore $q_u = 0.5 \times 8/7 = 4/7$ and $q_d = 0.5 \times 6/7 = 3/7$, which sum to 1 as they must. The call's payoff **in shares** is ${20/120 = 1/6}$ up and 0 down:

$$\frac{V_0}{S_0} = \frac{4}{7}\cdot\frac{1}{6} + \frac{3}{7}\cdot 0 = \frac{2}{21}, \qquad V_0 = \$100 \times \frac{2}{21} = \$9.523810$$

On 10,000 shares, \$95,238.10 again, and not approximately: ${10/1.05 = 1000/105 = 200/21}$ is an exact identity.

*The lesson: the probabilities went from 0.500 to 4/7, a 14% relative move, and the price did not budge by a cent.*

## The machinery: Radon-Nikodym as a ratio of numeraires

The reason that worked is worth stating precisely, because it is the one formula you actually carry around. For any two admissible numeraires $N$ and $M$, the density that translates between their measures is:

$$\left.\frac{dQ^N}{dQ^M}\right|_T = \frac{N_T/N_0}{M_T/M_0}$$

That is it. The [Radon-Nikodym derivative](/blog/trading/math-for-quants/radon-nikodym-densities-math-for-quants), which in the abstract is a per-outcome reweighting, is here just the ratio of how much each yardstick grew along that path. In worked example 1, $N$ was the stock and $M$ the money-market account, and the density was ${(S_T/S_0)/1.05}$, which is exactly the ${8/7}$ and ${6/7}$ you saw.

In continuous time this density is an exponential martingale, and feeding it through **Girsanov's theorem** tells you what happens to the Brownian motion driving your model: it picks up a drift equal to the volatility of the numeraire ratio. Nothing about the volatility changes, only the drift. The full derivation, with the market price of risk and why $\mu$ disappears from an option price, is in [Girsanov's theorem and the change of measure](/blog/trading/math-for-quants/girsanov-change-of-measure-math-for-quants).

### Why the price is invariant while the measure is not

Two things are going on at once, and only one of them is real.

The **price** is a number the market will pay, and it is invariant by construction: $V_t/N_t$ being a $Q^N$-martingale for every $N$ is a theorem about one no-arbitrage structure, not a new assumption per numeraire. Geman, El Karoui and Rochet proved the general statement in 1995. Numeraire and measure move in lockstep, and their product leaves the price alone.

The **measure** is bookkeeping. A $Q$-probability is not a forecast of anything. It is a price of a digital payoff divided by a discount factor, and when you change the discount factor you change the probability. Asking "what is the true probability" of a state is like asking whether a distance is truly 1.6 or truly 1.0 without saying kilometres or miles.

That is also why the trick is *safe*: a bad choice of numeraire cannot break a price, it can only make the algebra longer than it needed to be.

![A four row comparison matrix listing the money market account, the zero coupon bond, the annuity and the asset itself, with the measure each induces, what becomes a martingale under it, and which products it makes easy](/imgs/blogs/change-of-numeraire-math-for-quants-2.webp)

The first three rows are the workhorses: cash for vanillas, a bond maturing at $T$ when the discount factor is itself random, an asset when the payoff compares two assets. The annuity row is the swaption special case, and it comes at the end.

## The forward measure, where the trick pays rent

Here is the problem the forward measure solves. The general pricing formula is an expectation of a **product**: the random discount factor $D_T = 1/B_T$ times the payoff $X$. Expectations do not pass through products:

$$\mathbb{E}^Q[D_T X] = \mathbb{E}^Q[D_T]\,\mathbb{E}^Q[X] + \operatorname{Cov}^Q(D_T, X)$$

For an equity option this rarely matters: rates and the stock are close to independent over a short horizon, so the covariance is small. For a payoff *driven by rates* it is structural, and it always has the same sign, because a rate option pays most in exactly the states where rates are high and the discount factor is therefore low.

Switching to the $T$-forward measure $Q^T$, whose numeraire is the zero-coupon bond $P(t,T)$, makes that term vanish into the probabilities. Since $P(T,T) = 1$, the numeraire at maturity is a constant and the pricing formula collapses to:

$$V_0 = \mathbb{E}^{Q}\!\left[D_T X\right] = P(0,T)\,\mathbb{E}^{Q^T}\!\left[X\right]$$

A clean product of today's discount factor and one expectation, with no correction left to forget. The measure also earns its name: under $Q^T$ the $T$-forward price of any asset is a martingale, so forward prices are their own expectations.

#### Worked example 2: a caplet on \$100m, and the term the shortcut drops

A caplet on \$100m notional, strike 4.00%, on the one-year rate fixed at year 1 and paid at year 2. Today's one-year rate is 4.00% and is known. At year 1 the new one-year rate is 6.00% or 2.00%, each with $Q$-probability 0.500.

The payoff at year 2 is \$100m times the excess over 4.00%, so \$2,000,000 in the up state and \$0 in the down state. The discount factor from today to year 2 is path-dependent. Keep it as an exact fraction, because every identity below is exact and rounding one step early breaks all of them:

$$D_u = \frac{1}{1.04 \times 1.06} = \frac{625}{689} = 0.907112, \qquad D_d = \frac{1}{1.04 \times 1.02} = \frac{625}{663} = 0.942685$$

**Under the money-market measure**, price the product directly:

$$V_0 = \tfrac{1}{2}\cdot\tfrac{625}{689}\cdot\$2{,}000{,}000 + \tfrac{1}{2}\cdot\tfrac{625}{663}\cdot\$0 = \frac{\$625{,}000{,}000}{689} = \$907{,}111.76$$

**Under the forward measure**, first get today's two-year zero:

$$P(0,2) = \mathbb{E}^Q[D_2] = \tfrac{1}{2}\big(\tfrac{625}{689} + \tfrac{625}{663}\big) = \tfrac{2500}{2703} = 0.924898.$$
 The change-of-measure density is the ratio of numeraires, which in the up state is $\tfrac{625}{689} \div \tfrac{2500}{2703} = \tfrac{51}{52}$ exactly. So the forward-measure probabilities are $\tfrac{1}{2}\cdot\tfrac{51}{52} = \tfrac{51}{104} = 0.490385$ up and $\tfrac{53}{104} = 0.509615$ down, which makes $\mathbb{E}^{Q^2}[X] = \tfrac{51}{104} \times \$2{,}000{,}000 = \$980{,}769.23$. Then:

$$V_0 = P(0,2)\,\mathbb{E}^{Q^2}[X] = \tfrac{2500}{2703} \times \$980{,}769.23 = \$907{,}111.76$$

Same price, to the cent, from visibly different probabilities.

**The naive calculation** is the one a careful person does by accident: discount the expected payoff at today's curve. That is $\tfrac{2500}{2703} \times \$1{,}000{,}000 = \$924{,}898.26$. It is high by exactly a factor of $\tfrac{52}{51}$, which is 1.96%, or about \$17,787 of air on this one contract.

![A two state fan for the caplet showing the up state discount factor of 0.907112 with a two million dollar payoff and the down state discount factor of 0.942685 with a zero payoff, feeding three result boxes: the money market measure and the forward measure both at nine hundred seven thousand one hundred eleven dollars and seventy six cents in green, and the naive shortcut at nine hundred twenty four thousand eight hundred ninety eight dollars and twenty six cents in red](/imgs/blogs/change-of-numeraire-math-for-quants-3.webp)

Notice what the naive calculation got wrong. It did not use the wrong formula. $P(0,T)$ times an expected payoff is exactly the right shape. It used the right formula under the **wrong measure**, and the entire error is the covariance it therefore dropped. Here that covariance is exactly $-\$17{,}786.51$, and you can see where it comes from: under $Q$ the expected one-year rate at year 1 is 4.0000%, while under $Q^2$ it is $\tfrac{51}{104}(6\%) + \tfrac{53}{104}(2\%) = \tfrac{103}{2600} = 3.9615\%$, which is precisely today's forward rate. The 3.85 basis point gap between them is the whole correction.

*The lesson: the forward measure does not change the price, it changes where the covariance term is stored, from a correction you have to remember into probabilities you cannot forget.*

## Margrabe: the payoff that names its own numeraire

Some payoffs tell you which numeraire to use. The clearest is Margrabe's 1978 exchange option: the right, at time $T$, to give up asset 2 and receive asset 1, worth $\max(S_1(T) - S_2(T), 0)$.

Priced under the money-market measure this is unpleasant: two correlated lognormals, a two-dimensional integral over a region bounded by a diagonal, and an interest rate threaded through both drifts. Priced with **asset 2 as the numeraire** it is three lines. Divide the payoff by $S_2(T)$:

$$\frac{\max(S_1(T) - S_2(T), 0)}{S_2(T)} = \max\!\left(\frac{S_1(T)}{S_2(T)} - 1, 0\right)$$

The ratio $Z_t = S_1(t)/S_2(t)$ is a ratio of a traded asset to the numeraire, so under $Q^{S_2}$ it is a martingale by definition. The payoff is now a plain call on $Z$ struck at 1. One asset, one strike, zero interest rate, because the numeraire already absorbed all the discounting. The answer is Black-Scholes with the **spread volatility**:

$$V_0 = S_1(0)\,N(d_1) - S_2(0)\,N(d_2), \qquad \sigma^2 = \sigma_1^2 + \sigma_2^2 - 2\rho\,\sigma_1\sigma_2$$

$$d_1 = \frac{\ln\!\big(S_1(0)/S_2(0)\big) + \tfrac{1}{2}\sigma^2 T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

No $r$ appears. That is not an approximation and not a special case: the rate genuinely cancels, because you never converted anything into cash.

#### Worked example 3: swapping \$50m of value into growth

A portfolio holds \$50m of a value index at 100.00, which is 500,000 units, and wants the right in one year to swap the whole sleeve into a growth index, also at 100.00. Growth vol 25%, value vol 18%, correlation 0.65.

$$\sigma^2 = 0.25^2 + 0.18^2 - 2(0.65)(0.25)(0.18) = 0.0625 + 0.0324 - 0.0585 = 0.0364$$

so $\sigma = 19.08\%$. The two levels are equal so the log term is zero, and $d_1 = \sigma/2 = 0.095394$, $d_2 = -0.095394$. Then $N(d_1) = 0.537999$ and $N(d_2) = 0.462001$, giving

$$V_0 = 100 \times (0.537999 - 0.462001) = \$7.599805 \text{ per unit}$$

On 500,000 units that is **\$3,799,903**, or 7.60% of the \$50m sleeve. As a check, a one-year at-the-money Black-Scholes call on a \$100 underlying at that same spread volatility with $r=0$ returns \$7.599805 as well, matching to twelve digits, which is the collapse working.

Now move correlation to 0.90 and nothing else. The spread variance falls to ${0.0625 + 0.0324 - 0.0810 = 0.0139}$, so $\sigma = 11.79\%$, and the option is worth \$4.700737 per unit, or **\$2,350,369**. The same trade, the same vols, the same notional, and **\$1,449,534 less** because the two sleeves now move together.

![Line chart of the exchange option value in millions of dollars against the correlation between the two assets, falling from six point one two million at zero correlation through three point eight zero million at correlation zero point six five to one point four zero million at correlation one](/imgs/blogs/change-of-numeraire-math-for-quants-4.webp)

*The lesson: correlation enters the price through exactly one scalar, the spread volatility, so an exchange option is a pure correlation trade wearing a Black-Scholes costume.*

## The practitioner's version: quanto and the annuity

Two places where a desk meets this daily.

### The quanto correction is a numeraire change

A **quanto** pays in your currency off a foreign underlying, converted at a fixed rate agreed today rather than the market rate at maturity. That fixed conversion breaks the usual hedge, and the repair is a drift term.

The derivation is two lines of numeraire change. Let $X_t$ be the exchange rate quoted as domestic per unit of foreign. The foreign money-market account, valued domestically, is $X_t B^f_t$, so the density between the two currencies' risk-neutral measures is $\left(X_0/X_T\right)e^{(r_d-r_f)T}$: the exchange rate itself is the Radon-Nikodym derivative. Girsanov then shifts the Brownian motion driving the foreign asset, and its drift under the *domestic* measure comes out as

$$\mu_S = r_f - q - \rho\,\sigma_S\,\sigma_X$$

with $\rho$ the correlation between the foreign asset and $X$. The last term is the **quanto correction**, and it exists only because you changed currencies.

#### Worked example 4: a quanto call on \$50m

A quanto call on a foreign equity index at 2,000 points, struck at 2,000, one year, paying \$25,000 per index point, so \$50m of notional. Assumed inputs: foreign rate 1.00%, dividend yield 2.00%, domestic rate 4.00%, index vol 22%, FX vol 11%, and correlation $-0.40$ because the index tends to rally when its own currency weakens.

The correction is $-(-0.40)(0.22)(0.11) = +0.968\%$ per year, so the drift is ${-1.00\% + 0.968\% = -0.032\%}$ rather than $-1.00\%$. That moves the forward:

- ignoring the correction: $2000 \times e^{-0.0100} = 1{,}980.10$, and the call is worth **\$3,952,180**
- with the correction: $2000 \times e^{-0.00032} = 1{,}999.36$, and the call is worth **\$4,199,454**

The gap is **\$247,274**. That is only 0.49% of notional, which is why it survives a casual review, but it is 5.9% of the option's value, which is why it does not survive a month of trading against someone who computed it.

![Before and after ladder showing the quanto priced without the correction at a drift of minus one percent, a forward of 1980.10 and a value of three point nine five million dollars, against the corrected version at a drift of minus zero point zero three two percent, a forward of 1999.36 and a value of four point two zero million dollars, with a gap of two hundred forty seven thousand two hundred seventy four dollars](/imgs/blogs/change-of-numeraire-math-for-quants-5.webp)

*The lesson: a quanto correction is not a fudge factor bolted onto Black-Scholes, it is the drift you get for free when you write down the density between two currencies' numeraires.*

### Why swaptions quote an annuity measure

A payer swaption on a swap with fixed-leg dates $T_1,\dots,T_n$ pays $A_T\max(R_T - K, 0)$, where the forward swap rate and the annuity are

$$R_t = \frac{P(t,T_0) - P(t,T_n)}{A_t}, \qquad A_t = \sum_{i=1}^{n}\tau_i\,P(t,T_i)$$

The annuity is a positive portfolio of zero-coupon bonds, so it is an admissible numeraire, and $R_t$ is a ratio of traded assets to it, so $R_t$ is a martingale under $Q^A$. Divide the payoff by $A_T$ and you are left with $\max(R_T - K, 0)$, a plain call on a martingale. The price is $A_0$ times the Black formula on the swap rate. That, and nothing deeper, is why the swaption market quotes a **Black volatility on the swap rate** rather than a volatility on bond prices. Brigo and Mercurio build the LIBOR and swap market model family on this observation, and the short-rate models that feed it are surveyed in [short-rate models](/blog/trading/quantitative-finance/short-rate-models-vasicek-hull-white).

## Common misconceptions

**"The risk-neutral measure is the real probability, adjusted for risk."** It is not an adjusted forecast of anything. It is the bookkeeping that falls out of quoting prices in money-market units, and a different unit gives a different set of probabilities with equal standing. In worked example 2 the expected one-year rate was 4.0000% under $Q$ and 3.9615% under $Q^2$. Neither is a forecast of the rate. Both are prices in disguise.

**"Changing numeraire changes the price."** Three separate computations in this post gave \$907,111.76 for the same caplet, and two gave \$9.523810 for the same call. Invariance is a theorem, proved in general by Geman, El Karoui and Rochet, not a numerical coincidence. What changes is which term you have to remember and which one is absorbed into the probabilities.

**"The forward measure is a bond-market technicality."** It is needed the moment the discount factor is random *and* correlated with the payoff. That is every rate option, every credit product where default and rates co-move, every convexity-adjusted payment date, and increasingly every equity option collateralised in a currency whose rate co-moves with the underlying. The caplet above is a toy and still produced a 1.96% error.

**"Margrabe needs a correlation model layered on top of Black-Scholes."** It needs one number. Correlation enters only through $\sigma^2 = \sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2$, and once you have that scalar the pricing is a single univariate call. The modelling difficulty in exchange options is estimating $\rho$, not pricing given it.

## Sources and further reading

- H. Geman, N. El Karoui and J.-C. Rochet, "Changes of Numeraire, Changes of Probability Measure and Option Pricing," *Journal of Applied Probability* 32(2), 1995, pp. 443 to 458. The general theorem, including the admissibility conditions on a numeraire.
- W. Margrabe, "The Value of an Option to Exchange One Asset for Another," *Journal of Finance* 33(1), 1978, pp. 177 to 186. The original exchange-option paper.
- F. Jamshidian, "An Exact Bond Option Formula," *Journal of Finance* 44(1), 1989, pp. 205 to 209. The decomposition that makes bond options tractable under the forward measure.
- D. Brigo and F. Mercurio, *Interest Rate Models: Theory and Practice*, 2nd edition, Springer, 2006. Chapter 2 on change of numeraire, plus the quanto and swap-market-model appendices.
- J. Hull, *Options, Futures, and Other Derivatives*. The quanto chapter derives the drift correction used in worked example 4.
- On this site: [martingales and the risk-neutral measure](/blog/trading/math-for-quants/martingales-risk-neutral-measure-math-for-quants), [Girsanov's theorem](/blog/trading/math-for-quants/girsanov-change-of-measure-math-for-quants), [Radon-Nikodym derivatives](/blog/trading/math-for-quants/radon-nikodym-densities-math-for-quants), [Feynman-Kac and the Black-Scholes PDE](/blog/trading/math-for-quants/feynman-kac-black-scholes-pde-math-for-quants), and the [Black-Scholes deep dive](/blog/trading/quantitative-finance/black-scholes) for the Margrabe and Garman-Kohlhagen variants.

## In the interview room and on the desk

The question is usually phrased as a pricing request, not a theory question: *"Price me an option to exchange one asset for another."* Sometimes it is dressed up as a merger arb payoff, a best-of basket, or "the right to switch from our value sleeve into our growth sleeve."

A weak answer starts computing. It writes down two correlated geometric Brownian motions, sets up a double integral over the region where $S_1$ beats $S_2$, and grinds. That answer is not wrong, it is just twenty minutes long, and the interviewer has already learned what they wanted to know.

A strong answer picks the numeraire **in the first sentence**: "Use asset 2 as the numeraire." Then the rest follows in order, and each step is one line. Divide the payoff by $S_2(T)$ and it becomes $\max(Z_T - 1, 0)$ with $Z = S_1/S_2$. Under $Q^{S_2}$, $Z$ is a ratio of a traded asset to the numeraire, hence a martingale, hence driftless. Its volatility is $\sqrt{\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2}$ by Ito on the quotient. So it is a Black-Scholes call struck at 1 with zero rate, and the price is $S_1(0)N(d_1) - S_2(0)N(d_2)$. If the interviewer is any good, the next question is "where did the interest rate go," and the answer is that you never converted into cash, so there was nothing to discount.

The trap is treating $Q$ as privileged. A candidate who believes the risk-neutral measure is *the* measure will insist on discounting the Margrabe answer, will hunt for the $r$ that is supposed to be in the formula, and will be unable to say what changes when you switch measures and what does not. The same blind spot shows up as forgetting the quanto correction, and as discounting an expected payoff at today's curve when the payoff is rate-driven.

Jane Street weights the reasoning rather than the formula: expect to be asked *why* the ratio is a martingale. Citadel and Citadel Securities probe it through multi-asset and correlation books. Any rates or exotics seat treats fluency between the forward, annuity and share measures as table stakes, because the quoting conventions on their own screens are measure choices.

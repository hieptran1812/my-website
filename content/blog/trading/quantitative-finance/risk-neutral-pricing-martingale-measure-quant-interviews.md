---
title: "Risk-neutral pricing and the martingale measure for quant interviews"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A first-principles deep dive into why derivatives are priced as the discounted expected payoff under a risk-neutral measure, built on a single binomial tree, with worked dollar examples and five fully solved interview problems."
tags:
  [
    "risk-neutral-pricing",
    "martingale-measure",
    "no-arbitrage",
    "replication",
    "binomial-model",
    "change-of-measure",
    "girsanov",
    "monte-carlo",
    "quant-interviews",
    "derivatives-pricing",
    "quantitative-finance",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A derivative's fair price is the discounted average of its future payoffs, but the average is taken with a special set of "pricing probabilities" called the risk-neutral measure Q, not your real-world beliefs. The reason is pure no-arbitrage plus replication, and the whole thing fits on one binomial tree.
>
> - A one-step stock that goes from \$100 to either \$120 or \$80 lets you copy any option with shares plus a bond. The copy's cost *is* the option's price — no probabilities needed yet.
> - When you rearrange that price algebraically, your real-world probability `p` cancels out and a new number `q` appears. Here `q = 0.5`. That `q` is the risk-neutral probability.
> - Under `q`, the expected stock return equals the risk-free rate, and the discounted price is a *martingale*: its expected future value equals its value today.
> - The two fundamental theorems: no-arbitrage means a risk-neutral measure *exists*; market completeness means it is *unique*.
> - In continuous time the same recipe becomes `V_0 = e^(-rT) E^Q[payoff]`, and a real pricing desk computes that expectation by Monte Carlo — simulating paths with drift `r`, never the real drift.
> - The single fact to remember: **price = discounted expected payoff under Q**, and `q` is a pricing weight, not a forecast.

Here is a question that sounds simple and ends most junior interviews: *why do we price a derivative as a discounted expectation, and why under that strange "risk-neutral" probability instead of the real one?* The honest answer is not "because the textbook says so." It is that the price is forced on you by two ideas a market participant cannot escape — you cannot create money out of nothing (no-arbitrage), and you can build a copy of the option out of simpler instruments (replication). Everything else — the measure Q, the martingale property, Girsanov, Monte Carlo under Q — is a consequence of those two ideas. A strong candidate can derive the whole story on a single one-step tree and never wave their hands once.

![Risk-neutral pricing collapses to a discounted average of payoffs taken under a special probability measure Q](/imgs/blogs/risk-neutral-pricing-martingale-measure-quant-interviews-1.png)

The diagram above is the mental model, and it is worth staring at before any math. Read it left to right: start with the option's payoff at expiry, take its *average* — but average it under the risk-neutral measure Q, not under your real-world beliefs — then *discount* that average back to today at the risk-free rate. The output is the price. The single equation `V_0 = e^(-rT) E^Q[payoff]` is the entire field of arbitrage-free pricing compressed into one line. The rest of this article unpacks every word of it: what "no-arbitrage" buys you, why replication forces a unique price, where the measure Q comes from, why your real probability `p` evaporates, what "martingale" means in plain English, and how a desk actually computes that expectation in production. We will define every term from zero, work each idea in round dollar numbers, then end with five fully solved interview problems and the misconceptions that trip people up.

This is the conceptual backbone behind [Black-Scholes](/blog/trading/quantitative-finance/black-scholes), [derivatives pricing](/blog/trading/quantitative-finance/derivatives-pricing), and the [expected-value techniques](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews) that show up across quant interviews. If you only ever truly understand one pricing idea, make it this one — it is the load-bearing wall of the whole subject.

## Foundations: the four words you must define before anything else

A reader with no finance background can follow this entire article, but only if four ideas are nailed down first. None of them require a single formula. Each one is just a careful definition of a word the rest of the post will lean on hundreds of times.

**A derivative** is a contract whose payoff depends on the price of something else. The "something else" is the *underlying* — usually a stock, but it could be an index, a currency, or a commodity. The simplest derivative is a *call option*: a contract that gives you the right, but not the obligation, to buy one share at a fixed price `K` (the *strike*) at a fixed future date `T` (the *expiry*). If at expiry the stock is worth `S_T`, the call's payoff is `max(S_T − K, 0)` — you exercise and pocket `S_T − K` if the stock finished above the strike, and you walk away with \$0 if it finished below. The whole job of pricing is to answer: what is a fair amount to pay *today* for that future payoff?

**Arbitrage** is free money: a trade that costs nothing to put on, can never lose, and has a positive chance of paying off. Concretely, if you can assemble a portfolio that costs \$0 today and pays \$5 in some future state while never going negative, that is an arbitrage. The bedrock assumption of all of pricing is that *arbitrage does not exist in a functioning market* — or more precisely, that if it ever appears, traders pounce on it so fast it vanishes. This is not a claim that markets are perfectly efficient or that prices are "right." It is a far weaker claim: you cannot get a guaranteed something for nothing. That single weak claim is enough to pin down option prices exactly.

**Replication** is building a copy. If you can assemble a portfolio of simpler instruments — say, some shares of stock and a loan — whose payoff is *identical* to the option's payoff in every possible future state, you have *replicated* the option. The replicating portfolio is a perfect substitute: anywhere the option pays \$20, your portfolio pays \$20; anywhere the option pays \$0, your portfolio pays \$0.

**The law of one price** is the bridge between the two: if two portfolios produce the same payoff in every future state, they must cost the same today. Why? Because if they did not, you would buy the cheap one, sell the expensive one, and collect the price difference for free — and since the future payoffs cancel exactly, you carry no risk. That is an arbitrage, which we have assumed away. So no-arbitrage *forces* the law of one price, and the law of one price *forces* the option's price to equal the cost of its replica.

![No-arbitrage and the law of one price force every replicable payoff to a single fair price](/imgs/blogs/risk-neutral-pricing-martingale-measure-quant-interviews-6.png)

The figure above is the logical skeleton of the whole subject. No-arbitrage sits at the root. It branches two ways. Down one branch: the law of one price means "same payoff, same price," so if we can replicate a call with stock and a bond, the call's price *must* equal the cost of that replica. Down the other branch: if anyone quotes the option away from that replica cost, you buy the cheap leg, sell the dear leg, and lock in a risk-free profit instantly — which the market eliminates. Notice what is *not* in this picture: nobody's opinion about whether the stock will go up. We have not used a single probability yet, and we have already cornered the price. Hold onto that — it is the punchline the whole article builds toward.

Two more small definitions and we are fully equipped. The *risk-free rate* `r` is the interest you earn on a totally safe loan — think of a government bond or a bank deposit you are certain will be repaid. A *bond* in our toy world is just that safe loan: lend \$1 today at rate `r`, and at time `T` you get back `e^(rT)` dollars (continuous compounding) or `(1 + r)` dollars over one period (simple compounding). *Discounting* is running that backwards: a safe \$1 to be received at time `T` is worth `e^(−rT)` dollars today, because that is exactly how much you would need to deposit now to grow into \$1. Throughout the worked examples we will often set `r = 0` to keep the arithmetic clean — when the risk-free rate is zero, a dollar tomorrow is worth a dollar today, and discounting does nothing. That is a feature, not a cheat: it lets you see the probability machinery without the interest-rate arithmetic in the way, and we will switch `r` back on once the core idea is clear.

## The one-period binomial model: where the whole theory lives

Everything important about risk-neutral pricing is visible in the smallest possible model of an uncertain market: one stock, one bond, one time step, and exactly two possible futures. This is the *one-period binomial model*, and a remarkable fact about it is that it is not a toy — it contains the complete logical structure of continuous-time pricing, just without the calculus. If you understand the tree, you understand Black-Scholes; the continuous case is the tree taken to a limit of infinitely many infinitesimal steps.

![A 100 dollar stock that moves up to 120 or down to 80 in one step is the entire model](/imgs/blogs/risk-neutral-pricing-martingale-measure-quant-interviews-2.png)

Here is our world, drawn above. Today the stock trades at `S_0 = `\$100. Over one time step it can do exactly one of two things: jump *up* to `S_up = `\$120 (a +20% return) or fall *down* to `S_down = `\$80 (a −20% return). We write the up-factor `u = 120/100 = 1.2` and the down-factor `d = 80/100 = 0.8`. There is a real-world probability `p` that the up state occurs — maybe you think the stock is a great company and `p = 0.7`, maybe you are bearish and `p = 0.3`. Crucially, *we are not going to need `p`*. Set the risk-free rate to `r = 0` for now, so the bond does nothing: \$1 today is \$1 at expiry.

We want to price a one-period call with strike `K = `\$100. Its payoff is `max(S_T − 100, 0)`. In the up state the stock is \$120, so the call pays `120 − 100 = `\$20. In the down state the stock is \$80, so the call pays \$0. The whole pricing problem is: what is this call worth today?

#### Worked example: replicate the call with stock and a bond

We will copy the call's payoff exactly using `Δ` (delta) shares of stock plus `B` dollars in the bond. We want our portfolio to pay \$20 in the up state and \$0 in the down state — matching the call.

In the up state the portfolio is worth `Δ × 120 + B` (the bond is unchanged because `r = 0`). In the down state it is worth `Δ × 80 + B`. We need both to match the call:

$$
\Delta \cdot 120 + B = 20 \qquad \Delta \cdot 80 + B = 0
$$

Subtract the second equation from the first: `Δ × (120 − 80) = 20`, so `Δ × 40 = 20`, giving `Δ = 0.5`. The general formula is worth memorizing — delta is the *spread of the payoff divided by the spread of the stock*:

$$
\Delta = \frac{C_{up} - C_{down}}{S_{up} - S_{down}} = \frac{20 - 0}{120 - 80} = 0.5
$$

Plug `Δ = 0.5` back into the down equation: `0.5 × 80 + B = 0`, so `B = −40`. The negative sign means we *borrow* \$40 (a short bond position). So the replicating portfolio is: **buy 0.5 shares and borrow \$40.**

What does that cost to assemble *today*? Half a share costs `0.5 × 100 = `\$50, and borrowing \$40 gives you \$40 of cash, so the net cost is `50 − 40 = `\$10. By the law of one price, since this portfolio pays exactly what the call pays in both states, **the call must be worth \$10 today.** If `r = 0` and any other answer is quoted, there is free money on the table.

![Holding half a share and borrowing 40 dollars pays exactly the call payoff in both states](/imgs/blogs/risk-neutral-pricing-martingale-measure-quant-interviews-3.png)

The figure above checks the replica in both states so you can see the match with your own eyes. In the up state: the half-share is worth `0.5 × 120 = `\$60, you repay the \$40 you borrowed, and your portfolio is worth `60 − 40 = `\$20 — exactly the call payoff. In the down state: the half-share is worth `0.5 × 80 = `\$40, you repay \$40, and your portfolio is worth \$0 — again exactly the call payoff. Two states, two matches. The copy is perfect, so the price is forced: \$10. **The intuition this teaches: an option is not a gamble you price by guessing odds — it is a packaged combination of stock and borrowing, and its price is just the cost of that package.**

#### Worked example: the same price as a discounted expectation

Now comes the algebraic magic that the whole field is built on. We just found the price by computing the cost of a replica: \$10. Let us *rearrange* that calculation and watch a new object appear.

The replica costs `Δ × S_0 + B`. Substitute the formulas for `Δ` and `B`. After the dust settles (we will do this carefully), the price can be written as

$$
C_0 = e^{-rT}\big[\, q \cdot C_{up} + (1 - q) \cdot C_{down} \,\big] \quad\text{where}\quad q = \frac{e^{rT} S_0 - S_{down}}{S_{up} - S_{down}}.
$$

Do not take this on faith — let us derive `q` for our numbers and confirm it reproduces \$10. With `r = 0` (so `e^(rT) = 1`), `S_0 = 100`, `S_up = 120`, `S_down = 80`:

$$
q = \frac{1 \cdot 100 - 80}{120 - 80} = \frac{20}{40} = 0.5.
$$

Now plug `q = 0.5` into the pricing formula with the call payoffs `C_up = 20`, `C_down = 0`:

$$
C_0 = 1 \cdot \big[\, 0.5 \cdot 20 + 0.5 \cdot 0 \,\big] = 10.
$$

It matches. **The price equals the discounted, `q`-weighted average of the payoffs.** The weight `q = 0.5` looks like a probability — it is between 0 and 1, and `q` and `1 − q` sum to 1 — and we call it the *risk-neutral probability*. But look again at where it came from: it is built purely from the stock prices `S_0`, `S_up`, `S_down` and the rate `r`. **Your real-world belief `p` is nowhere in it.** The intuition this teaches: pricing by expectation and pricing by replication are the *same calculation* wearing two different costumes; the "probability" `q` is just the replication algebra rewritten to look like an average.

#### Worked example: price a digital option the same way

To prove the recipe is general and not a fluke of the vanilla call, price a *digital option* (also called a binary or cash-or-nothing option). A digital call pays a fixed \$1 if the stock finishes above the strike `K = `\$100, and \$0 otherwise. In the up state (\$120 > \$100) it pays \$1; in the down state (\$80 < \$100) it pays \$0.

We could replicate it from scratch — solve `Δ × 120 + B = 1` and `Δ × 80 + B = 0`, giving `Δ = 0.025` and `B = −2`, for a cost of `0.025 × 100 − 2 = 0.50`. But we already have the shortcut. The risk-neutral probability `q = 0.5` does not depend on the payoff — it is a property of the *stock*, computed once. So the digital's price is simply the discounted `q`-weighted average of its payoffs:

$$
D_0 = e^{-rT}\big[\, q \cdot 1 + (1 - q) \cdot 0 \,\big] = 1 \cdot 0.5 = 0.50.
$$

The digital call is worth \$0.50 today. Both methods agree, as they must.

![A digital call pays one dollar above the 100 dollar strike and zero below, so its price is just q](/imgs/blogs/risk-neutral-pricing-martingale-measure-quant-interviews-12.png)

The figure shows the digital's payoff: a flat \$0 below the strike (the red region, the down state) and a flat \$1 above it (the green region, the up state), with a sharp jump exactly at `K = `\$100. Because a digital pays \$1 precisely when the stock ends up, its price equals the risk-neutral probability of ending up (times the discount factor). In our world that is `q = 0.5`, so the price is \$0.50. **The intuition this teaches: a digital option is a direct bet on `q` itself — its price *is* the risk-neutral probability of finishing in the money, which is exactly why interviewers love it as a way to ask "what is `q`?" without saying so.**

## Why the real-world probability `p` drops out entirely

This is the single most counterintuitive fact in the subject, and a favorite interview probe. You spent the last section watching `p` never appear, and you might think it is an artifact of our tidy numbers. It is not. Let us prove that the price is *completely independent of `p`* by computing it under two wildly different real-world beliefs and getting the same answer.

#### Worked example: two different real probabilities, identical price

Suppose Trader A is a raging bull: she believes the up state is nearly certain, `p_A = 0.9`. Trader B is a deep bear: he believes the up state is unlikely, `p_B = 0.2`. They disagree violently about the stock's future. Do they disagree about the call's price?

Trader A's *real-world expected payoff* of the call is `0.9 × 20 + 0.1 × 0 = `\$18. Trader B's is `0.2 × 20 + 0.8 × 0 = `\$4. Those are miles apart — \$18 versus \$4. If pricing were "discount the real expected payoff," the two traders would quote the call at \$18 and \$4 respectively, and the market would be chaos.

But pricing is *not* discount-the-real-expected-payoff. Both traders can replicate the call with 0.5 shares and a \$40 loan, and that replica costs \$10 regardless of what either believes about `p`. If Trader A tried to sell the call at \$18, Trader B would buy it from her at, say, \$11, build the \$10 replica, and lock in \$1 of risk-free profit — and Trader A would have handed him \$18 minus \$11 minus... the point is the \$18 quote cannot survive. The replica cost is \$10 for both of them. **They must both quote \$10, no matter how much they disagree about the real odds.**

![The price uses q = 0.5, never the real-world p, so two different real beliefs give the same price](/imgs/blogs/risk-neutral-pricing-martingale-measure-quant-interviews-4.png)

The table above lays out why. There are three quantities floating around. The *real probability `p`* is your honest forecast — it can be 0.9, 0.2, anything — and it *does not enter the price at all*; it cancels in the replication algebra. The *risk-neutral probability `q`* is a pricing weight derived from no-arbitrage, fixed at `q = 0.5` here, and it is the *only* probability the price uses. The *discount rate `r`* (zero in this example) enters through the discount factor. The headline a candidate should be able to say out loud: *the real-world drift of the stock is already baked into today's stock price, so re-using your forecast `p` would double-count it.* The market has done the forecasting for you; the option price only needs the stock price and the rate.

Here is the deeper reason `p` cancels, stated cleanly. The replication argument never asks "how likely is the up state?" It only asks "*if* the up state happens, do the option and the replica pay the same? And *if* the down state happens, do they pay the same?" Since the answer is yes in *every* state — regardless of how probable each state is — the two are interchangeable, and interchangeable things cost the same. Probabilities weigh *how much each state matters to an expectation*; replication sidesteps expectations entirely by matching *state by state*. That is why a perfectly hedged book does not care about your market view: it has neutralized the view by construction.

## The risk-neutral measure and the martingale property

We have a number `q` that acts like a probability and prices everything. Time to name it properly and uncover the elegant property that makes it the natural language of pricing.

A *probability measure* is just a consistent assignment of probabilities to all the possible outcomes. The *real-world measure*, written P, assigns the true odds `p` and `1 − p` to the up and down states. The *risk-neutral measure*, written Q, assigns the pricing weights `q` and `1 − q` instead. Both are legitimate probability measures — both assign non-negative weights that sum to 1 — they just answer different questions. P answers "what will actually happen?" Q answers "what weights make today's prices arbitrage-free?" Risk-neutral pricing is the statement that prices are expectations under Q.

The defining property of Q is worth deriving, because it is the cleanest one-line characterization of the measure and a classic interview ask.

#### Worked example: under Q, the expected stock return equals the risk-free rate

Compute the expected stock price at expiry *under Q*, using `q = 0.5`:

$$
E^Q[S_T] = q \cdot S_{up} + (1 - q) \cdot S_{down} = 0.5 \cdot 120 + 0.5 \cdot 80 = 100.
$$

The Q-expected future stock price is \$100 — exactly today's price, because `r = 0` here means a dollar today equals a dollar tomorrow. So under Q the stock is expected to earn the risk-free rate (0%), *not* whatever return a bull or bear believes. This is the meaning of "risk-neutral": Q is the measure under which *every* asset — the risky stock and the safe bond alike — is expected to grow at exactly the risk-free rate. In a world where investors demanded no compensation for bearing risk, the real probabilities *would* be Q. They are not, of course — real investors do demand a risk premium, which is why `p` differs from `q` — but the *pricing* arithmetic behaves as if they did not, and that is the trick.

With a non-zero rate the statement generalizes to `E^Q[S_T] = e^(rT) S_0`: the Q-expected stock price grows at the risk-free rate. Equivalently, divide both sides by `e^(rT)`:

$$
e^{-rT} \, E^Q[S_T] = S_0.
$$

The *discounted* expected future price equals today's price. This is the famous *martingale property*, and it deserves a plain-English definition. A **martingale** is a process whose expected next value, given everything you know now, equals its current value — a "fair game" with no built-in drift up or down. A fair coin-flip betting game where you neither gain nor lose on average is a martingale; your expected bankroll after the next flip is exactly your bankroll now. The headline result of this whole subject: **under the risk-neutral measure Q, the discounted price of any tradable asset is a martingale.** The stock, the bond, the call, the digital — divide each by the bond (i.e. discount it) and under Q it becomes a fair game.

![Under Q the expected discounted future price equals todays price, so the trend is flat](/imgs/blogs/risk-neutral-pricing-martingale-measure-quant-interviews-5.png)

The figure shows the martingale property as a picture. The discounted stock starts at \$100 today. Under Q it goes to \$120 with weight 0.5 or \$80 with weight 0.5, and the `q`-weighted average lands back at \$100 — flat. That flatness *is* the martingale property: no expected drift under Q. Contrast this with the real measure P, under which a bull's `p = 0.9` would give an expected price of `0.9 × 120 + 0.1 × 80 = `\$116 — a clear upward drift, the risk premium the bull expects to earn for holding a risky stock. **The intuition this teaches: switching from P to Q is exactly the act of stripping out the risk premium so that every asset drifts at the same boring risk-free rate, which is what makes discounted prices into fair games and lets you price by simple averaging.**

Why does the martingale view matter beyond elegance? Because it turns pricing into a mechanical procedure. Once you know "discounted prices are Q-martingales," pricing *any* derivative is just: write down its discounted payoff, take the Q-expectation, done. There is no cleverness left — the cleverness was all front-loaded into finding Q. And for a vast class of models, finding Q is a solved problem. That is the payoff of the abstraction: it converts an open-ended "what is this worth?" into a definite integral.

## The two fundamental theorems of asset pricing

We have been speaking as if Q always exists and is always unique. When is that actually true? The answer is two of the most important results in mathematical finance, and an interviewer at a top shop will expect you to state both crisply. They are usually quoted informally — the precise versions involve measure-theoretic conditions we will skip — but the *content* is exactly what a quant needs.

**The First Fundamental Theorem of Asset Pricing:** a market is free of arbitrage *if and only if* there exists at least one risk-neutral measure Q. In words: "no free money" and "a pricing measure exists" are the same statement. This is why risk-neutral pricing is not an assumption you bolt on — it is *equivalent* to the no-arbitrage condition you already believe. If you accept that the market has no free lunches, you have already accepted that some Q exists; pricing by `E^Q` is then not a modeling choice but a logical consequence.

**The Second Fundamental Theorem of Asset Pricing:** an arbitrage-free market is *complete* if and only if the risk-neutral measure Q is *unique*. A market is *complete* when every possible payoff can be replicated by trading the available instruments — in our binomial world, with two states and two tradable assets (stock and bond), we *can* replicate any payoff, so the market is complete and Q is unique (`q = 0.5`, the only weight consistent with the prices). When a market is *incomplete* — too many possible states relative to tradable instruments — there are many valid Q's, and instead of a single price you get a whole *range* of arbitrage-free prices.

![The first theorem links no-arbitrage to a measure existing, the second links completeness to uniqueness](/imgs/blogs/risk-neutral-pricing-martingale-measure-quant-interviews-7.png)

The map above is the whole thing on one screen. Top row, the first theorem: no-arbitrage holds `⇔` a risk-neutral Q exists `⇒` you get discounted-expectation pricing. Middle row, the second theorem: the market is complete `⇔` that Q is unique `⇒` every claim has exactly one price. Bottom row, the failure mode: an *incomplete* market admits many Q's, so a derivative has not a single price but a *band* of arbitrage-free prices, and you need something beyond no-arbitrage — a model, a calibration, a risk preference — to pick one point in the band. A clean way to remember it: **existence ↔ no-arbitrage, uniqueness ↔ completeness.**

Why does completeness matter on a real desk? Because real markets are *not* perfectly complete. You cannot trade continuously, transaction costs are real, and some risks (a sudden jump, a spike in volatility) cannot be hedged with the instruments at hand. That incompleteness is exactly why two banks can hold genuinely different prices for the same exotic and both be "arbitrage-free" — they have chosen different Q's within the allowed band. Completeness is the idealization that makes a single price *exist*; the gap between that idealization and reality is where a lot of a derivatives desk's actual P&L and risk lives.

## The change of measure and Girsanov's intuition

So far P and Q have been two columns of numbers on a two-state tree. In continuous time they become two full probability distributions over price paths, and the act of swapping P for Q is called a *change of measure*. The theorem that governs it is *Girsanov's theorem*. You do not need its proof for an interview — you need its one-sentence intuition, and that intuition is genuinely simple.

In continuous time the standard model for a stock is *geometric Brownian motion*. Under the real-world measure P it is written as a stochastic differential equation:

$$
dS_t = \mu \, S_t \, dt + \sigma \, S_t \, dW_t^P.
$$

Term by term: `dS_t` is the tiny change in the stock over a tiny instant; `μ` (mu) is the *drift*, the average growth rate the stock actually has in the real world (say 8% a year); `σ` (sigma) is the *volatility*, how much the price jiggles randomly; and `dW_t^P` is the random shock — a *Brownian motion*, the continuous-time version of a coin flip — under measure P. The drift `μ` carries the real risk premium: a risky stock drifts up faster than the safe rate because investors demand to be paid for the risk.

Girsanov's theorem says: there is a change of measure from P to Q that *leaves the volatility `σ` untouched* and *changes only the drift*, replacing `μ` with the risk-free rate `r`. Under Q the very same stock follows

$$
dS_t = r \, S_t \, dt + \sigma \, S_t \, dW_t^Q.
$$

The randomness is identical — same `σ`, same kind of Brownian engine — but the average growth rate has been re-tilted from `μ` down to `r`. That is the entire content of the change of measure for pricing purposes: **it strips the risk premium out of the drift while preserving the noise.**

![The change of measure leaves volatility untouched and only re-tilts the average growth rate from mu to r](/imgs/blogs/risk-neutral-pricing-martingale-measure-quant-interviews-8.png)

The figure shows it directly. In the real world P (top band) the expected stock value climbs with the steeper drift `μ = 8%` per year. Under the risk-neutral measure Q (bottom band) the same stock climbs with the gentler drift `r = 2%` per year. The dashed arrow between the bands is the change of measure itself: it lowers the drift. And critically, the *spread* of outcomes — the volatility `σ` — is identical in both worlds; the bands have the same width. **The intuition this teaches: Girsanov does not change how *uncertain* the stock is, only how fast it is expected to grow; you re-weight the probabilities of the paths just enough to drag the average return down to the risk-free rate, and the option price falls out as a plain average under those re-weighted paths.**

This connects straight back to the binomial tree. Going from `p` to `q` on the two-state tree *is* a change of measure — the discrete-time baby version of Girsanov. On the tree we re-weighted two probabilities (from `p, 1−p` to `q, 1−q`) so that the stock's expected return became the risk-free rate. In continuous time we re-weight an entire continuum of paths to achieve the same thing. Same idea, more machinery. An interviewer who asks "what does Girsanov do?" wants exactly this: *it changes the drift from `μ` to `r` and leaves `σ` alone*, and the reason we want that is so the discounted price becomes a martingale and pricing reduces to averaging.

## Continuous-time pricing is the same discounted expectation

Once the stock evolves under Q with drift `r`, pricing any European derivative is the same one-line recipe as the binomial tree, just with the two-state average replaced by an integral over all possible final prices.

![The continuous-time price is still the discounted Q-expectation of the payoff, now with an integral](/imgs/blogs/risk-neutral-pricing-martingale-measure-quant-interviews-9.png)

Reading the pipeline above: start with the stock's SDE under Q (`dS = r S dt + σ S dW`), roll it forward to expiry `T` to get the distribution of the final price `S_T` (which turns out to be *lognormal*), compute the payoff `max(S_T − K, 0)` for each possible final price, average those payoffs under Q, then discount the average to today by multiplying by `e^(−rT)`. The result is the price:

$$
V_0 = e^{-rT}\, E^Q\big[\, \text{payoff}(S_T) \,\big].
$$

That single equation is the master formula of arbitrage-free pricing, and it is *literally the same* equation as the binomial `C_0 = e^(−rT)[q C_up + (1−q) C_down]` — the bracketed two-state average has just grown into a Q-expectation over a continuous range of `S_T`. For the specific case of a call on a lognormal stock, you can do the integral by hand and the closed form that pops out is the Black-Scholes formula. Black-Scholes is not a separate theory; it is this exact recipe with the integral evaluated. We have a full [Black-Scholes deep dive](/blog/trading/quantitative-finance/black-scholes) for the closed-form derivation and the Greeks — but structurally, it is `e^(−rT) E^Q[max(S_T − K, 0)]` with the expectation computed in closed form.

#### Worked example: price a one-period forward, two ways

A *forward contract* is an agreement to buy the stock at a fixed price `F` at time `T`, with no money changing hands today. The fair forward price is the `F` that makes the contract worth \$0 to enter. Let us price it under Q and confirm it matches the classic no-arbitrage answer, using a non-zero rate `r = 5%` over a one-year step so discounting actually does something. Take `S_0 = `\$100. With `r = 5%`, the up/down stock prices in a binomial step would be set by the model, but the forward price does not even need them — watch.

The payoff of a long forward at expiry is `S_T − F`. Its value today is the discounted Q-expectation:

$$
V_0 = e^{-rT}\, E^Q[S_T - F] = e^{-rT}\big(E^Q[S_T] - F\big).
$$

We know the martingale property gives `E^Q[S_T] = e^(rT) S_0`. Substitute:

$$
V_0 = e^{-rT}\big(e^{rT} S_0 - F\big) = S_0 - e^{-rT} F.
$$

The fair forward price is the `F` that makes `V_0 = 0`, so `F = e^(rT) S_0`. With `S_0 = `\$100 and `r = 5%` over one year, `F = e^(0.05) × 100 ≈ 1.0513 × 100 = `\$105.13. Now check it the elementary no-arbitrage way: to deliver a share at time `T`, buy the share today for \$100 with borrowed money, and you owe `100 × e^(0.05) ≈ `\$105.13 at expiry — so the fair forward must be \$105.13. **Both methods agree, and notice the risk-neutral route used nothing but the martingale property.** The intuition this teaches: the forward price is just today's price grown at the risk-free rate, because under Q the stock *is* expected to grow at exactly that rate — and the real probability `p` is irrelevant once again.

## In the interview room

These are the kinds of problems that actually get asked at Jane Street, Two Sigma, Citadel, and DE Shaw when they want to know whether you *understand* pricing or merely memorized a formula. Each is fully solved. Read the setup, try it, then check the working.

#### Worked example: solved problem 1 — price the call and explain why `p` is missing

*Setup.* A stock is at \$100 and over one period goes to \$120 or \$80. The risk-free rate is 0. Price a \$100-strike call, and then the interviewer asks: "I told you nothing about the probability the stock goes up. Why don't you need it?"

*Solution.* The risk-neutral probability is `q = (S_0 − S_down)/(S_up − S_down) = (100 − 80)/(120 − 80) = 20/40 = 0.5`. The call pays \$20 up and \$0 down, so its price is `q × 20 + (1 − q) × 0 = 0.5 × 20 = `\$10. *Why no `p`:* because the call is replicable. With 0.5 shares and a \$40 loan you copy the call's payoff in both states, and that replica costs \$10 regardless of how likely each state is. Probabilities only matter for expectations, and replication matches the payoff state-by-state, which is stronger than matching it on average. The real drift is already inside the \$100 price; using `p` would double-count it. **A candidate who can give that two-sentence "why" out loud has shown the interviewer the whole point of the subject.**

#### Worked example: solved problem 2 — back out `q`, then reprice under two beliefs

*Setup.* Same tree. Show explicitly that a bull (`p = 0.8`) and a bear (`p = 0.3`) price the call identically, and state what `q` would have to be for the call to be worth \$8.

*Solution.* The price is the replica cost, \$10, and it does not contain `p` — so both the bull and the bear quote \$10. To double-check there is no trickery, note the bull's *real* expected payoff is `0.8 × 20 = `\$16 and the bear's is `0.3 × 20 = `\$6, yet neither of those is the price; both are arbitraged away by the \$10 replica. For the second part: the price is `q × 20 = 8`, so `q = 0.4`. But `q` is pinned by the stock prices at 0.5, so a \$8 quote is *inconsistent with no-arbitrage* — it implies a `q` the market's stock price contradicts, which is precisely an arbitrage opportunity (the next problem cashes it in). **The lesson: `q` is not a free parameter; it is locked by the underlying's prices, and any option quote implies a `q` you can solve for and sanity-check.**

#### Worked example: solved problem 3 — find and execute the arbitrage when the call is mispriced

*Setup.* The fair value of the \$100-strike call is \$10 (from problem 1). The interviewer says the call is trading in the market at \$12. Construct the arbitrage and prove it is risk-free.

*Solution.* The call is \$2 too expensive, so *sell the rich call and buy the cheap replica*. Today: sell the call for \$12 (cash in +\$12); buy the replica — 0.5 shares for \$50 and borrow \$40 — for a net \$10 out (−\$10). Your net cash today is `12 − 10 = +`\$2, pocketed immediately. Now verify you carry no future risk. At expiry, if the stock is \$120: your replica pays \$20 (half-share worth \$60 minus the \$40 loan repaid), and the call you sold costs you \$20 to settle — net \$0. If the stock is \$80: your replica pays \$0, and the call you sold expires worthless, costing \$0 — net \$0. In *both* states the future cash flows cancel exactly, so the \$2 you banked today is pure profit with zero risk.

![Sell the rich call at 12, buy the 10 dollar replica, and pocket 2 dollars no matter which state occurs](/imgs/blogs/risk-neutral-pricing-martingale-measure-quant-interviews-11.png)

The figure traces the trade. On the left, today's three cash flows: receive \$12 for the call, pay \$10 for the replica, net +\$2 locked in. On the right, the two expiry states: in each one the replica's payoff and the sold call's cost cancel to \$0, so you keep the \$2 either way. **The intuition this teaches: "fair value" is not a soft opinion — it is the price below and above which a concrete, mechanical, risk-free trade exists, and being able to *write that trade down* is what separates knowing the formula from understanding it.**

#### Worked example: solved problem 4 — verify the risk-neutral drift and price a digital

*Setup.* On the same tree with `r = 0`, confirm that under Q the stock's expected return is the risk-free rate, then price a digital call that pays \$1 if the stock finishes at \$120.

*Solution.* Under Q with `q = 0.5`, the expected final stock price is `0.5 × 120 + 0.5 × 80 = `\$100, which equals `e^(rT) S_0 = 1 × 100`. So the Q-expected return is 0%, exactly the risk-free rate — confirming Q is the measure under which the stock earns `r`. The digital pays \$1 in the up state and \$0 in the down state, so its price is `e^(−rT)[q × 1 + (1 − q) × 0] = 1 × 0.5 = `\$0.50. Notice the digital's price *equals* `q` (times the discount factor) — a digital is the purest possible instrument for reading off the risk-neutral probability. **The lesson: "what is the price of a security that pays \$1 if `X` happens?" is the same question as "what is the risk-neutral probability of `X`?" — a substitution interviewers use constantly.**

#### Worked example: solved problem 5 — calibrate `q` from a non-zero rate and a forward

*Setup.* A trickier version. The stock is \$100, goes to \$110 or \$95 over one year, and the risk-free rate is `r = 4%` (simple, so the bond grows by a factor of 1.04). Find `q`, price a \$100-strike call, and state the no-arbitrage forward price.

*Solution.* With a non-zero rate the risk-neutral probability is `q = (S_0(1+r) − S_down)/(S_up − S_down) = (100 × 1.04 − 95)/(110 − 95) = (104 − 95)/15 = 9/15 = 0.6`. Sanity check the martingale property: `q × 110 + (1−q) × 95 = 0.6 × 110 + 0.4 × 95 = 66 + 38 = `\$104, which equals `S_0 × 1.04` — the stock earns the risk-free rate under Q, as required. The call pays `110 − 100 = `\$10 in the up state and \$0 in the down state, so its price is `[q × 10 + (1−q) × 0]/(1+r) = (0.6 × 10)/1.04 = 6/1.04 ≈ `\$5.77 — note the discount factor now does real work. The forward price is `S_0 × (1+r) = 100 × 1.04 = `\$104. **The lesson: turning on the rate changes two things — `q` shifts because the stock must out-drift cash, and the discount factor `1/(1+r)` actually bites — but the recipe is identical.**

## How the desk actually computes this: Monte Carlo under Q

A real exotic derivative does not live on a two-state tree — its payoff might depend on the whole path of the stock, on several correlated assets, or on conditions that have no closed-form integral. So how does a desk compute `E^Q[payoff]` when the math is intractable? It *simulates*. And the single most important rule of that simulation is the through-line of this entire article: **you simulate under Q, with drift `r`, never under the real measure with drift `μ`.**

![A pricing desk draws thousands of paths with drift r, averages the payoffs, then discounts once](/imgs/blogs/risk-neutral-pricing-martingale-measure-quant-interviews-10.png)

The pipeline above is exactly what a production Monte Carlo pricer does. Draw a large number `N` of random price paths — but evolve each path with the *risk-neutral* drift `r`, not the stock's real drift `μ`. Roll each path forward to expiry. Compute the derivative's payoff on each path. Average those payoffs across all `N` paths to estimate `E^Q[payoff]`. Discount the average once by `e^(−rT)`. That discounted sample mean is the price. The estimate's error shrinks like `1/√N`, so to halve the error you need four times as many paths — which is why production pricers obsess over variance-reduction tricks.

Here is the core loop in runnable Python, pricing our \$100-strike one-year call on a lognormal stock under Q. Note the drift is `r`, the real-world `mu` never appears, and the comment lines are indented so none of them start at column zero.

```python
import numpy as np

def price_call_mc(S0, K, r, sigma, T, n_paths=1_000_000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    #  Evolve the stock to expiry under Q: the drift is r, NOT the real mu.
    #  This is the whole point — risk-neutral simulation uses the risk-free rate.
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z
    S_T = S0 * np.exp(drift + diffusion)
    #  Payoff of a European call, then discount the sample mean once.
    payoffs = np.maximum(S_T - K, 0.0)
    price = np.exp(-r * T) * payoffs.mean()
    return price

if __name__ == "__main__":
    est = price_call_mc(S0=100, K=100, r=0.05, sigma=0.20, T=1.0)
    print(f"Monte Carlo call price under Q: {est:.4f}")
```

Run it and you get roughly \$10.45 — and if you plug the same inputs into the closed-form Black-Scholes call, you get \$10.45 too. They agree because they are computing the *same* object, `e^(−rT) E^Q[max(S_T − K, 0)]`, one by integration and one by sampling. If you made the rookie mistake of simulating with the real-world drift `μ = 8%` instead of `r = 5%`, your "price" would be too high and, worse, would be *arbitrageable* — it would not match the cost of the replicating hedge, and a counterparty could trade against it. The risk-neutral drift is not a modeling preference; it is the only drift consistent with no-arbitrage.

This is also where the framework earns its keep in dollars. A desk pricing thousands of exotics overnight cannot derive a new replication argument for each one. Instead it encodes *one* fact — "simulate under Q, average the payoff, discount" — into a pricing library, and every instrument from a vanilla call to a path-dependent Asian option to a multi-asset basket flows through the same engine. The risk-neutral measure is what makes that uniformity possible: it is the common coordinate system in which every payoff, however baroque, becomes a single expectation. For the broader machinery of simulating prices and validating models, see [derivatives pricing](/blog/trading/quantitative-finance/derivatives-pricing).

## Common misconceptions

These are the beliefs that get candidates rejected, and the corrections that get them hired. Each is a sentence you should be able to say with conviction.

**"`q` is the real probability the stock goes up."** No. `q` is a *pricing weight* derived from no-arbitrage, not a forecast. In our tree `q = 0.5` even if you are certain the stock will rise (`p = 0.9`). The proof is that `q` is built entirely from the stock prices and the rate — `q = (S_0(1+r) − S_down)/(S_up − S_down)` — with no input from anyone's view of the future. A digital option that pays \$1 when the stock rises is worth `q`, which is the price the market sets, not the odds you personally assign. Confusing `q` with `p` is the single most common error in the subject.

**"Risk-neutral pricing assumes investors are risk-neutral."** No — and this is the trap the name sets. Real investors are risk-*averse*; they demand a premium for holding risky assets, which is exactly why the real drift `μ` exceeds the risk-free rate `r`. Risk-neutral pricing does not assume away that aversion. It performs a *change of measure* that mathematically absorbs the risk premium into a re-weighting of probabilities, so that under the new measure Q everything appears to drift at `r`. The risk preferences are still there — they have been encoded into the difference between P and Q, not deleted. The phrase "risk-neutral" describes the *measure*, not the investors.

**"You discount the real expected payoff at a risk-adjusted rate."** This is the pre-modern way of thinking, and it is both harder and more error-prone. It requires you to (a) estimate the real probability `p`, (b) compute the real expected payoff, and (c) find the correct risk-adjusted discount rate for *that specific payoff* — and the right risk-adjusted rate is different for a call than for the stock than for a digital, because each has different risk. Risk-neutral pricing replaces all three guesses with one clean rule: take the Q-expectation and discount at the *risk-free* rate, the same rate for every instrument. It is not just more elegant; it removes two sources of estimation error.

**"The risk-neutral measure is some abstract mathematical fiction with no real meaning."** It has a very concrete meaning: it is the unique set of weights (in a complete market) under which today's traded prices are arbitrage-free, and it is exactly what a hedging desk implicitly uses. When a trader replicates and hedges an option, they are *living inside* Q — their hedged P&L does not depend on whether the stock actually goes up, only on whether their replication held. Q is the measure of the hedger, P is the measure of the speculator, and a derivatives desk is in the hedging business.

**"If two banks quote different prices for the same option, one of them is wrong."** Not necessarily. In an *incomplete* market — the real one, with jumps, stochastic volatility, and trading frictions — the second fundamental theorem says Q is *not* unique. There is a whole band of arbitrage-free prices, and two desks can sit at different points in that band by choosing different models or calibrations, with neither committing an arbitrage. The disagreement is not an error; it is the signature of incompleteness, and it is where a lot of real trading happens.

## How it shows up on a real desk

The ideas above are not academic. Here is where each one touches money, with concrete mechanisms.

**Every options market-maker is short the misconception in problem 3.** When a desk quotes a two-sided market on an option, the price it shows is anchored to a replication/hedging cost, not to a forecast of where the stock is going. The desk's edge is the bid-ask spread around fair value; its *risk* is that its replication is imperfect (the market does not move in nice binomial steps). The entire discipline of *delta-hedging* — continuously holding `Δ` shares against a short option — is the continuous-time version of the replicating portfolio in problem 1. The desk recomputes `Δ` constantly and trades the stock to stay matched, exactly as the tree prescribes, just in tiny steps.

**Monte Carlo pricing under Q is a nightly production job at every large derivatives house.** A bank holding a book of thousands of exotic derivatives — autocallables, baskets, path-dependent notes — values them overnight by simulating each under the calibrated risk-neutral measure, averaging payoffs, and discounting. The drift in every one of those simulations is the risk-free rate (plus dividend and funding adjustments), never the strategist's view of the market. A single bug that simulated under the real drift would mis-mark the whole book and could be caught only when a hedge failed to behave as the model predicted. The risk-neutral drift is enforced precisely because it is the only one consistent with the hedges the desk actually holds.

**Calibration is the act of choosing a point in the incompleteness band.** Because the real market is incomplete, a model like Black-Scholes (or its richer cousins) has free parameters — most importantly the volatility `σ`. Desks *calibrate*: they pick the parameters so the model reproduces the prices of liquid, observable options (the vanillas), and then use that calibrated Q to price the illiquid exotics consistently. The whole [implied-volatility surface](/blog/trading/quantitative-finance/black-scholes) is the market's collective choice of Q, read off from option prices. When two desks disagree on an exotic's price, they have usually calibrated to slightly different liquid instruments or chosen different model dynamics — different Q's, both arbitrage-free.

**The forward-price relationship from the forward example is the basis of the entire futures market.** The result `F = e^(rT) S_0` (adjusted for dividends and carry) is the *cost-of-carry* formula that links every futures price to its spot. When a futures price drifts away from `e^(rT) S_0`, *cash-and-carry arbitrageurs* step in — buy spot, sell the future, finance the position, deliver at expiry — exactly the trade in the forward worked example, and that pressure forces futures back to fair value. This is one of the most heavily traded, lowest-margin arbitrages in finance, and it is pure risk-neutral pricing with `r` as the only ingredient.

**The 2008 crisis was, in part, an incompleteness story.** Many credit derivatives were priced with models that implicitly assumed a more complete, more hedgeable market than existed. When liquidity vanished, replication broke down — you could not actually trade the hedge at the prices the model assumed — and the single "fair price" the models reported turned out to be one optimistic point in a very wide band. The lesson desks carry forward: a risk-neutral price is only as trustworthy as the replication it rests on, and when replication fails, so does the price. That is the second fundamental theorem biting in the real world.

## When this matters and where to go next

If you are preparing for a quant-researcher or derivatives interview, this is the single highest-leverage topic you can master, because it is the *why* underneath almost every pricing question they can ask. An interviewer who asks you to price an option on a tree is really testing whether you reach for replication and let `p` cancel; one who asks "what does Girsanov do?" wants the one-sentence drift-shift answer; one who asks "why discount at the risk-free rate?" wants the no-arbitrage-equals-existence-of-Q logic. All of it traces back to the \$100-to-\$120-or-\$80 tree you can now derive cold.

The practical habit to build: whenever you see a derivative, ask the three questions in order. *Can I replicate it?* (If yes, the market is locally complete and the price is forced.) *What is `q`?* (Solve it from the underlying's prices and the rate.) *What is `e^(−rT) E^Q[payoff]`?* (Average and discount.) That sequence prices a vanilla call, a digital, a forward, and — extended to many steps or to Monte Carlo — every exotic a desk holds.

For the natural next steps: [Black-Scholes](/blog/trading/quantitative-finance/black-scholes) is this exact recipe with the continuous-time integral evaluated in closed form, plus the Greeks that come from differentiating it. [Derivatives pricing](/blog/trading/quantitative-finance/derivatives-pricing) covers the replication and numerical machinery in more breadth. And the [expected-value techniques](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews) post sharpens the averaging skills that risk-neutral pricing leans on constantly. Master the tree first, though — everything else is the tree with more steps.

*This article is educational, not investment advice. Trading derivatives carries real risk of loss; the worked examples are simplified to teach the mechanism, not to recommend any trade.*

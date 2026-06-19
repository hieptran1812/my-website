---
title: "How a Rule Becomes a Price: Expectations, the Drift, and the Repricing"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The event-study toolkit applied to legal and policy events: how a rule's market impact splits into the pre-event run-up, the announcement gap, and the post-event drift or fade — and why the edge is measuring how much was already priced."
tags: ["regulation", "geopolitics", "event-study", "abnormal-return", "buy-the-rumor", "priced-in", "expectations", "repricing", "trading", "options", "drift"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A rule's market impact is not one move on the headline; it splits into a pre-event **run-up** as expectations build, an announcement **gap** that prices only the *surprise* versus what was already expected, and a post-event **drift** or **fade** as the effect is digested.
>
> - "Buy the rumor, sell the news" is this same pattern, formalized. The price already embeds the *expected* ruling; only the gap between the actual outcome and that priced-in baseline is new information.
> - The event-study toolkit measures it precisely: fit a normal-return model on a calm **estimation window**, then measure the **abnormal return** (actual minus expected) across the **event window**, and sum it into a **cumulative abnormal return** (CAR).
> - Whether a move keeps **drifting** (under-reaction to a complex, slow rule) or **fades** (over-reaction to a hyped, fully-anticipated event unwinds) is the difference between the drift trade and the fade trade.
> - The one number to remember: in the run-up to the January 2024 US spot-bitcoin-ETF approval, bitcoin rose roughly **+72%** from mid-October — and then *fell* about **15%** in the nine trading days *after* the approval. The good news was already in the price.

On the morning of 11 January 2024, the United States Securities and Exchange Commission — the federal agency that regulates stock and securities markets — let eleven *spot bitcoin exchange-traded funds* begin trading. An exchange-traded fund, or ETF, is just a fund whose shares trade on a stock exchange like any ordinary stock; a *spot* bitcoin ETF holds actual bitcoin, so buying one share is a regulated, brokerage-account way to own the coin without touching a crypto wallet. For a decade the SEC had refused to approve one. This was, by any plain reading of the headline, enormous, unambiguously bullish news for bitcoin.

Bitcoin fell. Over the next nine trading days it dropped from about \$46,640 to roughly \$39,530 — down about 15% — on the single most bullish regulatory headline in its history. To anyone watching only the news ticker, this looked insane: the market got exactly what it wanted and sold off. To anyone who had been watching the *price*, it was the most ordinary thing in the world. Bitcoin had already climbed about 72% from its mid-October low into the approval, as the rumor of approval hardened into near-certainty. By the day the rule actually changed, the good news was already *in the price*. There was nothing left to buy — only profits to take.

This is the single most important — and most counterintuitive — idea in trading laws, rulings, and policy: **markets price the expected rule before the rule exists.** A statute, an agency rule, a court decision, a tariff, or a sanction does not move a price when it is *announced*; it moves the price continuously, from the first credible rumor, as the market revises the *probability* and the *magnitude* of the change. By the time the gavel falls, most of the move is usually behind you. The practitioner's entire job is to measure how much is *already* priced, so they can trade the part that *isn't*. This post hands you the toolkit — the *event study* — that lets you do exactly that, and then turns it into four concrete trades.

![Three-phase price path with a run-up line, an announcement gap, and a forking drift up or fade down](/imgs/blogs/how-a-rule-becomes-a-price-expectations-drift-and-repricing-1.png)

## Foundations: efficient markets, expected returns, and the event study

Before we can measure a rule's impact, we need three building blocks, each defined from zero: *why* prices move before the event (efficient markets), *what* "normal" looks like so we can spot the abnormal (expected return), and *how* we isolate the rule's effect from everything else happening that week (the event study).

### What "efficient markets" actually claims

The **efficient market hypothesis** is a claim about *information*: that the price of a traded asset already reflects whatever is publicly known about it. The intuition needs no math. If everyone can see that a court is very likely to strike down a regulation that has been crushing a company's profits, then everyone wants to own that company *now*, before the ruling — and their buying pushes the price up *now*. By the time the ruling lands, the easy money has been competed away. A price is a giant, continuously-updated betting market on the future, and the bet is placed long before the result is known.

Economists slice the hypothesis into three strengths, and the distinction is the whole reason a *legal* event is tradeable at all:

- **Weak form** — prices already reflect all *past price* information. You cannot beat the market just by studying charts of past prices, because any pattern would already be arbitraged away.
- **Semi-strong form** — prices reflect all *publicly available* information: filings, news, the text of a published rule, a court's released opinion. The instant information becomes public, it is in the price. Under semi-strong efficiency, you cannot earn an *abnormal* return by trading on a public regulatory headline — by the time you read it, so did the machines.
- **Strong form** — prices reflect *all* information, including private, non-public information. Almost nobody believes this; it would mean insider information is worthless, which is exactly why insider trading is illegal and profitable.

Real markets sit *between* weak and semi-strong. That gap is the practitioner's playground. A regulatory event is *forecastable*: a comment period closes on a known date, a court hears arguments and signals which way it leans, an agency telegraphs a rule for months. The market *prices the forecast*, continuously, and the price you see is a weighted average over every possible outcome. The semi-strong claim is only about the moment information turns *public*; the run-up happens in the long pre-public window when the outcome is merely *probable*, not yet *certain*.

### Expected return: what "normal" looks like

To say a rule had an *abnormal* effect, you first need a number for the *normal* move — the move the stock would have made anyway, with no rule at all, just because the whole market was up or down that day. That baseline is the **expected return**, and the standard way to estimate it is the **market model**.

The market model says a stock's return on any day is a linear function of the market's return that day:

```
E(t) = alpha + beta * market_return(t)
```

Here **beta** measures how much the stock amplifies or dampens market moves — a beta of 1.5 means that when the market is up 1%, this stock tends to be up about 1.5%. **Alpha** is the stock's average drift independent of the market, usually tiny. You estimate alpha and beta by fitting this line over a calm historical stretch — the *estimation window* — when no event was happening. Then, on event days, you plug the actual market return into the formula to get the *expected* return: what the stock "should" have done given how the broad market moved.

### Abnormal return and cumulative abnormal return

Now the payoff. The **abnormal return** is the part of the day's move the market model *cannot* explain — the residual you attribute to the event:

```
AR(t) = actual_return(t) - E(t)
      = actual_return(t) - (alpha + beta * market_return(t))
```

If a stock rose 6% on a ruling day while the market model said it "should" have risen 1% given the market, the abnormal return is +5%. That 5% is your estimate of what the *rule* did, stripped of the market's own move. The **cumulative abnormal return**, or **CAR**, simply sums the daily abnormal returns over a window of days:

```
CAR = AR(-1) + AR(0) + AR(+1) + ...
```

CAR is the tool for measuring the *total* repricing across the run-up, the gap, and the drift — not just the one-day jump. The estimation window teaches you what normal is; the event window measures how far the event pushed the stock away from normal.

### Why legal events are special: the long, forecastable run-up

An event study works on *any* discrete event — an earnings report, a product launch, a merger. But legal and policy events have a structural feature that makes their run-ups uniquely long and uniquely tradeable: **the process is public, scheduled, and adversarial, long before the outcome is known.**

Compare the two. A quarterly earnings number is a near-instantaneous reveal: the company has the figure, the market does not, and at one moment it becomes public. The run-up to an earnings report is short and mostly speculative — you're guessing at a number nobody outside the company knows yet. The information *asymmetry* collapses in a single instant.

A regulatory or legal outcome is the opposite. A federal rule in the United States typically moves through a *notice-and-comment* process: the agency publishes a proposed rule (a "Notice of Proposed Rulemaking," or NPRM), opens a public comment period that can run 60 to 90 days or more, then issues a final rule that often takes effect months later — and the final rule is frequently litigated for years after that. A Supreme Court case is argued in open court, where the justices' questions telegraph their leanings, *months* before the opinion is released on a roughly knowable schedule (the Court clears its docket by late June). A merger's antitrust review has filing deadlines, a "second request" for information, and public statements from the reviewing agencies along the way. At every step, *information about the probable outcome leaks into public view* — not the outcome itself, but ever-sharper estimates of it.

That long, public, multi-stage process is *why* the run-up in a legal event can stretch over months and *why* it is forecastable by someone who reads the process carefully. The market is not guessing at a hidden number; it is continuously updating a probability as a visible process unfolds. The practitioner's edge is reading that process — the docket, the hearing transcript, the comment letters, the agency's public posture — better than the consensus does, and pricing the converging probability before it reaches certainty. We treat *where to find* these scheduled events as a separate skill in the regulatory-calendar post; here, the point is that the legal process is what makes the run-up phase both long and tradeable.

### The base rate: most rules are anticipated, and most surprises are small

There is one more foundational fact, and it is the one that keeps you humble: **the median regulatory event produces a small abnormal return**, because the median regulatory event comes out roughly as the market expected. The dramatic gaps — the surprise rulings, the shock approvals — are the *tail* of the distribution, not the body. The market is, most of the time, a good forecaster of legal outcomes precisely because the process is so public.

This matters for sizing and for sanity. If you expect every ruling to produce a tradeable jump, you will over-trade, paying spreads and fees on a series of non-events whose surprise was approximately zero. The discipline is to *wait for the events where you have a differentiated probability view* and to recognize that on most days the right position is no position. The base rate of "small move because it was priced" is not a market failure to exploit — it is efficiency doing exactly what it's supposed to, and it is the reason the priced-in check (below) is the gate every trade must pass through.

![Event-study timeline showing estimation window, run-up, event window, announcement day, and post-event drift, with abnormal return formulas](/imgs/blogs/how-a-rule-becomes-a-price-expectations-drift-and-repricing-2.png)

#### Worked example: an abnormal return around a ruling day

Suppose a healthcare company, MedCo, trades at \$80 the day before a federal appeals court rules on whether a regulation that capped its drug prices is lawful. Over the prior year — the **estimation window** — you fit the market model and find MedCo has **alpha = 0** (negligible) and **beta = 1.2**.

On the ruling day, the court strikes down the cap (good for MedCo). The broad market index that day is **up 0.5%**. MedCo closes at **\$85.60**, a **+7.0%** actual return.

First, the expected (normal) return given the market:

```
E = alpha + beta * market_return
  = 0 + 1.2 * 0.5%
  = 0.6%
```

So the market model says MedCo "should" have risen about 0.6% just from the market being up. The abnormal return — the part attributable to the *ruling* — is:

```
AR = actual - expected
   = 7.0% - 0.6%
   = +6.4%
```

In dollar terms, on a \$80 starting price, a 6.4% abnormal return is about **\$5.12** of value the ruling added per share, on top of the roughly \$0.48 the stock would have gained anyway from the up-market. The abnormal return strips out the market's move so you measure the rule's effect, not the day's weather.

#### Worked example: cumulative abnormal return over a 5-day window

One day rarely captures a rule's full repricing — the run-up and the drift both bleed across days. So we sum the abnormal returns into a **CAR**. Continuing with MedCo, here are the daily abnormal returns across a five-day event window centered on the ruling at t=0:

| Day | Actual return | Market return | beta × market | Abnormal return |
|---|---|---|---|---|
| t = −2 | +2.1% | +0.3% | +0.36% | **+1.74%** |
| t = −1 | +3.0% | −0.2% | −0.24% | **+3.24%** |
| t = 0 (ruling) | +7.0% | +0.5% | +0.60% | **+6.40%** |
| t = +1 | +1.5% | +0.4% | +0.48% | **+1.02%** |
| t = +2 | +0.6% | +0.6% | +0.72% | **−0.12%** |

The cumulative abnormal return is the sum of the abnormal-return column:

```
CAR(-2, +2) = 1.74% + 3.24% + 6.40% + 1.02% + (-0.12%)
            = +12.28%
```

Notice what the table reveals. Almost half the total repricing — the +1.74% and +3.24% on days t=−2 and t=−1 — happened *before* the ruling was public, as the market front-ran a likely outcome. That is the run-up. The +6.40% on t=0 is the gap. The +1.02% on t=+1 is a small post-event drift, and by t=+2 the abnormal return is essentially zero — the repricing is complete. The CAR is the single number that captures the whole repricing across all three phases.

## Phase 1 — the run-up: expectations building

The run-up is the slow climb (or slow slide) in the days, weeks, or even months *before* an event, as the market revises its estimate of the probability and magnitude of a rule change. It is not noise; it is the market doing its job — pricing in the *forecast*.

The mechanism is pure probability-weighting. Suppose a stock is worth \$100 if a favorable rule passes and \$70 if it fails. If the market thinks the rule has a 50% chance, the stock should trade near the probability-weighted value:

```
price = 0.50 * 100 + 0.50 * 70 = 85
```

Now suppose a credible leak, a favorable court hearing, or a regulator's public comment lifts the perceived probability to 80%. The fair price jumps:

```
price = 0.80 * 100 + 0.20 * 70 = 94
```

The stock rose \$9 — *before anything actually happened* — purely because the *odds* of the good outcome rose. Every new data point that nudges the probability nudges the price. This is the run-up: a sequence of small repricings as the market's estimate of the outcome converges toward what it eventually expects. By the time the rule is near-certain (probability ≈ 95%+), the price is already near \$100, and there is very little gap left to close.

There is a subtlety worth naming, because it explains *why being early has a cost*. The fair price during the run-up is the probability-weighted outcome *discounted* for time and for risk. A favorable outcome worth \$100 that won't be confirmed for another year is worth slightly less than \$100 *today*, because you must wait for it and because the wait carries uncertainty. As the event date approaches, two things happen at once: the probability sharpens (toward 0 or 1), and the discount shrinks (less time to wait). Both push the price toward its final outcome value. This is why the run-up often *accelerates* into the date — the closer you get, the faster the remaining uncertainty resolves and the discount unwinds. It also means an investor who buys *very* early pays for carrying the position through a long, uncertain wait; being right about the outcome but early about the timing ties up capital and bleeds opportunity cost. The run-up trade rewards being early, but only early *enough* — there is a real cost to being early by years.

#### Worked example: the priced-in fraction from the run-up

How much of the *eventual* move is already done by the time the event arrives? That fraction is the single most important number in this whole framework, because it tells you whether there is anything left to trade. Define:

```
priced-in fraction = (pre-event price - base price) / (full-outcome price - base price)
```

Take our \$70-to-\$100 stock. The **base price** (rule fails, the disappointing outcome) is \$70; the **full-outcome price** (rule passes, fully reflected) is \$100. Suppose by the day before the ruling the stock has run up to **\$94**. Then:

```
priced-in fraction = (94 - 70) / (100 - 70)
                   = 24 / 30
                   = 0.80, or 80%
```

Eighty percent of the favorable outcome is already in the price. If the rule then passes exactly as expected, the stock only has the last \$6 to gain (from \$94 to \$100) — a +6.4% move, not the +43% it would have made from the \$70 base. And if the rule *fails*, the stock falls all the way back to \$70, a brutal −25% drop from \$94. The asymmetry is the whole point: when 80% is priced in, the upside on confirmation is small and the downside on disappointment is large. The priced-in fraction is the number that tells you the risk is no longer worth the reward.

### The shape of the run-up tells you what the market knows

The run-up is not just a number; its *shape over time* is a readable signal. Three patterns recur, and each says something different about the market's information.

A **smooth, accelerating climb** into a known event date says the market is steadily converging toward a confident expectation — the probability is rising in an orderly way as the date approaches. This is the bitcoin-ETF shape: a long, fairly steady grind higher as approval moved from "possible" to "likely" to "near-certain." When you see it, assume the gap on the day will be *small*, because the smooth climb means the outcome is already well-anticipated.

A **sudden step** in the run-up — a one-day jump weeks before the actual ruling — marks a discrete *information event*: a leak, a favorable court hearing, a regulator's offhand public comment that sharply revised the odds. That step is the market repricing a probability in real time. If you can identify *what* caused the step (read the day's news), you learn what new fact moved consensus, and you can judge whether the market over- or under-reacted to it.

A **flat line** into a major event is itself information: it says the market either has no view (genuinely uncertain, a true coin-flip) or has decided the event is a non-event. A flat run-up before a binary ruling means the *full* surprise is still ahead of you — neither outcome is priced — and the gap on the day will be large in *either* direction. That is the setup where the implied move from options is widest and the gap trade has the most to work with.

Reading the run-up's shape is the qualitative complement to the priced-in fraction. The fraction tells you *how much* is priced; the shape tells you *how confidently* and *on what information*. Together they tell you whether the event ahead is likely to gap or fade.

This is precisely the bitcoin-ETF dynamic. Let me show it with the actual prices.

![Bitcoin price line around the 2024 spot-ETF approval, rising into 11 January then fading](/imgs/blogs/how-a-rule-becomes-a-price-expectations-drift-and-repricing-3.png)

From its mid-October 2023 level near \$27,150, bitcoin climbed to about \$46,640 by the approval — a run-up of roughly +72%, almost all of it driven by the market raising its estimate of the *probability* of approval (after a key court loss for the SEC in 2023 made approval look inevitable). By 11 January the approval was essentially certain. There was no probability left to reprice upward. So when the rule actually changed, the only thing left was profit-taking, and bitcoin *fell* about 15% over the next nine sessions. The headline was maximally bullish; the *surprise* was zero. The price had already done its work.

## Phase 2 — the gap: pricing only the surprise

The announcement gap is the price jump (up or down) on the day the rule actually becomes known. And here is the idea that separates people who understand markets from people who don't: **the gap prices the surprise, not the headline.** The move on the day is proportional to the difference between the *actual* outcome and what the market had *already priced in* — not to how good or bad the news is in absolute terms.

![Flow diagram showing priced-in baseline and actual ruling combining into the surprise, which drives the price move](/imgs/blogs/how-a-rule-becomes-a-price-expectations-drift-and-repricing-4.png)

Think of the priced-in baseline as a "consensus expectation" — the outcome the market is betting on. Information has value only to the extent it changes that bet. A ruling that comes out *exactly* as the consensus expected contains no new information, so it moves the price by approximately zero, no matter how dramatic the headline. A ruling that comes out *far* from consensus — a court strikes down a rule everyone thought it would uphold — is a huge surprise, and the price gaps violently to reprice the new reality.

This is why a stock can fall on objectively good news (the good news was *less good* than priced) and rally on objectively bad news (the bad news was *less bad* than feared). The sign of the gap is the sign of the *surprise*, not the sign of the *news*.

#### Worked example: extracting the expected move from a straddle into a binary date

Before a binary regulatory event — an FDA decision, a make-or-break court ruling, an ETF approval deadline — the *options market* tells you exactly how big a move it is pricing in. An **option** is a contract giving the right (not the obligation) to buy (a *call*) or sell (a *put*) the stock at a fixed *strike* price; the price you pay for that right is the *premium*. A **straddle** is buying both a call and a put at the same strike — a bet that the stock moves *a lot* in *either* direction. The straddle's cost is the market's price of the expected move.

Suppose RegPharma trades at \$50, and a binary FDA approval decision is due tomorrow. The at-the-money straddle (the \$50 call plus the \$50 put, both expiring just after the decision) costs **\$6.00** total (\$3.20 for the call, \$2.80 for the put). The rough rule of thumb for the **implied move** is:

```
implied move (in dollars) ~ straddle price
                          = 6.00

implied move (in %) ~ straddle price / stock price
                    = 6.00 / 50
                    = 12%
```

The options market is pricing in a roughly **±12%** move (about ±\$6) on the decision. That is the priced-in *magnitude* of the surprise. Now you have a yardstick: if you think the actual reaction to approval would be only +6%, the straddle is *overpriced* relative to your view, and you'd lean toward *selling* volatility (or fading the move). If you think a surprise rejection could send it down 25%, the straddle is *cheap*. The implied move converts the vague phrase "this event is a big deal" into a number you can trade against.

#### Worked example: a binary-event expected value from the priced-in probability

The straddle gives you the priced-in *magnitude*; the *spread* or the *price level* gives you the priced-in *probability*. Suppose a company's stock will be worth **\$100** if a merger is approved by regulators and **\$60** if it is blocked, and right now it trades at **\$88**. What probability of approval is the market pricing?

```
price = p * 100 + (1 - p) * 60
88    = 100p + 60 - 60p
88    = 40p + 60
40p   = 28
p     = 0.70, or 70%
```

The market is pricing a **70%** chance of approval. Now suppose *your* research — reading the regulator's public statements, the antitrust filings, the political backdrop — convinces you the true probability is **85%**. Your fair value is:

```
your fair value = 0.85 * 100 + 0.15 * 60
                = 85 + 9
                = 94
```

You think the stock is worth \$94 against a \$88 market price — a \$6, roughly 7%, edge — *because you disagree with the priced-in probability*, not because you think the merger is "good." The trade is the gap between your probability and the market's. This is the engine of merger arbitrage and every binary-event trade: you are never betting on the outcome, you are betting that the priced-in odds are wrong.

#### Worked example: is the abnormal return statistically real, or just noise?

A +6.4% abnormal return on a ruling day *looks* meaningful — but is it large relative to the stock's normal day-to-day wiggle, or is it the kind of move the stock makes on any random day? An event study answers this with a *t-statistic*: the abnormal return divided by the stock's normal daily volatility (its standard deviation of returns), measured on the calm estimation window.

Take MedCo again. On the estimation window, you measure MedCo's daily return standard deviation — its typical day-to-day swing — at **1.8%**. The ruling-day abnormal return was **+6.4%**. The t-statistic is:

```
t = abnormal_return / daily_return_stdev
  = 6.4% / 1.8%
  = 3.56
```

A t-statistic of 3.56 means the ruling-day move was about three and a half times the size of a normal daily swing. As a rough guide, a t above roughly 2 is conventionally treated as unlikely to be chance (about a 5% probability of occurring by luck); a t of 3.56 is well past that — this move is almost certainly the *ruling*, not noise. Now contrast a different name, ThinCo, with a noisy 5.0% daily standard deviation that posted a +6.0% abnormal return on its own event:

```
t = 6.0% / 5.0% = 1.20
```

ThinCo's t of 1.20 is *not* significant — a 6% move in a stock that routinely swings 5% a day could easily be noise, not the event. The lesson for trading: a raw abnormal-return *percentage* is meaningless without the stock's volatility for scale. A 6% jump is a thunderclap in a placid utility and a shrug in a meme stock. Always size the abnormal return against the normal swing before you believe the rule did it.

## Phase 3 — the drift and the fade: digesting the rule

The gap is rarely the end. After the announcement, the price keeps moving — and which way it moves separates two opposite trades. Either the market *under-reacted* and the price keeps **drifting** in the direction of the news for days or weeks, or the market *over-reacted* and the price **fades** back toward where it started.

![Branching diagram: announcement gap forks into a complex slow rule that drifts and a hyped event that fades](/imgs/blogs/how-a-rule-becomes-a-price-expectations-drift-and-repricing-6.png)

### Post-event drift: the under-reaction

**Drift** is the tendency of a price to keep moving the same direction *after* the event, because the market did not fully digest the news on day one. This is the regulatory cousin of *post-earnings-announcement drift*, one of the most robustly documented anomalies in finance. It happens when a rule is *complex*, its consequences are *slow* to play out, or *few analysts* cover the affected names — so the full implication takes time to be understood and priced.

The clearest example is a *re-rating*: a rule that permanently changes a sector's earnings power, digested over years rather than days.

![Defense-sector total return rebased to 100 on 23 February 2022, drifting up about 38% over three years](/imgs/blogs/how-a-rule-becomes-a-price-expectations-drift-and-repricing-7.png)

When Russia invaded Ukraine on 24 February 2022, it triggered a *structural* policy change across NATO: governments committed to permanently higher defense spending for years to come. That is not a one-day event — it is a slow re-rating of every defense contractor's future earnings. The US aerospace and defense index (rebased to 100 the day before the invasion) did not spike and fade; it *drifted upward*, reaching about 110 by the end of March 2022, 119 by year-end, and roughly 138 by the end of 2024 — a slow, grinding **+38%** re-rating as the market digested a multi-year shift in the rules of the game. A trader who understood that a structural spending regime is digested slowly could ride the drift long after the invasion headline had faded from the front page.

#### Worked example: a drift trade P&L

Suppose you identify a slow, complex rule change — a regulator quietly finalizes a rule that raises a utility's allowed return on capital, a wonky change that few generalist investors will model immediately. The stock, UtilCo, gaps from \$40 to \$42 (a +5% announcement gap) the day the rule is finalized, but you judge the *full* fair value is closer to \$46 once analysts rebuild their models over the coming weeks. You buy **1,000 shares at \$42**, betting the drift continues.

```
entry cost = 1,000 shares * $42 = $42,000
```

Over the next six weeks, analysts upgrade their estimates and the stock drifts to **\$45.50**. You sell:

```
exit value = 1,000 shares * $45.50 = $45,500
gross P&L  = $45,500 - $42,000     = $3,500
return     = 3,500 / 42,000        = +8.3%
```

You captured \$3,500, an 8.3% return, *after* the announcement had already happened — money that was available precisely because the market under-reacted to a complex rule on day one. The drift trade harvests the gap between the day-one reaction and the eventual full repricing. Its risk: if the market had it right on day one, there is no drift to harvest, and you've bought at the top.

### The fade: the over-reaction unwinds

The **fade** is the opposite. When an event is *hyped*, *fully anticipated*, and *crowded* — everyone has positioned for it in advance — the announcement often produces a *blow-off* move that immediately reverses, because the buyers who were going to buy have already bought, and now they take profits. This is "buy the rumor, sell the news" in its purest form: the run-up *was* the trade; the announcement is the exit.

The bitcoin-ETF chart above is a textbook fade. So, in a more violent form, was GameStop in January 2021.

![GameStop daily close on a log scale through January 2021, spiking to 347 then collapsing after broker halts](/imgs/blogs/how-a-rule-becomes-a-price-expectations-drift-and-repricing-5.png)

GameStop ran from \$17.25 on 4 January 2021 to a peak close of \$347.51 on 27 January — a roughly 20-fold repricing driven by a short squeeze and a frenzy of retail expectation. Then the *market-structure plumbing* intervened: on 28 January several brokers restricted buying of the stock (a regulatory and risk-management response to clearinghouse collateral demands), and the over-extended price collapsed about 44% in a single day, to \$193.60, then kept fading to \$53.50 by 4 February. The parabola and its immediate collapse are the signature of an over-reaction: a price detached from any fundamental anchor, fully priced with euphoria, with nothing left to buy. (We dig into the legal mechanics of those buy-button halts in the market-structure post; here, the point is the *shape* — the violent fade of an over-extended repricing.)

The diagnostic is simple. *Complex, slowly-understood, under-covered* rule → expect **drift** (the under-reaction). *Hyped, fully-anticipated, crowded* event → expect **fade** (the over-reaction). The same announcement can produce either, and reading which regime you're in is the difference between the drift trade and the fade trade.

### Second-order effects: the rule reprices more than its target

A rule almost never reprices a single name in isolation. Its real reach runs through the *web of related businesses*, and the second-order repricing is often where the slower, less-crowded drift lives — because the market reacts to the obvious first-order target on the headline and digests the ripple effects only later.

Trace the channels. A rule that hurts Company A often *helps* its direct competitor, Company B, which now faces a hobbled rival — so a ruling against A can be a *buy* for B. A rule that caps prices on a product reprices not just the maker but its *suppliers* (whose orders may fall) and its *customers* (whose input costs may fall). A regulation that raises compliance costs across an industry can be a *net positive* for the largest incumbents, who can absorb fixed compliance costs more easily than small rivals — so a "tough new rule" headline can paradoxically lift the dominant player and crush the fringe. And a sector-wide rule reprices the whole sector's *multiple* (the valuation the market is willing to pay per dollar of earnings), which is a slow, structural re-rating rather than a one-day gap.

The bitcoin-ETF approval is a clean multi-layer example. The first-order effect was on bitcoin itself (which faded). But the *second-order* effects rippled outward over the following months: the issuers who won approval (asset managers competing for fee revenue), the exchanges and custodians that service the funds, the crypto-adjacent equities, and eventually the broader question of which *other* crypto assets might get the same regulated wrapper. Those second-order repricings unfolded on a slower clock than the headline, and they were less crowded — which is exactly the drift setup.

The practical move: when you see a regulatory headline, don't stop at the obvious target. Map the competitors, suppliers, customers, and the sector multiple, and ask which of *those* the market has not yet repriced. The first-order trade is crowded by the time you read the headline; the second-order drift is where a careful reader still has an edge.

### Selection and confounding: how to misread a legal event

Two pitfalls quietly wreck event studies on legal events, and you must guard against both.

**Confounding** is when something *else* moves the stock during your event window, and you wrongly attribute it to the rule. A court rules on a company's antitrust case the same morning the Federal Reserve surprises markets with a rate decision — was the stock's move the ruling or the Fed? The fix is the market model itself: subtracting beta × market return removes the *market-wide* confounder. But it cannot remove a *company-specific* confounder (an earnings pre-announcement the same day). The discipline is to keep the event window *short* (often just t=−1 to t=+1) to minimize the chance of an unrelated event sneaking in, and to manually check the news for that window.

**Selection bias** is when you only study the events that "worked." If you collect the rulings where the stock moved a lot and conclude "rulings move stocks," you've selected on the outcome. Most regulatory events produce *small* abnormal returns precisely because they were *anticipated* — the run-up already happened. A clean study includes the duds: the rulings that came out as expected and moved nothing. Forgetting them makes you overestimate the tradeable opportunity in the *next* event.

## Common misconceptions

### Misconception 1: "The stock fell on good news, so the market is irrational"

This is the most common and the most expensive misreading. When bitcoin fell about 15% on its most bullish-ever headline — the January 2024 spot-ETF approval — the cry of "irrational market!" went up everywhere. But the market was perfectly rational: bitcoin had *already* risen roughly 72% in the run-up, from about \$27,150 to \$46,640, pricing in the approval as it became near-certain. With the good news fully in the price, the only remaining force was profit-taking. The number that explains the "irrational" drop is the *priced-in fraction*: when ~100% of the good news is already priced, a confirming announcement has zero positive surprise and the path of least resistance is *down*. The market wasn't irrational; you were looking at the headline instead of the run-up.

### Misconception 2: "The law is bullish, so buy it"

A rule being *good for a company* tells you nothing about whether to buy — what matters is whether the good news is *already priced*. Return to the priced-in math: a stock worth \$100 on a favorable rule and \$70 without it, trading at \$94 the day before, has 80% of the good outcome priced in. If you buy at \$94 expecting the bullish rule, your *upside on confirmation* is only \$6 (+6.4%), while your *downside on a surprise rejection* is \$24 (−25%). That is a terrible risk-reward — you're risking \$24 to make \$6 — even though the rule is unambiguously "bullish." The tradeable question is never "is this rule good?" It is "is this rule *better than what's priced*?" A bullish rule that's fully expected is a *sell*, not a buy.

### Misconception 3: "The move on the day is the whole move"

Many traders close the book after the announcement gap, assuming the repricing is done. But the *drift* can be larger than the gap. The post-Ukraine defense re-rating is the proof: the day-one and first-month reaction took the index from 100 to about 110, but the *drift* over the following years carried it to roughly 138 — meaning more than two-thirds of the total +38% repricing happened *after* the initial reaction window, as a complex multi-year spending regime was slowly digested. If you'd assumed "the move on the day is the whole move," you'd have left most of the trade on the table. For complex, slow rules, the gap is the *opening*, not the conclusion.

### Misconception 4: "If I read the news fast, I can beat the gap"

Reading a published rule or a released court opinion faster than the next human still loses to the machines and, more importantly, to the *run-up*. By the time a document is public, semi-strong efficiency has already collapsed the new information into the price in seconds. The edge is *not* in reaction speed on public information — it is in *forecasting* the outcome *before* it is public (reading the docket, the hearing, the political backdrop better than consensus) so you participate in the run-up, or in *understanding the second-order effects* of a complex rule better than consensus so you capture the drift. Speed-reading the headline is competing in the one phase where you have no edge.

### Misconception 5: "A big abnormal return on the day proves the rule mattered"

A large one-day move is tempting to read as proof the rule was decisive, but two things can fool you. First, the abnormal return must be *scaled by the stock's volatility* — a 6% jump in a placid stock (t ≈ 3.5) is a real signal, while the same 6% in a stock that swings 5% a day (t ≈ 1.2) is plausibly noise, as the significance worked example showed. Second, the move may have been a *confounder*: a Fed surprise, an earnings pre-announcement, or a sector-wide shock that happened to land in your event window and had nothing to do with the ruling. The market-model subtraction removes the *market-wide* confounder, but not a company-specific one. So "the stock moved 6% on the ruling day" proves nothing on its own. You need the move to be *large relative to normal volatility* (a significant t) *and* free of competing news in a *short* window before you can credibly attribute it to the rule. Skipping that check is how people build trading theses on coincidences.

## How it shows up in real markets

We've already walked the three signature shapes; here they are side by side as a pattern library you can pattern-match against the next event.

**A sell-the-news repricing (the fade).** Bitcoin into and out of the January 2024 spot-ETF approval is the archetype: a ~+72% run-up as approval became certain, a near-zero positive surprise on the day the rule actually changed, and a ~15% fade over the following nine sessions as the run-up trade was unwound. The tell was the *priced-in fraction* — by the approval, there was essentially no probability left to reprice upward, so the asymmetry pointed down.

**A positive-surprise gap.** When a court or regulator decides *against* the consensus — striking down a rule the market expected upheld, or approving a deal the market thought blocked — the price gaps hard, because the *surprise* is large. The MedCo worked example (a +6.4% abnormal return on a ruling day, a +12.28% five-day CAR) is the shape: most of the move concentrated in the surprise on the day, because the market had been pricing the *other* outcome. The bigger the gap between the actual ruling and the priced-in baseline, the bigger the gap on the chart.

**A slow re-rating drift (the under-reaction).** The post-Ukraine defense complex is the archetype: a structural rule-of-the-game change (permanently higher NATO defense budgets) that no single day could fully price, drifting +38% over roughly three years as the multi-year earnings implication was digested. Complex, slow, structural rules drift; the trade lives in the weeks and months *after* the headline, not on the day.

### Reading the shapes as a workflow

Put together, the three cases give you a repeatable diagnostic you can run on the next event before you commit a dollar. The workflow is four questions, in order:

1. **How much is priced in?** Compute the priced-in fraction from the run-up, or back out the implied probability from the price level, or read the implied move from the straddle. If the answer is "almost everything" (the bitcoin case), the *direction* of the remaining risk is usually *against* the headline — the setup is a fade, not a chase.
2. **What does the run-up's shape say?** A smooth accelerating climb means the outcome is confidently anticipated (small gap ahead); a flat line means the surprise is still entirely ahead (large gap, either way); a sudden step means a discrete information event already moved consensus (find out what it was).
3. **Is the gap a surprise or a confirmation?** Compare the actual outcome to the priced-in baseline, never to zero and never to the headline's emotional charge. A confirmation gaps small; a surprise gaps hard (the MedCo case).
4. **Will it drift or fade?** Read the regime: complex, slow, under-covered → drift (ride it, the defense case); hyped, anticipated, crowded → fade (the bitcoin and GameStop cases). The same machinery that built the run-up tells you which way the post-event path bends.

A reader who runs those four questions on every regulatory headline will trade a fraction of the events — most fail the first question, because most are already priced — but the ones that pass are the ones where the priced-in math says real money is still on the table. The discipline of *passing on the priced events* is as much of the edge as catching the mispriced ones.

The GameStop episode adds one more lesson that the price chart alone hides: *market-structure rules can be the event*. The 28 January 2021 collapse was not driven by a fundamental change at the company; it was driven by *brokers restricting the buy side* of the trade, a risk-and-compliance response to the collateral the clearinghouse demanded as volatility exploded. In other words, a piece of the market's *legal plumbing* — clearing rules, broker risk limits, the mechanics of who can buy when — became the price-moving event itself. That is a recurring pattern: when a price detaches far enough from any anchor, the *rules that govern trading the asset* (halts, position limits, margin requirements, settlement mechanics) can intervene and force the repricing, independent of any news about the asset. Always know whether a structural rule could become the catalyst, not just a fundamental one.

## How to trade it: the four repricing trades

Everything above resolves into four distinct trades, one per phase. Each has its own entry, its own *priced-in check* (the discipline that keeps you honest about how much is already in the price), and its own invalidation — the line that tells you the thesis is wrong and to get out.

![Matrix of the four repricing trades with entry, priced-in check, and invalidation for each](/imgs/blogs/how-a-rule-becomes-a-price-expectations-drift-and-repricing-8.png)

### The run-up trade

**The thesis:** the probability of a rule change is *rising* and not yet fully priced; buy (or sell) early and ride the repricing as the odds converge toward the outcome.

- **Entry.** Enter when you have an edge in *forecasting the probability* — you've read the docket, attended (or read transcripts of) the court hearing, tracked the comment period, or understood the political backdrop better than consensus. You want to be early, while the priced-in fraction is still low.
- **Priced-in check.** Estimate the priced-in fraction directly: where does the price sit between the "rule fails" base value and the "rule passes" full value? If it's already 80%+ priced, the run-up is largely over — there's little left to capture and the asymmetry has turned against you.
- **Invalidation.** Exit if the *odds* of the outcome fall (a bad hearing, an adverse leak) or the *calendar slips* (the ruling is delayed indefinitely, so your capital sits dead). The run-up thesis is about rising probability; falling or stalling probability is the exit.
- **The instrument matters.** For a run-up over months, the stock itself is fine — you want the slow, steady repricing and you don't want to pay for an option's time decay over a long hold. If the run-up is short (weeks into a known date) and you want defined downside in case the odds collapse, a call (for an upside thesis) or a put (downside) caps your loss at the premium while keeping the upside of the repricing. The longer and more uncertain the wait, the more the stock beats the option (no decay); the shorter and more binary the event, the more the defined-risk option earns its premium.

### The gap (surprise) trade

**The thesis:** the actual ruling will differ from the priced-in baseline; trade the *surprise*, sized by how far you expect the outcome to land from consensus.

- **Entry.** Position into the event *only* when you have a differentiated view of the *outcome versus consensus* — you think the priced-in probability is wrong. Use the implied-probability math: back out the market's odds from the price, compare to your own.
- **Priced-in check.** Always compare the ruling to *consensus*, never to zero or to the headline. A "great" ruling that merely confirms an 80%-priced expectation produces a small positive surprise; a "bad" ruling that's less bad than the priced-in disaster produces a *positive* gap. The straddle's implied move tells you the magnitude the market is already paying for.
- **Invalidation.** Stand down if the outcome matches what was priced — there's no surprise, hence no edge, and you should not have a position on at all. The gap trade only exists when you genuinely disagree with consensus.
- **The instrument matters.** A binary event with a large two-sided implied move is the natural home of options. If you have a *direction* view (the surprise will be on one side), buy that side and let the defined premium cap your loss if you're wrong. If you only believe the *real* move will be *smaller* than the implied move — that the event is over-hyped — you can *sell* the straddle to collect the premium, accepting that a true surprise can hurt you badly. The straddle's implied move is the line: trade the side of it your view disagrees with.

### The drift trade

**The thesis:** the market *under-reacted* to a complex, slow, or under-covered rule on day one; the price will keep drifting the same direction as the full implication is digested.

- **Entry.** Enter *after* the announcement gap, once the day-one reaction looks *incomplete* relative to the rule's full economic impact. You're not predicting the event; you're predicting that the market mispriced its consequences.
- **Priced-in check.** The drift signal is *complexity and thin coverage*: a wonky rule, a multi-year structural change, few analysts modeling it. These are the conditions under which a day-one reaction is reliably too small (the +8.3% UtilCo drift, the +38% defense re-rating).
- **Invalidation.** Cut if the price *stalls or reverses* — if the drift doesn't materialize within your expected window, the market digested the rule faster than you thought, and the thesis is wrong.

### The fade trade

**The thesis:** the market *over-reacted* to a hyped, fully-anticipated, crowded event; the blow-off move will reverse toward the pre-event level.

- **Entry.** Fade the spike when the run-up was extreme, the event was hyped, and positioning is crowded — the conditions under which the announcement is an *exit* for the crowd, not an entry (bitcoin's −15% post-approval fade, GameStop's −44% collapse).
- **Priced-in check.** Look for crowded positioning and an *implied move that dwarfs the real economic impact* — when the price has run far past anything fundamentals justify, the fade is the higher-probability path.
- **Invalidation.** This is the dangerous trade: cut *fast* if the price keeps running past the spike. A crowded over-reaction can extend further than you expect (squeezes feed on themselves), so the fade demands a tight, mechanical stop. Being right eventually but blown out first is the most common way to lose on a fade.
- **The instrument matters.** Fading an over-extended spike with the underlying directly (shorting the stock) carries unlimited theoretical loss if the squeeze keeps running — the GameStop way to get destroyed. Defined-risk structures (buying puts, or a put spread) cap the loss at the premium and let you survive the spike's last extension while still profiting from the fade. When you're betting *against* a euphoric crowd, capping the downside is not optional; the crowd can stay irrational longer than an uncapped short can stay solvent.

Across all four trades, notice that the *vehicle* tracks the *time horizon and the risk shape*: long, slow run-ups and drifts favor the underlying (no time decay to pay); short, binary gaps and crowded fades favor defined-risk options (the premium buys you a known maximum loss). Matching the instrument to the phase is the last piece of turning the three-phase model into a real position.

### Sizing the trade against the priced-in odds

Knowing *which* trade to put on is half the job; sizing it is the other half, and the priced-in math feeds directly into the size. The cleanest sizing tool for a binary event is to compare the trade's *expected value* against its *worst-case loss*, both expressed in dollars, using your probability versus the market's.

#### Worked example: sizing a binary-event position from your edge

Return to the merger stock: \$100 if approved, \$60 if blocked, trading at \$88 (a priced-in 70% approval probability). You believe the true probability is 85%. You decide to buy 1,000 shares at \$88.

```
position cost = 1,000 * $88 = $88,000
```

Your expected value per share, using *your* 85% probability:

```
your EV per share = 0.85 * 100 + 0.15 * 60 = $94
expected gain per share = 94 - 88 = $6
total expected gain = 1,000 * $6 = $6,000
```

But the *worst case* — the merger is blocked and the stock falls to \$60 — is a real and asymmetric loss:

```
loss per share if blocked = 88 - 60 = $28
total worst-case loss = 1,000 * $28 = $28,000
```

So this position risks \$28,000 to make an *expected* \$6,000, with a \$40,000 *swing* between the two outcomes (\$100 vs \$60). Even with a genuine edge (your 85% vs the market's 70%), the worst case is more than four times the expected gain — so you size it as a *fraction* of the portfolio you can afford to lose entirely if the merger breaks. A common discipline is to cap the worst-case loss at, say, 1–2% of the book: if your fund is \$2 million, a 1.4% cap is \$28,000, which is *exactly* this position's worst case — so 1,000 shares is your maximum, and a larger edge would *not* justify a larger position, only a different one. The priced-in probability sets both the edge *and* the downside; sizing must respect the downside, not just the edge.

The unifying discipline across all four trades is the *priced-in check*. Before any regulatory-event trade, force yourself to answer one question: *how much of this is already in the price?* Back out the implied probability from the level, the implied move from the straddle, the priced-in fraction from the run-up. The headline tells you the *direction* of the news; only the priced-in math tells you whether there's any money left in it — and how much you can afford to risk on it. That is the entire edge in trading the rules.

## Further reading & cross-links

- [How law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) — the master spine: how a rule travels from statute to policy to macro to price, of which this post is the *repricing mechanics*.
- [The regulatory calendar: trading the rulemaking clock](/blog/trading/law-and-geopolitics/the-regulatory-calendar-trading-the-rulemaking-clock) — how to find the events to run this toolkit on: comment periods, effective dates, court terms.
- [Anatomy of a news reaction: spike, fade, trend](/blog/trading/event-trading/anatomy-of-a-news-reaction-spike-fade-trend) — the same three-phase shape applied to all event types, with the spike-fade-trend taxonomy.
- [Consensus, expectations, and "priced in"](/blog/trading/event-trading/consensus-expectations-and-priced-in) — a deeper dive on the priced-in baseline and how to measure consensus before an event.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the policy mechanism behind many of the rate-decision confounders you must strip out of an event window.

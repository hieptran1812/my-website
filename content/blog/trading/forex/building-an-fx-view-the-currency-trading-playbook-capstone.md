---
title: "Building an FX View: The Currency-Trading Playbook"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The capstone of the series — a repeatable five-step process for forming a currency view from the rate gap, flows, valuation, vol and policy, plus the risk overlay that tells you when you are wrong."
tags: ["forex", "currencies", "fx-trading", "carry-trade", "rate-differentials", "positioning", "risk-management", "playbook", "usd-jpy"]
category: "trading"
subcategory: "Forex"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A currency view is not a hunch; it is the output of a repeatable process that nets five inputs — the rate gap, flows and positioning, valuation, volatility, and the policy regime — into one directional call with a level that proves you wrong.
>
> - Start from the spine of the whole series: an exchange rate is the relative price of two monies, pulled by the gap between their interest rates and the flow of money across borders. Everything else is a modifier.
> - Run the same five-step loop every time: gather the five inputs, weigh them for the current regime, form a view, size it by volatility, and write down the invalidation level *before* you trade.
> - The crisis lessons are the guardrails: carry has a fat left tail, the dollar is a wrecking ball that wins in boom and in panic, and pegs that look stable break in a single morning.
> - The one number to remember: in a base case, the rate gap and carry carry roughly 30% of the weight, flows and positioning 25%, and the other three inputs share the rest — but those weights are re-ranked by the regime, not fixed.

In the small hours of August 5, 2024, a single currency pair told the story of an entire market. Over the previous month, USD/JPY had fallen from about 161.9 to 141.7 — a move of more than twelve percent in the most heavily traded funding currency on earth. The yen had been the cheapest money in the world to borrow for a decade, and the whole planet had borrowed it to buy higher-yielding everything: Mexican pesos, Brazilian reais, US tech stocks, Nvidia. When the Bank of Japan finally nudged its policy rate up and the rate gap that had powered the trade began to close, the most crowded trade in macro reversed at once. The VIX spiked to an intraday 65.7. People who had never thought about Japanese monetary policy in their lives watched their portfolios convulse because of it.

If you had read the rest of this series, none of that morning was a surprise. You would have known that USD/JPY was riding a US–Japan rate differential that had blown out past four percentage points. You would have known that positioning was historically one-sided, that implied vol was suspiciously low, and that low vol invites leverage the way a quiet road invites speeding. You would have known that the BoJ was the one central bank still anchoring the short end near zero, and that its first hike would be the tripwire. You would not have predicted the *day* — nobody does — but you would have had a *view*: a thesis about why the pair was where it was, what would change it, and the level at which you were simply wrong.

That is what this final post is about. Across forty-odd posts we built the parts: how to read a quote, what moves a currency, how carry works and how it crashes, why the dollar sits on the other side of everything, how central banks defend a level and when they fail. This capstone assembles those parts into one thing: **a repeatable process for forming an accountable currency view.** Not a system that prints money — no such thing exists — but a discipline that turns scattered information into a single, falsifiable call.

![The five inputs to a currency view: rate gap, flows, valuation, vol, and policy](/imgs/blogs/building-an-fx-view-the-currency-trading-playbook-capstone-1.png)

## Foundations: The five inputs to an FX view

Before any process, the spine. The single sentence this entire series has woven through every post:

> An exchange rate is the relative price of two monies. You never own "a currency" in isolation — every position is a pair, a spread, a relative bet. What moves it is the gap between two countries' interest rates plus the flow of money across borders.

Hold onto that, because it tells you what *kind* of thing you are forecasting. When you say "I am long USD/JPY," you are not making a statement about America or about Japan. You are making a statement about the *difference* between them — the difference in their interest rates, in the flows pulling money in and out, in how rich or cheap each money looks, in how scared the options market is, and in what each central bank will tolerate. Five differences. Five inputs.

Let me define each one from zero, because the rest of the playbook is just learning to weigh them.

**Input 1 — the rate differential and carry.** The interest-rate gap between two countries is the gravitational center of their exchange rate. If you hold the currency that pays a higher interest rate, you earn that gap as income — that income is called *carry*. A currency whose central bank pays 5% draws money the way a 5% savings account draws deposits, all else equal. This is the master variable; we built it from the ground up in [interest-rate differentials, the master variable of FX](/blog/trading/forex/interest-rate-differentials-the-master-variable-of-fx). When you read a pair, the first question is always: *which side pays, and is that gap widening or narrowing?*

**Input 2 — flows and positioning.** A rate gap tells you where money *should* go; flows tell you where it is *actually* going, and positioning tells you who is *already there*. A currency is held up by real flows — a trade surplus, foreign direct investment, remittances, portfolio inflows chasing yield — and it can be knocked over when those flows reverse. Positioning is the other half: if everyone is already long the trade, there is no one left to buy and a small shock forces a stampede for the exit. The August-2024 yen unwind was a positioning event, not a fundamentals event.

**Input 3 — valuation (PPP and the real exchange rate).** Is the currency cheap or expensive relative to what its purchasing power says it should be worth? Purchasing-power parity (PPP) anchors the question: two monies that buy the same basket of goods should trade near the ratio of their price levels. The famous shorthand is the Big Mac index. The catch, which we labored in [purchasing power parity and the real exchange rate](/blog/trading/forex/purchasing-power-parity-and-the-real-exchange-rate), is that PPP anchors *decades, not days* — it tells you the tide, not the waves. Valuation is a slow input. It rarely tells you what happens this month, but it tells you which way the wind eventually blows.

**Input 4 — volatility and risk reversals.** The options market quotes a price for fear. Implied volatility tells you how big a move the market is paying up to be protected against; the *risk reversal* — the price gap between an out-of-the-money call and an out-of-the-money put — tells you which *direction* the fear points. We unpacked this in [risk reversals and the shape of fear in FX](/blog/trading/forex/risk-reversals-and-the-shape-of-fear-in-fx). Vol is the input that sizes your position and warns you when calm is a trap.

**Input 5 — the policy regime and intervention risk.** Finally: what will the central bank tolerate? A free-floating currency, a managed float, a crawling peg, and a hard peg are five different games with five different risk profiles. The State Bank of Vietnam steering USD/VND inside a band is a different animal from the yen floating freely, which is different again from a hard peg that one day breaks. The policy regime sets the *rules* of the pair, and intervention risk is the tail that can hand you a 30% gap overnight, as the Swiss franc did in 2015.

That is the whole framework in one breath: rate gap, flows, valuation, vol, policy. The five inputs above are the columns of every currency view you will ever build. The rest of this post is the process for turning them into a call — and the discipline of knowing when the call is wrong.

Why these five and not fifty? Because they are the *complete and non-overlapping* set of forces that move a relative price. Walk them in pairs and you see there is nothing left over. The rate gap and valuation are the two *anchors* — one is the price of money over the next year or two (fast), the other is the price of goods over the next decade (slow). Flows and positioning are the two *flow* forces — one is the real money actually crossing borders, the other is the speculative money already leaning on the trade. Vol and policy are the two *regime* forces — one is the market's own pricing of how violent the next move will be, the other is the institution that can override the market entirely. Anchors, flows, regime: that taxonomy is exhaustive. Everything else you read about currencies — terms of trade, current accounts, the Dornbusch overshoot, the cross-currency basis — is a *deeper view of one of these five*, not a sixth column. Terms of trade is a flow story (commodity exports are inflows). The basis is a vol-and-plumbing story. Keeping the framework to five forces you to ask the right five questions and prevents the most common analytical failure, which is drowning a simple relative-price judgment in a hundred data series that all measure the same two or three things.

## Input 1: The rate differential is your prior

Start where the money starts. The interest-rate differential is your *prior* — the default lean before you adjust for anything else. The reason is mechanical and we proved it from covered interest parity earlier in the series: a forward exchange rate is just spot adjusted for the rate gap, because otherwise you could borrow the cheap currency, lend the expensive one, and pocket a riskless spread. The rate gap is baked into the price of money across time.

The cleanest real-world picture of this is the yen over the last six years. As the US Federal Reserve hiked aggressively while the Bank of Japan held its policy rate near zero, the US–Japan two-year yield gap exploded from under half a percentage point to more than four — and the yen fell almost exactly in step.

![USD/JPY plotted against the US-Japan 2-year yield gap from 2019 to 2025](/imgs/blogs/building-an-fx-view-the-currency-trading-playbook-capstone-4.png)

Read that chart as a single claim: the rate gap drove the pair. When the gap was near zero in 2020, USD/JPY sat around 103. As the gap blew out to 4.35 points by end-2022, the pair ran to 131, then 144, then 157. The relationship is not perfect — nothing in FX is — but it is the dominant first-order force, which is exactly why it deserves the heaviest weight in your scorecard.

#### Worked example: pricing the carry on a \$1,000,000 yen-funded position

Suppose you want to express a long-USD/JPY view by funding in yen and holding dollars — the classic carry trade. Spot is \$1 = 150.00 yen, the US one-year rate is 5.0%, and the Japanese one-year rate is 0.5%. You put on a \$1,000,000 notional position.

The carry is the rate gap: 5.0% − 0.5% = **4.5% per year**. On \$1,000,000 of notional, that is \$45,000 of annual income just for *holding the position*, before the spot rate moves a single pip. Per day, that is roughly \$45,000 ÷ 360 ≈ \$125 dropping into your account.

But the forward already knows this. The one-year forward is spot adjusted for the gap: 150.00 × (1.005 ÷ 1.050) ≈ **143.6 yen per dollar**. The forward says the yen will *strengthen* to 143.6 — exactly enough to wipe out your 4.5% carry if it comes true. Carry is a bet that the forward is *wrong*, that the yen will not strengthen as much as covered parity implies. The whole trade lives or dies on the failure of that forecast — which is the heart of [the carry trade, getting paid to hold a currency](/blog/trading/forex/the-carry-trade-getting-paid-to-hold-a-currency). The carry pays you \$125 a day to take the other side of the market's own forward.

The takeaway from Input 1: the rate gap is the engine and your prior. But the engine has a known failure mode — it pays steadily, then craters — which is why it is a prior, not a conclusion. You weight it heavily, and then you check whether the other four inputs are about to mug it.

## Input 2: Flows and positioning decide the timing

The rate gap tells you the *direction* money is pulled. Flows and positioning tell you the *timing* and the *fragility*. This is the input that separates a forecast from a trade.

There are two layers here. The first is *real flows*: the actual cross-border money that funds a currency. A trade surplus means foreigners are net buyers of your goods and therefore net buyers of your money. Foreign direct investment, remittances from workers abroad, and portfolio inflows all push the same way. A currency with a structural inflow — Switzerland's chronic current-account surplus, Japan's, Vietnam's stack of FDI and remittances — has a floor under it that a deficit currency lacks. We made this concrete for an emerging market in [USD/VND and the managed float](/blog/trading/forex/usd-vnd-and-the-managed-float-how-the-sbv-runs-the-dong): the dong is held up by a trade surplus, FDI, and remittances, and pressed down by importers and dollar demand, with the State Bank standing in the middle.

The second layer is *positioning*: who is already in the trade. This is the one that produces violence. When a trade is *consensus* — when every macro fund, every CTA, and every retail account is leaning the same way — the rate gap can be perfectly correct and the trade can still detonate, because there is no marginal buyer left and any shock forces a simultaneous exit. The August-2024 yen unwind was not caused by a change in fundamentals; the rate gap barely moved. It was caused by *too many people holding the same correct view with too much leverage*.

#### Worked example: when a correct view is a crowded trade

Return to the long-USD/JPY carry position from before. The rate gap is 4.5% in your favor — the view is *fundamentally correct*. But suppose the market is so one-sided that leveraged funds collectively hold the equivalent of \$200 billion in short-yen carry, financed at, say, 7-to-1 leverage. Each \$1,000,000 of your notional is backed by roughly \$143,000 of capital.

Now the BoJ hikes by 0.15% and signals more. The rate gap narrows trivially — from 4.5% to maybe 4.3%. On fundamentals, nothing has happened. But the *marginal* funded position is now under water on a 3% spot move, and a 3% adverse move on 7-to-1 leverage is a 21% loss of capital. Margin calls force selling, the selling moves spot another 3%, and the loop feeds itself. Your \$1,000,000 position, earning \$125 a day in carry, can lose \$20,000 of spot value in a single session — 160 days of carry gone in one morning. That is the arithmetic of [carry crashes, picking up pennies in front of a steamroller](/blog/trading/forex/carry-crashes-picking-up-pennies-in-front-of-a-steamroller).

There is a subtlety worth internalizing about *which* flows matter on *which* horizon. Real flows — trade balances, FDI, remittances — move slowly and set the long-run floor or ceiling; they are the structural current that decides where a currency drifts over years. Portfolio and speculative flows move fast and set the near-term price; they are the gusts that decide the next month. A persistent current-account deficit, like America's chronic −3.3% of GDP, is a slow drag down that must be *financed* by capital inflows — and a currency held up only by hot portfolio inflows is fragile, because hot money is precisely the money that leaves first when the regime turns. The most dangerous configuration in all of FX is a currency with a structural deficit financed by speculative inflows at a fat carry: the real flows pull down, the speculative flows hold up, and when the speculative flows reverse there is nothing underneath. That is the anatomy of a sudden stop, and it is why an EM currency view weights *the quality of the flow*, not just its sign.

The lesson of Input 2: a crowded trade is a risk *even when the thesis is right*. Always ask not just "is my view correct?" but "is my view *consensus*, how levered is the consensus, and is the currency held up by sticky real flows or flighty hot money?" The more crowded the trade and the hotter the money behind it, the smaller you size and the tighter you watch the exit.

## Input 3: Valuation tells you the tide, not the wave

Valuation is the slow input. It almost never tells you what happens this week, and traders who anchor on it get run over for years. But it tells you the *direction of the eventual reversion*, and it tells you when a trade that is working has become *dangerous* because the currency has stretched far from any reasonable fair value.

The workhorse measure is purchasing-power parity. The intuition: if a basket of goods costs 100 dollars in the US and the same basket costs 14,000 yen in Japan, then PPP fair value is 140 yen per dollar. If the market trades at 157, the yen is "cheap" versus PPP — it buys more goods abroad than its exchange rate implies. The Economist's Big Mac index is the friendly version of this calculation, and across a basket of currencies it shows enormous, persistent gaps: the Swiss franc trades dear, the yen and several Asian currencies trade deeply cheap.

The chart below is the factor view of all this — but first, the valuation point itself. In our PPP series we showed USD/JPY sitting *far* below its OECD PPP-implied "fair value" for years: PPP implied roughly 100 yen per dollar while the market traded at 131, then 157. A valuation purist who shorted USD/JPY at 120 "because it was overvalued" was right about the destination and bankrupt about the journey.

So how do you *use* valuation without being run over by it? Two ways. First, as a *governor on size*: the further a trade stretches from fair value, the smaller you hold it and the more you respect the snap-back risk, because the rubber band is loaded. Second, as a *tiebreaker*: when the rate gap and flows are ambiguous, lean toward the cheaper currency, because over multi-year horizons real exchange rates do mean-revert. Valuation rarely starts a trade; it ends one, and it sizes one.

#### Worked example: valuation as a brake, not a trigger

You are long USD/JPY at 157, riding the 4.5% carry. PPP fair value is near 100 — the yen is roughly 36% cheap versus burgers-and-baskets ((157 − 100) ÷ 157 ≈ 36%). Does that mean short the yen-funded trade? No — valuation is not a trigger, and the carry plus the rate gap still favor the dollar. But it *does* change your risk math.

A 36% gap from fair value is a loaded spring. If a catalyst hits — a BoJ hike, a US recession that collapses the rate gap — the snap-back toward fair value could be 15-20% in weeks, exactly as it was in 1998 (USD/JPY fell from 147 to 112 in days) and again in 2024. So you cut your size: instead of a \$1,000,000 position you carry \$500,000, accepting \$62.50 a day in carry instead of \$125, because the asymmetry — small steady gains versus a large sudden loss — is worse the further you are from fair value. The 36% overvaluation does not tell you to exit; it tells you to *hold half*.

The takeaway: valuation is the tide. It will not time your entry, but it will tell you when the water has gone out far enough that you should stop standing on the seabed.

## Input 4: Volatility and risk reversals price the fear

Volatility is the input that turns a directional view into a *sized* position, and it is the early-warning system for the crowded-trade risk from Input 2. The options market is constantly quoting two things you need: how *big* a move it expects, and which *direction* it is afraid of.

The "how big" is implied volatility. A pair quoting 9.5% one-month implied vol is pricing a much wider range of outcomes than one quoting 7%. The "which direction" is the **risk reversal** — the price of an out-of-the-money call minus the price of an out-of-the-money put. When the market fears the yen *strengthening* (a carry-unwind), it bids up yen calls, and the USD/JPY risk reversal goes deeply negative. A calm USD/JPY risk reversal might sit near −0.8; in a yen-up-fear regime it gaps to −4.0. That move *is* the market pricing the steamroller.

Here is the counterintuitive part, and it is the most important single idea in carry trading: **low volatility is not safety; it is an invitation to leverage, which manufactures fragility.** When a pair is quiet, funds can hold more of it for the same risk budget, so they do — and the quiet itself becomes the setup for the crash. The August-2024 unwind was preceded by *unusually low* implied vol. The calm was the trap.

To see why volatility deserves a permanent seat in your scorecard rather than an afterthought, look at what carry actually does to capital over a long horizon. It is not a smooth compounding line; it is a staircase that occasionally falls down an elevator shaft.

![A G10 FX carry index from 2007 to 2025 showing steady gains punctuated by crashes](/imgs/blogs/building-an-fx-view-the-currency-trading-playbook-capstone-6.png)

That shape — patient up-grind, then a cliff — is the signature of every short-volatility strategy, and it is why vol is not optional. The index climbs steadily from 2010 to 2014, gives back a chunk in the 2015 franc shock, climbs again, and then takes its August-2024 hit. The drawdowns are not random noise; they cluster in exactly the moments when implied vol was lowest and leverage highest. A trader who treats volatility as a minor input — something to glance at after deciding the direction — is blind to the one variable that turns a winning carry position into a margin call. The vol input is not telling you *whether* the carry is positive; it is telling you *how violently the positive can reverse*, which is the only number that matters when the reversal comes.

The full opportunity set that a disciplined view harvests — and the role volatility plays in pricing each one — shows up when you treat FX styles as systematic factors:

![The four FX style factors a currency view harvests, ranked by stylised Sharpe ratio](/imgs/blogs/building-an-fx-view-the-currency-trading-playbook-capstone-8.png)

Carry has the highest stand-alone Sharpe in this stylised picture, around 0.55 — which is *exactly why it is dangerous*. A high Sharpe in a strategy that is short volatility means the returns look smooth right up until the tail event that defines them. Trend/momentum (≈0.45) and value/PPP (≈0.35) are the diversifiers that historically pay *when carry crashes*, because trend goes short the falling currency and value is already positioned for the reversion. The dollar factor (≈0.30) is its own beast, which we will come to.

#### Worked example: sizing a \$1,000,000 view by volatility

You want to risk a fixed \$10,000 per day of expected fluctuation on your USD/JPY view. USD/JPY one-month implied vol is 9.5% annualized. To convert that to a daily move, divide by the square root of the number of trading days in a year: 9.5% ÷ √252 ≈ 9.5% ÷ 15.9 ≈ **0.60% per day**.

So a \$1,000,000 position has an expected daily swing of about \$1,000,000 × 0.60% = \$6,000. That is *less* than your \$10,000 budget, so you could in principle hold \$1,667,000 (because \$10,000 ÷ 0.60% ≈ \$1,667,000). But now the risk reversal flips: it gaps from −0.8 to −4.0, the market screaming that it fears a yen rally. You treat that as a regime change and *cut*, not add. You size to \$1,000,000 — well under the budget — precisely because the cheap protection on yen calls tells you the tail just got fatter. Volatility didn't just size the position; the *shape* of the volatility told you to be smaller than the math alone allowed.

The takeaway of Input 4: vol sizes the trade, and risk reversals are the gauge that tells you the crowd is reaching for the exit before the exit gets crowded. When the cost of protection in your direction collapses and the cost in the *other* direction spikes, that is the market handing you a warning for free.

## Input 5: The policy regime sets the rules of the game

The final input asks: what game are you even playing? A currency's policy regime determines its entire risk profile, and ignoring it is how traders blow up on the one trade where the central bank, not the market, was always going to win — or lose.

The regimes run on a spectrum, from hard pegs (a currency board, or no own currency at all) through conventional pegs, crawling pegs and managed floats, all the way to free floats. By IMF count, the world's currencies are spread fairly evenly across this spectrum — dozens of hard pegs, dozens of managed floats, a few dozen genuine free-floaters. Each regime is a different bet.

- **Free float (EUR/USD, USD/JPY, GBP/USD):** the market sets the price, the central bank influences via rates and the occasional verbal or actual intervention. Here the five-input scorecard works in its purest form.
- **Managed float / crawling peg (USD/VND, USD/CNY):** the central bank steers the currency within a band or along a controlled glide path. The trade is not "where will the market take it" but "what will the central bank *allow*." We dissected exactly this for the dong in [USD/VND and the managed float](/blog/trading/forex/usd-vnd-and-the-managed-float-how-the-sbv-runs-the-dong): the State Bank sets a daily central rate, enforces a ±5% band, and burns reserves to defend the level — until the reserves get thin.
- **Hard peg:** the central bank promises a fixed rate. This is the highest-conviction-looking trade and the most dangerous, because pegs *break*, and when they break they break all at once.

The intervention risk is the tail that lives inside every non-floating regime — and even floating ones. Japan's Ministry of Finance has repeatedly sold dollars to defend the yen, spending \$63 billion in September-October 2022 alone and tens of billions more in 2024. The Swiss National Bank held a 1.20 floor under EUR/CHF for years, its balance sheet ballooning, until on January 15, 2015 it abandoned the floor without warning and the franc gapped roughly 30% in *minutes*, bankrupting brokers. We tell that whole story in [how central banks intervene in the currency market](/blog/trading/forex/how-central-banks-intervene-in-the-currency-market). The lesson is brutal and simple: a level that a central bank is defending is a coiled spring, and you never want to be the one standing under it when it lets go.

#### Worked example: pricing the intervention tail into a \$1,000,000 position

You are long USD/JPY at 157, and you read that the Japanese MoF has intervened twice in the past quarter, selling roughly \$37 billion in July alone to defend the yen. Each intervention has snapped USD/JPY down by 3-5 yen — call it a 2.5% adverse gap — in a single session, on no warning.

Price that tail. If there is, say, a 20% chance of an intervention this month that costs you 2.5% of spot, the *expected* drag from intervention risk is 0.20 × 2.5% = 0.5% on your \$1,000,000 position, or \$5,000 of expected loss for the month. Your carry earns about \$45,000 a year, or \$3,750 a month — so a single 20%-probable intervention has an expected cost larger than a month's carry. That does not mean abandon the trade, but it means you (a) size smaller, (b) keep dry powder to add *into* an intervention spike rather than getting stopped out by it, and (c) treat the intervention level as part of your invalidation map. The policy regime is not background color; it is a quantifiable line item in your P&L.

The takeaway of Input 5: always know the regime before you know the trade. A free float rewards the scorecard; a managed float rewards reading the central bank; a peg rewards humility about the morning it breaks.

## Weighting the five inputs: the scorecard is re-ranked by regime

You now have five inputs. The art is the weighting — and the single biggest mistake is to weight them the same way in every regime. In a base case, a reasonable default scorecard looks like this:

![The base-case weights of the five inputs in an FX scorecard](/imgs/blogs/building-an-fx-view-the-currency-trading-playbook-capstone-2.png)

Rate differential and carry carry the most weight (≈30%) because it is the dominant first-order force. Flows and positioning come next (≈25%) because they time the trade and flag the crowded-trade tail. Valuation, vol, and policy each take roughly 15% in a calm regime. But — and this is the whole point — **these weights are not constants. They are re-ranked by the regime you are in.**

- In a **calm, trending, free-float regime** (USD/JPY through most of 2022-2023), the rate gap dominates and you let carry do the work. The default weights apply.
- In a **stretched, crowded regime** (USD/JPY by mid-2024), positioning and vol jump to the front. The rate gap is still your prior, but it is the *least* informative input because it is already in the price and the risk is entirely about the exit.
- In a **managed-float or peg regime** (USD/VND, EUR/CHF pre-2015), policy and intervention risk dominate. You can have the rate gap and valuation both screaming a direction and still lose, because the central bank, not the scorecard, sets the price.
- In a **risk-off / crisis regime**, everything correlates and the dollar smile takes over (more on that below). Your single most important input becomes "is the dollar bid?", and the relative-value scorecard temporarily collapses into one question.

The discipline is to *re-rank consciously*, every time. Write down which regime you think you are in, then re-order the five inputs by relevance to *that* regime, then form your view from the top two or three. A scorecard you apply mechanically with fixed weights will be right in the calm and catastrophically wrong in the storm — which is precisely when being right matters.

## Turning the scorecard into a call: the decision tree

Weights are continuous, but a *decision* is discrete: long, short, or stand aside. The bridge from a weighted scorecard to an actual position is a decision tree, and walking it the same way every time is what keeps your process honest when your gut is screaming.

![A decision tree for a currency call starting from the rate gap](/imgs/blogs/building-an-fx-view-the-currency-trading-playbook-capstone-5.png)

Read it top to bottom. The root is always Input 1: *is the rate gap widening in your favor?* That sets the prior. From there, the tree forces you to confront the inputs that can override the prior. If the gap is widening *and* positioning is not yet crowded, you take the carry — but you size it small, because a widening gap is the very thing that attracts the crowd that later stampedes. If the gap is widening but positioning is *already crowded*, you stand aside or fade, because a correct-but-consensus trade is a trap, exactly as Input 2 warned. If the gap is narrowing or already priced, the prior is gone and the tree hands the decision to valuation and policy: a cheap currency with no peg risk is a patient value long; a currency facing high intervention or peg risk is one you *avoid entirely*, no matter how attractive the carry looks, because the central bank can hand you a 30% gap overnight.

Two features of the tree are deliberate. First, *every path that ends in "stand aside" is a win* — the hardest discipline in trading is not taking the trade, and the tree gives you explicit permission. Second, the tree never lets a single attractive input carry the decision alone. A juicy rate gap cannot override crowded positioning; a cheap valuation cannot override a breakable peg. The tree is a structured way of asking, at each fork, "what would make this trade a disaster?" — and routing you away from the disaster before you size the position. It is the scorecard's continuous weights distilled into the one binary you actually have to make.

## The repeatable process: five steps from inputs to invalidation

Here is the loop, end to end. It is deliberately mechanical, because the value of a process is that it runs the same way when you are calm and when you are scared — and you make your worst decisions when you are scared.

![The repeatable five-step process from gathering inputs to setting the invalidation level](/imgs/blogs/building-an-fx-view-the-currency-trading-playbook-capstone-3.png)

**Step 1 — Gather the five inputs.** For your pair, write down one line each: the rate gap and its direction; the real flows and the positioning; the valuation versus PPP; the implied vol and risk reversal; the policy regime and any intervention level. Five lines. No view yet — just the facts.

**Step 2 — Weigh by regime.** Decide what regime you are in (calm/trending, crowded/stretched, managed/peg, risk-off) and re-rank the five inputs by relevance to that regime. The top two or three drive the view.

**Step 3 — Form the directional view.** Net the weighted inputs into one sentence: "I am [long/short] [pair] because [the top one or two inputs], over [horizon]." If you cannot write that sentence, you do not have a view — you have a feeling. The horizon matters: a carry view is a multi-month grind; an event view is days.

**Step 4 — Size by volatility.** Convert your risk budget into a position size using implied vol (the √252 calculation from Input 4), then *cut* it for crowding (Input 2), stretch from fair value (Input 3), and intervention risk (Input 5). The base-case size is the ceiling, not the target.

**Step 5 — Set the invalidation level.** Before you trade, write the price and the conditions that prove you wrong. This is the single most important step and the one amateurs skip. The invalidation level is not a stop based on how much you are willing to lose; it is the level at which your *thesis* is broken — the rate gap closed, the positioning flipped, the central bank intervened. When it trips, you are out, no negotiation.

Then you loop. A view is not a one-time act; it is a standing hypothesis you re-test against the five inputs as they move.

One discipline binds the whole loop together: **match the horizon to the input that drives the view.** A carry view powered by the rate gap is a multi-month grind — you are collecting a slow income and you should expect to hold for weeks or months, re-testing the rate-gap and positioning tripwires as you go. A valuation view powered by PPP is a multi-*year* lean — you should size it as a small, patient position and never expect it to pay this quarter. An event view powered by a policy catalyst — a central-bank meeting, an intervention, a peg under attack — is a matter of *days*, sized larger and held briefly. The single most common way a good view becomes a bad trade is a horizon mismatch: holding a multi-year valuation lean as if it were a multi-day event trade, or panicking out of a slow carry grind on a single noisy session. The horizon is not an afterthought; it is part of the view, and it determines how you size, how long you hold, and which tripwires you watch most closely.

#### Worked example: the full end-to-end view on USD/JPY

Let me run the whole loop on one trade, with one \$1,000,000 position, the way you would in a notebook.

*Step 1 — gather.* (1) Rate gap: US 2Y minus JGB 2Y is +4.05 points, the widest in a generation, and stable. (2) Flows/positioning: leveraged funds are record-long the carry; the trade is crowded. (3) Valuation: USD/JPY at 157 is ~36% above PPP fair value of ~100 — extremely cheap yen, i.e. the trade is stretched. (4) Vol: one-month implied vol is a low 9.5%, but the risk reversal has crept from −0.8 toward −2.0 — the options market is starting to pay up for yen calls. (5) Policy: free float, but the MoF has intervened twice this quarter near 158-160.

*Step 2 — weigh.* The rate gap says "long." But the regime is *crowded and stretched*, so I re-rank: positioning and vol jump to the top, policy (intervention near 160) is a hard ceiling, valuation says the spring is loaded. The rate gap is real but already fully priced.

*Step 3 — view.* "I am long USD/JPY for the carry, but small and tactical, over weeks not months, because the rate gap supports it — while positioning, the drifting risk reversal, and intervention near 160 all say the upside is capped and the tail is fat."

*Step 4 — size.* My risk budget is \$10,000 daily fluctuation. At 0.60% daily vol, a \$1,000,000 position swings \$6,000 — inside budget. But crowding, the 36% PPP stretch, and intervention risk all say *cut*, so I hold \$1,000,000, not the \$1,667,000 the vol math would permit. I earn ~\$125/day in carry and keep powder dry.

*Step 5 — invalidate.* I am wrong if: (a) the BoJ hikes and the rate gap starts closing; (b) the risk reversal gaps past −3.5 (the crowd is bolting); or (c) the MoF intervenes in size. My price stop is a daily close below 152 — a 3% adverse move that would mark the start of an unwind. If any trip, I am flat or short, no debate.

That is an accountable view. Notice it is *long the carry but braced for the crash* — it respects the prior and the tail at once. When August 5 came and USD/JPY broke 152 on its way to 142, the invalidation level did its job: you were out near 152, not riding it to 142. The carry you gave up was \$125 a day; the loss you avoided was \$50,000 of spot. That asymmetry is the entire game.

## The risk overlay: what changes my mind

A view without an invalidation map is not a view; it is a prayer. The most professional thing in this entire playbook is the discipline of writing down, in advance, what would prove you wrong — and then *acting* on it without negotiating with yourself when it happens.

![The risk overlay showing how each input has a tripwire that forces the view to be cut or held](/imgs/blogs/building-an-fx-view-the-currency-trading-playbook-capstone-7.png)

Each of the five inputs carries its own tripwire. The view is live as long as none has tripped; the moment one does, you re-evaluate and usually cut.

- **Rate-gap tripwire:** the central bank that anchors your carry hikes or signals a turn, and the gap starts to close. Your *prior* is breaking. (BoJ, August 2024.)
- **Positioning tripwire:** positioning reaches a historic extreme and starts to reverse. Your trade is now a fire in a crowded theater.
- **Vol tripwire:** the risk reversal flips hard against you — the options market reprices the tail in your direction. Cheap protection on the other side is the market warning you.
- **Policy tripwire:** the central bank intervenes, or a defended level looks ready to break. The rules of the game just changed.

The reason to write these down *before* you trade is that human beings are catastrophically bad at cutting losers in real time. In the moment, every reason to hold sounds smart, and every reason is *post-hoc rationalization* of the position you already have. The pre-committed tripwire takes the decision out of the scared brain's hands. It is the same logic as a peg's reserve line, a fund's risk limit, or a bank-run depositor's queue — a coordination problem you solve by deciding in advance, which is exactly the territory of [the central bank game, credibility and commitment](/blog/trading/game-theory/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed).

#### Worked example: the invalidation level paying for itself

You hold the \$1,000,000 long-USD/JPY carry with a daily-close stop at 152. The position earns \$125 a day. For 60 days it works — you bank \$7,500 in carry and a little spot appreciation. Then the BoJ hikes (rate-gap tripwire) and positioning starts to bolt (positioning tripwire). USD/JPY closes at 151.5. Your invalidation level trips; you flatten.

Account for it. You earned ~\$7,500 in carry over 60 days. You exited near 152, taking a small spot loss of maybe \$30,000 from the 157 entry — a net loss of roughly \$22,500. Painful. But the pair kept falling to 142 over the next week. Had you *not* honored the invalidation level, the additional move from 152 to 142 is about 6.6%, or another ~\$66,000 of loss on \$1,000,000. The invalidation level did not make you money; it *saved* you \$66,000. That is what risk management actually is — not the trades you win, but the disasters you don't fully participate in. The discipline paid for itself many times over in one decision.

The takeaway of the risk overlay: your edge in FX is not predicting the future — nobody does that reliably. Your edge is *being wrong cheaply and being right with size*. The invalidation map is how you guarantee the first half.

## Common misconceptions

**"A strong economy means a strong currency."** Intuitive and frequently false. A currency is a *relative* price, so what matters is the economy *relative to its rate gap and the priced-in expectations*, not its absolute strength. The US grew strongly in 2023 *and* the dollar was strong — but because the Fed out-hiked everyone, not because GDP was high. Japan's economy was fine in 2024 while the yen collapsed, because the *rate gap*, not growth, drove the pair. Always translate "strong economy" into "what does it do to the rate differential and the flows?" before you trade it.

**"High-yielding currencies are good investments because you earn the carry."** The carry is real, but it is compensation for a fat left tail, not a free lunch. Across history, the G10 carry index grinds higher and then craters — 1998, 2008, 2015, August 2024. A stand-alone Sharpe near 0.55 hides the fact that the *distribution* is short-volatility: smooth gains punctuated by violent losses. You are being paid to insure other people against a currency crash, and occasionally the crash comes. Earning the carry without respecting the tail is picking up pennies in front of a steamroller.

**"The forward rate predicts where spot will go."** No. The forward is spot adjusted for the rate gap — it is a *no-arbitrage* price, not a forecast. In fact, empirically the forward is a *biased* predictor: the high-yield currency tends to fall *less* than the forward implies, which is precisely why the carry trade has a positive expected return. The failure of the forward to predict spot is not a bug; it is the entire reason carry exists. Treat the forward as the market's hedge price, never as its crystal ball.

**"PPP says the yen is cheap, so I should buy yen."** Valuation is the tide, not the wave. PPP can flag a currency as 36% cheap and that currency can get *cheaper* for years while the rate gap pulls it the other way. Valuation governs your *size* and serves as a multi-year *tiebreaker*; it does not time entries. The trader who shorted "overvalued" USD/JPY at 120 on PPP grounds was right about the destination and broke before arriving.

**"A pegged currency is a safe, low-volatility bet."** A peg is low-volatility right up until the morning it isn't. The franc's EUR/CHF floor was the calmest trade in FX for three years and then gapped roughly 30% in minutes. A peg converts continuous small risk into a rare enormous one — it does not remove risk, it *repackages* it into a tail. The calm of a defended level is the most dangerous calm there is.

**"More data and more indicators make a better currency view."** The opposite is usually true. Currencies are driven by a small number of forces — the five inputs — and most of the fifty indicators a junior analyst pulls up are just noisy proxies for those same five. Stacking ten current-account-related charts does not give you ten independent signals; it gives you one signal (the flow input) measured ten noisy ways, and the false confidence of "everything agrees." The discipline of the five-input framework is *subtractive*: it forces you to collapse the noise into five judgments and then weight them. A view built from five clear lines you can defend beats a view drowned in a hundred series you cannot, because the second one is really the same five judgments with the uncertainty hidden rather than confronted. Conviction should come from the *weighting being right for the regime*, not from the *volume of supporting charts*.

## How it shows up in real markets

Let me apply the full playbook to the case that opened this post — USD/JPY in 2024 — and then sketch how the same loop reads a managed float like the dong, so you see the framework flex across regimes.

**USD/JPY, 2024 — the playbook in the storm.** Run the five inputs as they stood through the first half of 2024. *Rate gap (Input 1):* the US–Japan 2Y differential sat above four points — the strongest possible "long USD/JPY" prior, and the dominant force that had taken the pair from 103 to 157 over four years. *Flows and positioning (Input 2):* this is where the warning was. Speculative positioning in short-yen carry was historically extreme; the trade was the consensus macro trade on earth. *Valuation (Input 3):* at 157, the yen was ~36% below PPP fair value — a maximally loaded spring. *Vol and risk reversals (Input 4):* implied vol was *low*, the classic trap, and the risk reversal was drifting toward yen-call fear. *Policy (Input 5):* the MoF had intervened twice and the BoJ was inching toward its first hike — both tripwires armed.

A trader running the scorecard mechanically with fixed weights would have stayed max-long the carry on the strength of Input 1 and been destroyed on August 5. A trader running the *regime-aware* playbook would have re-ranked: in a crowded, stretched, intervention-risked regime, Inputs 2, 4, and 5 dominate, and they all said *small and braced*. The rate gap was the prior, but it was the *least* actionable input because it was fully in the price. The view that survived August was "long the carry, but small, with an invalidation at 152" — and the 152 stop is the difference between giving up \$125/day of carry and eating the full 142-handle collapse.

This is also where the *dollar smile* earns its keep, because the August unwind was not just a yen story — it was a global risk-off, and in risk-off the dollar wins. We built this in [the dollar smile, why the dollar wins in boom and in panic](/blog/trading/forex/the-dollar-smile-why-the-dollar-wins-in-boom-and-in-panic): the dollar is strong when the US outperforms (the right of the smile) *and* when the world panics and everyone scrambles for the world's reserve currency (the left of the smile), and weak only in the calm, synchronized-growth middle. In a carry-unwind panic, the dollar's safe-haven bid is a sixth input that temporarily overrides the relative-value scorecard — which is exactly why "is the dollar bid?" jumps to the front in any risk-off regime.

**USD/VND — the same loop, a different game.** Now read the dong with the identical five inputs, and watch the weights re-rank for a managed float. *Rate gap:* Vietnam runs higher rates than the US at times and lower at others, but the gap matters far less than in a free float because the State Bank, not the market, sets the daily reference rate. *Flows:* this is the dominant input for an EM currency — Vietnam's trade surplus, ~\$25 billion of FDI, and ~\$16 billion of remittances are the real floor under the dong, and the importer/dollar-demand side is the pressure. *Valuation:* the dong sits deeply cheap on PPP (Vietnam runs near −45% on the Big Mac index), which is structural, not a trade signal. *Vol:* onshore vol is low *by construction* because the band suppresses it, so the real signal lives offshore in the NDF market, where a widening NDF-implied depreciation premium flags pressure the band is hiding. *Policy:* this is the *master* input for the dong — the ±5% band, the central reference rate, the reserves the State Bank burns to defend the level. When Vietnam's reserves fell from ~\$109 billion in 2021 to ~\$82 billion in 2024 — below the three-months-of-imports adequacy rule — that reserve drain *was* the most important number on the page, more than any rate gap. For a managed float, the playbook re-ranks policy and flows to the top and demotes the rate gap, exactly as the regime demands.

There is a third lesson hiding in the dong case that completes the crisis playbook: **the dollar is a wrecking ball, and a rising dollar is itself an EM stress gauge.** When the Fed hikes and the dollar strengthens, every emerging market that borrowed in dollars sees its debt burden rise in local-currency terms, its financial conditions tighten, and its currency pressed lower — a doom loop. That is why an EM currency view can never ignore the dollar even when the local fundamentals look fine. Vietnam's 2022 reserve drain from \$109 billion toward \$87 billion happened *because* the dollar was on a tear, not because anything broke domestically. The strong-dollar wrecking ball is the macro weather that an EM-FX view operates inside, and it is why "is the dollar bid?" is the question that frames the other five inputs for any currency outside the majors. The same force that makes the dollar a safe haven in a panic — the world's structural need for it — is what makes its strength a hammer for everyone who owes it.

The point of running these cases is that the *process* is identical and the *weights* are not. Five inputs, re-ranked by regime, netted into one view, sized by vol, fenced by an invalidation level. The framework is the constant; the regime is the variable. A free float in calm rewards the rate gap; a free float in panic rewards the dollar; a crowded free float rewards positioning and vol; a managed float rewards reading the central bank and its reserves. Same five questions, different order — and getting the order right *is* the skill.

## The takeaway: the repeatable process

Strip away every chart and worked example and here is what the whole series, and this capstone, comes down to.

An exchange rate is the relative price of two monies. You read it through five inputs — the rate gap, the flows and positioning, the valuation, the vol and risk reversals, and the policy regime. You weight those inputs *by the regime you are in*, not by a fixed formula. You net them into one falsifiable sentence — long or short, this pair, because of these one or two inputs, over this horizon. You size that view by volatility and cut it for crowding, stretch, and intervention risk. And before you trade a single unit, you write down the level and conditions that prove you wrong, and you honor them without negotiation when they trip.

That last step is the one that separates a professional from a tourist. Your edge in currencies is not a crystal ball — the August 5th *date* was unknowable, and anyone who tells you otherwise is selling something. Your edge is structural: respect the prior (the rate gap), respect the tail (carry crashes, the dollar wrecking ball, pegs that break), be wrong cheaply, and be right with size. The carry you forgo by cutting at your invalidation level is the insurance premium you pay to never be the one standing under the peg when it snaps.

There is one more thing the series taught that belongs in the final word: humility about the regime. Every catastrophe in this playbook — the carry crash, the dollar wrecking ball, the broken peg — shares a single root cause, which is a trader applying the *right framework with the wrong weights for the regime they were actually in*. The carry traders of August 2024 had the rate gap dead right and the regime dead wrong; they were in a crowded, stretched, intervention-risked market and they weighted it like a calm trending one. The PPP purists who shorted "overvalued" USD/JPY had valuation right and horizon wrong. The franc longs who trusted the floor had policy backwards — they treated a defended level as a law of physics. In every case the five inputs were visible and the *re-ranking* was the failure. So the deepest skill this series can leave you with is not knowing the five inputs — that is a week's work — but developing the judgment to name the regime correctly and re-rank the inputs accordingly, *especially* when the calm of a low-vol, high-carry, everyone-agrees market is whispering that the regime will never change. It always changes.

The market does not reward the cleverest forecast. It rewards the most *disciplined process* — the one that runs the same way in the calm and in the crash, because the crash is when an undisciplined view costs you everything and a disciplined one costs you \$125 a day in forgone carry. Build the view from the five inputs. Re-rank them honestly for the regime. Net them into one falsifiable sentence with a horizon. Size it by vol and cut it for the tails. Write the level that kills it. And honor that level without negotiation when it trips — because that one act of pre-committed discipline is the whole difference between giving up a little carry and being the one standing under the peg when it snaps. Build the view. Write the level that kills it. Size it small. And let the steamroller flatten someone else.

## Further reading & cross-links

This capstone is the hub of the whole series — every input above was built from the ground up in its own post. Follow the threads:

- **The master variable — Input 1:** [interest-rate differentials, the master variable of FX](/blog/trading/forex/interest-rate-differentials-the-master-variable-of-fx) builds the rate gap and covered interest parity from zero; [the carry trade, getting paid to hold a currency](/blog/trading/forex/the-carry-trade-getting-paid-to-hold-a-currency) turns the gap into a P&L.
- **The fat tail — Input 2 and the crisis lesson:** [carry crashes, picking up pennies in front of a steamroller](/blog/trading/forex/carry-crashes-picking-up-pennies-in-front-of-a-steamroller) is the steamroller; the crowded-trade exit dynamics are the formal model in [the central bank game, credibility and commitment](/blog/trading/game-theory/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed).
- **Valuation — Input 3:** [purchasing power parity and the real exchange rate](/blog/trading/forex/purchasing-power-parity-and-the-real-exchange-rate) explains why PPP anchors decades, not days.
- **Vol and fear — Input 4:** [risk reversals and the shape of fear in FX](/blog/trading/forex/risk-reversals-and-the-shape-of-fear-in-fx) reads the options market's directional fear gauge.
- **Policy and intervention — Input 5:** [how central banks intervene in the currency market](/blog/trading/forex/how-central-banks-intervene-in-the-currency-market) is the toolkit; [USD/VND and the managed float](/blog/trading/forex/usd-vnd-and-the-managed-float-how-the-sbv-runs-the-dong) shows a managed float in action.
- **The dollar as the sixth input:** [the dollar smile, why the dollar wins in boom and in panic](/blog/trading/forex/the-dollar-smile-why-the-dollar-wins-in-boom-and-in-panic) is why "is the dollar bid?" jumps to the front in any risk-off regime.
- **The macro and cross-asset lens:** [what moves exchange rates — rates, flows, carry](/blog/trading/macro-trading/what-moves-exchange-rates-rates-flows-carry) is the policy-mechanism companion to this playbook, and [FX, currencies, the relative-value layer](/blog/trading/cross-asset/fx-currencies-the-relative-value-layer) places the whole framework inside a multi-asset portfolio.

---
title: "Buy the Rumor, Sell the News: Public Signals and the Fed"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Why a 'good' number can sell off: the market adage is a game-theoretic result about private signals, common knowledge, and trading the surprise instead of the level."
tags: ["game-theory", "trading", "buy-the-rumor", "sell-the-news", "federal-reserve", "forward-guidance", "priced-in", "surprise", "common-knowledge", "positioning"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — "Buy the rumor, sell the news" is not folklore; it is what happens when private, probabilistic signals get bought into a position over time and a public announcement turns the outcome into common knowledge that is already in the price.
>
> - A **rumor** is diffuse, private information traders accumulate slowly; the **news** is the public confirmation that makes the outcome *common knowledge* — and once it is common knowledge, it is "priced in".
> - The price move is driven by the **surprise** (actual minus expected), not the level. At consensus, the surprise is zero, so a "good" number can do nothing — or sell off as early longs unwind.
> - The Fed is the cleanest case: forward guidance and the dot plot move markets *before* any rate changes; by the meeting the decision is priced in, and the reaction is the surprise versus the dots.
> - The one rule: **fade the consensus into the event, then trade the surprise and the positioning unwind on the announcement — never the headline.**

On a Wednesday afternoon in September 2024, the Federal Reserve cut interest rates by half a percentage point — a big, "dovish" move that should, by any naive reading, have sent stocks soaring. The S&P 500 popped on the headline, then spent the next half hour giving most of it back, and closed roughly flat. The next morning it ripped to a fresh record. If you only watched the headline, the market's behavior looked insane: a large rate cut, the thing equity bulls had been praying for, and the immediate reaction was a shrug followed by a fade.

It was not insane. It was the single most reliable pattern in all of trading, and it has a name that traders have repeated for two centuries: *buy the rumor, sell the news*. The rate cut was not a surprise — the market had been pricing a cut of roughly that size for weeks, accumulating long positions the whole way. By the time Jerome Powell stepped to the podium, the cut was already in the price. The only thing left to trade was the *surprise* — and on the day, there was barely any. The people who had "bought the rumor" over the prior month had nothing left to do but sell into the confirmation.

This post builds that adage from first principles as a game-theoretic result, not a piece of trading-floor wisdom. The mental model is below: an asset drifts up as a rumor gets accumulated, plateaus once the outcome is "priced in", and then — on the very news that confirms the good outcome — sells off as the early longs unwind. Our job is to explain *why* that shape is the equilibrium, not the exception.

![Buy the rumor sell the news price path with rumor accumulation, priced-in plateau, and a sell-off on the news](/imgs/blogs/buy-the-rumor-sell-the-news-public-signals-and-the-fed-1.png)

The chart above is the whole essay in one picture. The green region is the *rumor* phase: private, fragmented signals leak in, traders accumulate, the price drifts up. The amber region is the *priced-in* plateau: the outcome is now consensus, the marginal buyer is gone, the drift stalls. The dashed line is the *news* — the public announcement that makes the outcome common knowledge. And the red region is the *sell-the-news* unwind: with nothing left to anticipate, the early longs take profit, and the price can fall on good news. We will spend the rest of this post earning every word of that sentence.

## Foundations: rumor, news, priced-in, and the surprise

Before we can play this game we have to define its pieces from zero. None of these terms requires any finance background; they each describe a different kind of *information* and what a crowd of traders does with it.

### What is a "rumor"?

A **rumor**, in the precise sense we mean, is *diffuse, private, probabilistic* information. Break that into its three parts:

- **Diffuse**: it arrives in scattered fragments over time — a stronger-than-usual jobs report here, a hawkish speech there, a leaked headline, a shift in a single Fed official's tone. No single fragment is decisive.
- **Private**: each trader sees a slightly different slice of it, and crucially, *nobody is sure what everyone else has seen.* You might be 70% convinced a rate cut is coming; you don't know whether the trader on the other side of your screen is at 60% or 80%.
- **Probabilistic**: it is a *belief*, not a fact. "I think a cut is about 70% likely" is a rumor. "The Fed cut by 50 basis points at 2:00 pm" is not — that is news.

A *basis point* — we will use this repeatedly — is one hundredth of a percent, 0.01%. A 50-basis-point cut is a half-percentage-point cut. When traders say a rate move is "50 bps", they mean 0.50%.

Because a rumor is private and probabilistic, the rational thing to do with it is *accumulate a position gradually*. You don't bet the farm on a 70% guess; you build a long position scaled to your conviction, and you add to it as more fragments confirm your view. Every trader doing this — buying a little as their private estimate firms up — pushes the price up *before anything has officially happened*. That collective drift is what the green region of the cover figure shows.

### What is "the news"?

The **news** is the *public announcement* — the official number, the rate decision, the earnings release — that arrives at a single instant, that everyone sees at the same time, and that resolves the uncertainty. The news does something a rumor can never do: it makes the outcome **common knowledge**.

*Common knowledge* is a precise idea from game theory, and it is the hinge of this entire post. A fact is common knowledge when everyone knows it, *and* everyone knows that everyone knows it, *and* everyone knows that everyone knows that everyone knows it, all the way up. (We built this idea out in full in [Common Knowledge: I Know That You Know That I Know](/blog/trading/game-theory/common-knowledge-and-i-know-that-you-know-that-i-know); here we just need the punchline.) A rumor is *not* common knowledge — it is a tower of private guesses about what others believe. The news collapses that tower into a single shared fact in one instant.

And here is the load-bearing claim of the post: **what is common knowledge is already in the price.** The moment everyone knows the outcome, and knows everyone knows it, there is no one left who hasn't already adjusted. The marginal buyer — the trader whose purchase would push the price up — has already bought, because they did it during the rumor phase. So the news, by making the outcome common knowledge, *removes* the very thing that was driving the price: the anticipation.

### What does "priced in" mean?

"**Priced in**" is the everyday phrase for the state where the market's current price already reflects an expected future outcome. If the market is 90% confident the Fed will cut by 25 bps next week, then a 25-bp cut is roughly *priced in*: when it actually happens, the price barely moves, because the price already assumed it. The outcome was anticipated, accumulated, and absorbed before the event.

The cleanest way to make this concrete is with an **expected value**. The *expected value* of an uncertain outcome is the probability-weighted average of all the possible results — what you'd get "on average" if the same situation repeated many times. If a coin pays you \$1 on heads and costs you \$1 on tails, the expected value is $0.5 \times (+\$1) + 0.5 \times (-\$1) = \$0$. The market prices the *expected* outcome, then only moves by how far reality lands from that expectation.

#### Worked example: the expected move is what gets priced in

Suppose an FOMC meeting (the Federal Open Market Committee, the Fed's rate-setting body) is a day away, and the market sees three possible outcomes, each measured by how the 2-year Treasury yield would move on the decision. From overnight-index-swap pricing, the day-before probabilities are: a 25-bp cut is 70% likely (and would push the 2-year yield down a further 8 bps because most of the cut is already in), a hold is 25% likely (which would be *hawkish* relative to a priced-in cut, pushing yields up 6 bps), and a surprise 25-bp hike is just 5% likely (a shock that would send the 2-year up 30 bps).

The expected move already in the price is the probability-weighted average:

$$E[\text{move}] = 0.70 \times (-8) + 0.25 \times (+6) + 0.05 \times (+30)$$
$$= -5.6 + 1.5 + 1.5 = -2.6 \text{ bps}.$$

So the price has *already* moved down about 2.6 bps in anticipation. Now watch what happens to each outcome. If the cut lands (70% case), the *surprise* is $-8 - (-2.6) = -5.4$ bps — a small further drop, because most of the cut was priced. If instead the Fed holds (25% case), the surprise is $+6 - (-2.6) = +8.6$ bps — a sharp move *up* in yields, even though the Fed "did nothing". The hold is the violent outcome precisely because it was *not* what the price assumed.

![Expected move priced in around an FOMC meeting with the surprise as outcome minus expectation](/imgs/blogs/buy-the-rumor-sell-the-news-public-signals-and-the-fed-7.png)

The chart shows each outcome as a bar and the dashed line as the expected move (−2.6 bps) already in the price. The arrow marks the surprise for the hold case: the price doesn't react to where the bar *is*, it reacts to the gap between the bar and the dashed line. The intuition: the market never trades the level — it trades the distance between reality and what was already assumed.

### Why accumulating on a rumor is the rational thing to do

It is worth being precise about *why* a rational trader buys gradually on a rumor rather than waiting for certainty, because the gradual accumulation is what produces the price drift, and the drift is what makes the eventual plateau and unwind inevitable.

Start with how a rational trader updates a belief. You begin with a *prior* — your estimate before seeing today's fragment. Say you start the month thinking a rate cut is 50% likely. A fragment arrives: a soft inflation print, more consistent with cutting than with holding. You revise your belief upward — to 58%, say — using **Bayes' rule**, the arithmetic of updating a probability when new evidence arrives. (We don't need the formula here; the point is only that each fragment nudges your number, and the more fragments point the same way, the higher your estimate climbs.) Over the month, a stream of dovish fragments walks your private estimate from 50% to 70% to 85%.

Now overlay *position sizing*. A disciplined trader does not bet the same amount on a 50% belief and an 85% belief — they scale the position to the edge. The classic rule of thumb for sizing to an edge is the **Kelly criterion**, which says the fraction of your capital to risk grows with your edge. (We use it heavily across this series; the math lives in the quant-finance posts.) The practical consequence is simple: *as your private estimate of the cut rises, your position rises with it.* You're not flipping from flat to all-in at some threshold; you're adding continuously as conviction firms.

Multiply that across thousands of traders, each running their own Bayesian update on their own private fragments, each scaling their long as their number climbs. The aggregate is a *steady stream of net buying* that grows as the consensus firms — and steady net buying *is* a rising price. The rumor phase isn't a handful of insiders front-running a leak; it is the lawful, mechanical result of a crowd of rational traders sizing up a belief that is collectively converging toward "cut". The green run-up in the cover figure is what a market full of Bayesian, edge-sizing traders looks like from the outside.

This also explains the plateau without any extra assumption. When almost every trader's estimate has converged near the same number — when the cut is "85% across the board" — there is no one left whose estimate is still *rising*, so there is no one left adding to longs. Net new buying goes to zero, and the price stops drifting. The outcome is now *effectively common knowledge in advance*: everyone believes it, and everyone believes everyone believes it. The drift has done its work, and the only thing left is the news to make it official — and the unwind that follows.

### Surprise versus level — the distinction that runs the whole show

Here is the single most important sentence a new trader can internalize: **the market moves on the surprise, not the level.** A 50-bp cut is not "good news" in any absolute sense. It is good news *only if the market expected less than a 50-bp cut.* If the market expected exactly 50 bps, a 50-bp cut is a non-event. And if the market expected 75 bps, a 50-bp cut is *bad* news — the Fed was less dovish than hoped — and stocks can fall on a rate cut.

Define the surprise formally:

$$\text{surprise} = \text{actual} - \text{expected}.$$

The price reaction is, to a first approximation, *proportional* to this surprise:

$$\Delta P \approx \beta \times (\text{actual} - \text{expected}),$$

where $\beta$ (beta) is the market's sensitivity — how many percent the asset moves per unit of surprise. The level of the announcement enters this equation *only* through "expected", which already lives in the current price. This is the rigorous version of "buy the rumor, sell the news", and it is why the surprise framework is the backbone of event trading; we lean on the companion treatment in [Why News Moves Markets: The Surprise Framework](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework).

![Reaction versus surprise as a line through the origin with the move near zero at consensus](/imgs/blogs/buy-the-rumor-sell-the-news-public-signals-and-the-fed-2.png)

The reaction function above is the picture of that equation. The x-axis is the surprise (actual minus expected). The y-axis is the same-day price move. The line passes *through the origin*: when the surprise is zero — when the announcement exactly matches consensus — the move is zero, no matter how big or "good" the number is. A +2-unit upside surprise drives a +1.8% rally; a downside surprise drives a sell-off. The level of the announcement is nowhere on this chart. Only the surprise is.

#### Worked example: a "good" number that does nothing

You run a small fund and you're long an equity index into a CPI report (the Consumer Price Index, the main inflation gauge). The consensus forecast is for inflation to come in at +0.3% month-over-month. The number prints at +0.3% — bang in line. Inflation is, in an absolute sense, "fine". Your naive friend texts you: "In line! Should be a relief rally, right?"

Wrong. The surprise is $0.3\% - 0.3\% = 0$. With a sensitivity of, say, $\beta = 0.8\%$ of index move per 0.1% of CPI surprise, the expected reaction is $0.8\% \times 0 = 0\%$. The index doesn't rally on the relief — it sits still, and then often *drifts down* as traders who had hedged for a bad print remove those hedges and as the longs who bought the "soft-landing rumor" over the prior week take profit. A perfectly fine number, zero surprise, and a small sell-off. That is sell-the-news in miniature: the intuition is that "fine" was already the price.

## The game: who is on the other side of the rumor?

This series insists on one question above all others: *who is on the other side, and what game are you both playing?* "Buy the rumor, sell the news" is not a pattern that happens *to* the market — it is the equilibrium of a game between two kinds of traders, and you need to know which one you are.

### The two players: the anticipators and the headline-chasers

Call the first player the **anticipator**. The anticipator buys during the rumor phase, on a private, probabilistic signal, scaling in as conviction builds. They are *long the anticipation*. Their plan is explicit: accumulate before the event, and *sell into the confirmation* — distribute their position to whoever shows up to buy on the news.

Call the second player the **headline-chaser**. The headline-chaser waits for certainty. They don't want to bet on a 70% guess; they want to buy the *fact*. So they buy *on* the announcement, when the rate cut is official, when the earnings beat is confirmed. They are buying exactly when the anticipator is selling.

Now the game is obvious. The anticipator's exit *is* the headline-chaser's entry. The reason the price can fall on good news is that the supply of stock the anticipators want to sell, at the moment of confirmation, exceeds the demand from the headline-chasers who are buying the fact. The "good news" provides the liquidity event the early longs needed to get out. You are never trading against "the news"; you are trading against the *positioning* of the people who got there before you.

![Private rumor versus public news becoming common knowledge in a before and after layout](/imgs/blogs/buy-the-rumor-sell-the-news-public-signals-and-the-fed-4.png)

The before-and-after model above draws the information transition. On the left (the rumor state, amber for risk and uncertainty), each trader holds a private signal, nobody is sure what others know, and positions accumulate slowly. On the right (the news state, blue for the public instrument and green for the resolved outcome), the announcement is public, the outcome becomes common knowledge, and — because what is common knowledge is already priced — the unwind sells. The transition from left to right is the instant the anticipators have been waiting for.

### Why the equilibrium has a plateau

A natural objection: if everyone *knows* the cut is coming, why doesn't the price jump immediately to the fully-priced level and stay there? Why a gradual drift and then a plateau?

Because the rumor is *private and probabilistic*. Early on, the signal is weak and fragmented; only the most aggressive traders act, and they act small. As more fragments arrive, more traders cross their conviction threshold and add, and the ones already in scale up. The price drift *is* the aggregate of these staggered, conviction-weighted purchases. The drift slows and plateaus when the marginal trader's private estimate has converged to the consensus — when there's almost no one left whose estimate is rising. At that point the outcome is, for practical purposes, common knowledge *in advance*, and the price has nowhere left to go on anticipation alone. The amber plateau in the cover figure is the signature of a fully-priced-in outcome.

### How much of the move happens before the event?

This is the empirically striking part, and it varies with how *telegraphed* the outcome is. The more an event is anticipated — the more the central bank has guided, the more the analysts have forecast — the more of the total move happens *before* the announcement, and the smaller the reaction on the day.

![Share of the move that happens before versus on the announcement for three event types](/imgs/blogs/buy-the-rumor-sell-the-news-public-signals-and-the-fed-3.png)

The bar chart contrasts three event types. For a telegraphed Fed cut — one the central bank has been signaling for weeks — about 85% of the total move happens *before* the meeting, in the rumor phase, and only ~15% on the day. For a quarterly earnings beat, where the company gives less advance guidance, it is more balanced, roughly 60/40. For a genuine surprise rate hike — an outcome almost no one positioned for — the split inverts: only ~15% happens before, and ~85% slams in on the announcement, because there was no rumor to accumulate. The lesson: the better-telegraphed the event, the more the action is already over by the time the headline prints.

### The unwind is a coordination game: musical chairs into the news

Here is the part of the game that makes it genuinely strategic rather than mechanical, and it is the reason "buy the rumor, sell the news" is so reliable. Every anticipator wants to do the same thing: distribute their long *into* the confirmation, selling to the headline-chasers who show up to buy the fact. But there are only so many headline-chasers, and so much demand, on the news. If all the anticipators try to sell at the same instant, they overwhelm the available bids and the price collapses *before* most of them fill. It is a game of **musical chairs**: everyone is fine as long as the music plays, but when it stops, there are more sellers than chairs, and someone is left holding the position into the drop.

This turns the exit into a *coordination problem*, and coordination problems have a characteristic instability: if everyone expects everyone else to sell on the news, the rational move is to sell *just before* the news, to beat the rush. But if everyone reasons that way, the smart money sells a day before — so the truly smart money sells two days before — and the "sell-the-news" can migrate *backward in time*, away from the announcement itself. This is why, in heavily-anticipated events, you so often see the high print *days before* the event and a soft, drifting market into the announcement. The rumor of the news is enough; the actual news arrives to a market that has already started heading for the exits.

#### Worked example: why you can't all sell at the top

Suppose 1,000 anticipators each hold 100 shares they bought on the rumor, now worth \$110, and they all plan to sell into the news. The headline-chasers, in aggregate, are willing to buy 40,000 shares at around \$110 before their demand is exhausted and the price has to drop to find more buyers. But the anticipators want to sell $1{,}000 \times 100 = 100{,}000$ shares. There are 100,000 shares for sale and only 40,000 shares of price-insensitive demand: 60,000 shares have to clear at progressively lower prices.

So at most 40% of the anticipators get out near \$110; the other 60% sell into a falling market, averaging maybe \$106. The ones who *anticipated the coordination problem* — who sold a day early at \$109, before the crowd — did better than the ones who waited for the official news at \$110 and got caught in the cascade down to \$106. The intuition: in a crowded sell-the-news, being early to the *exit* matters as much as being early to the *entry*, because the exit is a chair-shortage, not a clean fill.

#### Worked example: buying the rumor versus buying the news

Let's put real dollars on the two strategies so the payoff structure is concrete. An asset trades at \$100. A favorable outcome (say a rate cut that helps this asset) is brewing, and over three weeks the rumor accumulates: the price drifts to \$108, then to \$110 as it gets fully priced in. The announcement confirms the cut exactly in line with expectations.

**The anticipator** bought at \$100 during the rumor phase and sells into the run-up at \$108 — a clean +\$8 per share, captured *before* the news even printed. They are flat by the time the headline hits.

**The headline-chaser** waited for certainty and bought at \$110 *on* the confirmation. But with the outcome now common knowledge and the anticipators selling, the price unwinds back to \$108. The headline-chaser is down −\$2 per share, having bought the most-priced-in moment in the entire cycle. They bought the news; the news was already the price.

![Buy the rumor versus buy the news payoff matrix across in-line and beat outcomes](/imgs/blogs/buy-the-rumor-sell-the-news-public-signals-and-the-fed-6.png)

The matrix shows the full payoff table. Rows are your strategy (buy the rumor early and sell into the news, or buy the news late). Columns are how the announcement lands (in line with consensus, or a genuine positive surprise). Buy-the-rumor wins in both columns: +\$8 if it lands in line (you sold the run-up) and +\$11 if it beats (you rode the run-up *and* the surprise pop). Buy-the-news is the weak hand: −\$2 if it lands in line (the unwind), and only +\$1 even on a beat (you paid the priced-in \$110 and captured just the small surprise). The intuition: the early position has the better payoff *whatever happens*, because it owns the run-up the late buyer paid for.

## The Fed: communication as equilibrium selection

Everything so far has been general. Now the cleanest, most important real case in all of markets: the Federal Reserve. The Fed is the purest example of "buy the rumor, sell the news" because the Fed has spent forty years deliberately turning itself into a *rumor machine* — and it did so on purpose, because controlling expectations *is* monetary policy.

### Forward guidance moves markets before any rate changes

The Fed sets one interest rate (the federal funds rate) eight times a year. But the federal funds rate is an overnight rate; almost nothing in the economy is priced off it directly. Mortgages, corporate bonds, and stocks are priced off the *expected path* of rates over years. So the Fed learned that it can move long-term rates — and therefore the whole economy — *without changing the overnight rate at all*, simply by changing the market's expectation of the future path. This is **forward guidance**: the Fed talks about what it intends to do, and the market prices it in immediately.

This is a genuinely strange and powerful idea: the Fed's most potent tool is *talk*. A single sentence in a speech — "the Committee anticipates that some further policy firming may be appropriate" versus "the Committee will be patient" — can move trillions of dollars of bonds and equities before a single rate has changed. We cover the full central-bank toolkit, including how guidance sits alongside the actual rate, QE, and QT, in [The Central Bank Toolkit: Rates, QE, QT, and Forward Guidance](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance); here we focus on the *game-theoretic* function of the talk.

In game-theory terms, forward guidance is **equilibrium selection**. Many self-consistent expectations about the future are possible; the economy could coordinate on "rates stay high" or "rates fall soon", and either can be self-fulfilling. By telling everyone what *it* expects, the Fed gives every trader a *focal point* — a shared expectation everyone can coordinate on, knowing everyone else is coordinating on it too. The Fed isn't just predicting the future; it is *choosing which shared belief the market settles into*, and thereby choosing which equilibrium the economy lands in. That is common-knowledge creation deployed as a policy lever.

![The Fed expectations game pipeline from forward guidance to priced in to the surprise versus the dots](/imgs/blogs/buy-the-rumor-sell-the-news-public-signals-and-the-fed-5.png)

The pipeline above traces the full game. Forward guidance — speeches, minutes, the dot plot — moves the curve with no rate change. That gets priced in: futures imply a path, consensus forms, and the move has already happened. Then the meeting arrives, the decision and new dots become common knowledge at once, and the reaction is the *surprise* — the decision and dots versus what was expected. If the Fed delivers exactly what it guided, the reaction is tiny or a sell-off; only a hawkish or dovish gap versus expectations causes real repricing.

### The dot plot: the Fed's published rumor

Four times a year, the Fed publishes the **dot plot** — a chart where each of the 19 FOMC participants marks a dot for where *they* think the federal funds rate should be at the end of this year, next year, and the year after. It is, quite literally, a published distribution of the rate-setters' private beliefs. It is the Fed handing the market its rumor in a single chart.

The market reads the *median dot* as the central expectation and prices toward it the instant it is released. This means the dot plot itself is a "news" event that creates a fresh round of common knowledge — and then a fresh round of "priced in". The famous post-meeting move is very often not about the rate decision at all (which is usually known in advance) but about whether the *dots shifted* relative to the prior dot plot. If the dots move up by one 25-bp increment versus what the market expected, that is a hawkish surprise, and bonds sell off and stocks can drop *even if the rate decision was exactly as priced*. The surprise lives in the dots, not the decision.

#### Worked example: the dovish cut that sells off

It's an FOMC day. The market is 95% priced for a 25-bp cut — it's a near-certainty, fully in the price. Treasury yields have already drifted down 12 bps over the prior two weeks anticipating it. You are long bonds (you profit when yields fall and prices rise) on the "they'll cut" rumor.

The Fed cuts 25 bps — exactly as expected. The surprise on the *decision* is zero. But the new dot plot shows the median participant now expects only *two* more cuts next year, where the prior dot plot and market consensus implied *four*. That is a hawkish shock: the Fed is cutting today but signaling a *shallower* path ahead. The 2-year yield, instead of falling, *jumps up* 10 bps on the hawkish dots. Your "dovish cut" long loses money on the day the Fed cut rates.

Walk the surprise arithmetic. The decision surprise is $25 - 25 = 0$ bps. The *path* surprise is two fewer cuts than priced, roughly $2 \times 25 = 50$ bps of removed easing spread over the next year, which the front end repriced sharply. The bond market traded the path surprise, not the decision. The intuition: at the Fed, the rate is the rumor and the dots are the news — and on a fully-priced cut, the dots are the only thing left to trade.

### Why the Fed wants the cut priced in

A subtle but crucial point: the Fed *prefers* its decisions to be fully priced in by the time it acts. A central bank that surprised the market at every meeting would create exactly the kind of volatility it exists to suppress. So the Fed deliberately guides expectations toward the decision in advance, precisely so that when the rate actually changes, the bond market barely moves — because the work was already done by the guidance.

This is why a well-run central bank produces *boring* meetings. When Powell delivers exactly what the dot plot and prior speeches led the market to expect, the reaction is a yawn — and that yawn is a *success*, not a failure. The Fed has transmitted its policy through expectations, smoothly, without a lurch. The violent FOMC reactions — the ones that make the news — are the *failures* of communication: a hawkish dot-plot shift the market didn't see coming, a press-conference comment that contradicted the guidance, a decision that broke from what was telegraphed. The signature of a credible central bank is that its biggest moves happen on its *words*, weeks before its actions, and its actions land as non-events.

This also means the Fed's most important asset is **credibility** — the market's belief that the Fed will actually do what it guides. If traders trust the guidance, they price it in fully, and the guidance becomes self-fulfilling: the Fed says "we'll be patient", the market prices a pause, financial conditions ease, and the pause becomes appropriate. If the market *doesn't* trust the guidance — because the Fed has cried wolf before — then traders won't price it in, the guidance loses its power, and the Fed is forced back onto the blunt instrument of actually moving rates. A central bank's whole forward-guidance toolkit runs on the credibility of its word, which is why central bankers guard their consistency so fiercely.

### "Priced in" before FOMC and the post-meeting reaction

By the morning of an FOMC meeting, fed funds futures and overnight-index swaps give you a precise, tradeable probability of every outcome. There is no mystery about what is "priced in" — you can read it off the screen to the basis point. This is why the professional framing of an FOMC trade is never "will they cut?" (the market already tells you the odds) but "what is the *surprise* relative to what's priced, and how is everyone *positioned* for it?" The decision is the rumor made public; the trade is the gap between the decision-plus-dots-plus-presser and the consensus, plus the unwind of whoever was offside.

This is also where [reflexivity](/blog/trading/game-theory/reflexivity-markets-that-watch-themselves) sneaks in: the Fed watches the market to gauge financial conditions, the market watches the Fed to gauge the rate path, and each is partly forecasting the other. A guidance signal that the market prices in *changes the conditions the Fed is responding to* — a feedback loop where the rumor and the news are not cleanly separable, because the market's anticipation is itself an input to the outcome.

### Reading the priced-in move from options

There is a precise, tradeable way to see how much of a move is "priced in" before an event: the **options market**. An option is a contract that pays off if the underlying moves past a certain level; the more a big move is expected, the more those contracts cost. The price of options straddling an event therefore encodes the market's *expected move* — a number you can read off the screen.

The standard construction is the **at-the-money straddle**: simultaneously owning a call (which profits if the price rises) and a put (which profits if it falls) at the current price, both expiring just after the event. The straddle pays off in either direction, so its price is roughly what the market thinks the event-driven move will be, in either direction. If the FOMC straddle costs 0.8% of the index, the market is pricing about a ±0.8% move on the decision.

This number is the bar for the *surprise*, the same way the consensus forecast is the bar for the *level*. If the actual post-meeting move is smaller than the priced straddle, the people who *bought* the straddle (betting on a big surprise) lose, and the people who *sold* it (collecting the premium, betting the event would be a non-event) win. And here is the recurring lesson: across a long run of fully-telegraphed events, the realized move tends to come in *below* the priced straddle, because the outcome was priced and the surprise was small. Selling volatility into telegraphed-but-uncertain events is a known (and dangerous, because the occasional surprise is enormous) edge for exactly the reason this whole post describes — most "news" is already the price.

#### Worked example: the straddle that overpaid for the surprise

An FOMC meeting is a day away. The at-the-money straddle on the index expiring the next day costs 1.0% of the index — the market is pricing a ±1.0% move on the decision. You judge the meeting is fully telegraphed: a 25-bp cut at 95% odds, dots unlikely to shift, Powell on-message. You think the *actual* move will be more like ±0.3%.

If you sell the straddle for 1.0% and the index moves only 0.3% on the news, you keep $1.0\% - 0.3\% = 0.7\%$ of the index as profit — a clean win, because the event was a non-event and you were paid for the volatility that never came. But if the dots shift hawkishly and the index gaps 2.5%, you lose $2.5\% - 1.0\% = 1.5\%$ — a loss five times your average win. The intuition: selling the priced-in event collects small, frequent premiums and pays out rare, brutal losses, which is the exact risk shape of betting that the news is already the price.

## Common misconceptions

**"Good news makes prices go up."** This is the deepest and most expensive misconception in trading. Prices respond to *surprises*, not to the quality of news. A blowout earnings beat that misses the *whisper number* — the unofficial, even-higher expectation that built up as a rumor — sells off. A terrible jobs report that is "less terrible than feared" rallies. The sign of the price move is the sign of (actual − expected), and "good" only enters through whether it beat the bar. Once you internalize this, the September 2024 "dovish cut that closed flat" stops being a paradox and becomes the default.

**"If everyone knows it, I can still profit by buying it."** No — if everyone knows it *and knows everyone knows it* (common knowledge), it is in the price, and your purchase at the priced-in level has no edge. The only profit in a fully-anticipated event comes from (a) having accumulated *before* it became common knowledge, or (b) correctly predicting the *surprise* — the part that wasn't priced. Buying the confirmed fact is buying at the worst moment of the cycle. The error is treating "true and important" as "tradeable"; only "true, important, and *not yet priced*" is tradeable.

**"I'll sell right before the news and avoid the unwind."** Everyone planning to "buy the rumor and sell the news" faces the same coordination problem as a game of musical chairs: if you all try to distribute into the announcement, the bids you're selling into get overwhelmed and the price gaps down *before* you fill. The unwind often *front-runs* the event by a day or two precisely because the smart money tries to beat the rush. This is why you frequently see the high *days before* the announcement, not on it — the sell-the-news can happen on the *rumor of the news*. Knowing the exit is crowded is itself part of the game.

**"The bigger the rate cut, the bigger the rally."** Only relative to expectations. A 50-bp cut when 75 was priced is a hawkish disappointment and equities can fall; a 25-bp cut when a hold was expected is a dovish shock and equities can rip. The *magnitude* of the policy move is almost irrelevant; the *gap* between the move and the priced-in expectation is everything. Traders who size off the headline number instead of the surprise consistently get the direction wrong.

**"Priced in means the price can't move on the event."** Not quite — "priced in" means the *expected* outcome can't move it, because that is already in the price. But the *surprise* always can, and the *positioning unwind* always can. A fully-priced event with zero surprise can still see a sharp move purely from the mechanical unwind of crowded positions — the longs who bought the rumor all heading for the same exit. "Priced in" kills the anticipation trade, not the volatility.

**"The pattern is so well known that it can't work anymore."** It is true that everyone knows the adage, and that knowledge does erode the simplest version — you can't just blindly short every confirmed piece of good news and expect to profit. But the *mechanism* survives, because it isn't a pattern that can be arbitraged away; it is a consequence of how information becomes common knowledge. As long as some traders accumulate on private signals before an event, the outcome will be partly priced in by the announcement, and the move will be driven by the surprise. What "everyone knowing it" changes is *where* the edge sits: it moves from the crude "short the news" toward the subtler skills of correctly reading what's priced, gauging whether positioning is actually crowded, and forecasting the surprise. The adage didn't stop working; it got harder, which is a different thing.

**"I can wait for the news to be sure, then act fast."** The premium you pay for certainty is the worst entry price in the cycle. By the time the outcome is confirmed and common knowledge, the anticipators have done their buying and are looking to sell *to you*. "Being sure" feels safe but is precisely the moment your edge is zero or negative, because the only thing certainty buys you is the right to transact at the fully-priced level. The uncomfortable truth of this game is that you get paid for bearing the *uncertainty* of the rumor phase, and you pay a toll for the *comfort* of the confirmed fact.

## How it shows up in real markets

These are real episodes; the mechanism from this post is the same one running underneath each.

**The September 2024 Fed cut.** On 18 September 2024 the FOMC cut the federal funds rate by 50 bps to a 4.75%–5.00% range — the first cut of the cycle and a larger-than-the-minimum move. Fed funds futures had been split between 25 and 50 bps going in, but a 50-bp cut was substantially priced after a dovish run of data and a *Wall Street Journal* report days earlier nudged expectations toward the larger cut. The S&P 500 ticked up on the headline, then faded during Powell's press conference to close roughly flat (down a hair), as the "buy the rumor" longs sold into a confirmation that contained little fresh surprise. The next day, 19 September, the index rallied over 1.5% to a record once traders digested that the cut was about normalization, not panic. Textbook sell-the-news on the day, with the real direction asserting itself only after the priced-in event cleared.

**The 2013 "Taper Tantrum".** In May 2013, Fed Chair Ben Bernanke merely *suggested* in congressional testimony that the Fed might begin to slow ("taper") its bond purchases later in the year. No policy changed — this was pure forward guidance, a rumor planted by the central bank itself. Yet the 10-year Treasury yield surged from about 1.6% to over 3.0% in the following months, and emerging-market currencies sold off hard. The *talk* moved markets violently because it shifted the expected path. Then, when the Fed *actually* began tapering in December 2013 — the "news" — markets barely flinched, because by then it was fully priced. The tantrum was on the rumor; the actual taper was a non-event. The entire move happened on the guidance, not the action.

**The Bitcoin spot-ETF approval, January 2024.** For over a year, a rumor accumulated that the U.S. SEC would approve spot Bitcoin exchange-traded funds. Bitcoin rallied from roughly \$25,000 in September 2023 to over \$48,000 by early January 2024 as the approval grew near-certain — the classic rumor accumulation. On 10–11 January 2024 the ETFs were approved and began trading: the news, made fully public and common knowledge. Bitcoin promptly *fell* about 20% over the next two weeks, to near \$38,000. The good news everyone had waited for marked the local top. The anticipators who bought the approval rumor sold into the launch-day buyers, and the price unwound exactly as the framework predicts.

**Earnings season and the "whisper number".** A company beats analyst consensus on earnings per share — unambiguously "good news" — and the stock drops 8% after hours. How? Because the *whisper number*, the unofficial higher bar that built up as a rumor among traders, was above the official beat. The official surprise was positive; the surprise versus the real, higher expectation was negative. This happens every quarter to richly-valued growth stocks, where the buy-side's whispered expectations run well ahead of the published consensus. The lesson recurs: the relevant "expected" is what the *positioning* implies, which can be far above the analyst number on the screen.

**OPEC production-cut announcements.** When OPEC is widely rumored to be planning a production cut, crude oil often rallies in the days *before* the meeting as traders position for tighter supply. Then, on the announcement of the cut — even a deep one — oil frequently *sells off*, because the cut was priced and the meeting also surfaces the messy details (compliance doubts, exemptions, a smaller-than-rumored figure). The "news" of a confirmed cut becomes a sell-the-news event because the *rumor* of the cut already did the work. Commodity desks treat the pre-meeting run-up, not the announcement, as the tradeable move.

**The 2024 "soft landing" rally and CPI prints.** Through 2024, equities rallied on the *rumor* of a soft landing — falling inflation without a recession. Individual in-line CPI prints, the periodic "news", frequently produced muted or negative same-day reactions even when inflation cooled, because cooling was the consensus and was already priced. The index level was driven by the slow accumulation of the soft-landing thesis (the rumor); the monthly prints that confirmed it were, on the day, near-non-events or mild fades. The macro trend was bought as a rumor over months, not on any single data point.

**The December 2018 "hawkish hike" sell-off.** On 19 December 2018 the Fed raised rates 25 bps — a move that was largely expected, so the *decision* surprise was small. But the market had been hoping the Fed would signal a *pause* given a stock-market wobble, and instead Powell's press conference and the projections came across as committed to further hikes and to shrinking the balance sheet "on autopilot". That was a hawkish *path* surprise. The S&P 500 fell about 1.5% on the day and slid into one of its worst Decembers on record, bottoming around Christmas Eve nearly 20% off its highs. The rate hike itself was priced; the market traded the surprise in the *guidance about what came next* — and got the opposite of the dovish pivot it had been positioning for. The eventual reversal came in January 2019 when the Fed signaled "patience", the dovish surprise the market had wanted, which kicked off a sharp rally. The whole episode is a clean read on the framework: small decision surprise, large path surprise, violent reaction driven entirely by the gap between the guidance and the positioning.

**The European Central Bank and "whatever it takes".** In July 2012, at the height of the euro-zone debt crisis, ECB President Mario Draghi said the bank would do "whatever it takes to preserve the euro", adding "and believe me, it will be enough". No policy changed that day — it was pure forward guidance, a verbal commitment. Yet Italian and Spanish bond yields, which had been spiking toward unsustainable levels, fell sharply over the following weeks and months, and the existential risk premium in European assets collapsed. The *talk* did the work; the actual backstop program (Outright Monetary Transactions) was announced later and was never even used. It is the most famous demonstration in modern central banking that the news a market needs is sometimes only a credible *expectation*, made common knowledge — the rumor of a backstop, believed, was the backstop.

## The playbook: how to play it

This is where every post in this series lands: who is on the other side, what game you're in, your edge, your invalidation, and your sizing and exit. "Buy the rumor, sell the news" is one of the few patterns with a genuinely repeatable structure, so the playbook is unusually concrete — but the edge is thinner than beginners think, because *everyone* knows the adage.

**Who is on the other side.** Into a telegraphed event, the headline-chasers are your counterparties — the slower money that wants to buy the confirmed fact. They provide the liquidity you sell into. *On* a genuine surprise, your counterparty flips: it is the offside positioning, the crowd that got the surprise wrong and must now reverse, who you trade *with* in the repricing. Know which of the two games — the anticipation unwind or the surprise repricing — you're in, because they require opposite trades.

**Your edge: fade the consensus into the event.** The first half of the playbook is to *anticipate*: build a position on the rumor while the signal is still private and probabilistic, scale it to your conviction, and plan to *distribute into the confirmation*. Your edge is being early and being right about what will become common knowledge — not buying the fact once it is. If you find yourself wanting to buy *because* it's now certain, you are the headline-chaser, and you are the liquidity.

**Trade the surprise, not the level.** The second half is the event itself. Before it, read what's priced — fed funds futures, options-implied moves, consensus forecasts — so you know the *bar*. Then trade only the gap: actual minus expected. If the number lands in line, expect a fade as positioning unwinds, and either stand aside or lean short into the crowded long unwind. If it surprises, trade *with* the surprise and the repricing it forces. The companion event-trading treatment in [Consensus Expectations and "Priced In"](/blog/trading/event-trading/consensus-expectations-and-priced-in) is the toolkit for reading the bar precisely.

#### Worked example: sizing the positioning-unwind trade

You've identified a fully-priced FOMC meeting: futures imply a 25-bp cut at 92% odds, the bond rally has run for two weeks, and positioning surveys show the long-bond trade is crowded. You judge the probability the meeting delivers a *hawkish surprise* (in-line cut but shallower dots) at 40%, in which case bonds fall and your *short* into the unwind makes +1.0% on your position; and a 60% chance the meeting is dovish-or-neutral and your short loses −0.5%.

Expected value of the short-into-the-unwind trade:

$$E = 0.40 \times (+1.0\%) + 0.60 \times (-0.5\%) = 0.40\% - 0.30\% = +0.10\%.$$

A positive but *thin* edge of +0.10% per unit — which tells you two things. First, size small: a thin edge does not justify a large position, because the variance dwarfs the edge. Second, the edge is real only because the long is crowded; if positioning were neutral, the unwind probability falls and the EV goes negative. The intuition: your edge in the sell-the-news trade is not the news — it is *other people's positioning*, and you must verify the crowd is actually offside before you fade them.

**Your invalidation.** The trade is wrong, and you exit, when (a) the event was *not* actually priced in — a real surprise lands and the move is the surprise, not an unwind, so a sell-the-news short gets run over; or (b) the positioning was *not* crowded, so there is no unwind to harvest and you're just short into a vacuum; or (c) the rumor is still building — you faded too early, the price hasn't plateaued, and the anticipation drift keeps going. The cleanest invalidation: if the price makes a *new high on the news itself*, the headline-chaser demand exceeded the anticipator supply, the unwind isn't happening, and you cover.

**Sizing and exit.** Because the edge is thin and the variance around events is large, size the event trade smaller than a trend trade — events are designed to be fair games once the surprise is the only unknown. Take profit on the unwind fast; these moves complete in hours, not weeks, and mean-revert as the offside positioning clears. The asymmetry to respect: buying the rumor early is a *position* trade with a comfortable run-up and a clean exit into the news; fading the news is a *fast* trade with a thin edge and a hard stop. Most of your money in this pattern should come from being early, not from being clever on the day.

#### Worked example: the full cycle P&L

Stitch the whole thing together with dollars. You buy 100 shares of an index ETF at \$100 three weeks before a telegraphed rate cut, on the rumor. Over the three weeks the price drifts to \$110 as it gets priced in — you're up $100 \times \$10 = \$1{,}000$ unrealized. The day before the event, with the price plateaued and the long crowded, you sell your 100 shares into the run-up at \$109 (you don't try to nail the top), banking $100 \times \$9 = \$900$ realized. You are now *flat into the news* — no longer the anticipator, not yet the chaser.

The cut lands exactly in line. The headline-chasers buy at \$110; the broader anticipator crowd distributes into them; the price unwinds to \$107 over the next hour. Because you're flat, the unwind costs you nothing — and if your positioning read was strong, you put on a small short at \$110 sized to the thin +0.10% EV, covering at \$107.50 for an extra $50 \times \$2.50 \approx \$125$ on a 50-share clip. Total: \$900 from the rumor, plus a small \$125 from the news. The intuition the whole post has been building to: the bulk of the money is in the rumor you accumulated, the news is where you *exit*, and the only edge left on the announcement itself is fading the crowd that bought the confirmation.

## Further reading & cross-links

- [Common Knowledge: I Know That You Know That I Know](/blog/trading/game-theory/common-knowledge-and-i-know-that-you-know-that-i-know) — the formal idea that the news creates and the rumor lacks; the hinge of why "priced in" means "common knowledge".
- [Why News Moves Markets: The Surprise Framework](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework) — the reaction-function math (move ≈ β × surprise) in full, with the event-study mechanics.
- [Consensus Expectations and "Priced In"](/blog/trading/event-trading/consensus-expectations-and-priced-in) — how to read the bar precisely: futures-implied odds, the whisper number, and what counts as the "expected".
- [The Central Bank Toolkit: Rates, QE, QT, and Forward Guidance](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) — how forward guidance sits alongside the actual policy levers, and why talk is the Fed's most potent tool.
- [Reflexivity: Markets That Watch Themselves](/blog/trading/game-theory/reflexivity-markets-that-watch-themselves) — the feedback loop where the market's anticipation of the Fed becomes an input to what the Fed does, blurring rumor and news.

*This is educational, not investment advice. Every event trade that can make money can lose it: a fully-priced event can still gap on a surprise, and fading a crowd that turns out not to be offside is a fast way to get run over. Size to the thin edge, not the strong story.*

---
title: "The Game-Theoretic Trading Playbook: Who, What, Edge, Sucker, Level"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "A repeatable pre-trade checklist that turns game theory into five questions you run before any position: who is on the other side, what game it is, where your edge is, whether you are the sucker, and what level you should reason at."
tags: ["game-theory", "trading", "expected-value", "edge", "counterparty", "level-k", "crowded-trades", "risk-management", "decision-framework"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A trade is not a bet against nature; it is a strategic interaction with a specific counterparty, so before you click you should be able to answer five questions, every time, in order.
>
> - **Who** is on the other side? Name them — hedger, informed, noise, market-maker, or forced. A fast fill from an informed counterparty is a warning, not a win.
> - **What** game am I in? Zero-sum (every derivative), positive-sum (equities over decades), or negative-sum (everything after costs)? A different game demands a different strategy.
> - Where is my **edge**, and is it real? Edge is expected value, not win-rate, and it only exists relative to a counterparty you can name.
> - Am I the **sucker**? If you can't name your edge and your counterparty, the seat at the table is yours.
> - What's my **level**? Reason one step above where the crowd actually reasons — not all the way to the textbook equilibrium, where nobody is.
> - **The one rule:** if you cannot complete the sentence "I make money because _____ is on the other side and they are wrong because _____," do not put the trade on.

You are about to buy something. Maybe it is 100 shares of a stock that just gapped up on an earnings beat. Maybe it is a short-dated call option because the chart "looks ready." Maybe it is a futures contract because a chart pattern triggered. Your finger is over the button. Stop, for thirty seconds, and ask: *who is selling this to me, and why are they so happy to let me have it at this price?*

That single question — taken seriously — is the difference between a trader and a tourist. The tourist thinks of the market as a slot machine: pull the lever, hope the number goes up. The trader knows that every share they buy was sold by a real human or a real algorithm who looked at the same screen, did their own math, and concluded that *selling to you, right now, at this price* was the smart move. One of you is wrong. The entire discipline of trading is figuring out, before you commit a dollar, whether that person is you.

This post is the field manual for that thirty seconds. It is the last working session of a long series, and it does one job: it turns everything we have built — counterparties, zero-sum versus positive-sum, expected value, the sucker principle, level-k reasoning, crowded trades, commitment, mixed strategies — into a single repeatable checklist you can run before any trade. We will define every term from scratch, walk a full worked trade through all seven questions with real numbers, and leave you with something you can pin above your screen. The diagram below is the whole loop; the rest of the post is how to run each step.

![Pre-trade checklist loop with seven questions who what edge sucker level crowded exit](/imgs/blogs/the-game-theoretic-trading-playbook-who-what-edge-sucker-level-1.png)

A quick honesty note before we start: this is educational, not financial advice. Nothing here tells you to buy or sell anything. It teaches you a way to *think* about whatever you are already considering, so that when you do act, you act with your eyes open rather than your eyes closed.

## Foundations: the five questions, from zero

Let us define the building blocks first, because the whole checklist rests on a handful of plain ideas. If you have read the rest of the series these will be familiar; if not, this section gets you to the starting line.

**A trade is a strategic interaction.** When you flip a coin, the coin does not care what you bet. Nature is indifferent; it has no plan. A market is the opposite. Every price you see exists because someone is willing to sell there and someone is willing to buy there, and both of them are *reacting to each other and to you*. The word for "a situation where my best move depends on what you do, and your best move depends on what I do" is a **game**, and the branch of math that studies it is **game theory**. The single most important consequence: you are not forecasting the weather, you are playing chess against an opponent who is also trying to beat you. This is the spine of the whole series, laid out in [the trade is a game](/blog/trading/game-theory/the-trade-is-a-game-why-markets-are-strategic-not-random).

**A counterparty is the person on the other side.** Every trade has exactly two sides. The *counterparty* is whoever takes the opposite side of yours — when you buy, they sell; when you sell, they buy. They are not an abstraction. They have a reason. Question one of the checklist is simply: *who are they, and what is their reason?*

**Expected value is the average outcome, weighted by probability.** Suppose a bet pays you \$10 if a coin lands heads and costs you \$5 if it lands tails. The *expected value* — written $EV$ — is the probability of each outcome times its payoff, summed:

$$EV = 0.5 \times (+\$10) + 0.5 \times (-\$5) = +\$2.50$$

That \$2.50 is what you would earn *on average per flip* if you could repeat the bet thousands of times. It is the single number that tells you whether a bet is worth making. Notice it has nothing to do with how *often* you win — you win only half the time here, but the bet is excellent because the wins are bigger than the losses. Hold that thought; it is the most misunderstood idea in trading.

**Win-rate is the fraction of trades that make money.** It feels like the important number. It is not. A strategy can win 90% of the time and still bleed you dry if the 10% of losses are large enough. Win-rate without payoff sizes is meaningless, the same way "I'm right 9 times out of 10" is meaningless until you know what the tenth time costs.

**The sucker principle.** There is an old poker line — usually attributed to the movie *Rounders* and to generations of card players before it — that goes: *if you've been at the table thirty minutes and you can't spot the sucker, you're the sucker.* In a zero-sum game (more on that shortly), the money has to come from somewhere. If you cannot identify whose pocket your profit is leaving, the unsettling answer is that it is leaving yours.

**Level-k reasoning.** Imagine a game where everyone secretly picks a number from 0 to 100, and whoever lands closest to two-thirds of the *average* of all guesses wins. A level-0 thinker picks randomly, say 50. A level-1 thinker reasons "if everyone picks 50, the average is 50, so I should pick two-thirds of that, about 33." A level-2 thinker says "but everyone else will think that too, so the average will be 33, and I should pick two-thirds of 33, about 22." Keep going and you spiral down toward 0, which is the textbook "perfectly rational" answer. The catch: real people stop after one or two steps. The winner is not the person who reasons all the way to 0 — that person loses, because almost nobody else got there. The winner reasons *one step deeper than the crowd actually does*. This is **level-k thinking**, and it is the most practical idea in the whole series. Full treatment in [the Keynesian beauty contest](/blog/trading/game-theory/the-keynesian-beauty-contest-and-level-k-thinking).

With those six ideas in hand, the checklist is just a disciplined way of applying them in sequence. Five core questions — Who, What, Edge, Sucker, Level — plus two follow-ups every position needs: *Is this crowded?* and *What is my exit?* Let us take them one at a time.

#### Worked example: the thirty-second sanity check

Suppose you are about to buy 100 shares of a stock at \$50, total cost \$5,000. You think it goes to \$55. Before you do, run a crude version of the loop in your head. *Who is selling?* You don't know — red flag. *What game?* Buying a stock outright is roughly positive-sum over years, so at least the structure is not stacked against you. *Edge?* "It looks ready" is not an edge; it is a feeling. *Sucker?* You cannot name your edge or your counterparty, so on the sucker test you have just failed two of two. *Level?* "It looks ready" is exactly what every chart-watcher sees, so you are at the crowd's level, not above it.

Four of five answers came back weak. That does not mean *never trade* — it means *this specific trade, framed this way, has no identified edge*, so if you take it you are gambling, and you should size it like a lottery ticket (money you can lose entirely) rather than like an investment. **The intuition: the checklist does not tell you what to do; it tells you honestly how much you actually know, so you can size to your ignorance instead of to your hope.**

## Question 1 — Who is on the other side?

The first question is the one almost nobody asks, and it is the most important. When your order fills, *someone took the other side*. Who, and why, tells you almost everything about whether that fill was good news or bad news.

There are essentially five kinds of counterparty, and a fill from each one means something different. The map below is the one to memorize.

![Counterparty map showing hedger noise informed market maker and forced traders](/imgs/blogs/the-game-theoretic-trading-playbook-who-what-edge-sucker-level-2.png)

**The hedger.** A *hedger* is someone trading to reduce a risk they already carry, not to make a directional bet. A farmer who sells wheat futures to lock in a price; an airline that buys oil futures so a fuel-price spike won't bankrupt them; a pension fund buying bonds to match its future payouts; an options dealer offloading inventory. The defining feature: the hedger is *happy to pay* to get rid of risk, the way you are happy to pay an insurance premium. When you take the other side of a hedger, you are the insurance company collecting the premium. This is the best counterparty in the world. Their willingness to trade is not information about value — it is a desire for safety. You can read the dealer's version of this in [how an options market maker thinks](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade).

**The noise trader.** A *noise trader* is someone who trades for reasons unrelated to the asset's actual value — they are following a tip, chasing momentum, rebalancing a portfolio mechanically, or simply bored. Their orders contain no information you should fear. A retail buyer piling into a meme stock because it is trending; an index fund forced to buy a stock the day it enters the index regardless of price. Trading against noise is, on average, profitable, because their flow is random with respect to value. The deeper treatment is in [noise traders and the limits of arbitrage](/blog/trading/game-theory/noise-traders-and-the-limits-of-arbitrage).

**The informed trader.** Now the scary one. An *informed trader* knows something you don't — an insider who knows earnings will miss, a quant fund whose model is faster and better than yours, a desk with a data feed you cannot afford. When an informed trader sells to you, they are selling *because they expect the price to fall*. The chilling tell: **a fast fill from an informed counterparty is bad news.** If you place a limit order to buy at \$50 and it fills *instantly and completely*, ask why someone was so eager to dump shares on you at exactly your price. The market did not do you a favor; it told you that someone who knows more than you wanted out at \$50, right now. This is the core of [adverse selection and the winner's curse](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news).

**The market maker.** A *market maker* (also called a dealer) is a professional who quotes both a price to buy (the *bid*) and a price to sell (the *ask*) at the same time, earning the gap between them — the *bid-ask spread* — as their fee for providing liquidity. They are not betting on direction; they are running a toll booth. When you cross the spread to trade immediately, you pay them. They are not your enemy, but they are a *cost*, and every round trip you make hands them a slice. The mechanics are in [who is on the other side of your trade](/blog/trading/game-theory/who-is-on-the-other-side-of-your-trade).

**The forced trader.** The most lucrative counterparty of all is the one who *has* to trade regardless of price. A leveraged fund hit with a margin call must sell to raise cash. A mutual fund facing redemptions must liquidate to pay departing investors. An index fund must buy a stock the moment it joins the index and sell the moment it leaves, at any price. An options dealer who is short gamma must buy as the price rises. None of these are trading because they think the price is right — they are trading because a rule, a contract, or a risk limit forces their hand. Being the willing, patient counterparty to a forced trader is one of the cleanest edges in markets, because you set the price and they accept it.

#### Worked example: reading a fast fill

You want to buy a thinly-traded stock. The current quote is \$20.00 bid, \$20.10 ask. You place a limit buy at \$20.05 — splitting the spread, trying to be clever. Two scenarios:

*Scenario A:* your order sits for three minutes, then fills in small pieces as natural sellers trickle in. This is a *good* fill. The slow, piecemeal nature suggests you traded against ordinary flow — likely noise or hedgers — not someone desperate to unload.

*Scenario B:* your order fills *instantly, all 5,000 shares, at \$20.05*. Now do the math on the counterparty's incentive. They had shares to sell and your \$20.05 bid was sitting below the \$20.10 ask. They could have waited for a buyer at \$20.10 and earned an extra \$0.05 × 5,000 = \$250. Instead they hit your lower bid immediately, *forgoing \$250*, to get done *now*. Why would anyone leave \$250 on the table? Either they are forced (good for you — a margin call doesn't haggle) or they know something that makes \$20.05 a *great* price to sell at because the stock is about to fall (very bad for you). You cannot tell which from the fill alone — but the instant, complete fill should make you nervous enough to recheck the news before you celebrate.

**The intuition: speed of fill is information. The market filled you fast because someone wanted out fast, and "wanting out fast" is exactly the behavior of someone who knows more than you do.**

## Question 2 — What game am I in?

Once you know *who* is across from you, ask *what kind of game* you are both playing — because the structure of the game determines whether winning is even possible on average, and how hard you have to work for it. There are three structures, and confusing them is one of the most expensive mistakes a beginner makes.

![Comparison of positive-sum equities zero-sum derivatives and negative-sum after costs](/imgs/blogs/the-game-theoretic-trading-playbook-who-what-edge-sucker-level-3.png)

**Positive-sum: the pie grows.** A *positive-sum* game is one where the total amount of money can increase, so it is possible for everyone to win. Owning a share of a real business over the long run is the canonical example. The company earns profits, reinvests, pays dividends, and grows. Historically, broad stock-market ownership has returned something like 7–10% per year nominal over many decades. Two people can both buy the same stock at different times and *both* make money, because the underlying value created by the business is real and growing. The buyer and the long-term holder are not fighting over a fixed pot; the pot itself is getting bigger.

**Zero-sum: pure transfer.** A *zero-sum* game is one where the total is fixed, so every dollar one player wins is a dollar another player loses. **Every derivative is zero-sum.** A futures contract, an option, a swap, a contract-for-difference, a spot FX trade — these are not claims on a growing business, they are *bets between two parties*. If you make \$1,000 on a futures contract, the person on the other side lost exactly \$1,000. There is no pie growing in the background to pay you both. The full argument is in [zero-sum, positive-sum, and the house](/blog/trading/game-theory/zero-sum-positive-sum-and-the-house-where-trading-profits-come-from). (For the precise mechanics of why a derivative payoff nets to zero, that post walks through the bookkeeping.) The practical consequence is severe: in a zero-sum game your profit *requires* someone else's loss, so the sucker question (Question 4) becomes life-or-death rather than optional.

**Negative-sum: the house always rakes.** Here is the part that turns the previous two on their head. Trading is not free. Every transaction pays the bid-ask spread to a market maker, a commission to a broker, slippage to the market, and — for leveraged or held positions — borrow costs, financing, and eventually taxes. These costs are the *rake*: the house's cut, the same way a poker room takes a slice of every pot. Once you subtract the rake, *every game becomes slightly negative-sum on average for the participants*. A zero-sum derivatives market with costs is a negative-sum game for the traders and a positive-sum business for the brokers and exchanges. Even the positive-sum stock market becomes a coin-flip-or-worse for the *active* trader who churns and pays the rake repeatedly, while the buy-and-hold investor who trades rarely keeps most of the pie's growth.

The test to tell the games apart is simple: *who gets paid if we both just hold and do nothing?* In equities, the company keeps earning, so the holder gets paid — positive-sum. In a derivative, holding does nothing; only the bet's resolution moves money — zero-sum. And in both, the broker gets paid every time you transact — the negative-sum overlay.

#### Worked example: the rake on an active trader

Suppose you trade a \$10,000 account, and you make 5 round-trip trades per week. Each round trip costs you, conservatively, 0.1% in spread plus slippage — that is \$10,000 × 0.001 = \$10 per round trip, or \$50 per week, \$2,600 per year. As a fraction of your \$10,000 account, that is **26% per year** bled to the rake before you have made a single correct prediction.

Now ask what edge you need just to break even. The long-run stock market returns maybe 8% a year. Your costs alone are 26%. So your trading skill must generate *more than 26% of alpha per year* — pure, costs-aside skill — just to match doing nothing. That is an extraordinary bar; the vast majority of active traders never clear it, which is exactly why most active accounts underperform a simple index fund. **The intuition: before you can beat a counterparty, you have to beat the rake, and the rake compounds with every click — so the single most reliable edge available to almost everyone is simply trading less.**

## Question 3 — Where is my edge, and is it real?

You know who is across from you and what game you are in. Now the question that separates a strategy from a hope: *do I actually have an edge, and can I prove it with a number?*

An **edge** is a positive expected value after all costs. Not a good feeling, not a clean chart, not a "high probability setup" — a positive number when you do the full $EV$ arithmetic. And here is the trap that catches nearly everyone: **edge is expected value, not win-rate.** A strategy that wins often can have a terrible edge, and a strategy that loses often can have a wonderful one. The chart below makes the point with two strategies side by side.

![Win rate bar chart versus expected value bar chart for two strategies](/imgs/blogs/the-game-theoretic-trading-playbook-who-what-edge-sucker-level-4.png)

Look at Strategy A on the left: it wins 80% of the time. It *feels* like a money machine — four out of five trades are green. Strategy B wins only 35% of the time; two out of three trades are red, which feels miserable. If you judged by win-rate alone, you would pick A every time. But look at the right panel, where we compute the actual expected value per trade. Strategy A's wins are tiny (+1 unit) and its rare losses are large (−6 units); Strategy B's losses are small (−1 unit) and its rare wins are large (+4 units). Run the arithmetic and A *loses* money on average while B *makes* it. The pretty win-rate was a trap.

The honest, hard truth embedded here connects to [expected value, edge, and variance](/blog/trading/game-theory/expected-value-edge-and-variance-thinking-like-the-house): an edge does not exist in a vacuum. It is always *relative to a counterparty*. Your edge is the gap between what you know (or how patiently you can wait, or how cheaply you can transact) and what the person across from you knows. "I have an edge because the stock will go up" is not an edge — it is a forecast, and forecasts are free, so everyone has them. "I have an edge because I am the willing buyer for a forced seller's margin liquidation, and I can hold through the volatility they cannot" — *that* is an edge, because it names the counterparty and the specific asymmetry between you. The honesty discipline behind win-rate-versus-edge is the same one technical analysts have to learn: a high win-rate strategy with negative expectancy is the most seductive way to lose money slowly.

#### Worked example: the 80%-win strategy that loses money

Let us put dollars on the two strategies in the chart. You risk \$1,000 per trade.

*Strategy A (sell cheap insurance):* wins 80% of the time, each win earning \$1,000 × 0.001 = \$1 per unit (call it +1 unit). The 20% of losses cost 6 units each. Expected value per trade:

$$EV_A = 0.80 \times (+1) + 0.20 \times (-6) = 0.80 - 1.20 = -0.40 \text{ units}$$

That is **−\$0.40 per trade**. Over 100 trades you lose \$40, despite winning 80 of them. The 80 little wins (+\$80 total) cannot cover the 20 painful losses (−\$120 total).

*Strategy B (ride the rare trend):* wins 35% of the time, each win earning +4 units; the 65% of losses cost 1 unit each. Expected value:

$$EV_B = 0.35 \times (+4) + 0.65 \times (-1) = 1.40 - 0.65 = +0.75 \text{ units}$$

That is **+\$0.75 per trade**. Over 100 trades you make \$75, despite losing 65 of them. The few large wins (+\$140) swamp the many small losses (−\$65).

**The intuition: your account does not care how often you are right, it cares how much you win when right versus how much you lose when wrong. Compute the $EV$ or you are flying blind, no matter how green your win-rate looks.**

## Question 4 — Am I the sucker?

This is the question that hurts, which is exactly why it is on the list. Questions 1 through 3 build toward it. If you have honestly answered who is across from you, what game it is, and where your edge is, then Question 4 is the final integrity check: *given all that, is the seat I'm sitting in the loser's seat?*

The rule, stated plainly: **if you cannot name your edge and your counterparty, assume you are the sucker.** Not "you might be" — *assume* you are, and refuse the trade or size it as a pure gamble. This is not pessimism; it is arithmetic. In a zero-sum-after-costs game, the money flows from those who don't know what they're doing to those who do. If you cannot articulate why you are in the second group, the prior probability says you're in the first.

It is worth being concrete about what "naming your edge" requires. It is a sentence with two blanks filled in honestly: *"I expect to profit because [my specific advantage], and the counterparty is willing to lose because [their specific reason]."* If you cannot fill both blanks with something real — not "the chart is bullish" but "I have a structural reason the forced flow is mispriced" — you have not found an edge, you have found a hope wearing an edge costume.

There is a second self-check that belongs right next to the sucker question: *am I being read?* In some games — especially execution and routing — your counterparty can profit not by knowing the asset's value but simply by *predicting your behavior*. If a predatory algorithm learns that you always route large orders to the same venue at the same time, it can sit there and front-run you. The defense is not to be smarter about value; it is to be *unpredictable*. The matrix below shows why this is a coin-flip game where the only safe play is to randomize.

![Matching pennies matrix for routing orders where the predator front runs predictable flow](/imgs/blogs/the-game-theoretic-trading-playbook-who-what-edge-sucker-level-6.png)

This little 2×2 is a famous game called *matching pennies*. You (the rows) choose which venue to route your order to, A or B. The predator (the columns) chooses which venue to lurk at. If they guess right, they front-run you: you lose 1, they win 1. If they guess wrong, you slip through: you win 1, they lose 1. Notice there is *no* fixed best move for you — if you always pick A, they learn it and camp at A, and you lose every time. The only stable strategy is to play A and B with equal 50/50 probability, so that no matter what the predator does, they cannot do better than break even against you. This is a **mixed strategy** — deliberately randomizing your own actions so that an opponent who is watching cannot exploit a pattern. The full development is in [mixed strategies and the value of being unpredictable](/blog/trading/game-theory/mixed-strategies-and-the-value-of-being-unpredictable).

#### Worked example: the cost of being readable

Suppose you route a 10,000-share order to the same venue, at 10:00 a.m., every single day. A predatory algorithm notices the pattern after a week and starts sitting at that venue at 9:59, buying ahead of you and selling you the shares a fraction higher. Say they extract \$0.02 per share. That is 10,000 × \$0.02 = **\$200 per day**, or about \$50,000 over a trading year, leaking straight out of your account into theirs — *purely because your behavior was predictable*, not because they knew anything about the stock.

Now you randomize: you vary the venue, the time, and the order size, splitting into unpredictable child orders. The predator can no longer camp profitably; against a true 50/50 mix their expected gain from guessing is zero. You have not gotten smarter about the stock — you have simply stopped handing a free \$200 a day to anyone patient enough to read you. **The intuition: in a game where the counterparty profits by predicting you, your edge is your own unpredictability, and a fixed routine is a recurring donation.**

The sucker question, then, has two faces. *Am I the sucker on value?* — checked by Questions 1 through 3. *Am I the sucker on behavior?* — checked by asking whether a pattern in my own actions is being harvested. Both must come back clean.

## Question 5 — What's my level?

The final core question is about *how deep to think*, and the surprising answer is: not as deep as you can. We met the beauty contest in the foundations — the "guess two-thirds of the average" game that spirals toward 0. The trap is believing that the smartest play is to reason all the way to the bottom. It is not. The chart below shows why.

![Beauty contest convergence showing the crowd at level one and your guess one step above](/imgs/blogs/the-game-theoretic-trading-playbook-who-what-edge-sucker-level-5.png)

The grey curve is the "perfectly rational" guess at each level of reasoning. Level 0 starts at 50 (a naive midpoint). Each additional level multiplies by two-thirds, marching down: 33, 22, 15, 10, and onward toward the red dashed line at 0, which is the textbook Nash equilibrium — the guess that would win *if everyone reasoned infinitely deeply*. But here is the empirical fact, observed in experiment after experiment: real people cluster around levels 1 and 2. The crowd, on average, guesses somewhere near 33 to 22. The person who confidently submits 0 — having reasoned "all the way" — *loses badly*, because they are betting on an equilibrium nobody else reached.

The winning move is the green dot: **reason one step above where the crowd actually reasons.** If the crowd is at level 1 (around 33), you play level 2 (around 22). Not level 6. Not 0. One step deeper than the herd, no more. This is the single most actionable idea in the series, because it reframes the entire job of a trader. You are not trying to be *right in some absolute sense*; you are trying to be *one level less naive than your counterparty*. The full treatment, with the experimental data, is in [the Keynesian beauty contest and level-k thinking](/blog/trading/game-theory/the-keynesian-beauty-contest-and-level-k-thinking).

In markets this shows up constantly. When everyone "knows" a stock is going up (level 0: the consensus), the level-1 thinker asks "if everyone knows, it's already priced in, so I should fade it." But the level-2 thinker asks "everyone who reads finance blogs now knows to fade the consensus, so the fade is itself crowded — what does the *third* layer do?" The skill is not reasoning forever; it is correctly estimating *where the marginal trader actually stops*, and then stepping exactly one level beyond.

#### Worked example: pricing one level above the crowd

The beauty-contest math is exact, so let us use it. With $p = 2/3$ and a starting anchor of 50, the level-$k$ guess is $50 \times (2/3)^k$:

- Level 0: $50 \times (2/3)^0 = 50.0$
- Level 1: $50 \times (2/3)^1 \approx 33.3$
- Level 2: $50 \times (2/3)^2 \approx 22.2$
- Level 3: $50 \times (2/3)^3 \approx 14.8$
- Level 6: $50 \times (2/3)^6 \approx 4.4$

Suppose you have good reason to believe the crowd of players in this particular contest is sophisticated — finance students, say — and tends to reason to level 1, landing near 33. The naive level-0 player guesses 50 and loses. The "genius" who reasons to level 6 guesses 4.4 and loses even worse, because the actual average will be near 33, and two-thirds of 33 is about 22 — so the *winning* guess is 22, which is exactly level 2: one step above the crowd's level 1.

Translate that to a trade: the "fair value" the crowd converges on is around 22, so if the asset is trading at 33 (the crowd's naive level) you fade toward 22, and if it is already at 22 you have no edge and stand aside. **The intuition: you do not need to find the true bottom of the reasoning chain; you need to find where the crowd stops and stand exactly one rung below them — deeper is not better, it is just lonelier and wronger.**

## Question 6 and 7 — Crowded? And what's my exit?

The five core questions tell you whether a trade has an edge. Two follow-ups decide whether you survive *holding* it. They matter most precisely when the edge is real, because a real edge attracts a crowd, and a crowd has to get out the same door.

**Is this crowded?** A *crowded trade* is one where so many participants have piled into the same position that the position's own popularity becomes its biggest risk. The danger is not that the thesis is wrong — it may be perfectly right — but that everyone holding it will want to exit at the same moment, and there is no one left to sell to. This is the *exit game*: getting in is easy because the crowd is buying alongside you, but getting out is a stampede through a single doorway. The full anatomy, including how to measure crowding via positioning data, is in [crowded trades and the exit game](/blog/trading/game-theory/crowded-trades-and-the-exit-game). The practical rule: *size a crowded trade for the stampede, not for the calm.* If your position is one everyone else also holds, assume the exit will be violent and size so you can survive a gap rather than be liquidated by it.

**What's my exit — and have I committed to it?** The hardest part of trading is not entering; it is leaving. In the moment, every loss whispers "just wait, it'll come back," and every winner whispers "let it run, don't be a coward." Both whispers are usually wrong, and both are why undisciplined traders give back their edge. The defense is a **commitment device** — a decision you make *in advance and bind yourself to*, when you are calm, so that the panicked or greedy version of you in the heat of the moment cannot override it. A hard stop-loss order resting in the market is a commitment device. A pre-written rule like "close at 50% of max profit, full stop if it breaks the level on volume" is a commitment device. The theory of why precommitment beats willpower is in [commitment devices and strategic precommitment](/blog/trading/game-theory/commitment-devices-and-strategic-precommitment-in-trading). The reason it belongs on the checklist is brutal and simple: an edge you cannot hold onto through the noise is not an edge, it is a story you tell yourself before the market takes it back.

#### Worked example: sizing for the exit, not the entry

You find a genuinely good trade: a 5% expected return over a month, with what you estimate as a 70% chance of working. Tempting to put 50% of your account on it. But it is also a popular, crowded position — you can see from positioning surveys that "everyone" is long it. So you ask the exit question.

In a crowded trade, the downside is not the calm scenario (a slow 5% drawdown you could trade out of); it is the *gap* — the morning the crowd all decides to leave at once and the price simply opens 20% lower with no chance to sell on the way down. If you had 50% of your account on, a 20% gap is a 10% account hit *in one morning*, on a position you could not exit. If you instead size to 10% of your account, the same 20% gap costs you 2% of the account — painful but survivable, and you live to trade again.

So you size to 10%, not 50% — not because the edge is weak, but because the *exit* is dangerous. **The intuition: in a crowded trade you size for the day everyone leaves at once, because that day is the only one that can actually ruin you, and it always comes without warning.**

## Common misconceptions

A handful of beliefs feel like wisdom but quietly cost people money. Each is corrected below with the number or the argument.

**"A high win-rate means a good strategy."** The most expensive myth in trading. As the worked example showed, an 80%-win strategy can have an expected value of −\$0.40 per trade while a 35%-win strategy makes +\$0.75. Win-rate without payoff sizes tells you nothing. The number that matters is expected value: probability times payoff, summed, after costs. A casino wins fewer than half its individual bets at the roulette wheel relative to a player on a hot streak, yet it owns the building — because its edge is in the payoff structure, not the hit rate.

**"If I get out before the music stops, the crowded trade is fine."** This is the bagholder's anthem. The problem is coordination: *everyone* in the crowd plans to get out first, and they cannot all be first. When the exit comes, it comes as a gap — the price jumps from \$100 to \$80 between one trade and the next, with no fills available in between. You do not get to sell at \$99 on the way down; you get to sell at \$80 after the gap, alongside everyone else who also planned to be early. The math of why you usually can't beat the stampede is the exit game in [crowded trades](/blog/trading/game-theory/crowded-trades-and-the-exit-game).

**"Reasoning more deeply is always better."** No — the beauty contest proves it. The player who reasons all the way to the Nash equilibrium of 0 loses to the player who reasons one step above the crowd at 22, because the crowd never got to 0. Over-thinking is its own failure mode: you end up betting on an equilibrium no one else reached. The skill is calibration — *where does the crowd actually stop?* — not raw depth.

**"The market is random, so it's just luck."** Markets have a large random component, true, but they are not random in the way a dice roll is. A dice roll has no opponent; a market is a strategic game against adaptive counterparties. The randomness you see is the *aggregate* of many players' strategic choices, not the indifference of nature. That distinction is the entire reason an edge is possible: you cannot out-think a die, but you can out-think a predictable counterparty. The framing is in [the trade is a game](/blog/trading/game-theory/the-trade-is-a-game-why-markets-are-strategic-not-random).

**"A fast fill at my price is the market being efficient and good to me."** Backwards. A fast, complete fill at your limit price means someone was *eager* to take exactly the price you offered, when they could have waited for a better one. Eagerness to trade at your price is information — usually that they know something or are forced. A good fill is often a slow, grudging one. The adverse-selection mechanism is in [adverse selection and the winner's curse](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news).

**"I don't need to know my counterparty; I just need to be right about the asset."** In a zero-sum-after-costs game, being right about the asset is necessary but not sufficient — you also need to be right *relative to the person on the other side*, and you need the structure of who is forced or informed to be in your favor. Two traders can both be "right" that a stock is undervalued; the one who bought from a forced seller at a panic price keeps more of the reward than the one who bought from an informed seller right before the next leg down.

## How it shows up in real markets

The checklist is not an abstraction. The same five questions, run honestly, would have flagged some of the most famous blow-ups and minted some of the most famous wins. Here are concrete episodes, each read through the framework.

**GameStop, January 2021.** Run the WHO question on the shorts: hedge funds short more than 100% of the float were, the moment the price started climbing, *forced* buyers — every uptick threatened margin calls that compelled them to buy back at any price. Run it on the longs: a coordinated retail crowd playing a coordination game, holding together to squeeze the forced shorts. The WHAT was layered — cash equity for the holders, but options dealers short gamma were forced to buy stock as it rose, amplifying the move. The trap for latecomers was Question 6: by late January the long side was *extremely* crowded, and the exit, when brokers restricted buying and the squeeze unwound, was a gap from roughly \$483 toward \$54. The edge belonged to whoever was early and sized for the stampede; the suckers were those who bought the top because "it was going up," with no named edge and no exit plan. The full anatomy is in [the GameStop case study](/blog/trading/game-theory/case-study-gamestop-2021-the-coordination-game-that-broke-wall-street).

**Long-Term Capital Management, 1998.** LTCM had genuine edges — Nobel-laureate models, real pricing advantages on bond spreads. They passed Questions 1 through 3 brilliantly. They failed Question 6 catastrophically. Their trades were enormously crowded (other banks ran the same models and the same positions), and highly leveraged, so when the spreads moved against them in the Russian-default panic, *everyone* tried to exit the same converging-spread trades at once. There was no one to sell to, the gap was the stampede, and the leverage turned a survivable drawdown into insolvency. The lesson is precisely Question 6 and 7: a real edge, crowded and over-sized, with no survivable exit, is a slow-motion ruin. See [the LTCM case study](/blog/trading/game-theory/case-study-ltcm-1998-the-crowded-genius-trade).

**The Volkswagen–Porsche squeeze, 2008.** For a few hours in October 2008, Volkswagen was briefly the most valuable company in the world — not on fundamentals, but because Porsche had quietly cornered the float via options, leaving short-sellers as *forced* buyers with almost no shares available to buy back. The shorts who had not run the WHO question discovered, too late, that they were the forced counterparty in a market with no liquidity, and the price spiked from around €200 toward €1,000 as they scrambled. The willing sellers — Porsche — set the price. Detailed in [the Volkswagen–Porsche squeeze](/blog/trading/game-theory/case-study-the-volkswagen-porsche-squeeze-of-2008).

**Selling options into a calm market.** A perennial real-market pattern: a strategy that sells out-of-the-money options collects small premiums month after month, racking up an 85–90% win-rate and a beautiful equity curve — until a single volatility spike delivers a loss larger than a year of premiums. This is the win-rate trap from Question 3 made flesh. The counterparty (Question 1) buying those options is often a hedger paying for insurance, which sounds good, but the seller's edge depends entirely on whether the *volatility risk premium* — the gap between implied and realized volatility — is genuinely positive after costs. When it is, the trade has a real edge; when the seller ignores the rare-but-huge loss in their $EV$ math, they are the sucker who confused a high win-rate for an edge.

**The everyday fast fill.** You do not need a famous crisis. Every retail trader who has placed a limit order on an illiquid small-cap and watched it fill *instantly and completely* has met the informed counterparty without knowing it. The stock that "let you in" at your exact price often had a seller who knew the next earnings would disappoint, or the next financing would dilute. The checklist's Question 1 turns that instant fill from a small victory into a prompt to recheck the news before celebrating.

## The playbook: one trade through all seven questions

Now the full field exercise. We take one concrete trade and run the entire checklist on it with numbers, exactly as you would before clicking. The trade: **selling a cash-secured put.** That means you agree to *buy* 100 shares of a stock at a chosen price (the *strike*) if the price falls below it by a set date, and in exchange you collect a *premium* — a cash payment up front — today. You hold enough cash to actually buy the shares if assigned, hence "cash-secured." It is a popular income strategy, which makes it a perfect specimen to dissect. The figure below is the whole checklist applied; the prose walks each cell.

![Seven question checklist applied to selling a cash secured put with numbers](/imgs/blogs/the-game-theoretic-trading-playbook-who-what-edge-sucker-level-7.png)

Concretely: the stock trades at \$52. You sell one put with a \$50 strike expiring in 30 days, and collect a \$200 premium (\$2.00 per share × 100 shares). You set aside \$5,000 (100 × \$50) in case you are assigned the shares. Now run the loop.

**1 — Who buys my put?** Someone is paying me \$200 for the right to sell me 100 shares at \$50. Who, and why? If the buyer is a *hedger* — someone who owns the shares and wants insurance against a drop — that is a good counterparty: they are paying me a premium for safety, like an insurance customer, and on average insurers profit. But if the buyer is *informed* — a desk that knows bad news is coming and wants the right to dump shares on me at \$50 — then I am the insurance company that just wrote a policy on a house that is already on fire. The tell, again: if my put sells *instantly* at a rich premium, I should ask why someone was so eager to buy downside protection right now. I cannot always know, so I treat the uncertainty as a cost.

**2 — What game is this?** An option is a *derivative*, so this is **zero-sum**: my \$200 premium is exactly the buyer's \$200 cost. There is no growing pie. And after the bid-ask spread on the option and the commission, it is slightly *negative-sum* — the market maker who facilitated the trade took a slice of my \$200 before I ever saw it. So I am not in the friendly positive-sum world of buy-and-hold equity; I am in a pure transfer game where my profit is literally someone's loss, which means Question 4 is going to matter a lot.

**3 — Where is my edge, and is it real?** My claimed edge is the *volatility risk premium*: historically, the implied volatility baked into option prices tends to run a bit higher than the volatility that actually materializes, so option *sellers* earn a small positive expected value on average — they are collecting an insurance premium that, over many trades, exceeds the claims they pay out. Let me put a number on it. Suppose the put expires worthless 85% of the time (I keep the full \$200 premium), and the other 15% of the time I am assigned and the stock has fallen far enough that I lose, on average, \$800 net of the premium. Expected value per contract:

$$EV = 0.85 \times (+\$200) + 0.15 \times (-\$800) = \$170 - \$120 = +\$50$$

A modest but real positive edge of roughly **+\$50 per contract** — *if* the volatility risk premium is genuinely positive after costs. Note how the structure mirrors the win-rate trap from Question 3's chart: I win 85% of the time but my single losing case is four times the size of a winning one, so the whole edge lives or dies on whether the premium I collect is rich enough to cover those rare large losses. The edge is real only conditional on that premium existing; if I am selling puts on a stock whose implied volatility is *fairly* priced or *cheap*, my edge evaporates and the $EV$ goes negative. This is the difference between a forecast and an edge: my edge is structural (sell insurance only when it is overpriced), not directional (the stock will go up).

**4 — Am I the sucker?** The integrity check. Can I fill in both blanks? *"I expect to profit because I am selling volatility insurance that is priced richer than the risk warrants, and the counterparty is willing to lose on average because they value the downside protection more than its fair cost (a hedger) — or because they are forced to pay up for it."* If the buyer is instead an informed trader dumping a stock they know is about to crater, then *they* are not the sucker, *I* am — I wrote cheap insurance on a fire. The defense: I only sell puts when I can identify a plausible reason the implied volatility is rich (an earnings event the crowd is over-fearing, a broad-market panic inflating all premiums), and I never sell into a single name right before its own known catalyst, because that is where the informed counterparty lives. If I cannot name why the premium is rich, I assume I am the sucker and stand aside.

**5 — What's my level?** The crowd of retail put-sellers does the obvious thing: sell the standard 30-delta put every single week, mechanically, on whatever is popular. That is level 1 — the naive consensus strategy, and it gets crowded and over-harvested. One step above: I sell *only* when the implied-volatility rank is elevated (say, in the top half of its one-year range), so I am collecting insurance premiums when they are genuinely rich, not when they are thin. I am not reasoning all the way to some exotic level-6 strategy nobody else runs; I am simply standing one rung above the mechanical weekly seller. The crowd sells always; I sell when paid well to.

**6 — Is this crowded?** Selling volatility is the textbook crowded trade. In calm markets, an enormous number of participants — funds, retail, structured products — are all short volatility, all collecting the same small premiums. When a shock hits, they all need to buy back protection at once, and volatility gaps higher in a stampede that turns a year of small premiums into a single catastrophic loss. So I treat this position not by its calm-market edge but by its stampede risk.

**7 — What's my exit, and have I committed to it?** Because of Questions 4 and 6, I bind my hands in advance: I risk no more than **1% of my account per trade**, so that even an assignment-and-crash scenario costs me 1%, not 10%. I write the rule before I enter — *close or roll the put when it has captured 50% of its maximum profit, and take the full loss (close the position) if the stock breaks below the strike on heavy volume* — and I let a resting order, not my in-the-moment willpower, enforce it. And to avoid being read by a predator, I vary the strikes, the expirations, and the timing rather than selling the identical put on the identical schedule every week. The verdict: this trade is *tradeable* — it has a real, named edge against an identifiable counterparty — but only *small, only when the premium is rich, and only with the exit precommitted*. Run carelessly, the same trade is a slow-motion way to be the sucker.

That is the entire discipline in one trade. Notice what the checklist did: it did not tell me "sell the put" or "don't sell the put." It forced me to convert a vague income idea into an honest accounting of who I'm trading against, whether my edge is real, whether I'm the mark, where the crowd is, and how I get out. The trade that survives all seven questions is rare — and that rarity is the point. **Most trades fail the checklist, which is precisely why most trades should not be made.**

#### Worked example: the same trade, run carelessly

For contrast, here is the identical \$50-strike put sale run by someone who skips the checklist on a \$10,000 account. They sell the put because "the stock looks strong and I'll collect easy income," committing the full \$5,000 of cash-securing — half their account — to a single name. They sell it the day before earnings, when the premium is fat (which they read as "good income" rather than "the market is pricing a big move because something is coming"). Earnings miss; the stock gaps from \$52 to \$42 overnight. They are assigned 100 shares at \$50, now worth \$42 — a \$800 loss on the shares, partially offset by the \$200 premium, for a net loss of \$600, *6% of the entire account in one night* — and worse, they are now holding \$4,200 of a falling stock they never wanted, with no stop possible because the damage happened in a gap while the market was closed.

Where did they go wrong? Question 1 (they sold into informed flow, right before a known catalyst, exactly where the informed counterparty lives). Question 4 (they could not name their edge — "easy income" is not an edge). Questions 6 and 7 (they committed half the account and had no exit, so the gap was unsurvivable). The disciplined version risked 1% and avoided the pre-earnings name entirely. The trade itself was not insane; the *carelessness* was. **The intuition: the same trade can be a disciplined small edge or an account-threatening gamble — and the only difference is whether you ran the checklist before you clicked.**

## Further reading and cross-links

This post is the synthesis; the depth lives in the rest of the series. Use these as the next stop for each question on the checklist.

- **The premise (why any of this works):** [The trade is a game: why markets are strategic, not random](/blog/trading/game-theory/the-trade-is-a-game-why-markets-are-strategic-not-random) — the foundational argument that you are playing against adaptive opponents, not betting against nature.
- **Question 1 — Who:** [Who is on the other side of your trade?](/blog/trading/game-theory/who-is-on-the-other-side-of-your-trade) and [Adverse selection and the winner's curse: why a fast fill is bad news](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news).
- **Question 2 — What game:** [Zero-sum, positive-sum, and the house: where trading profits come from](/blog/trading/game-theory/zero-sum-positive-sum-and-the-house-where-trading-profits-come-from).
- **Question 3 — Edge:** [Expected value, edge, and variance: thinking like the house](/blog/trading/game-theory/expected-value-edge-and-variance-thinking-like-the-house).
- **Question 5 — Level:** [The Keynesian beauty contest and level-k thinking](/blog/trading/game-theory/the-keynesian-beauty-contest-and-level-k-thinking).
- **Questions 6 and 7 — Crowded and exit:** [Crowded trades and the exit game](/blog/trading/game-theory/crowded-trades-and-the-exit-game), [Commitment devices and strategic precommitment in trading](/blog/trading/game-theory/commitment-devices-and-strategic-precommitment-in-trading), and [Mixed strategies and the value of being unpredictable](/blog/trading/game-theory/mixed-strategies-and-the-value-of-being-unpredictable).

When this touches your life: the next time your finger is over the buy button, do not ask "will it go up?" — that is the tourist's question and it has no answer. Ask the five: *Who is selling to me? What game is this? Where is my edge, and is it real? Am I the sucker? What level am I playing?* Then check the crowd and write the exit. If you can answer all of them honestly and the trade still looks good, you have earned the click. If you cannot, you have just saved yourself from being the sucker — which, over a trading lifetime, is worth more than any single winning trade.

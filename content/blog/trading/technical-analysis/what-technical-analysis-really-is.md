---
title: "What technical analysis really is: the honest case for and against reading charts"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Technical analysis is not prophecy but a way to place probabilistic bets on price, and this post shows honestly where any real edge could come from and where the whole idea falls apart."
tags: ["technical-analysis", "trading", "market-efficiency", "random-walk", "probability", "behavioral-finance", "order-flow", "backtesting", "risk-management", "expectancy"]
category: "trading"
subcategory: "Technical Analysis"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Technical analysis (TA) is the practice of studying past price and trading volume to make bets about future price. It is not fortune-telling; at best it gives you slightly-better-than-coin-flip odds, and you only ever know if it works by measuring it.
>
> - A *chart* is just a picture of past prices. By itself it carries no promise about the future — but past prices are the only raw material TA uses.
> - There are exactly three places a real edge could hide: recurring crowd behavior, supply-and-demand information leaking into price and volume, and price levels so widely watched they become self-fulfilling.
> - The honest case against is brutal: a pure coin-flip random walk produces charts that look full of trends and "support lines" that mean nothing, and the weak-form efficient-market hypothesis says past prices are already baked into today's price.
> - Most indicators are noise or curve-fitting. Test 20 random rules and one will look brilliant by luck alone — and that lucky survivor is the one most likely to fail on new data.
> - The defensible version of TA claims a *conditional probability* ("given this setup, the odds tilt to maybe 55%"), never a prediction. The edge is small, fragile, and must be measured — see the companion post on [why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies).

Here is a question worth sitting with. If you flip a fair coin 100 times and walk a price up by \$1 on heads and down by \$1 on tails, you will get a chart. Show that chart to a hundred chart-readers without telling them how it was made, and a large share of them will draw a trend line on it, name a "support level," and tell you where it is headed next. The chart was pure noise. There was nothing to read. So what, exactly, is technical analysis reading?

![A price print moves the crowd's beliefs, beliefs split into chasing and informed buying, those orders move price again, and the chart is the visible trace of the whole feedback loop](/imgs/blogs/what-technical-analysis-really-is-1.png)

The diagram above is the mental model: a price prints, the chart records it, that record shifts what the crowd *believes*, those beliefs turn into new buy and sell orders, and those orders move the price again. The chart is the visible trace of this feedback loop. Technical analysis is the claim that the trace contains usable information about the next loop. This post is an honest accounting of when that claim is true, when it is wishful thinking, and how to tell the difference with numbers instead of hope.

One disclaimer up front, stated once: this is an educational explanation of *mechanisms*, not financial advice. Nothing here tells you to buy or sell anything. The goal is for you to understand what is actually being claimed when someone "reads a chart," so you can judge it for yourself.

## Foundations: charts, auctions, and what an "edge" means

Before we can argue about whether technical analysis works, we have to agree on what the words mean. The reader who skips this section will spend the rest of the post confused, because almost every fight about TA is secretly a fight about definitions.

### What a price chart actually is

A *price chart* is a graph with time on the horizontal axis and price on the vertical axis. Each point records the price at which a *trade* — an actual exchange of an asset for money — happened. The most common form, a *candlestick chart*, compresses a slice of time (a minute, an hour, a day) into a single mark showing four numbers: the *open* (price at the start of the slice), the *high*, the *low*, and the *close* (price at the end). That is the entire raw material. A chart is a record of trades that already happened, drawn in a way that is easy for a human eye to scan.

It is worth being blunt about what a chart is *not*. It is not a measurement of the company's profits, its debt, its products, or its prospects. It is not news. It is a picture of one number — the last agreed price — over time, plus how many shares changed hands at each step (the *volume*). Everything technical analysis does is built out of those two streams: price and volume. If a method uses earnings, interest rates, or any outside fact, it is not technical analysis; it is something else.

### Volume and liquidity: the chart's second dimension

Price gets all the attention, but the second stream — *volume*, the number of shares (or contracts, or coins) that traded in each period — is where much of the real information hides, and a technician who ignores it is reading half a book. Volume tells you *how much conviction was behind a move*. A price that jumps \$2 on a hundred thousand shares means something very different from a price that jumps \$2 on a thousand shares: the first move had a lot of money behind it and is harder to reverse; the second was thin and can vanish as quickly as it came. This is why volume confirmation is one of the oldest ideas in the field — a breakout to a new high "on volume" is treated as more meaningful than the same breakout on a quiet day.

Closely related is *liquidity* — how easily you can buy or sell a meaningful quantity without moving the price much. A *liquid* market (a giant stock, a major currency pair) has tight bid-ask spreads and deep order books, so your trade barely nudges the price. An *illiquid* market (a tiny stock, an obscure token) has wide spreads and thin books, so even a modest order swings the price and the chart becomes jumpy and easy to manipulate. Liquidity matters enormously for whether any edge is *capturable*: a pattern that "works" in an illiquid market may evaporate entirely once you account for the price you actually move against yourself when you trade. We will see this cost bite hard later.

### Price as the outcome of an auction

Where does the price come from? A modern market is a continuous *auction*. At any instant there is a list of people willing to buy (the *bids*, each a price and a quantity) and a list of people willing to sell (the *asks* or *offers*). The highest bid and the lowest ask are the *best bid* and *best ask*; the gap between them is the *bid-ask spread* — the small cost you pay to trade immediately. A *trade* occurs when a buyer agrees to pay a seller's asking price, or a seller agrees to a buyer's bid. The price you see printed on the chart is simply the price of the most recent such agreement.

This matters because it tells you what a price *is*: the single number at which the marginal buyer and the marginal seller agreed, right now. It is not "the value of the company." It is the clearing price of an auction that never stops. When more money wants in than out at the current price, the price has to rise to find sellers; when more wants out, it falls to find buyers. *Supply and demand* — the quantity people want to sell versus the quantity people want to buy — set the price the same way they set the price of apples at a market, except the auction runs every millisecond and the participants can change their minds instantly.

### The two schools: technical versus fundamental

Two broad philosophies try to forecast price. *Fundamental analysis* asks "what is this asset actually worth?" — it studies earnings, cash flows, assets, debt, the economy, the management — and bets that price will eventually move toward that estimated worth. *Technical analysis* sidesteps the question of worth entirely. It says: all of that information, and the emotions of everyone acting on it, is already summarized in the price and volume. So instead of estimating worth, study the *behavior of the price itself* and bet on its next move.

A technician's core assumptions, stated plainly, are usually three: (1) price reflects everything knowable, (2) prices move in trends and patterns that tend to persist or repeat, and (3) history rhymes because human psychology does not change. Notice that assumption (2) is doing enormous work — and it is exactly the assumption the critics attack. The rest of this post is, in large part, a stress test of assumption (2).

### What "an edge" means

This is the most important definition in the post, so we will be careful. An *edge* is a measurable, repeatable tendency for your bets to make money on average, *after costs*, beyond what pure chance would give you. The phrase "on average" is load-bearing. Any single trade can win or lose for reasons that have nothing to do with skill. An edge is a statement about the *long-run distribution* of outcomes, not about the next trade.

The cleanest way to think about an edge is *expected value* (EV) — the average outcome of a bet if you could repeat it forever. If a bet wins \$300 with probability 0.40 and loses \$100 with probability 0.60, its expected value is `0.40 × (+$300) + 0.60 × (−$100) = +$120 − $60 = +$60` per bet. A positive expected value *after all costs* is an edge. A coin flip with even payoffs has an expected value of exactly \$0 — no edge. Most of the honest debate about technical analysis reduces to one question: **does it produce a positive expected value after costs, and can you prove it?** We will return to this number again and again. For the full treatment of why a high win rate can still lose money, see [why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies).

## The three places an edge could actually come from

Suppose, for the sake of argument, that some technical method really does beat a coin flip after costs. Where could that edge physically live? It cannot come from nowhere. Markets are competitive; if a pattern reliably made money, traders would pile into it until it stopped working. So any surviving edge has to be rooted in a real, persistent feature of how markets work. There are only three candidates.

![Any genuine technical edge must come from crowd behavior, from order flow leaking into price and volume, or from self-fulfilling watched levels](/imgs/blogs/what-technical-analysis-really-is-3.png)

The tree above is the whole defensible foundation of technical analysis. If an edge is not coming from one of these three sources, it is almost certainly coming from luck dressed up as skill. Let us take them one at a time.

### Source 1: recurring behavioral and reflexive patterns

Markets are made of people, and people are not random — they are *predictably* irrational in certain ways. When a price has been falling hard, fear spreads; people sell not because anything changed but because everyone else is selling, and the price *overshoots* downward past any reasonable value. When a price has been rocketing up, *fear of missing out* (FOMO) pulls latecomers in, and the price overshoots upward. These are *behavioral* patterns: regularities that come from human emotion rather than from the asset's worth.

There is a second, subtler version called *reflexivity*, a term popularized by the investor George Soros. Reflexivity means the act of observing and trading on a price *changes* the price, which changes beliefs, which changes trading — the feedback loop in figure 1. A rising price is itself a piece of news to many traders ("it's going up, get in"), and that interpretation makes it rise further, for a while, until it doesn't. If crowds reliably panic and chase in similar shapes, then a chart — which is a record of crowd behavior — might carry a faint, real signal about what the crowd will do next. This is the strongest honest argument for TA: **a chart is a behavioral fingerprint of the crowd, and crowds repeat themselves.**

The catch, which we will hammer later: "crowds repeat themselves" is much weaker than "this pattern predicts the price." Emotions are real but noisy, and the moment a behavioral pattern becomes well known, traders front-run it and it weakens.

It is worth naming the specific behavioral biases the field implicitly leans on, because they are documented in psychology, not invented by chartists. *Herding* — copying what others do because their action feels like information — drives the chasing and panicking. *Anchoring* — fixating on a reference number, like the price you paid or a recent high — is why prior price levels exert a pull on behavior. *Loss aversion* — feeling a loss about twice as painfully as an equal gain — is why selling cascades when a level breaks: people who were "holding for breakeven" all capitulate at once. The bet a behavioral technician is making is that these biases produce roughly *similar-shaped* crowd reactions across different episodes, so a chart that captured the shape last time carries a hint about this time. That bet is defensible in direction and weak in magnitude: the biases are real, but the noise around them is large, and any edge they leave is faint.

### Source 2: supply, demand, and order-flow information

The second source is colder and, to many quantitative traders, more respectable. Price and volume are the *only* public footprints of the actual order flow — the buying and selling pressure happening under the surface. Suppose a large institution wants to buy two million shares. It cannot do that all at once without spiking the price, so it buys in pieces over hours or days. That sustained buying leaves a footprint: the price drifts up on rising volume, and dips get bought quickly. A technician watching volume and price might *infer* the presence of that big buyer without ever knowing who it is. This is *order-flow* reading: using the visible price-and-volume trace to guess at the hidden supply and demand.

This is genuinely information, not superstition. A volume spike on a breakout, the speed at which a dip is bought, the way price stalls at a level — these can carry real information about who is trying to do what. The most successful quantitative firms in history, as we will see, built their entire businesses on extracting tiny statistical signals of exactly this kind from price and volume data. The honest caveat: the information is *small* and *decays fast*. Big players also know their footprints are visible and actively work to hide them (that is what *iceberg orders* and execution algorithms are for), so the signal is faint and adversarial.

### Source 3: self-fulfilling watched levels

The third source is almost circular, and that is the point. Some price levels matter *only because so many traders believe they matter*. Round numbers (\$100, \$50,000), the previous all-time high, last month's low, a widely cited moving average — huge numbers of traders place buy and sell orders, and especially *stop-loss orders* (automatic sell orders that trigger if price falls to a set level), clustered at these spots. Because the orders are really there, the price really does tend to react when it reaches them. The level "works" not because of any deep law but because the crowd's shared attention put real orders there.

![Support at $98 is just a wall of resting buy orders and resistance at $103 a wall of sell orders, and the chart only traces these walls](/imgs/blogs/what-technical-analysis-really-is-5.png)

The figure above is what a "support level" actually is under the hood: it is not a magic line, it is a thick band of resting buy orders sitting at, say, \$98, deeper than the bands above and below it. When falling price reaches \$98, those orders absorb the selling and the decline stalls — the level "holds." The same picture flipped gives *resistance* at \$103, a wall of resting sell orders that caps a rise. The chart drawn from prices only shows you the *trace* of these walls after the fact; the walls themselves live in the order book, which most chart-readers never see. A level holds only while its wall is bigger than the order flow hitting it — when a larger wave of selling arrives than the \$98 wall can absorb, the level "breaks," the remaining buyers pull back, and price falls fast through the now-empty space below.

This is the most defensible *and* the most fragile source at once. It is defensible because the orders are real — a wall of stop-losses just below \$98 will really get triggered if price touches \$98, and that cascade really moves the price. It is fragile because it depends entirely on coordination of belief; if attention shifts to a different level, the old one goes dead. A self-fulfilling level is real the way a bank run is real: the belief creates the fact, right up until the belief moves on.

There is a darker corner to this source worth flagging honestly: because the orders clustered at watched levels are *predictable*, sophisticated players sometimes deliberately push price into them to trigger the cascade and trade against the wreckage — a practice loosely called a *stop hunt*. Price probes just below the well-known support at \$98, the resting stop-losses fire, the brief flush of forced selling lets the instigator buy cheaply, and price snaps back. The self-fulfilling level "worked" — it did get tested and produce a reaction — but the trader who naively bought the bounce at support and the trader who placed an obvious stop just under it were both on the wrong side of someone who understood the level was a magnet, not a wall. The reflexive lesson stacks on top of the order-flow one: a level is information about where orders sit, and where orders sit is information someone else can exploit.

## The honest case against

Now the other side, told without flinching. If you only read the bullish case for TA, you will lose money, because the critiques below are correct often enough to bankrupt the careless. A serious person holds both the case for and the case against in mind at the same time.

### The random walk: convincing patterns from pure noise

Start with the most devastating critique, because it is the one most chart-readers have never honestly confronted. A *random walk* is a process where each step is an independent random move — like our coin flip, up \$1 on heads, down \$1 on tails. The crucial, counterintuitive fact is this: **a pure random walk produces charts that look exactly like real markets, complete with trends, channels, support, resistance, and "patterns" — and every one of those features is meaningless.**

![Both panels are pure coin flips with no predictable structure, yet the eye reads an uptrend on the left and a range on the right](/imgs/blogs/what-technical-analysis-really-is-2.png)

Look at the two panels above. Both were generated the identical way — a hundred coin flips, plus one on heads, minus one on tails, starting from \$100. The left panel wandered upward; if it were a real stock you would draw a trend line under it and call it an uptrend. The right panel wandered sideways; you would call it a trading range with support and resistance. Neither has any structure. The "uptrend" on the left is what a lucky run of more heads than tails looks like, and a lucky run tells you *nothing* about the next flip — the coin has no memory. Worked example (a) below makes this concrete with numbers.

The deep lesson is about *human pattern-finding*. Our brains are pattern-detectors that evolved to never miss a tiger in the grass, which means they cheerfully find tigers that are not there. Show a person random noise and they will find shapes, trends, and meaning, with total confidence. This is *apophenia* — perceiving real patterns in random data — and it is the default failure mode of every chart-reader, including disciplined ones. The first job of an honest technician is to fight their own eyes.

### Weak-form efficiency: the past is already in the price

The most cited academic objection is the *efficient-market hypothesis* (EMH), which says that prices already reflect available information, so you cannot reliably beat the market using that information. EMH comes in three strengths, and the distinction is the whole ballgame for TA.

![Weak-form efficiency says past prices are already priced in, which is exactly the claim a chart-based edge has to defeat](/imgs/blogs/what-technical-analysis-really-is-4.png)

As the table above lays out: *weak-form* efficiency says that all past *prices and volume* are already reflected in today's price, so studying past prices — which is exactly what technical analysis does — cannot give you an edge. *Semi-strong* form adds all public information (news, earnings, filings), which would also doom most fundamental analysis. *Strong* form adds even private inside information. You do not need to believe the strong form to be skeptical of TA; you only need to believe the *weak* form, the mildest version. And the weak form is the one with the most evidence behind it: decades of studies have found that simple price patterns, once trading costs are included, mostly do not predict future returns better than chance.

The honest rebuttal is that markets are not *perfectly* efficient — they are *mostly* efficient, with small, fleeting inefficiencies that disappear as soon as enough people exploit them. That gap is where any real edge lives. But notice how that rebuttal already concedes the critics' main point: if there is an edge, it is *small and temporary*, not a reliable money machine. "Markets are 98% efficient" is not a slogan a TA guru can sell, which is exactly why the gurus do not say it.

A more nuanced middle position, sometimes called the *adaptive markets* view, helps reconcile the two camps. It says efficiency is not a fixed property but a moving target: an inefficiency appears, traders learn to exploit it, their trading erases it, and the market becomes efficient *with respect to that pattern* — until conditions change and a new inefficiency opens elsewhere. On this view a technical edge is less like a law of physics and more like an ecological niche: real for a while, competed away over time, and never permanent. That is a far humbler claim than "patterns predict prices," and it is the only version of the bullish case that survives contact with the evidence. It also explains why honest practitioners are paranoid about *decay* — they assume any edge they find is on a clock and will need replacing, because the market is actively learning to neutralize it.

### Why most indicators are noise or curve-fitting

A technical *indicator* is a formula computed from price and volume — a moving average, a momentum oscillator, a band drawn some number of standard deviations from a mean. There are thousands of them, and you can invent a new one this afternoon. The problem is structural: with thousands of indicators and dozens of adjustable settings each, you can *always* find some combination that would have made money on any particular stretch of history. That is not a discovery; that is *curve-fitting* (also called *overfitting*) — tuning a rule so precisely to past data that it captures the random noise of that specific history, not any real, repeatable pattern.

A curve-fit rule has a tell: it looks spectacular on the data it was built on and falls apart on data it has never seen. The technical term for testing a rule on history is a *backtest*, and the term for the data the rule was *not* tuned on is *out-of-sample* data. A rule that shines in the backtest and dies out-of-sample was never a signal; it was a coincidence you mistook for a law. Worked example (c) below shows exactly how easy it is to manufacture one of these by accident.

### The guru survivorship illusion

Walk through finance social media and you will find people who called the last big move perfectly, with the screenshots to prove it. Surely *they* can read charts? Here is the trap, and it is pure statistics. If ten thousand people each loudly predict the market's direction, then by chance alone, after several predictions, a handful will have been right every single time — not because they are skilled but because *someone always is*, the way someone always wins the lottery. We only ever see the survivors. The thousands who made equally confident calls and were wrong deleted their posts, changed accounts, or quietly stopped. This is *survivorship bias*: judging a strategy by looking only at the winners who remain visible, ignoring the identical-looking losers who vanished.

The classic illustration is the "stock-tip scam." Send 10,000 people a free prediction — half told "it will rise," half "it will fall." Keep only the 5,000 you were right with, and split them again. After several rounds you have a few hundred people who have seen you call the market correctly five times straight and think you are a genius worth paying. You have zero skill; you have arithmetic. The visible TA guru with the perfect track record is, statistically, far more likely to be a lucky survivor than a skilled reader — and there is usually no way to tell which from the outside. The only defense is to demand the *full* record, including the losers, and to be deeply suspicious of anyone who shows you only wins.

## What a defensible version of TA actually claims

So is technical analysis worthless? No — but the defensible version claims far less than the popular version, and the gap between the two is where almost all the money is lost.

![The dishonest version of technical analysis predicts certainty while the defensible version only nudges a conditional probability and frames the risk](/imgs/blogs/what-technical-analysis-really-is-8.png)

### Conditional probability, not prediction

The honest claim is never "the price will go up." It is a *conditional probability*: "given that I observe this specific setup, the probability that price rises to my target before hitting my stop is, say, 55% instead of the 50% baseline." Read that carefully. It is a statement about *odds*, conditioned on an observation, and the edge is the tiny gap between 55% and 50%. It does not promise the next trade wins. It does not even promise *this kind of trade* wins most of the time — as we will see in worked example (d), you can have a positive edge with a *losing* win rate, if the wins are bigger than the losses.

The shift from prediction to conditional probability is the single most important idea in this post. A prediction is binary and falsifiable on one trade ("you said up, it went down, you were wrong"). A conditional probability is only testable over *many* trades ("you said 55%; over 200 trades did roughly 55% reach target?"). The guru who says "the chart says up" has made a prediction and will be humiliated by the first random loss. The honest technician who says "this tilts the odds to 55%, and I've sized the bet so a loss costs me 1% of my account" has made a probabilistic bet and can be wrong on any single trade without it meaning anything.

### Framing risk: the part that actually pays

Here is a quietly radical claim: the most valuable thing a chart gives a disciplined trader may have nothing to do with prediction at all. It is a *framework for risk*. A chart shows you, concretely, where you would be *wrong* — a level below which your reason for the trade no longer holds. That level becomes your *stop-loss*: the price at which you exit and admit the idea failed, capping the loss. Knowing your exit *before* you enter lets you size the position so that being wrong costs a small, survivable amount, and lets you compute the reward-to-risk ratio that decides whether the bet is even worth making. Even if the directional signal is pure noise, the *discipline of defining risk in advance* is genuinely valuable — and it is the part of TA that the gurus talk about least, because "manage your risk carefully" sells far worse than "this pattern prints money."

Make this concrete with *position sizing* — deciding how many shares to buy so that a loss is survivable. Suppose you have a \$10,000 account and a rule that you will never risk more than 1% of it on a single trade — that is \$100 of risk per trade. Your chart says the trade is wrong if price falls from your \$100 entry to your \$99 stop, a \$1 risk per share. Then the math forces your size: `shares = risk budget / risk per share = $100 / $1 = 100 shares`. If instead the stop were further away — wrong only below \$96, a \$4 risk per share — the same \$100 budget would let you buy only `$100 / $4 = 25 shares`. Notice what the chart did here: it never predicted anything, but it *defined the distance to "wrong,"* and that distance, combined with a fixed risk budget, mechanically determined your size. This is the difference between a trader who survives a long losing streak and one who blows up: the survivor risked a fixed small fraction each time, so twenty losses in a row cost a recoverable chunk; the gambler bet big on conviction and a normal cold streak ended the game. The chart's contribution to *not going broke* is, honestly, more reliable than its contribution to *being right*.

### The map is not the territory

A final, humbling point. A chart is a *map*, and a map is a radical simplification of the territory. The candlestick throws away everything except four prices per period; the indicator throws away even more. The map is useful precisely *because* it simplifies — but you must never confuse the map for the territory. The price did not move "because it hit resistance"; it moved because, at that level, more shares were offered for sale than demanded, for reasons the chart cannot see. The chart is a low-resolution shadow of the real auction. Treat it as a tool for framing probabilities and risk, never as a crystal ball, and you are using it honestly.

## Worked examples

Now we make all of this concrete with numbers. Each example uses round figures and shows every step, and ends with the one idea it was built to teach. These are the heart of the post — if you remember nothing else, remember the arithmetic here.

#### Worked example: a coin-flip "trend" out of pure randomness

You start with a price of \$100. Each step, you flip a fair coin: heads, the price goes up \$1; tails, down \$1. This is a pure random walk — the coin has no memory, so the next flip is 50/50 regardless of what came before.

Run 100 flips. The *expected* final position is exactly \$100, because each flip averages zero. But you almost never land exactly on \$100. The typical distance you drift from the start grows with the square root of the number of steps: roughly `√100 = 10` dollars. So after 100 flips, a drift of \$10 up or down — to \$110 or \$90 — is completely ordinary, *pure luck*. A drift of \$20 happens too, less often. Now picture the path that produced a \$12 gain: more heads than tails, with the surplus arriving in clumps (random data is clumpy, not evenly spread). On a chart that clumpy upward drift looks *exactly* like an uptrend with pullbacks — figure 2, left panel. You could draw a trend line under it, mark "support," and feel certain. And the next flip would still be 50/50.

Here is the kicker. If you computed the "win rate" of a rule like "buy after two up-steps" on this random path, you might find it won 53% of the time *on this specific path* — and that number is meaningless, because it came from a process with literally no edge. The drift was luck; the rule's apparent success was luck; the pattern your eye found was luck.

> Intuition: a convincing trend is exactly what randomness looks like over a short stretch, so the existence of a pattern proves nothing about whether it will continue.

#### Worked example: is a 55% win rate real, or luck?

Say a setup wins 55% of the time over 200 trades — 110 wins, 90 losses. Is that a real edge, or could a true 50/50 process produce a result that good by chance? This is the question that separates a measured edge from a story.

Model a no-edge process as 200 fair coin flips, each a 50% "win." The expected number of wins is `200 × 0.50 = 100`. The spread of that count, its *standard deviation* (a measure of how far results typically scatter from the average), is `√(200 × 0.5 × 0.5) = √50 ≈ 7.1` wins. Your result, 110 wins, is `(110 − 100) / 7.1 ≈ 1.41` standard deviations above the no-edge average.

How surprising is 1.41 standard deviations? For a result this far above the mean *or beyond*, the probability under pure chance is roughly 8%. In plain terms: even if your setup had *no edge at all*, you would see 110-or-more wins out of 200 about one time in twelve, just from luck. An 8% chance of a false positive is not nothing — it is far from the "this clearly works" feeling that 110 wins gives you. To be confident, you would want either many more trades (1,000 trades at 55% would be wildly unlikely under chance) or a bigger win-rate gap. This is *statistical significance*: the question of whether a result is large enough, given the sample size, to be unlikely under pure luck.

> Intuition: 55% over 200 trades feels like proof but is consistent with a coin flip about one time in twelve, so a small edge needs a large sample before you should believe it.

#### Worked example: data-mining a fake signal from 20 random rules

You decide to "discover" a trading rule. You write down 20 arbitrary rules — buy on Tuesdays, buy after three red candles, buy when volume is above its 13-day average, and so on — none with any real logic. You backtest all 20 on the same five years of history and keep the best one.

![Testing twenty random rules guarantees one looks great by luck, and that lucky survivor decays the moment it meets new data](/imgs/blogs/what-technical-analysis-really-is-6.png)

Here is the trap, with numbers, illustrated by the funnel above. Suppose each rule, being meaningless, has a true win rate of exactly 50% — no edge. By the same square-root math as before, a single rule tested over, say, 100 trades has a standard deviation of `√(100 × 0.25) = 5` wins, so a rule landing at 58–60 wins (a 58–60% win rate) is about 1.6–2 standard deviations up — individually about a 1-in-20 fluke. But you did not test *one* rule; you tested *twenty*, and kept the best. The chance that *at least one* of 20 independent no-edge rules clears that bar by luck is roughly `1 − (1 − 0.05)²⁰ ≈ 1 − 0.36 = 0.64` — about 64%. So you are *more likely than not* to "discover" a rule with a 58%+ win rate that is pure noise, simply because you looked at 20 of them.

This is the *multiple-comparisons problem*: every extra rule you test is another lottery ticket for a false positive, and reporting only the winner hides how many tickets you bought. The winner looks brilliant in the backtest precisely because it captured the random quirks of those five years. Run it forward on the next four years of out-of-sample data and the edge evaporates — figure 6's last stage — landing back near 51%, a coin flip. You did not find a signal; you found the loudest noise and mistook it for music.

> Intuition: the more rules you try, the more certain it becomes that one looks great by luck alone, so a backtest is only meaningful if you count every rule you tested, not just the survivor.

#### Worked example: framing one trade as a probabilistic bet

Now the example that ties it together — the one number that decides whether a trade is worth taking. Forget prediction; treat the trade as a bet and compute its expected value.

![A single trade has a probability of winning and a payoff if it does, and only their product, the expected value, tells you if the bet is worth taking](/imgs/blogs/what-technical-analysis-really-is-7.png)

You buy at \$100. Your chart-defined stop-loss is \$99 — if price hits it, you exit, losing \$1 per share. Your target is \$103 — if price gets there first, you exit with \$3 per share. So your *reward-to-risk ratio* (R:R), the size of the win versus the size of the loss, is `$3 / $1 = 3` to 1. Now suppose, honestly, that this setup only reaches the target before the stop *40% of the time* — your win rate is a *losing* 40%. Should you take the bet?

Compute the expected value per trade, scaling to \$100 risked per trade so the numbers are friendly (so a win is +\$300 and a loss is −\$100), as in the matrix above:

`EV = P(win) × win + P(lose) × loss = 0.40 × (+$300) + 0.60 × (−$100) = +$120 − $60 = +$60 per trade.`

The bet has a *positive* expected value of +\$60 per \$100 risked, even though you lose 60% of the time. This is the counterintuitive heart of risk-framing: **a low win rate can be highly profitable if the wins are big enough relative to the losses.** Flip it around: a setup that wins 70% of the time but only makes \$1 for every \$3 it risks has an EV of `0.70 × $100 − 0.30 × $300 = $70 − $90 = −$20` — a *loser* despite winning most of the time. The win rate alone tells you almost nothing; only `P(win)` and `R:R` together, multiplied into expected value, decide it. This is exactly why [win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies).

> Intuition: a single trade is a bet defined by a probability and a payoff, and only their product — the expected value — tells you whether taking it is rational, regardless of how often you win.

#### Worked example: the cost that quietly eats a small edge

Every example so far ignored what it costs to trade. That is where most paper edges go to die, so let us put the cost back in. Say you have found a setup with a genuine, real edge: it wins 52% of the time at even money (win \$100, lose \$100). Per trade, before costs, your expected value is `0.52 × $100 − 0.48 × $100 = $52 − $48 = +$4`. A real, positive edge — small, but real.

Now add the *frictions* the backtest pretended away. The *bid-ask spread* — the gap between the price you buy at and the price you can immediately sell at — costs you a little on entry and a little on exit; call it \$2 round-trip on this \$100 trade. *Slippage* — the difference between the price you expected and the price you actually got, because the market moved while your order filled — costs another \$1 on average. A *commission*, if your broker charges one, maybe \$1. Total cost: about \$4 per round-trip trade.

Subtract: `expected value after costs = +$4 − $4 = $0`. Your genuine 52% edge has been *exactly* erased by the cost of harvesting it. You would trade hundreds of times, be right more often than wrong, and end up with nothing — or, once you slightly overestimate your edge or underestimate your slippage, with a steady loss. This is not a hypothetical; it is the single most common reason real strategies that look great in a backtest lose money live. The edge has to clear the costs *with room to spare*, and the smaller the timeframe you trade, the more often you pay these costs and the larger your raw edge must be just to break even.

> Intuition: a small real edge and the cost of trading it are usually the same size, so a strategy only matters if its edge clearly survives the spread, slippage, and commissions a backtest leaves out.

## Common misconceptions

A short tour of the beliefs that cost beginners the most money. Each is wrong for a specific, fixable reason.

### "A pattern that appears means the price will move that way"

The single most common error. A *pattern* — a head-and-shoulders, a triangle, a flag — is a shape the eye finds in the chart. As the random-walk example showed, these shapes appear constantly in pure noise. The honest version is that *some* patterns *may* shift the odds *slightly*, *sometimes*, in *some* markets — and you only know which by measuring, with out-of-sample data, after costs. The appearance of a pattern is, at most, a weak conditional probability; it is never a promise.

### "More indicators means a better signal"

Beginners stack moving averages, oscillators, and bands until the chart is a wall of lines, believing more confirmation is safer. The opposite is true: every extra indicator is another knob to overfit and another correlated echo of the same price. Five indicators computed from one price stream are not five independent opinions; they are one opinion wearing five hats. Adding them mostly adds the *illusion* of confirmation while multiplying the ways you can fool yourself. Simplicity is a defense against curve-fitting.

### "It worked great in the backtest, so it will work live"

A backtest is a measurement on *past* data, and the past is the one stretch of history your rule had the unfair chance to be tuned to. The number that matters is *out-of-sample* performance — and even that overstates reality, because backtests usually ignore the *bid-ask spread*, *slippage* (getting a worse price than you expected when you actually trade), and the fact that your own orders move the price. A rule needs a *large* backtest edge just to survive the costs that the backtest pretended away.

### "The gurus with great track records have figured it out"

Survivorship bias, covered above. With enough people predicting, some will be right repeatedly by chance, and those are the only ones you see. A visible perfect record is weak evidence of skill and strong evidence of selection. Demand the losing trades too; skepticism here is not cynicism, it is arithmetic.

### "Technical and fundamental analysis are enemies"

They answer different questions and many serious participants use both: fundamentals to decide *what* to consider and *roughly what it is worth*, technicals to decide *when* to act and *where to place the risk*. Treating them as rival religions is a beginner's framing. The honest practitioner uses whatever carries information and discards the rest.

### "If it's not a sure thing, it's worthless"

The most corrosive misconception, because it sends people hunting for certainty that does not exist and into the arms of anyone who promises it. Markets do not offer certainty to anyone, ever. They offer *edges* — small, probabilistic tilts that, applied with discipline over many trades and sound risk control, can compound. The goal is not to be right; it is to have a positive expected value and to survive the variance long enough for it to play out.

## How it shows up in real markets

Theory is cheap. Here are four named, concrete ways the ideas above play out in real markets, with the mechanism from this post visible in each.

### Renaissance Technologies: statistical edges are real, tiny, and hidden

The most important counterpoint to "TA is worthless" is also the most misunderstood. Renaissance Technologies, founded by the mathematician Jim Simons, ran the Medallion fund, which posted returns over roughly three decades that are among the best ever recorded — reportedly averaging around 66% gross annually before fees in the period studied by outside writers. Crucially, Renaissance's edge came largely from finding *statistical regularities in price and volume data* — which is, in the broadest sense, technical analysis. So statistical price-based edges plainly *can* exist.

But read the lesson correctly, because the gurus get it exactly backwards. Renaissance's edges were *tiny* (they reportedly were right only slightly more than half the time on individual bets), *fleeting* (they decayed and had to be constantly replaced), and *extracted by PhDs in physics and mathematics using enormous data and computing power*, not by drawing trend lines by eye. They were also *capacity-constrained*: Medallion famously capped its size and returned outside money, because the edges were too small to deploy at scale. The honest takeaway is the opposite of inspiring: price-based edges are real but so small and so hard to find that the most successful effort in history needed a building full of scientists and still capped its assets. For the full story, see [Jim Simons and Renaissance](/blog/trading/finance/jim-simons-renaissance-quant-trading).

### The 2010s–2020s TA-influencer boom and selection bias

The rise of free brokerages and social media produced a wave of chart-reading influencers, especially around the 2020–2021 retail trading surge and the crypto cycles. Many showed screenshots of perfectly called moves. The mechanism is survivorship bias operating at industrial scale: when millions of people post predictions, a visible minority will have impressive recent records by chance, and the platform's algorithms *amplify exactly those people* because confident winners get engagement. The losers are not lying low out of shame; they are statistically invisible — there was never a feed collecting the failed calls. The lesson from the survivorship example applies directly: a great visible record, in a population this large, is close to what pure luck would produce, so it is almost no evidence of skill on its own.

### A famous failed technical call: the "Hindenburg Omen"

Technical analysis has a long history of named signals that sounded scientific and did not deliver. The "Hindenburg Omen," a market-breadth pattern that supposedly warned of crashes, drew waves of attention around 2010 and after. Studied honestly, signals like it fire far more often than crashes occur — they are riddled with *false positives*, predicting many crashes that never come. The mechanism is data-mining and multiple comparisons: a pattern selected because it preceded a couple of historical crashes will, like our 20-rules example, look meaningful in the backtest and fail to generalize, because there are only a handful of real crashes to fit to and endless ways to draw a line through them. A scary name and a couple of past hits is exactly the signature of a curve-fit, not a discovery.

### How a naive pattern decays out-of-sample

Researchers and practitioners have repeatedly taken simple, popular technical rules — a moving-average crossover, a momentum filter — and tested them properly: fit the parameters on one period, then evaluate untouched on a *later* period, with realistic trading costs subtracted. The recurring result is the data-mining funnel in figure 6, observed in the wild. Rules that looked strong in-sample shrink sharply out-of-sample, and a large fraction of the apparent edge turns out to be the *bid-ask spread and slippage* the in-sample test ignored. Some price-based effects (broad momentum across many assets is the most studied) have shown more persistence than pure data-mining would predict — which is why the foundations section listed real sources of edge — but even those are modest, vary across eras, and are the subject of genuine academic dispute. The honest summary across decades of this work: most naive technical rules do not survive honest out-of-sample testing with costs, and the few effects that partly survive are small. That is not a reason to give up; it is a reason to *measure*, ruthlessly, before risking money. The mathematics of why short-horizon price changes look so close to a coin flip is the subject of [Brownian motion and the random walk](/blog/trading/math-for-quants/brownian-motion-random-walk-math-for-quants).

### Round numbers as self-fulfilling magnets

The third source of edge shows up vividly around big round numbers. Markets visibly cluster attention and orders at psychologically round levels — a stock at \$100, an index at a round thousand, Bitcoin at \$10,000, \$20,000, \$50,000, \$100,000. Around these levels you reliably see thickened order books, stalls, and sharp reactions, not because the round number has any fundamental meaning (it does not; \$50,000 is not worth more or less than \$49,873) but because *everyone is watching the same number and has placed orders near it*. This is the self-fulfilling mechanism in its purest form: the level matters only because the crowd agreed to make it matter. The honest technician's edge here, if any, is small and conditional — "near a major round number, expect more order flow and bigger reactions" — and it is exactly the kind of edge that can be front-run and faked (the stop-hunt above). It is a real feature of markets and a weak basis for confident prediction at the same time, which is the recurring theme of this entire post.

### A liquidity vacuum: the 2010 Flash Crash

On May 6, 2010, U.S. stock indices plunged roughly 9% and recovered most of it within minutes — the "Flash Crash." No fundamental news caused it; the mechanism was a *liquidity vacuum*. A large automated sell order hit a market whose resting buy orders briefly evaporated, so each sale pushed price sharply lower into thinner and thinner demand, triggering more automated selling in a cascade. This is the order-flow source of edge turned violent: the chart's plunge was a faithful trace of real supply overwhelming real demand when liquidity vanished, exactly as the depth-chart picture in figure 5 implies it must. The lesson for a chart-reader is twofold. First, price *is* the outcome of the auction, and when one side of the auction disappears the price can do almost anything regardless of "support levels" — a drawn line is no defense against an empty order book. Second, the moves that look most dramatic on a chart are often liquidity events, not informational ones, and reading them as predictive signals about the future is a category error. The chart recorded what happened to supply and demand; it did not foretell it.

## When this matters to you and further reading

If you ever look at a price chart — to buy a stock, to trade crypto, or just to understand the financial news — the ideas here change what you are doing with your eyes. You stop asking "what is the chart telling me it will do?" and start asking "does this observation shift the odds at all, by how much, and how would I even know?" That single shift, from prediction to measured conditional probability, is the difference between using a chart and being used by one.

Three things to carry away. First, **respect the random walk**: assume by default that a pattern is noise until measurement, out-of-sample and after costs, says otherwise — your eyes will lie to you, confidently, every time. Second, **count your rules**: any edge "discovered" by trying many things and keeping the best is probably a false positive, and the only honest backtest is one that counts every attempt. Third, **the edge, if it exists, is small and fragile, so the durable value of TA is in framing risk** — knowing your exit before you enter, sizing the bet so a loss is survivable, and judging trades by expected value, not by being right.

To go deeper, the natural next step is the companion post on [expectancy and why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies), which turns the EV arithmetic from worked example (d) into a full framework for evaluating a strategy. To understand the chart itself — how a single printed price is actually born out of the order book — read [how a price chart is born](/blog/trading/technical-analysis/how-a-price-chart-is-born). For the mathematical bedrock of why price changes resemble coin flips over short horizons, [Brownian motion and the random walk](/blog/trading/math-for-quants/brownian-motion-random-walk-math-for-quants) is the foundation. And for proof that statistical price edges can exist while remaining tiny and brutally hard to capture, the story of [Jim Simons and Renaissance](/blog/trading/finance/jim-simons-renaissance-quant-trading) is the honest north star: not "charts predict the future," but "with enough rigor, a faint signal can be wrung from price and volume, and even then only barely."

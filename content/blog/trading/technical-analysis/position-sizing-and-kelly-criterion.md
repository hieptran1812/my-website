---
title: "Position Sizing and the Kelly Criterion: How Much to Bet When You Have an Edge"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How much you risk per trade matters more than where you enter. This is the honest math of bet sizing: the 1% rule, why over-betting a winning system still goes to zero, volatility drag and the recovery asymmetry, the Kelly criterion f* = p - q/b, and why pros run fractional Kelly."
tags:
  [
    "position-sizing",
    "kelly-criterion",
    "risk-management",
    "money-management",
    "fractional-kelly",
    "risk-of-ruin",
    "volatility-drag",
    "expectancy",
    "fixed-fractional",
    "technical-analysis",
    "trading-edge",
    "drawdown",
  ]
category: "trading"
subcategory: "Technical Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — how much you risk per trade decides whether a real edge compounds or blows up, and it matters *more* than where you enter. Get the size wrong and the best entry in the world cannot save you.
>
> - **Risk per trade is one number, in dollars.** It is the distance from your entry to your stop, times your position size. Call that amount **1R**. Everything in sizing is "how big do I make 1R?"
> - **The 1% rule:** risk a fixed small fraction — typically 0.5% to 2% — of your *current* account on each trade. Because the fraction is fixed, your bet automatically shrinks in a drawdown and grows on a winning streak.
> - **Over-betting a winning system still ruins you.** Wealth compounds *geometrically*, and a symmetric +50% / −50% swing nets a 25% loss, not zero. The deeper the loss, the more brutal the recovery: a 50% drawdown needs a **+100%** gain just to break even.
> - **The Kelly criterion** gives the growth-optimal bet: for a simple bet that wins with probability $p$ at payoff odds $b$-to-1, $f^* = p - \frac{1-p}{b}$, the edge divided by the odds. It maximizes long-run *geometric* growth — not expected dollars.
> - **Full Kelly is brutally volatile** and assumes you know your edge exactly, which you never do. Half-Kelly keeps about three-quarters of the growth with far shallower drawdowns, so almost every professional bets a *fraction* of Kelly. The 1% rule is, in practice, a very conservative fractional-Kelly bet.

A trader buys a system with a genuine, tested, positive edge — it really does make money on average, trade after trade. Six months later the account is at zero. Nothing was wrong with the entries. The signals fired correctly, the win rate held, the average winner was bigger than the average loser exactly as advertised. The trader did everything the strategy said. And the account still went to zero.

This is not a rare story. It is the *single most common* way a profitable trader loses money: not a bad edge, but a good edge bet too big. The entry tells you *whether* to trade. The position size tells you *how much*, and the "how much" is where the account actually lives or dies. You can survive mediocre entries with disciplined sizing. You cannot survive perfect entries with reckless sizing. This post is the math of that sentence.

![Three equity curves starting from the same ten thousand dollar account with the same sixty forty edge. The too-small curve barely grinds upward, the optimally sized curve compounds steeply into a green growth zone, and the too-big curve peaks early then collapses into a red ruin zone, showing that bet size alone separates the three outcomes.](/imgs/blogs/position-sizing-and-kelly-criterion-1.png)

The diagram above is the mental model for the entire post. All three curves come from the *exact same edge* — the same win rate, the same payoff, the same signals. The only thing that differs is the fraction of the account risked on each trade. Bet too little (the bottom curve) and you barely move; the edge is real but you are squandering it. Bet the right amount (the steep middle curve) and the account compounds. Bet too much (the top curve) and the account climbs for a while, peaks, and then a normal losing streak that a smaller bettor would have shrugged off takes it to zero. Same edge. Three completely different fates. Position sizing is the dial that chooses which one you get.

A note before we begin: this is educational. It explains the mechanics and the math of bet sizing so you can read any strategy's risk honestly. It is not advice to trade anything, and it is emphatically not advice to trade bigger. Every method that can compound an account can also empty one, and we will be specific about how. This post is the natural sequel to [why win rate lies and expectancy is what pays](/blog/trading/technical-analysis/expectancy-why-win-rate-lies): that post proved your edge is real; this one is about not blowing it up.

## Foundations: risk per trade, defined

Before we can argue about *how much* to risk, we need a precise, shared vocabulary for *what* "risk per trade" even means. We will build it from zero, in dollars, one term at a time. If you have read the expectancy post, the first two definitions will be review — but we restate them because everything here is built on top.

### The account and the trade

Your **account** (or **equity**, or **capital**) is the total money you are trading with. If you funded a brokerage account with \$10,000 and have not made a trade yet, your equity is \$10,000. After a winning trade of \$300 it is \$10,300; after a losing trade of \$200 it is \$10,100. Equity is a *live* number — it changes with every closed trade — and that fact, which sounds trivial, turns out to be the engine of the whole 1% rule. Hold onto it.

A **trade** is one round trip: you enter a position (buy, going *long*, or sell-short, going *short*) and later exit. We care about three prices on every trade:

- The **entry** — the price you got in at.
- The **stop** (or **stop-loss**) — the price at which you will admit the trade is wrong and exit for a loss. The stop is a *decision you make before entering*; it is the line that says "below here, my reason for the trade is gone." A trade without a predefined stop has no defined risk, and undefined risk cannot be sized.
- The **target** — the price at which you will take profit. (We will care less about the target here; sizing is mostly about the entry-to-stop distance.)

### 1R: the dollar amount you are risking

Here is the central definition of the whole post. The **risk per trade** — written **1R**, read "one R" — is the dollar amount you lose if the trade hits its stop. It is the distance from entry to stop, in dollars per share, multiplied by the number of shares you hold:

$$\text{1R} = (\text{entry} - \text{stop}) \times \text{shares}$$

Suppose you buy 100 shares of a stock at \$50.00 and place your stop at \$48.00. The entry-to-stop distance is \$2.00 per share. Your 1R is $2.00 \times 100 = \$200$. If the stop is hit, you lose \$200 — that is your "one R," your unit of risk on this trade.

The beauty of measuring risk in R is that it makes every trade speak the same language. A \$200 risk on a \$50 stock and a \$200 risk on a \$5,000 Bitcoin position are *the same size bet* — both are 1R — even though the share counts and the dollar prices are wildly different. When we later say "risk 1% of your account per trade," we mean: choose your share count so that 1R equals 1% of equity. Sizing is entirely about choosing how many shares to hold so that 1R comes out to the dollar figure you want. The entry and the stop set the *per-share* risk; the *share count* is the lever you pull to set the *total* risk.

This is worth saying slowly because beginners almost always get it backwards. They decide how many shares to buy first (often "as many as I can afford"), and the risk falls out as an afterthought. The disciplined order is the reverse: decide the dollar risk first (1R), measure the per-share risk (entry minus stop), and *divide* to get the share count. We will do exactly this arithmetic in the first worked example.

### Fixed-dollar versus fixed-fractional sizing

There are two broad ways to decide how big 1R should be.

**Fixed-dollar sizing** risks the same dollar amount on every trade, forever. "I risk \$100 per trade" — whether the account is \$10,000 or \$2,000 or \$40,000. It is simple and it has one fatal flaw: it does not adapt. On a \$2,000 account, risking \$100 is risking 5% per trade, which is aggressive; on a \$40,000 account, the same \$100 is 0.25%, which is timid. Worse, fixed-dollar sizing keeps risking the same absolute amount *as you lose*, so a losing streak eats a larger and larger fraction of a shrinking account — exactly the wrong direction.

**Fixed-fractional sizing** risks the same *percentage* of current equity on every trade. "I risk 1% of my account per trade." On \$10,000 that is \$100; after the account grows to \$12,000 it becomes \$120; after it falls to \$8,000 it becomes \$80. The bet *scales with the account*. This is the family that the 1% rule, and indeed the Kelly criterion, both live in. It has a self-correcting property we will spend a whole section on: it shrinks your risk automatically when you are losing and grows it when you are winning.

These two are not the only options — they sit at the base of a whole family of sizing rules, and it helps to see the family laid out before we go deep on any one branch.

![A taxonomy tree of position sizing rules answering the question how much to risk. It branches into a red fixed-dollar flat bet that does not adapt and a blue fixed-fractional percent of equity branch, which further splits into the green one percent rule risking half to two percent per trade, an external volatility-scaled branch sizing by ATR, and a red full Kelly branch labeled f-star equals edge over odds that itself leads down to a green fractional Kelly node betting a half or a quarter of f-star.](/imgs/blogs/position-sizing-and-kelly-criterion-8.png)

The tree above is the whole landscape of this post in one picture. Every sizing rule answers the same question — *how much to risk?* — and they ladder from the naive flat-dollar bet (which does not adapt and is the riskiest because it keeps betting the same amount as the account shrinks) up through fixed-fractional sizing, which is where everything good lives. Within fixed-fractional, the **1% rule** is the conservative default, *volatility-scaled* sizing (setting your stop and therefore your size by a measure of recent range like the **Average True Range**, or ATR — the typical distance price moves in a period) is a refinement, and **full Kelly** is the theoretical growth-optimal extreme — which, as we will see, you deliberately step *back* from into **fractional Kelly**. The green leaves (the 1% rule and fractional Kelly) are where survivors actually trade. We will earn every node of this tree over the rest of the post; for now, just note that the safest and the growth-optimal answers are *cousins*, both fixed-fractional, differing only in how aggressively they set the fraction.

### A one-paragraph refresher on expectancy

One last foundation, because sizing is meaningless without it. **Expectancy** is how much you make *on average per trade*, counting wins and losses together. If you win with probability $p$ and your average winner is $W$ (in R) against a 1R loss, your expectancy in R is $E[R] = p \cdot W - (1-p)$. A *positive* expectancy means the system makes money over many trades; a *negative* one means it bleeds, no matter how often it wins. (The full treatment — R-multiples, the breakeven win rate $\frac{1}{1+R}$, why a 40% system can beat a 70% one — is in [the expectancy post](/blog/trading/technical-analysis/expectancy-why-win-rate-lies).) The reason this matters here is stark: **position sizing only helps a positive-expectancy system.** If your edge is negative, sizing changes nothing except how fast you go broke. Bet small and you bleed slowly; bet big and you bleed fast; bet Kelly and Kelly tells you to bet *zero*. Everything below assumes you have first established a real, positive edge. Sizing is what you do *after* you have an edge, to make sure you live long enough to collect it.

## The 1% rule

The 1% rule is the simplest good answer to "how much should I risk?" It says: **risk a fixed small fraction of your current equity — typically between 0.5% and 2% — on every single trade.** That is the entire rule. The number most often quoted is 1%, which is why it has that name, but the *principle* is fixed-fractional sizing with a small fraction; the exact percentage is a dial you set based on how much volatility you can stomach.

Let us be concrete about what "risk 1%" means mechanically, because it is the thing beginners most often fumble. It does *not* mean "put 1% of your account into the trade." It means "size the position so that **if the stop is hit**, you lose 1% of your account." Those are very different. If you risk 1% with a stop 2% away from your entry, you are actually deploying 50% of your account into the position (because a 2% move against a 50% position is a 1% account loss). The 1% governs the *loss if you are wrong*, not the *capital deployed*.

![A six stage pipeline showing the one percent rule recomputing the bet from current equity each trade. Current equity ten thousand dollars flows into a fixed one percent risk fraction, then into one hundred dollars of risk dollars labeled one R, then is divided by a two dollar fifty cent stop distance to produce a position of forty shares, and finally loops to recompute on the next trade.](/imgs/blogs/position-sizing-and-kelly-criterion-2.png)

The pipeline above is the 1% rule as an algorithm you run before every trade. Start from your *current* equity (not your starting equity, not your peak — what the account holds right now). Multiply by your fixed fraction to get the risk dollars, your 1R for this trade. Measure the stop distance in dollars per share. Divide the risk dollars by the stop distance to get the share count. Place the trade. When it closes, your equity has changed, so the *next* trade recomputes from the new number. The risk dollars are never fixed; the *fraction* is. We will walk this exact pipeline with numbers in the first worked example.

### Why fixed-fractional sizing is self-correcting

Here is the property that makes the 1% rule quietly brilliant, and it follows directly from "the fraction is fixed but the dollars float."

**In a drawdown, your bet automatically shrinks.** Suppose you start at \$10,000 risking 1% (\$100) per trade, and you hit a rough patch that takes the account to \$8,000. Now 1% is \$80. You are *automatically* betting less, in dollars, precisely when you are losing — which means a losing streak decays your bets geometrically rather than chewing through a fixed amount. You can never be wiped out by fixed-fractional sizing in the way fixed-dollar sizing wipes people out, because each loss makes the next bet smaller. The account *asymptotically* approaches zero in the worst case but never quite reaches it from the sizing alone. (In practice other things — minimum position sizes, fees, psychological capitulation — end the story before the math does, but the direction is protective.)

**On a winning streak, your bet automatically grows.** Run the account up to \$15,000 and 1% is now \$150. You are betting more, in dollars, as you win — which is exactly how compounding happens. You are reinvesting your gains into larger positions without ever changing your *rule*. This is the engine behind the steep middle curve in the very first figure: a fixed *fractional* bet on a positive edge compounds, because each win enlarges the base that the next bet is a percentage of.

Notice what fixed-fractional sizing refuses to do: it never "bets more to make back" a loss. The single most destructive instinct in trading — increasing size after a loss to recover faster — is the *opposite* of what fixed-fractional sizing does. The rule mechanically reduces your bet after a loss. That is not a bug to override; it is the entire point.

### Choosing the fraction: 0.5%, 1%, or 2%?

The 1% in "1% rule" is a default, not a law. The right fraction depends on three things: how confident you are in your edge, how many trades you take (more trades means more chances for a bad streak, so smaller bets), and — most honestly — how much drawdown you can tolerate without abandoning the system. A useful rule of thumb: with a typical edge, **1% per trade** keeps your worst realistic drawdown in the 20–30% range over a long run, which most people can sit through. **2%** roughly doubles that drawdown depth and is near the upper edge of what disciplined discretionary traders use. **0.5%** is for large accounts, uncertain edges, or anyone who has noticed they make bad decisions when down. When in doubt, smaller is safer, and the cost of "too small" (slower growth) is recoverable, while the cost of "too big" (ruin) is not. We will quantify that asymmetry in the next two sections — it is the heart of the whole post.

## Why over-betting a winning system still ruins you

Now we arrive at the idea that surprises almost everyone, and that the first figure promised: **a system with a genuinely positive edge can still take your account to zero if you bet too big.** Being right on average is not enough. You can have the math of the edge completely on your side and still go broke, and the reason is not bad luck — it is arithmetic. Specifically, it is the arithmetic of *compounding*, which behaves very differently from the arithmetic of *averaging*.

### Wealth compounds geometrically, not additively

When you reason about an edge using expectancy, you are *adding up* outcomes: average win times win rate, minus average loss times loss rate. That is **arithmetic** averaging, and it is the right tool for answering "do I have an edge?" But it is the *wrong* tool for answering "how does my account actually grow?" Because your account does not add returns — it *multiplies* them. If you make 10% and then lose 10%, you do not end flat. You end at $1.10 \times 0.90 = 0.99$ — down 1%. Returns chain together by multiplication, and multiplication has a property that addition does not: it punishes volatility.

This is called **volatility drag** (or *variance drain*): the more your returns bounce around, the more the geometric compounding of those returns falls below their simple average. The bigger your bets — and therefore the bigger your up-and-down swings — the worse the drag. The single cleanest way to feel it is the +50% / −50% example.

![A bar chart showing the volatility drag of a fifty percent gain followed by a fifty percent loss. A blue ten thousand dollar starting bar grows to a green fifteen thousand dollar bar after a fifty percent gain, then collapses to a red seven thousand five hundred dollar bar after a fifty percent loss, ending twenty five percent below the start even though the percentage moves were symmetric.](/imgs/blogs/position-sizing-and-kelly-criterion-3.png)

### Worked example: the +50% / −50% round trip

You start with \$10,000. You have a great trade and the account jumps **+50%**, to \$15,000. Then you have an equally large losing trade, **−50%**. Symmetric, right? Up 50, down 50, back where you started?

No. The −50% applies to the *new, larger* base of \$15,000:

$$\$15{,}000 \times (1 - 0.50) = \$15{,}000 \times 0.50 = \$7{,}500.$$

You are at \$7,500 — down **25%** from your \$10,000 start, after a sequence whose percentages perfectly cancelled. The order does not matter either: −50% then +50% gives $\$10{,}000 \times 0.5 \times 1.5 = \$7{,}500$, the same. The intuition: a percentage *loss* takes a bite out of a bigger number than the equal percentage *gain* put on, because the gain was computed on the smaller pre-gain base. **Equal-percent swings do not cancel; they drag you down.** And the bigger the swings, the worse the drag — a +10%/−10% round trip only costs you 1%, but a +50%/−50% round trip costs you 25%, and a +90%/−90% round trip costs you a staggering 81%. This is precisely why betting bigger does not scale your growth linearly: past a point, the extra volatility drags more than the extra edge adds.

### The recovery asymmetry

Volatility drag has a vicious cousin: the deeper a hole you dig, the disproportionately larger the climb out. To recover from a loss of $L$ (as a fraction), the gain $G$ you need satisfies $(1 - L)(1 + G) = 1$, so:

$$G = \frac{L}{1 - L}.$$

![A steeply rising convex curve showing the gain needed to recover from a drawdown. A ten percent loss needs only an eleven percent gain marked in blue, but a fifty percent loss needs a one hundred percent gain marked in amber, and the curve explodes into a red deep-loss zone where recovery gains balloon toward nine hundred percent for a ninety percent loss.](/imgs/blogs/position-sizing-and-kelly-criterion-6.png)

The curve above is what that formula looks like, and it is one of the most important shapes in all of trading. For small losses the asymmetry is mild — lose 10%, need 11.1% to recover; lose 20%, need 25%. But it curves upward viciously:

| Drawdown (loss) | Gain needed to break even |
| --- | --- |
| 10% | 11.1% |
| 20% | 25.0% |
| 33% | 50.0% |
| 50% | 100.0% |
| 75% | 300.0% |
| 90% | 900.0% |

### Worked example: the cost of a 50% drawdown

Say a big over-sized bet (or a streak of them) takes your \$10,000 account down 50%, to \$5,000. To get back to \$10,000, you do not need a 50% gain — you need to *double* your money, a **+100%** gain:

$$\$5{,}000 \times (1 + 1.00) = \$10{,}000.$$

If your edge produces, say, an average +8% per month at a sane bet size, recovering 100% takes you roughly $\ln(2)/\ln(1.08) \approx 9$ months of *flawless* compounding — and that is just to get back to where you started, having made nothing. A 75% drawdown needs +300% (you must *quadruple*), which can be years. The lesson the curve teaches in one sentence: **deep drawdowns are not just painful, they are mathematically near-fatal, because the recovery they demand grows far faster than the loss that caused them.** Avoiding the deep hole is worth far more than any clever entry, and the only thing that keeps you out of the deep hole is bet size. (This is the same recovery math that makes a string of small, controlled −1R losses survivable while a single oversized loss is not; the [expectancy post](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) frames the same idea through risk of ruin.)

### Risk of ruin rises sharply with bet size

Put volatility drag and the recovery asymmetry together and you get **risk of ruin**: the probability that a string of losses takes your account below a level from which it cannot practically recover (often defined as some large percentage drawdown, or literally to zero). For a fixed positive edge, the risk of ruin is a steeply increasing function of bet size. Bet a tiny fraction and ruin is essentially impossible; bet a large fraction and ruin becomes likely *even with the edge in your favor*, because a normal-length losing streak now does irreversible damage.

![A matrix of risk of ruin across bet size and edge. Rows are risk per trade of one, five, twenty five, and fifty percent, and columns are edges of fifty two, fifty five, and sixty percent win rates. The top row of one percent risk is uniformly green at near zero ruin, the five percent row is mostly green and amber, but the twenty five and fifty percent rows turn red with ruin probabilities climbing to sixty six, eighty five, and beyond, showing ruin rises sharply with bet size even as the edge improves.](/imgs/blogs/position-sizing-and-kelly-criterion-7.png)

The matrix above holds the *edge* fixed within each column and varies only the *bet size* down the rows, and the color tells the story before you read a single number: the top rows (small bets) stay green and safe across every edge, while the bottom rows (large bets) march into red and near-certain ruin, *even at the strongest 60% edge*. Read the leftmost column — a thin 52% edge — and you see ruin go from essentially 0% at 1% sizing to roughly 66% at 25% sizing to about 85% at 50% sizing. The edge never changed; the size did all the damage. Reading across a row shows the gentler truth that a bigger edge does buy down ruin somewhat — but never enough to rescue a reckless bet size. **Size dominates edge for survival:** a great edge bet too big is more likely to ruin you than a thin edge bet small. The headline: there is a bet size above which a *winning* system becomes a *losing* proposition, purely from the geometry. The Kelly criterion is the math that finds the edge of that cliff — and tells you to stand well back from it.

## The Kelly criterion

So if betting too small wastes the edge and betting too big invites ruin, there must be a *right* amount somewhere in between — the bet size that makes the account grow as fast as possible in the long run without blowing up. There is, and it has a name: the **Kelly criterion**, derived by John Kelly Jr. at Bell Labs in 1956. It gives the single bet fraction that maximizes the long-run **geometric growth rate** of your wealth. That phrase — *geometric* growth rate, not expected dollars — is the whole subtlety, and we will unpack it.

### What Kelly maximizes (and what it does not)

Naively, you might think the best bet size is the one that maximizes your *expected dollars*. It is not — and that misconception is dangerous enough that it deserves a moment. If you only cared about expected dollars and you had any edge at all, the math would tell you to bet *everything* on every trade, because each bet has positive expected value and more money in means more expected money out. But "bet everything every time" guarantees ruin: sooner or later one loss takes the whole stack, and you are out forever. Expected-dollar maximization ignores that you only get to compound the wealth you *keep*.

Kelly maximizes something better: the **expected logarithm of wealth**, which is mathematically equivalent to the long-run geometric growth rate — the rate at which your account *actually* compounds over many bets. Maximizing expected log-wealth automatically respects the "you have to survive to compound" constraint, because the logarithm of zero is negative infinity: a strategy with *any* probability of going to zero has $-\infty$ expected log-wealth and is rejected outright. Kelly is the bridge between "I have an edge" and "I compound that edge as fast as is survivable." (For the full derivation through expected log-wealth, and seven solved interview-style sizing problems, see the quant-finance companion, [the Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews).)

### The formula, built up intuitively

For a simple bet — you wager a fraction $f$ of your bankroll, win with probability $p$ at odds of $b$-to-1 (a \$1 bet returns \$$b$ of profit on a win), and lose your wager with probability $q = 1 - p$ — the Kelly fraction is:

$$f^* = \frac{bp - q}{b} = p - \frac{q}{b} = p - \frac{1-p}{b}.$$

Let us read it as *edge over odds*, which is the intuition worth carrying around. The numerator $bp - q$ is your **edge**: the expected profit per \$1 wagered ($b$ dollars won with probability $p$, minus \$1 lost with probability $q$). The denominator $b$ is the **odds**. So:

$$f^* = \frac{\text{edge}}{\text{odds}}.$$

The shape of the formula makes sense if you push it to extremes. If you have **no edge** ($bp = q$, the bet is fair), the numerator is zero and $f^* = 0$ — bet nothing, exactly right. If your **edge grows**, $f^*$ grows — bet more when the math is more in your favor. If the **payoff odds $b$ get larger** (you win more per unit risked), the denominator grows and $f^*$ *shrinks* relative to the edge — a fat payoff means you need to risk less of your stack to capture the same growth. Every term points the way intuition says it should.

For the even-money case — a bet that pays 1-to-1, so $b = 1$ — the formula collapses to a clean one you should memorize:

$$f^* = p - q = 2p - 1.$$

At a 55% win rate on an even-money bet, $f^* = 2(0.55) - 1 = 0.10$ — Kelly says bet 10% of your bankroll. At 60%, $f^* = 0.20$. At 50% (no edge), $f^* = 0$. We will use the 55% case in a worked example below.

A caution about translating Kelly to trading: the clean formula above is for a *binary* bet with a fixed win and a fixed loss of your whole wager. Real trades have variable-sized wins and losses and you only risk to a stop, not your whole account. The trading-friendly version uses your win rate $p$ and your reward-to-risk ratio $b$ (average win in R), and it gives the *fraction of your account to put at risk* (your 1R as a fraction of equity), which is exactly the dial the 1% rule sets. So Kelly and the 1% rule are answering the *same question* in the *same units* — Kelly just computes the growth-optimal answer instead of picking a round, conservative number.

## Full Kelly is too wild; use a fraction

Here is the twist that separates people who have read about Kelly from people who have actually traded it: **the full Kelly fraction, $f^*$, is almost always too aggressive to use.** It is the growth-*optimal* bet only in a fantasy world where you know your edge with perfect precision and can tolerate gut-churning swings. In the real world, full Kelly is brutally volatile and dangerously sensitive to estimation error. Almost every professional who uses Kelly at all uses a *fraction* of it.

![The Kelly growth curve plotting long-run growth rate against bet fraction. Growth rises from zero, peaks at the Kelly fraction f-star of ten percent in a green maximum-growth zone, falls back to zero at twice f-star or twenty percent, and turns negative beyond into a red over-betting zone where the same edge destroys wealth.](/imgs/blogs/position-sizing-and-kelly-criterion-4.png)

### The shape of the growth curve

The figure above plots the long-run growth rate of your account against the fraction you bet, for a fixed positive edge. Read it carefully, because it contains nearly everything:

- At $f = 0$ you bet nothing and growth is zero — flat line, no risk, no reward.
- As you increase $f$, growth rises. The edge is real and betting more captures more of it.
- Growth peaks exactly at $f = f^*$, the **Kelly fraction**. This is the fastest your account can compound. Bet here and, over a long enough horizon, you out-grow every other constant fraction.
- Past the peak, growth *falls*, even though you are betting more. The volatility drag from the bigger swings is now eating the extra edge. You are taking more risk for *less* growth — the worst trade in finance.
- At $f = 2f^*$ — exactly *twice* the Kelly fraction — growth is back to **zero**. You have a positive edge, you are betting aggressively, and your long-run growth rate is nil. All that risk, no compounding.
- Past $2f^*$, growth goes **negative**. Now the same positive edge *shrinks* your account over time. This is the red zone in the figure, and it is where over-bettors live: a winning system, bet so big that it loses money. The very first figure's collapsing top curve is a path through this region.

The single most important feature of this curve is its **asymmetry around the peak**. Betting *half* of Kelly costs you very little growth, because the peak is rounded and flat on the left. Betting *double* Kelly costs you *all* of your growth. The penalty for under-betting is gentle; the penalty for over-betting is catastrophic. Given that you can never know your true edge exactly, you want to err on the *low* side of the peak — and that is the entire argument for fractional Kelly.

### Why half-Kelly is the practitioner's default

It turns out the math is unusually kind here. Because the growth curve is approximately a downward parabola near its peak, betting **half** the Kelly fraction gives you about **three-quarters** of the maximum growth rate — but it roughly *halves* your volatility and dramatically shallows your drawdowns. You give up a quarter of your growth to cut your risk in half. For almost anyone, that is a trade worth taking every day of the week.

![Two equity curves from the same edge starting at ten thousand dollars. The half Kelly curve climbs steadily and smoothly into a green steady zone, while the full Kelly curve, drawn dashed, reaches a similar height but through violent up and down swings and deep drawdowns into an amber wild-swings zone, illustrating that half Kelly keeps most of the growth with far calmer equity.](/imgs/blogs/position-sizing-and-kelly-criterion-5.png)

The two curves above tell the practical story. Both come from the same edge. Full Kelly (the jagged dashed line) reaches a great height — but look at the path: violent swings, deep drawdowns, stretches where the account is cut in half before recovering. Most humans cannot sit through that without abandoning the system at the worst possible moment, which converts a paper drawdown into a realized loss. Half-Kelly (the smooth line) gets *almost* as high with a fraction of the agony. The growth you "lose" by halving is small; the drawdown you avoid is large; and the drawdown you avoid is the thing that actually makes you quit.

### Estimation error makes full Kelly dangerous

There is a second, subtler reason to bet a fraction of Kelly, and it is the one professionals worry about most. The Kelly formula needs your *true* win rate $p$ and your *true* payoff $b$. You do not have those. You have *estimates* from a finite, noisy track record — and your estimates are almost certainly too optimistic, because of survivorship (you are looking at a strategy that worked so far), overfitting (you tuned it to past data), and plain small-sample luck. If your *true* edge is half of what you measured, then your *measured* full Kelly is actually *double* the true Kelly — and we just saw that double-Kelly has *zero* long-run growth. **Estimation error turns a confident full-Kelly bet into an accidental over-bet, and the over-bet side of the curve is the side that destroys you.** Fractional Kelly is your margin of safety against being wrong about your own edge. Betting a quarter or a half of your *estimated* Kelly means that even if your edge is substantially worse than you think, you are still on the safe, growing side of the peak.

Where does the 1% rule fit? It is, in practice, a *very* conservative fractional-Kelly bet. For a typical retail edge, full Kelly might suggest risking 5–15% of equity per trade — terrifyingly large to anyone who has felt a real drawdown. The 1% rule's 1% is often a *fifth or a tenth* of full Kelly. That is more conservative than even most professionals run, which is exactly why it is the standard advice for people still building confidence in their edge: it makes ruin essentially impossible while still compounding, and it forgives a great deal of estimation error. As your edge becomes better-measured and your discipline proven, you might inch toward quarter- or half-Kelly. You should essentially never run full Kelly, and never, ever exceed it.

## Worked examples

We have used small examples inline; here are four complete, numbers-shown walkthroughs that together cover the whole post. Round, friendly numbers throughout, so you can redo each in your head.

#### Worked example: the 1% rule, end to end

You have a **\$10,000** account and run the 1% rule. A setup appears: you want to buy a stock at **\$50.00** with a stop at **\$47.50**. How many shares?

1. **Risk dollars (1R).** 1% of \$10,000 is $0.01 \times \$10{,}000 = \$100$. That is your 1R — the most you will lose if the stop is hit.
2. **Per-share risk.** Entry minus stop is $\$50.00 - \$47.50 = \$2.50$ per share.
3. **Share count.** Divide risk dollars by per-share risk: $\$100 \div \$2.50 = 40$ shares.
4. **Sanity check the capital.** 40 shares at \$50 is \$2,000 deployed — 20% of the account in the position. But your *risk* is only \$100, because the stop is just \$2.50 (5%) below entry. Capital deployed and capital risked are different numbers; the 1% governs the second.

Now the part beginners miss. Say the trade loses and the account is **\$9,900**. The next trade's 1R is $0.01 \times \$9{,}900 = \$99$ — slightly smaller, automatically. Win it big and the account is **\$10,400**; the next 1R is \$104 — slightly larger, automatically. You never changed your rule; the rule scaled the bet for you. **The single sentence:** decide the dollar risk first, divide by the stop distance to get the share count, and let the fixed fraction resize the bet for you every trade.

#### Worked example: over-betting a 60% edge — 5% versus 50% of capital

Two traders share the *identical* edge: a coin-flip-style bet that wins **60%** of the time and, when it wins, returns the amount risked (even money, $b = 1$). The edge is clearly positive — expectancy per \$1 risked is $0.60 \times \$1 - 0.40 \times \$1 = +\$0.20$, a fat 20% edge. The full Kelly fraction is $f^* = 2(0.60) - 1 = 0.20$, or 20%.

Trader A risks a sane **5%** of capital per bet (a quarter of Kelly). Trader B over-bets at **50%** of capital per bet (two-and-a-half times Kelly — deep in the red zone of the growth curve).

Consider the *per-bet growth factors*. With even money, a win multiplies your account by $(1 + f)$ and a loss by $(1 - f)$.

- **Trader A (f = 5%):** a win gives $\times 1.05$, a loss gives $\times 0.95$. Over a long run, the geometric growth per bet is $1.05^{0.6} \times 0.95^{0.4} \approx 1.0088$ — about **+0.88% per bet**, compounding nicely. Starting at \$10,000, after 200 bets that is roughly $\$10{,}000 \times 1.0088^{200} \approx \$57{,}600$.
- **Trader B (f = 50%):** a win gives $\times 1.50$, a loss gives $\times 0.50$. The geometric growth per bet is $1.50^{0.6} \times 0.50^{0.4} \approx 0.967$ — about **−3.3% per bet**. *Negative.* The same 60% winning edge, bet this big, shrinks the account: starting at \$10,000, after 200 bets the expected geometric path is $\$10{,}000 \times 0.967^{200} \approx \$11$. Effectively zero.

Read that again. Same edge — 60% win rate, even money, +20% expectancy per bet. Trader A turns \$10,000 into roughly \$57,600. Trader B turns the same \$10,000 into *about eleven dollars*. The only difference is bet size. Trader B is not unlucky; the *expected* geometric outcome of betting 50% on this edge is ruin, because 50% is past $2f^* = 40\%$ where growth goes negative. **The single sentence:** a positive edge bet past twice the Kelly fraction has *negative* growth — over-betting does not just add risk, it flips a winning system into a losing one.

#### Worked example: Kelly for a 55% / 1-to-1 system — and why you bet half

You have an edge: you win **55%** of the time, and your winners and losers are about the same size (even money, $b = 1$). What does Kelly say, and what should you actually do?

1. **Full Kelly.** $f^* = 2p - 1 = 2(0.55) - 1 = 0.10$. Kelly says risk **10%** of your account per trade.
2. **Reality check the volatility.** Risking 10% per trade means a run of, say, five losses in a row — which happens routinely at a 45% loss rate (probability $0.45^5 \approx 1.8\%$, so roughly one such streak every ~55 trades) — takes you from \$10,000 to $\$10{,}000 \times 0.9^5 \approx \$5{,}905$, a 41% drawdown. Sit with that number. Most people abandon a system somewhere around a 30% drawdown, and full Kelly will hand you 40%+ drawdowns *routinely*.
3. **Half-Kelly.** Risk **5%** instead. You keep about three-quarters of the growth rate, but that same five-loss streak now costs $\$10{,}000 \times 0.95^5 \approx \$7{,}738$ — a 23% drawdown instead of 41%. Far more survivable.
4. **And remember estimation error.** Is your win rate *really* 55%, or did you measure 55% from 80 trades (where the true rate could easily be 50%)? If it is truly 50%, your edge is *zero* and full Kelly is *zero* — so your "10%" bet was an infinite over-bet. Half- or quarter-Kelly protects you from your own optimistic measurement.

**The single sentence:** Kelly tells you the *ceiling* (10% here); fractional Kelly tells you where to actually stand (5% or less), trading a little growth for much shallower drawdowns and a buffer against being wrong about your edge.

#### Worked example: the recovery math, in dollars

You let one position get too big and it gaps against you. The account falls from **\$10,000** to **\$5,000** — a 50% drawdown. What does it take to get back?

- Not a 50% gain. A 50% gain on \$5,000 is only $\$5{,}000 \times 1.5 = \$7{,}500$ — still \$2,500 short.
- You need to **double** the account: $\$5{,}000 \times 2.00 = \$10{,}000$, a **+100%** gain, just to break even. Using $G = L/(1-L) = 0.5/0.5 = 1.0 = 100\%$.
- Contrast a disciplined trader who never let a single loss exceed 1R. After ten consecutive −1R losses (a brutal streak) at 1% sizing, they are down roughly $1 - 0.99^{10} \approx 9.6\%$, to about \$9,040. To recover they need about +10.6% — a few good weeks, not a doubling.

**The single sentence:** the depth of your worst drawdown is set almost entirely by your bet size, and because recovery cost explodes with depth (50% needs +100%, 75% needs +300%), keeping bets small is worth more than any entry edge — it is the difference between a recoverable dip and a hole you may never climb out of.

## Common misconceptions

Sizing is where good traders quietly go broke, usually because of one of these beliefs.

**"Bet more when you're more confident."** This feels obvious and is mostly wrong. Your *confidence* in a setup is a notoriously unreliable estimate of its actual edge — humans are systematically overconfident, and the trades you feel best about are often the ones where you have talked yourself past your own rules. Worse, sizing up on "high-conviction" trades concentrates your risk exactly when your judgment is most compromised. The honest version is the reverse of intuition: keep your risk *constant* per trade (the 1% rule), and let your *edge*, measured over many trades, do the compounding. If you genuinely have a way to measure that some setups have higher expectancy, the disciplined response is a *modestly* larger fraction (say 1.5% instead of 1%), grounded in measured expectancy — never "I feel good about this one, so I'll risk 10%."

**"A winning system can't blow up."** This is the central error the whole post exists to correct. A positive edge is *necessary* but not *sufficient* for survival. Bet a winning system past twice its Kelly fraction and its long-run growth is negative — the system wins on average and the account dies anyway, from volatility drag and the recovery asymmetry. Profitability is about the edge; survival is about the size. They are separate problems and you must solve both.

**"Kelly says bet big."** Kelly has a reputation as an aggressive, swing-for-the-fences formula. The opposite is closer to the truth. Kelly's deepest message is *don't bet so big that you blow up* — its whole construction is built around the constraint that going to zero is infinitely bad (the log of zero is $-\infty$). The number Kelly produces is the *maximum* you should ever consider, and the asymmetry of the growth curve means the prudent play is to bet a *fraction* of even that. People who "blow up trading Kelly" were not trading Kelly; they were trading *multiples* of Kelly, which is the one thing Kelly most forbids.

**"Position size is less important than the entry."** Most trader education is 90% about entries — patterns, indicators, signals — and almost none about sizing. The weighting is backwards. A great entry with reckless sizing goes to zero (the first figure's top curve); a mediocre entry with disciplined sizing survives and compounds a modest edge. Entries determine *whether* you have an edge; sizing determines *whether you live to collect it*. The cruel asymmetry: a bad entry costs you 1R, recoverable; a bad sizing decision can cost you the account, not recoverable. Spend your study time accordingly.

**"I'll just use a tight stop, so I can size huge."** A tight stop does let you take a larger *share count* for the same 1R — that is correct arithmetic. But a stop placed too tight to give the trade room gets hit by normal noise, turning what would have been winners into a stream of −1R losses. You have not reduced your risk; you have just guaranteed you realize it more often. The stop belongs where the trade is *actually invalidated* (see [building one high-probability setup](/blog/trading/technical-analysis/building-one-high-probability-setup) for placing invalidation honestly); the share count then follows from the 1% rule. Sizing does not let you cheat the chart.

**"Risking 1% means putting 1% into the trade."** Covered above but worth repeating because it is so common. 1% is the loss *if the stop is hit*, not the capital deployed. With a 5%-away stop, risking 1% means deploying ~20% of the account. Confusing the two leads people to take positions far smaller than intended (if they think 1% = 1% deployed) or to mis-estimate their true exposure.

## How it shows up in real markets

The geometry above is not a textbook curiosity; it is the mechanism behind some of the most instructive blow-ups and best practices in real trading. As-of dates are given where they anchor the lesson; the principles are evergreen.

**The profitable trader ruined by oversizing.** The most common version never makes the news because it happens to individuals. A retail trader develops or buys a system with a genuine edge — positive expectancy, verified over hundreds of trades. Then they size it at 10–25% of capital per trade, often after a hot streak convinces them the edge is "even better than I thought." A normal losing streak — the kind their own backtest contained — arrives, and at that size it carves a 60–80% drawdown. By the recovery math, climbing out needs +150% to +400%, which the edge cannot deliver before the trader, demoralized, abandons the system at the bottom. The post-mortem always finds the edge was real. The size was the killer. This is the single most common path from "profitable strategy" to "blown account," and it is entirely a sizing failure.

**Long-Term Capital Management (1998).** LTCM was run by Nobel laureates and brilliant traders, and most of its individual trades had genuine positive expected value — small, high-probability convergence bets. The fatal flaw was size: leverage that turned tiny edges into enormous positions, roughly \$125 billion of positions on ~\$4.7 billion of equity (over 25-to-1, and far higher counting derivatives). At that size, a move that their models treated as a once-in-centuries event — the August 1998 Russian default and the correlated flight to quality — produced a drawdown the equity could not survive. They were, in effect, betting many multiples of Kelly on each "sure thing," and the over-bet side of the curve does not care how smart you are. A near-\$3.6 billion Fed-organized recapitalization wound them down. The lesson is exactly this post's: positive edge plus oversize equals ruin.

**Fractional Kelly at quant funds.** On the constructive side, sophisticated quantitative funds that *do* use Kelly-style sizing almost universally run a *fraction* of it. Ed Thorp — who pioneered applying Kelly to blackjack and then to markets through Princeton/Newport Partners — has written and spoken (e.g., in his memoir *A Man for All Markets*, 2017) about deliberately betting *fractions* of full Kelly to control drawdowns, precisely because full Kelly's volatility is intolerable for a fund that must keep investors calm through drawdowns. The pattern across the industry is consistent: estimate an edge, compute the Kelly bet, then bet a quarter or a half of it. The funds that survive decades are not the ones that bet biggest; they are the ones that sized to survive their own bad years.

**The 1% rule in proprietary trading firms.** Walk into most prop firms or look at the rules of the modern "funded trader" challenges (e.g., the evaluation programs run by firms like FTMO and others, popular through the 2020s), and you find hard-coded risk-per-trade and maximum-drawdown limits — frequently a 1–2% per-trade cap and a total drawdown limit (often ~5–10%) that ends the account if breached. These firms have watched thousands of traders and learned the same lesson empirically: the fastest way to lose a funded account is not bad entries, it is oversizing. The rules exist to enforce fixed-fractional sizing on people whose instincts would otherwise push them to bet bigger after a loss. The firm is, in effect, mandating the 1% rule because it has seen what happens without it.

**The single oversized bet.** Beyond systems, individual concentrated bets have ended careers and firms. A trader who has been disciplined for years takes one position far larger than their normal size — convinced, leveraged, "this is the one" — and a gap or a fast move against it does damage the rest of the year's edge cannot repair. The mechanism is the recovery asymmetry: a single −40% bet needs the rest of the book to make +67% just to break even. The discipline of *never* letting one trade risk more than a small, fixed fraction is precisely the protection against the one catastrophic bet, because by construction no single trade *can* be catastrophic. Many famous trading-desk losses reduce, at their core, to one position that was allowed to become too large relative to the capital behind it.

**The "Martingale" account that wins until it doesn't.** A recurring pattern in retail forex and options is the trader who *averages down* or doubles up after every loss — the "Martingale" approach, named after the old casino betting system of doubling your bet after each loss to recover everything plus one unit on the eventual win. On the surface it looks unstoppable: the equity curve is a long, smooth climb of small, reliable wins, because most sequences do end in a win before the doubling gets out of hand. But the position size is doing the opposite of fixed-fractional sizing — it *grows* the bet as losses mount, which is precisely the instinct the 1% rule forbids. The smooth climb is the build-up; the inevitable long losing streak, when it comes, compounds the doubled bets into a single catastrophic loss that erases months of gains in a day. Many accounts that show a "90%+ win rate" and a beautiful curve are running exactly this, and the curve's serenity is not a sign of edge — it is a sign that the ruin has not arrived *yet*. The math is the over-betting curve in disguise: each doubling pushes the effective bet fraction further past Kelly until one streak lands it deep in the negative-growth zone.

**Survivorship in the visible track records.** A quieter market reality: the sizing failures are *invisible* in the strategies you can see. The trader who bet 1% and compounded a modest edge for a decade is still around to show you their record; the trader with the *identical* edge who bet 20% blew up in year two and is gone, their account closed, their name forgotten. When you study "successful" sizing, you are looking at survivors, and survivors are disproportionately the ones who bet small — not because small betting was magic, but because large betting *removed its practitioners from the sample*. This is worth internalizing: the reason every grizzled professional preaches small per-trade risk is not superstition or excessive caution. It is that they watched the over-bettors leave, one ruined account at a time, and they are the ones who remained to give the advice.

## When this matters to you and further reading

If you take one thing from this post, make it the ordering: **find an edge, then size it to survive.** The edge is necessary, but the world is full of traders with real edges and empty accounts, ruined not by being wrong but by being right too big. Position sizing is the part of trading with the highest leverage on your survival and the least glamour, which is exactly why it is under-studied and over-punished.

Concretely, the dial you actually turn is the fraction of equity you risk per trade. The 1% rule — risk a fixed small fraction of *current* equity, sized so a stop-out costs you that fraction — is a deliberately conservative fractional-Kelly bet that makes ruin nearly impossible while still compounding a positive edge. It self-corrects: it shrinks your bet in drawdowns and grows it on winning streaks, and it mechanically refuses the most destructive instinct in trading, which is to bet more to make back a loss. Kelly tells you the absolute ceiling ($f^* = $ edge over odds); the rounded, asymmetric growth curve tells you to stand well below it, because under-betting costs you a little growth while over-betting costs you everything.

This is educational, not advice — but it is the honest math, and you can now read any strategy's risk claims through it. When someone shows you a system, the first question is no longer "what's the win rate?" but "how big are they betting, and what does that do to the drawdown?"

For the rest of this series, the natural next steps: [why win rate lies and what expectancy really measures](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) is the foundation this post builds on — establish a positive edge before you size it. [Risk, reward, and expectancy in practice](/blog/trading/technical-analysis/risk-reward-and-expectancy-in-practice) walks the same numbers through live trade management. [Building one high-probability setup, end to end](/blog/trading/technical-analysis/building-one-high-probability-setup) is where the stop — the entry-to-stop distance that sets your per-share risk — gets placed honestly at real invalidation. And for the full mathematical treatment of Kelly through expected log-wealth, with sequential-betting derivations and solved interview problems, see the quant-finance companion, [the Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews). Read in that order, the four posts are one argument: have an edge, measure it honestly, size it to survive, and let compounding do the rest.

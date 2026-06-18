---
title: "The HFT Latency Arms Race: A Prisoner's Dilemma at Light Speed"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "High-frequency trading's billion-dollar speed race is a textbook prisoner's dilemma where investing in speed dominates for every firm but leaves them all worse off than mutual restraint would."
tags: ["game-theory", "high-frequency-trading", "prisoners-dilemma", "market-microstructure", "latency", "batch-auctions", "market-design", "trading"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The high-frequency-trading speed race is a prisoner's dilemma at light speed: spending on speed *dominates* for each firm because speed lets you pick off stale quotes and win the queue, but speed is relative, so when everyone invests the edge cancels and only the cost remains — leaving every firm worse off than if they had all shown restraint.
>
> - The payoff matrix has **"invest in speed" as the dominant strategy** for each firm, so the unique Nash equilibrium is mutual investment, even though mutual restraint would make everyone better off.
> - Because speed is *relative*, the billions spent on co-location, microwave links, and FPGAs **cancel out in aggregate** — the social benefit is roughly zero while the bill keeps climbing.
> - Slow orders pay a **latency tax**: a fast firm picks off a stale quote before the slow market maker can cancel it. That is the "Flash Boys" sniping you've heard about.
> - The proposed fix is the **frequent batch auction** — clearing every fraction of a second at a uniform price — which makes being a microsecond faster worth nothing, ending the race.
> - The one number to remember: when speed is relative, the marginal social value of being faster is **near zero**, so the entire arms-race spend is deadweight loss the rest of us pay for in wider spreads.

In 2010, a company called Spread Networks finished a tunnel. Not a subway tunnel — a fiber-optic cable, laid in the straightest possible line through the Allegheny Mountains, from a data center in Chicago to one in northern New Jersey. The build reportedly cost around \$300 million. The entire point of the project was to shave the round-trip time for a signal between the Chicago futures market and the New York stock market from about 14.5 milliseconds down to roughly 13.3 milliseconds. A milliseconds is one thousandth of a second. The tunnel bought its customers a little over one of them.

Why would anyone spend \$300 million to save a single millisecond? Because in that millisecond, a trading firm can see that the price of a stock has just moved in Chicago and react in New York before everyone else does — buying or selling against quotes that are now, for a fraction of a heartbeat, *stale* (priced off old information). And here is the strange part: within a couple of years, that \$300 million tunnel was already obsolete. Microwave towers — beaming signals through the air in a straight line, since light travels faster through air than through glass fiber — had beaten it. Then came millimeter-wave and laser links. Then specialized chips that make trading decisions in well under a microsecond (a millionth of a second). Each new round made the previous round's enormous investment worthless, and forced every serious firm to spend again just to stay in the same place.

That pattern — everyone spends a fortune to get ahead, but because the advantage is *relative*, the spending cancels out and leaves everyone exactly where they started except poorer — is one of the most famous structures in all of game theory. It is the **prisoner's dilemma**: a situation where each player's individually rational choice leads to a collectively terrible outcome. The diagram below is the mental model for this entire post — a payoff matrix where "invest in speed" is the smart move for each firm no matter what the other does, yet both firms end up at the worst shared outcome.

![Prisoner's dilemma payoff matrix for the high-frequency-trading speed race showing invest as the dominant strategy](/imgs/blogs/the-hft-latency-arms-race-a-prisoners-dilemma-at-light-speed-1.png)

This post builds that claim from zero. We will define high-frequency trading, latency, the prisoner's dilemma, and frequent batch auctions for a reader with no finance or game-theory background; compute the actual payoff matrix; quantify the latency tax that slow orders pay; work through why no amount of "we should all just cooperate" survives contact with the temptation to defect; and end with how a trader, a long-term investor, and a market designer should each think about a market built on this dilemma. If you have only ever heard "HFT is bad" or "HFT is good," the goal here is to replace the slogan with the structure.

## Foundations: HFT, latency, the prisoner's dilemma, and batch auctions from zero

Before we can call the speed race a prisoner's dilemma, we need four building blocks, defined plainly.

### What is high-frequency trading?

**High-frequency trading (HFT)** is a style of automated trading where a computer program submits, modifies, and cancels orders in tiny fractions of a second, holding positions for milliseconds to minutes rather than days or years. The firms doing it — names like Citadel Securities, Virtu, Jump Trading, Hudson River Trading, Tower Research — are not betting on whether a company's earnings will be good next year. They are competing for two specific, repeatable sources of profit.

The first is **market making**: continuously posting a price at which they will buy (the *bid*) and a slightly higher price at which they will sell (the *ask*). The gap between them is the **bid-ask spread** — the difference between the price you can sell at and the price you can buy at, and the market maker's gross margin. If a market maker buys 100 shares at \$50.00 (the bid) and sells 100 shares at \$50.01 (the ask) a moment later, it pockets the one-cent spread, or \$1.00 on the round trip, for providing liquidity (the service of always being there to trade with).

The second source is **picking off stale quotes**, which is the heart of our story. When new information arrives — a price move in a related market, a news headline, a large order hitting another exchange — the "fair" value of a stock changes instantly, but the quotes already resting on the order book do not update until someone updates them. For a sliver of time, there are buy and sell orders sitting at prices that are now wrong. The fastest firm to notice can trade against those stale quotes, capturing the difference. This is sometimes called **latency arbitrage** (profiting from the time delay), and it is what the journalist Michael Lewis described as "sniping" in his 2014 book *Flash Boys*.

### What is latency?

**Latency** is the time delay between when something happens and when you can act on it. In trading it is measured in milliseconds (thousandths of a second), microseconds (millionths), and even nanoseconds (billionths). It has several components: the time for a signal to travel the physical distance (limited by the speed of light), the time for the exchange's computers to process an order, and the time for your own software and hardware to make a decision. Firms attack every component:

- **Co-location**: renting rack space for your servers *inside* the exchange's own data center, so your cable to the matching engine is meters long instead of miles. Exchanges sell this directly.
- **Faster physical links** between data centers: dedicated fiber, then microwave, then laser/millimeter-wave, each cutting the New York-to-Chicago round trip further.
- **Faster decision hardware**: moving the trading logic off general-purpose CPUs and onto **FPGAs** (field-programmable gate arrays — chips you can wire for one specific task) or custom ASICs (application-specific integrated circuits), so the "tick-to-trade" time — from receiving a price update to firing an order — drops below a microsecond.

The crucial fact, which we will hammer on, is that **latency only matters relative to your competitors**. There is no prize for being fast in absolute terms. The prize goes to whoever is *fastest* — first in the queue, first to snipe the stale quote. Being two microseconds slower than the winner is the same as being two hours slower: you lose the race.

### What is the prisoner's dilemma?

The **prisoner's dilemma** is the most famous game in game theory. The classic story: two suspects are arrested and held separately. Each is offered a deal — betray your partner (defect) and you get a lighter sentence, while your partner gets a heavy one. If both stay silent (cooperate), the police can only convict them of a minor charge, so both get a light sentence. If both betray, both get a moderately heavy sentence. The trap is that no matter what your partner does, betraying gives *you* a better personal outcome — so both betray, and both end up worse off than if they had both stayed silent.

The formal structure uses four payoffs, conventionally labeled with letters. For two players choosing **C** (cooperate) or **D** (defect):

- **R** = Reward for mutual cooperation (both stay silent)
- **T** = Temptation to defect (you betray while your partner cooperates — your best outcome)
- **S** = Sucker's payoff (you cooperate while your partner betrays — your worst outcome)
- **P** = Punishment for mutual defection (both betray)

A game is a prisoner's dilemma when $T > R > P > S$. That ordering does two things at once. Because $T > R$ and $P > S$, defecting beats cooperating *no matter what the other player does* — so defect is the **dominant strategy** (the choice that is best regardless of the opponent). And because $R > P$, mutual cooperation beats mutual defection — so the dominant-strategy outcome is collectively worse than the outcome both players could have reached. That gap between individually rational and collectively optimal is the whole tragedy.

The solution concept that pins down the outcome is the **Nash equilibrium** — a set of choices where no player can improve their own payoff by unilaterally changing strategy, given what everyone else is doing. In the prisoner's dilemma, *(defect, defect)* is the unique Nash equilibrium: from there, neither player wants to switch to cooperate, because cooperating while the other defects gives the sucker's payoff $S$, which is even worse than $P$. We covered Nash equilibrium in depth in [Nash equilibrium, best response, and the price as a truce](/blog/trading/game-theory/nash-equilibrium-best-response-and-the-price-as-a-truce); here we just need the one-line version: it is the outcome that is *stable*, the place where the game comes to rest, which is not always the place that's good for the players.

### What is a frequent batch auction?

We'll need this one for the fix, so define it now. Today's stock markets run as a **continuous limit order book**: orders arrive one at a time, and the matching engine processes each the instant it lands, in arrival order. **Time priority** — first order at a price gets filled first — is baked in, which is exactly what makes being a microsecond faster valuable.

A **frequent batch auction (FBA)** changes the clock. Instead of matching continuously, the market collects all orders arriving during a short window — say 100 milliseconds, or even shorter — and then, at the end of the window, clears them all together at a single **uniform price**: the price that maximizes the quantity traded. Crucially, within a batch, *the order you arrived at does not matter*. If two firms both want to buy at the same price, they are treated as having arrived "at the same time" and are pro-rated. The reward for being a microsecond faster than your rival evaporates, because the auction discretizes time into chunks and ignores the ordering inside each chunk. Hold that thought — it is the escape hatch from the dilemma.

With those four pieces in hand, we can now show that the speed race has exactly the prisoner's-dilemma shape.

## Building the payoff matrix: why "invest" dominates

Let's model two competing HFT firms, A and B, each deciding whether to **invest in speed** or **don't invest** (show restraint). To keep the arithmetic friendly we'll put dollar payoffs in millions per year, and we'll choose numbers that capture the real economics: investing buys you the ability to pick off the *other* firm's stale quotes and win the queue, while restraint saves the spend but leaves you exposed if your rival invests.

Here is the logic behind each cell, before we read the numbers off the matrix:

- **Both restrain (don't, don't)**: neither firm spends on speed. They compete on price and service like normal market makers, splitting the spread revenue from end investors. Call each firm's payoff **+\$20M** — a healthy, low-cost business. This is the cooperative outcome, $R$.
- **A invests, B restrains (invest, don't)**: A is now microseconds faster than B. A picks off B's stale quotes and jumps the queue ahead of B all day long. A captures B's market-making profits *plus* the sniping profits, netting **+\$30M** even after the cost of the speed gear; B, getting picked off and out-queued, nets **−\$30M**. A's +\$30M is the temptation payoff $T$; B's −\$30M is the sucker's payoff $S$.
- **B invests, A restrains (don't, invest)**: the mirror image. B gets +\$30M, A gets −\$30M.
- **Both invest (invest, invest)**: both firms now have the fast gear. Neither can pick the other off, because they react at the same speed — the relative edge *cancels*. All that's left is the cost of the arms race, which both paid. Each nets **−\$20M**. This is the mutual-defection payoff $P$.

Let me write the payoffs as two matrices, one for each firm's perspective, and feed them to the `nash_2x2` solver from the series' data module to confirm the equilibrium. Row 0 is "don't invest," row 1 is "invest"; the same for columns.

```
import data_gametheory as gt

>>> A = [[20, -30], [30, -20]]    # Firm A's payoff (row = A's choice)
>>> B = [[20,  30], [-30, -20]]   # Firm B's payoff (col = B's choice)
>>> gt.nash_2x2(A, B)
{'pure': [(1, 1)], 'mixed': None}
```

The solver returns one pure Nash equilibrium: `(1, 1)` — both firms invest. There is no mixed equilibrium and no cooperative equilibrium. The matrix figure at the top of the post shows exactly this: the green "both restrain" cell at +\$20M each is where both firms would *prefer* to be, but the only place the game comes to rest is the amber "both invest" cell at −\$20M each.

Now let's verify the dilemma ordering. We have $T = 30$, $R = 20$, $P = -20$, $S = -30$, so $T > R > P > S$ holds — this is a genuine prisoner's dilemma. And we can see *why* invest dominates by checking each firm's best response:

- If B restrains, A compares restraining (+\$20M) to investing (+\$30M). **Invest wins.**
- If B invests, A compares restraining (−\$30M) to investing (−\$20M). **Invest wins.**

No matter what B does, A is better off investing. By symmetry, the same is true for B. So both invest, both land at −\$20M, and both are \$40M per year worse off than they would be at mutual restraint. The advantage each was chasing cancels in aggregate; the cost does not.

#### Worked example: the dominance check in dollars

Suppose you run Firm A and you're tempted to be the good citizen and *not* spend on the new microwave link. You reason: "If B also restrains, I make \$20M; if I'd invested I'd make \$30M — so I'd give up \$10M of upside by restraining." That's the $T - R = 30 - 20 = $ \$10M temptation. "But surely it's worth it to avoid the arms race." Then you remember B faces the same temptation. So you ask the harder question: "What if B invests and I don't?" Now you make −\$30M (B snipes you all year) instead of −\$20M (you'd at least be fast enough to not get picked off). Restraining while B invests costs you an *extra* \$10M, the $P - S = -20 - (-30) = $ \$10M sucker penalty. Either way — whether B cooperates or defects — restraining costs you \$10M relative to investing. The intuition: you don't invest in speed because you're greedy; you invest because *not* investing is strictly worse in every single scenario, and you can't control what B does.

This is the cruel signature of the prisoner's dilemma: the firms are not being stupid or shortsighted. Each is making the correct individual decision. The structure itself, not any villain inside it, produces the bad collective outcome.

### Two prizes speed buys: the snipe and the queue

It's worth slowing down on *what exactly* speed wins, because the matrix lumps two distinct prizes into one "invest" payoff and they reward speed for different reasons. The first prize, which we've described, is the **snipe**: when fair value moves, the fastest firm gets to trade against quotes that are now stale before they can be cancelled. The second prize is **queue priority**, and it's quieter but arguably more valuable day to day.

In a continuous order book, orders at the same price are filled in the order they arrived — **price-time priority**. So when a market maker wants to post a buy order at, say, \$50.00, where it lands in the queue matters enormously. If it's first in line, it gets filled first when a seller arrives, captures the spread, and — crucially — gets out of the way *before* the rest of the queue, which means it's less likely to be the one holding the position when the price moves against everyone at that level. Being microseconds faster lets you grab the front of the queue the instant a new price level opens, and lets you cancel and re-post faster when the level gets risky. The order-book mechanics of this fight — who gets to the front, who gets stuck, how "make-take" rebates tilt it — are their own deep subject, covered in the series' look at [every market is an auction: the double auction of the order book](/blog/trading/game-theory/every-market-is-an-auction-the-double-auction-of-the-order-book) and in [order types as strategic moves: market, limit, hidden, and pegged](/blog/trading/game-theory/order-types-as-strategic-moves-market-limit-hidden-and-pegged). For our purposes the point is that *both* prizes — the snipe and the queue — pay off only in *relative* speed, which is what makes the investment a dominant strategy and the aggregate outcome a dilemma.

### What happens with more than two firms

Our matrix has two firms for clarity, but the real market has many, and the dilemma only sharpens as the number grows. With $N$ firms racing, the snipe and the front of the queue go to *one* winner — the single fastest — so $N - 1$ firms eat the cost of their speed gear and collect almost nothing for it in the rounds they lose. That makes the temptation to be the one who invests even sharper (the winner takes a larger relative share), while the punishment for collective investment is spread across more firms who are all burning money to win a single prize. More players also makes any tacit "let's all stop" agreement more fragile, because it takes only *one* defector to make restraint a losing strategy for everyone who restrained. A two-firm dilemma is the gentlest version of this game; the live market is the hardened one.

## Speed is relative: the spending that cancels out

The single most important fact in this whole story is that **latency is positional, not absolute**. If every firm gets twice as fast, no firm is any better off relative to the others — the queue still has the same order, the snipes still go to the same winner, the spreads are no tighter. The benefit of your speed investment to *you* depends entirely on the gap between you and the next firm, and that gap is something your rivals can erase by matching your spend.

The chart below makes the dynamic concrete. The amber line is each firm's *cumulative* spend on speed as the arms race goes through its rounds — co-location, then fiber, then microwave, then laser, then FPGAs. It climbs and never comes back down, because each upgrade is a fixed sunk cost on top of the last. The blue line is the *relative* latency edge any one firm holds: it jumps up the moment that firm deploys a new medium first, then collapses back toward zero the moment its rivals match. Over the full sequence, the spend goes up and to the right forever, while the edge oscillates around zero.

![Cumulative speed spend rising while the relative latency edge keeps canceling back to zero](/imgs/blogs/the-hft-latency-arms-race-a-prisoners-dilemma-at-light-speed-2.png)

This is what economists call a **positional arms race** or a "red queen" race, after the character in *Through the Looking-Glass* who has to run as fast as she can just to stay in the same place. It shows up far outside trading: rival nations buying more missiles, athletes taking performance-enhancing drugs, peacocks growing ever-heavier tails, parents bidding up tutoring to get kids into the same number of college slots. In every case, the resource that matters is *rank*, and rank is conserved — for every firm that moves up, another moves down. Spending can change who holds rank but not how much total rank exists, so the spending is, in aggregate, a transfer plus a bonfire.

#### Worked example: when getting twice as fast buys you nothing

Suppose Firm A's tick-to-trade latency is 5 microseconds and Firm B's is also 5 microseconds. They tie for the snipe roughly half the time, each capturing about \$25M a year of pick-off profit. Now A spends \$15M on an FPGA upgrade and gets to 2.5 microseconds — twice as fast. For one quarter, A wins almost every race and its sniping profit jumps to ~\$45M while B's falls to ~\$5M. A's upgrade looks brilliant: a \$15M spend bought \$20M of extra annual profit. But B is not a rock. B spends \$15M on the same FPGA and also reaches 2.5 microseconds. Now they tie again, back to ~\$25M each — exactly where they started — except both are \$15M poorer. The intuition: the \$15M didn't buy A any *durable* profit, because the only thing it bought was a speed gap, and a speed gap is precisely the thing a competitor can close by writing the same check.

If you internalize one idea from this post, make it this: **a relative advantage that any rival can buy is not an advantage you can keep — it is a recurring tax you and your rivals impose on each other.** That is the engine of the dilemma.

## The latency tax: what slow orders actually pay

So far we've talked about the cost the HFT firms impose on *each other*. But the speed race also extracts value from everyone else in the market — the pension funds, mutual funds, and retail investors whose orders are, by HFT standards, glacially slow. This is the **latency tax**, and it's worth quantifying because the number is often either wildly exaggerated or hand-waved away.

Here is the mechanism, step by step. A market maker — even a fast one — posts a quote: "I'll sell 1,000 shares at \$50.01." That quote is a standing promise, and it's based on the market maker's current estimate of fair value, say \$50.005. Now a piece of information arrives: the S&P 500 futures in Chicago tick up, implying this stock is really worth \$50.02 now. The fair value has moved, but the \$50.01 ask is still sitting there, stale. A faster firm sees the futures move first, realizes the \$50.01 ask is now underpriced, and buys those 1,000 shares at \$50.01 before the market maker can cancel and reprice. The faster firm immediately turns around and the position is worth \$50.02 — it just made about a cent a share, ~\$10 on the thousand shares, at the slow market maker's expense.

The chart below shows how the size of this tax depends on how far fair value moved before the quote could be cancelled. The horizontal axis is the price move during the latency window; the vertical axis is the value extracted per share. Below the half-spread (here, half a cent), there's no pick-off — the move is inside the bid-ask cushion, so the quote isn't yet underwater. Beyond that, every additional cent of move that the fast firm captures before the slow firm can react is a cent transferred from the slow side to the fast side.

![Latency tax extracted per share rising with how far fair value moved before the quote could be cancelled](/imgs/blogs/the-hft-latency-arms-race-a-prisoners-dilemma-at-light-speed-3.png)

#### Worked example: the latency tax on a pension fund's order

Imagine a pension fund needs to buy 500,000 shares of a stock trading around \$50. It can't just slam a market order — that would move the price against it — so it works the order over the day, posting limit orders and accepting fills. Suppose that over the day, fast firms detect short-term price moves and pick off the fund's stale resting orders (or the market makers the fund trades against widen their quotes to protect themselves from the same snipers). Say the effective extra cost is half a cent per share. Half a cent on 500,000 shares is $0.005 \times 500{,}000 = $ \$2,500 on that one order. That sounds small against a \$25M order (it's one basis point — one hundredth of one percent). But the fund trades billions of dollars a year, and the tax recurs on every order, every day. The intuition: the latency tax is tiny per share and invisible to any individual investor, which is exactly why it can be levied across the whole market without provoking a revolt — it's a thousand papercuts, not one stab wound.

A vital nuance, and one the honest version of this story must include: the latency tax has shrunk dramatically over the era of HFT. Bid-ask spreads on liquid US stocks are far narrower than in the human-floor and early-electronic days; explicit trading costs have fallen. So the slow side pays a latency tax *and* benefits from tighter spreads and cheaper execution. The critique of the arms race is not "spreads are wide because of HFT" — they're not. The critique, which we turn to next, is subtler and sharper: spreads could be *even narrower* if the market makers weren't forced to price in the sniping risk and to burn money on the speed race. The waste is real even when the headline costs are low.

## The social-cost critique: Budish, Cramton, and Shim

In 2015, three economists — Eric Budish, Peter Cramton, and John Shim — published a paper in the *Quarterly Journal of Economics* titled "The High-Frequency Trading Arms Race: Frequent Batch Auctions as a Market Design Response." It is the intellectual backbone of this post, and its argument is precise.

Their central empirical finding: at human time scales, the price of the S&P 500 futures (in Chicago) and the SPY ETF (which tracks the same index, traded in New York) are nearly perfectly correlated — they move together, as they should, since they represent almost the same thing. But zoom in to the *millisecond* scale and that correlation breaks down. For a few milliseconds after one moves, the other hasn't caught up. And in those tiny windows, there is a guaranteed, mechanical arbitrage: buy the cheaper one, sell the dearer one, pocket the difference. This isn't a clever prediction; it's a race to be the first to grab a known profit. Budish, Cramton, and Shim estimated this latency-arbitrage opportunity was worth on the order of hundreds of millions of dollars a year, *and that it had not shrunk over time despite massive speed investment* — because, of course, the prize goes to whoever is fastest, and getting faster doesn't make the prize disappear, it just resets who collects it.

Their argument has three moves that are worth separating:

1. **The arms race is a prisoner's dilemma.** Every firm would prefer a world where no one spent on the speed race, but each firm individually must spend to capture the latency arbitrage and avoid being the sucker. This is exactly the matrix we built.
2. **The spending is socially wasteful.** The latency arbitrage is a pure transfer — a fixed prize that exists because of the continuous-market design, captured by whoever is fastest. The billions spent racing for it produce no new information and make prices no more accurate. It's rent-seeking: spending real resources to capture an existing pie rather than to bake a bigger one.
3. **It's a market-design flaw, not a behavior problem.** You cannot regulate or shame the firms out of it, because each is behaving rationally. The fix has to change the *rules of the game* — the matrix itself.

The grouped-bar chart below illustrates the gap their critique identifies. As the arms race climbs through its rounds, the collective *spend* (amber) keeps rising, but the collective *benefit* — the incremental improvement in price discovery or efficiency the market gets from being even faster (green) — shrinks toward nothing, because the market was already fast enough that nobody's information reaches prices meaningfully sooner. The widening distance between the bars is the deadweight loss: real resources burned for no social return.

![Collective speed spend rising while collective price-efficiency benefit shrinks across arms-race rounds](/imgs/blogs/the-hft-latency-arms-race-a-prisoners-dilemma-at-light-speed-4.png)

#### Worked example: rent-seeking versus value creation

Consider two ways a firm can spend \$100M. **Value creation**: it spends \$100M building a better model that genuinely forecasts where a stock should trade next week, so its trading pushes prices toward fundamentals faster — society gets more accurate prices, and the firm earns the reward for producing that information. **Rent-seeking**: it spends \$100M on a microwave tower to be 100 microseconds faster than its rival at grabbing a latency arbitrage that *already exists* purely because the two correlated markets update at slightly different times. After the second \$100M, prices are no more accurate; the only thing that changed is which firm collected the fixed prize, and the bill. The intuition: the arms-race spend is the second kind. It's not that the firms are doing nothing — they're working incredibly hard — it's that the *marginal* dollar of speed buys rank, not knowledge, and rank is zero-sum while the cost is not.

This is the deepest point of the social-cost critique. A market can be extremely efficient at the human scale — tight spreads, fast fills, accurate prices — and *still* be sitting on a structural prisoner's dilemma that burns hundreds of millions a year, because the dilemma lives in the milliseconds where the continuous-clock design manufactures a prize.

It's worth being precise about *who ultimately pays* for the burned resources, because it's easy to assume "the HFT firms eat it, so why should I care." Follow the money. The market makers must recover the cost of staying competitively fast somewhere, and the only place to recover it is the spread they charge everyone who trades — so a sliver of the arms-race cost is embedded in the bid-ask spread that every investor crosses. The exchanges, meanwhile, are paid handsomely to *sell* the speed (co-location and fast data feeds), so the arms race is a revenue stream they actively cultivate, funded by the firms, funded in turn by the spread, funded by the end investor. The chain is long and each link is small, which is why nobody feels robbed. But the aggregate is a real social cost: the talented engineers, the fiber, the towers, and the silicon devoted to shaving microseconds are resources that produce nothing the economy is better off having — they exist solely to redistribute a fixed prize the continuous-clock design invented. That is the precise meaning of "deadweight loss": output we pay for and don't get.

A fair counter-argument deserves airtime here, because the honest version of this debate has two sides. Defenders of the status quo point out that the speed race also drove genuine technological progress (cheaper, faster networking and computing that spilled over to other industries), that competition among fast market makers is exactly what compressed spreads to historic lows, and that the estimated deadweight loss, while large in absolute dollars, is small relative to the total value the equity markets create. All of that can be true at the same time as the dilemma being real. The careful claim is not "HFT is bad and should be banned" — it's narrower and harder to dodge: *the specific latency-arbitrage slice of the speed race is a prisoner's dilemma whose spend is socially wasteful, and a market redesign could remove it without sacrificing the liquidity and tight spreads HFT genuinely provides.* Keep the baby; drain the bathwater.

## The repeated game: why "let's all just stop" never holds

A natural objection: these firms compete against each other every single day, for years. Game theory tells us that in *repeated* games, cooperation can emerge even when it can't in a one-shot game — players can sustain a cooperative deal by threatening to punish any defector in future rounds. So why don't the HFT firms simply, tacitly, agree to stop racing? Each would save \$40M a year (the move from −\$20M to +\$20M in our matrix). Why doesn't the repeated nature of the game rescue them?

This is a genuinely important question, and the math gives a sharp answer. In a repeated prisoner's dilemma, mutual cooperation can be sustained as a stable equilibrium if the players care enough about the future — specifically if their **discount factor** $\delta$ (delta), a number between 0 and 1 measuring how much they value next period's payoff relative to this period's, is high enough. The threshold, under a "grim-trigger" strategy (cooperate until someone defects, then defect forever), is:

$$\delta \geq \delta^* = \frac{T - R}{T - P}$$

where $T$, $R$, $P$ are the temptation, reward, and punishment payoffs from before. The logic: if you defect now, you grab the one-time gain $T - R$, but you trigger eternal punishment, costing you $R - P$ every period forever after. Cooperation holds only if the discounted future punishment outweighs the immediate temptation. Let's compute the threshold for our speed-race matrix using the data module:

```
import data_gametheory as gt

>>> T, R, P, S = 30, 20, -20, -30        # our speed-race PD ($M)
>>> gt.repeated_pd_delta_threshold(T, R, P, S)
0.2
```

So $\delta^* = (30 - 20) / (30 - (-20)) = 10/50 = 0.2$. The firms only need to value the future at $\delta \geq 0.2$ — a very low bar — for mutual restraint to be *theoretically* sustainable. These are long-lived firms; their effective $\delta$ is surely well above 0.2. So why doesn't cooperation happen?

The chart below plots the threshold $\delta^*$ as a function of the temptation $T$ — how much a firm gains in the round where it's the *only* one to deploy a new speed edge. As the temptation rises, the required patience rises with it: the more lucrative it is to be the lone defector for even one round, the more a firm must value the future to resist. Our baseline sits at the amber dot, $T = $ \$30M giving $\delta^* = 0.20$; above the blue curve, restraint survives; below it, the temptation wins.

![Required discount factor for cooperation rising with the temptation to defect in the repeated speed race](/imgs/blogs/the-hft-latency-arms-race-a-prisoners-dilemma-at-light-speed-6.png)

The threshold being low is exactly why the *failure* of cooperation here is so telling. The math says restraint *should* be sustainable. Three things break it in practice, and all three are about the real structure of the game rather than the firms' patience:

**Defection is unverifiable.** The grim-trigger strategy requires you to *detect* when a rival has defected, so you can punish it. But how would Firm A know that Firm B secretly upgraded its FPGA last Tuesday? Speed investments are private. By the time A notices it's losing races, it has already been the sucker for months. You cannot punish a defection you cannot see, and you cannot sustain cooperation if defection is invisible. This is the killer: the repeated-game theory assumes observable actions, and the speed race is fought in secret.

**The temptation spikes are huge and lumpy.** A genuinely new medium — the first microwave link, the first laser network — can let its owner dominate for a quarter or more before rivals catch up. In that window, $T$ is not \$30M; it's enormous. And our chart shows that as $T$ grows, the patience required to resist grows too. A one-time prize big enough can break any cooperative equilibrium no matter how patient the firms are.

**There are too many players, and entry is open.** Tacit cooperation among a handful of firms is fragile, but even if the incumbents managed it, a new entrant has every incentive to defect — buy the speed, pick off the cozy cooperators who didn't, and capture the sucker payoffs. With free entry, the cooperative equilibrium is not just hard to reach; it's not robust to a single ambitious newcomer.

#### Worked example: the patience math, and why it doesn't save them

Suppose the firms could somehow observe each other perfectly and there were only two of them. Firm A considers defecting this round. The one-time gain is $T - R = 30 - 20 = $ \$10M. The cost is being punished forever after: instead of +\$20M each future period, both get −\$20M, a loss of $R - P = 20 - (-20) = $ \$40M per period. With a discount factor $\delta = 0.9$ (firms value next year at 90% of this year), the present value of that eternal \$40M punishment is $40 \times \delta / (1 - \delta) = 40 \times 0.9 / 0.1 = $ \$360M. Defecting to grab \$10M now, to lose \$360M later, is insane — so cooperation holds *easily* when actions are observable. The intuition: the patience math overwhelmingly favors restraint, which proves the dilemma isn't sustained by impatience. It's sustained by *unverifiability* — A can't punish what it can't detect, so the \$360M threat is empty, and the \$10M temptation wins by default.

This is the bleak in the dam. The textbook escape from the prisoner's dilemma — repeat the game and punish defectors — requires that defections be *seen*. The speed race is engineered, by its nature, to be invisible, so the escape is sealed off. The firms are trapped not because they're impatient but because they're blind to each other's choices.

## The fix: frequent batch auctions and changing the game

If you can't behave your way out of a prisoner's dilemma, you change the rules so the dilemma no longer exists. That's the proposal at the heart of the Budish-Cramton-Shim paper: replace the continuous limit order book with a **frequent batch auction (FBA)**.

Recall the definition from Foundations. In a continuous market, the matching engine processes orders the instant they arrive, in arrival order, so time priority makes being a microsecond faster valuable. An FBA instead collects all orders arriving within a short window — say every 100 milliseconds, perhaps eventually much shorter — and clears them all at once, at a single uniform price, ignoring the order in which they arrived *within* the batch. The before-and-after diagram below contrasts the two regimes.

![Continuous sniping of stale quotes versus a frequent batch auction that clears at one uniform price](/imgs/blogs/the-hft-latency-arms-race-a-prisoners-dilemma-at-light-speed-5.png)

Why does this kill the arms race? Two reasons, both decisive:

**Time priority is gone within a batch.** Being 50 microseconds faster than your rival no longer wins you the queue, because both orders land in the same 100-millisecond batch and are treated as simultaneous. The marginal value of shaving a microsecond drops to zero. The whole reason to spend \$300M on a tunnel disappears.

**Stale-quote sniping is largely defused.** In a continuous market, when news hits, there's a race to pick off the stale quote *before* the market maker can cancel. In an FBA, when news hits mid-batch, *everyone* who saw it — the sniper *and* the market maker — gets to submit orders into the same batch, and they all clear together at the uniform price. The market maker can cancel or reprice its stale quote in the same batch the sniper tries to hit it. The sniper no longer has a guaranteed head start, because the batch collapses the relevant time difference to zero. Competition shifts from *who is fastest* to *who offers the best price* — which is the competition we actually want.

#### Worked example: a snipe that fails inside a batch

Take our stale-quote story from earlier. The market maker has a \$50.01 ask resting. The futures tick up, implying fair value is now \$50.02. In a continuous market, the sniper who sees the futures first buys at \$50.01 before the market maker can cancel — the sniper makes ~1 cent a share, and the speed gap decides it. Now run it in a 100-millisecond batch. The futures move 30 milliseconds into the batch. Both the sniper *and* the market maker observe it with 70 milliseconds left in the window. The market maker cancels its stale \$50.01 ask and reposts at \$50.02; the sniper submits a buy. When the batch clears, there is no \$50.01 ask to hit — it's gone, repriced within the same batch. The snipe captures nothing. The intuition: the snipe in the continuous market worked only because the sniper's order beat the cancel by microseconds; the batch makes "beat by microseconds" meaningless, so the only thing that survives is honest price competition.

FBAs are not a fantasy. The economic logic is well-developed, several proposals and venues have experimented with short auctions and "speed bumps" (deliberate small delays designed to neutralize latency arbitrage — the IEX exchange's famous 350-microsecond delay is a cousin of this idea), and the academic case is strong. But they face real obstacles: incumbent exchanges earn substantial revenue selling co-location and fast data feeds, so they have little incentive to adopt a design that makes speed worthless; coordination across fragmented venues is hard (if only one exchange batches, traders route elsewhere); and there are genuine open questions about the right batch length and how FBAs interact with the rest of market structure. The point for this post is conceptual: the dilemma is solvable, but only by redesigning the game, not by appealing to the players' better natures.

The arms race itself has a history, and seeing it laid out makes the "relative, not absolute" point visceral — each round shaved milliseconds off the same New York-to-Chicago link, and each forced everyone to match it. The timeline below traces the escalation.

![Timeline of the latency arms race from co-location through fiber, microwave, laser, and FPGA rounds](/imgs/blogs/the-hft-latency-arms-race-a-prisoners-dilemma-at-light-speed-7.png)

## Common misconceptions

**"HFT firms are getting rich off a secret — if I were faster I could do it too."** The profit from latency arbitrage is real but it is *competed away at the margin* by the arms race. The firms spend most of what they could make racing each other to be the one who captures it. The economic rent doesn't sit in some firm's pocket untouched; a huge fraction of it is burned on co-location fees, fiber, microwave towers, and engineering talent. That's the whole point of calling it a prisoner's dilemma — the prize exists, but the competition to grab it dissipates much of its value. You wouldn't get rich joining the race late; you'd just add another defector to the matrix.

**"The speed race makes prices more accurate, so it's worth it."** Up to a point, faster trading does incorporate information into prices more quickly, which is socially valuable. But the social-cost critique is specifically about the *margin*: once the market is already fast at the human and even the multi-millisecond scale, being faster still does not make prices meaningfully more accurate — it just changes which firm wins a mechanical race for a fixed arbitrage. The first microsecond of speed buys price discovery; the ten-thousandth buys rank. Conflating "some speed is good" with "more speed is always good" is the error.

**"If it's a prisoner's dilemma, the firms should just agree to stop."** They can't, and not because they're greedy. Cooperation in a repeated game requires that defection be observable so it can be punished, and speed investments are private and invisible until you're already losing. With $\delta^* = 0.20$ the patience bar is trivially low, yet cooperation still fails — which proves the obstacle is unverifiability and lumpy temptations, not impatience. You can't enforce a treaty whose violations you can't detect.

**"Frequent batch auctions would hurt liquidity and investors."** The opposite is the theoretical prediction. By defusing stale-quote sniping, FBAs let market makers quote tighter, because they no longer have to price in the risk of being picked off by someone microseconds faster. Tighter spreads and less adverse selection are good for the slow, ordinary investor. The parties who lose are the exchanges selling speed and the firms whose edge *is* speed — which is exactly why adoption is slow despite the favorable economics.

**"HFT is the same thing as latency arbitrage."** No. HFT is a broad category that includes genuine, valuable market making — continuously providing liquidity, tightening spreads, connecting buyers and sellers across fragmented venues. The arms-race critique targets the *latency-arbitrage* slice specifically, where speed is used to pick off stale quotes. A well-run market keeps the liquidity provision and removes the incentive to race; that's precisely what FBAs aim to do. Throwing out all of HFT to stop the arms race would be discarding the cure with the disease.

## How it shows up in real markets

**Spread Networks' Chicago-New York fiber (2010).** The canonical opening shot of the public arms race. The company spent a reported ~\$300M boring a near-straight fiber route through the Alleghenies to cut the round-trip latency to about 13.3 milliseconds, then leased capacity to trading firms at premium prices. Within roughly two years, microwave networks beat it — air carries signals faster than glass over the same path — and the fiber's speed advantage evaporated. The episode is the prisoner's dilemma made physical: a \$300M sunk cost rendered nearly worthless by the next round, exactly as the "speed is relative" chart predicts.

**The microwave land grab (2011-2013).** Firms and specialist networking companies raced to build line-of-sight microwave towers along the Chicago-New Jersey corridor, buying rooftop rights and tower sites, because microwave's straighter, faster-medium path beat fiber by roughly a millisecond. Latency on the link fell toward ~8.5 milliseconds. The catch: microwave degrades in heavy rain and fog, so firms then had to build redundant paths and fall back to fiber in bad weather — more spend, to defend an edge that any competitor could match by building their own towers. Pure positional racing.

**The Budish-Cramton-Shim ES-SPY study (2015).** Using direct-feed data, the authors documented that the futures-ETF arbitrage opportunity in those millisecond windows was worth on the order of hundreds of millions of dollars a year and had *not* declined over years of speed investment — the signature of a race for a fixed prize rather than a vanishing inefficiency. This is the empirical anchor for the claim that the arms-race spend is deadweight: the prize is constant, so faster racing redistributes it without shrinking it.

**IEX and the speed bump (2016).** The Investors Exchange, of *Flash Boys* fame, built a deliberate 350-microsecond delay (a coil of fiber, the "magic shoebox") in front of its matching engine, specifically to neutralize latency arbitrage against resting orders. The SEC approved it as a national exchange in 2016 after a contentious debate. IEX is not a full frequent batch auction, but it's a real-world, regulator-blessed instance of the same idea: change the rules so being microseconds faster stops paying, rather than asking firms to stop racing.

**FPGA and "tick-to-trade" competition (2017-present).** As the physical-link races saturated, the frontier moved into hardware. Firms moved trading logic onto FPGAs and custom silicon to get tick-to-trade times below a microsecond. The cost shifted from civil engineering (towers, tunnels) to chip engineering and elite hardware talent, but the structure is identical: a relative edge any rival can buy, so the spend recurs and the aggregate advantage cancels. The arms race didn't end; it changed medium.

**Exchange co-location revenue.** A revealing tell of why the dilemma persists: major exchanges earn substantial, recurring revenue selling co-location rack space and ultra-low-latency data feeds. The venues that would have to adopt batch auctions are the same venues monetizing the speed race. This is the institutional reason a profitable-for-everyone fix stays on the shelf — the referee is paid by the arms dealers.

## The playbook: how to play a market built on this dilemma

You're not an HFT firm, so the question isn't "should I join the arms race" — you can't and shouldn't. The question is how to act inside a market shaped by it. Here's the lens, by who you are.

**Who's on the other side, and what game you're in.** When you send a marketable order into a continuous market, you are, in a small way, the slow player in this dilemma. The other side may be a market maker pricing in the risk that *someone faster than it* will pick it off, and occasionally a latency arbitrageur reacting to a signal a beat before you. You are not going to out-speed them; that race is over and you didn't enter it. Your edge is not speed — it's *patience and design awareness*.

**For the everyday investor:** the latency tax on you is real but tiny — on the order of a fraction of a basis point per trade for liquid names — and it is swamped by the secular collapse in spreads and commissions over the HFT era. The practical move is the boring one: trade liquid instruments, use limit orders rather than market orders where you can, don't trade in the first or last seconds around volatile news (where stale-quote risk spikes), and don't pay up for "fast" retail products you don't need. The invalidation of worrying about HFT: if you're a long-term investor holding for years, the milliseconds are noise — your returns are decided by what you own, not by who got to the queue first.

**For the institutional trader:** this is where the dilemma touches you with real dollars. You move size, so you *are* the slow whale whose stale orders get picked off and whose footprint gets detected. Your defenses are execution-design choices, not speed: slice orders to hide your size, randomize timing so you're not predictable, use venues with speed bumps or auction mechanics (IEX-style delays, periodic auctions, midpoint dark pools) that neutralize latency arbitrage against you, and lean on the opening and closing auctions for size, since an auction is itself a batch that ignores microsecond ordering. We go deep on this in [execution as a game: VWAP, TWAP, and hiding from predators](/blog/trading/game-theory/execution-as-a-game-vwap-twap-and-hiding-from-predators). The connection to *why* a fast fill should worry you is in [adverse selection and the winner's curse: why a fast fill is bad news](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news) — if your resting order filled instantly, ask whether someone faster knew something you didn't.

**For the market designer or policy thinker:** the lesson is that you cannot legislate or shame players out of a prisoner's dilemma, because each is rational. You change the *rules*. Frequent batch auctions and speed bumps are the leading proposals; they work by deleting time priority within a window so speed stops paying. The trade-offs to weigh: batch length (too long hurts genuine price discovery, too short reintroduces the race), cross-venue coordination (a single batching venue gets routed around), and whose revenue you're disrupting (the exchanges selling co-location). The invalidation of the whole fix: if firms can predict the batch close and snipe at the boundary, you've just moved the race to the edge of the window — so the auction's timing must itself be hard to game.

**Sizing and exit, in dilemma terms.** The meta-point of this series is to reason one level deeper than your counterparty. In the speed race you can't win the speed level, so you exit it entirely — you compete on the dimensions where being slow doesn't lose: instrument selection, holding period, execution design, and venue choice. The size you should put into "fighting HFT" is roughly zero; the size you should put into *not being the predictable sucker* — limit orders, auctions, randomized slicing, long horizons — is most of it. The firms are trapped at −\$20M each in their matrix. You don't have to sit in any cell of it.

A market can be a marvel of efficiency at the scale a human cares about and a wasteful prisoner's dilemma at the scale a microwave tower cares about, both at once. Understanding which scale you're operating at — and refusing to play a relative-speed game you can only lose — is the whole edge here.

## Further reading & cross-links

- [The prisoner's dilemma in markets: why everyone sells at once](/blog/trading/game-theory/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once) — the foundational dilemma this post specializes; start here if the $T > R > P > S$ structure is new.
- [The trade is a game: why markets are strategic, not random](/blog/trading/game-theory/the-trade-is-a-game-why-markets-are-strategic-not-random) — the series' opening thesis that every trade has a thinking counterparty.
- [Execution as a game: VWAP, TWAP, and hiding from predators](/blog/trading/game-theory/execution-as-a-game-vwap-twap-and-hiding-from-predators) — the practical playbook for the slow whale defending against fast predators.
- [Adverse selection and the winner's curse: why a fast fill is bad news](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news) — why getting filled instantly often means someone faster knew more than you.
- [Nash equilibrium, best response, and the price as a truce](/blog/trading/game-theory/nash-equilibrium-best-response-and-the-price-as-a-truce) — the equilibrium concept that pins the dilemma's outcome.

*This article is educational, not financial advice. It explains the strategic structure of a market mechanism; it does not recommend any trade, venue, or product.*

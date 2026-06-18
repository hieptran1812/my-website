---
title: "The Market Maker's Game: Inventory, the Spread, and Fear of the Informed"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "How a real dealer plays the market-making game: quote both sides, earn the spread from uninformed flow, and price two fears — inventory risk and the informed counterparty — into every quote."
tags: ["game-theory", "market-making", "bid-ask-spread", "adverse-selection", "inventory-risk", "glosten-milgrom", "trading", "microstructure", "liquidity"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A market maker is not a toll booth charging you a fee; it is a player in a game whose spread is the price it must charge to survive two enemies, inventory risk and the informed counterparty, and the moment you understand that, you stop seeing the spread as a cost and start seeing it as a signal.
>
> - The dealer's job is to quote a two-sided market and earn the **bid-ask spread** on round-trips with uninformed flow, but every fill leaves it holding inventory it never wanted.
> - **Inventory risk:** a fill makes the dealer long or short, exposed to the next price move; it skews its quotes (quotes lower when long) to lean its position back toward flat — the reservation-price idea.
> - **Adverse selection:** the dealer fears the person hitting its quote *knows something*; it widens the spread to survive being picked off — the Glosten-Milgrom logic.
> - The spread is therefore a **strategic price covering both inventory and adverse-selection costs**, not a fee. When flow turns toxic, the spread must widen or the dealer goes broke.
> - The number to remember: in a Glosten-Milgrom market where the asset can be worth 110 or 90, a flow that is **50% informed forces a \$10 spread**; a flow that is **0% informed allows a zero spread**. The spread *is* the fear of the informed, priced.

You send a market order to buy 100 shares. It fills instantly at \$50.02. Half a second later you send a sell order and it fills at \$49.98. You just paid four cents — \$4 on the round-trip — to a counterparty you never saw, for a service you barely noticed. That counterparty is a market maker, and the four cents felt like a fee, like the spread a currency kiosk takes at the airport.

It is not a fee. The market maker did not decide it deserves four cents the way a shop decides on a markup. It arrived at four cents by solving a game — a strategic problem with at least two opponents who can hurt it — and four cents is the answer that lets it survive. If the dealer charged less, a smarter trader would bleed it dry; if it charged more, you would route your order somewhere cheaper and it would earn nothing. The spread is an equilibrium. The whole point of this post is to build, from zero, *why* that equilibrium sits where it sits, and what it tells you about who is really on the other side of your trade.

The diagram below is the mental model for the entire post: the spread is the dealer's answer to two fears stacked on top of each other. The naive view on the left treats the spread as a flat toll the dealer pockets. The game view on the right shows what the spread is actually paying for — the risk of being stuck with inventory, and the risk that you, the counterparty, know something the dealer does not.

![Before and after view of why a market maker sets the bid-ask spread, naive fee view versus inventory plus adverse selection](/imgs/blogs/the-market-makers-game-inventory-the-spread-and-fear-of-the-informed-1.png)

This is one of the most useful pieces of market intuition you can own, because the same two fears explain a hundred surface phenomena: why spreads gap open around earnings, why dealers vanish in a crash, why your "good fill" should sometimes scare you, why the close is the most aggressive time of the day, and why liquidity is never really free. We will build the dealer's game one piece at a time, with the arithmetic shown at every step, and end on how *you* — a trader on the other side — should play against it.

## Foundations: who the market maker is and the two fears that price the spread

Before any model, we need four terms defined from scratch. A reader with no finance background should be able to proceed from here without guessing.

A **market maker** (also called a *dealer* or *liquidity provider*) is a trader whose business is not to predict where prices go but to stand in the middle and quote both sides of a market continuously. It posts a **bid** — the price at which it is willing to *buy* from you — and an **ask** (or *offer*) — the price at which it is willing to *sell* to you. The ask is always higher than the bid. The gap between them is the **bid-ask spread**. If the dealer is willing to buy at \$49.98 and sell at \$50.02, its bid is \$49.98, its ask is \$50.02, and the spread is four cents. The **mid** (or *mid-price*) is the average of the two, \$50.00 here, and is the market's best guess of fair value at that instant.

The dealer's dream is a clean **round-trip**: someone sells to it at the bid (\$49.98), and a moment later someone else buys from it at the ask (\$50.02). The dealer never took a view on direction; it simply bought low and sold high within the same breath and kept the four-cent spread. Do that ten thousand times a day across thousands of names and the pennies become a serious business. This is the core of market making: **earn the spread on round-trips with flow that has no opinion.**

To see how thin the dealer's per-trade edge really is, put one number on it. Suppose a dealer earns a four-cent spread and turns over its inventory cleanly. On a \$50 stock, four cents is a 0.08% edge per round-trip — eight one-hundredths of one percent. The entire business is built on capturing that sliver, tens of thousands of times a day, while losing as little of it as possible to the two fears. A dealer that gives back even a quarter of that sliver per trade to inventory swings and informed traders has no business left. This is why the spread is set with such care: it is not a comfortable margin with room to spare, it is a knife-edge where a few basis points of leakage is the difference between a profitable franchise and a blown-up one.

That phrase — *flow that has no opinion* — is doing enormous work, and it is where the two fears come from.

The first fear is **inventory risk.** Round-trips do not arrive in neat pairs. Sometimes ten people sell to the dealer before anyone buys. Now the dealer is **long** (holding) a thousand shares it did not want, exposed to the next price move. If the price drops a nickel before it can sell, it loses \$50 per hundred shares, dwarfing the spread it was trying to earn. The dealer never wanted to bet on direction, but inventory *forces* a directional bet on it. Managing that unwanted exposure is half the game.

The second fear is **adverse selection.** This is subtler and more dangerous. "Adverse selection" means you tend to get *selected against* — the trades that come to you are disproportionately the ones that hurt you. When you quote a firm price to the whole world, who is most eager to take it? The person who knows your price is wrong. If a company is about to announce terrible news and you are still offering to buy its stock at \$50, the person who heard the news first will gleefully sell to you at \$50, and you will be holding a stock that is about to be worth \$45. You were *picked off*. The dealer cannot tell, at the moment of the fill, whether the seller is a retiree rebalancing or an insider who knows the building is on fire — so it must price in the *possibility* that every counterparty is informed.

These are the two fears in the cover diagram, and the rest of the post is the arithmetic of each one. The **solution concept** — the game-theory idea that ties it together — is simple to state: the dealer sets a spread wide enough that the money it loses to inventory swings and to informed traders is *recovered* by the spread it earns from everyone else. The spread is the equilibrium price of providing immediacy in a world where some of the people taking your quote know more than you do.

One more piece of vocabulary, because we will use it constantly. **Toxic flow** is order flow that is informed or adversely selected — flow that, on average, leaves the dealer worse off. **Benign flow** is uninformed — a saver, an index fund rebalancing, a tourist — flow the dealer is happy to trade against all day. The dealer's entire skill is reading which is which from the *footprint* of the order, and answering each with the right quote. We will get there.

## The spread that has nothing to do with information: order-processing cost

Start with the simplest possible world, then add realism one layer at a time. That is the only honest way to build a dealer's spread.

Imagine the friendliest possible market: every trader is uninformed, prices never move, and the dealer holds inventory for zero seconds because buys and sells arrive in perfect alternation. Even here, the spread is not zero. The dealer pays exchange fees, technology costs, and the cost of its own capital. It needs a few cents of edge just to keep the lights on. This is the **order-processing cost** — the part of the spread that is a genuine fee for a service, the part that *does* resemble the airport kiosk. In a deeply liquid, calm market it might be a cent or less per share. It is the floor of the spread, and it is the least interesting part, because it does not respond to who is on the other side. The other two components — inventory and adverse selection — are where the game lives, and they can dwarf the processing cost by a factor of ten or more.

Keep this decomposition in your head, because we will assemble the full spread out of exactly three pieces: **order-processing cost + inventory-risk cost + adverse-selection cost.** That sum is the spread. Each term is a number you can compute, and the rest of the post computes them.

## Adverse selection: the fear that the counterparty knows more (Glosten-Milgrom)

Let us tackle the deepest fear first, because it is the one that turns market making from a service into a game. The cleanest model of it is the **Glosten-Milgrom model**, named for Lawrence Glosten and Paul Milgrom, who in 1985 showed that a spread can exist *purely* from the fear of informed traders — even with zero inventory risk and zero processing cost. (This post links out to the dedicated treatment of that model rather than re-deriving every line; see [the bid-ask spread as an adverse-selection game](/blog/trading/game-theory/the-bid-ask-spread-as-an-adverse-selection-game-glosten-milgrom). Here we use it as the dealer's adverse-selection thermometer.)

The setup is a sealed, one-trade-at-a-time world. An asset is worth either a high value or a low value — say \$110 or \$90 — and nobody yet knows which; each is equally likely, so the fair mid is \$100. Traders arrive one at a time. A fraction of them are **informed** — they already know whether the true value is 110 or 90, and they will only buy when it is 110 and only sell when it is 90. The rest are **uninformed** (benign) — they buy or sell for reasons unrelated to value, a coin flip from the dealer's perspective.

Now stand in the dealer's shoes. Someone wants to buy from you. What does that tell you? An informed trader buys *only* when the value is high. An uninformed trader buys regardless. So a buy order is *evidence*, however weak, that the value is high — and the more of your flow is informed, the stronger that evidence. A rational dealer therefore cannot sell at the fair mid of \$100; it must sell at the *expected value of the asset given that someone wanted to buy it*, which is above \$100. Symmetrically, a sell order is evidence the value is low, so the dealer buys at a price below \$100. The gap between those two conditional expectations is the spread, and it exists **entirely because of the fear of the informed.** No inventory. No fees. Just inference.

The chart below is that spread, computed directly from the model as the fraction of informed flow rises from zero to one hundred percent.

![Adverse selection spread rising with the fraction of informed flow, computed from the Glosten-Milgrom model](/imgs/blogs/the-market-makers-game-inventory-the-spread-and-fear-of-the-informed-2.png)

Read the line. At the far left, **zero percent informed**, the spread is zero — if nobody knows anything, a buy order tells the dealer nothing, so it quotes both sides at the fair \$100 and earns no spread from information at all. At the far right, **one hundred percent informed**, the spread is the full \$20 range of the asset — every buyer is an insider who knows it is worth \$110, so the only safe ask is \$110, and the only safe bid is \$90. In between, the spread rises in a straight line: roughly \$20 times the fraction informed.

#### Worked example: the adverse-selection spread at 20% informed

Let us do the dealer's inference by hand, because seeing the arithmetic once makes the whole model click. Asset worth \$110 or \$90, equally likely, so the prior mean is \$100. Suppose 20% of traders are informed and 80% are uninformed (who buy or sell 50/50).

First, how likely is a buy order, and what does it imply? An informed trader buys only in the high state. So the probability you see a buy, *given* the true value is \$110, is the 20% informed (who all buy) plus half of the 80% uninformed: 0.20 + 0.40 = 0.60. Given the value is \$90, only the uninformed buy, and only half of them: 0.40. Since the two states are equally likely, the overall chance of a buy is (0.5 × 0.60) + (0.5 × 0.40) = 0.50 — a buy is a coin flip, as it must be by symmetry.

Now the key step. Given that a buy *did* arrive, how likely is the value actually \$110? By Bayes' rule, it is the chance of "high and a buy" divided by the chance of "a buy": (0.5 × 0.60) / 0.50 = 0.30 / 0.50 = 0.60. So conditional on a buy, there is a 60% chance the asset is worth \$110 and a 40% chance it is worth \$90. The dealer's fair **ask** is therefore the expected value given a buy: 0.60 × \$110 + 0.40 × \$90 = \$66 + \$36 = **\$102.** By the mirror-image argument, a sell drags the conditional expectation down to a **bid of \$98.** The spread is \$102 − \$98 = **\$4.**

You can confirm this against the model directly:

```
>>> import data_gametheory as gt
>>> r = gt.glosten_milgrom(110, 90, 0.5, 0.20)
>>> round(r["ask"], 2), round(r["bid"], 2), round(r["spread"], 2)
(102.0, 98.0, 4.0)
```

The intuition: a \$4 spread on a \$100 stock is not greed — it is exactly the amount the dealer must charge to break even against a flow that is one-fifth insiders.

#### Worked example: the spread when half the flow is informed

Now turn the fear up. Keep the asset at \$110 or \$90, but let **50% of traders be informed.** Redo the inference: the chance of a buy given the high state is 0.50 (all informed) + 0.25 (half the uninformed) = 0.75; given the low state it is 0.25. Conditional on a buy, the probability of the high state is (0.5 × 0.75) / 0.50 = 0.75. The ask is 0.75 × \$110 + 0.25 × \$90 = \$82.50 + \$22.50 = **\$105.** The bid is the mirror, **\$95.** The spread has **leapt to \$10** — two and a half times wider than the 20%-informed case, for a flow only two and a half times more toxic.

The lesson the dealer lives by: the spread scales with the *suspected* fraction of informed flow, and that fraction is not fixed — it spikes around news, earnings, and fast markets. A dealer who keeps a four-cent quote into an earnings release when half the incoming orders are informed is handing money to insiders. So it widens, or it stops quoting. The single sentence to carry: **the spread is the fear of the informed, converted into cents.**

This is also the bridge to the broader adverse-selection idea that should change how you read your own fills: if your order filled instantly and fully, ask *why the dealer was so happy to trade with you* — and whether the price is about to move against the dealer (good for you) or against you (the dealer was right to fear you was *not* the case, and you are the uninformed flow). That uncomfortable logic is the subject of [adverse selection and the winner's curse](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news).

## Inventory risk: every fill is an unwanted bet (Ho-Stoll, Avellaneda-Stoikov)

Now the second fear, which is mechanically simpler but just as deadly. Set information aside entirely — assume every trader is benign — and the dealer *still* has a problem, because fills do not arrive in matched pairs.

When a sell order hits the dealer's bid, the dealer is now **long** the shares. Until an offsetting buyer shows up, the dealer is a directional bettor: if the price ticks down, it loses; if it ticks up, it gains. It never wanted this bet. Worse, the longer it holds and the bigger the position, the more a random price wiggle can swamp the spread it was trying to earn. A dealer that lets inventory pile up is a dealer that has quietly become a leveraged momentum trader without meaning to.

The classic models here are **Ho and Stoll** (1981), who first framed the dealer as an inventory manager, and **Avellaneda and Stoikov** (2008), who turned it into the practical recipe high-frequency dealers actually use. The core idea is the **reservation price.** The reservation price is *the price at which the dealer is genuinely indifferent to its current inventory* — not the market mid, but the mid adjusted for how much the dealer wants to get flat. A simple, honest version:

$$r = s - q \cdot \kappa$$

where $s$ is the market mid, $q$ is the dealer's net inventory (positive when long), and $\kappa$ (kappa) is a per-share "inventory penalty" capturing how risk-averse the dealer is and how volatile the asset is. When the dealer is flat ($q = 0$), the reservation price equals the mid. When the dealer is long ($q > 0$), the reservation price drops *below* the mid — the dealer secretly values the asset less than the market does, because it already has too much of it.

The dealer then quotes its spread *around the reservation price, not around the mid.* Bid = $r$ − half-spread; ask = $r$ + half-spread. The figure below shows exactly how that drags both quotes when inventory builds.

![Inventory skew showing the reservation price and bid and ask sliding below the mid as the dealer gets long](/imgs/blogs/the-market-makers-game-inventory-the-spread-and-fear-of-the-informed-3.png)

Walk the chart. The dashed horizontal line is the true mid at \$100. When the dealer is flat (center), its quotes straddle \$100 symmetrically: bid \$99.90, ask \$100.10. As it gets **long** (rightward), the whole quote band slides *down*. As it gets **short** (leftward), the band slides *up*. The dealer is not changing the *width* of its spread here — it is changing the *center*. This is **inventory skewing**, and it is the dealer's steering wheel.

#### Worked example: skewing quotes when you are long 100 shares

Let us put numbers on it. Mid $s = \$100.00$, half-spread = \$0.10, and an inventory penalty $\kappa = \$0.01$ per share. The dealer just got filled and is now **long 100 shares.**

Reservation price: $r = 100.00 − 100 \times 0.01 = 100.00 − 1.00 = \$99.00.$ The dealer now centers its quote on \$99.00, not \$100.00. So its new **bid is 99.00 − 0.10 = \$98.90** and its **ask is 99.00 + 0.10 = \$99.10.**

Look at what this does to the world. The ask of \$99.10 is now *below* the true mid of \$100 — the dealer is practically *begging* someone to buy its excess inventory, offering shares a dime under fair value. Meanwhile its bid of \$98.90 is well below the old \$99.90, so it is *discouraging* further sellers — anyone who wants to sell to it now gets a worse price and will likely trade elsewhere. The skew simultaneously **attracts the flow it wants** (buyers, to take it back to flat) and **repels the flow it fears** (more sellers, who would deepen its unwanted long). One number, $\kappa$, does both jobs.

#### Worked example: when a short adverse move erases a session's spread

Now feel the danger. Suppose the dealer ignored skewing and let inventory run to **long 200 shares**, having booked \$5 of spread profit over the session. The price then drifts down by just **five cents** before it can flatten.

Inventory P&L = 200 shares × (−\$0.05) = **−\$10.** Net result for the session: +\$5 spread − \$10 inventory loss = **−\$5.** A nickel — a single tick on many stocks — turned a profitable day into a loss, purely from carrying inventory. The chart below makes the geometry vivid: the steeper the line, the bigger the inventory, and the faster a small move flips the dealer from green to red.

![Market maker session P and L versus price move for three inventory levels showing how inventory steepens the exposure](/imgs/blogs/the-market-makers-game-inventory-the-spread-and-fear-of-the-informed-4.png)

The flat green line is a dealer who stayed flat — its P&L is just the \$5 of spread it earned, immune to the price move. The blue line (long 50) tilts. The red line (long 200) is a near-vertical cliff: the dealer has accidentally become a leveraged directional trader, and a tiny adverse move wipes a whole session's grind. The intuition in one sentence: **inventory converts the dealer from a spread-earner into a price-bettor it never agreed to be, which is exactly why flattening is not optional.**

This is why real dealers obsess over inventory limits, why they widen the side of the spread that would deepen a bad position, and — the punchline of the section — why they **flatten into the close.** Holding inventory overnight means wearing the gap risk of every piece of news that breaks while the market is shut, with no ability to skew out of it. We will return to the close.

### What sets the inventory penalty: volatility, horizon, and risk appetite

The whole inventory model hangs on one number, $\kappa$, the per-share penalty that drags the reservation price away from the mid. So where does $\kappa$ come from? It is not a constant the dealer picks once and forgets; it is a live function of three things, and watching them move tells you when a dealer will skew gently versus violently.

The first input is **volatility.** The more the asset can move while the dealer is stuck with inventory, the more each share of exposure costs it, so $\kappa$ scales with the asset's variance. In the Avellaneda-Stoikov formulation, the inventory term carries a factor of $\sigma^2$ (the variance of returns): double the volatility and the inventory penalty roughly quadruples. This is why a dealer in a sleepy utility stock can sit on a thousand shares without flinching, while a dealer in a high-beta name with a thousand shares is sweating — the *same* inventory is a far bigger bet when the asset moves more.

The second input is the **time horizon to flatten,** $(T - t)$ — how long the dealer expects to be stuck holding before it can get flat. Early in the session, with hours left to find a natural offsetting trade, the dealer can afford to be patient and $\kappa$ is small. As the close approaches and the runway shortens, the penalty climbs, because there is less time for a benign buyer to show up and rescue the position. At the bell, the dealer either flattens or carries overnight gap risk, so $\kappa$ effectively spikes — which is exactly the formal reason the close is so aggressive.

The third input is the dealer's **risk appetite,** captured by a risk-aversion parameter (often written $\gamma$, gamma). A well-capitalized dealer with a big balance sheet can warehouse more inventory before it hurts, so its effective $\kappa$ is lower; a small or capital-constrained dealer has a high $\kappa$ and skews hard at the first sign of imbalance. When dealers' balance sheets get stressed — a margin call, a risk-limit breach, a regulator tightening capital rules — their effective risk aversion jumps and the whole market's liquidity thins, even with no change in volatility. The penalty $\kappa$ is, in one number, *volatility times horizon times risk appetite.*

#### Worked example: how doubling volatility doubles the skew

Take the same dealer from before — mid \$100, half-spread \$0.10 — now **long 100 shares**, and watch what happens to its skew as volatility changes. Suppose on a calm day the inventory penalty is $\kappa = \$0.01$ per share, so the reservation price is $100 − 100 \times 0.01 = \$99.00$ and the dealer's ask of \$99.10 sits a dime under fair value — a gentle nudge to attract a buyer.

Now volatility doubles. Because the penalty scales with variance, $\kappa$ roughly quadruples to \$0.04 per share. The reservation price drops to $100 − 100 \times 0.04 = \$96.00$, and the ask falls to \$96.10 — now a full **\$3.90 below the true mid.** The dealer is *desperate* to get flat: it is willing to sell its inventory almost four dollars under fair value rather than wear a position that has suddenly become four times as dangerous. Same 100 shares, same dealer, same stock — but a volatility spike turned a polite skew into a fire sale.

The intuition: inventory is not dangerous in itself; it is dangerous *in proportion to how much the asset can move and how little time you have left to escape it*, which is why dealers skew gently in calm markets and violently in fast ones.

## Putting it together: the spread is a stack of recovered costs

We now have all three pieces, so we can assemble the full spread and *see* that it is not a fee but a sum of costs the dealer must recover. The figure below stacks them across three market regimes.

![Spread decomposition bar chart showing order processing inventory and adverse selection components across calm normal and toxic regimes](/imgs/blogs/the-market-makers-game-inventory-the-spread-and-fear-of-the-informed-5.png)

The bottom slate block is the **order-processing cost** — a roughly fixed cent, the genuine fee. The blue block is the **inventory-risk cost** — it grows as the market gets more volatile, because a wiggle hurts more. The amber block on top is the **adverse-selection cost**, computed from the Glosten-Milgrom model, and it explodes as the flow turns informed.

#### Worked example: decomposing a 20-cent spread on a normal day

Take the middle bar. The asset is a \$20 stock that, on the next bit of news, could be worth \$20.40 or \$19.60 — a possible 40-cent move either way. On a normal day, suppose **20% of flow is informed.** Run Glosten-Milgrom on those numbers and the adverse-selection component of the spread is exactly **16 cents per share** (the model returns a \$0.16 spread; you can verify with `gt.glosten_milgrom(20.40, 19.60, 0.5, 0.20)`). Add the dealer's **inventory-risk cushion of 3 cents** and the **fixed processing cost of 1 cent**, and the quoted spread is 16 + 3 + 1 = **20 cents.**

Now flip to a calm tape where only **5% of flow is informed:** Glosten-Milgrom gives just **4 cents** of adverse selection, the inventory cushion shrinks to **1 cent** because the asset is barely moving, and with the **1-cent** processing floor the spread collapses to 4 + 1 + 1 = **6 cents.** And on a news day where **50% of flow is informed**, the adverse-selection component alone is **40 cents**; add a fat **6-cent** inventory cushion for the volatility and the **1-cent** floor and the dealer must quote a **47-cent** spread to break even.

The intuition: the *same dealer*, on the *same stock*, quotes 6 cents one hour and 47 cents the next, and nothing about its greed changed — only its fear did. The spread is a thermometer for toxicity and volatility, not a price list.

This decomposition is the heart of the post. Every time you see a spread, you can now mentally factor it: how much is the boring fee, how much is the dealer hedging the volatility, and how much is the dealer afraid of *you*?

## Reading the flow: toxic versus benign, and the quote that answers each

The dealer cannot directly observe who is informed — there is no badge. So it infers toxicity from the *footprint* of the flow and updates its quotes in real time. This is the live, adaptive part of the game, and it is where the best dealers earn their edge. The grid below lays out how the dealer reads the signals.

![Grid comparing benign and toxic flow across order size price behavior and the dealer quote response](/imgs/blogs/the-market-makers-game-inventory-the-spread-and-fear-of-the-informed-6.png)

Read it column by column. **Benign flow** (green, the flow the dealer wants) tends to be small, round-lot orders — 100 to 300 shares — that do not move the price; after the fill, the price drifts back, so the dealer keeps its spread cleanly. The right answer is a **tight quote**, both sides, all day. **Toxic flow** (red, the flow the dealer fears) tends to be large or aggressive — it sweeps through several price levels at once — and crucially, *after* the fill the price keeps moving in the same direction, because the trade was based on information. The dealer's fill was the first domino of a real move. The right answer is to **widen the spread or pull the quote entirely.**

The signal the dealer trusts most is the one in the middle row: **what the price does right after the fill.** If you sell to a dealer and the price immediately keeps falling, your sell was *informative* — you were ahead of a real move, and the dealer just caught a falling knife. Dealers measure this constantly; it is sometimes called *mark-out* (how the price "marks out" against the dealer a few seconds or minutes after the fill). A dealer with consistently bad mark-outs against a particular counterparty or order type will widen specifically against that flow. This is the dealer playing the repeated game: it cannot tell on any single trade whether you are informed, but over thousands of trades the *statistics* of your flow betray you, and the dealer prices accordingly.

#### Worked example: how one bad mark-out reprices the spread

Suppose a dealer is quoting a stock with a four-cent spread, mid \$50.00, and it just bought 100 shares from a seller at the bid of \$49.98. Over the next thirty seconds the price slides to \$49.90 — an eight-cent adverse mark-out. The dealer's inventory P&L on those 100 shares is 100 × (−\$0.08) = **−\$8**, against the **\$2** it would have earned had it completed a clean round-trip at the four-cent spread. That single fill was a 4-to-1 loser.

If the dealer's mark-out statistics say this kind of seller is informed even 30% of the time, the Glosten-Milgrom math we did earlier says the *break-even* spread is no longer four cents — it must widen toward the level where the gains from the benign 70% cover the losses to the informed 30%. So the dealer pulls its bid up only reluctantly and pushes its quoted spread out, say, to ten cents. The intuition: **a dealer does not need to know you are informed; it only needs its mark-outs to smell informed, and the spread you face widens for everyone who looks like you.**

This is the uncomfortable truth for aggressive traders. If your style — large, urgent, in-a-hurry orders right before moves — looks like informed flow, dealers will quote you worse *whether or not you actually have information.* You are paying the adverse-selection premium for the company you keep in the dealer's statistics.

## The dealer's loop: quote, get filled, re-mark, hedge, flatten

Market making is not a single decision; it is a loop that runs thousands of times a session. Stringing the pieces together, the dealer's actual algorithm looks like the pipeline below.

![Pipeline of the dealer loop quote get filled update inventory re-price reservation skew or hedge and flatten by close](/imgs/blogs/the-market-makers-game-inventory-the-spread-and-fear-of-the-informed-7.png)

Trace the loop. The dealer **quotes two sides** (99.90 / 100.10). It **gets filled** — say it buys 100 at 99.90. It **updates inventory** — now long 100. It **re-prices its reservation level** — $r = 100.00 − 1.00 = \$99.00$, dragging its quotes down. It then **skews or hedges**: it can lean its quotes (offer cheaper to attract a buyer) and/or **hedge** by selling a correlated instrument — a future, an ETF, a basket — to neutralize the directional risk while it waits for a natural buyer. And as the session winds down, it **flattens by the close**, so it carries zero overnight risk. Then the loop repeats from the top with new quotes.

Two pieces of that loop deserve a closer look.

**Hedging** is how a dealer separates the two fears. Inventory risk is *directional* — it is the risk that the price moves while the dealer is long or short. A dealer can offload that risk fast by trading something correlated: if it is long 100 shares of a stock it cannot immediately sell, it can short an equivalent dollar amount of the sector ETF or the index future, so a broad market drop no longer hurts its net book. What hedging *cannot* remove is the adverse-selection risk specific to *this* stock — if the seller knew this particular company is in trouble, the hedge against the index does nothing. So hedging neutralizes inventory risk but leaves the dealer still exposed to the very thing it most feared: a counterparty with stock-specific information. That residual is exactly the adverse-selection component of the spread. The options-market analog of this — a dealer dynamically hedging its directional (delta) risk while still wearing the volatility risk it cannot offload — is built out in [how an options market maker thinks](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade).

**Flattening into the close** falls out of the inventory model directly. Overnight, the market is shut but the world is not: earnings drop, wars start, central banks speak. A dealer holding inventory overnight wears all of that gap risk with no ability to skew or hedge out of it intraday. The reservation-price model says the inventory penalty $\kappa$ effectively spikes as the close approaches and the time-to-flatten shrinks, so the dealer skews harder and harder to dump its position. This is one reason the closing auction is the most strategic moment of the day — dealers and everyone else converge to get flat — and it connects to the broader [opening and closing auction](/blog/trading/game-theory/the-opening-and-closing-auction-the-most-strategic-moment-of-the-day) game.

## Where this connects to the deeper models

We have built the dealer's game from intuition and one clean model. Two deeper threads are worth flagging so you know where the rabbit holes go.

The **Kyle model** (Albert Kyle, 1985) attacks the same problem from the *informed trader's* side: how does a single trader with real information trade so as to hide inside the noise of uninformed flow, and how does the dealer's price impact ($\lambda$, lambda) emerge as the equilibrium response? In Kyle's world the dealer cannot tell the informed trader's orders from random noise, so it moves the price linearly with net order flow — and that price impact *is* the adverse-selection cost, viewed from the order-flow side rather than the single-trade side. If Glosten-Milgrom is "what does one buy order tell me," Kyle is "how should I move the whole price as net buying accumulates." They are two faces of the same fear. The full treatment is in [Kyle's model: how an informed trader hides in the noise](/blog/trading/game-theory/kyles-model-how-an-informed-trader-hides-in-the-noise).

The second thread is **competition.** Everything above assumed one dealer. In reality, many dealers compete to quote the same name, and competition compresses the spread toward the *break-even* level — the point where the spread just covers the three costs and no more. Glosten and Milgrom's result is a *competitive* spread for exactly this reason: if a dealer quoted wider than break-even, a rival would undercut it and steal the benign flow. So the spread you see is not the most a dealer could charge; it is the *least* it can charge and still survive the toxic flow. When dealers withdraw in a crisis, it is because at *any* spread they could quote, the toxic flow would still bleed them — so they rationally stop quoting, and liquidity evaporates. That is the dark side of the equilibrium.

#### Worked example: the competitive break-even spread

Let us make "break-even" concrete with a single round-trip's accounting. A dealer quotes a stock with mid \$50.00 and a four-cent spread, so it buys at \$49.98 and sells at \$50.02. Imagine 100 trades come through, and the dealer's mark-out statistics say 80 of them are benign and 20 are informed. On each *benign* trade the dealer captures the two-cent half-spread cleanly: 80 × \$0.02 × (say 100 shares) = 80 × \$2 = **\$160 earned.**

Now the informed 20. When an informed trader sells to the dealer at \$49.98, the price is genuinely heading lower — say it falls eight cents to \$49.90 before the dealer can react. The dealer loses the gap on each: 20 × \$0.08 × 100 shares = 20 × \$8 = **\$160 lost.** Net across all 100 trades: \$160 − \$160 = **zero.** The four-cent spread is the *break-even* spread — exactly wide enough that the money squeezed from the benign 80% pays back the money bled to the informed 20%, and no wider, because a competitor would undercut any fatter quote.

Now change one input: the informed move grows from eight cents to twelve cents (the news is bigger). The loss leg becomes 20 × \$0.12 × 100 = **\$240**, but the earn leg is still \$160 — the dealer is now losing \$80 over 100 trades at a four-cent spread. To get back to break-even it must widen the spread until the benign earnings rise to \$240, which means a six-cent spread (80 × \$0.03 × 100 = \$240). The intuition: the competitive spread is pinned to the *exact* level where the benign flow's payments offset the informed flow's damage, so any increase in either the informed fraction or the size of the informed move pushes the whole market's spread out for everyone.

## Common misconceptions

**"The spread is just the dealer's profit margin, like a shop's markup."** No. A shop's markup is pure margin over a known cost. The dealer's spread is a *loss-recovery* mechanism: a large chunk of it is handed straight back to informed traders who pick the dealer off, and another chunk pays for the inventory swings the dealer eats. The dealer's *net* edge after those losses is a sliver. In competitive, liquid names, dealers run on razor-thin per-trade margins and make money only on volume — exactly because competition has driven the spread down to the break-even level where the benign flow's payments just cover the toxic flow's damage.

**"A tighter spread always means a better, more liquid market."** Usually, but not always — and the exception is important. A spread can be *artificially* tight right before it gaps. If dealers have not yet realized the flow has turned toxic, they keep quoting a tight spread for a few seconds too long, and informed traders feast. The moment dealers wise up, the spread gaps wide. A persistently tight spread in a name where something big is brewing is not a sign of health; it can be a sign that the dealers have not repriced their fear yet, and you may be the benign flow about to be run over when they do.

**"If I get filled instantly at a great price, I won the trade."** This is the adverse-selection trap turned on you. A fast, full fill at a price you love means a dealer was *delighted* to take the other side — which should make you ask whether the dealer knows the price is about to move *your* way (lucky you) or whether you are simply the uninformed flow the dealer feeds on. A fill that is *hard* to get — where you have to chase the price, where liquidity keeps fading ahead of you — is often the fill that was actually *worth* getting, because the dealers were running from you. The discomfort of a hard fill and the comfort of an easy one are frequently backwards. This inversion is the core of the [fast-fill-is-bad-news](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news) argument.

**"Market makers want the price to move so they can make money."** Backwards. A pure market maker wants the price to *sit still* and the flow to be *balanced*, so it can earn the spread on clean round-trips with zero inventory risk. Price moves are the dealer's enemy — they are precisely what inventory risk and adverse selection are *about*. Dealers profit from *churn* (volume), not from *direction*. The traders who profit from direction are the informed ones on the other side, and they are the dealer's predators, not its friends.

**"Inventory skewing means the dealer is taking a directional view."** No — it means the opposite. Skewing exists to *remove* a directional position the dealer was forced into, not to express a view. When a long dealer quotes lower, it is not predicting the price will fall; it is trying to *get flat* by attracting buyers and repelling sellers. The skew is a steering correction back to neutral, not a bet. Confusing the two is the single most common misreading of dealer behavior.

## How it shows up in real markets

**The 2010 Flash Crash (May 6, 2010).** In a span of minutes, major US indices fell about 9% and recovered most of it, with some stocks trading at a penny and others at \$100,000. The mechanism was exactly the dealer's game breaking down. As a large, aggressive sell program hit the market, dealers' mark-outs turned catastrophically bad — every fill they took kept moving against them — so they did the rational thing the inventory and adverse-selection models predict: they widened spreads dramatically and many withdrew quotes entirely, some placing "stub quotes" at absurd prices like a penny just to nominally remain present. With the liquidity providers gone, there was no one to absorb inventory, and prices air-pocketed. It is the cleanest real-world demonstration that the spread is fear priced in: when fear goes to infinity, the spread does too, and the market simply stops.

**Earnings-announcement spread blowouts.** Watch any single stock in the seconds around its scheduled earnings release. The spread, which might be a penny or two during the day, routinely gaps to many cents or even dollars right before and after the print. Nothing about the dealer's greed changed at 4:00 p.m.; the *fraction of informed flow* spiked. Just before a known information event, the dealers reason that anyone trading aggressively might have a model, a leak, or faster news access — exactly the Glosten-Milgrom case where the informed fraction jumps — so the break-even spread jumps with it. Many dealers simply stop quoting tight markets across the announcement and resume once the information is public and the playing field is level again.

**The 2021 meme-stock episode (GameStop, late January 2021).** As retail and momentum flow flooded in and volatility exploded, spreads in the affected names widened enormously and some brokers restricted order types. From the dealer's lens, two things happened at once: inventory risk spiked because the names were moving 30%+ intraday (so $\kappa$ exploded), and the flow became hard to classify — a torrent of orders, some informed about the gamma-squeeze dynamics, some pure noise — so the adverse-selection cushion widened too. The eye-watering spreads were not a conspiracy against retail; they were the inventory-plus-adverse-selection model running at extreme inputs. The related dealer-flow mechanics — how options dealers' hedging amplified the move — are in [dealer gamma, charm and vanna](/blog/trading/options-volatility/dealer-gamma-charm-and-vanna-how-options-flows-move-the-spot).

**Treasury markets in March 2020.** Even the deepest, most liquid market on earth — US Treasuries — saw spreads widen sharply and dealer balance sheets seize up as the pandemic shock hit. Dealers, capacity-constrained and facing one-way selling from everyone needing cash at once, could not warehouse the inventory being dumped on them. The inventory side of the model dominated: with everyone selling and no offsetting buyers, dealers' inventory limits were hit, $\kappa$ effectively went to the moon, and the only rational response was to widen quotes and step back, until the Federal Reserve stepped in as the buyer of last resort. It is a reminder that "deep, liquid market" describes the *benign* state — the dealer's inventory constraint is always there, waiting for a one-sided flood.

**The everyday round-trip you never notice.** Most of the time, the model runs invisibly in your favor. When you buy 100 shares of a megacap like Apple and sell them a minute later, you pay a spread of a cent or less, because that flow is overwhelmingly benign, the name is so liquid that inventory clears in milliseconds, and dozens of dealers compete to quote it. You are the benign flow the whole system is designed to serve cheaply — and the penny you pay is the dealer's compensation for the *other* trades, the toxic ones, where it gets picked off. Your cheap fill is subsidized by the spread the informed traders force the dealer to charge everyone.

**Payment for order flow and the retail-wholesaler relationship.** When you trade through a commission-free retail broker, your order is often routed to a wholesale market maker that pays the broker for the privilege — *payment for order flow.* The dealer's game explains exactly why a wholesaler will *pay* to see your order: retail flow is, on average, the most benign flow in the market. A retiree buying 50 shares is almost never informed about the next tick, so the wholesaler can fill that order inside the public spread, capture a sliver, and face almost no adverse selection. The wholesaler segments the world precisely along the benign/toxic line from this post — it pays up for the demonstrably uninformed retail stream and quotes far more defensively against the anonymous institutional flow on the public exchange, which is where the informed traders hide. The arrangement is controversial, but mechanically it is the dealer's adverse-selection model deciding which flow is safe to pay for.

**Crypto market makers and the wider-by-default spread.** In many crypto venues, spreads are structurally wider than in mature equity markets, and the dealer's game explains why. Information is more asymmetric (insiders, protocol teams, and on-chain whales often know things before the tape), inventory is harder to hedge (fewer correlated instruments, fragmented venues), and volatility is higher — all three components of the spread inflate at once. A market maker quoting a thinly traded token is staring at a high informed fraction, a brutal $\kappa$, and weak hedging tools simultaneously, so the break-even spread is wide. It is the same three-term decomposition, with every term turned up.

## The playbook: how to play against the dealer

You are almost never the dealer; you are the flow on the other side. So the practical question is how to *play against* a player who has built its entire strategy around fearing you. Here is the playbook.

**Know which flow you are.** Be honest about whether your order is benign or toxic *from the dealer's seat.* If you are a long-term investor rebalancing, you are benign — the dealer is happy to fill you tight, and you should accept that the spread is a small, fair price for immediacy. If you are trading on a real, time-sensitive edge, you are toxic, and you should expect the dealer to fade you, quote you wider, and pull liquidity as you push — so size and slice accordingly. Misjudging which one you are is how traders bleed: a benign trader who panics and trades like an informed one pays the toxic premium for nothing.

**Your edge against the dealer is patience and disguise, not aggression.** The dealer's whole defense is reading your footprint. The more your order looks like benign flow — small, passive, patient, spread across time — the tighter the quotes you get and the less price impact you pay. This is the entire logic of execution algorithms (VWAP, TWAP, iceberg orders) that chop a large informed order into a stream that *looks* benign so the dealer does not widen against it. If you must move size, your job is to *not look informed*, which is the dealer-facing version of the [execution-as-a-game](/blog/trading/game-theory/execution-as-a-game-vwap-twap-and-hiding-from-predators) problem.

**Read the spread as a fear gauge, not a fee.** When the spread on a name suddenly widens with no obvious news, the dealers are telling you they smell toxicity or volatility — someone is trading like they know something, or the asset has gotten dangerous to warehouse. That is information. A widening spread ahead of an event is the market pricing in a higher informed fraction; treat it as a warning that the playing field is tilting, not as a random cost. Conversely, an unusually tight spread in a name where something is brewing can mean the dealers have not repriced yet — which can be an opportunity if *you* are the informed one, or a trap if you are not.

**Distrust the easy fill; respect the hard one.** Build the adverse-selection inversion into your reflexes. A fill that came instantly, fully, at a price you loved, is a fill a dealer wanted to give you — interrogate why. A fill you had to fight for, chasing fading liquidity, is often the one that was actually worth getting. This single reflex — *an easy fill is a question, not a reward* — separates traders who understand the game from those who think they "beat the spread."

**Your invalidation and your sizing.** The dealer's game tells you where your edge is real and where it is an illusion. If your strategy only works at the *quoted* spread and dies once you account for the *realized* spread on size (the price moving away as you trade), you do not have an edge — you have a backtest that ignored the dealer's response. Size such that your own footprint does not flip the dealers from quoting you benign to fading you toxic; the moment you are big enough to move the dealer's mark-out, your effective spread balloons and your edge can vanish. The invalidation is simple: if your fills consistently mark out *against* you — if the price keeps moving your way only *after* you finish buying — you are the toxic flow that *was* informed, congratulations; if it keeps moving against you *right after* every fill, you are the benign flow being run over, and you should slow down, pay the spread you are actually paying, and re-examine whether the edge survives it.

The deepest takeaway is the one the whole series is built on: the spread is not a number the market charges you, it is a *message* a specific, rational opponent is sending about how much it fears the person on your side of the trade. Learn to read that message and you stop paying the spread blindly and start using it as a window into who else is in the game.

This is educational, not advice. The models here are simplifications; real dealer behavior layers in regulation, fee tiers, latency, and a dozen frictions we have set aside to keep the game legible. The point is the *logic*, not a recipe to trade on.

## Further reading & cross-links

- [The bid-ask spread as an adverse-selection game (Glosten-Milgrom)](/blog/trading/game-theory/the-bid-ask-spread-as-an-adverse-selection-game-glosten-milgrom) — the full derivation of the adverse-selection spread we used as the dealer's thermometer here.
- [Kyle's model: how an informed trader hides in the noise](/blog/trading/game-theory/kyles-model-how-an-informed-trader-hides-in-the-noise) — the same fear from the informed trader's side, where price impact ($\lambda$) is the equilibrium response.
- [Adverse selection and the winner's curse: why a fast fill is bad news](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news) — the inversion that should change how you read your own fills.
- [How an options market maker thinks: the other side of your trade](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade) — the same inventory-and-adverse-selection game in options, where the residual risk is volatility rather than direction.

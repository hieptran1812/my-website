---
title: "Roll Yield and Why Long-Only Commodity ETFs Bleed"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why you can be right that oil went up and still lose money owning an oil ETF: the monthly roll, the contango bleed, and how to read a commodity vehicle's curve before you buy the brochure."
tags: ["commodities", "roll-yield", "contango", "backwardation", "commodity-etf", "uso", "futures", "carry", "natural-gas", "crude-oil", "long-only", "total-return"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — You cannot put a barrel of oil in your brokerage account, so a commodity fund holds **futures** and must **roll** them — sell the expiring front month, buy the next — every single month. That mechanical act, not the oil price, is what quietly decides your return.
>
> - In **contango** (the normal state for oil and especially natural gas) the next month is dearer than the one you are selling, so each roll is "sell low, buy high" — a **negative roll yield** that bleeds the fund even when spot is flat.
> - Total return splits into three pieces: **spot return + roll yield + collateral yield**. The headline price you watch on the news is only the first piece; the roll is the one that surprises people.
> - This is why "oil went up 20% but my oil ETF went *down*" is not a glitch — it is the roll. The famous USO fund was shredded by it in 2009 and almost destroyed by super-contango in April 2020.
> - The one habit to take away: **read the curve before you read the brochure.** A steep-contango market can cost a long-only holder on the order of **10-15% a year**; a backwardated one can pay you a similar amount.

In the spring of 2009, with crude oil having crashed from \$147 to the low \$30s the previous winter, a wave of ordinary investors decided oil was a screaming buy. The instrument they reached for was obvious: the United States Oil Fund, ticker USO, the largest and most heavily advertised oil exchange-traded fund on the market. It promised, in plain language, to track the price of West Texas Intermediate crude. Over the next twelve months the spot price of WTI did exactly what those investors hoped — it roughly *doubled*, climbing back from the low \$30s toward \$80. A perfect, correct, well-timed call on the direction of oil.

And USO? Over that same year it rose by a fraction of that. The fund's investors had been *right about oil* and still captured almost none of the move. Some who bought and held through the worst of it actually finished *down* while spot oil was *up*. They had nailed the price and been quietly robbed by something they had never heard of and the brochure never mentioned: the **roll**.

This post is about that robbery — why it happens, exactly how much it costs, and how to spot it before you put a dollar to work. It is the single most important thing a retail investor in commodities gets wrong, and once you see it you cannot unsee it. We will build the mechanism from zero, decompose a commodity fund's return into its three honest pieces, walk through the USO disasters of 2009 and the near-death of April 2020, and finish with the one habit that protects you: reading the shape of the forward curve before you read the marketing.

![The monthly roll where contango bleeds the fund and backwardation pays it](/imgs/blogs/roll-yield-and-why-long-only-commodity-etfs-bleed-1.png)

## Foundations: why a fund cannot just hold the barrel

Start with the most basic and most overlooked fact in all of commodity investing. **You cannot hold the commodity.**

When you buy a share of Apple, the fund behind your ETF buys actual Apple shares and holds them. They sit in a vault as electronic entries and do nothing but exist. There is no cost to holding a share of stock for a year; if anything it pays you a dividend. A stock is a *financial* asset — weightless, costless to store, infinitely patient.

A barrel of oil is none of those things. It is a physical, perishable-in-practice, dangerous, bulky thing that has to live in a steel tank somewhere, insured, guarded, and financed. A fund that wanted to "hold oil" the way it holds Apple would have to lease tank farms at Cushing, Oklahoma, hire people, buy insurance, and pay for the storage every single day. No retail fund does this, because the costs would be ruinous and the logistics absurd. The same is true, even more so, for natural gas (which you cannot store at all without enormous pressurized or refrigerated facilities), for cattle (which eat), for electricity (which cannot be stored), and for most of the complex.

So a commodity fund does the only practical thing: instead of holding the physical barrel, it holds a **futures contract** — a standardized paper promise to buy a barrel at a fixed price on a fixed future date. We covered what a futures contract actually is in [Spot vs futures](/blog/trading/commodities/spot-vs-futures-the-two-prices-of-the-same-barrel); here all you need is the headline. A futures contract lets you get the economic exposure to oil's price *without ever touching a barrel*. The fund buys the contract, posts a little cash as margin, and its profit or loss tracks the change in that contract's price. Clean, liquid, weightless. That is the whole reason commodity ETFs exist.

But here is the catch that the whole post hangs on. A futures contract **expires.** The front-month WTI contract delivers physical oil at Cushing in a specific month, and as that delivery date approaches, the fund faces a hard deadline. If it still holds the contract at expiry, it is legally obligated to take delivery of actual barrels of oil — the very thing it cannot store. So the fund must get out *before* expiry. And because it still wants to be long oil next month, it does not just sell and go to cash; it sells the expiring contract and *simultaneously buys the next month's contract*, pushing its exposure forward in time. That swap — sell the expiring front month, buy the next one — is called **rolling the position**, and it happens every single contract cycle, month after month, forever, for as long as the fund exists.

The roll is not optional. It is not a strategy choice. It is a structural necessity baked into the fact that you cannot hold the physical thing. And it is where all the money quietly leaks out.

It is worth pausing on *when* the roll happens, because the timing is itself a source of cost that the brochures never mention. The major front-month oil funds do not roll all at once on the last possible day — that would expose them to a single bad price and to predatory traders. Instead they roll over a published window, typically several business days before the front contract's expiry, selling a fixed fraction of the position each day. This sounds prudent, and it is, but it has a perverse consequence: the roll schedule is *public*. Other traders know exactly when a giant fund must sell the expiring contract and buy the next, so they position ahead of it — buying the next-month contract before the fund does, then selling it back to the fund a few cents dearer. That front-running is a small, recurring tax on top of the curve-shape cost, and for the largest funds it has been estimated to subtract a further fraction of a percent per roll. The fund is not just paying the slope of the curve; it is paying the cost of being a predictable elephant in a market full of nimble traders who can see exactly where it must step next.

There is one more operational reality to absorb. Because the fund holds futures rather than the physical, it does not spend the full value of its assets buying oil. A futures contract controls a large notional value of oil for a small margin deposit — often well under 10% of the contract's face value. So a fund with, say, \$1 billion of assets and \$1 billion of oil exposure might have only \$100 million tied up as margin at the exchange, leaving \$900 million sitting in cash. What the fund does with that idle \$900 million turns out to matter enormously, and it is the third piece of the return puzzle we build below. For now, hold two facts: the roll is mechanical and recurring, and most of the fund's money is *not* spent on oil at all.

### The two prices in every roll

To see why the roll costs money, you only need two prices: the price of the contract you are *selling* (the expiring front month) and the price of the contract you are *buying* (the next month out). Those two prices are rarely equal, and the gap between them is the whole story.

Recall from [The forward curve](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities) that a commodity does not have one price — it has a *menu* of prices, one for each delivery month, and connecting those dots gives you the **forward curve**. The shape of that curve has a name depending on its slope, and that name decides whether your roll makes money or loses it. We built the two shapes in depth in [Contango vs backwardation](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means); here is the one-line refresher.

**Contango** is an upward-sloping curve: each later delivery month costs *more* than the nearer one. The barrel you are selling (front month) is *cheaper* than the barrel you are buying (next month).

**Backwardation** is a downward-sloping curve: the prompt month is the *dearest*, and prices fall the further out you look. The barrel you are selling (front month) is *more expensive* than the barrel you are buying (next month).

Now overlay the roll on each shape and the punchline writes itself. In contango, you **sell the cheap front month and buy the dearer next month** — you sell low and buy high, every roll, by construction. In backwardation, you **sell the dear front month and buy the cheaper next month** — you sell high and buy low, every roll. The first bleeds you; the second pays you. That is roll yield, and the cover figure above lays the two cases side by side.

Notice what makes this so insidious: the loss is not a "fee" line item anywhere. No statement shows a charge labeled "roll cost." It hides inside the transaction itself — you simply own a slightly smaller economic position after each roll than before, because the dollars that bought your old front-month barrels now buy fewer of the dearer next-month barrels. The fund's share price absorbs it silently, month after month, and the only way to see it is to compare the fund's path to the spot price's path, which almost no retail investor ever does. A bank that charged you 2% a month would be a scandal; the curve charges exactly that in contango and nobody calls customer service, because there is no one to call. The carry is not a fee imposed by the fund manager — it is the price of physics, the cost of not being able to hold the barrel, paid invisibly and continuously through the simple mechanics of the monthly roll.

#### Worked example: the monthly roll-cost calculation

Make it concrete with numbers. Suppose it is roll week. The expiring front-month WTI contract is trading at \$72.00 a barrel. The next month's contract — the one the fund must buy to stay long — is trading at \$73.40. The curve is in mild contango: later is dearer.

The fund sells its front-month barrels at \$72.00 and buys the same notional amount of next-month barrels at \$73.40. To keep its dollar exposure the same, it can now afford *fewer* barrels, because each new barrel costs more. The fractional loss baked into this one roll is:

```
roll cost = (next month - front month) / front month
          = (73.40 - 72.00) / 72.00
          = 1.40 / 72.00
          = 0.0194  ->  about 1.9% per roll
```

That 1.9% is gone the instant the roll happens, regardless of what oil does next. It is not a trading mistake; it is the mechanical price of staying long without owning the physical barrel. Now annualize it: a 1.9% drag *every month* compounds to roughly 20-25% a year of headwind. **The intuition: in contango, the act of staying invested costs you a fixed percentage every month, and that fixed percentage is exactly how much dearer the next contract is than the one you are forced to sell.**

That single calculation — about two percent a month, twenty-plus percent a year, on a curve that is only *mildly* upward-sloping — is the entire reason long-only commodity funds have a reputation for disappointing the people who buy them.

## Decomposing the return: spot, roll, and collateral

The cleanest way to understand a commodity fund is to stop thinking of it as "a bet on the oil price" and start thinking of it as a machine that earns three separate things, which add up to your total return. Get this decomposition into your head and you will never again be surprised by a commodity ETF.

![Total return decomposed into spot return plus roll yield plus collateral yield](/imgs/blogs/roll-yield-and-why-long-only-commodity-etfs-bleed-4.png)

**Piece one: the spot return.** This is the change in the price of the commodity itself — the headline number on the news. If oil goes from \$72 to \$79, the spot return is roughly +10%. This is the piece everyone watches, and it is the piece people *assume* is the whole story. It is not even close.

**Piece two: the roll yield.** This is the gain or loss from rolling futures, which we just built. In contango it is negative (the bleed); in backwardation it is positive (the carry). It can be small, or it can be enormous — for natural gas in steep contango it can be the *dominant* term, swamping the spot return entirely. This is the piece people have never heard of, and it is the piece that does the damage.

**Piece three: the collateral yield.** Here is a piece that actually works in your favor and that almost nobody mentions. Because a futures position only requires posting a small margin, the fund is not spending all its cash on oil — most of the money sits in the bank as collateral. The fund parks that idle cash in safe, short-term instruments, typically U.S. Treasury bills. The interest those bills pay is real return that gets added back. When T-bills yield 5%, that is a meaningful 5% tailwind on the *full* notional of the fund. When they yield near zero, this piece vanishes.

So the honest formula for a long-only commodity fund is:

```
total return = spot return + roll yield + collateral yield
```

When people say "commodities returned X% over the decade," they are usually quoting the *total return* of an index, which already blends all three. And when an individual investor is baffled that their oil ETF underperformed the oil price, the answer is always hiding in the second term. The spot went up; the roll dragged it down; the net was disappointing. The brochure showed you the spot. The roll did the rest in the dark.

It helps to contrast this with the assets you already understand. A **stock** earns you two things: price appreciation and dividends, both of which can be positive, and neither of which charges you rent for holding. A **bond** earns you a coupon and pays you back at maturity; time is your friend, not your enemy. Hold either one and do nothing, and time generally works *for* you. A **commodity future** is structurally different and almost uniquely hostile in this regard: hold it through a contango curve and do nothing, and time works *against* you, draining value every roll even if the underlying never moves. There is no coupon, no dividend, no patient accrual — only the spot move you might or might not capture, the roll that may help or (usually) hurt, and the collateral yield on the cash you parked. This is why "buy and hold" — the soundest advice in all of stock investing — can be precisely the *wrong* instinct in commodities. The thing you are holding is not a patient store of value; it is a depreciating claim on a barrel you cannot keep, re-purchased at a markup every month.

That distinction is the deepest reason this series insists that commodities are *consumption and industrial assets, not monetary ones.* Gold, the great monetary commodity, is cheap to store and pays no one to hold the surplus, so a gold ETF can simply hold bars in a vault with no roll and no bleed — which is exactly why gold gets its own treatment and behaves like a currency rather than a barrel. Oil and gas, the great consumption commodities, are expensive to store and perishable in their economics, so a fund must rent its exposure month to month and pay the carry. The roll is not an accounting quirk; it is the financial shadow cast by the physical fact of storage. The barrel must live in a tank, the tank costs money, and that cost shows up in your return as the roll.

#### Worked example: the collateral-yield offset

Suppose you hold a broad long-only commodity fund for a year. The basket's spot prices rise 8%. But the curves across the basket are in mild contango, costing you a roll yield of −6% over the year. So far the fund's "commodity" return is +8% − 6% = +2%, a meager result for an 8% move in the underlying. But the year's T-bills yielded 5%, and the fund earns that on its collateral. The full return is:

```
total = spot + roll + collateral
      = (+8%) + (-6%) + (+5%)
      = +7%
```

The collateral piece nearly *tripled* the lousy +2% commodity return into a respectable +7%. **The intuition: in a high-interest-rate world the T-bill yield on the fund's idle cash is a quiet, reliable tailwind that can offset a meaningful chunk of the contango bleed — which is exactly why the same commodity fund can feel very different depending on whether rates are at zero or at five percent.** This is not a footnote; in the high-rate years of the early 1980s and again in 2023-2024, collateral yield was a major share of what commodity index investors actually earned.

## The money chart: same spot, falling fund

Words make the bleed sound theoretical. A chart makes it undeniable. Below is the single most important picture in this post: a spot price index that ends the period roughly where it started, plotted against the long-only total return of a fund holding that same commodity through a contango curve.

![Spot price index flat while the long-only fund total return falls steadily over five years](/imgs/blogs/roll-yield-and-why-long-only-commodity-etfs-bleed-2.png)

Read the gray line first. The spot price wobbles up and down over five years but ends almost exactly at 100 where it began — net, the commodity went *nowhere*. A casual investor watching the news would shrug: "Oil was flat, my oil fund should be roughly flat too." That is the trap.

Now read the red line. The fund holding that same flat commodity, rolling through contango month after month, has lost about 30% of its value. The shaded gap between the two lines is the bleed, and notice how it *widens every year*. This is the visual signature of compounding negative roll yield: a fixed monthly drag, applied over and over, on a base that the spot price never lifts. The spot was flat; the fund was destroyed; the gap is the roll.

This is the picture behind every "but oil was up and my ETF was down" complaint. The headline price and the investor's actual return are two different numbers, separated by a term the investor was never shown.

#### Worked example: the spot-up-but-ETF-down paradox

Let us reproduce the exact paradox that snared the 2009 USO buyers, with round numbers. You hold a front-month oil ETF for a year. Over that year:

- Spot oil rises a genuine **+20%** — you were right about the direction.
- But the curve sat in steep contango the whole time, averaging about a 2% bleed per month, which compounds to a roll yield of roughly **−25%** over the year.
- Rates were near zero, so collateral yield is about **0%**.

Your total return is:

```
total = spot + roll + collateral
      = (+20%) + (-25%) + (0%)
      = -5%
```

You called a 20% rally correctly and still *lost 5%* of your money. **The intuition: when the contango is steeper than the spot rally, the roll can flip a winning directional call into a losing position — being right about the price is necessary but not sufficient, because the curve is also charging you rent the entire time you hold.** This is not a hypothetical; it is approximately what happened to front-month oil ETF holders in 2009, when steep post-crash contango devoured a doubling in spot.

## Roll yield by regime: the sign of the slope decides

The roll yield is not a fixed villain. It is entirely determined by the *shape* of the curve, and the shape changes over time and varies wildly across commodities. The chart below puts an illustrative number on the carry for each regime, from steep contango through flat to steep backwardation.

![Roll yield by curve regime as a horizontal bar chart, negative in contango and positive in backwardation](/imgs/blogs/roll-yield-and-why-long-only-commodity-etfs-bleed-3.png)

Read it top to bottom. **Steep contango** — the chronic state of natural gas — can cost a long-only holder on the order of 12% a year. **Mild contango** still bleeds a few percent. **Flat** is the neutral case where the roll roughly washes out. Then the picture flips: **mild backwardation** *pays* the holder a few percent, and **steep backwardation** — the signature of a genuinely tight market like oil during a supply scare — can hand a long holder double digits *for free*, every year, on top of any spot move.

This is the deep point that separates a naive commodity investor from a literate one. The roll yield is not noise around the spot return — it is a *structural* return source with a predictable sign. If you know the shape of the curve, you know the sign of your carry before you have placed a single bet on direction. A long-only position in a backwardated market has a tailwind at its back; the same position in a contango market is rolling a boulder uphill. We treat the *positive* side of this — backwardation as a genuine, harvestable return premium — at length in [Backwardation as a structural return source](/blog/trading/commodities/backwardation-as-a-structural-return-source-the-carry-of-commodities). Here our concern is the negative side, the one that costs unsuspecting investors real money.

The asymmetry matters for vehicle selection. Markets that are *usually* backwardated (historically, crude oil and some metals during tight periods) have rewarded long-only holders over the long run; markets that are *usually* contango (natural gas above all, but also many agricultural products with seasonal storage) have punished them brutally. The shape is not random — it reflects the cost of carry and the convenience yield, which we unpack in [Convenience yield and the cost of carry](/blog/trading/commodities/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape). For the investor, the practical lesson is simply: find out which regime your chosen commodity tends to live in *before* you commit.

A useful way to read the regime is to watch the *first calendar spread* — the gap between the front month and the next month — because that one number is the cleanest local reading of the roll you are about to pay. A widening contango spread is a flashing red light: the bleed is getting steeper, the market is getting more glutted, and your long-only carry is getting worse. A spread that flips from contango to backwardation is the opposite signal — the market is tightening, the roll has turned from a headwind into a tailwind, and the carry now favors the holder. Practitioners track this prompt spread daily for exactly this reason; it is the single most informative number on the curve for anyone whose return depends on the roll. The level of oil tells you what the news cares about; the prompt spread tells you what your wallet should care about.

### The same mechanic, hiding in volatility products

Before we leave the regime question, it is worth seeing that the roll is not unique to barrels and bushels — it shows up, in identical form, anywhere a fund must hold a futures contract it cannot physically possess. The cleanest non-commodity example is **volatility**. You cannot hold "the VIX," the market's fear index, any more than you can hold a barrel; it is a calculated number, not a tradable thing. So the popular volatility ETPs (products like VXX) hold *VIX futures* and roll them — and the VIX futures curve is in **contango** the vast majority of the time, because fear is usually expected to mean-revert higher from a calm present. The result is exactly the bleed we have been describing: long-volatility products lose value relentlessly through the roll, so much so that several have lost over 99% of their value across their histories and survive only through repeated reverse splits. The mechanism is letter-for-letter the same as the oil ETF: hold a futures contract on something un-storable, roll it through a contango curve, and bleed. We trace that parallel in detail in [The term structure of volatility](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve). The lesson generalizes: *any* long-only futures-based vehicle on an un-storable underlying is a roll-yield machine, and the sign of the roll is the sign of that underlying's curve. Once you internalize the roll, you stop being surprised by any of them.

## Why oil ETFs bleed: rolling up the staircase

Let us slow down and watch the bleed happen on the actual curve, one roll at a time. Below is an illustrative contango forward curve for oil: a prompt price of \$72 climbing to about \$77 a year out. The fund lives on the left edge of this curve, and every month it must take one step up the slope.

![Contango forward curve where the fund sells the cheap front month and buys the dearer next month](/imgs/blogs/roll-yield-and-why-long-only-commodity-etfs-bleed-5.png)

Trace the green dot and the red dot. The green dot is where the fund *sells*: the expiring front month at \$72.00, the cheapest point on the curve. The red dot is where it *buys*: the next month at \$73.40, a step up the slope. The fund has just paid \$1.40 a barrel — about 1.9% — to move its exposure one month forward in time, and it has gained nothing for it. The oil is the same oil; only the calendar moved.

Now picture the next month. The whole curve may have shifted, but its *shape* — the upward slope — tends to persist in a structurally contango market. So next month the fund again sells near the bottom and buys a step up, paying the toll again. And again the month after. The fund is permanently climbing a down-escalator: every step up the curve is a step *down* in value relative to the spot price, because the price it paid for the deferred barrel converges back toward the (lower) spot as that contract becomes the new prompt.

That convergence is the engine of the bleed and worth stating precisely. The fund buys the next-month contract at \$73.40. As that contract ages and becomes the prompt, its price has to converge toward spot — and if spot is unchanged at around \$72, the fund's \$73.40 barrel "rolls down" to \$72, a \$1.40 loss realized purely through the passage of time. The fund bought high on the curve and watched the contract slide down to spot. Multiply by twelve rolls a year and you have the staircase that quietly carries the fund downstairs while the spot price stands still.

#### Worked example: a twelve-month compounding bleed

Watch \$100 of investment decay over a year in a steady contango, holding spot flat so we isolate the roll. Assume a 2% bleed per monthly roll and no offsetting collateral yield. Each month the value is multiplied by 0.98:

```
month  0:  100.00
month  1:  100.00 x 0.98 = 98.00
month  2:   98.00 x 0.98 = 96.04
month  3:   96.04 x 0.98 = 94.12
month  6:  about 88.6
month  9:  about 83.4
month 12:  100 x (0.98)^12 = 78.5
```

After a full year of a 2%-per-month roll cost — with the spot price *exactly flat* — your \$100 has become about \$78.50. You lost over a fifth of your money owning a thing whose price did not move. **The intuition: the contango bleed is not a one-time fee but a compounding monthly tax, so even a moderate per-roll cost grinds a long-only position down by roughly 20% a year, which is why these funds are corrosive to *hold* even when nothing dramatic happens to the commodity.** And remember, a 2% monthly bleed corresponds to only a *mild* contango — about \$1.40 on \$72. Steep contango is far worse, which brings us to the cruelest market of all.

## The natural-gas case: where the roll is brutal

If oil in contango is corrosive, natural gas is in a different category of pain. Henry Hub gas lives in **chronic, steep contango** for a structural reason: gas is intensely seasonal — demand spikes in winter for heating — and the only way to bridge summer surplus to winter need is storage, which is expensive and limited. The result is a forward curve that almost always slopes steeply upward from the cheap, glutted summer prompt to the dear, scarce winter months, and then resets and does it again. A long-only gas fund rolls up that steep slope every month, perpetually.

![Henry Hub natural gas spot price as a step chart from 2005 to 2025 showing the chronic contango market](/imgs/blogs/roll-yield-and-why-long-only-commodity-etfs-bleed-6.png)

The chart shows the *spot* price of Henry Hub gas from 2005 to 2025 — and even the spot tells a grim story, collapsing from around \$8/MMBtu in the shale-glut era to barely \$2 in the 2020s, with a brief 2022 supply spike. But the spot decline understates the disaster for a long-only gas ETF, because on *top* of that falling spot the holder paid the steep contango roll month after month. The most notorious natural-gas ETFs lost not 30% or 50% but well over **90%** of their value across these years, devastated by the combination of a falling spot price *and* a relentless contango bleed. Even adjusting for periodic reverse splits that mask the raw share-price collapse, the wealth destruction for buy-and-hold investors was close to total.

Why is gas so much worse than oil? The answer is storage economics taken to an extreme. Gas demand is violently seasonal — a cold winter can double consumption versus a mild shoulder season — but gas production is roughly steady year-round. The only bridge between flat supply and spiky demand is storage: injecting gas into underground caverns in summer and withdrawing it in winter. That storage is genuinely scarce and expensive, so the market must pay a large premium for winter-delivery gas over summer-delivery gas to incentivize injecting and holding it. The result is a curve that does not just slope up — it slopes up *steeply and repeatedly*, sawtoothing higher into every winter and resetting every spring. A long-only gas fund rolling the front month is perpetually selling cheap near-term gas and buying dear winter gas, paying the full seasonal premium over and over.

There is a second cruelty unique to gas. Because the front of the gas curve can swing so violently — a single forecast of a polar vortex can spike the prompt month — the prompt calendar spread is one of the most dangerous trades in all of commodities, nicknamed the "widow-maker" for the hedge funds it has destroyed. For the long-only ETF holder, the practical effect is that the gas roll is not only steep but *volatile*: in some months the roll cost is enormous, occasionally it briefly reverses, and the path is nerve-shredding. The average roll cost is brutal; the variance around it is worse.

This is the cautionary tale that anchors the whole subject. Natural gas is the market where a retail investor is most likely to think "gas prices are so low, surely they will bounce, I will buy the gas ETF and wait" — and it is precisely the market where waiting is most lethal, because the steep contango charges enormous rent for every month you hold. The spot might even rise, and you can *still* bleed out, because the roll cost in steep gas contango regularly exceeds 30-40% a year. We dig into why gas is structurally three seasonal markets in one in [Natural gas: Henry Hub, TTF, JKM](/blog/trading/commodities/natural-gas-henry-hub-ttf-jkm-and-the-most-seasonal-market); for the investor, the headline is blunt: a long-only natural-gas ETF is one of the most reliable wealth-destruction machines retail finance has ever produced, and the roll is the reason.

## The USO disaster: 2009, and the near-death of April 2020

No discussion of roll yield is complete without the story of the United States Oil Fund, the instrument that taught a generation of investors what contango does — the hard way, twice.

**2009.** As we opened, USO was designed to hold front-month WTI futures and roll them monthly. When oil crashed in late 2008, the curve flipped into steep contango — a glut had filled the tanks, the prompt was beaten down, the deferred months held up. At one point in early 2009 the gap between the front month and the contract a year out exceeded \$10 a barrel on a roughly \$40 prompt, a contango of over 25% across the strip. Through 2009, spot oil roughly doubled, but USO holders rolled up that steep contango curve every month and captured only a fraction of the rally. The fund's lag versus spot oil over 2009 ran to *double digits* of percentage points. For investors who had correctly called the bottom in oil, it was a brutal lesson: you can be right about the commodity and wrong about the vehicle, and the vehicle is what you actually own.

The 2009 episode was also where the front-running cost showed itself starkly. USO had grown so large relative to the front-month contract that its monthly roll was a market-moving event. Traders learned the fund's published roll schedule and bought the next-month contract ahead of USO, then sold it to the fund at a markup — adding a self-inflicted cost on top of the curve's natural contango. The fund's own size, in a sense, deepened the bleed it suffered: a giant, predictable, mechanical roller in a market full of opportunists is a turkey at a wolf convention. This is why the largest single-commodity funds eventually diversified their roll dates and contracts, and why an attentive investor checks not just a fund's strategy but its *size relative to the contract it trades*.

**April 2020.** Then came the day that nearly broke the fund entirely. In the spring of 2020, COVID lockdowns vaporized oil demand at the exact moment a Saudi-Russia price war flooded the market with supply. Every storage tank on Earth filled up. With nowhere to put physical barrels, the prompt WTI contract did something no one alive had seen: on April 20, 2020, the expiring May contract settled at **minus \$37.63** — sellers paid buyers to take the oil off their hands. Meanwhile, a barrel for delivery a year out still fetched over \$35. The curve was in *super-contango*: an almost vertical upward slope from a negative or near-zero prompt to a deferred price many times higher.

![Timeline of the April 2020 super-contango forcing USO to restructure down the curve](/imgs/blogs/roll-yield-and-why-long-only-commodity-etfs-bleed-7.png)

For USO, super-contango was an existential threat, as the timeline above traces. Rolling the front month in that environment meant selling a near-worthless prompt contract and buying a next-month contract that was vastly more expensive — locking in a catastrophic, near-total roll loss every cycle. The fund was so large that its own rolling activity was distorting the market it traded. Regulators and its broker imposed position limits. To survive, USO was **forced to restructure**: it abandoned the pure front-month strategy and spread its holdings *down the curve*, into a mix of contracts two through twelve months out, to escape the worst of the prompt-month super-contango. It changed what it *was* — mid-crisis, without asking shareholders — because the front-month roll had become un-survivable.

The scar was permanent. Investors who piled into USO in April 2020 betting on an oil rebound — and millions did, treating it like a lottery ticket on cheap oil — found that even as spot oil recovered strongly over the following year, their fund lagged badly, hobbled by the super-contango it had rolled through and the defensive restructuring it had been forced into. They had bought the cheapest oil in history and still trailed the rebound. The roll, once again, had done its quiet work.

#### Worked example: the super-contango roll loss

Put April-2020 numbers to the roll. Suppose the fund must roll out of an expiring prompt contract trading near \$20 (already recovered from the negative print, but still glutted) and into a next-relevant contract at \$26 — a super-contango gap. The per-roll cost is:

```
roll cost = (26 - 20) / 20
          = 6 / 20
          = 0.30  ->  30% in a single roll
```

A *thirty percent* loss in one roll, before oil does anything at all. Compare that to the ~1.9% of a normal contango roll. **The intuition: super-contango is the contango bleed turned up to a level that can destroy a fund in a handful of cycles, which is exactly why USO had to abandon its front-month mandate to survive — at a 30% per-roll loss, no fund holding the prompt could outlast the glut.** This is the most extreme illustration of the central thesis: in commodities, *the shape of the curve can matter infinitely more than the level of the price.*

## Common misconceptions

The roll generates more confident wrong beliefs than almost any topic in finance. Here are the five that cost people the most, each corrected with a number.

**Misconception 1: "An oil ETF tracks the price of oil."** No — a front-month oil ETF tracks the *front-month futures price minus the cumulative roll cost*. Over 2009, spot WTI roughly doubled while USO rose a fraction of that; the gap was the contango roll. The brochure says "designed to track the price of oil," and in a flat-curve world it nearly does, but in contango the tracking error compounds into a chasm. The fund tracks the *futures-roll experience*, not the spot price you see on the news.

**Misconception 2: "If the price goes up, I make money."** Not necessarily. As the worked example showed, a +20% spot move paired with a −25% contango roll leaves you down 5%. The sign of your return is `spot + roll + collateral`, and the roll term can be large enough to flip a correct directional call into a loss. Direction is necessary, not sufficient.

**Misconception 3: "The bleed only matters for short-term traders."** Backwards. The bleed *compounds*, so it punishes *long-term holders* most. A 2%-per-month contango costs you about 20% over a year and over 35% over two years. A day-trader holding for hours barely touches the roll; the buy-and-hold investor "waiting for the bounce" eats the full compounding tax. The roll is a holding cost, and holding is exactly what most retail investors do.

**Misconception 4: "Natural gas is cheap, so the gas ETF must be cheap too."** This conflates the commodity with the vehicle. Henry Hub spot fell from ~\$8 to ~\$2 over fifteen years, but a long-only gas ETF lost over 90%, because the steep, chronic contango roll piled on top of the spot decline. A "cheap" commodity in steep contango is one of the most expensive things you can hold. Cheapness of the underlying tells you nothing about the carry of the vehicle.

**Misconception 5: "Roll yield is always negative — commodities are a bad long-term hold."** Also wrong, in the other direction. Roll yield is negative in contango but *positive* in backwardation, and historically crude oil spent long stretches backwardated, paying long holders a structural carry of high single digits to low double digits a year. The honest statement is not "the roll always bleeds" but "the roll's sign is the sign of the curve's slope" — which is precisely why you must read the slope. We make the bullish case for the carry in [Backwardation as a structural return source](/blog/trading/commodities/backwardation-as-a-structural-return-source-the-carry-of-commodities).

## How it shows up in real markets: choosing a vehicle

So you have decided you want exposure to a commodity. The roll changes *everything* about how you should get it. Here is how the issue plays out across the actual menu of vehicles, and what the brochures conveniently underplay.

**Front-month ETFs (the USO model).** These hold the nearest contract and roll monthly. They give you the purest, most responsive exposure to the prompt price — and the maximum roll sensitivity. In backwardation they shine; in contango they bleed fastest. They are *trading* tools, not holding tools. If you are expressing a tactical view on the prompt price over days or a few weeks, a front-month ETF is responsive and honest. If you are "investing for the long run," it is usually the worst possible choice in a contango market.

**Spread-the-curve and optimized-roll ETFs.** The industry's answer to the bleed is to stop concentrating in the front month. Some funds hold contracts spread across many maturities (twelve-month "ladder" strategies); others use *optimized-roll* rules that algorithmically pick the contract on the curve with the *least* contango (or the most backwardation) to hold and roll. By rolling into deferred months where the curve is flatter, they cut the per-roll cost substantially. The trade-off: they track the prompt price less tightly and respond more sluggishly to spot moves. For a *holder*, that is usually a good trade — you give up some prompt sensitivity to stop the bleed. This is the same logic USO was *forced* into in April 2020, now offered as a deliberate product design.

The reason an optimized-roll fund helps is that the contango is almost always *steepest at the front of the curve* and flatter further out. The gap from month 1 to month 2 might be 2%, but the gap from month 11 to month 12 might be only a fraction of a percent, because most of the storage-and-glut pressure is concentrated in the prompt. By rolling between far-deferred months instead of the prompt, the optimized fund pays the shallow part of the slope rather than the steep part.

#### Worked example: front-month versus optimized-roll over a year

Compare two funds holding the same oil exposure for a year, spot flat, in a curve that is steep at the front and flat at the back. The front-month fund rolls the prompt every month at a 2% bleed. The optimized fund rolls between deferred months where the contango is only 0.4% per roll.

```
front-month fund:  100 x (0.98)^12   = about 78.5  (lost ~21%)
optimized fund:    100 x (0.996)^12  = about 95.3  (lost ~4.7%)
```

Same commodity, same flat spot, same year — and a 16-percentage-point difference in outcome, entirely from *which part of the curve each fund chose to roll through*. **The intuition: the optimized-roll fund does not escape the roll, it just pays the shallow back of the curve instead of the steep front, which is why for a long-term holder in a contango market the roll methodology can matter more than the fund's expense ratio by an order of magnitude.** When you compare two commodity funds, the headline expense ratios (say 0.6% versus 0.8%) are noise next to a roll-methodology difference that can be worth fifteen points a year.

**Broad commodity index funds.** Products tracking the S&P GSCI or Bloomberg BCOM hold a diversified basket and roll it on a published schedule. They blend many curves, so a backwardated oil curve can offset a contango gas curve. They also embed the collateral yield prominently, which matters enormously when rates are high. The catch is that the index's *roll rule is mechanical and predictable* — front-running of the GSCI roll by other traders is a documented cost — and the basket's overall roll yield still depends on the weighted shape of all the underlying curves. We unpack index construction and its energy tilt in [Commodity index investing](/blog/trading/commodities/commodity-index-investing-gsci-bcom-and-owning-the-basket).

**Equities and physically-backed vehicles.** For some commodities you can sidestep futures entirely. Buying shares in oil producers, miners, or commodity-trading houses gives you commodity-linked exposure with *no roll* — at the cost of taking on company-specific and equity-market risk. For the genuinely storable precious metals, physically-backed ETFs hold the actual bars in a vault, so there is no roll at all — which is one of several reasons gold and silver behave so differently from oil and gas as investments, a contrast we draw out in the gold series' [paper-versus-physical](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical) treatment. The deeper point is that gold is a *monetary* asset cheap to store, while oil is a *consumption* asset expensive to store — and storability is exactly what determines whether a vehicle suffers the roll.

The vehicle decision, then, reduces to a single question that the marketing rarely foregrounds: *what is this fund's roll methodology, and what shape is the curve it rolls through?* Front-month plus contango equals bleed. Optimized-roll or spread-the-curve softens it. A backwardated curve flips it to a tailwind regardless of method. The brochure will show you a chart of the commodity's spot price; your job is to look past it to the roll.

## The playbook: read the curve before the brochure

Here is the synthesis — the concrete habits that turn this knowledge into protection.

**1. Before buying any long-only commodity vehicle, read the curve shape first.** Pull up the forward curve for the commodity. Is it sloping up (contango, the roll fights you) or down (backwardation, the roll helps you)? This single check tells you the *sign* of your carry before you have taken any view on direction. It takes thirty seconds and it is the most important thing you will do. If the curve is in steep contango — gas, almost always; oil during a glut — a long-only hold is rolling a boulder uphill, and you should either not do it, size it small, or pick a vehicle built to dodge the roll.

**2. Separate your view on direction from your exposure to the roll.** "I think oil will rise" and "I will profit from owning an oil ETF" are *different statements* connected by the roll. You can be right on the first and lose on the second. If your directional view is strong but the curve is in contango, consider expressing it through a vehicle with low roll drag (optimized-roll, spread-the-curve, or producer equities) rather than a front-month ETF that will tax you the whole time you wait to be proven right.

**3. Match the holding period to the curve.** Front-month ETFs are *trading* instruments. If you genuinely want a multi-month or multi-year hold in a contango market, the compounding bleed is your enemy and you must use a roll-optimized or index vehicle, or accept producer-equity risk instead. The longer you intend to hold, the more the roll dominates and the more it should drive your vehicle choice.

**4. Treat steep, chronic contango as a near-prohibition on long-only holding.** Natural gas is the canonical case. A long-only gas ETF in steep contango can lose 30-40% a year to the roll alone, on top of any spot decline. "Gas is cheap, I will wait for the bounce" is the single most expensive sentence in retail commodity investing. If you must express a gas view, do it tactically and briefly, or through equities, never as a buy-and-hold ETF position.

**5. Remember the collateral tailwind — and that it is rate-dependent.** When short rates are high, the T-bill yield on a fund's collateral can offset a large chunk of a mild contango bleed; when rates are near zero, that cushion disappears and the bleed bites at full strength. The *same* commodity fund is a meaningfully better hold at 5% short rates than at 0%, purely because of the third term in the decomposition. Factor the rate environment into your expected total return, not just the spot view.

**6. When the curve is backwardated, the roll is a reason to hold, not just a tolerance.** Flip the whole frame in a tight, backwardated market: now the roll *pays* you a structural carry, year after year, on top of any spot appreciation. Backwardation is a tailwind a literate investor actively wants — the carry of commodities — and it is one of the few places in markets where a positive expected return shows up *mechanically* rather than as compensation for taking obvious risk. We make that case fully in [Backwardation as a structural return source](/blog/trading/commodities/backwardation-as-a-structural-return-source-the-carry-of-commodities), and the broader allocation question — how much, if any, commodity exposure belongs in a portfolio and through what sleeve — lives in [Energy: oil, gas, and the inflation engine](/blog/trading/cross-asset/energy-oil-gas-the-inflation-engine).

To make all of this operational, here is the five-line checklist to run before clicking "buy" on any commodity vehicle. First, **what is the curve shape today** — contango (headwind) or backwardation (tailwind)? Second, **what is the fund's roll methodology** — front-month (maximum roll sensitivity) or optimized/spread-the-curve (softened)? Third, **how big is the fund relative to the contract it trades** — a giant front-month roller pays a front-running tax a smaller fund avoids? Fourth, **what is the collateral yield environment** — are short rates high enough to cushion a mild bleed, or near zero? Fifth, **what is my intended holding period** — hours and days forgive the roll, months and years do not. Run those five questions and you will have done more diligence than the overwhelming majority of people who buy these products, who read the spot-price chart on the brochure and stop there.

A final reframing worth carrying away. The people who *systematically profit* from commodity markets are very often on the *other side* of the long-only retail roll. Producers selling forward, traders running cash-and-carry storage arbitrage, and speculators harvesting backwardation are, in part, collecting the carry that the uninformed long-only holder pays away every month. When you buy a contango-bleeding front-month ETF and hold it for a year, someone is the counterparty to your roll, and they are very glad you did not read the curve. The deepest protection this post offers is not a formula but a stance: in commodities you are not buying a price, you are *renting an exposure*, and the rent is set by the shape of a curve you can read for free in thirty seconds. The investor who reads it is the landlord; the one who reads the brochure is the tenant.

The thread back to this series' spine is exact. A commodity is a physical thing forced through a financial contract, and the forward curve — the cost of storage and the convenience yield made visible as a slope — is the gear that decides who profits from the roll. The long-only investor who ignores that gear is the counterparty paying the carry; the investor who reads it first is the one collecting it, or at least refusing to pay it. "Oil went up 20% but my oil ETF went down" is not bad luck or a broken product. It is the curve, doing exactly what its shape said it would do, to someone who read the brochure instead of the slope. Read the slope.

## Further reading & cross-links

- [Contango vs backwardation: what the shape of the curve means](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) — the two curve shapes in full, and why the shape beats the level for long-only returns.
- [Backwardation as a structural return source](/blog/trading/commodities/backwardation-as-a-structural-return-source-the-carry-of-commodities) — the *positive* side of the roll: harvesting the carry of commodities when the curve slopes down.
- [Commodity index investing: GSCI, BCOM, and owning the basket](/blog/trading/commodities/commodity-index-investing-gsci-bcom-and-owning-the-basket) — how the major indices weight, roll, and blend curves, and why an index is mostly an energy bet.
- [Convenience yield and the cost of carry](/blog/trading/commodities/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape) — *why* the curve takes its shape, the deep cause of the roll.
- [Spot vs futures: the two prices of the same barrel](/blog/trading/commodities/spot-vs-futures-the-two-prices-of-the-same-barrel) — what a futures contract is and why most never deliver.
- [Natural gas: Henry Hub, TTF, JKM](/blog/trading/commodities/natural-gas-henry-hub-ttf-jkm-and-the-most-seasonal-market) — why gas is structurally three seasonal markets and the home of chronic contango.
- [Energy: oil, gas, and the inflation engine](/blog/trading/cross-asset/energy-oil-gas-the-inflation-engine) — the allocation lens: how energy fits as one sleeve in a portfolio.
- [The term structure of volatility: contango, backwardation, and the VIX curve](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve) — the exact same roll problem in volatility products, where VIX ETPs bleed for the identical reason.
- [Gold futures: COMEX, contango, backwardation, and paper vs physical](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical) — how a storable monetary metal sidesteps the roll that crushes oil and gas funds.

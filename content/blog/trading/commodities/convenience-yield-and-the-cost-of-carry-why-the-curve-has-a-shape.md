---
title: "Convenience Yield and the Cost of Carry: Why the Curve Has a Shape"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The cost-of-carry model that explains why a forward curve slopes up or down: financing plus storage push it into contango, the convenience yield of holding the physical pulls it into backwardation, and reading the implied yield tells you how tight the market is."
tags: ["commodities", "cost-of-carry", "convenience-yield", "contango", "backwardation", "forward-curve", "storage", "futures", "no-arbitrage", "crude-oil"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A forward curve does not slope up or down by accident. Its shape is set by an accounting identity: the futures price equals the spot price plus the cost of carrying the physical (financing the cash tied up, paying for storage) minus the **convenience yield** — the hidden dividend you earn from actually having the stuff on hand. Read the slope and you read the balance of those three forces.
>
> - **Futures = Spot + financing + storage − convenience yield.** Or in compounding form, `F = S × e^((r + u − y)T)`. That single line explains the entire shape of the curve.
> - **Financing and storage push the curve UP** (contango): the future costs more because someone had to fund and warehouse the barrel until then. **Convenience yield pulls it DOWN** (backwardation): when the physical is precious *right now*, owning paper a year out is worth less than owning the real thing today.
> - **Storability decides where a commodity sits.** Cheap, dense, durable goods (gold, copper) sit near *full carry*. Hard-to-store goods (natural gas) swing wildly. Non-storable goods (electricity, perishables) break the model entirely — there is no carry to arbitrage.
> - The one number to remember: spot \$80, one-year financing 5% = \$4, storage \$3, gives a *full-carry* future of about **\$87**. If the actual one-year future trades at **\$83**, the market is paying you a **\$4 (≈5%) convenience yield** to hold physical — a direct, tradeable reading of how tight the barrel is *today*.

In the autumn of 2007, a curious thing happened in the oil market. Crude was expensive — Brent was pushing past \$90 a barrel on its way to the famous \$147 peak the following July — and yet the *shape* of the forward curve was screaming a different message than the price level. A barrel for delivery next month cost noticeably *more* than a barrel for delivery a year out. The curve sloped **downward**. To anyone who only watched the headline price, oil was simply "going up." To a trader who read the curve, the market was saying something far more precise: *we are short of physical barrels right now, and we will pay a premium to have one in our hands today rather than a promise of one next year.*

That premium has a name — the **convenience yield** — and together with the **cost of carry** it is the single idea that explains *why* a forward curve has the shape it has. In the previous posts of this series we learned to read the curve and we named its two shapes: [contango](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) (upward, back dearer than front) and backwardation (downward, front dearer than back). We described *what* the shapes look like and *what* they tell you about supply and demand. This post is the engine room. It answers the deeper question those posts deferred: *what actually sets the slope?* Why is one commodity's curve almost always gently upward while another's flips violently from up to down and back? Why can you store gold for a rounding error but not electricity at any price?

The answer is a piece of pure accounting dressed up as finance. There is no forecasting in it, no opinion, no "the market thinks oil will rise." It is a **no-arbitrage** relationship: if the futures price strayed too far from spot-plus-carry, you could buy the physical, store it, sell it forward, and pocket a risk-free profit — and the act of doing so would drag the prices back into line. Understanding that relationship turns the forward curve from a mysterious squiggle into a readable instrument panel. By the end of this post you will be able to look at any commodity curve and back out, in dollars, exactly how tight the physical market believes it is.

![The cost-of-carry stack: spot plus financing plus storage minus convenience yield equals the futures price](/imgs/blogs/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape-1.png)

## Foundations: the cost of carry, built from zero

Let us forget commodities for a moment and start with the simplest possible question: *if you could buy something today and sell it for delivery in one year, what should the forward price be?*

Suppose you and I agree, right now, that in exactly one year I will hand you one ounce of gold and you will hand me an agreed amount of cash. We are setting a **forward price** today for a delivery one year from now. What number should we write down?

Here is the trick that pins it down. I — the seller — have a choice. I could wait a year and then buy the gold in the market to deliver to you, taking my chances on whatever the price is then. Or I could buy the ounce *today*, at the spot price, lock it in a vault, and simply hand it over when the year is up. The second route removes all my price risk: I already own exactly what I owe you. So the forward price we agree should be *whatever it costs me to do that second thing* — buy now and carry the asset to the delivery date. That total — the spot price plus everything it costs to hold the asset until delivery — is the **cost of carry**, and the forward price built from it is called the **full-carry** or **fair-value** forward price.

What does it cost me to carry the gold for a year? Two things:

1. **Financing.** To buy the ounce today I either spend my own cash (which could otherwise have earned interest in the bank) or I borrow the money (and pay interest). Either way, tying up the purchase price for a year has a cost equal to the interest rate. If gold spots at \$2,000 and the one-year interest rate is 5%, financing the position costs me \$100 over the year. This is the **cost of the money**, and it applies to *every* asset, commodity or not.
2. **Storage.** A physical thing has to sit somewhere safe. The vault charges rent. There is insurance against theft, and for some goods, spoilage or shrinkage. Gold is cheap to store (it is dense and it does not rot), but it is not free. Call it \$5 for the year.

So my full-carry forward price for gold is roughly:

> forward = spot + financing + storage = \$2,000 + \$100 + \$5 = **\$2,105**.

That is the whole idea, and it works for anything you can buy and hold. The forward price of a financial asset, a barrel of oil, a tonne of copper — all of it starts from *spot plus the cost of holding it to the delivery date*. Write the same thing in the compact way the textbooks do, using continuous compounding so the algebra is clean:

```
F = S * exp( (r + u) * T )
where  F = futures (forward) price
       S = spot price today
       r = financing rate (interest), per year
       u = storage + insurance cost, expressed as a rate per year
       T = time to delivery, in years
```

Read it in plain English: the futures price is the spot price grown forward at the *carry rate* `(r + u)` for time `T`. Both `r` and `u` are costs of holding the thing, so they *push the futures price above spot*. If a commodity had only these two forces acting on it, every forward curve in the world would slope gently upward — every commodity would be in mild contango all the time, with the back of the curve dearer than the front by exactly the cost of carry.

But that is plainly not what we see. Oil spent most of 2007–2008 and 2011–2014 in *backwardation*, with the front month dearer than the back. The curve sloped the *wrong* way for a pure cost-of-carry asset. Something is missing from the equation — a third force that *pulls the curve down*. That force is the convenience yield.

#### Worked example: the full-carry forward price of gold

You hold cash and want to know the fair one-year forward price of gold. Spot is \$2,000/oz. Your borrowing cost (or the interest you forgo) is **5% per year**, so financing the \$2,000 for a year costs `0.05 × 2,000 = \$100`. The vault and insurance run **0.25%** of value, or `0.0025 × 2,000 = \$5`. Add them up: the cost of carrying one ounce for a year is `\$100 + \$5 = \$105`. The full-carry forward is therefore `2,000 + 105 = \$2,105`, an upward slope of `105 / 2,000 = 5.25%` over the year. **The intuition: with no reason to crave physical gold *today*, gold's forward price is simply spot grown forward at the cost of carry — which is exactly why gold curves sit so close to full contango, a point we will return to when we contrast it with oil.**

### The two forms of the equation, and why the exponential one is cleaner

You will meet the cost-of-carry relationship written two ways, and it is worth seeing why they are the same. The **simple, additive** form — the one we used for the gold example — just adds up the dollar costs:

```
F = S + financing + storage - convenience benefit
  = S * (1 + r*T + u*T - y*T)
```

This is intuitive and perfect for back-of-the-envelope work over a single period. But it has a subtle flaw over longer horizons: it assumes you pay financing only on the *original* spot price, not on the accumulating storage costs, and it does not compound. The **continuous-compounding** form fixes this and is what every desk and textbook actually uses:

```
F = S * exp( (r + u - y) * T )
```

Here `r`, `u`, and `y` are all expressed as *continuously compounded annual rates*, and `T` is in years. The two forms agree closely for small rates and short horizons — for a 5% rate over one year, `exp(0.05) = 1.0513` versus the simple `1.05`, a difference of 13 basis points. Over five years, though, the gap widens, and the exponential form is the correct one because it compounds the carry the way real money compounds. The other reason practitioners prefer it: taking the logarithm of both sides turns the multiplication into a clean *addition* of rates —

```
ln(F / S) = (r + u - y) * T
```

— so the **log of the futures-to-spot ratio, divided by time, is exactly the net carry rate** `(r + u − y)`. That single quantity — call it the *annualized basis* — is the number a trader reads off the curve. A positive annualized basis is contango; negative is backwardation; and whatever it is, it equals financing plus storage minus convenience yield. Everything in this post is just that one line, rearranged to solve for whichever term you do not already know.

### Where the financing rate actually comes from

The `r` in the equation is not a mysterious constant — it is the real, observable interest rate at which the carry trade is financed, and which rate you use matters. For a US-dollar commodity it is essentially the dollar funding cost: a short-term risk-free rate (the kind set by the central bank's policy, plus a small spread for the borrower's credit). When the Federal Reserve hikes rates, the financing leg of *every* commodity's carry rises in lockstep, which mechanically *steepens contango* across the whole complex — a higher `r` makes the future more expensive relative to spot. This is one of the quiet channels through which [monetary policy moves commodities](/blog/trading/macro-trading/how-monetary-policy-moves-commodities-real-rates-gold): not only through the level of prices, but through the *slope* of every forward curve. In a zero-rate world (2009–2021 for much of the time), the financing leg nearly vanished and curves flattened toward storage-and-convenience alone; in the 2022–2024 hiking cycle, the financing leg came roaring back and contango steepened across metals and energy.

A subtlety worth flagging: the relevant `r` is the *borrowing* rate of the marginal arbitrageur, not the central bank's headline number. A commodity trading house funds itself at a spread over the risk-free rate, and a hedge fund pays more still. So the "full carry" a given player can enforce depends on *their* cost of money — which is why the arbitrage that pins the curve is enforced first by the cheapest-funded participants (the big banks and trading houses) and why the curve can drift a little before the trade becomes worthwhile for everyone else.

## The third force: convenience yield, the dividend of having the physical

Imagine you run a refinery. Your business is turning crude oil into gasoline and diesel, and you make money on the *spread* between them — the [crack spread](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities). Your plant runs 24 hours a day. Now ask: what is a barrel of crude physically *in your tank, today* worth to you, compared with a paper promise of a barrel a year from now?

The answer is: *more*. A lot more, sometimes. Because if your tanks run dry, your refinery stops. A stopped refinery is a catastrophe — you lose the crack margin on every barrel you cannot process, you may breach supply contracts, you might have to restart the unit (expensive and risky). A paper future does not keep your plant running next Tuesday; only physical crude in the tank does. So you are willing to *pay a premium* to hold actual barrels rather than promises — and the size of that premium depends on how worried you are about running short.

That premium is the **convenience yield**. It is the benefit — the "convenience" — of physically possessing the commodity, over and above just owning a contract that pays off later. It is not paid in cash; it is a *yield in kind*, an implicit return you earn simply by having the stuff on hand when you might need it. Think of it as the dividend a physical inventory pays its holder: not in dollars, but in operational security and optionality.

There is a precise financial analogy that makes this click. When you hold a *stock*, you collect dividends — a cash yield that compensates you for owning it, and which lowers the fair forward price (a forward on a dividend-paying stock trades *below* spot-plus-financing by the dividend, because the buyer of the forward misses the dividends the holder collects). The convenience yield plays exactly that role for a commodity: it is the **"dividend" a physical inventory pays its holder**, except paid in operational security rather than cash. This is not a loose metaphor — it is the same term in the same equation. In options and forwards pricing, a stock's dividend yield and a commodity's convenience yield occupy the identical slot, both entering with a minus sign and both lowering the forward relative to spot-plus-financing. If you have read [what sets an option's price](/blog/trading/options-volatility/what-sets-an-options-price-the-five-inputs-and-the-intuition), the dividend-yield input there *is* the convenience yield here, wearing different clothes.

There is also an **embedded-option** flavor to the convenience yield that explains why it is so non-linear. Holding physical gives you the *option* to use it the instant you need it — to keep the refinery running, to meet a delivery, to seize an unexpected high-margin opportunity. Like any option, that flexibility is worth almost nothing when you have plenty (deep inventories, no chance of needing it urgently) and worth a great deal when you are near the edge (low inventories, a real chance of a costly stock-out). That option-like quality is *why* the convenience yield stays near zero through abundant inventories and then rises steeply as stocks drain — the same convexity an option shows as it moves toward the money. It is the optionality of physical possession, priced.

Crucially, the convenience yield behaves like a **negative storage cost**. Where storage `u` is a cost of holding (push the curve up), the convenience yield `y` is a *benefit* of holding (pull the curve down). It enters the equation with a minus sign:

```
F = S * exp( (r + u - y) * T )
where  y = convenience yield, per year (the benefit of holding physical)
```

Now the equation is complete, and it explains *both* curve shapes from a single line:

- When **`r + u > y`** (financing and storage outweigh the craving for physical), the exponent is positive, `F > S`, and the curve slopes **up**: **contango**. This is the normal state for a well-supplied market — there is no urgency to hold physical, so the future simply costs spot plus carry.
- When **`y > r + u`** (the convenience of physical outweighs the cost of carry), the exponent is negative, `F < S`, and the curve slopes **down**: **backwardation**. This is the signature of a *tight* market — physical is so precious right now that people will accept a lower price for delivery later just to free up barrels (or to avoid paying the premium of holding them).

This is the deepest payoff of the model. Contango and backwardation are not two separate phenomena needing two separate explanations. They are the *same equation* with the sign of `(r + u − y)` flipped. The whole drama of the forward curve reduces to a tug-of-war between two teams: **carry costs (financing + storage) pulling the curve up**, and **convenience yield pulling it down**. Whoever is winning sets the slope.

![Two forces on the curve: financing and storage push up into contango while convenience yield pulls down into backwardation](/imgs/blogs/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape-2.png)

#### Worked example: backing out the convenience yield from a real-looking curve

Suppose crude spot is **\$80**, the one-year financing rate is **5%**, and storage runs **\$3** for the year. The full-carry one-year future *would* be `80 + (0.05 × 80) + 3 = 80 + 4 + 3 = \$87`. That is what oil should cost a year out if nobody cared about holding physical today.

But suppose the actual one-year future trades at only **\$83**. The market is pricing the deferred barrel \$4 *cheaper* than full carry. Where did that \$4 go? It is the **convenience yield**: holders of physical crude are implicitly earning \$4 over the year (about `4 / 80 = 5%`) from the security of having barrels on hand, so they accept \$4 less for a barrel delivered later. We can solve for it directly: `convenience yield = financing + storage − (future − spot) = 4 + 3 − (83 − 80) = 7 − 3 = \$4`, or **5% per year**. **The intuition: the gap between the full-carry price the curve *should* show and the price it *actually* shows is a dollar-denominated thermometer of physical tightness — and here it reads \$4 of "the barrel is wanted now."**

Notice we computed the same number two ways: as the *difference between full carry and the observed future* (\$87 − \$83 = \$4), and as *carry costs minus the observed slope* (\$7 of carry − \$3 of upward slope = \$4 of downward pull). They must agree, because they are the same identity rearranged. That is the reassuring thing about an accounting relationship: there is no free parameter to fudge.

## The implied convenience yield is a thermometer

Here is the move that turns all this theory into a usable tool. In the real world you can *observe* three of the four quantities directly: the spot price `S`, the futures price `F` (off the screen), and the financing rate `r` (the relevant interest rate). Storage `u` you can estimate well for most commodities — tank rents, warehouse fees, and insurance are published or quotable. That leaves exactly *one* unknown: the convenience yield `y`. So you can **rearrange the equation and solve for it**:

```
implied convenience yield  y = r + u - (1/T) * ln(F / S)
```

The quantity `(1/T) × ln(F/S)` is just the *annualized slope* of the curve — how fast, in percent per year, the futures price grows (or shrinks) relative to spot. Subtract that slope from the carry rate `(r + u)` and what is left over is the market's *implied convenience yield*. You did not forecast it; you *backed it out* of prices that already exist. It is the residual — the part of the curve's shape that carry costs alone cannot explain.

And that residual is a remarkably good **thermometer for physical tightness**:

- A **high implied convenience yield** (often when the curve is backwardated) says the physical market is *tight*: inventories are low, buyers are scrambling, and a barrel-in-hand is precious. This is the curve shape during shortages, supply shocks, and demand surges.
- A **low or negative implied convenience yield** (curve in contango, futures above full carry) says the market is *loose*: there is plenty of physical sloshing around, nobody is desperate to hold it, and the future trades up at — or even *above* — full carry because tanks are filling and storage itself is getting scarce.

This is why professional desks watch the *implied convenience yield* (or its close cousin, the calendar-spread relative to estimated carry) as a real-time gauge. When it spikes, the physical market is signalling stress before the headlines catch up. The 2007 oil curve that opened this post was exactly this: a high implied convenience yield, backwardation, the market paying up for prompt barrels because they were genuinely short. The April 2020 collapse was the mirror image — a *super-contango* so steep that the implied convenience yield went deeply negative, because there was so much oil and so little tank space that holding a barrel was a *liability*, not a convenience.

![Illustrative contango curve with the implied annual carry between the front and the one year point](/imgs/blogs/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape-3.png)

#### Worked example: reading the implied carry off a contango curve

Take an illustrative WTI-style contango curve: the front month (M0) sits at **\$72.0** and the one-year contract (M12) at **\$76.6**. The curve slopes *up* by `76.6 − 72.0 = \$4.60`, an annualized slope of about `4.60 / 72.0 = 6.4%`. Now suppose financing is **5%** and storage is **2%** of price per year, so the carry rate `(r + u)` is **7%**. Plug in: `implied convenience yield = 7% − 6.4% = 0.6%`. The convenience yield is *nearly zero* — almost all of the upward slope is explained by the cost of carrying the barrel. **The intuition: a contango curve whose slope roughly matches the cost of carry is a market in calm, well-supplied balance — nobody is paying up for prompt barrels, so the future is just spot plus the rent on the tank and the interest on the cash.**

Contrast that with a backwardated curve where M0 is **\$85.0** and M12 is **\$78.9** — a *downward* slope of `−6.1 / 85.0 = −7.2%` per year. With the same 7% carry rate, `implied convenience yield = 7% − (−7.2%) = 14.2%`. That is an enormous yield: the market is paying holders the equivalent of 14% a year, in kind, to keep physical barrels available *now*. The thermometer is reading "very tight."

### Inventories are the knob that turns the convenience yield

If the convenience yield is the thermometer, **inventory** is the thing being measured. The relationship is one of the most robust empirical regularities in all of commodities, and it has a clean logic. The convenience yield is the value of having physical *on hand to avoid a stock-out*. When inventories are abundant — tanks brimming, warehouses full — the chance of running short is tiny, so the convenience of holding one more barrel is near zero. As inventories drain toward empty, the chance of a stock-out climbs, and the value of being the one who *has* the barrel rises sharply. So:

- **High inventories → low convenience yield → curve in contango.** A well-stocked market has nothing to fear from deferral, so the future trades at spot-plus-carry. In fact, when inventories are *so* high that storage space itself becomes scarce, the effective storage cost `u` spikes and the curve goes into *super-contango* — the future has to climb steeply enough to pay for ever-more-expensive tank space.
- **Low inventories → high convenience yield → curve in backwardation.** A market scraping the bottom of its tanks will pay a large premium for prompt physical, dragging the curve down into backwardation.

This gives the convenience yield its characteristic shape as a function of inventory: it is **flat and near-zero when stocks are high, then curves up steeply as stocks approach a critical minimum** — economists call this the *theory of storage*, and the steep tail is why backwardation tends to appear suddenly rather than gradually. A market can sit in placid contango for months as inventories slowly draw, then snap into sharp backwardation in a matter of days once stocks cross the threshold where stock-out fear takes over. The non-linearity is the reason oil and gas curves can flip so violently: it is not that the world changed overnight, it is that inventories crossed the knee of the convenience-yield curve.

The practical consequence: if you can see inventory data (the EIA publishes weekly US crude and product stocks; the LME publishes warehouse metal stocks; the USDA publishes grain stocks-to-use), you have an independent read on where the convenience yield *should* be — and you can compare it to the *implied* convenience yield you back out of the curve. When the two disagree, the curve is telling you something the headline inventory number is not (or the inventory data is stale), and that gap is where curve-trading edges live.

#### Worked example: the curve flips as inventories drain

Start with a calm, well-stocked oil market: spot **\$80**, financing **5%** (\$4), storage **\$3**, and inventories comfortable, so the convenience yield is a small **\$1**. The net carry is `4 + 3 − 1 = \$6`, so the one-year future trades at `80 + 6 = \$86` — gentle contango. Now run the tape forward three months as a supply disruption drains tanks. Spot rises to **\$90** as prompt buyers scramble, and the convenience yield jumps from \$1 to **\$12** because stocks have crossed the knee where stock-out fear takes over. Carry costs are roughly unchanged (`\$4.50 financing + \$3 storage = \$7.50`), but now the net carry is `7.50 − 12 = −\$4.50`. The one-year future trades at `90 − 4.50 = \$85.50` — *below* the new \$90 spot. The curve has flipped from contango to backwardation, not because anyone changed their forecast, but because the convenience yield crossed the carry. **The intuition: the same \$10 jump in spot that grabbed the headlines hid the more important story in the curve — a convenience yield that went from trivial to dominant as the tanks drew down, flipping the whole shape.**

### The term structure of the convenience yield

So far we have spoken of "the" convenience yield as one number, but in truth it has a *term structure* of its own — it can be high at the front of the curve and low further out, or vice versa, and that two-part reading is far richer than a single contango/backwardation label. A very common shape is **backwardation up front fading to contango in the back**: the prompt months are tight (high near-term convenience yield, downward slope), but a year or two out the market expects supply to recover and inventories to rebuild (convenience yield fades, carry takes over, the deferred curve tilts back up). This "hump" — down then up — is the market saying *tight now, easing later*.

The opposite hump — contango up front rolling into backwardation further out — is rarer but appears when the market is well supplied today but anticipates a structural shortage down the road (a depleting field, a looming mine closure). The point is that the convenience yield, like an interest rate, has a *curve*, and a sophisticated reader does not collapse it to one word. The first three or four months are the *prompt structure*, dominated by today's physical tightness; everything past a year is the *deferred structure*, anchored to the long-run marginal cost of production where the convenience yield fades toward zero and the curve approaches full carry. The two halves can — and routinely do — tell different stories.

![Illustrative backwardation curve where the convenience yield dominates the cost of carry](/imgs/blogs/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape-4.png)

## The arbitrage that enforces the relationship

A skeptic should be asking: *why* must the futures price obey this equation at all? Markets are full of opinions; why does an accounting identity win? The answer is the **cash-and-carry arbitrage**, and it is the disciplinarian standing behind the whole model. (We give it a full post of its own — see [cash-and-carry and storage arbitrage](/blog/trading/commodities/cash-and-carry-and-storage-arbitrage-locking-in-the-curve) — but the logic is worth seeing here, because it is *why* the curve has a shape at all.)

Suppose the one-year future were trading *above* full carry — say spot is \$80, full carry is \$87, but the future is quoted at \$92. A trader can do the following, today, with zero price risk:

1. **Borrow \$80** at the 5% financing rate and **buy one barrel** of physical crude at spot.
2. **Pay \$3** to store and insure it for the year.
3. **Sell one barrel forward** at the \$92 futures price, locking in the sale today.

A year later, the trader delivers the stored barrel against the future, collects \$92, repays the \$80 loan plus \$4 interest, and has paid \$3 storage — net cash in: `92 − 80 − 4 − 3 = \$5`, with *no exposure to the price of oil*. Every step was locked at the start. That \$5 is a risk-free profit, and the moment it exists, traders pile in: they *buy spot* (pushing S up) and *sell the future* (pushing F down) until the gap closes and the future sits back at \$87, full carry. So the futures price *cannot* sustainably exceed spot-plus-carry — the arbitrage is the ceiling.

![The cash and carry arbitrage that enforces the curve: buy spot, store, sell forward, lock the spread](/imgs/blogs/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape-5.png)

That explains the *ceiling*. What about the *floor*? Why can't the future trade *below* full carry without an arbitrage forcing it back up? Here is the beautiful asymmetry that *is* the convenience yield. The reverse arbitrage — sell the physical short today, invest the proceeds, buy it back forward cheaply — requires you to *borrow the physical commodity* to sell it. But you usually cannot borrow a barrel of oil or a tonne of copper the way you can borrow a share of stock. And even if you owned the barrel and sold it, you would give up the *convenience* of having it on hand. So the reverse trade is blocked or costly, and the future is *free* to trade below full carry by the amount of the convenience yield. **The convenience yield is precisely the measure of how far below full carry the curve can fall before the (impossible) reverse arbitrage would kick in.** It is the gap the arbitrage cannot close — which is exactly why it carries information the carry costs do not.

This asymmetry is the heart of why commodity curves behave so differently from financial-asset curves. A stock-index future is pinned *tightly* to spot-plus-carry in both directions, because you can borrow and short the index cheaply — there is no convenience to holding it, and the arbitrage works both ways. A commodity future is pinned from *above* (you cannot exceed full carry) but floats *freely below* it by the convenience yield, because you cannot easily short the physical. That is why financial futures are nearly always in tidy contango and commodity futures swing into deep backwardation whenever the physical gets tight.

#### Worked example: the cash-and-carry profit when the future is rich

Spot crude is **\$80**, one-year financing **5%** (\$4), storage **\$3**, so full carry is **\$87**. The market mistakenly quotes the one-year future at **\$92**. You execute the cash-and-carry: borrow \$80, buy the barrel, pay \$3 storage, and sell forward at \$92. One year on: receive \$92 on delivery, repay `80 + 4 = \$84` of principal-plus-interest, net of the \$3 already spent storing it. Profit `= 92 − 84 − 3 = \$5` per barrel, risk-free. **The intuition: a future trading above full carry is literally free money for anyone with a tank and a credit line — which is exactly why, in a functioning market, it never stays there, and the curve is pinned at full carry from above.**

## Storability decides everything: the spectrum from gold to electricity

The model now lets us answer the question we have been circling: *why does one commodity's curve barely move while another's whips around?* The answer is **storability** — specifically, how cheap, dense, durable, and shortable the physical is. Storability determines how the three forces balance, and it sorts every commodity along a spectrum.

At one end sit the **easy-to-store assets**: gold and the precious metals, and to a large degree the base metals like copper and aluminum on the LME warehouse system. Their physical is dense (a lot of value in a small space, so storage is a tiny fraction of price), durable (it does not rot, evaporate, or decay), and — for gold especially — abundant relative to its industrial use, so almost nobody is ever *desperate* for a physical ounce *this week*. Result: the convenience yield `y` is close to zero, storage `u` is tiny, and the curve sits at **full carry**, a gentle contango of roughly the interest rate. Gold's curve is the cleanest cost-of-carry curve in the world — which is exactly why [gold's COMEX curve is treated as a financing instrument](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical) and why gold gets its own series as a *monetary*, not industrial, asset.

In the **middle** sits crude oil. Storable, yes — in tanks, in salt caverns, on tankers at sea — but storage is real money (oil is bulky relative to its value), and crucially, oil is *consumed continuously* and supply can be disrupted (an outage, an embargo, a war). So oil's convenience yield is large and *variable*: low when tanks are full (contango, even super-contango), high when the prompt market is tight (backwardation). Oil's curve is the great seesaw of the commodity world precisely because its convenience yield is so volatile.

Further along, **natural gas** is hard and expensive to store (it must be compressed or kept in specialized underground reservoirs and salt domes, and it has sharp seasonal demand), so its curve shows huge, predictable *seasonal* humps — high winter prices, low summer prices — that no simple flat carry can describe. And at the far end sit the goods you essentially **cannot store at all**: **electricity** (which must be consumed the instant it is generated — there is no economic way to warehouse a megawatt-hour at grid scale), and many **perishable agricultural goods** (you cannot carry this year's lettuce into next year). For these, the cost-of-carry model *breaks down completely*: there is no physical inventory to buy, store, and sell forward, so there is no arbitrage to enforce a relationship between today's price and a forward price. The "forward" price of electricity is a forecast of supply and demand at that future hour, not spot-plus-carry — which is why power prices can be \$30 one hour and \$3,000 the next with no contradiction.

![Where commodities sit on the storability spectrum from easy to store metals to non storable electricity](/imgs/blogs/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape-6.png)

This spectrum is the practical takeaway of the whole model. Before you even glance at a commodity's curve, you can predict its *temperament* from how storable it is:

- **Cheap, dense, durable, abundant** (gold, base metals) → low and stable convenience yield → near full carry → curve mostly gentle contango, slope ≈ interest rate.
- **Bulky, consumed continuously, disruptable** (oil) → large, variable convenience yield → curve seesaws between deep backwardation (tight) and steep contango (glut).
- **Hard to store, seasonal** (natural gas) → storage cost dominates, with seasonal humps → curve is a repeating sawtooth.
- **Non-storable** (electricity, perishables) → no carry, no arbitrage → the model breaks; forwards are demand forecasts, not carry math.

#### Worked example: full carry for gold versus the wild card in oil

Gold spots at **\$2,000**, financing **5%**, storage a tiny **0.25%**. Full-carry one-year forward: `2,000 × (1 + 0.05 + 0.0025) = 2,000 × 1.0525 = \$2,105`. Because gold's convenience yield is near zero, the real forward sits *right at* \$2,105 — the curve is full carry, predictable to the dollar.

Now oil. Spot **\$80**, financing **5%** (\$4), storage **\$3**, full carry **\$87**. But oil's convenience yield is a wild card: in a glutted market (April 2020) `y` collapses toward zero or below and the future climbs toward or past \$87 (contango); in a tight market (2007, 2022) `y` jumps to \$8–\$12 and the future *falls below spot*, to \$72 or lower (backwardation). Same equation, same carry costs — but oil's convenience yield does all the moving while gold's sits still. **The intuition: the difference between a boring curve and a violent one is not the financing or the storage — those are similar in percentage terms — it is the volatility of the convenience yield, and that volatility is just a measure of how disruptable the physical supply is.**

## Decomposing the slope: where each dollar of the spread comes from

It helps to *see* the three forces as a budget. Take the illustrative contango curve again — spot near \$72, the one-year point near \$76.6, a spread of about \$4.60. Where does each dollar of that \$4.60 come from? Roughly: financing (5% of \$72 ≈ \$3.60) is the largest piece, storage (say \$1.40 for the year) adds the rest, and the convenience yield is near zero — so almost the entire upward spread is *carry*, the rent on money and tank space. That is the anatomy of a calm, well-supplied market.

Flip to a tight, backwardated market and the same budget looks completely different. Carry still wants to push the spread *up* by its \$5 or so, but the convenience yield is now so large — \$10, \$12 — that it overwhelms the carry and drags the net spread *negative*. The barrel a year out trades *below* the prompt barrel because the prompt barrel is the one everybody needs.

![Decomposing the futures spread into financing storage and convenience yield contributions](/imgs/blogs/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape-7.png)

This decomposition is exactly what a desk does when it asks "is this curve cheap or rich?" It estimates the financing leg (it knows the interest rate), estimates the storage leg (it knows tank rents), and whatever is *left over* in the observed spread is the market's implied convenience yield. If that residual looks too high relative to what the trader believes about inventories, the curve is "too backwardated" and there may be a trade; if it looks too low (or negative), the curve is "too contango'd." The cost-of-carry model is not just an explanation — it is the scaffolding for an entire style of relative-value trading on the *shape* of the curve, which we develop in the calendar-spread and roll-yield posts.

The reason the decomposition matters for *returns*, not just for understanding, is the **roll**. A long-only investor who holds a commodity through futures never takes delivery; they sell the expiring front contract and buy the next one out, repeatedly, "rolling" their position down the curve. In contango, each roll means selling a cheaper expiring contract and buying a dearer next one — a small loss baked in every month, the *negative roll yield* that quietly bleeds long-only commodity ETFs. In backwardation, the roll runs the other way: you sell a dearer front and buy a cheaper next, earning a *positive roll yield* just by holding. So the same `(u − y)` term that bends the curve is, dollar for dollar, the carry that a passive holder earns or pays. The shape is not academic — it is the single biggest determinant of whether a long-only commodity position makes money over time, often swamping the move in the spot price itself. That is why a serious investor reads the convenience yield before they read the price forecast: the forecast tells you where spot might go, but the carry tells you what you earn or lose just for *being there* while you wait.

## Common misconceptions

**"Contango means the market expects prices to rise."** No — this is the single most common error, and it confuses the *shape* of the curve with a *forecast*. A contango curve does not predict that spot will climb to meet the back month. It says that *holding the commodity costs money* (financing + storage exceed convenience yield), so the future is priced higher to compensate the holder. In fact, in a steep contango, the typical outcome is the *reverse*: the back-month price falls toward the (lower) spot as time passes — which is exactly the "roll bleed" that erodes long-only commodity ETFs. The curve's slope is about the cost of carry, not a price target. The expectation of future spot is a *separate* thing layered on top.

**"Backwardation means the market expects prices to fall."** Same error, mirror image. A downward-sloping curve does not forecast a decline; it reflects a high convenience yield — physical is precious *now*. Markets in steep backwardation (oil in 2007, 2022) frequently kept *rising* for months. Backwardation is a statement about present tightness, not a bet on the future direction of spot.

**"Storage cost and convenience yield are two unrelated numbers."** They are mirror images of each other — storage is the *cost* of holding, convenience yield the *benefit* of holding, and they enter the equation with opposite signs. The *net* of the two (`u − y`) is what actually bends the curve. When inventories are high, storage dominates and the net is positive (contango); when inventories are low, convenience yield dominates and the net goes negative (backwardation). It is more useful to track a single "net carry" that swings with inventory than two separate constants.

**"The cost-of-carry model works for every commodity."** It works only for things you can *buy, store, and re-sell forward*. For non-storable goods — electricity above all, but also fresh produce, and even some financial-like exposures — there is no inventory to arbitrage, so the model simply does not apply. An electricity "forward" is a forecast of the supply-demand balance at that hour, which is why power forwards show shapes (huge intraday and seasonal peaks) that no carry math could ever produce. Knowing *when the model breaks* is as important as knowing how it works.

**"A near-zero convenience yield means the commodity is unimportant."** It means the commodity is *abundant and easy to store* relative to its consumption — which is true of gold, where the above-ground stock dwarfs annual mine supply and industrial use. A near-zero convenience yield is the signature of a *monetary-like* asset (you hold it for its value, not because you fear running out), versus the high, volatile convenience yield of a *consumption* asset like oil or wheat. This is the deepest reason gold sits in its own [monetary-metal framing](/blog/trading/gold/is-gold-money-a-commodity-or-a-currency-the-framing-that-decides-everything) and not in the industrial-commodity bucket.

## How it shows up in real markets

**Oil, 2007–2008: backwardation as a tightness signal.** Through late 2007 and into the 2008 super-spike, the WTI curve was backwardated even as prices marched from \$90 toward \$147. The implied convenience yield was high — the prompt physical market was genuinely tight, OPEC spare capacity was thin, and refiners were paying up for barrels in hand. A trader watching only the price level saw "expensive oil." A trader reading the curve saw *expensive-and-tight*: the backwardation was the market's own confession that it had no slack. When the demand-destruction crash came in late 2008, the curve flipped violently into contango — physical suddenly piled up faster than anyone could use it, convenience yield collapsed, and storage became the binding constraint.

**Oil, April 2020: super-contango when the tanks filled.** The mirror image, and the most extreme carry event in modern markets. As the pandemic destroyed demand, crude piled up until storage at Cushing, Oklahoma — the WTI delivery point — physically ran out. With nowhere to put a barrel, the convenience yield went *deeply negative*: holding a barrel was a liability, not a convenience, because you might be forced to take delivery with no tank to put it in. The front WTI contract settled at **minus \$37.63** on 20 April 2020 while the one-year-out contract sat near **\$36.50** — a super-contango of more than \$70 between prompt and deferred. The cost-of-carry model explains it perfectly: when storage itself becomes scarce, `u` (the cost of storage) spikes toward infinity and the prompt price collapses below the deferred. The negative price was not the market saying "oil is worthless"; it was the market saying "the cost of *storing* this specific barrel right now exceeds its value."

**Natural gas: the seasonal carry.** Gas is the textbook case of storage-driven shape. Because gas is expensive to store and demand spikes in winter (heating), the curve shows a repeating sawtooth: winter contracts trade at a premium to summer ones, year after year. This is not a forecast that gas "will rise into winter" — it is the cost-of-carry model with a seasonal storage constraint baked in. Utilities inject gas into underground storage in the cheap summer and withdraw it in the expensive winter, and the summer-to-winter spread must, by arbitrage, roughly equal the cost of carrying the gas across those months. There is a famous trade buried in this structure — the *widow-maker*, the spread between March (the last cold month) and April (the first warm one), which blows out spectacularly when a cold snap threatens to empty storage before spring. That spread is pure convenience yield: the value of a molecule of gas in March, when you might freeze without it, versus April, when you do not need it. In the 2022 European energy crisis, after the Russian pipeline supply was cut, the TTF winter convenience yield exploded — Dutch month-ahead gas spiked toward **EUR 340/MWh** in August 2022 against a pre-crisis level near **EUR 9** the year before — because the entire continent was terrified of running short before spring. That was not a price forecast; it was the convenience yield of survival molecules going vertical.

**Copper and the LME: full carry with occasional squeezes.** Industrial metals sit nearer the gold end of the spectrum — dense, durable, and stored cheaply in the [LME warehouse system](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities) — so copper's curve usually sits close to full carry, in mild contango equal to financing plus the modest warehouse rent. But copper is *consumed* (unlike gold, which is mostly hoarded), so when a specific delivery location runs short — a warehouse with little on-warrant metal, a sudden Chinese restock — the prompt convenience yield can spike and throw the front of the curve into sharp backwardation, even while the back stays in placid contango. The LME's daily "cash-to-three-month" spread is essentially a direct quote of copper's prompt convenience yield, and traders watch it as a tightness gauge exactly the way oil desks watch the prompt time spread. The contrast with gold is the whole lesson of the spectrum: same warehouse economics, but copper's *consumption* gives it a live convenience yield that gold's *hoarding* does not.

**Gold: the full-carry benchmark.** Gold's curve is the control experiment. Because its convenience yield is near zero — nobody's factory stops if they cannot get a physical ounce this week — gold trades at almost exactly full carry, a gentle contango equal to the financing rate minus the tiny lease/storage costs. When gold *does* briefly slip into backwardation (it has, in stressed funding markets), it is a genuine alarm: it means physical gold has suddenly become hard to borrow, the rarest of convenience-yield events for a monetary metal, and it tells you something is broken in the funding plumbing rather than in the metal's fundamentals.

## The takeaway: how to read the curve like a carry trader

You now have the master key. A forward curve's *shape* is not a mystery and it is not a forecast — it is the visible output of a tug-of-war between the **cost of carry** (financing + storage, pushing up into contango) and the **convenience yield** (the dividend of physical, pulling down into backwardation). Here is how to put it to work:

- **Decompose, do not just label.** Don't stop at "it's in contango." Estimate the carry — the interest rate plus the storage cost — and compare it to the *actual* slope. If the slope roughly equals carry, the market is calm and well supplied. If the slope is steeper than carry (super-contango), storage itself is getting scarce — a glut. If the slope is *flatter* than carry, or negative (backwardation), a convenience yield is at work and the physical market is tightening.

- **Treat the implied convenience yield as a tightness thermometer.** Back it out with `y = r + u − (1/T)·ln(F/S)`. A rising implied convenience yield is an early, price-based warning that the physical market is getting short — often before the inventory data or the headlines confirm it. A falling or negative one says supply is piling up.

- **Predict temperament from storability.** Before you trade a commodity, ask how cheap, dense, durable, and shortable its physical is. Cheap-and-durable (gold, metals) → boring full-carry curve. Bulky-and-consumed (oil) → violent seesaw. Hard-to-store-and-seasonal (gas) → sawtooth. Non-storable (power, perishables) → the carry model breaks, so do not even try to read it as carry.

- **Remember the asymmetry.** The future is pinned at full carry *from above* by the cash-and-carry arbitrage, but it can float freely *below* full carry by the convenience yield, because you cannot easily short the physical. That asymmetry — borrowable financial assets pinned both ways, un-borrowable commodities pinned only from above — is *why* commodity curves swing into deep backwardation while financial-asset curves stay in tidy contango.

- **Watch for when storage itself becomes the constraint.** The model's tidy ceiling — "the future cannot exceed full carry" — assumes storage capacity is available at a known price. When tanks physically fill, that assumption breaks: storage stops being a fixed rent and becomes a scarce, auctioned resource whose price can spike toward infinity. That is the regime that produced April 2020's negative prices, and it is the one extreme worth holding in your head, because it inverts everything. A negative spot price is not a paradox in the model — it is the cost of storage `u` exceeding the value of the commodity itself, so that someone will pay you to take a barrel off their hands because they have nowhere to put it. When you hear a commodity price has gone negative, do not reach for "the thing is worthless"; reach for "the storage has run out," and the cost-of-carry equation will have told you so before the headline did.

- **Connect it to the spine.** This is the whole series in one equation. A commodity is a physical thing forced through a financial contract, and the cost of carry plus the convenience yield are the gears that translate the physical balance — how much is around, how badly it is wanted now, how expensive it is to hold — into the *shape* of the paper curve. Master those gears and the curve stops being a chart you look at and becomes an instrument you read. From here, the natural next steps are to *trade* the shape directly ([calendar spreads](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities) and the [cash-and-carry](/blog/trading/commodities/cash-and-carry-and-storage-arbitrage-locking-in-the-curve)) and to understand why the same shape that explains the curve also quietly determines *who profits from the roll*.

The next time you see a commodity curve, don't ask "what does the market think the price will be?" Ask instead: *what is the cost of carrying this thing, and how badly does someone need it in their hands today?* The slope is just the answer to that question, written in prices.

## Further reading & cross-links

**Within this series — the curve, and how to trade its shape:**

- [The forward curve: the most important chart in commodities](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities) — the chart this post explains the *shape* of.
- [Contango vs backwardation: what the shape of the curve means](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) — the *what*; this post is the *why* underneath it.
- [Cash-and-carry and storage arbitrage: locking in the curve](/blog/trading/commodities/cash-and-carry-and-storage-arbitrage-locking-in-the-curve) — the arbitrage that *enforces* the cost-of-carry relationship, in full.

**The monetary metal — the control experiment for a near-zero convenience yield:**

- [Gold futures, COMEX: contango, backwardation, and paper vs physical](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical) — gold's curve as the cleanest full-carry curve in the world.
- [Is gold money, a commodity, or a currency?](/blog/trading/gold/is-gold-money-a-commodity-or-a-currency-the-framing-that-decides-everything) — why a near-zero convenience yield is the signature of a *monetary* asset.

**The pricing math — convenience yield is to a commodity what dividends are to a stock option:**

- [What sets an option's price: the five inputs and the intuition](/blog/trading/options-volatility/what-sets-an-options-price-the-five-inputs-and-the-intuition) — carry, dividends, and the cost of holding, in the options context.

**The macro lens — what moves the level the curve sits on:**

- [How monetary policy moves commodities: real rates and gold](/blog/trading/macro-trading/how-monetary-policy-moves-commodities-real-rates-gold) — the financing leg of the carry is the interest rate, which policy sets.

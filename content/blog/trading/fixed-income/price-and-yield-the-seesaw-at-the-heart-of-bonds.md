---
title: "Price and yield: the seesaw at the heart of bonds"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Why a bond's price and its yield always move in opposite directions, how much they move, and why a longer or lower-coupon bond moves more — built from a single $1,000 bond."
tags: ["fixed-income", "bonds", "price-yield", "yield-to-maturity", "discounting", "interest-rates", "duration", "convexity", "treasuries"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **The one idea:** a bond pays a *fixed* stream of cash, so when market interest rates change, the only thing that can adjust is the bond's *price* — and it adjusts in the opposite direction. That is the seesaw at the heart of all of fixed income.
> - A bond's price is just the present value of its future coupons and principal, discounted at the market's required yield.
> - When the required yield goes **up**, the price goes **down**, and vice versa. This is not a rule of thumb; it is arithmetic.
> - **Yield to maturity** is the single discount rate that makes that present value equal the price you pay — the bond's internal rate of return if you hold it to the end.
> - Price above par means you locked in a *premium* (your coupon beats the market); below par means a *discount* (your coupon trails it).
> - A **1% move in rates is not a 1% move in price.** The size of the price move depends on maturity and coupon — the seed of *duration*, the most important number in the whole field.

Suppose you buy a brand-new bond today. It is a clean, simple deal: you hand over \$1,000, and in return the issuer promises to pay you \$40 a year for five years and then give your \$1,000 back. That \$40 is 4% of \$1,000, so we call it a 4% coupon. Nothing about that contract will ever change. The \$40 checks will arrive on schedule, and the \$1,000 will come back at the end, no matter what happens in the world.

Now imagine that one week later, every *new* bond being issued pays 5% instead of 4%. The world's interest rate has nudged up. Your bond still pays only \$40 a year. If you wanted to sell it, who would pay you the full \$1,000 for a bond that pays \$40 when they could buy a fresh one that pays \$50? Nobody. To sell, you would have to drop your price until your bond's *return to the buyer* matches the 5% they can get elsewhere. Your bond just lost value — not because the issuer did anything wrong, but because the rest of the world got a better deal.

That is the entire drama of fixed income in one example. A bond's promised cash is frozen. The market's required return floats. When the two disagree, the bond's **price** moves to reconcile them — and it always moves *against* the rate. Rates up, price down. Rates down, price up. They sit on opposite ends of a seesaw.

![A seesaw showing that when market interest rates rise the price of an existing bond falls, and when rates fall the price rises](/imgs/blogs/price-and-yield-the-seesaw-at-the-heart-of-bonds-1.png)

The diagram above is the mental model to carry through this whole post: rate on one side, price on the other, a fixed bond as the fulcrum. Press down on rates and price pops up; press down on price and the bond's effective yield pops up. Everything that follows — *par, premium, discount, yield to maturity, the curved price–yield line, duration* — is just a more precise way of describing that one seesaw. We will build it from zero, ground every claim in a running \$1,000 bond, and finish by showing why a 1% rate move can mean a tiny price wiggle on one bond and a brutal 20% loss on another.

If you want the policy-and-macro view of why "the price of money" matters for the whole economy, the sibling post [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) takes that angle. Here, we stay inside the bond and watch the mechanism work.

## Foundations: the words you need before the seesaw makes sense

Before we can be precise, we have to agree on a handful of terms. None of them are hard. Each one is just a name for something you already understand intuitively.

A **bond** is a tradable loan. When a government or a company needs money, instead of borrowing from a single bank it can borrow from thousands of investors at once by selling bonds. Each bond is a slice of that loan, written down as a contract. You, the investor, are the lender. The issuer is the borrower.

The **face value** (also called **par value** or **principal**) is the amount printed on the bond — the sum the issuer promises to repay at the end. The market standard we will use is **\$1,000**. This is the loan amount that gets returned to you on the final day.

The **coupon** is the fixed interest the bond pays each year, quoted as a percentage of face value. A 4% coupon on a \$1,000 bond pays \$40 a year. (The name is literal: old paper bonds had little coupons you clipped off and mailed in to collect each payment.) The crucial word is *fixed* — once the bond is issued, the coupon never changes, which is exactly why the price has to do all the adjusting later.

The **maturity** is when the loan ends and the face value comes back. A "5-year bond" returns your \$1,000 five years from issue. We will mostly use a 5-year bond, then stretch it to 30 years to show how maturity changes everything.

The **market interest rate** — sometimes called the **required yield** or the **prevailing rate** — is the return investors currently demand to lend money for that length of time and risk. It is set by the wider market: by the central bank's policy rate, by inflation expectations, by how badly everyone wants safe assets. This is the number that floats. When people say "rates went up", this is the rate they mean.

The **price** is what the bond actually trades for *right now*, which can differ from its \$1,000 face value. Quoting convention: bond prices are usually expressed per 100 of face value, so a price of "98.50" means 98.5% of face, or \$985 on a \$1,000 bond. We will use dollars to keep it concrete.

The **yield** is the return you actually earn given the price you pay. Buy a \$40-a-year bond for \$1,000 and your yield is 4%. Buy that same \$40-a-year bond for \$800 and your yield is higher than 4%, because you are getting the same \$40 on a smaller outlay. Price and yield are two views of the same trade — which is the seesaw, stated plainly.

One more unit, because traders live in it: a **basis point** is one-hundredth of a percentage point — 0.01%. So when "the 10-year yield rose 40 basis points," it climbed 0.40%, say from 4.10% to 4.50%. Traders use basis points because the moves that matter are often small fractions of a percent.

With those eight words — bond, face value, coupon, maturity, market rate, price, yield, basis point — you have everything you need. Now we make the seesaw exact.

## Why the price has to move: a bond is a frozen promise in a moving world

Hold the two halves of the contract side by side. On one side: a fixed schedule of cash. On the other: a market rate that drifts. The whole mechanism falls out of the fact that one side cannot move, so the other must.

Take our running example — call it the **Northwind 4s of 2031**, a \$1,000 par, 4% coupon, 5-year bond issued by a fictional company called Northwind Corp. ("4s of 2031" is bond shorthand: a 4% coupon maturing in 2031.) Its promise, written out, is just a list of dated payments:

| Year | Payment | What it is |
|---|---|---|
| 1 | \$40 | coupon |
| 2 | \$40 | coupon |
| 3 | \$40 | coupon |
| 4 | \$40 | coupon |
| 5 | \$1,040 | coupon + \$1,000 principal |

That list never changes. Not when the Fed hikes, not when inflation spikes, not when Northwind's stock doubles. The issuer owes those exact dollars on those exact dates.

Now, what is that list *worth* today? That is the only open question, and the answer depends entirely on what return the market currently demands. If the market is happy with 4%, the list is worth exactly \$1,000 — because 4% is precisely the coupon, so a buyer paying \$1,000 earns 4% and is satisfied. But the moment the market wants 5%, that same list is worth *less* than \$1,000, because to earn 5% on a bond that only pays \$40 a year, you have to buy it for less than face. And if the market will accept 3%, the list is worth *more* than \$1,000, because a 4%-paying bond is now generous and buyers will pay up for it.

The cash is frozen. The required return floats. The price is the variable that absorbs the difference. That is why bond prices move at all — and why they always move opposite to rates.

#### Worked example: the same bond at three different market rates

Let's make "worth less" and "worth more" into actual dollars. We will price the Northwind 4s — \$40 a year for five years, plus \$1,000 back at year five — at three different market rates. To price a bond, we discount each future payment back to today (the next section explains exactly how); for now, just watch what the market rate does to the answer.

- **Market rate = 4%** (equal to the coupon). Price ≈ **\$1,000.00**. You pay face, you earn the coupon, everyone is even. This is *par*.
- **Market rate = 5%** (above the coupon). Price ≈ **\$956.71**. The market wants 5% but the bond only pays 4%, so the price drops \$43 below face to make up the gap. Your \$40 coupons plus a \$43 "bonus" at maturity (buy at \$957, get \$1,000 back) together work out to a 5% return. This is a *discount*.
- **Market rate = 3%** (below the coupon). Price ≈ **\$1,045.80**. The bond's 4% coupon now beats the market's 3%, so buyers compete and bid the price \$46 above face. You pay \$1,046, collect generous \$40 coupons, and accept getting only \$1,000 back at the end — netting 3%. This is a *premium*.

Notice the direction every single time: rate **up** to 5% pushed price **down** to \$957; rate **down** to 3% pushed price **up** to \$1,046. The seesaw is not a metaphor, it is the arithmetic of discounting.

![A before and after comparison of a one thousand dollar four percent bond, on the left priced at par when the market rate is four percent and on the right repriced to about nine hundred fifty seven dollars when the market rate rises to five percent](/imgs/blogs/price-and-yield-the-seesaw-at-the-heart-of-bonds-3.png)

The figure puts the 4%-versus-5% case side by side: the same five payments, the same \$40 coupons and \$1,000 principal, but the *left* panel discounts them at 4% (summing to \$1,000, par) and the *right* panel discounts them at 5% (summing to \$956.71, a discount). The cash didn't change; only the discount rate did, and the price absorbed the entire difference — a \$43.29 drop. That \$43.29 is not arbitrary: it is exactly the amount that turns a 4%-coupon bond into a 5% return for the buyer. The market is, in effect, handing the new buyer a discount large enough to make up for the stingy coupon.

A useful way to feel this: the price gap *is* the make-good. The bond underpays by 1% a year (\$10 a year versus the \$50 a fresh 5% bond pays) for five years, but the gap isn't a flat \$50 — it is the *present value* of those shortfalls, which discounting compresses to about \$43. Whenever a bond's coupon trails the market, the price falls by just enough to compensate the buyer for the shortfall, dollar for present-value dollar.

*The same frozen cash flows are worth more when the market is stingy with returns and less when the market is generous — and the bond's price is what carries that difference, down to the cent.*

## The engine under the seesaw: discounting

To see *why* the math always tips the seesaw that way, we need the one idea that powers all of bond pricing: **the time value of money**. A dollar today is worth more than a dollar next year, because a dollar today can be invested and grow. So a payment arriving in the future is worth *less* than its face amount today — and the further away it is, the less it is worth right now.

The tool that converts a future dollar into its value today is **discounting**. If the market rate is $r$, then a dollar arriving in one year is worth $\frac{1}{1+r}$ today, a dollar in two years is worth $\frac{1}{(1+r)^2}$, and so on. In general, a payment $C$ arriving in $t$ years has a **present value** of:

$$PV = \frac{C}{(1+r)^t}$$

- $C$ — the future cash payment (a coupon, or the principal)
- $r$ — the market's required annual return (the discount rate)
- $t$ — the number of years until the payment arrives
- $PV$ — what that future payment is worth in today's dollars

A bond is just a bundle of these payments, so the bond's price is the sum of the present values of every coupon plus the principal:

$$P = \sum_{t=1}^{N} \frac{C}{(1+r)^t} + \frac{F}{(1+r)^N}$$

- $P$ — the bond's price today
- $C$ — the annual coupon in dollars (here \$40)
- $F$ — the face value repaid at maturity (here \$1,000)
- $N$ — the number of years to maturity (here 5)
- $r$ — the market's required yield

Stare at where $r$ sits in that formula: in the **denominator**, raised to a power. Make $r$ bigger and every denominator gets bigger, so every present value gets smaller, so the price $P$ gets smaller. Make $r$ smaller and every denominator shrinks, so each present value swells, so $P$ rises. The inverse relationship between price and yield is not a market behavior we observe and hope holds — it is baked into the structure of the formula. $r$ is downstairs; push it up and you push the price down.

![The discounting pipeline showing each future coupon and the principal divided by one plus the market rate raised to its year, then summed into the bond price](/imgs/blogs/price-and-yield-the-seesaw-at-the-heart-of-bonds-7.png)

The figure traces the machine: each dated payment on the left goes through its own discount factor (divide by $(1+r)^t$), and the discounted pieces pile up on the right into the price. Because the discount rate $r$ lives in every denominator, turning that one dial moves the whole stack the opposite way. That is the seesaw, drawn as plumbing.

For the deep mechanics of pricing — clean vs dirty price, accrued interest, day-count conventions — the sibling post [discounting cash flows: how a bond is priced](/blog/trading/fixed-income/discounting-cash-flows-how-a-bond-is-priced) goes step by step. If you want the heavier, formal treatment, [bond pricing](/blog/trading/quantitative-finance/bond-pricing) has the full math. Here we keep just enough to see the seesaw turn.

#### Worked example: discounting the Northwind 4s at 5%, payment by payment

Let's actually grind the \$956.71 from earlier so the formula stops being abstract. Market rate $r = 5\%$, so each year's payment gets divided by $(1.05)$ raised to that year:

| Year | Payment $C$ | Discount factor $1/(1.05)^t$ | Present value |
|---|---|---|---|
| 1 | \$40 | 0.9524 | \$38.10 |
| 2 | \$40 | 0.9070 | \$36.28 |
| 3 | \$40 | 0.8638 | \$34.55 |
| 4 | \$40 | 0.8227 | \$32.91 |
| 5 | \$1,040 | 0.7835 | \$814.87 |
| | | **Sum = price** | **\$956.71** |

Three things to notice. First, the sum is \$956.71 — exactly the discount price from before, now built from its parts. Second, the far-off year-5 payment of \$1,040 contributes the lion's share (\$814.87), because principal dwarfs a single coupon — which is a hint about *why* longer bonds are more rate-sensitive (more of their value sits in a distant, heavily-discounted payment). Third, redo this with $r = 4\%$ and every factor rises, the sum climbs back to \$1,000.00; redo it with $r = 3\%$ and it climbs to \$1,045.80. Same payments, one dial.

*A bond's price is nothing more than the sum of its discounted payments, and because the discount rate sits in every denominator, raising it lowers every piece — and the total.*

### A real-world wrinkle: most bonds pay twice a year

We have been using one neat \$40 coupon a year because it keeps the arithmetic clean. Real U.S. Treasuries and most corporate bonds pay **semiannually** — half the coupon every six months. Our 4% bond would pay \$20 every six months for ten periods, not \$40 every year for five. The seesaw doesn't change one bit, but the bookkeeping does: you discount ten half-year payments using a half-year rate, and the convention for quoting the yield is to take the per-period rate and double it (a "bond-equivalent yield").

This wrinkle is small but worth seeing once, because it is why a bond's quoted yield and its true compounded annual return differ slightly, and why pricing in a spreadsheet uses periods, not years. Here is the whole calculation as a few lines of runnable Python — the same formula, just looped over periods:

```python
def price_bond(face, coupon_rate, years, market_rate, freq=2):
    """Price a vanilla bond. freq=2 means semiannual coupons (the U.S. norm)."""
    n = years * freq                      # total number of payments
    c = face * coupon_rate / freq         # cash per coupon (e.g. $20)
    r = market_rate / freq                # per-period discount rate
    price = 0.0
    for t in range(1, n + 1):
        cash = c + (face if t == n else 0)   # add principal at the last period
        price += cash / (1 + r) ** t
    return price

print(price_bond(1000, 0.04, 5, 0.04))   # our running bond, semiannual -> 1000.00 (par)
print(price_bond(1000, 0.04, 5, 0.05))   # -> 956.24   (discount)
print(price_bond(1000, 0.04, 5, 0.03))   # -> 1046.11  (premium)
```

Notice the semiannual prices (\$956.24, \$1,046.11) are within a dollar of our annual ones (\$956.71, \$1,045.80). The compounding frequency nudges the second decimal; it does not touch the direction of the seesaw. Throughout this post we will keep using the clean annual numbers, because the *mechanism* — rate up, price down — is identical either way, and the round numbers are easier to hold in your head. Just know that when you price a real Treasury, you are looping over half-years, not years.

*Real bonds pay twice a year and the math runs in half-year periods, but the inverse price–yield relationship is exactly the same — frequency moves the pennies, not the seesaw.*

## Yield to maturity: reading the seesaw backwards

So far we have gone from a known market rate to a price. In real life, the market quotes you the *price*, and you want to know the *yield* — the actual return you would earn buying at that price and holding to the end. That number is the **yield to maturity (YTM)**, and it is the single most important quote in fixed income.

YTM is defined as the one discount rate that makes the present value of all the bond's cash flows equal to its current market price. It is the same pricing formula, run backwards: instead of plugging in $r$ to get $P$, you take the observed $P$ and solve for the $r$ that fits.

$$P_{\text{market}} = \sum_{t=1}^{N} \frac{C}{(1+y)^t} + \frac{F}{(1+y)^N}$$

- $P_{\text{market}}$ — the price the bond actually trades at
- $y$ — the yield to maturity (the unknown we solve for)
- everything else as before

There is no clean algebraic formula to isolate $y$ — you find it by trial and error or with a solver (a spreadsheet's `YIELD` or `IRR` function, a calculator, or a few rounds of guessing). But conceptually it is simple: **YTM is the internal rate of return on the bond if you buy at today's price and hold to maturity, collecting and reinvesting every coupon at that same rate.** It folds three sources of return into one number: the coupons you collect, any pull-to-par gain or loss (the difference between what you pay and the \$1,000 you get back), and the assumed reinvestment of coupons.

This is why "yield" and "price" are interchangeable ways of quoting a bond. Tell a trader the price and they know the yield; tell them the yield and they know the price. They are the two ends of the seesaw, and YTM is the precise hinge that connects them.

The relationship between coupon, price, and yield collapses into a clean rule of thumb you can memorize:

| If the bond trades at... | then its YTM is... | and we call it a... |
|---|---|---|
| **par** (\$1,000) | **equal** to the coupon (4%) | par bond |
| a **discount** (below \$1,000) | **above** the coupon (>4%) | discount bond |
| a **premium** (above \$1,000) | **below** the coupon (<4%) | premium bond |

The logic is the seesaw again. A discount price means you pay less than \$1,000 but still get \$1,000 back, so you earn the coupon *plus* a capital gain — your yield beats the coupon. A premium price means you pay more than \$1,000 and only get \$1,000 back, so you earn the coupon *minus* a capital loss — your yield trails the coupon. (There are several other "yields" — current yield, yield to call, yield to worst — that mean different things; the sibling post [the many yields](/blog/trading/fixed-income/the-many-yields-current-yield-ytm-and-yield-to-call) sorts them out. YTM is the one that matters most.)

#### Worked example: the same bond bought at \$957 vs \$1,046

Two investors buy the Northwind 4s on the same day at different prices, and we read their yields.

**Investor A** buys at the discount price, \$956.71. She pays \$956.71, collects \$40 a year for five years (\$200 total), and gets \$1,000 back. Her total cash in is \$956.71; her total cash out is \$1,200, of which \$43.29 is a capital gain on top of \$200 of coupons. Solve the pricing formula for the rate that fits \$956.71 and you get **YTM = 5.00%** — above the 4% coupon, exactly as the table promises. *She bought cheap, so she earns more than the coupon.*

**Investor B** buys at the premium price, \$1,045.80. He pays \$1,045.80, collects the same \$40 a year (\$200 total), but gets only \$1,000 back — a \$45.80 capital loss that eats into his coupons. Solve for the fitting rate and you get **YTM = 3.00%** — below the coupon. *He paid up, so he earns less than the coupon.*

Same bond, same \$40 coupons, same \$1,000 at the end. The only difference is the price each paid, and that price is what set each one's yield. Price and yield are one trade seen from two sides.

*Yield to maturity is the price, translated into the language of return — the single rate that reconciles what you pay with everything the bond will pay you.*

### Pull to par: why the seesaw stops mattering at the end

There is a quiet feature of YTM that resolves a lot of confusion: **no matter what the price does in between, a bond converges to exactly its face value as maturity approaches.** This is called the **pull to par**. On the final day, the bond is worth precisely \$1,000, because that is what the issuer pays and there is no future left to discount. So a discount bond bought at \$957 must climb back to \$1,000 over its life, and a premium bond bought at \$1,046 must drift down to \$1,000. The price isn't random noise around a fixed number; it is on a leash that gets shorter every day, tied to par at the end.

Why does this matter? Because it is the precise reason a buy-and-hold investor earns the YTM they locked in, regardless of the price swings along the way. If you buy at a 5% YTM and hold to maturity, you *get* 5% (assuming no default and that you can reinvest coupons near that rate). The interim losses on your statement — caused by rates rising after you bought — are real marks, but they reverse themselves as the price gets pulled back to par. The seesaw rocks hardest in the middle of a bond's life and goes still at the end.

#### Worked example: a discount bond climbing back to par

Investor A bought the Northwind 4s at \$956.71 to yield 5%. Suppose market rates stay at 5% for her whole holding period. Watch her bond's price each year, holding the yield fixed at 5% and just shortening the remaining maturity:

| Years left | Price at 5% yield | Year's price change |
|---|---|---|
| 5 | \$956.71 | — |
| 4 | \$964.54 | +\$7.83 |
| 3 | \$972.77 | +\$8.23 |
| 2 | \$981.41 | +\$8.64 |
| 1 | \$990.48 | +\$9.07 |
| 0 | \$1,000.00 | +\$9.52 |

Even with the *yield never changing*, her bond's price rose every year — from \$956.71 to \$1,000 — purely because of the pull to par. That \$43.29 capital gain, spread over five years, is exactly the "extra" return on top of her \$40 coupons that lifts her total return from the 4% coupon to the 5% YTM. The discount wasn't a loss to be feared; it was a built-in gain she collected as the leash pulled the price home.

*A bond's price is on a leash to par that tightens as maturity nears — which is why holding to maturity delivers the yield you bought, no matter how violently the price swings in between.*

## The price–yield curve: the seesaw drawn as a line

We have priced our bond at 3%, 4%, and 5%. If we price it at *every* possible yield from, say, 0% up to 10%, and plot price on the vertical axis against yield on the horizontal axis, we get the single most important picture in fixed income: the **price–yield curve**.

![The price-yield curve for a bond, sloping downward and curving, with the par point marked where price equals one thousand and zones labeled premium above par and discount below par](/imgs/blogs/price-and-yield-the-seesaw-at-the-heart-of-bonds-2.png)

Three features carry the entire chapter. First, the curve **slopes down**: every step right (higher yield) is a step down (lower price). That downward slope *is* the seesaw — the inverse relationship, drawn. Second, the curve **passes through par**: where yield equals the 4% coupon, price equals exactly \$1,000. Everything above that point (lower yields) is the *premium* zone, where price > \$1,000; everything below it (higher yields) is the *discount* zone, where price < \$1,000. Third — and this is the subtle, important one — the curve is **not a straight line; it bows.** It is steeper on the left (low yields) and flatter on the right (high yields). It curves toward the origin. That curvature has a name, **convexity**, and it has real consequences we will come back to.

Why does it curve instead of being a straight diagonal? Because of those denominators again. At low yields, the discount factors are large and a small change in $r$ swings them a lot; at high yields, the factors are already small and the same change in $r$ moves them less. The bond's price falls as yield rises, but it falls at a *decelerating* rate — so the line bends. A straight-line approximation (which is essentially what *duration* gives you) is a decent guess for small moves but increasingly wrong for big ones, precisely because the truth is curved.

#### Worked example: walking down the price–yield curve in 1% steps

Let's tabulate the actual curve for the Northwind 4s so the shape is concrete, not just described:

| Yield | Price | Change from previous |
|---|---|---|
| 1% | \$1,145.81 | — |
| 2% | \$1,094.27 | −\$51.54 |
| 3% | \$1,045.80 | −\$48.47 |
| 4% | \$1,000.00 | −\$45.80 |
| 5% | \$956.71 | −\$43.29 |
| 6% | \$915.75 | −\$40.96 |
| 7% | \$876.95 | −\$38.80 |

Read down the "change" column. Each 1% rise in yield drops the price, but by a *shrinking* amount: \$51.54, then \$48.47, then \$45.80, then \$43.29… The drops get smaller as yields rise. That decay is the convexity — the curve flattening out. If the relationship were a straight line, every step would be the same size. It isn't, and that gap between "straight-line guess" and "curved reality" is exactly what duration and convexity measure. The full treatment lives in [convexity: why duration is not the whole story](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story); here, just notice that the seesaw is curved, and the curve is friendly — prices fall slower than the straight line would predict, and rise faster.

That "friendly" word hides something genuinely useful, so let's make it precise. Because the curve bows toward the origin, the price *gains* from a yield drop are slightly larger than the price *losses* from an equal-sized yield rise. Look at the table around par: cutting the yield from 4% to 3% gains \$45.80, while raising it from 4% to 5% loses only \$43.29. Same 1% move, but the upside is bigger than the downside — by \$2.51. That asymmetry is convexity working *for* the bondholder, and it is why convexity is often described as a free good: for the same duration, a more convex bond gives you a little extra on the way up and loses a little less on the way down. (The notable exception is mortgage-backed bonds, whose convexity flips negative because homeowners refinance when rates fall — covered later in the series.) For now, the lesson is that the seesaw is not symmetric: the curve tilts the odds gently in the holder's favor, and the further you are from par, the more that curvature matters.

*The price–yield curve turns the seesaw into a picture: a downward, bowed line that crosses par exactly where the yield meets the coupon — and bows just enough that gains outrun losses.*

## The influence channel: rates move, and bond prices move against them — every day

So far we have changed the market rate by hand. In the real world, that rate moves on its own, all day, every day — driven by the Federal Reserve, by inflation data, by growth, by global flows of safe money. The benchmark everyone watches is the **10-year U.S. Treasury yield**, the closest thing finance has to a "price of money" for the medium term. When that yield moves, the price of every existing bond moves against it, mechanically, through exactly the discounting we just built. This is the bond market's heartbeat.

![A two-line time chart showing the market yield rising and falling while an existing bond's price moves in the opposite direction, illustrating the inverse correlation between yields and bond prices over time](/imgs/blogs/price-and-yield-the-seesaw-at-the-heart-of-bonds-4.png)

The figure shows the two lines over an illustrative stretch of time: the market yield (the line that rises) and the price of an already-issued bond (the line that mirrors it downward). Where the yield climbs, the price sinks; where the yield rolls over and falls, the price recovers. The two lines are near-perfect mirror images — that mirroring *is* the correlation. A bond and the prevailing interest rate have a correlation close to −1: when one zigs, the other zags. This is why a total-return bond index falls in a rising-rate year even though the underlying bonds are paying their coupons exactly as promised. The coupons are fine; the *market value* of the claim on those coupons drops because the discount rate underneath it rose.

This is the influence thread of the whole series. Bonds are the price of money, and that price is set on the seesaw. When the 10-year yield rises, mortgage rates rise (a mortgage is priced off the same curve), the present value of every future corporate cash flow falls (so stocks tend to wobble), and the dollar's pull on global capital changes. The little inverse relationship inside one \$1,000 bond, scaled up to a \$28-trillion Treasury market and a \$130-trillion global bond market, is one of the most powerful forces in finance. For how that reaches your mortgage and the real economy, see [from the 10-year yield to your mortgage](/blog/trading/fixed-income/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates); for the allocation view of bonds as the risk-free anchor, see [government bonds: the risk-free anchor and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration).

There is one more layer of correlation worth naming, because it is the part that surprises people. It is not only that *one bond* moves against rates — it is that *every* bond moves against the *same* rate at once, in lockstep. The discount rate isn't a private number attached to your bond; it is the market's required return for that maturity and credit, shared across the whole market. So when the 10-year yield jumps, the price of every 10-year-ish bond on earth gets marked down together, in the same direction, on the same day. That is why "the bond market" can have a good day or a bad day as a single thing, and why bond returns within a maturity bucket are so highly correlated with each other: they all hang from the same discount rate, and that one rate is doing the moving. The corollary is that bonds give you very little diversification *against other bonds* of similar maturity — to diversify a bond portfolio you have to vary maturity (the first lever) or credit, not just buy more bonds. The cross-asset angle — how this same rate also drives stocks and the 60/40 portfolio — is taken up in [stock–bond correlation: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine).

#### Worked example: a 40-basis-point day on a real-sized position

Suppose you run a \$10,000,000 position in a 5-year Treasury-like note, and overnight the relevant yield rises 40 basis points (0.40%) — a big but not extraordinary one-day move, the kind that happens around a hot inflation print. A 5-year bond has a price sensitivity (its modified duration, which we will meet properly in a later post) of roughly 4.5; that means a 1% rise in yield costs about 4.5% of price, so a 0.40% rise costs about $4.5 \times 0.40\% = 1.8\%$.

On \$10,000,000, that is a one-day mark-to-market loss of about **\$180,000** — gone, on a "safe" government bond, from a single data release, while every coupon was paid exactly on time. If instead the yield had *fallen* 40 bps, you would have *made* roughly \$180,000. Same instrument, same coupons, opposite outcomes, all driven by which way the seesaw tipped.

*"Safe" means free of default risk, not free of price risk: a Treasury can hand you a six-figure daily swing purely because the discount rate underneath it moved.*

## Not all bonds tip the same: maturity is the first lever

Here is where the seesaw gets interesting. We have been treating "the price moves against the rate" as if it were one fixed amount. It isn't. *How much* the price moves for a given rate change depends on the bond — and the two biggest levers are **maturity** and **coupon**. Get these, and you have understood the intuition behind duration before we ever define it.

Start with maturity. A longer bond's cash is locked away for longer, so more of its value sits in payments that are far in the future — and far-future payments are exactly the ones most sensitive to the discount rate, because they get divided by $(1+r)$ raised to a *big* power. Crank the rate up a notch and a payment 30 years out shrinks dramatically; the same notch barely touches a payment due next year. So a long bond's price swings far more than a short bond's for the identical rate change.

![Two price-yield curves on the same axes, a steep one for a thirty-year bond and a shallow one for a two-year bond, showing the long bond's price reacts much more to the same yield change](/imgs/blogs/price-and-yield-the-seesaw-at-the-heart-of-bonds-5.png)

The figure stacks two price–yield curves on the same axes: a steep one for a 30-year bond and a gentle one for a 2-year bond. For the *same* horizontal move in yield, the long bond's curve plunges; the short bond's barely dips. Same seesaw, very different leverage. This is why, in a rising-rate scare, the long end of the bond market gets crushed while T-bills barely flinch — and why investors who want to bet on rates falling buy long bonds, while those who want to hide from rate risk hug the short end.

#### Worked example: a 2-year vs a 30-year bond, both hit with the same +1%

Take two 4%-coupon bonds, both starting at par (\$1,000), and raise the market yield from 4% to 5% on both.

- The **2-year** bond falls from \$1,000 to about **\$981.41** — a loss of roughly **1.86%**.
- The **30-year** bond falls from \$1,000 to about **\$846.28** — a loss of roughly **15.37%**.

Same coupon, same starting price, same 1% rate move. The 2-year shrugs it off with a 2% scratch; the 30-year takes a brutal 15% hit. The only difference is how long the cash is locked away. *Maturity is leverage on the seesaw — the longer the bond, the more violently its price reacts to the same change in rates.* This single fact is the reason long-term bond funds can lose 20–30% in a bad year and is the seed of [duration, the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income).

## The second lever: coupon size, and the extreme case of the zero

The other lever is the **coupon**. Two bonds can have the same maturity but very different price sensitivity if their coupons differ. The logic: a high coupon returns more of your money sooner (in fat early payments), so the bond's "center of gravity" is closer to today, and it behaves like a shorter bond. A low coupon pushes more of the value to the distant principal, so its center of gravity is further out, and it behaves like a longer bond — more sensitive to rates.

The extreme case makes it vivid: a **zero-coupon bond** pays no coupons at all. You buy it at a deep discount and get a single payment — the face value — at maturity. All of its value sits in one far-future payment, so a zero is the *most* rate-sensitive bond of its maturity. Nothing comes back early to cushion the blow.

![Two price-yield curves comparing a high-coupon bond with a shallow curve against a zero-coupon bond with a much steeper curve, showing the zero is far more sensitive to yield changes](/imgs/blogs/price-and-yield-the-seesaw-at-the-heart-of-bonds-6.png)

The figure contrasts a high-coupon bond (gentle curve, value returns early, less sensitive) with a zero-coupon bond of the same maturity (steep curve, all value at the end, most sensitive). For the same yield move, the zero's price moves the most of any bond at that maturity. This is why zero-coupon Treasuries (STRIPS) and long zero-coupon bond funds are the most ferocious rate bets available — they are pure, uncushioned duration.

#### Worked example: a 4% coupon bond vs a zero, both 10-year, both hit with +1%

Take two 10-year bonds, each priced to yield 4% today, and bump the yield to 5%.

- The **4%-coupon** 10-year bond falls from \$1,000 to about **\$922.78** — a loss of roughly **7.7%**.
- The **zero-coupon** 10-year bond (priced at \$1,000/$(1.04)^{10} \approx$ \$675.56 today) falls to about \$613.91 — a loss of roughly **9.1%**.

Same maturity, same starting yield, same 1% move. The zero loses more because none of its value comes back early — every dollar is parked in the single year-10 payment, the most rate-exposed spot on the schedule. *Lower coupons mean later money means a longer effective life means a bigger price swing — coupon is the second dial on the seesaw, and the zero turns it all the way up.*

## Putting both levers together: the four corners of rate sensitivity

Maturity and coupon combine. Pull them in the same direction and the effects compound. The most rate-sensitive bond you can build is a **long-maturity, low-coupon** (ideally zero) bond — everything locked away, nothing returned early. The least sensitive is a **short-maturity, high-coupon** bond — money back fast, in big early chunks. A useful 2×2 to keep in your head:

| | High coupon | Low / zero coupon |
|---|---|---|
| **Short maturity** | Least sensitive (T-bill-like; price barely moves) | Mildly sensitive |
| **Long maturity** | Moderately sensitive | **Most sensitive** (long zero; price whips around) |

This table is, in spirit, the entire concept of duration — a single number that blends maturity and coupon into one measure of "how hard does this bond's price react to rates?" When we formalize it later, the bottom-right corner will have the highest duration and the top-left the lowest. For now, the intuition is enough: *longer and lower-coupon means more sensitive; shorter and higher-coupon means more stable.*

It helps to have a single shortcut number, even before we define duration properly. For a quick mental estimate, **a bond's percentage price change is roughly its duration times the yield change, with a minus sign in front**: a duration-7 bond loses about 7% if yields rise 1%, and gains about 7% if yields fall 1%. Duration, in turn, is a little less than maturity for a coupon bond (the coupons pull the center of gravity in) and almost exactly equal to maturity for a zero (nothing pulls it in). So a 10-year 4% coupon Treasury has a duration around 8, and a 10-year zero has a duration of 10 — which is precisely why the zero in our last example fell more. You can do a surprising amount of bond reasoning with just this one rule and a rough sense of where a bond sits in the 2×2: estimate its duration, multiply by the rate move, flip the sign, and you have the price impact within a fraction of a percent for small moves. The exact mechanics — Macaulay versus modified duration, DV01, and how to hedge a real position in dollars — are the subject of [duration, the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income) and [modified duration and DV01](/blog/trading/fixed-income/modified-duration-and-dv01-measuring-and-trading-rate-risk).

#### Worked example: building a defensive vs an aggressive bond sleeve

Say you have \$1,000,000 and a view. You think rates are about to fall hard, and you want maximum upside if you are right.

- The **aggressive** play: put it in 30-year zero-coupon Treasuries (STRIPS). If the 30-year yield drops 1%, a 30-year zero can gain on the order of **25%+** — roughly \$250,000 — because its duration is near 30. If you are wrong and rates rise 1%, you lose about the same. It is a high-octane bet on the seesaw.
- The **defensive** play: put it in 1-year T-bills. If rates move 1% either way, your price barely flinches — under **1%**, a few thousand dollars — and you simply collect your yield and roll into the next bill. You have almost taken the seesaw out of the equation.

Same million dollars, same rate move, wildly different outcomes — chosen entirely by where you sit on the maturity-and-coupon grid. *You don't have to accept a bond's rate sensitivity; you choose it, by choosing maturity and coupon.*

## Common misconceptions

**"If rates rise 1%, my bond loses 1%."** No — and this is the single most common beginner error. A 1% rise in rates is *not* a 1% price loss. The price loss depends on the bond's maturity and coupon: roughly 2% for a 2-year, roughly 8% for a 10-year, roughly 15%+ for a 30-year, and 25%+ for a 30-year zero. The "1% for 1%" intuition comes from thinking of rates like a simple interest rate on a savings account; bonds amplify the move through duration. The right mental model is "price change ≈ −(duration) × (rate change)", and duration is usually a number between 2 and 30, not 1.

**"A bond's yield is fixed because the coupon is fixed."** The *coupon* is fixed; the *yield* is not. The coupon is a fixed dollar amount (\$40 a year). The yield is that amount measured against the *current price*, and the price moves all day. Buy the bond cheaper and your yield rises; the bond's quoted YTM in the market changes second by second as the price changes, even though no coupon ever changes.

**"A bond trading below par is a bad bond / a sign of trouble."** Not necessarily. A bond trades at a discount mostly because market rates rose *after* it was issued, making its older, lower coupon less attractive — which has nothing to do with the issuer's health. A perfectly safe U.S. Treasury issued at 2% in 2021 trades at a steep discount today simply because new Treasuries pay much more. Discount means "issued in a lower-rate world", not "defaulting". (Credit deterioration *can* also push a price down, but that shows up as a widening credit spread, a separate story covered in [credit risk](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back).)

**"If I just hold to maturity, the price moves don't affect me."** Half-true, and worth getting precise. If you hold a non-defaulting bond to maturity, you get exactly the YTM you locked in at purchase regardless of the price gyrations in between — the price always "pulls to par" as maturity approaches, because at maturity the bond is worth exactly its \$1,000 face. So the *interim* losses are paper losses if you truly never sell. But two real catches: you might be *forced* to sell (a bank needing liquidity, as in 2023), and you face reinvestment risk on the coupons. The price risk is real even for a buy-and-hold investor the moment circumstances make holding impossible.

**"Higher yield always means a better bond."** A higher yield means a higher *expected* return, but it usually means you are being paid for more risk — longer maturity (more rate risk) or weaker credit (more default risk). A 9% bond is not "better" than a 4% Treasury; it is a different risk. The seesaw explains the rate-risk part: the high-yield, long-duration bond that looks generous today will fall the hardest if rates rise. Yield without context is half the picture.

**"Bond math is just for institutions; it doesn't touch me."** The price–yield seesaw sets your mortgage rate, your car loan, the return on your savings, and the value of the bond funds in your retirement account. When the news says "the 10-year jumped", the present value of every future cash flow in the economy just got marked down a notch. You are on the seesaw whether you trade bonds or not.

## How it shows up in real markets

**The 2022 bond rout — the seesaw at full violence.** In 2022, U.S. inflation hit 40-year highs and the Federal Reserve hiked its policy rate from near zero to over 4% in a single year. The 10-year Treasury yield roughly doubled, from around 1.5% to around 3.9%. The seesaw did exactly what the formula promises: bond *prices* collapsed. The Bloomberg U.S. Aggregate Bond Index — the broad "safe" bond benchmark — fell about **13%**, its worst year in modern history. Long-dated Treasuries fell far more: the 20+ year Treasury index dropped roughly **30%**. Investors who believed "bonds are safe" learned the hard lesson of the maturity lever: safe from *default* is not safe from *duration*. Every coupon was paid; the prices still cratered, because the discount rate underneath them soared.

**Silicon Valley Bank, 2023 — the "hold to maturity" catch made real.** SVB had parked a huge share of its deposits in long-dated, low-coupon Treasuries and mortgage bonds bought when rates were near zero. When the Fed hiked through 2022, those bonds' market prices fell hard — a paper loss, in principle, since the bonds would pay par at maturity. But when depositors rushed to withdraw, SVB was *forced* to sell those bonds before maturity, crystallizing roughly **\$1.8 billion** in losses and triggering a bank run that ended the bank in days. This is the precise mechanism behind the "if I hold to maturity it doesn't matter" misconception: it only doesn't matter if you can actually hold. See [SVB and Credit Suisse: the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs) for the full anatomy.

**The 40-year bond bull market, 1981–2021 — the seesaw running in reverse for a generation.** In 1981, U.S. inflation and yields peaked — the 10-year Treasury yielded over **15%**. Then, for four decades, yields ground steadily downward to near zero by 2020. The seesaw means that a multi-decade *fall* in yields was a multi-decade *rise* in bond prices: long-bond investors enjoyed both fat coupons and relentless capital gains. It was one of the great bull markets in any asset, and it happened purely because the discount rate fell year after year. The 2022 rout was, in part, that bull finally reversing.

**The UK gilt / LDI crisis, 2022 — duration plus leverage.** In September 2022, a UK budget announcement spooked the bond market and gilt (UK government bond) yields spiked sharply in days. UK pension funds running "liability-driven investment" strategies held heavily leveraged long-duration bond positions; the violent price drop (the long-maturity lever at work) triggered margin calls, forcing them to sell gilts into a falling market, which pushed yields *higher still* — a doom loop. The Bank of England had to step in with emergency bond buying. The episode is a master class in the maturity lever: long-duration positions move so much that, with leverage on top, a few days of rising yields can threaten the financial system.

**Japan's decades near zero — the seesaw stuck at the bottom.** For most of the 2000s and 2010s, Japan's government bond yields sat near or even below zero. With yields already at the floor, the seesaw had little room to push prices higher — and Japanese bond investors earned almost nothing in yield while taking on enormous price risk if rates ever rose (which they began to in 2023–24). Japan shows the flip side of the seesaw: when yields are already at rock bottom, bonds offer thin income and asymmetric downside, because the only big move left for rates is *up*.

**TIPS and "real" rates — the seesaw on inflation-adjusted bonds.** Treasury Inflation-Protected Securities pay a coupon plus an inflation adjustment to principal, and their price moves against the *real* (inflation-adjusted) yield rather than the nominal one. In 2022, real yields rose sharply from deeply negative to clearly positive, and long-dated TIPS — high-duration, low-real-coupon instruments — fell hard right alongside nominal Treasuries. It is the same seesaw, just measured in real terms; the [real yields](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything) post takes that angle further.

## When this matters to you, and where to go next

You are on this seesaw whether or not you ever buy a bond directly. The mortgage rate you are quoted, the yield on your savings, the value of the bond funds in your 401(k) or pension, and even the level of the stock market (which is priced by discounting future earnings at a rate set in the bond market) all ride on the price–yield relationship you just learned. When you hear "the 10-year yield rose today," you now know the second half of the sentence the news leaves out: *and so the price of every existing bond fell, and the present value of every future dollar in the economy got marked down a little.*

This post is the foundation the rest of the series builds on. From here, the natural next steps:

- **The mechanics of pricing**, in full: [discounting cash flows: how a bond is priced](/blog/trading/fixed-income/discounting-cash-flows-how-a-bond-is-priced) and [why bond prices move when rates move — and by how much](/blog/trading/fixed-income/why-bond-prices-move-when-rates-move-and-by-how-much).
- **The other yields** you will see quoted: [current yield, YTM, and yield to call](/blog/trading/fixed-income/the-many-yields-current-yield-ytm-and-yield-to-call).
- **Duration**, the number that quantifies everything we did by hand here: [duration, the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income), and its curvature correction, [convexity](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story).
- **The wider world**: how the seesaw sets [the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) and reaches [your mortgage](/blog/trading/fixed-income/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates), and for the heavy math, [bond pricing](/blog/trading/quantitative-finance/bond-pricing).

Master the seesaw and you have the key to the entire bond market: a frozen promise in a moving world, with price as the variable that makes the two agree. (This is educational, not investment advice — it explains how bonds behave, not what you should buy.)

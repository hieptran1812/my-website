---
title: "The price of money: how bonds set every other price"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into the single idea that unifies all of finance: every asset is worth its future cash flows discounted at the risk-free rate plus a risk premium, so when the bond market moves the risk-free rate, stocks, real estate, credit, and the dollar all reprice at once."
tags: ["fixed-income", "bonds", "risk-free-rate", "discount-rate", "valuation", "10-year-yield", "cap-rate", "equity-risk-premium", "asset-pricing", "interest-rates"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Every asset on earth is worth the future cash it will throw off, discounted back to today at *the risk-free rate plus a premium for its risk* — and bonds set that risk-free rate, which is why the bond market quietly prices everything else.
>
> - The master equation of finance is simple: **value = future cash flows ÷ (risk-free rate + risk premium)**. The risk-free rate is the floor every investment is measured from, and the yield on safe government bonds *is* that floor. That is why we call bonds "the price of money."
> - The **US 10-year Treasury yield** is treated as the single discount rate for the whole system. Move it, and the denominator of the master equation moves for *every* asset at once — which is why a bond-market move can reprice stocks, houses, and currencies on the same afternoon.
> - A stock's fair price-to-earnings multiple, a building's value through its **cap rate**, a mortgage rate, a corporate bond's yield, and the level of the dollar are all just "the risk-free rate, plus a spread." We will revalue the *same* steady-cash-flow stock and the *same* rental property as the risk-free rate climbs from 2% to 5% and watch both lose roughly **40–60%** of their value.
> - Warren Buffett's line is the whole post in six words: **"interest rates are gravity for asset prices."** When the safe rate is near zero, gravity is weak and risky assets float to high valuations; when it rises, gravity strengthens and everything is pulled back down to earth.
> - Bonds set this baseline because the bondholder is the **senior claim** — paid first, with a fixed promise. Every riskier asset must out-promise the bond to attract a dollar, so the bond yield is the hurdle every other return is measured against.

Here is a fact that sounds impossible the first time you hear it. In late 2021, a share of a profitable, slow-growing company might trade at 30 times its annual earnings. Eighteen months later, with the company earning *exactly the same profit*, that same share might trade at 18 times earnings — a 40% collapse in price — and nothing about the business had changed. No lost customers, no scandal, no recession in its own revenue. What changed was somewhere else entirely: the yield on a government bond went from about 1.5% to about 4.5%.

How can the price of a bond — a sleepy, predictable IOU from the government — reach across the entire financial system and knock 40% off the value of a company it has nothing to do with? That is the question this post answers. And the answer is the deepest idea in all of finance, the one that, once you see it, makes the whole market click into place: **bonds are the price of money, and the price of money sets every other price.**

![A central risk-free rate at the hub of a wheel, with cash flows being discounted along spokes that fan out to a stock, a house, a corporate bond, and the dollar](/imgs/blogs/the-price-of-money-how-bonds-set-every-other-price-1.png)

The figure above is the mental model for the entire post, and indeed for this whole series. At the center sits one number — the risk-free rate, set by the bond market. Every other asset hangs off it. A stock, a building, a loan, a currency: each is worth its own future cash flows, but each of those cash flows is discounted back to today using the risk-free rate as the starting point. Move the center, and everything on the spokes reprices at once. This is post #34 of 42, and it is the payoff of the series' thesis. Every earlier post taught you a piece of the machine — what a bond is, how it's priced, what its yield means. This is where we step back and see what the machine *does* to the rest of the world. Let us build it from zero.

## Foundations: the one equation behind every price

Before we can watch bonds move stocks and houses, we need a shared vocabulary. A reader with no finance background needs four ideas firmly in place: what "cash flows" are, what "discounting" does, what a "risk premium" is, and what "risk-free" really means. A practitioner can skim this; a beginner cannot proceed without it.

### Every asset is a claim on future cash

Strip away the ticker symbols and the jargon, and **every investment is the same thing: a claim on money you will receive in the future.** A bond pays you coupons and then your principal back. A stock pays you a share of the company's profits, today or eventually, as dividends or as a higher resale value built on those profits. A rental property pays you rent, month after month, minus expenses. A startup pays you nothing for years and then, you hope, a flood of cash. Different shapes, same essence: you hand over money now in exchange for a stream of money later.

This means the question "what is this asset worth?" is always the same question: *what is a stream of future money worth to me today?* And the moment you ask that, you run into the single most important idea in finance — the **time value of money.**

### Discounting: why a dollar later is worth less than a dollar now

A dollar in your hand today is worth more than a dollar promised a year from now. Not because of inflation (that is a separate haircut), but because today's dollar can be put to work *risk-free* and grow. If you can earn 5% with no risk, then \$100 today becomes \$105 in a year. Run that backwards: a guaranteed \$105 a year from now is worth exactly \$100 today, because \$100 is all you'd need to set aside to reproduce it. We say the future \$105 has been **discounted** back to a **present value** of \$100.

The machine that does this is one line of arithmetic. To find the present value (PV) of a cash flow $C$ arriving in $t$ years, when the relevant interest rate is $r$:

$$PV = \frac{C}{(1+r)^{t}}$$

Here $C$ is the future cash, $t$ is how many years away it is, and $r$ is the **discount rate** — the return you could earn instead, which is exactly what you give up by tying your money in this asset. The rate $r$ lives in the denominator, and that single fact is the engine of this entire post: **when $r$ rises, the denominator gets bigger, so the present value falls.** Future money is worth *less today* when safe rates are *higher*, because the alternative — just earning the safe rate — got more attractive.

For an asset with many cash flows (coupons, dividends, rents) arriving over many years, you discount each one and add them up:

$$\text{Value} = \frac{C_1}{(1+r)^{1}} + \frac{C_2}{(1+r)^{2}} + \frac{C_3}{(1+r)^{3}} + \cdots$$

That sum *is* the value of the asset. This is called a **discounted cash flow** (DCF), and it is not a model for bonds, or for stocks, or for property — it is *the* model for all of them. (The dedicated post [discounting cash flows: how a bond is priced](/blog/trading/fixed-income/discounting-cash-flows-how-a-bond-is-priced) walks the bond case in full detail.)

### The risk-free rate and the risk premium

There is one more layer. Government bonds are treated as **risk-free** — meaning free of default risk, the chance you don't get paid back (not free of *all* risk; their prices still swing when rates move). When you discount the cash flows of a *safe* government bond, you use the risk-free rate $r_f$ and nothing else.

But a stock or a rental property is *not* safe. Tenants leave; profits fall; companies go bankrupt. To compensate you for taking that risk, a risky asset must offer a higher expected return — and so you discount its cash flows at a *higher* rate. That extra slice on top of the risk-free rate is the **risk premium**: the reward, per dollar, for bearing uncertainty instead of buying the safe bond. The full discount rate for any asset is therefore:

$$r = \underbrace{r_f}_{\text{risk-free rate}} + \underbrace{\text{(risk premium)}}_{\text{extra for this asset's risk}}$$

This is the master equation of valuation, and it is worth memorizing in plain English: **the value of anything is its future cash flows, discounted at the risk-free rate plus a premium for its risk.** A *basis point* — one hundredth of a percent, 0.01% — added to either piece ripples through the whole present value.

The risk-free rate $r_f$ is the *common* term in every asset's discount rate. The stock has its own premium, the property has its own, the junk bond has its own — but they *all* sit on top of the *same* $r_f$. That shared foundation is the secret. When $r_f$ moves, it doesn't move one asset's denominator; it moves *everybody's* denominator at once. And $r_f$ is set by the bond market. That is the entire mechanism of this post, and we will now watch it work.

### Where the risk-free rate itself comes from

One natural question before we go on: *why* is the risk-free rate the number it is? Why 2% in one year and 5% in another? The bond market sets it, but the bond market is itself weighing two things. The first is the **expected path of the central bank's policy rate** over the bond's life — if the market thinks the Fed will keep short-term rates around 4% for the next decade, the 10-year will sit near there. The second is **inflation**: a lender who will be repaid in dollars ten years from now demands compensation for the purchasing power those dollars will lose. So the risk-free rate is, roughly, *expected real growth in the economy plus expected inflation plus a small term premium* for tying money up for a long time. (The drivers are the subject of [what moves the yield curve: the Fed, growth, inflation, and supply](/blog/trading/fixed-income/what-moves-the-yield-curve-the-fed-growth-inflation-and-supply), and the policy lens is in [central-bank toolkit: rates, QE, QT, and forward guidance](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance).)

This matters for our story because it means the risk-free rate is not arbitrary — it carries information. When it rises because the economy is overheating and inflation is hot, it reprices every asset *and* signals that the central bank intends to cool things down. That double role — gravity *and* signal — is why a single bond-market number commands so much attention. For the rest of this post we'll take $r_f$ as given and trace its effects, but keep in the back of your mind that the number itself is the market's verdict on the future of growth and inflation.

### Real vs nominal: a quick but load-bearing distinction

One last term, because it lurks under every example. The yield quoted on a Treasury is a **nominal** rate — it includes expected inflation. Strip inflation out and you get the **real** rate, the true reward for waiting in terms of purchasing power. The relationship is roughly: nominal rate ≈ real rate + expected inflation. For most of this post we'll use nominal rates (that's what's quoted and what discounts nominal cash flows), but the deepest version of "the price of money" is the *real* rate — which is why allocators obsess over it (see [real yields: the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything)). When the real risk-free rate rises, it's an unambiguous tightening of gravity; when a nominal rate rises only because inflation expectations rose, the effect on assets is murkier. Keep the distinction in your pocket; we'll flag it where it changes the answer.

## Why the bond yield *is* the risk-free rate

We keep saying "the risk-free rate is set by the bond market," but where exactly does it come from? It is not a number a committee announces. It is, quite literally, the **yield on safe government bonds**, and above all the yield on the **US 10-year Treasury note** — a loan to the US government that gets paid back in ten years.

### Why government bonds set the floor

Recall that yield is the return you actually earn on a bond given its price (when the price falls, the yield rises — the seesaw at the heart of all fixed income, covered in [price and yield: the seesaw at the heart of bonds](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds)). The US government is treated as the safest borrower on earth: it has never failed to pay its dollar debt, and it borrows in a currency it alone can create. So the yield on its bonds is the closest thing the world has to a *pure* price of time — what money costs with no default risk attached. That number is the $r_f$ in the master equation. (See [US Treasuries: the risk-free benchmark of the world](/blog/trading/fixed-income/us-treasuries-the-risk-free-benchmark-of-the-world) for why the world settled on US debt as its yardstick.)

Why the *10-year* specifically, out of the whole range of maturities? Because it is the natural reference point for *long-lived* assets. A stock has cash flows stretching out indefinitely; a 30-year mortgage runs three decades; a building lasts generations. None of these are well-matched by a 3-month Treasury bill, whose rate the Fed controls almost directly. The 10-year sits in the sweet spot: long enough to reflect the market's view of growth and inflation over a meaningful horizon, short enough to be the deepest, most-traded long bond in the world. By convention and convenience, the 10-year yield became *the* discount rate of finance — the number every other return is quoted against.

### "Plus a spread" — the universal phrasing

Once you see this, you start noticing it everywhere. Listen to how finance professionals actually quote prices, and almost every one is phrased as *"the Treasury, plus a spread."*

- A corporate bond yields "the 10-year, plus 150 basis points" for its default risk.
- A 30-year mortgage rate is roughly "the 10-year, plus 150–200 basis points" for prepayment and credit risk.
- An emerging-market government borrows at "US Treasuries, plus a country-risk spread."
- A stock's *earnings yield* (profits ÷ price, the inverse of the P/E ratio) tends to sit "above the 10-year, by an equity risk premium."
- A commercial building's **cap rate** (its yearly net income ÷ its price) trades "above the 10-year, by a real-estate risk premium."

Every one of these is the master equation in disguise. The Treasury yield is the base; the spread is the risk premium. So when the base moves, every one of those prices is forced to move with it — unless the spread happens to move the other way by exactly the same amount, which it rarely does. *This is why the bond is the price of money:* it is the common term in every quote in the building.

#### Worked example: the same bond, repriced by a rate move

Let's ground the seesaw in a single number before we move on to bigger assets. Take a plain US Treasury note: \$1,000 par, a 4% coupon (so \$40 a year), 10 years to maturity. When it was issued, the 10-year yield was 4%, so it traded at exactly par: \$1,000. The market's required return *was* the coupon.

Now the 10-year yield jumps to 5%. New buyers can get \$50 a year of risk-free income elsewhere, so nobody will pay \$1,000 for your \$40-a-year bond. Its price has to fall until its yield matches the market's 5%. Discounting the ten \$40 coupons and the \$1,000 principal at 5% instead of 4% gives a price of roughly **\$923** — a drop of about 7.7% from a one-percentage-point rate move. Go the other way: if the 10-year falls to 3%, the same bond reprices *up* to about **\$1,085**.

*Even the safest, simplest asset reprices the instant the risk-free rate moves — and everything riskier sits downstream of this exact mechanism, only with more leverage to the move.*

## Rates are gravity: the centerpiece

Now the payoff. We have established that (1) every asset is its discounted future cash flows, and (2) the discount rate for all of them shares one common term, $r_f$, set by the bond market. Put those together and you get the single most important picture in finance: **the bond yield fans out and reprices everything.**

![The 10-year Treasury yield at the center fanning out to a stock P/E, a property cap rate, the mortgage rate, the credit spread, and the dollar, each shown as a downstream price that moves when the yield moves](/imgs/blogs/the-price-of-money-how-bonds-set-every-other-price-2.png)

The figure above is the heart of the post. One number on the left — the 10-year yield — feeds into the denominator of every asset on the right. Raise it, and stock multiples compress, cap rates rise (property prices fall), mortgage rates climb, credit gets more expensive, and the dollar tends to strengthen. Lower it, and the whole fan runs in reverse. The bond market doesn't *ask* these markets to move; the arithmetic of discounting *forces* them to.

### Buffett's gravity metaphor

Warren Buffett put it more memorably than any textbook. *"Interest rates are to asset prices like gravity is to the apple. They power everything in the economic universe."* And: *"When interest rates are 13%, the gravitational pull on asset values is enormous… When interest rates are very low, the gravitational pull is reduced."*

The metaphor is exactly right, and it is just the discounting equation in physical clothing. When the risk-free rate is near zero, the denominator of the master equation is tiny, so far-off future cash flows keep almost all their present value. Gravity is weak; risky assets — growth stocks whose payoff is decades away, speculative property, unprofitable startups — float up to dizzying valuations. When the risk-free rate rises, the denominator swells, far-off cash flows get crushed, and gravity yanks everything back down. The assets that float highest in low-gravity (long-dated, low-current-cash-flow assets) are also the ones that fall hardest when gravity returns — because their value lives furthest out in the future, where discounting bites hardest.

![A downward-sloping curve showing the present value of a fixed future cash flow falling as the discount rate on the horizontal axis rises from 0 to 8 percent](/imgs/blogs/the-price-of-money-how-bonds-set-every-other-price-3.png)

The figure above is the shape of gravity itself. On the horizontal axis is the discount rate; on the vertical axis is the present value of one fixed, far-off cash flow. The curve slopes *down* and to the right — higher rate, lower value — and it is convex: each additional percentage point of rate does more damage when rates are already high, but the steepest *proportional* damage to a long-dated cash flow comes from the first moves off zero. That curve is the single most important relationship in finance, and it is the reason every other figure in this post points the way it does. Notice it never touches zero — even at a punishing rate, a future dollar is worth *something* — but it falls relentlessly as the rate climbs.

Why convex, not a straight line? Because discounting compounds. Discounting a cash flow $t$ years out divides it by $(1+r)^t$, and that exponent makes the effect accelerate. This is the same curvature that bond traders call **convexity** — the reason a bond's price-yield relationship bends rather than running in a straight line (the full treatment is in [convexity: why duration is not the whole story](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story)). For our purposes the lesson is simpler: the longer-dated the cash flow, the steeper and more curved its line, so the more violently it reprices when the bond market moves the rate.

#### Worked example: gravity crushing a far-off dollar

Make the gravity concrete with a single cash flow. Suppose an asset will pay you exactly \$1,000, once, 20 years from now, and nothing in between (a "growth" asset — all the payoff is far in the future).

At a risk-free rate of 2%, that future \$1,000 is worth $1{,}000 / (1.02)^{20} \approx \$673$ today.

Raise the risk-free rate to 5%, and the same \$1,000 is worth $1{,}000 / (1.05)^{20} \approx \$377$ today — a **44% collapse** in value, with the cash flow *completely unchanged*.

Now compare a "value" asset that pays you \$1,000 next *year*. At 2% it's worth \$980; at 5% it's worth \$952 — barely a 3% drop. *The further out an asset's cash flows sit, the harder rising rates crush its present value: that is why a rate spike hammers long-duration growth assets and barely scratches near-term cash flows.* This is the same idea as **duration** in bonds — the longer the wait for your money, the more rate-sensitive its price (see [duration: the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income)).

## How a stock reprices when the risk-free rate moves

Let's now run the master equation on a stock, because this is where the "impossible fact" from the opening becomes obvious arithmetic.

### From earnings yield to the P/E multiple

A simple, durable company earns a steady profit each year. Forget growth for a moment and treat it as a perpetual stream: it earns $E$ dollars per share, forever. The value of a perpetual stream of $E$ dollars, discounted at rate $r$, is a famous one-line formula:

$$\text{Value} = \frac{E}{r}$$

where $r = r_f + \text{equity risk premium}$. Flip it over and you get the **price-to-earnings (P/E) multiple** — how many dollars of price you pay per dollar of annual earnings:

$$\text{P/E} = \frac{\text{Price}}{E} = \frac{1}{r}$$

So a stock's fair P/E is just *one divided by its discount rate.* And its discount rate is the risk-free rate plus the equity risk premium (ERP) — the extra return investors demand for holding stocks instead of safe bonds, historically averaging somewhere around 4–5%. The earnings yield (the inverse of the P/E) is therefore roughly "the 10-year, plus the ERP." Now you can see why the bond yield reaches straight into the stock market: it is sitting *inside* the P/E.

![A before and after of one stock's fair price, with the same earnings repriced from a high multiple at a 2 percent risk-free rate to a much lower multiple at a 5 percent risk-free rate](/imgs/blogs/the-price-of-money-how-bonds-set-every-other-price-4.png)

#### Worked example: a steady-earnings stock revalued from 2% to 5%

Meet **Steady Co.**, a boring, reliable company that earns \$5.00 per share every year and is expected to keep doing so. Investors demand an equity risk premium of 4% over the risk-free rate. We'll revalue it as the risk-free rate climbs.

*When the 10-year yield is 2%:* the discount rate is $r = 2\% + 4\% = 6\%$. The fair P/E is $1 / 0.06 \approx 16.7$, so the fair price is $\$5.00 \times 16.7 = \$83.3$ per share.

*When the 10-year yield rises to 5%:* the discount rate is now $r = 5\% + 4\% = 9\%$. The fair P/E is $1 / 0.09 \approx 11.1$, so the fair price is $\$5.00 \times 11.1 = \$55.6$ per share.

The earnings never changed — Steady Co. still earns \$5.00 a share. But its fair value fell from \$83.3 to \$55.6, a **33% drop**, purely because the risk-free rate rose three percentage points. *A stock's price is its earnings divided by a discount rate that has the bond yield baked inside it, so when the bond yield rises, the stock falls even though the business is identical.* This is the arithmetic behind the opening paragraph — and for a fast-growing company, whose earnings sit even further in the future, the effect is far larger.

### Why growth stocks fall hardest

Steady Co. dropped 33%. A high-growth company drops *much* more from the same rate move, and now you know why: its cash flows are concentrated far out in the future, where discounting bites hardest (the gravity example above). When you raise the discount rate on a stream that pays almost nothing for ten years and then a fortune, you crush the part that matters most. This is exactly why, in 2022, the profitless tech and "long-duration" growth names fell 50–80% while steady dividend payers and energy stocks barely moved or rose. Same rate move, wildly different gravity, because of *where* in time each company's cash flows lived. (The dedicated stock–bond post [bonds vs stocks: discount rates, the 60/40, and correlation](/blog/trading/fixed-income/bonds-vs-stocks-discount-rates-the-60-40-and-correlation) goes deeper on this link.)

There's a deeper way to say this that ties the whole series together: a growth stock has a long **duration**, exactly the way a long-maturity bond does. Duration, in fixed income, measures how far in the future an asset's cash flows sit, and therefore how sharply its price moves when rates change. A 30-year zero-coupon Treasury and a profitless growth stock both pay you almost nothing now and everything later — so both have enormous duration, and both get hammered when the risk-free rate rises. A short-dated bond and a high-dividend "value" stock both pay you mostly soon — short duration, low rate sensitivity. Once you see that *every* asset has a duration, the entire market sorts itself onto a single ruler: how far out are your cash flows, and therefore how hard does gravity pull on you when the bond market moves the rate? That single lens explains why the same rate shock can be a flesh wound for one portfolio and a catastrophe for another.

## How real estate reprices: the cap rate

Property is the easiest place to *see* the master equation, because real-estate professionals quote it directly, every day, in a single number: the **cap rate**.

### What a cap rate is

The capitalization rate, or **cap rate**, is a building's yearly net operating income (rent minus operating expenses) divided by its price:

$$\text{Cap rate} = \frac{\text{Net operating income}}{\text{Price}}$$

It is the property world's version of an earnings yield — the income return per dollar of price. And just like the earnings yield, it is fundamentally *"the risk-free rate, plus a real-estate risk premium"* (compensation for vacancy, illiquidity, maintenance, and tenant default). Flip the equation around and a building's value is its income *divided by* the cap rate:

$$\text{Price} = \frac{\text{Net operating income}}{\text{Cap rate}}$$

That is the master equation again, dressed in real-estate clothes. The cap rate is the discount rate; income is the cash flow. When the risk-free rate rises, cap rates are pulled up with it — and since the cap rate sits in the *denominator* of the price equation, a rising cap rate means a falling price, even if the rent never changes.

![A chart with the 10-year Treasury yield on the horizontal axis and the property cap rate on the vertical axis, showing the cap rate rising roughly in step with the bond yield above a steady risk-premium gap](/imgs/blogs/the-price-of-money-how-bonds-set-every-other-price-5.png)

The figure above shows the relationship traders take for granted: cap rates track the 10-year yield, riding a roughly constant risk-premium spread above it. The spread isn't perfectly fixed — in a boom it can compress, in a panic it widens — but the *direction* is reliable. Bond yields up, cap rates up, property values down.

#### Worked example: an apartment building revalued from 2% to 5%

Meet a small apartment building, **Maple Court**, that produces \$100,000 a year in net operating income. Real-estate investors in its market demand a risk premium of 3% over the risk-free rate.

*When the 10-year yield is 2%:* the cap rate is $2\% + 3\% = 5\%$. The building's value is $\$100{,}000 / 0.05 = \$2{,}000{,}000$.

*When the 10-year yield rises to 5%:* the cap rate is now $5\% + 3\% = 8\%$. The building's value is $\$100{,}000 / 0.08 = \$1{,}250{,}000$.

The rent collected didn't change by a dollar — Maple Court still nets \$100,000. But its value fell from \$2.0 million to \$1.25 million, a **38% loss**, purely because the bond market repriced money. *A building is worth its income divided by a cap rate that rides on top of the bond yield, so when the bond yield rises, property reprices down exactly like a long-duration bond.* This is not a thought experiment: it is precisely why commercial real estate values fell sharply across 2022–2023 as the 10-year tripled, even where buildings stayed fully rented.

## The risk-premium stack: the same base under every asset

We've now revalued three things — a Treasury, a stock, a building — and every time the *base* was the same risk-free rate and the only difference was the premium stacked on top. It's worth seeing that structure as a single picture, because it is the architecture of the whole financial system.

![A layered stack showing the risk-free Treasury rate as the base, with the credit spread, equity risk premium, real-estate premium, and emerging-market premium stacked on top to form each asset's total required return](/imgs/blogs/the-price-of-money-how-bonds-set-every-other-price-6.png)

Read the stack from the bottom up. At the base is the **risk-free rate** — the 10-year Treasury yield, the price of pure time with no default risk. On top of it, each asset adds its own premium: an investment-grade corporate bond adds a small credit spread; a high-yield ("junk") bond adds a fat one; a stock adds the equity risk premium; a building adds a real-estate premium; an emerging-market bond adds a country-risk premium. The *total height* of each column is that asset's required return — its discount rate.

The crucial insight is what happens when the *base layer* moves. If the risk-free rate rises by one percentage point, every single column gets one percentage point taller — *simultaneously*. The required return on every asset class goes up together, so the price of every asset class goes down together. This is why, in a sharp rate-rise year like 2022, stocks *and* bonds *and* real estate *and* credit *and* crypto can all fall at once, breaking the comforting old rule that "when stocks fall, bonds protect you." When the move comes from the shared base — the risk-free rate itself — there is nowhere downstream to hide. (The classic diversification engine and when it fails is the subject of [bonds vs stocks: discount rates, the 60/40, and correlation](/blog/trading/fixed-income/bonds-vs-stocks-discount-rates-the-60-40-and-correlation) and the allocation view in [stock–bond correlation: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine).)

### Why the bond, as the senior claim, gets to set the base

There's a reason the bond — not the stock — sits at the *base* of the stack and defines the hurdle for everyone else. It comes down to the order in which a company (or a government) pays out its cash.

When money flows out of a business, it goes in a strict order of seniority: lenders (bondholders) get paid *first*, in full, on a fixed schedule; only what's left over flows to the owners (equity). In a bankruptcy, the same order holds — bondholders are senior, shareholders are wiped out last and least. (This pecking order is the subject of [seniority, recovery, and the capital structure](/blog/trading/fixed-income/seniority-recovery-and-the-capital-structure).) The bondholder is taking the *least* risk of anyone in the capital structure, so the bond yield is the *lowest* required return — the floor. Everyone above the bondholder in risk must be paid *more* to compensate. That is what makes the bond the natural baseline: it is the safest claim, so its return is the one every riskier claim is measured against.

Put differently: the risk-free bond is the option every investor *always* has. You can simply lend to the government and earn $r_f$ with near-certainty. So no rational investor will take on a risky asset unless it promises *more* than $r_f$. The bond yield is the universal *opportunity cost* — the return you give up to hold anything else — which is exactly why it anchors the price of everything else. Raise the floor, and every ceiling above it has to rise too, or the risky asset isn't worth holding.

#### Worked example: the same dollar of income priced three ways

Take one dollar of *annual income* and price it as three different assets, all at a 2% risk-free rate, to see the stack in action.

- As a **risk-free perpetual** (a government claim): discount rate 2%, value $= \$1 / 0.02 = \$50$.
- As a **corporate bond's** coupon stream, with a 2% credit spread: discount rate $2\% + 2\% = 4\%$, value $= \$1 / 0.04 = \$25$.
- As a **stock's** earnings, with a 4% equity risk premium: discount rate $2\% + 4\% = 6\%$, value $= \$1 / 0.06 \approx \$16.7$.

The exact same dollar of yearly income is worth \$50, \$25, or \$16.7 depending only on how much risk premium sits on top of the risk-free base. *The risk-free rate sets the floor value of a dollar of income, and every risk premium is a discount the market applies for uncertainty — which is why safe income is the most expensive income on earth.*

## Two more spokes: mortgage rates and credit spreads

We've priced assets — things you own. But the same base rate also sets the price of the *loans* people take, because a loan is just a bond from the borrower's point of view. Two spokes on the wheel are loans rather than assets, and they're worth walking through because they're where the bond yield touches ordinary life and the corporate economy most directly.

### Mortgage rates: the 10-year, plus a spread

The 30-year fixed mortgage is the loan most households know, and its rate is one of the cleanest "Treasury plus a spread" relationships in finance. It tracks the **10-year Treasury yield** — not the Fed's overnight rate — because a 30-year mortgage, in practice, gets paid off or refinanced in roughly a decade on average, so its effective life matches the 10-year far better than an overnight rate. On top of the 10-year sits a spread, usually around **1.5 to 2.0 percentage points**, that pays the lender for two risks: the chance the borrower defaults, and the chance the borrower *prepays* (refinances when rates fall, handing the lender back their money at the worst possible time — the negative-convexity problem of [mortgage-backed securities: bonds with negative convexity](/blog/trading/fixed-income/mortgage-backed-securities-bonds-with-negative-convexity)).

So when the 10-year yield moves, your potential mortgage rate moves with it — not because your local bank decided to charge more, but because the bank funds that loan in a market priced off Treasuries. The bond yield reaches all the way to your kitchen table.

#### Worked example: the same house, twice the payment

Suppose the 10-year yield is 1.5% and the mortgage spread is 1.8%, so the 30-year mortgage rate is $1.5\% + 1.8\% = 3.3\%$. On a \$400,000 loan, the monthly principal-and-interest payment is about **\$1,750**.

Now the 10-year rises to 4.5% (the spread holds at 1.8%), so the mortgage rate is $4.5\% + 1.8\% = 6.3\%$. The monthly payment on the *same* \$400,000 loan jumps to about **\$2,475** — an increase of roughly **41%** for the identical house.

The house didn't change. The bond market did, and the buyer's monthly cost rose by \$725 — which prices many buyers out entirely and freezes the housing market. *Your mortgage payment is the 10-year Treasury yield plus a spread, run through an amortization formula, which is why a bond-market move can shut a household out of a home it could have afforded a year earlier.* (The full chain from the bond desk to your loan is in [from the 10-year yield to your mortgage: the transmission of rates](/blog/trading/fixed-income/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates).)

### Credit spreads: the cost of corporate borrowing

The same logic prices what companies pay to borrow. A corporate bond yields "the Treasury, plus a **credit spread**" — the extra yield that compensates lenders for the chance the company defaults (covered in full in [credit spreads: pricing the probability of default](/blog/trading/fixed-income/credit-spreads-pricing-the-probability-of-default)). When the risk-free rate rises, a company's all-in borrowing cost rises with it, *even if the company's own creditworthiness hasn't changed a bit* and its spread is unchanged. A rate move is a tax on every borrower in the economy at once.

Worse, the two pieces often move *together* in a downturn: a recession that pushes the Fed to cut the risk-free rate also raises default fears, *widening* credit spreads — so corporate borrowing costs don't always fall when the risk-free rate does. The base and the premium can pull in opposite directions, which is one reason the master equation is a framework, not a guarantee.

#### Worked example: a company's borrowing cost when only the base moves

Northwind Corp, a solid investment-grade company, borrows at "the 10-year, plus a 1.5% credit spread."

*When the 10-year is 2%:* Northwind's borrowing cost is $2\% + 1.5\% = 3.5\%$. On \$500 million of debt, that's about \$17.5 million a year in interest.

*When the 10-year rises to 5%* (spread unchanged): Northwind's cost is $5\% + 1.5\% = 6.5\%$. On the same \$500 million, that's about \$32.5 million a year — an extra **\$15 million** in annual interest, with Northwind's business and credit quality completely unchanged.

That \$15 million has to come from somewhere — fewer hires, less investment, lower profit — which is one channel through which higher rates slow the whole economy. *A company's cost of capital is the risk-free rate plus its credit spread, so a move in the bond market reprices corporate borrowing across the entire economy at once, even for companies whose own risk never changed.*

## Putting it together: one rate, every price at once

We've revalued assets one at a time. Now let's line them up side by side and watch them *all* reprice as the risk-free rate climbs — because the punchline of this whole series is that this happens *simultaneously*, on the same afternoon, from one move in the bond market.

![A matrix showing a stock's fair P/E and price, an apartment building's value, and a long-dated cash flow's present value at risk-free rates of 2, 3, 4, and 5 percent, all falling together as the rate rises](/imgs/blogs/the-price-of-money-how-bonds-set-every-other-price-7.png)

The matrix above takes our running examples — Steady Co.'s fair P/E and price, Maple Court's value, and the present value of that far-off \$1,000 — and computes each at risk-free rates of 2%, 3%, 4%, and 5%. Read down any column and you see one consistent rate environment; read across any row and you watch a single asset deflate as the rate rises. The whole table moves together because the same number, $r_f$, is in every cell's denominator.

#### Worked example: the system-wide repricing from 2% to 5%

Let's total it up. Hold the cash flows of every asset perfectly constant and move *only* the risk-free rate from 2% to 5%:

| Asset | Cash flow | Value at $r_f$ = 2% | Value at $r_f$ = 5% | Change |
|---|---|---|---|---|
| Steady Co. stock | \$5/yr earnings (ERP 4%) | \$83.3 (P/E 16.7) | \$55.6 (P/E 11.1) | **−33%** |
| Maple Court building | \$100k/yr income (premium 3%) | \$2.00M (5% cap) | \$1.25M (8% cap) | **−38%** |
| Far-off \$1,000 (20 yrs) | \$1,000 once in 20 yrs | \$673 | \$377 | **−44%** |
| 10-year Treasury (4% coupon) | \$40/yr + \$1,000 | \$1,180 | \$923 | **−22%** |

Nothing in the *businesses* changed. No tenant left, no profit fell, no coupon was missed. Yet a stock lost a third of its value, a building lost nearly 40%, a long-dated claim lost 44%, and even the safe Treasury lost a fifth — all from one number, the price of money, moving three percentage points. *This single table is the entire thesis of the post: the bond market sets the risk-free rate, and the risk-free rate is the gravitational constant of the financial universe — change it, and every price in the system falls into a new orbit at once.*

## How the dollar fits in

The one asset on our wheel we haven't priced is the dollar itself — the currency. It doesn't have "cash flows" in the same way, but it still bends to the bond yield, through a different but related channel: **the hunt for yield across borders.**

Money is global and restless. If US bonds suddenly pay 5% while German and Japanese bonds pay 2%, global investors sell euros and yen to buy dollars so they can capture the higher US yield. That extra demand for dollars pushes the dollar *up*. So a rising US risk-free rate tends to strengthen the dollar, and a falling one tends to weaken it. The mechanism is the *interest-rate differential* — the gap between what US bonds pay and what other countries' bonds pay — and capital flows toward the higher rate.

This matters far beyond currency traders. A stronger dollar makes US exports more expensive and squeezes any company or country that borrowed in dollars (their debt got heavier in their home currency). It pressures commodity prices (most are priced in dollars) and emerging markets (which often borrow in dollars). So the bond yield reaches the dollar, and the dollar reaches the entire global economy — one more spoke on the wheel, all turning from the same hub. (The allocation lens on this is [real yields: the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything), and the macro lens is [interest rates: the price of money, the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).)

#### Worked example: a yield gap pulling capital into the dollar

Suppose you manage a global bond fund and can buy either a German 10-year at 2% or a US 10-year at 2%. The yields are equal; currency aside, you're indifferent. Now the US 10-year rises to 5% while Germany's stays at 2%. The US bond now pays you **3 percentage points more per year** of nearly risk-free income. To capture it, you (and thousands of funds like you) sell euros and buy dollars — say you move \$100 million; that's \$100 million of euro-selling, dollar-buying pressure, repeated across the market. The dollar strengthens until the *expected* return (after accounting for expected currency moves) equalizes again. *A rising US risk-free rate doesn't just reprice US assets — it pulls global capital toward the dollar, which is why the bond market sets the price of the currency too.*

## A second look at the influence wheel

We've now traced every spoke: stock multiples, property cap rates, mortgage rates, credit spreads, and the dollar, each fanning out from the 10-year yield. Step back and the picture is almost eerie in its simplicity. The financial world looks dizzyingly complex — thousands of asset classes, millions of securities, a wall of jargon. But underneath, almost all of it is one equation (value = cash flows ÷ discount rate) sharing one number (the risk-free rate, set by bonds).

That is what the whole "The Bond Market, From the Ground Up" series has been building toward, and why we say bonds are *the price of money.* The bond market isn't one corner of finance among many. It is the corner that sets the price every other corner has to pay. When a central banker, a stock trader, a homebuyer, or a finance minister wants to know what the future is worth, they all start at the same place: the yield on a government bond. Everything else is that number, plus a spread.

There is one honest caveat worth stating plainly before the misconceptions, because the model is powerful enough to be dangerous if taken too literally. The master equation describes the *gravitational pull* of rates, not a precise predictor of any single price on any single day. Three things can muddy the link in the short run. First, **the risk premium moves too** — a falling equity risk premium can offset a rising risk-free rate, and a panic can widen credit spreads even as the safe rate falls. Second, **cash flows aren't always constant** — a rate rise driven by a booming economy may come *with* rising earnings, so a stock can absorb higher rates if its numerator grows fast enough. Third, **the *reason* rates moved matters**: a real-yield rise (genuine tightening) is pure gravity, while a rise driven only by higher inflation expectations is a softer headwind. None of this breaks the framework; it just reminds you that the wheel turns through a fog, not a vacuum. The direction is reliable; the daily magnitude is not. Hold the model as the *map* of how money prices everything, and the exceptions become legible rather than confusing.

## Common misconceptions

**"The stock market and the bond market are separate worlds."** They feel separate — different exchanges, different people, different news. But they're joined at the denominator. Every stock's fair value has the bond yield buried inside its discount rate. That's why a "bond move" can be the biggest single driver of a stock-market day, and why equity investors who ignore the 10-year are flying half-blind.

**"If a company's earnings don't change, its stock shouldn't move."** Earnings are only the *numerator* of value. The *denominator* — the discount rate — moves with the risk-free rate, and it moves prices just as powerfully. A company can earn exactly the same profit two years running and see its stock halve, simply because rates rose. The business didn't change; the price of money did.

**"Higher interest rates are just bad for banks and borrowers."** Higher rates raise the discount rate for *every* asset, so they pull down the value of stocks, bonds, real estate, and credit all at once — far beyond anyone's loan payment. The reach of rates is systemic, not sectoral. That breadth is exactly what makes the 10-year the master variable.

**"Cash flows are what matter; the discount rate is just a technicality."** For long-lived assets — growth stocks, real estate, 30-year mortgages — the discount rate often matters *more* than near-term cash flows, because most of the value lives far in the future where discounting bites hardest. A small change in $r$ swamps a large change in next year's earnings. The "technicality" is frequently the whole story.

**"The risk premium absorbs rate moves, so assets are protected."** Sometimes risk premiums *do* move to cushion a rate rise (a falling equity risk premium can offset a rising risk-free rate). But there is no rule that says they must, and in a typical rate-rise shock the premium stays roughly put or even *widens* (as fear rises with rates), so the asset takes the full force of the move. Counting on the premium to save you is a hope, not a mechanism.

**"Bonds are the boring, safe part of a portfolio that has nothing to do with the exciting assets."** Bonds are the *opposite* of a sideshow. They set the reference rate that prices the exciting assets. The boring asset is the one writing the rules for all the interesting ones — which is the entire reason this series treats the bond market as the foundation of finance, not a footnote to it.

## How it shows up in real markets

**The 2022 everything-selloff.** This is the cleanest real-world demonstration of the thesis you will ever get. As the Federal Reserve fought inflation, the US 10-year yield roughly tripled, from around 1.5% at the start of 2022 to over 4% by autumn. Per the master equation, every asset's denominator swelled at once — and so the S&P 500 fell about 18% for the year, long-dated Treasuries fell *more* than 25% (their long duration meant huge rate sensitivity), investment-grade and high-yield bonds both fell, commercial real estate values began a sharp slide, and Bitcoin fell around 65%. The "diversification" of stocks-plus-bonds failed precisely because the shock came from the *shared base* — the risk-free rate itself — leaving nowhere downstream to hide. It was gravity strengthening across the whole system at once.

**The 2020–2021 zero-rate melt-up.** The mirror image. With the 10-year yield crushed near 0.5% during the pandemic, gravity went nearly weightless. Long-duration assets — profitless tech, SPACs, speculative crypto, "story" stocks whose payoff was a decade away — floated to extraordinary valuations, because a tiny discount rate barely touched their far-off cash flows. The same arithmetic that crushed those assets in 2022 had inflated them in 2020–2021. Nothing about the businesses justified the round trip; the price of money did all the work in both directions.

**The 2023 commercial real-estate repricing.** As the 10-year settled near 4% (after years near 2%), cap rates were dragged up with it, and the price equation (value = income ÷ cap rate) did the rest. Office and apartment values fell sharply across the US even where buildings stayed leased and rents were stable, because the *denominator* moved. Refinancing loans taken out in the low-rate era at the new, higher cap rates exposed huge value gaps — the direct, mechanical consequence of the bond yield repricing the income stream. (The bank-stress angle is in [SVB & Credit Suisse: the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

**The 1980s, when gravity was enormous.** Buffett's own example. With the 10-year yield in the teens in the early 1980s (after [Paul Volcker's rate shock](/blog/trading/finance/paul-volcker-1980-rate-shock-killing-inflation) crushed inflation), the gravitational pull on asset prices was immense: discount rates above 13% meant even solid earnings were worth a low multiple, and stock P/Es languished in the single digits and low teens. As yields fell over the following four decades — from the teens to near zero by 2020 — gravity weakened relentlessly, and that secular decline in the risk-free rate is a large part of the great bull market in nearly every asset class over that span. One number, falling for forty years, lifting everything.

**The dollar's 2022 surge.** As the Fed hiked and US yields raced ahead of Europe's and Japan's, the interest-rate differential blew out, and global capital poured into dollars to capture the higher US yield. The dollar index (DXY) rose roughly 20% over 2022 to a two-decade high — putting the euro briefly *below* parity with the dollar and the yen to multi-decade lows. The strong dollar then squeezed dollar-borrowers worldwide and pressured commodities and emerging markets. The bond yield reached the currency, and the currency reached the globe — the wheel turning, exactly as the model says.

**Mortgage rates and the housing freeze.** When the 10-year rose through 2022–2023, the 30-year mortgage rate (the 10-year plus a spread) jumped from under 3% to over 7%. Home *prices* are sticky, but *affordability* collapsed: the monthly payment on the same house nearly doubled, transaction volume froze, and the housing market seized — a household-level illustration that the bond yield doesn't stay on Wall Street. (The full chain is in [from the 10-year yield to your mortgage: the transmission of rates](/blog/trading/fixed-income/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates).)

## When this matters to you and further reading

This is not abstract. The price of money is in your life whether or not you own a single bond. The rate on your mortgage, the value of your home, the level of your retirement portfolio, the cost of your car loan, even your job security in a rate-sensitive industry — all of them sit downstream of the yield on a government bond you've probably never thought about. The next time the financial news leads with "the 10-year yield jumped today," you'll know that it isn't a sleepy bond-market footnote. It's the gravitational constant of finance shifting, and everything you own is quietly being repriced around it.

This is educational, not investment advice — the point is to understand the *mechanism*, not to time any market with it.

Where to go next in this series and its siblings:

- The pieces that built up to this: [discounting cash flows: how a bond is priced](/blog/trading/fixed-income/discounting-cash-flows-how-a-bond-is-priced), [price and yield: the seesaw at the heart of bonds](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds), and [duration: the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income).
- The next stops on the influence track: [bonds vs stocks: discount rates, the 60/40, and correlation](/blog/trading/fixed-income/bonds-vs-stocks-discount-rates-the-60-40-and-correlation) and [from the 10-year yield to your mortgage: the transmission of rates](/blog/trading/fixed-income/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates).
- The allocation view: [government bonds: the risk-free anchor and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration), [stock–bond correlation: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine), and [real yields: the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything).
- The macro lens: [interest rates: the price of money, the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).
- The heavy math, if you want it: [bond pricing](/blog/trading/quantitative-finance/bond-pricing) and [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics).

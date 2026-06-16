---
title: "The many yields: current yield, YTM, and yield to call"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into what a bond's yield actually means — coupon rate, current yield, yield to maturity, yield to call, and yield to worst — and why they order differently for discount and premium bonds."
tags: ["fixed-income", "bonds", "yield-to-maturity", "current-yield", "yield-to-call", "yield-to-worst", "coupon-rate", "bond-pricing", "reinvestment-risk"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **The one-sentence thesis:** a bond does not have a single "yield" — it has a family of them, and which one you should look at depends entirely on the question you are asking.
> - The **coupon rate** is fixed at issue and never changes; it tells you the dollar income, not your return.
> - **Current yield** = annual coupon ÷ price you paid — a quick income snapshot that ignores the gain or loss you book at maturity.
> - **Yield to maturity (YTM)** is the single discount rate that makes the present value of every future cash flow equal the price — the bond's internal rate of return, and the number quotes screens show.
> - **Yield to call (YTC)** redoes the YTM math but ends at the issuer's first call date, and **yield to worst (YTW)** is the lowest of all those scenarios — the conservative number.
> - For a **discount** bond (price below par) the order is YTM > current yield > coupon; for a **premium** bond (price above par) the order flips to coupon > current yield > YTM.

You look up a bond and the screen throws four or five numbers at you, all labeled some flavor of "yield," all slightly different. A 5% coupon bond is quoted at a 6.9% yield. The "current yield" says 5.4%. There is a "yield to worst" that is lower than the "yield to maturity." And a friend who owns the same bond swears they are earning 5%. Who is right?

They all are. Each number answers a different question, and the confusion that sends new investors spinning is almost always the result of comparing two numbers that were never meant to mean the same thing. "Yield" in the bond market is not one quantity — it is a whole vocabulary, and the entire game of fixed income starts with knowing which word you are using.

![A comparison table showing four different yield measures for a single 1000 dollar par 5 percent coupon bond bought at a 920 discount, with the coupon rate at 5 percent, current yield at 5.43 percent, and yield to maturity at 6.95 percent](/imgs/blogs/the-many-yields-current-yield-ytm-and-yield-to-call-1.png)

The table above is the mental model for this entire post: one bond, four yields, four different stories. By the end you will be able to glance at that fan of numbers and know instantly what each one is counting, which one you actually care about, and what the *spread between them* secretly tells you about whether the bond is cheap or expensive. This is the capstone of the series' opening track — once "yield" clicks, every later post about duration, credit spreads, and the yield curve has solid ground to stand on.

We will build one running example and never let go of it: a plain **\$1,000 par, 5% coupon bond from a fictional issuer called Northwind Corp**, and we will price it at a discount, at par, and at a premium to watch all the yields move. Wherever it sharpens the lesson we will put it next to a real **US Treasury**, the benchmark every other bond is measured against.

## Foundations: the words before the math

Before we can compare yields we need the handful of terms a bond is built from. If you have read the earlier posts in this series, this is review; if this is your entry point, do not skip it — every later sentence leans on these definitions.

A **bond** is a loan you make to a borrower (a government, a company, a city) in exchange for a written promise: pay me interest on a schedule, and give me my money back on a fixed future date. That is the whole instrument. The pieces:

- **Face value** (also **par value** or **principal**): the amount the borrower promises to repay at the end. The market standard for quoting is \$100 or \$1,000 of face value; we will use **\$1,000** throughout. Crucially, face value is *not* what you pay to buy the bond — it is what you get back at the end.
- **Coupon**: the fixed interest payment, quoted as a percentage of *face value*. A "5% coupon" on a \$1,000 bond means \$50 of interest per year, forever fixed, regardless of what you paid. In the US, coupons are almost always split into **two semiannual payments** — so a 5% coupon really pays \$25 every six months. We will use annual payments in the simplest examples to keep the arithmetic clean, then switch to semiannual when it matters.
- **Maturity**: the date the borrower repays the face value and the bond ceases to exist. A "5-year bond" matures in five years; a "10-year Treasury" in ten.
- **Price**: what the bond trades for *today* in the market. This floats. It can be **below** face value (a *discount*), **equal** to it (*at par*), or **above** it (a *premium*). The single most important fact in all of fixed income is that **price moves opposite to yield**: when market interest rates rise, the price of an existing bond falls, and vice versa. (We unpack the mechanism of that seesaw in the companion post on [why bond prices move opposite to rates](/blog/trading/quantitative-finance/bond-pricing); here we take it as given.)
- A **basis point** (*bp*) is one hundredth of a percent — 0.01%. "Yields rose 40 bps" means they rose 0.40%. Bond people count in basis points the way the rest of us count in dollars; get comfortable with the unit now.

That is the entire toolkit. Notice what is *fixed* the moment the bond is issued: the face value, the coupon dollars, and the maturity date. The borrower's promise never changes. The only thing that moves is the **price** — and because price moves, the *yield* moves with it. Everything in this post is a different way of converting "I paid this price for that fixed promise" into "so my return is this."

One more piece of plumbing worth a sentence, because it explains *why* prices move at all. Bonds trade in a vast secondary market: after a bond is issued, holders buy and sell it among themselves, the way shares of a stock change hands. When new bonds come out paying a higher coupon, nobody will pay full price for an old bond paying less — so the old bond's *price* falls until its effective return (its yield) matches what is newly available. When new bonds pay less, the old higher-coupon bonds become prized and their prices rise. The bond's *promise* is frozen; the market's *re-pricing* of that promise is what generates discounts and premiums, and therefore the whole spread of yields we are about to dissect. If this seesaw is new to you, the dedicated walkthrough is in the [bond pricing](/blog/trading/quantitative-finance/bond-pricing) post; for our purposes the takeaway is that a bond's price is the market's verdict on its fixed promise, re-rendered every day.

### Why one bond needs more than one yield number

Here is the core tension. A bond pays you in two completely different ways:

1. **Income** — the coupon checks that arrive on schedule.
2. **Capital gain or loss** — the difference between what you paid and the \$1,000 you collect at maturity. Buy at \$920, get back \$1,000, and you book an \$80 gain on top of the coupons. Buy at \$1,080, get back \$1,000, and you book an \$80 *loss* that quietly eats into your coupon income.

A yield measure that counts only the income (current yield) will mislead you about a discount or premium bond, because it ignores that gain or loss. A yield measure that bundles income *and* the pull toward par into one number (yield to maturity) tells the fuller story but hides an assumption about reinvesting your coupons. There is no single number that is "the yield" because there is no single question. The skill is matching the number to the question. Let us take them one at a time, simplest first.

## The coupon rate: the fixed number that is not your return

The **coupon rate** is the easiest yield to understand and the easiest to misuse. It is simply the annual coupon divided by the *face value*, set once at issue and frozen forever:

$$\text{coupon rate} = \frac{\text{annual coupon}}{\text{face value}}$$

For Northwind Corp's bond, that is \$50 ÷ \$1,000 = 5.00%. It will read 5.00% in year one and 5.00% in year five. It does not care what you paid.

That last point is where beginners trip. The coupon rate tells you the **dollar income** the bond throws off — \$50 a year, no more, no less — but it tells you almost nothing about your **return**, because your return depends on your *price*, and the coupon rate is blind to price. The coupon rate is a property of the bond as written on the certificate. Your return is a property of the *deal you struck* when you bought it.

#### Worked example: the coupon rate ignores your price

*Two investors, one bond.* Alice buys Northwind's 5% bond at issue for \$1,000 (par). Bob buys an identical Northwind bond two years later, after rates have risen, for \$920. Both bonds carry the same 5.00% coupon rate. Both collect the same \$50 a year.

But Alice paid \$1,000 and Bob paid \$920. On her \$1,000, Alice's income alone is \$50 ÷ \$1,000 = 5.00% a year. On his \$920, Bob's income is \$50 ÷ \$920 = 5.43% a year — and on top of that, Bob will collect \$1,000 at maturity for a bond he bought at \$920, an extra \$80 gain Alice will not get. Same coupon rate, two very different returns.

*The coupon rate is a fact about the bond; your return is a fact about the price you paid for it — never confuse the two.*

## Current yield: income measured against what you actually paid

The first repair to the coupon rate's blind spot is to divide the coupon by the **price you paid** instead of by face value. That is the **current yield** (sometimes called the **running yield**):

$$\text{current yield} = \frac{\text{annual coupon}}{\text{market price}}$$

For Bob's discount bond: \$50 ÷ \$920 = **5.43%**. For a premium buyer at \$1,080: \$50 ÷ \$1,080 = **4.63%**. For anyone buying at par: \$50 ÷ \$1,000 = 5.00%, exactly the coupon rate.

Current yield answers a real and useful question: *"If I just want the cash income relative to what I put in, what am I getting?"* It is the bond equivalent of a stock's dividend yield. An income investor living off coupons cares about it directly — it is the cash hitting their account each year as a percentage of their outlay.

But current yield has a gaping hole: it completely ignores the gain or loss waiting at maturity. Bob's bond will repay \$1,000 on a \$920 purchase — that \$80 is real money, spread over the life of the bond, and current yield does not count a cent of it. So current yield *understates* a discount bond's true return and *overstates* a premium bond's. It is a snapshot of income, not a measure of total return.

#### Worked example: current yield flatters a premium bond

*You are tempted by a high coupon.* You see a Northwind bond with a fat 5% coupon trading at \$1,080. Its current yield is \$50 ÷ \$1,080 = 4.63% — still a decent-looking income number. But here is what current yield hides: you paid \$1,080 for a bond that will hand you back exactly \$1,000 at maturity. That is an \$80 capital *loss*, guaranteed, baked in the day you buy. Spread that \$80 loss over the bond's remaining life and it claws back a chunk of every coupon. Your true return — the number that nets the income against the certain loss — is far below 4.63%. (We are about to compute it: it is 3.24%.)

*Current yield tells you the income, but for a premium bond it quietly omits the loss you are guaranteed to take, so it always reads higher than your real return.*

## Yield to maturity: the one rate that ties it all together

Now we reach the number that quote screens actually mean when they say "yield," the one professionals trade on, and the one beginners most often misunderstand: **yield to maturity**.

Strip away the jargon and YTM is a beautifully simple idea. A bond is a stream of future cash flows — coupons every period, then the face value at the end. Those future dollars are worth less than dollars today (a dollar next year is worth less than a dollar now, because you could have invested today's dollar). To compare future cash to the price you pay today, you have to *discount* the future cash flows back to the present. The yield to maturity is **the single discount rate that makes the present value of all those future cash flows exactly equal to the bond's price.**

![A cash flow timeline showing a 920 dollar payment today, then five 50 dollar coupons and a 1000 dollar principal repayment at year five, each discounted back at 6.95 percent so their present values sum to exactly 920 dollars](/imgs/blogs/the-many-yields-current-yield-ytm-and-yield-to-call-2.png)

The figure above is the whole concept in one picture. You pay \$920 today (the red outflow on the left). In return you collect five \$50 coupons and your \$1,000 face value at year five (the green and blue inflows). Each of those future inflows is discounted back to today at one rate — and the magic number, the rate that makes those discounted inflows add up to *exactly* the \$920 you paid, is the YTM: **6.95%**.

In formula form, for a bond with price $P$, coupon $C$ per period, face value $F$, and $N$ periods to maturity:

$$P = \sum_{t=1}^{N} \frac{C}{(1+y)^t} + \frac{F}{(1+y)^N}$$

where $y$ is the yield to maturity — the one unknown we solve for. $P$ is the price you pay, $C$ is each coupon dollar amount, $F$ is the face value repaid at maturity, $N$ is the number of periods, and $t$ indexes each period from 1 to $N$.

You cannot solve that equation for $y$ with algebra — it is a polynomial of degree $N$. In practice a computer (or a spreadsheet's `RATE` or `YIELD` function) finds it by trial and error: guess a rate, compute the present value, see if it is too high or too low, adjust, repeat until it lands on the price. That iterative answer is the YTM. The deeper machinery — Newton's method, day-count conventions, the semiannual compounding convention — lives in the [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics) post; here we care about what the number *means*.

What it means is this: **YTM is the bond's internal rate of return if you hold it to maturity.** It folds the coupon income *and* the pull toward par into a single annualized number. For Bob's \$920 discount bond, the 5% coupon income plus the \$80 gain at maturity together work out to 6.95% a year. That is why YTM (6.95%) sits above current yield (5.43%) for a discount bond — the current yield missed the gain, and YTM caught it.

#### Worked example: solving a YTM by squeezing it

*Let us find the YTM ourselves, the way a computer does.* Take Northwind's 5-year, \$1,000 par, \$50-annual-coupon bond priced at \$920. We want the rate $y$ that discounts five \$50 coupons plus the final \$1,000 back to \$920.

Guess 1: try $y$ = 5%. Discount everything at 5% and the present value comes to exactly \$1,000 (because at a 5% yield a 5% coupon bond is worth par). That is way above \$920 — our guess is too low, so we need a *higher* yield (higher yield, lower price).

Guess 2: try $y$ = 7%. Discount at 7%: the five coupons are worth about \$205 and the \$1,000 face is worth \$1,000 ÷ (1.07)⁵ ≈ \$713, summing to roughly \$918 — just a hair below \$920. Too high a yield by a touch.

Guess 3: split the difference and land near $y$ = 6.95%. Discount at 6.95% and the cash flows sum to \$920.00 — bullseye. (Component check: the coupons discount to \$46.75 + \$43.71 + \$40.87 + \$38.22, and the final \$50 coupon plus \$1,000 face discounts to \$750.44. Add them: \$920.00.)

*YTM is just the rate that "squeezes" the future cash flows down until their present value equals the price — there is nothing mystical about it, only iteration.*

#### Worked example: the same bond at a premium

*Now flip the price.* The market loves Northwind and bids the identical 5% bond up to \$1,080. The coupons are unchanged at \$50, and the face value still repays only \$1,000. Now you are overpaying by \$80 relative to what you get back, so your total return must be *below* the 5% coupon. Run the same iteration — find the rate that discounts five \$50 coupons plus \$1,000 down to \$1,080 — and it lands at **3.24%**. The 5% income is real, but the guaranteed \$80 loss at maturity drags the all-in return down to 3.24%.

*For a premium bond, YTM falls below both the coupon rate and the current yield, because it is the only one of the three that subtracts the certain loss you take when the bond redeems at par.*

### Semiannual coupons and the bond-equivalent yield

There is one wrinkle in how the US market quotes YTM that you will meet the moment you look at a real bond, and it is worth understanding so the numbers reconcile. US bonds pay coupons **twice a year**, so the discounting math runs on *half-year* periods with *half* the coupon, and the rate that solves the equation is a *semiannual* rate. To turn that into the annual number you see quoted, the convention is simply to **double the semiannual rate** — a quirk called the **bond-equivalent yield (BEY)** or *semiannual bond basis*. It is not the same as the true compound annual rate, because doubling ignores the half-year of compounding in between, but it is the universal convention, so every US bond yield is quoted this way and they are all comparable to each other.

The practical upshot: when you compare a US Treasury yield to, say, a European bond that pays annually, you are comparing two slightly different conventions, and a careful analyst converts one to the other before declaring one "higher." For everything in this post the difference is small — our \$920 discount bond yields 6.95% on an annual basis and 6.92% on a semiannual bond-equivalent basis, a 3-basis-point gap — but on large institutional trades that gap is real money, and mixing conventions is a classic rookie error.

#### Worked example: the same bond, annual versus semiannual quoting

*Convention changes the headline number a little.* Take Northwind's \$920 discount bond. If it paid one \$50 coupon a year, the YTM that solves the pricing equation is 6.95%. Now pay it the realistic way — \$25 every six months, ten payments over five years — and solve for the semiannual rate, then double it for the bond-equivalent yield: you get 6.92%. Same bond, same price, same total dollars; the only difference is the compounding convention baked into how we annualize. The 3-basis-point gap comes entirely from the fact that doubling a semiannual rate slightly understates the effect of compounding twice a year.

*When you compare two bonds' yields, first make sure they are quoted on the same compounding convention — otherwise you are comparing two different units that only look alike.*

## The influence figure: how the yields relate across the price range

We now have three numbers — coupon rate, current yield, YTM — and we have seen them at three prices. The single most useful thing you can internalize is **how they move relative to each other as the price slides from a deep discount to a deep premium.** This relationship is the secret decoder ring: once you see it, you can glance at any bond's quoted yields and instantly tell whether it trades cheap or rich.

![A line chart with market price on the horizontal axis from 800 to 1200 dollars and yield on the vertical axis, showing a flat 5 percent coupon line, a gently falling current yield line, and a steeply falling yield to maturity line, with all three crossing at the 1000 dollar par price](/imgs/blogs/the-many-yields-current-yield-ytm-and-yield-to-call-3.png)

Read the chart from left to right — from cheap to expensive:

- The **coupon rate** is a flat line at 5%. It is fixed; price does not touch it. This is your reference line.
- The **current yield** slopes gently downward. As price rises, the same \$50 coupon is divided by a bigger number, so the income percentage shrinks — but only gently, because it is a simple ratio.
- The **YTM** slopes steeply downward and crosses both other lines at par. It falls fast because it captures *both* the shrinking income *and* the swing from a capital gain (at a discount) to a capital loss (at a premium).

All three lines meet at one point: **\$1,000 par, 5% yield.** This is not a coincidence — it is a theorem. At par, you pay exactly what you get back, so there is no capital gain or loss; the only return is the coupon income; and so coupon rate, current yield, and YTM are all identical at 5%. Par is the pivot where the order of the yields flips.

To the **left of par** (discount, shaded green — a bargain): YTM > current yield > coupon rate. You are buying below face value, so the gain at maturity boosts your total return above the income, which is itself above the fixed coupon rate.

To the **right of par** (premium, shaded red — you are paying up): coupon rate > current yield > YTM. The loss at maturity drags your total return below the income, which is itself below the coupon rate.

This is the relationship the team asked us to make unmissable, so let me state it as a rule you can carry forever: **the position of YTM relative to the coupon rate tells you the price without your ever seeing the price.** If a screen shows a YTM above the coupon, the bond is at a discount. If YTM is below the coupon, it is at a premium. If they are equal, it is at par. You have decoded the price from the yields alone.

#### Worked example: reading the price off the yields

*A bond quote with no price.* A colleague texts you: "Northwind 5s, current yield 4.63%, YTM 3.24% — worth a look?" There is no price in the message. But you do not need one. The coupon is 5%, the current yield (4.63%) is below it, and the YTM (3.24%) is below *that*. The order is coupon > current > YTM — the premium ordering. You text back: "That's a premium bond, you're paying up — probably around \$1,080, and your real return is 3.24%, not the 5% coupon." You read the price, the premium, and the true return off three numbers and a rule.

*The gap and the ordering between coupon, current yield, and YTM is a fingerprint: it identifies a discount, par, or premium bond before you ever look at the price tag.*

### The "pull to par" that the chart is really drawing

The steep slope of the YTM line in the chart has a name worth knowing: the **pull to par**. A bond, no matter what price it trades at today, must converge to exactly \$1,000 on its maturity date, because that is what gets repaid. A discount bond at \$920 *must* drift up to \$1,000 over its remaining life; a premium bond at \$1,080 *must* drift down to \$1,000. That convergence is not a market opinion — it is a contractual certainty, the one price move in all of finance you can predict with total confidence.

The pull to par is exactly the gain or loss that current yield ignores and YTM captures. For the discount bond, the \$80 climb from \$920 to \$1,000 is a built-in tailwind that lifts YTM above the income; for the premium bond, the \$80 slide from \$1,080 to \$1,000 is a built-in headwind that drags YTM below the income. The closer a bond gets to maturity, the stronger this pull, which is why a bond's price grinds toward par as its final day approaches almost regardless of what rates are doing — the pull-to-par force overwhelms small rate moves when there is little time left.

#### Worked example: watching the pull to par at work

*A discount bond on its way home.* Bob's \$920 bond has five years left. Suppose market yields do not move at all over the next year. What happens to the price? It does not stay at \$920 — it rises, because with one fewer year to maturity the bond is closer to its \$1,000 payday. Re-price the now-four-year bond at the same 6.95% yield and it is worth about \$934. Bob earned his \$50 coupon *and* an extra \$14 of price appreciation purely from the calendar advancing, with no change in rates at all. That \$14 is the pull to par made visible — the slice of return that current yield (which only saw the \$50) completely missed.

*A discount bond appreciates toward par as time passes even if rates never move, which is precisely the extra return YTM counts and current yield throws away.*

## Yield to call: the issuer's escape hatch

So far we have assumed the bond runs cleanly to its stated maturity. Many corporate and municipal bonds — and the mortgage-backed bonds that dominate the US market — come with a catch: the issuer can **call** the bond, meaning repay it early, on or after a set date, at a set price (often par or a small premium above it). A *callable* bond is a bond with an embedded escape hatch that belongs entirely to the borrower.

Why would an issuer want that hatch? The same reason a homeowner refinances a mortgage: if interest rates fall, the issuer can call the old high-coupon bond, repay the holders, and reissue new bonds at the lower rate, pocketing the difference. The call option is *good for the issuer and bad for the holder* — it gets exercised precisely when you would least want it, after rates have dropped and your high-coupon bond has become valuable.

To price that risk, we compute the **yield to call (YTC)**: the exact same discounting math as YTM, but we cut the cash-flow stream short at the **first call date** and use the **call price** instead of the face value as the final payment. It answers: *"What return do I earn if the issuer calls this bond at the earliest opportunity?"*

![A side by side comparison of a 1080 dollar premium callable bond showing that held to a ten year maturity it yields 4.01 percent, but if called early at year three it yields only 2.83 percent, which becomes the yield to worst](/imgs/blogs/the-many-yields-current-yield-ytm-and-yield-to-call-4.png)

The figure contrasts the two scenarios for a premium callable bond. On the left, the bond runs to its 10-year maturity and yields 4.01%. On the right, the issuer calls it at year three for \$1,020 — and because you paid \$1,080 and lose most of your premium fast while collecting fewer coupons, your return collapses to just **2.83%**. The call hurts you, and for a premium bond it almost always *will* be exercised if rates allow, because the issuer is sitting on cheap-to-refinance debt.

#### Worked example: the call eats a premium bond's return

*You bought the high coupon; the issuer takes it away.* You pay \$1,080 for a Northwind callable bond — 5% coupon, 10 years to final maturity, first callable in 3 years at a call price of \$1,020. You are dreaming of a decade of \$50 coupons.

If it runs to maturity, the YTM is 4.01% — you collect ten years of coupons and \$1,000 back, and the 4.01% reflects the slow bleed of your \$80 premium over ten years.

But suppose rates fall. Northwind calls the bond at year three, hands you \$1,020, and refinances cheaper. Now you have collected only three \$50 coupons and gotten back \$1,020 on a bond you paid \$1,080 for — a near-instant loss of your premium. Run the YTC math (discount three \$50 coupons plus \$1,020 down to \$1,080) and the return is only **2.83%**. You expected 4%, you got 2.83%, and you never had a say.

*On a premium callable bond, the call is the bad outcome, so the yield to call — not the yield to maturity — is the honest number to plan around.*

## Yield to worst: the conservative number professionals quote

If a callable bond can end several ways — at final maturity, or at any of several call dates — which yield should you trust? The professional answer is brutally simple: assume the *worst* one happens to you. The **yield to worst (YTW)** is the lowest yield among the yield to maturity and the yields to every possible call date:

$$\text{YTW} = \min(\text{YTM}, \text{YTC}_1, \text{YTC}_2, \dots)$$

where $\text{YTC}_1, \text{YTC}_2, \dots$ are the yields computed to each call date on the bond's call schedule. YTW assumes the issuer will act in its own interest at your expense — which, for the call option, is exactly what it will do. It is the number a careful buyer underwrites to, because it is the floor: in most scenarios you do at least this well.

![A decision tree showing how yield to worst is computed by finding the yield to maturity and a yield to call for each call date, then taking the minimum, with a premium bond example where the first call at year three gives the lowest yield of 2.83 percent](/imgs/blogs/the-many-yields-current-yield-ytm-and-yield-to-call-5.png)

The tree shows the procedure: compute the YTM, compute a YTC for *every* call date on the schedule, then keep the smallest. For our premium bond, the first call (year 3 at \$1,020) gives 2.83%, a later call (year 5 at \$1,010) gives 3.42%, and final maturity gives 4.01%. The minimum is 2.83% — so YTW = 2.83%, and that is the number you quote.

Here is the subtlety that trips up even experienced people: **the worst case is not always the call.** It depends on whether the bond is at a premium or a discount.

- For a **premium** callable bond, calling *early* is worst (you lose your premium fast), so YTW = the earliest YTC — usually lower than the YTM.
- For a **discount** callable bond, running to *maturity* is worst (an early call at par would *accelerate* a gain you wanted), so YTW = the YTM, and the call would actually help you.

#### Worked example: when maturity, not the call, is the worst case

*A discount callable bond flips the logic.* Take a Northwind callable bond now trading at a discount of \$940 — same 5% coupon, 10-year maturity, first callable in 3 years at \$1,020. Compute both:

- YTM (to year 10): discount ten \$50 coupons plus \$1,000 down to \$940 → **5.81%**.
- YTC (to year 3 at \$1,020): discount three \$50 coupons plus \$1,020 down to \$940 → **7.94%**.

Here the call is the *good* outcome — being repaid \$1,020 in three years for a bond you bought at \$940 hands you a big gain quickly, lifting the YTC to 7.94%. The *worst* you can do is have the bond run to maturity at 5.81%. So YTW = min(5.81%, 7.94%) = **5.81%**, the YTM. And in practice the issuer will *not* call a bond trading at a discount, because its coupon is below current market rates — there is nothing to refinance. So you will probably get the 5.81%, and the call is a free upside lottery ticket.

*Yield to worst means lowest, not earliest — for a discount bond the worst outcome is the bond simply surviving to maturity.*

## The hidden assumption: YTM and reinvestment

We have treated YTM as "the return you earn if you hold to maturity," and that is the standard shorthand — but it hides an assumption that matters enormously in the real world. The YTM equation discounts every coupon at the rate $y$. The mathematical mirror of that is a promise: **YTM only equals your actual realized return if you can reinvest every coupon you receive at the YTM rate, all the way to maturity.**

![A pipeline showing that buying a par bond at 5 percent yield to maturity gives a realized return of 5 percent only if each 50 dollar coupon is reinvested at 5 percent, ending at 1276 dollars, but holding coupons as cash drops the realized return to 4.56 percent](/imgs/blogs/the-many-yields-current-yield-ytm-and-yield-to-call-6.png)

Think about why. Your bond pays you \$50 a year. That \$50 does not just sit in your pocket earning nothing in the YTM's accounting — the math assumes you take each coupon and put it to work earning the same yield until the bond matures. The coupons earn interest, and that interest earns interest. If reality cooperates and you can reinvest at 5%, the YTM comes true. If you cannot — if rates have fallen, or you spend the coupons, or you let them sit in cash — your *realized* return falls short of the quoted YTM. This is called **reinvestment risk**, and it is the single biggest gap between the yield on the screen and the return in your account.

#### Worked example: reinvestment makes or breaks the YTM

*Two ways to hold the same bond.* You buy Northwind's 5-year, \$1,000 par bond at par for \$1,000, quoted at a 5% YTM. You will receive five \$50 coupons and your \$1,000 back. The question is what you do with each \$50 as it arrives.

**Scenario A — reinvest at 5%.** Each coupon goes straight back into something yielding 5%, compounding to year 5. The first coupon (received at year 1) compounds for 4 years; the last (year 5) not at all. Summing the compounded coupons gives \$276.28, plus your \$1,000 face = **\$1,276.28** total. Turn that into an annual growth rate: (1,276.28 ÷ 1,000)^(1/5) − 1 = **5.00%**. The YTM came true, exactly.

**Scenario B — hold coupons as cash (0%).** You stuff each \$50 in a drawer earning nothing. At year 5 you have five \$50 coupons (\$250) plus the \$1,000 face = **\$1,250**. Growth rate: (1,250 ÷ 1,000)^(1/5) − 1 = **4.56%**. You earned 4.56%, not the 5% the screen promised — a shortfall of 44 basis points, purely from not reinvesting.

*The quoted YTM is a promise conditional on reinvesting your coupons at that same yield; let the coupons sit idle and your real return drifts below the number you were quoted.*

The cash drawer in Scenario B is a deliberately extreme stand-in for the realistic case: rates *fall* after you buy, and you are forced to reinvest each coupon at a lower yield than the one your bond is quoted at. This is the situation that bites real investors, because falling rates and a high quoted YTM tend to go together — you lock in a great-looking yield right before the reinvestment environment for your coupons deteriorates.

#### Worked example: reinvestment risk when rates fall

*The yield you locked in versus the yield you live with.* You buy the same 5-year par bond at a 5% YTM, fully intending to reinvest every coupon. But the day after you buy, market rates drop to 2% and stay there. Now each \$50 coupon, as it arrives, can only be reinvested at 2%, not 5%. Compound the five coupons at 2% instead of 5% and you accumulate about \$260 of coupons-plus-interest instead of \$276; add the \$1,000 face and you end with roughly \$1,260, a realized return near 4.7% rather than the 5% on the screen. You did everything right — you held to maturity and reinvested diligently — and still fell short of the quoted YTM, because the YTM silently assumed a 5% reinvestment world that vanished.

*Reinvestment risk means a falling-rate environment can quietly shave your realized return below the YTM you were quoted, even if you hold to maturity exactly as planned.*

This is also why **zero-coupon bonds** — bonds that pay no coupons at all, just a single lump at maturity — have *no* reinvestment risk. With no coupons to reinvest, their quoted YTM is exactly the return you lock in. The price you pay for that certainty is that all your money is tied up until maturity, and their prices swing more violently when rates move (a topic for the duration post).

## Putting it together: discount versus premium, side by side

We have now met every member of the yield family. The cleanest way to lock it in is to see all three core yields for the same bond at three prices in one grid.

![A comparison matrix showing the same 5 percent bond at three prices, where a 920 discount gives YTM above current yield above coupon, par at 1000 makes all three equal at 5 percent, and a 1080 premium gives coupon above current yield above YTM](/imgs/blogs/the-many-yields-current-yield-ytm-and-yield-to-call-7.png)

Read the matrix top to bottom and the whole post snaps into a single picture:

- **Discount (\$920):** coupon 5.00% < current yield 5.43% < YTM 6.95%. The bargain price means a gain at maturity, so total return (YTM) is the highest of the three.
- **Par (\$1,000):** all three equal 5.00%. No gain, no loss — income is the only return, so every measure agrees.
- **Premium (\$1,080):** coupon 5.00% > current yield 4.63% > YTM 3.24%. Paying up means a loss at maturity, so total return (YTM) is the lowest.

Memorize the direction, not the digits: **a discount fans the yields upward toward YTM; a premium fans them downward toward YTM; par collapses them to a point.** The coupon rate sits frozen in the middle of the fan as the reference line. That single image is the most portable thing in this post.

#### Worked example: the par bond as the anchor

*Why par is the dividing line.* Consider why the at-par row is so clean. You pay \$1,000 and you get back \$1,000 — the principal washes out, contributing zero gain or loss. The *only* return left is the \$50 coupon on your \$1,000, which is 5%. Current yield is \$50/\$1,000 = 5%. And YTM, with no capital gain or loss to fold in, also lands on 5%. There is genuinely no daylight between the three numbers because there is no second source of return for them to disagree about. Every discount or premium is just a deviation from this anchor: the further the price strays from \$1,000, the wider the yields fan apart.

*Par is the one price at which all the yield measures tell the same story, which is exactly why it is the reference point the others are defined against.*

### A quick comparison table

| Yield measure | Formula (in words) | What it counts | What it ignores | Best for answering |
|---|---|---|---|---|
| Coupon rate | coupon ÷ face value | the fixed dollar income | your price entirely | "How big is the coupon check?" |
| Current yield | coupon ÷ market price | income vs. what you paid | the gain/loss at maturity | "What income am I getting on my money?" |
| Yield to maturity | rate where PV(cash flows) = price | income + pull to par | call risk; reinvestment shortfall | "What's my total return if held to maturity?" |
| Yield to call | YTM math, ending at first call | return if called early | scenarios after the call | "What if the issuer calls it?" |
| Yield to worst | min(YTM, all YTCs) | the least favorable ending | upside scenarios | "What's my floor?" |

Use the right tool for the question. An income-focused retiree leans on current yield; a buy-and-hold investor lives by YTM; anyone touching callable or mortgage bonds quotes yield to worst and never the YTM.

## Why "yield" is the price of money for everyone

It is tempting to treat all this as inside baseball for bond traders. It is not. The yield to maturity on a bond is, quite literally, **the price of money for a given borrower over a given horizon**, and that price ripples out to touch nearly every financial decision you will ever make.

Start with the mechanism. The YTM on a US Treasury is the return demanded for lending to the US government — the closest thing to a risk-free rate that exists. Every other interest rate in the economy is built on top of it. A corporate bond yields the Treasury YTM *plus a spread* for default risk. A mortgage rate tracks the yield on long-term Treasuries and mortgage bonds. The rate on your car loan, your credit card, your savings account — all are anchored, directly or indirectly, to the yields set in the bond market every single day. When commentators say "the 10-year yield rose," they are reporting a change in the YTM of one specific bond, and that change reprices trillions of dollars of mortgages, loans, and stocks. (The macro view of how the level of those yields gets set lives in [interest rates as the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).)

This is why the *distinction* between the yield measures matters beyond a trading desk. A homeowner deciding whether to refinance is, without knowing it, exercising a call option exactly like the issuer of a callable bond — and the bank that holds the mortgage suffers the same reinvestment risk we worked through. A pension fund matching its future payouts to bond cash flows must reckon with whether the YTM it locks in will survive reinvestment. A saver comparing two CDs is comparing two yields that may be quoted on different compounding conventions. The vocabulary of this post is not specialized trivia; it is the grammar of how money is priced across time, and once you read it fluently you see it everywhere.

#### Worked example: your mortgage is a callable bond you issued

*Flip the perspective.* When you take a \$300,000 30-year mortgage at 6%, you are the *issuer* of a callable bond, and your bank is the holder. You promise the bank fixed monthly payments for 30 years — your coupons. But your mortgage contract lets you prepay (refinance or sell the house) whenever you like — that is your embedded call option. If rates fall to 4%, you refinance: you "call" the 6% loan, pay the bank back early, and reissue at 4%. The bank is now holding cash it must reinvest at 4% instead of the 6% it was earning — its realized yield is the yield-to-call, not the yield-to-maturity it underwrote. The bank knew this could happen, which is why mortgage rates carry a small premium over comparable Treasury yields: the bank is charging you for the call option you hold over it.

*Every fixed-rate mortgage in America is a callable bond from the lender's point of view, which is why the yield-to-call and reinvestment ideas in this post quietly govern the largest debt market on earth.*

## Common misconceptions

**"A higher coupon means a higher return."** No — the coupon is the income, not the return. A 5% coupon bond bought at \$1,080 returns only 3.24%, while a 3% coupon bond bought at a deep enough discount could return more. The coupon tells you the size of the check; the *price* tells you the return. Chasing fat coupons without checking the price is how investors talk themselves into low-yielding premium bonds.

**"Current yield is the bond's yield."** Current yield is *an* income snapshot, not *the* yield. It ignores the single biggest non-coupon component of return — the gain or loss at maturity. Quoting current yield as "the yield" overstates premium bonds and understates discount bonds, sometimes by hundreds of basis points. The market means YTM (or YTW) when it says "yield."

**"YTM is the guaranteed return if I hold to maturity."** Only if you reinvest every coupon at the YTM rate. The YTM is a return *conditional on reinvestment*; if rates fall after you buy, you reinvest your coupons at lower rates and your realized return drifts below the quoted YTM — that is reinvestment risk, and our worked example showed it costing 44 bps just from holding coupons as cash. The only bond whose YTM is truly locked is a zero-coupon bond, because it has no coupons to reinvest.

**"Yield to worst means the bond gets called early."** YTW means the *lowest* yield outcome, which is not always the earliest call. For a discount callable bond, the worst case is the bond surviving to maturity, and an early call would actually *help* you. "Worst" is about the yield, not the timing.

**"All these yields are basically the same number, so it doesn't matter which I use."** They are the same *only at par*. The further a bond trades from \$1,000, the more they diverge — for our premium bond the coupon (5.00%) and the YTW (2.83%) differ by 217 basis points, which on a large position is real money. Picking the wrong measure can make a bad bond look good. The whole point of knowing the vocabulary is that the differences are decision-grade, not cosmetic.

**"A callable bond and a regular bond with the same coupon and price are equivalent."** They are not — the callable bond is worse for you, because you have *sold* the issuer an option to take it back when it suits them. That embedded option is why callable bonds offer a slightly higher yield than otherwise-identical non-callable bonds: you are being paid a little extra to bear the call risk. If two bonds yield the same and one is callable, the callable one is the worse deal.

## How it shows up in real markets

**The 2020–2021 corporate refinancing wave.** When the Federal Reserve slashed rates to near zero in the COVID crisis, investment-grade companies rushed to call their old, high-coupon callable bonds and refinance at record-low rates. Holders who had bought those bonds at a premium for the juicy coupons suddenly got handed back the call price and watched their bonds disappear — their realized return was the yield to call, not the yield to maturity they may have been eyeing. It was a textbook, market-wide demonstration that on a premium callable bond, the YTC is the number that comes true. Anyone underwriting to YTM that year was unpleasantly surprised; anyone quoting yield to worst was not. The episode is part of why the [credit market](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads) treats YTW as the default quote for callable paper.

**US Treasuries: clean YTM, no call games.** Modern US Treasury notes and bonds are non-callable, which is part of why they are the world's benchmark — their quoted yield *is* the yield to maturity, with no call schedule to muddy it. When you read "the 10-year Treasury yield is 4.2%," that is a clean YTM on a bond that will run exactly ten years. (The Treasury did issue callable bonds decades ago and stopped, precisely to keep its yields unambiguous.) This is why every other bond's yield is quoted as a *spread* over the comparable Treasury — the Treasury gives a call-free, default-free YTM to measure against. The [government bond post](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration) builds out why that clean benchmark anchors the whole system.

**Mortgage-backed securities and prepayment as a giant call.** The largest callable market on earth is not corporate bonds — it is US mortgage-backed securities, where millions of homeowners hold the call option in the form of the right to prepay (refinance) their mortgages. When rates fall, homeowners refinance en masse, the mortgage bonds get "called" through prepayment, and investors get their principal back early to reinvest at lower rates — the same reinvestment problem we worked through, at a \$10-trillion scale. MBS investors live and die by yield-to-worst-style analysis and elaborate prepayment models, because the "maturity" of an MBS is a moving target driven by the rate environment. It is the clearest real-world proof that reinvestment risk and call risk are the same animal wearing different costumes.

**The premium-bond trap in retail brokerage.** Brokerage platforms have historically led with the **current yield** or even the coupon when displaying bonds, because those numbers look highest and most appealing. A retail investor sees a "5% bond" trading at \$1,080, anchors on the 5%, and buys — collecting a real but smaller 4.63% current yield while quietly locking in a 3.24% total return as the premium bleeds back to par. Regulators and better platforms now push **yield to maturity and yield to worst** to the front for exactly this reason. The lesson from this post is the defense: always find the YTM (and YTW if callable) before the coupon seduces you.

**Municipal bonds and the de minimis trap.** The US municipal bond market — where states and cities borrow, often with tax-free coupons — is full of callable bonds, and many trade at premiums precisely because investors prize the high tax-free coupon. A muni buyer who fixates on the coupon and ignores the yield to worst can badly overpay, because the issuer will call the bond the moment refinancing is cheaper, handing back the call price and ending the tax-free income stream early. There is an extra twist unique to munis: a tax rule (the *de minimis* rule) can turn part of a deeply discounted muni's gain into ordinary income rather than a capital gain, which lowers the *after-tax* yield below the screen's pre-tax YTM. It is a reminder that for some bonds even the YTM is not the final word — the after-tax, after-call yield is what you actually keep, and getting there starts with knowing which yield the screen is showing you.

**The 2022 bond rout and yields that finally looked like yields.** For most of the 2010s, yields were so low that the distinctions in this post felt almost academic — a 1.5% Treasury yield is a 1.5% Treasury yield, and there was not much spread between the measures to argue about. Then 2022 happened: the Fed raised rates at the fastest pace in four decades, the 10-year Treasury yield rose from roughly 1.5% to over 4%, and bond *prices* fell hard (remember the seesaw). Suddenly, bonds issued at low coupons in 2020–2021 traded at steep discounts, and their YTMs leapt above their coupons in exactly the discount ordering this post describes. Investors who had ignored the gap between coupon and YTM during the placid years got a vivid, painful refresher: the bonds they thought of as "my 2% bonds" were now "my deep-discount bonds yielding 4.5% to a new buyer." The episode reset a generation's intuition about why the yield, not the coupon, is the number that matters — and the [2022 case study](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) on stocks and bonds falling together traces the wider damage.

**Volcker's high-coupon Treasuries and the discount-to-premium swing.** When Paul Volcker drove rates into the high teens in the early 1980s to break inflation, the Treasury issued bonds with coupons of 13%, 14%, even 15%. As rates collapsed over the following two decades, those bonds soared far above par — a 14% coupon bond in a 6% world traded at an enormous premium. Investors who held them watched their *coupon rate* stay frozen at 14% while their *YTM* fell toward the prevailing 6%, a living illustration of the premium ordering: coupon rate far above current yield far above YTM. Those who bought at the premium for the eye-popping coupon earned the modest YTM, not the headline 14%. (The episode is the centerpiece of the [Volcker case study](/blog/trading/macro-trading/sovereign-debt-and-the-bond-vigilantes) on how policy reshapes the bond market.)

## When this matters to you, and where to go next

The next time you look at a bond — in a brokerage app, a fund's holdings, or a headline about Treasury yields — you now have the decoder. Find the coupon (the income), the current yield (income on your money), and the YTM or YTW (your real total return), and read the *ordering* between them: it tells you instantly whether you are looking at a discount or a premium, and whether the screen's flashiest number is flattering you. The single habit worth keeping is to never let a high coupon stand in for a high return — convert to YTM, and if the bond is callable, to yield to worst, before you decide anything.

This is educational, not investment advice; the goal is to make the mechanics legible, not to point you at any particular bond.

Where this leads next in the series: the YTM you now understand is the raw material for **duration** — how much a bond's price moves when yields change — which is the next foundational idea and the bridge from "what is yield" to "what is risk." From there the series builds out the [yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) (yields across every maturity at once) and [credit spreads](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads) (the extra yield for taking default risk). If you want the full mathematical machinery behind the YTM solve and the day-count conventions we waved past, the [bond pricing](/blog/trading/quantitative-finance/bond-pricing) deep dive is the place to go. And if you want the macro view of why the level of yields moves in the first place, start with [interest rates as the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).

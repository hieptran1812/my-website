---
title: "Immunization and duration matching: how pensions and insurers hedge"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into the biggest job in fixed income — asset-liability management — covering immunization, the duration gap, cash-flow matching versus duration matching, liability-driven investing, and how an unmanaged gap brought down Silicon Valley Bank in 2023."
tags: ["fixed-income", "bonds", "immunization", "duration-matching", "asset-liability-management", "liability-driven-investing", "pensions", "insurance", "duration-gap", "svb-2023"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — pensions, insurers, and banks hold bonds to fund future promises, and *immunization* means matching the duration of those bonds to the duration of the promises, so a rate move hits both sides equally and the surplus is protected.
> - A bond's price moves opposite to rates, and the *size* of that move is its **duration** — so if your assets and your liabilities have the same duration, a rate move changes both by the same percentage and your **funded ratio** barely budges.
> - The number that matters is the **duration gap** (asset duration minus liability duration): a gap of zero is immunized, a positive gap bets that rates fall, a negative gap bets that they rise.
> - **Cash-flow matching** (dedication) buys bonds whose payments land exactly when each liability is due; **duration matching** (immunization) is cheaper and more flexible but must be **rebalanced** as the match drifts.
> - **Liability-driven investing (LDI)** is this idea scaled up to a whole pension fund — manage the assets relative to the liabilities, not relative to a stock index.
> - **Silicon Valley Bank in 2023** is the textbook failure: long, fixed-rate bonds funded by deposits that could leave in an afternoon — a giant unmanaged duration gap that a rate rise turned into a loss bigger than the bank's entire capital.

Here is a question that sounds simple and turns out to run the financial lives of hundreds of millions of people. A pension fund has promised to pay a retiree \$10,000,000 worth of benefits, but not today — in roughly seven years. The fund has the money to cover it, parked in bonds. Then interest rates jump two percentage points overnight. Is the pension still able to keep its promise, or has it just quietly gone broke?

The honest answer is: *it depends entirely on how the bonds were chosen.* With one portfolio, the rate shock is a non-event — the pension wakes up exactly as funded as it went to sleep. With another portfolio holding the very same dollar amount, the same shock blows a hole in the fund that takes years and extra contributions to repair. Nothing about the promise changed. Nothing about the amount of money changed. The only difference is whether someone matched the *timing sensitivity* of the assets to the *timing sensitivity* of the liability.

![A balance beam showing a bond portfolio on the left and a pension liability on the right, both labeled with a duration of seven years, staying level when a rate move pushes down on the fulcrum because both sides fall by the same amount](/imgs/blogs/immunization-and-duration-matching-how-pensions-and-insurers-hedge-1.png)

The diagram above is the mental model for the whole post: a balance beam with your bonds on one side and your future promise on the other. When the two sides have the same *duration* — the same sensitivity to rates — a rate move pushes down on both pans equally and the beam stays level. That is **immunization**, and it is the single most important practical job in all of fixed income. Pensions do it. Life insurers do it. Banks are supposed to do it. When they do it well, you never hear about them. When they get it wrong, you get Silicon Valley Bank. (Everything here is educational, not investment advice — the goal is to understand the mechanism that quietly governs trillions of dollars of retirement and insurance money.)

This is the capstone of the duration story. In the earlier posts we built [duration as a bond's center of gravity and its rate sensitivity](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income), turned it into dollars with [modified duration and DV01](/blog/trading/fixed-income/modified-duration-and-dv01-measuring-and-trading-rate-risk), corrected it with [convexity](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story), and saw [the two faces of yield — price risk versus reinvestment risk](/blog/trading/fixed-income/reinvestment-risk-and-the-two-faces-of-yield). Now we put all of it to work on the biggest real-world job those tools were invented for: making sure the money is there when the promise comes due.

## Foundations: liabilities, assets, and the funded ratio

Let's build every term from zero, because the whole subject is just three ideas stacked on top of each other.

A **liability** is a promise to pay money in the future. When a pension fund tells a worker "we will pay you a benefit," that is a liability. When a life insurer sells you a policy that pays \$500,000 to your family when you die, that is a liability. When a bank takes your \$10,000 deposit, that is a liability — the bank owes you that \$10,000 back. A liability is, in plain terms, *money you owe later.*

An **asset** is something you own that is worth money. For these institutions, the assets are overwhelmingly **bonds** — tradable loans that pay a fixed stream of cash. (If the word "bond" is new, the very first post in this series, [why bonds rule the world](/blog/trading/fixed-income/why-bonds-rule-the-world-fixed-income-introduction), builds it from scratch.) A pension buys bonds; the bonds pay coupons and principal; that incoming cash is what the pension uses to pay the retiree.

The institution's job is to make sure the **assets** are enough to cover the **liabilities**. The ratio of the two has a name you should memorize, because it is the scoreboard for this entire field:

$$\text{Funded ratio} = \frac{\text{value of assets}}{\text{value of liabilities}}$$

- A funded ratio of **100%** means assets exactly equal liabilities — the promise is fully covered, no more, no less.
- Above 100% is a **surplus** (more assets than promises). Below 100% is a **deficit** (a shortfall the institution must close).

Here is the subtle part that makes this whole topic hard, and interesting. Both the assets *and* the liabilities have a present value, and **both move when interest rates move.** We are used to thinking of a debt as a fixed number — "I owe \$10,000,000." But the *present value* of that promise — what it would cost you today to set aside enough money to cover it — depends on the interest rate you can earn in the meantime. The higher rates are, the less you need to set aside today, because the money you set aside grows faster. So the liability has a present value that *falls when rates rise*, exactly like a bond.

### Why the liability behaves like a bond

This is the keystone idea, so let's nail it with the running example.

A pension owes **\$10,000,000 in seven years.** What is that promise worth *today*? You discount it back at the prevailing interest rate. At a 5% rate:

$$\text{PV} = \frac{\$10{,}000{,}000}{(1.05)^{7}} = \frac{\$10{,}000{,}000}{1.4071} = \$7{,}106{,}813$$

So the pension needs about **\$7.11 million** today, invested at 5%, to grow into \$10,000,000 in seven years. That \$7.11 million is the present value of the liability — and notice it behaves *exactly* like a 7-year zero-coupon bond. A zero-coupon bond pays one lump sum at the end and nothing in between; so does this liability. They are mathematically the same animal. That means the liability has a **duration**, just like a bond does.

If rates rise to 6%, the same promise is suddenly worth less today:

$$\text{PV} = \frac{\$10{,}000{,}000}{(1.06)^{7}} = \frac{\$10{,}000{,}000}{1.5036} = \$6{,}650{,}571$$

The promise didn't shrink — the pension still owes \$10,000,000 in seven years. But its *present value* dropped by about \$456,000, roughly 6.4%, because higher rates mean you need less set aside today. **The liability got cheaper to fund when rates rose.** Hold that thought, because it is the entire trick: if your assets *also* get cheaper by the same percentage when rates rise, the two cancel and your funded ratio doesn't move.

*The intuition: a future promise is a negative bond — discount it like a bond, and it gains and loses value with rates exactly like a bond.*

### Three institutions, three shapes of promise

Before we go further, it helps to see *who* runs these duration-matching books and why their liabilities look so different — because the shape of the promise dictates the shape of the hedge.

A **defined-benefit pension fund** promises retirees a stream of monthly payments for the rest of their lives, often indexed to inflation. Its liabilities are *long* (decades), *numerous* (one stream per member), and *uncertain* (nobody knows exactly how long each retiree will live — that is **longevity risk**, the chance people live longer than assumed and the fund pays out for more years than it budgeted). A pension's liability duration is typically 12 to 20 years, longer than almost any single bond it can buy, which is why pensions are structurally *short* duration relative to what they owe.

A **life insurer** is the mirror image with extra precision. Selling annuities and whole-life policies, it owes large sums far in the future, and it prices those policies assuming it can earn a certain return on the premiums in the meantime — so it is acutely sensitive to rates. Insurers are the natural buyers of the very longest bonds (30-year and beyond) precisely because their liabilities reach that far out. Where a pension can be a little loose, a regulated insurer must hold capital against its duration gap, so its ALM is tighter and more rule-bound.

A **bank** is the odd one out, and the most dangerous. Its liabilities are *short* — deposits that can be withdrawn on demand, with a duration near zero — while its assets (loans and bonds) are *long*. A bank therefore runs a *positive* duration gap by its very nature: it is in the business of "borrowing short and lending long," capturing the gap between short and long rates (the term premium). That business model is profitable in normal times and lethal when rates rise fast and depositors flee at once — which is the whole SVB story we will reach later.

The common thread: each institution owns bonds not to speculate, but to make a set of future cash promises come true. The differences in how *long* and how *certain* those promises are determine how each one hedges. Now, the tool that measures all of it.

### Duration, in one paragraph, because everything hinges on it

You met duration in the earlier posts; here is the one-paragraph version you need. **Duration** is how sensitive a present value is to a change in interest rates, measured in years. A duration of 7 means: for every 1% (100 basis point) rise in rates, the value falls by roughly 7% (more precisely, by the *modified* duration, which is the plain duration divided by one plus the periodic yield — a small adjustment we will use below). A *basis point* is one hundredth of a percent — 0.01% — and you will see rate moves quoted in "bps" constantly. The longer the duration, the bigger the price swing for the same rate move. A 7-year zero-coupon liability has a duration of about 7 years. A portfolio of bonds has a duration too — the value-weighted average of its bonds' durations. **Immunization is the act of making those two durations equal.**

## Asset-liability management: the job that bonds were invented for

Step back and look at *why* these institutions exist and what they are actually doing, because it reframes the whole point of bonds.

A retail investor buys bonds to earn income or to diversify a stock portfolio. A pension, an insurer, or a bank buys bonds for a completely different reason: **to fund specific future promises.** Their bonds are not there to "beat the market" — they are there to be worth the right amount at the right time. This discipline has a name: **asset-liability management**, usually shortened to **ALM**. The goal of ALM is not to maximize return. It is to make sure that whatever happens to interest rates, the assets keep pace with the liabilities so the funded ratio stays healthy.

This is why the *biggest* buyers of bonds in the world are exactly these liability-driven institutions — we covered the whole roster in [who buys bonds](/blog/trading/fixed-income/who-buys-bonds-the-global-demand-for-safe-income). A pension fund managing \$50 billion is not trying to guess where rates go. It has made tens of thousands of promises to retirees stretching out fifty years, and its entire job is to own a bond portfolio that tracks the value of those promises.

The mistake that ruins institutions is to manage the *assets in isolation* — to ask "did my bonds go up?" instead of "did my bonds go up *as much as my liabilities went up?*" A bond portfolio that lost 10% looks like a disaster on its own. But if the liabilities it was funding *also* fell 10% in present value, the institution is exactly as funded as before — nothing bad happened at all. Conversely, a portfolio that *gained* 5% looks great in isolation, but if the liabilities gained 12%, the institution just got dramatically less funded. **The asset's return in isolation is almost meaningless. What matters is the asset's return relative to the liability.**

![A side by side comparison of a matched book and a mismatched book showing that when asset duration equals liability duration the duration gap is zero and the surplus is protected, but when asset duration exceeds liability duration the gap is positive and the surplus craters on a rate rise](/imgs/blogs/immunization-and-duration-matching-how-pensions-and-insurers-hedge-3.png)

#### Worked example: the same shock to a matched and a mismatched book

Take our pension owing \$10,000,000 in seven years, with a liability present value of \$7.11 million and a liability duration of 7.0 years. The fund holds exactly \$7.11 million of bonds, so today it is **100% funded.** Now rates jump **+2% (200 basis points).** Watch what happens under two different bond portfolios.

**Portfolio A — matched (asset duration 7.0).** The bonds have the same duration as the liability. The liability's present value falls by its modified duration times the rate move: modified duration is $7.0 / 1.05 \approx 6.67$, so the liability falls by roughly $6.67 \times 2\% \approx 13.3\%$, from \$7.11M to about \$6.16M. The assets have the same duration, so they fall by the same $\approx 13.3\%$, from \$7.11M to about \$6.16M. New funded ratio: $\$6.16\text{M} / \$6.16\text{M} = 100\%$. **Nothing happened.** The shock that vaporized 13% of the asset value also vaporized 13% of the liability value, and they cancelled perfectly.

**Portfolio B — mismatched (asset duration 3.0).** Same \$7.11M, but invested in short bonds with a duration of only 3.0. Now the assets fall by only $\approx (3.0/1.05) \times 2\% \approx 5.7\%$, to about \$6.70M. But the liability still falls 13.3%, to \$6.16M. New funded ratio: $\$6.70\text{M} / \$6.16\text{M} \approx 109\%$. The pension just became *over*-funded by a rate rise — which sounds nice until you realize it means the bet works in reverse too.

*The intuition: with a matched book the rate move cancels out; with a mismatched book the rate move becomes a bet, and you only profit if rates go the way you happened to be positioned for.*

That last point is the dagger. A mismatch is not "safe versus risky" — it is a *bet on the direction of rates*, whether you intended to make one or not. Portfolio B profits if rates rise (short assets, long liability), and it *loses* if rates fall — the liability would balloon faster than the short assets. Let's prove that the matched book really is flat in *both* directions.

## The funded ratio is the scoreboard: matched stays flat, mismatched swings

The cleanest way to see immunization is to plot the funded ratio against the change in rates. A matched book draws a flat horizontal line — the funded ratio is roughly 100% no matter what rates do. A mismatched book draws a steep sloped line — the funded ratio swings wildly with every rate move.

![A chart with the change in market rates in basis points on the horizontal axis and the funded ratio as a percentage on the vertical axis, showing a flat blue line at one hundred percent for the matched book and a steep orange line for the mismatched book that swings from eighty percent to one hundred sixteen percent](/imgs/blogs/immunization-and-duration-matching-how-pensions-and-insurers-hedge-2.png)

The blue flat line is the whole goal. It says: *whatever rates do, our funded ratio stays put.* The institution has converted an unknown — "where will rates go?" — into a non-issue. The orange line is the danger: the same institution, mismatched, sees its funding fortune yoked to a variable it cannot control or predict.

#### Worked example: the mismatch cuts both ways

Continue with Portfolio B (asset duration 3, liability duration 7), starting at 100% funded with \$7.11M on each side. We already saw that **+2%** pushed the funded ratio up to ~109%. Now suppose instead rates fall **−2% (−200 bps).**

The assets *rise* by $\approx (3.0/1.05) \times 2\% \approx 5.7\%$, to about \$7.51M. The liability rises by $\approx (7.0/1.05) \times 2\% \approx 13.3\%$, to about \$8.05M. New funded ratio: $\$7.51\text{M} / \$8.05\text{M} \approx 93\%$. The pension just fell into a **7% deficit** purely because rates dropped — it now owes more (in present value) than it owns, and somebody has to make up the \$0.54 million difference.

*The intuition: a duration mismatch is a hidden bet on rates; falling rates are the pension's enemy because liabilities are long-duration and a rate fall inflates them faster than short assets can keep up.*

This is exactly why pension funds *hate* falling rates, which surprises people. Lower rates are supposed to be "good for bonds." But a pension's liabilities are usually *longer* than its assets, so a rate fall raises the present value of what it owes faster than the value of what it owns. Many corporate pension plans that looked healthy in 2007 were suddenly deep in deficit by 2012 — not because their assets crashed, but because rates collapsed and their long-dated liabilities ballooned. The assets held up fine. The *gap* did the damage.

## The duration gap: the one number that tells you the bet

We can compress everything above into a single number. The **duration gap** is the duration of the assets minus the duration of the liabilities:

$$\text{Duration gap} = D_{\text{assets}} - D_{\text{liabilities}}$$

- **Gap = 0** → immunized. A rate move hits both sides equally; the surplus is protected.
- **Gap > 0** (assets longer than liabilities) → you *lose* when rates rise and *gain* when rates fall. You are implicitly long duration — betting rates fall.
- **Gap < 0** (assets shorter than liabilities) → you *gain* when rates rise and *lose* when rates fall. You are implicitly short duration — betting rates rise.

A bank typically runs a **positive** gap: it lends long (30-year mortgages, multi-year loans) and funds short (deposits that can leave any day). So a bank is structurally hurt by rising rates unless it hedges — a fact that will matter enormously when we get to SVB. A pension typically runs a **negative** gap: its liabilities (decades of future benefits) are longer than the bonds it can easily buy, so it is structurally hurt by *falling* rates. Each institution's *natural* gap points in a different direction, and ALM is the work of pushing that gap back toward zero — or deliberately, and consciously, leaving a small gap when the institution wants to take a measured view on rates.

#### Worked example: sizing a bank's duration gap in dollars

A small bank has **\$1 billion in assets** with an average duration of **5 years**, funded by **\$900 million in deposits** with an average duration of **0.5 years** (deposits reprice fast — they are nearly cash) plus **\$100 million of equity** (the owners' capital). The dollar duration of the assets is $\$1{,}000\text{M} \times 5 = \$5{,}000$ million-years. The dollar duration of the liabilities is $\$900\text{M} \times 0.5 = \$450$ million-years. The mismatch in dollar terms is \$4,550 million-years.

Now rates rise **+1% (100 bps).** Assets fall by $5 \times 1\% = 5\%$ of \$1,000M = **−\$50M.** Liabilities fall by $0.5 \times 1\% = 0.5\%$ of \$900M = **−\$4.5M.** The bank's equity absorbs the difference: it drops by $\$50\text{M} - \$4.5\text{M} = \$45.5\text{M}$, from \$100M to about \$54.5M. A single 1% rate move just erased **45% of the bank's capital.** A 2% move would erase about 90% of it. This is not a hypothetical — it is precisely the arithmetic that broke SVB.

*The intuition: a bank's equity is a thin sliver sitting on top of a giant duration gap, so even a modest rate move, amplified by the leverage, can swallow the capital whole.*

## Immunization: the precise recipe

Now let's make the recipe exact. **Immunization** (the term comes from the 1950s actuary Frank Redington, who wanted to "immunize" a life insurer's surplus against rate moves) means structuring the assets so the surplus is protected against a rate change. The classic two conditions are:

1. **Match present values.** The present value of the assets must equal the present value of the liabilities. (Otherwise you are under- or over-funded before rates even move.)
2. **Match durations.** The duration of the assets must equal the duration of the liabilities. This is what makes the *first-order* effect of a rate move cancel.

There is a third, more advanced condition that the best practitioners add:

3. **Make asset convexity ≥ liability convexity.** Duration is only a straight-line approximation; **convexity** is the curvature correction (the full story is in [the convexity post](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story)). If the assets are *more* convex than the liabilities, then for a large rate move in *either* direction the assets fall a little less and rise a little more than the liabilities — the mismatch works slightly in your favor. This is why immunizers often hold a **barbell** (some very short bonds and some very long bonds) rather than a single bond at the target duration: a barbell has the same duration but *more* convexity than a single bullet bond, so it gives a small free cushion against big moves.

#### Worked example: building the immunizing portfolio

Our pension owes \$10,000,000 in seven years. We computed the liability present value at 5%: **\$7,106,813**, and its duration: **7.0 years.** To immunize, we need a bond portfolio worth \$7.11 million with a duration of 7.0.

We could buy a single 7-year bond — that satisfies conditions 1 and 2. But to also pick up the convexity cushion in condition 3, we build a **barbell** out of two bonds:

- A **3-year note** with a duration of 3.0 years.
- A **12-year bond** with a duration of 12.0 years.

We need the value-weighted average duration to equal 7.0. Let $w$ be the fraction in the 3-year note:

$$w \times 3.0 + (1 - w) \times 12.0 = 7.0$$

$$12.0 - 9.0\,w = 7.0 \quad\Rightarrow\quad w = \frac{5.0}{9.0} \approx 0.556$$

So put about **56% in the 3-year note** and **44% in the 12-year bond.** In dollars on a \$7.11M base: roughly **\$4.0M in the 3-year** and **\$3.1M in the 12-year.** Check the duration: $(\$4.0\text{M} \times 3.0 + \$3.1\text{M} \times 12.0) / \$7.1\text{M} = (12.0 + 37.2)/7.1 = 49.2/7.1 \approx 6.9 \approx 7.0$ years. Matched.

![A worksheet matrix showing the pension liability of ten million dollars in seven years with a present value of seven point one one million and a duration of seven years, matched by a barbell portfolio of four million in three year notes and three point one million in twelve year bonds whose value weighted duration equals seven years](/imgs/blogs/immunization-and-duration-matching-how-pensions-and-insurers-hedge-7.png)

*The intuition: immunizing is just funding the present value of the promise with a bond portfolio whose duration equals the promise's duration — and a barbell does it with a convexity bonus that a single bond cannot.*

## Measuring the liability: the discount-rate fight that decides everything

There is a quiet, ferocious argument buried inside that present-value formula, and it determines whether a pension is reported as healthy or broke. To get the present value of the liability, you have to *choose a discount rate.* We used 5%. But where does that 5% come from, and who gets to pick it?

Recall the mechanics: the present value of a fixed future promise *falls* as the discount rate *rises*. So a pension that discounts its \$10,000,000 obligation at 7% reports a much smaller liability — and therefore a much healthier funded ratio — than the same pension discounting at 4%. **The choice of discount rate moves the reported funded ratio by tens of percentage points without anyone touching a single asset or changing a single promise.** This is not an academic nicety; it is one of the most consequential accounting choices in finance.

Two philosophies fight over it. The **market-consistent** view (used by insurers and by corporate pension accounting in much of the world) says: discount the liabilities at the yield of high-quality bonds, because that is what it would actually *cost* to defease the promise by buying matching bonds today. Under this view the liability is honest and rate-sensitive, and immunization is the natural response — match the assets to a liability that moves with market yields. The **expected-return** view (long used by many US public pension plans) says: discount the liabilities at the *return you expect to earn on your assets*, often 7% or more. This makes liabilities look smaller and lets the plan justify holding more stocks. Critics argue it hides risk: it lets a plan call itself fully funded on the strength of *hoped-for* stock returns, and it understates the true, market-value cost of the promises.

#### Worked example: how the discount rate swings the funded ratio

Our pension owes \$10,000,000 in seven years and holds \$7.11M of assets — 100% funded *at a 5% discount rate.* Now hold the assets and the promise fixed, and only change the assumed discount rate.

- At **4%**: liability PV $= \$10{,}000{,}000 / (1.04)^7 = \$7{,}599{,}178$. Funded ratio $= \$7.11\text{M} / \$7.60\text{M} \approx 94\%$ — a deficit appears.
- At **5%**: liability PV $= \$7{,}106{,}813$. Funded ratio $= 100\%$.
- At **7%**: liability PV $= \$10{,}000{,}000 / (1.07)^7 = \$6{,}227{,}497$. Funded ratio $= \$7.11\text{M} / \$6.23\text{M} \approx 114\%$ — a comfortable surplus.

The *same* pension, with the *same* assets and the *same* promise, is reported as 94% funded, 100% funded, or 114% funded depending purely on a number chosen by an actuary. *The intuition: before you can immunize a liability you have to agree on what it is worth, and the discount rate that fixes its value is a choice with enormous power to flatter or expose a plan.*

This is also why immunization and the market-consistent view go hand in hand. If you discount your liabilities at market bond yields, then your liability moves exactly like a bond, and a matching bond portfolio hedges it perfectly. If you discount at an expected equity return, your "liability" doesn't move with rates in any clean way, and no bond portfolio can immunize it — you have priced yourself into a position where hedging is impossible by construction. How the underlying market rates that anchor all of this are set is the subject of [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).

## Cash-flow matching (dedication): the brute-force alternative

Immunization matches *durations* and accepts that the assets and the liability are not literally the same cash flows — it relies on the durations staying matched as rates move. There is a more direct, more expensive cousin: **cash-flow matching**, also called **dedication.** Instead of matching the average sensitivity, you buy bonds whose actual coupons and principal land *exactly* on the dates you owe money.

If you owe \$1,000,000 on January 1 of each of the next ten years, cash-flow matching means buying a ladder of bonds engineered so that each year's incoming coupons and maturing principal sum to \$1,000,000 right when you need it. The cash is *already there*, scheduled, on the day it is due. You never have to sell a bond into the market at an uncertain price, and you barely care what rates do in between, because you are holding everything to maturity.

The advantage is precision and peace of mind: a cash-flow-matched portfolio has almost no **reinvestment risk** (you are not relying on reinvesting coupons at unknown future rates) and almost no need to rebalance. The disadvantage is cost and rigidity: you need to find bonds maturing on exactly the right dates in exactly the right amounts, which is often impossible, so you over-buy and leave yield on the table. For a *single, large, fixed* payout — a lottery jackpot annuity, a legal settlement, the wind-down of a closed pension — dedication is often worth the cost. For a *big, fuzzy, decades-long* stream of liabilities, duration matching is cheaper and far more practical.

![A comparison matrix of cash flow matching versus duration matching across how it works, rebalancing needed, reinvestment risk, cost and flexibility, and what each is best for, showing dedication as precise but expensive and immunization as flexible but needing rebalancing](/imgs/blogs/immunization-and-duration-matching-how-pensions-and-insurers-hedge-4.png)

#### Worked example: dedication versus immunization on a single payment

You owe exactly \$10,000,000 in seven years. **Dedication:** buy \$10,000,000 face value of a 7-year zero-coupon Treasury STRIP (a stripped Treasury that pays one lump sum at maturity). At a 5% yield it costs \$7,106,813 today, and in seven years it pays exactly \$10,000,000 — the precise amount, on the precise day. You are done; you can put it in a drawer and ignore every rate move for seven years. **Immunization:** buy the \$7.11M barbell with duration 7.0. It also covers the liability, costs the same today, and may yield slightly more — but you must watch it and rebalance as the durations drift apart.

*The intuition: when the liability is a single known payment, a matching zero-coupon bond is the perfect, maintenance-free hedge; immunization is what you reach for when the liabilities are too numerous or too irregular to dedicate exactly.*

## Why the match drifts, and the discipline of rebalancing

Here is the catch that separates immunization on paper from immunization in practice: **a matched book does not stay matched.** Two forces pull the durations apart.

**Time passes — but not at the same speed for both sides.** As a year goes by, the liability's duration falls by about a full year (a 7-year promise becomes a 6-year promise). But a *coupon-paying* bond portfolio's duration falls by *less* than a full year, because the coupons keep the average payment date from sliding as fast. So even with rates frozen, the asset duration and liability duration drift apart over time, opening a gap.

**Rates move — and convexity re-weights the portfolio.** When rates change, the present values of the individual bonds change by different amounts (that is convexity at work), which shifts the value-weighted average duration of the portfolio. The barbell you carefully set to duration 7.0 might be 6.7 after a 1% rate move, while the liability is now 6.4 — a fresh gap.

So immunization is not a trade you do once and forget. It is a **process.** Periodically — monthly, quarterly, or whenever the gap exceeds a tolerance — the manager **rebalances**: buys or sells bonds to re-pin the asset duration to the new liability duration. If the gap has gone positive (assets too long), the manager sells some long bonds and buys shorter ones. If it has gone negative, the reverse. This rebalancing is the daily work of an LDI desk, and it is where the transaction costs and the skill live.

There is a genuine tradeoff hiding in the word "periodically," and it is the kind of thing that separates a careful manager from a careless one. Rebalance *too often* and you bleed money on transaction costs — every trade crosses the **bid-ask spread** (the gap between the price you can buy at and the price you can sell at), and for less-liquid bonds that spread is real money paid away on every round trip. Rebalance *too rarely* and you let the gap drift wide enough that an interim rate move can do real damage before you catch it. The practical answer is a **tolerance band**: the manager picks a gap they can live with — say, plus or minus 0.5 years of duration — and only trades when the gap breaches it, rather than chasing perfection every day. This is the same logic a thermostat uses: don't fire the furnace for every fractional degree, but don't let the room get genuinely cold either. The width of that band is a real decision with real money on both sides, and it is one of the levers an LDI desk tunes for each client's risk appetite and the liquidity of the bonds involved.

![A timeline showing an immunized book at year zero with a zero duration gap, the gap drifting positive as time passes and as rates move, a rebalancing trade that sells long bonds and buys short bonds, and the duration gap restored to zero, repeating each quarter](/imgs/blogs/immunization-and-duration-matching-how-pensions-and-insurers-hedge-5.png)

#### Worked example: the gap drifting and the rebalancing trade

At setup, the pension's assets and liability both have duration 7.0 — gap zero. A year later, the liability has shortened to about **6.0 years** (a 7-year promise is now a 6-year promise). The coupon-paying asset barbell, meanwhile, has only shortened to about **6.4 years** — its coupons slowed the slide. The gap is now $6.4 - 6.0 = +0.4$ years. Then rates jump +100 bps and convexity re-weights the barbell up to **6.7 years** while the liability sits at **6.0** — the gap widens to **+0.7 years.** The pension is now mildly betting that rates fall. To restore the match, the manager **sells some of the 12-year bonds and buys 3-year notes**, dragging the asset duration back down to 6.0. Gap zero again — until next quarter.

*The intuition: immunization is a treadmill, not a destination; the match decays with every passing day and every rate move, so the hedge has to be tended, not set and forgotten.*

## The limits of duration matching: what a matched book still cannot hedge

Duration matching is powerful but it is not omnipotent, and a serious practitioner knows exactly where it stops working. Three gaps matter.

**Non-parallel curve shifts.** Plain duration assumes the *whole* yield curve moves up or down together by the same amount — a "parallel shift." But the curve also *twists*: short rates can rise while long rates fall (a flattening), or the reverse (a steepening). A barbell immunizer is especially exposed here. Recall our barbell: \$4.0M in 3-year notes and \$3.1M in 12-year bonds, matched to a 7-year liability. If the curve *steepens* — short rates fall, long rates rise — the 12-year leg loses value while the 3-year leg gains, and the *liability*, anchored at 7 years, sits in between. The portfolio duration says "matched," but the actual dollar outcome no longer cancels, because the assets are exposed to *parts* of the curve where the liability is not. The fix is **key-rate durations**: instead of one duration number, measure sensitivity to each segment of the curve (2-year, 5-year, 10-year, 30-year) separately, and match the liability's exposure at each key point. A true LDI program matches key-rate durations, not just the single aggregate number.

**Convexity under big moves.** Duration is a straight-line approximation; for a large rate move the curvature (convexity) matters, and if the asset convexity does not at least equal the liability convexity, the match leaks on big shocks. We built the barbell partly to *over*-supply convexity for exactly this reason, but a single-bond bullet immunizer can be caught out by a violent move.

**Inflation-linked liabilities.** A pension that has promised *inflation-indexed* benefits owes a real, not a nominal, amount — the dollar figure grows with the cost of living. A portfolio of *nominal* bonds, however well duration-matched, does not hedge that: if inflation surprises higher, the nominal bonds are fixed but the liability swells. The correct hedge is **inflation-linked bonds** (TIPS in the US), whose payments rise with inflation, matched in *real* duration to the real liability. Matching nominal duration to a real liability is a classic, expensive mistake.

#### Worked example: a curve twist defeats a perfectly matched aggregate duration

Take the barbell again: \$4.0M in 3-year notes (duration 3) and \$3.1M in 12-year bonds (duration 12), aggregate duration 6.9, matched to a \$7.11M liability of duration 7.0. Now the curve *steepens by 1% at the long end and falls 1% at the short end* — not a parallel move. The 12-year leg falls by about $12 \times 1\% = 12\%$, a loss of roughly \$0.37M. The 3-year leg *rises* by about $3 \times 1\% = 3\%$, a gain of roughly \$0.12M. Net asset change: about **−\$0.25M.** But the 7-year liability, discounted off the *middle* of the curve where rates barely moved, is roughly unchanged. The funded ratio just fell from 100% to about $\$6.86\text{M} / \$7.11\text{M} \approx 96\%$ — a 4-point hit that the aggregate-duration match swore could not happen.

*The intuition: matching one duration number protects you only against the curve moving in one piece; the moment the curve bends or twists, you need key-rate durations to stay truly hedged.*

## Liability-driven investing: immunization for a whole pension

Scale the single-promise example up to an entire pension fund with tens of thousands of retirees and a liability stream stretching out fifty years, and immunization gets a grander name: **liability-driven investing**, or **LDI.** The philosophy is identical — *manage the assets relative to the liabilities, not relative to a stock-market benchmark* — but the machinery is bigger.

An LDI program starts by modeling the entire liability cash-flow profile: how much the fund expects to pay in each future year, discounted to a present value and a duration (often a very long one, 12 to 20 years, because of all those distant payments). Then it builds an asset portfolio — usually long-dated government bonds plus interest-rate **derivatives** (swaps and futures that let you add duration without tying up cash) — whose duration matches the liabilities. The point of the derivatives is **capital efficiency**: a pension that also wants to hold stocks for growth can use a small amount of cash in interest-rate swaps to get the duration it needs to hedge the liabilities, freeing the rest of its capital to chase returns. This is the "have your cake and eat it" promise of LDI: hedge the rate risk *and* keep the growth assets.

To see why a swap is the workhorse of LDI, it helps to know what one *is*, in plain terms. An **interest-rate swap** is a contract where one side agrees to pay a fixed rate and receive a floating rate (or vice versa) on an agreed notional amount, with no principal changing hands up front. A pension that "receives fixed" on a long-dated swap has, in effect, bought the rate sensitivity of a long bond *without paying for the bond* — it has added duration using almost no cash, just a margin deposit. That is the magic and the menace in one sentence: the fund can hedge \$1 billion of liability duration while keeping nearly all \$1 billion of cash invested in return-seeking assets like stocks. The duration is rented, not bought.

#### Worked example: hedging duration with a swap instead of a bond

A pension needs to add 7 years of duration to \$10 million of liability exposure. **Option one — buy bonds:** spend \$10 million on 7-year bonds. The duration is hedged, but all \$10 million is now tied up in low-returning bonds. **Option two — a receive-fixed swap:** enter a 7-year swap on \$10 million notional, posting perhaps \$0.5 million of collateral. The swap delivers the same ~7 years of rate sensitivity on the \$10 million, while \$9.5 million of cash stays free to invest in equities for growth. On a +1% rate move, the swap loses about $7 \times 1\% = 7\%$ of \$10M, or \$0.7M — exactly offsetting the rise in the funded position — but that \$0.7M must be posted *as cash collateral, immediately.* *The intuition: a swap lets a pension hold both the hedge and the growth assets at once, but it converts a quiet mark-to-market move into an urgent, real-cash collateral call.*

That capital efficiency is also LDI's hidden fault line. Because the duration is achieved with **leverage** (derivatives controlling far more notional than the cash backing them), a sharp move in the *wrong* direction can trigger margin calls — demands to post more cash *right now* — forcing the fund to sell its growth assets at the worst possible moment. That is precisely the trap that sprang on UK pension funds in September 2022, which we will come to in the real-markets section. The lesson is not that LDI is bad — it is the correct framework — but that leverage turns a smooth hedge into a fragile one when liquidity disappears. The macro mechanics of how policy and rates drive these episodes are covered in [how monetary policy moves bonds](/blog/trading/macro-trading/how-monetary-policy-moves-bonds-duration-convexity).

## Silicon Valley Bank, 2023: the textbook unmanaged gap

Everything in this post comes together in one spectacular failure. Silicon Valley Bank (SVB) was, in March 2023, the most vivid illustration imaginable of an unmanaged duration gap — and the fact that it happened to professionals who *had every tool described here* is the most important lesson of all.

![A before and after comparison showing Silicon Valley Bank in 2022 holding about one hundred twenty billion dollars of long dated bonds with a duration of five to six years funded by on demand deposits with a duration near zero, then in March 2023 a rate rise creating an unrealized loss bigger than equity, a deposit run of forty two billion in one day, and the bank seized on March tenth](/imgs/blogs/immunization-and-duration-matching-how-pensions-and-insurers-hedge-6.png)

Here is the setup. During 2020 and 2021, SVB's tech-startup clients flooded it with deposits — the bank's deposits roughly tripled. SVB had to do *something* with that cash, so it bought bonds: roughly \$120 billion of them, heavily weighted toward **long-dated, fixed-rate** US Treasuries and agency mortgage-backed securities, much of it parked in a "held-to-maturity" accounting bucket. The asset side of the balance sheet therefore had a duration of roughly **five to six years.**

Now the liability side. SVB's funding was overwhelmingly **deposits** — money its clients could withdraw on demand, any day, with a few keystrokes. The duration of on-demand deposits is essentially **zero**: they reprice (or leave) instantly. So SVB was running an enormous **positive duration gap**: long assets, zero-duration funding. In the language of this post, it was making a massive, unhedged, unconscious bet that rates would *not* rise.

Then, in 2022 and into 2023, the Federal Reserve raised rates by roughly **4 to 5 percentage points** — the fastest hiking cycle in forty years. With a five-to-six-year asset duration, a ~4% rate move implies a bond-price fall on the order of 20%+. SVB's bond portfolio developed an **unrealized loss estimated above \$15 billion** — larger than the bank's entire common equity. On paper, the bank was already insolvent; it was only the "held-to-maturity" accounting that let the loss stay hidden, on the fiction that the bonds would be held to maturity and never sold.

The fiction broke when the *liability* side moved. Depositors — a concentrated, sophisticated, herd-prone set of venture-backed startups — got nervous, and on **March 9, 2023, attempted to withdraw about \$42 billion in a single day.** To meet the run, SVB had to *sell* the long bonds, which crystallized the loss that accounting had been hiding. The hole was real, the capital was gone, and regulators seized the bank on **March 10, 2023.** A textbook positive duration gap, left unmanaged, met a rising-rate cycle and a deposit run, and the bank died in 48 hours. We trace the broader 2023 banking stress, including Credit Suisse, in [the SVB and Credit Suisse bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).

#### Worked example: how a 4% rate move ate SVB's capital

Approximate SVB with round numbers: **\$200 billion of assets** at an average duration of **5.5 years**, funded by **\$185 billion of deposits** at a duration near **0**, leaving **\$15 billion of equity.** Rates rise **+4% (400 bps).** The asset value falls by roughly $5.5 \times 4\% = 22\%$ of \$200B — about **−\$44 billion.** The deposits barely move in present value. The entire loss lands on the \$15 billion equity cushion, which is overwhelmed three times over. The bank is deeply insolvent on a mark-to-market basis; all that remains is for someone to notice and demand their money back.

*The intuition: SVB's failure was not a credit problem — the bonds were US Treasuries and agency MBS that will pay every penny at maturity — it was a pure duration-gap problem, the most basic ALM mistake there is, magnified by leverage and a fast run.*

The deepest lesson of SVB is that the mismatch was not exotic and not unknowable. Any first-year ALM analyst, handed the balance sheet, could have computed the duration gap and the loss-on-a-rate-shock in an afternoon. The tools in this post are not advanced finance — they are the *table stakes* of running a bank, an insurer, or a pension. SVB's failure was a failure to *use* them, or to act on what they said. That is what makes it the perfect closing case for the duration story.

## Common misconceptions

**"If my bonds are US Treasuries, they're safe."** Safe from *default*, yes — the US government will pay every coupon. But not safe from *duration*. SVB held Treasuries and agency MBS and still failed, because a Treasury's price falls just as hard as any bond's when rates rise. Credit risk and interest-rate risk are two completely different dangers; a bond can be perfectly safe on one and lethal on the other. "Risk-free" means free of *default* risk, not free of *price* risk.

**"A pension with enough assets to cover its liabilities is fine."** Only if the durations match. A pension can be 100% funded today and 90% funded after a rate move that didn't touch the dollar amount of a single promise — purely because the present value of its long liabilities rose faster than its shorter assets. Funded *level* without duration *matching* is a snapshot that the next rate move can erase.

**"Falling rates are good for bond investors, so they're good for pensions."** For a pension, falling rates are usually *bad*. A pension's liabilities are typically longer-duration than its assets, so a rate fall inflates what it owes faster than what it owns. Many corporate pension deficits in the 2010s came not from market crashes but from rates grinding lower and lower, ballooning the present value of decades of future benefits.

**"Immunization eliminates all risk."** It eliminates *first-order interest-rate risk* — the risk from small, parallel rate moves. It does not eliminate convexity risk (large moves), reinvestment risk (coupons reinvested at unknown rates), credit risk (the bonds defaulting), liquidity risk (being forced to sell into a thin market), or the risk that the liabilities themselves change (people live longer, inflation surprises). Immunization is a powerful, specific tool, not a magic shield.

**"Cash-flow matching is always better because it's exact."** It is more precise, but it is also more expensive and often impossible — you rarely find bonds maturing on exactly the right dates in exactly the right amounts. The "exactness" can also lock you into lower-yielding bonds. For a single fixed payout, dedication shines; for a large, fuzzy, evolving liability stream, duration matching is cheaper and more practical, and the small imprecision is worth the flexibility.

**"The duration gap only matters for banks."** Every institution that owns assets to fund liabilities has a duration gap, whether it measures it or not. Pensions, life insurers, endowments with spending commitments, even a household saving for a known future expense — all face the same arithmetic. Banks are just the most leveraged and therefore the most spectacular when the gap goes wrong.

## How it shows up in real markets

**Life insurers and the long-duration scramble.** A life insurer selling annuities and whole-life policies owes money decades into the future — its liabilities can have durations of 10 to 20 years. To immunize, insurers are among the only buyers willing to hold 30-year bonds and very long corporate debt in size, which is part of why a deep long-end of the bond market exists at all. When long rates fall, insurers' liabilities balloon, and they scramble for any long-duration asset they can find — a structural demand that helps explain why very long yields can stay stubbornly low even when short rates are high. Their hedging need quietly shapes the whole long end of the curve.

**UK pensions and the 2022 LDI crisis.** In September 2022, the UK government announced a large unfunded tax-cut package, and gilt (UK government bond) yields spiked violently — the 30-year gilt yield rose more than a full percentage point in days. UK defined-benefit pensions, almost all running leveraged LDI programs, had used interest-rate derivatives to hedge their long liabilities. The sudden yield spike triggered **collateral calls** — demands to post cash against the derivatives — and to raise that cash, the funds sold gilts, pushing yields *higher*, triggering *more* collateral calls: a self-reinforcing doom loop. The Bank of England had to step in with emergency gilt purchases to stop it. The irony is sharp: the funds were *hedged* against rates in the long run, but the *leverage* in the hedge made them fragile to a fast move in the short run. LDI was right; leveraged LDI without a liquidity buffer was dangerous.

**Silicon Valley Bank, March 2023.** Covered in detail above — the cleanest modern example of an unmanaged positive duration gap. Long fixed-rate bonds funded by on-demand deposits, a 4–5% rate hiking cycle, an unrealized loss bigger than equity, and a deposit run that forced the losses into the open. The whole failure is a duration-gap story from start to finish, and it triggered a wider re-examination of how every regional bank manages its bond book.

**Defined-benefit pension de-risking.** Over the past two decades, thousands of corporate pension plans — having been burned by the funded-ratio swings of the 2000s — have shifted from chasing stock returns to LDI, deliberately matching the duration of their bond portfolios to their liabilities and even buying long-dated bonds and derivatives specifically to close the duration gap. Some go all the way to a "buyout," paying an insurer to take the liabilities entirely. This decades-long migration toward liability-matching is one of the largest sustained sources of demand for long-dated bonds in the world, and it is immunization applied at industrial scale.

**Banks after SVB: the great rate-hedge audit.** In the months after SVB, regulators and bank treasurers everywhere re-examined their duration gaps and the size of their unrealized bond losses. Many banks moved to hedge more of their fixed-rate assets with interest-rate swaps (which convert fixed-rate exposure to floating, shrinking the gap), held more cash, and watched deposit "stickiness" far more carefully. The episode was a violent, expensive reminder that ALM is not optional paperwork — it is the thing that keeps a bank alive when rates move.

## When this matters to you, and where to go next

If you have a pension, a life-insurance policy, or money in a bank, the duration gap of an institution you have never thought about is quietly determining whether the promise made to you will be kept when a rate shock hits. Immunization is the invisible engineering behind "your money is safe" — and SVB is what happens when the engineering is skipped. Understanding it changes how you read every "the bank is fine / the pension is fully funded" headline: the real question is never *how much* they own, it is whether what they own is matched to what they owe.

To go deeper from here:

- The mechanics underneath this post — [duration](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income), [modified duration and DV01](/blog/trading/fixed-income/modified-duration-and-dv01-measuring-and-trading-rate-risk), and [convexity](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story) — are the tools this whole discipline is built on.
- For the allocator's view of bonds as duration ballast in a portfolio, see [government bonds: the risk-free anchor and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration) and [the stock-bond correlation engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine).
- For the heavy quantitative machinery — the term-structure models that price these hedges precisely — see [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics).
- For how the rate moves that drive all of this get set in the first place, see [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).

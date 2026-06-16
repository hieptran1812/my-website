---
title: "Anatomy of a bond: par, coupon, maturity, and the issuer"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A from-zero walkthrough of the bond contract — face value, coupon, maturity, and the issuer — using one running example bond to show how price, par, premium, and discount all connect."
tags: ["fixed-income", "bonds", "par-value", "coupon", "maturity", "issuer", "indenture", "zero-coupon", "floating-rate", "bond-pricing"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — A bond is a written promise to repay borrowed money on a fixed schedule, and the whole contract reduces to four numbers: who borrows (the issuer), how much they repay (par), what they pay you along the way (the coupon), and when it ends (maturity). Learn those four and you can read any bond on earth.
>
> - **Par (face value)** is the amount the issuer repays at the end — almost always \$1,000 per bond. It is the contract's anchor, not the price you pay.
> - **The coupon** is the periodic interest, quoted as a rate on par. A 4% coupon on a \$1,000 bond pays \$40 a year — in the US, split into two \$20 payments six months apart.
> - **Maturity** is the date the loan ends and par comes back. A bond's whole risk character — how violently its price moves — flows from how far away that date is.
> - **The price you pay is a separate thing from par.** When the market's prevailing rate equals the coupon, the bond trades at par. When the rate is lower, the fixed coupon looks generous and the bond trades at a *premium* (above \$1,000). When the rate is higher, it trades at a *discount* (below \$1,000). That single relationship is the seed of all of fixed income.
> - **The issuer's identity is the credit risk.** The same four numbers from the US Treasury and from a shaky company are not the same promise — one is treated as risk-free, the other can default.

Suppose a friend asks to borrow \$1,000 from you. You agree, but you write down terms: she'll pay you \$40 of interest each year, and she'll give you the original \$1,000 back in exactly five years. You shake hands, and now you own something. Not the cash — you gave that away — but a *claim* on a stream of future payments. If a third person offered to buy that claim from you next week, what is it worth?

That handshake is a bond. Strip away the trading desks, the Bloomberg terminals, and the jargon, and a bond is nothing more than a loan written down with enough precision that the claim can be bought and sold. The reason bonds matter — the reason the global bond market is roughly \$140 trillion, larger than every stock market on earth combined — is that this little contract is how governments, companies, and cities borrow, and the price the market sets on it *is the price of money itself*. When you understand the four parts of one bond, you have the key that unlocks mortgages, the dollar, the stock market's valuation, and the entire machinery of interest rates.

![An annotated bond with the issuer, par value, coupon, and maturity all labeled on a five-year cash-flow timeline](/imgs/blogs/anatomy-of-a-bond-par-coupon-maturity-issuer-1.png)

The diagram above is the mental model we'll build the whole post around. There is an **issuer** who borrows, a **bondholder** (you) who lends, a **par value** of \$1,000 that comes back at the end, a **coupon** that drips in along the way, and a **maturity** date when the whole thing wraps up. Everything else in fixed income — yield, duration, credit spreads, the yield curve — is a refinement of these four ideas. Let's earn each one from scratch.

## Foundations: the four words that define every bond

Before we go deep, let's nail down the vocabulary. Every term here will reappear hundreds of times across the bond market, so it's worth defining each one carefully and concretely. We'll use one running example all the way through the post so the numbers compound instead of resetting.

**Our running bond.** Throughout this post we'll follow a single security: a **5-year, \$1,000 par note with a 4% annual coupon**, issued by a fictional company we'll call **Northwind Corp**. Where it helps, we'll put it next to a real **US Treasury note** — the benchmark "risk-free" bond against which every other bond on earth is measured. Keep these two in your head: Northwind (a company that *can* default) and a Treasury (a borrower the market treats as certain to pay).

### The issuer — who is borrowing

The **issuer** is the entity that borrows the money and owes you the payments. When you buy a bond, you are lending to the issuer. The issuer could be a national government (the US Treasury, the German *Bund*, the Japanese government), a company (Apple, Northwind Corp), a city or state (a *municipal* issuer), or a government agency. The issuer's identity is not a footnote — it *is* the bond's credit risk, the chance you don't get paid back. A 4% coupon from the US Treasury and a 4% coupon from a struggling company are wildly different promises, and the market prices them differently. We have a whole post on [who issues bonds and why](/blog/trading/fixed-income/who-issues-bonds-and-why-governments-companies-and-cities); for now, just hold the idea that *who* is on the other side of the loan is half the story.

### Par value (face value) — the amount repaid at the end

The **par value**, also called the **face value** or **principal**, is the amount the issuer promises to repay when the bond matures. By overwhelming convention, par is **\$1,000 per bond** in the US corporate and Treasury markets (Treasuries are technically sold in \$100 increments, but the math is identical — we'll use \$1,000 throughout). Par is the *anchor* of the contract: the coupon is calculated as a percentage of par, and par is the lump sum that comes back at the end.

Here is the single most common beginner confusion, so let's kill it immediately: **par is not the price.** Par is a fixed contractual number — \$1,000, written into the indenture, never changing. The *price* is what the bond trades for in the market today, and it bounces around: \$1,000, \$1,094, \$877, whatever buyers and sellers agree on. They happen to be equal at the moment of issuance for many bonds, which is why people conflate them, but they are different things. We'll spend a whole section on why the price drifts away from par.

### The coupon — the interest paid along the way

The **coupon** is the periodic interest payment. It's quoted as an annual rate on par — our Northwind bond has a "4% coupon," which means it pays 4% of \$1,000 = **\$40 per year**. The word "coupon" is a historical artifact: old paper bonds had little detachable tickets ("coupons") around the edge, and you'd literally clip one off and mail it in to collect each interest payment. The paper is gone; the name stuck.

Two details matter enormously and trip up beginners:

1. **Frequency.** In the US, bonds pay coupons **semiannually** — twice a year — by default. So our 4% bond pays its \$40 as two \$20 payments, six months apart. (European bonds often pay annually; this convention difference matters when you compare yields across markets.)
2. **The coupon rate is fixed at issuance and (usually) never changes.** This is the heart of why bond prices move. The issuer locked in "\$40 a year" the day it sold the bond. If interest rates in the world change afterward, the bond's payment does *not* adjust — and that frozen payment is exactly what makes the bond's price rise or fall. (The exception is a *floating-rate* bond, which we'll meet shortly.)

A *basis point* — a unit you'll see everywhere in this series — is one hundredth of a percent, 0.01%. So a coupon that rises from 4.00% to 4.25% has risen 25 basis points, usually written "25 bps." Bond people quote everything in basis points because the moves that matter are small.

### Maturity — when the loan ends

The **maturity** is the date the bond's life ends: the issuer repays par, makes the final coupon, and the contract is extinguished. Our Northwind bond matures in 5 years. **Term to maturity** (or just "term") is the time *remaining* until that date — it shrinks every day. A bond issued with a 5-year term has a 5-year term to maturity on day one, a 3-year term to maturity two years later, and so on. People classify bonds by term: roughly, *bills* mature in under a year, *notes* in 2–10 years, and *bonds* in more than 10 years (the names are loose, but that's the US Treasury convention).

Maturity is not just a calendar fact. It's the single biggest driver of how *risky* a bond's price is. A bond maturing tomorrow is almost as safe as cash — you'll get your \$1,000 back in a day, so its price can't wander far. A bond maturing in 30 years is a long, exposed bet: a lot can happen to interest rates over three decades, and its price swings violently. That sensitivity has a name — *duration* — and it's [the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income). For now, just internalize: **longer maturity means a wilder price.**

### Putting the four together: a bond is a cash-flow schedule

With those four words, a bond becomes completely legible. It is a *schedule of cash flows*: you pay the price today (an outflow), then receive a fixed series of coupons (inflows), and finally receive par plus the last coupon at maturity (a big inflow). That's it. The entire discipline of fixed income is the study of that schedule — what it's worth, how its value moves, and how likely you are to actually receive it.

![A cash-flow timeline of the five-year four-percent bond showing ten coupon payments and the return of principal at maturity](/imgs/blogs/anatomy-of-a-bond-par-coupon-maturity-issuer-2.png)

#### Worked example: the cash flows of our running bond

Let's lay out every dollar our Northwind bond pays, so the abstraction becomes concrete.

- **Today (t = 0):** you buy the bond. Assume for now it's issued at par, so you pay **−\$1,000**.
- **Every six months for five years:** you collect a coupon. The coupon is 4% of \$1,000 = \$40 a year, paid semiannually, so **+\$20** each time. Over five years that's 10 payments of \$20 = **+\$200** in coupons.
- **At maturity (year 5):** along with the final \$20 coupon, the issuer returns your **+\$1,000** principal.

Add it up: you paid \$1,000 and received \$200 in coupons plus \$1,000 back = \$1,200 total, for a \$200 profit over five years. That profit, expressed as an annual rate, is your *yield* — and because we bought at par, the yield equals the coupon rate of 4%. (Yields get subtler the moment the price isn't par; that's [the seesaw at the heart of bonds](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds) and [the many kinds of yield](/blog/trading/fixed-income/the-many-yields-current-yield-ytm-and-yield-to-call), each its own post.)

*A bond bought at par simply pays you its coupon rate as your return — no more, no less; everything interesting happens when the price differs from par.*

## Par value: the anchor that never moves

Let's go deeper on par, because almost every beginner mistake about bonds traces back to confusing par with price.

Par is the **denomination** of the loan — the unit the contract is written in. The issuer borrowed in \$1,000 chunks and will repay in \$1,000 chunks. Three things are pinned to par and never change over the bond's life:

1. **The redemption amount.** At maturity you get par back, always \$1,000 per bond, regardless of what you paid or what the bond traded for in between.
2. **The coupon calculation.** The coupon rate is applied to par, not to the market price. A 4% coupon always pays \$40 a year even if the bond's price collapses to \$700 — the \$40 is computed on the \$1,000 face, not on what you paid.
3. **The legal claim in a default.** If the issuer defaults, your claim is for the par amount (plus accrued coupon), not the market price. This becomes very important in the section on the indenture.

What is *not* pinned to par is the **price**. The price is set by the market every second of every trading day, and it answers a different question: *"Given today's interest rates and Northwind's creditworthiness, what is this exact stream of \$20 coupons and \$1,000 redemption worth right now?"* That present-value calculation — [discounting the cash flows](/blog/trading/fixed-income/discounting-cash-flows-how-a-bond-is-priced) — is the subject of the next post in this series. The key idea to carry forward is that price and par are two different numbers that only coincide under one specific condition (when the market rate equals the coupon rate).

Bond prices are quoted as a **percentage of par**, which is why you'll hear a trader say a bond is "trading at 97.4" or "at 105." That means 97.4% of par (= \$974) or 105% of par (= \$1,050). A price of exactly **100** means par. This convention is universal and worth memorizing: *the price is a percentage, par is the 100% baseline.*

#### Worked example: par versus price in a default

Imagine Northwind hits trouble and its 5-year bond's market price falls to **\$600** (quoted as "60"). You bought at \$1,000. Now Northwind formally defaults, and after a restructuring, bondholders recover **40 cents on the dollar** — meaning the legal claim is settled at 40% of *par*, not of price.

- Your recovery is 40% of par: 0.40 × \$1,000 = **\$400**.
- Notice the recovery is computed on the \$1,000 face value — *par* — even though the bond was trading at \$600 right before default.
- If you had panic-bought the distressed bond at \$600 and then recovered \$400, you'd have lost \$200. If you'd bought it even cheaper, at \$300, and recovered \$400, you'd have *made* \$100 — which is exactly the bet distressed-debt investors make.

*Par is the legal anchor for both your coupon and your recovery; the price is just the market's running opinion about whether you'll actually collect.*

## The coupon: fixed, floating, or zero

The coupon clause is where bonds get their variety. The default — and our running bond — is a **fixed-rate** coupon: a set rate, locked at issuance, paid until maturity. But there are two other major designs, and the contrast between them teaches you something deep about interest-rate risk.

![A comparison matrix of fixed-rate, floating-rate, and zero-coupon bonds across what the coupon pays, price stability, return source, who wants it, and the main risk](/imgs/blogs/anatomy-of-a-bond-par-coupon-maturity-issuer-3.png)

**Fixed-rate bonds** pay the same coupon every period for life. Our Northwind 4% bond is fixed-rate: \$20 every six months, no matter what happens to interest rates. Simplicity is the appeal — you know exactly what you'll receive and when. The cost is that the fixed payment becomes a liability when rates move: if new bonds start paying 6%, your locked-in 4% looks stingy and your bond's price falls so its *effective* yield catches up. Fixed-rate bonds carry the full force of **interest-rate risk**.

**Floating-rate bonds** (FRNs, "floaters") have a coupon that *resets* every period to track a reference interest rate plus a fixed spread. A typical floater might pay "**SOFR + 1.00%**" — where SOFR (the Secured Overnight Financing Rate) is a benchmark short-term US interest rate that the market resets daily. When SOFR is 3%, the coupon is 4%; when SOFR jumps to 5%, the coupon climbs to 6% at the next reset. Because the coupon chases the market, a floater's price barely moves — it stays near par. The trade-off: your income is unpredictable, and it *falls* when rates fall. Banks love floaters because their own funding costs float too, so the floating coupon hedges them.

**Zero-coupon bonds** pay no coupon at all. Instead, you buy them at a deep **discount** to par and collect the full par at maturity — the entire return is baked into the gap between purchase price and \$1,000. A 5-year zero priced to yield 4% costs about **\$820** today and pays \$1,000 in five years; the \$180 of "interest" is the discount, earned by waiting. Zeros are the purest, most rate-sensitive bonds: with no coupons to cushion you, *all* of your money is locked up until maturity, so their prices swing the most for a given change in rates. US Treasury "STRIPS" are real zero-coupon Treasuries; savings-bond-style instruments and some pension liabilities are matched with zeros precisely because a known lump sum arrives on a known date.

#### Worked example: pricing a zero-coupon bond

Let's price a 5-year zero-coupon bond that the market wants to yield 4% per year. The price is just par discounted back five years:

$$ P = \frac{1000}{(1 + 0.04)^5} = \frac{1000}{1.2167} \approx 821.93 $$

where $P$ is the price today, the numerator \$1,000 is par, 0.04 is the annual yield, and 5 is the years to maturity.

- You pay about **\$821.93** today.
- You receive **\$1,000** in exactly five years.
- Your gain is \$178.07, and because \$821.93 grown at 4% for five years (\$821.93 × 1.04⁵) lands back on \$1,000, your annualized return is exactly **4%** — the same yield as our coupon bond, achieved with zero coupons.
- Now suppose right after you buy, the market yield jumps to 5%. The zero's price drops to \$1,000 ÷ 1.05⁵ ≈ **\$783.53** — a fall of about **4.7%** from \$821.93. A 5-year *coupon* bond would fall less, because its near-term coupons aren't discounted as harshly.

*A zero-coupon bond is interest-rate risk in its purest form: with no coupons to soften the blow, the entire price rides on one discounting calculation.*

## Maturity, term, and why time is the master risk

We said maturity drives a bond's wildness. Let's make that precise with intuition before we ever touch the duration math (which lives in [its own post](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income)).

Think about *when* your money comes back. For our 5-year coupon bond, small amounts (\$20 coupons) come back soon, but the big chunk — the \$1,000 par — doesn't arrive for five years. For a 30-year bond, that \$1,000 is locked away for three decades. The further out a cash flow is, the more its present value reacts to a change in the discount rate, because you're compounding the discount over more periods. So a long bond's price is hypersensitive to rate moves, and a short bond's price is sluggish.

A clean way to feel this: a bond's price sensitivity is roughly proportional to its remaining term (for a zero, it's *exactly* the term). A 2-year bond might lose about 2% if yields rise 1 percentage point; a 10-year bond loses about 9%; a 30-year bond can lose over 20%. Same 1-point move, wildly different damage — and the only thing that changed is how far away maturity is.

#### Worked example: same rate move, three maturities

The market rate rises by exactly **1 percentage point** (100 bps). What happens to three 4%-coupon bonds of different maturity? Using the rule of thumb that a bond's percentage price change is approximately minus its duration times the yield change, and noting duration is a bit below maturity for a coupon bond:

| Bond | Approx. duration | Price change for +1% yield | New price (from \$1,000) |
|---|---|---|---|
| 2-year, 4% coupon | ~1.9 years | −1.9% | ~\$981 |
| 10-year, 4% coupon | ~8.1 years | −8.1% | ~\$919 |
| 30-year, 4% coupon | ~17.3 years | −17.3% | ~\$827 |

- The 2-year holder barely notices: a \$19 paper loss on \$1,000.
- The 30-year holder is down \$173 — nine times as much — from the *identical* 1-point rate move.
- Reverse the move (rates fall 1 point) and the 30-year bond *gains* the most. Long maturity is leverage on rates, in both directions.

*Maturity is the dial that sets how much interest-rate risk you're taking; everything about a bond's volatility starts with how far away its final payment sits.*

## The price–par relationship: par, premium, and discount

Now we connect the coupon to the wider world — the influence thread that runs through this entire series. **A bond's price is a tug-of-war between its fixed coupon and the market's current interest rate.** This single relationship is why the bond market is the price of money, and it's worth slowing down for.

Here's the intuition. Your Northwind bond pays a fixed 4%. Now imagine the world's interest rates change *after* you bought it:

- **If new bonds of similar risk now pay 6%,** your 4% suddenly looks miserly. Nobody will pay full price for a bond paying 4% when they could buy a fresh one paying 6%. So your bond's price must *fall* until its effective return (its yield) rises to match the 6% the market demands. Your bond now trades at a **discount** — below par.
- **If new bonds now pay only 2%,** your 4% looks generous. Buyers will compete for your bond and bid its price *up* until its effective yield falls to the 2% market rate. Your bond trades at a **premium** — above par.
- **If new bonds pay exactly 4%** — the same as your coupon — your bond is neither better nor worse than a fresh one, so it trades at exactly **par**, \$1,000.

That's the whole logic of premium and discount: *the price adjusts so that the bond's yield always equals what the market currently demands, no matter what coupon it happens to carry.* The coupon is frozen; the price does the moving.

![A chart showing how the prevailing market interest rate determines whether the bond trades at par, a premium, or a discount, with the curve crossing par exactly where the rate equals the four-percent coupon](/imgs/blogs/anatomy-of-a-bond-par-coupon-maturity-issuer-4.png)

The figure above is the influence picture for this post. The horizontal axis is the prevailing market interest rate; the vertical axis is what our 4%-coupon bond is worth. The curve slopes down — higher rates mean lower bond prices, the famous inverse relationship — and it crosses par (\$1,000) at exactly the point where the market rate equals the 4% coupon. To the left, where rates are below the coupon, the bond is a *premium*. To the right, where rates are above the coupon, it's a *discount*. This downward-sloping curve is the seed of [the price–yield seesaw](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds) that defines the entire asset class.

Why does this ripple out to the wider world? Because *the same logic prices everything that has future cash flows.* A stock is a claim on future earnings; a house is a claim on future rent or shelter. When the market rate (set in the bond market) rises, the present value of every future dollar falls — so bonds, stocks, and houses all reprice downward together. The bond market is where that master rate is discovered, which is why it's called [the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).

#### Worked example: par, premium, and discount on one bond

Take our Northwind 5-year bond with its fixed 4% coupon (paying \$40/year, treated annually here for clean arithmetic), and price it under three different market rates. The price is the present value of \$40 a year for five years plus \$1,000 at year five, discounted at the market rate $y$:

$$ P = \sum_{t=1}^{5} \frac{40}{(1+y)^t} + \frac{1000}{(1+y)^5} $$

where $P$ is price, \$40 is the annual coupon, \$1,000 is par, $y$ is the market yield, and $t$ runs over the five years.

- **Market rate = 4% (equal to the coupon):** $P \approx \$1{,}000$. The bond trades at **par**. The coupon and the market agree.
- **Market rate = 2% (below the coupon):** $P \approx \$1{,}094$. The bond trades at a **premium** of \$94. Buyers pay extra for the above-market 4% coupon.
- **Market rate = 6% (above the coupon):** $P \approx \$916$. The bond trades at a **discount** of \$84. Buyers demand a price cut to make up for the below-market coupon.

Notice the symmetry isn't perfect — the premium (\$94) is a bit larger than the discount (\$84) for the same 2-point move. That gentle asymmetry is **convexity**, a second-order effect that gets [its own post](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story). For now, the headline is the direction: rate down, price up; rate up, price down.

*A bond's coupon is fixed forever, so the only way the market can change the bond's yield is to change its price — premium when the coupon beats the market, discount when it lags.*

## The issuer is the credit: same coupon, different promise

We've been pricing our Northwind bond as if the only thing that matters is the level of interest rates. That's true for a US Treasury, because the market treats the US government as certain to pay. But for a *company* like Northwind, there's a second force on the price that we've been quietly ignoring: the chance the issuer simply doesn't pay. That chance is **credit risk**, and it's the reason the issuer's identity is not decoration but a core part of the contract.

Here's the cleanest way to see it. A corporate bond's yield can be split into two pieces:

$$ \text{corporate yield} = \text{Treasury yield} + \text{credit spread} $$

where the Treasury yield is the compensation for lending money at all (the pure time-value of money, set by the risk-free benchmark), and the **credit spread** is the *extra* yield investors demand for taking the risk that this particular issuer might default. The spread is the price of the issuer's reputation, quoted in basis points. A pristine company might pay 50 bps over Treasuries; a shaky one might pay 500 bps or more. The bigger the default risk, the bigger the spread, the lower the price for the same coupon. We unpack this split fully in [credit spreads, pricing the probability of default](/blog/trading/fixed-income/credit-spreads-pricing-the-probability-of-default).

This is also why two bonds with the *identical* coupon, par, and maturity can trade at completely different prices: if one is issued by the Treasury and the other by Northwind, the market discounts Northwind's cash flows at a higher rate (Treasury rate + spread), so Northwind's price is lower. The four numbers look the same on paper; the fifth fact — *who is promising* — pulls them apart.

There's a feedback loop worth naming. When a company's prospects deteriorate, the market demands a wider spread, which pushes its bond price *down* — and a falling bond price (a rising yield) makes it more expensive for the company to borrow again, which can deepen the trouble. This is the mechanism behind a "credit spiral," and it's how a bond market can discipline a borrower, or push a wobbling one over the edge. The same dynamic at the level of *whole countries* is the story of [sovereign debt and the bond vigilantes](/blog/trading/macro-trading/sovereign-debt-and-the-bond-vigilantes).

#### Worked example: a Treasury and a corporate, same coupon, different price

Two 5-year bonds, both with a 4% annual coupon and \$1,000 par. One is a US Treasury; one is Northwind Corp. The risk-free 5-year Treasury rate is 4%, and Northwind's credit spread is 200 bps (2%).

- **The Treasury** is discounted at the risk-free 4%. Since its coupon (4%) equals the discount rate (4%), it trades at **par, \$1,000**.
- **Northwind** is discounted at 4% + 2% = **6%**. Its coupon (4%) is now *below* its discount rate (6%), so by the premium/discount logic above, it trades at a **discount** — about **\$916**, the same number we computed earlier for a 6% market rate.
- The \$84 price gap is entirely the credit spread at work. Same coupon, same par, same maturity — the only difference is the name of the issuer, and it's worth \$84 per bond.
- If Northwind's outlook worsens and its spread widens to 400 bps, it's now discounted at 8%, and its price falls further to about **\$840** — even if Treasury rates never moved. A corporate bondholder is exposed to *both* the rate dial and the credit dial.

*For a government bond the issuer is an afterthought, but for a corporate bond the issuer's creditworthiness is a second engine of price movement, captured in the spread it pays over the risk-free benchmark.*

### How the issuer's type changes everything

The issuer's *category* — not just its name — reshapes the bond:

- **Sovereign (Treasury) issuers** borrow in their own currency and are treated as risk-free in that currency, because a government that prints its own money can always nominally repay. Their bonds carry interest-rate risk but essentially no default risk, which is why they anchor the whole system as [the risk-free benchmark](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration).
- **Corporate issuers** can and do default, so their bonds carry credit risk on top of rate risk, and their indentures lean heavily on covenants to protect lenders.
- **Municipal issuers** (US states and cities) often pay coupons that are exempt from federal income tax, which means a 3% muni coupon can be worth more *after tax* than a 4% corporate coupon — the issuer's tax status changes the effective return without changing the stated coupon at all.
- **Agency and supranational issuers** (like government-sponsored enterprises or the World Bank) sit in between, with implicit or explicit backing that compresses their spreads toward the sovereign.

The practical upshot: when you read a bond, the issuer line tells you which *risks* you're being paid to bear, and therefore how to interpret every other number on the page. A 6% coupon from a sovereign is a story about rates; a 6% coupon from a junk-rated company is a story about survival.

## The indenture: turning a promise into an enforceable contract

So far we've treated the bond as four friendly numbers. But a loan is only as good as your ability to enforce it, and that's the job of the **indenture** — the formal legal contract behind every public bond. The indenture is often a hundred-plus pages, and it's where the bond stops being a handshake and becomes a security with teeth.

![A tree diagram of the indenture showing its economic terms and its covenants enforced by a trustee on behalf of all bondholders](/imgs/blogs/anatomy-of-a-bond-par-coupon-maturity-issuer-5.png)

The indenture has two halves. The first is the **economic terms** we've already met — par, coupon, maturity, payment dates, the day-count convention, and the bond's *seniority* (where it ranks if the issuer goes bust, a topic for the [capital-structure post](/blog/trading/fixed-income/seniority-recovery-and-the-capital-structure)). The second half is the **covenants** — the issuer's binding promises about how it will behave while it owes you money. Covenants come in two flavors:

- **Affirmative covenants** are things the issuer *must* do: pay coupons and principal on time, file financial reports, maintain insurance, keep its corporate existence. These are the "stay alive and stay honest" promises.
- **Negative covenants** (also called restrictive covenants) are things the issuer *must not* do: take on more than a certain amount of new debt, pay out dividends beyond a limit, sell off key assets, or let financial ratios deteriorate past a threshold. These exist to stop the company from quietly making your loan riskier after you've handed over your money.

Tying it together is the **trustee** — a bank or trust company that represents *all* the bondholders as a group. Individual bondholders are scattered and can't coordinate, so the trustee monitors compliance and, if the issuer breaches a covenant or misses a payment, declares an **event of default** and acts on the bondholders' behalf. The event of default is the trigger that lets bondholders demand immediate repayment ("acceleration") and pursue the issuer's assets. Without the indenture and the trustee, a "bond" would be a polite suggestion; with them, it's an enforceable senior claim.

#### Worked example: a covenant that protects you

Suppose Northwind's indenture has a negative covenant capping total debt at **3× annual earnings** (a common leverage covenant). Northwind earns \$100 million a year, so it can owe at most \$300 million. You bought the bond when Northwind owed \$200 million — comfortably inside the limit.

- A year later, Northwind's CEO wants to borrow another \$200 million to fund an acquisition. That would push total debt to \$400 million = 4× earnings, *breaching* the 3× covenant.
- Because of the covenant, Northwind legally *cannot* take that loan without bondholder consent. Your claim can't be quietly diluted by a pile of new debt ranking alongside yours.
- If Northwind borrowed anyway, the trustee would declare an event of default, and bondholders could accelerate — demand their \$1,000 par back immediately.

*The coupon tells you what you earn; the covenants tell you whether you'll actually collect it — they are the reason a bond is an investment and not just a hope.*

## Coupon frequency: why timing nudges your real return

We mentioned US bonds pay semiannually. It's worth seeing *why* the payment frequency matters, because it introduces the difference between a *stated* rate and an *effective* rate — a distinction that runs through all of finance.

The intuition: if two bonds both have a 4% stated coupon, but one pays it all at year-end and the other pays half at mid-year, the mid-year payer is slightly better. Why? Because you get some of your money *sooner*, and you can reinvest that early cash to earn a little more. More frequent payments → more reinvestment → a slightly higher *effective annual yield*, even though the stated rate is identical.

![A grid showing how the same four-percent coupon produces a slightly higher effective annual yield as the payment frequency rises from annual to semiannual to quarterly to monthly](/imgs/blogs/anatomy-of-a-bond-par-coupon-maturity-issuer-6.png)

The effective annual rate for a stated rate $r$ paid $n$ times a year is:

$$ \text{EAR} = \left(1 + \frac{r}{n}\right)^{n} - 1 $$

where $r$ is the stated annual coupon rate (4% here), $n$ is the number of payments per year, and EAR is the effective annual rate that accounts for reinvesting each payment.

#### Worked example: the frequency premium on a 4% coupon

Hold the stated rate fixed at 4% and crank up the frequency on \$1,000 of par:

- **Annual** ($n=1$): EAR = $(1 + 0.04/1)^1 - 1 = 4.000\%$. You get one \$40 payment. Baseline.
- **Semiannual** ($n=2$): EAR = $(1 + 0.04/2)^2 - 1 = 4.040\%$. Two \$20 payments; the mid-year \$20 reinvests for half a year. Worth about **+\$0.40** a year versus annual.
- **Quarterly** ($n=4$): EAR = $(1 + 0.04/4)^4 - 1 = 4.060\%$. Four \$10 payments.
- **Monthly** ($n=12$): EAR = $(1 + 0.04/12)^{12} - 1 = 4.074\%$. Twelve \$3.33 payments.
- **Continuous** (the theoretical limit): EAR = $e^{0.04} - 1 = 4.081\%$. This is the ceiling — you can't squeeze out more than about 4.081% from a 4% stated rate, no matter how often you pay.

So the entire range from annual to continuous is just **8 basis points**. The frequency effect is real but small at low rates — which is exactly why it's an easy detail to overlook and an easy way to get a yield comparison subtly wrong. The lesson generalizes: whenever someone quotes a rate, ask how often it compounds, or you might be comparing two numbers that aren't the same thing.

*A stated coupon rate and an effective yield are different numbers; paying more often lets you reinvest sooner, which is why frequency quietly lifts your real return.*

### Day-count conventions: how "a year" is actually counted

There's one more piece of plumbing hiding inside the coupon: how the market counts the *days* in a period. It sounds pedantic, but it determines the exact size of each coupon and of the accrued interest you'll pay when you trade mid-period, so it's worth a paragraph.

When a coupon period isn't a clean six months — say you need the interest earned over 47 days — you need a rule for turning "47 days" into a fraction of a year. That rule is the **day-count convention**, and different markets use different ones:

- **30/360** pretends every month has 30 days and every year has 360 days. It's the convention for most US corporate and municipal bonds because it makes coupon math clean: each semiannual coupon is always exactly half the annual coupon, no matter how the calendar falls.
- **Actual/actual** counts the real number of days in the period over the real number of days in the year. US Treasuries use this, so a Treasury's accrued interest depends on the literal calendar.
- **Actual/360** counts real days but divides by a 360-day year — common in money-market instruments, and it quietly makes the effective rate a touch higher than the stated one.

The differences are small per trade but they're *real money* at scale, and getting the convention wrong is a classic source of settlement disputes. The headline for a beginner: when a bond says "4% coupon," the precise dollars also depend on a day-counting rulebook agreed in the indenture — another reminder that the contract, not your intuition, defines the cash flows.

#### Worked example: accrued interest under two day-counts

You buy our 4% Northwind bond (\$20 semiannual coupon) **47 days** into a coupon period, and you want to know the accrued interest the seller is owed.

- Under **30/360**, the half-year period is treated as 180 days, so accrued = (47 / 180) × \$20 ≈ **\$5.22**.
- Under **actual/actual**, suppose the actual period has 184 days; accrued = (47 / 184) × \$20 ≈ **\$5.11**.
- The 11-cent gap looks trivial on one bond — but on a \$100 million position it's about \$11,000, which is exactly why trading desks automate the convention and never eyeball it.

*The coupon rate sets the size of the payment, but the day-count convention sets exactly how it's sliced across time — a detail that's invisible until you trade mid-period.*

## Settlement and the bond's lifecycle

A bond doesn't only exist on the day it's bought and the day it matures. It has a full lifecycle, and a couple of mechanical facts about *trading* it matter for any real investor.

![A timeline of a bond's lifecycle from issuance through coupon payments to maturity and final redemption](/imgs/blogs/anatomy-of-a-bond-par-coupon-maturity-issuer-7.png)

A bond is **born** in the **primary market**: the issuer sells it to investors and receives cash. This is the moment Northwind gets its \$1,000 and you become a lender. After that, the bond trades in the **secondary market** — you can sell your bond to another investor at any time before maturity, at whatever the market price is that day. The issuer isn't involved in secondary trades; it just keeps paying coupons to whoever happens to own the bond on each payment date. Most of the "bond market" you hear about is this secondary trading, which we cover in [how bonds actually trade](/blog/trading/fixed-income/how-bonds-actually-trade-otc-dealers-and-treasury-market-structure).

Through its **middle life**, the bond simply pays coupons on schedule. Some bonds are **callable**, meaning the issuer has the right to redeem them early (usually when rates fall and it wants to refinance cheaper) — a feature that caps your upside and gets [its own treatment](/blog/trading/fixed-income/the-many-yields-current-yield-ytm-and-yield-to-call). Our Northwind bond is not callable, so it runs its full term. At **maturity**, the issuer makes the final coupon, repays par, and the bond ceases to exist. The contract is extinguished.

Two settlement mechanics are worth knowing:

- **Settlement date (T+1).** When you buy a bond, the trade *executes* today but *settles* — cash and bond actually change hands — one business day later for US Treasuries ("T+1"). This is why a bond's "settlement date" can differ from its "trade date."
- **Accrued interest (clean vs dirty price).** Coupons accrue continuously but pay only twice a year. If you buy a bond halfway between coupon dates, the seller has earned half a coupon they won't receive, so you pay them that **accrued interest** on top of the quoted price. The quoted price is the **clean price**; the clean price plus accrued interest is the **dirty price** (what you actually pay). For our 4% bond bought exactly three months into a six-month coupon period, you'd pay about \$10 of accrued interest (half of the \$20 coupon) on top of the clean price. We unpack clean versus dirty price fully in [why bond prices move when rates move](/blog/trading/fixed-income/why-bond-prices-move-when-rates-move-and-by-how-much).

#### Worked example: buying between coupon dates

You buy Northwind's bond on a day exactly **3 months** after its last coupon, when the clean (quoted) price is **\$1,005**. The bond pays \$20 every 6 months.

- The seller has held the bond for 3 of the 6 months in this coupon period, so they've earned half the upcoming \$20 coupon: accrued interest = $\frac{3}{6} \times \$20 = \$10$.
- You pay the **dirty price** = clean price + accrued = \$1,005 + \$10 = **\$1,015**.
- Three months later, the full \$20 coupon arrives — but it comes to *you*, the current owner. You collect \$20, of which \$10 reimburses the accrued interest you paid and \$10 is the coupon you genuinely earned over your 3 months of ownership.
- The clean price (\$1,005) is what's quoted on screens precisely so that this sawtooth of accrued interest doesn't make the price look like it's jumping around at every coupon date.

*Bonds quote a clean price but trade at a dirty one; accrued interest just makes sure each owner is paid for exactly the time they held the bond.*

## Common misconceptions

**"Par is what I pay for the bond."** No — par is what you get *back* at maturity, and what the coupon is calculated on. The price you pay is set by the market and is usually *not* par; it's par only when the market rate happens to equal the coupon rate. Most bonds trade at a premium or discount most of the time.

**"A higher coupon means a better investment."** Not necessarily. A bond with a fat coupon usually costs more (trades at a premium), so you're paying upfront for that income, and you'll take a small capital loss as the price drifts back toward par by maturity. What matters for your total return is the *yield* — coupon and price change together — not the coupon alone. A 6% coupon bond bought at \$1,100 and a 4% coupon bond bought at \$1,000 can have the exact same yield.

**"Bonds are safe because you get your money back."** Two big caveats. First, you only get par back *if the issuer doesn't default* — that's credit risk, and it's the whole reason a Treasury and a junk bond paying the same coupon are not the same investment. Second, even a default-free Treasury can lose you serious money if you sell before maturity after rates have risen: in 2022, long Treasuries fell over 20% in price even though every coupon was paid on time. "Get your money back at maturity" and "can't lose money" are different claims.

**"The coupon rate is my return."** Only if you buy at par and hold to maturity and reinvest every coupon at the same rate. Buy at a discount and your return beats the coupon; buy at a premium and it trails the coupon; reinvest coupons at a different rate and your realized return shifts again. The single number that bundles all this is *yield to maturity*, which we devote a [whole post](/blog/trading/fixed-income/the-many-yields-current-yield-ytm-and-yield-to-call) to.

**"A floating-rate bond can't lose money."** Floaters resist *interest-rate* losses because their coupon resets, so their price stays near par. But they still carry *credit* risk — if the issuer's quality deteriorates, the floater's price falls even though its coupon is floating. And when rates fall, a floater's income drops with them, which can be its own kind of pain for an income-dependent holder.

**"The bond's price and the company's stock price move together."** Often they diverge. When a company gets *safer* (less likely to default), both its bonds and stock can rise. But when interest rates rise economy-wide, the company's bonds fall (rate risk) even if the business is doing fine and the stock is climbing. Bonds answer to two masters — the level of interest rates and the issuer's credit — and only the second is shared with the stock.

## How it shows up in real markets

**The US Treasury market: the world's benchmark bond.** Every concept in this post is most cleanly visible in US Treasuries — bills, notes, and bonds issued by the US government. A 10-year Treasury note has a par of \$1,000 (sold in \$100 units), a fixed semiannual coupon set at auction, and a fixed maturity. Because the US government is treated as default-free, a Treasury's price moves *only* with interest rates — there's no credit story muddying it — which makes it the purest demonstration of the price–par seesaw. The 10-year Treasury *yield* is the single most-watched number in global finance: it's the reference rate off which mortgages, corporate bonds, and stock valuations are all priced. When commentators say "the 10-year hit 4.5%," they're describing exactly the rate on the right-hand axis of our influence figure. See [US Treasuries, the risk-free benchmark](/blog/trading/fixed-income/us-treasuries-the-risk-free-benchmark-of-the-world) for the full anatomy.

**Apple's 2013 bond sale: a blue-chip issuer at work.** In 2013, Apple — sitting on a mountain of cash — issued \$17 billion of bonds, then the largest corporate bond sale in history. Why borrow when you're flush? Because most of Apple's cash was overseas, and borrowing at rock-bottom rates was cheaper than repatriating it. The bonds had the same four parts as our Northwind note: par, fixed coupons (some tranches around 2–4%), specific maturities (3 to 30 years), and Apple as the issuer. The 30-year tranche is a textbook lesson in maturity risk: as rates rose in subsequent years, those long Apple bonds fell sharply in price — not because Apple got riskier, but purely because of the maturity dial we discussed. A pristine issuer's *long* bond is still a leveraged bet on interest rates.

**2022: when "safe" bonds had their worst year ever.** For decades, bonds were sold as the boring, safe ballast in a portfolio. Then 2022 happened: the US Federal Reserve raised its policy rate from near zero to over 4% in a single year to fight inflation, and the prevailing market rate in our influence figure lurched violently to the right. Every fixed-coupon bond's price slid down the curve. The broad US investment-grade bond index fell about **−13%**, its worst year on record, and long-maturity Treasuries fell more than 20%. Investors who thought "I get my money back" learned the asterisk: yes, *at maturity* — but if you needed to sell in 2022, you sold at a steep discount. This episode is the clearest modern illustration that a fixed coupon plus a maturity equals interest-rate risk. We dig into it from the allocation angle in [the case study of 2022](/blog/trading/cross-asset/case-study-2022-stocks-and-bonds-both-fell).

**Floating-rate notes during the 2022 rate hikes.** While fixed-rate bonds were getting hammered in 2022, floating-rate notes barely budged. As the Fed hiked, SOFR climbed, and floaters' coupons reset upward in lockstep — so their prices held near par while their income rose. A bank holding floaters earned *more* as rates climbed, exactly the hedge floaters are designed to provide. The same year that punished fixed-coupon holders rewarded floating-coupon holders, a vivid demonstration that the coupon clause — fixed versus floating — is not a technicality but a fundamental choice about which risk you want to bear.

**Zero-coupon Treasuries (STRIPS) and pension matching.** Pension funds and insurers owe fixed sums on known future dates — a retiree's payment in 2045, say. To match that, they buy zero-coupon bonds (Treasury STRIPS) that pay one lump sum on exactly that date, with no coupons to reinvest. This is *immunization* — using a bond's maturity to lock down a future obligation — and it's why zeros, despite being the most rate-sensitive bonds, are prized by these institutions: held to maturity, a zero delivers a guaranteed amount on a guaranteed day. The same volatility that scares a trader is irrelevant to a matcher who never sells. We cover this in [immunization and duration matching](/blog/trading/fixed-income/immunization-and-duration-matching-how-pensions-and-insurers-hedge).

**From the 10-year Treasury to your mortgage rate.** The most direct way the four parts of a bond touch ordinary life runs through the 30-year mortgage. Mortgage lenders price home loans off the 10-year Treasury yield (plus a spread for credit and prepayment risk), because a pool of mortgages behaves like a long bond. So when the 10-year Treasury's *price* falls and its *yield* climbs — exactly the seesaw from our influence figure — mortgage rates follow within days. In 2020–2021, with the 10-year yield near 0.7%, US 30-year mortgage rates fell under 3%, and a \$400,000 loan cost roughly \$1,680 a month. By late 2023, with the 10-year yield above 4.8%, the same mortgage rate topped 7.5% and that same loan cost about \$2,800 a month — nearly \$1,100 more, on an identical house, driven entirely by where the benchmark bond's yield sat. The fixed coupon, par, and maturity of a Treasury note are not abstractions; they set the monthly check millions of families write. This is the transmission mechanism we trace in [from the ten-year yield to your mortgage](/blog/trading/fixed-income/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates).

**The 2007 paper-bond curiosity.** Until surprisingly recently, the word "coupon" was literal. Bearer bonds with physical coupons circulated into the late 20th century; you'd clip a coupon and present it at a bank to collect your interest. The US largely ended new bearer-bond issuance in 1982 (the TEFRA rules) for tax-enforcement reasons, moving to "registered" bonds where the issuer knows who owns each one. The vocabulary we still use — "coupon," "clipping coupons" as slang for living off bond income — is a fossil of that paper era, a reminder that today's electronic \$140-trillion market grew out of literal slips of paper.

## When this matters to you, and further reading

You now own the foundational vocabulary of the largest market on earth. The next time you hear "the 10-year yield rose," you'll know that means existing bonds' prices fell, that the world's benchmark price of money went up, and that mortgages and stock valuations are about to feel it. The four parts — issuer, par, coupon, maturity — are the lens through which every later concept in this series will make sense.

This touches your life more than you'd think. The rate on your mortgage, your car loan, and your savings account is downstream of the bond market's price of money. Your retirement fund almost certainly holds bonds, and whether it gained or lost in 2022 came straight from the seesaw we drew. Even your job can hinge on it: when bond yields rise, companies borrow less and hire less.

A final way to hold all of this together: a bond is the simplest possible financial instrument and, at the same time, the most consequential. It is just a promise to pay fixed amounts on fixed dates — issuer, par, coupon, maturity — yet the price the market sets on that promise is the price of time itself, and the price of time prices everything else. Master the four parts of one \$1,000 note and you have not learned a niche corner of finance; you have learned the rate that quietly sits underneath your house, your savings, and the value of every other asset you'll ever own.

Where to go next in this series:

- The inverse price–yield relationship in full: [Price and yield, the seesaw at the heart of bonds](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds).
- How a bond is actually priced: [Discounting cash flows, how a bond is priced](/blog/trading/fixed-income/discounting-cash-flows-how-a-bond-is-priced).
- The number that measures price sensitivity: [Duration, the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income).
- Who's on the other side of the loan: [Who issues bonds and why](/blog/trading/fixed-income/who-issues-bonds-and-why-governments-companies-and-cities).
- The macro lens on all of this: [Interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).
- The heavy math, when you want it: [Bond pricing](/blog/trading/quantitative-finance/bond-pricing).

*This is educational material, not investment advice. It explains how bonds work and what risks they carry; it does not recommend buying or selling any security.*

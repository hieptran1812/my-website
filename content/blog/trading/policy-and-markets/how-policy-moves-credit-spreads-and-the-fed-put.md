---
title: "How Policy Moves Credit Spreads and the Fed Put"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How central-bank policy prices the cost of corporate money — the credit spread as fear gauge and canary, the refinancing wall, and the Fed put that compresses risk premiums even before a single bond is bought."
tags: ["monetary-policy", "credit-spreads", "fed-put", "corporate-bonds", "central-banks", "asset-valuation", "investment-grade", "high-yield", "refinancing-risk", "fixed-income", "quantitative-easing", "risk-premium"]
category: "trading"
subcategory: "Policy & Markets"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A credit spread is the extra yield a company pays to borrow above the risk-free Treasury rate, and policy moves it from both ends: higher policy rates raise the refinancing bill that can push a firm toward default, while the belief that the central bank will backstop credit in a crisis — the "Fed put" — quietly compresses spreads in calm times.
>
> - A corporate bond yield = the risk-free Treasury yield + a credit spread, and the spread itself is compensation for two things: default risk and illiquidity.
> - Spreads are the market's fear gauge and the canary in the coal mine — credit tends to crack before equity does, so a blowout in high-yield is an early stress signal.
> - The refinancing channel is how higher policy rates bite: when a 3% bond matures and rolls into a 7% world, interest expense jumps and weak borrowers tip toward default.
> - The number to remember: on March 23, 2020 the Fed announced it would buy corporate bonds for the first time ever, and IG spreads (then 373 bp) began tightening that day — months before it bought a single bond.

On the morning of March 23, 2020, the corporate-bond market was seizing up. Investment-grade (IG) spreads — the extra yield over Treasuries that the safest companies in America pay to borrow — had blown out from about 100 basis points in February to 373 basis points. High-yield (HY) spreads, the cost for riskier "junk"-rated firms, had rocketed past 1,000 basis points to 1,087. A basis point is one hundredth of a percentage point, so 1,087 basis points means high-yield borrowers were paying nearly 11 full percentage points over Treasuries. New bond issuance had frozen. Healthy companies could not refinance debt that was coming due. The plumbing of corporate finance was clogging, and a wave of defaults looked plausible.

Then, before the stock market even opened, the Federal Reserve announced something it had never done in its 106-year history: it would create two special vehicles — the Primary Market Corporate Credit Facility (PMCCF) and the Secondary Market Corporate Credit Facility (SMCCF) — to buy corporate bonds directly. The Fed had bought Treasuries and mortgage-backed securities before, but never the debt of private companies. The announcement was an unmistakable signal: the central bank would stand behind the credit market.

Spreads turned that day. They did not wait for the Fed to actually buy anything. The first SMCCF purchase did not settle until May 12, almost two months later, and the facilities ultimately bought only about \$14 billion of bonds and ETFs — a rounding error in a \$10-trillion market. Yet by late summer IG spreads were back near 140 basis points and companies were issuing record amounts of new debt at low rates. The Fed had repriced the entire cost of corporate borrowing with a press release. That is the purest demonstration of two ideas this post is about: the **credit spread** as the price of corporate money, and the **Fed put** — the market's belief that the central bank will catch credit when it falls.

![Policy levers reach the credit spread through the cost of corporate money and reprice corporate bond values](/imgs/blogs/how-policy-moves-credit-spreads-and-the-fed-put-1.png)

This whole series follows one spine: a policy lever pulls, it travels through a transmission channel, and it shows up as a change in what an asset is worth. For credit, the lever is a policy rate, the quantity of money the central bank creates (QE/QT), or an explicit backstop; the channel is the **cost of corporate money**; and the asset that reprices is a corporate bond, whose value is the mirror image of its spread. Pull the lever and you change the spread a company pays; change the spread and you change the price of every bond it has outstanding and the viability of every refinancing it must do. Let us build that machine from the ground up.

## Foundations: what a credit spread actually is

Begin with something everyone has done: lending money to a friend versus lending it to a stranger. If your most reliable friend asks to borrow \$100 for a year, you might lend it for a token thank-you, because you're sure you'll be paid back. If a stranger with a patchy reputation asks for the same \$100, you'd want more — maybe \$110 back, maybe \$120 — to compensate for the chance they vanish. And if you might need that cash back in an emergency, you'd charge the stranger even more, because getting your money out of a shaky loan early is hard. The *extra* you charge the stranger over what you'd charge your reliable friend — that gap — is exactly what a credit spread is. The reliable friend is the U.S. Treasury; the stranger is a corporation; and the gap is the price of doubt plus the price of being stuck.

With that picture in hand, the formal version is easy. Start with the simplest building block in finance: the risk-free rate. When the U.S. Treasury borrows, it is assumed never to default, because it issues the currency it borrows in. The yield on a Treasury bond is therefore the price of *time alone* — pure compensation for lending money for, say, five or ten years with no risk of not being repaid. Every other dollar-denominated interest rate in the economy is built on top of this floor.

A corporation is not the U.S. government. It can go bankrupt. Its bonds can be hard to sell in a panic. So when a company borrows by issuing a bond, lenders demand a yield *higher* than the matching-maturity Treasury. That extra yield is the **credit spread**. If a five-year Treasury yields 4.0% and a five-year bond from a solid company yields 5.2%, the spread is 1.2 percentage points, or 120 basis points.

The spread is not arbitrary. It is the sum of two distinct compensations:

- **The default premium.** Some fraction of similar bonds will not be paid back in full. Lenders need extra yield across the whole pool to break even after losses. The default premium is roughly the *probability of default* times the *loss given default* (how much you lose when it happens, after recovering some pennies on the dollar in bankruptcy).
- **The illiquidity premium.** A Treasury can be sold instantly in size at a razor-thin bid-ask. A specific corporate bond may trade rarely; selling it in a hurry means accepting a worse price. Lenders demand extra yield to hold something they cannot dump cheaply. This premium balloons in a crisis, when everyone wants to sell at once and no one wants to buy.

![A corporate yield stacks a default premium and an illiquidity premium on top of the same risk-free Treasury floor](/imgs/blogs/how-policy-moves-credit-spreads-and-the-fed-put-3.png)

The most-watched measure of the spread is the **option-adjusted spread (OAS)**, which strips out the effect of any embedded options (like a call feature that lets the issuer repay early) so you are comparing pure credit risk. When traders say "IG spreads are at 90" they mean the ICE BofA U.S. Corporate Index OAS is 90 basis points. We will use OAS throughout, because that is the series the data is reported in.

Two grades of corporate borrower matter:

- **Investment grade (IG)** — bonds rated BBB−/Baa3 and above by the rating agencies. These are large, stable companies (think Apple, Johnson & Johnson, a regulated utility). Their spreads are typically 80–150 basis points in calm times.
- **High yield (HY)**, also called "junk" or "speculative grade" — bonds rated BB+/Ba1 and below. These are smaller, more leveraged, or troubled firms. Their spreads run 300–500 basis points in calm times and can blow past 1,000 in a crisis.

The line between these two grades is not a smooth slope; it is a cliff. The reason is institutional. Many of the largest pools of capital in the world — insurance companies, pension funds, regulated banks — are *required by their mandates or by regulation* to hold mostly investment-grade bonds. A bond rated BBB− is eligible; the same bond downgraded one notch to BB+ is suddenly off-limits for a huge swath of natural buyers. So when a company is downgraded across that boundary — a so-called **fallen angel** — forced selling by mandate-constrained investors can blow its spread out far more than the one-notch change in default risk would justify. The 2020 stress was so dangerous partly because a record amount of debt sat right at the BBB edge, threatening a wave of fallen angels that would have overwhelmed the smaller high-yield market. The Fed explicitly extended its facilities to cover recent fallen angels precisely to defuse this cliff. The lesson: spreads are not just about fundamentals; they are about *who is allowed to buy*, and policy can change that overnight.

It is also worth being precise about the rating agencies themselves. Moody's, S&P, and Fitch assign letter grades (Aaa/AAA down to C/D) that map to historical default rates. These ratings are slow — they lag the market, which is why the *spread* usually moves before the *rating* does; by the time a bond is officially downgraded, its spread has often already repriced. Credit traders treat the rating as a coarse, sticky label and the spread as the live, continuous price. When a downgrade finally lands, it can still matter — because of the mandate cliff above — but it rarely surprises the market about the underlying risk. The market knew first; the canary sang before the agency confirmed it.

#### Worked example: a corporate bond's yield and price

Suppose the five-year Treasury yields **4.0%**. A solid investment-grade company issues a five-year bond with a credit spread of **1.2%** (120 basis points). The bond's yield to maturity is therefore:

```
corporate yield = risk-free yield + credit spread
                = 4.0% + 1.2%
                = 5.2%
```

Now price it. The bond has a \$1,000 face value and pays a 5.2% annual coupon (\$52 a year) for five years, returning the \$1,000 at the end. If the market discount rate equals the 5.2% yield, the bond prices exactly at par — \$1,000 — because the coupon equals the yield. The present value of the five \$52 coupons plus the \$1,000 principal, each discounted at 5.2%, sums to \$1,000.

The point is the decomposition. Of that 5.2% yield, **4.0% is the price of time** (set by Treasury markets and, behind them, the Fed) and **1.2% is the price of this company's credit risk and the illiquidity of its bond**. When policy moves the 4.0% floor, every corporate yield moves with it; when fear or policy moves the 1.2% spread, the *relative* cost of corporate money moves. The spread is the part this post is about.

## How spreads become the market's fear gauge

Here is the property that makes credit spreads so useful: they are a remarkably honest, real-time price of fear. When investors grow worried about the economy, they demand more compensation to hold risky corporate debt, and spreads widen. When they relax, spreads compress. Because the corporate-bond market is enormous and dominated by sophisticated institutional money — pension funds, insurers, asset managers running their own credit research — the spread aggregates a great deal of careful thinking about who is likely to default and when.

Spreads are also a **canary in the coal mine**: credit tends to crack *before* equity does in a stress episode. The reason is structural. Bondholders sit *above* shareholders in the capital structure — if a company fails, bondholders are paid first and equity holders are wiped out last. So a bondholder's entire job is to worry about the downside; a stockholder is partly dreaming about the upside. When the economic weather turns, the people whose job is to worry about default — credit investors — react first. A widening in high-yield spreads while the stock market is still near its highs is one of the classic late-cycle warning signs.

![Investment-grade and high-yield spreads blow out together in every crisis, with high-yield as the louder canary](/imgs/blogs/how-policy-moves-credit-spreads-and-the-fed-put-2.png)

The chart above plots IG and HY spreads across two decades. Notice three things. First, the two lines move together — when fear rises, all credit widens — but HY moves far more, because junk bonds have far more default risk to reprice. Second, the spikes line up exactly with the crises everyone remembers: the 2008 global financial crisis (HY to 1,971 bp), the March 2020 COVID shock (HY to 1,087 bp), the 2022 hiking cycle (HY to 552 bp), and the April 2025 tariff shock (HY to 461 bp). Third, between crises spreads grind down to remarkably low levels — IG near 80 bp, HY near 300 bp — which is itself a clue about the Fed put we will get to.

There is a second reason credit leads equity, beyond the seniority point: the corporate-bond market is where companies actually *fund themselves*. When spreads widen far enough, new issuance dries up — a company that planned to issue a bond simply cannot, or can only at a punishing rate. That is not an abstract signal; it is a real tightening of financing conditions that feeds back into the economy. A frozen bond market means deferred investment, cancelled buybacks, and, at the extreme, an inability to roll maturing debt. So a credit blowout is both a *thermometer* (measuring fear) and a *cause* (tightening real financing). Equity is mostly the thermometer; credit is both. That dual role is why central banks watch spreads so closely and why they were willing, in 2020, to intervene in the corporate-bond market directly when they had never done so before — a frozen credit market is a self-reinforcing economic problem, not just a sentiment reading.

The *shape* of a widening carries information too. When IG and HY widen together in roughly fixed proportion, the market is repricing broad recession risk — everyone gets a bit more cautious. When HY widens dramatically faster than IG, the market is worried specifically about the *weak tail*: the leveraged, low-rated borrowers most exposed to a refinancing wall or a sector shock. The ratio of HY to IG spreads is itself a gauge of how much the stress is concentrated in fragile credits versus spread broadly. A 2008-style event pushes both to extremes; a targeted shock (an energy-sector default scare, a single large bankruptcy) widens HY while IG barely moves. Reading the difference tells you whether the market fears a systemic problem or a localized one.

For a deeper statistical treatment of *how tightly* spreads correlate with equity drawdowns and lead them in time, see the companion piece on [credit spreads as the risk correlation and the canary](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary). This post owns the *mechanism and the case studies*; that one owns the *betas and correlations*.

## The discount-rate channel: why a wider spread means a lower price

A spread is a yield, and a yield is the inverse of a price. When the spread on a company's bonds widens, the price of those bonds *falls* — and the size of the fall depends on the bond's **duration**, a measure of how sensitive its price is to a change in yield. As a rule of thumb, a bond's price changes by approximately its duration times the change in yield, with the opposite sign.

This matters because a spread that "only" widens by a couple of percentage points can take a brutal bite out of the price of a long-dated bond. A pension fund holding investment-grade corporate bonds is not exposed to default in any single name very often — defaults among IG issuers are rare. But it is enormously exposed to *spread widening*, because a market-wide repricing marks down the value of its whole portfolio at once. That mark-to-market loss is real: it shows up on balance sheets, triggers margin calls, and can force selling that widens spreads further — the doom-loop that froze the market in March 2020.

#### Worked example: a +200 bp spread widening on a 7-year IG bond

Take an investment-grade bond with a **modified duration of 6.5 years** (typical for a 7-year maturity), trading at par, \$1,000 face. The economy turns and its spread widens by **200 basis points** (2.0 percentage points) — a sharp but not catastrophic move, roughly the IG widening from late 2021's 88 bp toward a recession-grade 290 bp.

The approximate price change is:

```
price change percent = - duration x yield change
                      = - 6.5 x 2.0%
                      = - 13.0%
```

So the bond falls from \$1,000 to about **\$870** — a loss of roughly \$130 on \$1,000, or 13%, with *no default at all*. The company is still paying its coupons; nothing has actually gone wrong with the firm. The loss is pure repricing of risk. For the convexity-minded, the second-order term softens this slightly (a real 7-year bond might fall ~12%), but the lesson stands: **spread risk, not default, is what hurts most IG investors most of the time.** A 2-point spread move on a long bond is a double-digit drawdown.

Now you can see why the canary matters so much for the people who own credit. They are not mostly afraid that any one company will fail; they are afraid the *price of corporate risk* will reset higher and mark their whole book down overnight.

Two refinements sharpen this. First, **duration is why long bonds suffer most in a spread shock**. A 2-year bond has a duration near 2; a 30-year bond a duration near 18 or more. The same 200 bp spread widening that costs the 7-year bond 13% costs the 30-year bond closer to 30–35%. That is why, in a credit sell-off, the long end of the spread curve and the longest-dated bonds get hit hardest — and why investors who fear a widening shorten the duration of their credit books, accepting lower yield for less spread sensitivity. Second, **the policy rate and the spread can move in opposite directions and partly cancel**. In a growth scare the Fed cuts (the risk-free rate falls, lifting bond prices) while spreads widen (lowering them). For a high-quality IG bond the rate cut can dominate, so the bond actually *gains* even as its spread widens — Treasuries and IG can rally in a recession. For a junk bond, the spread widening dominates and the bond falls hard despite the rate cut. The grade of the bond decides which force wins, which is why investment-grade credit behaves more like a Treasury and high-yield behaves more like equity. Knowing which regime a bond lives in is half of understanding how policy will move its price.

## The refinancing channel: how higher policy rates actually bite

The first way policy moves credit is through that risk-free floor. When the central bank raises its policy rate, Treasury yields rise, and every corporate yield rises with them even if spreads do not budge. But there is a second, slower, and more dangerous channel: **refinancing**.

Companies rarely repay their bonds out of cash. They roll them over — when a bond matures, they issue a new one to pay off the old one. This is fine when rates are stable. It becomes a vice when rates have risen sharply, because the company is forced to replace cheap old debt with expensive new debt. The whole stock of corporate debt does not reprice at once; it reprices *as it matures*, bond by bond, year by year. The schedule of when a company's (or a whole market's) debt comes due is called the **maturity wall**, and the years after a big hiking cycle are when that wall does the most damage.

This is the mechanism by which the Fed's policy rate — which the Fed only directly sets at the very front end — eventually reaches a company that borrowed for ten years. The company is insulated until its bond matures. Then it hits the wall. A firm that financed itself at 3% in 2021 and must refinance in a 7% world in 2025 sees its interest expense on that slice of debt more than double. If its earnings have not grown to match, its **interest coverage ratio** (earnings divided by interest expense — how many times over it can pay its interest bill) collapses, and the rating agencies and bond investors notice. Its spread widens *because* its ability to service debt has deteriorated. Higher rates do not just make new borrowing expensive; they manufacture credit risk in companies that were previously fine.

![Higher policy rates widen spreads through the refinancing wall while the Fed put compresses them through reach-for-yield](/imgs/blogs/how-policy-moves-credit-spreads-and-the-fed-put-5.png)

The diagram above splits the two policy forces. On the left, a rising policy rate flows into the refinancing wall and falling coverage, which widen the spread. On the right, the belief in a backstop caps tail risk and pulls yield-hungry investors into credit, which compresses the spread. Both forces converge on the bond's value, where spread times duration translates into a price move. In normal times the right-hand force dominates and spreads sit low; in a crisis the left-hand force takes over and they explode — until, sometimes, the central bank intervenes and the right-hand force snaps back.

#### Worked example: refinancing \$500M from a 3% coupon to 7%

A mid-sized company has **\$500 million** of bonds maturing, originally issued at a **3.0%** coupon. To pay them off it must issue new five-year bonds, and the world has changed: the risk-free rate is higher and its own spread has widened, so the new coupon is **7.0%**.

Old annual interest on this slice:

```
500,000,000 x 3.0% = 15,000,000   (15 million dollars a year)
```

New annual interest after refinancing:

```
500,000,000 x 7.0% = 35,000,000   (35 million dollars a year)
```

The refinancing adds **\$20 million a year** of interest expense — the bill more than doubles. Suppose the company earns \$80 million a year in operating profit (EBIT) and this \$500M was its only debt. Its interest coverage falls from:

```
old coverage = 80,000,000 / 15,000,000 = 5.3x
new coverage = 80,000,000 / 35,000,000 = 2.3x
```

A 5.3x coverage ratio is comfortably investment-grade. A 2.3x ratio is borderline — the kind of number that gets a company put on negative watch and pushed toward high-yield. **The same firm, the same earnings, became riskier purely because the policy rate rose while its old debt matured.** That is the refinancing channel in one example: policy didn't change the business; it changed the cost of the business's money, and that change *is* credit risk.

This is also why a hiking cycle's damage is *delayed*. The 525 basis points of hikes in 2022–23 did not blow up corporate credit in 2022; the maturity wall pushed the pain into 2024–2026 as cheap pandemic-era debt came due. The canary sings on a lag.

Three features of the refinancing channel are worth making explicit, because they govern *who* gets hurt and *when*:

- **It is concentrated in the weak.** A strong investment-grade company refinancing from 3% to 7% sees its interest bill rise, but its earnings are large relative to its debt, so coverage stays comfortable. The same percentage-point increase is fatal for a highly leveraged high-yield borrower whose coverage was already thin. This is why the refinancing channel widens *high-yield* spreads far more than IG spreads — the wall selectively demolishes the most fragile borrowers, which is exactly what the canary is supposed to detect.
- **It is front-loaded by floating-rate debt.** Not all corporate debt is fixed-rate bonds. Leveraged loans, which fund private-equity-owned companies, mostly carry *floating* rates that reset with the policy rate every few months. For those borrowers there is no maturity wall to wait for — their interest bill jumped almost immediately when the Fed hiked in 2022. That is one reason the private-credit and leveraged-loan corner of the market felt the hiking cycle first, even as the public high-yield bond market, with its longer fixed maturities, lagged.
- **It interacts with the put.** When refinancing stress builds, the market starts to ask whether the central bank will ease to relieve it — reviving the Fed put. In a low-inflation world, the mere expectation of cuts can compress spreads and let companies refinance before the wall does real damage. In a high-inflation world, the Fed cannot ride to the rescue, and the wall stands. So the same refinancing wall can be a non-event or a crisis depending entirely on whether inflation lets the central bank ease — another reminder that the inflation backdrop is the master switch over the whole credit channel.

The practical upshot for 2024–2026 was that the wall was large but the damage was contained, because inflation fell fast enough that the Fed could start cutting in late 2024 and through 2025 (three cuts, to a 3.50–3.75% range). Companies refinanced into a *falling*-rate environment rather than a still-rising one, the put was live, and spreads stayed historically tight even as cheap debt rolled off. Had inflation stayed high and forced rates to keep climbing, the same maturity wall would have produced a far uglier default cycle. The wall was the same; the policy backdrop decided whether it bit.

## The liquidity channel: QE, QT, and the "everything bid"

There is a third lever, and it works on the illiquidity-premium part of the spread. When the central bank does **quantitative easing (QE)** — creating reserves to buy bonds — it does two things to credit. Directly, if it buys corporate bonds (as in 2020) it bids up their prices and compresses spreads. Indirectly, and far more powerfully, it floods the financial system with cash that has to go *somewhere*. With Treasuries yielding almost nothing, investors are pushed out the risk curve into corporate bonds to earn anything at all. This "reach for yield" is a relentless bid under credit that compresses spreads even when the Fed is not buying a single corporate bond. It is the same "everything bid" that lifts stocks and crypto in an easy-money regime.

![Fed assets ballooned in 2020 as QE plus the corporate-bond backstop put a floor under credit](/imgs/blogs/how-policy-moves-credit-spreads-and-the-fed-put-6.png)

The Fed's balance sheet, charted above, is the footprint of this lever. It went from \$0.9 trillion before 2008 to \$4.5 trillion after the QE1–QE3 programs, then exploded from \$4.2 trillion to \$7.4 trillion in 2020 and peaked at \$8.97 trillion in April 2022. Each expansion coincided with a compression in spreads; the **quantitative tightening (QT)** that followed — letting the balance sheet run off toward \$6.55 trillion by late 2025 — drained liquidity and was one reason spreads were jumpier in 2022–25.

It is worth separating the two ways QE reaches credit, because they are often conflated. The *direct* effect — the Fed actually buying corporate bonds — happened only once, in 2020, and was small in dollar terms. The *indirect* effect — flooding the system with reserves so that yield-starved investors pile into credit — is the constant, structural force. Through most of the 2010s the Fed never bought a single corporate bond, yet QE relentlessly compressed spreads by removing safe-yielding Treasuries from the market and forcing the marginal investor out the risk curve. An insurance company that needs a 5% return to meet its liabilities cannot earn it on a 1% Treasury; it must reach into corporate bonds, and its reaching bids spreads tighter. Multiply that across the whole institutional world and you get the "everything bid" — a regime in which spreads sit far below where default fundamentals alone would put them, simply because there is too much money chasing too little yield. This is the same force that lifts equities and even crypto in an easy-money era; credit just happens to be the most direct beneficiary because it sits one rung above Treasuries on the risk ladder.

QT runs the movie backward. As the Fed lets bonds mature without replacing them, reserves drain, the marginal reach-for-yield investor pulls back, and spreads lose their structural bid. QT does not usually *cause* a spread crisis on its own, but it removes the cushion, leaving credit more exposed to whatever shock comes next — which is part of why the 2022–25 period, dominated by QT, saw sharper spread reactions to the hiking cycle and the tariff shock than the QE-soaked 2010s did. For the mechanism of how balance-sheet policy lifts and drains the whole risk-asset complex, see [the liquidity channel: QE, QT, and the everything bid](/blog/trading/policy-and-markets/the-liquidity-channel-qe-qt-and-the-everything-bid) and the macro-trading treatment of [QE versus QT](/blog/trading/macro-trading/qe-vs-qt-how-balance-sheet-policy-moves-markets).

## The Fed put: a backstop you mostly never use

Now we can name the most important and most subtle force in credit: the **Fed put**.

In options language, a "put" is the right to sell something at a set price — a floor under your losses. The "Fed put" is the market's belief that the central bank will, in effect, sell the market a floor: that if asset prices fall far enough or credit seizes badly enough, the Fed will cut rates, restart QE, or stand behind credit directly, catching the fall. The term started life describing the equity market (the "Greenspan put" after 1987), but it is most literally true in credit, because in 2020 the Fed put a floor under corporate bonds *explicitly* by promising to buy them.

The deep point is that a credible backstop changes prices **in normal times, not just in the crisis**. If investors believe the worst-case scenario for credit is capped — that the central bank will not let a 2008-style cascade of corporate defaults happen — then they will price less compensation for that tail risk. The default premium shrinks. Spreads sit lower than the underlying default fundamentals alone would justify. The backstop is paid for in the form of permanently tighter spreads, a kind of insurance premium the market collects from itself in advance. This is why IG spreads spent much of 2024–2025 near 80–85 basis points, historically tight, even with a maturity wall looming: the market priced a Fed that would not allow disorder.

The reason it works without much spending is the **announcement effect**. The Fed's most powerful tool is not its checkbook; it is its credibility. When it announces a backstop, sophisticated investors instantly reprice the tail risk, and *they* do the buying — chasing the spread tighter before the Fed has to. The Fed essentially conscripts the market to enforce the floor for it. Mario Draghi proved the limiting case in 2012 with three words and a never-used program; the Fed proved it in 2020 with \$14 billion of actual purchases doing the work of what looked like it would need hundreds of billions.

This belief in a backstop is the credit cousin of the expectations channel — the idea that what the central bank promises about the future moves prices today as much as what it does. For the general version, see [the expectations channel: forward guidance and credibility](/blog/trading/policy-and-markets/the-expectations-channel-forward-guidance-and-credibility) and the macro-trading note on [forward guidance and the Fed put](/blog/trading/policy-and-markets/forward-guidance-and-the-fed-put).

#### Worked example: the announcement effect — spreads compress before any purchase

Walk through the arithmetic of March 2020 for a single long-dated IG bond, **\$1,000 face, modified duration 8 years**, trading near par.

On March 23, with IG spreads at **373 bp**, this bond was priced roughly as if its yield were 3.7 percentage points over the matching Treasury. Over the following weeks, on the announcement alone, the IG index compressed toward **140 bp** by late summer — a tightening of about **233 basis points** (2.33 percentage points). The approximate price gain:

```
price change percent = - duration x spread change
                     = - 8 x (-2.33%)
                     = + 18.6%
```

The bond gained roughly **18.6%** — from \$1,000 to about **\$1,186** — driven entirely by the spread compressing, with the Fed having bought essentially none of it on the day the repricing began. An investor who bought that bond on March 23, simply *believing the announcement*, earned a double-digit capital gain on top of the coupon as the spread ground in. That is the announcement effect made concrete: **the put repriced the asset before the put was ever exercised.** The Fed changed what the bond was worth by changing what investors believed about the floor under it.

This is the most important sentence in the post: in credit, the central bank's promise to act is often worth more than the action, because credible promises are self-fulfilling — the market front-runs the backstop and does the buying itself.

### A short history of the put, and its limits

The idea did not start with corporate bonds. The phrase "Greenspan put" was coined after the 1987 stock-market crash, when Fed chair Alan Greenspan flooded the system with liquidity and the market recovered fast — teaching investors that the Fed would cushion big declines. The same pattern repeated through the 1998 Long-Term Capital Management rescue, the 2008 crisis, and the 2010s, with each chair (Bernanke, Yellen, Powell) seeming to confirm that the central bank would act when markets fell far enough. Over decades this congealed into a deep market belief: *don't fight the Fed, because the Fed will catch you*. The 2020 corporate-bond facilities were the most literal expression of that belief — a put written not on the stock market but explicitly on credit.

But the put has hard limits that are easy to forget in a calm market. **First, it is conditional on low inflation.** The central bank can only ease to rescue credit if doing so does not stoke an inflation it is fighting. In 2022, with inflation at 9%, the put was suspended — the Fed hiked into falling markets and let spreads widen, because price stability outranked the backstop. **Second, the Fed backstops liquidity, not solvency.** It will catch a market that is freezing for lack of buyers; it will not make whole the bondholders of a genuinely insolvent company whose business has failed. The put compresses *systemic* tail risk, not *idiosyncratic* default risk. **Third, the put is a legal and political act.** The 2020 corporate facilities required the Treasury to provide a capital cushion under the CARES Act and were controversial; the Fed cannot simply buy corporate bonds whenever it likes without backing and authorization. Knowing where the put stops is as important as knowing it exists — leaning on a backstop the Fed cannot legally or politically deploy is how investors get hurt.

There is also a darker structural consequence. Because a credible put compresses spreads in calm times, it lowers the cost of borrowing for risky companies and rewards leverage. Cheap credit invites more debt, weaker covenants, and riskier lending — the very fragility the put later has to rescue. Each rescue strengthens the belief in the next one, which compresses spreads further, which encourages still more leverage. This is the moral-hazard critique of the Fed put: it is not a free lunch but a slow accumulation of system risk, paid for in the form of artificially tight spreads today and a larger backstop required tomorrow. The spread you observe in a calm market is therefore partly a *fundamental* price of credit risk and partly an *artifact* of the rescue everyone expects in the next crisis — and you cannot fully separate the two by looking at the number alone.

## Common misconceptions

**"Wider spreads just mean more defaults are coming."** Not exactly. Spreads are compensation for *expected* losses plus a risk and liquidity premium that swings with sentiment. In a panic, spreads widen far more than realized defaults end up justifying — March 2020 priced a default wave that the policy response prevented, so investors who bought the panic were paid handsomely. Spreads overshoot in both directions; they are a fear gauge, not a default forecast you can read off mechanically.

**"The risk-free rate and the spread move together, so it's all one thing."** They are distinct and often move *opposite*. In a growth scare the Fed cuts (the risk-free rate falls) while credit spreads widen (fear rises) — so a corporate bond's total yield can be roughly unchanged even as its spread blows out. Treasuries rally as credit sells off. Confusing the two leads you to misread what the bond market is saying.

**"IG bonds are safe because the companies rarely default."** True about default, dangerously wrong about risk. As the duration example showed, a 200 bp spread widening can knock 13% off a long IG bond with zero defaults. The risk most IG investors actually run is *spread mark-to-market*, not credit losses — and that risk is highly correlated across the whole market, so it cannot be diversified away.

**"The Fed put means credit can't lose."** The put compresses spreads in normal times and caps the worst tail, but it is not a guarantee and it has limits. The Fed will not backstop a *solvency* problem caused by genuinely bad fundamentals, only a *liquidity* problem caused by a market freeze. And the put can be in conflict with the inflation mandate: in 2022 the Fed was *hiking into* falling markets because inflation tied its hands, and spreads widened with no rescue. The put is real but conditional — it depends on inflation being low enough to let the Fed ease.

**"A small Fed program can't move a multi-trillion-dollar market."** The 2020 facilities bought ~\$14 billion and moved a \$10-trillion market by hundreds of basis points. The size of the checkbook is almost irrelevant; the *credibility of the promise* and the announcement effect do the work. This is the single most counterintuitive fact about the Fed put.

## Case studies: how policy moved spreads in five regimes

Theory is cheap. Here is the spread doing what the mechanism predicts, across five real, dated regimes. The bar chart below puts the peak spreads side by side so the magnitudes are concrete before we walk through each one.

![Peak investment-grade and high-yield spreads across five regimes, from calm to the 2008 crisis](/imgs/blogs/how-policy-moves-credit-spreads-and-the-fed-put-8.png)

### 2008: the global financial crisis — what happens with no backstop, then a partial one

The 2008 crisis is the control case: a credit panic in which the modern corporate-bond backstop did not yet exist. As Lehman Brothers failed in September 2008 and the financial system's plumbing froze, the price of corporate risk went vertical. High-yield OAS reached **1,971 basis points** by December 2008 — nearly 20 percentage points over Treasuries, meaning the market was pricing a default catastrophe. Investment-grade spreads hit **555 basis points**, an extraordinary level for the safest companies in America. Even blue-chip firms could barely borrow.

The policy response was historic but aimed mostly at *banks*, not corporate bonds directly: the Fed cut rates to zero, launched QE1 to buy Treasuries and mortgage-backed securities, and the Treasury recapitalized the banking system through TARP. There was no PMCCF/SMCCF for corporate bonds — that tool was invented in 2020. Spreads compressed over 2009 as the bank backstop stabilized the system and QE's liquidity tide reached credit indirectly, but the path down was slow and the peak was the widest in modern history.

The contrast with 2020 is the whole point. In 2008 the backstop reached credit *indirectly* — by saving the banks and flooding the system with liquidity — and so it worked slowly: spreads took the better part of a year to normalize from their December peak. In 2020 the Fed went *directly* at the corporate-bond market, and spreads turned in a single day. The difference was not the size of the intervention but its *aim*: a backstop pointed straight at the frozen market repriced it instantly, while a backstop pointed at the banking system trickled through over months. The lesson policymakers took from 2008 — that you can stop a credit cascade if you intervene fast and aim directly at the seizing market — is exactly what they applied with terrifying speed in 2020. The 2008 peak of 1,971 bp in high-yield remains the modern record precisely because that was the last major crisis fought *without* a direct corporate-bond backstop in the toolkit. Every crisis since has had a smaller credit peak, not because the shocks were milder but because the put got more direct.

### March 2020: the COVID backstop — the Fed put made explicit

We opened with this one because it is the cleanest experiment in the whole literature. In about a month, the COVID shock drove IG spreads from ~100 bp to **373 bp** and HY from ~360 bp to **1,087 bp**. The S&P 500 fell 33.9% from its February peak to the March 23 low — and that low was the *same day* the Fed announced the corporate-bond facilities. Equities and credit bottomed together, on the announcement.

![The 2020 backstop turned spreads on the announcement day, months before the first purchase settled](/imgs/blogs/how-policy-moves-credit-spreads-and-the-fed-put-7.png)

The timeline above is the heart of the case. On March 23 the Fed announced PMCCF (to buy newly issued bonds) and SMCCF (to buy existing bonds and bond ETFs). Spreads turned that day. The first actual SMCCF purchase did not settle until May 12. By the time the facilities wound down they had bought roughly \$14 billion — and IG spreads were back near 140 bp, HY back toward 400 bp, and companies had issued a *record* amount of new investment-grade debt in 2020 because borrowing was suddenly cheap again. The announcement effect, the worked example above, and the Fed put are all the same story told here: a promise priced the market before the checkbook opened. Combined with the CARES Act's ~\$2.2 trillion of fiscal support and unlimited Treasury/MBS QE, this monetary-plus-fiscal bazooka turned a 33.9% crash into a V-shaped recovery in which the S&P doubled off the low by 2021.

![Investment-grade and high-yield spreads peak on the March 23 2020 announcement and tighten for months](/imgs/blogs/how-policy-moves-credit-spreads-and-the-fed-put-4.png)

Zooming into the daily path makes the timing undeniable. The chart above traces IG and HY spreads from February through June 2020. Both lines climb almost vertically into March 23 — IG to 373 bp, HY to 1,087 bp — and that day is the peak. From the announcement onward the lines bend down and keep falling for months, through a period in which the Fed bought essentially nothing. There is no second leg up when the actual purchases begin in May; the repricing is already done. If you wanted a single picture to explain why central bankers obsess over *credibility* rather than *firepower*, this is it: the market did the Fed's buying for it the moment it believed the Fed would.

One subtlety worth dwelling on: the facilities were deliberately designed to *never need to be used much*. By announcing a credible buyer of last resort, the Fed removed the worst-case fear that was driving the illiquidity premium sky-high. Private investors, no longer terrified of being unable to sell, came back as buyers — and their buying compressed spreads before the Fed's program was operationally ready. The \$14 billion the Fed eventually spent was almost a formality to prove the promise was real. This is the announcement effect functioning exactly as intended: the threat of intervention substitutes for the intervention itself.

### 2022: the hiking cycle — the put in abeyance

2022 is the case that proves the Fed put is *conditional*. Inflation had surged to 9%, and the Fed responded with the fastest hiking cycle since Volcker: 525 basis points in sixteen months, eleven hikes, the upper bound of the funds rate going from 0.25% to 5.50%. This time the Fed was not catching falling markets — it was *causing* the fall, deliberately, to break inflation. There was no put. Credit had to reprice for higher rates and recession risk on its own.

High-yield spreads widened to **552 basis points** by October 2022, and IG to **165**. These are recession-grade levels, though nowhere near 2008 or 2020 — partly because the *labor market* stayed strong, so default fundamentals never deteriorated as much as a 552 bp spread might imply. The deeper damage of 2022 was the refinancing wall it built: 525 bp of hikes meant that every bond issued at pandemic lows would eventually refinance into a far more expensive world, loading credit risk into 2024–2026. And 2022 was the year stocks *and* bonds fell together (the S&P −19.4%, the Agg bond index −13.0%, the classic 60/40 portfolio −16.0%) — because the rising discount rate hit every asset at once. In March 2023 the strain surfaced as regional-bank stress: Silicon Valley Bank collapsed on March 10, 2023, having taken duration losses on its bond portfolio exactly as the duration math above predicts. The connection from policy rates to bank solvency runs straight through the spread-and-duration mechanism this post describes; for the regulatory response, see [macroprudential and regulatory policy](/blog/trading/policy-and-markets/macroprudential-and-regulatory-policy).

### 2023 regional-bank stress and the 2025 tariff shock — the modern canary

Two recent episodes show the canary working in real time. The March 2023 regional-bank failures (SVB, Signature, then First Republic) caused a sharp, brief spread widening as the market worried the stress would spread to credit broadly. It did not — the Fed's emergency lending facility (the Bank Term Funding Program) and deposit guarantees walled it off — and spreads quickly settled back. That is the put working in a *targeted* way: a backstop sized to the specific problem, which contained it.

The 2025 tariff shock is the newest case and a textbook spread move. When the "Liberation Day" tariffs were announced on April 2, 2025 — a 10% universal tariff plus steep "reciprocal" rates on dozens of partners — the market repriced the risk of a trade-war recession in days. The S&P 500 fell about 12% in four sessions and 18.9% from its February peak to the April 8 trough. Credit followed the canary script: IG OAS widened from ~83 bp at end-2024 to **121 bp** in April, and HY from ~292 bp to **461 bp**. Then, on April 9, the administration announced a 90-day pause on most reciprocal tariffs, the S&P ripped 9.5% in a single day, and spreads began to compress back — IG toward 85 bp and HY toward 300 bp by year-end as deals were struck (including Vietnam's 46% rate negotiated down to 20% in July). Note that this widening, painful as it was, peaked at 121/461 bp — far short of 2020 or 2008. The market priced a policy *scare*, not a solvency crisis, and the spread told you so. For how tariffs reprice assets through multiple channels, see [tariffs and trade policy as a market force](/blog/trading/policy-and-markets/tariffs-and-trade-policy-as-a-market-force).

### A European footnote: Draghi's "whatever it takes" — a backstop that was never used

The single cleanest proof that a *promise* compresses spreads comes not from corporate credit but from sovereign credit, in Europe, in 2012. The euro-zone debt crisis had pushed the borrowing costs of Spain and Italy to levels that looked terminal: Spain's 10-year yield hit 7.6% in July 2012 and Italy's 6.6% — spreads over German bunds wide enough that the market was pricing a break-up of the euro. These are sovereign spreads, not corporate, but the mechanism is identical: compensation for the risk that the borrower defaults or that the currency you're repaid in collapses.

On July 26, 2012, European Central Bank president Mario Draghi said the ECB would do "whatever it takes to preserve the euro — and believe me, it will be enough." In September the ECB unveiled Outright Monetary Transactions (OMT), a program to buy the bonds of stressed governments without a preset limit. Spanish and Italian spreads began collapsing immediately, and Spain's 10-year yield fell to 1.6% by 2014. The remarkable part: **OMT was never used. Not a single bond was ever bought under it.** Three words and a credible, never-deployed backstop did the entire job. It is the purest possible demonstration of the announcement effect and the put: credibility, not bond-buying, did the work. The Fed studied this template carefully before 2020 — and when it announced the corporate facilities, it was applying Draghi's lesson to American credit.

### The five regimes in one sentence

The regimes together tell one story: spreads widen in rough proportion to the *severity and policy-treatability* of the shock. A trade scare with a fast policy off-ramp peaks near 460 bp in HY; a deliberate hiking cycle near 550; a liquidity freeze with a backstop near 1,090; an unbackstopped solvency crisis near 1,970. The number on the screen is the market's estimate of how bad it is and how much help is coming.

## What it means for asset values: the playbook

Pull the threads together into something you can act on.

**Which assets reprice, and in what direction.** When the central bank eases (cuts, QE, or signals a backstop), the risk-free floor falls *and* the spread compresses through reach-for-yield and the put — a double tailwind for corporate bonds, with high-yield and long-duration bonds gaining the most. When the central bank tightens (hikes, QT, or steps back from the put), the floor rises and spreads widen through the refinancing wall and re-priced tail risk — a double headwind, worst for the longest-duration and lowest-rated credit. The magnitude is governed by the duration math: a bond's price moves roughly its duration times the total yield change, so a long bond in a big regime shift moves double digits.

**The signal to watch.** High-yield OAS is the single best real-time fear gauge in the market — cheaper and faster than waiting for earnings or economic data. Watch the *level* against history (sub-350 bp HY is complacent, 500+ is stress, 800+ is crisis) and the *rate of change* (a fast widening while equities are still high is the canary singing). Watch the *IG–HY relationship*: when HY widens much faster than IG, the market is worried specifically about the weak, leveraged tail — a more ominous signal than a uniform widening. And watch the *maturity wall*: a hiking cycle's damage shows up two to three years later as cheap debt refinances expensive.

**What would invalidate the read.** The Fed put is conditional on the inflation mandate. The whole "central bank will catch credit" thesis breaks when inflation is high enough to force the Fed to hike *into* a falling market, as in 2022 — then spreads can widen with no rescue. So the master switch is inflation: a low-inflation world is a world where the put is live and spreads sit artificially tight; a high-inflation world is a world where the put is suspended and credit must stand on its own fundamentals. Before leaning on the Fed put, check whether the Fed is even *able* to use it.

**The structural risk.** Because the put compresses spreads in calm times, it also encourages more leverage and riskier lending — the very fragility it later has to rescue. Each rescue makes the next backstop more expected, which compresses spreads further, which builds more leverage. This is the moral-hazard critique of the Fed put, and it is why every intervention is also a future liability. The spread you see in calm times is partly an artifact of the rescue you expect in the next crisis.

Put it all together and the discipline is simple to state, if hard to practice. A credit spread is the single cleanest price of fear and of policy belief in the market: it stacks a default premium and an illiquidity premium on a Treasury floor, it widens before equity cracks, it falls when the central bank promises a floor, and it reprices every corporate bond and every refinancing along the way. To read it, ask three questions in order. *Where is the risk-free floor going* — is the Fed hiking or cutting? *Is the put live* — is inflation low enough that the Fed can ease if credit seizes? *And where is the maturity wall* — when does cheap debt roll into the current rate? Answer those, and the spread stops being a mysterious number on a screen and becomes a legible readout of the cost of corporate money, set by policy, and the belief that policy will catch it when it falls.

For how this channel sits alongside the others — the discount rate on equities, the currency, the liquidity tide — start from [how policy sets asset prices: the transmission map](/blog/trading/policy-and-markets/how-policy-sets-asset-prices-the-transmission-map), and for the front-end rate path that anchors the whole risk-free floor, see [how policy sets the bond market: the yield curve](/blog/trading/policy-and-markets/how-policy-sets-the-bond-market-the-yield-curve).

## Further reading & cross-links

Within this series:

- [How policy sets asset prices: the transmission map](/blog/trading/policy-and-markets/how-policy-sets-asset-prices-the-transmission-map) — the master picture of every lever and channel.
- [Macroprudential and regulatory policy](/blog/trading/policy-and-markets/macroprudential-and-regulatory-policy) — how bank rules and supervision shape credit supply and contain stress (the 2023 regional-bank case).
- [How policy sets the bond market: the yield curve](/blog/trading/policy-and-markets/how-policy-sets-the-bond-market-the-yield-curve) — the risk-free floor that every spread stacks on.
- [The liquidity channel: QE, QT, and the everything bid](/blog/trading/policy-and-markets/the-liquidity-channel-qe-qt-and-the-everything-bid) — the balance-sheet lever that bids up credit.
- [The expectations channel: forward guidance and credibility](/blog/trading/policy-and-markets/the-expectations-channel-forward-guidance-and-credibility) — why a promise reprices the asset before the action.
- [Forward guidance and the Fed put](/blog/trading/policy-and-markets/forward-guidance-and-the-fed-put) — the credibility mechanism behind the backstop.
- [Tariffs and trade policy as a market force](/blog/trading/policy-and-markets/tariffs-and-trade-policy-as-a-market-force) — the 2025 shock that widened spreads.

Elsewhere on the site:

- [Credit spreads: the risk correlation and the canary](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary) — the statistical betas and lead-lag with equities.
- [QE versus QT: how balance-sheet policy moves markets](/blog/trading/macro-trading/qe-vs-qt-how-balance-sheet-policy-moves-markets) — the trader's-lens version of the liquidity channel.
- [Central-bank toolkit: rates, QE, QT, forward guidance](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) — the full set of levers behind the put.
- [Quantitative easing explained](/blog/trading/finance/quantitative-easing-explained-printing-money) — the from-scratch primer on the balance-sheet lever.

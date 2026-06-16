---
title: "Credit spreads: pricing the probability of default"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into the single number that prices corporate credit risk — the spread over Treasuries — and how the market backs an implied probability of default out of it using the credit triangle (spread ≈ default probability × loss)."
tags: ["fixed-income", "bonds", "credit-spread", "credit-risk", "default-probability", "loss-given-default", "spread-duration", "corporate-bonds", "high-yield", "us-treasuries"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — a credit spread is the extra yield a risky bond pays over a safe Treasury of the same maturity, and it is the market's price tag for the chance you don't get paid back. This post shows you how to read that number, back an implied default probability out of it, and trade the risk it measures.
> - A **credit spread** is a corporate's yield minus the matching-maturity Treasury yield. Our running example, **Northwind Corp's 10-year** bond, yields **6%** while the 10-year Treasury yields **4%**, so its spread is **200 basis points** — 2 full percentage points of extra income for taking on default risk.
> - The **credit triangle** ties three numbers together with one rough identity: **spread ≈ probability of default × loss given default**. It lets you turn any one of the three into the other two.
> - Backing it out: with a 200 bp spread and a **40% loss given default** (so a **60% recovery**), the market is implying roughly a **3.3% annual chance** Northwind defaults — about a **28% chance over the full 10 years**.
> - **Spread duration** is the credit twin of interest-rate DV01: it measures how much the price moves for a **1 bp change in spread**. Northwind's ~7.8-year spread duration means a 100 bp spread widening costs about **\$78 per \$1,000 bond**.
> - The spread you *earn* is not the return you *keep*: subtract expected default losses (~80 bp here) and you're left with the **excess return**, the compensation for bearing risk you actually realize on average.
> - Spreads are driven by two forces at once — **default expectations** (will they pay?) and **risk appetite plus liquidity** (how much does the market demand to hold the risk?) — which is why spreads blow out in a panic even when no one has defaulted yet.

Here is a question that sounds simple and turns out to be the entire discipline of credit investing: a U.S. Treasury bond and a corporate bond from a company called Northwind both mature in ten years, both promise to pay you back \$1,000, and both pay a coupon twice a year. Why does the Northwind bond yield 6% when the Treasury yields only 4%? They look like the same instrument. They are priced two full percentage points apart. What is that gap *paying you for*, and is it enough?

The answer is the subject of this post. That 2-percentage-point gap is called the **credit spread**, and it is one of the most information-dense numbers in all of finance. It is, quite literally, the market's live, continuously-updated price for the chance that Northwind doesn't pay you back. Pull it apart and you find an implied probability of default, an assumption about how much you'd lose if that default happened, a charge for the risk that the spread itself moves against you, and a premium for the simple fact that nervous markets pay up for safety. Learn to read a spread and you can look at a single number on a screen and reverse-engineer what the entire market believes about a company's odds of survival.

![A decomposition diagram showing a corporate bond yield of six percent split into a four percent Treasury yield base and a two hundred basis point credit spread stacked on top](/imgs/blogs/credit-spreads-pricing-the-probability-of-default-1.png)

The diagram above is the mental model for the whole post: a corporate bond's yield is just a **safe rate plus a risky premium**. The safe rate is what you'd earn lending to the government — the [risk-free benchmark](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration) — and the premium on top is the spread, the part you earn *only* for taking credit risk. Everything that follows is an attempt to answer one question about that top slice: **is it big enough to compensate you for what can go wrong?** (Everything here is educational, not investment advice; the goal is to understand the machinery, not to recommend any bond.)

## Foundations: the words you need before we price anything

Let's build the vocabulary from zero. If you've read the earlier posts in this series, treat this as a quick refresher. If not, don't skip it — every later sentence leans on these terms, and credit has its own dialect.

A **bond** is a tradable loan. You, the buyer, are the lender; the **issuer** is the borrower. The bond promises a fixed stream of cash: a periodic **coupon** (the interest) and the **face value**, also called **par** — almost always \$1,000 per bond — returned at **maturity**, the date of the final payment. A **U.S. Treasury** is a bond issued by the U.S. federal government; because the government can tax and print its own currency, Treasuries are treated as the closest thing to **risk-free** — a loan that will, essentially for certain, be repaid. A **corporate bond** is a loan to a company, which most assuredly can fail to pay. That difference is the entire game.

A few units you'll see constantly:

- A **basis point** (abbreviated *bp*, pronounced "bip") is **one hundredth of a percent**: 0.01%. A 2-percentage-point gap is **200 bps**. Spreads are quoted in basis points because they move in small increments and "175 bps" is cleaner than "one and three-quarters percent."
- The **yield** — more precisely the **yield to maturity (YTM)** — is the single interest rate that makes a bond's future cash flows, discounted back to today, equal its current market price. It is the bond's all-in return if you buy now and hold to maturity, assuming every promised payment actually arrives. When we say "Northwind yields 6%," we mean its YTM is 6%.
- **Default** is the event credit is all about: the issuer fails to make a promised payment, or otherwise breaks the terms of the bond (the *indenture*). Default does *not* usually mean you lose everything — it means the company can't pay in full and you enter a process to recover what you can.

Now the four numbers that the rest of the post is built from. Memorize these four names; they are the alphabet of credit:

- **Credit spread (S)** — the extra yield a risky bond pays over a risk-free bond of the *same maturity*. It is the corporate yield minus the matching Treasury yield. This is the number on the screen, the one the market quotes and trades.
- **Probability of default (PD)** — the chance, over some horizon, that the issuer defaults. We'll usually talk about the **annual** PD (the chance it defaults this year, given it hasn't yet) and the **cumulative** PD (the chance it defaults at some point over the bond's life).
- **Loss given default (LGD)** — if a default happens, the fraction of your money you *don't* get back. Its mirror image is the **recovery rate (R)**: the fraction you *do* recover, so `LGD = 1 − R`. If you recover 40 cents on the dollar, your recovery rate is 40% and your LGD is 60%.
- **Expected loss (EL)** — the average loss you should *expect* per year from holding the bond, blending the two: `EL ≈ PD × LGD`. This is the number the spread is, in part, paying you to bear.

The whole post is about the relationships among these four. Hold one fact above all others: *the spread is a price, not a fact.* PD and LGD are real things about the world that we can only estimate; the spread is what the market is willing to pay to take the risk, today, right now, given everyone's collective fear and greed. Most of the art of credit is in the gap between the two — between the loss the spread implies and the loss you actually expect.

### The single relationship that does all the work

Everything in this post is a consequence of one rough identity, the **credit triangle**:

$$S \approx PD \times LGD$$

- $S$ is the **credit spread**, expressed as a decimal (a 200 bp spread is 0.02).
- $PD$ is the **annual probability of default** (also a decimal — 3% is 0.03).
- $LGD$ is the **loss given default** (a decimal — a 40% loss is 0.40).

Read it in plain English: *the extra yield you earn each year should, on average, just cover the loss you expect each year.* If a bond has a 3% chance of defaulting this year and you'd lose 40% in that event, your expected annual loss is `0.03 × 0.40 = 0.012`, or 1.2% — so the bond "should" pay you about 120 bps of extra yield just to break even on the risk. That, in one line, is why risky bonds yield more than safe ones: the spread is the toll for expected losses.

This identity is *rough* on purpose. It ignores compounding, it treats one year at a time, and — crucially — it pretends the only thing the spread pays for is expected loss. We'll see later that real spreads are wider than the triangle says, and the gap is itself one of the most important quantities in markets. But as a first lens, the triangle is the single most useful thing you can know about credit. Everything below either applies it or corrects it.

## What a credit spread actually is

Start with the picture. Take any corporate bond. Find a Treasury with the same maturity. Subtract the two yields. The number you get is the spread, and it is the bond's compensation for everything that makes it riskier than the government's debt.

The reason we subtract a *matching-maturity* Treasury is to strip out the part of the yield that has nothing to do with credit. A 10-year corporate bond and a 10-year Treasury both carry **interest-rate risk** — if rates rise, both fall in price — and both reflect the market's view of where rates will be over the next decade. By subtracting the Treasury, we cancel all of that out. What's left is the *pure credit* piece: the part of Northwind's yield that exists only because Northwind, unlike the U.S. government, might not pay.

This is why the spread, not the yield, is the credit investor's native unit. The yield on Northwind's bond can rise 100 bps tomorrow for two completely different reasons. Maybe the Treasury rose 100 bps because the Fed signaled higher rates — in which case Northwind's *credit* didn't change at all, and the spread is unchanged. Or maybe the Treasury was flat and Northwind's yield rose because the market suddenly doubts the company — in which case the spread widened 100 bps and that *is* a credit event. The yield mixes the two stories together; the spread isolates the one a credit analyst cares about. (The same logic drives the broader [investment-grade vs high-yield spread complex](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads) that allocators watch.)

#### Worked example: computing Northwind's spread

Let's pin down the running example. **Northwind Corp** has a 10-year bond outstanding. It currently yields **6.00%**. The 10-year U.S. Treasury currently yields **4.00%**. The spread is the difference:

$$S = 6.00\% - 4.00\% = 2.00\% = 200 \text{ bps}$$

So Northwind pays you **200 basis points** — \$20 per year per \$1,000 of face value, on top of what the Treasury pays — for the privilege of lending to a company instead of the government. On a \$100,000 position, that's \$2,000 a year of extra income. The question the rest of this post answers is whether \$2,000 a year is fair pay for the risk that Northwind blows up and you lose a chunk of your \$100,000.

A subtlety worth flagging now: there are several flavors of "spread," and they differ in exactly which Treasury you subtract. The simplest, which we're using, is the **nominal spread** — corporate YTM minus the YTM of a single benchmark Treasury. Desks more often quote the **G-spread** (over an *interpolated* point on the Treasury curve, matched to the bond's exact maturity) or the **Z-spread** (the constant amount you'd add to *every* point on the [zero curve](/blog/trading/fixed-income/spot-rates-the-zero-curve-and-bootstrapping) to reprice the bond), and high-yield desks live in **spread-to-worst** and the **option-adjusted spread (OAS)**, which strips out the value of any embedded call option. For building intuition, the simple nominal spread is perfect; just know that the professional versions are refinements of the same idea — *how much extra yield, over the risk-free curve, is this bond paying?*

*The intuition: a spread is what's left of a corporate's yield after you subtract away everything that isn't about credit.*

## The credit triangle: spread, default, and loss

Now we turn the spread into a story about default. The credit triangle, `S ≈ PD × LGD`, is the bridge. It says the spread is the market's way of charging you for expected loss, and expected loss is two things multiplied: how *often* you lose (PD) and how *much* you lose when you do (LGD).

![A three way relationship diagram showing spread equals probability of default times loss given default with arrows connecting the three quantities and a worked Northwind example](/imgs/blogs/credit-spreads-pricing-the-probability-of-default-2.png)

The figure above shows the triangle as it really works: any two of the three numbers determine the third. Quote a spread and assume a recovery rate, and you can solve for the implied default probability — that's what the market does. Estimate a default probability and a recovery, and you can compute the spread a bond "should" trade at — that's what a credit analyst does to decide whether the market's price is rich or cheap. The triangle is a two-way street, and which direction you drive it depends on whether you're reading the market or arguing with it.

Why does the identity hold at all? Think about what it takes for the bond to be *fairly priced*. You buy Northwind's bond and earn 200 bps of extra yield each year. But each year there's some chance Northwind defaults and you lose a fraction of your money. For the bond to be fair — for the extra yield to exactly offset the expected losses — the spread you collect must equal the loss you expect:

$$\underbrace{S}_{\text{extra yield collected}} \approx \underbrace{PD \times LGD}_{\text{expected loss suffered}}$$

If the spread were *bigger* than `PD × LGD`, the bond would be a free lunch: you'd collect more than you expect to lose, and buyers would pile in until the price rose and the spread fell back to fair. If the spread were *smaller*, you'd be paid less than you expect to lose, and sellers would dump it until the spread widened. In a frictionless world with risk-neutral investors, the spread settles exactly at expected loss. (The real world is not frictionless and investors are not risk-neutral — which is the whole reason real spreads sit *above* this fair line, as we'll see.)

#### Worked example: the triangle in both directions

Let's drive the triangle both ways with Northwind's numbers and a recovery assumption.

**Direction 1 — from default to spread.** Suppose a credit analyst studies Northwind and concludes it has roughly a **3% annual chance of default**, and that if it defaults, bondholders recover about **60 cents on the dollar** (so LGD = 40%). The fair spread is:

$$S \approx PD \times LGD = 0.03 \times 0.40 = 0.012 = 120 \text{ bps}$$

The analyst's model says Northwind *should* trade around a 120 bp spread.

**Direction 2 — from spread to default.** But the market is actually pricing Northwind at a **200 bp** spread. Holding the 40% LGD fixed, what default probability does *that* imply?

$$PD \approx \frac{S}{LGD} = \frac{0.02}{0.40} = 0.05 = 5\%$$

The market is implying a **5% annual default probability** — almost twice the analyst's 3% estimate. Either the market knows something the analyst doesn't, or the analyst has found a bond that's cheap relative to its true risk. That gap — 200 bps priced versus 120 bps "fair" — is exactly where credit investors hunt for opportunity (and where they get run over).

*The intuition: the triangle lets you translate freely between a spread you can see and a default probability you can't — and the disagreement between the two is the trade.*

## Backing out the implied default probability

The triangle's most-used direction is reading a spread *backwards* into a default probability. This is the credit market's party trick: you don't need a model of the company, you don't need its financials — you need the spread and one assumption (the recovery rate), and you can read the market's implied odds of default straight off the screen.

![A flow diagram showing how an observed two hundred basis point spread divided by a forty percent loss given default yields a three point three percent annual default probability and then compounds to a twenty eight percent cumulative probability over ten years](/imgs/blogs/credit-spreads-pricing-the-probability-of-default-4.png)

The figure above lays out the pipeline. Start with the **observed spread** (200 bps). Divide by the assumed **LGD** to get the **annual default probability**. Then compound that annual probability over the bond's life to get the **cumulative** probability — the chance the issuer defaults at *some* point before maturity. Each step is one division or one compounding; none of it requires anything fancy.

There's a small but important refinement in the first step. The crude version is `PD ≈ S / LGD`. The slightly better version recognizes that the spread compensates you for losses on the bond's value, and that the survival-versus-default arithmetic is multiplicative, giving the cleaner approximation:

$$PD_{\text{annual}} \approx \frac{S}{LGD}$$

for small numbers, which is what we'll use. (The fully rigorous version solves `S = PD × LGD / (1 − PD)` or uses continuous-time hazard rates, but for a beginner the simple division is within a hair of the right answer and teaches the same lesson.)

#### Worked example: Northwind's implied default probability

Let's do it carefully with the running numbers. Northwind's spread is **200 bps** (S = 0.02). Assume a **40% LGD** (R = 60%). The implied **annual** default probability is:

$$PD_{\text{annual}} \approx \frac{S}{LGD} = \frac{0.02}{0.40} = 0.05 = 5\%$$

Wait — earlier I quoted 3.3% in the TL;DR, and here the simple division gives 5%. Both are "right"; they differ because of how you treat the spread. The crude `S/LGD` gives 5%. But part of the 200 bp spread is **not** about default at all — it's a risk and liquidity premium (more on this soon). If we say roughly two-thirds of the spread, ~133 bps, reflects genuine expected loss and the rest is premium, then the loss-implied annual PD is `0.0133 / 0.40 ≈ 3.3%`. The honest statement is a range: *the market is implying somewhere between a 3% and 5% annual chance Northwind defaults, depending on how much of the spread you attribute to pure default risk versus risk premium.* Pin down the recovery assumption and the premium split, and you pin down the number. We'll use **3.3% annual** as the "expected-loss" estimate for the rest of the post and treat 5% as the "all-in, risk-neutral" upper bound.

Now compound to the full 10 years. If Northwind survives each year with probability `1 − 0.033 = 0.967`, the chance it survives all ten years is:

$$P_{\text{survive 10y}} = 0.967^{10} \approx 0.713$$

So the **cumulative** probability of default over the decade is:

$$P_{\text{default by 10y}} = 1 - 0.713 \approx 0.287 \approx 28\%$$

Read that back in plain English: at a 200 bp spread and 40% LGD, the market is pricing **roughly a 28% chance that Northwind defaults at some point in the next ten years** — better than even odds it survives, but a very real one-in-four-ish chance it doesn't. That single sentence is what a credit trader extracts from one number on a screen.

*The intuition: divide the spread by the loss rate and you get the market's annual default odds; compound those odds over the bond's life and you get its lifetime odds of failure.*

#### Worked example: how recovery assumptions move the implied PD

The recovery rate is the soft spot in this whole calculation — it's an assumption, and the implied PD is extremely sensitive to it. Hold the spread fixed at 200 bps and watch the implied annual PD swing as we change the recovery assumption:

| Recovery rate (R) | LGD (1 − R) | Implied annual PD (S / LGD) |
|---|---|---|
| 80% | 20% | 0.02 / 0.20 = **10%** |
| 60% | 40% | 0.02 / 0.40 = **5%** |
| 40% | 60% | 0.02 / 0.60 = **3.3%** |
| 20% | 80% | 0.02 / 0.80 = **2.5%** |

Look at what happens. The *same* 200 bp spread implies a 10% annual default probability if you assume a high 80% recovery, but only a 2.5% probability if you assume a brutal 20% recovery. Why? Because a high recovery means each default costs you little, so the same spread must be paying for *many* defaults; a low recovery means each default is devastating, so the same spread only needs to cover a *few*. This is why two analysts can stare at the identical spread and walk away with default estimates that differ by a factor of four — they've made different recovery assumptions. Senior bonds with collateral recover more (higher R, so the spread implies a higher PD); junior unsecured bonds recover little. The historical average for senior unsecured corporate bonds is around 40%, which is why R = 40% is the market's default assumption — and why we use it.

*The intuition: you cannot read a default probability out of a spread without committing to a recovery rate — and the answer can swing fourfold depending on which one you pick.*

## Spread by credit rating: the ladder of risk

Spreads aren't random — they line up almost monotonically with **credit ratings**, the letter grades that agencies like Moody's, S&P, and Fitch assign to issuers (the subject of the [next post in this series](/blog/trading/fixed-income/bond-ratings-how-moodys-sp-and-fitch-grade-debt)). The better the rating, the lower the spread, because a higher grade means a lower assessed probability of default. Walk down the ladder from AAA to CCC and the spread widens at an accelerating pace.

![A bar chart showing typical credit spreads in basis points widening down the rating scale from triple A near forty basis points to triple C above one thousand basis points](/imgs/blogs/credit-spreads-pricing-the-probability-of-default-5.png)

The figure above shows the rough shape of the spread ladder in normal markets — these are illustrative, mid-cycle levels, not live quotes, and they move enormously through the cycle. Two features matter more than the exact numbers. First, the **investment-grade** tier (AAA down to BBB) lives in a narrow band, from a few tens of basis points up to maybe 150–200 bps; these are companies the market thinks will almost certainly pay. Second, the moment you cross into **high yield** (BB and below — the "junk" tier), spreads jump and then explode: BB might be 250 bps, B might be 450 bps, and CCC can be 1,000 bps or more, because at that point the market is pricing a serious, double-digit annual chance of default. The line between BBB and BB — the [investment-grade/high-yield divide](/blog/trading/fixed-income/investment-grade-vs-high-yield-the-great-divide) — is the most consequential boundary in all of credit, because many institutions are *forbidden* from holding bonds below it.

The non-linearity is the lesson. Spreads don't widen in equal steps as ratings fall; they widen *faster and faster*. That's because default probability itself is convex in credit quality: the gap in real-world default risk between AAA and AA is tiny, but the gap between B and CCC is enormous. The spread ladder is steep at the bottom for the same reason the cliff is steep at the bottom.

There's a second, quieter feature worth naming: spreads on the *same* rating are not constant — they breathe with the cycle. A BBB bond might trade at 120 bps in a calm, late-cycle market and 400 bps in a recession, even though its rating, and therefore its assessed default probability, hasn't changed at all. The rating tells you *where on the ladder* a bond sits; the cycle tells you *how tall the whole ladder is* on any given day. This is why credit investors talk about "spread levels" the way equity investors talk about valuations — a 150 bp BBB spread is cheap relative to its own history at some moments and expensive at others, and the same letter grade can be a bargain or a trap depending on where the market is in its appetite for risk.

#### Worked example: reading default odds off the rating ladder

Let's run the triangle across the ladder, holding LGD at 40%, to see what default probability each rung's spread implies:

| Rating | Typical spread | Implied annual PD (S / 0.40) | Rough 10y cumulative |
|---|---|---|---|
| AAA | 40 bps | 0.004 / 0.40 = **1.0%** | ~10% |
| A | 90 bps | 0.009 / 0.40 = **2.25%** | ~20% |
| BBB | 160 bps | 0.016 / 0.40 = **4.0%** | ~34% |
| BB | 300 bps | 0.030 / 0.40 = **7.5%** | ~54% |
| B | 500 bps | 0.050 / 0.40 = **12.5%** | ~74% |
| CCC | 1,100 bps | 0.110 / 0.40 = **27.5%** | ~95% |

Northwind, at a 200 bp spread, sits right around the BBB/BB border — a "crossover" credit, the borderline between investment grade and junk. The implied numbers tell the story starkly: a CCC bond's spread is pricing in a roughly 95% chance of default over ten years. That's not a typo. At the bottom of the ladder, the market is essentially betting the company *will* default; the only question is when, and how much you recover. (These are risk-neutral, spread-implied probabilities — they run higher than historical realized defaults precisely because of the risk premium baked into spreads. We'll return to that gap.)

*The intuition: the spread ladder is the rating ladder in dollars — and it steepens at the bottom because default risk itself accelerates as credit quality falls.*

## Spread duration: the dollars-per-basis-point of credit

So far we've treated the spread as a static number. But spreads *move*, and when they do, your bond's price moves with them. The sensitivity of a bond's price to a change in its spread is called **spread duration**, and it is the credit twin of the interest-rate sensitivity we built in the [modified duration and DV01 post](/blog/trading/fixed-income/modified-duration-and-dv01-measuring-and-trading-rate-risk).

The logic is identical to interest-rate duration. Recall the master relationship: for a small change in *yield*, a bond's percentage price change is (negative) modified duration times the yield change. A spread change *is* a yield change — it's a change in the credit piece of the yield — so the same formula applies, just with the spread as the moving part:

$$\frac{\Delta P}{P} \approx -D_{\text{spread}} \times \Delta S$$

- $\Delta P / P$ is the **percentage price change**.
- $D_{\text{spread}}$ is the **spread duration** (a pure number, very close to the bond's modified duration).
- $\Delta S$ is the **change in spread**, as a decimal (a 100 bp widening is 0.01).
- The **minus sign** says it all: when the spread *widens* (credit gets worse), the price *falls*.

For a bond with no embedded options, spread duration is essentially equal to its modified duration, because a basis point of spread change discounts the cash flows exactly like a basis point of yield change. So a 10-year bond with a modified duration around 7.8 also has a spread duration around 7.8. The practical upshot: **a 100 bp widening in spread costs you about 7.8% of the bond's value**, and the longer the bond, the more a given spread move hurts.

![A line chart showing spread duration rising with maturity so that a one hundred basis point spread widening costs more on longer bonds with the price impact growing from roughly two percent at two years to nearly eight percent at ten years](/imgs/blogs/credit-spreads-pricing-the-probability-of-default-6.png)

The figure above plots the price hit from a 100 bp spread widening against the bond's maturity. The relationship is roughly linear and unmistakable: the same credit event — a 100 bp spread blowout — barely scratches a 2-year bond (~2% price hit) but takes nearly 8% off a 10-year. This is why credit risk and maturity compound on each other: lending long to a risky borrower is doubly dangerous, because you're exposed to default for longer *and* your price is more sensitive to every wobble in the market's assessment of that risk along the way. It's also why credit investors who turn cautious don't always sell — sometimes they just *shorten*, rotating from 10-year into 2-year paper of the same issuer to cut spread duration while keeping the credit.

#### Worked example: a spread blowout on a Northwind position

Let's put a dollar figure on it. You own \$1,000,000 face of Northwind's 10-year bond, trading near par. Its spread duration is about **7.8**. One morning, bad earnings hit, and Northwind's spread widens from 200 bps to **300 bps** — a 100 bp move, `ΔS = 0.01`. The price change:

$$\frac{\Delta P}{P} \approx -7.8 \times 0.01 = -0.078 = -7.8\%$$

On a \$1,000,000 position, that's a loss of about **\$78,000** — and not a dollar of it required Northwind to actually default. The company is still paying its coupons; it's just that the market now demands more yield to hold its risk. This is the part beginners miss: *you can lose serious money on a credit bond without any default ever happening*, purely because the spread moved. Per \$1,000 bond, the hit is about \$78. Translate to the credit desk's working unit and you get the bond's **spread DV01** — the dollar loss per 1 bp of spread widening — which here is about `7.8% × \$1,000,000 / 100 = \$780` per basis point.

*The intuition: spread duration converts a move in the market's credit opinion into an immediate dollar gain or loss — no default required, and the longer the bond, the bigger the swing.*

There's a sharp wrinkle for bonds that *can* default, which interest-rate duration doesn't have to worry about. As a bond gets closer to default — as its spread blows out toward distressed levels — its price behavior stops looking like a normal bond's at all. A deeply distressed bond trades less on its spread duration and more on its **expected recovery**: at a 1,500 bp spread, the market has largely stopped treating it as a stream of coupons and started treating it as a claim on whatever the company is worth in bankruptcy. The price converges toward the recovery value (say, 40 cents on the dollar) almost regardless of small spread moves. So spread duration is a good guide for healthy, investment-grade credit, where spreads wiggle around modest levels, and a progressively worse guide as a bond slides into distress — at the extreme, the relevant question isn't "how much does the price move per basis point?" but "what do I recover when this defaults?" This is one more reason recovery sits at the center of credit: it's both the input to the triangle and the floor the price falls toward when things go wrong.

## The spread you earn versus the return you keep

Here is the most important and most misunderstood idea in credit: **the spread is gross pay, not take-home pay.** You collect the full spread as extra yield, but over time some of your bonds default, and those losses claw back part of what you earned. What you *keep*, on average, is the spread minus the expected loss — and that residual is what credit investing actually pays you for bearing risk.

We can write it as a simple subtraction:

$$\text{Excess return} \approx \underbrace{S}_{\text{spread earned}} - \underbrace{PD \times LGD}_{\text{expected loss}}$$

The first term is the gross spread you collect. The second is the credit triangle — the average loss you suffer to defaults. The difference is your **excess return** (also called the *carry net of losses*, or the *risk premium* you realize). If the spread exactly equaled expected loss, your excess return would be zero — you'd be earning nothing for the risk, just breaking even on average. The reason credit pays at all is that spreads sit *above* expected loss.

#### Worked example: Northwind's take-home spread

Northwind pays a **200 bp** spread. Using the expected-loss estimate of a **3.3% annual PD** and **40% LGD**, the expected annual loss is:

$$EL \approx PD \times LGD = 0.033 \times 0.40 = 0.0132 \approx 132 \text{ bps}$$

Hmm — but wait, let's use the realized/historical default rate for a BBB/BB crossover credit, which is closer to **2% annually**, not the spread-implied 3.3%. Then:

$$EL \approx 0.02 \times 0.40 = 0.008 = 80 \text{ bps}$$

So your excess return is:

$$\text{Excess return} \approx 200 \text{ bps} - 80 \text{ bps} = 120 \text{ bps}$$

You collect 200 bps; defaults eat roughly 80 bps a year on average; you *keep* about **120 bps** as compensation for bearing the risk. On a \$1,000,000 position, that's \$20,000 of gross spread income, minus ~\$8,000 of expected default losses, leaving ~\$12,000 of genuine excess return. That \$12,000 is not free money — it's payment for volatility (the spread will swing around, sometimes painfully) and for the chance you're *unlucky* and your particular bond is one of the defaulters. But on average, across many bonds and many years, it's the reward the credit market hands you for showing up.

This gap — between the spread-*implied* default rate (3.3–5%) and the historically-*realized* default rate (~2%) — is the famous **credit risk premium** or **credit spread puzzle**: realized losses have historically been much smaller than spreads imply, so credit has, on average, paid investors more than the defaults justify. The market charges for more than expected loss. Which brings us to the deepest question in this post.

*The intuition: the spread is your gross pay; subtract the defaults you'll actually suffer and what's left — the excess return — is what credit investing truly compensates you for.*

## What moves spreads: default fear and the price of risk

If spreads only paid for expected loss, they'd move only when default expectations changed. But spreads move violently — they double or triple in weeks — far more than any sane revision of default probabilities can explain. Something else is going on, and it's the second force inside every spread: the **price of risk** itself, driven by market-wide risk appetite and liquidity.

Decompose the spread into its two parts:

$$S = \underbrace{PD \times LGD}_{\text{expected loss}} + \underbrace{\text{risk premium}}_{\text{compensation for bearing uncertainty}}$$

The first piece, expected loss, moves slowly — it's about the actual creditworthiness of issuers, which changes at the speed of earnings and balance sheets. The second piece, the risk premium, moves *fast* — it's about how much the market, collectively, demands to be paid for holding risky things at all. In a calm, greedy market, investors will hold credit risk for a thin premium, so spreads compress. In a panic, everyone wants safety at once, no one wants to hold risk, and the premium explodes — spreads gap wider even though no company's fundamentals changed overnight. Add in **liquidity**: when markets freeze, you can't sell a corporate bond without taking a haircut, and that illiquidity demands its own premium, which also spikes in a crisis.

This is the single most important thing to understand about spreads: **most of the violence in spreads is the risk premium moving, not default expectations moving.** When the 2008 or 2020 spread blowouts happened, the market wasn't suddenly forecasting that 30% of investment-grade companies would default. It was demanding enormous compensation to hold *any* risk, in a moment when everyone wanted cash and safety. The default expectations barely moved; the price of risk went vertical.

### The centerpiece: spreads move with fear, not just default

Here's the cleanest evidence that spreads are about risk appetite as much as default: **credit spreads and equity volatility move together, in lockstep, spiking at the exact same moments.** Equity volatility — measured by the VIX index, the market's "fear gauge" — has nothing directly to do with corporate default probabilities. It's a measure of how scared the stock market is. Yet when the VIX spikes, credit spreads spike right alongside it, because both are reading the same underlying variable: **the market's appetite for risk.**

![A two line time series chart showing credit spreads and equity volatility moving together over time with both spiking sharply in the 2008 financial crisis and the 2020 pandemic and compressing in calm years between](/imgs/blogs/credit-spreads-pricing-the-probability-of-default-3.png)

The figure above is the centerpiece of this post, and it's illustrative — the shapes are faithful to history but the exact levels are stylized, not live data. It plots two lines over roughly two decades: credit spreads (the average investment-grade spread, in basis points) and equity volatility (the VIX, in volatility points). Watch what they do. In calm years — the mid-2000s, the mid-2010s, 2017 — both lines sit low and quiet. Then, in 2008, both *erupt*: spreads blow out past 600 bps as the VIX rockets above 60. They calm together through the 2010s. Then, in March 2020, both spike again in a matter of weeks — the fastest spread widening in history — as the pandemic hit. The two lines are not identical, but they are unmistakably the same shape, driven by the same force.

Why do they move together so tightly? Because they are two windows onto a single quantity: **how much risk the market wants to hold right now.** When fear rises, investors flee *everything* risky at once — they dump stocks (driving volatility up) and they dump corporate bonds (driving spreads up). The companies haven't changed; the market's willingness to own risk has collapsed. This is also why credit is not the diversifier some investors hope for: in exactly the crisis when you want your bonds to protect you, corporate spreads blow out alongside the stocks you were trying to hedge against. Genuinely safe bonds — Treasuries — rally in a crisis; *credit* bonds fall, because the spread component is, at heart, a risk asset wearing a bond's clothing. (This is the deep reason behind the [stock–bond correlation](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) flips that wreck naive diversification.)

#### Worked example: decomposing a spread blowout

Let's make the two-force decomposition concrete with a crisis scenario. Going into a crisis, Northwind's spread is 200 bps, decomposed as roughly 80 bps of expected loss (the 2% realized PD × 40% LGD) plus 120 bps of risk premium. A panic hits. Northwind's spread blows out to **500 bps**. How much of that 300 bp widening is the market forecasting more defaults, versus just demanding more for risk?

Suppose, in the panic, the realized default expectation for a crossover credit roughly doubles, from 2% to 4% annually — a genuine deterioration. The expected-loss component rises to `0.04 × 0.40 = 160 bps`. So of the new 500 bp spread:

$$\underbrace{160 \text{ bps}}_{\text{expected loss}} + \underbrace{340 \text{ bps}}_{\text{risk premium}} = 500 \text{ bps}$$

The expected-loss piece rose by 80 bps (from 80 to 160). But the *total* spread rose by 300 bps. That means **220 of the 300 bps — nearly three-quarters of the widening — was the risk premium exploding, not default fears rising.** Even though default expectations literally doubled, the lion's share of your loss came from the market repricing the cost of risk itself. This is the anatomy of almost every credit selloff: a modest deterioration in fundamentals, amplified several times over by a collapse in risk appetite.

*The intuition: when spreads blow out, most of the move is the market's panic about risk in general, not its specific forecast of who will default.*

## Putting it all together: the Northwind spread table

We've built every piece. Let's assemble them into one table that shows what a single observed spread tells you about Northwind across a range of scenarios — the kind of cheat-sheet a credit analyst keeps at hand.

![A matrix table showing for Northwind at spreads of one hundred two hundred three hundred and five hundred basis points the implied annual default probability the ten year cumulative probability and the excess return after expected losses](/imgs/blogs/credit-spreads-pricing-the-probability-of-default-7.png)

The figure above is the master table: read across any row to see, for a given observed spread, what default probability the market is implying (at 40% LGD), what that compounds to over ten years, and what excess return you'd keep after subtracting realized losses. It's the whole post in one grid.

#### Worked example: walking the Northwind spread table

Let's compute each row so the figure is fully grounded in arithmetic. Hold LGD at 40% throughout; use the spread-implied PD for the "implied" columns and a realized PD of roughly *half* the implied for the excess-return column (reflecting the historical premium).

**Row 1 — S = 100 bps (Northwind upgraded to solid investment grade).** Implied annual PD = `0.01 / 0.40 = 2.5%`. Survival over 10 years = `0.975^10 ≈ 0.776`, so cumulative PD ≈ **22%**. Realized loss ≈ `1.25% × 0.40 = 50 bps`, so excess return ≈ `100 − 50 = 50 bps`.

**Row 2 — S = 200 bps (Northwind today).** Implied annual PD = `0.02 / 0.40 = 5%`. Survival = `0.95^10 ≈ 0.599`, cumulative PD ≈ **40%** (the risk-neutral figure; the lower 28% we computed earlier used the expected-loss PD of 3.3%). Realized loss ≈ `2.5% × 0.40 = 100 bps`, excess return ≈ `200 − 100 = 100 bps`.

**Row 3 — S = 300 bps (Northwind slipping into high yield).** Implied annual PD = `0.03 / 0.40 = 7.5%`. Survival = `0.925^10 ≈ 0.458`, cumulative PD ≈ **54%** — now better-than-even odds of default over the decade. Realized loss ≈ `3.75% × 0.40 = 150 bps`, excess return ≈ `300 − 150 = 150 bps`.

**Row 4 — S = 500 bps (Northwind in distress).** Implied annual PD = `0.05 / 0.40 = 12.5%`. Survival = `0.875^10 ≈ 0.263`, cumulative PD ≈ **74%**. Realized loss ≈ `6.25% × 0.40 = 250 bps`, excess return ≈ `500 − 250 = 250 bps`.

Notice the pattern: as the spread widens, both the implied default odds *and* the potential excess return rise — which is exactly the trade-off at the heart of credit. Wider spreads pay you more, but they pay you more *because* the odds of getting hurt are higher. The skill is judging whether a given spread overpays or underpays for the actual risk — whether the market's implied PD is too high (a buying opportunity) or too low (a value trap). There is no spread so wide it can't be too narrow for the risk, and none so narrow it can't be too wide.

*The intuition: every spread is simultaneously a default forecast and a payout — and credit investing is the art of deciding whether the payout is worth the forecast.*

## Common misconceptions

**"A higher yield means a better bond."** No — a higher yield on a corporate bond usually means a *riskier* bond. The extra yield is the spread, and the spread exists because the market thinks you might not get paid back. A 10% bond isn't "better" than a 4% bond; it's pricing a much higher chance of default. The yield is the bait; the spread is the warning label. Chasing yield without reading the spread is how investors buy risk they didn't understand they were taking.

**"If there's no default, I earn the whole spread."** Only if you hold to maturity *and* survive the journey. Along the way, the spread moves, and through spread duration those moves hit your price immediately. You can lose 8% on a 10-year credit bond in a week — without any default — purely because the spread widened. Credit returns are realized through both carry (the spread you collect) and price moves (the spread you suffer or enjoy), and the second can dwarf the first over short horizons.

**"The spread tells me the probability of default."** It tells you the *market's implied* default probability, and only after you commit to a recovery assumption — and that implied number is systematically *higher* than the default rate that actually shows up, because the spread also contains a risk premium. Reading a 5% implied PD off a spread and treating it as "the company has a 5% chance of failing" overstates the real-world odds, usually by roughly half. The spread is a price, not a forecast.

**"Spreads widen because companies are about to default."** Sometimes — but most of the time, especially in a crisis, spreads widen because the market's *appetite for risk* collapsed, not because anyone's fundamentals deteriorated. The 2020 spread blowout happened in weeks, far faster than any company's creditworthiness could plausibly change. Most of a spread's volatility is the price of risk moving, not the probability of default moving.

**"Investment-grade bonds are safe."** Safe-*er*, not safe. Investment-grade spreads are narrow because default risk is low — but "low" isn't "zero," and IG bonds carry real spread duration, so they lose value when spreads widen even if they never default. In 2008 and 2020, investment-grade bonds fell hard. The label means the market assesses a low default probability; it does not mean the price won't move or that the company can't be downgraded.

**"Recovery doesn't matter much."** Recovery is half the credit triangle, and the implied default probability is wildly sensitive to it — the same spread implies a 10% PD at 80% recovery or a 2.5% PD at 20% recovery, a fourfold swing. Where a bond sits in the [capital structure](/blog/trading/fixed-income/seniority-recovery-and-the-capital-structure) — secured versus subordinated — changes its recovery and therefore everything you infer from its spread. Ignoring recovery is ignoring half the picture.

## How it shows up in real markets

**The 2008 financial crisis — spreads as a panic gauge.** Going into 2008, investment-grade corporate spreads sat around 150–200 bps. As Lehman Brothers failed in September 2008 and the financial system seized, IG spreads blew out to roughly 600 bps and high-yield spreads to nearly 2,000 bps by late 2008 — levels that, taken literally through the credit triangle, implied default rates that would have wiped out a huge fraction of corporate America. They were never realized; actual defaults, while elevated, came nowhere close. The lesson is the one this post has hammered: most of that spread spike was the price of risk and the collapse of liquidity, not a sober forecast of mass default. Investors who bought IG credit at 600 bps in early 2009 and held earned spectacular returns as the premium normalized — not because defaults were low (though they were) but because the *risk premium* mean-reverted.

**March 2020 — the fastest blowout ever, and the Fed backstop.** When the pandemic hit, IG spreads gapped from about 100 bps to over 370 bps in roughly three weeks — the most violent widening on record — and the corporate bond market briefly stopped functioning, with even high-quality issuers unable to trade. Almost none of this was a default forecast; it was a global dash for cash. The Federal Reserve responded by announcing, for the first time ever, that it would buy corporate bonds (and even high-yield ETFs) directly. Spreads began collapsing the moment the announcement hit — *before the Fed bought a single bond* — because the backstop restored risk appetite and liquidity. It was a live demonstration that spreads price the *willingness to hold risk*, and that willingness can be restored by a credible promise alone. (This is the [central bank's bond-market toolkit](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) at its most dramatic.)

**The energy sector, 2014–2016 — when default fear was justified.** Not every spread blowout is an overreaction. When oil crashed from over \$100 to under \$30 a barrel in 2014–2016, spreads on energy high-yield bonds — especially shale drillers — exploded past 1,000 and then 2,000 bps. This time the spreads were pricing a real, fundamental deterioration: dozens of energy companies genuinely could not service their debt at \$30 oil, and a wave of actual defaults followed in 2015–2016. The lesson is the mirror image of 2008 and 2020: sometimes the spread is *right*, and the implied default probability is realized. Telling the two apart — overreaction versus justified fear — is the entire job of a credit analyst.

**Fallen angels and the BBB/BB cliff.** In March 2020, a wave of companies — Ford most famously — were downgraded from BBB (the lowest investment-grade rung) to BB (the highest junk rung), becoming "fallen angels." The instant they crossed the line, their spreads gapped wider, not because their fundamentals changed overnight but because index-tracking and mandate-constrained investors were *forced* to sell bonds that had dropped out of investment-grade indices. This is the [BBB/BB divide](/blog/trading/fixed-income/investment-grade-vs-high-yield-the-great-divide) in action: the spread jump at a downgrade is partly mechanical, driven by forced selling, which is why the crossover zone where Northwind sits is one of the most treacherous — and occasionally most rewarding — neighborhoods in credit.

**The credit spread puzzle in academic finance.** Decades of research (going back to work by Elton, Gruber, and others) have documented that investment-grade corporate spreads have historically been several times larger than realized default losses can justify. A typical BBB spread of 150–200 bps over the cycle has corresponded to realized credit losses of well under 50 bps. The gap is the credit risk premium — compensation for the *volatility* of spreads, the *illiquidity* of corporate bonds, and the fact that defaults cluster in bad times (exactly when losses hurt most). This puzzle is why credit, held diversified and through the cycle, has historically been a paid risk — and why the discipline of separating the loss the spread implies from the loss you'll actually suffer is worth real money.

**Sovereign spreads and the Eurozone crisis.** Spreads aren't just a corporate phenomenon. During the 2010–2012 Eurozone crisis, the spread of Greek, Italian, and Spanish government bonds over German Bunds — the regional risk-free benchmark — blew out to extraordinary levels, with Greek 10-year spreads exceeding 3,000 bps at the peak. The same credit triangle applied: the market was pricing a real probability of sovereign default and restructuring (which Greece ultimately did, with bondholders taking large losses). The episode ended only when the ECB's Mario Draghi promised to do "whatever it takes" — once again, a credible backstop collapsing a spread by restoring the market's willingness to hold the risk. (See [sovereign debt and the bond vigilantes](/blog/trading/macro-trading/sovereign-debt-and-the-bond-vigilantes) for the policy dimension.)

## When this matters to you and further reading

The credit spread reaches into your life more than you'd guess. The rate on your mortgage is a spread over the Treasury or swap curve; the rate on your car loan and credit card embeds a spread for *your* probability of default; the yield on a corporate-bond fund in your retirement account is, after fees, a spread harvested across hundreds of issuers. Every time a lender quotes you a rate above the risk-free rate, they are running the credit triangle on you — estimating your PD, assuming a recovery, and charging a spread. Understanding what's inside that number is understanding the price of your own credit risk.

For the next layer of depth, the natural next steps within this series are [how Moody's, S&P, and Fitch grade debt](/blog/trading/fixed-income/bond-ratings-how-moodys-sp-and-fitch-grade-debt) (where the spread ladder's letter grades come from), [investment grade versus high yield](/blog/trading/fixed-income/investment-grade-vs-high-yield-the-great-divide) (the BBB/BB cliff that makes the crossover zone so dangerous), and [seniority, recovery, and the capital structure](/blog/trading/fixed-income/seniority-recovery-and-the-capital-structure) (where the recovery rate that drives the whole calculation actually comes from). For the dollars-and-cents mechanics, revisit [modified duration and DV01](/blog/trading/fixed-income/modified-duration-and-dv01-measuring-and-trading-rate-risk), whose logic spread duration borrows directly. And for the allocation lens — how spreads fit into a portfolio alongside stocks and Treasuries — see [corporate credit, investment grade and high yield](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads) and the [stock–bond correlation engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine). For the heavy quantitative machinery behind hazard rates and credit modeling, the [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics) deep dive is the place to go next.

The one thing to carry away is the triangle. A spread is a price for risk; that price equals, roughly, the chance of loss times the size of loss, plus a premium for the discomfort of bearing it. Hold that identity in your head, and a wall of incomprehensible bond quotes resolves into a single readable question, asked over and over, of every borrower on earth: *is the extra yield enough?*

---
title: "Bond ratings: how Moody's, S&P, and Fitch grade debt"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into bond ratings: what the AAA-to-D ladder actually means, how the three big agencies grade debt, why a rating is an opinion and not a price, the issuer-pays conflict at its heart, and the spectacular failures that proved its limits."
tags: ["fixed-income", "bonds", "credit-ratings", "moodys", "s-and-p", "fitch", "investment-grade", "high-yield", "credit-risk", "fallen-angels", "us-treasuries"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — a bond rating is one agency's *opinion* of how likely a borrower is to miss a payment, ranked on a single ladder from AAA (safest) down to D (already in default) — it is not a price, not a guarantee, and not a buy or sell signal.
> - Three firms — **Moody's, S&P, and Fitch** — dominate. They use different symbols (Moody's writes `Aaa`/`Baa`; S&P and Fitch write `AAA`/`BBB`), but the rungs map one-to-one and the scales agree on where the cliff is.
> - The one cliff that matters is the **investment-grade / high-yield line** (S&P's `BBB-` vs `BB+`). Above it, big regulated funds may hold the bond; below it, many *must* sell — so crossing it is a market event, not just a relabel.
> - History shows the ladder genuinely **ranks default risk**: roughly 0.1% of AAA issuers default within ten years versus about half of CCC issuers. The scale works — *on average, over the cycle.*
> - It is paid for by the borrower being graded (the **issuer-pays conflict**), it changes slowly (downgrades lag reality), and in 2008 it failed catastrophically on structured finance — thousands of bonds stamped AAA were downgraded to junk or wiped out.
> - A **\$1,000** bond downgraded from `BBB` to `BB` — a "fallen angel" — typically loses around **\$70** of price as its credit spread widens and forced sellers pile out.

Why does one company borrow at 4% while another, in the same industry, pays 9% — and how did the lenders decide that gap before a single payment was ever made or missed? Almost always, the answer starts with three letters printed next to the bond: a *rating*. A handful of firms read a borrower's finances, form a view on how likely it is to pay you back, and compress that view into a symbol — `AAA`, `BBB`, `BB`, `CCC`. Trillions of dollars move on those symbols. Pension funds are *legally required* to consult them. And in 2008, when the symbols turned out to be wrong about the riskiest corner of the market, they helped take down the global financial system.

This post is about what those symbols actually mean, who assigns them, and — just as important — what they do *not* mean. The single most common mistake a beginner makes is to read a rating as a *price* or a *recommendation*. It is neither. A rating is a relative, through-the-cycle opinion about one narrow question: **will this borrower default?** Not "is this a good investment," not "will the price go up," just — will the coupons and the principal arrive on time.

![A ladder of bond ratings from AAA at the top down to D at the bottom, branching into an investment grade group, a high yield junk group, and default, with the investment grade to high yield boundary drawn as a hard line](/imgs/blogs/bond-ratings-how-moodys-sp-and-fitch-grade-debt-1.png)

The diagram above is the mental model for the whole post: one ladder, ranked from safest to default, with a single famous line drawn across it. Everything above that line is *investment grade* — "safe enough that a conservative institution can hold it." Everything below is *high yield*, or, less politely, *junk* — speculative debt that pays more precisely because it might not pay at all. The rest of this post explains how that ladder is built, who builds it, why the one line in the middle is load-bearing, and where the whole apparatus has failed. (Everything here is educational, not investment advice — the goal is to understand the mechanism, not to tell you what to buy.)

## Foundations: the building blocks you need first

Before we can talk about ratings, we need a few terms defined from zero. If you have read [the anatomy of a bond](/blog/trading/fixed-income/anatomy-of-a-bond-par-coupon-maturity-issuer), this is a refresher; if not, do not skip it, because every later sentence leans on these ideas.

**A bond is a loan you can trade.** When you buy a bond you are lending money to the *issuer* — the borrower. In return the issuer promises a fixed schedule: a periodic *coupon* (the interest) and then the *face value* (also called *par*, almost always \$1,000 per bond) returned at *maturity* (the final date). Our running example all post is a fictional company, **Northwind Corp**, that has issued a **5-year \$1,000 par bond with a 5% coupon** — it pays \$50 a year for five years, then \$1,000 back. Alongside it sits a real benchmark: the **US Treasury**, the bond of the US government, treated by the market as the closest thing to risk-free.

**Default is the thing a rating is about.** A *default* is when a borrower fails to keep its promise — it misses a coupon, can't repay the principal at maturity, or files for bankruptcy and restructures the debt for less than face value. Default is not always total loss: when a bond defaults, holders usually recover *something* (the *recovery rate*), often 30–50 cents on the dollar for senior corporate debt. But default is the event the whole rating system is built to forecast. A rating is, at its core, a *probability-of-default* opinion dressed up as a letter.

**A basis point** is one hundredth of a percent — 0.01%. Rates and spreads are quoted in *basis points* ("bps") because the moves are small: 100 bps is one full percentage point. We'll use it constantly.

**A credit spread is the price of risk.** The Treasury is (treated as) risk-free, so its yield is the baseline cost of money. Any riskier borrower must pay *more* to compensate lenders for the chance of default. That extra yield, above the equivalent Treasury, is the *credit spread*. If a 5-year Treasury yields 4% and Northwind's 5-year bond yields 5.5%, Northwind's credit spread is 1.5%, or 150 bps. The spread *is* the market's own price of the credit risk — and, as we'll see, it usually moves before the rating does.

**Yield and price move on a seesaw.** The coupon printed on a bond never changes. What changes every second is the *price* and, mirror-image to it, the *yield* — the single annual return that makes the bond's future cash flows worth exactly its current price. When the market demands a higher yield (because it now sees more risk), the price of the existing bond *falls*. That inverse link is the [price–yield seesaw](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds), and it is why a downgrade — which raises the yield investors demand — *lowers* the bond's price. Hold that mechanism; the running fallen-angel example turns on it.

With those five terms in hand, here is the one sentence that motivates everything: **a rating is a ranked opinion about default probability, and the market translates that opinion into a credit spread, which sets the bond's price.** The rating is the qualitative judgment; the spread is the quantitative price; the two are linked but not identical, and the gap between them is where a lot of the interesting behavior lives.

## What a rating actually is (and four things it is not)

Let's nail the definition, because almost every misunderstanding flows from a fuzzy one. A bond rating is **a forward-looking opinion about the relative creditworthiness of an issuer or a specific debt obligation** — its capacity and willingness to meet its financial commitments in full and on time. Read that slowly. Four words are doing the heavy lifting: *opinion*, *relative*, *creditworthiness*, *forward-looking*.

It is an **opinion**, not a fact and not a measurement. The agencies say so themselves, in the fine print, partly because that legal framing has historically shielded them under free-speech protections. A rating is a considered professional judgment — but it is a judgment, and judgments can be wrong.

It is **relative**. A AAA bond is not "guaranteed safe"; it is "safer than a AA, which is safer than an A." The ladder *ranks*. The numbers attached to each rung (default rates) are observed averages, not promises. AAA does not mean zero default risk — it means the lowest default risk on the scale.

It is about **creditworthiness** — the probability of default and, to varying degrees across the agencies, the expected loss if default happens. It is emphatically *not* about anything else. It says nothing about whether the bond's price will rise, whether the yield compensates you for the risk, how liquid the bond is, or whether interest rates are about to move. A AAA Treasury can lose 20% of its price in a year if rates spike — that is *interest-rate risk*, which the rating does not address at all.

And it is **forward-looking** and *through-the-cycle*: the agencies try to rate a borrower's ability to survive a *typical downturn*, not just its condition on a sunny day. This is why ratings are deliberately *sticky* — they are not supposed to twitch with every quarter's results. That stickiness is a feature when it filters out noise and a bug when it lags a genuine collapse, as we'll see.

Now the four things a rating is **not**:

| A rating is NOT… | …because |
|---|---|
| a **price** | the price/yield/spread is set by the market every second; the rating is a slow opinion that *influences* the price |
| a **buy/sell recommendation** | it judges default risk only — not whether the yield compensates you for that risk |
| a **guarantee** | even AAA issuers can default; the rating is a probability, not a promise |
| a measure of **interest-rate or liquidity risk** | a AAA bond can still fall hard when rates rise or markets freeze |

#### Worked example: a high rating and a bad investment, at the same time

*Setup.* In 2020, a 10-year US Treasury yielded about 0.6% — and it was rated AAA, the safest rating in existence. You buy \$10,000 of it.

*Step 1 — the rating's promise.* The AAA rating says: the US government will almost certainly pay you your coupons and your \$10,000 back. On that narrow question, the rating was correct — no default.

*Step 2 — what happened to the price.* Over 2022, market yields on that same maturity rose toward 4%. By the price–yield seesaw, a long bond's price falls hard when yields quadruple. A 10-year Treasury bought near 0.6% lost roughly **15–20%** of its market value if you needed to sell.

*Step 3 — reconcile.* The rating never wavered — AAA throughout — yet you'd have lost real money. The rating was *right about default and silent about price.*

*A AAA rating tells you the issuer will pay you back; it tells you nothing about what your bond will be worth before then.*

## How the agencies actually decide a rating

So where does the letter come from? It is not a single number popped out of a black box. An analyst (or a committee of them) builds a picture of the borrower along two big axes, then combines them.

The first axis is **business risk** — qualitative, judgmental, hard to quantify. How stable is the industry? (A regulated water utility has steadier cash flows than a fashion retailer.) How strong is the company's competitive position — its scale, its brand, its diversification across products and geographies? How good and how prudent is management, and how disciplined is its financial policy (does it hoard cash, or buy back stock with borrowed money)? Two companies with identical balance sheets can sit two notches apart purely on business risk: the boring, predictable one rates higher because its cash flows are more likely to survive a recession.

The second axis is **financial risk** — quantitative, ratio-driven. This is where the agencies lean on a handful of credit metrics, and they're worth knowing because they tell you *what the letter is really measuring*:

- **Leverage**: how much debt relative to earnings, usually *debt / EBITDA* (EBITDA is earnings before interest, taxes, depreciation, and amortization — a rough proxy for the cash a business throws off). A company with debt at 2× EBITDA is far safer than one at 6×.
- **Interest coverage**: how many times over the company's earnings cover its interest bill — *EBITDA / interest*. Coverage of 8× means the company earns eight dollars for every dollar of interest it owes; coverage of 1.5× means a bad year could leave it unable to pay.
- **Cash-flow adequacy**: does the business generate enough *free cash flow* (cash left after running and maintaining itself) to service and repay its debt without constantly refinancing?
- **Liquidity**: can the company cover its near-term obligations — the bonds maturing next year, its credit lines — without a crisis if markets freeze?

The agency runs these through its published *methodology* (each firm has detailed, public criteria documents), forms a preliminary rating, and then a *rating committee* — not a lone analyst — votes on the final grade. The committee structure exists precisely to dilute any single analyst's bias or pressure. The output is the letter, an *outlook*, and a written rationale that lays out the reasoning.

#### Worked example: the same letter from two different paths

*Setup.* Two companies are both rated `BBB`. Stable Utility Co earns \$1,000 million of EBITDA, carries \$3,000 million of debt (3× leverage), and covers interest 7×. Cyclical Builder Inc earns the same \$1,000 million of EBITDA but carries only \$2,000 million of debt (2× leverage) and covers interest 9×.

*Step 1 — the puzzle.* On the ratios alone, Cyclical Builder looks *safer* — less leverage, better coverage. Yet both land at `BBB`. Why?

*Step 2 — the business-risk offset.* The utility's cash flows are regulated and recession-proof, so the agency tolerates more leverage at a given rating. The builder's cash flows swing violently with the housing cycle — in a downturn that \$1,000 million of EBITDA could halve — so the agency demands *lower* leverage just to hold the same letter.

*Step 3 — read it.* The same `BBB` encodes a *trade* between business risk and financial risk: the steadier the business, the more debt it's allowed at a given rating.

*A rating is not a ratio — it's a judgment that blends how risky the business is with how much debt it carries, which is why two very different companies can earn the identical letter.*

## The ladder, rung by rung

Now the scale itself. Every agency uses the same underlying idea — an ordered ladder — but the symbols differ. Here is the S&P/Fitch version, top to bottom, with the plain-English meaning of each rung:

- **AAA** — the highest. Extremely strong capacity to meet commitments. Vanishingly few corporate issuers qualify; the US Treasury is the archetype.
- **AA** — very strong. A small notch below AAA, still extremely safe.
- **A** — strong, but somewhat more susceptible to adverse conditions.
- **BBB** — adequate capacity, but more vulnerable to bad economic conditions. **This is the lowest investment-grade rung** — one notch above junk.
- **BB** — the highest *speculative* (junk) rung. Faces major ongoing uncertainties; can pay now, but exposed to a downturn.
- **B** — more vulnerable; currently paying, but a deterioration would likely impair its ability to pay.
- **CCC** — currently vulnerable and dependent on favorable conditions to keep paying. Default is a real possibility.
- **CC / C** — highly vulnerable; default is near or a bankruptcy filing has begun but payments continue.
- **D** — in default. A payment has been missed (beyond any grace period) or the issuer has filed.

Within most of those letters, the agencies add a finer gradation called a *notch*. S&P and Fitch append `+` and `−`: `A+`, `A`, `A−`, then `BBB+`, `BBB`, `BBB−`. Moody's appends numbers: `A1`, `A2`, `A3`, then `Baa1`, `Baa2`, `Baa3`. So `A−` (S&P) and `A3` (Moody's) mean the same thing: the weakest of the three `A`-level notches. A *notch* is the smallest unit of rating change — when people say "a two-notch downgrade," they mean a move like `A` to `BBB+`.

The single most consequential boundary on the whole ladder is between **BBB− and BB+** (Moody's: `Baa3` and `Ba1`). That is the **investment-grade / high-yield line**. It is not just a label change. A huge swath of the world's capital — pension funds, insurers, money-market funds, bond index funds, bank treasuries — operates under rules (regulatory or self-imposed in their mandates) that say "investment grade only." When a bond is downgraded from `BBB−` to `BB+`, those holders may be *forced* to sell, regardless of price. That forced selling is why crossing the line is a violent event, not a gentle slide. We'll trace exactly what it costs in the running example.

#### Worked example: reading a rating off the ladder

*Setup.* Northwind Corp's 5-year bond is rated `BBB` by S&P and `Baa2` by Moody's. A competitor, Southgate Inc, is rated `BB+` by S&P.

*Step 1 — translate.* `BBB` (S&P) = `Baa2` (Moody's): both say "adequate, lowest tier of investment grade." Northwind is *one notch* above the line (`BBB−` is the floor of IG; `BBB` is one above it).

*Step 2 — place Southgate.* `BB+` is the *top* rung of high yield — the first rung *below* the line. So Southgate is junk, but barely: it sits immediately under the boundary.

*Step 3 — the gap that matters.* Northwind and Southgate are only *two notches* apart (`BBB` → `BBB−` → `BB+`), but those two notches straddle the IG/HY line. Northwind is in every investment-grade index; Southgate is in none of them. The buyer base is completely different.

*Two bonds can be a hair apart in credit quality and a world apart in who is allowed to own them — the IG/HY line, not the raw distance, decides that.*

## Three agencies, one ladder: reading the notation

The three dominant firms — **Moody's Investors Service, S&P Global Ratings, and Fitch Ratings** — together account for the overwhelming majority of all outstanding ratings (a combined share usually cited around 95%). They are the "Big Three." The US regulator designates a small set of approved agencies (the NRSRO list — *Nationally Recognized Statistical Rating Organizations*); there are a handful of smaller ones (DBRS Morningstar, Kroll, and others), but the Big Three set the conventions everyone reads.

Their scales agree on the structure but differ on the symbols. The most visible difference: **S&P and Fitch use the letter style** (`AAA`, `AA`, `A`, `BBB`…), while **Moody's uses a mixed-case style** (`Aaa`, `Aa`, `A`, `Baa`…). The figure below lines them up rung for rung.

![A side by side comparison table of Moody's, Standard and Poor's, and Fitch rating symbols showing how the scales map to each other and to plain meaning, with the investment grade to high yield boundary falling at the same rung for all three](/imgs/blogs/bond-ratings-how-moodys-sp-and-fitch-grade-debt-2.png)

The table makes the key point: the symbols are cosmetic, the ranking is the same, and crucially **the IG/HY line falls at the same place for all three** — Moody's draws it between `Baa3` and `Ba1`; S&P and Fitch between `BBB−` and `BB+`. So when an index or a fund mandate says "investment grade," it means the same set of bonds whichever agency rated them.

Two practical wrinkles follow from having three opinions:

**Split ratings.** A single bond can carry different ratings from different agencies — say `BBB−` from S&P but `Ba1` (junk) from Moody's. This is a *split rating*. How does the market decide which side of the line the bond is on? Index providers and many mandates use rules like "the lower of two" or "the middle of three." A bond that is IG by one agency and HY by another is in a genuinely awkward limbo, and which rule applies can determine billions in forced flows.

**The outlook and the watch.** A rating is rarely just a static letter. Each agency also publishes an *outlook* (the likely direction over the medium term: *positive*, *stable*, or *negative*) and, when an event is imminent, a *CreditWatch* / *Review* (a near-term flag that a change is actively being considered). A `BBB negative outlook` is the market's early warning that a fallen angel may be coming. Smart credit investors watch the outlook and the watch far more than the headline letter, because those move first.

#### Worked example: a split rating decides who must sell

*Setup.* Northwind's bond is downgraded by S&P to `BB+` (junk) but Moody's keeps it at `Baa3` (the lowest IG rung). It is now split: junk by one, IG by the other.

*Step 1 — the index rule.* A major investment-grade index uses "if rated by two agencies, take the lower." Lower of `BB+` and `Baa3` is `BB+` — junk. The bond is *ejected* from the IG index at the next month-end.

*Step 2 — the mandate rule.* An insurer's mandate says "IG if rated IG by *any* NRSRO." By that rule the bond is still IG (Moody's keeps it `Baa3`), so the insurer may keep holding it.

*Step 3 — the clash.* Index funds tracking the IG index must sell; the insurer need not. The split rating splits the buyer base right down the middle, and the selling pressure depends entirely on which rule a given holder follows.

*With three opinions, the rule for combining them — not any single letter — often decides whether a bond gets dumped.*

## The proof the scale works: default rates by rating

A skeptic should ask: does the ladder actually *rank* default risk, or is it astrology with letters? This is the most important empirical question in the whole subject, and the answer — averaged over decades and thousands of issuers — is a clear yes. The agencies publish long-run *default studies* every year, and the pattern is unmistakable: as you walk down the ladder, the historical default rate climbs, gently at first and then ferociously.

![A bar chart of ten year cumulative default rates by rating, with bars near zero for AAA and AA, low single digits for A and BBB, then rising steeply through BB and B to roughly half for CCC, with the investment grade to high yield boundary marked](/imgs/blogs/bond-ratings-how-moodys-sp-and-fitch-grade-debt-3.png)

The chart shows roughly what the published studies find for *cumulative ten-year default rates* (illustrative, rounded figures consistent with long-run S&P and Moody's averages): **AAA around 0.1%, AA around 0.5%, A around 1.5%, BBB around 3.5%** — and then, across the IG/HY line, the curve goes near-vertical: **BB around 12%, B around 28%, CCC around 50%**. The staircase is steep on purpose. Half of all CCC issuers don't make it ten years; almost no AAA issuers fail. That gap is the scale earning its keep.

Two things to read off this carefully:

**Investment grade is genuinely safe — collectively.** A BBB default rate of ~3.5% over a decade means about 96–97% of BBB issuers paid in full over ten years. That is why the IG/HY line is meaningful: above it, default is the rare exception; below it, it is a material, plannable risk.

**The curve is convex, not linear.** Notice that each step *down* the ladder roughly *multiplies* the default rate rather than adding a fixed amount. AAA to BBB is a 35× jump (0.1% to 3.5%); BBB to CCC is another ~14× jump. Risk doesn't rise smoothly as you descend — it explodes once you cross into junk and accelerates toward CCC. This is why the high-yield market is a different animal: it is priced for, and lives with, real default.

#### Worked example: turning a rating into expected loss

*Setup.* You hold \$100,000 face of single-`B` corporate bonds. The historical ~5-year cumulative default rate for `B` is around 15% (let's use that), and senior unsecured recovery in default averages around 40 cents on the dollar.

*Step 1 — expected defaults.* 15% of \$100,000 = **\$15,000** of face value is expected to default over five years (statistically, across many such bonds).

*Step 2 — loss given default.* On the defaulted \$15,000 you recover ~40%, so you lose 60%: 0.60 × \$15,000 = **\$9,000** of expected credit loss.

*Step 3 — what the yield must cover.* Spread that \$9,000 loss over five years on \$100,000 and it's about **1.8% per year** of expected loss. So a `B` bond must pay you *at least* ~1.8%/year over Treasuries just to break even on average credit losses — and more, to compensate for the *risk* around that average.

*A rating isn't just a letter; combined with recovery assumptions it converts directly into the minimum extra yield the bond must pay to be worth owning.*

## The rating and the spread: when the market disagrees

We've now met both halves of the credit picture: the **rating** (a slow opinion, updated occasionally by a committee) and the **credit spread** (a live price, updated every second by buyers and sellers). They are supposed to tell the same story — riskier borrower, lower rating, wider spread. Most of the time they do, and the chart of "average spread by rating" is a clean staircase that mirrors the default-rate staircase from earlier. But the interesting cases are precisely the ones where they *disagree*, and learning to read that gap is the difference between a tourist and a resident in the credit market.

The spread moves first because it is a market price and the rating is a deliberate, sticky judgment. So a useful mental model is: **the spread is the market continuously voting on creditworthiness; the rating is the agency's considered verdict published with a lag.** When they part company, one of two things is usually happening.

**The spread is wider than the rating implies.** A bond rated `BBB` but trading at a spread normally seen on `BB` paper is what traders call a *crossover* or a "`BBB` trading like junk." The market is telling you it expects a downgrade — it is pricing risk the rating hasn't acknowledged yet. Sometimes the market is right and the downgrade follows (Enron, WorldCom). Sometimes the market is over-fearful and the bond is genuinely cheap — a `BBB` paying a `BB` yield, still inside investment-grade mandates, is exactly the kind of mispricing active credit funds hunt for.

**The spread is tighter than the rating implies.** A bond rated `BB` but trading like `BBB` means the market is *more* comfortable than the agency. This often happens to *rising stars* — junk issuers improving toward an upgrade — where the market front-runs the agency's slow machinery and bids the bond up before the upgrade is official.

#### Worked example: the crossover that pays you to be early

*Setup.* A `BBB−` bond (lowest IG rung) trades at a 350 bps spread — far wider than the ~150 bps typical of `BBB−`, and squarely in `BB` territory. You judge the company will *not* be downgraded; the market is simply panicking.

*Step 1 — the carry.* You buy \$100,000 of it. You're earning roughly 350 bps over Treasuries — about **\$3,500/year** more than a Treasury, and ~\$2,000/year more than a "normal" `BBB−` would pay.

*Step 2 — the repricing if you're right.* If the panic fades and the spread tightens back to 150 bps, that's a 200 bps move. With a duration of ~4.5, the price rises ~4.5 × 2.0% ≈ **9%** — about **\$9,000** of capital gain on your \$100,000.

*Step 3 — the risk if you're wrong.* If instead it *is* downgraded to junk and the spread blows out further, forced sellers crush the price and you take a loss — the same mechanism as the fallen-angel example, against you. The fat carry was the market paying you to take exactly that risk.

*When the spread and the rating disagree, the market is offering you a bet — extra yield in exchange for the chance the agency's slow opinion is about to catch down to the market's fast one.*

This gap is also why sophisticated investors treat the rating as *one input among several*, not the answer. The outlook and watch tell you the agency's *direction*; the spread tells you the *market's* verdict and how urgently it's being repriced; and your own analysis of the business and the capital structure tells you whether either of them is wrong. The rating is the floor of the analysis, never the ceiling.

## Who pays for the rating: the issuer-pays conflict

Here is the structural problem sitting at the center of the entire industry, and it is so simple it's almost a joke: **the borrower pays the agency to rate the borrower's own debt.** This is the *issuer-pays* model, and it has been the dominant business model since the 1970s.

![A flow showing an issuer paying a fee to a rating agency, the agency assigning a rating, and the rating being published free to investors who rely on it and bear the default risk](/imgs/blogs/bond-ratings-how-moodys-sp-and-fitch-grade-debt-4.png)

Walk the flow. A company — Northwind — wants to sell bonds and benefits enormously from a high rating (a higher rating means a lower yield, which means cheaper borrowing). Northwind *hires and pays* one or more of the Big Three to rate the deal. The agency assigns the grade. The rating is then published *free* to the whole market, and investors — who pay nothing and bear all the default risk — rely on it. The customer and the entity being judged are the same party. The people the rating is supposed to protect are not in the transaction at all.

Why does the industry work this way at all? It wasn't always so. Until the early 1970s the agencies used an *investor-pays* model — they sold rating manuals to investors by subscription. Two things broke it: cheap photocopiers (subscribers shared the manuals, so the agencies couldn't capture the value) and the 1970 collapse of Penn Central, a large issuer whose default shook confidence and made issuers *want* to pay for a credible third-party stamp to reassure buyers. Issuer-pays solved the agencies' revenue problem — and created the conflict that has haunted ratings ever since.

The conflict has teeth in several ways:

- **Rating shopping.** An issuer can quietly ask agencies for a preliminary indication and then *hire only the ones that offer the best grade*. The agencies know this, which creates a subtle pressure to be issuer-friendly.
- **The repeat-customer relationship.** A big issuer is a recurring client worth millions in fees. An analyst who is too harsh risks the firm losing the account to a more accommodating competitor.
- **Structured finance made it worse.** In the 2000s, a handful of investment banks were the *repeat customers* for vast volumes of mortgage-bond ratings. The fees were huge and concentrated, and the banks could and did pressure agencies, who competed for the business. We'll see the result in the case studies.

To be fair, the model has real counter-pressures. An agency's *entire franchise* is its reputation; ratings that prove systematically wrong destroy the brand the whole business rests on. Post-2008 reforms (the Dodd-Frank Act in the US) added oversight, disclosure, and rules to curb the worst rating-shopping. Investor-pays challengers exist. But the core conflict — the graded party writes the check — has never been removed, only managed. Keep it in mind every time you read a rating: it is an opinion, produced by a firm paid by the borrower.

#### Worked example: how a single notch pays for itself many times over

*Setup.* Northwind wants to issue \$500 million of 5-year bonds. At `BBB` its credit spread is 150 bps; the agencies hint that at `BBB+` (one notch higher) the spread would be ~120 bps.

*Step 1 — the interest savings.* One notch saves 30 bps of yield. On \$500M that's 0.30% × \$500,000,000 = **\$1,500,000 per year**.

*Step 2 — over the bond's life.* Five years of \$1.5M = **\$7,500,000** saved (ignoring discounting, for a back-of-envelope figure).

*Step 3 — versus the rating fee.* The fee to get the deal rated is typically a few basis points of issuance — call it 5 bps on \$500M, about \$250,000 *total* across agencies.

*A single notch of rating is worth millions to the issuer while the rating fee is a rounding error — which is exactly why the incentive to get a friendly grade is so powerful.*

## How ratings change: transitions, downgrades, and notching

Ratings are not set once and frozen. They migrate — usually slowly, occasionally in a rush. The agencies study this with a *transition matrix*: a table that says, of all issuers at a given rating at the start of a year, what fraction sat at each rating a year later. It is the single most useful tool for understanding how credit risk actually evolves.

![A one year rating transition matrix showing that most issuers keep their rating, with small percentages migrating up or down, and the BBB to BB cell highlighted as the fallen angel transition](/imgs/blogs/bond-ratings-how-moodys-sp-and-fitch-grade-debt-5.png)

Read the matrix row by row. The defining feature is the **dominant diagonal**: most issuers, most years, stay where they are. A typical one-year study finds ~90% of `AAA` issuers still `AAA` a year later, ~89% of `BBB` still `BBB`, and so on. Ratings are *sticky* — that's the through-the-cycle design at work. But the off-diagonal cells are where the action is:

- **Downgrade drift.** Lower-rated issuers are *more likely to be downgraded than upgraded* — the matrix is asymmetric, leaning toward deterioration, because companies that get into trouble tend to keep getting into trouble. The lower you start, the bigger this tilt.
- **Fallen angels and rising stars.** A *fallen angel* is an issuer downgraded from investment grade (the lowest IG rung, `BBB`) to high yield (`BB`). It is the most-watched transition on the matrix because of the forced-selling cliff. The mirror image — junk upgraded to IG — is a *rising star*, and it triggers forced *buying* by index funds that must now add it.
- **The CCC trap.** Look at the bottom row: a large slice of CCC issuers either default within the year or fall further. Once you're near the bottom, the transition matrix is brutal — there's little room left to fall except into default.

*Notching* is a related idea. Even within one issuer, different bonds can carry different ratings depending on where they sit in the capital structure. A senior secured bond (first in line in a default) might be rated a notch or two *above* the issuer's general rating, while a subordinated or junior bond (last in line, except equity) is *notched down*. Same company, different ratings, because the *recovery* in a default differs by seniority. This is why you'll sometimes see one issuer's bonds rated `BBB+`, `BBB`, and `BBB−` simultaneously — the agency is notching for seniority.

#### Worked example: a fallen angel and the \$70 hit

*Setup.* This is the running example, paid off at last. Northwind's 5-year \$1,000 bond starts at `BBB`, priced at par (\$1,000) with a 5.5% yield — a 150 bps spread over a 4% Treasury. It is downgraded to `BB+` — a fallen angel.

*Step 1 — the spread widens.* Crossing into junk, the market re-prices Northwind's risk. Say the spread widens from 150 bps to 270 bps — a 120 bps increase. The new yield investors demand is 4% + 2.7% = **6.7%**.

*Step 2 — the price falls (seesaw + duration).* The bond's modified duration is about 4.5 (a 5-year bond). A 120 bps (1.2%) rise in yield drops the price by roughly duration × yield change = 4.5 × 1.2% ≈ **5.4%** — call it ~7% once you include the steeper repricing and the fact that the move isn't tiny. On \$1,000 that's about a **\$70 loss**: the bond falls to roughly **\$930**.

*Step 3 — forced selling deepens it.* Now the cliff bites. IG-only funds must dump the bond into a thinner junk market, pushing the price *below* where fundamentals alone would put it — sometimes another point or two — before high-yield buyers step in at the cheaper level.

*A one-notch label change that crosses the IG/HY line turns into a real, ~\$70-per-bond loss, amplified by forced sellers who don't care about price — the rating didn't change the company overnight, but it changed who is allowed to own it.*

The figure below lays out that same downgrade as a before-and-after on the numbers.

![A matrix comparing a bond before and after a downgrade from BBB to BB, showing the credit spread widening, the yield rising, the price of a one thousand dollar bond falling about seventy dollars, and investment grade funds being forced to sell](/imgs/blogs/bond-ratings-how-moodys-sp-and-fitch-grade-debt-7.png)

Read it left to right: the spread widens ~120 bps, the yield demanded rises to ~6.7%, the \$1,000 bond reprices to ~\$930, the mark-to-market hit is about −7%, and the buyer base flips from "IG funds may hold" to "IG funds must sell." Every box ties back to the worked example above — the rating is the trigger, the spread is the mechanism, the price is the consequence.

### Issuer ratings versus issue ratings: not the same thing

One more distinction that trips up beginners and matters for exactly the notching we just met. There are two kinds of rating, and people blur them constantly.

An **issuer rating** (also called a *counterparty* or *corporate family* rating) grades *the company as a whole* — its overall ability to pay its debts. An **issue rating** grades *one specific bond* that the company has sold. A single company has *one* issuer rating but can have *many* issue ratings, one per bond, and they can differ — because two bonds from the same borrower can have very different *recovery prospects* if the company defaults.

Why does recovery differ within one company? Because in a bankruptcy, creditors are paid in a strict order — the *capital structure* or *seniority* — and the rating reflects where your bond sits in that line:

- **Secured debt** is backed by specific collateral (a factory, a fleet of planes). If the company fails, secured creditors get first claim on that asset, so their recovery is high — often 60–80 cents on the dollar. Agencies *notch this up* relative to the issuer.
- **Senior unsecured debt** has no specific collateral but ranks ahead of subordinated debt — the typical investment-grade corporate bond. Recovery averages around 40 cents.
- **Subordinated (junior) debt** is paid only after senior creditors are made whole — recovery is low, sometimes near zero. Agencies *notch this down*.
- **Equity** (the shareholders) is dead last and usually gets nothing in a default — which is why a bond, even a risky one, is structurally safer than the same company's stock.

So a single `BBB` issuer might have a `BBB+` secured bond, a `BBB` senior unsecured bond, and a `BBB−` subordinated bond outstanding at the same time — three issue ratings, one issuer rating, all internally consistent. The lesson for an investor: *always check whether the rating you're reading is for the company or for your specific bond, and where your bond sits in the line.* A high issuer rating doesn't help much if you own the most junior slice.

#### Worked example: same company, two bonds, two recoveries

*Setup.* Riverside Manufacturing defaults owing \$1,000 on each of two bonds you could have bought: a *senior secured* bond backed by its factory, and a *subordinated* bond with no collateral. The factory and remaining assets are sold and there's enough to cover senior claims fully but little left over.

*Step 1 — the senior secured bond.* First in line, backed by the factory. Recovery: say 75 cents on the dollar. On your \$1,000 you get back **\$750** — a \$250 loss.

*Step 2 — the subordinated bond.* Paid only after the senior creditors are whole, and there's almost nothing left. Recovery: say 10 cents. On your \$1,000 you get back **\$100** — a \$900 loss.

*Step 3 — read it.* Same company, same default, same \$1,000 invested — but a \$650 difference in outcome purely from seniority. That's why the agency rated the secured bond two notches above the subordinated one.

*The default probability is a property of the company, but the loss you actually take is a property of your specific bond — which is why issue ratings, not just the issuer rating, are what you must read.*

## The limits: lag, procyclicality, and the 2008 catastrophe

Now the uncomfortable part. The ladder *ranks* default risk well on average and over the cycle — but it has serious, structural limits, and ignoring them is how investors get hurt.

**Ratings lag.** Because they are deliberately through-the-cycle and sticky, ratings are *slow*. The agencies do not want to downgrade a company on one bad quarter and re-upgrade it on the next. The cost of that stability is that when a borrower is genuinely deteriorating, the rating often confirms what the market already knows rather than warning ahead of it. The credit spread — the live market price — typically moves *before* the rating. By the time `Enron` was downgraded to junk in 2001, it was four days from bankruptcy; the market had been pricing distress for weeks. The rating was a lagging indicator dressed as a forward-looking one.

**Ratings are procyclical.** Because downgrades trigger forced selling and higher borrowing costs, a downgrade can *worsen* the very distress it describes — a company downgraded in a crisis faces higher rates exactly when it can least afford them, which makes default more likely, which justifies further downgrades. Ratings can amplify the cycle they're trying to measure. This is the dark side of the IG/HY cliff: it concentrates pain at the worst moment.

**Ratings were hard-wired into regulation — which made the failures systemic.** For decades, the law itself leaned on ratings. Bank capital rules let a bank hold *less* capital against a `AAA` bond than a `BBB` one; insurance regulators set reserve requirements by rating; money-market funds were restricted to top-rated paper. This *regulatory reliance* meant a rating wasn't just an opinion investors could take or leave — it was plumbing baked into the rulebook, so an error didn't just fool one buyer, it mispriced risk for the entire regulated system at once. Worse, it created a perverse incentive: institutions bought the *highest-yielding* bond *within* each rating bucket — the riskiest `AAA`, the one whose extra spread hinted the market doubted the letter — because the rules treated all `AAA`s as identical while the market did not. Post-2008, Dodd-Frank explicitly directed US regulators to *strip mandatory rating references* out of the rules and replace them with independent risk assessments. The reliance has been reduced, not eliminated, but the lesson stuck: when a flawed opinion is written into law, its mistakes scale.

**Sovereigns are slow and political.** Rating a *country* (a sovereign) is harder and slower still. Agencies were criticized for being late to downgrade Greece before the 2010 euro crisis, then criticized for downgrading too fast once the crisis hit and deepening it. The 2011 S&P downgrade of the United States from `AAA` to `AA+` — the first ever — was a political earthquake; ironically, Treasury *yields fell* (prices rose) right after, because in a panic investors still fled *to* US debt, which is a vivid reminder that a rating is an opinion and the market can disagree.

And then there is the failure that dwarfs all the others.

![A before and after comparison of a 2008 mortgage backed security, showing a senior tranche stamped AAA at issuance on the assumption that home prices would not all fall together, and the same tranche downgraded to junk or defaulted after nationwide home prices fell in unison](/imgs/blogs/bond-ratings-how-moodys-sp-and-fitch-grade-debt-6.png)

**The 2008 structured-finance catastrophe.** In the mid-2000s, Wall Street took pools of thousands of subprime mortgages, sliced them into layers (*tranches*), and asked the agencies to rate the layers. The senior tranches — first in line to be paid from the mortgage cash flows — were stamped `AAA`, "as safe as Treasuries." Investors worldwide, mandated to hold safe assets, bought them by the trillion.

The ratings rested on a model assumption that turned out to be catastrophically wrong: that home prices across different regions of the country were only *weakly correlated* — that California and Florida and Ohio wouldn't all fall at once. As long as defaults were scattered and independent, the senior tranche was genuinely protected, because it would take an improbable simultaneous wave of defaults to eat through the junior layers beneath it. The math, on that assumption, really did support `AAA`.

But in 2007–2009, US home prices fell *nationwide, together*. Correlation that the models assumed near zero went, in effect, to one. The "improbable simultaneous wave" happened. Junior tranches were wiped out and the losses ate straight into the supposedly bulletproof senior `AAA` tranches. The agencies mass-downgraded thousands of structured securities — many fell from `AAA` to junk in *months*, some defaulted outright. It was the single largest ratings failure on record, and it had three compounding causes: a flawed correlation model, the issuer-pays conflict at its most concentrated (a few banks paying enormous fees for vast volumes of these ratings), and the false equivalence of slapping a `AAA` *corporate-scale* symbol on a *structured* product whose risk behaved nothing like a `AAA` company. A `AAA` Treasury and a `AAA` subprime CDO tranche shared a label and almost nothing else.

#### Worked example: why "low correlation" was the whole ballgame

*Setup.* A simplified mortgage pool funds two tranches: a junior tranche that absorbs the *first* 10% of losses, and a senior tranche (90% of the deal) that only loses money if losses exceed 10%. Each underlying mortgage has, say, a 5% chance of defaulting with zero recovery.

*Step 1 — the low-correlation world (the model's assumption).* If defaults are *independent*, pool-wide losses cluster tightly near 5% — almost never above 10%. So the senior tranche almost never takes a loss. The model says: `AAA`, comfortably.

*Step 2 — the high-correlation world (reality in 2008).* If a national housing bust makes mortgages default *together*, the pool's loss isn't a tidy 5% — it's all-or-nothing-ish: either a calm year near 0%, or a bust where 30%, 40% of the pool defaults at once. Now losses blow *far* past the 10% junior buffer and slam the senior tranche.

*Step 3 — the lesson.* The senior tranche's safety depended *entirely* on the correlation assumption. Change one number — correlation — from "near zero" to "near one," and a `AAA` becomes a near-total loss. The rating was only as good as that single hidden input.

*The 2008 `AAA` failure wasn't bad arithmetic — it was the right arithmetic fed one disastrously wrong assumption, which is exactly the kind of error a single letter can never reveal.*

## Common misconceptions

**"AAA means the bond can't lose money."** No. AAA means the lowest *default* risk on the scale — the issuer will almost certainly pay you back. It says nothing about *price*. A AAA Treasury can fall 15–20% in a year when interest rates rise. Default risk and price risk are different animals, and the rating only addresses the first.

**"A rating is an objective fact."** It is an *opinion* — a considered professional judgment, but a judgment, produced by a firm paid by the borrower it's grading. The agencies themselves frame it as opinion, partly for legal protection. Treat it as one informed view among several, not as a measured constant.

**"All AAAs are equally safe."** A `AAA` corporate bond, a `AAA` government, and (pre-2008) a `AAA` structured product shared a symbol and wildly different real risk. The 2008 crisis was, in part, a lesson that a single scale was being stretched across products whose risks behaved nothing alike. Always ask *what* is rated, not just the letter.

**"The rating tells me when to sell."** It usually tells you *after* the market already knows. Credit spreads — the live price of risk — typically move days or weeks before the rating catches up. If you wait for the downgrade to act, you're often selling into the crowd that already repriced the bond.

**"Investment grade is safe, junk is dangerous, end of story."** The line is real and consequential, but it's a probability boundary, not a wall. Plenty of `BBB` companies have defaulted; plenty of `BB` companies pay every coupon for decades. The line decides *who is allowed to own the bond* (and thus the forced-flow dynamics) at least as much as it decides the underlying risk.

**"The agencies caused the 2008 crisis."** They were a crucial enabler, not the sole cause. The flawed correlation model, the issuer-pays conflict, and the explosion of subprime lending all combined. The ratings turned bad mortgages into "safe" assets that institutions worldwide could and did buy — but the rot started in the loans, the leverage, and the incentives, not only in the letters.

## How it shows up in real markets

**The Ford and GM downgrades of 2005.** In May 2005, S&P cut both Ford and General Motors to junk — at the time, the two largest fallen angels in history, with around \$450 billion of debt between them suddenly crossing the IG/HY line. The forced selling was enormous: IG funds dumping, the high-yield market suddenly asked to absorb a wall of new junk paper it wasn't sized for. Spreads gapped, and the episode is a textbook illustration of the cliff — the companies didn't collapse overnight, but the *ownership base* of their debt had to turn over violently.

**Enron, 2001.** Enron's bonds were rated investment grade until *four days* before it filed for bankruptcy. The market had been pricing distress for weeks — the stock had cratered, the spreads had blown out — but the rating held at IG almost to the end. It became the canonical example of ratings *lagging* reality, and it fed directly into the post-Enron reforms and the later, harsher scrutiny of the agencies.

**The 2008 subprime CDOs.** The defining failure. Trillions of dollars of mortgage-backed and CDO tranches rated `AAA` were downgraded en masse in 2007–2009, many to junk within months, many to total loss. Pension funds, money-market funds, and banks across the world held them precisely *because* of the `AAA` stamp — that's what made it systemic. The episode triggered the Dodd-Frank reforms, removed some hard-wired regulatory reliance on ratings, and permanently changed how the market treats structured-finance ratings.

**The US downgrade, 2011 (and again later).** When S&P stripped the United States of its `AAA` in August 2011 — the first time ever — markets braced for chaos in Treasuries. Instead, in the surrounding flight-to-safety, Treasury *prices rose and yields fell*. The world's investors disagreed with the agency about the safest asset on earth, and voted with their money. It is the clearest possible demonstration that a rating is an *opinion* the market is free to override. (Fitch followed with its own US downgrade in 2023, to similar muted market effect.)

**Greece and the euro crisis, 2010–2012.** Agencies were attacked from both sides: too slow to flag Greece's deteriorating finances before the crisis, then accused of *deepening* it by cutting Greece and other periphery sovereigns rapidly once it began — each downgrade raising borrowing costs and triggering collateral and mandate effects that made the fiscal hole deeper. It is the procyclicality problem in its rawest sovereign form, and it spurred the EU to build its own ratings oversight (ESMA).

**The 2020 "fallen angel" wave.** When COVID hit, a record volume of `BBB`-rated debt was at risk of being cut to junk — the IG market had grown enormously at its lowest rung, so a wave of fallen angels threatened to flood a much smaller high-yield market. The Federal Reserve took the extraordinary step of announcing it would buy *recently downgraded* fallen-angel debt, explicitly to prevent the forced-selling cliff from turning a downgrade wave into a fire sale. It was a vivid, real-time demonstration of how much market plumbing hangs on that one line — enough that a central bank intervened to defend it.

**The "BBB bulge" of the late 2010s.** Through the cheap-money decade after 2008, companies loaded up on debt, and the `BBB` tier — the lowest rung of investment grade — swelled to become the *largest* slice of the entire IG market, several trillion dollars. Analysts warned of a structural trap: in the next recession, even a modest wave of `BBB` downgrades would dump more junk onto the high-yield market than it could digest, because so much debt was parked one notch above the cliff. The 2020 Fed intervention was, in effect, the bulge's first stress test — and the reason it was such a live worry was the asymmetry of the IG/HY line. It is the clearest modern illustration that the *distribution* of ratings across the market, not just any single rating, is itself a systemic risk.

**WorldCom, 2002.** A year after Enron, WorldCom — a telecom giant — went from investment grade to the largest bankruptcy in US history at the time, after an \$11 billion accounting fraud surfaced. Like Enron, its bonds were rated IG embarrassingly close to the collapse. Coming back-to-back, Enron and WorldCom shattered confidence that the ratings caught fraud or aggressive accounting, and together they drove the Sarbanes-Oxley Act and intensified the scrutiny that would later, after 2008, produce direct oversight of the agencies themselves. The pattern — a fraud the ratings missed until the market had already figured it out — is the recurring shape of every ratings-lag story.

## When this matters to you, and where to go next

If you own a bond fund, a target-date retirement fund, or your employer's pension is invested in credit, ratings touch your money whether you think about them or not — they decide what those funds are allowed to hold and how they behave when a borrower stumbles. Understanding that a rating is a slow, paid-for *opinion* about default — not a price, not a guarantee, not a sell signal — is the difference between trusting the letter blindly and reading it for what it is.

The deeper lesson of this post is the one the credit market learned the hard way: a rating compresses a borrower's entire future into three letters, and compression always loses information. Use it as a *starting point* — a fast, useful, mostly-reliable ranking of default risk — and then look at the things the letter can't show you: the live credit spread, the outlook and watch, the seniority of *your* bond in the capital structure, and the assumptions behind any structured product.

To go deeper, the natural next steps are the dedicated case study on the agencies themselves, [credit rating agencies: Moody's, S&P, and Fitch](/blog/trading/finance/credit-rating-agencies-moodys-sp-fitch); the allocation-lens view of the IG/HY split in [corporate credit: investment grade, high yield, and spreads](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads); the mechanics of how a downgrade flows into price through the [price–yield seesaw](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds); and, for the macro backdrop that moves spreads in the first place, [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable). The bond market runs on these three letters; now you know what they're hiding.

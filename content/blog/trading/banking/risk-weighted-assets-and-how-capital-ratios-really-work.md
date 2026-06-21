---
title: "Risk-Weighted Assets and How Capital Ratios Really Work: Why the Denominator Decides Everything"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "A from-scratch guide to risk-weighted assets, the standardized and internal-model approaches, the denominator games banks play, and the leverage-ratio backstop and output floor that catch them."
tags: ["banking", "risk-weighted-assets", "capital-ratio", "cet1", "basel", "leverage-ratio", "irb", "output-floor", "bank-regulation", "rwa-density"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A bank's capital ratio is capital divided by *risk-weighted assets*, not total assets, and the bank gets a lot of say over that denominator — which is why two banks with identical balance sheets can report wildly different capital strength.
>
> - **Risk-weighting** scales every asset by how risky it is: cash gets a 0% weight, a prime mortgage maybe 35%, a plain corporate loan 100%, a soured loan 150%. RWA is the weighted sum.
> - The headline ratio everyone quotes — **CET1 ratio = CET1 capital / RWA** — can be flattered by shrinking RWA without adding a dollar of capital. That is the *denominator game*.
> - Banks shrink RWA by tilting toward low-weight assets and by using **internal models (IRB)** to assign lower weights than the regulator's standardized table would. Same loans, smaller RWA.
> - Two backstops fence this in: the **leverage ratio** (capital / total assets, un-weighted, ignores the models entirely) and the Basel **output floor** (IRB RWA can't fall below 72.5% of the standardized number).
> - The one number to remember: **RWA density = RWA / total assets**. A bank at 25% density and one at 60% are not in the same business, even if both report "12% CET1."

In April 2023, a few weeks after Silicon Valley Bank failed, analysts noticed something uncomfortable about the banks still standing. Several of them reported Common Equity Tier 1 ratios — the gold-standard measure of capital strength — north of 12%, comfortably above every regulatory minimum. On paper, they looked like fortresses. And yet their stock prices were collapsing, their deposits were walking out the door, and a few of them would be dead within months.

How can a bank be "well capitalized" by the official measure and falling apart at the same time? The answer is buried in one word that does an astonishing amount of work in modern banking: *risk-weighted*. The capital ratio that regulators bless, that ratings agencies cite, that bank CEOs put on the first slide of every earnings call, is not capital divided by the bank's assets. It is capital divided by its *risk-weighted assets* — a number the bank substantially constructs itself, asset by asset, model by model. Change the recipe and the ratio changes, even though not one dollar of real equity moved.

This post is about that denominator. We will build risk-weighted assets from absolute zero — what a risk weight is, where it comes from, how the sum is computed — then show you the two competing ways banks calculate it (the regulator's fixed table versus the bank's own models), the games that calculation invites, and the two crude-but-honest backstops regulators bolted on to stop the games from getting out of hand. By the end, when you see "CET1 ratio: 13.4%" on a bank's slide, you will know exactly which questions to ask before you believe it.

![Pipeline from raw assets through risk weights to RWA to the capital ratio](/imgs/blogs/risk-weighted-assets-and-how-capital-ratios-really-work-1.png)

The diagram above is the mental model for the whole post: raw assets go in, each one gets multiplied by a risk weight, the weighted total is RWA, and the capital ratio is capital divided by *that* number — never the raw assets. Hold onto the shape of it. Everything else is detail hung on this frame.

This connects straight back to the spine of this whole series. A bank is a leveraged, confidence-funded maturity-transformation machine: it borrows short, lends long, and survives only as long as its thin equity cushion absorbs losses faster than they arrive. The capital ratio is the single number that is *supposed* to tell you how thick that cushion is relative to the trouble it might have to absorb. Risk-weighting is the attempt to make the comparison fair — to say a dollar of cash and a dollar of a junk loan should not count the same when you measure how much cushion you need. The trouble, as we will see, is that "how risky is this asset" is exactly the kind of question a bank has every incentive to answer optimistically.

## Foundations: assets, risk weights, RWA, and the ratios that judge a bank

Let's define every term from scratch. If you have read [bank capital and leverage](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion), some of this will be a refresher; if not, you can proceed from here cold.

**An asset** is anything the bank owns that has value — cash in its vault and at the central bank, the loans it has made (a mortgage you owe the bank is *its* asset, your liability), the bonds it holds, the buildings it occupies. The asset side of a bank's balance sheet is, overwhelmingly, loans and securities. Those are the things that earn the bank money and, crucially, the things that can lose value.

**Capital**, in the sense we mean here, is the bank's own money — its equity, the cushion that absorbs losses before depositors and other creditors are touched. The highest-quality, most loss-absorbing slice is **Common Equity Tier 1 (CET1)**: ordinary shares plus retained earnings, the stuff that takes the first hit when a loan goes bad. There are softer layers above it (Additional Tier 1, Tier 2) that we cover in [Basel I, II, III and the capital rules](/blog/trading/banking/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank); for this post, when we say "capital," picture CET1 unless noted.

**A risk weight** is a percentage the regulator (or the bank's model) assigns to an asset to reflect how likely it is to lose value and how much would be lost. Cash can't default, so it gets a **0%** weight — it consumes no capital. A loan to a blue-chip government in its own currency, also effectively 0%. A residential mortgage, well-collateralized by a house, might get **35%**. A standard loan to an unrated company gets **100%** — the benchmark, "average" risk. A loan that is already past-due and unsecured can get **150%**, meaning regulators want you to hold capital as if it were *more* than a full dollar of risk. The risk weight is the dial that says "treat this dollar as if it were 0, 35, 100, or 150 cents of true exposure."

**Risk-weighted assets (RWA)** is just the sum, across every asset, of (asset amount × its risk weight). It is the denominator of every capital ratio that matters. If you hold \$100 of cash, that contributes \$0 to RWA. If you hold \$100 of unrated corporate loans, that contributes \$100. If you hold \$100 of prime mortgages, that contributes \$35. RWA is, loosely, "how much truly-risky exposure does this bank carry, expressed in benchmark dollars."

**The capital ratio** is capital divided by RWA. The flagship is the **CET1 ratio = CET1 capital / RWA**. There are wider ones — Tier 1 ratio (Tier 1 capital / RWA), Total Capital ratio (all capital / RWA) — but they share the same denominator. When a bank says it is "well capitalized," it almost always means its CET1 ratio clears the regulatory minimum plus its buffers, which for a large global bank lands somewhere around 10–11% all-in (a 4.5% bare minimum plus a 2.5% conservation buffer plus a surcharge for being systemically important).

**The leverage ratio** is the deliberately dumb cousin: capital / *total* assets (un-weighted). It ignores risk weights entirely. A dollar of cash and a dollar of junk loan count the same. Its whole job is to be un-foolable by the modeling that goes into RWA. Basel sets a 3% minimum; the US imposes 5–6% on its largest banks via the enhanced supplementary leverage ratio.

**The output floor** is a Basel III rule that caps how much a bank's internal models can shrink RWA below what the standardized table would produce. The floor is **72.5%**: a bank using its own models cannot report RWA lower than 72.5% of the standardized RWA on the same portfolio. It exists precisely because internal models, left unchecked, drifted toward implausibly low risk weights.

**RWA density** is RWA divided by total assets — the share of the balance sheet that "counts" as risky. A bank with \$1,000 of assets and \$400 of RWA has a 40% density. Density is the single most revealing number in this whole field, because it strips away the games and asks: of every dollar you hold, how many cents do you admit are at risk? We will return to it again and again.

### The standardized approach versus the internal-ratings-based approach

There are two licensed ways to compute risk weights, and the gap between them is the heart of this post.

The **standardized approach** is the regulator's fixed lookup table. The Basel Committee (and your national regulator) publishes risk weights by asset class and, in the latest version, by credit quality. Cash: 0%. High-grade sovereigns: 0%. Residential mortgages: 20–70% depending on loan-to-value. Investment-grade corporates: 50–65%. Unrated corporates: 100%. Past-due exposures: up to 150%. The bank looks up each asset and applies the prescribed number. It is blunt — every unrated corporate borrower gets 100% whether it's a rock-solid family firm or a wobbly startup — but it is *firm*. The bank has almost no discretion.

The **internal-ratings-based approach (IRB)** lets a bank, with regulatory approval, estimate the risk inputs itself and feed them into a Basel-supplied formula that spits out a risk weight. The inputs are the credit-risk trinity covered in [credit risk management: PD, LGD, EAD](/blog/trading/banking/loan-pricing-cost-of-funds-risk-premium-and-the-capital-charge) — the probability of default, the loss given default, and the exposure at default. A bank with decades of its own loan data can argue, "our prime mortgages default far less than the standardized table assumes, and we recover almost all of the loan when they do, so the true risk weight should be 12%, not 35%." If the regulator buys the models, the bank gets the lower weight — and a smaller RWA, and a higher capital ratio, on the very same loans.

That is the central tension. The standardized approach is crude but hard to game. IRB is risk-sensitive but soft — and softness, in a system where every basis point of capital ratio affects the share price, gets exploited. The rest of this post is what happens in that tension and what regulators did about it.

### Where the whole idea came from, and why

It helps to know why risk-weighting exists at all, because the history explains both its purpose and its built-in flaw. Before the first Basel Accord in 1988, there was no internationally agreed way to measure whether a bank had enough capital. Different countries used different rules, and most of them anchored to a simple, un-weighted leverage measure: hold equity equal to some flat percentage of total assets. That was honest but blind. It treated a dollar of cash and a dollar of a speculative loan identically, which created two problems at once. It let a reckless bank look as safe as a careful one, and — more perversely — it actively *pushed* banks toward risk. If a safe loan and a risky loan both consume the same flat capital, the rational move is to chase the risky one's higher yield, because you get paid more for the same capital cost. Flat leverage rules quietly subsidize danger.

Risk-weighting was the fix. The 1988 Accord introduced a small set of buckets — 0% for cash and OECD sovereigns, 20% for claims on banks, 50% for mortgages, 100% for everything else — so that capital requirements would finally track risk. A bank that took more risk had to hold more capital; a bank that played it safe got relief. That is the genuine, good idea at the bottom of all this, and it is worth keeping in mind, because the games we are about to describe are not arguments *against* risk-weighting. They are the predictable consequence of letting the institution whose ratio is being measured help set the weights. The 1988 buckets were crude and gameable in their own way (everything corporate was 100%, so banks engineered exposures to land in the 50% or 20% buckets), and the entire arc from Basel I to today is the regulators chasing ever-more-sophisticated ways to measure risk while the banks chase ever-more-sophisticated ways to lower the measured number. We pick up that arms race in the standardized-versus-IRB story below.

### The three RWA buckets: credit, market, and operational

One more piece of foundation before the arithmetic. Everything above has talked about *credit* risk — the chance a borrower doesn't pay. Credit risk is the largest piece of RWA for almost every bank, often 80%+ of the total, which is why it dominates the discussion. But total RWA is the sum of three separate buckets, and a complete picture needs all three.

**Credit RWA** captures the risk that loans and bonds lose value because borrowers default or are downgraded. This is the bucket the standardized table and IRB models we have been discussing apply to. For a commercial bank it is the overwhelming majority of RWA.

**Market RWA** captures the risk that the bank's *trading book* — the securities and derivatives it holds to trade, not to hold to maturity — loses value as prices move. This is computed from value-at-risk-style models or a standardized market-risk framework (the post-2008 overhaul is called the Fundamental Review of the Trading Book, or FRTB). For a pure commercial lender, market RWA is tiny; for an investment bank with a huge trading operation, it can be a meaningful slice.

**Operational RWA** captures the risk of losses from failed processes, fraud, lawsuits, cyber-attacks, and the like — the rogue trader, the mis-selling fine, the systems outage. Basel III replaced the old, model-based operational-risk charge with a single standardized formula driven mainly by the bank's income, precisely because the internal models for operational risk had become another denominator game. For a large bank, operational RWA is often 10–15% of the total.

Two things matter here. First, when you see a bank's total RWA, it is credit + market + operational, and a sudden change in any one of them moves the ratio — a fat conduct fine, for example, can spike operational RWA and dent the CET1 ratio with no change to the loan book at all. Second — and this is the deep point that the SVB collapse drove home — **interest-rate risk in the banking book is in none of these buckets.** The risk that a bank's hold-to-maturity bonds lose market value as rates rise sits outside the RWA framework entirely. RWA can say a bank is a fortress while a risk it doesn't even measure is hollowing it out. Keep that gap in mind; it is the single most important limitation of the whole apparatus, and we return to it with SVB.

## How RWA is actually computed, step by step

Let's make the abstraction concrete with the simplest possible bank and build up.

Picture a bank with exactly \$100 of assets, spread across a few buckets. We apply the standardized risk weight to each, sum the weighted amounts, and out comes RWA. The risk weights by asset class are the dial we keep coming back to:

![Standardized risk weights by asset class from zero to 150 percent](/imgs/blogs/risk-weighted-assets-and-how-capital-ratios-really-work-2.png)

Notice the spread. Cash and high-grade sovereign debt sit at 0% — they consume no capital at all, which is exactly why banks loaded up on government bonds (a decision that, ironically, helped sink SVB through a different channel: interest-rate risk, not credit risk). The "average" asset, an unrated corporate loan, sits at 100% by construction — that is the benchmark the whole scale is calibrated around. And the genuinely troubled assets climb to 150%, where regulators demand you hold capital as if the dollar of exposure were a dollar and a half.

Now the arithmetic.

#### Worked example: computing RWA from assets and risk weights

A bank holds exactly \$100 of assets:

- \$20 of cash and central-bank reserves, risk weight **0%**
- \$40 of prime residential mortgages, risk weight **35%**
- \$30 of unrated corporate loans, risk weight **100%**
- \$10 of past-due, unsecured loans, risk weight **150%**

Multiply each bucket by its weight and add:

- Cash: \$20 × 0% = **\$0**
- Mortgages: \$40 × 35% = **\$14**
- Corporate: \$30 × 100% = **\$30**
- Past-due: \$10 × 150% = **\$15**

RWA = \$0 + \$14 + \$30 + \$15 = **\$59**.

So a bank with \$100 of assets carries \$59 of RWA. Its RWA density is 59 / 100 = **59%**. The intuition: even though it has \$100 of stuff on the balance sheet, regulators only make it hold capital against \$59 of "benchmark risk" — the cash is treated as harmless, and the mortgages count for barely a third of their face value, while the small slug of bad loans punches above its weight.

(For the cover figure and the worked examples that follow, I round this stylized bank's RWA to a clean \$55 by trimming the past-due bucket slightly — the exact cents don't change the lesson, and round numbers are easier to track. The mechanics are identical.)

Once you have RWA, the capital ratio is a single division.

#### Worked example: the CET1 ratio is capital divided by RWA

Take the bank above with **\$55 of RWA** and suppose it holds **\$6.60 of CET1 capital** — real common equity and retained earnings.

CET1 ratio = CET1 / RWA = \$6.60 / \$55 = **12.0%**.

Twelve percent comfortably clears the ~10–11% all-in demand a large bank faces, so this bank reports as well capitalized.

Here is the part that matters. The same \$6.60 of capital, measured against the bank's \$100 of *total* assets, is only 6.6%. The capital ratio looks nearly twice as strong as the leverage ratio, purely because the denominator shrank from \$100 to \$55. The intuition: the capital ratio is not "how much equity per dollar of stuff you own" — it is "how much equity per dollar of *admitted* risk," and the bank gets to do a lot of the admitting.

This is the engine of everything that follows. If you can legitimately move the denominator down — by holding more 0% and 35% assets, or by convincing the regulator your 100% loans deserve a 60% weight — your reported ratio rises with no new capital. Let's see exactly how.

## Standardized versus IRB: same loans, different RWA

Now we put the two approaches head to head on one portfolio. This is where the "two banks, same assets, different ratios" puzzle gets resolved.

Take a bank with \$100 of loans — \$40 of mortgages, \$35 of corporate loans, \$25 of retail and small-business loans. Under the standardized table it gets one RWA. If it has regulatory approval to use internal models, it gets to substitute its own (lower) estimates and gets a different, smaller RWA. Same loans. Same borrowers. Same balance sheet. Two numbers.

![Same one hundred dollars of loans producing two different RWA totals under standardized and IRB](/imgs/blogs/risk-weighted-assets-and-how-capital-ratios-really-work-3.png)

#### Worked example: the same portfolio under standardized and IRB

Standardized weights (representative): mortgages 35%, corporate 100%, retail/SME 75%.

- Mortgages: \$40 × 35% = **\$14**
- Corporate: \$35 × 100% = **\$35**
- Retail/SME: \$25 × 75% = **\$18.75**
- **Standardized RWA ≈ \$67.75** (call it \$68)

Now the same bank's internal models, fed its own PD and LGD history, produce lower effective weights — say 15% on mortgages, 60% on corporate, 48% on retail:

- Mortgages: \$40 × 15% = **\$6**
- Corporate: \$35 × 60% = **\$21**
- Retail/SME: \$25 × 48% = **\$12**
- **IRB RWA = \$39**

Suppose the bank holds \$7 of CET1 either way. Under standardized, its ratio is \$7 / \$68 = **10.3%** — adequate but unspectacular. Under IRB, the *identical* bank reports \$7 / \$39 = **17.9%** — a fortress. The intuition: nothing about the bank's actual loss exposure changed between those two numbers. The only thing that changed was who was allowed to set the risk weights, and the bank's own models are systematically kinder than the regulator's table.

This is not a hypothetical. Empirically, large European and US banks that run IRB models have historically reported RWA roughly **20–40% lower** than the standardized approach would produce on the same exposures, and the dispersion *between* banks modeling the *same* type of risk has at times been enormous. The Basel Committee ran exercises in the mid-2010s handing identical hypothetical portfolios to dozens of banks and asking each to compute the RWA with its own approved models. The answers varied by a factor of two or more for the same assets. That is not risk sensitivity; that is a measurement system whose readings depend on who is holding the instrument.

### Why IRB exists at all (the case for it)

It would be easy to read this as pure regulatory capture, but the case for IRB is real. The standardized approach is genuinely crude. A 100% weight on every unrated corporate loan means a bank that lends carefully to strong mid-sized firms is forced to hold the same capital as a bank that lends recklessly to weak ones. That punishes prudence and creates a perverse incentive: if a careful loan and a reckless loan consume the same capital, you might as well chase the reckless one's higher yield. Risk-sensitive weights, done honestly, fix that — they reward the bank that actually has lower losses with lower capital, which is what you want.

The problem is the word "honestly." Models are built by the institution whose capital ratio they determine, validated by regulators who are outgunned on data and talent, and rebuilt whenever a model can be tuned to a friendlier answer. The incentive gradient points one way: down. Over a decade, even with no single act of bad faith, the whole population of models drifts toward optimism, because every borderline modeling choice that lowers RWA gets made and every one that raises it gets challenged. That is the denominator game in its most respectable form — not fraud, just gravity.

### What the IRB formula actually does

It is worth peeking inside the IRB machinery, because the levers a bank pulls only make sense once you see what the formula consumes. The IRB risk weight for a credit exposure is driven by three inputs, the credit-risk trinity:

- **PD** — the probability of default over one year, expressed as a percentage. A AAA borrower might have a PD of 0.01%; a CCC borrower, 26%.
- **LGD** — loss given default, the fraction of the exposure the bank actually loses if the borrower defaults, after recoveries from collateral. A senior secured loan might have an LGD of 25% (you recover 75 cents on the dollar); a subordinated, unsecured one, 65%.
- **EAD** — exposure at default, how much will be owed at the moment of default (which for a credit line includes the portion the borrower is likely to draw down on the way to trouble).

The Basel formula takes PD and LGD, runs them through a calibrated curve designed to capture losses at a severe (99.9th-percentile) one-year stress, and produces a risk weight. The crucial property is that the output is *monotone* in the inputs — lower PD, lower LGD, or shorter maturity all push the risk weight down. So every modeling choice that nudges an estimated PD or LGD lower flows directly into a smaller risk weight, a smaller RWA, and a higher ratio. That is why "RWA optimization" is so often really "PD and LGD optimization": recognize a bit more collateral and your LGD drops; reclassify a borrower one notch better and your PD drops; the formula does the rest.

#### Worked example: total RWA across all three buckets

A mid-sized universal bank has three RWA components:

- **Credit RWA: \$80** (the loan and bond book, computed as above)
- **Market RWA: \$8** (its modest trading desk)
- **Operational RWA: \$12** (the standardized op-risk charge on its income)

Total RWA = \$80 + \$8 + \$12 = **\$100**. With \$11 of CET1, its ratio is \$11 / \$100 = **11.0%**.

Now suppose the bank loses a major lawsuit and its operational-risk charge jumps from \$12 to \$22 as the loss event feeds the formula. Nothing about its loans changed. But total RWA rises to \$110, and the ratio falls to \$11 / \$110 = **10.0%** — a full percentage point gone. The intuition: RWA is a sum of three risks, and a shock to any one of them moves the headline ratio. A capital ratio that drops without any deterioration in the loan book often points to market or operational RWA, not credit — which is exactly where a careless reader forgets to look.

## The denominator games: how to flatter a ratio without raising capital

There are three broad ways to lift a capital ratio. Only one of them involves actually getting stronger.

The honest way is to **raise the numerator** — issue shares, retain earnings, cut the dividend, sell a business and keep the proceeds as equity. This genuinely thickens the cushion. It is also expensive and dilutive, which is why management treats it as a last resort.

The two denominator games **shrink RWA** instead, and the ratio rises just the same on the page even though the cushion is unchanged in absolute dollars.

![Graph showing two levers cut RWA which lifts the capital ratio with capital unchanged](/imgs/blogs/risk-weighted-assets-and-how-capital-ratios-really-work-8.png)

The first lever is **portfolio tilt**: shift the asset mix toward low-weight assets. Sell the 100%-weighted corporate loans and buy 0%-weighted government bonds. Originate more mortgages (35%) and fewer unsecured business loans (100%). Push exposures into anything that carries a low number in the table. This is partly legitimate balance-sheet management and partly cosmetic — and it has a dark side, because the lowest-weight assets aren't actually risk-free. A bank that piles into 0%-weighted long-dated government bonds has zero *credit* RWA on them but has loaded up on *interest-rate* risk that RWA doesn't capture at all. That is, very nearly, the SVB story.

The second lever is **model optimization**: get internal models approved that assign lower weights, then keep refining them downward at every model review. Reclassify exposures into lower-risk buckets. Recognize more collateral and guarantees so the effective LGD drops. Each individual move is defensible; the cumulative effect is a steadily shrinking denominator. This is "RWA optimization," and at large banks it is a staffed, budgeted, KPI-tracked discipline with a target measured in basis points of CET1 ratio.

#### Worked example: shrinking RWA lifts the ratio with zero new capital

Start with our bank: \$6.60 of CET1, \$55 of RWA, CET1 ratio = 12.0%.

Management wants to report a 16% ratio for the next earnings call but doesn't want to issue equity. So it works the denominator. It sells \$10 of 100%-weighted corporate loans (removing \$10 of RWA) and gets a model refinement approved that trims another \$5 of RWA off the mortgage and retail books. RWA falls from \$55 to **\$40**.

New CET1 ratio = \$6.60 / \$40 = **16.5%**.

The bank just went from "adequately capitalized" to "fortress" — a 4.5-percentage-point jump — and its actual equity cushion is the *same \$6.60 it always was*. The intuition: a capital ratio can rise for two completely different reasons, and only one of them — more capital — makes the bank safer. The other just makes the denominator smaller. If you can't tell which happened, the ratio is telling you almost nothing.

This is why a single-period jump in a bank's CET1 ratio should make you *more* curious, not less. Did equity actually grow, or did RWA shrink? You find the answer in the RWA walk — the reconciliation, disclosed in the bank's Pillar 3 report, that breaks the period's RWA change into book growth, model and methodology changes, asset-quality migration, and FX. A big "model and methodology" line is the denominator game showing up in print.

There is a third family of denominator moves that sits between the two clean levers and deserves its own mention: **risk transfer**. A bank can keep a loan on its books but buy protection against its default — through a guarantee, a credit-default swap, or a "significant risk transfer" securitization in which the riskiest first-loss tranche is sold to outside investors. Done genuinely, this *does* lower the bank's risk and so legitimately lowers RWA: if someone else now bears the first losses, the bank really is less exposed. But the same structures can be engineered so the bank quietly retains most of the economic risk while shedding the regulatory RWA, paying a fee to a willing counterparty purely to rent a lower capital charge. Regulators scrutinize significant-risk-transfer deals precisely because the line between "I sold the risk" and "I rented a lower number" is thin and lucrative to blur. When you see a bank's RWA drop sharply alongside a wave of synthetic securitization, the question is always whether the risk actually left the building or just the RWA did.

## RWA density: the number the games can't fully hide

Because the levers above all work by shrinking RWA relative to the assets, the cleanest defense against being fooled is to look at the ratio of the two directly: **RWA density = RWA / total assets**. It tells you, of every dollar the bank holds, how many cents it treats as risky. The capital ratio can be flattered; density makes the flattering visible.

![Bar chart of RWA density across four bank profiles from twenty-two to seventy-eight percent](/imgs/blogs/risk-weighted-assets-and-how-capital-ratios-really-work-7.png)

The spread across real banks is staggering. A mortgage-heavy bank running aggressive internal models might report RWA density in the low 20s — meaning it claims that of every \$100 of assets, only about \$22 is genuinely at risk. A diversified universal bank lands nearer 35–40%. A plain-vanilla commercial lender on the standardized approach can be 55%+. And an emerging-market lender full of unrated borrowers can run 75–80% density, because almost everything it holds carries a high weight.

#### Worked example: two banks, same assets, very different density

Bank A and Bank B each hold exactly \$1,000 of assets and each report a **12% CET1 ratio**. Identical on the first slide.

Bank A runs IRB and reports **RWA density of 25%**, so its RWA is \$1,000 × 25% = \$250. A 12% CET1 ratio on \$250 of RWA means CET1 capital = 12% × \$250 = **\$30**.

Bank B runs the standardized approach and reports **RWA density of 50%**, so its RWA is \$1,000 × 50% = \$500. A 12% CET1 ratio on \$500 of RWA means CET1 capital = 12% × \$500 = **\$60**.

Same assets, same headline ratio — but Bank B holds **twice the capital** (\$60 versus \$30) against the same \$1,000 balance sheet. Measured the honest, un-weighted way, Bank A's equity is just 3% of assets and Bank B's is 6%. The intuition: the CET1 ratio normalized away exactly the thing you most wanted to know — how much real cushion sits under the assets. Density, and the leverage ratio it implies, hands it back.

This is the resolution to the puzzle we opened with. When two banks report the same capital ratio, they are not making the same claim, because they are dividing by different denominators built on different assumptions. The CET1 ratio is a *relative* statement — capital relative to the bank's own assessment of its risk. Density and leverage are *absolute* statements — capital relative to the assets, full stop. A careful reader always wants both.

## The leverage ratio: the un-foolable backstop

Regulators understood the denominator problem long before SVB, and their first fix was deliberately stupid by design. The **leverage ratio** is capital divided by total assets (technically, total exposure including some off-balance-sheet items), with **no risk weights at all**. Every dollar counts as a dollar. It cannot be gamed by tilting the portfolio or tuning a model, because it doesn't look at risk weights — it looks at the raw size of the balance sheet.

![Before and after panels comparing the risk-weighted ratio and the un-weighted leverage ratio on one bank](/imgs/blogs/risk-weighted-assets-and-how-capital-ratios-really-work-4.png)

The figure shows the two views of the very same bank. On the left, the risk-weighted view: \$6.60 of CET1 over \$55 of RWA is a 12% ratio that screams "well capitalized." On the right, the un-weighted view: the same \$6.60 of Tier 1 capital over \$100 of total assets is a 6.6% leverage ratio — a far thinner-looking cushion. Nothing about the bank changed between the two panels. The leverage ratio simply refuses to give the bank credit for its low-weight assets.

The Basel minimum leverage ratio is **3%**, with a surcharge for global systemically important banks. The US applies a tougher **enhanced supplementary leverage ratio** of 5% at the holding-company level and 6% at the insured-bank level for its largest institutions. The point of these floors is to act as a *backstop*: the risk-weighted ratio is the primary, risk-sensitive measure, but if a bank games its RWA down far enough, the leverage ratio becomes the *binding* constraint — the one that actually stops it — regardless of what the models say.

![Bar chart comparing a stylized bank CET1 ratio and leverage ratio against their regulatory minimums](/imgs/blogs/risk-weighted-assets-and-how-capital-ratios-really-work-5.png)

The chart above makes the headroom visible. The bank's CET1 ratio of 12% sits comfortably above its ~7% all-in risk-weighted demand — lots of room. But its leverage ratio of 6.6% is only just above the 3% Basel minimum (and would be far closer to a binding constraint under the US 5–6% enhanced floor). The risk-weighted measure says "plenty of capital"; the leverage measure says "watch the size of this balance sheet." When those two disagree, the disagreement *is* the information.

The relationship between the two ratios is worth making precise, because it tells you exactly when the leverage ratio takes over. The leverage ratio equals the risk-weighted ratio multiplied by RWA density: leverage ratio = (capital / RWA) × (RWA / total assets) = capital / total assets. So a bank with a 12% CET1 ratio and 55% density has a leverage ratio of 12% × 55% = 6.6% — exactly our figure. The lower the density, the wider the wedge between the impressive risk-weighted number and the sober un-weighted one. A bank reporting 14% CET1 at 25% density has a leverage ratio of just 3.5%; the same 14% CET1 at 60% density gives 8.4%. The two banks could not be more different in real cushion, yet their headline ratios are identical. This is the single most useful piece of arithmetic in the post: multiply any bank's risk-weighted ratio by its RWA density and you recover its leverage ratio, which is the number the games cannot touch.

#### Worked example: when the leverage ratio becomes the binding constraint

A bank plays the denominator game to the hilt. It pushes RWA density down to 20% by loading up on 0%-weighted government bonds and squeezing its models. It holds \$5 of capital against \$100 of total assets.

Risk-weighted view: RWA = \$100 × 20% = \$20. CET1 ratio = \$5 / \$20 = **25%**. By the risk-weighted measure this bank looks like one of the best-capitalized institutions on earth.

Leverage view: \$5 / \$100 = **5% leverage ratio**. Under the US enhanced floor of 5–6%, this bank is *at or below its binding constraint* despite the heroic-looking 25% CET1 ratio.

The intuition: once a bank shrinks RWA aggressively enough, the un-weighted leverage ratio takes over as the rule that actually limits it. The risk-weighted ratio becomes a vanity metric — high, impressive, and no longer the thing keeping the bank honest. The leverage ratio is what catches a flattered headline number, which is exactly why it exists.

The leverage ratio has a famous weakness, too, and it cuts the other way. Because it ignores risk entirely, it gives a bank no capital relief for holding genuinely safe assets, and it can actively *discourage* low-risk, low-margin, high-volume businesses — like holding government bonds, clearing trades, or warehousing safe collateral — because those bloat total assets without bloating RWA. In stressed moments, banks have pulled back from exactly these safe activities to protect their leverage ratio, which can make markets *less* liquid when liquidity is most needed. The leverage ratio is the backstop, not the primary measure, precisely because being risk-blind is both its strength and its flaw.

## The output floor: capping how far the models can go

The leverage ratio backstops the *whole* balance sheet against being shrunk to nothing. But it doesn't directly police the gap between a bank's internal models and the standardized table — a bank can have a comfortable leverage ratio and still be running implausibly low risk weights on specific books. So Basel III's final reforms (the package the industry calls "Basel IV," finalized in 2017 and phasing in through the late 2020s) added a second, more targeted fence: the **output floor**.

The rule is simple to state. A bank using internal models cannot report total RWA below **72.5%** of what the standardized approach would produce on the same portfolio. Whatever your models say, your RWA is floored at 72.5% of the standardized number. It caps the maximum discount internal models can deliver at 27.5%.

![Matrix comparing standardized, internal models, and the output floor across who sets the weight, effect on RWA, and main risk](/imgs/blogs/risk-weighted-assets-and-how-capital-ratios-really-work-6.png)

The matrix lays the three approaches side by side. Standardized: the regulator sets the weight from a fixed table, RWA comes out higher but firm, and the main risk is crudeness. IRB: the bank sets the weight from its own models, RWA comes out lower, and the main risk is that it understates the danger. The output floor: it doesn't compute weights at all — it computes the standardized RWA in parallel, takes 72.5% of it, and uses that as a hard floor under whatever the models produced. It is the explicit admission that internal models, left alone, will drift too far.

#### Worked example: the output floor catching an over-optimistic model

A bank's portfolio produces **\$100 of standardized RWA**. Its internal models, after years of optimization, produce only **\$60 of RWA** — a 40% discount.

The output floor is 72.5% of standardized: 72.5% × \$100 = **\$72.50**.

Because the model RWA (\$60) is *below* the floor (\$72.50), the bank must report the floor: its RWA is bumped up to **\$72.50**, not \$60. The floor just added \$12.50 of RWA the models had stripped out.

Now run it through the ratio. With \$8 of capital: pre-floor the bank claimed \$8 / \$60 = **13.3%**; post-floor it must report \$8 / \$72.50 = **11.0%**. The output floor knocked 2.3 percentage points off the headline ratio without any change to the bank's actual risk or capital — it simply refused to honor the last slice of the model discount. The intuition: the floor doesn't ask whether the models are right; it caps how much benefit they're allowed to deliver, on the theory that a discount bigger than ~27.5% is more likely to be optimism than insight.

The output floor is the most consequential capital reform of the post-2008 era for the big model-using banks, because it directly raises the RWA — and therefore lowers the reported ratios — of exactly the institutions that had pushed their models hardest. European banks, which lean heavily on IRB, fought it for years and won a long phase-in (the floor ratchets up over several years to the full 72.5%), but the direction is set. The era of unbounded model discounts is ending. You can read the broader regulatory arc in [BIS and Basel bank regulation](/blog/trading/finance/bis-and-basel-bank-regulation).

## Common misconceptions

**"The capital ratio is capital divided by the bank's assets."** It is not — it is capital divided by *risk-weighted* assets, which can be far smaller than total assets. In our running example, \$6.60 of capital is a 12% capital ratio (against \$55 of RWA) but only a 6.6% leverage ratio (against \$100 of assets). Anyone quoting "12% capital" without saying which denominator is, knowingly or not, quoting the more flattering one. The two numbers can differ by a factor of two or more.

**"A higher capital ratio always means a safer bank."** Only if the higher ratio came from more capital. A ratio that rose because RWA shrank — through portfolio tilt or model optimization — reflects a smaller denominator, not a thicker cushion. In the worked example, the bank went from 12% to 16.5% with the *same \$6.60 of capital*. You must check the RWA walk to know which lever moved.

**"Internal models are more accurate, so IRB ratios are more trustworthy."** Risk-sensitive does not mean accurate when the modeler's incentive points one direction. The Basel Committee's own benchmarking exercises handed identical portfolios to many banks and got RWA estimates that varied by more than two-to-one. A measurement that disagrees with itself by a factor of two across users is not precise; it is plastic. That plasticity is exactly why the output floor exists.

**"The leverage ratio is a worse measure than the risk-weighted ratio, so it doesn't matter."** It is a *cruder* measure, deliberately, and it matters enormously because it cannot be gamed. It is the binding constraint for the most aggressive banks and the honest cross-check for all of them. Crude and un-foolable beats sophisticated and gameable when the stakes are a bank's survival. The best read on a bank uses both.

**"A 0% risk weight means an asset is risk-free."** It means it is treated as carrying no *credit* risk for capital purposes — not that it cannot lose value. Government bonds carry a 0% credit weight but enormous *interest-rate* risk, which RWA largely ignores. SVB held a huge book of 0%-weighted, AAA-rated securities and was destroyed when rising rates crushed their market value. The risk weight measured the wrong risk. Zero on the RWA table is not zero in the real world.

**"RWA is just the loan book — it's all credit risk."** Credit RWA is the biggest piece, often more than 80% of the total, but RWA is the sum of credit, market, and operational risk. A bank's ratio can fall because a trading loss inflated market RWA or a conduct fine inflated operational RWA, with the loan book untouched. When you see a CET1 ratio drop, your first question should be *which bucket moved* — and the answer is in the RWA breakdown in the Pillar 3 report, not the loan disclosures.

**"The output floor makes all banks' ratios comparable now."** It narrows the gap, but it does not close it. The floor caps the *total* model discount at 27.5% relative to standardized; it does not force every bank to compute risk the same way below that cap, and it phases in over years with national variations. A standardized-approach bank and a floored IRB bank reporting the same CET1 ratio still are not making identical claims. Density and the leverage ratio remain the cross-checks; the floor just keeps the IRB discount from running to infinity.

## How it shows up in real banks

**The Basel benchmarking shock (mid-2010s).** The Basel Committee and national regulators ran "hypothetical portfolio exercises": they constructed identical loan and trading portfolios and asked dozens of large banks to compute the RWA using their own approved internal models. For the same assets, the implied risk weights varied by a factor of two or more across banks — some banks' models said a given exposure deserved a 20% weight while others said 45%. There was no way both could be right about identical assets. This finding, more than any single scandal, is what convinced regulators that internal models had become a source of *unwarranted variability* in capital ratios, and it is the direct intellectual origin of the output floor.

**SVB and the 0%-weight trap (March 2023).** Silicon Valley Bank reported strong regulatory capital ratios right up to its collapse, in part because so much of its balance sheet sat in 0%-weighted Treasury and agency securities that carried almost no credit RWA. The bank's risk-weighted ratios looked fine; its problem lived entirely outside the credit-RWA framework, in the interest-rate risk of long-dated bonds whose market value fell roughly \$17 billion as rates rose. RWA measured the credit risk of those bonds (near zero) and was blind to the rate risk that actually killed the bank. The lesson is not that the ratios lied — it is that a 0% weight measures one specific risk, and a bank can die of a different one. The full mechanics are in [the SVB and Credit Suisse 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).

**RWA optimization as a profession.** At every large bank, "RWA optimization" or "capital management" is a permanent, well-staffed function with a target measured in basis points of CET1 ratio and dollars of freed-up capital. Its toolkit is entirely legitimate on paper: recognize more eligible collateral to lower LGD, clean up data so exposures aren't conservatively defaulted to higher buckets, use credit-risk transfers and synthetic securitizations to offload the riskiest tranches, refine models at each review. A bank that frees up, say, \$5 billion of RWA can either lift its ratio or redeploy that capital into new lending. None of this is fraud — but the whole apparatus exists to make the denominator smaller, and its very existence is why a sophisticated reader treats a falling RWA with the same scrutiny as a rising one.

**The European fight over the output floor.** When Basel finalized the output floor at 72.5%, European banks objected loudly, because they rely on internal models far more than US banks and stood to see their RWA — and thus their reported ratios — rise materially as the floor bit. Studies at the time estimated the fully-loaded floor would raise the aggregate RWA of the largest European banks by double-digit percentages, forcing them to either raise capital or shrink lending. The EU responded with a long phase-in stretching toward the end of the decade and some carve-outs, but accepted the principle. It is the clearest real-world proof that the output floor does exactly what it was built to do: it takes capital relief away from the banks that had modeled the most aggressively.

**Shrinking the balance sheet to hit a ratio.** After 2008, several troubled European banks faced regulators demanding higher capital ratios at a moment when raising fresh equity was almost impossible — their share prices were on the floor, so issuing shares meant catastrophic dilution. Their answer was to attack the denominator directly: announce multi-year programs to cut hundreds of billions of RWA by winding down trading books, selling non-core lending portfolios, and exiting whole business lines. Deutsche Bank, for instance, ran exactly this kind of multi-year RWA-reduction and "non-core unit" wind-down through the 2010s, shedding large blocks of risk-weighted assets to lift its CET1 ratio without a proportional equity raise. The honest reading is mixed: shrinking genuinely risky books *does* make a bank safer, so this is not pure cosmetics — but it also means the headline ratio improvement came from the bank getting *smaller*, not from the cushion getting *thicker* per dollar of remaining risk. The same maneuver can shade into the cosmetic when what gets sold is low-risk, capital-light business kept only because it bloated total assets. Always ask whether a ratio rose because the bank got safer or merely because it got smaller; they are not the same thing, and only the disclosures tell you which.

**Window-dressing the denominator at period-end.** Because the reported ratio is a snapshot on the last day of the quarter, banks have at times managed RWA *down* specifically around reporting dates — unwinding trades, parking exposures, or compressing positions just before the snapshot, then putting them back on after. Regulators have repeatedly flagged this "window dressing," particularly in repo and securities-financing books where positions can be moved off the balance sheet overnight. The leverage ratio, which keys off total exposure, is the measure most vulnerable to this trick, which is one reason supervisors moved toward *average* (rather than period-end) exposure measurement for the leverage ratio. The lesson for a reader: a ratio is a point estimate on a chosen day, and the chosen day is not random. Comparing period-end ratios to intra-period averages, where disclosed, occasionally reveals a bank that looks tidier on the 31st than it does on the 15th.

**Reading the RWA walk in a Pillar 3 report.** Every large bank discloses, in its quarterly Pillar 3 report, an "RWA flow" or "RWA walk" that reconciles the change in RWA over the period into components: asset size and book growth, model and methodology updates, asset-quality migration (borrowers up- or down-graded), acquisitions and disposals, and FX. This table is where the denominator game becomes visible. A bank whose CET1 ratio rose this quarter while its RWA fell, and whose RWA walk shows a large negative "model and methodology" line, did not get safer — it remodeled. A bank whose RWA fell on the "asset size" line shrank its book; one whose RWA rose on "asset-quality migration" is watching its borrowers get downgraded, which is an early-warning sign worth heeding. Analysts who know to read this line are never surprised by a ratio that turns out to be hollow, and they catch deterioration a quarter or two before it reaches the headline.

## The takeaway: how to read a capital ratio like a skeptic

Strip everything down and one idea remains: a capital ratio is a fraction, and the bank controls a great deal of the denominator. The numerator — capital — is hard, audited, and roughly comparable across banks. The denominator — risk-weighted assets — is partly a regulatory rulebook and partly a modeling choice the bank makes about its own risk, with every incentive to make that choice optimistically. So the ratio is only as trustworthy as the denominator under it, and the denominator is the softest number in banking.

That gives you a concrete way to read any bank's capital disclosure. Never accept a capital ratio alone. Pair it with three companions. First, the **RWA density** (RWA / total assets) — it tells you how much of the balance sheet the bank admits is risky, and a suspiciously low density (low 20s for anything other than a pure mortgage lender) is a flag that the models are doing heavy lifting. Second, the **leverage ratio** (capital / total assets) — the un-weighted cross-check that cannot be gamed; if it is thin while the risk-weighted ratio is fat, the gap is the story. Third, the **RWA walk** in the Pillar 3 report — it tells you *why* the ratio moved, and a ratio that rose on a shrinking denominator is a different animal from one that rose on retained earnings.

A practical habit ties these together. Whenever you read a capital ratio, do the one-line multiplication: ratio × density gives you the leverage ratio, the number no model can flatter. If a bank's slide proudly shows 13% CET1 but doesn't mention density, find density in the Pillar 3 (RWA divided by total assets) and run the product yourself. If the answer is a leverage ratio comfortably above 5–6%, the headline is probably honest. If the product lands near 3–4% despite a fat risk-weighted number, you have found a balance sheet that is far thinner than it advertises — and you found it with grade-school arithmetic, not a risk model. Two banks both shouting "13% CET1" can sit on opposite sides of that line. The whole reason this post exists is that the shout is not the answer; the denominator is.

The reason all of this matters traces straight back to the spine of this series. A bank is a leveraged machine running on a thin equity cushion, and it lives or dies by whether that cushion absorbs losses faster than they arrive. The capital ratio is our best single attempt to measure how thick the cushion is relative to the danger — but it measures the danger using a yardstick the bank helped calibrate. The standardized approach, the leverage ratio, and the output floor are all the same admission in different forms: that we cannot fully trust a bank's own measurement of its own risk, so we bolt on crude, un-gameable floors underneath the sophisticated number. When you look at a bank, do what the regulators learned to do the hard way: respect the risk-weighted ratio, but never let it out of the room without its un-weighted chaperone. The denominator is where the truth hides, and the truth is usually a little thinner than the headline.

## Further reading and cross-links

- [Basel I, II, III and the capital rules that govern every bank](/blog/trading/banking/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank) — the capital stack (CET1 / Tier 1 / Tier 2), the buffers, and the full evolution of the rules whose denominator this post dissects.
- [Bank capital and leverage: why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) — the numerator side: what capital is, why a few percent of equity backs the whole balance sheet, and how leverage amplifies a small loss into insolvency.
- [Loan pricing: cost of funds, risk premium, and the capital charge](/blog/trading/banking/loan-pricing-cost-of-funds-risk-premium-and-the-capital-charge) — where PD, LGD, and EAD come from and how the capital charge on a single loan flows back to RWA and pricing.
- [BIS and Basel bank regulation](/blog/trading/finance/bis-and-basel-bank-regulation) — the system-level view of who writes these rules, why they are global, and how the Basel framework keeps evolving.

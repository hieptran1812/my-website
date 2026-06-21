---
title: "Basel I, II, III and the Capital Rules That Govern Every Bank"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "A from-zero deep dive into the Basel capital framework: what regulatory capital really is, how the CET1, AT1 and Tier 2 stack absorbs losses, what the buffers and minimum ratios demand, and how the rules played out in the Credit Suisse AT1 wipeout."
tags: ["banking", "basel", "capital-requirements", "cet1", "at1", "tier-2", "regulation", "risk-weighted-assets", "coco-bonds", "bank-capital"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — The Basel framework is the global rulebook that tells every bank how much loss-absorbing capital it must hold against its risks, and it has spent forty years getting steadily stricter because the last version was always too lenient when the next crisis hit.
>
> - Capital comes in tiers ranked by how reliably they absorb loss: **CET1** (common equity, the gold standard) takes losses first and always; **AT1** (perpetual CoCo bonds) converts or writes off at a trigger; **Tier 2** (subordinated debt) only absorbs at the point of failure.
> - The headline minimum is a **CET1 ratio of 4.5% of risk-weighted assets**, but the buffers stacked on top mean a big global bank effectively needs roughly **8.5% to 11% CET1** before it can pay dividends freely.
> - The **leverage ratio** (3% of total exposure, regardless of risk weighting) is a backstop that catches banks that game the risk-weighting in the main rules.
> - In March 2023, Credit Suisse's regulator wrote **CHF 16 billion** of AT1 bonds to zero while shareholders still received **CHF 3 billion** from UBS — a vivid, contested demonstration of exactly where these instruments sit in the loss queue.

In March 2023, bondholders who thought they understood the pecking order of a failing bank got a brutal lesson. Credit Suisse, a 167-year-old pillar of global finance, was collapsing. Switzerland's regulator engineered an emergency takeover by UBS over a single weekend. And in the small print of that deal, the regulator did something that shocked the entire fixed-income world: it wrote **CHF 16 billion** of a specific kind of bank bond — Additional Tier 1, or AT1 — all the way down to zero, while the bank's *shareholders*, who normally sit beneath bondholders in any wind-up, walked away with **CHF 3 billion** from UBS.

To most people that sounds like a violation of the most basic rule of finance: debt gets paid before equity. But it wasn't a violation at all. It was the Basel capital framework working exactly as written. Those AT1 bonds had a clause buried in their contract that said, in effect, "if the regulator declares this bank non-viable, you, the bondholder, can be wiped out before the shareholders are." The buyers had been paid a fat coupon for years precisely *because* they were taking that risk. Most of them just hadn't read the fine print.

This post is about that fine print — the global rulebook that decides how much capital a bank must hold, what counts as capital, and who absorbs the losses when things go wrong. It is one of the most consequential, least understood systems in modern finance, and once you see how the pieces fit, the headlines about CET1 ratios, buffers, and bondholder wipeouts stop being jargon and start telling you a clear story about whether a bank can survive a bad year.

![Capital stack layers CET1 minimum buffers AT1 and Tier 2 loss queue](/imgs/blogs/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank-1.png)

The diagram above is the mental model for the whole post. A bank's required capital is built in layers. At the bottom sits the hardest, first-loss capital — common equity, called CET1 — with a minimum of 4.5% of the bank's risk-weighted assets. On top of that the regulators pile *buffers* you must hold to operate freely. And behind common equity, in the loss queue, sit two weaker tiers of capital — AT1 and Tier 2 — that absorb losses only after the equity is gone. Everything in this post is an elaboration of that one figure.

## Foundations: what capital is, why a bank needs rules about it, and the vocabulary

Before any Basel number makes sense, we need to build the idea of bank capital from zero. Let's start with an everyday example and then add the financial machinery one layer at a time.

### Capital is the owner's stake, not a pile of cash

The single most common misunderstanding about bank capital is that it's a vault of money the bank keeps in reserve "just in case." It is not. *Capital* is the portion of a bank's assets that is funded by its owners rather than borrowed from someone else. It's an accounting relationship, not a stockpile.

Take a corner shop. The owner buys \$100,000 of inventory and equipment. She puts in \$10,000 of her own savings and borrows \$90,000 from the bank. Her **capital** — her equity stake — is \$10,000. That \$10,000 isn't sitting in a drawer; it's been spent on inventory. But it's the *cushion*: if the shop's assets lose \$8,000 of value, she absorbs that loss and still owes the full \$90,000. Her equity shrinks to \$2,000, but the lender is untouched. Only if losses exceed \$10,000 does the lender start to take a hit.

A bank works the same way, just with bigger numbers and a different funding mix. Its assets are loans and securities. Most of the funding for those assets is borrowed — overwhelmingly from depositors, plus some bonds and interbank borrowing. A thin slice is the owners' stake: **equity capital**. That slice is what stands between the bank's losses and its depositors. When a loan goes bad, the loss eats into equity first. As long as the equity cushion is bigger than the losses, depositors are made whole and the bank survives. When losses blow through the equity, the bank is insolvent — it owes more than it owns — and someone (depositors, the deposit insurer, the taxpayer) eats the difference.

That is the entire reason capital rules exist. A bank is, as this series keeps insisting, a **leveraged, confidence-funded maturity-transformation machine**: it borrows short and lends long, runs on a sliver of equity, and survives only as long as that sliver absorbs losses faster than they arrive. Left to themselves, banks would run on the thinnest equity they could get away with, because thin equity juices the return on that equity — until a bad year wipes them out and takes the depositors' money with it. Capital rules are society's way of forcing a minimum cushion so that this doesn't happen so often.

If you want the full intuition for why equity is the thin cushion and how leverage amplifies both returns and ruin, the companion post [bank capital and leverage: why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) builds it brick by brick. Here we'll take that as given and focus on the *rules*.

### Risk-weighted assets: the denominator that makes the rule risk-sensitive

Here's the first subtlety. You can't just say "hold 8% capital against your assets," because not all assets are equally risky. A \$1 million loan to a blue-chip government is far safer than a \$1 million unsecured loan to a struggling startup. If you charged the same capital against both, you'd push banks toward the riskier loan — same capital cost, higher yield.

So Basel measures capital not against raw assets but against **risk-weighted assets (RWA)**. Each asset gets a *risk weight* reflecting how likely it is to lose money. Roughly: cash and top-rated government bonds get a 0% weight (they count as zero RWA), a well-secured residential mortgage might get 35%, an ordinary corporate loan 100%, and a high-risk exposure can be weighted above 100%. You multiply each asset by its weight, add them up, and that's your RWA — the denominator for almost every capital ratio in this post.

A worked feel for it: a bank with \$100 billion of total assets — say \$20 billion in government bonds (0% weight), \$30 billion in mortgages (35%), and \$50 billion in corporate loans (100%) — has RWA of \$0 + (\$30bn × 0.35) + \$50bn = \$10.5bn + \$50bn = **\$60.5 billion**. So a 10% capital ratio here means \$6.05 billion of capital against \$100 billion of actual assets — about 6% of real assets, but 10% of *risk-weighted* assets. The whole machinery of how risk weights are computed (standardized formulas versus banks' own internal models, and the games that get played in the denominator) is a deep topic in its own right, covered in [risk-weighted assets and how capital ratios really work](/blog/trading/banking/risk-weighted-assets-and-how-capital-ratios-really-work). For now, just hold onto this: **a capital ratio is capital ÷ risk-weighted assets.**

### The three tiers of capital, ranked by how reliably they absorb loss

Not all capital is equally good at its job. The job of capital is to absorb losses *while the bank keeps operating* — what regulators call a "going-concern" basis. Common equity does this perfectly: when a loan goes bad, equity just falls, no permission needed, no maturity to worry about, no coupon the bank is obliged to pay. Other instruments absorb loss only partially, or only at the very end. So Basel ranks capital into tiers:

- **Common Equity Tier 1 (CET1)** — common shares plus retained earnings, minus a few deductions. This is the purest, hardest capital. It has no maturity (it never has to be repaid), no obligatory dividend (the bank can cut the dividend to zero without defaulting), and it absorbs losses first and continuously. CET1 is the number the market, the regulators, and the rating agencies actually watch.
- **Additional Tier 1 (AT1)** — perpetual bonds with a special trick: they convert into equity or get written down when the bank's CET1 falls below a trigger. These are the famous "CoCos" (contingent convertibles). They count toward Tier 1 capital along with CET1, but they're weaker, because they only absorb loss once a trigger is hit. CET1 + AT1 = **Tier 1 capital**.
- **Tier 2** — subordinated debt with a remaining maturity of at least five years. It absorbs losses only at the *point of non-viability* — essentially, once the bank has failed. It's a gone-concern cushion, not a going-concern one. Tier 1 + Tier 2 = **Total capital**.

We'll dissect each tier in detail later, but this ranking — CET1 best, AT1 in the middle, Tier 2 weakest — is the spine of the capital stack.

### The minimum ratios and the buffers

Basel III sets three core minimum ratios, all as a percentage of RWA:

- **CET1 ≥ 4.5%** of RWA
- **Tier 1 ≥ 6.0%** of RWA (so CET1 plus at most 1.5% of AT1)
- **Total capital ≥ 8.0%** of RWA (so Tier 1 plus at most 2% of Tier 2)

But those are floors. On top of them sit **buffers** — additional CET1 you must hold to operate without restrictions:

- **Capital conservation buffer: 2.5%** of RWA, always required.
- **Countercyclical buffer: 0% to 2.5%**, switched on by national regulators when credit is booming, to lean against the cycle.
- **G-SIB surcharge: 1% to 3.5%**, an extra charge on globally systemically important banks (the JPMorgans and HSBCs) because their failure would do the most damage.

Buffers are not hard minimums — a bank *can* dip into them — but dipping in triggers automatic restrictions on dividends, buybacks, and bonuses. So in practice banks treat the top of the buffer stack as their real operating floor. We'll see exactly how that tripwire works.

### The leverage ratio: a backstop with no risk-weighting

Finally, because risk-weighting can be gamed, Basel III added a **leverage ratio** as a backstop: Tier 1 capital ÷ *total exposure* (essentially all assets, plus some off-balance-sheet items, with no risk weights at all). The Basel minimum is **3%**, with bigger banks (US global banks under the enhanced supplementary leverage ratio) facing 5% to 6%. The leverage ratio doesn't care how safe you *say* your assets are; it just asks, "how much real stuff are you standing on top of per dollar of capital?" When a bank has piled into assets it claims are nearly riskless, the leverage ratio is the rule that bites.

That's the vocabulary. Now let's see how this system was built, one crisis at a time.

## How the rules were built: Basel I, II, III, and the endgame

The Basel framework is named after Basel, Switzerland, home of the Bank for International Settlements (BIS) — the "central bank for central banks." The rules are written by the Basel Committee on Banking Supervision, a club of the world's major bank regulators. Crucially, the Basel rules are not law. The committee writes a standard; each country then translates it into its own legislation (the EU's Capital Requirements Regulation, the US's capital rules, and so on). That's why implementation dates and details differ across jurisdictions. For the system-level view of how the BIS and Basel fit into global bank regulation, see [BIS and Basel: how banks are regulated globally](/blog/trading/finance/bis-and-basel-bank-regulation). Here we go deep on the *mechanics* of each accord and why each one was written.

![Basel timeline from 1988 accord through the 2008 crisis to the 2023 endgame](/imgs/blogs/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank-4.png)

The timeline above is the story in one line: each accord was a fix for the failure the previous one allowed. Let's walk it.

### Basel I (1988): one crude number

Before 1988, there was no international agreement on how much capital a bank needed. Different countries had wildly different standards, which meant a bank in a lax jurisdiction could undercut everyone else by running on thin equity. The first Basel Accord, agreed in 1988 and phased in by 1992, fixed that with a single, deliberately simple rule: a bank must hold capital equal to at least **8% of its risk-weighted assets**.

Basel I introduced the two ideas we still use: risk-weighting (with just four crude buckets — 0%, 20%, 50%, 100%) and tiers of capital (a Tier 1 and a Tier 2, with Tier 1 having to be at least half the total). It was a triumph of getting *something* agreed across borders. But it was crude. A loan to a blue-chip multinational and a loan to a near-bankrupt firm both sat in the 100% bucket and attracted the same capital. There was no recognition of credit quality within a bucket, no charge for the risk in a bank's trading book at first, and no concept of operational risk at all. Banks quickly learned to *arbitrage* the buckets — keep the riskiest 100%-weighted assets (highest yield for the same capital), and offload the safe ones. The rule meant to make banks safer was nudging them toward risk.

The committee patched the most glaring hole in **1996** with the Market Risk Amendment, which added a capital charge for the trading book — the portfolio of securities and derivatives a bank holds to trade, not to hold to maturity. That amendment introduced something quietly radical: it let banks use their own **Value-at-Risk (VaR) models** to size the market-risk charge. It was the first time the framework handed the measurement of risk to the regulated bank's own models, and it set the template — and the trap — that Basel II would then extend across the whole balance sheet. The lesson of Basel I's crude simplicity was real, but the cure (let the banks model it) planted the seed of the next failure. That tension — between a rule simple enough to be gameable and a rule sophisticated enough to be gamed — runs through every accord that follows.

#### Worked example: a CET1 ratio from scratch

Let's compute the central number of this whole framework — a CET1 ratio — using friendly figures, because everything else builds on it.

Suppose a bank has **\$60 billion of risk-weighted assets** (the same denominator we built in the Foundations section). Its balance sheet shows \$5.4 billion of common shares and retained earnings. After the regulatory deductions (Basel makes you subtract things like goodwill and certain deferred tax assets, because they wouldn't actually absorb losses), assume \$5.1 billion of that qualifies as CET1.

The CET1 ratio is simply:

$$\text{CET1 ratio} = \frac{\text{CET1 capital}}{\text{RWA}} = \frac{\$5.1\text{bn}}{\$60\text{bn}} = 8.5\%$$

So this bank has a CET1 ratio of **8.5%**. Against the 4.5% minimum that looks generous — almost double. But hold that thought: once we stack the buffers on top, 8.5% turns out to be roughly the *real* operating floor for a big bank, not a comfortable surplus. The one-sentence intuition: a capital ratio is just the owners' first-loss cushion expressed as a fraction of the risk the bank is carrying, and the whole Basel system is an argument about how big that fraction must be.

### Basel II (2004): risk-sensitivity, three pillars, and the seeds of the crisis

Basel II, finalized in 2004, was an attempt to fix Basel I's crudeness by making capital genuinely *risk-sensitive*. Its great innovation was to let sophisticated banks use their own **internal models** to estimate the riskiness of their assets — the Internal Ratings-Based (IRB) approach — rather than the regulator's standardized buckets. A bank could now say, in effect, "our models show these mortgages are very safe, so we should hold less capital against them," and, subject to supervisory approval, do so.

Basel II was organized around **three pillars**, a structure that survives today:

- **Pillar 1: minimum capital requirements** — the formulas for credit, market, and (newly) operational risk.
- **Pillar 2: supervisory review** — regulators assess each bank's risks and can demand *more* capital than Pillar 1 implies (the bank-specific add-on).
- **Pillar 3: market discipline** — mandatory disclosure, so the market can see each bank's risk profile and impose its own discipline.

The intention was sound: align capital with actual risk. The execution was a disaster of timing and incentives. By letting banks model their own risk weights, Basel II handed them a powerful lever to *reduce* the very RWA denominator that capital ratios divide by. A bank that could shave its modeled risk weights could report a healthier capital ratio without raising a dollar of new capital. Across the mid-2000s, risk weights drifted down and reported ratios looked reassuring — right up until 2007–08, when the assets everyone's models had blessed as low-risk (AAA-rated mortgage securities, in particular) turned out to be anything but.

When the crisis hit, two things became horribly clear. First, banks simply did not have enough capital — and what they did have was often low quality, padded with instruments that didn't really absorb loss. Second, the system had no answer for liquidity: even solvent-looking banks died when their short-term funding evaporated. Basel II had made capital *sound* risk-sensitive while leaving it *too thin, too low-quality, and silent on liquidity*. Those three failures defined the next accord.

### Basel III (2010–2019): quality, quantity, buffers, and liquidity

Basel III, agreed in stages from 2010 and phased in over the following decade, was the most sweeping rewrite. It attacked each Basel II failure directly.

**Quality.** Basel III ruthlessly redefined what counts as capital. It elevated common equity — CET1 — to the centre and demanded that the bulk of required capital be CET1, the kind that actually absorbs losses while a bank keeps running. Hybrid instruments that had counted as capital under Basel II but failed to absorb loss in the crisis were squeezed out or recategorized. The clean tiering we use today — CET1, AT1, Tier 2 — is a Basel III construction.

**Quantity.** The minimums rose in substance. The headline CET1 minimum became 4.5% of RWA, Tier 1 became 6%, total capital stayed at 8% — but the real lift came from the **buffers** stacked on top.

**Buffers.** Basel III introduced the capital conservation buffer (2.5%), the countercyclical buffer (0–2.5%), and the G-SIB surcharge (1–3.5%). These transformed the effective requirement, as we'll quantify in a moment.

**A leverage backstop.** Recognizing that risk-weighting had been gamed, Basel III added the non-risk-weighted leverage ratio (3% minimum) as a floor that no amount of model wizardry could lower.

**Liquidity, for the first time.** Basel III added two liquidity rules — the Liquidity Coverage Ratio (LCR) and the Net Stable Funding Ratio (NSFR), both with a 100% minimum — to make sure a bank could survive a funding squeeze, not just a solvency hit. Those rules are their own deep topic, covered in [liquidity management: LCR, NSFR and the liquidity buffer](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer); the key Basel-III insight is simply that capital and liquidity are *different* problems, and a bank can be solvent and still die of thirst.

![Bar chart of Basel III minimum ratios CET1 Tier 1 total capital conservation buffer and leverage](/imgs/blogs/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank-3.png)

The chart above lays out the core Basel III numbers side by side: the 4.5% CET1 minimum, the 6% Tier 1 minimum, the 8% total-capital minimum, the 2.5% conservation buffer that sits on top, and the 3% leverage-ratio backstop (which is measured against total exposure, not RWA — a different denominator, hence its own colour). Notice that the leverage ratio of 3% looks small next to the 8% total-capital number, but it's not comparable: 3% of *everything* can bite harder than 8% of *risk-weighted* assets when a bank has loaded up on assets it claims are low-risk.

### The 2023 endgame: putting a floor under the models

Even Basel III left one big loophole open: banks using internal models could still report dramatically lower risk weights than banks using the standardized approach for the same exposures. Two banks could hold identical assets and report capital ratios that differed by a wide margin, purely because one had a friendlier model. That undermined the whole point of a common standard, and it made capital ratios hard to compare across banks and countries.

The final piece of Basel III — agreed in 2017 and often called the **"Basel III endgame"** or, in Europe, simply the finalization — closes that gap with an **output floor**. The rule: a bank's total RWA, computed using its own internal models, cannot fall below **72.5%** of what the RWA would be under the standardized approach. Put concretely, internal models can save you at most about a quarter of your capital requirement relative to the standard formulas; beyond that, the floor catches you. The endgame also overhauled the standardized approaches for credit and operational risk and tightened the rules for market risk (the Fundamental Review of the Trading Book, FRTB).

Implementation has been politically contentious and staggered. Different jurisdictions are phasing the endgame in across roughly 2023–2028, and the calibration (especially in the US, where the original proposal was scaled back after fierce industry pushback) has been heavily debated. The direction of travel, though, is unmistakable: from one crude number, to risk-sensitivity, to quality-and-buffers, to a hard floor under the cleverness. Every step traded simplicity for robustness, written in the language of the last crisis.

#### Worked example: building the effective CET1 demand for a G-SIB

Here's where the buffers transform the headline number. The 4.5% CET1 minimum is almost never the binding constraint for a large bank. Let's build the *effective* demand for a globally systemic bank, taking the components straight from the framework.

Start with the components:

- CET1 minimum: **4.5%** of RWA
- Capital conservation buffer: **+2.5%** of RWA
- G-SIB surcharge (a representative mid-bucket bank): **+1.5%** of RWA

Add them up:

$$4.5\% + 2.5\% + 1.5\% = 8.5\% \text{ of RWA}$$

So a typical G-SIB needs about **8.5% CET1** before it can distribute profits without restriction — and that's before any countercyclical buffer is switched on (which could add up to another 2.5%) and before any Pillar 2 add-on the supervisor demands. For the most systemic banks, with surcharges near the top of the 1–3.5% range and a live countercyclical buffer, the effective CET1 demand can run to **11% or more**.

![Stacked bar of G-SIB CET1 demand 4.5 percent minimum plus 2.5 buffer plus 1.5 surcharge](/imgs/blogs/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank-2.png)

The stacked bar above shows exactly this build: the 4.5% floor (blue), the 2.5% conservation buffer (amber), and a 1.5% surcharge (lavender) sum to 8.5% of RWA. The one-sentence intuition: the headline 4.5% is a floor you must never touch, while the buffers above it are the real operating range — which is why a big bank reporting "12% CET1" is comfortable but a bank reporting "8% CET1" is already in the danger zone, even though 8% is well above the stated minimum.

## The capital stack in detail: CET1, AT1, and Tier 2

Now let's go deep on the three tiers, because the differences between them are where the most consequential — and most misunderstood — mechanics live.

### CET1: the gold standard

Common Equity Tier 1 is what a bank is really made of, from a safety point of view. It consists of:

- **Common shares** issued by the bank, and
- **Retained earnings** — the profits the bank kept rather than paying out,
- plus other comprehensive income and a few qualifying reserves,
- **minus regulatory deductions**: goodwill, other intangibles, certain deferred tax assets, and a few other items that wouldn't actually absorb a loss in a crunch.

Three properties make CET1 the gold standard. It is **permanent** — there's no maturity date, no obligation ever to repay it. It is **fully discretionary** — the bank can cut or skip the dividend entirely without defaulting on anything. And it is **first-loss and continuous** — when the bank loses money, CET1 falls dollar for dollar, immediately, with no trigger and no permission required. Because of these properties, CET1 absorbs losses on a true going-concern basis: the bank keeps operating while the cushion shrinks. This is why every serious analysis of a bank's strength leads with the CET1 ratio. It's the only capital number the market fully trusts.

The regulatory **deductions** are worth dwelling on, because they're where reported book equity and regulatory CET1 part ways, and the gap can be large. Basel makes a bank subtract, from its raw common equity, anything that wouldn't actually be there to absorb a loss in a crunch. The big deductions are:

- **Goodwill and other intangibles.** When a bank pays more than book value to acquire another bank, the excess is recorded as goodwill — an accounting asset with no resale value in a fire sale. It can't absorb a loss, so Basel deducts it from CET1 entirely. A bank that's grown by expensive acquisitions can carry tens of billions of goodwill that simply vanishes from its regulatory capital.
- **Deferred tax assets that rely on future profitability.** A DTA is, roughly, a tax refund the bank can claim *if it makes money later*. A failing bank won't make money later, so the part of a DTA that depends on future profits is deducted (above a threshold), because it's worthless exactly when you'd need it.
- **Significant investments in other financial institutions** and a few other items, above set thresholds, to stop capital being double-counted across the system (one bank's capital propping up another's).

The practical upshot: a bank can report, say, \$70 billion of common shareholders' equity on its balance sheet but only \$58 billion of regulatory CET1 after deductions. When you read a bank's capital, the number that matters is the *post-deduction* CET1, not the headline book equity — the two are not the same, and the difference is precisely the stuff that wouldn't show up when you needed it.

### AT1: the contingent middle tier

Additional Tier 1 is the strange, hybrid middle. AT1 instruments — almost always **contingent convertible bonds, or "CoCos"** — are designed to look like bonds in good times and behave like equity in bad times. They are:

- **Perpetual** — no maturity date (though typically callable by the bank after five years).
- **Loss-absorbing on a trigger** — when the bank's CET1 ratio falls below a contractual trigger (commonly **5.125%**, sometimes 7%), the bond either *converts into common shares* or is *written down* (partially or fully), instantly boosting the bank's equity by reducing its debt.
- **Discretionary on coupons** — the bank can skip the coupon payments without triggering a default, much like it can skip a dividend.

The appeal to the bank is obvious: AT1 counts toward Tier 1 capital (it can fill up to 1.5% of the 6% Tier 1 minimum) but is cheaper than issuing common shares, and in good times it doesn't dilute existing shareholders. The appeal to investors is the fat coupon — AT1 yields are high precisely because the investor is taking equity-like risk for bond-like cash flows. The danger, as Credit Suisse holders discovered, is that the contract can let the regulator wipe out AT1 *before* common equity in a non-viability event, depending on how the specific bond and the local resolution law are written. AT1 is the tier where the fine print matters most, and where investors most often misjudge their place in the loss queue.

#### Worked example: an AT1 trigger and conversion

Let's make the AT1 mechanism concrete. Take our bank with **\$60 billion of RWA**. Suppose it has \$5.1 billion of CET1 (an 8.5% ratio, as before) and **\$1.2 billion of AT1** CoCo bonds outstanding, with a contractual conversion trigger at a CET1 ratio of **5.125%**.

Now the bank takes a brutal year — say \$3.6 billion of losses on bad loans and write-downs. Those losses hit CET1 first:

$$\text{New CET1} = \$5.1\text{bn} - \$3.6\text{bn} = \$1.5\text{bn}$$
$$\text{New CET1 ratio} = \frac{\$1.5\text{bn}}{\$60\text{bn}} = 2.5\%$$

The CET1 ratio has crashed to 2.5%, far below the 5.125% trigger. The CoCos now activate: the \$1.2 billion of AT1 either converts into new common shares or is written down. Suppose it converts in full. The bank's CET1 jumps back up:

$$\text{CET1 after conversion} = \$1.5\text{bn} + \$1.2\text{bn} = \$2.7\text{bn}$$
$$\text{CET1 ratio after conversion} = \frac{\$2.7\text{bn}}{\$60\text{bn}} = 4.5\%$$

The conversion has dragged the bank back to the 4.5% minimum — still wounded, but no longer in free-fall. The AT1 holders have absorbed \$1.2 billion of loss (their bonds became shares in a battered bank, or vanished altogether), exactly as the instrument was designed to do. The one-sentence intuition: AT1 is loss-absorbing capital that *sleeps as a bond until a trigger wakes it as equity* — which is why it pays more than senior debt and why its holders are, whether they realise it or not, standing much closer to the fire than ordinary bondholders.

### Tier 2: the gone-concern cushion

Tier 2 is the weakest tier. It's **subordinated debt** — bonds that rank below the bank's depositors and senior creditors but above shareholders in a wind-up — with a remaining maturity of at least five years (and it amortizes out of capital recognition in its final five years, losing 20% of its eligibility per year). Tier 2 absorbs losses only at the **point of non-viability**: essentially, once the bank has failed and is being resolved. It's a *gone-concern* cushion, there to protect the deposit insurer and senior creditors in a wind-up, not to keep the bank alive. Tier 2 can fill up to 2% of the 8% total-capital minimum. Because it only bites at the very end, Tier 2 is cheaper than AT1 and far cheaper than equity — but it does very little for the bank's day-to-day resilience.

There's a related layer that sits just outside the capital stack but is built from the same logic: **TLAC** (Total Loss-Absorbing Capacity), and its European cousin **MREL**. After 2008, regulators wanted to be able to wind down a failing G-SIB *without* taxpayer money, by imposing losses on a deep enough pool of capital and bail-in-able debt. TLAC requires the biggest banks to hold loss-absorbing instruments — CET1, AT1, Tier 2, *plus* a layer of senior bail-in-able debt — totalling roughly **18% of RWA** (and at least 6.75% of the leverage exposure). The idea is that if a G-SIB fails, the resolution authority can write down or convert this whole stack to recapitalise the bank overnight, keeping the critical functions running while the losses fall on investors who signed up for the risk, not on depositors or the state. TLAC is the framework's answer to "too big to fail": it doesn't stop a giant bank from failing, but it tries to make the failure survivable without a bailout. For the deep treatment of the deposit-insurance and lender-of-last-resort safety net that backstops all of this, see the series' coverage of deposit insurance and moral hazard.

![Pipeline of the loss queue losses hit CET1 first then AT1 converts then Tier 2 absorbs](/imgs/blogs/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank-7.png)

The pipeline above is the loss queue in motion. A loss arrives and eats into CET1 first, dollar for dollar. If CET1 falls to the AT1 trigger, the CoCos convert or write off, refilling equity. Only if the bank reaches the point of non-viability — the regulator declares it can't survive — does Tier 2 get written down. The strict ordering is the whole point: each tier is a different line of defence, deployed in sequence, and the price each instrument pays (its coupon) reflects exactly how far back in the queue it sits.

![Matrix comparing CET1 AT1 and Tier 2 on loss absorption trigger and permanence](/imgs/blogs/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank-8.png)

The matrix above pins down the differences that the loss queue glosses over. Read across the rows: CET1 absorbs first and always, needs no trigger, and is permanent. AT1 converts or writes off at a CET1 trigger or a non-viability call, is perpetual but callable, and lets the bank skip coupons. Tier 2 absorbs only at failure, is dated debt, and amortizes in its final years. Read down the columns and you can see why CET1 is the only tier the market treats as fully reliable: it's the only one that is permanent *and* first-loss *and* trigger-free.

## The buffers in detail: the tripwire that caps payouts

The buffers deserve their own deep treatment, because they're where the rules actually shape bank behaviour day to day. The minimums (4.5% / 6% / 8%) are hard floors — breach them and you're in resolution territory. But the buffers are different: they're a *zone of restriction*, not a cliff. A bank can operate inside its buffers; it just loses the freedom to pay out profits.

### The mechanism: the combined buffer and the Maximum Distributable Amount

Stack the buffers on top of the 4.5% CET1 minimum and you get the **combined buffer requirement**. For a non-systemic bank with only the conservation buffer, that's 4.5% + 2.5% = a **7% threshold**. For a G-SIB with a surcharge and a live countercyclical buffer, it's higher.

When a bank's CET1 dips below that combined threshold — into the buffer zone — an automatic constraint kicks in: the **Maximum Distributable Amount (MDA)**. The MDA caps the share of profits the bank may pay out as dividends, buybacks, or discretionary staff bonuses. The cap tightens the deeper you go into the buffer: the buffer is split into quartiles, and a bank in the *lowest* quartile of its buffer can distribute essentially **0%** of its profits; the next quartile up allows 20%, then 40%, then 60%. The idea is to force a struggling bank to *retain* its earnings and rebuild capital rather than shovelling it out to shareholders.

This is one of the most elegant features of the post-2008 framework. Before Basel III, a bank limping toward trouble could keep paying dividends to reassure the market — right up until it collapsed, having bled out the very capital it needed. The MDA makes capital rebuild *automatic and non-negotiable* once a bank enters its buffer.

![Before and after comparison of a bank above the buffer versus inside the buffer with payouts capped](/imgs/blogs/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank-5.png)

The before-and-after above shows the tripwire. On the left, a bank with a 12% CET1 ratio sits comfortably above the 7% combined threshold, with five points of headroom, and pays dividends and buybacks freely. On the right, the same bank after losses has fallen to a 6.5% CET1 ratio — *inside* the buffer. It's still above the 4.5% hard minimum, so it isn't in resolution. But it has tripped the MDA: in the lowest buffer quartile, its payouts are capped at zero. The dividend is frozen, the buyback is halted, and discretionary bonuses are off. The bank is being forced to heal.

#### Worked example: the MDA payout cap by quartile

Let's quantify the MDA tripwire, because the graduated cap is the bit people get wrong. Take a non-systemic bank whose combined buffer requirement is the standard **2.5%** conservation buffer sitting on the **4.5%** CET1 minimum — so the combined threshold is **7.0%**, and the buffer zone runs from 4.5% up to 7.0%, a band of 2.5 percentage points. The MDA splits that band into four equal quartiles of 0.625 points each:

- CET1 in **6.375%–7.0%** (top quartile of the buffer): may distribute up to **60%** of eligible profits.
- CET1 in **5.75%–6.375%**: up to **40%**.
- CET1 in **5.125%–5.75%**: up to **20%**.
- CET1 in **4.5%–5.125%** (bottom quartile): up to **0%** — nothing.

Now suppose our bank earns \$1 billion of distributable profit in a year and finds itself at a CET1 ratio of **6.0%** — that's in the second quartile from the top (5.75%–6.375%), where the cap is 40%. The maximum it may pay out is:

$$\text{MDA} = 40\% \times \$1\text{bn} = \$400\text{ million}$$

So \$600 million of that year's profit is *forced* to stay in the bank, rebuilding CET1, whether management likes it or not. Had the bank slipped to a 5.0% CET1 ratio (bottom quartile), the cap would be 0% and the entire \$1 billion would have to be retained. The one-sentence intuition: the MDA turns "rebuild your capital" from a polite supervisory request into an automatic, formula-driven seizure of the dividend — the deeper you sink into the buffer, the harder the rule clamps down on payouts, which is exactly why bank investors watch the *distance to MDA* almost as closely as the CET1 ratio itself.

#### Worked example: the leverage-ratio backstop binding

The leverage ratio usually sits in the background, but sometimes it — not the risk-based ratio — becomes the binding constraint. This is exactly what it's designed for: catching a bank that has loaded up on assets its models call near-riskless.

Take a bank that has gone all-in on government bonds and central-bank reserves, which carry a **0% risk weight**. Suppose it has **\$1,000 billion of total exposure** (real assets plus off-balance-sheet items) but, because so much of it is 0%-weighted, only **\$300 billion of RWA**. It holds **\$30 billion of Tier 1 capital**.

Check the risk-based ratio first:

$$\text{Tier 1 ratio} = \frac{\$30\text{bn}}{\$300\text{bn RWA}} = 10\%$$

That's a healthy 10% — comfortably above the 6% Tier 1 minimum. By the risk-based rule, this bank looks strong. Now check the leverage ratio:

$$\text{Leverage ratio} = \frac{\$30\text{bn}}{\$1{,}000\text{bn total exposure}} = 3.0\%$$

It's sitting *exactly* on the 3% Basel minimum — and if it were a US G-SIB facing a 5% enhanced supplementary leverage ratio, it would already be in breach. The risk-based rule says "you're fine"; the leverage backstop says "you're standing on a thousand billion of real assets with only thirty billion of capital, and we don't care how safe you think those assets are." The one-sentence intuition: the leverage ratio is the rule that doesn't believe your models — it catches the bank that has converted genuine thinness into a flattering risk-weighted ratio, which, not coincidentally, is the exact trap that broke Silicon Valley Bank, whose "safe" long-dated Treasuries were a 0%-credit-weight asset that nonetheless detonated.

## Common misconceptions

Capital rules generate more confusion than almost any other corner of banking. Here are the beliefs that trip people up most often, each corrected with a number.

**"Bank capital is cash the bank keeps in reserve."** No — this is the big one. Capital is a *funding* concept, not a stockpile. The 4.5% CET1 requirement doesn't mean the bank parks 4.5% of its assets in a vault; it means at least 4.5% of its risk-weighted assets are funded by the owners' equity rather than borrowed. The actual cash a bank keeps for day-to-day liquidity is a completely separate thing, governed by the liquidity rules (LCR and NSFR), not the capital rules. Confusing capital with cash is the single most common error people make about banks.

**"A higher capital ratio always means a safer bank."** Mostly true, but with a sharp caveat: the ratio is only as honest as its denominator. Two banks can both report a 12% CET1 ratio while holding very different real risk, if one has used internal models to shrink its RWA. That's precisely why Basel added the leverage ratio (which ignores risk weights) and the output floor (which caps how far models can shrink RWA). A 12% risk-based ratio paired with a 3% leverage ratio is a thinner bank than a 12% ratio paired with a 6% leverage ratio.

**"AT1 bonds are bonds, so they're safer than the bank's stock."** Dangerously wrong, and Credit Suisse holders learned it the hard way. AT1 (CoCo) bonds are explicitly designed to absorb losses, and under certain resolution frameworks they can be wiped out *before* common shareholders. In March 2023, CHF 16 billion of Credit Suisse AT1 was written to zero while shareholders received CHF 3 billion. The fat coupon on an AT1 bond is compensation for sitting near the front of the loss queue, not a free lunch.

**"Hitting the 4.5% CET1 minimum is fine — it's the rule, after all."** No. A bank at 4.5% CET1 is in deep trouble. It has blown clean through all its buffers (the combined threshold is at least 7%), which means its dividends, buybacks, and bonuses are frozen by the MDA, and it is one bad quarter from resolution. The minimum is a floor you must never approach, not a target you're allowed to sit on. Healthy big banks run with CET1 in the 12–15% range, leaving real headroom above the buffers.

**"The buffers are just extra minimums."** Not quite — and the difference is the whole point. Breaching a *minimum* (4.5% CET1) is a regulatory failure that pushes a bank toward resolution. Dipping into a *buffer* is permitted; it simply triggers automatic, graduated restrictions on payouts via the MDA. The buffers are deliberately *usable* — they exist to be drawn down in a stress so the bank keeps lending — whereas the minimums are bright lines. Treating them as the same thing misses the elegant design: buffers bend, minimums don't.

## How it shows up in real banks

The abstractions above became blunt reality in a handful of episodes. Here's the framework in action, with real dates and real numbers.

### Credit Suisse, March 2023: the AT1 wipeout

This is the definitive real-world demonstration of where AT1 sits in the loss queue — and a reminder that the fine print, not intuition, governs. Credit Suisse had spent a decade leaking trust through a parade of scandals and losses; by late 2022 depositors were fleeing, with roughly **CHF 110 billion** of outflows in the fourth quarter of 2022 alone. The Swiss National Bank threw a **CHF 100 billion** liquidity line at the problem, but liquidity support can't restore confidence once it's gone. Over the weekend of 18–19 March 2023, the authorities forced a takeover by UBS.

![Bar chart of Credit Suisse 2023 AT1 written off UBS price SNB liquidity line and deposit outflows in CHF billion](/imgs/blogs/basel-i-ii-iii-and-the-capital-rules-that-govern-every-bank-6.png)

The chart above shows the four numbers that defined the resolution. UBS paid about **CHF 3 billion** for the equity — far below book value, but not zero. And Switzerland's regulator, FINMA, invoked a clause in the AT1 contracts and Swiss emergency law to write the entire **CHF 16 billion** of AT1 CoCos down to zero. Shareholders got something; AT1 bondholders got nothing.

To bondholders trained on the normal hierarchy — depositors, then senior bonds, then subordinated bonds, then equity, with equity wiped first — this looked like an inversion of the natural order. It wasn't. The Credit Suisse AT1 prospectuses, and the Swiss legal framework, explicitly allowed a full write-down of AT1 in a "viability event" or when extraordinary government support was provided, without requiring shareholders to be zeroed first. The instruments did exactly what their contracts said. The episode taught the market two enduring lessons: first, AT1 is genuinely equity-like risk wearing a bond's clothing; and second, *resolution law varies by country*, so the same-looking CoCo can rank differently in Zurich than in Brussels. The AT1 market briefly seized, then reopened at higher yields — investors repricing a risk they'd always nominally been paid for but never quite believed. The full anatomy of the collapse is in [Credit Suisse 2023: the slow death of trust and the AT1 wipeout](/blog/trading/banking/credit-suisse-2023-the-slow-death-of-trust-and-the-at1-wipeout); here the point is narrower and sharper: the capital stack is a contract, and the contract is what binds.

### The buffers in 2020: built to be used

The early-2020 pandemic shock was, in a quiet way, a vindication of the buffer design. When economies locked down in March 2020, regulators feared banks would respond to looming loan losses by hoarding capital and choking off credit exactly when the economy needed it most. Their answer was to *let banks use the buffers*. The Bank of England, the European Central Bank, the US Federal Reserve, and others explicitly encouraged banks to draw down their capital conservation and countercyclical buffers to keep lending, and many regulators cut the countercyclical buffer back toward zero outright.

This is precisely what the buffers were built for. A buffer that can never be touched isn't a shock absorber; it's just a higher minimum. By signalling that dipping into the buffers was acceptable in a system-wide stress — and by pairing that with restrictions on dividends and buybacks to make sure the freed-up capital went to lending and resilience rather than shareholders — regulators turned the Basel III architecture into a counter-cyclical tool. Banks kept lending through 2020, the feared credit crunch didn't materialise on the scale once worried about, and the buffers refilled as conditions normalised. It was the rare case of a financial-stability rule working as designed under live fire.

### Basel II's risk-weight arbitrage: the slow-motion failure

The clearest illustration of *why* the rules kept tightening is Basel II's IRB models in the run-up to 2008. By letting banks model their own risk weights, Basel II created a quiet incentive to make the models optimistic. Across the mid-2000s, average risk weights on similar portfolios drifted lower, which flattered reported capital ratios without a dollar of new equity. The most notorious case was the treatment of AAA-rated structured credit — tranches of mortgage securities that the models, fed by benign historical data and rating-agency blessings, assigned tiny risk weights. Banks loaded up on assets that looked nearly capital-free.

When defaults came, those "low-risk" assets generated catastrophic losses against almost no capital. The lesson was that *risk-sensitivity is only as good as the risk model*, and a system that lets the regulated party calibrate its own model has a built-in optimism bias. That single failure runs straight through to the 2017 endgame's output floor: the 72.5% floor exists to cap exactly the model-driven RWA shrinkage that made the 2000s ratios a fiction. The arc from Basel II's discretion to the endgame's floor is the framework learning, expensively, that you cannot fully outsource the measurement of risk to the people holding the risk.

### The market reads the stack through AT1 spreads and the distance to MDA

The capital framework isn't only a regulatory exercise — it's a live market signal, and fixed-income investors trade on it daily. Because AT1 (CoCo) bonds convert or write off when CET1 hits a trigger, their price is exquisitely sensitive to how close a bank is sitting to its buffers. When a bank's CET1 ratio is comfortably high, its AT1 bonds trade at a modest spread over senior debt — investors are confident the trigger is remote. When trouble brews and the CET1 ratio drifts toward the buffer zone, AT1 spreads blow out, because the probability of conversion or write-off — and of skipped coupons — has just jumped.

That's why, in the months before Credit Suisse's collapse, its AT1 bonds were already trading at distressed levels: the market was pricing the rising odds of exactly what happened. Watching AT1 spreads, and a bank's **distance to MDA** (how many percentage points of CET1 stand between it and the buffer-zone payout cap), is one of the sharpest early-warning tools an analyst has. A bank can publish a reassuring annual report while its own AT1 bonds are screaming that the loss queue is about to start moving. When the two disagree, trust the bonds — they're the ones with money on the line at the precise point in the stack the framework governs. The capital rules, in other words, don't just sit in a regulator's filing cabinet; they're embedded in the price of every CoCo bond, turning the abstract loss queue into a number that updates by the second.

### Silicon Valley Bank, 2023: when the safe asset was the risk

SVB is usually told as a liquidity-and-duration story, and it is. But it's also a capital-rules story, and a cautionary one about what the rules *don't* catch. SVB had piled into long-dated US Treasuries and mortgage-backed securities — assets with a **0% credit risk weight**, because the US government won't default on them. By the risk-based rules, those holdings demanded essentially no capital. But "no credit risk" is not "no risk": when rates rose sharply in 2022, the market value of those long bonds collapsed, leaving SVB sitting on roughly **\$17 billion** of unrealised losses on its securities — losses that, in the US framework for a bank of its size, didn't fully flow through its reported capital ratios.

The risk-based capital rules, by design, ignored the interest-rate risk that actually killed SVB, because credit risk weighting is blind to it. The leverage ratio would have caught more of it (it counts the real assets regardless of weight), and the interest-rate-risk-in-the-banking-book framework was supposed to flag it — but the binding constraints for SVB's regulatory category had been loosened in the years before. The episode is a reminder that capital ratios measure the risks the rules are looking at, and a bank can die of a risk the rules aren't weighting. The deep mechanics of how a duration mismatch detonates a balance sheet are in this series' coverage of interest-rate risk; the capital-rules takeaway is humbler — a 0% risk weight is a statement about *credit*, not about safety.

## The takeaway: how to read a bank through its capital stack

Step back and the Basel framework resolves into a single, coherent idea, and it's the spine of this whole series. A bank is a leveraged, confidence-funded machine that survives only as long as its thin equity cushion absorbs losses faster than they arrive. The Basel rules are, in their entirety, an attempt to make that cushion *big enough, good enough, and honestly measured* — and forty years of revisions are forty years of discovering that the previous version fell short on at least one of those three.

So here is how to actually use this when you look at a bank. Lead with the **CET1 ratio**, because it's the only capital number that is permanent, first-loss, and trigger-free — the only one the market fully trusts. Then ask three questions. *How much headroom is there above the buffers?* A bank at 13% CET1 with a 9% combined buffer threshold has four points of room before the MDA freezes its dividend; a bank at 8% has none, regardless of how far above the 4.5% minimum it nominally sits. *What does the leverage ratio say?* If the risk-based ratio looks strong but the leverage ratio is scraping 3%, the bank has converted genuine thinness into a flattering number, and the unweighted backstop is the truth-teller. *And where do its AT1 and Tier 2 sit in the loss queue under its home country's resolution law?* Because, as Credit Suisse proved, the contract and the jurisdiction — not the intuition that bonds rank above equity — decide who gets wiped.

The deepest lesson of Basel isn't any single ratio. It's that capital adequacy is a *moving target written in the language of the last failure*. Each accord was the right rule for the previous crisis and an incomplete rule for the next one. Basel I was too crude; Basel II was too trusting of models; Basel III added quality and buffers and liquidity; the endgame put a floor under the cleverness. The next revision, whatever triggers it, will close a gap nobody is worried about today — probably one we'll only see clearly the morning after some bank we thought was well-capitalised turns out not to have been. Read a bank's capital not as a pass-fail score but as a measure of how much room it has to be wrong before the cushion runs out. That number, more than any other, tells you whether the machine can survive a bad year.

## Further reading & cross-links

- [Risk-weighted assets and how capital ratios really work](/blog/trading/banking/risk-weighted-assets-and-how-capital-ratios-really-work) — the deep dive on the denominator: standardized versus internal-model risk weights, the games played in RWA, and the output floor that the endgame uses to stop them.
- [Bank capital and leverage: why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) — the first-principles intuition for why a bank runs on a sliver of equity and how leverage turns a small asset move into a wipeout.
- [BIS and Basel: how banks are regulated globally](/blog/trading/finance/bis-and-basel-bank-regulation) — the system-level one-pager on the Bank for International Settlements, the Basel Committee, and how a non-binding standard becomes national law.
- [Credit Suisse 2023: the slow death of trust and the AT1 wipeout](/blog/trading/banking/credit-suisse-2023-the-slow-death-of-trust-and-the-at1-wipeout) — the full anatomy of the collapse that turned the AT1 loss queue from theory into a CHF 16 billion reality.
- [Liquidity management: LCR, NSFR and the liquidity buffer](/blog/trading/banking/liquidity-management-lcr-nsfr-and-the-liquidity-buffer) — the other half of Basel III: why a solvent bank can still die of a funding squeeze, and how the liquidity rules try to prevent it.

*This is educational material about how the banking capital framework works, not financial advice. Regulatory figures and minimums reflect the Basel III framework as published by the BIS; case-study numbers cite the as-of date of the event.*

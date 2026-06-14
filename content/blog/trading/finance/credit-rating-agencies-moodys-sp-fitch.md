---
title: "Credit Rating Agencies: How Three Companies Grade the Creditworthiness of the World"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How Moody's, S&P, and Fitch turn the question of who will pay you back into a single letter grade, why that grade is wired into law and contracts, and how three private firms abused that gatekeeper power in 2008."
tags: ["credit-ratings", "moodys", "standard-and-poors", "fitch", "bonds", "credit-risk", "subprime-crisis", "sovereign-debt", "financial-regulation", "financial-institutions", "conflict-of-interest"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Moody's, S&P, and Fitch turn the messy question of "will this borrower pay me back?" into a single letter grade from AAA down to D, and because regulators and investment contracts treat that letter as if it were a law of nature, three private companies became gatekeepers for the entire global bond market.
>
> - A *credit rating* is one firm's opinion of how likely a borrower is to default; the scale runs AAA (safest) to D (already defaulted), with a hard cliff between **investment grade** and **junk**.
> - The Big Three control roughly 95% of the market, and they are paid by the borrowers they grade, which is the *issuer-pays* conflict of interest at the center of every scandal.
> - The gap between a AAA and a BBB grade can be worth tens of millions of dollars a year on a single large bond, which is exactly why issuers fight for every notch.
> - In 2008 the agencies stamped trillions of dollars of subprime mortgage bonds AAA, then downgraded them en masse; that one failure was the central enabler of the financial crisis.
> - Ratings are not advice, not guarantees, and not predictions of price — they are an opinion about default, and the agencies are legally protected as if they were just publishing it.

Here is a fact that should be more disturbing than it is: when the United States government borrows money, when Apple issues a bond, when your state pension fund decides what it is allowed to own, and when a bank calculates how much of a cushion it must hold against a loan, the answer often turns on a letter grade assigned by one of three private companies you have probably never thought about. Not a regulator. Not a court. Not a public agency. Three for-profit firms — Moody's, S&P Global Ratings, and Fitch — sit between the world's borrowers and the world's lenders and decide, in effect, who is creditworthy and who is not. And the borrower being judged is the one who pays the bill.

The diagram below is the mental model to keep in your head for this entire piece: the credit rating system is a giant translation machine. On one side is a genuinely hard question — *how likely is it that this borrower fails to pay me back?* On the other side is a comically simple output: a few letters. Everything interesting, and everything dangerous, happens in the compression between the two.

![The credit rating scale from AAA down to D, split into investment grade and junk](/imgs/blogs/credit-rating-agencies-moodys-sp-fitch-1.png)

The scale at the top, `AAA`, means "we believe there is almost no chance this borrower defaults." The bottom, `D`, means "this borrower has already failed to pay." Between them runs a single, fateful line that splits the whole bond universe in two — **investment grade** above, **junk** (the polite term is *high yield*) below. That one line, as we will see, moves more money than almost any other distinction in finance, because it is written into the rules that govern who is *allowed* to buy what. By the end of this piece you will understand exactly how that letter is produced, why a private company gets to produce it, what each notch is worth in real dollars, and how the system failed so spectacularly in 2008 that it helped freeze the entire global economy.

## The basics: credit risk, default, and what a rating actually is

Before any of this makes sense, we need four ideas, built from zero. None of them require finance background; they require only the everyday experience of lending a friend money and wondering whether you will see it again.

### Lending, interest, and the thing that can go wrong

When you put money in a savings account, you are lending it to the bank. When you buy a government *bond*, you are lending money to the government. A **bond** is just a formal IOU: you hand over a lump sum today (say \$1,000), and in exchange the borrower promises to pay you regular *interest* (called the *coupon*) and to return your \$1,000 (the *principal*, or *par value*) on a fixed future date (the *maturity*). A bond paying a 5% coupon on \$1,000 pays you \$50 a year, then gives back your \$1,000 at the end.

Lending always carries one fundamental danger: the borrower might not pay you back. That danger is **credit risk** — the risk that a borrower fails to make a promised payment. When a borrower actually misses a payment or formally cannot meet its obligations, that event is a **default**. Default does not always mean you lose everything; often you *recover* some fraction (creditors of a bankrupt company might get back 40 cents on the dollar). But default is the bad outcome the whole system is built to anticipate.

So the single most important question any lender asks is: *how likely is this borrower to default, and if it does, how much will I lose?* A bank can answer that for a single home loan by examining one borrower's income and savings. But the bond market is enormous and impersonal. A pension fund in California might lend to a German carmaker, a Brazilian utility, and the government of Indonesia all in the same week. It cannot personally investigate each one. It needs a shorthand.

### The rating: an opinion compressed into a letter

That shorthand is the **credit rating**: a standardized opinion, produced by a specialist firm, of how likely a particular borrower (or a particular bond) is to default. The rating is expressed as a letter grade so that any investor anywhere can compare a Brazilian utility to a German carmaker on the same scale, without doing the underlying analysis themselves.

The scales of the three big firms are nearly identical, with cosmetic spelling differences. S&P and Fitch use the same letters; Moody's uses its own capitalization. From safest to most distressed:

| Tier | S&P / Fitch | Moody's | Plain meaning |
|---|---|---|---|
| Prime | AAA | Aaa | Default is almost unthinkable |
| High grade | AA+, AA, AA- | Aa1, Aa2, Aa3 | Extremely strong |
| Upper medium | A+, A, A- | A1, A2, A3 | Strong, some sensitivity to conditions |
| Lower medium | BBB+, BBB, BBB- | Baa1, Baa2, Baa3 | Adequate — the *last rung* of investment grade |
| --- the line --- | --- | --- | **Investment grade above / junk below** |
| Speculative | BB+, BB, BB- | Ba1, Ba2, Ba3 | Faces major uncertainties |
| Highly speculative | B+, B, B- | B1, B2, B3 | Vulnerable; depends on favorable conditions |
| Substantial risk | CCC, CC, C | Caa, Ca, C | Default is a live possibility or near-certain |
| Default | D | (no notch) | Has already defaulted |

The small `+`/`-` (or `1`/`2`/`3` at Moody's) are **notches** — finer gradations within a letter. The distance from `BBB-` to `BB+` is a single notch, but it is the most consequential notch in finance, because it is the one that crosses **the line**.

### Investment grade vs. junk: why the line is load-bearing

Everything from `BBB-`/`Baa3` and above is **investment grade**: considered safe enough that a conservative, regulated institution (a pension, an insurer, a money-market fund) is permitted to own it. Everything `BB+`/`Ba1` and below is **speculative grade**, universally nicknamed **junk** and marketed as **high yield**. The bonds are not worthless — they simply carry a real, acknowledged chance of default, so they must pay a higher coupon to attract buyers.

The line matters because it is not just a description; it is a *rule*. A vast amount of institutional money operates under **investment mandates** — written contracts and laws that say things like "this pension fund may only hold investment-grade bonds." The moment a bond is downgraded from `BBB-` to `BB+`, it can become *illegal* for a whole class of investors to keep holding it. They are forced to sell, often all at once, into a market where everyone else under the same rule is also selling. We will return to this "cliff effect" because it is one of the most dangerous mechanics in the entire system.

### What a rating measures — and what it pointedly does not

It is worth being precise about the question a rating answers, because most beginners (and plenty of professionals in 2008) get it wrong. A rating is an opinion about *creditworthiness* — the probability of default and, depending on the agency and the rating type, the expected loss if default occurs. S&P's headline ratings focus on the *probability* of default; Moody's ratings are framed around *expected loss*, which blends the chance of default with how much you would recover afterward. The distinction sounds academic but it explains why the two firms occasionally land on different notches for the same bond.

What a rating does *not* measure is just as important, and the list is long. A rating says nothing about whether a bond's *price* will rise or fall — a `AAA` bond loses value when interest rates rise, just like any other bond, because the rating is about default, not about market price. It says nothing about *liquidity* — whether you will be able to sell the bond quickly without crushing the price. It is not a *recommendation* to buy or sell. And it is explicitly *not a guarantee*: the agency is offering a forward-looking opinion, and like all opinions about the future it can be wrong. When you see a bond described as "rated `AA`," the honest mental translation is "one firm's current opinion that this borrower is very unlikely, but not certain, to miss a payment." Everything else — price, timing, whether it suits your goals — is on you.

There is one more subtlety that shapes how ratings behave in a crisis: the agencies aim to rate *through the cycle* rather than to the moment. They deliberately try to assign a grade that reflects a borrower's creditworthiness across good times and bad, rather than jerking the rating up and down with every quarter's results. That philosophy has a virtue — ratings are stable enough to anchor long-term contracts — and a vice we will see repeatedly: ratings are *sticky*, changing in discrete steps on the agency's schedule, while market prices reprice continuously. In a fast-moving crisis the official grade is frequently the last to admit what the market has already concluded.

### Sovereign ratings: grading entire countries

The agencies do not only grade companies. They grade governments — this is the **sovereign rating**, an opinion on how likely a country is to default on its debt. The United States, Germany, and Singapore have historically sat at or near the top; Greece, Argentina, and a long list of others have been downgraded into junk. A sovereign's rating also acts as a soft ceiling: companies inside a country are rarely rated higher than the government itself, on the logic that if the state defaults it can drag everyone down with it. So when an agency moves a country's rating, it can reprice the borrowing cost of every business and bank inside that country at once. Three private firms, in other words, can raise the interest bill of an entire nation with a single press release.

## The Big Three: an oligopoly with a government seal

Now that you know *what* a rating is, the obvious question is: who decides, and why these three?

### How three firms came to grade the world

Credit rating began respectably enough. John Moody published the first bond ratings in 1909, selling thick manuals to investors who paid for the analysis — the *investor-pays* model. Standard Statistics and Poor's (later merged into S&P) and Fitch followed. For decades these were research firms whose customers were the people doing the lending. That detail will matter enormously later.

Today, Moody's, S&P, and Fitch together account for roughly 95% of all outstanding ratings, with S&P and Moody's each holding the lion's share and Fitch a distant but still dominant third. Dozens of smaller agencies exist, but they are rounding errors. This is a textbook **oligopoly** — a market controlled by a handful of firms — and it is an unusually durable one.

![A matrix comparing S&P, Moody's, and Fitch by top grade, investment-grade floor, market share, and owner](/imgs/blogs/credit-rating-agencies-moodys-sp-fitch-2.png)

The comparison above shows how interchangeable the three look from the outside and how concentrated the market is: S&P holds roughly half, Moody's about a third, and Fitch around an eighth. Their scales are aligned to the notch, which is by design — the whole point of a rating is comparability, and a market with three subtly different scales would be useless.

### The NRSRO: how the government locked the oligopoly in

Why can't a sharp new analytics firm just enter and compete? Because the most valuable thing the Big Three own is not a model. It is a government designation.

In 1975 the U.S. Securities and Exchange Commission, looking for a way to set capital rules for broker-dealers, decided to lean on credit ratings — but only ratings from agencies it trusted. It created the category **NRSRO**: *Nationally Recognized Statistical Rating Organization*. From that moment, when a regulation said "investment grade" or "AAA," it implicitly meant "as rated by an NRSRO." The SEC initially blessed exactly three firms: Moody's, S&P, and Fitch.

This was the original sin of the modern system. The government took a private opinion and wired it into the machinery of law. And because the rules referenced *NRSRO ratings* specifically, a new entrant faced an impossible bootstrap problem: investors only valued NRSRO ratings, but you could not easily become an NRSRO without already being "nationally recognized." The designation built a moat around the incumbents and handed them a franchise. The agencies were no longer just publishers of opinions; they had become, in the words of one famous description, the issuers of "regulatory licenses" — the letter grade was now a key that unlocked legal permissions, and only three firms could cut the key.

## How they actually make money: the issuer-pays conflict

Here is the part that surprises almost everyone, and it is the rotten beam at the center of the building.

### The customer is the company being graded

Remember that ratings began as an *investor-pays* business: lenders bought research. Starting around the early 1970s, that flipped. Today the dominant model is **issuer-pays**: the borrower — the company or government issuing the bond — *hires and pays the agency to rate it*. A typical corporate issuer pays an agency a fee scaled to the size of the deal, often running from tens of thousands of dollars for a small issue up to roughly \$1 million to \$2 million or more for a large, complex structured deal, plus ongoing surveillance fees to keep the rating current.

Read that again. The entity whose creditworthiness is being judged is the entity writing the check to the judge. Imagine if students paid the examiner directly, chose which examiner to use, and could decline to publish the grade if they did not like it. That is roughly the structure of the bond rating market.

![The issuer-pays model as a pipeline: issuer pays a fee to the agency, which issues a rating sold to investors](/imgs/blogs/credit-rating-agencies-moodys-sp-fitch-3.png)

The pipeline above traces the money and the product in opposite directions. The borrower selects an agency, pays the fee, receives a rating, and sells the now-graded bond to investors. The investors — the people the rating is supposed to protect — pay the agency nothing in the dominant model. The agency's actual paying customer is the party it is grading.

### Why issuer-pays exists at all

It is worth being fair about why the industry switched, because the conflict was not adopted out of pure greed. Two forces pushed it. First, the **photocopier problem**: once cheap copying arrived, an investor-pays research report could be photocopied and passed around for free, so only the first buyer paid while everyone read it — a classic free-rider problem that gutted the investor-pays revenue model. Second, issuers genuinely *want* ratings: a bond without a rating is hard to sell, and an unrated borrower pays more. So issuers were happy to pay for the service that made their debt marketable. The model is not irrational. It is just dangerously conflicted.

### The conflict, drawn as a loop

The danger is not that an analyst takes a bribe. It is subtler and structural. Because the issuer chooses which agency to hire and can solicit multiple agencies, an issuer can effectively *shop* for the most favorable grade. If one agency is stricter, the issuer can take its business to a more lenient one. Every agency knows this. So every agency faces a quiet, persistent pressure: be too tough and you lose the fee; the deal — and the revenue — walks across the street.

![A directed graph showing how issuer shopping and fee revenue pull every agency toward leniency](/imgs/blogs/credit-rating-agencies-moodys-sp-fitch-5.png)

The graph above shows the mechanism as a forward chain, not a single villain. The intuition is that no individual has to be corrupt for the system to drift: an issuer wanting a high rating shops the agencies; the one offering `AAA` wins the deal and the fee; the stricter agency that refused loses the revenue; and over thousands of deals, the rational response of every competitor is to drift toward leniency rather than lose share. Internal emails surfaced after 2008 captured this with brutal clarity — one S&P analyst wrote that a deal "could be structured by cows and we would rate it," and another quipped that a deal was rated and the model could have been built "by monkeys." The conflict is not a bug that occasionally fires; it is the gravitational field the whole industry sits inside.

#### Worked example: the issuer-pays fee per deal

Let's make the incentive concrete. Suppose an investment bank is assembling a \$1,000,000,000 (one billion dollar) structured mortgage deal — exactly the kind of product at the heart of 2008. The agencies' fees for rating structured finance were notably higher than for plain corporate bonds, and could run on the order of several basis points of the deal size. A **basis point** is one hundredth of one percent — 0.01%.

Suppose the fee is 5 basis points of the deal:

- 5 basis points = 0.05% = 0.0005 as a decimal.
- Fee = \$1,000,000,000 x 0.0005 = \$500,000 for that single deal.

Now suppose the bank brings the agency *forty* such deals in a year. That is \$500,000 x 40 = \$20,000,000 of revenue from one client's structured-finance pipeline, before any ongoing surveillance fees. If pushing back on one deal's `AAA` rating risks losing that client's future business, the analyst is not weighing a \$500,000 fee — they are implicitly weighing a \$20,000,000 relationship.

**The intuition:** issuer-pays does not require a bribe to bend behavior; the size and repeat-business nature of the fee is enough to make saying "no" institutionally expensive.

## How a rating is actually made

It is tempting to assume a rating simply pops out of a spreadsheet, but the process is more human — and more contestable — than that. Understanding it explains both why ratings carry real analytical weight and where the conflict has room to operate.

When an issuer wants to bring a bond to market, it engages an agency and hands over a trove of information: financial statements, business plans, projections, and details of the specific bond's structure (its seniority, its collateral, its covenants — the promises the borrower makes to protect lenders). A lead analyst studies the borrower against the agency's published **methodology** — a documented framework that scores factors like leverage (how much debt relative to earnings), cash-flow stability, competitive position, and the legal structure of the bond. For a corporate issuer this is heavily qualitative; for a structured product like a mortgage pool it leans on a quantitative model that simulates how the pool performs under different default and recovery assumptions.

Crucially, the final grade is not set by the analyst alone. It is decided by a **rating committee** — a group of analysts who vote, in principle to guard against any single person being captured or pressured. The issuer is typically allowed to see the proposed rating *before* it is published and can appeal it, supplying more information to argue for a higher grade. In structured finance, this dynamic became something closer to *co-design*: bankers would iterate the structure of a deal — adjusting how much cushion sat below the senior tranche — running it through the agency's model until the senior slice cleared the `AAA` threshold. The agency was no longer simply judging a finished product; it was, in effect, helping engineer the product to its own model's specifications. When the customer can see the test, knows the answer key (the published methodology), and gets to resubmit until it passes, the line between rating and tailoring blurs.

Once published, a rating is not frozen. The agency conducts ongoing **surveillance**, reviewing the borrower and changing the grade as circumstances shift. A formal warning that a change may be coming is an **outlook** (e.g. "negative outlook") or, more urgently, a **credit watch** or "rating under review." These signals matter to markets in their own right: a negative outlook can move a bond's price before any actual downgrade arrives, because investors front-run the expected cut. This is the surveillance machinery that, in 2008, fired the mass downgrades — and in calmer times, quietly reprices the borrowing cost of thousands of issuers as the world changes around them.

## The rating scale, notch by notch — and what each notch is worth

A rating is not a vibe. Each grade maps, statistically, to a historical probability that a bond at that grade defaults over a given horizon. The agencies publish these "default studies" every year. The rough shape, using illustrative cumulative 10-year default rates from long-run historical data (these move year to year, so treat them as approximate orders of magnitude):

| Grade (S&P) | Plain meaning | Illustrative 10-yr default rate |
|---|---|---|
| AAA | Essentially riskless | well under 1% |
| AA | Extremely strong | roughly 1% |
| A | Strong | a few percent |
| BBB | Last rung of IG | mid-single-digit percent |
| BB | Speculative | low-to-mid teens percent |
| B | Highly speculative | around 25-30% |
| CCC | Substantial risk | roughly half or more |

Notice the curve is not linear — it bends sharply downward in quality as you descend. The jump in default risk from `AAA` to `BBB` is modest; the jump from `BBB` to `B` is enormous. This is why the *price* of credit — the extra yield a borrower must pay — also bends sharply as ratings fall.

That extra yield has a name: the **credit spread**, the additional interest a borrower pays over and above a truly safe benchmark (like a top-rated government bond) to compensate lenders for taking on default risk. A `AAA` borrower might pay only a tiny spread; a `B` borrower might pay several percentage points more. The rating, in other words, is not academic — it sets the price.

#### Worked example: AAA vs. BBB borrowing cost on \$1,000,000,000

This is the example that makes a CFO care about a single notch. Imagine two companies that each want to borrow \$1,000,000,000 (one billion dollars) for ten years. The only difference between them is their rating.

- Company A is rated `AAA`. Investors demand a credit spread of, say, 0.40% (40 basis points) over the safe benchmark.
- Company B is rated `BBB`. Investors demand a spread of, say, 1.60% (160 basis points) over the same benchmark.

The difference in spread is 1.60% - 0.40% = 1.20%, or 120 basis points. Apply that to the principal:

- \$1,000,000,000 x 1.20% = \$1,000,000,000 x 0.012 = \$12,000,000 per year.

Company B pays \$12,000,000 more in interest *every single year* than Company A, purely because of the letters on its bond. Over the 10-year life of the bond, ignoring compounding, that is \$120,000,000 in extra interest — for the same billion dollars, from the same investors, at the same moment in history.

**The intuition:** a rating is not a description of a company; it is a price tag stapled to its borrowing, and the gap between two grades can be worth nine figures over a single bond's life — which is exactly why issuers fight for every notch and exactly what makes the issuer-pays conflict so combustible.

#### Worked example: the capital relief a bank gets from a AAA asset

The rating does not only set what borrowers pay. It sets how much cushion lenders must hold — and that is where the agencies' power reaches inside the banking system itself. Bank capital rules (the international **Basel** framework) require a bank to hold a slice of its own money — *capital*, essentially shareholder equity — against its assets, sized to how risky each asset is. The riskiness is measured by a **risk weight**, and under the rules that prevailed before the crisis, the risk weight was read straight off the credit rating.

![A stack showing how a rating sets the risk weight, which sets required capital and bank equity](/imgs/blogs/credit-rating-agencies-moodys-sp-fitch-7.png)

The stack above shows the chain: the agency's letter sets the risk weight, the risk weight sets the required capital, and required capital determines how much equity the bank must lock away instead of lending out. Suppose a bank holds \$1,000,000,000 of a bond, and the baseline capital requirement is 8% of risk-weighted assets.

- If the bond is rated `AAA`, its risk weight under the old standardized rules was 20%. Risk-weighted assets = \$1,000,000,000 x 20% = \$200,000,000. Required capital = \$200,000,000 x 8% = \$16,000,000.
- If the same bond were rated `BB` (junk), its risk weight could be 100% or more. At 100%: risk-weighted assets = \$1,000,000,000. Required capital = \$1,000,000,000 x 8% = \$80,000,000.

So a `AAA` stamp let the bank hold \$16,000,000 of capital against that billion-dollar position instead of \$80,000,000 — freeing up \$64,000,000 of capital to lend out and earn more profit on.

**The intuition:** a high rating is not just cheaper borrowing for the issuer; it is *capital relief* for the banks that hold the bond — which gave banks a powerful incentive to want AAA stamps to exist on as many assets as possible, and gave the agencies a buyer hungry for exactly the product that blew up in 2008.

## 2008: the AAA that wasn't

If you want to understand why credit ratings are not a dry technical footnote but a central character in modern financial history, you have to understand what happened to structured finance in the 2000s. This is the crisis's central enabler, and it runs directly through the issuer-pays conflict and the capital-relief incentive we just built.

### How a pile of risky loans became "AAA"

Start with a single subprime mortgage — a home loan to a borrower with weak credit, who therefore carries a high chance of default. On its own, no one would call that loan safe. But the financial engineering of the 2000s performed a kind of alchemy. Thousands of these loans were pooled together into a **mortgage-backed security (MBS)**, and the cash flowing in from all those borrowers was sliced into layers called **tranches** (from the French for "slice"). The trick was *seniority*: the **senior tranche** got paid first, out of the very first dollars that came in; the **junior** or **equity tranche** got paid last and absorbed the first losses.

The argument went like this: even if many subprime borrowers default, *some* will keep paying, and the senior tranche — first in line — will still get its money. So the senior slice of a pool of junk-rated loans could itself be rated `AAA`. Then bankers went further and built **collateralized debt obligations (CDOs)**: they took the leftover, riskier middle tranches of many different mortgage deals, pooled *those* together, and re-sliced *that* pile — producing, astonishingly, brand-new `AAA` tranches out of bonds that were already mediocre. It was risk laundering through repackaging, and the agencies' models blessed it.

![Before-and-after: a subprime tranche rated AAA in 2006 versus its real risk revealed in 2008](/imgs/blogs/credit-rating-agencies-moodys-sp-fitch-4.png)

The before-and-after above is the whole tragedy in one image. On the left, in 2006, a senior tranche of pooled subprime loans carries a `AAA` stamp and is priced as if default were near-impossible — which is exactly why pension funds, insurers, and banks (chasing yield *and* capital relief) bought it by the trillion. On the right, in 2008, home prices fall, the supposedly diversified loans default *together*, and the same tranche is cut to junk, leaving investors recovering pennies. Repackaging never removed the risk; the rating only hid it.

### The fatal modeling error

Why did the models say `AAA`? The senior-tranche logic only holds if the underlying loans default *independently* — if one borrower's default tells you nothing about the next. The single number at the heart of the whole exercise was the **default correlation**: how much one borrower's failure raises the odds that the next one fails too. If correlation is low, pooling works beautifully — a few scattered defaults barely dent a large pool, and the senior tranche is genuinely safe. If correlation is high, pooling is an illusion — the loans tend to fail together, in clumps, and even a senior slice can be wiped out. The agencies' models leaned on correlation assumptions, calibrated largely on a period of *rising* home prices, that badly understated how much subprime defaults would move together when prices turned.

But a nationwide fall in home prices is not a collection of independent accidents; it is a single common cause hitting every borrower at once. A subprime borrower in Florida and one in Nevada had little in common — except that both had bought houses they could only afford if prices kept rising and they could refinance. When that one shared assumption broke, they defaulted *for the same reason at the same time*. Defaults did not arrive at the modeled trickle; they arrived in a correlated wave, and they tore through the senior tranches that were never supposed to be touched. The agencies had, in effect, rated a portfolio assuming a calm that the structure itself made impossible — and they had done it with models the issuers could see and optimize against, on deals the issuers paid for.

A second-order failure compounded the first. The CDO-of-CDO structures — pools built from the leftover slices of other pools — were extraordinarily *sensitive* to the correlation assumption. A small error that nudged a plain mortgage tranche from `AAA` to `AA` could, when that tranche was re-pooled and re-sliced, swing the resulting CDO tranche from `AAA` all the way to junk. The agencies were stacking model error on top of model error, and the higher they stacked, the more violently the structure would move when reality diverged from the assumption. The same mathematics that manufactured the `AAA` on the way up manufactured the avalanche on the way down.

### The downgrade avalanche

Then came the second act, which turned a bad situation into a systemic collapse. Once reality became undeniable, the agencies did not adjust gently — they downgraded *en masse*. Securities that had been `AAA` were cut by ten or more notches, sometimes straight to junk, in waves through 2007 and 2008. Those downgrades were not cosmetic. They triggered exactly the cliff effects the system was riddled with: investors with investment-grade-only mandates were forced to dump bonds; banks holding the assets suddenly faced far higher capital requirements (the relief reversed); and the value of these securities as collateral evaporated. The downgrades did not just record the crisis; they propagated it.

#### Worked example: a AAA CDO tranche cut to junk

Put yourself in the seat of a conservative investor — a town pension fund, say — in 2006. You are allowed to hold only investment-grade bonds, and you want a little extra yield, so you buy \$100,000,000 of a `AAA`-rated senior CDO tranche. The whole point, you believe, is safety.

- You pay \$100,000,000 for the tranche, at par (100 cents on the dollar), trusting the `AAA` stamp.
- Through 2007 the agencies begin cutting these securities. By 2008 your tranche is downgraded from `AAA` to deep junk (`CCC` or below).
- Two things now happen at once. First, your mandate forbids holding junk, so you are a *forced seller*. Second, the market price has collapsed because the underlying loans are defaulting and everyone is selling at the same time. Senior subprime CDO tranches that had been priced near par traded down toward 20 cents on the dollar or worse during the worst of it.
- If you must sell at 20 cents: you receive \$100,000,000 x 0.20 = \$20,000,000. Your loss is \$100,000,000 - \$20,000,000 = \$80,000,000 — roughly 80 cents on every dollar, on the asset you bought *because* it was rated safest.

**The intuition:** a `AAA` rating that is wrong is more dangerous than no rating at all, because it lures exactly the most conservative, most constrained investors into the riskiest assets — and then the same rating system that lured them in forces them to sell at the bottom.

The financial damage was staggering, the agencies' role was central, and yet — a point we will return to — the firms paid no meaningful penalty proportional to the harm. A U.S. government commission later concluded the crisis could not have happened without the rating agencies, and S&P eventually paid a settlement of roughly \$1.5 billion to the U.S. Department of Justice and states in 2015, with Moody's settling for around \$864 million in 2017. Large numbers — and small ones next to the trillions of dollars in losses across the system.

## Sovereign downgrades: when the agencies grade governments

The same machinery the agencies point at companies, they point at countries — and the stakes scale up to the level of entire national budgets.

### The 2011 downgrade of the United States

On August 5, 2011, S&P did something it had never done: it stripped the United States of its top `AAA` rating, cutting it one notch to `AA+`. The trigger was a self-inflicted political crisis — a standoff in Congress over raising the federal *debt ceiling* (the legal cap on how much the government may borrow) that flirted with a technical default not because the U.S. could not pay, but because its lawmakers nearly chose not to authorize payment. S&P's reasoning was less about the math of solvency and more about the dysfunction of the politics. (There was even an embarrassing \$2 trillion arithmetic error in S&P's initial analysis, which the firm had to correct; it downgraded anyway, on the political reasoning.) Moody's and Fitch held the U.S. at `AAA` at the time — a reminder that the three do not always agree.

The reaction was paradoxical and revealing. In theory, downgrading a borrower should *raise* its borrowing costs. In practice, the opposite happened: terrified investors fled *toward* U.S. Treasuries as the safest asset they knew, and Treasury yields *fell*. The downgrade told you more about the limits of the agencies' authority over the world's reserve currency than about U.S. creditworthiness. (Fitch would later also downgrade the U.S. to `AA+` in 2023, and Moody's would strip the last `AAA` in 2025 — but the market's "flight to the thing being downgraded" reflex persisted, because there was nowhere safer to flee.)

### Greece's fall into junk

For a smaller country without a reserve currency, a downgrade is not a paradox; it is a noose. In April 2010, S&P cut Greece's sovereign rating to `BB+` — junk — at the start of the European debt crisis. Unlike the U.S., Greece could not print the currency it owed (it was inside the euro), and it depended on foreign lenders who now demanded punishing yields. As Greece was downgraded deeper, its borrowing costs spiraled, which worsened its finances, which justified further downgrades — a doom loop. Greek 10-year yields rocketed from single digits toward 30% and beyond at the peak, locking the country out of markets and forcing a series of international bailouts. The downgrades did not merely describe Greece's distress; by raising its borrowing cost they helped *deepen* it.

#### Worked example: a one-notch sovereign downgrade and a country's interest bill

Suppose a mid-sized country has \$500,000,000,000 (five hundred billion dollars) of government debt, and each year it must refinance — roll over — about \$100,000,000,000 of it as old bonds mature and new ones are issued. Now an agency downgrades the country one notch, and investors respond by demanding an extra 0.50% (50 basis points) of yield on its new borrowing.

- The extra cost applies to the new debt issued at the higher yield: \$100,000,000,000 x 0.50% = \$100,000,000,000 x 0.005 = \$500,000,000 per year in additional interest, on that year's refinancing alone.
- As more of the old, cheaper debt matures and is replaced at the higher yield, that extra annual cost compounds across the stock of debt. If eventually the full \$500,000,000,000 reprices 50 basis points higher, the added interest bill is \$500,000,000,000 x 0.005 = \$2,500,000,000 per year — \$2.5 billion of taxpayer money redirected to lenders, every year, from one notch.

**The intuition:** for a government, a single downgrade is a tax increase imposed by a private company — money that must come from higher taxes or lower spending — and for a country that cannot print its own currency, the resulting rise in borrowing cost can be self-fulfilling, deepening the very distress the downgrade flagged.

## Regulatory over-reliance and the reforms that didn't reform

We have now seen the conflict and the failure. The remaining puzzle is why the system survived essentially intact.

### Why regulators outsourced judgment to three firms

For decades, regulators around the world did something seductive and lazy: instead of judging the riskiness of banks' assets themselves, they wrote the agencies' ratings directly into the rulebook. "Hold less capital against `AAA` assets." "Money-market funds may only hold short-term debt rated in the top tiers." "Insurers face lower charges on investment-grade bonds." This is **regulatory over-reliance** — the practice of treating private ratings as if they were objective regulatory facts.

The consequence was that a rating stopped being merely *information* and became a *trigger* with legal force. When the agencies were right, this was efficient. When they were wrong — as in 2008 — the error did not stay contained in one investor's bad bet; it was hardwired into the capital position of every regulated institution at once, so a wave of downgrades drained capital from the entire banking system simultaneously. The regulators had built the agencies' fallibility into the foundation of the financial system.

### Rating triggers and the cliff effect

The same hardwiring shows up in private contracts, and it has its own name: the **rating trigger**. A loan agreement or derivatives contract may contain a clause that fires automatically if a party's rating drops below some threshold — for example, "if your rating falls below `A-`, you must immediately post additional collateral," or "this funding line is cancelled if you fall below investment grade." Individually these clauses look prudent. Collectively they create a **cliff**: a single downgrade can simultaneously trigger demands for billions in collateral, cancel funding lines, and force asset sales — all at the worst possible moment, when the firm is already weak. The downgrade does not just reflect distress; it *manufactures* a liquidity crisis. This is precisely what helped destroy AIG in 2008, whose downgrades triggered tens of billions of dollars in immediate collateral calls it could not meet.

![A timeline of the 2007-2011 ratings crisis: subprime downgrades, AIG triggers, Greece and the US downgrade](/imgs/blogs/credit-rating-agencies-moodys-sp-fitch-6.png)

The timeline above lines up the sequence: the first subprime downgrades in 2007, the mass downgrades and rating-trigger detonations of 2008 (Lehman and AIG), Greece's fall to junk in 2010, and the symbolic stripping of the U.S. `AAA` in 2011. Read together, it is a four-year demonstration that the agencies' grades were not passive descriptions — each downgrade was a live wire connected to capital rules, mandates, and contract triggers, and pulling it sent current through the whole system.

### Dodd-Frank and the reform that wasn't

After 2008, the obvious targets for reform were the issuer-pays conflict and regulatory over-reliance. The 2010 U.S. **Dodd-Frank Act** took aim at both. It directed federal regulators to *remove* references to credit ratings from regulations wherever an alternative standard could be found. It stripped the agencies of an old legal shield. It created an SEC office to supervise the NRSROs and required more disclosure of methodologies and track records. It even floated the idea of a board that would *assign* which agency rated a given structured deal, to break the issuer-shopping dynamic — a proposal that was studied and quietly shelved.

And yet the model survived. Why?

The honest answer is that no one built a better mousetrap. The conflict-free alternative — investor-pays — still suffers the free-rider problem that killed it in the first place. Government-assigned ratings raise the question of who is accountable when the *government's* assignment is wrong. Removing ratings from regulation forced regulators to invent their own risk measures, which is exactly the hard, expensive judgment they outsourced to avoid. The Big Three's franchise — built on the network effect of universal comparability and the moat of the NRSRO designation — proved far more durable than the political will to dislodge it. The agencies leaned harder into a legal defense that has protected them throughout: that a rating is merely an **opinion**, a form of speech, and they are no more liable for a wrong rating than a newspaper is for a wrong editorial. That defense — not perfect, but resilient — is a large part of why three firms that helped detonate the global economy are, today, more profitable than ever.

### The "it's just an opinion" defense

The free-speech argument deserves its own moment, because it is the keystone that holds the whole structure up. For decades the agencies argued in court that their ratings are constitutionally protected opinions — comparable to a journalist's editorial — and therefore shielded from the liability an auditor or an investment bank faces for getting things wrong. Investors who lost billions on `AAA`-rated paper found this maddening: the rating was treated by regulators as an objective, license-granting fact when it suited the system, and as a mere casual opinion when someone tried to sue. The agencies cannot have it both ways, critics argued, yet for a long time they effectively did. Dodd-Frank chipped at the shield — exposing agencies to a slightly lower liability bar for some statements and removing one statutory exemption — but the core defense largely held, which is why the post-crisis settlements, large in absolute dollars, were negotiated rather than imposed by a jury verdict that established broad liability.

### The industry today

Strip away the scandals and the Big Three are, financially, among the best businesses in the world. They sell an intangible product — a letter — at enormous margins, into a market protected by regulation and network effects, with pricing power over customers who *must* be rated to sell their bonds. Moody's and S&P routinely post operating margins that ordinary industrial companies can only dream of, and their stock prices have far outrun the broad market since 2008. The lesson is uncomfortable: the gatekeeper role that made the agencies so dangerous in the crisis is the very thing that makes them such durable, profitable franchises. Reform aimed at the *behavior* (more disclosure, supervision, conflict policies) while leaving the *structure* (issuer-pays, oligopoly, regulatory reliance) intact — and structure, not behavior, is where the power lives. Outside the U.S., regulators built their own oversight regimes (Europe's ESMA now supervises agencies operating in the EU), but the same three firms dominate globally, and no challenger has cracked the franchise.

## Common misconceptions

**"A AAA rating is a guarantee — or even a prediction — that the bond won't lose money."** It is neither. A rating is an *opinion about the probability of default*, nothing more. A `AAA` bond can fall sharply in price (if interest rates rise, all bond prices fall regardless of rating), can be downgraded, and — as 2008 proved at scale — can default if the opinion was wrong. The agencies themselves insist the rating speaks only to default risk, not to price, liquidity, or whether the bond is a good investment.

**"The agencies are regulators or government bodies."** They are private, for-profit companies whose shares trade on the stock market. The confusion is understandable, because the government *delegated* a quasi-regulatory role to them via the NRSRO designation. But no one elected them, and their primary duty runs to their shareholders, not to investors or the public.

**"The investor who relies on the rating is the one paying for it."** In the dominant issuer-pays model, the opposite is true: the *borrower being graded* pays. The investors the rating is supposed to protect typically pay nothing for the headline rating. This inversion — paying customer is the rated party — is the single fact that explains most of the system's pathologies.

**"After the 2008 disaster, the agencies were broken up or reformed away."** No. The Big Three remain a tight oligopoly, the issuer-pays model is intact, and the firms are highly profitable. Dodd-Frank trimmed the edges — more disclosure, an SEC supervisory office, a push to strip ratings out of some rules — but left the core business model untouched, because no superior model commanded political consensus.

**"All three agencies always agree, so the rating is objective truth."** They frequently disagree, sometimes by multiple notches, and the disagreements are revealing. In 2011 S&P downgraded the U.S. while Moody's and Fitch did not. A bond can be `BBB-` (investment grade) at one agency and `BB+` (junk) at another — a *split rating* that lands it right on the most consequential line in finance, with different investors treating it differently depending on which opinion their mandate honors.

**"A downgrade just records bad news that already happened."** Often it actively *causes* further harm. Through forced selling (mandate cliffs), higher capital charges (banks), and contractual rating triggers (collateral calls), a downgrade can drain liquidity from a borrower precisely when it is most fragile, turning a warning into a self-fulfilling prophecy. The information and the trigger are fused.

## How it shows up in real markets

**The 2008 subprime CDO ratings.** This is the defining episode. Between roughly 2004 and 2007, the agencies stamped `AAA` on trillions of dollars of mortgage-backed securities and CDOs whose underlying collateral was subprime loans. The grades were essential fuel: without them, conservative buyers (pensions, insurers, foreign banks) could not have bought the bonds, and banks could not have claimed the capital relief that made warehousing them so profitable. When U.S. home prices fell nationally — the one scenario the correlation models discounted — the supposedly safe senior tranches absorbed losses they were never modeled to face, and the agencies downgraded en masse. The U.S. Financial Crisis Inquiry Commission concluded the agencies were "essential cogs in the wheel of financial destruction" and a "key enabler" of the meltdown. The mechanism from this entire piece — issuer-pays leniency, plus regulatory and capital reliance, plus the cliff effect of mass downgrades — fired all at once.

**The 2011 S&P downgrade of the United States.** On August 5, 2011, S&P cut the U.S. from `AAA` to `AA+`, citing not insolvency but the political dysfunction of the debt-ceiling standoff. The lasting lesson was about the *limits* of agency power: rather than driving up U.S. borrowing costs, the downgrade triggered a flight to safety *into* Treasuries, and yields fell. For the issuer of the world's reserve currency, the agencies' opinion mattered less than the market's lack of any safer alternative — a stark contrast to what a downgrade does to a small, indebted country.

**Greece and the Eurozone debt crisis.** Beginning in 2009-2010, the agencies repeatedly downgraded Greece, Portugal, Ireland, Spain, and Italy as the European sovereign-debt crisis unfolded. Greece's cut to junk by S&P in April 2010 was a pivotal moment. For countries inside the euro — unable to print their own currency to pay debts — the downgrades fed a doom loop: lower ratings meant higher yields meant worse finances meant lower ratings. European politicians, furious at the *procyclical* role the agencies played (downgrading into a crisis and amplifying it), even tried to launch a European rating agency to break the American firms' grip; it went nowhere, for the same reason every alternative does.

**Enron, rated investment grade days before it failed.** In late 2001, the energy-trading giant Enron — riddled with accounting fraud — was still rated comfortably **investment grade** by all three agencies until just four days before it filed for the largest U.S. bankruptcy of its time. The agencies, like nearly everyone, were fooled by fraudulent financials. But the episode exposed a structural weakness baked into the system by the rating triggers: Enron's debt contained clauses that, on a downgrade to junk, would force immediate repayment the company could not make. The agencies, knowing a downgrade could *itself* push Enron over the edge, hesitated — illustrating how rating triggers can perversely make agencies *slower* to downgrade, because the downgrade is the kill shot. The system's wiring corrupted the very judgment it depended on.

**Rating-trigger "cliff" effects: AIG in 2008.** The insurer AIG had sold enormous volumes of credit protection (credit default swaps) on mortgage securities, with contracts containing rating triggers: a downgrade obligated AIG to post more collateral. When the agencies cut AIG's rating in September 2008, the triggers fired simultaneously, generating collateral demands of tens of billions of dollars that AIG could not meet, forcing an emergency U.S. government bailout of roughly \$182 billion to prevent its collapse from cascading through every counterparty. The downgrade was not a passive observation; it was the pin that pulled the grenade. This is the cliff effect at full, systemic scale — and the clearest possible demonstration that a credit rating, in the modern system, is an *action*, not just an opinion.

**The 2023 regional-bank and Credit Suisse stress.** More recently, the agencies' actions around the 2023 banking stress — downgrades of U.S. regional banks and the unraveling of Credit Suisse — showed the same tension: downgrades that arrived fast enough to accelerate funding runs, but were criticized as lagging the market's own real-time verdict (bank stock and bond prices had already collapsed). The recurring critique is that ratings are *sticky and slow* — they change in discrete notches on the agency's timetable, while markets reprice continuously — so the official grade is often the last to admit what prices already know. For the deeper mechanics of that episode, see [the SVB and Credit Suisse bank runs of 2023](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).

## When this matters to you / further reading

You may never personally hire a rating agency, but their letters touch your life through almost every pool of capital that holds your money. The bond funds in your retirement account are governed by mandates written in these letters. The money-market fund where your cash sits is restricted to top-rated short-term debt. The pension that owes you a future check buys and sells under investment-grade rules. The insurer behind your policy holds capital sized to these grades. And the borrowing cost of your own government — which sets the backdrop for your mortgage and your taxes — moves when these three firms move.

The single most useful habit this piece can leave you with is skepticism about the word *rated*. When you read that a bond, a fund, or a structured product is "`AAA`-rated" or "investment grade," translate it in your head to what it actually is: *one private company's paid opinion about default probability, expressed on a coarse scale, subject to a structural conflict of interest, and possibly slow to change.* That is genuinely useful information — ratings do correlate with default risk over the long run, and ignoring them entirely would be foolish. But it is an opinion bought by the borrower, not a guarantee, not a price forecast, and not a substitute for understanding what you actually own. The investors who learned that lesson the hard way in 2008 are the ones who had outsourced the most judgment to the fewest firms.

If you want to go deeper into how these pieces connect, three companion reads build directly on this one. To see the rating failure play out inside a single collapsing firm, read [how Lehman Brothers and the 2008 financial crisis unfolded](/blog/trading/finance/lehman-brothers-2008-financial-crisis). To understand the banks that *manufactured* the rated CDOs and earned the fees on both ends, read [how an investment bank actually makes money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money). And to place the rating agencies among the other players — banks, insurers, asset managers, market infrastructure — that each wield a slice of financial power, read [the field guide to financial institutions](/blog/trading/finance/field-guide-to-financial-institutions). The agencies are not the largest institutions in finance, but they may be the most quietly powerful: three companies whose alphabet decides the price of trust for the entire borrowing world.

*This article is educational, not investment advice. Ratings, default statistics, fees, and market figures cited here are illustrative or approximate and as-of mid-2026; the agencies update their default studies and methodologies regularly, and live spreads and yields move daily.*

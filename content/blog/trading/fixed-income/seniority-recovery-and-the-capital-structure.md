---
title: "Seniority, recovery, and the capital structure: where a bondholder actually sits when things go wrong"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into the capital structure: the waterfall that pays secured debt first and common equity last, the absolute-priority rule in bankruptcy, why two bonds from the same company can carry very different risk, and how recovery rates by seniority turn into the loss-given-default that drives credit spreads."
tags: ["fixed-income", "bonds", "credit-risk", "seniority", "recovery-rate", "loss-given-default", "capital-structure", "bankruptcy", "secured-debt", "corporate-credit", "us-treasuries"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — when a company runs out of money, it does not pay everyone a little; it pays its claims in a strict order — a *waterfall* — and the people at the top get all of their money before the people below get a single cent.
> - The order, top to bottom, is **secured debt → senior unsecured → subordinated → preferred equity → common equity**. This ranking is the *capital structure*, and where you sit in it is the single biggest driver of what you recover.
> - In bankruptcy this order is enforced by the **absolute-priority rule**: a lower tier gets nothing until the tier above it is paid *in full*. That is why two bonds from the very same issuer can have wildly different risk.
> - **Recovery rate** is the cents-on-the-dollar a lender gets back in a default; **loss-given-default (LGD) = 1 − recovery**. Senior secured lenders have historically recovered roughly **\$0.50–0.70** on the dollar; subordinated bonds far less; equity, almost always **\$0**.
> - Expected loss has three parts: *probability of default* × *loss-given-default* × *exposure*. Seniority moves the middle term — same default, very different loss — so it feeds straight into the **credit spread** you are paid.
> - Running example: **Northwind Corp** defaults with **\$600M of assets** against **\$1,000M of claims**. We trace who recovers what, and watch the money run out partway down the stack.

Two people lend the very same company the very same \$1,000, on the very same day. One gets back \$1,000 with interest. The other gets back \$120 and a thank-you note. Neither did anything wrong. They simply signed different pieces of paper — one sat higher in the company's *capital structure* than the other — and when the company failed, that single difference decided almost everything.

This is the part of bond investing that beginners almost never see until it is too late, because it only matters when things go wrong. When a company is healthy, every one of its bonds pays the same promised coupon, and they all look like roughly the same thing. It is only in default — when there is not enough money to go around — that the hidden ranking inside the company snaps into focus and starts paying some lenders in full while leaving others with pennies. Understanding that ranking, *before* you need it, is what separates a lender who knows what they own from one who is about to be surprised.

![The capital structure drawn as a vertical waterfall of stacked layers from secured debt at the top down through senior unsecured, subordinated, preferred equity, and common equity at the bottom, with paid-first at the top and paid-last at the bottom](/imgs/blogs/seniority-recovery-and-the-capital-structure-1.png)

The diagram above is the mental model for everything that follows. The company's claims are stacked like floors in a building, and money fills the building from the top down. Secured debt is the penthouse: it gets paid first, and almost always in full. Below it, in order, sit senior unsecured bonds, then subordinated bonds, then preferred shareholders, and at the very bottom — the basement that floods first and dries out last — the common shareholders, the owners. When the company is making money, the cash flows *up* from the basement to the penthouse as profit. When the company dies, the assets flow *down* from the penthouse to the basement as repayment, and they very often run out before they reach the bottom. This post is about reading that stack: what each floor is, why the order is enforced, and how to turn "where you sit" into a number — the *recovery rate* — that prices the bond. (Everything here is educational, not investment advice; the goal is to understand the mechanism, not to tell you what to buy.)

## Foundations: the building blocks you need first

Let's assemble the vocabulary from zero. Most of these terms appear together for the first time only in a default, so even readers comfortable with healthy bonds often meet them here.

**A bond is a loan, and a loan is a claim.** When you buy a bond you lend the issuer money; in return you hold a *claim* — a legal right to be paid a fixed schedule of coupons and then your principal back at maturity. The healthy-times mechanics of that schedule are covered in [the anatomy of a bond](/blog/trading/fixed-income/anatomy-of-a-bond-par-coupon-maturity-issuer). What matters today is that a claim is not just "money owed" — it is money owed *with a rank*. Two claims for the same dollar amount against the same company can have completely different ranks, and rank is destiny when the money is short.

**Equity is the residual claim — the leftovers.** A company is funded by two kinds of money: *debt* (lenders, who are promised a fixed amount) and *equity* (owners, who get whatever is left after the lenders are paid). Equity is called the *residual claim* for exactly this reason: shareholders are last in line, entitled only to the residue. In good times the residue is enormous — all the profit, all the growth — which is why owning equity can make you rich. In a default the residue is usually nothing, which is why equity is almost always wiped out first.

**The capital structure is the full ranked list.** Stack every claim against a company from the safest (paid first) to the riskiest (paid last) and you have its *capital structure* (often shortened to *cap structure*, or drawn as the *capital stack*). The canonical order is: senior secured debt, then senior unsecured debt, then subordinated (or "junior") debt, then preferred equity, then common equity. We will define each rung in its own section; for now, just hold the shape — a stack, paid top to bottom.

**Seniority means "who gets paid first."** A claim is *senior* to another if it must be paid in full before the other gets anything. The opposite is *subordinated* (or *junior*): a subordinated claim agrees, in advance and in writing, to stand behind the senior claims. Seniority is purely about *order*, not about size or interest rate. A small senior loan outranks a giant subordinated bond.

**Secured vs unsecured: is there collateral?** A *secured* claim is backed by specific *collateral* — a particular asset (a factory, a fleet of planes, the company's receivables) that the lender can seize and sell if it is not paid. An *unsecured* claim has no such pledge; it is a general promise backed only by the company's overall ability to pay. Security is a different axis from seniority — a bond can be senior *and* unsecured — but secured claims effectively sit above unsecured ones for the value of their collateral.

**Default is the broken promise; bankruptcy is the court process that follows.** A *default* is the event of failing to honor the bond's terms — most often missing a coupon or principal payment, sometimes breaching a covenant. *Bankruptcy* is the formal legal process (in the United States, usually a filing under Chapter 11 for reorganization or Chapter 7 for liquidation) that then sorts out who gets paid what. The capital structure is the script the court follows. The full taxonomy of *why* a default happens lives in the credit-risk discussion; here we pick up at the moment the music stops and the assets get divided.

**Recovery rate and loss-given-default.** When the dust settles, a lender gets back some fraction of what they were owed. That fraction is the *recovery rate*, quoted in cents on the dollar — a 40% recovery means you got \$0.40 back for every \$1 of claim. Its mirror image is *loss-given-default*, written **LGD = 1 − recovery**: the fraction you *lost*. A 40% recovery is a 60% LGD. These two numbers are the bridge from "where you sit in the stack" to "what a default actually costs you," and they are the spine of the entire post.

**Why does this order exist at all?** It is worth pausing on, because the ranking is not arbitrary — it is a deal everyone agreed to in advance, and it makes the whole system work. Lenders who accept being paid *first* are willing to lend *cheaply*, because their downside is small. Lenders who accept being paid *later* demand a *higher* coupon, because their downside is large. Equity holders, paid *last*, demand the *most* — all of the upside — because they bear the most risk. By letting a company sell claims at every point along this risk ladder, the capital structure lets it raise money from the widest possible set of investors at the lowest possible blended cost: cautious money buys the senior debt, adventurous money buys the equity, and everyone in between finds a rung that matches their appetite. The order is the price of that arrangement. Without an enforceable ranking, no one would accept the cheap-but-safe senior position, because there would be nothing stopping a junior claimant from grabbing assets first — and the company's cost of capital would be far higher. Seniority, in short, is the machinery that lets risk be sliced and sold.

With those seven ideas in hand, here is the one sentence that motivates everything: **a bond's promised coupon is the same whether it is senior or subordinated, but its recovery in default is not — and the gap between those recoveries is most of the reason one bond is riskier than another from the same company.**

## The capital-structure waterfall, floor by floor

Let's walk the stack from top to bottom, defining each rung and noting, as we go, why it sits where it sits.

**Senior secured debt (the top floor).** This is debt backed by specific collateral *and* ranked senior. The classic example is a bank loan secured by the company's assets, or a *first-lien* bond. A *lien* is a legal claim on a specific asset; a *first lien* is the first in line against that asset. If the company defaults, secured lenders can seize and sell their collateral and keep the proceeds up to the amount they are owed, *before* anyone else touches that money. This is why secured debt recovers the most — it has a dedicated pile of assets reserved for it. Many companies also have *second-lien* debt: secured, but standing behind the first lien on the same collateral, so it only gets the collateral's value left over after the first lien is satisfied.

**Senior unsecured debt (the next floor down).** This is the workhorse of the corporate bond market: a plain bond, senior in ranking, but with no specific collateral pledged. Most investment-grade corporate bonds you can buy are senior unsecured. In default these lenders are paid from the company's *general* assets — everything not already pledged to the secured lenders — and they share that pool *pari passu* (a Latin phrase meaning "on equal footing") with each other. Senior unsecured sits below secured because it has no dedicated collateral, but above everything below it because nothing junior can be paid until senior unsecured is paid in full.

**Subordinated (junior) debt.** This is debt that has explicitly agreed, in its own bond contract, to rank *below* the senior debt. It is still debt — it pays a fixed coupon and has a maturity — but in a default it gets nothing until both the secured and senior unsecured lenders are paid in full. To compensate for sitting lower, subordinated debt pays a higher coupon. *Mezzanine* financing and many bank *Tier 2* instruments live here. Because the money so often runs out before it reaches this floor, subordinated debt's recoveries are dramatically lower than senior debt's.

**Preferred equity.** Now we cross the line from debt into equity. *Preferred* shares are a hybrid: they pay a fixed dividend (like a coupon) and rank above common shares, but they sit below *all* debt. They are equity in a default — they only get paid after every lender, senior and junior, is made whole. In practice, in a hard default, preferred is almost always wiped out alongside common.

**Common equity (the basement).** The owners. The residual claim. In a healthy company, common shareholders capture all the upside — every dollar of profit and growth after the lenders are paid. In a default, they are dead last, entitled only to whatever is left after every other claim is satisfied in full, which is almost always nothing. The *absolute-priority rule* (which we will define in a moment) means equity holders should receive zero until every creditor is paid 100 cents on the dollar — a bar that defaulting companies, by definition, rarely clear.

One nuance worth flagging, because it confuses many newcomers: this single five-rung stack is the *clean textbook* version, and real capital structures are often more layered. A large company might have a first-lien term loan, a second-lien term loan, senior secured notes, senior unsecured notes, senior subordinated notes, junior subordinated notes, *and* multiple classes of preferred stock — a dozen rungs, not five. The principle never changes: it is still a strict order, paid top to bottom, and each additional rung simply inserts another floor in the waterfall. When you analyze a real bond, your first job is to locate its exact rung in *that company's* specific stack, because "senior unsecured" at one company might have two secured layers above it and at another might have none. There is also a parallel set of claims that sit *outside* the voluntary capital structure entirely and often jump ahead of everyone in bankruptcy — *administrative claims* (the lawyers and advisors running the case), certain *tax* claims, and sometimes employee wages and pension obligations. These "super-priority" claims are paid off the very top, before even the secured lenders in some respects, which is one more reason real recoveries can come in below the clean waterfall's prediction.

#### Worked example: the same company, two very different bonds

*Setup.* Northwind Corp has two bonds outstanding, each with \$1,000 face value and an 8% coupon. Bond A is *senior secured*, backed by Northwind's main factory. Bond B is *subordinated*. In healthy years, both pay \$80 a year and look almost identical on a brokerage screen.

*Step 1 — the default hits.* Northwind misses a payment and files for bankruptcy. Its assets, when sold, are worth far less than its total claims. The court applies the waterfall.

*Step 2 — Bond A's outcome.* The secured lenders, including Bond A's holder, are paid first from the factory's sale proceeds. There is enough to cover them in full. Bond A recovers \$1,000 — a 100% recovery, 0% LGD.

*Step 3 — Bond B's outcome.* By the time the waterfall reaches the subordinated floor, the money is gone. Bond B recovers \$120 — a 12% recovery, an 88% LGD.

*Step 4 — read the gap.* Same issuer, same face value, same coupon, same default. Bond A lost nothing; Bond B lost 88 cents on the dollar. *The coupon told you nothing about the risk — the rank in the capital structure told you everything.*

## The absolute-priority rule: why the order is enforced

The capital structure would be a polite suggestion if nothing made companies honor it. What makes it binding is the **absolute-priority rule** (APR): in a liquidation, each tier must be paid *in full* before the next tier down receives anything at all. It is "absolute" because it admits no partial sharing across tiers — a junior creditor cannot receive a cent while a senior creditor is still owed a dollar.

Picture the assets as a fixed pool of water poured into a tower of buckets, from the top. The top bucket (secured) fills completely before a drop spills into the second bucket (senior unsecured). That fills completely before the third (subordinated) gets anything, and so on down to the bottom bucket (common equity). The pour stops when the water runs out — and in a default, it almost always runs out partway down. Every bucket below the water line gets exactly nothing. There is no "everyone takes a 30% haircut"; there is "the top buckets are full and the bottom buckets are dry."

![The bankruptcy waterfall drawn as a pipeline of asset value flowing through five claim tiers in priority order, secured then senior unsecured then subordinated then preferred then common, each tier filled until the asset pool runs dry partway down](/imgs/blogs/seniority-recovery-and-the-capital-structure-3.png)

The figure above traces the pour for Northwind. The \$600M asset pool enters at the top, fills the secured bucket, fills most of senior unsecured, and runs dry there. Subordinated, preferred, and common are all below the water line — they recover nothing.

The reason the rule is so strict is again the deal-in-advance logic from the Foundations section. The whole point of buying senior debt is that you are *promised* you will be paid before the junior holders — that is exactly what you accepted a lower coupon for. If a court could simply override that and split the assets evenly, the promise would be worthless, and no one would ever buy senior debt at a senior price again. The absolute-priority rule protects the bargain that lets companies sell cheap senior debt in the first place. It is the legal enforcement of the contract you signed when you chose your rung on the ladder.

There is one important real-world caveat, and honesty requires stating it: in practice, in a *reorganization* (Chapter 11, where the company keeps operating rather than being liquidated), the absolute-priority rule is sometimes bent. To get a plan approved quickly, senior creditors occasionally agree to hand a small "tip" to junior creditors or even equity — a few cents — in exchange for their cooperation and to avoid a long, value-destroying fight. These are called *APR violations* or *deviations*, and academic studies have found they happen in a meaningful minority of large Chapter 11 cases. They are small in size but real, and they are the reason equity in a bankrupt company is occasionally worth a little more than zero. The rule is the strong default; the deviations are the negotiated exceptions.

#### Worked example: pouring \$600M into the Northwind tower

*Setup.* Northwind's full capital structure, by claim size, top to bottom: secured \$250M, senior unsecured \$400M, subordinated \$200M, preferred \$75M, common (residual). Total debt-and-preferred claims: \$925M. The assets, liquidated, fetch only \$600M.

*Step 1 — fill the secured bucket.* Secured is owed \$250M; \$600M is available. Pay it in full. Recovery: 100%. Remaining pool: \$600M − \$250M = \$350M.

*Step 2 — fill senior unsecured.* Senior unsecured is owed \$400M; only \$350M remains. Pay all \$350M to them. Recovery: \$350M / \$400M = 87.5%. Remaining pool: \$0.

*Step 3 — everything below is dry.* Subordinated (\$200M), preferred (\$75M), and common all receive \$0. Recovery: 0%, 0%, 0%.

*Step 4 — read the result.* The water reached the second floor and stopped. Secured got everything, senior unsecured took a modest 12.5% haircut, and the bottom three tiers were wiped out completely. *Absolute priority means the loss is not shared — it is concentrated entirely on the tiers the money never reached.*

## Recovery rates by seniority: the empirical picture

The waterfall tells you the *order*; history tells you the *numbers*. Over decades of defaults, rating agencies and researchers have measured what each tier actually recovered, and the pattern is exactly what the waterfall predicts: recovery falls monotonically as you descend the stack.

The broad, widely-cited averages (these are long-run figures across many cycles, and any single default can land far from the average) look roughly like this:

| Tier | Typical recovery | Typical LGD | Why |
|---|---|---|---|
| Senior secured (bank loans / 1st lien) | ~60–70% | ~30–40% | dedicated collateral, top of the waterfall |
| Senior unsecured bonds | ~40% | ~60% | general assets, no collateral |
| Subordinated bonds | ~25–30% | ~70–75% | paid only after all senior debt |
| Preferred equity | ~5–10% | ~90–95% | below all debt |
| Common equity | ~0% | ~100% | residual; almost always wiped out |

![A bar chart of average recovery rate by seniority tier showing secured loans recovering the most at around 65 percent, senior unsecured around 40 percent, subordinated around 28 percent, preferred near 8 percent, and common equity near zero](/imgs/blogs/seniority-recovery-and-the-capital-structure-2.png)

The bars above are the single most important empirical fact in this post: **recovery is a staircase that descends as you go down the capital structure.** This is the real, measured link between where you sit and what you get back. It is not a coincidence or a market mood — it is the absolute-priority rule playing out, default after default, across decades of data. The secured lender, with first claim on dedicated collateral, recovers the most; the equity holder, with the residual claim, recovers essentially nothing.

A few honest caveats on these numbers, because recovery is one of the most variable quantities in all of finance:

- **Recoveries are cyclical.** In a recession, *more* companies default at once, *and* their assets fetch lower prices in a glutted market — so recovery rates fall exactly when default rates rise. This double-whammy is why credit losses cluster so violently in downturns. Senior unsecured recovery, ~40% on average, has dipped toward 20% in the worst years and exceeded 60% in benign ones.
- **Industry matters enormously.** A company with hard, sellable assets (utilities, real estate, equipment-heavy businesses) recovers more than an asset-light one (a software firm whose "assets" are people who walk out the door). Collateral you can actually sell is worth more than a promise.
- **Capital structure mix matters.** Counterintuitively, a company with *more* debt above you lowers your recovery, but a company with more debt *below* you (more subordinated cushion) can *raise* your recovery, because that junior debt absorbs the first losses. Where you sit is relative to what is stacked around you.

### How recovery is actually measured (and why the number you read can mean two different things)

"Recovery rate" sounds like a single, clean number, but there are two very different ways to measure it, and confusing them is a classic mistake.

The first is *ultimate recovery*: the total value a creditor finally receives when the bankruptcy is fully resolved — cash, plus any new bonds or new equity handed out in the reorganization — discounted back and compared to the original claim. This is the "true" recovery, but it can take *years* to know, because a Chapter 11 case can drag on for two or three years before creditors get their final distribution.

The second is *trading-price recovery* (or "post-default price"): the price the defaulted bond trades at in the market about a month after default, when distressed-debt investors are actively buying and selling the claim. This is available almost immediately and is what most index-level recovery statistics (and the recovery assumptions baked into credit-default-swap settlements) use. It is the market's *forecast* of the ultimate recovery, expressed as a price today.

These two numbers usually point the same way but can differ meaningfully. A bond might trade at 35 cents a month after default (trading-price recovery) and ultimately pay out 45 cents three years later (ultimate recovery) because the reorganization went better than the market feared — or the reverse. When someone quotes you a recovery rate, it is worth knowing which one they mean, because a strategy built on one (say, buying distressed bonds at the post-default price) lives or dies on the gap between the two.

There is a further subtlety that trips people up: recovery is conventionally measured against the bond's *face value*, not its market price the day before default. A bond already trading at 60 cents (because the market saw trouble coming) that recovers 40 cents of face has *lost* 20 cents from where a recent buyer paid, even though the headline "40% recovery" sounds like a partial save. Recovery is a fraction of par; your actual loss depends on what you paid.

#### Worked example: recovery as a staircase

*Setup.* Imagine a company defaults and you hold \$1,000 face of one of its bonds. Apply the long-run average recoveries to see what \$1,000 of claim is worth at each tier.

*Step 1 — senior secured.* At a 65% recovery, your \$1,000 claim returns \$650. You lose \$350. LGD = 35%.

*Step 2 — senior unsecured.* At a 40% recovery, your \$1,000 claim returns \$400. You lose \$600. LGD = 60%.

*Step 3 — subordinated.* At a 28% recovery, your \$1,000 claim returns \$280. You lose \$720. LGD = 72%.

*Step 4 — common equity.* At a ~0% recovery, your \$1,000 of equity returns roughly \$0. You lose everything. LGD ≈ 100%.

*Step 5 — read the staircase.* Same default, same \$1,000 at stake, four wildly different outcomes: \$650, \$400, \$280, \$0. *The only thing that changed was the rank of the paper you held — and it changed your loss by more than \$600.*

## Secured vs unsecured: what collateral actually buys you

Secured and senior are easy to confuse, but they answer different questions. *Seniority* asks "what order are claims paid in?" *Security* asks "is there a specific asset reserved for this claim?" A secured claim brings its own pile of assets to the waterfall — a pile that other creditors cannot touch until the secured lender is satisfied.

![A before and after comparison showing an unsecured lender sharing the whole asset pool with everyone versus a secured lender with a dedicated collateral claim reserved on a specific asset before the general pool is divided](/imgs/blogs/seniority-recovery-and-the-capital-structure-4.png)

The figure contrasts the two. On the left, the unsecured lender throws their claim into the general pool and shares it with every other unsecured creditor; if the pool is small, everyone takes a proportional haircut. On the right, the secured lender has carved out a specific asset — the factory, the planes, the receivables — as *theirs first*. They take the value of that collateral off the top, before the general pool is even divided. If the collateral is worth more than the loan, they recover in full and the excess flows back into the general pool for everyone else. If it is worth less, they recover the collateral's value as a secured claim and become an *unsecured* creditor for the shortfall — they drop into the general pool for the rest.

That last point is the subtle mechanic worth dwelling on. Security does not guarantee 100% recovery; it guarantees *first claim on a specific asset's value*. If you lent \$500 secured by a machine that turns out to be worth only \$300, you recover \$300 as a secured claim and join the unsecured queue for the remaining \$200 — where you might recover, say, 40 cents, or \$80. Total recovery: \$380 on \$500, or 76%. Collateral is a floor, not a ceiling, and the floor is only as high as the collateral's resale value in a fire sale.

And "fire sale" is the operative phrase. Collateral is almost never sold at its calm, going-concern value, because by definition it is being sold when the borrower has failed — often when the whole *industry* is in trouble, which is exactly when buyers are scarce and prices are low. A specialized factory worth \$200M to a running business might fetch \$120M from the only buyer willing to take it during a downturn. This is why the *type* of collateral matters so much. Cash and marketable securities are worth nearly the same in a crisis as in calm times. Real estate, inventory, and standard equipment hold a good fraction of their value. But highly specialized assets — a half-built semiconductor fab, a single-purpose pipeline, brand-and-people "assets" with no resale market — can collapse to a fraction of their book value precisely when you need to sell them. The strength of a secured claim is the *liquidity* of its collateral as much as the size of it. A first lien on cash is worth more than a first lien on a dream.

#### Worked example: when the collateral does not cover the loan

*Setup.* Northwind borrowed \$300M secured by its factory. In the default, the factory — sold quickly, in a weak market — fetches only \$220M. The general unsecured pool, separately, pays 40 cents on the dollar.

*Step 1 — the secured portion.* The lender takes the factory's \$220M first. That covers \$220M of the \$300M claim at 100 cents.

*Step 2 — the shortfall becomes unsecured.* The remaining \$80M of the claim is now an *unsecured* claim, standing in the general pool with everyone else.

*Step 3 — recover on the shortfall.* At 40 cents on the dollar, the \$80M unsecured shortfall returns \$32M.

*Step 4 — total recovery.* \$220M + \$32M = \$252M on a \$300M claim, a recovery of 84%. *Security is a reserved first claim on an asset's value, not a magic guarantee of par — its worth depends entirely on what the collateral fetches.*

## Structural subordination: the rank that hides in the org chart

There is a second, sneakier way to be junior, and it catches even experienced investors off guard because it is invisible in the bond's own contract. It is called *structural subordination*, and it comes not from a clause that says "I rank below the senior bonds" but from *where in the corporate family tree* your borrower sits.

Big companies are not single legal entities; they are trees of them. There is usually a *holding company* (the "HoldCo") at the top, which owns one or more *operating companies* (the "OpCos") underneath. The OpCos are where the real business happens — the factories, the contracts, the cash. The HoldCo mostly just owns the shares of the OpCos. Here is the trap: a creditor who lends to the HoldCo has a claim only on the HoldCo's assets, and the HoldCo's main asset is *the equity of the OpCos* — which, you now know, is the residual claim, paid last. A creditor who lends directly to the OpCo, by contrast, has a claim on the actual operating assets, ahead of the HoldCo entirely.

So even if a HoldCo bond is labeled "senior unsecured," it is *structurally subordinated* to all the debt at the OpCo level. In a default, the OpCo's own lenders are paid in full from the OpCo's assets first; only the leftover equity value of the OpCo flows up to the HoldCo, where the HoldCo's "senior" bondholders finally get their turn. The word "senior" on a HoldCo bond describes its rank *among HoldCo creditors* — which can be a thin slice of nothing if the OpCo debt soaks up all the value below it.

Picture the family tree: value is generated at the OpCo, claimed first by OpCo creditors, and only the residue rises to the HoldCo for its bondholders. This is why analysts always ask *where* in the structure a bond is issued, not just what it is called. The same logic explains *guarantees* and *upstream/downstream* arrangements: a HoldCo bond that carries a guarantee from the OpCos effectively pulls itself back up to rank alongside the OpCo debt, curing the structural subordination. Reading whether a bond has such guarantees is a core part of credit analysis.

#### Worked example: the HoldCo bond that looked senior

*Setup.* Northwind reorganizes into a HoldCo that owns one OpCo. The OpCo runs the whole business and has borrowed \$400M directly (OpCo senior debt). The HoldCo issued \$300M of bonds labeled "senior unsecured." In a default, the OpCo's assets fetch \$450M.

*Step 1 — pay the OpCo lenders first.* The OpCo's \$400M of direct lenders are paid from the OpCo's \$450M of assets. They recover 100%, leaving \$50M of OpCo value.

*Step 2 — what flows up to the HoldCo.* That leftover \$50M is the OpCo's equity, which the HoldCo owns. It flows up to the HoldCo as the HoldCo's only real asset.

*Step 3 — pay the HoldCo bonds.* The HoldCo's \$300M of "senior" bonds now divide \$50M. Recovery: \$50M / \$300M = 16.7%.

*Step 4 — read the trap.* The HoldCo bond said "senior unsecured" and recovered just 17 cents, while the OpCo lender — never labeled "senior" at all — recovered 100 cents. *Structural subordination means the family-tree position can outrank the word printed on the bond; always ask where the debt sits, not just what it is called.*

## Covenants, intercreditor agreements, and the documents that set the order

The capital-structure waterfall is not folklore — it is written down, claim by claim, in the bond's legal documents. Two kinds of documents do most of the work, and knowing they exist changes how you read a bond.

**Covenants** are promises the borrower makes in the bond contract (the *indenture*). Some protect your *seniority* directly. A *negative pledge* covenant, for example, promises the company will not later pledge its assets as collateral to *other* lenders — protecting an unsecured bondholder from waking up one day to find that new secured lenders have jumped ahead of them in the queue. A *limitation on indebtedness* covenant caps how much additional debt the company can pile on, protecting you from being buried under new claims. Weak or absent covenants ("covenant-lite" loans, which became common in the 2010s) mean the company can erode your position after you have already lent the money — a real, if slow-moving, risk to your eventual recovery.

**Intercreditor agreements** are contracts *among the lenders themselves* that spell out the pecking order explicitly: who gets paid first, who can seize which collateral, who must stand aside in a default. When a company has first-lien and second-lien debt on the same collateral, an intercreditor agreement is what makes the second lien actually second. These agreements have become a battleground in modern restructurings — the infamous "liability management exercises" and "creditor-on-creditor violence" of recent years are fights over exactly these documents, where some creditors find clever ways to move *ahead* of others who thought they were safe. The lesson is the same one this whole post teaches, sharpened: your place in the stack is only as solid as the documents that define it.

#### Worked example: a negative pledge that saved a recovery

*Setup.* You hold \$1,000 of Northwind senior unsecured bonds, expecting to share the general asset pool with other unsecured creditors. The general pool, in a default, would pay 40 cents on the dollar. Your bond has a negative-pledge covenant.

*Step 1 — what could have happened without the covenant.* Without a negative pledge, Northwind could later borrow \$200M from a new lender and pledge its best assets as collateral. Those assets would be carved out of the general pool, shrinking what is left for you. Suppose that would have dropped the general-pool recovery to 25 cents.

*Step 2 — what the covenant prevents.* The negative pledge bars Northwind from making that pledge. The best assets stay in the general pool, available to all unsecured creditors.

*Step 3 — the recovery difference.* With the covenant, you recover \$400 (40%); without it, you would have recovered \$250 (25%). The covenant was worth \$150 of recovery on your \$1,000 bond.

*Step 4 — read the result.* *Covenants are not boilerplate — they are the rules that protect your rank from being eroded after you have lent, and their strength shows up directly in recovery.*

## From recovery to loss-given-default — and to the spread

Now we connect this back to the thing every bond investor is actually paid for: the *spread*. Recall the master decomposition of credit risk — the expected loss on a bond is the product of three things:

$$\text{Expected Loss} = PD \times LGD \times EAD$$

where $PD$ is the *probability of default* (the chance the issuer fails to pay), $LGD$ is the *loss-given-default* (the fraction you lose *if* it fails, which is $1 - \text{recovery}$), and $EAD$ is the *exposure at default* (how much you have at risk, usually the face value). The first term — how *likely* default is — is the same for every bond from one issuer; a company either defaults or it doesn't, and when it does, all its bonds default together. **Seniority does not change** $PD$. **What seniority changes is** $LGD$. The senior secured bond and the subordinated bond from Northwind have the *same* probability of default — but the secured bond loses ~35 cents in that default while the subordinated bond loses ~72 cents.

That difference in expected loss is exactly what the *credit spread* compensates. The spread is the extra yield a corporate bond pays over a [risk-free Treasury](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration) of the same maturity, and to a first approximation it must at least cover the expected loss: $\text{spread} \approx PD \times LGD$. Hold $PD$ fixed and push $LGD$ up (move down the capital structure) and the spread must widen to compensate. This is the precise, mechanical reason a subordinated bond yields more than a senior bond from the same issuer — not because the company is "more likely to fail" for the junior holder, but because the *loss is bigger* if it does.

A quick honesty check on that formula, because it is a floor, not the whole story. The real spread is wider than $PD \times LGD$ for two reasons beyond the raw expected loss. First, investors demand a *risk premium* for bearing the *uncertainty* of default — not just the average loss, but the fact that defaults cluster in bad times when you can least afford them, so credit losses hurt more than their average size suggests. Second, corporate bonds are less *liquid* than Treasuries — harder to sell quickly without moving the price — and that illiquidity earns its own slice of spread. So the full spread is roughly expected loss, *plus* a risk premium, *plus* a liquidity premium. But the seniority effect runs cleanly through the first term: holding the issuer and the maturity fixed, the *difference* in spread between two of its bonds is overwhelmingly the difference in their loss-given-default, because the risk and liquidity premia are similar for two bonds of the same name. Seniority is the cleanest, most isolated driver of the spread gap *within* an issuer.

![A matrix showing loss given default by seniority tier as one minus recovery, with secured at thirty five percent, senior unsecured at sixty percent, subordinated at seventy two percent, preferred at ninety two percent, and common equity at one hundred percent](/imgs/blogs/seniority-recovery-and-the-capital-structure-5.png)

The matrix above flips the recovery staircase into its mirror image: loss-given-default. It is the same information — LGD is just 1 minus recovery — but framed as "what you lose," which is what feeds the spread formula. Read it as the second factor in the expected-loss product, the one that seniority controls.

![An XY chart with seniority on the horizontal axis and credit spread in basis points on the vertical axis, showing the spread rising as the bond moves from secured to senior unsecured to subordinated, illustrating that lower seniority means a wider spread for the same issuer](/imgs/blogs/seniority-recovery-and-the-capital-structure-6.png)

The chart makes the link visible: hold the issuer (and so the default probability) fixed, and the spread climbs as you descend the capital structure. The secured bond, with the smallest loss-given-default, sits at the bottom-left with the tightest spread; the subordinated bond, with the largest LGD, sits at the top-right with the widest. The slope of that line *is* seniority being priced. For the full machinery of how spreads are quoted, decomposed, and traded, see [investment-grade and high-yield credit spreads](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads).

#### Worked example: pricing the seniority gap into the spread

*Setup.* Northwind's bonds have a one-year probability of default of 5%. The senior secured bond has an LGD of 35%; the senior unsecured bond, 60%; the subordinated bond, 72%. Use the rough rule that the credit spread must cover the expected loss.

*Step 1 — the secured bond's expected loss.* $PD \times LGD = 0.05 \times 0.35 = 0.0175$, or 1.75% — about **175 basis points** of expected annual loss. (A *basis point* is one hundredth of a percent, 0.01%.) The spread must be at least ~175 bps to break even.

*Step 2 — the senior unsecured bond's expected loss.* $PD \times LGD = 0.05 \times 0.60 = 0.030$, or 3.0% — about **300 basis points**. Same default odds, deeper loss, so the spread sits between the secured and subordinated bonds.

*Step 3 — the subordinated bond's expected loss.* $PD \times LGD = 0.05 \times 0.72 = 0.036$, or 3.6% — about **360 basis points**. The spread must be at least ~360 bps.

*Step 4 — the seniority premium.* The subordinated bond must pay roughly 360 − 175 = **185 bps more** than the secured bond, purely to compensate for the deeper loss. Same company, same default odds; the extra yield is entirely the price of sitting lower.

*Step 5 — read the result.* *The probability of failure was identical for all three bonds; the spread gap between them is the market pricing the loss-given-default that seniority controls.* (Real spreads also embed liquidity, risk premia, and the chance of default rising — this is the expected-loss floor, not the whole spread.)

## Putting it all together: the full Northwind waterfall

Let's run the complete recovery waterfall once, end to end, so every number in the post lands in one place. This is the figure to keep when you forget everything else.

![A matrix laying out the full Northwind recovery waterfall with each tier as a row showing claim amount, dollars paid from the six hundred million asset pool, recovery percentage, and loss given default, from secured at one hundred percent down to common equity at zero](/imgs/blogs/seniority-recovery-and-the-capital-structure-7.png)

The matrix above is the whole story on one grid. Read it row by row, top to bottom, watching the "paid" column drain the \$600M pool:

| Tier | Claim | Paid from \$600M | Recovery | LGD |
|---|---|---|---|---|
| Senior secured | \$250M | \$250M | 100% | 0% |
| Senior unsecured | \$400M | \$350M | 87.5% | 12.5% |
| Subordinated | \$200M | \$0 | 0% | 100% |
| Preferred equity | \$75M | \$0 | 0% | 100% |
| Common equity | residual | \$0 | 0% | 100% |
| **Total** | **\$925M + equity** | **\$600M** | — | — |

#### Worked example: cents on the dollar, tier by tier

*Setup.* This is the same \$600M-against-\$925M Northwind default, now read as cents on the dollar for a holder of \$1,000 face in each tier.

*Step 1 — secured holder.* Owns \$1,000 of the \$250M secured tranche, which recovered 100%. Gets back \$1,000. Loss: \$0.

*Step 2 — senior unsecured holder.* Owns \$1,000 of the \$400M senior tranche, which recovered 87.5%. Gets back \$875. Loss: \$125.

*Step 3 — subordinated holder.* Owns \$1,000 of the \$200M subordinated tranche, which recovered 0%. Gets back \$0. Loss: \$1,000.

*Step 4 — the spread that should have been there.* The senior holder's 12.5% loss, if it carried (say) a 5% default probability, implies an expected loss of $0.05 \times 0.125 = 0.625\%$, about 63 bps — modest. The subordinated holder's 100% loss in this default implies $0.05 \times 1.00 = 5\%$, about 500 bps. *The same \$600M shortfall produced a 12.5% haircut at one floor and a total wipeout one floor below — and that cliff is exactly what the extra subordinated yield was paying for.*

## Common misconceptions

**"A bond from a strong company is safe regardless of which bond it is."** No — the *issuer's* health sets the probability of default, but *your bond's* place in the capital structure sets your loss if default comes. A subordinated bond from an investment-grade company can still hand you an 80% loss in the rare event it defaults. Strong issuer, weak position is a real and underappreciated combination.

**"Secured means I am guaranteed to get my money back."** No — secured means you have *first claim on a specific asset's value*, not a guarantee of par. If the collateral fetches less than your loan in a fire sale, you recover the collateral's value and become an unsecured creditor for the rest. Security is a floor set by resale value, not a ceiling at 100 cents.

**"In bankruptcy, everyone takes a proportional haircut."** No — the absolute-priority rule means losses are *concentrated, not shared*. Senior tiers are paid in full while junior tiers get nothing; there is no across-the-board 30% cut. The loss falls entirely on the floors the money never reached. (Reorganizations sometimes hand junior creditors a small negotiated tip, but that is a deviation, not proportional sharing.)

**"Higher coupon means higher quality."** Often the reverse. Within one issuer, the *subordinated* bond pays a higher coupon precisely because it ranks lower and would recover less in default. A fatter coupon is compensation for risk, not a sign of safety. Always check seniority before reading a coupon as good news.

**"Recovery rates are stable, so I can plug in 40% and forget it."** No — recovery is one of the most cyclical numbers in finance. It falls in recessions (more defaults, lower asset prices, at the same time) and varies hugely by industry and by how much debt sits around you. The 40% average for senior unsecured spans a real-world range from ~20% to ~60%-plus depending on the cycle.

**"Preferred stock is basically a safe bond because it pays a fixed dividend."** No — "preferred" describes its rank *above common equity*, not above debt. In a default, preferred sits below *every* bond, senior and subordinated alike, and is almost always wiped out with the common. The fixed dividend makes it *feel* bond-like in good times; the capital structure makes it equity-like in bad ones.

**"If I hold the senior bond, I do not care about the company's other debt."** You should care a great deal. Your recovery is *relative* to everything stacked around you. New secured debt issued ahead of you (absent a negative-pledge covenant) lowers your recovery by carving assets out of your pool. A thick layer of subordinated debt *below* you can *raise* your recovery, because that junior cushion absorbs the first losses before they reach you. The phrase "senior" only has meaning in the context of the full stack — you cannot judge your position without reading the whole structure, including the debt that has not been issued yet but could be.

**"A default means I lose everything."** Usually not, if you are senior. The whole point of this post is that default and total loss are different events: a senior secured lender often recovers the majority of their money in a default, while only the equity holder reliably loses it all. Conflating "the company defaulted" with "my bond is worthless" is the single most common beginner error — the recovery staircase exists precisely because the loss depends entirely on where you sat.

## How it shows up in real markets

**Lehman Brothers, 2008 — the seniority cliff in real time.** When Lehman filed for bankruptcy in September 2008, its capital structure split apart exactly as the waterfall predicts. Senior unsecured bondholders ultimately recovered something on the order of 20–40 cents on the dollar through the long liquidation that followed (well below the long-run ~40% average, because it was the worst possible time to be selling assets), while Lehman's subordinated debt and its common equity were effectively wiped out. Two creditors of the same failed bank, separated only by a line in the capital structure, walked away with radically different outcomes — the textbook illustration of why rank, not issuer, decided the loss.

**General Motors, 2009 — when the absolute-priority rule got bent.** GM's bankruptcy is the most famous modern example of an APR deviation. In the politically-charged restructuring, the company's senior unsecured bondholders were offered a recovery that many of them argued was worse, relative to their rank, than what certain junior stakeholders (notably the union retiree health-care trust) received. The episode became a lasting case study in how, in a large reorganization, negotiation and policy can bend the strict priority order — and a reminder that the rule is a strong default, not an iron law, once a case enters the messy reality of Chapter 11.

**The 2020 energy default wave — collateral and cyclicality together.** When oil prices collapsed in early 2020, a wave of US shale and energy companies defaulted at once. The episode showcased two of the post's caveats simultaneously: recoveries fell sharply because *everyone* was defaulting and selling assets into the same glutted market (cyclicality), and recoveries varied wildly by *what* the company owned — firms with hard, sellable reserves and pipelines recovered more than asset-light drillers. Senior secured reserve-based loans recovered far better than the senior unsecured bonds stacked above the equity but below the bank debt.

**Credit Suisse AT1 bonds, 2023 — the order turned upside down.** In the emergency takeover of Credit Suisse by UBS, roughly \$17 billion of the bank's *Additional Tier 1* (AT1) contingent-convertible bonds were written down to *zero* while shareholders — who sit *below* those bonds in the normal capital structure — received some value in the UBS share exchange. This stunned the market because it appeared to invert the waterfall: a debt-like instrument was wiped out while equity below it was paid. The fine print of AT1 bonds (a special bank-capital instrument designed to absorb losses *before* a bank fails) and Swiss law permitted it, but the episode was a violent, real-world lesson that the "normal" order can be rewritten by the specific contract terms — and that you must read what your paper actually says, not assume the textbook stack. (For the broader story, see [the SVB and Credit Suisse bank runs of 2023](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

**Leveraged buyouts and the loan-versus-bond split.** When a private-equity firm buys a company with borrowed money — a *leveraged buyout* — it almost always stacks the debt deliberately: a big layer of *senior secured term loans* at the bottom of the risk pile, topped by *senior unsecured high-yield bonds*, sometimes with subordinated or "PIK" (payment-in-kind) notes above that. The loans get the collateral and the covenants; the bonds get a higher coupon for sitting behind them. When such a company struggles — the retail and telecom LBOs of the 2010s are full of examples — the recovery split is stark and predictable: the secured term-loan lenders recover most of their money while the unsecured bondholders take heavy losses. The capital structure was engineered on day one, and the recovery outcome was baked in from the start. This is the everyday, non-headline version of the Northwind example, playing out in the leveraged-finance market every cycle.

**Why ratings agencies grade bonds, not just companies.** Every major rating agency assigns ratings at the *instrument* level, not only the issuer level — and they explicitly *notch* a subordinated bond's rating below the issuer's senior rating to reflect its lower recovery. A company might carry a senior unsecured rating of BBB while its subordinated bond is rated BB, two notches lower, for the exact mechanism in this post: same default probability, higher loss-given-default. This is the capital-structure waterfall encoded directly into the rating scale. (See [how the rating agencies work](/blog/trading/finance/credit-rating-agencies-moodys-sp-fitch).)

## When this matters to you, and where to go next

Seniority is invisible in good times and decisive in bad ones, which is exactly why it is so easy to ignore until it bites. If you ever buy a single corporate bond, a high-yield bond fund, or a preferred-stock fund, you are taking a position in someone's capital structure — and the recovery you would get in a default is set the day you buy, by the rank of the paper, not by the headlines. The practical takeaway is small and durable: before you judge a bond by its coupon or its issuer's name, find out where it sits in the stack, because that placement is most of the difference between getting your money back and getting a thank-you note.

From here, the natural next steps deepen each thread this post opened. To see how recovery and default probability combine into the *price* of credit, read the cross-asset treatment of [corporate credit and spreads](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads) and the anchor it floats above, [government bonds as the risk-free benchmark](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration). To see seniority encoded into a grading scale, read [how the rating agencies work](/blog/trading/finance/credit-rating-agencies-moodys-sp-fitch). For the macro backdrop — why default and recovery cluster in cycles driven by [interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) — step up a level to the policy lens. And for the quantitative machinery that turns these ideas into priced instruments, the heavy-math companion on [bond pricing](/blog/trading/quantitative-finance/bond-pricing) carries the formulas the rest of the way.

---
title: "Bankruptcy and restructuring law: Chapter 11, the creditor stack, and the fulcrum security"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "When a company fails, bankruptcy law decides exactly who gets paid and who is wiped — and distressed investors trade that law directly by finding the fulcrum security."
tags: ["bankruptcy", "chapter-11", "distressed-debt", "restructuring", "creditor-stack", "fulcrum-security", "absolute-priority", "credit", "regulation", "law-and-markets"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — When a company can no longer pay its debts, bankruptcy law — not negotiation, not luck — decides exactly who gets paid and who gets wiped, and distressed investors trade that law directly.
>
> - The capital structure is a strict priority ladder: secured debt is paid first, then senior unsecured bonds, then subordinated debt, then preferred equity, and common shareholders last. The **absolute-priority rule (APR)** says no junior layer recovers a cent until the layer above it is paid in full.
> - The **fulcrum security** is the layer where the company's value runs out — the claim that gets paid partly, and that typically converts into the equity of the reorganized firm. Own the fulcrum and you own the recovery.
> - Chapter 11 reorganizes a viable business; Chapter 7 liquidates a dead one. The automatic stay freezes creditors, DIP financing keeps the lights on with super-priority, and cramdown lets a court bind dissenting creditors to a plan.
> - The one number to remember: in a typical large-corporate default, **secured first-lien debt has historically recovered around 60–70 cents on the dollar, senior unsecured around 40 cents, and subordinated debt under 20** — the stack is a gradient of pain, and the fulcrum sits right at the edge of it.

On the morning of September 15, 2008, Lehman Brothers filed the largest bankruptcy petition in American history: more than \$600 billion in assets, over a hundred thousand creditors. In the months that followed, something strange happened in the trading pits. Lehman's senior bonds — debt of a firm that had just publicly imploded — did not go to zero. They traded. First at \$0.35 on the dollar, then \$0.20, then settling for years in a band as lawyers fought over the carcass. Hedge funds bought them by the hundreds of millions. Why would anyone pay real money for the IOUs of a bankrupt company?

Because the bonds were not a bet on Lehman surviving. They were a bet on a number: how much an estate worth tens of billions would ultimately return to *that specific layer* of the capital structure once the law had finished sorting out who stood ahead of whom. The buyers were not gambling. They were pricing a legal waterfall. When Lehman's plan finally paid out, senior unsecured creditors recovered roughly \$0.21 on the dollar — and the funds that bought at \$0.12 or \$0.15 in the darkest months made multiples of their money. They had read the law correctly when the market was reading the headlines.

This is the heart of distressed investing, and it is one of the purest examples of a theme that runs through this whole series: **a rule, not a sentiment, sets the price.** When a company fails, bankruptcy law is the rulebook that determines, claim by claim, dollar by dollar, who recovers what. The investors who win in distress are the ones who can build the recovery waterfall in their heads, find the exact layer where the money runs out, and buy *that* security — the fulcrum — before everyone else figures out it is the one that matters.

![Pipeline from a failed company through bankruptcy law to the creditor stack the fulcrum security and the distressed trade](/imgs/blogs/bankruptcy-and-restructuring-law-chapter-11-and-the-creditor-stack-1.png)

## Foundations: the capital structure, absolute priority, and the two kinds of bankruptcy

Before we can trade any of this, we need the vocabulary. Every term below is something a distressed analyst uses ten times before lunch, so let us build each one from zero.

### The capital structure (the creditor stack)

A company funds itself with a mix of **debt** (money it borrows and must repay) and **equity** (ownership stakes it sells). But not all debt is equal, and that inequality is the entire game. The **capital structure** — traders call it the **creditor stack** or the **cap stack** — is the ranked list of everyone who has a claim on the company, ordered by who gets paid first if things go wrong.

Think of it as a ladder. From top (safest, paid first) to bottom (riskiest, paid last):

1. **Secured debt** — loans backed by specific **collateral** (assets pledged to the lender, like real estate, equipment, or receivables). A **first-lien** term loan or revolving credit line sits here. If the company defaults, the secured lender has a legal right to seize and sell its collateral. This is the safest rung.
2. **Senior unsecured debt** — bonds and loans with no specific collateral, but a contractual right to be paid before junior creditors. "Senior" means senior to the layers below, not that it has collateral.
3. **Subordinated (junior) debt** — debt that has contractually agreed to stand *behind* senior debt. Often called "sub debt" or "mezzanine." It accepts a worse position in exchange for a higher interest rate.
4. **Preferred equity** — a hybrid: technically ownership, but with a fixed dividend and a claim that ranks ahead of common shareholders. Paid only after all debt.
5. **Common equity** — the ordinary shareholders. They own the upside if the company thrives, but in bankruptcy they are the **residual claimant**: they get whatever is left after *everyone* else is paid, which in most failures is nothing.

![Five-layer creditor stack from secured debt at top to common equity at bottom showing paid first and paid last](/imgs/blogs/bankruptcy-and-restructuring-law-chapter-11-and-the-creditor-stack-2.png)

The deeper you sit in this stack, the higher the interest rate you demanded when you lent — because you are first in line for the losses. That is the iron law of credit: **you are paid for your place in the line.** A first-lien loan might yield 6 percent; the subordinated notes of the same company might yield 12 percent. The extra 6 points is the market pricing the extra risk of being lower in the stack. We cover this seniority gradient in detail in the fixed-income companion piece on [seniority, recovery, and the capital structure](/blog/trading/fixed-income/seniority-recovery-and-the-capital-structure).

### The absolute-priority rule (APR)

The rule that gives the ladder its teeth is the **absolute-priority rule**, codified in Section 1129(b) of the U.S. Bankruptcy Code. It states, in plain terms: **a junior class cannot receive anything until every senior class above it has been paid in full.** Not "mostly paid." In full.

This is what makes the stack a *strict* order rather than a rough guideline. If the secured lenders are owed \$300 million and the available value is \$250 million, the secured lenders take all \$250 million and *everyone below them gets zero* — the senior unsecured, the subordinated, the preferred, the common, all of it, zero. APR is the difference between bankruptcy being a lottery and being a calculation. Distressed investing only works because APR makes the outcome computable.

> [!note]
> APR is "absolute" in theory but bends in practice. Senior creditors sometimes hand a junior class a token recovery — a "tip" or "gift" — to buy their cooperation and speed the plan along, even though strict APR says they deserve nothing. Courts have wrestled with these "gifting" deals for decades. The principle holds; the edges are negotiated.

There is one more wrinkle that matters for trading: claims are grouped into **classes**, and the entire voting and cramdown machinery operates class-by-class, not creditor-by-creditor. A class is a bucket of similar claims — all the first-lien lenders are one class, all the senior unsecured bondholders another, and so on. A class is **impaired** if the plan gives it less than it is legally owed (anything short of payment in full, or a change to its rights). Only impaired classes get to vote; unimpaired classes are deemed to accept because they are made whole. This is why the *fulcrum* class is the politically powerful one in a Chapter 11: it is impaired (so it votes), it is the most senior impaired class (so it usually controls the plan), and it is the class that absorbs the conversion to equity. The classes above it are unimpaired and irrelevant to the negotiation; the classes below it are wiped and have no leverage except nuisance litigation. Almost every important fight in a restructuring is a fight *within or about the fulcrum class*.

### Administrative claims and the true top of the stack

One subtlety the clean five-layer ladder hides: there is a layer *above* even the secured debt, and distressed investors ignore it at their peril. **Administrative claims** — the professional fees of the lawyers, bankers, and restructuring advisors, plus post-petition trade debt and the DIP loan — are paid *first*, ahead of pre-petition secured creditors, as the cost of running the case. In a contested, multi-year bankruptcy these fees can run into the hundreds of millions. A waterfall that ignores admin claims overstates every recovery below it. When you build the stack, the real top is not "secured debt" — it is "DIP loan + administrative claims," and only then the pre-petition secured layer. The longer and more litigious the case, the more this top layer eats, which is precisely why a dragged-out bankruptcy is bad for *every* impaired creditor: the fee meter runs ahead of all of them.

### Chapter 11 versus Chapter 7

U.S. bankruptcy comes in two main flavors for businesses, named for chapters of the Bankruptcy Code:

- **Chapter 11 — reorganization.** The company keeps operating as a **going concern** (a living business, not a pile of assets) while it restructures its debts. Management usually stays in control. The goal is to emerge as a smaller, leaner company that can survive. This is what large, viable-but-overleveraged companies file.
- **Chapter 7 — liquidation.** The company stops operating. A court-appointed **trustee** sells off the assets piece by piece and distributes the cash down the creditor stack. This is for companies with no viable business left — the value is in the assets, not the enterprise.

The distinction is enormous for recovery. A reorganized airline that keeps flying is worth far more than the same airline sold off as gates, planes, and landing slots. A going concern preserves the value of the business as a whole; a liquidation realizes only the **fire-sale** value of the parts.

![Comparison matrix of Chapter 11 reorganization versus Chapter 7 liquidation across goal firm management recovery and best use](/imgs/blogs/bankruptcy-and-restructuring-law-chapter-11-and-the-creditor-stack-8.png)

### The automatic stay

The moment a company files for bankruptcy, an **automatic stay** snaps into place (Section 362). It is an instant, court-enforced freeze on *all* collection activity: no creditor may seize assets, foreclose on collateral, continue a lawsuit, or even call demanding payment. The lights stay on; the chaos of a creditor stampede is halted.

The stay is the single most important protection in Chapter 11. Without it, the first creditor to grab an asset would win and the rest would be left fighting over scraps — a destructive scramble that would destroy the going-concern value everyone is trying to preserve. The stay buys the time to do the math and write a plan. It is, in effect, a legal time-out that converts a riot into an orderly process.

The stay has one famous exception that matters enormously for trading desks: certain financial contracts — derivatives, repos, securities-lending, and swaps — enjoy a **safe harbor** that lets counterparties terminate and net out their positions *despite* the stay. This was a central drama of the Lehman collapse: counterparties raced to close out their swaps and seize collateral the instant Lehman filed, while ordinary creditors were frozen. The safe harbor exists to stop a single firm's bankruptcy from cascading through the derivatives market, but it also means a sophisticated counterparty's claim can behave very differently from a plain bondholder's. When you build a waterfall for a financial firm, the safe-harbored contracts are effectively a separate, faster line that bypasses the stay entirely.

### Debtor-in-possession (DIP) financing

A bankrupt company still needs cash to operate — to make payroll, pay suppliers, keep the doors open. But who would lend to a company that just defaulted? The answer is **debtor-in-possession (DIP) financing**: new loans extended *during* the bankruptcy, which the court grants **super-priority** (Section 364). Super-priority means the DIP lender jumps to the *very top* of the stack — ahead of even the pre-petition secured lenders — for the new money it advances.

DIP lending is one of the safest forms of credit in finance precisely because of this legal jump. The lender is first in line, the loan is short, and the company's survival depends on it. We will price a DIP loan's return below; for now, hold the idea that **the law can re-order the stack by creating a new, higher rung.**

### The plan of reorganization and cramdown

The output of Chapter 11 is a **plan of reorganization** — a court-approved document that says exactly what each class of creditors receives: cash, new debt, new equity, or some mix. Creditors vote on the plan by class. A class **accepts** if creditors holding at least two-thirds of the dollar amount and a majority in number vote yes.

But what if a class votes no? Here is where **cramdown** comes in (the colorful name for the power in Section 1129(b)). A court can confirm a plan over the objection of a dissenting class — "cram it down their throats" — as long as the plan is "fair and equitable," which essentially means it respects absolute priority. Cramdown is the stick that forces holdouts to accept a plan they dislike, so long as the plan does not violate the priority order. It is the legal mechanism that prevents a single stubborn creditor from blocking a restructuring everyone else has agreed to.

### Recovery rate and the fulcrum security

Two final terms, and they are the ones the whole post turns on.

A **recovery rate** is the percentage of a claim's face value that the holder ultimately gets back. A bond with a \$1,000 face value that pays out \$400 has a 40 percent recovery, or "40 cents on the dollar." Recovery rates fall as you go down the stack: secured debt recovers a lot, common equity usually recovers nothing.

The **fulcrum security** is the specific layer of the capital structure where the company's value *runs out* — the claim that is only partially paid. Everything above the fulcrum recovers in full (it is "money-good"); everything below it recovers nothing (it is "out of the money"); and the fulcrum itself sits on the knife's edge, recovering somewhere between zero and one hundred cents. Because the fulcrum is the layer that bears the marginal loss, it is also, under APR, the layer that typically **converts into the new equity** of the reorganized company. Find the fulcrum, and you have found the security that will own the business when it emerges.

That is the entire thesis of distressed investing in one sentence: **buy the fulcrum, control the recovery.**

## How the recovery waterfall works

Let us make the abstraction concrete with the single most important mechanic in the field: the **recovery waterfall**.

When a company is restructured, the first step is to estimate its **enterprise value (EV)** — what the business is worth as a going concern, independent of how it is financed. (Usually EV is estimated as a multiple of cash flow, or via a comparable-company analysis, or a discounted-cash-flow model.) That EV is the pool of value available to distribute. Then you pour it down the stack, top to bottom, paying each claim in full until the pool is empty. The layer where it empties is the fulcrum.

![Recovery waterfall showing enterprise value of 600 million cascading down secured senior subordinated and equity layers with a cutoff line marking the fulcrum](/imgs/blogs/bankruptcy-and-restructuring-law-chapter-11-and-the-creditor-stack-3.png)

#### Worked example: a full recovery waterfall

Consider a manufacturer, Atlas Industrial, that has defaulted. Its capital structure:

- Secured first-lien term loan: \$300 million claim
- Senior unsecured bonds: \$400 million claim
- Subordinated notes: \$200 million claim
- Common equity (the shareholders)

Total debt claims: \$900 million. Now a restructuring banker values the going-concern business at an **enterprise value of \$600 million**. We pour that \$600 million down the stack:

1. **Secured first lien (\$300M claim):** paid first, in full. Pays out \$300 million. Recovery = 300 / 300 = **100 percent**. Remaining value: 600 − 300 = \$300 million.
2. **Senior unsecured (\$400M claim):** next in line. Only \$300 million is left, but they are owed \$400 million. They take all \$300 million. Recovery = 300 / 400 = **75 percent**. Remaining value: 300 − 300 = \$0.
3. **Subordinated notes (\$200M claim):** nothing left. Recovery = **0 percent**.
4. **Common equity:** nothing left. **Wiped out** — recovery = 0.

The senior unsecured bonds are the **fulcrum**: the value ran out *inside* their claim. They are paid partly (75 cents), so they are neither money-good nor worthless — they are the marginal claim. Everything above them is whole; everything below is zero.

The intuition: enterprise value is a fixed pool that fills the stack from the top, and the fulcrum is simply the layer the water level stops at.

Note what *did not* matter: the coupons, the maturities, the original yields. Once a company is being restructured, the only things that count are the **face amount of each claim** and its **priority**. A bond paying a 4 percent coupon and a bond paying a 9 percent coupon, both senior unsecured with the same face value, recover identically — they are the same claim in the waterfall. This is why distressed desks stop quoting bonds in yield and start quoting them in price: yield assumes you collect coupons, and a defaulted bond pays no coupons, only a recovery. The whole pricing language of the market shifts the moment a company crosses into distress.

#### Worked example: from a credit spread to an implied recovery

Before a company ever files, the bond market is already pricing its default and recovery through the **credit spread** — the extra yield a risky bond pays over a risk-free Treasury. There is a rough but powerful relationship: spread ≈ probability of default × loss given default, where loss given default is (1 − recovery rate).

Suppose Atlas's senior bonds trade at a spread of **900 basis points** (9 percentage points) over Treasuries, and the market's estimate of the annual default probability is **15 percent**. We can back out the recovery the market is assuming:

- Spread ≈ default probability × (1 − recovery).
- 0.09 ≈ 0.15 × (1 − recovery).
- 1 − recovery ≈ 0.09 / 0.15 = 0.60.
- Implied recovery ≈ **40 percent** = \$0.40 on the dollar.

If your own waterfall says the senior bonds are the fulcrum and will recover **75 cents**, the bond market — even before any filing — is mispricing the recovery by 35 points. You can buy the bonds *now*, while they still trade as performing credit, and capture the gap as the market wakes up to the true recovery. The intuition: the credit spread is the recovery rate in disguise, and a distressed analyst who can compute the waterfall can spot when the spread implies a recovery that the capital structure cannot support. This spread-to-default math is developed fully in the fixed-income piece on [credit spreads and the probability of default](/blog/trading/fixed-income/credit-spreads-pricing-the-probability-of-default).

#### Worked example: identifying the fulcrum when the value is different

The fulcrum is not a fixed bond — it *moves* with the enterprise value. Take the same Atlas stack, but now suppose the business is worth only **\$250 million** (a worse outcome):

1. Secured first lien (\$300M claim): only \$250M available. Takes all \$250M. Recovery = 250 / 300 = **83 percent**. Now the *secured* debt is the fulcrum.
2. Senior unsecured: \$0 left. Recovery = **0 percent** — they are now wiped out too.
3. Subordinated, common: zero.

And if instead the business is worth **\$1.0 billion** (a great outcome, more than all the debt):

1. Secured (\$300M): paid in full, 100 percent. \$700M left.
2. Senior unsecured (\$400M): paid in full, 100 percent. \$300M left.
3. Subordinated (\$200M): paid in full, 100 percent. \$100M left.
4. Common equity: receives the leftover \$100 million. Even the shareholders recover something.

Notice the fulcrum walked all the way down the stack — from the secured loan (at \$250M EV) to the senior bonds (at \$600M EV) to the common equity (at \$1B EV). **The single most important judgment in distressed investing is estimating where the enterprise value will land, because that estimate tells you which security is the fulcrum.** Get the EV right and the fulcrum falls out of the arithmetic.

## Finding the fulcrum and why it becomes the new equity

Here is the elegant part. Under the absolute-priority rule, the claims *above* the fulcrum are money-good — in a reorganization, they get reinstated or paid in cash, and they walk away. The claims *below* the fulcrum are out of the money — they get nothing, and their claims are cancelled. The fulcrum is the only layer that is impaired but not wiped: it is owed more than it will get, but it gets the *first dollars* of whatever value exists above the layers below it.

So when the company emerges from Chapter 11 with a fresh, deleveraged balance sheet, who gets the equity of the new company? Logically, it must go to the fulcrum holders. The senior layers are paid off in cash or new debt; the junior layers are gone; the only claimants left with skin in the residual value are the fulcrum creditors. **The fulcrum debt converts into the new equity.** A bondholder who bought senior notes at 50 cents walks out of bankruptcy owning shares in a company with no debt above them — and if the reorganized business does well, those shares can be worth far more than the bonds ever were.

![Before and after capital structure showing the senior fulcrum bonds converting into 100 percent of the new equity while junior layers are cancelled](/imgs/blogs/bankruptcy-and-restructuring-law-chapter-11-and-the-creditor-stack-5.png)

This is the "loan-to-own" or "fulcrum trade," and it is how distressed funds — Oaktree, Elliott, Apollo, Centerbridge, and dozens of others — actually make their returns. They are not lending to healthy companies; they are buying the fulcrum security of failing ones, at a discount, so that when the dust settles they own the equity of a cleaned-up business. Many of these funds sit inside the broader distressed-debt and special-situations world that we cover in the [hedge-fund strategy guide](/blog/trading/hedge-funds/choosing-your-strategy-the-operating-lens).

### The valuation fight is a fight over who is the fulcrum

Because the fulcrum is determined entirely by enterprise value, and enterprise value is an *estimate*, the single most contested number in any large Chapter 11 is the valuation. And here the incentives flip in a way that is almost comic once you see it. The creditors *above* the likely fulcrum want a **low** valuation — if the business is worth little, value runs out high in the stack, and they get the equity of the reorganized company cheap. The creditors *below* the likely fulcrum want a **high** valuation — if the business is worth a lot, value cascades further down and reaches their layer, giving them a recovery instead of a wipeout.

So in the courtroom you see senior creditors hiring bankers to argue the company is barely worth its secured debt, while junior creditors hire competing bankers to argue it is a hidden gem. Each side's expert produces a discounted-cash-flow model and a comparable-company analysis that, miraculously, supports its client's position. The judge picks a number somewhere in between, and that number decides who is the fulcrum and therefore who walks away owning the company. **A distressed investor is not just betting on the business; they are betting on where the judge will land in the valuation fight.** This is why the best distressed funds employ restructuring lawyers and former bankers, not just credit analysts — the edge is as much legal and procedural as it is fundamental.

#### Worked example: how a 1-turn valuation swing reassigns the fulcrum

Atlas generates \$100 million of annual cash flow (EBITDA). The senior creditors argue it should trade at a **5.0× multiple** (a distressed multiple for a weak manufacturer); the junior creditors argue **8.0×** (a healthy peer multiple).

- Senior view: 5.0 × \$100M = **\$500M** enterprise value. Pour it down: secured \$300M (full), senior unsecured takes the remaining \$200M of their \$400M claim → 50 percent recovery, fulcrum sits in the senior bonds, junior debt and equity get zero.
- Junior view: 8.0 × \$100M = **\$800M** enterprise value. Pour it down: secured \$300M (full), senior unsecured \$400M (full), subordinated takes the remaining \$100M of their \$200M claim → 50 percent recovery, the fulcrum has moved *down* into the subordinated notes.

A three-turn swing in the multiple — entirely a matter of expert opinion — moves the fulcrum from the senior bonds to the subordinated notes, transferring the equity of the reorganized company from one set of creditors to another. The intuition: in distress, "enterprise value" is not a fact you look up; it is a negotiated and litigated number, and the fulcrum trade is partly a wager on how that fight resolves.

#### Worked example: pricing a bond off an implied recovery

You are a credit analyst. Atlas Industrial's senior unsecured bonds (face value \$1,000 each, \$400 million outstanding) are trading at **\$0.45 on the dollar** — \$450 per bond. The market is pricing in a 45 percent recovery. Your own waterfall, using an enterprise value of \$600 million, says these bonds are the fulcrum and should recover **75 cents**. Should you buy?

- Market price: \$450 per \$1,000 bond.
- Your estimated recovery: 0.75 × \$1,000 = \$750 per bond.
- Upside if you are right: (750 − 450) / 450 = **+67 percent**.

But run the downside too. If the business is worth only \$450 million instead of \$600 million:

- Secured takes \$300M; \$150M left for the senior bonds' \$400M claim.
- Senior recovery = 150 / 400 = **37.5 percent** = \$375 per bond.
- Loss from \$450: (375 − 450) / 450 = **−17 percent**.

So at \$450 you have roughly +67 percent of upside against −17 percent of downside — a favorable, asymmetric payoff *if* your enterprise-value range is honest. The entire trade lives or dies on the EV estimate. The intuition: a distressed bond price is just the market's implied recovery, and the trade is a bet that your waterfall is more accurate than the consensus baked into the price.

#### Worked example: a DIP loan's super-priority return

A distressed fund provides Atlas with a **\$100 million DIP loan** to fund operations during its eight-month Chapter 11. The terms (typical for DIP facilities): an interest rate of 9 percent per year, a 2 percent upfront commitment fee, and super-priority status — first in line, ahead of every pre-petition claim.

- Upfront fee: 2 percent × \$100M = \$2 million, earned on day one.
- Interest over 8 months: 9 percent × (8/12) × \$100M = \$6 million.
- Total income: 2 + 6 = \$8 million on \$100 million over eight months.
- Annualized return: 8 / 100 × (12/8) = **12 percent**, and the principal sits at the very top of the stack, the safest position in the entire structure.

DIP lenders are paid an equity-like 12 percent for taking a near-risk-free position — that is the premium the law's super-priority rung commands. The intuition: bankruptcy creates the safest loan in finance, and the lender who can write that check during the crisis captures an outsized yield for almost no priority risk.

There is a strategic dimension here too. Pre-petition secured lenders often *want* to provide the DIP, even at thin economics, because controlling the DIP means controlling the case. A DIP loan comes with covenants, milestones, and a budget that the company must hit; the DIP lender effectively dictates the timetable and the shape of the eventual plan. Distressed funds sometimes provide a DIP not for the 12 percent yield but to seize the steering wheel of a restructuring whose fulcrum they already own. And a feature called a **roll-up** — where the DIP lender is allowed to convert some of its *old, pre-petition* debt into the new super-priority DIP — lets a creditor drag a chunk of its existing claim up to the very top of the stack as a condition of providing new money. The roll-up is one more example of the central theme: the law, applied through the right document, lets a creditor change its own place in the line.

## The Chapter 11 process from filing to emergence

To trade any of this you need to understand the clock. Chapter 11 is a sequence with predictable milestones, and each milestone is a potential repricing event.

![Chapter 11 process timeline from petition and automatic stay through DIP financing plan negotiation confirmation and emergence](/imgs/blogs/bankruptcy-and-restructuring-law-chapter-11-and-the-creditor-stack-4.png)

1. **The petition (Day 0).** The company files. The automatic stay snaps on instantly. Trading in the debt often spikes in volume as holders who cannot or will not own a bankrupt name dump their bonds — frequently the best moment to buy, because forced sellers depress the price below fundamental recovery.
2. **First-day motions and DIP approval (Week 1).** The court approves "first-day" motions to keep the business running — pay employees, honor critical vendors — and approves the DIP loan. The DIP terms reveal who the senior players are and how much liquidity the estate has.
3. **Claims and valuation fights (months).** Creditors file claims; the company and the creditors' committees argue over the enterprise value. This is the war that determines the fulcrum, because the fulcrum *is* a function of EV. Senior creditors argue for a low valuation (so value runs out higher and they take the equity); junior creditors argue for a high valuation (so the value reaches down to them). The valuation fight is a fight over who is the fulcrum.
4. **The plan and the vote (6–18 months).** A plan of reorganization is filed, classes vote, and the company seeks confirmation. If a class dissents, the company can pursue **cramdown** to confirm over its objection.
5. **Confirmation and emergence (exit).** The court confirms the plan, the old equity is cancelled, the fulcrum debt converts to new equity, and the company emerges deleveraged. The new shares begin trading; the fulcrum holders are now the owners.

The whole process can run anywhere from a 30-day "prepackaged" bankruptcy (where the plan is negotiated *before* filing) to a multi-year brawl. The longer and more contested it is, the more the legal mechanics — not the operating business — drive who ends up with what.

## Liability-management exercises and creditor-on-creditor violence

For most of bankruptcy's history, the creditor stack was assumed to be stable: a contract said you were senior, and senior you stayed. Over the past decade that assumption has been demolished by a wave of aggressive maneuvers known as **liability-management exercises (LMEs)** — out-of-court transactions that *re-order the stack* by exploiting loopholes in loan documents. Practitioners gave them a grimly accurate nickname: **creditor-on-creditor violence.**

The setup: when a company is distressed but not yet bankrupt, its credit documents often allow amendments approved by a *majority* of lenders (by dollar amount) to bind everyone. A clever sponsor (often a private-equity owner trying to save its equity) recruits a majority of lenders into a deal that benefits them at the direct expense of the minority. There are two main flavors.

### Uptiering (priming)

In an **uptiering** transaction, a majority of lenders agree to amend the credit agreement to create a *new, super-senior tranche* of debt. The majority lenders exchange their existing loans into this new top tranche — and in doing so, they **prime** (subordinate) the lenders who were not invited. Yesterday everyone held the same first-lien loan, pari passu (equal in priority). Today the majority is super-senior and the minority has been pushed down to second or third lien, their position stripped of value.

![Graph of an uptiering exchange where a majority of lenders form a super senior tranche that primes the non participating minority down to third lien](/imgs/blogs/bankruptcy-and-restructuring-law-chapter-11-and-the-creditor-stack-6.png)

The landmark case is **Serta Simmons (2020)**. A majority of Serta's first-lien lenders provided new money and uptiered themselves above the non-participating lenders, who sued. After years of litigation, in late 2024 a federal appeals court ruled that the maneuver was *not* permitted by the "open market purchase" provision Serta had relied on — a major (if late) win for the primed minority and a warning shot to the LME machine. But by then dozens of copycat deals had already been done.

### Drop-downs

In a **drop-down**, the company moves its most valuable collateral — a brand, a subsidiary, intellectual property — *out* of the reach of existing lenders by transferring it to a new, **unrestricted subsidiary** that the existing debt does not have a claim on. The company then raises new debt against that asset, leaving the original lenders' collateral hollowed out.

The original sin here is **J.Crew (2017)**, which transferred its brand IP to an unrestricted subsidiary and borrowed against it — coining the verb "to J.Crew" your lenders. **Serta** and the 2019–2020 **Neiman Marcus** "MyTheresa" transfer (which moved a fast-growing e-commerce unit beyond creditors' reach) became the other canonical examples. Together they taught the entire credit market that the loan document is a weapon, and that "first lien" means nothing if the docs let the company move the collateral or prime your claim.

#### Worked example: an uptiering LME repricing winners and losers

A company has \$1.0 billion of first-lien term loans, all equal in priority, trading at \$0.70 on the dollar (the market expects a 70 percent recovery). A majority group holding \$600 million of the loans cuts an uptiering deal:

- The \$600M majority exchanges into a new **super-senior** tranche and provides \$200M of new money. Their new position is first in line.
- The \$400M minority is pushed down to **third lien** — behind \$800M of debt that did not exist above them yesterday.

Suppose the company's recoverable value is \$700 million. After the deal:

- Super-senior (\$600M old + \$200M new = \$800M): takes the first \$700M. Recovery on the old \$600M portion is now near 100 percent — call it ~\$0.95+ once you account for the new-money priority. They gained.
- Third-lien minority (\$400M): \$0 left after the super-senior is paid. Recovery falls from an expected ~70 cents to **near zero**. They lost roughly **\$0.70 × \$400M = \$280 million** of value — transferred, in effect, to the majority.

The arithmetic is brutal and zero-sum: the same \$700M of value now flows to a different set of creditors because the *documents* let a majority re-rank the stack. The intuition: in the LME era, your recovery depends not only on where you sit in the stack but on whether the loan documents let someone move you — seniority is now contractual, contestable, and worth fighting over before the bankruptcy ever begins.

## Priming, intercreditor agreements, and the fights that follow

LMEs are possible because the relationships *between* creditors are themselves governed by contracts — chiefly the **intercreditor agreement (ICA)**, the document that spells out who is senior to whom, who controls the collateral, and who can do what in a default. When two classes of lenders (say first-lien and second-lien) lend to the same company, the ICA is the treaty between them.

**Priming** is the act of inserting new debt *ahead* of existing debt — exactly what an uptiering does, and what DIP financing does with court blessing. The question of whether priming is allowed without unanimous consent is the central legal battlefield of modern credit. Older loan documents were drafted loosely, assuming good faith; the lawyers who drafted them never imagined a majority would turn on a minority. Newer documents include "**J.Crew blockers**," "**Serta protections**," and "**anti-uptiering**" clauses precisely to close these loopholes — but every new clause invites a new workaround. It is a permanent arms race between document drafters and the funds that hunt for openings.

For an investor, the lesson is that **reading the credit agreement is now part of pricing the bond.** Two bonds with identical coupons and identical seniority on paper can have wildly different real-world protection depending on whether the docs permit drop-downs and uptiers. The covenant package — once an afterthought — is now a first-order driver of recovery. This connects directly to how the market prices default risk in the first place, the subject of the fixed-income piece on [credit spreads and the probability of default](/blog/trading/fixed-income/credit-spreads-pricing-the-probability-of-default).

### Why this changed, and why it keeps escalating

LMEs did not emerge by accident. Two structural shifts created the opening. First, the long era of low interest rates from 2010 to 2021 flooded the market with **covenant-lite** loans — loans with few protective covenants, written in a borrower's market where lenders competed to deploy capital and surrendered protections to win deals. Those loose documents are the loopholes the LME machine now exploits. Second, the rise of large, sophisticated private-credit and distressed funds created players with the legal firepower and the appetite to engineer these transactions — and, crucially, the willingness to do to *other creditors* what was once unthinkable in a clubby market that ran on relationships.

The escalation is genuinely an arms race. Each new maneuver names itself after the company that pioneered it, and each spawns a defensive clause that the next maneuver routes around. After J.Crew came "J.Crew blockers." After Serta came "Serta protections" and "**cooperation agreements**" — pacts where a majority of lenders pre-agree to act as a bloc so that *no* subset can be picked off and primed. The newest variants, sometimes called "**double-dips**" and "**pari plus**" structures, find ways to give favored creditors *two* claims on the same collateral or a structurally senior claim through an intercompany loan. The specifics matter less than the meta-lesson: **in modern credit, seniority is not a fixed attribute of a bond — it is a contested, contractual position that can be attacked and defended, and the value of your claim depends on which side of the next maneuver you are on.** A first-lien loan in a company with weak docs and a desperate sponsor is not a senior claim; it is a senior claim wearing a target.

## Sovereign debt restructuring: the no-court analog

Everything so far assumes a bankruptcy *court* — a judge with the power to impose the automatic stay, approve a DIP loan, and cram down a plan. But what happens when the borrower is a *country*? There is no bankruptcy court for nations. No judge can seize Argentina's assets or freeze its creditors. So how does a sovereign restructure its debt?

The answer reveals what bankruptcy law actually *does*: it solves a coordination problem. When a company defaults, hundreds of creditors would each prefer to be paid in full while everyone else takes a haircut. If they all hold out, nothing gets restructured and the value evaporates. Bankruptcy law forces a collective solution. Sovereigns, lacking a court, have to manufacture the same collective binding through *contract*.

![Before and after of a sovereign restructuring contrasting the holdout problem without collective action clauses against a binding supermajority vote with CACs](/imgs/blogs/bankruptcy-and-restructuring-law-chapter-11-and-the-creditor-stack-7.png)

### Collective action clauses and the holdout problem

A **collective action clause (CAC)** is a provision written into sovereign bonds that says: if a supermajority of bondholders (say 75 percent) agrees to a restructuring, the new terms bind *all* holders, including those who voted no. It is, in effect, a contractual cramdown — the country's substitute for what a bankruptcy court would do automatically.

Without CACs, sovereign restructurings are plagued by **holdouts**: creditors who refuse the deal, demand payment in full, and sue. The defining saga is **Argentina**. After its \$100 billion default in 2001 (the largest sovereign default in history at the time), Argentina offered creditors an exchange worth roughly 30 cents on the dollar. About 93 percent accepted over two rounds. But a group of holdout funds — led by Elliott Management's NML Capital — refused, bought the defaulted bonds cheap, and litigated for over a decade.

Their weapon was the **pari passu** ("equal footing") clause in the old bonds. A U.S. court ruled in 2012 that Argentina could not pay the exchange bondholders *anything* unless it also paid the holdouts in full — effectively blocking the entire restructuring until Argentina settled. In 2016, Argentina paid the holdouts roughly \$4.65 billion to end the standoff. The holdouts who bought defaulted debt for pennies earned a return estimated at several times their investment. The case terrified the sovereign-debt market and accelerated the global adoption of stronger, "single-limb" CACs that make holdout strategies far harder to execute.

The deep point: sovereign debt restructuring is bankruptcy law's mechanics — binding the minority, halting the scramble, allocating recovery — reconstructed entirely out of contract terms because no court exists to supply them. The CAC is the automatic stay and the cramdown, fused into a bond clause. For more on how country risk feeds into bond pricing, see the fixed-income coverage of [emerging-market and sovereign debt](/blog/trading/fixed-income/emerging-market-and-sovereign-debt-yield-with-country-risk).

#### Worked example: the holdout math under a CAC

A country has \$10 billion of bonds outstanding and offers an exchange worth **40 cents on the dollar** (a \$6 billion haircut). The bonds carry a CAC with a **75 percent** activation threshold. A holdout fund buys \$2 billion face value of the bonds in the secondary market at the distressed price of \$0.30, hoping to block the deal and litigate for par.

- The fund holds \$2B of \$10B = **20 percent** of the issue.
- To block the CAC, holdouts need to control more than 25 percent (so the 75 percent supermajority cannot be reached). At 20 percent, the fund alone *cannot* block — it needs allies controlling another 5-plus percent.
- If 80 percent of holders accept and the CAC binds the rest, the fund's \$2B is crammed into the exchange and recovers 40 cents = **\$800M** on its \$600M cost (\$0.30 × \$2B). That is a +33 percent gain — a win, but a modest one.
- Compare the *pre-CAC* world: if the fund could hold out and litigate to full par, \$2B at par on a \$600M cost would be a **+233 percent** gain. That is the prize the CAC destroys.

The arithmetic shows exactly why CACs were adopted and why holdout funds fight their spread: a high activation threshold and a large enough cooperating majority turn a potential triple into a modest exchange gain. The intuition: a CAC converts the holdout's litigation lottery into an ordinary, capped recovery — it is the contractual reason a country with well-drafted bonds restructures in months while Argentina, with old un-CAC'd bonds, fought for fifteen years.

## Common misconceptions

### "Bankruptcy means zero for everyone"

The single most common error. Bankruptcy is not a wipeout; it is a *redistribution* according to priority. In our Atlas example at \$600 million enterprise value, the secured lenders recovered **100 percent** and the senior bonds recovered **75 percent** — only the subordinated debt and common equity went to zero. Historically, across large U.S. corporate defaults, first-lien debt has recovered on the order of **60–70 cents on the dollar** and senior unsecured around **40 cents**. The Lehman senior bonds that opened this post paid roughly **21 cents** — a brutal loss, but a long way from zero, and a fortune for the funds that bought them at \$0.12. "Bankrupt" is a statement about the equity, not about every claim in the stack.

### "Equity always gets wiped, so common stock is worthless once a company files"

Usually true — but not always. When the enterprise value exceeds the total debt (a "solvent" reorganization, or a business that recovers faster than expected), the old equity can receive a recovery, often a "**stub**" — a small slice of the new equity or warrants. In our \$1 billion Atlas scenario, the common equity recovered \$100 million even after all \$900 million of debt was paid. Real cases happen: Hertz filed for bankruptcy in 2020 with its stock seemingly doomed, but a surge in used-car values during the 2021 recovery pushed the enterprise value above the debt, and Hertz shareholders received cash and equity worth several dollars per share — an almost unheard-of outcome that minted profits for the retail traders who bought the "worthless" stock. The lesson is not that equity usually survives — it usually does not — but that "always zero" is a recovery assumption, and recovery assumptions can be wrong when the value comes in high.

### "Secured means safe"

Secured debt is *safer*, not *safe*. Its protection is only as good as the **collateral coverage** — the value of the pledged assets relative to the loan. A first-lien loan of \$300 million secured by assets worth \$200 million is *under-secured*: it recovers the \$200 million from its collateral and then stands as an *unsecured* creditor for the remaining \$100 million, fighting for scraps with everyone else. In our second Atlas scenario (\$250M enterprise value), the "safe" secured lenders recovered only **83 cents**. And as the LME era proved, even a contractually first-lien position can be *primed* or have its collateral *dropped down* out of reach — Serta's non-participating first-lien lenders watched their "secured" claim get pushed to third lien overnight. Secured is a starting position, not a guarantee; the recovery depends on collateral value and on whether the documents let someone re-rank you.

## How it shows up in real markets

### A distressed bond trading at an implied recovery

When a company's bonds fall from \$0.95 to \$0.40, the market is not saying "this company is bad." It is making a precise statement: *the expected recovery on this claim is about 40 cents on the dollar.* A bond trading at \$0.40 with a \$1,000 face value is a \$400 lottery ticket on a legal waterfall. Distressed desks quote bonds in **price** (cents on the dollar) rather than **yield**, because once a company is near default, yield becomes meaningless — there is no reliable coupon stream, only a recovery. The entire distressed market is a market in *implied recovery rates*. A bond at \$0.40 versus \$0.55 is the market disagreeing about where the fulcrum sits.

### The fulcrum trade in action

When **Hertz** filed in 2020, distressed funds raced to identify the fulcrum. Early on, with car values depressed, the unsecured bonds looked like the fulcrum — likely to convert to equity at a steep discount. Funds bought the bonds in the \$0.40s. As used-car prices exploded through 2021, the enterprise value soared, the fulcrum migrated *down* the stack toward the equity, the unsecured bonds rallied toward par, and the old equity — supposedly worthless — got a recovery. The funds that bought the right layer at the right time made multiples. The fulcrum trade is precisely this: identify the security that will absorb the marginal value, buy it before the market agrees, and ride it as it converts to equity.

### An LME repricing winners and losers in real time

When news breaks that a company is pursuing an uptiering or drop-down, the market reprices the stack *instantly*. The participating lenders' loans jump (they just moved up the ladder); the non-participating lenders' loans crater (they just got primed). When Serta's uptiering was announced, the non-participating first-lien loans fell sharply as the market recognized they had been subordinated; the participating tranche traded up. A modern credit trader watches not just earnings and leverage but the *threat* of an LME — a company with loose loan documents and a desperate private-equity owner is a company where your seniority can be revoked by a phone call between a sponsor and a majority of your fellow lenders.

### The Lehman waterfall, settled

Return to the post's opening trade. Lehman's holding-company estate took years to resolve because the claims were tangled across hundreds of subsidiaries and jurisdictions — exactly the kind of contested, fee-heavy case where administrative claims and litigation eat into recoveries. When the dust settled, senior unsecured creditors of the holding company recovered roughly **21 cents on the dollar**. To a buyer at par, that is a catastrophic 79-point loss. But to a fund that bought the senior bonds at **\$0.12** during the panic of late 2008, the math was different: a recovery of \$0.21 on a \$0.12 cost basis is a gain of (21 − 12) / 12 ≈ **75 percent**, earned by reading the waterfall when the market was reading the obituary. The same estate paid different subsidiaries' creditors anywhere from near-zero to nearly full recovery, depending on where in the corporate structure their claim sat — a vivid reminder that "Lehman bonds" was never one price; it was dozens of prices, one per place in the stack.

### A distressed exchange that avoids the courthouse entirely

Not every restructuring reaches a courtroom. A **distressed exchange** is an out-of-court deal where bondholders voluntarily swap old bonds for new ones with a lower face value, a later maturity, or a more senior position — accepting a haircut to avoid the cost and uncertainty of a formal bankruptcy. The rating agencies treat a coercive distressed exchange as a default, but it can leave creditors better off than a Chapter 11 would. The fulcrum logic still governs: the exchange terms reflect where the value runs out, and the most senior impaired class extracts the best terms. For a trader, a looming distressed exchange is a catalyst to price — the announcement reprices every layer of the stack as the market learns who gets the new senior paper and who gets crammed into a longer, junior instrument.

## How to trade it: the distressed and credit playbook

This is the payoff. Here is how a distressed analyst actually turns bankruptcy law into a position.

**1. Build the waterfall.** Lay out the full capital structure — every tranche, its face amount, its seniority, its collateral. Estimate the enterprise value with a *range* (bear, base, bull), because EV is the most uncertain and most important input. Pour each EV scenario down the stack and record the recovery for every layer. The output is a recovery matrix: security × EV scenario → cents on the dollar.

**2. Locate the fulcrum.** In your *base-case* EV, find the layer where value runs out — that is your fulcrum. Confirm it is the layer that converts to equity under APR. Then check how the fulcrum *moves* across your EV range: a fulcrum that stays put across scenarios is a high-conviction trade; a fulcrum that jumps layers with a small change in EV is a coin flip you should size down.

**3. Price the trade off implied recovery.** Compare your estimated recovery to the market price. If the fulcrum bond trades at \$0.45 and your base case says \$0.70 with a \$0.37 floor, you have asymmetric upside — roughly +55 percent against −18 percent. Demand a margin of safety: the price should sit *below* your bear-case recovery, so that even if you are wrong on EV, you do not lose money. The whole edge is buying the fulcrum cheaper than its worst-case recovery.

**4. Read the documents before you read the balance sheet.** In the LME era, the credit agreement is part of the price. Check for J.Crew blockers, anti-uptiering language, unrestricted-subsidiary capacity, and "open market purchase" loopholes. A first-lien claim with weak docs is not a first-lien claim — it is a first-lien claim *that can be primed*. Discount your recovery for document risk.

**5. Watch the catalysts.** The price moves at the legal milestones: the filing (forced selling = buying opportunity), the DIP approval (reveals liquidity and the senior players), the valuation fight (determines the fulcrum), and confirmation (the conversion to equity). Position ahead of the milestone you have an edge on, not after the market has digested it.

**6. Know what invalidates the thesis.** The trade is wrong if: (a) the enterprise value comes in below your bear case (the fulcrum moves above you and you are wiped); (b) an LME re-ranks the stack and primes your "senior" claim; (c) the process drags on so long that legal and advisory fees — which are paid ahead of everyone, as administrative claims — eat the recovery; or (d) a fraudulent-conveyance or other litigation claim reshuffles the priorities. If any of these fires, your waterfall is no longer the waterfall, and you exit.

The discipline is the same one that runs through every post in this series: a legal rule sets the price, you read the rule before the crowd does, you size the repricing, and you know exactly what would prove you wrong. In distress, the rule is bankruptcy law, the price is cents on the dollar, and the edge is finding the fulcrum before the market admits where the value runs out. For the broader lens on how legal rules become market prices, start with the series spine on [how law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain), and for how this same machinery shows up as a systemic risk during a banking failure, see [deposit insurance, the lender of last resort, and the anatomy of a bank run](/blog/trading/law-and-geopolitics/deposit-insurance-lender-of-last-resort-and-the-anatomy-of-a-bank-run).

## Further reading & cross-links

- [How law moves markets: the transmission chain](/blog/trading/law-and-geopolitics/how-law-moves-markets-the-transmission-chain) — the series spine: how a rule becomes a price.
- [Deposit insurance, the lender of last resort, and the anatomy of a bank run](/blog/trading/law-and-geopolitics/deposit-insurance-lender-of-last-resort-and-the-anatomy-of-a-bank-run) — bankruptcy's systemic cousin: when the failing entity is a bank.
- [Regulatory risk as an asset-pricing factor](/blog/trading/law-and-geopolitics/regulatory-risk-as-an-asset-pricing-factor) — how legal and regulatory uncertainty gets priced into every asset.
- [Seniority, recovery, and the capital structure](/blog/trading/fixed-income/seniority-recovery-and-the-capital-structure) — the fixed-income foundation for the creditor stack.
- [Credit spreads: pricing the probability of default](/blog/trading/fixed-income/credit-spreads-pricing-the-probability-of-default) — how the market prices default and recovery before bankruptcy ever happens.
- [Emerging-market and sovereign debt: yield with country risk](/blog/trading/fixed-income/emerging-market-and-sovereign-debt-yield-with-country-risk) — the sovereign analog, where CACs replace the bankruptcy court.
- [Choosing your strategy: the operating lens](/blog/trading/hedge-funds/choosing-your-strategy-the-operating-lens) — where distressed and special-situations funds fit in the hedge-fund universe.

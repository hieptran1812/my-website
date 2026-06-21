---
title: "Debt vs Equity: The Two Ways to Raise Capital"
date: "2026-06-21"
publishDate: "2026-06-21"
description: "Every dollar a company raises is either borrowed or sold as ownership. Here is the fundamental fork — what each claim is, who gets paid first, what each costs, and why firms blend them."
tags: ["capital-markets", "debt", "equity", "capital-structure", "wacc", "seniority", "convertibles", "corporate-finance", "tax-shield", "bankruptcy"]
category: "trading"
subcategory: "Capital Markets"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — There are exactly two ways to raise long-term capital: sell a piece of ownership (equity) or borrow and promise to repay (debt). Everything else is a blend of those two.
>
> - **Equity** is a residual claim: unlimited upside, downside to zero, no obligation to pay, but you give up ownership, votes, and a share of every future dollar.
> - **Debt** is a fixed senior claim: it must be repaid with interest and carries covenants, but it is cheaper, tax-favoured, and dilutes nobody.
> - **Seniority is the whole game.** In good times interest is paid before dividends; in bankruptcy the order is secured debt → senior unsecured → subordinated → preferred → common — and common usually gets zero.
> - The one number to remember: in 2023 the US issued roughly **\$23 trillion of Treasury debt and \$1.4 trillion of corporate debt**, dwarfing the **~\$30 billion** raised in traditional IPOs. The world runs far more on borrowing than on selling shares.

On the morning of December 11, 2020, Airbnb's bankers set the IPO price at \$68 a share. By the time the stock opened for trading it was at \$146 — it had more than doubled before a single public investor could buy at the offering price. Airbnb raised about \$3.5 billion that day by selling new shares: a permanent slice of the company, handed to whoever was willing to pay. The founders kept control through a special class of supervoting stock, but they had just given strangers a claim on every future dollar Airbnb would ever earn.

That same year, Apple — a company with more cash than most governments — quietly *borrowed* \$5.5 billion by selling bonds at coupons under 2.5%. Apple did not need the money. It borrowed because debt was almost free, the interest was tax-deductible, and issuing bonds handed no ownership and no votes to anyone. The bondholders got a fixed promise: a coupon twice a year and their principal back on a fixed date. No upside if Apple's market value tripled. No vote in the boardroom. Just a senior IOU.

Two companies, two completely different ways of answering the same question: *where does the money come from?* This post is about that fork in the road — the single most consequential decision in corporate finance, and the one that quietly shapes everything else in the capital markets machine.

![The capital stack seniority ladder from secured debt at the top to common equity at the bottom](/imgs/blogs/debt-vs-equity-the-two-ways-to-raise-capital-1.png)

This series has a spine: a capital market is a **machine that turns savings into long-term investment**, running on a primary market that *creates* securities and a secondary market that *trades* them. When a company raises money in the primary market, it is forced to choose what kind of security to create — and there are only two base ingredients. Debt and equity are the two atoms from which every financial instrument in existence is built. Get this fork right and the rest of the machine — pricing, trading, bankruptcy, regulation — clicks into place.

## Foundations: what a claim actually is

Before any jargon, start with a kitchen-table version. Suppose your friend Mai wants to open a coffee shop and needs \$100,000 she does not have. She can get it from you in exactly two ways.

**Way one — you lend it.** You hand Mai \$100,000 and she signs a piece of paper: "I will pay you 6% a year and return your \$100,000 in five years." That paper is a *debt claim*. You are now a **creditor**. You do not own the coffee shop. You do not care whether it sells ten cups a day or ten thousand — as long as Mai pays your interest and returns your principal, you are made whole and not a dollar richer. If she gets rich, you still get exactly 6%. If she struggles, you can take her to court and seize the espresso machine before she keeps anything for herself. Your claim is *fixed* and it is *senior*.

**Way two — you buy in.** You hand Mai \$100,000 and she signs a different paper: "You now own 40% of this coffee shop." That paper is an *equity claim*. You are now a **shareholder**, a part-owner. There is no interest. There is no repayment date. If the shop makes \$50,000 in profit this year and decides to distribute it, you get 40% of it — \$20,000. If it makes nothing, you get nothing and cannot sue. But if Mai's shop becomes a chain worth \$10 million, your 40% is worth \$4 million. Your claim is *residual* — you get what is left *after* everyone else, including the lender, has been paid — and in exchange for standing last in line, you get all the upside.

That is the entire idea, and everything technical in this post is a consequence of it. A **security** (covered in [what a security actually is](/blog/trading/capital-markets/what-a-security-actually-is-claims-you-can-sell)) is just a standardised, transferable version of one of those two pieces of paper, scaled up so it can be sold to thousands of investors and traded tomorrow. When a corporation does this at scale, the lending paper becomes a **bond** or a **loan**, and the ownership paper becomes a **share** of **stock**.

Two definitions to lock in before we go deeper:

- **Residual claim (equity):** a right to whatever is left over after all fixed claims are satisfied. Open-ended on the upside, worthless if there is nothing left.
- **Fixed claim (debt):** a right to a pre-agreed stream of cash (interest) plus the return of a principal amount, ranking *ahead* of the residual claim.

The genius of the capital market is that it does not force a company to pick one source. It lets a firm slice its funding into a *stack* of claims — some fixed, some residual, some in between — and sell each slice to whichever investor wants exactly that risk. That stack is where we are headed, but first let's go deep on each atom.

Why does the world settle on *exactly two* base claims rather than three or seventeen? Because there are only two fundamental questions a financier can answer differently. The first is: *do you get a fixed amount or a share of whatever's left?* That single split — fixed versus residual — is the deepest division in finance, and it is the line between debt and equity. The second is: *if there isn't enough money, who eats the loss first?* That question produces the *ordering* of claims — seniority — which slots the various flavours of debt and equity into a ladder. Every financial instrument ever invented is some answer to those two questions. A plain bond says "fixed amount, near the top of the ladder." A common share says "whatever's left, at the very bottom." A convertible says "fixed amount for now, with the right to switch to whatever's left later." Hold those two questions in mind and no instrument will ever surprise you — it is always just a particular answer to *fixed-or-residual* and *who-loses-first*.

## Equity: selling a piece of the company

Equity is ownership made tradeable. When you buy a share, you are buying a fractional, transferable slice of a company's residual value. Four features define what that slice is, and each one is a double-edged sword.

**1. It is a residual claim.** Shareholders are last in line. Every period, the company pays its suppliers, its employees, its lenders, and the tax authority *first*. Whatever is left — the residual — belongs to the equity. In a great year that residual is enormous. In a bad year it is zero, and shareholders simply absorb the disappointment. There is no court they can run to, because they were never *promised* anything.

**2. The upside is unlimited; the downside stops at zero.** This asymmetry is the heart of equity's appeal. A bond can, at best, pay you back your principal plus the coupon you were promised — that is the ceiling. A share has no ceiling. If you bought \$1,000 of a company at its IPO and it became the next Amazon, your \$1,000 could become \$500,000. But the floor is firm: a share can fall to zero and *no further*. Thanks to **limited liability** — a legal invention as important as the steam engine — a shareholder who put in \$1,000 can lose at most that \$1,000, even if the company goes bankrupt owing billions. You are never on the hook beyond what you invested.

**3. There is no obligation to pay.** A company is never legally required to pay a dividend. Apple paid no dividend for 17 years and nobody could force it to. This is equity's great operational advantage to the *issuer*: the money raised never has to be returned, and there is no fixed bill to meet each quarter. In a recession, a firm financed entirely by equity simply cuts or skips its dividend and survives. A firm drowning in debt that misses an interest payment is in *default* — a far more dangerous place.

**4. It carries control and suffers dilution.** Shares usually vote. One share, one vote, is the default; that vote elects the board, who hire and fire the CEO. So selling equity literally sells slices of control. And every time a company issues *new* shares to raise more money, the existing slices shrink — this is **dilution**. If a company has 1 million shares and issues 250,000 new ones, the old owners now hold 1,000,000 / 1,250,000 = 80% of a company they used to own 100% of.

![Comparison matrix of equity versus debt across claim, obligation to pay, upside and control](/imgs/blogs/debt-vs-equity-the-two-ways-to-raise-capital-3.png)

#### Worked example: the dilution math when you issue new shares

A company has **1,000,000 shares** outstanding and the market values it at **\$50,000,000** — so each share is worth **\$50**. It wants to raise **\$10,000,000** in fresh equity to build a factory.

At \$50 a share, it must issue **\$10,000,000 / \$50 = 200,000 new shares**. Now there are **1,200,000 shares**.

What happened to an existing owner who held 100,000 shares (10% of the company)?

- Before: 100,000 / 1,000,000 = **10.0% ownership**.
- After: 100,000 / 1,200,000 = **8.33% ownership**.

Their stake fell by a sixth. But the company's value rose to \$50M + \$10M cash = \$60M, and 8.33% of \$60M = **\$5,000,000** — exactly the same \$5M their 10% of \$50M was worth before. *Fair* dilution does not destroy value; it trades a bigger slice of a smaller pie for a smaller slice of a bigger one. The danger is *unfair* dilution: if the new shares are sold for less than \$50 (say in a desperate down-round), existing owners lose real value. **The intuition: dilution is only theft when the new shares are sold too cheap.**

This is why equity issuance dominates in two situations: when a company is young and cannot promise the steady cash a lender demands, and when it is so richly valued that selling a small slice raises an enormous sum. The chart below shows the *stock* of all that ownership — the total US equity market — compounding (and crashing) over a decade.

![US equity market capitalization at year end from 2014 to 2024 in trillions of dollars](/imgs/blogs/debt-vs-equity-the-two-ways-to-raise-capital-6.png)

Notice the shape: equity value *compounds*. From \$26 trillion in 2014 to \$58 trillion in 2024, with a brutal 2022 drawdown when the market value of all US public companies fell roughly \$8 trillion in a year. That volatility *is* the residual claim showing its teeth — equity holders bear the full swing of the economy's fortunes, up and down. A bond's value would never gyrate like that, because a bond's payoff is capped.

### Not all equity is the same

Even within "equity" there is structure, and it matters for control. Companies can issue *different classes* of shares with different voting rights. Dual-class structures — common among founder-led tech firms — give insiders shares that carry, say, ten votes each while public investors get one vote each. This is how Airbnb's founders, and Mark Zuckerberg at Meta, kept voting control of companies in which they own a minority of the *economic* value. The public buys the cash-flow rights; the founders keep the steering wheel. It is a reminder that "ownership" splits into two things — a claim on the *money* and a claim on the *decisions* — and equity can hand out each separately.

Equity also gets raised at different stages through different channels, and each is a primary-market transaction that creates new shares:

- **Venture / private equity rounds:** a private company sells new shares to professional investors. No public market yet; the price is negotiated deal by deal.
- **The IPO (initial public offering):** the company sells shares to the public for the first time and lists on an exchange, converting private ownership into freely tradeable securities. This is the headline event, but it is just *one* equity raise.
- **Follow-on / seasoned offerings:** an already-public company issues *more* new shares to raise additional capital — and this is where dilution and signalling bite hardest.

In every case the mechanic is identical to the kitchen-table version: the company creates new ownership paper and sells it. Who structures and sells that paper — the underwriters, the syndicate, the bookbuilding process — is the work of the investment bank, covered in [inside an investment bank](/blog/trading/finance/inside-an-investment-bank-how-they-make-money); our point here is simply that *all* of it is equity, all of it is a residual ownership claim, and all of it dilutes whoever already owns the company.

## Debt: borrowing instead of selling

Debt is the opposite bargain in every respect. When a company borrows — whether from a bank as a *loan* or from the market as a *bond* — it creates a fixed senior claim and sells it to a lender. Four features, again, each a mirror image of equity.

**1. It is a fixed senior claim.** The lender is promised specific cash on specific dates: periodic *interest* (the coupon) and the return of *principal* (the face value) at *maturity*. That promise ranks *ahead* of equity. Interest must be paid before any dividend can be; principal must be repaid before any shareholder sees a liquidation cent. Seniority is the lender's protection against the open-ended risk they declined to take.

**2. It must be repaid — missing a payment is default.** This is debt's defining discipline and its defining danger. A company cannot simply decide not to pay interest the way it can skip a dividend. Miss a coupon and you are in **default**: the lenders can accelerate the loan (demand all of it back at once), seize collateral, or force the company into bankruptcy. Debt turns a soft obligation into a hard one, and that hardness is exactly why lenders accept a lower return.

**3. The upside is capped.** A bondholder's best case is being paid in full, on time. If the company you lent to becomes the next Apple, you still get your 5% coupon and your principal back — not a penny of the equity bonanza. This cap is the price of seniority and safety. You traded the lottery ticket for the IOU.

**4. It carries covenants, not votes.** Lenders do not get a board seat or a vote. Instead they protect themselves with **covenants** — contractual promises baked into the loan: "you will not let your debt exceed 3× earnings," "you will not pay a dividend if your cash falls below \$X," "you will maintain this much collateral." Covenants are a *negative* form of control: lenders cannot direct the company, but they can stop it from doing things that endanger their claim, and breaching a covenant can itself trigger default.

The single most important fact about debt is its *scale*. People intuitively think of "raising capital" as IPOs and stock — the glamorous part. But the debt market is far larger. The chart below shows US debt issuance in a single year.

![US debt issuance by type in 2023 on a log scale, Treasury dominating corporate municipal and ABS](/imgs/blogs/debt-vs-equity-the-two-ways-to-raise-capital-2.png)

Note the **log scale** — each gridline is 10× the one below. US Treasury issuance alone was about **\$23.2 trillion** in 2023 (most of it short-term bills constantly rolled over), corporate bonds about **\$1.4 trillion**, mortgage-backed securities \$1.5 trillion. For comparison, traditional US IPOs raised roughly **\$19 billion** in 2023 — about *one seventy-fourth* of corporate debt issuance, and a rounding error against Treasuries. The world finances itself overwhelmingly by borrowing. When this series says capital markets turn savings into investment, *most* of that flow runs through debt, not equity.

Issuance is a *flow* — how much was raised in one year. The *stock* — how much is outstanding in total — tells the same story. Globally, the bond market and the equity market are roughly the same enormous size, with bonds slightly ahead.

![Global stock of capital comparing the bond market and equity market in trillions of dollars](/imgs/blogs/debt-vs-equity-the-two-ways-to-raise-capital-4.png)

The global bond market is around **\$140 trillion** outstanding against roughly **\$115 trillion** of global equity market cap; in the US the two are closer to even at about **\$55 trillion** each. Two lessons hide in those bars. First, debt is *not* the junior partner in the capital markets — by the size of the claims outstanding it is the senior partner, the larger pool of capital. Second, the reason bonds can be issued in such volume is that they are *safer and more standardised* than equity: a pension fund or insurer that must match predictable liabilities (pensions to pay, claims to settle) wants exactly the fixed, senior, predictable cash a bond delivers, and there is a vast amount of such "patient, safety-seeking" money in the world. Equity, with its open-ended risk, attracts a different and ultimately smaller pool of capital. The two markets are roughly matched in size precisely because savers come in two flavours — those who want safety and a fixed return, and those who want ownership and upside — and the capital market builds an instrument for each.

#### Worked example: \$100M raised as debt vs equity

Mai's coffee chain, now a real company called Brewline, needs **\$100,000,000** to expand. Compare the two financings, holding everything else equal. Suppose the business will earn **\$15,000,000** of operating profit (EBIT) next year.

**Option A — raise \$100M as debt** at a 6% coupon:

- Annual interest = \$100,000,000 × 6% = **\$6,000,000**, owed no matter what.
- Pre-tax profit after interest = \$15M − \$6M = **\$9,000,000**.
- The founders still own **100%** of the company. No dilution, no new votes.
- But Brewline now *must* find \$6M of cash a year, and repay \$100M at maturity. A bad year that drops EBIT to \$5M means it cannot fully cover interest from earnings — danger.

**Option B — raise \$100M as equity** by selling shares. Say the company is worth \$300M pre-money, so it sells \$100M / (\$300M + \$100M) = **25% of the company**:

- No interest. Pre-tax profit stays **\$15,000,000**.
- But the founders now own **75%**, and 25% of every future dollar — including that \$15M and all the upside of a chain worth billions someday — belongs to new shareholders.
- There is no payment to miss, so a bad year is survivable: skip the dividend and carry on.

**The intuition: debt keeps the whole pie but adds a mandatory bill that can sink you; equity removes the bill but permanently gives away a slice of every future pie.** Which is "better" depends entirely on how stable Brewline's cash flows are and how cheap its debt is — the trade-off we build toward below.

### The spectrum of debt

Just as equity has classes, debt comes in many shapes, and the differences all reduce to *how much risk the lender takes* and *how much control they keep*. The main dials:

- **Loan vs bond.** A *loan* is private debt, usually from a bank, often with tight covenants and a relationship lender who can renegotiate. A *bond* is debt sold to many investors in the public market, more standardised and tradeable, with looser per-investor control. Loans dominate for smaller and private borrowers; bonds dominate for large, well-known issuers who can tap thousands of buyers at once.
- **Secured vs unsecured.** Secured debt is backed by specific collateral the lender can seize on default; unsecured debt has only a general claim. Secured ranks higher in the stack and therefore carries a lower rate.
- **Senior vs subordinated.** As the stack shows, subordinated debt agrees to stand behind senior debt and pays a higher coupon for the privilege of being riskier.
- **Maturity.** Debt can be overnight or thirty years. Short-term borrowing is cheaper but must be constantly refinanced — a deadly trap if markets freeze, as we will see in the Vietnam case. The split between short-term *money-market* debt and long-term *capital-market* debt is itself a defining divide, covered in [money market vs capital market](/blog/trading/capital-markets/money-market-vs-capital-market-where-short-meets-long).
- **Fixed vs floating rate.** A fixed coupon locks the cost for the life of the bond; a floating rate moves with market interest rates, shifting rate risk from issuer to lender or vice versa.

Every one of these is a knob on the same instrument — a fixed senior claim — letting the issuer and lender agree on exactly how much risk changes hands. But all of them sit *above* equity in the one place it matters most: the line at which a company runs out of money.

## The capital stack: who gets paid first

Real companies do not pick one financing and stop. They build a **capital stack** — a layered ladder of claims, each with a different seniority, sold to a different kind of investor. The stack is the single most useful mental model in all of corporate finance, because it answers the two questions that govern who actually makes money: *who gets paid first in good times,* and *who gets paid first when the company dies.*

From the top (most senior, safest, paid first) to the bottom (most junior, riskiest, paid last):

1. **Secured debt** — loans and bonds backed by specific collateral (a building, equipment, receivables). First claim. If the company fails, secured lenders can seize their collateral before anyone else touches it.
2. **Senior unsecured debt** — ordinary corporate bonds and loans with no specific collateral, but a general first claim on the company's assets ahead of everything below.
3. **Subordinated (junior) debt** — debt that has contractually agreed to stand *behind* senior debt. Still debt (fixed claim, must be paid), but it only gets paid after senior debt is whole. Higher coupon to compensate.
4. **Preferred stock** — a hybrid. It pays a *fixed* dividend like debt, ranks ahead of common stock, but sits *below* all debt. Preferred holders usually cannot vote and have no upside beyond their fixed dividend.
5. **Common equity** — the residual, last in line, the true owners. They eat all the losses first and all the gains last.

The ladder runs in *both* directions. In good times, cash flows *down* the stack: the company pays interest on secured debt first, then senior, then subordinated, then preferred dividends, and only what remains can be paid to common shareholders or reinvested. In bad times, *losses* flow *up* the stack from the bottom: common equity is wiped out first, then preferred, then subordinated debt takes a haircut, and only in a catastrophe do senior secured lenders lose anything.

This is the "stack" figure that opened the post — keep it in your head. Risk rises as you go down; expected return rises with it; and the price each investor pays reflects exactly where on the ladder they sit.

#### Worked example: the bankruptcy payout order on \$300M

Brewline fails. After liquidating everything — selling the stores, the equipment, the brand — the trustee recovers **\$300,000,000** to distribute. The claims against the company are:

- Secured debt: **\$200,000,000** (backed by the store properties)
- Senior unsecured bonds: **\$150,000,000**
- Subordinated debt: **\$50,000,000**
- Preferred stock: **\$30,000,000**
- Common equity: whatever is left

The waterfall pays strictly top-down, each tier in full before the next gets anything:

1. **Secured debt** gets its **\$200M** in full. Pool remaining: \$300M − \$200M = **\$100M**.
2. **Senior unsecured** is owed \$150M but only \$100M is left. It gets the **\$100M** — a recovery of 100/150 = **67 cents on the dollar**. Pool remaining: **\$0**.
3. **Subordinated debt, preferred stock, and common equity** all get **\$0**.

Notice what just happened: the subordinated *lenders* — who hold debt, who were "promised" repayment — recovered nothing, while the senior lenders got most of their money and the secured lenders were untouched. **The intuition: in bankruptcy the word "debt" means nothing on its own; only your rung on the seniority ladder decides whether you are paid in full, take a haircut, or get wiped out.** This is why "senior secured" bonds yield less than "subordinated" bonds from the very same company — they are simply safer claims on the identical business.

![Bankruptcy waterfall showing a 300 million dollar liquidation pool paid top down through the seniority ladder](/imgs/blogs/debt-vs-equity-the-two-ways-to-raise-capital-5.png)

## The cost of each: why equity is more expensive than debt

Here is a fact that surprises beginners: **equity is more expensive than debt for the company.** It feels backwards — debt has a visible interest bill and equity seems "free" because you never repay it. But "cost of capital" means the *return investors demand*, and equity investors demand far more.

Three reasons, all flowing directly from the stack:

**1. Equity is riskier, so it demands a higher return.** Equity holders are last in line and bear the full swing of the company's fortunes. No rational investor accepts that risk for a bond-like 5%. They demand a much higher *expected* return — historically equities have returned ~7–10% a year versus ~2–5% for high-grade bonds — precisely because they might get zero. The cost of equity is the price of standing last in line.

**2. Debt is contractually senior and often secured.** A lender's downside is protected by seniority, covenants, and sometimes collateral. Less risk borne means a lower return required. The cost of debt is roughly the interest rate the company pays on its bonds — and you can read that rate straight off the bond market.

**3. The tax shield makes debt cheaper still.** This is the big one. In most countries, *interest is tax-deductible but dividends are not.* A company's interest payments reduce its taxable income; its dividend payments do not. This subsidy — the **interest tax shield** — lowers the *effective* cost of debt by the corporate tax rate.

#### Worked example: the interest tax shield on \$100M at 6%

Brewline borrows **\$100,000,000** at a **6%** coupon. Its corporate tax rate is **21%**.

- Annual interest paid = \$100,000,000 × 6% = **\$6,000,000**.
- Because interest is tax-deductible, that \$6M reduces taxable income, saving tax of \$6,000,000 × 21% = **\$1,260,000** a year.
- So the *after-tax* cost of the debt is \$6,000,000 − \$1,260,000 = **\$4,740,000**, an effective rate of just **4.74%**, not 6%.

The general formula: after-tax cost of debt = coupon × (1 − tax rate) = 6% × (1 − 0.21) = **4.74%**. **The intuition: every dollar of interest comes with a government rebate equal to the tax rate, so leverage quietly lowers a profitable company's tax bill — which is a huge part of why almost every large firm carries some debt even when it does not need the cash.** (For how the *level* of interest rates is set across maturities — the input to every cost-of-debt calculation — see [the yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance).)

### A high-level look at WACC

If a company uses both debt and equity, its overall cost of capital is the blend of the two, weighted by how much of each it uses. That blend is the **Weighted Average Cost of Capital (WACC)**:

```
WACC = (E / V) * Re  +  (D / V) * Rd * (1 - Tax)
```

where `E` is the market value of equity, `D` the market value of debt, `V = E + D` the total, `Re` the cost of equity, `Rd` the pre-tax cost of debt, and `Tax` the corporate tax rate.

You do not need to memorise the math — the equity-research side of this blog owns the detailed derivation and how WACC feeds a discounted-cash-flow valuation, and you should read that for the rigorous version rather than re-deriving it here. What matters for *us* is the intuition: because the cost of debt (after tax) is lower than the cost of equity, adding *some* debt to an all-equity company *lowers* the blended WACC. A lower WACC means future cash flows are discounted less harshly, which means the company is worth more. That single mechanical fact is what tempts every firm toward leverage — and it sets up the central tension of the whole post.

### Leverage amplifies the return on equity

There is a second, more visceral reason firms borrow: leverage magnifies the return earned *by the equity holders*, as long as the business earns more than the cost of the debt. This is the engine behind private-equity buyouts and it is worth seeing in numbers.

#### Worked example: how borrowing amplifies (and endangers) equity returns

An investor buys a building for **\$100,000,000** that produces **\$8,000,000** of operating income a year — an 8% unlevered return.

- **All equity:** put in \$100M, earn \$8M → **8.0% return on equity.**
- **50% debt at 5%:** put in \$50M of equity, borrow \$50M. Interest = \$50M × 5% = \$2.5M. Profit to equity = \$8M − \$2.5M = **\$5.5M**, on \$50M of equity → **11.0% return on equity.**
- **80% debt at 5%:** put in \$20M, borrow \$80M. Interest = \$4M. Profit to equity = \$8M − \$4M = **\$4M**, on \$20M → **20.0% return on equity.**

Borrowing turned an 8% asset into a 20% equity return. But the amplifier runs both ways. If a bad year drops the building's income to **\$3,000,000**, the 80%-levered investor owes \$4M of interest against only \$3M of income — a **\$1M shortfall** they must cover from their own pocket or default. The all-equity owner simply earns 3% and is never in danger. **The intuition: leverage is a multiplier on the gap between what the asset earns and what the debt costs — wonderful when that gap is positive and stable, ruinous the moment it goes negative.**

## The capital-structure trade-off: why not 100% debt?

If debt is cheaper than equity, and adding debt lowers WACC and raises firm value, the obvious question is: *why doesn't every company finance itself entirely with debt?* This is one of the great questions of finance, and the answer earned two economists, Franco Modigliani and Merton Miller, a Nobel Prize.

### Modigliani–Miller: the starting point

In 1958, Modigliani and Miller proved a startling result. In a *perfect* world — no taxes, no bankruptcy costs, no asymmetry of information — **the capital structure does not matter at all.** A company worth \$1 billion is worth \$1 billion whether it is financed with all equity, all debt, or any mix. The reasoning: the total cash the business generates is fixed by its operations, not by how you slice the claims on it. Rearranging the slices cannot change the size of the pie. If you lever up, equity holders bear more risk per share and demand a higher return, which exactly offsets the cheaper debt. The WACC stays put. The value stays put.

This sounds like it makes the whole question pointless — but the genius of M&M is the *opposite*. By proving capital structure is irrelevant in a frictionless world, they told us exactly where to look for why it *does* matter in the real world: **the frictions.** Capital structure matters *only* to the extent that taxes, bankruptcy costs, and information gaps break the perfect-world assumptions. Each friction is a real-world reason to lean one way or the other.

### Friction 1: taxes push toward debt

Add corporate taxes and the symmetry breaks. As the tax-shield example showed, interest is deductible and dividends are not, so each dollar of debt creates a tax saving the company keeps. In the pure tax version of the theory, this pushes a firm toward *more* debt — the more you borrow, the bigger the shield, the higher the after-tax value. Taken literally, this argues for 100% debt. Obviously firms do not do that, which means there must be a force pulling the other way.

### Friction 2: financial distress pushes toward equity

That counterforce is the **cost of financial distress.** As a firm piles on debt, the probability that it cannot meet its fixed payments rises. And the costs of getting close to default are real and large:

- **Direct costs:** lawyers, bankers, and court fees in a restructuring or bankruptcy — easily 3–5% of firm value.
- **Indirect costs**, which dwarf the direct ones: customers stop buying (would you buy a car from a manufacturer that might not honour the warranty?), suppliers demand cash up front, the best employees leave, and management spends its time fighting fires instead of building the business. A distressed firm bleeds value long before it ever files.

So the more debt you add, the bigger the tax shield *but* the bigger the expected distress cost. Somewhere between zero debt and all debt is the level that maximises firm value — the **optimal capital structure**.

![Trade-off graph showing more debt creating a tax shield and lower WACC but rising distress risk toward an optimal structure](/imgs/blogs/debt-vs-equity-the-two-ways-to-raise-capital-8.png)

This is the **trade-off theory**: a firm should borrow up to the point where the marginal tax benefit of one more dollar of debt equals the marginal expected cost of distress it creates. It predicts that stable, profitable, asset-heavy firms (utilities, telecoms) should carry lots of debt — their cash flows are predictable, their distress risk low, and they have hard assets to pledge — while volatile, asset-light firms (early-stage tech, biotech) should carry little, because their cash flows are too uncertain to safely promise fixed payments.

### Friction 3: information and the pecking order

There is a second, subtler force. Managers usually know more about their company's true prospects than outside investors do — an **information asymmetry**. This shapes financing choices in a way the trade-off theory misses, captured by the **pecking-order theory**:

1. **Internal funds first.** Firms prefer to fund investment from their own retained earnings — no signalling, no fees, no dilution.
2. **Debt second.** If they must raise outside money, they prefer debt, because issuing a bond signals confidence ("we are sure we can make these payments") and does not give away ownership.
3. **Equity last.** They issue equity only as a last resort, because doing so sends a bad **signal**.

Why is issuing equity a bad signal? Because managers issue stock when they think it is *overvalued* — selling something for more than it is worth is a good deal for existing owners. Investors know this, so the announcement of a new share issue often makes the stock *fall*: the market reads it as "management thinks we are expensive." This signalling effect is real and measurable. It is a big reason mature, profitable firms lean on debt and retained earnings, and why a seasoned equity offering is treated as something close to bad news.

#### Worked example: which financing does Brewline pick?

Brewline needs \$100M and is choosing between debt and equity. Its EBIT is a stable **\$15M** a year, it owns its store properties, and its founders value control. Walk the frameworks:

- **Trade-off theory:** stable cash flow + hard assets + a 21% tax shield (\$1.26M/yr on \$100M at 6%) → it can safely carry debt; the shield is valuable; distress risk is modest. *Lean debt.*
- **Pecking order:** raising equity would signal the stock is overpriced and dilute the founders; debt signals confidence and keeps 100% ownership. *Lean debt.*
- **Control:** the founders refuse to give up votes. *Lean debt.*

A young, money-losing version of Brewline — no profits to shield, no stable cash to promise, no hard assets to pledge — would flip every one of these and be forced toward equity, which is exactly why startups raise venture *equity* and mature utilities issue *bonds*. **The intuition: the "right" mix is not a formula but the answer to three questions — how stable is the cash, how valuable is the tax shield, and how much does the owner care about control.**

## Hybrids: the instruments between debt and equity

We have been treating debt and equity as two sharply separate boxes. In practice they are the two ends of a *continuum*, and the most interesting instruments live in between. These **hybrids** let an issuer fine-tune exactly how much "debt-ness" and "equity-ness" to sell.

**Preferred stock** is the simplest hybrid. It is legally equity (it sits below all debt in the stack) but behaves like debt: it pays a *fixed* dividend, usually carries no vote, and has no upside beyond that dividend. Companies use it when they want money that does not count as debt on their balance sheet (so it does not breach loan covenants) but does not dilute common voting control. Banks issue a lot of it for exactly this regulatory reason.

**Convertible bonds** are the most elegant hybrid and worth understanding in detail. A convertible is a bond — it pays a coupon and ranks as debt — *but* it gives the holder the right to convert it into a fixed number of shares. It is a bond with a free equity lottery ticket stapled on. The issuer loves it because the equity option lets them pay a *lower* coupon than a plain bond would require. The investor loves it because they get downside protection (it is debt; if the stock tanks, they still collect the coupon and principal) plus upside (if the stock soars, they convert and ride the equity).

#### Worked example: a convertible's conversion arithmetic

Brewline issues a **\$1,000** convertible bond with a **3%** coupon and a **conversion ratio of 20** — meaning the holder can swap the bond for 20 shares whenever they choose. The implied **conversion price** is \$1,000 / 20 = **\$50 per share**. Brewline's stock currently trades at \$40.

- **If the stock stays at \$40:** converting gives 20 × \$40 = \$800, *less* than the \$1,000 bond. So the holder does *not* convert; they keep collecting the 3% coupon (\$30/yr) and get \$1,000 back at maturity. The bond floor protects them.
- **If the stock rises to \$80:** converting gives 20 × \$80 = **\$1,600** — far more than holding the \$1,000 bond. The holder converts and pockets the \$600 gain. They rode the equity upside.
- **The breakeven** is the conversion price, \$50: above it, conversion beats holding; below it, the bond is worth more.

For Brewline, the prize is the coupon: it borrowed at **3%** instead of the **6%** a plain bond would have cost, saving \$30/yr per \$1,000 — \$3,000,000 a year on a \$100M issue — in exchange for handing investors the option to become shareholders at \$50 if things go well. **The intuition: a convertible lets a company borrow cheaply today by promising to sell equity later, but only at a price it is happy with — the best of both atoms, paid for with potential future dilution.**

**Mezzanine financing** sits between senior debt and equity, common in private deals and leveraged buyouts. It is typically subordinated debt with an equity "kicker" — warrants or a conversion feature — so the lender earns a high coupon *plus* a slice of the upside, compensating them for ranking near the bottom of the stack. It is expensive money (often 12–20% all-in) used to bridge the gap when senior lenders will not lend enough and the owners do not want to sell more equity.

The chart below shows the same idea at the level of an entire pool of loans — a securitization, where one stream of cash flows is sliced into a senior (debt-like) tranche and an equity (first-loss) tranche.

![Securitization tranche stack splitting a 100 unit deal into senior mezzanine and equity first-loss claims](/imgs/blogs/debt-vs-equity-the-two-ways-to-raise-capital-7.png)

The **senior tranche** (80 of 100 units here) is paid first and rated AAA — it is the "debt" of the structure. The **equity tranche** (the bottom 5 units) absorbs the first losses and earns the residual — it is the "equity" of the structure. The mezzanine sits between. It is the exact same logic as a single company's capital stack, applied to a basket of loans. We go deep on how this machinery is built in [how banks turn loans into securities](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities); the point here is that **debt and equity are not a binary — they are a dial, and the whole craft of structured finance is choosing where to set it.**

## Common misconceptions

**"Equity is free money because you never pay it back."** No — equity is the *most expensive* form of capital. You never repay the principal, true, but you hand over a permanent claim on *all* future profits and *all* the upside, plus voting control. A profitable company that issues equity is selling its most valuable asset — its future — at whatever the market will pay. Investors demand 7–10% expected returns on equity precisely because of that open-ended risk; debt costs a third of that after tax.

**"Debt is dangerous, so the safest company has none."** Also wrong, in two ways. First, *zero* debt means a company is leaving the tax shield — free value — on the table, and is likely under-using its borrowing capacity. Second, "safe" is about *cash-flow stability*, not the existence of debt: a utility with steady revenue can carry 60% debt safely, while a biotech with no revenue is reckless at 10%. The danger is not debt itself but *too much fixed obligation relative to how reliable your cash is.*

**"Bondholders and shareholders want the same thing."** They are often in direct conflict. Shareholders, with unlimited upside and limited downside, *like* risk — a wild bet that might 10× the stock costs them little if it fails (they were standing last anyway) and pays hugely if it works. Bondholders, with capped upside, *hate* risk — they get nothing extra from a moonshot but lose everything in a bankruptcy. This "asset-substitution" conflict is why covenants exist: bondholders contractually stop shareholders from gambling with their collateral.

**"A company that issues stock is raising cash for itself."** Only in a *primary* offering (an IPO or a follow-on sale of *new* shares). When you buy Apple stock on the exchange, *Apple gets nothing* — you are buying existing shares from another investor in the **secondary market**. The company raised its money once, when those shares were first created. This is the primary/secondary split at the heart of this series, and it is the reason secondary-market liquidity matters so much: nobody would buy a newly issued share in the primary market if they could not sell it to someone else tomorrow. (See [what is a capital market](/blog/trading/capital-markets/what-is-a-capital-market-how-money-finds-its-best-use) for the two-engine machine in full.)

**"More leverage always means higher returns."** Leverage *amplifies* returns in both directions. Borrowing to invest magnifies your gains when things go well and magnifies your losses when they do not — and adds a fixed bill that can force a sale at the worst possible moment. The 2008 crisis and the 1998 collapse of [Long-Term Capital Management](/blog/trading/finance/ltcm-1998-when-genius-failed) were both, at root, stories of too much borrowed money meeting an unexpected loss. Leverage is a tool, not a free lunch.

## How it shows up in real markets

**Apple's "borrow to avoid tax" bonds (2013–2020).** For years Apple held hundreds of billions of dollars in cash *overseas*, which it could not bring home without paying US repatriation tax. So instead of using its own cash to fund dividends and buybacks, Apple *borrowed* tens of billions in the US bond market at sub-3% coupons. It was a pure capital-structure play: borrow cheaply at home, leave the taxed cash abroad, and capture the interest tax shield in the process. A company with more cash than it could spend chose *debt* — the clearest possible proof that capital structure is a deliberate choice, not a necessity.

**The 2021 IPO boom and 2022 bust.** In 2021, with interest rates near zero and equity valuations sky-high, companies rushed to *sell equity*: US IPOs raised about \$142 billion across 397 deals — a record. Then in 2022, as rates spiked and valuations cratered, the equity window slammed shut: just \$8 billion across 71 IPOs, a 94% collapse in proceeds. Companies that needed money in 2022 could not sell equity at acceptable prices, so they leaned on debt and private credit instead. This is the pecking order and the trade-off playing out in real time: *when does the equity window open?* When stocks are expensive and risk appetite is high. When it shuts, debt becomes the only door.

**WeWork: the cost of the wrong structure.** WeWork raised billions in equity at a private valuation of \$47 billion in early 2019, then saw its IPO collapse months later when investors balked at its losses and governance. Stripped of the equity it was counting on, it was left with enormous fixed lease obligations — effectively debt-like commitments — against cash flows that could not cover them. It eventually filed for bankruptcy in 2023. The lesson is the trade-off in reverse: a company with deeply uncertain cash flows had taken on huge *fixed* obligations, and when the equity window shut, the fixed bills sank it. Match the *type* of capital to the *stability* of the cash, or the structure eventually breaks the company.

**The 2008 securitization blow-up.** The same tranche logic in Figure 7 was applied to subprime mortgages on an enormous scale through the mid-2000s, with senior tranches rated AAA and sold as nearly risk-free debt. When the underlying loans defaulted far beyond what the thin equity tranches could absorb, losses tore *up* the stack into tranches everyone had treated as safe. It was a brutal lesson that the seniority ladder only protects you if the loss estimates beneath it are honest — the structure was sound; the inputs were not.

**The private-equity buyout playbook.** A leveraged buyout is the debt-equity fork wielded as a strategy. A private-equity firm buys a company using a small slice of its own *equity* and a large slab of *debt* secured against the target's own assets and cash flows — often 60–70% of the purchase price. As the leverage example above showed, this magnifies the return on the firm's equity if the business performs. The catch is the fixed bill: the acquired company now carries heavy interest obligations, so PE targets are chosen precisely for *stable, predictable* cash flows that can service that debt — mature businesses, not startups. When a buyout fails, it usually fails for the textbook reason: the debt load was sized for good times and the business hit a bad one. Toys "R" Us, loaded with roughly \$5 billion of buyout debt, could not carry the interest through a retail downturn and liquidated in 2018. The same leverage that promised outsized equity returns became the rope it hanged on. It is the trade-off theory's central warning made concrete: match the fixed obligation to the *reliability* of the cash, or the obligation wins.

**Vietnam's corporate-bond squeeze (2022).** Vietnam's capital market shows the same fork with local color. For years property developers funded themselves with a flood of corporate *bonds* (debt). When a 2022 crackdown on bond issuance abruptly shut that door, developers who had leaned entirely on debt faced a wall of maturities they could not refinance, and several defaulted — even as the equity market (the VN-Index, which fell from ~1,498 at end-2021 to ~1,007 at end-2022) offered no easy escape either. A market that over-relies on one financing channel is fragile when that channel closes; resilient firms keep both the debt door and the equity door usable. (For how foreign capital flows into and out of that market, see [foreign flows and the index effect in Vietnam](/blog/trading/vietnam-stocks/foreign-flows-etfs-and-the-index-effect-vietnam).)

## The takeaway: two atoms, infinite combinations

Step back and the whole landscape simplifies. Every financing decision a company ever makes, every instrument an investment bank ever structures, every claim that trades in the capital markets, is built from just two atoms:

- **Equity** — a residual ownership claim. Last in line, unlimited upside, no obligation to pay, but it costs control and dilutes, and it is the *most expensive* capital because investors demand a high return for bearing all the risk.
- **Debt** — a fixed senior claim. First in line (in its rung), capped upside, must be repaid on schedule or you default, but it is *cheaper*, *tax-favoured*, and dilutes nobody.

Between them sits a continuum — subordinated debt, preferred stock, convertibles, mezzanine, securitization tranches — that lets an issuer dial in exactly how much risk to transfer and to whom. The **capital stack** organises all of it into a seniority ladder that governs who gets paid first in good times and who absorbs losses first in bad. The **cost of each** flows directly from that ladder: lower rung, lower risk, lower required return. And the **mix** a firm chooses is the answer to three real-world questions the perfect-world M&M theorem taught us to ask — how big is the tax shield, how costly is distress, and what does management know that the market doesn't.

**How to read any company through this lens.** The next time you look at a real business, ask the three questions in order. *How is it financed?* Find the split between debt and equity on its balance sheet — a software company at 5% debt and a regulated utility at 60% debt are telling you how stable each one thinks its cash flows are. *Where do the claims sit?* Within the debt, note how much is secured and senior versus subordinated, and whether there is preferred stock above the common — that ladder tells you who absorbs a loss first and therefore who is really at risk. *Can it meet its fixed bill?* Compare the annual interest owed to the operating cash the business throws off; if interest eats most of the cash, the company is one bad year from trouble no matter how big it looks. Those three reads — the mix, the ladder, and the coverage — turn a wall of financial statements into a single sentence about how the firm is funded and how fragile that funding is. You now have the vocabulary to write that sentence for any company in the world.

Here is the insight to carry into the rest of this series. A capital market exists to *match* savers who want different risks with companies that need to fund different things. The debt-versus-equity fork is how that matching happens at the level of a single firm: the firm slices its future into claims of varying seniority, and the market sorts those claims to the investors whose risk appetite fits each rung — the pension fund buys the senior bond, the venture fund buys the equity, the hedge fund buys the distressed mezzanine. Get the slicing right and capital flows to its best use cheaply. Get it wrong — too much fixed debt against shaky cash, or too much equity given away too cheap — and the firm either chokes on its obligations or sells its future for a song. Two atoms, one ladder, and the entire art of corporate finance is choosing where to stand on it.

And the choice is never made in a vacuum. It bends with the price of money — when rates are near zero, debt is so cheap that even cash-rich firms borrow, and when rates spike the equity window can be the only one open. It bends with the maturity of the business — startups raise equity because no lender will accept their uncertainty, while mature cash machines lever up to harvest the tax shield. It bends with what management knows that the market does not, and with how much the founders care about keeping the steering wheel. The next time you read that a company "raised \$500 million," do not stop at the headline. Ask which atom it sold, what rung of the ladder it created, and what that choice reveals about how its managers see their own future. That single question — debt or equity, and where on the stack — is the seam where the whole capital-markets machine connects to a real company's fate.

## Further reading & cross-links

- [What is a capital market: how money finds its best use](/blog/trading/capital-markets/what-is-a-capital-market-how-money-finds-its-best-use) — the savings→investment machine and the primary/secondary split this post sits inside.
- [What a security actually is: claims you can sell](/blog/trading/capital-markets/what-a-security-actually-is-claims-you-can-sell) — why a standardised, transferable claim is the whole invention behind both shares and bonds.
- [Money market vs capital market: where short meets long](/blog/trading/capital-markets/money-market-vs-capital-market-where-short-meets-long) — how the *maturity* of a claim, not just its debt-or-equity nature, splits the market.
- [The yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) — how the level and shape of interest rates (the input to every cost-of-debt figure) is set across maturities.
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — the intermediaries who actually structure and sell these debt and equity claims.
- [Securitization: how banks turn loans into securities](/blog/trading/banking/securitization-how-banks-turn-loans-into-securities) — the tranche stack in Figure 7, built at industrial scale.
- [LTCM 1998: when genius failed](/blog/trading/finance/ltcm-1998-when-genius-failed) — what happens when leverage meets an unexpected loss.

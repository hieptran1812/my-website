---
title: "Corporate bonds: lending to companies"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into the corporate bond market: why companies borrow instead of selling stock, how a new bond is underwritten and priced, why it trades so differently from a stock, what covenants and embedded options like calls and puts actually do, and how the credit spread over Treasuries gets set and moves with the economy."
tags: ["fixed-income", "bonds", "corporate-bonds", "credit-spread", "covenants", "callable-bonds", "underwriting", "corporate-credit", "us-treasuries"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 43
---

> [!important]
> **TL;DR** — a corporate bond is a loan to a company sliced into tradable pieces; it pays the same shape of cash flows as a government bond, but you demand extra yield — the *credit spread* — because the company, unlike the government, can actually fail to pay you back.
> - Companies issue bonds because debt is usually **cheaper than equity** (lenders accept a smaller return than owners) and because the interest is **tax-deductible**, which lowers the true cost further. They also keep control: bondholders are not owners.
> - A new bond is sold through an **underwriting syndicate** that *builds a book* of investor orders, then prices the deal a touch cheap — the **new-issue concession** — so the bonds trade up on day one.
> - The secondary market for corporates is a **dealer / over-the-counter** market: you trade by phone or chat with a bank, not on a lit exchange, and it is far **less liquid** than the stock market.
> - **Covenants** are promises in the bond contract that protect lenders (limit new debt, dividends, asset sales). **Incurrence** covenants test only when the company acts; **maintenance** covenants test every quarter. Bonds often carry **embedded options** — *callable* (issuer can repay early), *puttable* (you can demand early repayment), *convertible* (swap into stock), *sinking funds* (forced gradual repayment).
> - Where a corporate sits versus a Treasury is the **spread**: corporate yield = risk-free Treasury yield + credit spread. That spread **widens when the economy weakens** and tightens when it recovers.
> - Running example: **Northwind Corp** issues a **\$500M 10-year bond at a 150 bp spread**, callable after 5 years. We trace the coupon, the proceeds, and the call decision step by step.

Why does a profitable, well-known company — one that could clearly afford to — go out and *borrow* half a billion dollars from strangers, when it already has cash in the bank and could simply sell more shares to raise the rest? And why would thousands of pension funds, insurers, and ordinary savers line up to lend it that money at a fixed rate for ten years, when they could buy the stock instead and share in all the upside?

The answer to both questions is the same single idea, and it is the idea this entire post is built around: **a corporate bond is a loan, and a loan is a fundamentally different deal from ownership.** When you lend a company money, you are not buying a piece of its future; you are buying a *promise* — a fixed schedule of payments, on fixed dates, that the company is legally obligated to honor before its owners see a cent. That promise is worth less than ownership when things go well (you only ever get your money back plus interest, never the windfall), and worth far more when things go badly (you stand ahead of the owners in line). The corporate bond market is the machinery that turns that promise into something you can buy, sell, price, and trade — roughly **\$8–9 trillion** of it in the United States alone.

![A side by side comparison showing a ten year US Treasury on the left and a Northwind Corp ten year bond on the right, with the same cash flow shape but the corporate bond adding a credit spread on top of the risk free yield](/imgs/blogs/corporate-bonds-lending-to-companies-1.png)

The diagram above is the mental model for everything that follows. On the left is a US Treasury — the government's promise, treated as effectively certain, paying the *risk-free rate*. On the right is the very same shape of promise from a single company: identical coupons, identical par repayment, identical maturity. The only difference is risk — the company *might* not pay — and so the buyer demands a little extra yield to compensate. That extra yield is the **credit spread**, and almost everything interesting about corporate bonds is really a story about what sets that spread, how it moves, and what protections you can negotiate to shrink it. (Everything here is educational, not investment advice; the goal is to understand the mechanism, not to tell you what to buy.)

## Foundations: the building blocks you need first

Let's assemble the vocabulary from zero. A few of these terms you may have met in healthy-bond contexts; here we sharpen them for the corporate world, where the company on the other side of the promise can genuinely disappoint you.

**A bond is a loan split into tradable units.** When a company "issues a bond," it borrows a large sum by selling many small, identical IOUs. Each one has a *face value* (also called *par*), almost always **\$1,000** in the corporate market, that the company promises to repay at a set *maturity* date. In between, it pays *coupons* — periodic interest, usually twice a year. Buy one bond and you have lent the company \$1,000; buy a thousand and you have lent it \$1,000,000. The full anatomy of par, coupon, and maturity is covered in [the anatomy of a bond](/blog/trading/fixed-income/anatomy-of-a-bond-par-coupon-maturity-issuer); what matters here is that a corporate bond is exactly that structure, issued by a *company* rather than a government.

**Debt versus equity — the two ways to fund a company.** A company raises money in exactly two flavors. *Equity* is ownership: sell shares and the buyers become part-owners, entitled to a slice of all future profits forever, but with no promise of any particular payment. *Debt* is lending: sell bonds (or take loans) and the buyers are owed a fixed schedule of payments, but own nothing. Owners are paid *last* and get *everything left over*; lenders are paid *first* and get *only what they were promised*. Every company chooses a *mix* of the two — its *capital structure* — and we will see shortly why most large companies deliberately use a lot of debt.

**The issuer is the borrower; the bondholder is the lender.** The *issuer* is the company that sells the bonds and owes the money. A *bondholder* (or *investor*, or *creditor*) is whoever currently owns a bond and is owed by the issuer. Ownership of a bond can change hands many times over its life — the original buyer can sell to someone else — but the issuer's obligation stays the same: pay whoever holds the bond on each coupon date.

**Yield, and the risk-free rate.** A bond's *yield* is the annual return you earn if you buy it at today's price and hold it to maturity, accounting for both the coupons and any gap between price and par. (The full machinery of yield is in [the many yields](/blog/trading/fixed-income/the-many-yields-current-yield-ytm-and-yield-to-call).) The *risk-free rate* is the yield on a US Treasury of the same maturity — "risk-free" because the US government is assumed certain to pay, since it can always print the dollars it owes. The risk-free rate is the baseline price of money; it is set, ultimately, by the Federal Reserve and the market's expectations, a story told in [interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).

**The credit spread — the heart of the matter.** A *credit spread* is the *extra* yield a corporate bond pays over a Treasury of the same maturity, quoted in *basis points*. A **basis point** (bp) is one hundredth of a percent — 0.01% — so 150 basis points is 1.50%. If a 10-year Treasury yields 4.50% and a 10-year corporate bond yields 6.00%, the credit spread is 150 bp. The spread is your compensation for two things the Treasury doesn't have: the risk the company *defaults* (fails to pay), and the fact that corporate bonds are harder to sell quickly (less *liquid*). Everything in this post eventually feeds the spread.

**Default, and credit risk.** A *default* is the event of a company failing to honor its bond — missing a coupon, missing the principal repayment, or breaching a key term. *Credit risk* is the chance of that happening, and the chance of how much you lose if it does. We treat the full mechanics of default and recovery elsewhere — see [credit risk](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back) and [seniority and recovery](/blog/trading/fixed-income/seniority-recovery-and-the-capital-structure) — but the one-line version is: the bigger the chance of default and the worse the expected loss, the wider the spread you demand.

**Investment grade versus high yield.** Rating agencies grade corporate issuers from very safe to very risky (the system is explained in [bond ratings](/blog/trading/fixed-income/bond-ratings-how-moodys-sp-and-fitch-grade-debt)). The market splits into two great bands at a single dividing line. *Investment grade* (IG) is BBB−/Baa3 and above: solid companies, low default rates, tight spreads. *High yield* (HY), also called *junk*, is BB+/Ba1 and below: shakier companies, real default rates, wide spreads. The great divide between them is so consequential it has its own post: [investment grade vs high yield](/blog/trading/fixed-income/investment-grade-vs-high-yield-the-great-divide).

With those eight ideas in hand, here is the sentence that motivates the whole post: **a corporate bond is just a Treasury's cash-flow promise made by a company instead of a government — and the entire price difference, the credit spread, is the market's running estimate of how likely that company is to break the promise, plus a premium for how hard the bond is to sell.**

## Why companies issue bonds instead of selling stock

Before we get into mechanics, settle the "why" — because it is genuinely counterintuitive that healthy companies choose to be in debt on purpose. There are three reasons, and they compound.

**Reason one: debt is cheaper than equity.** This sounds backwards — isn't borrowing expensive? But compare the two sources head to head. An equity investor buys your stock expecting, over the long run, something like a 7–10% annual return, because they are taking on all the risk of the business: if you fail, their shares can go to zero, so they demand a fat expected return to compensate. A bond investor, by contrast, only wants their money back plus interest. Because they are paid *first* and own a *promise*, their downside is far smaller, so they accept a far smaller return — say 6% for a solid company. From the company's point of view, that 6% bond is *cheaper capital* than the 9% the equity market effectively demands. A company funds itself at the lowest blended cost by using both: cheap, safe debt up to the point where more borrowing starts to look risky, and equity for the rest.

**Reason two: interest is tax-deductible.** Here is the quiet subsidy that tilts the whole corporate world toward debt. When a company pays *interest* to bondholders, it deducts that interest as an expense before calculating its taxes — so the government effectively pays part of the interest bill. When a company pays *dividends* to shareholders, there is no such deduction; dividends come out of after-tax profit. This makes debt cheaper still, by roughly the tax rate. The plumbing of this *tax shield* is worth a worked example.

**Reason three: debt doesn't dilute ownership or surrender control.** Sell new shares and you create new owners who get votes and a permanent claim on future profits — the existing owners' slices shrink (this is *dilution*). Sell bonds and you create lenders who get paid back and then go away; they have no vote and no claim on the upside. A founder or controlling shareholder who wants to grow without giving up the company will reach for debt first. The flip side, of course, is that debt is a *hard* obligation: miss a coupon and you can be forced into bankruptcy, whereas you can always skip a dividend. Debt is cheaper precisely *because* it is unforgiving.

#### Worked example: why a company prefers the bond to new shares

*Setup.* Northwind Corp needs to raise **\$500M** to build a new plant. Its corporate tax rate is **21%**. It can either issue a 10-year bond at a **6% coupon**, or sell new stock to equity investors who require a **9%** expected return.

*The equity path.* New shareholders put in \$500M and expect 9% a year, or about **\$45M** of value transferred to them annually (in dividends plus the share of growth they now own). None of it is tax-deductible. And every existing owner's stake is now permanently smaller.

*The debt path, before tax.* The bond costs 6% on \$500M = **\$30M** of interest a year. Already cheaper than the \$45M equity cost, because lenders demand less.

*The debt path, after the tax shield.* That \$30M of interest is deductible. At a 21% tax rate, the deduction saves Northwind \$30M × 21% = **\$6.3M** in taxes. So the *true* after-tax cost of the debt is \$30M − \$6.3M = **\$23.7M** a year — an effective rate of 23.7 / 500 = **4.74%**, not 6%.

*The comparison.* Debt costs Northwind about \$23.7M a year after tax; the equity would have cost it roughly \$45M a year in returns surrendered to new owners, plus dilution of the founders' stake. The bond is the obvious choice for the next \$500M.

*Intuition: debt is cheaper than equity because lenders accept less for standing first in line — and the tax deduction on interest makes it cheaper still.*

There is, of course, a limit. Each additional dollar of debt makes the company a little riskier — more fixed payments it *must* make, less cushion before trouble — so beyond some point lenders start demanding wider spreads and the cheapness fades. The art of *capital structure* is borrowing up to that point and no further. But for most large, stable companies, that point is a long way out, which is why the corporate bond market is so enormous.

## How a new corporate bond is born: underwriting and the new-issue process

A company doesn't just print bonds and email them to savers. A new bond goes through a well-worn industrial process, run by investment banks, that takes it from "we need \$500M" to "the bonds are trading." Understanding it explains a lot of otherwise-mysterious market behavior — especially the *new-issue concession*.

![A pipeline showing a new corporate bond moving from the issuer through an underwriting syndicate and book building and pricing to investors and then into the dealer secondary market](/imgs/blogs/corporate-bonds-lending-to-companies-2.png)

The pipeline above is the life of a new bond. Walk it left to right.

**Step one: the issuer decides to borrow.** Northwind's treasurer, with the board's blessing, decides to raise \$500M of 10-year debt. They pick a maturity, a rough size, and a target use of proceeds (the new plant), and they hire banks.

**Step two: the underwriting syndicate.** The company appoints one or more investment banks as *underwriters* — typically a *lead* (or "bookrunner") plus several *co-managers*, together called the *syndicate*. The word *underwrite* means the banks **commit to buy the whole deal from the issuer and resell it to investors** — they take the risk that the bonds might not all sell. (In practice most deals are "best efforts" or are de-risked by pre-marketing, but the principle of the bank standing between issuer and market is real.) For this service the banks earn an *underwriting fee* (or "gross spread"), usually a fraction of a percent of the deal size for an investment-grade bond, more for high yield.

**Step three: book-building.** This is the heart of pricing. The syndicate announces the deal to investors with *initial price talk* — a first guess at the spread, say "150 to 160 basis points over Treasuries." Investors who are interested place *orders* — "I'll buy \$20M at 150" — and the banks assemble these into a *book* of demand. If the book is huge (the deal is *oversubscribed* — more orders than bonds), the banks *tighten* the price talk: "actually we'll price at 150, maybe 148." If demand is thin, they widen it to attract buyers. The final spread is whatever clears the book.

**Step four: pricing and the new-issue concession.** When the book is set, the deal *prices*: the final coupon and spread are locked in. Crucially, banks almost always price a new bond a touch *cheap* — at a slightly wider spread (and so a slightly lower price) than where the company's existing bonds trade. That deliberate discount is the **new-issue concession**, typically **5–15 bp** for an investment-grade deal. Why give money away? Because it guarantees the deal sells, rewards the investors who committed early, and makes the bonds *tick up* in price on the first day of trading — a "successful" deal everyone feels good about. The concession is, in effect, the price of certainty for the issuer and a small day-one gift to buyers.

**Step five: settlement and the secondary market.** A few days after pricing, the deal *settles*: investors pay, the issuer receives the cash (minus fees), and the bonds are theirs. From that moment the bond lives in the *secondary market*, where it trades among investors for the rest of its life — a market we'll examine next.

#### Worked example: Northwind's \$500M deal prices

*Setup.* The 10-year Treasury yields **4.50%**. Northwind is a solid BBB-rated company; its existing 10-year bonds trade at about a **140 bp** spread. It wants to raise \$500M.

*Initial price talk.* The lead bank announces "10-year senior notes, initial price talk +160 area" — deliberately wide of the 140 where Northwind already trades, to draw a big book.

*Book-building.* Investors love the name; orders pour in and the book reaches \$2 billion — four times the \$500M on offer. With that much demand, the bank tightens: "revised guidance +150, the number." The deal is *4× oversubscribed*.

*Pricing.* The bond prices at **+150 bp**, for a coupon of 4.50% + 1.50% = **6.00%**. Note the 150 versus the 140 on existing bonds: that **10 bp** gap is the new-issue concession. Northwind paid 10 extra basis points — about \$500,000 a year — to guarantee the deal and reward buyers.

*Proceeds.* The underwriting fee is **0.50%** of \$500M = **\$2.5M**. Northwind receives \$500M − \$2.5M = **\$497.5M** in cash; the banks keep \$2.5M for placing the deal.

*Day one.* Because the bond priced 10 bp cheap, it trades *up* immediately as the secondary market pulls it toward the 140 fair value — early buyers are pleased, and Northwind has its money.

*Intuition: the new-issue concession is a few basis points the issuer gives up to make the deal sell smoothly and reward the investors who showed up.*

## The secondary market: why bonds trade nothing like stocks

Once a bond is issued, it trades — but in a market that would feel alien to anyone used to buying stocks on an app. Three differences matter.

**Corporate bonds trade over-the-counter, through dealers.** A stock trades on a *lit exchange*: a central order book where buyers' and sellers' prices meet, visible to all, with continuous prices. A corporate bond trades *over-the-counter* (OTC): you call (or, today, electronically message) a bank's *dealer desk*, ask for a price, and trade directly with that bank, which holds the bond on its own books for a while before selling it on. There is no single visible price; the market is a web of dealer quotes. The bank makes money on the *bid-ask spread* — the gap between the price at which it will buy from you (the *bid*) and sell to you (the *ask*) — which for a bond can be far wider than for a stock. (The full ecosystem of who trades and who holds bonds is in [who buys bonds](/blog/trading/fixed-income/who-buys-bonds-the-global-demand-for-safe-income).)

**Corporate bonds are far less liquid.** *Liquidity* is how easily you can sell a sizable position quickly without moving the price against yourself. A big-company stock is extremely liquid — millions of shares change hands every minute. A specific corporate bond may not trade *at all* for days or weeks. A single company that has one stock typically has *dozens* of different bonds outstanding (issued in different years, at different maturities, with different coupons), and the buyers are mostly large institutions that buy and *hold to maturity*. So on any given day, most individual bonds simply sit. When you do want to sell, you may have to accept a meaningfully worse price — and part of the credit spread you earned was *paying you for exactly this illiquidity*.

**Prices are quoted as a percentage of par, and you pay accrued interest.** A bond price isn't a dollar figure like a stock; it's quoted as a *percentage of face value*. "Northwind trades at 98.5" means \$985 per \$1,000 bond. "At 102" means \$1,020. Par is 100. And because coupons are paid only twice a year, when you buy a bond mid-period you also pay the seller the interest that has *accrued* since the last coupon — *accrued interest*, added on top of the quoted price.

#### Worked example: selling a bond into a thin market

*Setup.* You own \$1,000,000 face value of Northwind's 6% bond (1,000 bonds). The "fair" price is 100 (par), so it's notionally worth \$1,000,000. You need cash and want to sell today.

*The stock-market intuition (wrong here).* If this were a megacap stock, you'd sell instantly at the screen price, losing maybe a penny to the spread.

*The bond-market reality.* You message three dealers. The fair mid-price is 100, but each dealer quotes you a *bid* below it because they'll have to hold your bonds until they find a buyer, and the bond is illiquid. Best bid: **99.25**. You sell at 99.25 → \$992,500. The 0.75-point haircut — **\$7,500** — is the cost of demanding immediacy in an illiquid market.

*The patient alternative.* If instead you'd worked the order over a few days, advertised it quietly, and waited for a natural buyer, you might have gotten 99.85 — saving most of that \$7,500. Liquidity is something you pay for when you need it now.

*Intuition: in the corporate bond market, the price you can actually transact at depends heavily on how quickly you need to trade and how often that specific bond changes hands.*

## The credit spread and the business cycle

We've defined the credit spread as the extra yield over Treasuries. Now the crucial dynamic: **the spread is not constant — it breathes with the economy.** This is the single most important thing to internalize about corporate bonds, because it's where they connect to the wider world.

The logic is a chain. When the economy weakens — growth slows, profits fall, layoffs rise — more companies edge toward trouble, and the *probability of default* goes up. Investors, seeing this, demand *more* compensation to hold corporate risk, so credit spreads *widen*. When the economy strengthens, defaults fade, confidence returns, and spreads *tighten*. Corporate spreads are therefore a real-time barometer of economic fear: tight in good times, blowing out in recessions and panics.

![An X Y chart showing the business cycle in blue with growth strong then crashing into recession then recovering, and the credit spread in red moving opposite, spiking to about six hundred basis points at the recession trough then tightening as growth returns](/imgs/blogs/corporate-bonds-lending-to-companies-3.png)

The chart above is the influence picture for the whole post. The blue line is the business cycle — strong growth, a crash into recession, then recovery. The red line is the credit spread, and notice how it moves *opposite* the cycle: as growth tumbles toward the recession trough, the spread spikes (here to roughly 600 bp, a severe-recession level), and as growth comes back, the spread tightens again. The dotted vertical line marks the recession, where the two are at their most extreme. This counter-cyclical behavior is why credit spreads are watched as closely as the [yield curve](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) for signs of stress, and it ties directly to the allocator's view in [corporate credit, investment grade, high yield, and spreads](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads).

The decomposition is worth stating cleanly. Your corporate yield is built in layers:

$$y_{\text{corp}} = r_{\text{risk-free}} + s_{\text{credit}}$$

where $y_{\text{corp}}$ is the yield you earn on the corporate bond, $r_{\text{risk-free}}$ is the Treasury yield of the same maturity, and $s_{\text{credit}}$ is the credit spread. The risk-free part moves with the Fed and the economy's rate expectations; the spread part moves with credit fear. The two can move *together* (a recession often brings both falling Treasury yields *and* widening spreads) or *against* each other, which is exactly why a corporate bond's price can behave in ways a pure Treasury never would.

#### Worked example: the same bond in a boom and a bust

*Setup.* Northwind's 10-year bond was issued at par with a 6% coupon: 4.50% Treasury + 150 bp spread. You hold it. Two years pass.

*Scenario A — the boom.* The economy is strong; defaults are rare; investors are hungry for yield. Northwind's spread tightens from 150 bp to **100 bp**. Suppose Treasury yields are unchanged at 4.50%, so Northwind's yield falls from 6.00% to **5.50%**. Because [price moves opposite yield](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds), the bond's price *rises* — for a bond with roughly 7 years of duration left, a 0.50% yield drop lifts the price about 0.50% × 7 ≈ **3.5%**, to about **103.5**. The spread tightening made you money.

*Scenario B — the bust.* Two years later a recession hits. Defaults climb; investors flee corporate risk. Northwind's spread blows out from 150 bp to **400 bp**. Even if the Fed cuts and Treasury yields *fall* to 3.50%, Northwind's yield is now 3.50% + 4.00% = **7.50%** — up from 6.00%. The price *falls*: a 1.50% yield rise on ~7 years of duration is about −10.5%, to roughly **89.5**. The widening spread cost you more than the falling Treasury yield helped.

*Intuition: with a corporate bond you are exposed to two moving parts — the risk-free rate and the credit spread — and in a recession the spread can hurt you even while falling Treasury yields would have helped a government bond.*

### What actually sets the *level* of the spread

We've said the spread compensates for default risk and illiquidity, but it's worth seeing the arithmetic, because it explains why spreads are where they are. The default-related part of the spread is, in essence, the *expected loss* per year from default, plus a *risk premium* on top for bearing that uncertainty. Expected loss itself has two pieces:

$$\text{expected loss} \approx p_{\text{default}} \times \text{LGD}$$

where $p_{\text{default}}$ is the annual *probability of default* and $\text{LGD}$ is the *loss given default* — the fraction you *lose* if the company defaults, which is one minus the recovery rate. (The mechanics of recovery and seniority are in [seniority and recovery](/blog/trading/fixed-income/seniority-recovery-and-the-capital-structure); spreads as a price for default probability get their own treatment in [credit spreads](/blog/trading/fixed-income/credit-spreads-pricing-the-probability-of-default).) Walk a concrete case: if a bond has a 2% annual chance of defaulting and you'd lose 60% of your money if it did, the expected loss is 2% × 60% = **1.2% a year** — about 120 bp. That alone would justify roughly a 120 bp spread; the *actual* spread is usually somewhat wider, because investors demand extra for the *uncertainty* (defaults cluster in bad times, exactly when you can least afford them) and for illiquidity.

Two features fall out of this decomposition. First, **the spread is sensitive to both pieces, and they move together in a downturn**: in a recession, default probabilities rise *and* recovery rates fall (assets fetch less in a fire sale), so expected loss can jump more than either piece alone — which is why spreads gap so violently in crises rather than drifting. Second, **the equity cushion underneath your bond is your real margin of safety**. A bondholder is paid before the owners, so the more equity value sits below your debt, the more the company's assets can fall before *you* take a loss. A company financed 70% by equity and 30% by debt can lose well over half its value before bondholders are impaired; one financed 20% equity and 80% debt has almost no cushion, so its bonds carry a far wider spread for the same business. When you size up a corporate bond, you are really asking: how thick is the equity cushion beneath me, and how stable are the cash flows that maintain it?

#### Worked example: turning a default probability into a spread

*Setup.* You're sizing two bonds. Bond X is from a stable, low-leverage firm: you estimate a **0.5%** annual default probability and, because it's senior with good asset coverage, a **40%** loss-given-default. Bond Y is a high-yield name: **5%** annual default probability and **65%** loss-given-default.

*Bond X's expected loss.* 0.5% × 40% = **0.20% a year** = 20 bp. Add a modest risk-and-liquidity premium and the bond might trade around an **80 bp** spread — comfortably investment grade.

*Bond Y's expected loss.* 5% × 65% = **3.25% a year** = 325 bp of pure expected loss. Because default risk this size also carries a big uncertainty premium, the bond might trade at a **500 bp** spread or more.

*The check.* The \$ math lines up with the market: on \$100,000 of Bond Y, a 5% default chance and 65% loss implies you'd *expect* to lose about \$3,250 a year to defaults — so a coupon spread under ~325 bp wouldn't even cover the expected losses, let alone pay you for the risk. That's why junk bonds *must* pay wide spreads: a large slice of that fat coupon is simply replacing the money lost to the defaults that *will* happen across a portfolio.

*Intuition: a credit spread is, at its core, the market charging you the expected annual loss from default plus a premium for the uncertainty — so a high coupon on a risky bond is largely pre-payment for losses, not free extra income.*

## Covenants: the promises that protect lenders

When you lend a company money for ten years, a lot can change. The company could pile on *more* debt that ranks ahead of you, sell off the very assets that backed your loan, or pay all its cash out to shareholders and leave you holding an empty shell. *Covenants* are the contractual promises, written into the bond's legal document (the *indenture*), that limit what the company can do — they are how a lender fences in the borrower's behavior for the life of the loan.

Covenants come in two great families, and the difference between them is one of the most practically important distinctions in all of credit.

![A matrix comparing incurrence covenants typical of bonds against maintenance covenants typical of bank loans, across when each is tested, what it restricts, and what it protects the lender from](/imgs/blogs/corporate-bonds-lending-to-companies-5.png)

**Incurrence covenants test only when the company *acts*.** An *incurrence* covenant is triggered by a specific *action* the company chooses to take. The classic form: "you may not *incur* (take on) new debt if doing so would push your leverage above 4× EBITDA." Note the trigger — the test only runs *when the company tries to borrow more, pay a dividend, sell assets, or grant a lien to another lender*. If the company simply sits still and its numbers drift the wrong way, an incurrence covenant does nothing. These are the dominant covenants in *bonds*, because bondholders are dispersed and can't easily renegotiate, so they prefer rules that only bite on discrete corporate decisions.

**Maintenance covenants test *continuously*.** A *maintenance* covenant requires the company to *maintain* a financial ratio within limits *every reporting period*, no matter what it does. The classic forms: "net debt / EBITDA must stay below 4× *at the end of every quarter*," or "interest coverage must stay above 3×." If the company's earnings slip and the ratio breaches the limit — even with no action on the company's part — that's a *covenant breach*, which is a *default* under the loan and forces a renegotiation. Maintenance covenants live mostly in *bank loans*, where a small group of lenders can sit down with the company and rework terms. Because they catch deterioration *early*, they are far more protective than incurrence covenants.

The practical upshot: a *bank loan* with maintenance covenants gives lenders an early seat at the table when a company starts to slip; a *bond* with only incurrence covenants lets the company drift a long way before the lenders have any leverage. This is one reason bank loans often recover more than bonds in a default — the lenders got to act sooner. (The seniority side of that story is in [seniority and recovery](/blog/trading/fixed-income/seniority-recovery-and-the-capital-structure).)

A few specific covenants you'll meet by name:

- **Limitation on indebtedness** — caps how much more debt the company can take on (often as a leverage ratio).
- **Restricted payments** — limits dividends, buybacks, and other cash paid *out* to shareholders, so cash isn't drained away from lenders.
- **Negative pledge** — the company promises not to grant *liens* (secured claims) to other lenders that would jump ahead of you on specific assets.
- **Limitation on asset sales** — restricts selling off the assets, or requires the proceeds be used to repay debt.
- **Change of control put** — if someone buys the company, *you* get the right to sell your bonds back at (typically) 101, so a new owner can't load the company with debt at your expense.

#### Worked example: a covenant earns its keep

*Setup.* You hold Northwind's bond. It carries an incurrence covenant: Northwind may not issue new debt if total leverage would exceed **4.0× EBITDA**. Today Northwind has \$2 billion of debt and \$600M of EBITDA — leverage of 2 billion / 600 million = **3.3×**.

*The temptation.* Northwind's owners want to pay themselves a giant \$1 billion dividend, funded by *new* borrowing. That would add \$1B of debt, taking total debt to \$3B and leverage to 3,000 / 600 = **5.0×** — well above the 4.0× cap.

*The covenant bites.* Because issuing that debt is an *action* that would breach the 4.0× incurrence test, the covenant *forbids* it. Northwind cannot do the full dividend without your consent. The most it could borrow and stay at 4.0× is debt of 4.0 × \$600M = \$2.4B — only \$400M more, not \$1B.

*Without the covenant.* If your bond had *no* leverage covenant, Northwind could borrow the full \$1B, pay the dividend, and leave you holding a bond from a company now levered 5.0× — much riskier, and your bond's price would fall to reflect it. The covenant protected the *credit quality* you bought.

*Intuition: a covenant is a pre-agreed limit that stops the company from doing the things that would quietly make your bond riskier after you've already lent the money.*

## Embedded options: calls, puts, converts, and sinking funds

Many corporate bonds aren't plain "pay coupons then par at maturity" instruments. They carry *embedded options* — rights, baked into the bond, that let one side change the deal under certain conditions. These options are not free; they show up in the coupon. The most important by far is the *call*.

### The callable bond: the issuer's option to repay early

A *callable* bond gives the **issuer** the right (not the obligation) to repay the bond early, at a set price, after a set date. Northwind's bond is *callable after 5 years at 102.5* — meaning after year 5, Northwind can hand bondholders \$1,025 per \$1,000 bond and cancel the rest of the debt. Why would a company want that? Because if interest rates (or its own spread) *fall*, it can call the expensive old bond and refinance with a cheaper new one — exactly like a homeowner refinancing a mortgage when rates drop.

That right is great for the issuer and *bad* for you, the bondholder, and the bond's behavior shows it.

![An X Y chart of bond price against market yield showing a straight bond price rising freely as yields fall while a callable bond flattens and caps at its call price of one thousand twenty five dollars, illustrating negative convexity](/imgs/blogs/corporate-bonds-lending-to-companies-4.png)

The chart above is the key picture. The horizontal axis is the market yield, rising to the right; the vertical axis is the bond's price. The blue line is a *straight* (non-callable) bond: as yields fall (move left), its price rises freely and ever more steeply — the convex curve that makes ordinary bonds so valuable when rates drop. The red line is the *callable* bond. At high yields (right side) it behaves just like the straight bond. But as yields fall and the price approaches the **\$1,025 call price**, the curve *flattens and caps*: the price can't rise much above the call price, because everyone knows that if it did, Northwind would simply *call* the bond at \$1,025 and you'd lose the rest. That ceiling — where the price curve bends the *wrong way*, away from you — is **negative convexity**. (Convexity in general, and why it's normally your friend, is the subject of [convexity](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story).)

Because the call option hurts the bondholder, a callable bond must pay you *more* — a higher coupon, or equivalently a wider spread — than an otherwise-identical non-callable bond. You are, in effect, *selling* the issuer an option, and the extra yield is your premium. This is also why callable bonds are quoted on a *yield-to-call* as well as a yield-to-maturity (see [the many yields](/blog/trading/fixed-income/the-many-yields-current-yield-ytm-and-yield-to-call)): if rates fall, you should assume you'll be called away at the worst time for you.

#### Worked example: Northwind decides whether to call

*Setup.* Five years have passed. Northwind's \$500M bond pays a 6% coupon (\$30M a year) and has 5 years left. It is callable now at **102.5** — to call it, Northwind must pay bondholders \$500M × 1.025 = **\$512.5M**, a call premium of **\$12.5M** over par. Meanwhile, rates and Northwind's spread have fallen: it could issue a *new* 5-year bond today at only a **5%** coupon.

*The savings from refinancing.* A new 5% bond on \$500M costs \$25M a year, versus \$30M on the old 6% bond — a saving of **\$5M a year**. Over the 5 remaining years, that's \$5M × 5 = **\$25M** of interest saved.

*The cost of calling.* To capture that, Northwind must pay the \$12.5M call premium today (plus some new issuance fees, say \$2.5M) — roughly **\$15M** of cost.

*The decision.* \$25M saved versus \$15M to capture it → a net gain of about **\$10M**. Northwind *calls* the bond: it pays you \$1,025 per bond, retires the 6% debt, and issues new 5% debt.

*Your side of it.* You got \$1,025 back — a small premium over par — but your lovely 6% bond is *gone*, exactly when rates have fallen and you can only reinvest the cash at 5% or less. This is *reinvestment risk* (covered in [reinvestment risk](/blog/trading/fixed-income/reinvestment-risk-and-the-two-faces-of-yield)) in its purest form: the call took your high coupon away at the worst possible moment for you.

*Intuition: the issuer calls a bond when refinancing saves more than the call premium costs — which is precisely when keeping your high coupon would have been most valuable to you.*

### Puttable, convertible, and sinking-fund bonds

Three more embedded features round out the toolkit; each shifts value between issuer and holder.

**Puttable bonds — your option to demand early repayment.** A *puttable* bond is the mirror image of a callable one: it gives *you*, the bondholder, the right to *sell the bond back* to the issuer at a set price (usually par) on set dates. This protects you if rates *rise* or the company *deteriorates* — you can hand the bond back and get your money. Because the put benefits *you*, a puttable bond pays a *lower* coupon than a plain bond (you're paying for the protection by accepting less yield). Puts are far rarer than calls in the corporate market.

**Convertible bonds — swap the bond for stock.** A *convertible* bond gives you the right to *convert* it into a fixed number of the company's shares. It's a hybrid: a bond with a coupon and a maturity, *plus* a built-in equity call option. If the stock soars, you convert and capture the upside; if it doesn't, you keep collecting coupons and get your par back like any bond. Because that equity option is valuable to you, convertibles pay a *low* coupon — companies (especially younger, higher-growth ones) like them because they can borrow cheaply, and investors like the "bond floor with equity upside" shape. The cost is dilution if the bonds convert.

**Sinking funds — forced gradual repayment.** A *sinking fund* requires the issuer to retire a portion of the bond issue *each year* before maturity — by buying bonds back in the market or calling a slice at par by lottery — rather than repaying the whole thing in one balloon at the end. For the issuer it's a discipline; for the lender it *reduces credit risk* (the debt shrinks steadily, so there's less to default on at maturity) but adds a little uncertainty about exactly when your particular bond gets retired.

Here is the unifying principle for every embedded option: **an option that benefits the issuer (a call) makes the bond pay you more; an option that benefits you (a put, a convert) makes the bond pay you less.** The coupon always prices the option.

#### Worked example: the convertible's split personality

*Setup.* Northwind issues a 5-year *convertible* bond, \$1,000 par, with only a **2% coupon** (versus 6% on its straight bond). Each bond converts into **20 shares** of Northwind stock. The stock is at \$40 today, so 20 shares are worth \$800 — less than the \$1,000 bond, so no one would convert yet. The *conversion price* is \$1,000 / 20 = **\$50** per share.

*If the stock stagnates.* The stock stays around \$40. You never convert; you collect your 2% coupons and get \$1,000 back at maturity. You earned less than the straight bond's 6% — that's the price you paid for the option.

*If the stock soars.* Northwind's stock jumps to **\$80**. Now 20 shares are worth 20 × \$80 = **\$1,600**. You convert: your \$1,000 bond becomes \$1,600 of stock — a **60% gain**, far beyond anything a plain bond could deliver. The equity option paid off.

*The issuer's view.* Northwind borrowed at 2% instead of 6% — cheap money — at the cost of issuing new shares (dilution) only if the stock does very well, which is a nice problem to have.

*Intuition: a convertible is a low-coupon bond that quietly contains a call option on the company's stock — you give up yield in exchange for a shot at the equity upside.*

## The shape of the corporate market: ratings and sectors

Step back and look at the whole market. It is organized first by *credit quality* and then by *sector* — and where a bond falls in that grid largely determines its spread.

![A grid of the corporate bond market split into a large investment grade band and a smaller high yield band, then broken down by sectors such as banks technology energy retail utilities and leveraged buyouts, with the spread paid in each band shown below](/imgs/blogs/corporate-bonds-lending-to-companies-6.png)

The grid above lays it out. The first split is the great divide between **investment grade** (BBB− and up, roughly \$7 trillion in the US, low default, tight spreads of about 80–150 bp) and **high yield** (BB+ and below, roughly \$1.3 trillion, real default risk, wide spreads of 300–600+ bp). Within each band, the *sector* matters enormously, because sectors differ in how stable and how levered their cash flows are.

- **Banks and financials** are the single biggest investment-grade issuers — they borrow constantly to fund their own lending.
- **Technology, healthcare, and consumer staples** issue IG debt backed by stable, recurring cash flows.
- **Utilities** are steady and predictable but very sensitive to interest rates (they carry a lot of long debt).
- **Energy, retail, media, telecom, and leveraged buyouts** populate the riskier and high-yield end — cyclical demand, thin margins, or heavy debt loads make their cash flows shakier, so they pay wide spreads.

When a credit analyst sizes up a bond, the *sector* is the starting point: a utility and a shale-oil driller with the *same* leverage are not the same risk, because the utility's revenue is regulated and steady while the driller's swings with the oil price. The spread reflects both the rating *and* the sector's inherent volatility.

#### Worked example: two BBB bonds, two different spreads

*Setup.* Two companies are both rated **BBB** (the lowest rung of investment grade). One is **Northwind Utility**, a regulated electric utility; the other is **Drillco**, an oil-and-gas producer. Both issue 10-year bonds when the Treasury yields 4.50%.

*The utility.* Northwind Utility's revenue is set by a regulator and barely moves with the economy. Investors see low cash-flow risk and price the bond at a **110 bp** spread → a 5.60% coupon.

*The driller.* Drillco's revenue swings violently with the oil price; a downturn could halve its cash flow. Even at the *same* BBB rating, investors demand more for that volatility and price its bond at a **200 bp** spread → a 6.50% coupon.

*The lesson.* Same rating, **90 bp** of spread difference — about \$4.5M a year more interest on a \$500M deal — entirely because of *sector* cash-flow stability. The rating is a starting point, not the whole story.

*Intuition: the credit spread reflects not just the rating but the predictability of the cash flows behind it, which is why sector matters so much.*

## Putting it together: Northwind from issue to call

We've met every piece; now watch them combine in one running deal.

![A matrix tracing Northwind's five hundred million dollar deal showing how the credit spread sets the coupon and annual interest, how the underwriting fee sets the net proceeds, and how the call decision is made when refinancing saves more than the call premium costs](/imgs/blogs/corporate-bonds-lending-to-companies-7.png)

The matrix above is the whole worked example on one page. Read it row by row.

- **The coupon row.** The 10-year Treasury yields 4.50%; Northwind's 150 bp spread sets the coupon at 6.00%; on \$500M of par that's **\$30M of interest every year**, paid until the bond matures or is called.
- **The proceeds row.** The bond is issued at par, but the 0.50% underwriting fee — \$2.5M — comes out, so Northwind actually receives **\$497.5M**, with the syndicate keeping \$2.5M for placing the deal.
- **The call row.** The bond is callable after year 5 at 102.5. When rates later fall and Northwind could refinance at ~5%, the math (saving \$5M/yr × 5 years = \$25M, versus a \$12.5M call premium) says **call it** — Northwind hands you \$1,025 and refinances cheaper, and your high coupon disappears.

#### Worked example: your total return as a Northwind bondholder

*Setup.* You bought \$100,000 face of the Northwind 6% bond at issue (par, so \$100,000). You expected 6% a year for 10 years. Then it gets called after year 5.

*Years 1–5: the coupons.* You collect 6% on \$100,000 = **\$6,000 a year**, for 5 years = **\$30,000** of coupons.

*Year 5: the call.* Northwind calls at 102.5. You receive \$100,000 × 1.025 = **\$102,500** — your \$100,000 back plus a **\$2,500** call premium.

*Your realized return.* Over 5 years you received \$30,000 of coupons + \$2,500 of premium = \$32,500 of income on a \$100,000 investment, plus your principal back. That's a healthy return — but *not* the 10 years of 6% you signed up for.

*The reinvestment problem.* You now have \$102,500 to redeploy, but rates have fallen — the best comparable bond now yields 5%. Your *next* five years will earn 5%, not 6%, so your *forward* income drops by about \$1,000 a year. The call captured the issuer's gain at your expense.

*Intuition: the call option means your best-case is "get your money back early when reinvesting is least attractive" — which is why callable bonds must pay you extra up front.*

## Common misconceptions

**"Bonds are safe; only stocks lose money."** A corporate bond can absolutely lose money — and in two distinct ways. If the company *defaults*, you may recover only cents on the dollar (often \$0.40–0.60 for senior unsecured debt, less for subordinated). And even with no default, if interest rates *rise* or the credit spread *widens*, the bond's market price falls, and selling locks in the loss. A "safe" bond is only safe if the issuer pays *and* you hold to maturity *and* you don't mind the price swings in between. Investment-grade bonds are *lower* risk than stocks, not *no* risk.

**"A higher coupon means a better bond."** The coupon is set at issue to make the bond sell at par given the issuer's risk *at that moment*. A bond paying a 9% coupon isn't generous — it's *risky*, and the high coupon is compensation for that risk (or for an embedded call that hurts you). A 3% coupon from a rock-solid issuer and a 9% coupon from a shaky one can be priced *fairly* against each other. Yield, not coupon, is the comparable number — and even yield must be read alongside the *risk* you're taking to earn it.

**"Bonds trade like stocks — I can sell instantly at the screen price."** There is no single "screen price" for a corporate bond, and most specific bonds don't trade every day. You trade by asking dealers for quotes, you pay a bid-ask spread that can be far wider than a stock's, and in stressed markets liquidity can vanish entirely — exactly when you most want to sell. Part of the spread you earn is *payment for this illiquidity*; it's not a free lunch.

**"A bond's rating tells me everything about its risk."** A rating is a useful summary, but it's coarse, it lags reality (agencies often downgrade *after* trouble is obvious), and it ignores sector dynamics, covenant quality, and embedded options. Two BBB bonds can carry very different spreads because one is a stable utility and the other a cyclical driller, or because one has strong covenants and the other none. The market's *spread* often prices risk faster and finer than the rating does — see [bond ratings](/blog/trading/fixed-income/bond-ratings-how-moodys-sp-and-fitch-grade-debt).

**"If I hold to maturity, price moves don't matter."** True for *default-free* bonds held to maturity — but a *corporate* bond can default before maturity, in which case "holding to maturity" means holding through a restructuring and recovering a fraction. And if you might *need* to sell early (most investors do, sometimes), the interim price absolutely matters. The "hold to maturity, ignore the noise" mantra quietly assumes the issuer survives the whole term — which for high-yield bonds is not a safe assumption.

**"A callable bond is just a normal bond that might pay me back early — that's fine."** The call is not a neutral convenience; it is an option *you sold to the issuer*, and it will be exercised against you at the worst time — when rates have fallen and reinvesting is least attractive. That's why callable bonds pay a higher coupon: you're being compensated for handing the issuer a valuable right. Never analyze a callable bond on its yield-to-maturity alone; check the yield-to-call.

## How it shows up in real markets

**The 2008 financial crisis: spreads blow out.** In the depths of the 2008–09 crisis, investment-grade corporate spreads, which had sat around 100–150 bp in the calm of 2006–07, exploded to roughly **600 bp**, and high-yield spreads reached an astonishing **~2,000 bp** at their peak in late 2008. The mechanism is exactly the influence chart in this post: as the economy collapsed and defaults surged, investors demanded enormous compensation to hold corporate risk, so spreads gapped wider and corporate bond *prices* fell hard — even as Treasury yields *fell* in the flight to safety. The lesson burned into a generation of credit investors: in a true crisis, the credit spread, not the risk-free rate, drives your corporate bond's price, and it can move violently.

**The March 2020 COVID shock and the Fed's backstop.** When the pandemic hit, the corporate bond market briefly *froze* — even high-quality bonds became almost impossible to sell, and IG spreads jumped from ~100 bp to over **350 bp** in weeks. What broke the spiral was unprecedented: in March 2020 the **Federal Reserve announced it would buy corporate bonds directly** (through emergency facilities) for the first time in its history. The mere announcement — before it had bought much at all — caused spreads to snap tighter and the new-issue market to reopen, with companies issuing a record wave of bonds to shore up cash. It was a vivid demonstration that the corporate market's liquidity, in a panic, can depend on a central-bank backstop, and that the [central bank's toolkit](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) now reaches into corporate credit.

**Apple's first bond in 2013: a cash-rich company borrows anyway.** In 2013, Apple — sitting on well over \$100 billion of cash — issued **\$17 billion** of bonds, then the largest corporate deal in history. Why would the world's most cash-rich company borrow? Because most of that cash was held *overseas* and would have been taxed heavily to bring home, while issuing cheap debt domestically (at the time, coupons around 2–4%) to fund buybacks and dividends was *cheaper* than repatriating the cash. It's the tax-and-cost-of-capital logic of this post writ large: even a company that doesn't *need* money issues bonds when debt is the cheapest available dollar.

**The callable high-yield refinancing wave.** When the Fed slashed rates to near zero after 2008 and again in 2020, a flood of high-yield issuers *called* their existing bonds and refinanced at dramatically lower coupons — a company that had issued at an 8% coupon in 2009 might call it and reissue at 5% in 2013. Bondholders got their par (plus a small call premium) back and were forced to reinvest at the new, lower yields — the reinvestment risk of the call worked example, playing out across the whole market. It's why experienced high-yield investors obsess over *call protection*: the period after issuance during which a bond *cannot* be called, which preserves their high coupon.

**The 2015–16 energy bust: same rating, very different fates.** When oil crashed from over \$100 to under \$30 a barrel in 2014–16, the *energy sector* of the high-yield market saw a wave of defaults, while non-energy high-yield held up far better. Two bonds rated the same B at the start of 2014 — one from an oil driller, one from a cable company — ended up worlds apart: the driller's bond might have lost half its value or defaulted, the cable bond barely moved. It's the "sector matters" worked example in real life: the rating was identical, but the cash-flow exposure to the oil price was not, and the market's *spread* had been signaling that gap all along.

**The covenant-lite era.** In the long, low-rate decade after 2010, a striking shift occurred in the *loan* market: more and more leveraged loans were issued *covenant-lite* — stripped of the maintenance covenants that traditionally protected lenders, carrying only bond-style incurrence covenants. Borrowers (and the private-equity sponsors behind many of them) had enough negotiating power, in a yield-starved world, to demand looser terms. The consequence, as this post's covenant section predicts, is that lenders lost their early seat at the table: in the next downturn, with no maintenance covenant to trip, troubled companies could drift much further before lenders had any leverage — and recoveries on cov-lite debt have generally come in *lower* than on traditionally-covenanted loans. The protection you negotiate up front determines what you can do when things go wrong.

## When this matters to you, and further reading

If you have a bond fund in your retirement account, a chunk of it is almost certainly corporate bonds — and now you know what you actually own: a portfolio of promises from companies, each paying a spread over Treasuries that breathes with the economy. When you read that "corporate spreads widened" in the financial news, you'll know it means investors got more worried about corporate health, that corporate bond prices fell, and that it's often an early-warning sign of economic stress. When a company you've heard of "issues bonds" or "refinances its debt," you'll understand the underwriting, the concession, and the cost-of-capital logic driving it. And when you compare two bonds, you'll know to look past the coupon to the yield, the spread, the rating *and* sector, the covenants, and any embedded call.

To go deeper from here: for the heavy math of pricing these cash flows, see [bond pricing](/blog/trading/quantitative-finance/bond-pricing) and [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics). For where corporates sit in a whole portfolio, see [corporate credit, investment grade, high yield, and spreads](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads). For the credit-risk side — default probability, recovery, and how spreads price it — continue with [credit risk](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back), [credit spreads](/blog/trading/fixed-income/credit-spreads-pricing-the-probability-of-default), and [seniority and recovery](/blog/trading/fixed-income/seniority-recovery-and-the-capital-structure). And to see how the agencies that grade all of this actually operate, read [credit rating agencies](/blog/trading/finance/credit-rating-agencies-moodys-sp-fitch). The corporate bond is the bridge between a company's balance sheet and the price of money — and once you can read it, a huge amount of the financial world snaps into focus.

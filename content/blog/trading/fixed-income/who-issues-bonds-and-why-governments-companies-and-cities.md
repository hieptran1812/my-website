---
title: "Who issues bonds, and why: governments, companies, and cities"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-to-deep tour of who borrows in the bond market and why: the US Treasury funding the deficit, agencies and GSEs, corporations choosing debt over equity, and cities issuing tax-free municipal bonds — plus how a new bond is actually sold."
tags: ["fixed-income", "bonds", "bond-issuers", "us-treasury", "corporate-bonds", "municipal-bonds", "capital-structure", "debt-vs-equity", "primary-market", "bond-auctions"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Almost every bond on earth is issued by one of four borrowers — the sovereign government, government agencies, corporations, or states and cities — and each one borrows for a different reason that explains the bond's risk and its yield.
>
> - The **US Treasury** is the biggest issuer on the planet. It sells bonds to fund the federal deficit — the gap between what the government spends and what it collects in taxes. When the deficit jumps, Treasury issuance jumps with it.
> - **Corporations** issue bonds because debt is usually *cheaper* than equity: interest is tax-deductible, lenders accept a lower return than shareholders, and borrowing doesn't dilute the owners. The catch is that interest is a hard promise — miss it and you can go bankrupt.
> - **Municipalities** (states, cities, school districts) issue bonds to build infrastructure, and in the US the interest is often **exempt from federal tax**, which lets them borrow more cheaply than a company of the same credit quality.
> - A bond is born in the **primary market** — Treasuries by auction, corporates and munis by an underwritten deal run by a bank — and then trades for the rest of its life in the **secondary market**.
> - The thread that runs through the whole series: the price a borrower pays to issue a bond *is* an interest rate, and that rate ripples out into every mortgage, car loan, and stock valuation in the economy.

Here is a question almost nobody asks but everybody should: when a country, a company, or a city needs a giant pile of money — more than it has, more than any single bank wants to lend — where does that money actually come from?

The answer is the bond market. A *bond* is just a tradable loan: the borrower (the **issuer**) sells you a piece of paper that promises to pay you interest for a number of years and then hand back the amount you lent. We covered that contract line by line in the previous post on [the anatomy of a bond](/blog/trading/fixed-income/anatomy-of-a-bond-par-coupon-maturity-issuer). What we have not yet done is ask the more human question: *who* signs that paper, and *why* would they choose to borrow this way instead of going to a bank or selling a stake in themselves?

That "who and why" turns out to organize the entire bond market. The diagram below is the mental model for the whole post: nearly every bond in existence is issued by one of four families of borrower — the sovereign government's Treasury, government agencies, corporations, and municipalities — and each family borrows for a reason that tells you almost everything about how risky its bonds are and what yield they pay.

![A tree diagram splitting the bond universe into four issuer families: the US Treasury, agencies and GSEs, corporations, and municipalities, each with examples and what backs the promise to repay](/imgs/blogs/who-issues-bonds-and-why-governments-companies-and-cities-1.png)

We will meet a recurring character along the way: a fictional company called **Northwind Corp**, a mid-sized manufacturer that needs \$100 million to build a new factory. Watching Northwind decide *how* to raise that money — sell stock, or sell bonds — will make the single most important idea in corporate finance concrete. And we will keep one real benchmark in view the whole time: the US Treasury, the borrower against which every other borrower on earth is measured.

## Foundations: the words you need before we start

Before we can talk about issuers, we need a small, shared vocabulary. A practitioner can skim this; if any of these terms is new, read it slowly, because everything later is built on them.

**A bond is a loan you can sell.** When you buy a newly issued bond, you are lending money to the issuer. In return you get a contract that specifies three things: the **par value** (also called face value — the amount you get back at the end, conventionally \$1,000 per bond), the **coupon** (the interest, quoted as an annual percentage of par — a 4% coupon on a \$1,000 bond pays \$40 a year), and the **maturity** (the date the loan ends and par is repaid). The thing that makes a bond different from a bank loan is that you can sell it to someone else any day you like. That tradability is what creates a *market*.

**Yield is the return, not the coupon.** The coupon is fixed forever the day the bond is issued. The **yield** is what the bond actually earns you given the price you paid for it. Because bond prices move up and down after issuance, the yield moves too — and it moves *opposite* to price. Pay more than par and your yield falls below the coupon; pay less than par and your yield rises above it. We devote a whole later post to [the price–yield seesaw](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds); for now, just hold onto: **yield = the borrower's cost and the lender's return, expressed as an annual percentage.**

**A basis point is one hundredth of a percent.** Bond people talk in *basis points* (bps) because the differences that matter are small. 0.25% is 25 bps. A "spread of 150 basis points" means 1.5 percentage points. When a bond yields 1.5% more than a Treasury, we say it trades "150 over."

**Credit risk is the chance you don't get paid back.** A US Treasury bond is treated as having essentially zero credit risk — the government can always print the dollars it owes — so its yield is called the **risk-free rate**. Every other borrower is riskier, so it must pay *more* than the Treasury to attract lenders. That extra is the **credit spread**, and it is the market's price for the chance the borrower defaults. We will lean on this idea constantly: a bond's yield is the risk-free Treasury yield *plus* a spread that grows with the issuer's riskiness.

**Equity vs debt.** A company can raise money two ways. It can sell **equity** — shares of ownership, a permanent claim on its profits, with no promise to ever pay anything back. Or it can sell **debt** — a bond or loan, a temporary claim with a fixed schedule of interest and a repayment date. The choice between them is the heart of this post, and it is the heart of corporate finance.

**The primary vs secondary market.** When a bond is first sold by the issuer to raise money, that happens in the **primary market** — the issuer gets cash, the first buyers get the bonds. After that, those bonds change hands among investors in the **secondary market**, where the issuer is no longer involved and gets none of the proceeds. New issuance happens in the primary market; everything else is secondary trading.

With those six ideas in hand, we can meet the four families of borrower.

## The US Treasury: the biggest borrower on earth

Start with the giant. The single largest issuer of bonds in the world is the government of the United States, borrowing through its **Treasury department**. As of 2024, US federal debt held by the public is roughly \$28 trillion, and the Treasury market — the bills, notes, and bonds the government has issued — is the deepest, most liquid securities market that has ever existed. (These are illustrative round numbers; the exact figure climbs every day. Always check the as-of date for live debt numbers.)

Why does the government borrow so much? Because it spends more than it taxes. In any year the government collects revenue (mostly income and payroll taxes) and spends on the military, Social Security, Medicare, interest, and everything else. When spending exceeds revenue, the difference is the **deficit**, and the deficit has to be financed somehow. The government does not have a giant savings account. It borrows the gap — by selling bonds. The accumulated total of all past deficits is the **national debt**.

This is the most direct version of the influence thread that runs through this whole series. The deficit is a policy choice made in Congress; the *funding* of that deficit is a mechanical certainty made in the bond market. Every dollar of deficit becomes a dollar of new Treasury bonds that somebody, somewhere, has to be persuaded to buy.

### The three kinds of Treasury

The Treasury slices its borrowing by maturity:

- **Treasury bills (T-bills)** mature in one year or less. They pay no coupon; you buy them below \$1,000 and get \$1,000 back, and the difference is your return. These are the short-term cash-management tool.
- **Treasury notes (T-notes)** mature in 2 to 10 years and pay a coupon every six months. The 10-year note is the single most-watched interest rate in the world — it anchors mortgage rates and corporate borrowing costs alike.
- **Treasury bonds (T-bonds)** mature in 20 or 30 years, also paying semiannual coupons. These are the long-term debt.

There is also a fourth, smaller flavor — **TIPS** (Treasury Inflation-Protected Securities), whose principal grows with inflation — and **floating-rate notes**, but we will save those for [the Treasury sector deep-dive](/blog/trading/fixed-income/us-treasuries-the-risk-free-benchmark-of-the-world).

#### Worked example: financing one year of deficit

Suppose in a given fiscal year the federal government spends \$6.5 trillion and collects \$5.0 trillion in revenue. The deficit is:

$$\text{Deficit} = \$6.5\text{T} - \$5.0\text{T} = \$1.5\text{ trillion}$$

That \$1.5 trillion gap does not vanish. The Treasury must raise it by selling bonds. If it sells the whole amount as 10-year notes carrying a 4% coupon, the government has just committed to pay:

$$\$1.5\text{T} \times 4\% = \$60\text{ billion per year in interest}$$

every year for ten years, on top of interest on all the debt it already owes. That \$60 billion is real money diverted from other spending — and it is precisely why a higher interest rate makes deficits more painful: the same borrowing costs more to carry.

*The intuition: a deficit is not just spending today — it is a stream of interest payments the bond market will collect for years, and the size of that stream depends on the yield the market demands.*

This is the cleanest place to see how issuance and yields relate. The figure below shows the relationship illustratively: the annual federal deficit and the net new Treasury borrowing that funds it move together, year by year, and both exploded in the 2020 pandemic year when the government ran enormous deficits to support the economy.

![A line chart over fiscal years 2015 to 2024 showing the federal deficit and net Treasury issuance rising and falling together, both spiking sharply in the 2020 pandemic year and staying elevated afterward](/imgs/blogs/who-issues-bonds-and-why-governments-companies-and-cities-4.png)

The two lines track each other because they are two sides of the same coin: the deficit *is* the amount of new borrowing required, so net issuance has to rise to meet it. (The numbers are illustrative — drawn to be realistic about the shape and the 2020 spike, not exact to the dollar.) And here is the second-order effect that the macro series develops in depth: when the government floods the market with new bonds, it is selling a *lot* of supply, and to clear that supply it sometimes has to offer a higher yield — which pushes up borrowing costs for everyone. That mechanism, deficit → supply → yields, is the bridge from fiscal policy to your mortgage rate; the macro post on [deficits, debt, and bond supply](/blog/trading/macro-trading/deficits-debt-bond-supply-why-issuance-moves-yields) is the place to go deep on it.

### Why Treasuries are the safest bond there is

A US Treasury is treated as the **risk-free** asset for one structural reason: the debt is denominated in dollars, and the United States controls the printing of dollars. It cannot be *forced* to default on a dollar obligation, because it can always create the dollars. (Whether it *should* — and what inflation that might cause — is a different and important question, but the *mechanical* default risk is essentially nil.) That is why every other bond is priced as "the Treasury yield plus a spread." The Treasury is the floor of the whole system: the price of money with no credit risk attached.

This is why the broader series calls bonds "the price of money" — and why the [risk-free anchor](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration) is the reference point for valuing every other asset, from corporate bonds to stocks.

### Who holds all that Treasury debt?

It's natural to wonder who on earth is willing to lend the US government tens of trillions of dollars. The answer is "almost everyone," and the breakdown matters because it tells you how stable that funding is. Treasury debt is held by, roughly: foreign governments and central banks (Japan and China are the two largest foreign holders, parking their export earnings in the world's safest asset); the **Federal Reserve** itself (which buys Treasuries when it conducts quantitative easing — more on that in a moment); US pension funds, insurers, banks, and money-market funds; and households, directly or through their mutual funds and retirement accounts.

This distribution is itself part of the influence thread, because it means a US deficit is funded by *global* savings. When a Japanese insurer or a Saudi sovereign-wealth fund buys US Treasuries, it is choosing to lend to America rather than to its own government or to invest elsewhere — and that demand keeps US borrowing costs lower than they would otherwise be. The flip side is a vulnerability: if foreign buyers ever pulled back sharply, the Treasury would have to offer higher yields to attract domestic buyers to fill the gap, raising borrowing costs across the whole economy. This is the "exorbitant privilege" of the dollar — and its quiet fragility.

The **Federal Reserve** is a special case worth a sentence on its own. During crises (2008, 2020) the Fed bought enormous quantities of Treasuries, creating new money to do so — this is **quantitative easing (QE)**. By becoming a giant buyer, the Fed pushes Treasury prices up and yields down, deliberately lowering borrowing costs to stimulate the economy. When it reverses — letting bonds mature without replacing them, or selling them — that is **quantitative tightening (QT)**, which removes a buyer and tends to push yields up. The point for now: the largest issuer (the Treasury) and one of the largest buyers (the Fed) are both arms of the same government, and the interplay between them is the [central-bank toolkit](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) that sets the price of money for everyone.

## Agencies and GSEs: the half-step below the Treasury

Sitting just below the Treasury in safety is a set of issuers most people have never consciously thought about but whose bonds quietly fund a huge share of American life: **federal agencies** and **government-sponsored enterprises** (GSEs).

The big names are **Fannie Mae** and **Freddie Mac** (the two GSEs that buy mortgages from banks, bundle them, and guarantee them) and **Ginnie Mae** (an actual government agency that guarantees mortgages backed by federal programs). When you get a 30-year fixed mortgage in the US, there is a very good chance your loan ends up inside a bond issued or guaranteed by one of these entities.

Why do they issue bonds? To fund the mortgage market. Here is the chain: a bank lends you money to buy a house. The bank does not want to wait 30 years to get it back, so it sells your loan to Fannie or Freddie. Fannie and Freddie pool thousands of such loans together and sell **mortgage-backed securities (MBS)** — bonds whose coupons come from all those homeowners' monthly payments. The money raised by selling the MBS flows back to the banks so they can make more loans. The whole machine recycles capital so that home lending never runs dry.

These bonds carry an **implicit government backing**. Legally, the government does not *promise* to stand behind Fannie and Freddie debt the way it stands behind a Treasury — but the market has long assumed it would never let them fail, and in 2008 that assumption was proven right when the government took them into "conservatorship" and made their bondholders whole. So agency and GSE bonds yield a little *more* than Treasuries (because the backing is implicit, not explicit) but a lot *less* than corporate bonds. They are the half-step below the risk-free floor. We give MBS their own treatment — including the strange way they behave when rates fall — in the [mortgage-backed securities post](/blog/trading/fixed-income/mortgage-backed-securities-bonds-with-negative-convexity).

There is a subtle point here about *who is really the issuer*. With a Treasury, the borrower (the government) and the entity whose creditworthiness you're relying on are the same. With an MBS, the picture is layered: the ultimate borrowers are thousands of individual homeowners, the GSE bundles and guarantees their loans, and *you* — the bond buyer — are lending to the pool while relying on the GSE's guarantee against the homeowners defaulting. This is your first encounter with **securitization**, the financial alchemy of turning a heap of small, illiquid loans into a single tradable bond. It is one of the most consequential inventions in modern finance, and it gets a full deep-dive later in the series; for now, just notice that "the issuer" is sometimes a pass-through structure standing between you and a crowd of underlying borrowers.

#### Worked example: how the mortgage machine recycles capital

Walk through one cycle of the machine with round numbers. A bank has \$300 million of deposits it can lend. It writes 1,000 mortgages of \$300,000 each — \$300 million lent, and now the bank is fully invested with no cash to make new loans.

The bank sells those 1,000 mortgages to Freddie Mac for \$300 million. Freddie pools them and issues a \$300 million MBS to investors — pension funds, insurers, foreign central banks — who want the steady stream of homeowner payments. Freddie passes the \$300 million from the MBS buyers back to the bank.

Now the bank has \$300 million of fresh cash again. It writes *another* 1,000 mortgages. The same \$300 million of original deposits has now funded **2,000 mortgages**, and it can keep going. Without securitization, the bank's lending would stop at \$300 million; with it, the capital cycles through again and again.

*The intuition: agency bonds exist to keep capital moving — by turning a bank's frozen 30-year loans back into cash, they let the same dollars fund home after home, which is why agency debt is one of the largest issuer families of all.*

## Corporations: why a company borrows instead of selling itself

Now we arrive at the most interesting issuer, because here the borrower has a genuine *choice*. A government has to fund its deficit; it can't sell shares of America. A company, by contrast, can raise money two completely different ways — and the decision between them is one of the deepest in all of finance.

Meet **Northwind Corp**. It is a profitable mid-sized manufacturer, and it has spotted a chance to build a new factory that will cost \$100 million and, management believes, earn a healthy return. Northwind does not have \$100 million in the bank. It needs to raise it. It has two options.

**Option A — sell equity.** Northwind can issue \$100 million of new stock. New investors hand over \$100 million and receive shares — slices of ownership in Northwind. Those shareholders now own a piece of *all future profits*, forever. There is no repayment date and no promise of any particular return; if Northwind thrives, the shareholders get rich, and if it struggles, they simply earn less. The cost of equity is high precisely *because* the payoff is uncertain — shareholders demand a fat expected return (say 9–11%) to compensate for bearing that risk. And there is a second cost: the original owners are now **diluted** — they own a smaller fraction of the company and share control with the new investors.

**Option B — sell debt.** Alternatively, Northwind can issue \$100 million of bonds. Lenders hand over \$100 million and receive a contract: a fixed coupon (say 5% a year, so \$5 million annually) and a promise to repay the \$100 million principal at maturity. Crucially, the bondholders own *none* of the company. They have no vote, no share of the upside, no permanent claim. If Northwind triples its profits, the bondholders still get their \$5 million a year and not a penny more. The original owners keep 100% of the ownership.

The figure below lays the two choices side by side.

![A side-by-side comparison of Northwind Corp raising 100 million by selling equity versus by selling debt, contrasting dilution, cost, tax treatment, and repayment obligation](/imgs/blogs/who-issues-bonds-and-why-governments-companies-and-cities-2.png)

So why would a healthy company so often choose debt? Three reasons, each worth understanding on its own.

### Reason 1: debt is usually cheaper than equity

Lenders accept a *lower* return than shareholders, for a simple reason: they take less risk. A bondholder gets paid before shareholders in good times (a fixed coupon) and ahead of them in bad times (bondholders are repaid before shareholders if the company is liquidated). Because the bondholder's outcome is more certain, they are willing to accept a smaller expected return. Shareholders, who get paid last and bear the full swing of the business, demand more.

That gap — the cost of equity above the cost of debt — is one of the most reliable facts in finance, and it is why "lever up with cheap debt" is a perennial temptation.

#### Worked example: Northwind's cost of capital

Suppose investors would demand a **9% return** to own Northwind's stock, but Northwind can issue bonds at a **5% coupon**. Funding the \$100 million factory:

- With equity, the implicit cost is \$100M × 9% = **\$9 million per year** of expected return that the owners are effectively giving up to new shareholders.
- With debt, the cost is \$100M × 5% = **\$5 million per year** in coupon.

On the surface, debt looks \$4 million a year cheaper. *The intuition: because lenders are paid before owners and bear less risk, they accept a lower return — so borrowing is usually the cheaper way to fund a project, as long as the project earns more than the coupon.*

### Reason 2: the tax shield makes debt cheaper still

Here is the part that surprises people. In most tax systems, including the US, the **interest a company pays on its debt is tax-deductible** — it comes out of profit *before* tax is calculated. Dividends paid to shareholders are not deductible; they come out of *after-tax* profit. This asymmetry makes debt cheaper than its headline coupon suggests.

#### Worked example: the tax shield in dollars

Northwind pays a 5% coupon on \$100 million, so \$5 million of interest a year. Suppose Northwind's corporate tax rate is **21%**. Because that \$5 million of interest is deductible, it reduces Northwind's taxable profit by \$5 million, which saves it:

$$\$5\text{M} \times 21\% = \$1.05\text{ million in taxes}$$

So the *real*, after-tax cost of the debt is not \$5 million — it is \$5M − \$1.05M = **\$3.95 million**, an after-tax rate of about **3.95%**. The formula is worth memorizing:

$$\text{After-tax cost of debt} = \text{coupon rate} \times (1 - \text{tax rate})$$

where the *coupon rate* is the interest the firm pays and the *tax rate* is its marginal corporate tax rate.

*The intuition: the government effectively subsidizes corporate borrowing by letting firms deduct interest — so the true cost of debt is meaningfully below the coupon, widening its advantage over equity.*

The chart below puts the two costs side by side across different rate environments. The key visual: the after-tax cost of debt stays below the cost of equity in every environment — but the *gap* between them shrinks as rates rise, because debt gets more expensive while equity costs move less.

![A line chart comparing the cost of equity and the after-tax cost of debt across low, normal, and high interest-rate environments, with debt always below equity but the gap narrowing as rates climb](/imgs/blogs/who-issues-bonds-and-why-governments-companies-and-cities-6.png)

This is exactly why the debt-vs-equity decision is *rate-dependent*. When rates are near zero, debt is almost free and the temptation to borrow is overwhelming — which is a big part of why the 2010s saw a corporate-borrowing boom. When rates rise sharply, as they did in 2022–2023, the cheap-debt advantage narrows and some companies discover that the borrowing they took on at 2% is brutal to refinance at 6%.

### Reason 3: debt preserves ownership and control

The third reason is not about money at all — it's about *power*. When Northwind's founders sell equity, they give away a permanent slice of the company and invite new voices into the boardroom. When they sell debt, they keep every share and every vote. The bondholders are creditors, not partners: as long as the coupons get paid, they have no say in how Northwind is run. For a founder who believes in the business and doesn't want to share the upside or the control, debt is enormously attractive.

### The catch: debt is a hard promise

None of this means debt is free or safe. The coupon is a *contractual obligation*. If Northwind's factory flops and cash gets tight, it can quietly cut or skip its *dividend* to shareholders with no legal consequence — but it cannot skip its *bond coupon*. Miss a coupon, and the company is in **default**, which can trigger bankruptcy, hand control to the creditors, and wipe out the original owners entirely. Equity is patient money; debt is impatient money. The more debt a company piles on (the more **leverage** it takes), the more its fixed obligations grow, and the smaller the cushion before a bad year turns into a crisis. This is the tradeoff at the center of [the capital structure](/blog/trading/fixed-income/seniority-recovery-and-the-capital-structure), where we look at who gets paid first when a company actually fails.

#### Worked example: when leverage helps and when it hurts

Northwind funds the \$100M factory with 5% debt, costing \$5 million a year in coupon. Consider two scenarios for what the factory earns:

- **Good year:** the factory earns \$12 million. After paying \$5 million in coupon, \$7 million is left over for the owners — and the owners put in *none* of their own new money. Their return on the project is effectively infinite leverage on a \$7 million gain. Borrowing magnified the win.
- **Bad year:** the factory earns only \$3 million. The coupon is still \$5 million. Northwind must find \$2 million from elsewhere in the business just to avoid default. If it can't, the bondholders can force the issue. Borrowing magnified the loss.

*The intuition: debt is a lever — it amplifies returns in both directions, which is why a profitable, stable company can safely carry debt that would sink a volatile one.*

### How much debt is too much: the optimal capital structure

If debt is cheaper than equity, why doesn't every company fund itself entirely with debt? Because the *cheapness* of debt is not constant — it rises with the amount of debt. The first slug of borrowing is safe and cheap; lenders barely blink. But as a company piles on more and more leverage, lenders get nervous: the fixed obligations grow, the cushion shrinks, and the probability of default climbs. So they demand a higher and higher yield for each additional dollar lent. Past some point, the spread the company has to pay rises so fast that more debt actually makes its overall cost of capital go *up*, not down.

This produces a U-shaped tradeoff. Too little debt and the company is leaving the cheap tax-shielded financing on the table. Too much debt and the rising default risk makes everything — both its debt and its equity — more expensive, because both lenders and shareholders now price in the danger of bankruptcy. Somewhere in between is a sweet spot, the **optimal capital structure**, where the firm has borrowed enough to capture the tax benefit but not so much that financial distress becomes likely. There is no exact formula for it; in practice it depends on how stable and predictable the company's cash flows are.

That is why a steady utility — which collects predictable, regulated revenue every month — can comfortably carry a lot of debt, while a volatile biotech with no reliable revenue should carry almost none. The right amount of leverage is a function of how reliably the company can cover its coupons. Rating agencies exist precisely to grade this — to tell the market how likely a given issuer is to keep paying — and we devote a full post to [how Moody's, S&P, and Fitch grade debt](/blog/trading/fixed-income/bond-ratings-how-moodys-sp-and-fitch-grade-debt).

#### Worked example: when more debt raises the cost of capital

Northwind currently has very little debt and can borrow at a 5% coupon. Suppose it considers loading up — borrowing not \$100 million but \$500 million, doubling its total debt load relative to its earnings. Now lenders see a company whose fixed coupon obligations have ballooned, and they demand **7%** instead of 5% to compensate for the higher default risk.

The cost of *all* its new debt is now 7%, not 5%. And because the company is now visibly riskier, its shareholders also demand a higher return — say the cost of equity rises from 9% to 11%, because equity holders, who get paid last, are now closer to being wiped out in a bad year. Both legs of financing got more expensive. The extra leverage didn't lower Northwind's cost of capital; past the sweet spot, it raised it.

*The intuition: debt is cheap only in moderation — borrow past the point where default becomes a real worry and the rising risk premium makes every dollar of financing, debt and equity alike, more expensive.*

### A wrinkle: callable and other embedded options

Real corporate bonds often come with extra features baked into the contract. The most common is a **call provision**: the right for the *issuer* to repay the bond early, before maturity, at a set price. Why would a company want that? Because if interest rates fall after it issues, it can call the old high-coupon bonds and reissue new ones at the new, lower rate — refinancing, exactly like a homeowner refinancing a mortgage when rates drop.

That option is valuable to the company and costly to the bondholder, who loses the high coupon just when it has become most attractive. So callable bonds must pay a *higher* yield to compensate. The mirror image is a **puttable bond**, which gives the *bondholder* the right to sell it back to the issuer early — valuable to the holder, so it pays a *lower* yield. These embedded options are why "the yield" on a corporate bond is more complicated than on a plain Treasury, and why later in the series we have to distinguish yield-to-maturity from [yield-to-call and yield-to-worst](/blog/trading/fixed-income/the-many-yields-current-yield-ytm-and-yield-to-call). For now, the lesson is simply that an issuer can shape its bonds — adding features that shift risk between itself and its lenders, and paying or saving yield accordingly.

## Municipalities: cities, states, and the tax-free advantage

The fourth family is **municipalities** — states, counties, cities, school districts, transit authorities, water districts, and the like. In the US these are called **munis**, and they issue bonds for one main reason: to fund **public infrastructure** that costs far more up front than current tax revenue can cover. A new bridge, a school, a water-treatment plant, a light-rail line — these are multi-decade assets, and it makes sense to spread their cost over the decades that will use them, by borrowing now and repaying over time.

There is a fairness logic to borrowing for infrastructure, beyond just affordability. A bridge built today will serve drivers for fifty years. If the city paid for it entirely out of this year's taxes, today's taxpayers would bear the whole cost of an asset that future taxpayers will enjoy. By borrowing and repaying over decades, the city spreads the cost across the same generations that benefit — a principle called **intergenerational equity**. Debt, used this way, is not a burden dumped on the future; it is a way of matching who pays to who uses. (The danger, of course, is borrowing for things that *don't* last — funding day-to-day operating costs with long-term debt, which leaves future taxpayers paying for services already consumed. That is how cities get into trouble.)

What makes munis special is a tax quirk that turns out to matter enormously. In the US, the interest paid on most municipal bonds is **exempt from federal income tax** (and often from state tax too, for residents of the issuing state). For a high-income investor, that exemption is worth a lot, and it lets cities borrow more cheaply than their credit quality alone would justify.

#### Worked example: the tax-equivalent yield

Suppose a city issues a municipal bond yielding a **tax-free 3.5%**, and you, the investor, are in the **35% federal tax bracket**. Compare it to a taxable corporate bond. To match the muni's after-tax return, the taxable bond would have to yield:

$$\text{Tax-equivalent yield} = \frac{\text{muni yield}}{1 - \text{tax rate}} = \frac{3.5\%}{1 - 0.35} = \frac{3.5\%}{0.65} \approx 5.38\%$$

where the *muni yield* is the bond's tax-free coupon yield and the *tax rate* is your marginal income-tax rate.

So a 3.5% tax-free muni is worth as much to you, after tax, as a **5.38% taxable bond**. *The intuition: the tax exemption means a muni can pay a lower headline yield than a corporate and still leave a high-bracket investor better off — which is precisely why cities can borrow cheaply.*

That advantage is also why munis are a tax-driven asset: they make sense mostly for investors with high tax bills, and the exemption is, in effect, a federal subsidy for local infrastructure. We go deeper on the math and the market in the dedicated [municipal bonds post](/blog/trading/fixed-income/municipal-bonds-tax-free-income-and-the-muni-market).

### Two flavors of muni: general obligation vs revenue

Within munis there is a crucial distinction that decides how risky the bond is: what, exactly, is promised to repay it?

A **general obligation (GO) bond** is backed by the issuer's **full taxing power**. If a city issues a GO bond to build schools, it is pledging that it will, if necessary, raise property taxes to make the payments. GO bonds are usually voter-approved and represent the broadest, strongest promise a municipality can make — so they yield less.

A **revenue bond** is backed *only* by the cash flow of the specific project it funded. A toll-road authority issues a revenue bond to build the road, and it pledges only the *tolls* to repay it. If the tolls fall short — too few cars, an economic slump — the bondholders take the hit, because they have *no* claim on the city's general taxes. Revenue bonds are therefore riskier and yield more.

The figure below contrasts the two.

![A side-by-side comparison of a general-obligation municipal bond backed by the issuer's full taxing power versus a revenue bond backed only by a specific project's cash flow, showing why revenue bonds carry more risk and yield more](/imgs/blogs/who-issues-bonds-and-why-governments-companies-and-cities-7.png)

#### Worked example: GO vs revenue yields

Imagine a single city issues two bonds in the same week. A 10-year **GO bond** to build schools yields **3.2%**, backed by the city's full taxing power. A 10-year **revenue bond** to build a new parking garage yields **3.9%**, backed only by the garage's parking fees.

The 70-basis-point difference (3.9% − 3.2% = 0.70%) is the market's price for the extra risk: if the garage sits half-empty, the revenue bondholders are exposed in a way the GO holders are not. On a \$10,000 investment, that's:

- GO bond: \$10,000 × 3.2% = **\$320 a year**, with the safety of the whole city's tax base behind it.
- Revenue bond: \$10,000 × 3.9% = **\$390 a year**, but riding on one parking garage's success.

*The intuition: even from the same issuer, the bond's backing — broad taxing power versus a single project's revenue — decides its risk and therefore its yield.*

## Putting the families in proportion

Now that we've met all four issuer families, it helps to see how big each one is. The US bond market is roughly \$55 trillion in total — far larger than the US stock market. The figure below breaks it down by issuer with illustrative sizes.

![A grid showing the US bond market by issuer family with illustrative sizes: Treasuries around 27 trillion, mortgage and agency debt around 12 trillion, corporate bonds around 11 trillion, and municipal bonds around 4 trillion, with what backs each](/imgs/blogs/who-issues-bonds-and-why-governments-companies-and-cities-3.png)

Two things jump out. First, **Treasuries dominate** — the single biggest slice, and growing fastest as deficits persist. Second, **mortgage-related debt is enormous** — bigger than the entire corporate bond market — which is why the health of the housing-finance machine matters so much to the whole financial system. Munis, though they loom large in the public imagination as "safe income," are actually the smallest of the four families. (All figures are round and illustrative; the exact numbers shift constantly and you should check the as-of date for any live total.)

This proportionality is itself a lesson in the influence thread. The Treasury market is the biggest because the US government is the biggest borrower; and because it is the biggest and safest, its yield becomes the reference rate that prices everything else. The size *is* the power.

## How a new bond is actually sold: the primary market

We've covered who issues bonds and why. The last piece is *how* — the mechanics of getting a brand-new bond from the issuer's decision to a buyer's portfolio. This happens in the **primary market**, and it works differently for Treasuries than for corporates and munis.

The figure below traces the full pipeline.

![A pipeline showing a bond's journey from the issuer's decision to borrow, through the primary market (auction for Treasuries, underwritten deal for corporates and munis), price discovery, cash to the issuer, and finally trading in the secondary market](/imgs/blogs/who-issues-bonds-and-why-governments-companies-and-cities-5.png)

### Treasuries: the auction

The Treasury raises money by **auction**. On a regular, published schedule, it announces it will sell, say, \$40 billion of 10-year notes. A select group of large banks called **primary dealers** are *required* to bid in every auction; other investors can bid too. Bidders submit the yield they are willing to accept, and the Treasury fills the lowest-yield (highest-price) bids first, working up until the whole amount is sold. The yield at which the last bond clears becomes the yield for the entire issue — everyone pays the same clearing price. This is a **single-price (Dutch) auction**, and its sheer scale and regularity are part of why the Treasury market is so deep and liquid.

The auction is also a real-time referendum on the government's borrowing. A "strong" auction (lots of demand, low clearing yield) signals that investors are happy to fund the deficit cheaply. A "weak" auction (tepid demand, higher yield, the dealers stuck holding the leftovers) is a warning sign that the market is demanding more compensation to absorb all that supply — the deficit-to-yields link in action, watched obsessively by traders.

### Corporates and munis: underwritten syndication

A company like Northwind cannot run a Treasury-style auction; it issues too infrequently and is too small to command a standing market. Instead, it hires an **underwriter** — an investment bank — to manage the whole process. The underwriter does several jobs at once:

1. **Structures the deal** — advises Northwind on the maturity, the size, and roughly what coupon the market will demand given Northwind's credit quality (its yield will be the Treasury yield *plus* a spread reflecting its default risk).
2. **Markets the bonds** — takes the deal to its network of investors (pension funds, insurers, bond funds), gauges demand, and **builds a book** of orders.
3. **Sets the price** — once it knows how much demand there is, it sets the final coupon. Strong demand lets it set a lower coupon (cheaper for Northwind); weak demand forces a higher one.
4. **Takes on risk** — in a *firm-commitment* underwriting, the bank actually buys the entire issue from Northwind at an agreed price and then resells it to investors, pocketing the spread and bearing the risk that it can't resell at a profit.

#### Worked example: the underwriting spread

Northwind wants to raise \$100 million. Its underwriter agrees to buy the whole \$100 million issue at **99% of par** (\$99 million) and resell it to investors at **par** (\$100 million). The 1% difference — the **underwriting spread** — is the bank's fee:

$$\$100\text{M} \times 1\% = \$1\text{ million to the underwriter}$$

So Northwind actually receives \$99 million in cash and owes the bondholders the full \$100 million at maturity plus all the coupons in between. The \$1 million is the cost of having a bank find buyers and guarantee the sale.

*The intuition: an underwriter is paid to convert the issuer's promise into cash today and to absorb the risk that buyers don't show up — a service governments don't need because their auctions are deep enough to clear themselves.*

For big, frequent issuers like a large municipality or a blue-chip corporation, several banks may form a **syndicate** to spread the risk and pool their investor networks — hence "underwritten syndication." Either way, the result is the same: the bonds get placed, the issuer gets its cash, and the bonds begin their long life in the secondary market.

### The secondary market: where bonds live the rest of their lives

Once the primary sale is done, the issuer is out of the picture. From then on, the bonds trade between investors in the **secondary market** — and here's a fact that surprises newcomers: most bonds do *not* trade on an exchange like stocks do. They trade **over-the-counter (OTC)**, dealer to dealer, by phone and electronic message. A pension fund that wants to sell a Northwind bond calls a dealer, who quotes a price, and the trade is done bilaterally. Treasuries trade in a hyper-liquid OTC market with razor-thin spreads; an obscure corporate or muni bond might not trade for weeks.

The secondary market is where everything we'll cover in the rest of the series plays out: as interest rates rise and fall, as Northwind's credit improves or deteriorates, the *price* of its bonds moves up and down in secondary trading — even though the coupon and the maturity never change. That price movement is the entire subject of [the price–yield seesaw](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds) and the duration posts that follow it. The microstructure of how that trading actually works — dealers, liquidity, the on-the-run/off-the-run split — gets its own deep-dive later in the series.

The secondary market also matters *back to the issuer*, even though the issuer isn't a party to those trades. Here's why: when Northwind wants to issue *new* bonds next year, the underwriter will price them off where Northwind's *existing* bonds are trading in the secondary market. If investors have soured on Northwind and its old bonds now yield 8% in secondary trading, the new bonds will have to offer something near 8% too — the secondary market sets the benchmark for the next primary deal. So a company that lets its credit deteriorate pays for it twice: once when its outstanding bonds fall in price, and again when it has to borrow fresh money at the new, higher rate. The market's ongoing verdict on an issuer, expressed every day in secondary prices, is the cost of its next loan.

This is also where the influence thread closes its loop. The Treasury's secondary-market yield — the famous 10-year — is the reference rate off which corporate and muni new issues are priced. So when the 10-year yield rises, *every* issuer's new borrowing gets more expensive, from the federal government down to the smallest city. The price discovered in the deepest, most liquid secondary market on earth flows outward into the borrowing cost of everyone else. That is what it means to say bonds are the price of money.

## Common misconceptions

**"Government bonds and corporate bonds are basically the same thing."** No — and the difference is the whole point of the credit spread. A US Treasury has essentially zero default risk because the government controls the currency it owes in. A corporate bond can default if the company runs out of cash. That is why a corporate always yields *more* than a Treasury of the same maturity: the extra yield is the market's price for the chance you don't get repaid. Treating them as interchangeable is how investors get blindsided when a company they lent to goes bankrupt.

**"Companies issue bonds because they're short of cash or in trouble."** Usually the opposite. The healthiest, most creditworthy companies are the *best* bond issuers, because lenders will fund them cheaply. Apple, with hundreds of billions in cash, has issued tens of billions in bonds — not because it needs the money but because borrowing at a low after-tax rate and keeping its cash invested elsewhere is good financial engineering. Issuing debt is a sign of access to capital, not desperation.

**"Municipal bonds are risk-free because the government issues them."** Munis are *not* backed by the federal government — they are backed by the issuing city or authority, which absolutely *can* default. Detroit filed for bankruptcy in 2013; Puerto Rico defaulted on tens of billions of muni debt in 2016. A revenue bond can fail entirely if its project doesn't generate the expected cash. The federal *tax exemption* makes munis attractive, but it is not a federal *guarantee*.

**"The interest rate on a bond is set by the issuer."** The issuer proposes; the market disposes. A Treasury *auction* lets buyers set the yield directly. A corporate underwriting sets the coupon based on the demand the bank can find — strong demand means a lower coupon, weak demand a higher one. The issuer can only choose the maturity and size; the *price of the money* is set by what lenders will accept, which is the risk-free Treasury rate plus a spread the market decides.

**"Buying a bond at issuance means lending to the issuer; buying one later means the same."** Only the *first* sale (in the primary market) actually funds the issuer. When you buy a 5-year-old Northwind bond in the secondary market, your money goes to whoever sold it to you, not to Northwind — the company got its cash years ago. This trips up people who think buying corporate bonds always "supports the company"; in the secondary market you are just trading an existing IOU with another investor.

## How it shows up in real markets

**The US Treasury auction calendar.** Every week the Treasury sells tens of billions of dollars of bills, and several times a month it sells notes and bonds, on a schedule published in advance. In 2024, with the federal deficit running near \$1.8 trillion (an illustrative round figure — check the current number, as it moves constantly), the *size* of these auctions has grown so large that traders watch each one for signs of strain. When a 2023 long-bond auction drew weak demand and the dealers were left holding an outsized share, yields jumped and the stock market wobbled — a vivid demonstration of the deficit → supply → yields chain that this whole series keeps returning to.

**Apple's debt issuance.** Apple is the textbook case of a cash-rich company that issues bonds anyway. Starting in 2013 it began selling tens of billions of dollars of bonds despite sitting on a mountain of cash, partly because much of that cash was held overseas and bringing it home would have triggered taxes, and partly because it could borrow at rates so low that it was cheaper to issue debt than to repatriate cash. It even issued bonds to fund dividends and buybacks. The lesson: the most creditworthy issuers borrow not out of need but because the after-tax math of cheap debt is too good to ignore — exactly the cost-of-capital logic from the Northwind example.

**The Detroit bankruptcy, 2013.** Detroit filed the largest municipal bankruptcy in US history, with roughly \$18 billion in liabilities. Holders of its bonds — including supposedly safe general-obligation bonds — faced losses, shattering the myth that munis cannot default. The episode forced the entire muni market to re-price the credit risk of struggling cities and made investors look much harder at *what* actually backs a given muni: full taxing power, a specific revenue stream, or, as Detroit showed, a promise a distressed city may not be able to keep.

**Puerto Rico's default, 2016 onward.** Puerto Rico, a US territory, had borrowed heavily in the tax-exempt muni market — its bonds were triple-tax-exempt and so were eagerly bought by mainland funds. When the island could no longer pay, it defaulted on more than \$70 billion of debt, leading to a years-long restructuring. It was a brutal real-world lesson that the tax exemption is not a safety guarantee, and that a borrower's *ability* to repay matters more than the tax treatment of its coupons.

**The 2008 GSE rescue.** When Fannie Mae and Freddie Mac neared collapse in September 2008, the US government placed them into conservatorship and effectively stood behind their debt — confirming the "implicit guarantee" the market had always assumed. Holders of Fannie and Freddie *bonds* were protected; holders of their *equity* were largely wiped out. The episode crystallized exactly where agency debt sits in the hierarchy: a half-step below Treasuries, safer than corporates, riding on a government backing that was implicit right up until the moment it became explicit.

**The 2020–2021 corporate borrowing boom.** When the pandemic hit and the Federal Reserve slashed rates to near zero and intervened directly in credit markets, corporate borrowing costs collapsed. Companies issued a record wave of bonds — over \$2 trillion of US investment-grade corporate debt in 2020 alone (an illustrative round figure) — locking in ultra-low coupons. Some of it was defensive (raising cash to survive the shock), but much of it was opportunistic: when debt is nearly free, the debt-vs-equity math tilts overwhelmingly toward debt, exactly as the cost-of-capital chart in this post predicts. The hangover came in 2022–2023, when rates rose and that cheap debt started to mature into a far more expensive refinancing world.

## When this matters to you, and where to go next

You don't have to be a bond trader for this to touch your life. The interest rate on your mortgage is a near-cousin of the agency MBS yield. The rate on your car loan and credit card floats above the Treasury curve. The 401(k) you own almost certainly holds Treasuries, agency MBS, corporate bonds, and munis — the four families from this post, in proportions not far from the market sizes we drew. And the deficit debates you hear about in the news are, ultimately, debates about how many new Treasury bonds the market will have to absorb, and at what yield.

The next step is to follow the money in the other direction: we've met the borrowers, so now meet the lenders. The next post, on [who buys bonds](/blog/trading/fixed-income/who-buys-bonds-the-global-demand-for-safe-income), looks at the banks, insurers, pensions, central banks, and foreign governments on the other side of every issue — and why the world has such an insatiable appetite for safe income. From there the series turns to the mechanics that the rest of fixed income is built on: how a bond's [price and yield](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds) move on a seesaw, and how that one relationship ripples out to set the price of money for everyone.

*This is educational material, not investment advice. The dollar figures in the worked examples and the sizes in the figures are illustrative and rounded; for live debt, deficit, and market-size numbers, always check the current as-of data.*

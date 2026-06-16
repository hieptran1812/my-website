---
title: "From Enterprise Value to Price Per Share: The Bridge Everyone Rushes"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A from-zero guide to the EV-to-equity bridge: why a DCF gives you the value of the whole business, how to subtract every claim that ranks ahead of common shareholders, why the right share count is diluted not basic, and how to land on a defensible price per share."
tags: ["equity-research", "corporate-finance", "enterprise-value", "equity-value", "valuation", "dcf", "net-debt", "diluted-shares", "treasury-stock-method", "fundamental-analysis"]
category: "trading"
subcategory: "Equity Research"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A discounted cash flow on the firm's free cash flow values the *whole business*, not the stock. To get from there to a price per share you must walk a specific bridge, and people rush it and get the answer wrong even when the DCF was perfect.
>
> - A DCF on **free cash flow to the firm (FCFF)** discounted at WACC gives you **enterprise value** — the value of the operating business to *everyone* who funded it, lenders and shareholders alike. It is not the value of the equity.
> - To reach **equity value** you walk the **EV-to-equity bridge**: subtract everything that ranks *ahead of* common shareholders (debt, minority interest, preferred stock, pension shortfalls, other debt-like claims) and add back what is *not part of the operating business* (cash and marketable securities, non-operating stakes).
> - Then you divide by the **right share count** — the **fully diluted** count, not the basic count. Options, restricted stock, and convertibles add shares; using the basic count silently overstates value per share.
> - The **treasury stock method** is how options enter the count: their strike proceeds buy back shares, so only the *net* new shares dilute you. Convertibles enter by the **if-converted** method.
> - Run the bridge **backwards** — equity value (market cap) plus debt minus cash — to build the enterprise value that EV/EBITDA and EV/Sales multiples demand. The same logic, in reverse.

Picture two analysts who have just finished the same discounted cash flow model. They forecast the same free cash flows, they used the same WACC, and they computed — to the dollar — the same number: an enterprise value of \$1.5 billion. Then they each divide to get a price per share, and they hand in answers that differ by 30%. One says the stock is worth \$15; the other says \$10. Neither made an arithmetic error inside the DCF. The entire discrepancy lives in the few lines *after* the DCF — the bridge from the value of the business to the value of a single share. One analyst walked that bridge carefully; the other rushed it.

This is the most under-taught, over-skipped step in all of valuation. Textbooks lavish chapters on forecasting revenue, on the terminal value, on the cost of capital — and then, when the hard modeling is done and the enterprise value is sitting there, they wave a hand: *"subtract net debt and divide by shares."* That airy sentence hides at least eight distinct decisions, any one of which can move the per-share answer by 10% or more. Which claims rank ahead of common equity? Is the cash all spare, or is some of it needed to run the business? Do leases count as debt? Which share count — the one on the cover of the 10-K, or the diluted one? How do options and convertibles enter that count? Get any of these wrong and a flawless DCF produces a wrong stock price.

The reason the bridge matters is structural, not cosmetic. A DCF built on FCFF answers a specific question: *how much is the operating business worth to all the people who funded it?* That pool of value belongs to the lenders and the shareholders together. The lenders have first claim; the shareholders get the residual. So to find what the shareholders' slice is worth, you must hand the lenders their share first — and the preferred holders, and the minority owners of any consolidated subsidiary, and the pensioners owed more than the plan holds. Only what is left, divided among the actual shares that will exist, is the common stock's value. The bridge is not bookkeeping; it is the act of separating the residual claimant from everyone ahead of them.

![A side by side comparison of enterprise value as the worth of the whole operating business to every capital provider versus equity value as only the slice left for common shareholders after all senior claims are paid](/imgs/blogs/enterprise-value-to-per-share-the-bridge-1.png)

We will use **Northwind Industries**, the fictional industrial-machinery maker that runs through this series, so every number compounds. If you have read the companion on [free cash flow: FCFF vs FCFE](/blog/trading/equity-research/free-cash-flow-fcff-vs-fcfe) you already know that FCFF discounted at WACC produces enterprise value; this post is what happens *next*. By the end you will be able to walk the full bridge from enterprise value to equity value, compute a fully diluted share count using the treasury stock and if-converted methods, derive a defensible price per share, sanity-check it against the market, and run the whole bridge backwards to build the enterprise value that relative-valuation multiples require. Let us start with the two ideas the entire bridge rests on.

## Foundations: enterprise value, equity value, and who has a claim

Before we can build the bridge we have to be precise about its two endpoints — enterprise value and equity value — and about the single organizing principle that decides every line in between: *who has a claim on the business, and in what order.*

### Enterprise value is the worth of the operating business to everyone

**Enterprise value (EV)** is the value of a company's *core operating business* — the machines, the brands, the customer relationships, the going concern that produces cash — to *all* of the people who funded it. That "all" is the key word. A business is funded by two broad groups: **lenders**, who hold its debt, and **shareholders**, who hold its equity. (Sometimes a third group sits between them — preferred shareholders — and sometimes outside investors own a piece of a subsidiary the company controls.) Enterprise value is the worth of the operating business to that entire coalition of funders, before any of them is paid.

Two consequences follow immediately, and both are deliberate design choices baked into how EV is defined.

First, **enterprise value is independent of how the business is financed.** Two identical factories producing identical cash flows have the same enterprise value whether one is funded 90% by debt and the other 90% by equity. The operating business is the same; only the split of its value among funders differs. This is exactly why a DCF on FCFF — which is computed *before* interest and is therefore capital-structure-neutral — produces enterprise value rather than equity value. The cash flow ignores financing, so the value it produces ignores financing too.

Second, **enterprise value excludes cash.** This trips up nearly everyone the first time. The intuition: enterprise value is meant to capture the worth of the *operating* business, and a pile of spare cash sitting in a bank account is not part of operations — it earns a tiny return and could be paid out tomorrow without affecting how the factory runs. So EV is built to measure the operating engine alone, and the spare cash is treated as a separate, non-operating asset. (This is why, in the bridge, cash gets *added back* when we move from EV to equity value: it was deliberately left out of EV, so it must be put back in for the shareholders, who do own it.) We will return to this repeatedly, because the cash treatment is where a surprising amount of the bridge's subtlety lives.

### Equity value is only what is left for common shareholders

**Equity value** (also called **market value of equity**, or, when you observe it in the market, **market capitalization**) is the value of the slice that belongs to **common shareholders alone** — the residual claimants, the people who are last in line. They are entitled to whatever is left of the business *after* every other claimant has been satisfied: the lenders have been repaid, the preferred holders have taken their fixed claim, the minority owners of subsidiaries have their share, the pensioners are made whole. Equity value is the residual.

This residual quality is the whole reason equity is the riskiest capital and the reason it is the *last* thing you compute. You cannot know what is left until you have set aside everything that comes first. The DCF gives you the size of the whole pie (enterprise value); the bridge sets aside every slice that belongs to someone ahead of the common shareholders; equity value is the crumbs — sometimes a very large pile of crumbs, sometimes nothing at all — that remain. Divide those crumbs by the number of shares and you have the value of one share.

It is worth being explicit about *why* the bridge exists at all, because there is a path that skips it. A DCF can be built two ways. Build it on **free cash flow to the firm (FCFF)** — the cash available to all funders, before interest — discount at WACC, and you get *enterprise value*; then you must walk this bridge to reach equity value. Or build it on **free cash flow to equity (FCFE)** — the cash left for shareholders *after* the lenders are paid — discount at the cost of equity, and you arrive at *equity value directly*, with no bridge needed. So why is the FCFF-and-bridge route the overwhelming default in practice? Because FCFF is capital-structure-neutral and far more stable to forecast: it does not lurch every time the company refinances or changes its debt mix, and it lets you value the operating business once and then handle financing as a separate, transparent step. The bridge is the price you pay for that cleaner forecast — and it is a price worth paying, precisely because the bridge is mechanical and auditable while a leverage-sensitive FCFE forecast is treacherous. The relationship between the two free cash flows is developed in [free cash flow: FCFF vs FCFE](/blog/trading/equity-research/free-cash-flow-fcff-vs-fcfe); for this post, the point is simply that the bridge is the structural consequence of choosing the more robust FCFF path.

### The organizing principle: the claim hierarchy

Everything in the bridge is governed by one idea — the **capital structure hierarchy**, or **waterfall of claims**. Think of the business as generating a pool of value, and imagine that value being poured down a series of steps. At the top, the most senior claimants drink first: secured lenders, then unsecured lenders. Below them sit the hybrid claimants — preferred shareholders, who are owed a fixed amount before common holders but rank below debt. Beside them, in a consolidated subsidiary, the minority (non-controlling) owners have a claim on their slice. And only at the very bottom, after everyone above has been satisfied, do the common shareholders collect whatever is left.

The bridge is nothing more than walking down this waterfall on paper. Start with the whole pool (enterprise value), hand each senior claimant their due (subtract it), add back anything that was excluded from the operating pool but does belong to the owners (cash, non-operating assets), and what remains is the common shareholders' value. Once you internalize the hierarchy, you never have to memorize whether a given item is added or subtracted — you just ask: *does this rank ahead of common equity (subtract it as a claim) or is it spare value the owners are entitled to (add it back)?* The sign falls out of the answer.

With those three ideas — EV as the operating business for everyone, equity value as the residual for common holders, and the claim hierarchy that orders them — we can build the bridge.

## The EV-to-equity bridge, line by line

Here is the bridge written as a formula. Read it not as something to memorize but as a sentence in the claim hierarchy:

$$\text{Equity value} = \text{EV} - \text{total debt} + \text{cash \& securities} - \text{minority interest} - \text{preferred} - \text{other claims} + \text{non-operating assets}$$

Every term has the same justification, applied with the right sign. Subtract anything that is a **claim ahead of common equity**. Add anything that is **value the owners are entitled to but that was excluded from operating enterprise value**. That is the entire logic. Let us walk each term.

![A vertical waterfall starting from enterprise value of fifteen hundred million, subtracting debt, adding cash, subtracting minority interest and preferred stock, and ending at an equity value of eleven hundred fifty million](/imgs/blogs/enterprise-value-to-per-share-the-bridge-2.png)

**Subtract total debt.** Lenders rank ahead of shareholders, so their claim comes out of the pool first. Use *total* debt — short-term borrowings, the current portion of long-term debt, long-term bonds and loans, and increasingly the capitalized lease liabilities that accounting now puts on the balance sheet. Use the *market value* of debt where it differs materially from book (for distressed companies it can differ enormously), though for investment-grade companies book value is usually a fine approximation because the debt trades near par.

**Add cash and marketable securities.** Because enterprise value was deliberately defined to *exclude* cash (it is not part of the operating business), and because that cash genuinely belongs to the shareholders, you add it back when moving to equity value. Marketable securities — short-term investments the company could liquidate at will — get the same treatment. (We will see shortly that not *all* cash should be added back; some is needed to run the business and is not truly spare. That is the single biggest nuance in the bridge.)

**Subtract minority interest (non-controlling interest).** When a company *controls* a subsidiary it does not fully own — say it owns 80% of a unit — accounting rules make it *consolidate* the whole subsidiary onto its statements: 100% of the subsidiary's revenue, 100% of its assets, and crucially, 100% of the cash flows that feed the FCFF and hence the enterprise value. But the parent's shareholders do not own that last 20%; outside investors do. **Minority interest** is the line that represents those outside owners' claim, and because the enterprise value captured 100% of the subsidiary while the parent's shareholders are entitled to only 80%, you must subtract the minority's slice to avoid handing the parent's shareholders value that belongs to someone else.

**Subtract preferred stock.** Preferred shareholders hold a hybrid security — junior to debt, senior to common. They are typically owed a fixed dividend and rank ahead of common holders in a wind-up. Their claim comes out of the pool before the common holders collect, so subtract it. (Whether to use book or market value of the preferred depends on the situation; for straight preferred, the liquidation or redemption value is usually the right figure.) Two refinements matter when preferred is large. First, if the preferred is **cumulative** and the company has skipped dividends, the **dividends in arrears** — the unpaid back dividends that must be cleared before common holders see a cent — are an additional senior claim and belong in the subtraction. Second, if the preferred is **convertible**, you face the same in-the-money-or-not switch as a convertible bond: subtract it as a senior claim when conversion is out of the money, or treat it as common shares (add them to the diluted count, drop the preferred from the bridge) when conversion is in the money. The hybrid nature of preferred is exactly what makes it easy to mishandle.

A subtle point on the minority-interest line is *how to value it*. The balance-sheet minority-interest figure is a **book** number, and book often understates economic value badly for a profitable, growing subsidiary. A more rigorous approach values the minority's stake at its economic worth — for instance, by applying the subsidiary's own valuation multiple to the minority's share of its earnings, or by a separate small DCF of the minority's slice. For a quick valuation the book figure is a serviceable approximation; for a company whose value turns on a partly-owned crown-jewel subsidiary, valuing the minority interest properly can change the bridge materially. The principle is unchanged — you are removing the value that belongs to the subsidiary's outside owners — but the *amount* you remove deserves more than a glance at the book line.

**Subtract other claims that rank ahead of common equity.** This is the catch-all that separates careful analysts from sloppy ones. The most common members: **underfunded pension and other post-employment obligations** (if the pension plan owes its retirees more than the plan's assets hold, the shortfall is a real, debt-like obligation of the company that ranks ahead of shareholders), **operating-lease liabilities** if not already in your debt figure, certain **provisions and contingent liabilities**, and sometimes **deferred consideration** owed on a past acquisition. Each is a claim on the business that the common shareholders do not get to keep, so each is subtracted.

**Add the value of non-operating assets.** The mirror image of cash: if the company holds something of value that is *not* part of its operating business and was therefore *not* captured by the FCFF-based enterprise value, you add its value back because the shareholders own it. The classic example is a **minority stake in another company** carried as an investment (an "associate") — its value never flowed into the operating FCFF, so it would be missed entirely if you did not add it. Other examples: excess real estate the business does not use, a non-core division valued separately, or a stake in a publicly traded company.

![A three column grid showing each bridge item, whether it ranks ahead of common equity, and whether the bridge action is to subtract it as a senior claim or add it as value owed to owners](/imgs/blogs/enterprise-value-to-per-share-the-bridge-3.png)

Notice the symmetry in the figure above. Every "subtract" line is a claim that ranks *ahead of* common equity. Every "add" line is value that belongs to the owners but sits *outside* the operating enterprise value. You never have to memorize the signs — you derive them from the claim hierarchy.

#### Worked example: walking Northwind's bridge from EV to equity value

Northwind's DCF on FCFF, discounted at WACC, produced an **enterprise value of \$1,500M** for the operating business. Now we walk the bridge to the common shareholders. From the latest balance sheet and footnotes:

- **Total debt:** \$390M (a mix of bank loans, a bond, and capitalized leases).
- **Cash and marketable securities:** \$150M.
- **Minority interest:** \$60M (outside owners of an 80%-owned distribution subsidiary).
- **Preferred stock:** \$50M (a straight preferred issue at its \$50M redemption value).

The bridge:

- **Start: enterprise value = \$1,500M.** The whole operating business, owed to all funders.
- **Subtract total debt: −\$390M → \$1,110M.** Lenders are paid first.
- **Add cash and securities: +\$150M → \$1,260M.** Spare cash, excluded from EV, belongs to owners.
- **Subtract minority interest: −\$60M → \$1,200M.** The 20% of the subsidiary the parent does not own.
- **Subtract preferred stock: −\$50M → \$1,150M.** A senior claim ahead of common.
- **Equity value = \$1,150M.**

So of the \$1,500M of operating-business value, **\$1,150M belongs to Northwind's common shareholders**, after the lenders, the minority owners, and the preferred holders have each been handed their claim. That \$1,150M is the number we will divide by shares to get a price per share.

*The bridge is just the claim hierarchy on paper: hand each senior claimant their slice, add back the spare cash that was never part of operations, and what is left is the common shareholders' value.*

#### Worked example: adding a non-operating stake Northwind almost missed

Suppose that, buried in the footnotes, Northwind also holds a **30% minority stake in a parts supplier**, carried on the balance sheet as an associate. Because Northwind does not control the supplier, it does *not* consolidate it — only Northwind's *share of the supplier's profit* appears, in a single line below operating income, and crucially **none of the supplier's cash flow ever entered Northwind's operating FCFF.** The DCF therefore valued Northwind's operating business at \$1,500M and saw nothing of the stake.

A careful analyst estimates the stake is worth **\$80M** (perhaps 30% of the supplier's own market value, or a small DCF of Northwind's share of its dividends). Because that value sits *outside* the operating enterprise value, it must be added:

- **Equity value before the stake:** \$1,150M.
- **Add non-operating stake: +\$80M.**
- **Adjusted equity value = \$1,230M.**

Miss the stake and you understate Northwind's equity by \$80M — about 7% of its value — not because the DCF was wrong but because a real asset never entered it. This is the mirror of the cash logic: anything of value the FCFF did not capture must be added back by hand.

*Non-operating assets are the easiest value to leave on the table, because the DCF never sees them; the bridge is where you put them back.*

## Net debt is subtler than "debt minus cash"

The two largest lines in most bridges — debt and cash — are usually collapsed into a single figure called **net debt** (total debt minus cash and equivalents), and the bridge is often written as the compact *equity value = EV − net debt − minority − preferred*. That compression is fine for a clean company, but it hides four traps. Each can move the bridge by tens of millions, and each separates an analyst who *understands* net debt from one who copies it off a screener.

![A three column grid of net debt nuances comparing the naive view to the careful view for operating versus excess cash, leases, restricted cash, and gross versus net debt](/imgs/blogs/enterprise-value-to-per-share-the-bridge-7.png)

**Operating cash versus excess cash.** Not all cash on the balance sheet is spare. A business needs some minimum working balance just to operate — to make payroll, to smooth the timing of receipts and payments, to hold a buffer against a bad month. That **operating cash** is not truly available to shareholders; it is as committed to the business as the inventory. Only the **excess cash** above that operating minimum is genuinely spare and belongs in the add-back. A company reporting \$150M of cash but needing \$30M to operate has only \$120M of excess cash to add back. Many analysts add back the full reported cash, which overstates equity value. The harder the company's business is to run on a thin cash balance (seasonal, capital-intensive, volatile), the larger the operating cash you should carve out.

**Leases.** Under modern accounting (IFRS 16 and the equivalent US standard), most operating leases now sit on the balance sheet as a **lease liability** with a matching right-of-use asset. Economically a lease is debt-like: a fixed obligation to pay, senior to shareholders. If your enterprise value was built on a cash flow *before* lease payments (an EBITDAR or pre-lease basis), the lease liability belongs in your debt figure in the bridge. The trap is *inconsistency* — counting leases as debt in the bridge while having already subtracted lease payments inside the cash flow, which double-counts them. Pick one treatment and apply it consistently between the cash flow and the bridge.

**Restricted cash.** Some cash is pledged — held in escrow, posted as collateral, locked up by a covenant, or trapped in a subsidiary that cannot remit it. **Restricted cash** is not free for the owners and should *not* be added back as if it were spare. Treating restricted cash as excess cash inflates equity value with money the shareholders cannot actually touch.

**Gross versus net debt.** The default is to net cash against debt, but there are cases where you should use **gross debt** — that is, *not* net the cash. The most important: cash that is trapped offshore and would be heavily taxed to repatriate, or cash earmarked for a specific obligation. If \$100M of "cash" can only reach shareholders after a 20% tax hit, netting it dollar-for-dollar against debt overstates its value. In these cases use gross debt and add back only the *after-friction* value of the cash.

#### Worked example: Northwind's careful net debt

Northwind's screener shows total debt of \$390M and cash of \$150M, for a naive net debt of \$240M. But a careful read of the footnotes reveals:

- **\$30M of the cash is operating cash** the business needs to run — not spare.
- **\$20M of the cash is restricted**, held in escrow against a warranty claim — not free for owners.
- The remaining **\$100M is genuinely excess cash.**

So the cash truly available to shareholders is \$100M, not \$150M. Redoing the relevant bridge lines:

- **Subtract total debt: −\$390M.**
- **Add only excess cash: +\$100M** (not the full \$150M).
- The net effect on the bridge is **−\$290M**, not the naive **−\$240M.**

That \$50M difference — the operating and restricted cash a careless analyst would have added back — flows straight into equity value. Across 115M diluted shares, it is about \$0.43 per share, roughly 4% of the stock. A "trivial" cash line, read carelessly, moves the answer by 4%.

*Net debt is not a number you copy; it is a judgment about which cash is genuinely free and which claims are genuinely debt-like.*

## The share count: why diluted, not basic

You have the equity value. The last step looks like the easiest — *divide by the number of shares* — and it is where a startling number of valuations quietly go wrong. The trap is using the **basic share count** (the shares outstanding today) when you should use the **fully diluted share count** (the shares that will exist once all the obligations to issue more stock are honored). Using basic instead of diluted *overstates* the value per share, because it divides the same equity value by too few shares.

### Basic versus diluted, defined from zero

**Basic shares outstanding** is the number of common shares that exist right now — the figure on the cover of the 10-K. **Diluted shares** is the larger number that will exist once every claim on future shares is exercised: employee and executive **stock options**, **restricted stock units (RSUs)** that vest into shares, **warrants**, and **convertible bonds or preferred** that can turn into common stock. Every one of these is a promise to hand someone a share at some point, and each promise, when honored, increases the number of shares the equity value must be split across.

The reason this matters is that these instruments are *not* free to the existing shareholders. An option lets an employee buy a share, often at a price below the market — diluting the existing owners. A convertible bond lets a lender swap their loan for shares — diluting the existing owners. If you value the equity at \$1,150M and divide by today's 100M basic shares, you get \$11.50, but that ignores the 15M shares that options and convertibles will add. The true claim per *existing* share is lower, because the pie is sliced into more pieces. Valuing on the basic count is a systematic upward bias, and it is larger the more option-heavy and convertible-heavy the company is — which is to say, largest exactly where it matters most, in growth and technology companies that pay employees in equity.

![A bar chart comparing eleven dollars fifty per share on a basic count of one hundred million shares against ten dollars on a diluted count of one hundred fifteen million shares for the same equity value](/imgs/blogs/enterprise-value-to-per-share-the-bridge-5.png)

### The treasury stock method: how options enter the count

The naive way to count options would be to add every option to the share base. But that overstates dilution, because options are not free to exercise — the holder must pay the **strike price** (also called the exercise price) to the company, and that cash is real. The standard, accounting-blessed way to handle this is the **treasury stock method (TSM)**, and its logic is elegant once you see it.

When option holders exercise, two things happen. First, the company issues new shares (dilution). Second, the company *receives* the strike proceeds — cash it can use to buy back shares in the market at the current price (anti-dilution). The treasury stock method nets these against each other: of the gross new shares from exercise, the company is assumed to use the strike proceeds to repurchase as many shares as that cash will buy at the market price, and only the *net* new shares actually dilute. The mechanics:

1. **Gross new shares** = the number of in-the-money options (and only in-the-money ones — out-of-the-money options will not be exercised and add nothing).
2. **Strike proceeds** = number of options × strike price.
3. **Shares repurchased** = strike proceeds ÷ current market price.
4. **Net new shares** = gross new shares − shares repurchased.

The net new shares are what you add to the basic count. The deeper the options are in the money (the further the market price is above the strike), the larger the net new shares, because each option's proceeds buy back proportionally less. Out-of-the-money options drop out entirely.

![A before and after comparison showing the naive method adding every option share against the treasury stock method where strike proceeds buy back shares so only the net new shares dilute](/imgs/blogs/enterprise-value-to-per-share-the-bridge-4.png)

#### Worked example: Northwind's options under the treasury stock method

Northwind has **10M employee stock options outstanding** with a **\$15 strike price**. The current share price is **\$30**. The options are deeply in the money (\$30 > \$15), so they will be exercised and must be counted. Apply the treasury stock method:

- **Gross new shares:** 10M (all in the money).
- **Strike proceeds:** 10M × \$15 = **\$150M** of cash the company receives.
- **Shares repurchased:** \$150M ÷ \$30 (the market price) = **5M shares**.
- **Net new shares:** 10M − 5M = **5M shares**.

So Northwind's 10M options add only **5M shares** to the diluted count, not 10M. The treasury stock method has cut the dilution in half, because the \$150M of strike cash buys back 5M shares at the \$30 market price. Had the share price been higher — say \$60 — the same proceeds would buy back only 2.5M shares, and the net dilution would be 7.5M. The further in the money, the worse the dilution.

*The treasury stock method says options dilute you only by their net effect — the new shares minus the shares the strike cash can repurchase — which is why deep-in-the-money options hurt the most.*

### Convertibles: the if-converted method

A **convertible bond** is debt that the holder can swap for a fixed number of shares. While it is still a bond, it pays interest and sits in your debt figure. But if the stock is high enough that conversion is attractive, the holder will convert, and the bond becomes shares — diluting the common holders. The standard treatment is the **if-converted method**: assume the bond converts, add the shares it would become to the diluted count, and *remove the bond from your debt figure* (since it is no longer debt). You also add back the after-tax interest the company would no longer pay, though for the bridge the share-count and debt effects dominate.

The crucial discipline with convertibles is **avoiding double-counting**. A convertible cannot be *both* debt (subtracted in the bridge) *and* equity (adding shares to the count). You pick one state. If conversion is in the money, treat it as if-converted: add the shares, drop the debt. If conversion is out of the money, treat it as straight debt: subtract it, add no shares. Counting it as debt *and* adding its shares subtracts its value twice; ignoring it entirely misses real dilution. The if-converted method forces the consistent choice.

#### Worked example: Northwind's convertible bond, if-converted

Northwind issued a **\$100M convertible bond** that converts into **10M shares** (a conversion price of \$10 per share). With the stock at \$30, conversion is deeply in the money — a holder converting receives 10M shares worth \$300M for a \$100M bond, so they will convert. Apply the if-converted method:

- **Add the conversion shares:** +10M shares to the diluted count.
- **Remove the bond from debt:** the \$100M convertible leaves the debt figure (it is becoming equity, not staying a loan).

Note the offsetting bridge effects: the equity value *rises* by \$100M (because \$100M of debt is no longer subtracted), but it is now split across 10M more shares. The \$100M of removed debt, spread across the larger base, is roughly \$0.87 per share of added value — but the 10M new shares dilute the existing 105M-ish base by nearly 9%. For deeply in-the-money convertibles, the dilution from the shares usually dwarfs the relief from the removed debt, which is why a convertible that has gone deep in the money is a genuine overhang on the common stock.

*A convertible is either debt or shares, never both; the if-converted method makes you commit, add the shares, and drop the debt — and for deep-in-the-money converts, the dilution bites harder than the debt relief helps.*

### RSUs and warrants: the rest of the dilution

Options and convertibles get the spotlight, but two other instruments commonly add to the diluted count, and a complete share count must include them. **Restricted stock units (RSUs)** are promises to deliver shares to employees once a vesting condition is met (usually the passage of time). Unlike options, RSUs have no strike price — the employee pays nothing — so they bring in *no* proceeds to repurchase shares. That makes RSUs **simpler but more dilutive than options**: every unvested RSU that will eventually vest adds a *full* share to the diluted count, with no treasury-stock offset. A company heavy in RSUs (the modern norm at large technology firms) carries dilution that the treasury stock method does not soften at all.

**Warrants** are economically options issued to outside investors or lenders rather than employees — they confer the right to buy shares at a set price. They are handled by exactly the same treasury stock method as employee options: count only in-the-money warrants, let their strike proceeds buy back shares, and add the net new shares. Warrants frequently appear attached to financing deals, in companies emerging from restructuring, and in vehicles like SPACs, where they can be a large and easily overlooked source of dilution. The discipline is the same across all of these instruments: every promise to issue future shares — option, RSU, warrant, or convertible — must be run through the right method (treasury stock for those with a strike, if-converted for convertibles, full-count for no-strike RSUs) and folded into the diluted denominator. Leave any of them out and you divide by too few shares.

## From equity value to price per share

Now we assemble the final number. The price per share is simply:

$$\text{Value per share} = \frac{\text{Equity value}}{\text{Fully diluted shares}}$$

The numerator is what the bridge produced. The denominator is the basic count plus every net new share from the treasury stock method (options, warrants, RSUs) and the if-converted method (convertibles). Let us put Northwind together.

#### Worked example: Northwind's price per share, and the error from using basic shares

Northwind's bridge produced an **equity value of \$1,150M** (we will use the base bridge, before the non-operating stake, to keep the share-count story clean). Its share count:

- **Basic shares outstanding:** 100M.
- **Net new shares from options (treasury stock method):** +5M.
- **Shares from the convertible (if-converted):** +10M.
- **Fully diluted shares:** 100M + 5M + 10M = **115M.**

The **correct** value per share, on the diluted count:

$$\frac{\$1{,}150\text{M}}{115\text{M shares}} = \$10.00 \text{ per share.}$$

Now watch the error. A rushed analyst divides by the **basic** count of 100M:

$$\frac{\$1{,}150\text{M}}{100\text{M shares}} = \$11.50 \text{ per share.}$$

The basic-share answer is **\$11.50**; the correct diluted answer is **\$10.00**. The basic count overstates the value per share by **\$1.50, or 15%** — enough to flip a stock from "fairly valued" to "15% undervalued" and trigger a buy that should never have happened. And note that the equity value did not change at all between the two calculations; the entire 15% error came from dividing by the wrong number of shares. The DCF was perfect. The bridge's last line was rushed.

*Using basic shares instead of diluted shares divides the right equity value by too few shares, and the resulting overstatement is largest exactly where options and convertibles are largest — in the equity-heavy growth companies where the stakes are highest.*

#### Worked example: when the convertible is out of the money

To see the if-converted discipline bite the other way, change one fact: suppose Northwind's stock is **\$8**, not \$30, and the convertible's conversion price is \$10. Now conversion is *out of the money* — a holder converting would receive 10M shares worth \$80M for a \$100M bond, so they will *not* convert. The if-converted method says: treat it as straight debt.

- **Convertible stays in debt:** subtract the \$100M in the bridge (it is a loan, not shares).
- **Add no conversion shares:** the 10M shares do not enter the diluted count.

If you had blindly applied if-converted regardless of the stock price — adding 10M shares *and* keeping the \$100M as debt — you would have double-counted the convertible: subtracting its \$100M as debt while also diluting by its 10M shares. The diluted count would be wrong (115M instead of 105M) and the debt would be overstated. The rule is mechanical: **a convertible is debt when out of the money and shares when in the money — never both, never neither.**

*The if-converted method is a switch, not a default: the convertible flips between debt and equity depending on whether conversion pays, and applying it the wrong way double-counts the security.*

## The reverse bridge: equity value back to enterprise value

So far we have run the bridge in one direction: from the enterprise value a DCF produces *down* to equity value and a price per share. But you will run it just as often in the *other* direction — starting from the equity value the market hands you (the market capitalization) and walking *up* to enterprise value. You need this whenever you compute a multiple that has enterprise value in the numerator: **EV/EBITDA, EV/Sales, EV/EBIT, EV/FCF**. These multiples compare the value of the *operating business* to an operating metric, and the operating metric (EBITDA, sales) belongs to all funders — so it must be paired with the all-funder value, enterprise value, not equity value.

The reverse bridge is the forward bridge with every sign flipped:

$$\text{EV} = \text{market cap} + \text{total debt} + \text{minority interest} + \text{preferred} - \text{cash \& securities}$$

You **add** the claims you previously subtracted (debt, minority, preferred), because going from the residual back to the whole pool means putting the senior claims *back in*. And you **subtract** the cash you previously added, because cash is not part of the operating enterprise value. The market cap is your starting point — the share price times the diluted share count, observed directly on any finance website.

![A before and after comparison showing the reverse bridge from market capitalization to enterprise value by adding back debt minority interest and preferred and subtracting cash for use in EV multiples](/imgs/blogs/enterprise-value-to-per-share-the-bridge-6.png)

#### Worked example: Northwind's reverse bridge for an EV/EBITDA multiple

Suppose Northwind trades at exactly our computed value — \$10 per share on 115M diluted shares — for a **market capitalization of \$1,150M**. We want its EV/EBITDA multiple, and EBITDA (an all-funder, operating number) requires enterprise value. Run the bridge backwards:

- **Start: market cap = \$1,150M.**
- **Add total debt: +\$390M → \$1,540M.** Lenders' claim goes back in.
- **Add minority interest: +\$60M → \$1,600M.** The subsidiary's outside owners go back in.
- **Add preferred: +\$50M → \$1,650M.** The preferred claim goes back in.
- **Subtract cash: −\$150M → \$1,500M.** Cash is not part of the operating business.
- **Enterprise value = \$1,500M.**

We are back to the \$1,500M the DCF started from — the bridge is reversible, as it must be. Now, if Northwind's EBITDA is \$200M, its **EV/EBITDA = \$1,500M ÷ \$200M = 7.5×**. Had we sloppily used market cap in the numerator (\$1,150M ÷ \$200M = 5.75×), we would have computed a meaningfully cheaper-looking multiple — and compared Northwind against peers on an inconsistent, equity-vs-operating basis. The reverse bridge is what makes EV multiples comparable across companies with different debt loads. We develop this fully in the companion on [multiples: P/E, EV/EBITDA, P/B, P/S, PEG](/blog/trading/equity-research/multiples-101-pe-ev-ebitda-pb-ps-peg).

*The reverse bridge exists because EV multiples must compare an all-funder value to an all-funder metric; pairing market cap with EBITDA mismatches a shareholder number against an everyone number, and the multiple lies.*

## Sanity-checking the per-share answer

A DCF-and-bridge per-share value is an *estimate*, and like every estimate it should be cross-examined before you trust it. Three sanity checks catch most blunders.

**Check it against the current market price.** Your \$10 per share is a claim about intrinsic value; the market is quoting some price right now. If your number is within a sensible band of the market price (say, ±25%), the disagreement is an investment thesis you can articulate. But if your number is \$10 and the stock trades at \$2, or at \$40, the *first* hypothesis should be that *you* made an error — a missed claim in the bridge, a wrong share count, a runaway terminal value — not that the entire market is wrong by 4×. Extreme disagreements are far more often bridge errors than genuine mispricings. Reconcile the gap before you act on it.

**Check the implied market cap against the reported market cap.** Multiply your diluted share count by the *current* market price and compare to the reported market capitalization. If your 115M diluted shares × the market price differs wildly from the market cap quoted on a data provider, you have probably miscounted shares — most often by using basic instead of diluted, or by botching the treasury stock or if-converted math. The market's own market cap is a free check on your denominator.

**Check that each bridge line ties to a balance-sheet or footnote number.** Every subtraction and addition in the bridge should trace to a specific figure in the financial statements or the footnotes — the debt to the debt schedule, the minority interest to its balance-sheet line, the pension shortfall to the pension footnote, the options to the equity-compensation footnote. A bridge line you cannot source is a bridge line you invented. The footnotes are where the bridge is sourced; the post on [reading the 10-K footnotes and the MD&A](/blog/trading/equity-research/reading-the-10k-footnotes-and-mda) is the companion on where to find each one.

#### Worked example: Northwind's per-share value fails, then passes, the market check

Suppose Northwind trades at \$10.20. Our diluted per-share value is \$10.00 — a 2% gap, well within reason, and the implied market cap (115M × \$10.20 = \$1,173M) ties closely to the reported \$1,150M–\$1,180M range. The valuation passes: the small gap is a defensible thesis (the stock is roughly fairly valued, perhaps a hair rich).

Now imagine a junior analyst on the same company reports a per-share value of **\$15** while the stock trades at \$10.20 — a 47% claimed upside. Before declaring a screaming buy, the sanity check should make them suspicious. Re-examining, they find the error: they divided the \$1,150M equity value by the **75M basic shares** they pulled off an out-of-date cover page (and missed the options and convertible entirely): \$1,150M ÷ 75M = \$15.33. Counting the full 115M diluted shares brings it back to \$10. The "47% upside" was never real; it was a share-count error the market check flagged in seconds.

*The market is a free auditor of your bridge: when your per-share number disagrees with the price by a lot, suspect your own share count and missed claims long before you suspect a mispricing.*

## Common misconceptions

**"Enterprise value is just a fancier market cap."** No — they answer different questions and differ by net debt plus other claims. Enterprise value is the worth of the *operating business to all funders*; market cap is the worth of the *equity to shareholders alone*. For a debt-heavy company the two diverge enormously: a company with a \$1B market cap and \$2B of net debt has a \$3B enterprise value, and an EV/EBITDA multiple computed on the \$1B would be three times too cheap. The whole point of EV is to value the business in a way that does not depend on how it happens to be financed.

**"Just subtract net debt and divide by shares — it's one line."** That airy instruction hides at least eight decisions: which claims rank ahead of equity (minority, preferred, pensions), which cash is truly spare versus operating versus restricted, whether leases are debt, gross versus net debt, basic versus diluted shares, the treasury stock method for options, the if-converted method for convertibles, and the value of non-operating assets. Each can move the per-share answer by several percent. "One line" is exactly the attitude that produces a 15% error from a perfect DCF.

**"Cash is always added back at face value."** Only *excess* cash is freely available to shareholders. Operating cash is committed to running the business; restricted cash is pledged and untouchable; offshore cash may carry a repatriation tax. Adding back the full reported cash, as if every dollar were spare, systematically overstates equity value — most for cash-rich, capital-intensive, or multinational companies.

**"Diluted shares barely differ from basic, so basic is fine."** For a mature company with few options, the gap is small. But for the option-and-convertible-heavy growth companies where valuation matters most, diluted can exceed basic by 10–20% or more, and the per-share overstatement from using basic is exactly that large. The error is biggest precisely where the stakes are highest. Stock-based compensation also keeps *adding* to the option pool every year, so the dilution is a recurring drag, not a one-time event.

**"A convertible is debt, so it always gets subtracted."** Only when it is out of the money. When the stock is high enough that conversion pays, the if-converted method treats the convertible as *equity* — you add its shares and remove it from debt. Counting it as debt *and* its shares double-counts; treating an in-the-money convertible as pure debt misses real dilution. The convertible flips between the two states depending on the stock price.

**"If my DCF is right, my stock price is right."** The DCF gives you enterprise value; the *bridge* gives you the stock price, and the bridge has its own dozen ways to go wrong. A flawless DCF feeding a rushed bridge produces a wrong per-share number with full confidence — which is more dangerous than an obviously rough estimate, because the precision of the DCF lends false authority to the careless bridge.

## How it shows up in real markets

**Cash-rich balance sheets and the EV discount.** For years, the largest technology companies carried enormous net cash — Apple, Alphabet, and Microsoft have at various points held cash and securities exceeding \$50–100B (illustrative orders of magnitude, not point-in-time figures). For such companies the EV-to-equity gap runs the *other* way: enterprise value is *below* market cap, because the cash add-back exceeds the debt. An analyst who computes EV/EBITDA without subtracting that mountain of cash overstates the multiple badly and makes these companies look more expensive than they are on an operating basis. The cash line, so often treated as a rounding detail, is the difference between a right and a wrong multiple for the most valuable companies in the world.

**Stock-based compensation and the diluted-share creep.** High-growth technology and biotech companies pay a large share of compensation in equity, issuing options and RSUs every year. The diluted share count at such companies can sit 10–20% above basic, and it *grows* annually as new grants pile up. Investors who value these companies on the basic count — or who ignore the steady issuance of new stock — systematically overpay, because the value per *existing* share is being diluted every year. The recurring nature of stock-based dilution is one of the most under-modeled drags on per-share value in growth investing, and it is precisely the line the bridge forces you to confront.

**Convertible bonds and the dilution overhang.** Companies that fund themselves with convertible bonds — common among growth firms and, famously, among some that later struggled — carry a hidden dilution overhang. When the stock rises and the convertibles go deep in the money, the if-converted share count jumps, and the per-share value the common holders can claim falls even as the business does well. A convertible-heavy capital structure means the common shareholders share more of the upside than the basic count suggests — a subtlety that only the if-converted method makes visible, and one that catches investors who valued the stock on its basic shares.

**Minority interest in conglomerates and holding companies.** Sprawling holding companies and conglomerates often consolidate subsidiaries they only partly own. Their reported revenue and EBITDA include 100% of those subsidiaries, but the parent's shareholders own less. An analyst who values such a company on consolidated EBITDA without subtracting minority interest in the bridge hands the parent's shareholders value that belongs to the minority owners — a systematic overstatement that is largest for the most acquisitive, structurally complex companies. The minority-interest line, easy to overlook on a clean-looking balance sheet, is the correction.

**Pension shortfalls as hidden debt.** Old-economy companies with large defined-benefit pension plans — industrials, airlines, automakers — can carry pension obligations that exceed their plan assets by billions. That **underfunded pension** is a real, senior, debt-like claim on the business, but it sits in a footnote rather than the debt schedule, so a bridge that stops at "debt minus cash" misses it entirely. During market downturns, when plan assets fall and the shortfall widens, this hidden claim can swing the equity value of a pension-heavy company sharply — and it is invisible to anyone who did not read the pension footnote into the bridge.

## When this matters and further reading

The EV-to-equity bridge is where intrinsic valuation meets the actual stock price, and it is the step that separates a model that *looks* rigorous from one that *is*. The DCF earns the headlines; the bridge earns the answer. Rush it and a perfect DCF produces a per-share number that is confidently wrong — overstated by the cash you added too generously, the claims you forgot to subtract, the diluted shares you replaced with basic. Walk it carefully — subtract every senior claim, add only the truly spare value, count the diluted shares with the treasury stock and if-converted methods, and sanity-check against the market — and the bridge turns the value of a business into a defensible value for a share.

It matters most where the capital structure is complex: companies with meaningful debt, preferred stock, minority interests, pension shortfalls, large option pools, or convertible bonds. For a debt-free company with no options, the bridge collapses to "add cash, divide by shares" and the rush does no harm. But those companies are the exception. For everyone else, the bridge is where the value of a share is actually determined, and rushing it is the most common way a good valuation produces a bad number.

To go deeper, the natural next steps in this series are:

- [Free cash flow: FCFF vs FCFE](/blog/trading/equity-research/free-cash-flow-fcff-vs-fcfe) — why FCFF gives you enterprise value (and therefore why you need this bridge at all), and how FCFE gets you to equity value directly without it.
- [Building a DCF, part 2: cost of capital (WACC and CAPM)](/blog/trading/equity-research/building-a-dcf-part-2-cost-of-capital-wacc-capm) — the discount rate that turns FCFF into the enterprise value this bridge starts from.
- [The balance sheet: what a company owns, owes, and is worth](/blog/trading/equity-research/balance-sheet-what-a-company-owns-owes-and-is-worth) — where every bridge line (debt, cash, minority interest, preferred) lives on the statements.
- [Multiples 101: P/E, EV/EBITDA, P/B, P/S, PEG](/blog/trading/equity-research/multiples-101-pe-ev-ebitda-pb-ps-peg) — where the *reverse* bridge does its work, turning market cap into the enterprise value that EV multiples demand.
- [Reverse DCF and sensitivity analysis](/blog/trading/equity-research/reverse-dcf-and-sensitivity-analysis) — testing how the per-share answer moves when the bridge's assumptions (excess cash, dilution, discount rate) flex.

Get the bridge right and the rest of valuation has somewhere to land. Get it wrong and the most careful DCF in the world produces a stock price that is precisely, confidently, wrong.

---
title: "Stock-based compensation, buybacks, and EPS optics"
date: "2026-08-05"
publishDate: "2026-08-05"
description: "How stock-based compensation hides a real cost inside adjusted metrics, how buybacks flatter earnings per share while issuance quietly offsets them, and the six tests that separate a genuine return of capital from a cosmetic one."
tags: ["stock-based-compensation", "share-buybacks", "earnings-per-share", "dilution", "non-gaap", "forensic-accounting", "treasury-stock-method", "executive-compensation", "financial-statements", "earnings-quality", "excise-tax"]
category: "trading"
subcategory: "Forensic Accounting"
author: "Hiep Tran"
featured: true
readTime: 58
---

> [!important]
> **TL;DR** — Stock-based compensation is a real cost that shows up almost nowhere except the share count, and buybacks are the tool that hides it there. Read the denominator, not the buyback press release.
>
> - Stock-based compensation (SBC) is charged to existing shareholders as a transfer of ownership, not to the company as a payment of cash. Because it is non-cash, it is added back in the cash flow statement — and then, in most companies' investor decks, added back a *second* time in "adjusted" profit.
> - A buyback shrinks the share count; issuing shares to employees grows it. Only the net matters. In fiscal 2024 Alphabet spent \$62.0 billion repurchasing 379 million shares and ended the year with 249 million fewer shares outstanding — an effective cost of about \$249 per share of real reduction, roughly 1.5× the \$163.71 average price it actually paid.
> - Salesforce spent \$32.0 billion on buybacks over fiscal 2023–2026 and reduced its diluted share count by 41 million shares. Snowflake spent \$3.4 billion over fiscal 2024–2026 and its diluted share count went *up* by 18 million against its pre-buyback fiscal 2023 baseline.
> - EPS accretion is arithmetic, not value. A buyback raises earnings per share whenever the earnings yield exceeds the after-tax cost of the money used — which says nothing about whether the shares were worth buying.
> - The single number to track: **diluted weighted-average shares outstanding, over five years.** In the fourth quarter of 2024, S&P Dow Jones Indices found that only 11.9% of S&P 500 companies had cut that number by as much as 4% year over year — against a record \$942.5 billion of buybacks for the year ([S&P Dow Jones Indices, 19 March 2025](https://press.spglobal.com/2025-03-19-S-P-500-Q4-2024-Buybacks-Increase-7-4-and-2024-Expenditure-Sets-New-Record-by-Increasing-18-5-Earnings-Per-Share-Increases-from-Buybacks-Decline-for-the-Quarter,-as-Q1-2025s-Impact-is-Expected-to-Increase)).

In fiscal year 2024, Alphabet spent \$62.0 billion buying back its own stock. It purchased 379 million shares. At the end of the year it had 249 million fewer shares outstanding than at the start.

Those two numbers are both true, and the gap between them is the subject of this article. One hundred and thirty million shares — worth roughly \$21 billion at the average price Alphabet paid — went out the door to employees while the buyback was bringing shares in. The buyback was real. The share reduction was real. But a third of the money bought nothing at all, in the sense that it purchased shares that were immediately re-created and handed to somebody else.

This is not fraud. Alphabet disclosed every figure in this paragraph in its [Form 10-K for the year ended 31 December 2024](https://www.sec.gov/Archives/edgar/data/1652044/000165204425000014/goog-20241231.htm), which is exactly where I got them. It is not even unusual — it is the normal operating pattern of most large technology companies, and something close to it happens at a large fraction of the S&P 500. But it is an *optic*: a set of true statements arranged so that the reader draws a conclusion the arrangement does not support. "We returned \$62 billion to shareholders" and "our share count fell 2%" describe the same year, and they feel like they should be the same size. They are not.

![An illustrative before-and-after showing that a stock grant transfers ownership from existing shareholders rather than costing the company cash](/imgs/blogs/stock-based-compensation-buybacks-and-eps-optics-1.webp)

The diagram above is the mental model for the whole article, and it is worth sitting with before we go anywhere near a filing. A company grants shares to its employees. No cash leaves the business. The business is worth exactly what it was worth a second ago. And yet something was definitely paid, because the employee is definitely richer. The payment came out of the only place it could have come from: the percentage of the company that everyone else owns. That is the entire mechanism, and everything else in this article — the add-backs, the treasury-stock method, the accretion arithmetic, the excise tax — is bookkeeping wrapped around it.

This article is the fourth in a sequence about metrics companies construct. If you have not read [non-GAAP and adjusted EBITDA: the metrics companies invent](/blog/trading/forensic-accounting/non-gaap-and-adjusted-ebitda-the-metrics-companies-invent), that piece covers the general machinery of adjustment; this one takes the single largest adjustment in modern corporate reporting and follows it all the way down.

## The foundations: how earnings per share is actually built

Before we can talk about how earnings per share gets manipulated, we need to be precise about what it is. If you already know, skim — but the precision matters later, because most of the tricks live in definitions people assume they understand.

### Earnings per share is a fraction, and fractions have two levers

**Earnings per share (EPS)** is a company's profit divided by the number of shares it has outstanding. If a company earns \$100 million and has 100 million shares, each share earned \$1.00.

$$\text{EPS} = \frac{\text{Net income available to common shareholders}}{\text{Weighted-average shares outstanding}}$$

The numerator, **net income**, is the profit left after every expense — cost of goods, salaries, rent, research, interest on debt, and tax. The denominator, **shares outstanding**, is how many slices the company has been cut into.

Here is the thing about fractions that drives everything below: there are two ways to make one bigger. Increase the top, or decrease the bottom. A company that earns 10% more with the same share count reports 10% higher EPS. A company that earns exactly the same money but retires 10% of its shares reports about 11% higher EPS. From the outside, in a headline, in an algorithmic screen, in a comparison to the analysts' consensus estimate, those two look identical.

They are not identical. The first is a better business. The second is the same business, divided differently.

### What stock-based compensation actually is

**Stock-based compensation (SBC)** is pay delivered in shares rather than in cash. It comes in a few flavors, and the differences matter:

- A **stock option** gives the employee the right, but not the obligation, to buy a share at a fixed price (the **strike price** or **exercise price**) at some future date. If the strike is \$10 and the stock later trades at \$25, the employee can buy at \$10 and immediately be \$15 better off. If the stock trades at \$8, the option is worthless and the employee simply does not exercise it.
- A **restricted stock unit (RSU)** is a promise to hand over an actual share once the employee has worked long enough to earn it (the **vesting period**, typically three to four years). Unlike an option, an RSU has value even if the stock falls, because the employee receives the share itself rather than the right to buy it. RSUs are now the dominant form of equity pay at large US technology companies.
- A **performance share unit (PSU)** is an RSU whose vesting depends on hitting a target — a revenue number, a total-shareholder-return ranking, or, importantly for this article, an EPS number.
- An **employee stock purchase plan (ESPP)** lets employees buy shares at a discount, usually 15%, through payroll deduction.

All four have the same economic shape: the company hands out claims on itself instead of handing out cash.

### What a buyback actually is

A **share buyback** (or **share repurchase**) is a company using its own money to buy its own shares on the open market and cancel them, or hold them in **treasury** — a kind of corporate limbo where the shares still legally exist but are not counted as outstanding and receive no dividends.

Economically, a buyback is a way of returning cash to shareholders. A dividend gives every shareholder cash and leaves their ownership percentage unchanged. A buyback gives cash to the shareholders who sell, and leaves the ones who stay owning a larger fraction of a slightly smaller company. In a world with no taxes and no mispricing, they are equivalent.

We do not live in that world, which is why buybacks are interesting.

### The one sentence that makes SBC a cost

Here is the argument that took the accounting profession thirty years to accept.

Suppose a company grants an employee shares worth \$100,000. Now suppose instead it had sold those same shares on the open market for \$100,000 and used the cash to pay the employee a bonus. The employee ends up in the identical position. The company ends up in the identical position — same shares in the world, same employee paid, same cash balance. Every other shareholder ends up in the identical position.

The two transactions are the same transaction. The second one obviously has a \$100,000 expense in it. So the first one does too.

Warren Buffett put it more bluntly in the 1992 Berkshire Hathaway chairman's letter, in a passage worth quoting in full because the whole industry spent a decade arguing with it:

> If options aren't a form of compensation, what are they? If compensation isn't an expense, what is it? And, if expenses shouldn't go into the calculation of earnings, where in the world should they go?

#### Worked example: what a grant costs you, in dollars

Let us make the transfer concrete with round, illustrative numbers.

You own 100 shares of a company that has 1,000 shares outstanding and is worth \$10,000 in total. Your ownership is 100 ÷ 1,000 = 10%, and the value of your stake is 10% × \$10,000 = \$1,000.

The company grants 100 new shares to its employees.

- Shares outstanding: 1,000 + 100 = **1,100**
- Company value: still **\$10,000**. No cash left the business, so the business is worth what it was worth.
- Your shares: still **100**
- Your ownership: 100 ÷ 1,100 = **9.09%**
- Your stake value: 9.09% × \$10,000 = **\$909**

You are \$91 poorer. Nothing happened to the company's bank account, its revenue, its factories, or its customers. The 100 new shares are collectively worth 100 ÷ 1,100 × \$10,000 = \$909, and that \$909 came from the existing shareholders in exact proportion to what they held.

**The intuition:** SBC is not free because it is non-cash. It is paid in ownership, and ownership is the thing you own.

There is one honest complication worth naming. In practice, granting equity is not always pure loss for the outside shareholder, because the employee does something in return — they build the product. The question is never "did this cost me something" (it did) but "did I get more than \$91 of value back". That is a real question with a real answer that varies by company. What is *not* a real question is whether a cost occurred.

## Why SBC is a real expense — and why it took until 2006 to say so

The accounting history here is not trivia. It explains why the modern presentation is shaped the way it is, and why so many investors still reason about SBC using a mental model that the rules abandoned twenty years ago.

### Three decades of arguing about one number

Under **APB Opinion No. 25**, the rule that governed US stock compensation from 1972, options were measured at their **intrinsic value** — the amount by which the stock price exceeded the strike price on the day of grant. A company that granted options with a strike equal to the current market price had, by this measure, granted something worth exactly zero. It recorded no expense at all.

This was obviously wrong, and everyone knew it was obviously wrong, because if at-the-money options were worthless nobody would have wanted them. But it was convenient, and it produced an era in which a company could pay a substantial fraction of its workforce entirely in an item that never touched the income statement.

The Financial Accounting Standards Board tried to fix this in 1995 with **SFAS 123**, which established fair-value measurement as *preferable* — and then, under heavy pressure from corporate lobbying, permitted companies to keep using APB 25 as long as they disclosed the fair-value effect in a footnote as pro-forma information. Almost every company chose the footnote.

The fix came in December 2004, when FASB issued **SFAS 123R, *Share-Based Payment***, which eliminated the APB 25 alternative and required that the compensation cost of share-based payments be recognized in the income statement at fair value. FASB set the effective date at the first interim or annual reporting period beginning after 15 June 2005; the SEC then deferred the compliance date for registrants to the first *fiscal year* beginning after that date. For most calendar-year filers, that meant fiscal 2006. When the standards were codified in 2009 the requirement moved to **ASC 718, *Compensation — Stock Compensation***, which is the citation you will see in modern filings and in SEC enforcement orders.

Read that date again. **Stock compensation has only been an income-statement expense in the United States since 2006.** Almost the entire body of investor folklore about "non-cash charges" predates the rule that made this one an expense at all.

### Where the \$100 goes

Now that SBC is an expense, follow it.

![A flow diagram tracing \$100 of stock-based compensation through the income statement, the cash flow statement, the balance sheet, and adjusted EBITDA, showing the two add-backs](/imgs/blogs/stock-based-compensation-buybacks-and-eps-optics-2.webp)

The figure traces one hundred dollars of SBC expense through the accounts. Follow each branch:

**The income statement.** The \$100 is allocated to whichever function the employee works in — cost of revenue, research and development, sales and marketing, or general and administrative. It reduces operating income by \$100 and net income by \$100 (ignoring tax effects, which complicate the picture without changing it). This is the one place where SBC behaves like any other expense.

**The cash flow statement.** The cash flow statement in its standard "indirect" form starts at net income and reverses out everything that reduced net income without moving cash. SBC is a textbook example, so a line reading "stock-based compensation \$100" is added straight back. Net effect on cash from operations: **zero**. This add-back is entirely correct — the cash flow statement's job is to report cash, and no cash moved. But it means that a company with enormous SBC and negative net income can report strongly positive operating cash flow, and both numbers are honest.

**The balance sheet.** The other side of the entry credits **additional paid-in capital** — the equity account that records money (or value) contributed by shareholders above par value. Additional paid-in capital rises \$100; retained earnings fall \$100 through the reduced net income. Total equity is unchanged. So the balance sheet, too, shows no scar.

**The non-GAAP presentation.** Here is where it stops being neutral. In most companies' quarterly investor decks, the same \$100 is added back a *second* time, in "adjusted EBITDA", "non-GAAP net income", "non-GAAP operating margin", or "adjusted EPS". After that second add-back, the adjusted figure is exactly where it started before the compensation existed.

Add up the branches and you get the uncomfortable conclusion: after the income statement, the cash flow statement, the balance sheet, and the adjusted metrics have all had their say, the only place the cost of SBC durably lives is **the number of shares**. Which is precisely the number a buyback is designed to move.

### The "it's non-cash so ignore it" fallacy, stated carefully

The defense of the second add-back usually runs: *SBC does not consume cash, so excluding it gives a cleaner view of the cash-generating capacity of the business.*

The first half of that sentence is true. The second half does not follow, for a reason that becomes obvious once you ask what happens next year. A company that pays engineers in stock must keep paying engineers in stock, or it must start paying them in cash. There is no third option in which the engineers work for free. So the SBC line is not a one-time item, not a legacy artifact, and not an accounting fiction — it is a recurring operating cost that the company has chosen to settle in a currency it prints itself.

The honest way to state the case for excluding it is narrower: *this metric answers the question "how much cash did operations throw off", and SBC does not belong in that answer.* Fine. But then you must ask the follow-up — how much did the shareholders pay for that cash — and the answer is in the share count.

#### Worked example: the GAAP-to-non-GAAP bridge, using real filings

Snowflake Inc. is a useful case because the numbers are large enough relative to the business that the mechanism is impossible to miss. Everything below comes from Snowflake's Forms 10-K; its fiscal year ends 31 January.

For **fiscal 2025** (the year ended 31 January 2025):

| Line | Amount |
| --- | --- |
| Revenue | \$3,626M |
| GAAP net loss | (\$1,286M) |
| Stock-based compensation | \$1,479M |
| Net cash provided by operating activities | \$960M |

Now build the bridge, one step at a time:

1. Start at the GAAP net loss: **(\$1,286M)**.
2. Add back stock-based compensation: (\$1,286M) + \$1,479M = **+\$193M**.
3. That is the entire distance from "loss-making" to "profitable". The add-back is not a component of the swing. It *is* the swing.

Two ratios make the scale legible:

- SBC ÷ revenue = \$1,479M ÷ \$3,626M = **40.8%**. For every dollar of revenue, forty-one cents of stock went to employees.
- SBC ÷ cash from operations = \$1,479M ÷ \$960M = **154%**. The stock issued to employees was worth half again as much as all the cash the business generated.

The following year, fiscal 2026 (ended 31 January 2026), the ratios improved but stayed extreme: revenue \$4,684M, SBC \$1,600M (34.2% of revenue), operating cash flow \$1,222M (SBC at 131% of it), GAAP net loss (\$1,332M).

**The intuition:** when SBC exceeds operating cash flow, "profitable on an adjusted basis" and "profitable" are describing different companies. Nothing here is hidden — Snowflake reports all of it — but a reader who takes the adjusted number at face value has silently agreed that a cost larger than the company's entire cash generation is not a cost.

For a fuller treatment of how these adjusted metrics are constructed and where the reconciliations hide, see [reading the income statement and the quality of earnings](/blog/trading/forensic-accounting/reading-the-income-statement-and-the-quality-of-earnings).

## The share count is the only honest scoreboard

If the cost of SBC ultimately settles in the share count, then the share count is where a forensic reader should look. This section is about reading it properly, which is harder than it sounds because there are at least four different share counts in every annual report and they do not agree with each other.

### Basic, diluted, and why the gap exists

**Basic shares outstanding** counts the shares that actually exist right now.

**Diluted shares outstanding** counts the shares that exist *plus* the shares that would exist if everyone holding a claim on future shares — option holders, RSU holders, convertible bondholders — converted those claims today. It is the pessimistic count, and it is the one that matters, because those claims will in fact be exercised if the company does well.

The gap between them is the size of the outstanding promises. In Alphabet's fiscal 2024, basic weighted-average shares were 12,319 million and diluted were 12,447 million — a gap of 128 million shares, about 1.0%.

Two further complications:

- Both figures are **weighted averages** over the year, not point-in-time counts. If a company repurchases heavily in December, the weighted average barely moves even though the year-end count drops sharply. This is why a buyback announced late in the year does almost nothing to that year's EPS and a lot to next year's.
- The **year-end** count — the one on the balance sheet and the cover of the 10-K — is a point-in-time number and will differ from both weighted averages. Forensic readers should look at all three and understand why they differ.

### The treasury-stock method: how options become shares on paper

The rule for turning outstanding options into diluted shares is the **treasury-stock method**, codified in the US in ASC 260, *Earnings Per Share*. It is elegantly cynical, and worth understanding precisely because it systematically understates dilution in one specific way.

The method assumes that when options are exercised, the company receives the exercise proceeds in cash and immediately uses all of that cash to buy back shares at the average market price during the period. Only the shortfall — the shares it could not buy back — counts as dilutive.

![The treasury-stock method worked at two share prices, showing that option dilution grows as the stock rises](/imgs/blogs/stock-based-compensation-buybacks-and-eps-optics-4.webp)

#### Worked example: the treasury-stock method at two prices

Illustrative arithmetic, with round numbers. A company has 1,000,000 basic shares and 100,000 employee options outstanding with a \$10 exercise price.

**Case A — the average market price during the year is \$25.**

1. Exercise proceeds if all options are exercised: 100,000 × \$10 = **\$1,000,000**
2. Shares the company could buy back with that money: \$1,000,000 ÷ \$25 = **40,000 shares**
3. Incremental dilutive shares: 100,000 − 40,000 = **60,000**
4. Diluted shares: 1,000,000 + 60,000 = **1,060,000**

Dilution: 6.0%.

**Case B — the average market price during the year is \$50.** The options are the same options. The strike is the same \$10.

1. Exercise proceeds: still 100,000 × \$10 = **\$1,000,000**
2. Shares buyable: \$1,000,000 ÷ \$50 = **20,000 shares**
3. Incremental dilutive shares: 100,000 − 20,000 = **80,000**
4. Diluted shares: 1,000,000 + 80,000 = **1,080,000**

Dilution: 8.0%.

**The intuition:** option dilution is not a fixed number of shares — it grows as the stock rises, because the fixed exercise proceeds buy back fewer and fewer shares. Exactly when a company is doing well and its investors are least inclined to worry, the dilution from its option overhang is quietly expanding.

Two refinements a careful reader should hold:

- **RSUs have no exercise proceeds.** An RSU holder pays nothing to receive the share. So under the treasury-stock method the offset is limited to any unrecognized compensation cost, and essentially the entire RSU count is dilutive. As the industry moved from options to RSUs over the last fifteen years, dilution became both larger and more predictable.
- **Anti-dilutive securities are excluded.** If including a security would *increase* EPS — an underwater option, for example — accounting rules require excluding it. This is correct in principle but means that in a bad year the reported diluted count understates the overhang that will reappear when the stock recovers.
- **In a loss-making year, diluted equals basic.** This follows from the rule above: when the numerator is negative, adding shares to the denominator makes the loss per share *smaller*, so every potential share is anti-dilutive and every one of them is excluded. A company reporting a GAAP net loss therefore publishes a "diluted" share count that contains no dilution at all. Snowflake is a clean example — its basic and diluted weighted-average counts are identical in each of fiscal 2023 through fiscal 2026 (318.7M, 328.0M, 332.7M, 337.5M), because it lost money in all four years. Every figure quoted for Snowflake later in this article is therefore the *undiluted* count, and the real overhang is larger than the series shows.

### The roll-forward: the single most useful table you will build

Every serious analysis of this topic ends up as the same four-line table. Beginning share count, plus issuance, minus repurchases, equals ending share count. The company will not present it this way. You have to build it.

![A share-count roll-forward for Alphabet's fiscal 2024 showing 12,460 million shares at the start, 379 million repurchased, 130 million issued, and 12,211 million at the end](/imgs/blogs/stock-based-compensation-buybacks-and-eps-optics-3.webp)

#### Worked example: Alphabet's fiscal 2024 share-count roll-forward

All figures from Alphabet Inc.'s Form 10-K for the year ended 31 December 2024 and the accompanying results release dated 4 February 2025.

The disclosed facts:

| Item | Figure |
| --- | --- |
| Shares issued and outstanding, 31 Dec 2023 | 12,460M |
| Shares issued and outstanding, 31 Dec 2024 | 12,211M |
| Shares repurchased and retired during 2024 | 379M |
| Value of stock repurchased and retired during 2024 | \$62,047M |
| Repurchases line in the cash flow statement | \$62,222M |
| Diluted weighted-average shares, 2023 | 12,722M |
| Diluted weighted-average shares, 2024 | 12,447M |
| Net income, 2024 | \$100,118M |
| Diluted EPS, 2024 | \$8.04 |

Two of those lines are close together and are not the same thing, which is worth pausing on because the distinction recurs across every company in this article. The **\$62,047 million** is the aggregate value of the shares repurchased and retired, reported in the statement of stockholders' equity and the repurchase note — it is the consideration for the 379 million shares, and so it is the figure to divide by a share count. The **\$62,222 million** is the *cash* line in the financing section, which differs because of settlement timing and because it also carries the buyback excise tax. Alphabet's own disclosure breaks that line into about \$61.8 billion of cash paid for repurchases plus \$447 million of excise tax paid in the fourth quarter. Use the first number for per-share arithmetic and the second when you are tracing cash.

Now the roll-forward:

1. Opening count: **12,460M**
2. Less shares repurchased and retired: **−379M** → 12,081M
3. The closing count is 12,211M. So shares issued during the year must be 12,211 − 12,081 = **+130M**
4. Closing count: **12,211M**

Net reduction for the year: 12,460 − 12,211 = **249M shares**, or 2.0%.

Now the cost arithmetic, which is where the optic lives:

- Average price paid per share repurchased: \$62,047M ÷ 379M = **\$163.71**
- Effective cost per share of *net* reduction: \$62,047M ÷ 249M = **\$249.19**
- Ratio: \$249.19 ÷ \$163.71 = **1.52×**

Alphabet paid roughly \$164 a share for the shares it bought, but each share of genuine reduction in the count cost about \$249 — half again as much — because 130 million of the 379 million shares it retired were immediately replaced by shares issued to employees. Roughly 34% of the buyback (130 ÷ 379) went to standing still.

Using the diluted weighted averages instead of the point-in-time counts gives a slightly gentler version of the same picture: the diluted count fell 275 million (12,722 → 12,447), for an effective \$62,047M ÷ 275M = **\$226** per net share retired, or 1.38× the average price paid. Weighted averages and point-in-time counts will never reconcile exactly, because the weighting spreads each transaction across the part of the year that remains; both versions are honest and both tell the same story.

**The intuition:** the buyback press release quotes the numerator. The share count is the denominator. Divide one by the other and you find out what a share of real reduction actually cost.

One last piece of arithmetic, because Alphabet is genuinely the *good* case here and it deserves saying:

- Actual fiscal 2024 diluted EPS: \$8.04, up from \$5.80 in 2023 — growth of **38.6%**
- Counterfactual EPS if the diluted share count had stayed at 12,722M: \$100,118M ÷ 12,722M = **\$7.87**, growth of **35.7%**

So of Alphabet's 38.6 percentage points of EPS growth, about 2.9 points came from the smaller share count and about 35.7 came from the business earning more money. That is a company whose earnings growth is overwhelmingly real, and whose buyback is a modest supplement. Keep that ratio in mind — it is the benchmark against which the next few cases should be read.

## The anti-dilutive buyback: paying cash to stand still

There is a category of buyback that has a name inside companies and almost no visibility outside them: the **anti-dilutive buyback**, whose stated purpose is to offset the dilution from employee equity. Its economics are worth being blunt about.

If a company issues \$1 billion of stock to employees and then spends \$1 billion buying stock back, the combined transaction is: \$1 billion of cash left the company, employees received \$1 billion of value, and the share count is unchanged. That is a cash bonus. It is a cash bonus routed through the equity market, reported as a non-cash expense on the income statement, added back in operating cash flow, added back again in adjusted EBITDA, and then presented in the capital-returns slide as money returned to shareholders.

Every step of that is legal and disclosed. The net effect is still a cash bonus that appears in none of the places a reader looks for cash bonuses.

![A comparison table of Alphabet, Salesforce and Snowflake showing buyback dollars, diluted share counts, net change, cost per net share retired, and SBC as a percentage of revenue](/imgs/blogs/stock-based-compensation-buybacks-and-eps-optics-5.webp)

The table above puts three companies side by side, all figures from their Forms 10-K. The spread is the point: the same activity — "we are buying back stock" — produces wildly different results depending on how much stock is going out the other door.

#### Worked example: Salesforce — \$32.0 billion for 41 million shares

Salesforce, Inc. began repurchasing shares in fiscal 2023 under pressure from activist investors. Its fiscal year ends 31 January. From its Forms 10-K:

| Fiscal year | Buybacks | SBC | Diluted weighted-average shares |
| --- | --- | --- | --- |
| FY2022 (ended Jan 2022) | \$0M | \$2,779M | 974M |
| FY2023 | \$4,000M | \$3,279M | 997M |
| FY2024 | \$7,620M | \$2,787M | 984M |
| FY2025 | \$7,829M | \$3,183M | 974M |
| FY2026 (ended Jan 2026) | \$12,596M | \$3,509M | 956M |

The arithmetic:

1. Cumulative buybacks, FY2023 through FY2026: \$4,000M + \$7,620M + \$7,829M + \$12,596M = **\$32,045M**, call it \$32.0 billion.
2. Diluted weighted-average shares, FY2023 to FY2026: 997M → 956M. Net reduction: **41M shares**, or 4.1%.
3. Effective cost per net share retired: \$32,045M ÷ 41M = **\$782**.

Note carefully what \$782 is and is not. It is not a price Salesforce paid — nobody paid \$782 for a Salesforce share. It is the total cash divided by the net shares removed, and the gap between it and the actual market price is a direct measure of how much of the program went to absorbing issuance rather than reducing the count.

The comparison against the pre-buyback baseline is starker still. In FY2022, before the program started, the diluted count was 974 million. Four years and \$32.0 billion later, it was 956 million — a net reduction of **18 million shares**, less than 2%. Over the same four years the company recognized \$12,758 million of stock-based compensation.

**The intuition:** the right denominator for "how much did the buyback cost per share retired" is the *net* change in the count, and when issuance is large that denominator can be a small fraction of the shares actually purchased.

#### Worked example: Snowflake — \$3.4 billion, and the count went up

![A bar chart of Snowflake's annual buyback spending against its diluted share count, showing the share count rising from 319 million to 337 million despite \$3.4 billion of repurchases](/imgs/blogs/stock-based-compensation-buybacks-and-eps-optics-6.webp)

The limiting case. From Snowflake's Forms 10-K, fiscal years ending 31 January:

| Fiscal year | Buybacks | SBC | Diluted weighted-average shares |
| --- | --- | --- | --- |
| FY2023 | \$0M | \$862M | 319M |
| FY2024 | \$592M | \$1,168M | 328M |
| FY2025 | \$1,932M | \$1,479M | 333M |
| FY2026 | \$874M | \$1,600M | 337M |

1. Cumulative buybacks, FY2024 through FY2026: \$592M + \$1,932M + \$874M = **\$3,398M**, or \$3.40 billion.
2. Diluted weighted-average shares, FY2023 to FY2026: 319M → 337M.
3. Net change: **+18 million shares**. The count rose 5.6%.

Cost per net share retired: undefined, because no shares were net retired. The company spent \$3.4 billion and finished with more shares than it started with.

**The intuition:** a buyback is not a share reduction. It is a bid against issuance, and the bid can lose.

This is not a claim that Snowflake did anything improper — the company discloses its repurchase program, its SBC, and its share counts in every filing, and it has never claimed the program shrank the count. The point is about what a reader infers from a headline number. "Snowflake repurchased \$1.9 billion of stock" is true of fiscal 2025 and, standing alone, implies something that did not happen.

## EPS accretion arithmetic versus value creation

Companies describe buybacks as "accretive to earnings per share", and analysts model the accretion, and the accretion is usually correct. It is also nearly meaningless as a test of whether the buyback was a good idea. This section explains why, because the confusion between the two is where a great deal of shareholder money goes to die.

### The accretion test, in one line

A buyback increases EPS whenever the earnings the company gives up to fund it are less than the earnings attributable to the shares it retires.

If the buyback is funded with cash sitting on the balance sheet, the earnings given up are the after-tax interest that cash was producing. If it is funded with debt, the earnings given up are the after-tax interest on the new debt. Either way:

$$\text{Buyback is accretive to EPS} \iff \text{Earnings yield} > \text{After-tax cost of funds}$$

where **earnings yield** is EPS divided by the share price — the inverse of the price-to-earnings ratio. A stock at 20× earnings has a 5% earnings yield. A stock at 50× earnings has a 2% earnings yield.

![An illustrative comparison of a buyback funded with cash and the same buyback funded with debt, showing EPS accretion under both and the rule that accretion depends on earnings yield versus the after-tax cost of funds](/imgs/blogs/stock-based-compensation-buybacks-and-eps-optics-7.webp)

#### Worked example: the accretion test, run both ways

Illustrative arithmetic. A company earns \$100 million a year, has 100 million shares, and trades at \$20 a share.

- EPS: \$100M ÷ 100M = **\$1.00**
- Price-to-earnings ratio: \$20 ÷ \$1.00 = **20×**
- Earnings yield: \$1.00 ÷ \$20 = **5.0%**

It repurchases \$200 million of stock — 10 million shares at \$20 — leaving 90 million shares.

**Funded with balance-sheet cash earning 2% after tax:**

1. Forgone interest income: \$200M × 2% = **\$4M**
2. New net income: \$100M − \$4M = **\$96M**
3. New EPS: \$96M ÷ 90M = **\$1.067**
4. Accretion: +6.7%

**Funded with new debt at 6% pre-tax, 4.5% after tax:**

1. New interest expense, after tax: \$200M × 4.5% = **\$9M**
2. New net income: \$100M − \$9M = **\$91M**
3. New EPS: \$91M ÷ 90M = **\$1.011**
4. Accretion: +1.1%

Both are accretive, because in both cases the 5.0% earnings yield exceeds the after-tax cost of funds. Now notice what the calculation never asked: **whether \$20 was a sensible price to pay.** The arithmetic works identically if the shares are worth \$8 and identically if they are worth \$40. Accretion is a statement about two interest rates. It is not a statement about value.

**The intuition:** every buyback of a stock trading below roughly the reciprocal of the company's cost of funds is accretive, including all the terrible ones.

### What actually creates value

A buyback creates value for the continuing shareholders under exactly one condition: the shares were bought for less than they were worth. That is it. The continuing shareholders' gain is the difference between intrinsic value and price, multiplied by the shares retired, and it comes directly out of the pockets of the shareholders who sold.

Which produces an uncomfortable corollary. A buyback is a transaction between the company and its exiting shareholders, negotiated by management, in which management has vastly better information than the counterparty. If management buys well, continuing holders gain at the expense of sellers. If management buys badly, the sellers won. There is no version in which everybody wins, and the direction depends entirely on price.

### Buying high: the timing problem is structural

If buybacks created value only when management bought cheap, you would expect buyback spending to peak when stocks are cheap. It does the opposite, and the reason is not stupidity — it is that buybacks are funded from cash flow and authorized by boards, and both are most abundant exactly when business is best and stocks are most expensive.

According to S&P Dow Jones Indices (press release of 19 March 2025), S&P 500 buyback expenditure reached a trailing-twelve-month peak of **\$1.005 trillion in June 2022** — in the middle of a year in which the index fell sharply. Full-year 2024 set a new annual record of **\$942.5 billion**, up 18.5% from \$795.2 billion in 2023, near the top of a two-year advance. S&P Dow Jones Indices data likewise show that buybacks peaked around **\$589 billion in 2007** and collapsed to roughly **\$138 billion in 2009** — the corporate sector spending most aggressively immediately before the worst equity market in eighty years, and least aggressively at its bottom.

That pattern is procyclical by construction. A company generating record cash in a boom has record cash to spend, and a board approving a repurchase authorization in a good year is approving it at good-year prices.

### The debt-funded case, and where it ends

Funding a buyback with debt converts a flexible obligation (equity, which can pay nothing) into a rigid one (interest, which must be paid). At moderate leverage this is defensible — debt is cheaper than equity and interest is tax-deductible. At high leverage it removes the company's ability to absorb a bad year.

Oracle Corporation ran the aggressive version, and its filings show where it goes. From Oracle's Forms 10-K (fiscal years ending 31 May):

| Fiscal year | Buybacks | Cash from operations | Diluted shares | Total stockholders' equity |
| --- | --- | --- | --- | --- |
| FY2018 | \$11,347M | \$15,386M | 4,238M | \$46,372M |
| FY2019 | \$36,140M | \$14,551M | 3,732M | \$21,785M |
| FY2020 | \$19,240M | \$13,139M | 3,294M | \$12,074M |
| FY2021 | \$20,934M | \$15,887M | 3,022M | \$5,238M |
| FY2022 | \$16,248M | \$9,539M | 2,786M | (\$6,220M) |

In fiscal 2019 alone Oracle spent \$36.1 billion on buybacks against \$14.6 billion of operating cash flow — a ratio of about 2.5× — and \$11.1 billion of net income. Over five years the diluted share count fell from 4,238 million to 2,786 million, a genuine and very large **34% reduction**. Total stockholders' equity went from \$46.4 billion to *negative* \$6.2 billion, because a company that repurchases more than it earns eventually buys back its own book value.

Negative equity is not automatically alarming for a business with durable cash flows — it is an accounting consequence, not an insolvency test. But it is the marker of a company that has spent its financial flexibility, and it constrains what happens next.

What happened next is the most instructive part. Oracle's buybacks collapsed: \$1,300M in FY2023, \$1,202M in FY2024, \$600M in FY2025, \$95M in FY2026. And the diluted share count promptly went back up — 2,766M in FY2023 to **2,914M in FY2026**, a rise of 148 million shares, while SBC grew from \$3,547M to \$4,811M over the same span.

**The intuition:** the buyback was not reducing the share count so much as suppressing it. Remove the buyback and the underlying issuance reappears immediately, because it never stopped.

## The incentive: why management wants the number to move

None of this happens by accident. To understand a reporting pattern, look at what the people producing it are paid for, which is a question the proxy statement answers directly and in public.

### EPS as a bonus metric

Executive incentive plans pay out against metrics. When one of those metrics is earnings per share, management is being paid, in part, for moving a fraction — and a fraction has two levers.

FW Cook's *2024 Top 250 Annual Incentive Plan Report* — the standard annual survey of pay practice at the largest US public companies — places earnings per share among the most commonly used financial metrics in annual incentive plans, at roughly one company in five. That proportion comes from secondary summaries of the report rather than from a figure I could re-verify in the report itself, so read it as an order of magnitude rather than a precise share. The order of magnitude is the whole point: for a large minority of big US companies, a fraction that a buyback mechanically improves sits directly in the bonus formula.

Well-designed plans anticipate this. A compensation committee that is paying attention will either adjust the EPS target for the effect of repurchases, define the metric on a fixed share count, or use a metric that a buyback cannot move — return on invested capital, revenue growth, operating income, or relative total shareholder return. **Whether the plan contains that adjustment is a fact you can look up**, in the Compensation Discussion and Analysis section of the annual proxy statement (Form DEF 14A). It takes about ten minutes and it is the single highest-information-per-minute item in this entire article.

### The evidence that this changes behavior

The suspicion that managers buy back stock to hit EPS targets is not merely a suspicion; it has been tested carefully, three times, over twenty years, with the results pointing the same way.

Daniel Bens, Venky Nagar, Douglas Skinner and Franco Wong took the first careful pass in "Employee stock options, EPS dilution, and stock repurchases" (*Journal of Accounting and Economics*, volume 36, 2003, pages 51–91). They found that repurchase activity rises with the dilutive overhang from employee stock options and with the size of the EPS shortfall a firm is facing — the two forces this article has been describing separately turn out to be a single, measurable behaviour in the data.

Paul Hribar, Nicole Jenkins and Bruce Johnson sharpened it in "Stock repurchases as an earnings management device" (*Journal of Accounting and Economics*, volume 41, 2006, pages 3–27). Their finding is the cleanest statement of the mechanism: EPS-accretive repurchases cluster among firms that would otherwise have missed the consensus estimate. The repurchases are not randomly distributed with respect to the analyst forecast; they show up exactly where a penny is needed.

In "The real effects of share repurchases" (*Journal of Financial Economics*, volume 119, issue 1, 2016, pages 168–185), Heitor Almeida, Vyacheslav Fos and Mathias Kronlund used a regression discontinuity design around the point where a firm would just barely miss the consensus EPS forecast. Their finding: the probability of an EPS-increasing repurchase is sharply higher for firms that would have just missed the forecast without it. And those particular repurchases are associated with **reductions in employment and investment**.

That second half is what makes the result matter. It is not simply that managers reallocate cash toward buybacks when they need a penny; it is that they appear willing to trade real investment and real headcount for the accounting outcome. The buyback is not free. It is funded, at the margin, by the things that would have produced next year's earnings.

#### Worked example: buying your way past a penny

Illustrative arithmetic showing how little it takes.

A company is on track for net income of \$500.0 million with 500.0 million diluted shares. That is EPS of \$1.000. Consensus is \$1.01, so it is about to miss by a penny — the kind of miss that moves a stock several percent and costs a bonus.

Management wants reported EPS of at least \$1.01. Solving for the required share count:

1. Required shares: \$500.0M ÷ \$1.01 = **495.05M shares**
2. Shares to retire: 500.0M − 495.05M = **4.95M shares**
3. At a \$40 share price, cost: 4.95M × \$40 = **\$198M**

But the shares must be retired early enough in the period to move the *weighted average*. Shares repurchased with one month left in the year contribute only about one-twelfth of their effect to the annual weighted average, so an equivalent late repurchase would need roughly twelve times as many shares — an implausible \$2.4 billion. This is why EPS-motivated repurchases cluster earlier in the quarter and the year, and why the timing of a repurchase relative to the period-end is itself a signal worth noticing.

Also note step 1 assumed net income is fixed. It is not, quite: spending \$198 million of cash forgoes some interest income, which reduces net income slightly and requires retiring slightly more shares. The circularity is small here but it is the reason careful models solve for it iteratively.

**The intuition:** closing a one-cent gap on a \$1.00 EPS costs about 1% of the share count, which for most companies is a rounding error against an existing authorization — the cheapest earnings management available to a company that has already announced a buyback.

## The 2023 excise tax, and what it accidentally revealed

In August 2022 the United States enacted the Inflation Reduction Act, which added **Section 4501** to the Internal Revenue Code: a **1% excise tax on the fair market value of stock repurchased** by publicly traded US corporations, applying to repurchases after **31 December 2022**. The Internal Revenue Service published final regulations implementing it on **24 November 2025**.

The tax itself is small. What is interesting is how Congress wrote it.

### The netting rule is the thesis of this article, in statute

Section 4501(c)(3) contains what practitioners call the **netting rule**: the amount subject to the excise tax is reduced by the fair market value of stock the corporation *issues* during the same taxable year — explicitly including stock issued or provided to employees, or to employees of a specified affiliate.

Read that again in plain English. The US tax code, when it needed to define how much stock a company had really repurchased, did not use the gross figure. It used repurchases minus issuance. The drafters of a revenue statute independently arrived at the same conclusion this article has been arguing from the accounting side: **gross buyback dollars do not measure anything; the net change does.**

There is a straightforward reason for it — Congress did not want to tax a company for buying back the shares it had just handed to its own staff — but the implication stands regardless of motive. When real money is at stake, the gross number is not the one anyone uses.

### How big is the bite

Small, so far. S&P Dow Jones Indices calculated (press release of 19 March 2025) that the 1% tax on net buybacks reduced S&P 500 operating earnings by **0.44% for full-year 2024** and 0.37% in the fourth quarter alone.

You can see it directly in a filing. Alphabet's fiscal 2024 cash flow statement shows a repurchases line of \$62,222 million, which its own footnote breaks out as \$61.8 billion of cash paid for repurchases in the full year plus **\$447 million of excise tax payments** made during the fourth quarter of 2024.

At 1%, the tax is a friction, not a deterrent. A company that believes its shares are cheap will pay it without much thought, and a company running an anti-dilutive program will treat it as a small increase in the cost of standing still. Various proposals to raise the rate have circulated since 2023; Alphabet's own 10-K risk factors note that the 1% rate "could potentially increase in the future". If it ever rose to a level that changed behavior, the netting rule means the burden would fall hardest on exactly the companies described in this article — the ones repurchasing heavily while issuing heavily would owe tax on a base that nets down, while a company doing a clean, non-offsetting reduction would pay on nearly the full amount.

## Where the legal line actually is

This article has said "this is legal" several times, which invites the obvious question: what would not be? The answer is narrower than most readers expect, and knowing where the boundary sits is what keeps the six tests below in proportion.

Repurchases themselves are governed by **Rule 10b-18**, adopted in 1982 and amended on 10 November 2003. It is a *non-exclusive safe harbour*: a company that keeps its repurchases inside the rule's conditions on manner, timing, price and volume will not be deemed to have manipulated the market by reason of those purchases alone. That is all it does. It confers no protection for trading on material non-public information, no protection against a manipulation charge built on other conduct, and no protection for how the buyback is described to investors. A company can be comfortably inside 10b-18 and still be in trouble.

What you actually get to see is set by **Item 703 of Regulation S-K**, which requires the repurchase table in periodic reports — aggregated by *month*. The SEC tried to replace that with day-by-day data in the Share Repurchase Disclosure Modernization rule, adopted 3 May 2023. The Fifth Circuit vacated it, effective 19 December 2023, and the Commission published technical amendments on 8 April 2024 reverting the rule text to what existed before. So the daily detail does not exist, and the monthly aggregate is what a forensic reader has to work with. Anyone citing that rule as current requirements is a rule behind.

Two enforcement actions mark the boundary, and what is instructive about both is what they were *not* charged as. In November 2023 the SEC settled with **Charter Communications** for a **\$25 million** penalty over its stock buybacks: from 2017 to 2021 Charter used nine 10b5-1 trading plans containing "accordion" provisions that let it change the dollar amounts and timing after the plans took effect, which meant the plans did not satisfy Rule 10b5-1 and the buybacks did not match what the board had authorised. The charge was a violation of the internal accounting controls requirement of Exchange Act Section 13(b)(2)(B). In January 2025 the SEC settled with **Celsius Holdings** for **\$3 million** over stock-based compensation: the company extended vesting for departing employees, which ASC 718 treats as a modification requiring the awards to be re-valued, and it did not account for the modifications correctly. The charges were Sections 13(a), 13(b)(2)(A) and 13(b)(2)(B) and the related reporting rules.

Neither case was charged as fraud. Both were books-and-records, reporting and internal-controls matters — the SEC's finding was that the companies had failed to build the machinery that would have kept the accounting right, not that anyone set out to deceive. That is precisely the register this whole subject lives in. The optics described in this article are not on the wrong side of the line; the line is somewhere else entirely, and it is mostly about controls. Which is exactly why the reader has to do the arithmetic — nobody is going to be charged for the thing that is actually costing you money.

## How to detect it: six tests from the filings

Everything above reduces to a short procedure. All six tests use public documents, and five of them use numbers that are printed in the annual report without any adjustment.

![A six-row detection dashboard listing each test, what to compute, and the red-flag threshold](/imgs/blogs/stock-based-compensation-buybacks-and-eps-optics-9.webp)

### Test 1 — plot the diluted share count over five years

**Where:** the income statement of each Form 10-K, the line "diluted weighted-average shares outstanding" (or the EPS note).

**What:** just the series. Five annual figures, in order.

**Red flag:** the count is flat or rising while the company describes a substantial buyback program. If the line does not go down, the buyback did not reduce anything, no matter how many dollars were spent.

This is the whole article in one test. If you do only one, do this one. For scale: S&P Dow Jones Indices reported (press release of 19 March 2025) that in the fourth quarter of 2024 only **11.9% of S&P 500 companies** had reduced their EPS share count by at least 4% year over year — in a year of record \$942.5 billion of aggregate buybacks. The overwhelming majority of that money did not move share counts materially.

### Test 2 — SBC as a percentage of revenue

**Where:** the cash flow statement, the add-back line "stock-based compensation". Divide by revenue from the income statement.

**Red flag:** above 10% deserves attention; above 25% means the equity issuance is a first-order feature of the business model, not a detail. For calibration, each pair below is taken from that company's Form 10-K for the fiscal year named: IBM, \$468M of SBC on \$81,741M of revenue for 2015 (0.6%); Alphabet, \$22,785M on \$350,018M for fiscal 2024 (6.5%); Salesforce, \$3,509M on \$41,525M for fiscal 2026 (8.4%); Snowflake, \$1,600M on \$4,684M for fiscal 2026 (34.2%) and \$1,479M on \$3,626M for fiscal 2025 (40.8%).

### Test 3 — SBC against cash generation

**Where:** the same SBC line, divided by net cash provided by operating activities.

**Red flag:** above 30% is meaningful. If SBC exceeds operating cash flow — as at Snowflake in fiscal 2025, where \$1,479M of SBC ran against \$960M of operating cash flow for 154%, per its Form 10-K for the year ended 31 January 2025 — then the entirety of the company's "adjusted profitability" is the add-back, and you should evaluate the company on the GAAP figures or not at all.

### Test 4 — the GAAP-to-non-GAAP EPS gap, tracked over time

**Where:** the quarterly earnings release reconciliation. Regulation G — adopted 22 January 2003, effective 28 March 2003 — requires a reconciliation to the most directly comparable GAAP measure whenever a non-GAAP measure is publicly disclosed, and Item 10(e) of Regulation S-K imposes the parallel requirement inside filings such as the 10-K and 10-Q. The staff's Compliance and Disclosure Interpretations, updated in May 2016 and again on 13 December 2022, govern which adjustments are permissible.

**What:** non-GAAP EPS minus GAAP EPS, each year, and that difference as a percentage of GAAP EPS.

**Red flag:** a gap that widens year after year, or one that exceeds roughly 30% of GAAP EPS. A stable gap that consists of genuinely non-recurring items is defensible. A growing gap made mostly of SBC is the company telling you its compensation cost is growing faster than its profit.

### Test 5 — buyback efficiency

**Where:** the financing section of the cash flow statement (cash spent on repurchases) and the share-count series from Test 1.

**What:** buyback dollars divided by the net reduction in shares. Compare the result to the average market price over the period.

**Red flag:** a ratio above about 1.3×. At 1.0× the buyback retired every share it bought. At 1.5×, as with Alphabet in fiscal 2024, a third of the program went to offsetting issuance. Above 3× the program is essentially an issuance-absorption facility with a share reduction attached, and if the net change is zero or positive — Snowflake — the ratio does not exist.

### Test 6 — read the proxy

**Where:** the Compensation Discussion and Analysis in the annual proxy statement, Form DEF 14A.

**What:** the list of metrics in the annual bonus and the performance-share plan.

**Red flag:** an EPS or adjusted-EPS target with no stated adjustment for the effect of share repurchases. That is a plan that pays management for buying back stock, whether or not buying back stock was the right use of the money.

#### Worked example: running all six tests on one company

Take Alphabet's fiscal 2024 — every figure from its Form 10-K for the year ended 31 December 2024, assembled above — and run the numbers as a reader would.

1. **Share count trend:** diluted weighted-average shares 13,159M (2022) → 12,722M (2023) → 12,447M (2024) → 12,230M (2025). Falling steadily, about 2% a year. **Pass.**
2. **SBC ÷ revenue:** \$22,785M ÷ \$350,018M = **6.5%**. Below the 10% threshold, and roughly flat as revenue grew — SBC rose only 1.4% in 2024 (\$22,460M → \$22,785M) while revenue rose 13.9%. **Pass, and improving.**
3. **SBC ÷ operating cash flow:** \$22,785M ÷ \$125,299M = **18.2%**. Below 30%. **Pass.**
4. **Non-GAAP gap:** Alphabet does not present a non-GAAP EPS at all — its only non-GAAP measures are free cash flow and constant-currency revenue. There is no gap to widen. **Pass, notably.**
5. **Buyback efficiency:** \$62,047M ÷ 249M net shares = **\$249**, against \$163.71 average paid — a ratio of **1.52×**. This is the one test Alphabet does not clear comfortably: about a third of the program offsets issuance.
6. **Proxy metrics:** a check of the compensation discussion in the DEF 14A for whether any EPS-linked target exists and whether it is share-count-adjusted.

Five of six clean, one amber. That is what a company with large but well-controlled equity compensation looks like from the outside, and it is a useful reference point — the tests are not designed to find villains, they are designed to size a cost.

**The intuition:** the tests take about twenty minutes per company and produce a number, not a verdict. What you do with a 1.5× buyback efficiency ratio depends on what you are paying for the stock.

## Common misconceptions

**"SBC is non-cash, so it doesn't affect valuation."** It affects valuation through the share count, which is the denominator of every per-share figure you use. If you value a company at \$100 billion and then divide by a share count that grows 3% a year, your per-share value falls 3% a year. Practitioners handle this by either treating SBC as a cash expense in the cash flow forecast or by explicitly forecasting future dilution — never by ignoring it. Doing both is double-counting; doing neither is the standard error.

**"A buyback returns cash to shareholders, so it's like a dividend."** It returns cash to the shareholders who *sell*. If you hold, you receive nothing — your share of a slightly smaller company simply becomes slightly larger. Whether that is worth more than the cash depends entirely on whether the price paid was below intrinsic value. A dividend has no such dependency, which is why dividends are boring and buybacks are contentious.

**"EPS accretion means the buyback was a good use of money."** Accretion means the earnings yield exceeded the after-tax cost of funds. As the worked example above shows, that condition is satisfied by essentially every buyback of a stock trading at a normal multiple with normal interest rates, including the ones destroying value. A buyback of overvalued stock funded with cheap debt is accretive to EPS *and* value-destructive, simultaneously and without contradiction.

**"Diluted share count already accounts for the dilution, so I don't need to worry about it."** Diluted share count accounts for the claims outstanding *today*. It does not account for the grants the company will make next year, which for a company running SBC at 10% of revenue will be substantial. The diluted count is a snapshot of the overhang, not a forecast of it.

**"The company bought back more shares than it issued, so dilution isn't a problem."** Possibly, but check what it cost. If a company issued 100 million shares to employees and repurchased 150 million, the count fell by 50 million — and the cash spent was three times what a pure 50-million reduction would have cost. The reduction was real; two-thirds of the money still went to compensation. Test 5 exists to size exactly this.

**"Companies buy back stock when management thinks it's cheap."** Sometimes. In aggregate, buybacks peak with cash flow and with the market — the S&P 500 trailing-twelve-month record of \$1.005 trillion was set in June 2022 and the annual record of \$942.5 billion in 2024, while the 2009 trough of roughly \$138 billion came at the market's low. The aggregate pattern is procyclical, which is the opposite of buying cheap.

**"High SBC is just how technology companies work, so it's priced in."** "Priced in" is doing a lot of work in that sentence. The cost is disclosed, and sophisticated investors do adjust for it; the question is whether *you* have. The test is mechanical — compute the per-share value after forecast dilution, not before — and it takes an afternoon.

## How it shows up in real markets

### IBM and Roadmap 2015: the shrinking denominator

![An indexed chart of IBM from 2011 to 2015 showing revenue falling 24%, the diluted share count falling 19%, and diluted EPS rising 3%](/imgs/blogs/stock-based-compensation-buybacks-and-eps-optics-8.webp)

In 2010, under then-CEO Sam Palmisano, IBM set out a five-year plan that came to be known as Roadmap 2015, whose headline commitment was operating (non-GAAP) earnings per share of "at least \$20" by 2015, supported by a large program of returning cash through dividends and repurchases. As reported at the time and widely covered since, CEO Ginni Rometty abandoned the \$20 target on 20 October 2014, alongside a disappointing third-quarter result.

The intervening years are one of the cleanest natural experiments available. From IBM's Forms 10-K:

| Year | Revenue | Diluted shares | Diluted EPS | Buybacks |
| --- | --- | --- | --- | --- |
| 2011 | \$106,916M | 1,214M | \$13.06 | \$15,046M |
| 2012 | \$102,874M | 1,155M | \$14.37 | \$11,995M |
| 2013 | \$98,367M | 1,103M | \$14.94 | \$13,859M |
| 2014 | \$92,793M | 1,010M | \$11.90 | \$13,679M |
| 2015 | \$81,741M | 983M | \$13.42 | \$4,609M |

Cumulative buybacks 2011–2015: \$59,188 million, or **\$59.2 billion**.

Over those five years:

- Revenue fell from \$106.9B to \$81.7B — **down 23.5%**
- Diluted share count fell from 1,214M to 983M — **down 19.0%**
- Diluted EPS went from \$13.06 to \$13.42 — **up 2.8%**

Now the counterfactual. IBM's 2015 net income was \$13,190 million. Had the share count remained at its 2011 level of 1,214 million:

$$\text{2015 EPS at a flat share count} = \frac{\$13{,}190\text{M}}{1{,}214\text{M}} = \$10.86$$

So reported 2015 EPS of \$13.42 contains **\$2.56 — about 19% of it — supplied by the share count rather than the business**. Against 2011's \$13.06, the flat-count figure of \$10.86 is a decline of 16.8%. The reported figure was an increase of 2.8%. A shareholder reading only EPS saw a company treading water. A shareholder reading revenue saw one losing a quarter of itself.

Worth noting what this case is *not*: IBM's SBC was tiny — \$468 million in 2015, 0.6% of revenue. This was not an anti-dilutive buyback mopping up issuance. It was a genuine, very large reduction in the share count, executed exactly as announced. That is what makes it instructive. Even a completely real share reduction, honestly disclosed and fully delivered, can hold a per-share metric flat while the business underneath it contracts. The optic does not require any issuance games at all.

### Alphabet: the same mechanics, the opposite conclusion

Alphabet's fiscal 2024 is the counter-case, and running both through the same tests is the best way to see that these are diagnostics rather than accusations. Revenue grew 13.9% to \$350.0 billion. Net income grew 35.7% to \$100.1 billion. SBC grew 1.4% to \$22.8 billion — meaning SBC as a share of revenue *fell*. The company presents no non-GAAP EPS at all. The share count fell about 2%.

And still, Test 5 shows \$62.0 billion buying 249 million net shares at an effective \$249 against \$163.71 paid. The issuance offset is real and large in absolute terms — roughly \$21 billion of stock at the average repurchase price. The difference from the other cases is proportion: at Alphabet the buyback is a supplement to enormous genuine earnings growth, and about 2.9 of the 38.6 percentage points of EPS growth came from the denominator. At IBM, the denominator was carrying the number.

### High-SBC software: when the add-back is the profit

The Snowflake numbers above are a category, not an outlier. A software company early in its life competes for engineers against companies that pay in stock, so it pays in stock; the expense lands on the income statement, gets added back in cash flow, gets added back again in the adjusted metrics presented to investors, and the share count absorbs it. Fiscal 2025: revenue \$3,626M, GAAP net loss (\$1,286M), SBC \$1,479M, operating cash flow \$960M. Buybacks of \$1,932M that year against a diluted share count that rose from 328M to 333M.

The forensic point is not that this is illegitimate. It is that "adjusted operating margin" for such a company is a number with the largest cost removed, and the reader who compares that margin to a mature software company's GAAP margin is comparing two different quantities with the same name.

### Salesforce: activism, buybacks, and what actually changed

Salesforce's repurchase program began in fiscal 2023 amid activist pressure over margins and capital allocation, and the program was large and genuine — \$32.0 billion over four fiscal years, rising to \$12.6 billion in fiscal 2026 alone. The diluted share count did come down, from 997 million to 956 million.

What the four-year picture shows is the scale mismatch. Against \$32.0 billion of repurchases sat \$12,758 million of stock-based compensation over the same four years. The company also grew — revenue went from \$31.4 billion to \$41.5 billion — so this is not a story of a shrinking business. It is a story about how much cash it takes to move a share count when a large equity-compensation program is running in the opposite direction, and about the difference between a headline of "\$12.6 billion returned" and an outcome of 18 million shares fewer than before the program started.

### Oracle: what happens when the buyback stops

Oracle's arc, laid out in the table earlier, is the clearest available demonstration that a buyback suppresses a share count rather than solving it. Five years of very aggressive repurchase — \$36.1 billion in fiscal 2019 alone, 2.5× that year's operating cash flow — took the diluted count down 34%, from 4,238 million to 2,786 million, and took stockholders' equity from \$46.4 billion to negative \$6.2 billion.

Then the program wound down to \$95 million by fiscal 2026, and the share count rose to 2,914 million — up 148 million from the fiscal 2023 trough — while SBC climbed from \$3,547 million to \$4,811 million. The underlying issuance had been there the whole time. The buyback was the thing standing on top of it.

The rest of the story confirms the reading. With repurchases switched off and retained earnings accumulating, stockholders' equity climbed straight back out of its hole — negative \$6.2 billion in fiscal 2022, then \$1.1 billion, \$8.7 billion, \$20.5 billion and \$42.5 billion by fiscal 2026. Oracle was never insolvent; it had simply spent its equity buying its own shares, and it stopped. The share count went up the moment it did.

### The aggregate: a record year that moved few share counts

The market-wide version is the same pattern at scale. In 2024 the S&P 500 spent a record \$942.5 billion on buybacks, up 18.5% from \$795.2 billion in 2023, and in the fourth quarter of that year only 11.9% of index members had reduced their EPS share count by as much as 4% year over year. Nearly a trillion dollars of repurchases, and roughly nine companies in ten did not achieve a share-count reduction that would register as more than noise in a valuation model.

That single juxtaposition is the argument of this article, aggregated: the money is real, the share reduction mostly is not, and the difference is compensation.

## When this matters to you

If you own an index fund, this is the machinery underneath a meaningful share of your returns, and the useful takeaway is modest: EPS growth for the market as a whole is part business and part arithmetic, and the arithmetic part is smaller than the buyback headlines suggest.

If you analyze individual companies, the practical change is a single habit. Before reading any per-share figure, pull five years of diluted weighted-average shares from the filings and plot them. It takes five minutes, requires no judgment, and reframes everything that follows — a company whose share count fell 20% and one whose count rose 5% are not comparable on EPS growth, no matter what the growth rates say. Then run Tests 2, 3 and 5 to size the compensation cost that the share count is absorbing.

If you work at a company that pays you in equity, the mechanics cut the other way and are worth understanding on their own terms: your RSUs dilute the shareholders, the buyback partially offsets that dilution, and the value of your grant depends on a share price that the buyback is helping to support. None of that is a reason for guilt — it is compensation you earned — but it is useful to know which levers your employer is pulling and why the vesting schedule is shaped the way it is.

Everything in this article is explanation, not investment advice. The tests size a cost; they do not tell you what a company is worth or what to do about it.

The natural next steps in this series are [reading the cash flow statement: why cash beats net income](/blog/trading/forensic-accounting/reading-the-cash-flow-statement-why-cash-beats-net-income), which covers the add-back mechanics that make the SBC line behave the way it does, and [non-GAAP and adjusted EBITDA: the metrics companies invent](/blog/trading/forensic-accounting/non-gaap-and-adjusted-ebitda-the-metrics-companies-invent), which handles the second add-back and the reconciliation rules that govern it.

## Sources & further reading

**Company filings (primary)**

- Alphabet Inc., Form 10-K for the fiscal year ended 31 December 2024, and the results release furnished as Exhibit 99.1 on 4 February 2025 — for revenue, net income, stock-based compensation, repurchases of stock, the share repurchase table (379 million shares for \$62,047 million in 2024; 528 million for \$62,184 million in 2023), the \$447 million of excise tax payments, share counts, and diluted EPS: [sec.gov](https://www.sec.gov/Archives/edgar/data/1652044/000165204425000014/goog-20241231.htm)
- Snowflake Inc., Forms 10-K for the fiscal years ended 31 January 2023 through 31 January 2026 — for revenue, net loss, stock-based compensation, operating cash flow, repurchases and diluted share counts. Figures retrieved from the SEC's XBRL company-concept data: [data.sec.gov](https://data.sec.gov/api/xbrl/companyconcept/CIK0001640147/us-gaap/ShareBasedCompensation.json)
- Salesforce, Inc., Forms 10-K for the fiscal years ended 31 January 2022 through 31 January 2026 — for buybacks, stock-based compensation, revenue and diluted share counts: [data.sec.gov](https://data.sec.gov/api/xbrl/companyconcept/CIK0001108524/us-gaap/PaymentsForRepurchaseOfCommonStock.json)
- International Business Machines Corporation, Forms 10-K for 2011 through 2015 — for revenue, diluted share counts, diluted EPS, net income, stock-based compensation and repurchases: [data.sec.gov](https://data.sec.gov/api/xbrl/companyconcept/CIK0000051143/us-gaap/WeightedAverageNumberOfDilutedSharesOutstanding.json)
- Oracle Corporation, Forms 10-K for the fiscal years ended 31 May 2018 through 31 May 2026 — for repurchases, operating cash flow, diluted share counts, stockholders' equity and stock-based compensation: [data.sec.gov](https://data.sec.gov/api/xbrl/companyconcept/CIK0001341439/us-gaap/StockholdersEquity.json)

**Accounting standards and tax law**

- Financial Accounting Standards Board, *Statement of Financial Accounting Standards No. 123 (revised 2004), Share-Based Payment*, issued December 2004; superseded APB Opinion No. 25 and required income-statement recognition of share-based payment at fair value. FASB's effective date was the first interim or annual reporting period beginning after 15 June 2005; the SEC subsequently deferred the compliance date for registrants to the first fiscal year beginning after that date, which for calendar-year filers was fiscal 2006.
- FASB Accounting Standards Codification Topic 718, *Compensation — Stock Compensation* — the current codification of SFAS 123R, including the modification-accounting requirement at issue in the Celsius Holdings enforcement action.
- FASB Accounting Standards Codification Topic 260, *Earnings Per Share* — the requirement to present both basic and diluted EPS, the treasury-stock method for computing diluted shares, and the exclusion of anti-dilutive securities, which is why diluted equals basic in a loss-making year.
- Internal Revenue Code Section 4501, added by the Inflation Reduction Act of 2022 — the 1% excise tax on repurchases of corporate stock, applying to repurchases after 31 December 2022, with the netting rule at Section 4501(c)(3) reducing the base by the fair market value of stock issued during the taxable year, including stock issued to employees: [taxnotes.com](https://www.taxnotes.com/research/federal/usc26/4501)
- Congressional Research Service, *The 1% Excise Tax on Stock Repurchases (Buybacks)*, Report R47397: [congress.gov](https://www.congress.gov/crs-product/R47397)
- Internal Revenue Service and Department of the Treasury, *Excise Tax on Repurchase of Corporate Stock*, final regulations published in the Federal Register on 24 November 2025: [federalregister.gov](https://www.federalregister.gov/documents/2025/11/24/2025-20721/excise-tax-on-repurchase-of-corporate-stock)
- U.S. Securities and Exchange Commission, Regulation G and Item 10(e) of Regulation S-K, adopted 22 January 2003 and effective 28 March 2003, together with the Division of Corporation Finance's Compliance and Disclosure Interpretations on non-GAAP financial measures (updated May 2016 and 13 December 2022) — the requirement to reconcile a non-GAAP measure to the most directly comparable GAAP measure, and the staff's view of which adjustments are permissible: [sec.gov](https://www.sec.gov/rules-regulations/staff-guidance/corporation-finance-interpretations/non-gaap-financial-measures)

**Securities regulation and enforcement**

- U.S. Securities and Exchange Commission, Rule 10b-18 under the Securities Exchange Act of 1934, adopted 1982 and amended 10 November 2003 — the non-exclusive safe harbour covering the manner, timing, price and volume of issuer repurchases. It does not provide protection for trading on material non-public information or for manipulative conduct.
- Item 703 of Regulation S-K — the issuer repurchase table required in periodic reports, aggregated monthly.
- U.S. Securities and Exchange Commission, *Share Repurchase Disclosure Modernization*, Release No. 34-97424, adopted 3 May 2023 (Federal Register, 1 June 2023) — **vacated by the U.S. Court of Appeals for the Fifth Circuit effective 19 December 2023**. The Commission published technical amendments reverting the rule text on 8 April 2024, so the pre-existing Item 703 monthly disclosure remains the operative requirement: [federalregister.gov](https://www.federalregister.gov/documents/2024/04/08/2024-06187/share-repurchase-disclosure-modernization)
- U.S. Securities and Exchange Commission, "Charter Communications to Pay \$25 Million Penalty for Stock Buyback Controls Violations", press release 2023-235, 14 November 2023 — settled charges under Exchange Act Section 13(b)(2)(B) over nine 10b5-1 trading plans containing "accordion" provisions used between 2017 and 2021. Charged as an internal accounting controls failure, not as fraud: [sec.gov](https://www.sec.gov/newsroom/press-releases/2023-235)
- U.S. Securities and Exchange Commission, *In the Matter of Celsius Holdings, Inc.*, Exchange Act Release No. 34-102227, AAER-4559, 17 January 2025 — \$3,000,000 civil penalty for failing to apply ASC 718 modification accounting to extended vesting for departing employees; charged under Exchange Act Sections 13(a), 13(b)(2)(A) and 13(b)(2)(B) and Rules 12b-20, 13a-11, 13a-13 and 13a-15, again with no antifraud charge: [sec.gov](https://www.sec.gov/files/litigation/admin/2025/34-102227.pdf)

**Market data**

- S&P Dow Jones Indices, press release of 19 March 2025, "S&P 500 Q4 2024 Buybacks Increase 7.4% and 2024 Expenditure Sets New Record by Increasing 18.5%" — for the 2024 record of \$942.5 billion, the 2023 figure of \$795.2 billion, the trailing-twelve-month peak of \$1.005 trillion in June 2022, the finding that 11.9% of companies reduced EPS share counts by at least 4% year over year in Q4 2024, and the 0.44% reduction in 2024 operating earnings from the buyback excise tax: [press.spglobal.com](https://press.spglobal.com/2025-03-19-S-P-500-Q4-2024-Buybacks-Increase-7-4-and-2024-Expenditure-Sets-New-Record-by-Increasing-18-5-Earnings-Per-Share-Increases-from-Buybacks-Decline-for-the-Quarter,-as-Q1-2025s-Impact-is-Expected-to-Increase)
- S&P Dow Jones Indices buyback series as reported in contemporaneous coverage for the 2007 peak of approximately \$589 billion and the 2009 trough of approximately \$138 billion. These two figures are attributed rather than taken from a primary release and should be read as approximate.

**Research and commentary**

- Daniel Bens, Venky Nagar, Douglas J. Skinner and M. H. Franco Wong, "Employee stock options, EPS dilution, and stock repurchases", *Journal of Accounting and Economics*, volume 36, issues 1–3 (2003), pages 51–91 — the finding that repurchase activity rises with the dilutive overhang from employee stock options and with the size of the EPS shortfall the firm faces.
- Paul Hribar, Nicole Thorne Jenkins and W. Bruce Johnson, "Stock repurchases as an earnings management device", *Journal of Accounting and Economics*, volume 41, issues 1–2 (2006), pages 3–27 — the finding that EPS-accretive repurchases cluster among firms that would otherwise have missed the consensus estimate.
- Heitor Almeida, Vyacheslav Fos and Mathias Kronlund, "The real effects of share repurchases", *Journal of Financial Economics*, volume 119, issue 1 (2016), pages 168–185 — the regression-discontinuity finding that firms which would just miss the consensus EPS forecast are markedly more likely to make EPS-increasing repurchases, and that those repurchases are associated with reductions in employment and investment: [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0304405X15001476)
- Warren E. Buffett, Berkshire Hathaway Inc. chairman's letter for 1992 — the passage on option accounting quoted in this article: [berkshirehathaway.com](https://www.berkshirehathaway.com/letters/1992.html)
- FW Cook, *2024 Top 250 Annual Incentive Plan Report*, summarised on the Harvard Law School Forum on Corporate Governance (5 October 2024) — for the prevalence of EPS as an annual incentive metric. The Harvard summary reports the general finding that companies use multiple financial measures in annual incentive plans but does not itself state an EPS percentage, and the underlying FW Cook report is not publicly retrievable at the link it gives. The "roughly one company in five" figure in this article is therefore attributed and approximate rather than verified against a primary document: [corpgov.law.harvard.edu](https://corpgov.law.harvard.edu/2024/10/05/2024-top-250-annual-incentive-plan-report)
- IBM's Roadmap 2015 target of "at least \$20" of operating earnings per share, set out at the company's 2010 investor briefing and withdrawn on 20 October 2014, is drawn from contemporaneous press and analyst coverage rather than a single primary filing; the financial figures used in the case study above are taken from IBM's Forms 10-K.

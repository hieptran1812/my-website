---
title: "The Income Statement of a Bank: Net Interest Income, Fees, and Provisions"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a bank actually makes money, line by line, from the interest spread and fee income down through operating costs and loan-loss provisions to net income."
tags: ["banking", "net-interest-income", "fee-income", "efficiency-ratio", "loan-loss-provisions", "pre-provision-profit", "bank-profitability", "income-statement", "financial-analysis"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A bank's income statement is built around one engine, the *interest spread*: it earns more on its loans and securities than it pays on its deposits and borrowings, and that gap, scaled across a giant balance sheet, is *net interest income*. Add fee income, subtract operating costs to get *pre-provision profit*, then subtract loan-loss *provisions* and tax to reach net income.
>
> - **Net interest income (NII)** is interest earned minus interest paid — usually 55-70% of a bank's revenue. It is the spread business in dollars.
> - **Fee income** is everything that is not the spread: cards, advice, asset management, payments. It needs little capital and smooths out the rate cycle.
> - The **efficiency ratio** (operating cost ÷ revenue) is the bank's cost discipline in one number; below ~60% is good, above ~70% is a problem.
> - **Provisions** are a charge for expected loan losses. They are tiny in good years and enormous in a recession — and because they hit the bottom line directly, a provision spike is how a profitable bank suddenly reports a loss.
> - The number to remember: a healthy bank earns about **1% on its assets** (return on assets). On a \$1 trillion balance sheet, that is roughly \$10 billion of net income — and a single bad credit year can cut it in half.

In the third quarter of 2020, JPMorgan Chase — the largest bank in the United States — did something that looked, on the surface, like a catastrophe and a triumph at the same time. Earlier that year, as the pandemic shut down the economy, the bank had set aside more than \$28 billion across a few quarters to cover loans it feared would go bad. That single act of caution turned what would have been a record-profit year into a string of mediocre ones. Then, as it became clear the feared wave of defaults never fully arrived, the bank *released* much of those reserves back into profit, and its earnings exploded to records.

Nothing about the bank's actual business — the loans, the deposits, the branches, the bankers — changed much between those two moments. What changed was a single accounting line called *provisions*, a guess about the future that flows straight to the bottom line. To understand a bank, you have to understand that its reported profit is not just what it earned. It is what it earned, minus what it *thinks* it is about to lose.

That is the strange and wonderful thing about a bank's income statement. A normal company sells a product, counts the cash, subtracts its costs, and reports the difference. A bank's profit is woven out of spreads measured in fractions of a percent, multiplied by a balance sheet so large that those fractions become billions — and then adjusted by a forward-looking estimate of losses that can swing the whole result. This post takes that income statement apart, line by line, and rebuilds it from zero.

![Bank income statement flow from interest income through net interest income fees operating expense provisions and tax to net income](/imgs/blogs/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions-1.png)

The diagram above is the mental model for this entire post: money comes in as interest and fees, costs go out as funding, operations, and credit losses, and what survives at the bottom is net income. Every section below zooms into one of those boxes. By the end, you will be able to read a bank's earnings report the way an analyst does — and know which line is the real engine, which line is the cost discipline, and which line is the one that can quietly turn a good year into a disaster.

## Foundations: the language of a bank's profit, from zero

Before we build anything, we need a shared vocabulary. A bank, at its heart, is a *maturity-transformation machine*: it borrows money short-term (your deposit, which you can withdraw any time) and lends it long-term (a 30-year mortgage), and it pockets the difference between what it earns on the loan and what it pays on the deposit. That difference — the *spread* — is the whole business. (If you want the deep version of why this trade is both profitable and structurally fragile, see [what a bank actually does](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread).) The income statement is just the spread, scaled up and adjusted for everything else, written down for one period of time.

Let me define every term you will need, in plain English, before any of them appears in a formula.

**Interest income.** This is the money a bank *earns* from lending. When you take out a car loan at 8%, the bank's interest income is that 8% on the balance you owe. The same goes for the bonds and Treasury securities a bank holds — they pay interest, and that interest is income. Interest income is the bank's *top line on its lending business*: the gross yield on everything it has put to work. The technical name for the assets that produce it is *earning assets* — loans plus securities plus the cash it lends out overnight.

**Interest expense.** This is the money a bank *pays* to fund itself. Your savings account that pays 4%? That 4% is the bank's interest expense. So is the interest on the bonds the bank itself issues, and on the money it borrows from other banks. Interest expense is the *cost of the raw material*, and the raw material of a bank is money. A bank that can fund itself cheaply — with lots of checking accounts that pay almost nothing — has a structural advantage that shows up right here.

**Net interest income (NII).** This is the headline. NII is simply interest income *minus* interest expense — the spread, in dollars, for the whole bank. If a bank earns \$100 of interest and pays \$35 of interest, its NII is \$65. This single line is usually the largest source of revenue at a typical commercial bank, somewhere between 55% and 70% of the total. When you hear that "banks make money on the spread", NII is the spread, counted up. The *rate* version of the same idea — NII divided by earning assets — is the **net interest margin (NIM)**, which we will return to constantly. (NIM gets its own full treatment in [net interest margin and the spread business](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained).)

**Fee income (also called non-interest income).** This is everything a bank earns that is *not* the interest spread. Card fees, advisory fees for arranging a merger, fees for managing a wealthy client's money, fees for moving a corporation's cash around the world, account maintenance fees, currency-exchange fees. Fee income is prized because, unlike lending, it usually requires very little of the bank's own capital and does not rise and fall with interest rates the way the spread does.

**Operating expense (also called non-interest expense or "opex").** This is what it costs to *run* the bank: salaries and bonuses for everyone from tellers to traders, the rent and upkeep on branches, the enormous technology budget, the legal and compliance army, marketing. It is the equivalent of a normal company's overhead. The single most-watched summary of it is the **efficiency ratio**, which is just operating expense divided by total revenue — how many cents of cost the bank spends to generate a dollar of revenue. Lower is better.

**Provisions (provision for credit losses, or PCL).** This is the most distinctive line on a bank's income statement, and the one that trips up newcomers. A provision is *not* a loss that already happened. It is a charge the bank takes *now* to cover loans it expects to go bad *later*. Each period, the bank looks at its loan book, estimates how much of it will not be repaid, and books that estimate as an expense. The money goes into a reserve called the *allowance for loan losses*. When a loan actually defaults and is written off, the loss is absorbed by that reserve — it does not hit the income statement again. So the income statement records the *expectation* of loss, and the balance sheet holds the *cushion* against it.

**Pre-provision operating profit (PPOP).** Also called *pre-provision net revenue*. This is total revenue (NII + fees) minus operating expense, *before* you subtract provisions. PPOP is arguably the truest measure of a bank's underlying earning power, because it strips out the volatile, forward-looking provision guess and shows you what the franchise earns in a normal world. Analysts love PPOP because it answers the question: "When the credit cycle is calm, how much does this machine actually print?"

**Net income.** The bottom line. Take PPOP, subtract provisions, subtract tax, and what remains is net income — the profit that belongs to the bank's shareholders. Divide it by the bank's assets and you get **return on assets (ROA)**; divide it by the bank's equity and you get **return on equity (ROE)**. (Those two ratios, and how leverage links them, are the subject of [ROE, ROA, and the leverage identity](/blog/trading/banking/roe-roa-and-the-leverage-identity-how-a-bank-is-judged).)

That is the entire cast of characters. Now let us watch them interact.

### Why a bank's income statement looks nothing like a normal company's

Here is the cleanest way to see what is different. A coffee chain's income statement reads: revenue from selling coffee, minus the cost of beans and cups and baristas and rent, equals operating profit; subtract tax, get net income. Simple, linear, intuitive. The "cost of goods" is a real, physical thing you can point to.

A bank's "cost of goods" is *money itself* — the interest it pays to get the funds it lends out. Its "revenue" is *interest plus fees*. And then it has this extra, almost philosophical line — provisions — where it deducts profit for losses that have not occurred and may never occur. No coffee chain takes a charge today because it thinks some customers might not pay next year. A bank does exactly that, every single quarter, and the size of that charge can dwarf everything else on the page.

There is a second deep difference. A coffee chain with \$1 billion of revenue might keep \$100 million of net income — a 10% margin on sales. A bank's profitability is measured against its *assets*, not its revenue, because its balance sheet is its business. A bank earning 1% on \$1 trillion of assets is doing well — but that 1% is a razor-thin margin that only becomes large money because the balance sheet is colossal and funded mostly with other people's money. This is why bank earnings are so sensitive: when you operate on a 1% margin against a giant, leveraged balance sheet, a small change in the spread or a small spike in losses moves the bottom line enormously.

A third difference is worth naming because it confuses almost everyone the first time. On a normal company's income statement, "interest" is a small line near the bottom — the cost of whatever debt it happens to carry. On a *bank's* income statement, interest is the top line on *both* sides: interest income is the main revenue, and interest expense is the main cost. The thing that is a footnote for a coffee chain is the entire business for a bank. That is because a bank's product *is* credit and its raw material *is* money, so the price of money — interest — sits at the center of everything it reports. Once you internalize that inversion, a bank's financials stop looking strange and start looking inevitable.

One more piece of plumbing before we proceed: the income statement and the balance sheet are two views of the same machine. The balance sheet is a *snapshot* — what the bank owns and owes on one day. The income statement is a *movie* — what flowed through the bank over a period (a quarter, a year). Interest income is the yield *on the assets* the balance sheet lists; interest expense is the cost *of the liabilities* the balance sheet lists; provisions feed the reserve that *sits on the balance sheet*. You cannot fully read one without the other. (We build the snapshot side from zero in [reading a bank balance sheet](/blog/trading/banking/reading-a-bank-balance-sheet-assets-liabilities-and-equity); this post is the movie.)

## Net interest income: the engine room

Let us start where most of the money comes from. Net interest income is the spread business, and it is worth slowing down to feel how it works.

Imagine the simplest possible bank. It takes in \$1,000 of deposits, on which it pays 1% interest, and it lends that \$1,000 out as a loan at 5% interest. Over one year:

- It earns 5% × \$1,000 = \$50 of interest income.
- It pays 1% × \$1,000 = \$10 of interest expense.
- Its net interest income is \$50 − \$10 = \$40.

That \$40 is the spread, in dollars. The *margin* — the rate version — is \$40 ÷ \$1,000 = 4.0%. That percentage is the net interest margin, and it is the heartbeat of the bank. Notice what makes NII grow: a wider spread (charge more on loans, pay less on deposits) *or* a bigger balance sheet (lend out more dollars at the same spread). Banks pull both levers.

Now, real banks do not lend out every dollar at one rate. They hold a mix of high-yielding credit-card balances, mid-yielding mortgages, and low-yielding government bonds, all funded by a mix of free checking accounts, interest-bearing savings, and more expensive wholesale borrowing. NII is the sum of every asset's yield minus the sum of every funding source's cost. But the principle is exactly the one in the toy example: earn more on assets than you pay on liabilities, multiplied by the size of the book.

There is one more source of spread that the toy example quietly contained but did not name: *free funding*. A chunk of a bank's assets is paid for not by interest-bearing deposits but by money the bank pays nothing for — checking-account balances that earn no interest, and the bank's own equity. Every dollar of an asset funded by free money earns its full yield with no offsetting expense. This is why a bank's net interest margin is usually *wider* than the simple loan-rate-minus-deposit-rate spread: a slice of the book is funded for free. It is also why the *deposit franchise* — the ability to gather large balances that pay little or nothing — is the most valuable thing a bank owns. A bank with a fortress of cheap checking deposits has a structurally higher margin than a rival that must pay up for every dollar it funds, even if they lend at identical rates.

The amount of free funding a bank enjoys is captured by a metric called the **CASA ratio** — the share of deposits held in *current and savings accounts* (the cheap, low-rate kind) rather than expensive term deposits. A high CASA ratio means cheap funding, a wide margin, and resilience when rates rise, because the bank does not have to pass much of the rate increase through to its depositors. When you see one bank earn a much higher NIM than another with a similar loan book, the answer is usually buried in the funding mix, not the lending.

![US commercial bank net interest margin from 2010 to 2024 showing the ZIRP trough and post-hike recovery](/imgs/blogs/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions-3.png)

The chart above shows why NII is not a steady annuity but a cyclical engine. The aggregate net interest margin of US commercial banks fell from 3.76% in 2010 to a trough of 2.56% in 2021, then jumped back above 3.2% as the Federal Reserve raised rates in 2022-2023. When the Fed cut rates to near zero (the "ZIRP" era — zero interest-rate policy), banks could not lower deposit rates below zero, so their funding cost had a floor while their loan yields kept falling — the spread compressed. When rates rose, loan yields repriced up faster than deposit rates, and the margin widened. A bank's NII can swing by billions of dollars between these regimes without management lifting a finger; the rate environment does it for them.

#### Worked example: building net interest income for a mid-sized bank

Let us scale up to something realistic. Suppose a bank has \$50 billion of earning assets yielding an average of 5.5%, and \$45 billion of interest-bearing liabilities costing an average of 1.8%. (The gap between \$50bn of assets and \$45bn of liabilities is funded by non-interest-bearing checking accounts and by the bank's own equity — free money, which is part of why the margin beats the simple spread.)

- Interest income = 5.5% × \$50 billion = \$2.75 billion.
- Interest expense = 1.8% × \$45 billion = \$0.81 billion.
- Net interest income = \$2.75 billion − \$0.81 billion = \$1.94 billion.
- Net interest margin = \$1.94 billion ÷ \$50 billion = 3.88%.

Now watch what one rate cycle does. Say the bank's deposit costs rise to 3.0% (depositors demand more after the Fed hikes) while loan yields rise only to 6.2% (loans reprice slowly). Interest expense becomes 3.0% × \$45bn = \$1.35bn; interest income becomes 6.2% × \$50bn = \$3.10bn; NII = \$1.75bn. The bank's loans are now earning more, yet its NII *fell* by nearly \$200 million — because its funding cost rose faster than its asset yield. **The intuition: NII is not about high rates or low rates; it is about the gap, and the gap depends on how fast each side reprices.**

## Fee income: the part of the bank that does not care about rates

If NII is the engine, fee income is the part of the car that keeps running when the engine stalls. Fee income — formally, non-interest income — is everything a bank earns that is not the spread. It matters for two reasons. First, it usually requires very little of the bank's scarce capital, so each dollar of fee profit is "cheaper" than a dollar of lending profit. Second, much of it does not move with interest rates, so it cushions the bank when the spread compresses.

![Matrix of four fee income families cards and payments advisory and markets asset and wealth management and transaction services](/imgs/blogs/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions-5.png)

There are four broad families of fee income, shown above:

**Cards and payments.** Every time you swipe a credit or debit card, the merchant pays a small fee, and a slice of that — *interchange* — goes to the bank that issued your card. Add annual card fees, late fees, and foreign-transaction fees, and card income becomes a large, steady stream that scales with consumer spending and uses almost no balance sheet. (The mechanics of who gets paid on a card swipe are a whole topic; the foundations are in [how money is created and moves through banks](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier).)

**Advisory and capital markets.** This is the investment-banking side: fees for advising a company on a merger, for underwriting a bond or stock issue, for trading on behalf of clients. These fees are extremely high-margin but *lumpy* — a single mega-deal can make a quarter, and a quiet market can starve it. This is why a universal bank's earnings are smoother than a pure investment bank's: the steady spread and card income absorb the lumpiness of deal fees. (For the full picture of this business, see [inside an investment bank](/blog/trading/finance/inside-an-investment-bank-how-they-make-money).)

**Asset and wealth management.** When a bank manages money for clients — mutual funds, private wealth, custody, trust services — it charges a fee, usually a small percentage of the assets it oversees. This income is recurring and sticky: clients rarely move their portfolios, and the fee base grows automatically as markets rise. Banks love it precisely because it is so predictable.

**Transaction services.** The unglamorous plumbing: charging corporations to move money, manage their cash, process payroll, handle wire transfers and foreign exchange. It is a low-key annuity, and critically, it brings in *cheap deposits* — corporations leave balances in their accounts, which the bank then funds its lending with. So transaction banking quietly feeds the NII engine too.

It is worth dwelling on why analysts and investors place a *higher value* on a dollar of fee income than on a dollar of spread income, because it is not obvious. The reason is capital. To earn the spread, a bank must hold an asset (a loan) on its balance sheet, and regulators require it to back that asset with expensive equity capital. Fee income — especially advisory, asset-management, and payments fees — uses little or no balance sheet, so it earns its profit without tying up the bank's scarce capital. A bank that earns \$1 of profit from fees can do so with almost no equity behind it, whereas \$1 of profit from lending demands a chunk of equity to support the loan. That makes the fee dollar more "capital-efficient", and capital efficiency is what drives return on equity. This is the deeper reason the great universal banks spent decades building card networks, wealth-management arms, and payments rails: those businesses let them grow profit without growing the capital-hungry balance sheet.

The flip side is that fee income is not free of risk; it is just a *different* risk. Advisory and trading fees vanish when markets freeze, exactly when the bank can least afford it. Card fee income falls when consumers stop spending — again, in a recession. So fee income diversifies the bank away from pure interest-rate risk, but it does not make the bank recession-proof; some fee lines are deeply cyclical in their own right. The smoothest banks are the ones whose fee mix leans toward the *recurring* kinds — asset-management and transaction fees that grind on through a downturn — rather than the *episodic* kinds that depend on a hot deal market.

#### Worked example: how fee income changes a bank's quality

Take two banks that each report \$2 billion of total revenue. Bank A earns \$1.8 billion from NII and only \$0.2 billion from fees — it is almost purely a spread machine. Bank B earns \$1.2 billion from NII and \$0.8 billion from fees — 40% of its revenue is fee-based.

Now a rate cycle compresses everyone's margin by 15%. Bank A's NII falls by 15% × \$1.8bn = \$0.27bn, so its revenue drops to \$1.73bn — a 13.5% hit. Bank B's NII falls by 15% × \$1.2bn = \$0.18bn, but its \$0.8bn of fees is untouched, so its revenue drops to \$1.82bn — only a 9% hit. **The intuition: fee income is a shock absorber. Two banks with identical revenue today can have very different resilience tomorrow, and the difference is how much of that revenue depends on the spread.** This is exactly why analysts prize a high non-interest-income share, and why banks fought so hard to build card, wealth, and payments franchises.

## Operating expense and the efficiency ratio

Revenue is only half the story. A bank can earn enormous NII and fees and still be a poor business if it spends too much to do it. Operating expense — staff, branches, technology, compliance, marketing — is the cost of running the machine, and the way the whole industry summarizes it is the *efficiency ratio*.

The efficiency ratio is one of the most useful numbers in all of banking, and it is delightfully simple:

$$\text{Efficiency ratio} = \frac{\text{Operating expense}}{\text{Total revenue}}$$

where total revenue is NII plus fee income. It tells you how many cents of cost the bank spends to generate one dollar of revenue. A bank with an efficiency ratio of 55% spends 55 cents to make a dollar — and keeps 45 cents as pre-provision profit. A bank at 75% spends 75 cents to make a dollar and keeps only 25. Because the denominator is revenue, *lower is better* — which catches people out, since for most ratios higher is better.

![Horizontal bar chart comparing efficiency ratios across five illustrative banks from lean to struggling](/imgs/blogs/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions-6.png)

The chart above shows the rough industry range. A lean, technology-led retail franchise might run in the low 50s; a strong universal bank in the high 50s; the industry average sits around 60-63%; a sprawling bank with too many branches and legacy systems drifts into the high 60s; and a struggling bank can exceed 75%, at which point too little revenue survives the cost line to absorb credit losses and still pay shareholders. The rough rule of thumb: below 60% is good, 60-65% is acceptable, above 70% is a warning sign.

There is a subtlety worth flagging. The efficiency ratio is sensitive to revenue, not just costs. When the rate cycle widens NII, revenue jumps and the efficiency ratio *improves* even if the bank's actual spending did not change at all. So a falling efficiency ratio can mean genuine cost discipline — or just a tailwind from rates. Good analysts watch the absolute level of expenses alongside the ratio, to tell the two apart.

It helps to know what actually sits inside operating expense, because the components behave very differently. The biggest piece, usually around half, is *compensation* — salaries and, crucially, the bonus pool, which flexes with revenue (a great trading year automatically raises the comp bill). The second piece is *occupancy and equipment* — the branches, offices, and ATMs, a cost the industry has spent two decades shrinking as banking moved online. The third, and the fastest-growing, is *technology* — the core systems, the data centers, the cybersecurity, the cloud migrations. The fourth is *compliance, legal, and regulatory* — the armies of staff who run anti-money-laundering checks, file regulatory reports, and handle the consequences when something goes wrong; this line ballooned after the 2008 crisis and barely shrinks. And lurking in expense are the one-off items: litigation settlements, restructuring charges, and fines, which can blow a quarter's efficiency ratio sky-high. A bank that keeps reporting "adjusted" results that strip out these items quarter after quarter is telling you something: the one-offs are not one-offs.

The reason cost discipline is existential, not cosmetic, is the chain we have been building. Operating expense sets PPOP; PPOP is the buffer that absorbs provisions; and provisions are unavoidable when the cycle turns. So a bank that lets its cost base bloat in good times is quietly shrinking the shield it will need in bad times. The efficiency ratio is, in the end, a measure of how much of each revenue dollar a bank manages to keep in reserve for the day the credit cycle comes for it.

#### Worked example: the efficiency ratio and what it leaves behind

A bank reports \$6 billion of NII and \$4 billion of fee income, for \$10 billion of total revenue. Its operating expense is \$6.2 billion.

- Efficiency ratio = \$6.2 billion ÷ \$10 billion = 62%.
- Pre-provision operating profit = \$10 billion − \$6.2 billion = \$3.8 billion.

Now suppose the bank invests heavily in technology and lets its expenses rise to \$7.0 billion while revenue holds at \$10 billion.

- Efficiency ratio = \$7.0 billion ÷ \$10 billion = 70%.
- PPOP = \$10 billion − \$7.0 billion = \$3.0 billion.

That 8-percentage-point deterioration in the efficiency ratio vaporized \$800 million of pre-provision profit — money that would otherwise be available to absorb loan losses and reward shareholders. **The intuition: the efficiency ratio is not a vanity metric; it directly sets the size of the buffer (PPOP) that stands between the bank and a loss. A bloated cost base is a thin shield.**

## Pre-provision operating profit: the truest measure of the engine

We have now assembled enough pieces to compute the most important interim number on a bank's income statement: pre-provision operating profit.

$$\text{PPOP} = \underbrace{(\text{NII} + \text{Fee income})}_{\text{total revenue}} - \text{Operating expense}$$

PPOP is the bank's earnings *before* the volatile provision line and tax. Think of it as the durable, repeatable earning power of the franchise — what it makes in a normal world, before the credit cycle has its say. Analysts love PPOP precisely because it sidesteps the two most manipulable and most volatile lines: provisions (a forward-looking estimate that management has discretion over) and tax (which depends on one-off items and jurisdiction). PPOP is the closest thing to "this is what the machine actually produces."

It is also the bank's *first line of defense* against losses. Here is the crucial relationship: in any given year, the bank's PPOP is the amount of loan losses it can absorb *and still break even*. If PPOP is \$4 billion and provisions come in at \$4 billion, the bank earns zero before tax but does not lose money — the engine's profit was exactly enough to cover the credit losses. Only when provisions *exceed* PPOP does the bank report a pre-tax loss and start eating into its capital. So a bank with a fat PPOP can survive a far worse credit cycle than a bank with a thin one, even if they have identical loan books. PPOP is the moat.

#### Worked example: building a full bank P&L from the lines up

Let us assemble an entire income statement, using the scaled "income = 100 units" framework from the cover chart so the arithmetic is clean. (Picture each unit as, say, \$100 million for a mid-sized bank.)

- Interest income: **+100**
- Interest expense: **−35**
- **Net interest income = 100 − 35 = 65**
- Fee income: **+35**
- Total revenue = 65 + 35 = **100**
- Operating expense: **−60** (efficiency ratio = 60 ÷ 100 = 60%)
- **Pre-provision operating profit = 100 − 60 = 40**
- Provisions: **−10**
- Pre-tax profit = 40 − 10 = **30**
- Tax at ~21%: **−6**
- **Net income = 30 − 6 = 24**

So out of every 100 units of interest income, this bank keeps 24 units of net income. Notice how each line carves into the total: funding cost takes 35, operations take 60 of the revenue, credit losses take 10, and tax takes 6. **The intuition: a bank's profit is what survives a gauntlet of four big subtractions, and the order matters — revenue must first beat operating cost (clearing PPOP) before it even meets the provision line.**

![Waterfall bar chart building a bank P and L from interest income to net income with income scaled to 100 units](/imgs/blogs/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions-2.png)

The waterfall above is that exact same P&L, drawn. Read it left to right: the tall green bar of interest income, knocked down by the red interest-expense bar to the blue net-interest-income subtotal of 65; lifted by the green fee bar to 100; cut down hard by the red operating-expense bar to the blue PPOP of 40; then trimmed by provisions and tax to the final blue net-income bar of 24. The picture makes the structure obvious in a way a table never can: operating expense is the single largest bite, and net income is a small slice of where you started.

## Provisions: the line that turns profit into loss

Now we arrive at the line that makes a bank's income statement genuinely different from any other company's — and the line responsible for the JPMorgan whiplash we opened with.

A provision (provision for credit losses, PCL) is a charge the bank books *today* for loan losses it *expects* in the future. It is not cash leaving the building. It is an accounting recognition that some of the loans on the books will not be repaid, made *before* anyone actually defaults. The money is moved into a reserve on the balance sheet — the *allowance for credit losses* — and when a loan finally goes bad and is written off, the loss is absorbed from that pre-funded reserve, so it does not hit the income statement a second time.

The modern accounting rules made this even more forward-looking. Under the US standard called **CECL** (current expected credit losses) and the international standard **IFRS 9**, a bank must estimate the *lifetime* expected losses of its loans and provision for them up front, using forecasts of the economy. The practical effect: the moment the economic outlook darkens, banks must book large provisions *immediately*, even before a single borrower misses a payment. That is why, in the first half of 2020, US banks took tens of billions in provisions almost overnight — the models, fed a recession forecast, demanded it. (The mechanics of provisioning, staging, and expected loss are a deep topic of their own; for the credit-risk side, see [credit risk: the chance you don't get paid back](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back).)

Here is why this single line is so dangerous to the bottom line: provisions sit *below* PPOP and are subtracted directly from it. PPOP is relatively stable — the spread and fees do not collapse overnight. But provisions are violently cyclical: near zero (or even negative, when reserves are released) in good times, and a multiple of normal in a recession. So the volatility of a bank's *net income* comes overwhelmingly from this one line. A bank can have a perfectly steady, profitable PPOP for a decade and still report a thumping loss in a single bad year, entirely because provisions spiked.

![Before and after comparison showing how a provision spike turns a normal year profit into a recession year loss](/imgs/blogs/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions-7.png)

The before-and-after above shows the mechanism with the numbers from our P&L. In a normal year, the bank has PPOP of 40, provisions of only 10, pre-tax profit of 30, and net income of +24. In a recession year, the *same bank* with the *same revenue* still earns PPOP of about 40 — but provisions surge to 55 as defaults rise and the bank builds reserves against the gloomy forecast. Now PPOP of 40 minus provisions of 55 is a pre-tax loss of −15. There is no tax on a loss, so net income is −15. The bank's equity cushion absorbs the hit. Nothing about the franchise changed; the credit cycle simply turned, and the provision line did the rest.

### Provision vs allowance vs charge-off: three words people confuse

To read a bank's credit story correctly, you have to keep three related words straight, because they live in different places and mean different things.

- The **provision** (PCL) is the *expense on the income statement* this period — the new charge the bank takes to top up (or, if released, draw down) its reserve.
- The **allowance for credit losses** is the *running reserve on the balance sheet* — the accumulated cushion built up by all past provisions, sitting ready to absorb losses. It is a stock; the provision is the flow that feeds it.
- The **charge-off** (or write-off) is the moment a specific loan is finally declared uncollectible and removed from the books. The loss is *absorbed by the allowance*, not re-charged to the income statement. A *recovery* is when the bank later claws back some cash on a charged-off loan, replenishing the allowance.

The link between them: `ending allowance = beginning allowance + provision − net charge-offs`. So in a calm year a bank might charge off about as much as it provisions, leaving the allowance roughly flat. In a deteriorating year it provisions *more* than it charges off, building the allowance ahead of the losses it sees coming — which is exactly the front-loading CECL forces. And in a recovering year it can provision *less* than it charges off, or even take a *negative* provision (a *reserve release*) that flows back into profit. The reserve release is the mechanical reason a bank's earnings can jump when the outlook brightens, even though nothing about its actual lending changed.

#### Worked example: tracking the allowance through a downturn and recovery

A bank starts a year with an allowance of \$8.0 billion. Over the year it charges off \$3.0 billion of bad loans and recovers \$0.5 billion on old ones, for net charge-offs of \$2.5 billion. The economy is worsening, so it provisions \$6.0 billion to build the reserve ahead of expected losses.

- Ending allowance = \$8.0bn + \$6.0bn − \$2.5bn = **\$11.5 billion**. The cushion grew because the bank provisioned far more than it lost — front-loading the bad news.

The next year, the recession proves milder than feared. Net charge-offs are \$2.0 billion, but the bank now needs a smaller reserve, so it takes a *negative* provision of −\$1.0 billion (a reserve release).

- Ending allowance = \$11.5bn − \$1.0bn − \$2.0bn = **\$8.5 billion**. And that −\$1.0bn provision *added* \$1.0 billion to pre-tax profit instead of subtracting from it.

**The intuition: the provision line can be a tailwind, not just a headwind. When a bank over-builds reserves in a scare and then releases them, the release inflates a later year's profit — which is why you should read net income alongside the allowance, not on its own.** A bank "beating estimates" on the back of a big reserve release has not earned more from banking; it has merely un-feared a fear.

#### Worked example: how a provision spike flips the bottom line

Let us make the swing concrete with realistic dollars. A bank has a stable PPOP of \$5 billion. In a normal year, its provisions run at \$1.2 billion (about 0.4% of a \$300 billion loan book).

- Pre-tax profit = \$5.0bn − \$1.2bn = \$3.8bn. Tax at 21% = \$0.80bn. **Net income = \$3.0bn.**

Now a recession hits. Expected losses jump, and under CECL the bank must front-load lifetime losses on its weakening book. Provisions surge to \$6.5 billion (about 2.2% of the loan book — roughly the peak experienced in severe downturns).

- Pre-tax result = \$5.0bn − \$6.5bn = −\$1.5bn. No tax is owed on a loss. **Net loss = \$1.5 billion.**

The bank swung from +\$3.0bn to −\$1.5bn — a \$4.5 billion turnaround — even though its PPOP, the underlying engine, never moved. **The intuition: PPOP tells you how much credit pain a bank can eat before it bleeds; provisions tell you how much pain arrived. The gap between them is the whole story of a bank's profit in any given year.** And the cruel twist of CECL is that the provision is recognized *all at once* when the outlook sours, so the loss is front-loaded and the recovery — the reserve release — comes later, which is exactly the whiplash JPMorgan showed in 2020-2021.

## Tax, net income, and what it all means per dollar of assets

The last two subtractions are quick. Tax is the corporate income tax on the bank's pre-tax profit — in the US, a federal rate of 21% plus state taxes, though one-off items and tax-exempt municipal-bond income can move the effective rate around. Subtract tax from pre-tax profit and you have *net income*: the profit that belongs to shareholders.

But the raw net-income number is almost meaningless on its own, because banks are enormous. \$10 billion of net income is spectacular for a \$200 billion bank and mediocre for a \$3 trillion one. To judge a bank, you scale net income by the size of the balance sheet:

$$\text{Return on assets (ROA)} = \frac{\text{Net income}}{\text{Total assets}}$$

The classic benchmark for a healthy bank is an ROA of about **1%** — one cent of net income per dollar of assets. It sounds tiny, and it is: it reflects the fact that banking is a low-margin, high-volume, leveraged business. A bank earns a sliver on each dollar but commands an immense pile of dollars, most of them borrowed.

![US banking industry return on assets from 2010 to 2024 with the one percent benchmark and crisis year troughs marked](/imgs/blogs/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions-4.png)

The chart above shows the US banking industry's ROA from 2010 to 2024, with the 1% benchmark drawn in. In good years — 2018, 2019, 2021 — the industry sat comfortably above 1%. In the crisis years, it cratered: barely 0.65% in 2010 as post-2008 losses worked through, and 0.72% in 2020 when the pandemic provisioning wave hit. Look at what drove those troughs: it was not the spread collapsing (NII held up reasonably). It was provisions surging. The ROA chart is, in large part, a chart of the provision cycle in disguise.

#### Worked example: from net income to ROA, and why leverage makes 1% feel like 12%

A bank has \$1 trillion of total assets and reports \$11 billion of net income.

- ROA = \$11 billion ÷ \$1,000 billion = 1.1%. A healthy result.

Now here is why that thin 1.1% is attractive to a shareholder. The bank funds those \$1 trillion of assets with only about \$100 billion of its own equity — the other \$900 billion is deposits and borrowings. So the *leverage* is roughly 10×. The return to shareholders is:

$$\text{ROE} = \text{ROA} \times \text{leverage} = 1.1\% \times 10 = 11\%$$

That is the magic and the menace of a bank. A 1.1% return on assets becomes an 11% return on equity because the balance sheet is leveraged about ten to one. But the same leverage works in reverse: if provisions push ROA to −0.5% in a bad year, that becomes −5% on equity, and the thin equity cushion takes the hit. **The intuition: a bank's income statement produces a tiny margin on assets, but leverage multiplies it into a respectable return on equity — and an alarming one when the margin goes negative.** (This identity, and how investors use it to value a bank, is the heart of [ROE, ROA, and the leverage identity](/blog/trading/banking/roe-roa-and-the-leverage-identity-how-a-bank-is-judged).)

## How the four levers fit together

Step back and the whole income statement resolves into four levers that management can pull, and one — the credit cycle — that pulls back.

![Graph showing the four drivers of bank net income two revenue engines and two drains feeding pre provision profit and net income](/imgs/blogs/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions-8.png)

The diagram above maps the drivers. Two *revenue engines* feed the top line: net interest income (the spread) and fee income. Together they make total revenue. One *cost drain* — operating expense — is subtracted to produce pre-provision profit, the durable engine. Then two more drains, provisions and tax, carve out the rest to leave net income.

What can management actually control? It can widen the spread (price loans better, gather cheaper deposits). It can grow fees (build card, wealth, and payments businesses). It can cut the cost line (close branches, automate, control headcount), improving the efficiency ratio. And it can underwrite credit more carefully, which lowers provisions *over the cycle*. What it cannot control is the timing of the credit cycle itself: when a recession arrives, provisions spike whether management likes it or not, and the only defense is the PPOP and capital it built in the good years.

Notice, too, how the levers interact rather than work in isolation. Cheap deposits do not just widen the spread — they come from transaction and retail relationships that *also* generate fees. A bigger fee base lifts revenue, which mechanically improves the efficiency ratio without a single dollar of cost being cut. Disciplined underwriting lowers provisions *and* lets the bank lend with less capital, raising its return on equity. The best-run banks pull all four levers in a way that reinforces: a sticky, cheap deposit franchise feeds wide margins and fee income, low costs preserve the buffer, and careful credit keeps the provision line from ever overwhelming it. The worst-run banks get the reverse spiral — they chase loan growth by paying up for hot deposits, which narrows the margin, then take more credit risk to compensate, which inflates provisions exactly when the buffer is thinnest. The income statement is where you watch which of those two spirals a bank is in.

This is also why bank earnings are famously *pro-cyclical* — they swing further than the economy does. Late in an expansion, all four levers point the same happy way at once: margins are healthy, fee businesses hum with active markets, costs look lean against booming revenue, and provisions are near zero (often releasing reserves into profit). Reported earnings look magnificent. Then the cycle turns and every lever reverses together: the margin compresses, markets-related fees dry up, the efficiency ratio worsens as revenue falls, and provisions explode. The amplitude of a bank's earnings swing is far larger than the amplitude of the recession that caused it — which is the income-statement signature of leverage and the credit cycle compounding each other.

This is the whole spine of the series in one income statement. A bank is a leveraged, confidence-funded maturity-transformation machine. The income statement is where that machine's spread shows up as money (NII), where its diversification shows up as fees, where its discipline shows up as the efficiency ratio, where its risk-taking shows up as provisions, and where its leverage turns a 1% return on assets into a double-digit return on equity. Read the income statement well and you can see, line by line, exactly how a bank lives — and where it would die.

## Common misconceptions

**"A bank's revenue is the interest it charges on loans."** No — its *revenue* is net interest income plus fees. The interest it charges (interest income) is a gross figure; the bank has to pay interest to fund those loans, and only the *spread* is revenue. Quoting interest income alone overstates a bank's earnings by the cost of its funding, which is often a third or more of the gross. In our worked P&L, interest income was 100 units but NII was only 65.

**"Higher interest rates are always good for banks."** Not necessarily. Rates affect NII through the *gap* between what assets earn and what funding costs — and that gap can widen *or* narrow when rates rise, depending on how fast each side reprices. A bank funded by sticky cheap deposits benefits when rates rise (loan yields climb, deposit costs lag). But a bank that must compete hard for deposits can see its funding cost rise faster than its loan yields, compressing the margin even as rates go up — exactly the second case in our NII worked example. Rates are not a simple tailwind; the repricing structure is everything.

**"Provisions are losses the bank has already suffered."** This is the single most common error. Provisions are an *estimate of future* losses, booked in advance under CECL/IFRS 9. They can be reversed (a *reserve release*) if the outlook improves, which is why JPMorgan's profit surged in 2021 as it released the reserves it had built in 2020. The actual loss event — a loan defaulting and being written off — is absorbed by the pre-funded reserve and does not hit the income statement again. The provision line is a forecast, not a tombstone.

**"A bank with rising profits is getting safer."** Often the opposite, late in a cycle. When the economy is booming, provisions fall to near zero (and reserves get released into income), so reported profits look fantastic *precisely when* the bank is making the loans that will go bad in the next downturn. Bank earnings are *pro-cyclical*: they peak right before the cycle turns. The time to worry about a bank is often when its provisions look suspiciously low and its loan growth looks suspiciously fast.

**"A high efficiency ratio means the bank is efficient."** Backwards. The efficiency ratio is cost *divided by* revenue, so *lower* is better. A 55% efficiency ratio is excellent; a 75% efficiency ratio means the bank burns 75 cents to make a dollar and is in trouble. The name is genuinely confusing, and even experienced people slip on the direction.

## How it shows up in real banks

**JPMorgan, 2020-2021 — the provision whiplash.** As the pandemic struck, JPMorgan booked enormous provisions — over \$28 billion across the worst quarters — front-loading expected losses under CECL based on a grim economic forecast. Its underlying engine (PPOP) stayed strong, but the provision avalanche crushed net income. Then, as the feared default wave failed to materialize thanks to massive government support, the bank *released* much of those reserves, and 2021 earnings hit records. Same loans, same spread, same franchise — the entire swing was the provision line moving from a huge build to a release. It is the cleanest real-world illustration of why net income is so much more volatile than PPOP.

**Silicon Valley Bank, 2023 — when the income statement hid the danger.** SVB's income statement looked fine right up until it failed. It had positive NII and reported profits; provisions were low because its borrowers (venture-backed tech firms) were not defaulting. The danger was not on the income statement at all — it was the unrealized loss on its long-dated bond portfolio sitting in the footnotes, and a deposit base that could flee in hours. SVB is the reminder that the income statement is necessary but not sufficient: a bank can be profitable on paper and still be killed by the balance sheet and liquidity risks the P&L does not capture. (The full anatomy is in [SVB and Credit Suisse, 2023](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

**The ZIRP decade — the great margin squeeze.** From roughly 2010 to 2021, the Federal Reserve held rates near zero, and the industry's net interest margin ground down from 3.76% to 2.56% (visible in our NIM chart). Banks could not lower deposit rates below zero, so their funding-cost floor held while loan yields kept falling. The response across the industry was to lean harder on fee income and to obsess over the efficiency ratio — squeezing costs because the spread engine was running cold. It is a real-world demonstration of how the three revenue-and-cost levers compensate for one another.

**Wells Fargo and the cost of conduct.** A bank's operating-expense line is not just branches and salaries; it also carries the cost of legal settlements, fines, and remediation. Wells Fargo's fake-accounts scandal drove billions of dollars through its expense line over several years (roughly \$5.5 billion in fines and penalties across 2016-2022), bloating its efficiency ratio and dragging its returns. It is a reminder that conduct risk shows up on the income statement as an expense, and a bank with a chronic legal bill is structurally less efficient than its peers.

**The 2008-2010 trough — provisions as the destroyer of ROA.** Industry ROA fell to about 0.65% in 2010 (our ROA chart). The proximate cause was not collapsing NII — the spread held up reasonably — but a tidal wave of provisions as mortgage and consumer losses materialized. The episode is the textbook case of the fourth lever overpowering the other three: a bank can do everything right on the spread, fees, and costs, and still post a dismal return because the credit cycle turned and the provision line swallowed the profit.

**The efficiency-ratio gap between rivals.** Compare a digitally-led franchise running an efficiency ratio in the low 50s against a sprawling universal bank in the high 60s. On identical revenue, the leaner bank keeps far more as PPOP — a structurally larger buffer to absorb credit losses and a higher through-cycle ROE. Over a decade, that cost-discipline gap compounds into a meaningfully different valuation, which is why the efficiency ratio is one of the first numbers a bank analyst checks.

## The takeaway / How to use this

When you next open a bank's earnings report, read it in the order this post built it, and ask a question at each line.

Start with **net interest income**: is it growing because the spread widened, because the balance sheet grew, or is it shrinking because deposit costs are outrunning loan yields? That one line tells you whether the core engine is healthy. Then look at **fee income** as a share of revenue: the higher it is, the more resilient the bank is to a rate squeeze, because fees do not move with the spread. Next, the **efficiency ratio**: below 60% is a disciplined operator; above 70% is a bank whose cost base is eating the buffer that should protect it. Multiply revenue minus expenses out to **PPOP** — that is the bank's durable earning power and the amount of credit loss it can eat before it bleeds.

Then comes the line that matters most for the future: **provisions**. Are they suspiciously low while loan growth is fast? That is a late-cycle warning, not a sign of safety — the loans being made now are the losses of the next recession. Are they spiking? Then compare the spike to PPOP: if provisions stay below PPOP, the bank merely earns less; if they exceed it, the bank reports a loss and starts eating capital. Finally, scale net income to **ROA** (aim for ~1%) and remember that leverage of roughly 10× turns that 1% into a low-double-digit ROE — and a negative ROA into a frightening hit to the thin equity cushion.

The deepest lesson is the relationship between two numbers: **PPOP and provisions**. PPOP is what the bank earns when the world is calm; provisions are what the world charges it when the cycle turns. A bank with a fat PPOP and a clean loan book can survive a brutal recession with a bad-but-survivable year. A bank with a thin PPOP — squeezed by a weak spread, low fees, and a bloated cost base — has no margin for error, and the first serious provision cycle pushes it into a loss and toward its capital. That is the income statement reading of the series' spine: a bank lives on a razor-thin margin against a giant, leveraged, confidence-funded balance sheet, and the gap between its pre-provision profit and its loan losses is the difference between a quiet year and a fight for survival.

*This is educational material about how to read a bank's financials, not investment advice.*

## Further reading & cross-links

- [What a bank actually does: maturity transformation and the spread](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread) — the business model the income statement reports on.
- [Reading a bank balance sheet: assets, liabilities, and equity](/blog/trading/banking/reading-a-bank-balance-sheet-assets-liabilities-and-equity) — the other half of the financial statements; the assets that earn interest and the funding that costs it.
- [Net interest margin and the spread business explained](/blog/trading/banking/net-interest-margin-and-the-spread-business-explained) — the deep dive on NIM, deposit beta, and asset repricing through a rate cycle.
- [ROE, ROA, and the leverage identity: how a bank is judged](/blog/trading/banking/roe-roa-and-the-leverage-identity-how-a-bank-is-judged) — how net income becomes the returns investors value.
- [Credit risk: the chance you don't get paid back](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back) — the risk that provisions are an estimate of.
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — the advisory and capital-markets fee engine in depth.

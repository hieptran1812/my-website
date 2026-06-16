---
title: "Liquidity and Solvency: Can the Company Survive a Bad Year?"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A from-zero deep dive into the two survival tests every business faces — liquidity (can it pay this year's bills?) and solvency (can it carry its debt for the long haul?), with the current, quick, cash, debt-to-equity, net-debt-to-EBITDA, and Altman Z-score ratios worked out in dollars."
tags: ["equity-research", "corporate-finance", "liquidity", "solvency", "current-ratio", "quick-ratio", "net-debt-to-ebitda", "altman-z-score", "covenants", "bankruptcy"]
category: "trading"
subcategory: "Equity Research"
author: "Hiep Tran"
featured: true
readTime: 43
---

> [!important]
> **TL;DR**
> - Profitability is about thriving; **liquidity and solvency are about surviving.** Most companies that die do not die from low profits — they die from running out of cash while the income statement still says "profitable."
> - **Liquidity** asks a short-term question: can the company pay the bills coming due in the next twelve months? You answer it with the **current ratio**, the **quick (acid-test) ratio**, and the **cash ratio** — three rungs of a ladder, each throwing out a less-certain asset.
> - **Solvency** asks a long-term question: can the company carry all its debt and avoid bankruptcy over many years? You answer it with **debt-to-equity**, **debt-to-assets**, the **equity ratio**, and the single most-used number on the buy side, **net debt to EBITDA**.
> - The balance sheet shows *book* solvency, but survival depends on **cash flow and the ability to refinance**. A solvent company with a **maturity wall** — too much debt coming due in one year — can be forced into default if credit markets are shut.
> - **Covenants** are promises in the loan agreement (e.g. "net leverage stays below 4.0x"). Trip one and the lender can demand all the money back *today*, turning a slow problem into an instant crisis.
> - The **Altman Z-score** rolls five balance-sheet and earnings ratios into one number that sorts a firm into *safe*, *grey*, or *distress* — a fast, decades-old smoke detector for bankruptcy.

A company's income statement can glow with profit right up to the week it files for bankruptcy. That sentence sounds like a paradox, and unlearning the paradox is the whole job of this post. We spend most of our time as investors asking whether a business is *good* — does it earn high margins, does it return strong [returns on capital](/blog/trading/equity-research/returns-on-capital-roic-roe-roa), is it growing? Those are questions about thriving. But before a business can thrive, it has to *survive*, and survival is a completely different test. It does not run on profit. It runs on cash, on the timing of obligations, and on whether lenders will keep lending. A wildly profitable company that cannot make Friday's payroll is in exactly as much trouble as an unprofitable one — arguably more, because nobody saw it coming.

So this post is about the two survival questions, and they are not the same question. The first is **liquidity**: can the company pay the bills that land in the next year — suppliers, wages, interest, the slice of debt coming due? The second is **solvency**: can the company carry the *whole* burden of its debt over the long run, or does it owe more than the business is fundamentally worth? Liquidity is a question about *this year's cash*. Solvency is a question about *the structure of the whole balance sheet*. A firm can pass one and fail the other, and the two failures look and behave completely differently — one is a curable cash crunch, the other is a terminal condition.

![Two columns comparing liquidity which asks whether a company can pay this year's bills against solvency which asks whether it can survive its long term debt over many years](/imgs/blogs/liquidity-and-solvency-can-the-company-survive-1.png)

The figure above is the mental model for the entire post. On the left is liquidity — the short fuse, measured in months, answered with the cash and near-cash you can scrape together against the bills due soon. On the right is solvency — the long fuse, measured in years, answered by comparing the total debt against the equity, the assets, and the earning power that has to service it. Keep both columns in your head as we go, because almost every real corporate failure is a story about one column collapsing while the other looked fine.

Throughout, we will use two recurring fictional companies. **Northwind Industries** — the same sturdy firm from [the balance sheet post](/blog/trading/equity-research/balance-sheet-what-a-company-owns-owes-and-is-worth) — is our fortress: profitable, cash-rich, lightly levered. Against it we will set **Riverstone**, a company that on the surface looks similar but is quietly fragile: more debt, less cash, and a single year where everything comes due at once. By the end you will be able to open a real 10-K, find the balance sheet and the debt-maturity footnote, and compute every survival ratio in this post — and, more importantly, know which ones actually predict whether the lights stay on.

## Foundations: the building blocks of survival

Before any ratio, we need a shared vocabulary. Every term below is defined from zero, because the ratios are just arithmetic on these definitions, and if the definitions are fuzzy the ratios are meaningless.

### Current vs non-current: the one-year line that splits the balance sheet

The balance sheet draws an invisible line called the **one-year boundary**, and everything to do with liquidity lives around it.

A **current asset** is anything the company expects to turn into cash within twelve months: actual cash, marketable securities (stocks and bonds it can sell quickly), accounts receivable (money customers owe and will pay soon), inventory (goods it expects to sell), and prepaid expenses (things it paid for in advance, like a year of insurance). A **non-current asset** — a factory, machinery, a patent, goodwill — is expected to stay in the business for years.

A **current liability** is anything the company must pay within twelve months: accounts payable (unpaid supplier invoices), accrued expenses (wages and taxes owed but not yet paid), the current portion of long-term debt (the slice of a multi-year loan due this year), and short-term borrowings. A **non-current liability** is owed further out: long-term bonds, multi-year loans, long-dated lease and pension obligations.

That one-year line is the dividing wall between liquidity and solvency. **Liquidity is almost entirely a story about the current section** — current assets versus current liabilities. **Solvency is about the whole balance sheet** — all the debt, current *and* non-current, against everything that backs it. Hold that distinction and the ratios fall into place.

### Cash flow vs profit: why a profitable company can run dry

A company records **revenue** when it delivers a product, not when the customer pays. It records an **expense** when it consumes a resource, not when the cash leaves. The gap between when profit is *recorded* and when cash actually *moves* is the entire reason liquidity is a separate problem from profitability — a theme we developed at length in [accruals vs cash](/blog/trading/equity-research/accruals-vs-cash-why-earnings-are-an-opinion) and in the [cash flow statement](/blog/trading/equity-research/cash-flow-statement-where-the-cash-really-comes-from) post.

Picture a company that books a \$10 million sale on credit. The income statement immediately shows \$10 million of revenue and, say, \$2 million of profit. But not a single dollar has arrived — the customer will pay in 90 days. Meanwhile the company already paid its suppliers and workers in cash to make the product. On the income statement it is thriving. In its bank account it is bleeding. If enough sales pile up as receivables faster than cash comes in, the company can be *growing, profitable, and insolvent on cash* all at once. This is called **growing broke**, and it is the single most common way healthy-looking small companies die.

The lesson to carry through the whole post: **profit is an opinion recorded under accrual rules; cash is a fact in a bank account.** Survival runs on the fact, not the opinion.

### Debt, equity, and the order of who gets paid

When a company is funded, the money comes from exactly two places: **lenders** (who provide debt and must be paid back on a schedule, with interest, no matter what) and **owners** (who provide equity and get only what is left over after everyone else is paid). We built this two-source picture in detail in [the balance sheet](/blog/trading/equity-research/balance-sheet-what-a-company-owns-owes-and-is-worth); here we need only its survival consequence.

Debt is a **fixed claim**: the \$10 million bond is due on its date whether the company had a great year or a terrible one. Equity is a **residual claim**: shareholders get the leftovers. In a wind-down, the order is strict — secured lenders first, then unsecured lenders, then preferred stock, then common equity last. This ordering is why debt is dangerous: it does not flex with the business. A bad year cuts profit, which cuts what flows to equity, but the debt payments do not shrink to match. **Leverage** — using borrowed money — magnifies good years and *also* magnifies bad ones, and the bad-year magnification is what can kill you.

### Insolvency: two precise meanings

The word "insolvent" gets used loosely, but it has two specific technical meanings, and both matter.

**Balance-sheet insolvency** means liabilities exceed assets — the company has *negative equity*. If you sold everything it owns and paid everyone it owes, there would not be enough; the owners' stake is below zero. **Cash-flow insolvency** (also called *practical insolvency*) means the company simply cannot pay its debts as they fall due, regardless of what the balance sheet says. A firm can be balance-sheet *solvent* (assets worth more than debts) but cash-flow *insolvent* (no money this Friday) — that is precisely the illiquid-but-solvent case, and it is curable. A firm that is balance-sheet *insolvent* is in much deeper trouble, because no amount of fresh cash fixes a business that owes more than it is worth.

With these foundations — the one-year line, cash versus profit, fixed versus residual claims, and the two flavors of insolvency — we can now build every ratio.

## Liquidity: can the company pay this year's bills?

Liquidity is the easier of the two questions to compute and the more *immediate* of the two to kill you. We measure it by stacking up the assets that can become cash soon against the bills due soon. There are three standard ratios, and the smartest way to understand them is as a **ladder**: each rung throws out a less-trustworthy asset and asks a harsher version of the same question.

![A vertical ladder of three liquidity ratios where the current ratio counts all current assets the quick ratio removes inventory and prepaids and the cash ratio keeps only cash and securities](/imgs/blogs/liquidity-and-solvency-can-the-company-survive-2.png)

The ladder in the figure runs from loosest at the top to strictest at the bottom. The **current ratio** is generous — it counts every current asset, including inventory you have not sold yet and prepaid insurance you cannot spend. The **quick ratio** is suspicious of inventory and prepaids and throws them out. The **cash ratio** trusts nothing but money you already have. Reading all three together tells you not just *whether* a company can pay its bills but *how much it has to scramble* to do it.

### The current ratio: the broadest first look

The **current ratio** is the most basic liquidity test:

$$\text{Current ratio} = \frac{\text{Current assets}}{\text{Current liabilities}}$$

It asks: for every dollar of bills due within a year, how many dollars of soon-to-be-cash does the company have lined up? A current ratio of **2.0** means \$2 of current assets for every \$1 of current liabilities — a comfortable cushion. A ratio **below 1.0** means current liabilities exceed current assets, which is a warning: at face value, the company owes more in the near term than it has in near-term resources to cover it.

But "below 1.0 is bad, above 1.0 is good" is a beginner's rule that breaks constantly, and the reasons it breaks are where the real understanding lives. We will get to those after we compute one.

#### Worked example: Northwind's current, quick, and cash ratios

Recall Northwind's current section from [the balance sheet](/blog/trading/equity-research/balance-sheet-what-a-company-owns-owes-and-is-worth) (all figures in millions). Its **current assets** are: cash & equivalents \$120, marketable securities \$30, accounts receivable \$90, inventory \$110, and prepaid expenses \$20 — totaling **\$360**. Its **current liabilities** are: accounts payable \$70, accrued expenses \$40, deferred revenue \$50, and short-term debt \$40 — totaling **\$200**.

The **current ratio**:

$$\text{Current ratio} = \frac{\$360}{\$200} = 1.80$$

For every \$1 of near-term bills, Northwind has \$1.80 of near-term resources. That is healthy for most industries.

The **quick ratio** (acid-test) strips out inventory (\$110) and prepaids (\$20), because inventory might not sell at full value and prepaids cannot be turned back into spendable cash:

$$\text{Quick ratio} = \frac{\$360 - \$110 - \$20}{\$200} = \frac{\$230}{\$200} = 1.15$$

Even after throwing out the soft assets, Northwind still has \$1.15 of quick assets per \$1 of bills. It does not need to sell a single widget to cover the next year.

The **cash ratio** keeps only cash and securities — money already in hand:

$$\text{Cash ratio} = \frac{\$120 + \$30}{\$200} = \frac{\$150}{\$200} = 0.75$$

Northwind can cover **75%** of all its near-term bills with cash sitting in the bank right now, before collecting a single receivable or selling a single unit. That is a fortress liquidity position.

*The three ratios tell one story at three levels of paranoia: Northwind is comfortable on the loosest test, comfortable on the middle test, and still covers three-quarters of its bills under the harshest test.*

### The quick (acid-test) ratio: liquidity when inventory is a trap

The **quick ratio** exists because inventory lies. In a healthy business, inventory is worth roughly what the books say. But the moment a company is in trouble, inventory is the *first* thing to become worthless: a fashion retailer's unsold winter coats in March, a chipmaker's last-generation processors, perishable goods past their shelf life. When you most need to convert inventory to cash, it is precisely when nobody wants to buy it, or wants it only at a fire-sale discount.

$$\text{Quick ratio} = \frac{\text{Cash} + \text{Securities} + \text{Receivables}}{\text{Current liabilities}}$$

The quick ratio is "what can I turn into cash *quickly and reliably*?" — cash already is cash, securities sell in a day, and receivables are money customers genuinely owe (though even these can go bad if a big customer fails). It deliberately excludes inventory and prepaids. For inventory-heavy businesses — retailers, manufacturers — the gap between the current ratio and the quick ratio is huge, and the quick ratio is the number that actually matters. For a software company that holds almost no inventory, the two ratios are nearly identical.

### The cash ratio: the doomsday test

The **cash ratio** is the most conservative liquidity measure: only cash and marketable securities over current liabilities. It assumes the worst — that you cannot collect a single receivable and cannot sell a single unit of inventory — and asks what fraction of your bills you could still pay from money already in the bank.

$$\text{Cash ratio} = \frac{\text{Cash} + \text{Marketable securities}}{\text{Current liabilities}}$$

Almost no company has a cash ratio above 1.0, and they should not — holding enough cash to cover *every* near-term bill with zero collections is wildly inefficient. But the cash ratio is the number that matters in a true panic, when receivables stop coming in and nobody buys inventory. In the 2008 crisis and the March 2020 cash scramble, the cash ratio is what separated companies that calmly rode it out from companies that begged the Fed for a lifeline.

### Why a *high* current ratio can be a bad sign

Here is the counterintuitive part that separates a real analyst from a checklist. A current ratio of 4.0 sounds wonderful — \$4 of current assets per \$1 of bills! — but it can actually be a symptom of mismanagement. The current assets that pile up to make the ratio high are **idle working capital**: cash earning nothing, inventory gathering dust in a warehouse, receivables the company has failed to collect. Every dollar tied up in a swollen current ratio is a dollar *not* invested in the business, *not* returned to shareholders, *not* earning a return. A company with a current ratio of 4.0 may simply be terrible at collecting its bills and managing its stock.

The best-run companies often run *low* current ratios — sometimes below 1.0 — on purpose, because they have such tight control of cash and such reliable cash generation that they do not need a fat cushion. Think of a giant retailer that gets paid by customers instantly in cash but pays its suppliers 60 days later: it is effectively financing its operations with supplier money (a *negative cash conversion cycle*), and its current ratio can sit below 1.0 while it is one of the safest businesses on earth. **The current ratio measures a cushion, but a cushion can be either prudent safety or lazy capital. You have to know which.**

#### Worked example: why a current ratio of 1.0 can be safer than a current ratio of 3.0

Compare two firms. **GrocerCo** is a supermarket chain. It collects cash from shoppers the instant they swipe a card, holds fast-moving inventory, and pays suppliers 45 days later. Its current assets are \$1,200M (mostly inventory that turns over every two weeks), current liabilities \$1,250M — a current ratio of **0.96**. **SlowMakerCo** is a struggling manufacturer with \$3,000M of current assets (of which \$1,800M is slow-moving inventory and \$700M is aging receivables nobody is chasing) against \$1,000M of current liabilities — a current ratio of **3.0**.

By the naive rule, SlowMakerCo (3.0) is three times safer than GrocerCo (0.96). The reality is the opposite. GrocerCo generates cash every single day; its low ratio reflects a brilliant, supplier-financed model, not weakness. SlowMakerCo's high ratio reflects \$2,500M of capital frozen in inventory it cannot sell and receivables it cannot collect — assets that will evaporate the moment the business sours. If both hit a rough quarter, GrocerCo keeps printing daily cash while SlowMakerCo discovers its "current assets" were a mirage.

*A liquidity ratio is only as good as the quality of the assets inside it; a low ratio backed by daily cash beats a high ratio backed by dead inventory.*

### Right ranges by industry: there is no universal "good" number

Because of everything above, there is no single correct liquidity ratio — it depends entirely on the business model. A few rough industry norms (illustrative, not exact):

- **Software / SaaS**: low inventory, recurring cash, often current ratios of 1.0–2.0 and quick ratios nearly identical. Deferred revenue (cash collected upfront) inflates current liabilities but is "good" debt — it is a service owed, not cash owed.
- **Retail / grocery**: high inventory, fast cash collection, often current ratios near or below 1.0 by design (supplier financing). The quick ratio is the one to watch.
- **Heavy manufacturing**: large inventory and receivables, current ratios of 1.5–2.5 are typical; you scrutinize inventory aging.
- **Utilities**: stable, predictable cash, can run lean current ratios because revenue is contractual.
- **Banks**: liquidity is measured completely differently (regulatory ratios, deposit runoff) — the current ratio is meaningless for them.

The discipline is: **never judge a liquidity ratio without knowing the industry's working-capital rhythm.** A 0.9 current ratio is alarming for a manufacturer and unremarkable for a grocer.

## Solvency: can the company survive the long haul?

Liquidity asks about this year. **Solvency** asks the deeper question: across all the years ahead, can the company carry the *total* weight of its debt — or has it borrowed so much that the business owes more than it can ever be worth? Solvency ratios look at the *whole* balance sheet, not just the current section, and they compare the total debt against the three things that have to back it: the owners' equity, the assets, and the cash earnings.

![A matrix of solvency ratios debt to equity debt to assets net debt to EBITDA equity ratio and interest coverage each with a comfortable zone a watch zone and a danger zone](/imgs/blogs/liquidity-and-solvency-can-the-company-survive-3.png)

The matrix above is your solvency dashboard. No single number in it condemns a company — every business and industry sits somewhere different — but the column structure is the point. A firm with one ratio drifting into the amber "watch" zone is normal. A firm with *several* ratios sliding into the red "danger" column at the same time is what bankruptcy looks like on a balance sheet, quarters before it happens. Let us build each ratio.

### Debt-to-equity: how the funding splits between lenders and owners

The **debt-to-equity ratio** (D/E) compares total debt to shareholders' equity — the borrowed money against the owners' money:

$$\text{Debt-to-equity} = \frac{\text{Total debt}}{\text{Shareholders' equity}}$$

A D/E of **1.0** means lenders and owners have put in equal amounts. A D/E of **2.0** means there is twice as much borrowed money as owner money — the firm is heavily leveraged. A D/E **below 1.0** means owners fund most of the business, a conservative structure.

Why this matters for survival: equity is a *shock absorber*. When the business loses money, those losses come out of equity first; debt is untouched until equity is gone. The more equity relative to debt, the more bad years the company can absorb before lenders are at risk and the company is forced to restructure. A high D/E means a thin shock absorber — a couple of bad years can wipe out the equity entirely and push the firm into the lenders' hands. We will go far deeper on the *cost* of that leverage in the forthcoming [leverage and coverage](/blog/trading/equity-research/leverage-and-coverage-debt-that-compounds-vs-kills) post; here the point is purely structural survival.

### Debt-to-assets and the equity ratio: two views of the same cushion

The **debt-to-assets ratio** asks what fraction of *everything the company owns* was funded by debt:

$$\text{Debt-to-assets} = \frac{\text{Total debt}}{\text{Total assets}}$$

A ratio of **0.3** means debt funded 30% of the assets; the other 70% came from owners (and operations). The **equity ratio** is the mirror image — the fraction funded by owners:

$$\text{Equity ratio} = \frac{\text{Shareholders' equity}}{\text{Total assets}}$$

These two (plus the small piece from non-debt liabilities like payables) roughly add to one. The equity ratio is, in a sense, the most fundamental solvency number: it is the size of the cushion as a fraction of the whole business. An equity ratio of **0.5** means half the assets are owner-funded — the company can lose up to half its asset value before it is balance-sheet insolvent. An equity ratio of **0.1** means a 10% drop in asset value wipes out the owners entirely.

### Net debt: gross debt minus the cash you could use to pay it

Before the most important solvency ratio, one refinement. A company's *gross* debt overstates how leveraged it really is if it is also sitting on a pile of cash, because that cash could be used to pay down debt tomorrow. **Net debt** corrects for this:

$$\text{Net debt} = \text{Total debt} - \text{Cash and marketable securities}$$

A company with \$500M of debt and \$500M of cash has *zero* net debt — it could clear its entire debt with the cash on hand. A company with \$500M of debt and \$50M of cash has \$450M of net debt — the real burden. Net debt is the number that actually matters for the leverage ratios professionals use, because it reflects the true claim against the business after accounting for the firm's own war chest.

### Net debt to EBITDA: the number the whole credit market lives on

If you learn one solvency ratio, learn this one. **Net debt to EBITDA** is the single most-quoted leverage metric on the buy side, in credit ratings, and in loan covenants. **EBITDA** is *Earnings Before Interest, Taxes, Depreciation, and Amortization* — a rough proxy for the annual cash earnings a business throws off before financing and accounting items. The ratio is:

$$\frac{\text{Net debt}}{\text{EBITDA}}$$

and it has a beautifully intuitive meaning: **roughly how many years of cash earnings it would take to repay all the net debt**, if every dollar of EBITDA went to debt repayment. A ratio of **1x** means the company could clear its debt in about a year — barely levered. A ratio of **5x** means it would take five years of flawless, fully-applied earnings — heavily levered, usually junk-rated, and dangerously exposed to any stumble. A ratio of **8x** means the company can essentially never repay from earnings and survives only by continually borrowing new money to refinance the old.

![A four firm comparison table showing net debt EBITDA and net debt to EBITDA ratios ranging from under one for a fortress firm to eight for a distressed firm](/imgs/blogs/liquidity-and-solvency-can-the-company-survive-7.png)

The table above shows the same arithmetic across four firms, and it makes the key point: **the same dollar of debt is safe on one firm and lethal on another.** What determines safety is not the absolute size of the debt but the size relative to the cash the business generates. \$600M of debt is comfortable on a firm earning \$250M of EBITDA (2.4x) and a slow-motion catastrophe on one earning \$75M (8x).

#### Worked example: what 5x net-debt-to-EBITDA really means for Riverstone

Meet **Riverstone**, our fragile firm. It has \$1,000M of total debt and only \$100M of cash, so its net debt is:

$$\text{Net debt} = \$1{,}000\text{M} - \$100\text{M} = \$900\text{M}$$

Its EBITDA — annual cash earnings before financing and accounting items — is \$180M. So:

$$\frac{\text{Net debt}}{\text{EBITDA}} = \frac{\$900\text{M}}{\$180\text{M}} = 5.0\times$$

What does 5x actually mean? It means that if Riverstone took *every single dollar* of cash earnings and applied it to debt — paid no interest, no taxes, no capital spending, no dividends, nothing — it would still take **five full years** to clear the debt. In reality it cannot do that, because interest alone eats a large chunk of EBITDA and the business needs to reinvest to survive. So 5x means the debt is not getting repaid from earnings in any realistic horizon; it is getting *refinanced* — rolled into new debt when it comes due. That is fine as long as credit markets stay open and Riverstone's EBITDA holds. But if EBITDA drops to \$120M in a recession, the ratio jumps to \$900M / \$120M = **7.5x** without Riverstone borrowing a single extra dollar — the leverage got worse purely because earnings fell. Compare Northwind: \$340M debt − \$150M cash = \$190M net debt, against \$220M EBITDA, for a ratio of **0.86x** — Northwind could clear its entire net debt in under a year.

*Net debt to EBITDA is the credit market's clock on a company: at 1x it is a year from debt-free, at 5x it is permanently dependent on lenders agreeing to keep lending.*

### Book solvency vs the ability to refinance: the distinction that kills people

Here is the most important idea in the solvency half of this post, and it is the one beginners miss. **The balance sheet measures *book* solvency** — does the company own more than it owes, on paper, today? But survival depends on something the balance sheet does not show: **the ability to refinance**. Most large companies never actually repay their debt from earnings; they *roll it over*. A \$300M bond comes due, and the day it matures the company issues a *new* \$300M bond to pay off the old one. The debt is not extinguished — it is renewed. As long as the company can keep finding lenders willing to renew, it survives indefinitely, even with enormous debt.

This means a balance-sheet-solvent company can die if the renewal stops. Refinancing depends on two things outside the balance sheet: **the credit markets being open** (sometimes they slam shut for everyone, as in 2008 and briefly in March 2020) and **lenders still trusting this specific company** (its credit rating, its earnings trend, its sector). When either fails, a debt that "everyone knew would be refinanced" suddenly cannot be, and the company is forced to repay cash it does not have. The balance sheet looked solvent the whole time. The problem was never the book value — it was the rollover.

### The maturity wall and rollover risk

This brings us to one of the deadliest patterns in corporate finance: the **maturity wall**. A maturity wall is a year in which an unusually large amount of debt comes due all at once. In normal years the company refinances small, staggered tranches without trouble. But if it has stacked a huge chunk of debt into a single year, it has concentrated all its refinancing risk into that one window — and if credit markets happen to be shut that particular month, the company is forced into default on a debt it could have rolled over easily in any normal year.

![A debt maturity timeline showing small manageable maturities in early years then a three hundred million dollar wall in one year that forces a default when credit markets are shut](/imgs/blogs/liquidity-and-solvency-can-the-company-survive-4.png)

The timeline above is the maturity wall made visible. In 2025 and 2026, small tranches come due and get refinanced without anyone noticing. Then 2027 is the wall — \$300M maturing in a single year. The company is *solvent*: its assets are worth far more than its debts. But if the credit market is frozen the month that bond matures, there is no buyer for a new bond, no lender for a new loan, and the \$300M must be repaid in cash the company does not have. A profitable, solvent firm files for bankruptcy not because it was worth too little, but because its debt was scheduled badly. Good treasurers spend enormous effort *smoothing* the maturity profile precisely to avoid walls — staggering maturities across many years so no single year can sink the company.

#### Worked example: a solvent Riverstone forced into default by a maturity wall

Riverstone is, on the balance sheet, comfortably solvent. Its assets are worth \$2,000M and its total debt is \$1,000M — it owns twice what it owes; equity is a healthy \$1,000M and falling nowhere. By every book-solvency test, Riverstone is fine.

But look at the debt schedule. Of the \$1,000M, a single \$300M bond matures in 2027. In 2025 and 2026 only \$40M and \$60M come due — trivially refinanced. Riverstone's treasurer always assumed the \$300M would be rolled over the same way: issue a new bond, pay off the old one. Then in early 2027, a financial shock freezes the credit markets — no company can issue new bonds for several months. The \$300M comes due in March 2027. Riverstone has \$100M of cash. It cannot raise \$300M because no one is buying any bonds from anyone. It cannot sell \$300M of assets in a few weeks without crushing fire-sale losses. The bond defaults. Riverstone — solvent, profitable, worth twice its debt — files for bankruptcy protection.

The post-mortem is brutal in its simplicity: Riverstone did not die because it owed too much relative to what it owned. It died because it scheduled \$300M to come due in a single month and got unlucky about which month. Had that bond been split into three \$100M tranches maturing in 2026, 2028, and 2030, no single frozen-market window could have killed it.

*A maturity wall converts a market-wide liquidity freeze into a company-specific death sentence; the cure is boring — stagger your maturities so no year can sink you.*

### Covenants: the trip-wires in the loan agreement

There is one more solvency mechanism that turns slow problems into instant crises: **covenants**. A covenant is a promise the borrower makes in the loan agreement, and breaking it gives the lender powerful rights. There are two kinds. **Affirmative covenants** require the borrower to *do* things (deliver audited financials on time, maintain insurance). **Negative covenants** forbid things (don't take on more debt above a limit, don't sell major assets, don't pay dividends above a cap). The most dangerous category is **financial maintenance covenants** — promises to keep a financial ratio within a limit, *tested every quarter*. The classic one: "net debt to EBITDA shall not exceed 4.0x" or "interest coverage shall stay above 2.0x."

Here is why covenants are a survival issue. If the company *trips* a covenant — its net leverage rises above 4.0x because EBITDA fell — it is in **technical default**. The debt has not missed a payment; the company simply broke a promise. But technical default gives the lender the right to **accelerate** the loan: declare the entire balance due *immediately*. A \$500M loan that was due in five years can become \$500M due *today*. The company that was slowly sliding now faces an instant demand for cash it does not have. In practice lenders often grant a **waiver** (for a fee, and often tighter terms) rather than detonating the loan — but they hold that right, and in a crisis they may use it, or extract painful concessions for not using it. A tripped covenant is how a gradual deterioration becomes a sudden cliff.

#### Worked example: Riverstone trips a net-leverage covenant

Riverstone's main \$500M term loan carries a financial maintenance covenant: **net debt to EBITDA must stay below 4.0x**, tested at the end of each quarter. When the loan was signed, Riverstone's net debt was \$900M and EBITDA was \$250M, for net leverage of \$900M / \$250M = **3.6x** — comfortably under the 4.0x limit.

Then a soft year hits. Demand weakens and Riverstone's trailing-twelve-month EBITDA falls from \$250M to \$180M. Its net debt is roughly unchanged at \$900M. The covenant test at quarter-end:

$$\frac{\text{Net debt}}{\text{EBITDA}} = \frac{\$900\text{M}}{\$180\text{M}} = 5.0\times$$

That is **above 4.0x** — the covenant is breached. Riverstone has not missed a single payment; it has simply earned less. But the breach puts it in technical default, and the lender now has the right to demand the entire \$500M back immediately. Riverstone has \$100M of cash. It cannot repay \$500M on demand. It is now negotiating from a position of weakness: the lender agrees to a waiver but only in exchange for a higher interest rate, a fee, and even tighter future covenants — and the breach itself, once disclosed, frightens Riverstone's other lenders and suppliers, who tighten their own terms. A single number drifting from 3.6x to 5.0x, driven entirely by an earnings dip, has turned a five-year loan into a present-tense emergency.

*A covenant is a clock the company doesn't control: a bad quarter can hand the lender the right to demand everything back at once, which is how distress accelerates from slow to sudden.*

## The Altman Z-score: one number for bankruptcy risk

We now have a dashboard of individual ratios. In 1968, NYU professor **Edward Altman** asked whether those ratios could be combined into a single number that *predicts* bankruptcy. He took a sample of firms that went bankrupt and a sample that survived, and used statistics to find the weighted combination of ratios that best separated the two groups. The result — the **Altman Z-score** — has been a standard bankruptcy smoke detector ever since, and it remains startlingly effective for a half-century-old formula.

![A five row table of the Altman Z-score components each ratio its meaning and its weighted contribution summing to a Z-score of one point five which falls in the distress zone](/imgs/blogs/liquidity-and-solvency-can-the-company-survive-5.png)

The Z-score (the classic version, for public manufacturers) is a weighted sum of five ratios:

$$Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E$$

where each letter is a ratio capturing a different kind of fragility:

- **A = Working capital / Total assets.** Short-term liquidity cushion relative to the firm's size. A firm with negative working capital is already on thin ice.
- **B = Retained earnings / Total assets.** Cumulative lifetime profitability plowed back into the business. A young firm or one that has paid out everything has a low B; a mature, self-funded firm has a high B.
- **C = EBIT / Total assets.** The core earning power of the assets, before financing and taxes — the engine. This carries the biggest weight (3.3) because earning power is the deepest determinant of survival.
- **D = Market value of equity / Total liabilities.** How much market-priced cushion sits above the debt. This uses *market* equity, so the stock market's collective judgment of the firm feeds in.
- **E = Sales / Total assets.** Asset turnover — how efficiently the assets generate revenue.

The weighted sum sorts the firm into three zones:

- **Z > 2.99**: the **safe** zone — bankruptcy is unlikely in the next two years.
- **1.81 ≤ Z ≤ 2.99**: the **grey** zone — caution, the model cannot say cleanly.
- **Z < 1.81**: the **distress** zone — elevated bankruptcy risk; the model is flashing red.

#### Worked example: computing Riverstone's Altman Z-score into the distress zone

Let us score Riverstone. From its financials (in millions): total assets \$2,000, current assets \$300, current liabilities \$300 (so working capital = \$300 − \$300 = \$0), retained earnings \$160, EBIT \$120, total liabilities \$1,000, market value of equity \$500 (the stock has fallen as worries grew), and sales \$1,540.

Compute the five ratios:

- **A** = working capital / total assets = \$0 / \$2,000 = **0.00**
- **B** = retained earnings / total assets = \$160 / \$2,000 = **0.08**
- **C** = EBIT / total assets = \$120 / \$2,000 = **0.06**
- **D** = market equity / total liabilities = \$500 / \$1,000 = **0.50**
- **E** = sales / total assets = \$1,540 / \$2,000 = **0.77**

Now apply the weights:

$$Z = 1.2(0.00) + 1.4(0.08) + 3.3(0.06) + 0.6(0.50) + 1.0(0.77)$$

$$Z = 0.00 + 0.11 + 0.20 + 0.30 + 0.77 = 1.38$$

(Rounding the components shown in the figure to two decimals gives ≈1.50; carrying full precision gives ≈1.38 — both land firmly in distress.) Riverstone's Z-score of roughly **1.4** is **below 1.81** — squarely in the **distress zone**. The score tells us *why*, too: the two biggest drags are the near-zero working capital (A contributes nothing) and the weak earning power (C contributes only 0.20 despite its heavy 3.3 weight). Riverstone is being held up almost entirely by its sales turnover (E) and a still-positive market cushion (D) — and that market cushion is exactly what evaporates first when sentiment turns.

For comparison, run the same formula on Northwind with its fortress balance sheet — strong working capital, deep retained earnings, robust EBIT, a large market cap relative to modest liabilities — and the Z-score lands comfortably above 3.0, in the safe zone. Same formula, opposite verdict, because the underlying business is genuinely sturdier.

*The Z-score is not magic — it is just five ratios with weights chosen to separate survivors from casualties — but as a single, fast, hard-to-fool smoke detector, it flags fragile firms quarters before they fail.*

### What the Z-score can and can't do

The Z-score is a smoke detector, not a crystal ball. Its limits matter. The classic 1.2/1.4/3.3/0.6/1.0 weights were fit on public *manufacturers*; Altman published separate variants (the **Z'-score** for private firms, the **Z''-score** for non-manufacturers and emerging markets) because the original misfires on asset-light service and tech businesses that have few tangible assets and little working capital. It uses *market* equity (component D), so it inherits whatever the stock market believes — useful, but it means a bubble-priced stock can flatter a fragile firm's score until the bubble pops. And it says nothing about the *timing* of the maturity wall or covenants — a firm with a great Z-score can still be torpedoed next quarter by a single covenant breach. Use it as a fast first screen that says "look harder here," never as a final verdict.

## Why cash flow, not the balance sheet alone, determines survival

We have built ratios off the balance sheet, but the deepest truth of this whole post is that **the balance sheet alone cannot tell you whether a company will survive** — because survival is a question about *flows*, and the balance sheet is a *snapshot*. The balance sheet tells you what the company owns and owes at one instant. It does not tell you whether cash is coming in faster than it is going out. Two companies with identical balance sheets can have completely different fates: one generating \$200M of operating cash a year, the other burning \$50M a quarter. The first survives almost anything; the second is on a countdown timer no balance-sheet ratio reveals.

This is why the **cash flow statement** — covered in depth in [its own post](/blog/trading/equity-research/cash-flow-statement-where-the-cash-really-comes-from) — is the survival document. Specifically, **operating cash flow** (cash generated by the actual business) and **free cash flow** (operating cash minus capital spending) are what service debt and fund the maturity wall when refinancing stops. A company with strong, stable free cash flow can carry high leverage safely, because it can pay down debt from earnings if it ever has to. A company with weak or negative free cash flow is fragile even at modest leverage, because it has no internal engine to fall back on — it lives entirely at the mercy of lenders. The ratio that fuses the two worlds is the one we already met: **net debt to EBITDA** puts the balance-sheet stock (net debt) over the cash-flow flow (EBITDA), which is exactly why the credit market lives on it rather than on debt-to-equity.

#### Worked example: identical balance sheets, opposite survival odds

Take two firms with the *same* balance sheet: each has \$1,000M of debt, \$100M of cash, \$2,000M of assets, \$1,000M of equity. Every solvency ratio we have computed is identical for the two: same debt-to-equity (1.0), same debt-to-assets (0.5), same net debt (\$900M).

Now look at the cash flows the balance sheet does not show. **CashCo** generates \$300M of operating cash flow a year and spends \$80M on capital, leaving \$220M of free cash flow. **BurnCo** generates only \$60M of operating cash and spends \$90M on capital — *negative* \$30M of free cash flow; it burns \$30M a year just to keep running. When the \$300M maturity wall arrives and markets are jittery, CashCo can plausibly self-fund a big chunk of it from two years of free cash flow and refinance the rest from a position of strength. BurnCo cannot fund a dollar of it from operations — it is already burning cash — so it is entirely dependent on the credit market saying yes. Same balance sheet, same ratios, opposite survival odds, and the difference is invisible on the balance sheet.

*The balance sheet tells you the size of the obligation; only the cash flow statement tells you whether the company has an engine to meet it — survival is a flow question dressed up as a stock question.*

## Liquidity crises vs solvency crises: illiquid-but-solvent vs insolvent

We can now draw the cleanest and most important distinction in the whole field of corporate survival: the difference between a **liquidity crisis** and a **solvency crisis**. They look superficially similar — in both, the company cannot pay something — but they are fundamentally different conditions, with different cures, and confusing them is how both companies *and* their rescuers make catastrophic mistakes.

![Two paths to failure where a liquidity crisis at a solvent firm can be cured by a bridge loan but a solvency crisis at an insolvent firm cannot be fixed by lending more](/imgs/blogs/liquidity-and-solvency-can-the-company-survive-6.png)

A **liquidity crisis** happens at a company that is **solvent but illiquid**: its assets are genuinely worth more than its debts, but it cannot lay hands on cash *right now* to meet an obligation. The Riverstone maturity-wall story is a pure liquidity crisis — solvent firm, frozen markets, no cash this month. The cure for a liquidity crisis is **a bridge of cash**: a short-term loan, an asset sale, a credit line, a central bank stepping in. Lend the solvent firm enough to get past the crunch and it survives and repays you, because the underlying business was sound all along. This is the classic role of a **lender of last resort** — central banks exist partly to provide cash to solvent-but-illiquid firms so a temporary panic does not destroy fundamentally good businesses.

A **solvency crisis** is different in kind. Here the company is **insolvent**: its debts genuinely exceed the value of everything it owns. Equity has been wiped out — by accumulated losses, by asset write-downs, by a collapse in the value of what it holds. The defining feature is that **lending it more money does not help** — it makes things worse. You cannot fix a business that owes \$1,000M and is worth \$700M by lending it another \$200M; you have just added \$200M of debt to a firm that already owes more than it is worth. A solvency crisis can only be resolved by **restructuring** — wiping out or converting debt, usually wiping out the old shareholders — or by **bankruptcy**. The old equity almost always ends up worth zero, because by definition there is nothing left for the residual claimant once the debts exceed the assets.

The reason this distinction is the most important one in the post: **the cure for one crisis is poison for the other.** Lend cash to a solvent-but-illiquid firm and you save it. Lend cash to a genuinely insolvent firm and you have thrown good money after bad and deepened the eventual loss. The single hardest judgment in any financial crisis — for the company, for its lenders, for the central bank — is telling the two apart *in the moment*, when everyone is panicking and the truth is murky. Was Bear Stearns illiquid or insolvent in March 2008? Was Lehman? Trillions of dollars and the shape of the [2008 financial crisis](/blog/trading/finance/lehman-brothers-2008-financial-crisis) turned on getting that one diagnosis right or wrong.

#### Worked example: the same \$300M shortfall, two completely different diagnoses

A company faces a \$300M obligation it cannot meet from cash on hand. The CFO and the would-be rescuer must diagnose: liquidity or solvency?

**Case A — liquidity crisis.** The firm's assets are worth \$2,000M, its total debt is \$1,000M; it is solidly solvent (\$1,000M of equity cushion). It simply cannot raise \$300M *this month* because markets are frozen. Diagnosis: illiquid but solvent. The right action is a **\$300M bridge loan** — and a rational lender should happily make it, because the firm is worth twice its debt; the bridge gets repaid when markets reopen. Cost of getting it wrong (refusing to lend): a perfectly good business is destroyed for the want of a temporary loan.

**Case B — solvency crisis.** Same \$300M shortfall, but now the firm's assets have collapsed to \$900M against \$1,000M of debt — it is insolvent by \$100M; equity is already gone. Lending it \$300M does not save it; it creates a \$1,300M debt against \$900M (plus \$300M cash, immediately spent) of assets. Diagnosis: insolvent. The right action is **restructuring** — the lenders take a haircut or convert debt to equity, the old shareholders are wiped out, and the business continues with a debt load it can actually carry. Cost of getting it wrong (lending more): you have converted a \$100M hole into a \$400M hole and merely delayed the reckoning.

*The exact same cash shortfall demands opposite responses depending on whether the firm is worth more or less than it owes — which is why "is it liquidity or solvency?" is the first and most consequential question in any crisis.*

## Common misconceptions

**"A profitable company can't go bankrupt."** This is the central myth the whole post dismantles. Profit is recorded on an accrual basis; bankruptcy is caused by running out of cash or breaching an obligation. A company can book record profits while its cash drains into uncollected receivables and unsold inventory, or while a maturity wall it cannot refinance bears down. Profitable companies file for bankruptcy every single year — and they almost always do it for a balance-sheet or cash-flow reason, never an income-statement one.

**"A high current ratio means a company is safe."** A high current ratio can mean a fat safety cushion — or it can mean idle, mismanaged working capital: uncollected receivables, dead inventory, cash earning nothing. The best-run companies often run *low* current ratios on purpose because their cash generation is so reliable they do not need a cushion. The ratio measures the size of the cushion, not the quality of what is in it; you have to look inside.

**"Solvency just means assets exceed liabilities on the balance sheet."** Book solvency is necessary but nowhere near sufficient. A firm can be balance-sheet solvent and still die from a liquidity crisis — a maturity wall it cannot refinance, a covenant it trips, a market that freezes. Survival depends on cash flow and the ability to roll debt over, neither of which appears in the snapshot of assets-minus-liabilities. The graveyard is full of solvent companies.

**"More cash on the balance sheet is always better."** Within reason, cash is a fortress — but a company hoarding enormous cash relative to its needs is failing to deploy capital. That idle cash earns a low return and drags down [return on equity](/blog/trading/equity-research/returns-on-capital-roic-roe-roa). The art is holding *enough* cash to survive a frozen market and a bad year, and not so much that the balance sheet becomes a low-yielding savings account that shareholders could invest better themselves.

**"Net debt to EBITDA is a fixed property of the company."** Leverage measured against EBITDA moves even when debt does not, because EBITDA is in the denominator. A firm at 4.0x leverage in a good year can be at 6.0x in a recession *without borrowing a dollar more* — earnings simply fell. This is why covenant-heavy, highly-levered firms are so dangerous: a downturn worsens their leverage ratio at the exact moment they can least afford a covenant breach. Leverage is a moving target, and it moves against you precisely when you are weakest.

**"Bankruptcy means the business is worthless and disappears."** Bankruptcy is usually a *financial* restructuring, not a liquidation of a worthless business. In a reorganization, the operating business often continues largely intact — same factories, same employees, same customers — while the *capital structure* is rebuilt: debt is reduced or converted to equity, and the *old shareholders* are wiped out. The business survives; the owners do not. That is exactly why solvency is an equity-holder's problem first: in a solvency crisis, the residual claimant — the shareholder — is the one who gets zero.

## How it shows up in real markets

The patterns in this post are not abstractions; they are the recurring cause of death in real corporate history. The numbers below are rounded and illustrative of well-documented episodes, used to show the *mechanism*, not as exact audited figures.

**Lehman Brothers (2008): the liquidity-vs-solvency question, with the world watching.** Lehman is the textbook case of the distinction this post is built on. It funded enormous holdings of mortgage assets with vast short-term borrowing that had to be rolled over *constantly* — a maturity profile so short it was effectively one perpetual rollover. When confidence cracked, lenders refused to renew, and Lehman faced a liquidity crisis of historic scale. The trillion-dollar question was whether Lehman was *illiquid but solvent* (in which case a bridge of cash would have saved it) or genuinely *insolvent* (in which case lending more was throwing money into a hole). The judgment came down that its assets were worth far less than its debts — insolvent — and no rescue came. Its September 2008 bankruptcy, the largest in U.S. history, turned a slowing crisis into a global one, and the whole drama hinged on the single diagnosis this post teaches: liquidity or solvency? We trace the full anatomy in the [Lehman case study](/blog/trading/finance/lehman-brothers-2008-financial-crisis).

**Enron (2001): hidden leverage and the maturity wall behind the fraud.** [Enron](/blog/trading/finance/enron-2001-accounting-fraud) is remembered as an accounting fraud, but the engine of its collapse was solvency. Enron had hidden enormous debt in off-balance-sheet entities, so its *reported* leverage looked manageable while its *true* leverage was extreme. Worse, much of that hidden debt had **triggers** — covenants and rating-linked clauses that demanded immediate repayment if Enron's credit rating or stock price fell below thresholds. When the fraud unraveled and the rating was cut, those triggers fired all at once, creating an instant, unpayable maturity wall. A company that had looked solvent collapsed in weeks because its real leverage and its covenant triggers had been concealed. The lesson: the solvency ratios only protect you if the debt they measure is the *real* debt — read the footnotes.

**Working-capital crunches at growing companies.** Less famous but far more common than the headline frauds: fast-growing companies that "grow broke." A company doubling its revenue must fund a doubling of inventory and receivables *before* the cash from those sales arrives. Profit on the income statement soars while the bank account drains. Many promising businesses — especially in manufacturing and retail — have been forced to sell out cheaply or fold not because they were unprofitable but because growth itself consumed cash faster than profit replaced it. This is why a banker looks at the cash conversion cycle and the quick ratio of a high-growth firm far more nervously than at its earnings.

**The 2020 dash for cash.** In March 2020, the pandemic froze credit markets for a few terrifying weeks. Suddenly the cash ratio — usually an academic curiosity — was the only ratio that mattered. Companies with real cash on the balance sheet calmly rode out the freeze; companies that had optimized away every dollar of "idle" cash scrambled to draw down credit lines, slash dividends, and beg for emergency funding. The episode was a live demonstration that liquidity buffers, which look like wasteful drag in good times, are exactly what keeps you alive when the rollover machine stops. Many CFOs permanently raised their target cash buffers afterward — paying a small, constant efficiency cost to insure against the rare frozen-market month that kills the unprepared.

**Forensic cases and fake liquidity.** Sometimes the liquidity on the balance sheet is not even real. In the [Wirecard fraud](/blog/trading/finance/wirecard-the-german-fintech-fraud), roughly €1.9 billion of "cash" that propped up the company's apparent liquidity simply did not exist. A company can show a beautiful cash ratio and quick ratio and still be a fraud if the cash is fictitious — which is the bridge from this survival-ratios post back to the [quality-of-earnings and 10-K-footnote](/blog/trading/equity-research/reading-the-10k-footnotes-and-mda) work: a ratio is only as trustworthy as the audited number you feed it.

## When this matters and further reading

Liquidity and solvency are the questions you ask *before* you ask whether a stock is cheap, because a business that does not survive the next bad year has no intrinsic value to be cheap relative to. Run the liquidity ladder (current → quick → cash) to judge whether the company can pay this year's bills; run the solvency dashboard (debt-to-equity, debt-to-assets, equity ratio, and above all net debt to EBITDA) to judge whether it can carry its debt for the long haul; check the **debt-maturity footnote** in the 10-K for a wall; check the loan covenants for trip-wires; and run the Altman Z-score as a fast smoke detector. Then — and this is the part beginners skip — go to the **cash flow statement**, because survival is ultimately a flow question, and only operating and free cash flow tell you whether the company has an engine to meet the obligations the balance sheet reveals.

To go deeper, read these companion posts:

- [The balance sheet: what a company owns, owes, and is worth](/blog/trading/equity-research/balance-sheet-what-a-company-owns-owes-and-is-worth) — where net debt, working capital, and book solvency come from, line by line.
- [Leverage and coverage: debt that compounds vs kills](/blog/trading/equity-research/leverage-and-coverage-debt-that-compounds-vs-kills) — the next step: how the *cost* of debt (interest coverage, the leverage multiplier) magnifies both returns and ruin.
- [The cash flow statement: where the cash really comes from](/blog/trading/equity-research/cash-flow-statement-where-the-cash-really-comes-from) — the survival document, and why operating and free cash flow, not the balance sheet, decide who lives.
- [Lehman Brothers and the 2008 financial crisis](/blog/trading/finance/lehman-brothers-2008-financial-crisis) — the liquidity-vs-solvency diagnosis played out at world scale.

The single sentence to carry away: **companies do not usually die from low profits — they die from running out of cash while still profitable, or from owing more than the business can ever be worth.** Liquidity tells you whether they survive this year; solvency tells you whether they survive the decade. Learn to read both, and you will see the bankruptcies coming while the income statement is still glowing green.

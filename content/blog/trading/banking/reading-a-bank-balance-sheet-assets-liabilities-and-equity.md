---
title: "Reading a Bank Balance Sheet: Assets, Liabilities and Equity"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "A from-zero guide to the bank balance sheet: why loans are assets, your deposit is the bank's IOU, and the thin slice of equity decides whether the bank survives a bad year."
tags: ["banking", "balance-sheet", "bank-equity", "bank-deposits", "leverage", "loans", "accounting-identity", "finance-explained"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A bank balance sheet is the same two-sided ledger as your household one, but with the signs flipped: the deposit you think of as your money is, on the bank's books, money it *owes you*, and the loan you think of as a debt is, on the bank's books, an *asset it owns*.
>
> - **Assets = liabilities + equity.** Always. The two sides are equal by construction, not by luck. Assets are what the bank owns (loans, securities, cash); liabilities are what it owes (mostly your deposits); equity is the leftover sliver that belongs to shareholders.
> - **Your deposit is the bank's liability.** When you deposit \$100, the bank records a \$100 IOU to you and is free to lend most of it back out. Your asset is its debt — the same dollar, two opposite signs.
> - **Equity is a thin cushion, and that thinness is the whole story.** A typical bank funds roughly 92% of its assets with other people's money and only ~8% with its own. That means a loss of just a few percent of assets can wipe out the owners entirely.
> - **The one number to remember:** at ~8% equity, a bank runs about **12× leverage** — every \$1 of its own money supports about \$12 of assets, so an 8% fall in asset value erases the equity. That single ratio is why banks are both enormously profitable and structurally fragile.

In March 2023, depositors tried to pull **\$42 billion** out of Silicon Valley Bank in a single day, with roughly another **\$100 billion** queued for the next morning. The bank had been declared profitable weeks earlier. Its loans were mostly performing. Nothing about its income statement screamed "about to die." And yet 36 hours later it was the second-largest bank failure in US history. To understand how a "healthy" bank can vanish in a day, you do not start with the dramatic part. You start with the boring part: the balance sheet.

The balance sheet is the single most important page a bank produces, and almost nobody outside finance has ever really read one. That is a shame, because once you can read it, an enormous amount of finance stops being mysterious. You stop wondering why banks pay you a little interest and charge borrowers a lot. You stop being surprised that a bank with billions in profit can collapse over a weekend. You start seeing the trade that every bank on earth is making — and exactly where it can go wrong.

This post builds the bank balance sheet from absolute zero. No accounting background assumed. We will define every term the first time it appears, build the intuition with everyday examples before any formula, and then go deep enough that you could pick up a real bank's annual report and know where to look. The promise of the series is one sentence, and we will keep returning to it: *a bank is a leveraged, confidence-funded maturity-transformation machine — it borrows short, lends long, earns the spread, and survives only as long as depositors trust it and its thin equity cushion absorbs losses faster than they arrive.* The balance sheet is where you can see all of that at once.

![bank balance sheet assets equal liabilities plus equity two column matrix](/imgs/blogs/reading-a-bank-balance-sheet-assets-liabilities-and-equity-1.png)

The figure above is the mental model for the whole post. A bank balance sheet is two columns of equal height. On the left is everything the bank *owns* — its assets. On the right is everything the bank *owes* plus what is left over for its owners — its liabilities and equity. The two columns are the same height because they describe the same pile of money from two angles: what it was spent on, and where it came from. Equity is that thin green slice at the bottom right, and the entire rest of this article is, in one way or another, about how thin it is.

## Foundations: assets, liabilities, equity and the one identity that never breaks

Let us start with the three words that appear on every balance sheet ever written, and define them with no jargon at all.

An **asset** is something you *own* that has value — something that is either worth money now or will bring you money later. Your car is an asset. The cash in your wallet is an asset. Money other people owe you is an asset, because they will pay you back. For a bank, the big assets are the loans it has made (people will repay them, with interest), the securities it has bought (bonds that pay it coupons), and the cash it holds.

A **liability** is the opposite: something you *owe* — a claim that someone else has on you. Your mortgage is a liability. Your unpaid credit-card bill is a liability. For a bank, the giant liability is *deposits* — yes, deposits — because every dollar in a customer's account is a dollar the bank has promised to give back on demand. We will come back to this inversion because it trips everyone up at first.

**Equity** (also called *net worth*, *capital*, or *book value*) is what is left for the owners after you subtract everything you owe from everything you own. It is the answer to the question: "If I sold all my assets at the recorded value and paid off all my debts, what would be left over for me?" For a person, that leftover is your net worth. For a bank, equity is the shareholders' stake — the money the owners put in, plus the profits the bank kept over the years.

Those three definitions give us the single most important equation in all of accounting. It is called the **accounting identity** (an *identity* is an equation that is always true by definition, not just sometimes):

$$\text{Assets} = \text{Liabilities} + \text{Equity}$$

Read it out loud: everything you own equals everything you owe plus what is yours. This is not a law of nature you have to verify; it is true by construction. Rearrange it and it says the same thing a different way:

$$\text{Equity} = \text{Assets} - \text{Liabilities}$$

Equity is the *plug* — the number that makes the two sides balance. Whatever your assets are worth, and whatever you owe, equity is simply the difference. That is why a balance sheet always *balances*: the left column (assets) must equal the right column (liabilities + equity), because equity is defined as exactly the amount needed to make them equal.

#### Worked example: building the world's smallest bank

Let us build a tiny bank from scratch so the identity stops being abstract.

You and a few friends start "Tiny Bank." You put in **\$8** of your own money. That \$8 is the bank's **equity** — it is yours, the owners'. With it, the bank opens its doors.

Now customers walk in and deposit **\$92** in total. Those deposits are not the bank's money; the bank owes every cent of it back. So \$92 is a **liability**.

The bank now has \$8 (yours) + \$92 (depositors') = **\$100 in cash**. Cash is an **asset**. Check the identity:

$$\underbrace{\$100}_{\text{assets (cash)}} = \underbrace{\$92}_{\text{liabilities (deposits)}} + \underbrace{\$8}_{\text{equity}} \quad\checkmark$$

It balances. Now the bank does what banks do: it lends. It keeps \$10 of cash on hand for people who want withdrawals and lends out **\$90** to borrowers. The asset side changes shape — \$10 of cash plus \$90 of loans — but the total is still \$100, and the right side has not moved at all:

$$\underbrace{\$10 \text{ cash} + \$90 \text{ loans}}_{\$100 \text{ assets}} = \underbrace{\$92}_{\text{deposits}} + \underbrace{\$8}_{\text{equity}} \quad\checkmark$$

The intuition: lending does not change the total size of the balance sheet here — it just converts one kind of asset (cash) into another kind (a loan that earns interest). The bank's whole business is choosing *which* assets to hold against the money it owes.

### Why the two sides are really one pile of money seen twice

The reason the identity can never break is worth sitting with, because it is the key to reading any balance sheet. The left side and the right side are not two separate facts that happen to be equal. They are *the same money described twice*.

The right side answers: **where did the money come from?** Some came from owners (equity). Some was borrowed (liabilities). Those are the only two ways any business can get funds: someone gives it to you and keeps an ownership stake, or someone lends it to you and you owe it back.

The left side answers: **what did you do with that money?** You bought things — you hold cash, you made loans, you bought bonds. Every dollar that came in (right side) had to go *somewhere* (left side). It cannot evaporate. So the total of "where it came from" must equal the total of "what we did with it." That is the identity. It holds for a lemonade stand, a tech giant, and a global bank with five trillion dollars on the books.

So when you look at a bank's balance sheet, you are looking at the same pile of money from two directions. The funding directions (right) and the uses (left) are always equal because they are the same pile.

## What a bank owns: the asset side, from loans to cash

Now we can fill in the real content. Start with the left column — what a bank actually owns. There is a remarkable consistency here across almost every commercial bank on earth: the bulk of the assets are loans, a chunk is securities, a slice is cash, and the rest is everything else.

![bank asset mix loans securities cash trading share of total assets](/imgs/blogs/reading-a-bank-balance-sheet-assets-liabilities-and-equity-2.png)

For a typical large universal bank, the asset side breaks down roughly like the chart above: about **52% loans**, **22% securities**, **13% cash and reserves**, and **13% trading and other assets**. These are representative figures, not one specific bank, but the shape is universal — loans dominate, and everything else fills in around them. Let us define each.

**Loans** are the heart of the bank and usually its single biggest asset. A loan is an asset because the borrower will pay it back, with interest. When you take out a \$300,000 mortgage, *you* think of it as a debt — and to you it is. But on the bank's books, that mortgage is a \$300,000 asset, because the bank is owed \$300,000 plus years of interest payments. Loans are the highest-earning asset a bank holds — the interest on loans is most of how a bank makes money — but they are also the *least liquid* (a *liquid* asset is one you can turn into cash quickly without losing much value). You cannot sell a portfolio of small-business loans this afternoon at full price the way you can sell a Treasury bond. And loans carry **credit risk** — the risk that the borrower does not pay back. That combination, high yield but hard to sell and capable of going bad, is what makes loans both the engine and the danger of a bank.

**Securities** are financial instruments the bank buys and holds — overwhelmingly bonds, especially government bonds (like US Treasuries) and mortgage-backed securities. A *bond* is itself just a loan in tradeable form: the bank lends money to a government or company and receives interest plus the principal back at maturity. Securities earn less than loans but have two virtues: they are far more liquid (a Treasury can be sold in seconds), and most carry little or no credit risk. Banks hold securities partly as a place to park money they have not lent out, and partly as a liquidity buffer they can sell in a pinch. As we will see when we discuss SVB, the way a bank *accounts for* its securities can hide enormous risk in plain sight.

**Cash and reserves** are the most liquid asset of all — actual cash, plus money the bank keeps on deposit at the central bank (these central-bank deposits are called *reserves*). Cash earns the least (sometimes nothing), but it is what the bank pays out when you make a withdrawal. A bank that runs out of cash, even if all its loans are perfectly good, is in mortal danger — and that distinction between running out of cash and actually being insolvent is one of the most important in all of banking. We will return to it.

**Trading and other assets** is a catch-all: positions held by the bank's trading desk, derivatives, the bank's own buildings and equipment, goodwill from acquisitions, and so on. For a plain retail bank this slice is small; for a big universal bank with a large markets division it can be substantial.

![bank asset side stack loans securities cash yield versus liquidity](/imgs/blogs/reading-a-bank-balance-sheet-assets-liabilities-and-equity-5.png)

The figure above arranges the asset side by the trade-off that defines it: **yield versus liquidity**. At the top, loans earn the most but are the hardest to sell. At the bottom, cash earns the least but can be paid out instantly. Securities sit in the middle — decent yield, sellable in a day. Every banker on earth spends their career managing the position on this ladder: hold too much cash and you earn nothing; hold too many loans and you cannot meet a wave of withdrawals. The asset side is a constant balancing act between getting paid and staying able to pay.

#### Worked example: a more realistic asset side

Let us scale Tiny Bank up to \$1,000 and split its assets the realistic way, so the percentages become concrete.

Take a bank with **\$1,000** in total assets, split using the typical mix:

- Loans: 52% of \$1,000 = **\$520**
- Securities: 22% of \$1,000 = **\$220**
- Cash and reserves: 13% of \$1,000 = **\$130**
- Trading and other: 13% of \$1,000 = **\$130**

Total: \$520 + \$220 + \$130 + \$130 = **\$1,000** ✓

Now suppose the loans yield 6% a year, securities yield 4%, cash yields 2%, and the trading book yields 3%. The annual income from assets is:

$$(\$520 \times 6\%) + (\$220 \times 4\%) + (\$130 \times 2\%) + (\$130 \times 3\%)$$
$$= \$31.20 + \$8.80 + \$2.60 + \$3.90 = \$46.50$$

So this \$1,000 of assets throws off about **\$46.50** a year — a blended yield of roughly 4.65%. Notice that the loans, just over half the assets, produce \$31.20, or about **two-thirds** of the income. The intuition: a bank earns most of its money from the riskiest, least liquid corner of its balance sheet. That is not an accident; it is the business model. The yield is the reward for taking the credit risk and the liquidity risk.

### The three asset types side by side

Because the asset side is really a menu of trade-offs, it helps to lay the three big categories out together. Each row answers the same questions: what it earns, how fast you can turn it into cash, and what can go wrong.

| Asset | What it is | Yield | Liquidity | Main risk |
|---|---|---|---|---|
| Loans | Money lent to households and firms | Highest | Lowest — hard to sell fast | Credit risk: the borrower defaults |
| Securities | Tradeable bonds the bank holds | Medium | High — sellable in a day | Market/rate risk: price falls when rates rise |
| Cash & reserves | Physical cash + deposits at the central bank | Lowest | Highest — instantly available | Almost none, but earns little |

Read down the "main risk" column and you can already see the two ways a bank's assets get it into trouble. Loans can go bad — *credit risk* — when borrowers stop paying, which writes value off the asset directly. Securities can fall in price — *market risk*, and specifically *interest-rate risk* — when rates rise, even though the issuer never misses a payment. These are genuinely different failure modes. A 2008-style crisis is mostly a *credit* event: loans turning bad on a massive scale. The 2023 SVB episode was mostly a *rate* event: perfectly safe Treasuries falling in price after the Fed hiked. A reader of a balance sheet has to watch both columns of risk, not just the dramatic one.

It is also worth understanding why loans are so illiquid in the first place, since that illiquidity is half of what makes a bank fragile. A loan is a private contract, tailored to one borrower, whose true quality only the bank really knows. There is no deep market where you can sell a single small-business loan at a fair price on a Tuesday afternoon — a buyer would have to re-underwrite it from scratch and would demand a steep discount for the uncertainty. Securities, by contrast, are standardized and traded by thousands of participants, so a Treasury bond has a known price every second of the day. This asymmetry — assets you cannot sell quickly, funded by money that can leave quickly — is the structural tension we will keep coming back to. The asset side is where the bank's money is *locked up*; the funding side is where it can *walk out*.

## What a bank owes: the funding side, where your deposit becomes its debt

Now flip to the right column — where the money comes from. This is where banks differ most dramatically from any company you might be familiar with, and where the famous inversion lives.

![bank funding mix deposits wholesale debt equity share of total funding](/imgs/blogs/reading-a-bank-balance-sheet-assets-liabilities-and-equity-3.png)

For the same typical large bank, the funding side looks like the chart above: about **71% deposits**, **10% wholesale and repo funding**, **7% long-term debt**, **4% other liabilities**, and just **8% equity** — that thin green sliver on the far right. Stare at that for a moment. The bank funds roughly 92 cents of every dollar of assets with *other people's money*, and only about 8 cents with its own. We will spend a whole section on why that matters, but first let us define the pieces.

**Deposits** are by far the largest source of funding, and they are the crux of the whole inversion. When you deposit \$1,000 in your checking account, you experience that as "I have \$1,000 in the bank — that's my money, sitting there." But that is not what is happening on the bank's side. On the bank's books, your \$1,000 deposit is a **liability** — a \$1,000 IOU. The bank has taken your cash, promised to give it back whenever you ask, and is now free to lend most of it out to someone else. *Your deposit is the bank's debt to you.* That is the inversion: the thing you call "my money in the bank" is, from the bank's perspective, money it borrowed from you and now owes back.

This is not a trick or an abuse. It is the entire point of a bank. A bank is, at its core, a machine for borrowing from many people who want their money available on demand and lending to a few who want it for years. The technical name is **maturity transformation** — turning short-term funding (deposits you can withdraw any time) into long-term assets (loans that last for years) — and it is the subject of the [first post in this series, on what a bank actually does and the spread it earns](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread). The balance sheet is simply the snapshot of that machine at one moment.

**Wholesale and repo funding** is money the bank borrows from other financial institutions and markets rather than from retail customers. *Repo* (short for "repurchase agreement") is a way of borrowing cash for a short period by pledging securities as collateral — the bank sells a bond today and agrees to buy it back tomorrow at a slightly higher price, the difference being the interest. Wholesale funding is convenient and can be raised in large size fast, but it is *fickle*: the institutions providing it can yank it the instant they smell trouble. Banks that lean heavily on wholesale funding are more fragile, a lesson Northern Rock and Lehman Brothers both taught in blood.

**Long-term debt** is bonds the bank itself issues — it borrows from investors for several years at a fixed schedule. This funding is stable (it cannot be withdrawn on a whim) but more expensive than deposits. The general principle of the funding stack: the cheaper a funding source, the more likely it is to run; the more stable a source, the more it costs.

**Other liabilities** covers the miscellaneous: accrued expenses, amounts owed to suppliers, certain derivative positions, and so on. For most banks it is small.

**Equity**, the last sliver, is the owners' stake — and it is the only entry on the right side that is *not* owed to anyone. Everything else (deposits, wholesale, debt) is somebody else's money that must be paid back. Equity is the buffer that stands between those creditors and any losses. We are about to see why that 8% sliver decides everything.

There is a hierarchy hidden in the funding side, and it is one of the most useful things to internalize about banks. Funding sources differ along two axes that move together: **cost** and **stickiness**. The general principle, which holds across almost every bank, is that the *cheaper* a funding source, the more dangerous it can be in a crisis, because cheap usually means it can leave fast. Ranking the right side of our typical bank from stickiest to flightiest:

| Funding source | Cost to the bank | Stickiness | Behavior in a panic |
|---|---|---|---|
| Insured retail deposits | Very cheap | Very sticky | Stay put — protected by deposit insurance |
| Uninsured large deposits | Cheap | Moderate | Can leave fast if confidence cracks |
| Wholesale & repo | Moderate | Low | Yanked at the first sign of trouble |
| Long-term debt | More expensive | Locked in | Cannot run before maturity |
| Equity | Most expensive | Permanent | Never runs; absorbs losses instead |

This table is, in many ways, a survival map. A bank funded mostly by small insured deposits — the first row — is buying the cheapest *and* the safest funding there is, which is why a stable retail deposit base is often described as the most valuable thing a bank owns. A bank that has reached for cheap funding in the wrong places — large uninsured deposits and overnight wholesale money — has bought cheapness at the price of fragility, and it will discover the bill the day depositors get nervous. The reason SVB died and most banks did not, despite holding similar underwater bonds, is almost entirely a story about which rows of this table dominated their funding. SVB's was top-heavy with the second and third rows — uninsured, concentrated, flighty money. The role of *deposit insurance* (in the US, the FDIC's \$250,000-per-depositor guarantee) is precisely to convert flighty money into sticky money by removing the depositor's reason to run, and we will see in a later post how central that guarantee is to keeping the whole system from being permanently run-prone.

A useful exercise when you look at any real bank: estimate what fraction of its deposits sit *above* the insurance limit. Those uninsured deposits are the part of the funding base with a live incentive to flee, and a bank with a very high uninsured share is carrying a hidden liquidity risk that never shows up in its profit numbers.

### The inversion, drawn out: your asset is the bank's liability

Because this single idea confuses almost everyone the first time, it deserves its own picture.

![household balance sheet versus bank balance sheet deposit asset and liability](/imgs/blogs/reading-a-bank-balance-sheet-assets-liabilities-and-equity-9.png)

On the left is *your* balance sheet, the household view. Your deposit at the bank is an **asset** — money you own. Your mortgage is a **liability** — money you owe. On the right is the *bank's* balance sheet for the very same two items. Your deposit is now a **liability** — money the bank owes you. Your mortgage is now an **asset** — money the bank is owed. Same two dollars, mirror-image signs.

This is why your bank balance and the bank's books are perfect opposites, line by line. Every deposit that is an asset to a household is a liability to the bank. Every loan that is a liability to a borrower is an asset to the bank. The bank's balance sheet is, quite literally, the inverted reflection of all its customers' balance sheets stacked together. Once this clicks, a lot of confusing finance headlines resolve themselves. "Deposits are flowing out of banks" sounds like banks losing their assets; it actually means banks losing their *funding* — their liabilities walking out the door, forcing them to sell assets to pay.

#### Worked example: tracing one \$100 deposit through the books

Let us follow a single \$100 deposit and watch it appear on two balance sheets at once.

You walk into a bank with **\$100** in cash and deposit it. Three things happen simultaneously:

1. **On your household balance sheet:** your cash (an asset) of \$100 turns into a deposit (also an asset) of \$100. Your net worth is unchanged — you just swapped one asset for another.
2. **On the bank's balance sheet, left side:** the bank's cash (an asset) goes *up* by \$100. It now physically holds your \$100.
3. **On the bank's balance sheet, right side:** the bank records a \$100 deposit liability — it owes you \$100. The identity still balances: assets rose \$100, liabilities rose \$100, equity unchanged.

Now the bank lends \$90 of it to a small business and keeps \$10 as reserve. On the bank's books, \$90 of cash (asset) becomes a \$90 loan (asset). Total assets still \$100 (\$10 cash + \$90 loan). The \$100 you are owed has not changed. The intuition: the same \$100 is simultaneously an asset to you, a liability to the bank, and — once lent out — the funding behind a \$90 asset the bank owns. One dollar wears three hats at once, and that is exactly how a banking system multiplies a deposit into credit. (The system-wide version of this, where the lent-out \$90 gets re-deposited and re-lent, is [money creation and the money multiplier](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier).)

![pipeline deposit becomes bank liability funds loan that earns interest](/imgs/blogs/reading-a-bank-balance-sheet-assets-liabilities-and-equity-7.png)

The pipeline above traces that same journey as a flow. You deposit \$100 — your asset. The bank books it as a liability it owes you. It sets aside \$10 as a reserve buffer and lends the remaining \$90 to a borrower, at which point that \$90 becomes the bank's *asset*. The borrower then pays interest, which is income on that asset. Read left to right, this single picture contains the entire business: a liability (your deposit) is transformed into an asset (a loan) that earns more than the liability costs. The gap between the two is the *spread*, and the spread is how the whole machine pays for itself.

## Equity: the thin cushion that decides everything

We have now seen the two columns. Time to focus on the line that matters most for survival, even though it is the smallest: equity.

Go back to the funding chart. Deposits were 71%, and equity was 8%. That means for every \$100 of assets, only about \$8 belongs to the owners; the other \$92 is borrowed in one form or another. Compare that to a typical household: a family with a \$500,000 house and a \$300,000 mortgage has \$200,000 of equity — 40% of their assets. A conservatively financed company might have 30–50% equity. A bank runs on roughly **8–12%**. Banks are, by the standards of any other business, astonishingly thinly capitalized.

Why so thin? Because thin equity is what makes banking profitable. The less of your own money you use to fund a given pile of assets, the higher the return on the money you *did* put in — as long as nothing goes wrong. This is **leverage**, and it is the amplifier at the center of the entire business. We will quantify it in a moment. But first, understand what equity *does*: it is the **loss-absorbing cushion**. When a loan goes bad and an asset loses value, that loss has to come out of *somewhere*. By the rules of the identity, it cannot come out of deposits — the bank still owes depositors every cent. It comes out of equity. Equity is the shock absorber, the part of the balance sheet that takes the hit so that depositors do not have to.

### How a loss eats the cushion

This is the single most important mechanism in bank risk, so let us make it vivid.

![before and after balance sheet a loan loss is subtracted from equity](/imgs/blogs/reading-a-bank-balance-sheet-assets-liabilities-and-equity-4.png)

The figure shows a bank before and after a loss. *Before:* assets of 100, liabilities of 92, equity of 8 — it balances. Then \$5 of loans go bad and have to be written off. *After:* assets fall to 95 (the \$5 of loans is gone). But liabilities are *unchanged* at 92 — the bank still owes depositors and creditors every cent it owed them yesterday; their loans did not get cheaper just because the bank had a bad year. So where does the \$5 loss go? Straight out of equity. Equity drops from 8 to **3**. The new identity: 95 = 92 + 3 ✓.

Read that again, because it is the crux of everything. A 5% loss in *asset value* caused a **62.5% loss in equity** (from 8 down to 3, a fall of 5/8). The thinness of the cushion is exactly what makes the bank's owners' stake so sensitive. Losses land entirely on equity, and equity is small, so small asset losses become large equity losses. This is leverage working in reverse, and it is why a bank that looks rock-solid in good years can be hollowed out shockingly fast in a bad one.

#### Worked example: a 5% loan loss eats the equity

Let us put real dollars on it with our \$1,000 bank.

Start with a bank that has **\$1,000** in assets, funded by **\$920** of liabilities (mostly deposits) and **\$80** of equity. Check: \$1,000 = \$920 + \$80 ✓. The equity ratio is \$80 / \$1,000 = **8%**.

Now a recession hits and **5% of the bank's loans default** and must be written off. Recall loans were 52% of assets, so loans were \$520; a 5% loss on those loans is:

$$\$520 \times 5\% = \$26 \text{ written off}$$

Assets fall from \$1,000 to **\$974**. Liabilities stay at **\$920** — the bank still owes its depositors the full \$920. So equity absorbs the entire \$26 hit:

$$\text{New equity} = \$974 - \$920 = \$54$$

Equity fell from \$80 to \$54 — a drop of \$26, which is **32.5%** of the owners' stake (\$26 / \$80), caused by a loss of just 2.6% of total assets (\$26 / \$1,000). The new equity ratio is \$54 / \$974 = **5.5%**.

The intuition: because equity is the plug that absorbs all losses, a modest hit to assets becomes a large hit to the owners. Now imagine the loss had been bigger. If 16% of the loans defaulted — \$520 × 16% = \$83.20 — the loss would exceed the entire \$80 of equity. The bank would be **insolvent**: its assets (\$916.80) would be worth less than what it owes (\$920). The owners would be wiped out, and depositors would be looking at a hole. That is what "a few percent from death" really means.

## Leverage: the amplifier hiding in the equity ratio

We have been circling leverage; now let us define it cleanly and measure it. **Leverage** is the ratio of how much you control to how much is actually yours. For a bank, the simplest measure is:

$$\text{Leverage} = \frac{\text{Total assets}}{\text{Equity}}$$

It tells you how many dollars of assets each dollar of the owners' money is supporting. It is the exact inverse of the equity ratio: if equity is 8% of assets, leverage is 1 / 0.08 = 12.5×. If equity were a thin 4%, leverage would be 1 / 0.04 = 25×, and so on.

![leverage equals assets over equity bar across equity ratio scenarios](/imgs/blogs/reading-a-bank-balance-sheet-assets-liabilities-and-equity-6.png)

The chart above turns equity ratios into leverage multiples. At the realistic **8% equity** of our base case (a typical large bank after modern capital rules), leverage is about **12×** — every \$1 of owners' money supports roughly \$12 of assets. At a thin 4%, it balloons to 25×. At 12%, a well-capitalized bank runs around 8×. And a household with 20% equity in its home runs a mere 5×. The pattern is the engine of the whole industry: *less equity means more leverage means a bigger multiplier on both gains and losses.*

Here is why leverage is a double-edged sword. Suppose the bank's assets earn 1% more than its funding costs — a 1% return on assets, which is roughly the long-run norm for US banking. At about 12× leverage, that 1% on assets becomes:

$$1\% \times 12.5 = 12.5\% \text{ return on equity}$$

A 1% return on assets, magnified 12.5 times, is a 12.5% return on the owners' money. That is a strong result, and it is precisely why banking is profitable enough to attract trillions in capital. But the magnifier works both ways. A 1% *loss* on assets, at 12.5× leverage, is a 12.5% loss of equity. And an 8% loss on assets — at 12.5× leverage — wipes out **100%** of the equity. The bank is gone.

#### Worked example: leverage amplifies a gain and then a loss

Let us run the magnifier in both directions with concrete numbers.

Take a bank with **\$1,000** of assets and **\$80** of equity, so liabilities are \$920 and leverage is \$1,000 / \$80 = **12.5×**.

**The good year.** Assets earn a net 1% after all funding costs and expenses: \$1,000 × 1% = **\$10** of profit. That \$10 flows to the owners, whose stake was \$80. Their return on equity is:

$$\frac{\$10}{\$80} = 12.5\%$$

A 1% return on assets became a 12.5% return on equity. Leverage turned a modest spread into a strong result.

**The bad year.** Now assets *lose* 8% — say a wave of defaults knocks \$80 off their value: \$1,000 × 8% = **\$80**. Liabilities are unchanged at \$920. Equity absorbs all of it:

$$\text{New equity} = \$920 - \$920 = \$0$$

Assets are now \$920, liabilities are \$920, equity is **zero**. The owners are wiped out by a loss of only 8% of assets. The intuition: leverage does not change *whether* you win or lose; it changes *how much*. At 12.5×, an 8% move in either direction is the difference between a great year and total destruction. This is why regulators obsess over capital ratios, and why every bank failure in this series traces back, one way or another, to the moment losses grew faster than this thin cushion could absorb them. For the full treatment of why the cushion is set where it is, see [bank capital and leverage: why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion).

## Marking the assets: why the recorded value can be a fiction

So far we have treated the value of the assets as a fixed, knowable number. In reality, the *value at which a bank records its assets* is one of the most consequential and slippery parts of reading a balance sheet — and it is exactly where SVB's risk was hiding.

There are two broad ways a bank can carry an asset on its books. It can record it at **fair value** (also called *mark-to-market*) — the price the asset would fetch if sold today. Or it can record it at **amortized cost** (sometimes called *held-to-maturity*, or HTM) — essentially the price it paid, adjusted over time, on the assumption it will hold the asset until it matures and never sell it.

This sounds like a dry accounting choice. It is not. Consider a bank that bought \$100 of 10-year bonds when interest rates were near zero. Then rates rose sharply. Bond prices move *opposite* to interest rates (when new bonds pay more, old low-rate bonds are worth less — see [price and yield: the seesaw at the heart of bonds](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds)). So those \$100 bonds might now be worth only \$83 if sold. A bank carrying them at fair value would show an asset of \$83 — and the \$17 loss would have already eaten into equity. But a bank carrying them as held-to-maturity gets to keep showing **\$100** on the balance sheet, as if the loss did not exist, *as long as it never sells*. The loss is real; it is just invisible, footnoted as an "unrealized loss" rather than booked against equity.

This is the trap SVB fell into. As of late 2022 it carried roughly **\$91 billion** of securities as held-to-maturity, and the unrealized loss buried in its securities books was around **\$17 billion** — a number large enough to swallow most of its equity. On paper, the balance sheet balanced and the bank looked solvent. But the *fair-value* balance sheet — the one that mattered the moment the bank was forced to sell securities to fund withdrawals — was a different, far weaker creature. When depositors ran and SVB had to sell, the paper loss became a real loss, and the gap between the booked equity and the true equity was exposed all at once.

#### Worked example: an unrealized loss that hollows out a bank

Let us see how a held-to-maturity loss can make a "solvent" bank actually insolvent.

A bank shows assets of **\$1,000** on its balance sheet, liabilities of **\$940**, and equity of **\$60**. By the books, it is solvent (\$1,000 = \$940 + \$60) and its equity ratio is 6%.

But \$400 of those assets are long-dated bonds carried at amortized cost (held-to-maturity). Rates have risen, and at fair value those bonds are now worth only \$320 — an **\$80 unrealized loss** sitting in the footnotes.

The *true* economic balance sheet, if you mark everything to market:

$$\text{Real assets} = \$1,000 - \$80 = \$920$$
$$\text{Real equity} = \$920 - \$940 = -\$20$$

The bank's real equity is **negative \$20** — it is economically insolvent. Its assets, honestly valued, are worth \$20 less than what it owes. Yet its published balance sheet still proudly shows \$60 of equity. The intuition: the held-to-maturity label does not make the loss disappear; it only delays the day it is recognized. A bank can be "well-capitalized" on the page and underwater in reality, and the only thing keeping the fiction alive is *not having to sell*. The instant a deposit run forces sales, the fiction collapses. This is why a careful reader of a bank balance sheet always hunts down the unrealized-loss footnote — it is where the real risk often lives.

## How the balance sheet connects to how a bank lives and dies

We now have all the pieces. Let us assemble them into the picture of the whole machine, because the balance sheet is not just a snapshot — it is a portrait of a trade, and the trade has three pressure points.

**Pressure point one: liquidity.** The asset side ranges from instant-cash to can't-sell-fast loans, while the liability side is dominated by deposits that can leave *today*. A bank is, structurally, promising to repay on demand money it has tied up for years. As long as not too many depositors want their money at once, this works — the law of large numbers smooths it out. But if enough depositors lose confidence simultaneously, the bank cannot turn its loans into cash fast enough to pay them. This is a **bank run**, and the terrifying thing is that it can sink a bank whose loans are *all perfectly good*. Running out of cash and being insolvent are different deaths, and a bank can suffer the first without the second.

The distinction between *illiquidity* and *insolvency* is worth slowing down on, because confusing them is how people misread bank failures. **Insolvency** is a balance-sheet condition: your assets, honestly valued, are worth less than what you owe — equity is negative. **Illiquidity** is a timing condition: your assets may be worth plenty, but you cannot convert them to cash quickly enough to meet payments coming due *right now*. A perfectly solvent bank — one whose loans are all good and whose net worth is solidly positive — can still be killed by illiquidity if a run forces it to sell long-dated assets into a panicked market at fire-sale prices. And here is the cruel twist: the act of selling assets in a hurry can *turn* an illiquidity problem into an insolvency one, because fire-sale prices crystallize losses that erode the equity cushion. The two conditions are separated by a thin membrane, and a run is what tears it. This is why central banks act as a *lender of last resort* — lending freely against good collateral in a panic — so that a solvent-but-illiquid bank does not have to dump assets and convert a survivable squeeze into a fatal hole.

**Pressure point two: solvency.** Solvency is the equity question — are the assets worth more than the liabilities? A bank becomes insolvent when losses on its assets exceed its equity, as we saw in the worked examples. Solvency is a problem of the asset side losing value (bad loans, bond losses) eating through the thin cushion.

**Pressure point three: leverage ties them together.** Because the cushion is thin, the bank is always close to both edges. A liquidity problem (a run) forces asset sales, which crystallizes losses, which threatens solvency. A solvency problem (rumored losses) frightens depositors, which triggers a run, which is a liquidity problem. The two failure modes feed each other, and leverage is the gearing that makes the loop spin fast. This is the spine of the whole series stated in balance-sheet terms: *a bank borrows short and lends long, earns the spread, and survives only as long as confidence holds and its thin equity absorbs losses faster than they arrive.* Every line of the balance sheet is a parameter in that survival equation.

This is also why the *income statement* and the *balance sheet* must be read together. The income statement tells you how much spread the bank is earning — its profit engine — while the balance sheet tells you how much shock it can withstand before that engine stops mattering. A bank can be wildly profitable on the income statement and one bad asset-write-down away from insolvency on the balance sheet. For the profit-engine side, see [the income statement of a bank: net interest income, fees and provisions](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions).

## Common misconceptions

**"The money I deposit just sits in the bank waiting for me."** No. The moment you deposit, the bank records an IOU to you and lends most of your money out. At any time, a bank holds only a small fraction of total deposits as actual cash — roughly the 13% cash-and-reserves slice we saw, and often less in physical form. The other ~87% is out working as loans and securities. This is not fraud; it is the design, and it is why a bank could never repay *all* depositors at once even if every loan it made was perfectly good. The system relies on the fact that depositors do not all show up on the same day.

**"A bank with billions in profit is safe."** Profit and safety are different axes. Profit lives on the income statement; safety lives on the balance sheet, specifically in the equity cushion and the liquidity of the assets. SVB reported a profit shortly before it failed. A bank can earn a fat spread for years and still be destroyed in days if its assets carry hidden losses or its funding runs. Reading only the income statement and ignoring the balance sheet is how analysts miss the next failure.

**"Equity is cash the bank keeps in a vault for emergencies."** Equity is not a pile of money sitting anywhere. It is an *accounting concept* — the difference between assets and liabilities. It tells you how much of the assets the owners financed, not where any specific dollars are. When a bank "uses its capital to absorb a loss," nothing physically moves; the loss simply reduces the assets, and equity is recomputed as the smaller leftover. The "cushion" is a measure of how much asset value can disappear before depositors are at risk, not a stash.

**"More leverage is just greedy banks taking risks."** Leverage is not an abuse layered on top of banking; it *is* banking. A bank that funded its loans 50% with equity would be far safer but would earn its owners a fraction of the return, could not pay competitive deposit rates, and would lend far less to the economy. Society wants banks to be leveraged enough to be useful and capitalized enough to be safe — and the entire apparatus of bank regulation (Basel capital rules) exists to fix where on that spectrum banks must sit. The debate is never "leverage: yes or no," it is "how much."

**"If a bank's assets equal its liabilities plus equity, it must be fine."** The identity *always* balances — that is what makes it an identity. A balanced balance sheet tells you nothing about health; even an insolvent bank's balance sheet balances (equity just goes negative). What matters is the *quality and honest valuation* of the assets and the *stability* of the funding. A balance sheet can balance perfectly while being a work of fiction, if assets are carried above their true worth or if 30% of funding is hot money that will flee at the first headline.

## How it shows up in real banks

**Silicon Valley Bank, March 2023 — the duration trap made visible on the balance sheet.** SVB had ballooned during the tech boom, taking in vast deposits from startups and parking the money in long-dated bonds when rates were near zero. By early 2023 it held about **\$209 billion** in assets, of which roughly **\$91 billion** was securities classified as held-to-maturity. When the Fed hiked rates aggressively, those bonds fell in value, leaving an unrealized loss of around **\$17 billion** — enough to erase most of the bank's equity if recognized. The balance sheet, using the held-to-maturity fiction, still balanced and looked solvent. But the bank's deposit base was extraordinarily concentrated and **94% uninsured** (above the FDIC's \$250,000 protection limit), so when fear spread, depositors had every reason to run. They pulled **\$42 billion** in a day, forcing SVB to sell securities and turn the paper loss into a real one. The balance sheet that had "balanced" was revealed to be missing its cushion. Every element of this post — held-to-maturity accounting, the thin equity slice, deposits as a runnable liability — is visible in that 36-hour collapse. (A fuller treatment lives in [the SVB and Credit Suisse 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

**Lehman Brothers, September 2008 — leverage at 30× and a balance sheet that could not bend.** Lehman was an investment bank, not a deposit-taker, but the balance-sheet logic is identical and more extreme. At the end of 2007 Lehman ran leverage of about **30.7×** — meaning roughly \$30 of assets for every \$1 of equity, an even thinner cushion than a commercial bank. With \$639 billion of assets, an equity slice that small could be wiped out by an asset decline of barely 3%. When the value of its mortgage-related assets fell and its short-term wholesale funding evaporated (it had little in sticky retail deposits to fall back on), there was almost no equity to absorb the losses and almost no stable funding to buy time. Lehman even used an accounting maneuver known as **Repo 105** — temporarily moving about \$50 billion of assets off the balance sheet at quarter-end to make leverage look lower than it was. The lesson: when the cushion is that thin and the funding is that fickle, the balance sheet has no slack, and a small move in asset values is fatal.

**The 2023 industry context — \$17 billion was not unique.** SVB's hidden bond losses were a symptom of a system-wide condition. After years of near-zero rates followed by the fastest hiking cycle in decades, US banks collectively sat on hundreds of billions of dollars of unrealized losses on their securities books — losses that were perfectly legal to keep off the equity line as long as the bonds were labeled held-to-maturity and never sold. The reason most banks survived and SVB did not comes straight from the balance sheet and its funding side: banks with stickier, more insured, more diversified deposits could simply hold their underwater bonds to maturity and wait. SVB, with concentrated, uninsured, flighty deposits, could not. Same asset problem, different funding stability — and the funding side decided who lived.

**The scale of these balance sheets — why a 1% error is enormous.** It is easy to lose perspective on how large these ledgers are.

![horizontal bar chart total assets of the largest banks in trillions](/imgs/blogs/reading-a-bank-balance-sheet-assets-liabilities-and-equity-8.png)

The largest banks in the world carry truly staggering balance sheets, as the chart shows: ICBC of China tops the list around **\$6.3 trillion** in total assets, with the other Chinese megabanks close behind, and JPMorgan Chase leading the US field near **\$4.0 trillion**. At these sizes, the thinness of equity becomes vertiginous. A bank with \$4 trillion of assets and 8% equity has about \$320 billion of capital — which sounds enormous until you realize it is being asked to absorb losses across \$4 trillion of loans, bonds, and trading positions. A 1% error in the valuation of those assets is \$40 billion. An 8% loss erases the entire equity of the fifth-largest bank on earth. The bigger the balance sheet, the more the thin-cushion arithmetic should focus the mind, not reassure it.

**Continental Illinois and the long history of the same trade.** Long before SVB, Continental Illinois — once one of the largest US banks, with about \$40 billion in assets — failed in 1984 in a wholesale-funding run after losses on its loan book ate into a cushion that was, as always, thin. The phrase "too big to fail" entered the language because of it. Across forty years and a complete change of technology, the failure mechanism is the same one this post describes: bad assets eroding a small equity slice, then frightened funding fleeing faster than the bank could sell. The balance sheet is timeless; only the speed of the run changes.

## The takeaway / How to use this

If you take one habit away from this post, let it be this: whenever you encounter a bank — as a depositor, an investor, a citizen reading the news — picture its balance sheet as those two columns and ask three questions in order.

First, **how is it funded?** Look at the right side. What share is deposits versus wholesale and short-term borrowing? Deposits, especially insured retail deposits, are sticky and cheap; wholesale and uninsured money is hot and will run. A bank funded 90% by sticky deposits is a fundamentally different animal from one funded 40% by wholesale money, even if their asset sides look identical. The funding side is where runs begin.

Second, **how thick is the cushion?** Find the equity, divide it by total assets, and invert it to get leverage. At 8% equity you are looking at about 12× leverage and a bank that dies on an 8% asset loss; at 12% you have an 8× bank that can take a much bigger hit. Then go one step further than the headline number and hunt for the unrealized-loss footnote on the securities book — because, as SVB taught, the *published* equity can be a fiction and the *real* equity, marked to market, can be far thinner or even negative.

Third, **how liquid are the assets against how runnable are the liabilities?** Match the left side's liquidity (cash and securities you can sell today versus loans you cannot) against the right side's runnability (how fast the funding can leave). The danger is a bank with illiquid assets funded by runnable liabilities — long bonds and loans funded by uninsured deposits and overnight repo. That mismatch is the duration trap, and it is the most common cause of sudden death in banking.

Notice that none of these three questions appears on the income statement. Profit tells you how the engine is running; the balance sheet tells you whether the vehicle can survive a crash. A bank can post record profits and be one footnote away from insolvency, and the only way to see it is to read the two columns and the notes beneath them. That is the durable skill: not memorizing ratios, but seeing the balance sheet as the portrait of a single, fragile, profitable trade — borrowing short, lending long, balancing on a sliver of equity. Every function we will explore in the rest of this series, and every great failure, is a variation on how that trade was managed or mismanaged.

The balance sheet, in the end, is honest about exactly one thing: a bank is mostly other people's money, lent out and turned into assets, with a thin layer of the owners' money standing between a bad year and a catastrophe. Read it that way and you will rarely be surprised by which banks live and which ones die.

*This is educational material about how bank balance sheets work, not investment advice.*

## Further reading & cross-links

- [What a bank actually does: maturity transformation and the spread](/blog/trading/banking/what-a-bank-actually-does-maturity-transformation-and-the-spread) — the series intro and the spine: borrowing short, lending long, and earning the spread that the balance sheet is built to capture.
- [Bank capital and leverage: why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) — the deep dive on the equity slice we focused on here: how regulators set it, why 3% can wipe a bank, and the full leverage math.
- [The income statement of a bank: net interest income, fees and provisions](/blog/trading/banking/the-income-statement-of-a-bank-net-interest-income-fees-and-provisions) — the profit-engine companion to this post; read the two statements together to judge a bank.
- [How money is created: banks, central banks and the money multiplier](/blog/trading/finance/how-money-is-created-banks-central-banks-money-multiplier) — the system-level view of how one deposit, lent and re-deposited across many balance sheets, multiplies into the money supply.
